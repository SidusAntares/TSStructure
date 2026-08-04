from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from methods.structure_da.joint_trainer import (
    JointStructureDATrainingConfig,
    joint_structure_da_train_step,
    validation_structure_contributions,
)
from methods.structure_da.phase_aware_objective import (
    PhaseAwareTaskLossWeights,
    PhaseAwareTaskObjective,
)
from methods.structure_da.quality_fusion import TwoScaleQualityObjective
from tests.structure_da.test_full_model import _model


def _sample(batch: int, length: int, *, labels: bool = True):
    torch.manual_seed(1000 + length)
    sample = {
        "pixels": torch.randn(batch, length, 2, 4),
        "valid_pixels": torch.ones(batch, length, 4, dtype=torch.bool),
        "positions": torch.linspace(0, 300, length).round().long().expand(batch, -1),
    }
    if labels:
        sample["label"] = torch.arange(batch) % 3
    return sample


def _config(**overrides):
    values = dict(
        epochs=1,
        steps_per_epoch=1,
        lr=1e-3,
        weight_decay=0.0,
        log_step=1,
        progress_bar="off",
        classes=("a", "b", "c"),
    )
    values.update(overrides)
    return JointStructureDATrainingConfig(**values)


def _objectives(config):
    quality = TwoScaleQualityObjective(
        config.quality_classification_weight,
        config.quality_domain_weight,
    )
    task = PhaseAwareTaskObjective(
        PhaseAwareTaskLossWeights(
            classification=config.classification_weight,
            quality=config.quality_weight,
            source_shape=config.source_shape_weight,
            source_raw=config.source_raw_weight,
            global_domain=config.global_domain_weight,
            target_semantic=config.target_semantic_weight,
        )
    )
    return quality, task


def _step(model=None, config=None):
    model = model or _model()
    config = config or _config()
    quality, task = _objectives(config)
    task_optimizer = torch.optim.Adam(model.task_parameters(), lr=config.lr)
    geometry_optimizer = torch.optim.Adam(model.geometry_parameters(), lr=config.lr)
    result = joint_structure_da_train_step(
        model,
        _sample(3, 5),
        _sample(2, 7, labels=False),
        task_optimizer,
        geometry_optimizer,
        quality,
        task,
        config,
        torch.device("cpu"),
    )
    return model, result


def test_training_config_has_only_phase_aware_weights_and_validates_them() -> None:
    names = {field.name for field in fields(JointStructureDATrainingConfig)}
    assert {
        "geometry_weight",
        "classification_weight",
        "quality_weight",
        "source_shape_weight",
        "source_raw_weight",
        "global_domain_weight",
        "target_semantic_weight",
        "quality_classification_weight",
        "quality_domain_weight",
    } <= names
    assert not {
        "task_weight",
        "alignment_weight",
        "structural_classification_weight",
        "structural_domain_weight",
        "component_classification_weight",
        "component_domain_weight",
    } & names
    with pytest.raises(ValueError, match="classification_weight"):
        _config(classification_weight=-1)
    with pytest.raises(ValueError, match="lr"):
        _config(lr=0)
    assert _config(amp=True, amp_dtype="bfloat16", progress_bar="auto").amp


def test_joint_step_uses_new_losses_and_different_domain_lengths() -> None:
    model, result = _step()
    assert result.source_batch_size == 3
    assert result.target_batch_size == 2
    assert result.quality.total_loss.ndim == 0
    assert result.mean_alpha_trend.ndim == 0
    assert result.mean_alpha_structure.ndim == 0
    expected = result.losses.task.total_loss + result.losses.geometry.total_loss
    torch.testing.assert_close(result.losses.reported_total_loss, expected)
    assert all(torch.isfinite(value) for value in result.diagnostics.scalars.values())
    assert all(parameter.grad is None for parameter in model.geometry_parameters())


def test_geometry_step_precedes_task_forward_and_state_updates_once_after_both() -> None:
    model = _model()
    config = _config()
    quality, task = _objectives(config)
    task_optimizer = torch.optim.Adam(model.task_parameters(), lr=config.lr)
    geometry_optimizer = torch.optim.Adam(model.geometry_parameters(), lr=config.lr)
    events: list[str] = []

    original_geometry_step = geometry_optimizer.step
    original_task_step = task_optimizer.step
    original_forward = model.forward_from_backbone
    original_update = model.update_source_state_from_output

    def geometry_step(*args, **kwargs):
        events.append("geometry_step")
        return original_geometry_step(*args, **kwargs)

    def task_step(*args, **kwargs):
        events.append("task_step")
        return original_task_step(*args, **kwargs)

    def task_forward(*args, **kwargs):
        events.append("task_forward")
        return original_forward(*args, **kwargs)

    def state_update(*args, **kwargs):
        events.append("state_update")
        return original_update(*args, **kwargs)

    geometry_optimizer.step = geometry_step
    task_optimizer.step = task_step
    model.forward_from_backbone = task_forward
    model.update_source_state_from_output = state_update

    joint_structure_da_train_step(
        model,
        _sample(3, 5),
        _sample(2, 7, labels=False),
        task_optimizer,
        geometry_optimizer,
        quality,
        task,
        config,
        torch.device("cpu"),
    )
    assert events == [
        "geometry_step",
        "task_forward",
        "task_forward",
        "task_step",
        "state_update",
    ]


def test_source_state_is_updated_once_and_target_has_no_update_api() -> None:
    model, _ = _step()
    assert model.temporal_features.core.trend_template.num_updates > 0
    assert model.temporal_features.core.structure_diagnostic_template.num_updates > 0
    assert model.prototype_alignment.trend_update_count.sum() == 3
    assert model.prototype_alignment.structure_update_count.sum() == 3
    assert not hasattr(model, "update_target_state")


def test_cpu_bfloat16_amp_step_keeps_phase_semantics_finite() -> None:
    _, result = _step(config=_config(amp=True, amp_dtype="bfloat16"))
    assert torch.isfinite(result.losses.reported_total_loss)
    assert all(torch.isfinite(value) for value in result.diagnostics.scalars.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_float16_grad_scaler_step_is_finite() -> None:
    device = torch.device("cuda")
    model = _model().to(device)
    config = _config(amp=True, amp_dtype="float16")
    quality, task = _objectives(config)
    task_optimizer = torch.optim.Adam(model.task_parameters(), lr=config.lr)
    geometry_optimizer = torch.optim.Adam(model.geometry_parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    result = joint_structure_da_train_step(
        model,
        _sample(3, 5),
        _sample(2, 7, labels=False),
        task_optimizer,
        geometry_optimizer,
        quality,
        task,
        config,
        device,
        task_scaler=scaler,
    )
    assert torch.isfinite(result.losses.reported_total_loss)


def test_validation_counterfactual_reuses_one_model_pass_per_batch() -> None:
    model = _model()
    calls = 0
    original = model.forward_details

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.forward_details = counted
    loader = [_sample(2, 5), _sample(2, 7)]
    metrics = validation_structure_contributions(
        model, loader, torch.device("cpu"), ("a", "b", "c")
    )
    assert calls == len(loader)
    modes = {"full", "no_shape", "trend_only", "structure_only", "shape_only"}
    assert {f"{mode}_{suffix}" for mode in modes for suffix in ("loss", "f1")} <= set(metrics)
    assert metrics["delta_shape"] == metrics["full_f1"] - metrics["no_shape_f1"]
    assert metrics["delta_trend"] == metrics["full_f1"] - metrics["structure_only_f1"]
    assert metrics["delta_structure"] == metrics["full_f1"] - metrics["trend_only_f1"]
