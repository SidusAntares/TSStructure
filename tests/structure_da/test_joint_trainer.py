from __future__ import annotations

import torch

from methods.structure_da.joint_trainer import (
    JointStructureDATrainingConfig,
    joint_structure_da_train_step,
    validation_structure_contributions,
)
from methods.structure_da.quality_fusion import HierarchicalQualityObjective
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
        epochs=1, steps_per_epoch=1, lr=1e-3, weight_decay=0.0,
        task_weight=1.0, geometry_weight=1.0, alignment_weight=1.0,
        structural_classification_weight=1.0, structural_domain_weight=1.0,
        component_classification_weight=1.0, component_domain_weight=1.0,
        log_step=1, progress_bar="off", classes=("a", "b", "c"),
    )
    values.update(overrides)
    return JointStructureDATrainingConfig(**values)


def _objective(config):
    return HierarchicalQualityObjective(
        config.structural_classification_weight,
        config.structural_domain_weight,
        config.component_classification_weight,
        config.component_domain_weight,
    )


def test_joint_step_preserves_losses_alignment_and_residual_path() -> None:
    model = _model()
    config = _config()
    task_optimizer = torch.optim.Adam(model.task_parameters(), lr=config.lr)
    geometry_optimizer = torch.optim.Adam(model.geometry_parameters(), lr=config.lr)
    result = joint_structure_da_train_step(
        model, _sample(2, 5), _sample(2, 7, labels=False),
        task_optimizer, geometry_optimizer, _objective(config), config,
        torch.device("cpu"),
    )
    assert torch.isfinite(result.losses.total_loss)
    assert torch.isfinite(result.alignment.loss)
    assert result.mean_alpha_residual.ndim == 0
    assert result.source_batch_size == result.target_batch_size == 2


def test_state_update_uses_detached_three_dimensional_backbone() -> None:
    model = _model()
    backbone = model.forward_backbone(*(_sample(2, 5)[key] for key in ("pixels", "valid_pixels", "positions")))
    detached = model.detach_backbone_for_state(backbone)
    assert detached.tokens.shape == (2, 5, 4)
    assert not detached.tokens.requires_grad
    assert not detached.decomposition.residual.requires_grad
    model.update_source_state_from_backbone(detached, _sample(2, 5)["positions"])


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
    assert set(metrics) == {
        "full_loss", "raw_only_loss", "full_f1", "raw_only_f1", "delta_structure"
    }
    assert metrics["delta_structure"] == metrics["full_f1"] - metrics["raw_only_f1"]


def test_training_config_still_controls_amp_and_progress() -> None:
    config = _config(amp=True, amp_dtype="bfloat16", progress_bar="auto")
    assert config.amp and config.amp_dtype == "bfloat16"
    assert config.progress_bar == "auto"
