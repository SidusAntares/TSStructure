from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from methods.structure_da.joint_trainer import (
    JointStructureDATrainingConfig,
    joint_structure_da_train_step,
    train_joint_structure_da,
)
from methods.structure_da.phase_aware_objective import (
    PhaseAwareTaskObjective,
)
from methods.structure_da.quality_fusion import TwoScaleQualityObjective
from tests.structure_da.test_full_model import _model
from tests.structure_da.test_joint_trainer import _config, _objectives, _sample


def _run_step(model, config, source, target):
    quality, task = _objectives(config)
    task_optimizer = torch.optim.Adam(model.task_parameters(), lr=config.lr)
    geometry_optimizer = torch.optim.Adam(model.geometry_parameters(), lr=config.lr)
    return joint_structure_da_train_step(
        model,
        source,
        target,
        task_optimizer,
        geometry_optimizer,
        quality,
        task,
        config,
        torch.device("cpu"),
    )


def test_two_complete_steps_activate_phase_aware_source_state_and_losses() -> None:
    model = _model()
    config = _config()
    source = _sample(3, 5)
    target = _sample(3, 7, labels=False)
    first = _run_step(model, config, source, target)
    second = _run_step(model, config, source, target)

    assert model.temporal_features.core.trend_template.num_updates > 0
    assert model.temporal_features.core.structure_diagnostic_template.num_updates > 0
    assert model.prototype_alignment.trend_prototype_ready.any()
    assert model.prototype_alignment.structure_prototype_ready.any()
    assert second.losses.geometry.valid_candidate_sample_count >= 0
    assert second.losses.geometry.source_center_count >= 0
    for result in (first, second):
        assert torch.isfinite(result.losses.reported_total_loss)
        assert torch.isfinite(result.losses.task.total_loss)
        assert torch.isfinite(result.losses.geometry.total_loss)

    output = model.forward_details(
        source["pixels"], source["valid_pixels"], source["positions"]
    )
    quality = output.representation.quality
    expected = torch.cat(
        [quality.weighted_trend, quality.weighted_structure, quality.shape_feature],
        dim=-1,
    )
    torch.testing.assert_close(output.representation.fused_feature, expected)
    assert not hasattr(output, "z_phase")
    assert not hasattr(output.representation, "dynamics_embedding")
    assert not hasattr(quality, "beta_trend_temporal")


def test_checkpoint_roundtrip_restores_state_and_eval_logits() -> None:
    model = _model()
    config = _config()
    source = _sample(3, 5)
    _run_step(model, config, source, _sample(3, 7, labels=False))
    model.eval()
    with torch.no_grad():
        expected = model(
            source["pixels"], source["valid_pixels"], source["positions"]
        )
    state = model.state_dict()
    restored = _model()
    restored.load_state_dict(state)
    restored.eval()
    with torch.no_grad():
        actual = restored(
            source["pixels"], source["valid_pixels"], source["positions"]
        )
    torch.testing.assert_close(actual, expected)
    assert restored.temporal_features.core.trend_template.num_updates.equal(
        model.temporal_features.core.trend_template.num_updates
    )
    assert restored.prototype_alignment.q_update_count.equal(
        model.prototype_alignment.q_update_count
    )
    assert restored.alignment.grl.iteration.equal(model.alignment.grl.iteration)


def test_one_epoch_joint_training_saves_phase_aware_checkpoint(tmp_path) -> None:
    model = _model()
    config = JointStructureDATrainingConfig(
        epochs=1,
        steps_per_epoch=2,
        lr=1e-3,
        weight_decay=0.0,
        progress_bar="off",
        log_step=1,
        classes=("a", "b", "c"),
    )
    checkpoint_path = tmp_path / "model.pt"
    source_loader = [_sample(3, 5)]
    target_loader = [_sample(3, 7, labels=False)]
    val_loader = [_sample(3, 5)]
    best = train_joint_structure_da(
        model,
        source_loader,
        target_loader,
        val_loader,
        config,
        None,
        torch.device("cpu"),
        checkpoint_path,
    )
    assert torch.isfinite(torch.tensor(best))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["training_config"]["classification_weight"] == 1.0
    restored = _model()
    restored.load_state_dict(checkpoint["state_dict"])


def test_cli_exposes_only_phase_aware_high_level_options() -> None:
    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for option in (
        "--shape_dim",
        "--lambda_source_shape",
        "--lambda_source_raw",
        "--lambda_target_semantic",
        "--lambda_q_to_raw_target",
        "--warp_num_candidates",
    ):
        assert option in result.stdout
    for option in (
        "--structure_dim",
        "--lambda_task",
        "--lambda_alignment",
        "--lambda_structural_cls",
        "--lambda_component_cls",
    ):
        assert option not in result.stdout


def test_train_protocol_reports_cpu_bfloat16_amp_as_enabled() -> None:
    source = Path("train.py").read_text(encoding="utf-8")

    assert 'device.type == "cuda"' in source
    assert 'training_config.amp_dtype == "bfloat16"' in source


def test_joint_training_builds_only_new_objectives(monkeypatch, tmp_path) -> None:
    quality_calls = 0
    task_calls = 0
    original_quality = TwoScaleQualityObjective.__init__
    original_task = PhaseAwareTaskObjective.__init__

    def quality_init(self, *args, **kwargs):
        nonlocal quality_calls
        quality_calls += 1
        original_quality(self, *args, **kwargs)

    def task_init(self, *args, **kwargs):
        nonlocal task_calls
        task_calls += 1
        original_task(self, *args, **kwargs)

    monkeypatch.setattr(TwoScaleQualityObjective, "__init__", quality_init)
    monkeypatch.setattr(PhaseAwareTaskObjective, "__init__", task_init)
    config = _config()
    train_joint_structure_da(
        _model(),
        [_sample(3, 5)],
        [_sample(3, 7, labels=False)],
        [_sample(3, 5)],
        config,
        None,
        torch.device("cpu"),
        tmp_path / "model.pt",
    )
    assert quality_calls == 1
    assert task_calls == 1
