from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from methods.structure_da import Stage2Objective, Stage2ObjectiveConfig


def test_stage2_objective_is_timematch_style_two_branch_loss() -> None:
    objective = Stage2Objective(
        num_classes=3,
        config=Stage2ObjectiveConfig(lambda_target=1.0, focal_gamma=1.0),
    )
    source_logits = torch.randn(3, 3, requires_grad=True)
    target_logits = torch.randn(2, 3, requires_grad=True)
    output = objective(
        source_to_target_logits=source_logits,
        source_labels=torch.tensor([0, 1, 2], dtype=torch.long),
        native_target_logits=target_logits,
        stable_target_labels=torch.tensor([1, 2], dtype=torch.long),
    )
    output.total.backward()
    assert output.source_count == 3
    assert output.target_count == 2
    assert source_logits.grad is not None
    assert target_logits.grad is not None


def test_stage2_objective_accepts_stable_target_labels_but_not_geometry_losses() -> None:
    parameters = inspect.signature(Stage2Objective.forward).parameters
    assert "stable_target_labels" in parameters
    assert "native_target_logits" in parameters
    assert "source_prototype_bank" not in parameters
    assert "domain_shape_state" not in parameters
    assert "synthetic_q" not in parameters


def test_stage2_trainer_has_native_target_student_path_and_no_adversarial_path() -> None:
    import re

    text = Path("methods/structure_da/stage2_trainer.py").read_text(encoding="utf-8")
    for pattern in (r"gradient reversal", r"\bGRL\b", r"\bDANN\b"):
        assert re.search(pattern, text, flags=re.IGNORECASE) is None
    assert "def _target_forward_native(" in text
    target_api = inspect.signature(
        __import__(
            "methods.structure_da.stage2_trainer",
            fromlist=["Stage2Trainer"],
        ).Stage2Trainer._target_forward_native
    )
    assert "temporal_positions_override" not in target_api.parameters
    assert "native target positions only" in text


def test_train_wires_timematch_style_stage2_controls() -> None:
    text = Path("train.py").read_text(encoding="utf-8")
    for flag in (
        "--stage2_config",
        "--stage2_registration_lambda",
        "--stage2_phase_confirmation_patience",
        "--stage2_shape_confirmation_patience",
        "--stage2_lambda_target",
        "--stage2_focal_gamma",
        "--stage2_lr",
        "--stage2_target_time_keep_ratio",
        "--stage2_ema_decay",
        "--stage2_lambda_delta",
    ):
        assert flag in text
    assert "create_target_stage2_train_loader" in text
    assert "target_stable_label_loader=target_stable_label_loader" in text
    assert "target_train_loader=target_stage2_train_loader" in text
    assert "|stable_label_scan=" in text
    assert "|native_student=" in text
    assert 'load_structure_da_state_dict(model, stage1_checkpoint["model_state_dict"])' in text
    assert "configure_stage2_parameter_policy(model)" in text
    assert "Stage2EMATeacher.from_student" in text
    assert "CosineAnnealingLR" in text
    assert "scheduler=stage2_scheduler" in text
    assert 'lr=stage2_lr' in text
    assert "run_stage2_training(" in text


def test_formal_timematch_stage2_config_freezes_optimization_recipe() -> None:
    import json

    payload = json.loads(
        Path("configs/stage2_timematch_v1.json").read_text(encoding="utf-8")
    )
    assert payload["stage2_lr"] == pytest.approx(1e-4)
    assert payload["stage2_ema_decay"] == pytest.approx(0.9999)
    assert payload["stage2_focal_gamma"] == pytest.approx(1.0)
    assert payload["stage2_lambda_target"] == pytest.approx(1.0)
    assert payload["stage2_target_time_keep_ratio"] == pytest.approx(0.8)


def test_stage2_only_checkpoint_defaults_to_current_fold(tmp_path) -> None:
    pytest.importorskip("zarr")
    import train as train_module

    config = type("Config", (), {})()
    config.fold_dir = str(tmp_path / "fold_0")
    config.stage1_checkpoint = None
    config.stage2_only = True
    config.num_folds = 1
    assert train_module._resolve_stage1_checkpoint_path(config, 0) == str(
        tmp_path / "fold_0" / "stage1_best.pt"
    )


def test_stage2_only_wiring_skips_stage1_and_reuses_formal_boundary() -> None:
    text = Path("train.py").read_text(encoding="utf-8")
    stage1_call = text.index("train_source_classification(", text.index("def main(config):"))
    resume_guard = text.rfind('if not getattr(config, "stage2_only", False):', 0, stage1_call)
    checkpoint_load = text.index("stage1_checkpoint = torch.load", stage1_call)
    source_scan = text.index("source_bank = build_source_prototype_bank", checkpoint_load)
    stage2_call = text.index("stage2_result = run_stage2_training", source_scan)

    assert resume_guard != -1
    assert resume_guard < stage1_call < checkpoint_load < source_scan < stage2_call
    assert "STAGE2_RESUME|" in text
    assert "--stage2_only" in text
    assert "--stage1_checkpoint" in text


def test_stage2_diagnostic_only_is_guarded_and_skips_training() -> None:
    text = Path("train.py").read_text(encoding="utf-8")
    assert "--stage2_diagnostic_only" in text
    assert 'raise ValueError("--stage2_diagnostic_only requires --stage2_only")' in text
    diagnostic_call = text.index("run_stage2_statistics_diagnostic(stage2_trainer)")
    training_call = text.index("stage2_result = run_stage2_training", diagnostic_call)
    continue_pos = text.index("continue", diagnostic_call)
    assert diagnostic_call < continue_pos < training_call
    assert "STAGE2_DIAGNOSTIC_ONLY_EXIT|training_skipped=true" in text
