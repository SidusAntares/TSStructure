from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from methods.structure_da import (
    DomainShapeStatus,
    Stage2Objective,
    Stage2ObjectiveConfig,
)
from tests.structure_da.test_stage2_objective import _bank, _source, _weights


def test_phase_only_stage2_objective_does_not_require_fake_confirmed_delta() -> None:
    source_logits, source_fused, labels, q, support, valid = _source()
    synthetic_logits = torch.randn(3, 3, requires_grad=True)
    objective = Stage2Objective(
        num_classes=3,
        config=Stage2ObjectiveConfig(
            lambda_src_proto=0.0,
            lambda_src_cons=0.0,
            lambda_syn=1.0,
            lambda_syn_cons=0.1,
            tau_q=0.1,
            fused_margin=0.1,
        ),
    )
    output = objective(
        source_logits=source_logits,
        source_fused_repr=source_fused,
        source_labels=labels,
        source_q=q,
        source_q_support=support,
        source_q_valid=valid,
        source_prototype_bank=_bank(),
        integration_weights=_weights(),
        synthetic_logits=synthetic_logits,
        synthetic_labels=labels,
        synthetic_q=q.detach(),
        synthetic_q_support=support,
        synthetic_q_valid=valid,
        domain_shape_state=None,
        lambda_delta=None,
    )
    output.total.backward()
    assert output.synthetic_count == 3
    assert synthetic_logits.grad is not None


def test_stage2_objective_has_no_target_label_input() -> None:
    parameters = inspect.signature(Stage2Objective.forward).parameters
    assert "target_labels" not in parameters
    assert "stable_target_labels" not in parameters


def test_round7_trainer_contains_no_target_ce_or_adversarial_path() -> None:
    text = Path("methods/structure_da/stage2_trainer.py").read_text(encoding="utf-8")
    forbidden = ("target_ce", "gradient reversal", "GRL", "DANN")
    for token in forbidden:
        assert token not in text
    assert "Stage2Objective" in text


def test_train_wires_stage2_without_scientific_defaults() -> None:
    text = Path("train.py").read_text(encoding="utf-8")
    for flag in (
        "--stage2_config",
        "--stage2_registration_lambda",
        "--stage2_phase_confirmation_patience",
        "--stage2_shape_confirmation_patience",
        "--stage2_lambda_src_proto",
        "--stage2_lambda_src_cons",
        "--stage2_lambda_syn",
        "--stage2_lambda_syn_cons",
        "--stage2_objective_tau_q",
        "--stage2_fused_margin",
        "--stage2_ema_decay",
        "--stage2_lambda_delta",
    ):
        assert flag in text
    assert 'default=None' in text
    assert 'load_structure_da_state_dict(model, stage1_checkpoint["model_state_dict"])' in text
    assert "configure_stage2_parameter_policy(model)" in text
    assert "Stage2EMATeacher.from_student" in text
    assert "run_stage2_training(" in text


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
