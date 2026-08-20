from pathlib import Path

import torch

from methods.structure_da.timematch_nonlinear_phase import (
    candidate_phase_grid,
    canonical_grid,
    evaluate_phase_grid,
    inverse_phase_on_canonical_grid,
)


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "scripts/train_timematch_phase.py"
LAUNCHER = ROOT / "scripts/run_timematch_phase_4tasks_4gpu_seed1.sh"


def test_fixed_source_phase_is_inverse_of_initial_target_phase():
    u = canonical_grid(129)
    rho = u + 0.025 * torch.sin(torch.pi * u) * u * (1 - u)
    target_to_source = candidate_phase_grid(delta_days=-18, alpha=.5, rho_dom=rho)
    source_to_target = inverse_phase_on_canonical_grid(target_to_source)
    recovered = evaluate_phase_grid(evaluate_phase_grid(u, source_to_target), target_to_source)
    assert torch.max(torch.abs(recovered - u)).item() < 5e-4


def test_single_stage2_entry_uses_alpha_candidates_instead_of_legacy_modes():
    text = TRAINER.read_text(encoding="utf-8")
    assert "--alpha-candidates" in text
    assert "requires_nonlinear_phase" in text
    for obsolete in ("ORIGINAL_TIMEMATCH", "ALPHA_ZERO_SANITY", "NONLINEAR_PHASE_SEARCH"):
        assert obsolete not in text
    for obsolete_import in (
        "visualize_stage2_phase_alignment",
        "diagnose_sample_level_phase_validity",
        "stage2_trainer",
    ):
        assert obsolete_import not in text


def test_alpha_zero_skips_geometry_and_nonlinear_bank_builds_it_once():
    text = TRAINER.read_text(encoding="utf-8")
    assert "nonlinear_enabled = requires_nonlinear_phase(alpha_bank)" in text
    assert "if nonlinear_enabled:" in text
    assert text.count("_build_frozen_residual(") == 2
    assert '"registration_refresh_count": 1 if nonlinear_enabled else 0' in text


def test_trainer_keeps_delta_then_alpha_search_and_native_target_student():
    text = TRAINER.read_text(encoding="utf-8")
    delta = text.index("current_delta = tm.estimate_shift")
    alpha = text.index("current_alpha, _, alpha_summary = _alpha_select", delta)
    assert delta < alpha
    assert "for delta in" not in text
    assert "student, subset, shift_days=0.0" in text
    assert "target truth leaked into TimeMatch nonlinear Phase training" in text


def test_stage1_checkpoint_embeds_runtime_configuration():
    text = (ROOT / "scripts/train_original_timematch_source.py").read_text(encoding="utf-8")
    assert "source_val_loader, _ = create_evaluation_loaders(config.source" in text
    assert "_, target_test_loader = create_evaluation_loaders(config.target" in text
    assert 'checkpoint["runtime_config"]' in text
    assert '"classes": list(config.classes)' in text
    assert "PseLTae" in text and "FocalLoss" in text
    for forbidden in ("shape_loss", "geometry_loss", "decomposition_loss", "z_shape"):
        assert forbidden not in text


def test_four_gpu_launcher_runs_one_task_per_gpu_and_defaults_to_alpha_zero():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'ALPHA_CANDIDATES="${ALPHA_CANDIDATES:-0}"' in text
    for task in ("AT1_DK1", "DK1_FR2", "FR1_AT1", "FR2_FR1"):
        assert task in text
    for gpu in ("$GPU0", "$GPU1", "$GPU2", "$GPU3"):
        assert gpu in text
    assert "train_timematch_phase.py" in text
    assert "train_stage2_original_timematch.py" not in text
    assert "--epochs 20" in text and "--steps-per-epoch 500" in text


def test_final_target_test_and_frozen_geometry_guards_are_present():
    text = TRAINER.read_text(encoding="utf-8")
    assert 'runtime["source"], classes, source_val' in text
    assert 'output / "source_val_metrics.csv"' in text
    assert 'output / "target_val_metrics.csv"' not in text
    assert "TMNP_TEST" in text
    assert 'out / "model.pt"' in text
    assert "geometry_guard.assert_frozen()" in text
    assert "geometry_pse state changed across a Stage2 epoch" in text
