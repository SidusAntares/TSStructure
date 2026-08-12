from pathlib import Path

import torch

from methods.structure_da.oracle_gamma_harm_diagnostic import (
    classifier_margin_and_competitor,
    geometry_margin,
    hard_transition,
    shrink_gamma_toward_identity,
    value_space_residual_diagnostic,
    warp_value_function_gamma,
)


def test_gamma_shrinkage_preserves_endpoints_and_monotonicity():
    gamma = torch.tensor([0.0, 0.08, 0.32, 0.70, 1.0], dtype=torch.float64)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = shrink_gamma_toward_identity(gamma, alpha)
        assert abs(float(out[0])) < 1e-12
        assert abs(float(out[-1]) - 1.0) < 1e-12
        assert torch.all(out[1:] > out[:-1])
    assert torch.allclose(
        shrink_gamma_toward_identity(gamma, 0.0),
        torch.linspace(0.0, 1.0, gamma.numel(), dtype=torch.float64),
    )
    assert torch.allclose(shrink_gamma_toward_identity(gamma, 1.0), gamma)


def test_classifier_margin_and_geometry_margin_use_strongest_competitor():
    margin, competitor = classifier_margin_and_competitor(
        torch.tensor([0.2, 1.0, 0.7]), true_class=1
    )
    assert competitor == 2
    assert abs(margin - 0.3) < 1e-6

    gmargin, gcompetitor = geometry_margin(
        torch.tensor([0.6, 0.2, 0.4]), true_class=1
    )
    assert gcompetitor == 2
    assert abs(gmargin - 0.2) < 1e-6


def test_hard_transition_has_no_artificial_gain_threshold():
    assert hard_transition(False, True) == "beneficial_hard"
    assert hard_transition(True, False) == "harmful_hard"
    assert hard_transition(True, True) == "stable_correct"
    assert hard_transition(False, False) == "stable_wrong"


def test_value_warp_uses_target_values_at_gamma_positions():
    values = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float64)
    gamma = torch.tensor([0.0, 0.125, 0.5, 0.875, 1.0], dtype=torch.float64)
    warped = warp_value_function_gamma(values, gamma)
    assert torch.allclose(warped[:, 0], torch.tensor([0.0, 0.5, 2.0, 3.5, 4.0], dtype=torch.float64))


def test_analysis_affine_fit_explains_global_scale_and_channel_offsets():
    source = torch.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0], [4.0, 7.0]], dtype=torch.float64
    )
    # source = 2 * target + channel-wise offset [1, 2]
    target = (source - torch.tensor([1.0, 2.0], dtype=torch.float64)) / 2.0
    support = torch.ones(4, dtype=torch.float64)
    weights = torch.tensor([0.125, 0.375, 0.375, 0.125], dtype=torch.float64)
    diag = value_space_residual_diagnostic(target, source, support, support, weights)
    assert diag.value_residual > 0
    assert diag.affine_residual < 1e-10
    assert diag.affine_residual_ratio < 1e-8
    assert abs(diag.affine_scale - 2.0) < 1e-8


def test_08_script_is_zero_dp_and_does_not_add_forbidden_mechanisms():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "diagnose_oracle_gamma_harm_mechanism.py"
    ).read_text(encoding="utf-8")
    assert "solve_t_only_registrations" not in script
    assert "optimum_reparam_curve" not in script
    assert '"exact_dp_calls": 0' in script
    assert '"production_legality_filter": False' in script
    assert '"phase_grouping": False' in script
    assert '"stable_label": False' in script
    assert '"teacher_student": False' in script
    assert '"parameter_update": False' in script


def test_08_launcher_has_no_exact_dp_fallback():
    launcher = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_oracle_gamma_harm_mechanism_at1_dk1_seed1.sh"
    ).read_text(encoding="utf-8")
    assert "ORACLE_GAMMA_08_MISSING_INPUT" in launcher
    assert "exact_dp_fallback=false" in launcher
    assert "run_sample_level_phase_validity" not in launcher
