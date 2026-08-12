from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from methods.structure_da.phase_geometry import phase_distance
from methods.structure_da.shared_domain_phase_diagnostic import (
    center_phase_distance_matrix,
    class_balanced_weights,
    phase_distance_to_identity_for_gamma,
    sample_equal_weights,
    weighted_frechet_mean_gamma,
    weighted_objective,
)


def _gamma(power: float, k: int = 65) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, k, dtype=torch.float64)
    return grid.pow(power)


def test_class_balanced_weights_give_equal_total_mass_per_class():
    labels = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int64)
    weights = class_balanced_weights(labels).numpy()
    masses = [weights[labels == c].sum() for c in (0, 1, 2)]
    assert np.allclose(masses, [1 / 3, 1 / 3, 1 / 3], atol=1e-12)
    assert np.isclose(weights.sum(), 1.0)


def test_leave_one_class_weights_assign_zero_to_held_out_and_rebalance():
    labels = np.asarray([0, 0, 1, 1, 1, 2], dtype=np.int64)
    weights = class_balanced_weights(labels, held_out_class=1).numpy()
    assert np.allclose(weights[labels == 1], 0.0)
    assert np.isclose(weights[labels == 0].sum(), 0.5)
    assert np.isclose(weights[labels == 2].sum(), 0.5)
    assert np.isclose(weights.sum(), 1.0)


def test_sample_equal_weights_are_uniform():
    weights = sample_equal_weights(7)
    assert torch.allclose(weights, torch.full((7,), 1 / 7, dtype=torch.float64))


def test_weighted_frechet_mean_of_identical_gammas_is_same_gamma():
    gamma = _gamma(1.3)
    gammas = torch.stack([gamma, gamma, gamma], dim=0)
    result = weighted_frechet_mean_gamma(gammas, torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64))
    assert result.converged
    assert torch.max(torch.abs(result.gamma - gamma)).item() < 1e-10
    assert result.objective < 1e-12


def test_weighted_frechet_mean_is_monotone_and_beats_identity_objective():
    gammas = torch.stack([_gamma(1.15), _gamma(1.25), _gamma(1.35), _gamma(1.45)], dim=0)
    weights = sample_equal_weights(gammas.shape[0])
    result = weighted_frechet_mean_gamma(gammas, weights)
    assert result.gamma[0].item() == 0.0
    assert result.gamma[-1].item() == 1.0
    assert torch.all(result.gamma[1:] > result.gamma[:-1])
    identity = torch.linspace(0.0, 1.0, gammas.shape[1], dtype=torch.float64)
    assert result.objective <= weighted_objective(gammas, identity, weights) + 1e-10


def test_class_balancing_is_invariant_to_replication_inside_one_class():
    a = _gamma(0.8); b = _gamma(1.25)
    labels_small = np.asarray([0, 1], dtype=np.int64)
    gammas_small = torch.stack([a, b])
    center_small = weighted_frechet_mean_gamma(gammas_small, class_balanced_weights(labels_small)).gamma

    labels_big = np.asarray([0, 0, 0, 0, 1], dtype=np.int64)
    gammas_big = torch.stack([a, a, a, a, b])
    center_big = weighted_frechet_mean_gamma(gammas_big, class_balanced_weights(labels_big)).gamma
    assert phase_distance(center_small, center_big).item() < 1e-8


def test_sample_equal_center_can_change_under_class_imbalance():
    a = _gamma(0.8); b = _gamma(1.3)
    labels = np.asarray([0, 0, 0, 0, 0, 1], dtype=np.int64)
    gammas = torch.stack([a, a, a, a, a, b])
    balanced = weighted_frechet_mean_gamma(gammas, class_balanced_weights(labels)).gamma
    sample_equal = weighted_frechet_mean_gamma(gammas, sample_equal_weights(len(labels))).gamma
    assert phase_distance(balanced, sample_equal).item() > 1e-3


def test_center_phase_distance_matrix_is_symmetric():
    gammas = torch.stack([_gamma(0.8), _gamma(1.0), _gamma(1.25)])
    matrix = center_phase_distance_matrix(gammas)
    assert torch.allclose(matrix, matrix.T, atol=1e-12)
    assert torch.allclose(torch.diag(matrix), torch.zeros(3, dtype=torch.float64))


def test_phase_distance_to_identity_matches_formal_phase_distance():
    gamma = _gamma(1.25)
    identity = torch.linspace(0.0, 1.0, gamma.numel(), dtype=torch.float64)
    assert abs(phase_distance_to_identity_for_gamma(gamma) - phase_distance(gamma, identity).item()) < 1e-12


def test_experiment_10_launcher_has_no_registration_fallback_or_training():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "run_class_balanced_shared_domain_phase_leave_one_class_out_at1_dk1_seed1.sh").read_text(encoding="utf-8")
    assert "registration_calls=0" in text
    assert "clustering=false" in text
    assert "--stage-a-cache" in text
    assert "train.py" not in text


def test_experiment_10_script_does_not_import_clustering_or_registration_solver():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "diagnose_class_balanced_shared_domain_phase_leave_one_class_out.py").read_text(encoding="utf-8")
    forbidden = ("KMeans", "HDBSCAN", "silhouette_score", "optimum_reparam_curve", "solve_t_only_registrations")
    for token in forbidden:
        assert token not in text
    assert "weighted_frechet_mean_gamma" in text
    assert "align_target_positions_to_source" in text
