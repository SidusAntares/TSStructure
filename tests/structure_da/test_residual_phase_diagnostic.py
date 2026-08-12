from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from methods.structure_da.phase_geometry import phase_distance
from methods.structure_da.residual_phase_diagnostic import (
    build_residual_population,
    compose_phase,
    deterministic_class_folds,
    inverse_phase,
    residual_center_summary,
    residual_from_shared,
)


def _gamma(power: float, k: int = 65) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, k, dtype=torch.float64)
    value = grid.pow(power)
    value[0] = 0.0
    value[-1] = 1.0
    return value


def test_phase_compose_and_inverse_follow_source_to_target_function_order():
    shared = _gamma(1.15)
    residual = _gamma(0.90)
    sample = compose_phase(shared, residual)
    recovered = residual_from_shared(sample, shared)
    rebuilt = compose_phase(shared, recovered)
    assert float(phase_distance(sample, rebuilt)) < 1e-7
    identity = compose_phase(shared, inverse_phase(shared))
    expected = torch.linspace(0.0, 1.0, shared.numel(), dtype=torch.float64)
    assert float(phase_distance(identity, expected)) < 1e-7


def test_residual_population_hard_reconstruction_gate_passes_exact_factorization():
    shared = {0: _gamma(1.10), 1: _gamma(0.95)}
    residuals = [_gamma(0.92), _gamma(1.04), _gamma(1.08), _gamma(0.88)]
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    sample = torch.stack([compose_phase(shared[int(c)], r) for c, r in zip(labels.tolist(), residuals)])
    audit = build_residual_population(sample, labels, shared, reconstruction_tolerance=1e-7)
    assert audit.fail_count == 0
    assert audit.max_error < 1e-7
    assert audit.residuals.shape == sample.shape


def test_class_folds_are_reproducible_balanced_and_order_independent():
    ids = np.asarray([20, 10, 50, 40, 30, 120, 110, 150, 140, 130], dtype=np.int64)
    labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    folds = deterministic_class_folds(ids, labels, n_folds=5, seed=17)
    folds_again = deterministic_class_folds(ids, labels, n_folds=5, seed=17)
    assert np.array_equal(folds, folds_again)
    for class_id in (0, 1):
        assert sorted(folds[labels == class_id].tolist()) == [0, 1, 2, 3, 4]
    order = np.asarray([4, 0, 2, 1, 3, 9, 5, 7, 6, 8])
    reordered = deterministic_class_folds(ids[order], labels[order], n_folds=5, seed=17)
    mapping = {int(sid): int(fold) for sid, fold in zip(ids[order], reordered)}
    assert [mapping[int(sid)] for sid in ids] == folds.tolist()


def test_residual_center_summary_uses_fisher_rao_frechet_center():
    residuals = torch.stack([_gamma(0.95), _gamma(1.00), _gamma(1.05), _gamma(1.02)])
    summary = residual_center_summary(3, residuals)
    assert summary.class_id == 3
    assert summary.center.shape == residuals.shape[1:]
    assert summary.dispersion_mean_squared >= 0.0
    assert summary.distance_q25 <= summary.distance_median <= summary.distance_q75 <= summary.distance_q90


def test_experiment_11_script_forbids_mechanism_expansion_and_requires_reconstruction():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "diagnose_shared_phase_residual_class_structure_and_generalization.py").read_text(encoding="utf-8")
    assert "build_residual_population" in text
    assert "automatic_route_decision\": None" in text
    assert '"registration_calls": 0' in text
    assert '"clustering": False' in text
    assert '"class_specific_alpha": False' in text
    assert '"residual_gating": False' in text
    assert '"training_updates": False' in text
    assert "weighted_frechet_mean_gamma" in text
    assert "solve_t_only_registrations" not in text
    assert "optimum_reparam_curve" not in text


def test_experiment_11_launcher_freezes_five_fold_protocol():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "run_shared_phase_residual_class_structure_and_generalization_at1_dk1_seed1.sh").read_text(encoding="utf-8")
    assert "--cv-folds 5" in text
    assert "--expected-valid-count 10634" in text
    assert "registration_calls=0" in text
    assert "clustering=false" in text
