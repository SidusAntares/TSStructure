from pathlib import Path
import ast

import numpy as np
import torch

from methods.structure_da.phase_geometry import phase_distance
from methods.structure_da.phase_population_diagnostic import (
    common_language_effect,
    deterministic_equal_class_subset,
    deterministic_random_subset,
    exact_phase_knn,
    gamma_to_unit_phase_vectors,
    knn_label_statistics,
    phase_distance_to_identity,
    phase_pair_distances,
    sample_within_cross_pairs,
)


def _gammas():
    grid = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    return torch.stack([grid, grid.square(), torch.sqrt(grid)], dim=0)


def test_vectorized_phase_distance_matches_formal_phase_geometry():
    gammas = _gammas()
    vectors = gamma_to_unit_phase_vectors(gammas)
    pairs = phase_pair_distances(vectors, [0, 0, 1], [1, 2, 2])
    expected = np.asarray([
        float(phase_distance(gammas[0], gammas[1]).item()),
        float(phase_distance(gammas[0], gammas[2]).item()),
        float(phase_distance(gammas[1], gammas[2]).item()),
    ])
    assert np.allclose(pairs, expected, atol=1e-10)
    identity = phase_distance_to_identity(vectors)
    assert float(identity[0]) < 1e-8


def test_exact_phase_knn_uses_formal_geometry_without_labels():
    gammas = _gammas()
    vectors = gamma_to_unit_phase_vectors(gammas)
    indices, distances = exact_phase_knn(vectors, k_max=2, device=torch.device("cpu"), block_size=2)
    assert indices.shape == (3, 2)
    assert distances.shape == (3, 2)
    for row in range(3):
        assert row not in indices[row].tolist()
        formal = sorted(
            (float(phase_distance(gammas[row], gammas[col]).item()), col)
            for col in range(3) if col != row
        )
        assert indices[row].tolist() == [item[1] for item in formal]


def test_knn_labels_are_posthoc_statistics_only():
    neighbors = np.asarray([[1,2],[0,2],[1,0],[2,1]], dtype=np.int64)
    labels = np.asarray([0,0,1,1], dtype=np.int64)
    stats = knn_label_statistics(neighbors, labels, k_values=(1,2))
    same1, entropy1 = stats[1]
    assert np.allclose(same1, [1,1,0,1])
    assert np.allclose(entropy1, 0.0)
    same2, entropy2 = stats[2]
    assert np.all((same2 >= 0.0) & (same2 <= 1.0))
    assert np.all(entropy2 >= 0.0)


def test_pair_sampling_is_reproducible_and_keeps_both_strata():
    labels = np.asarray([0,0,0,1,1,2,2,2,2], dtype=np.int64)
    a = sample_within_cross_pairs(labels, count_each=50, seed=7)
    b = sample_within_cross_pairs(labels, count_each=50, seed=7)
    for left, right in zip(a, b):
        assert np.array_equal(left, right)
    wl, wr, cl, cr = a
    assert np.all(labels[wl] == labels[wr])
    assert np.all(labels[cl] != labels[cr])


def test_effect_size_direction_and_subset_sampling():
    effect = common_language_effect(np.asarray([0.1,0.2,0.3]), np.asarray([0.7,0.8,0.9]))
    assert effect["prob_within_smaller"] == 1.0
    assert effect["rank_biserial"] == 1.0
    r1 = deterministic_random_subset(100, 20, 3)
    r2 = deterministic_random_subset(100, 20, 3)
    assert np.array_equal(r1, r2)
    labels = np.repeat(np.arange(4), 10)
    eq = deterministic_equal_class_subset(labels, per_class=4, seed=5)
    assert [int(np.sum(labels[eq] == c)) for c in range(4)] == [4,4,4,4]


def test_09_script_has_no_population_fitting_calls():
    script = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_oracle_true_class_phase_population_structure.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    forbidden = {"KMeans", "HDBSCAN", "silhouette_score", "davies_bouldin_score", "sqrt_mean_gamma", "sqrt_median_gamma"}
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    assert forbidden.isdisjoint(called)
    text = script.read_text(encoding="utf-8")
    assert "registration_solver_called\":False" in text.replace(" ", "")
    assert "clustering_performed\":False" in text.replace(" ", "")
    assert "group_count_selected\":False" in text.replace(" ", "")
