from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from methods.structure_da.candidate_structure_diagnostic import (
    complementarity_diagnostics,
    empirical_percentiles_by_class,
    exact_cosine_knn,
    fixed_quantile_grid,
    fixed_top_intersections,
    loo_class_centroid_distances,
    source_knn_observables,
    target_knn_observables,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "diagnose_stage2_missing_semantic_structure_13a2.py"
LAUNCHER = REPO_ROOT / "scripts" / "run_stage2_missing_semantic_structure_13a2_at1_dk1_seed1.sh"


def test_loo_source_distance_does_not_use_own_sample_in_centroid():
    features = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [-1.0, -1.0],
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])
    distances, centers = loo_class_centroid_distances(features, labels, 2)
    # For sample [1,0], the LOO center is mean([0,1],[1,1])=[0.5,1],
    # which is not the full class center [2/3,2/3].
    loo_expected = 1.0 - 0.5 / np.sqrt(1.25)
    full_expected = 1.0 - (2.0 / 3.0) / np.sqrt(8.0 / 9.0)
    assert abs(distances[0] - loo_expected) < 1e-8
    assert abs(distances[0] - full_expected) > 1e-3
    assert np.allclose(centers[0], [2 / 3, 2 / 3])


def test_empirical_percentile_is_class_conditioned():
    reference = np.array([0.1, 0.2, 0.3, 10.0, 20.0, 30.0])
    classes = np.array([0, 0, 0, 1, 1, 1])
    query = np.array([0.25, 25.0])
    query_class = np.array([0, 1])
    got = empirical_percentiles_by_class(reference, classes, query, query_class, 2)
    assert np.allclose(got, [2 / 3, 2 / 3])


def test_exact_target_knn_excludes_self_and_returns_group_boundary_distances():
    features = np.array([
        [1.0, 0.0],
        [0.99, 0.1],
        [0.0, 1.0],
        [0.1, 0.99],
    ], dtype=np.float32)
    groups = np.array([0, 0, 1, 1])
    result = exact_cosine_knn(
        features, features, k_max=2, device=torch.device("cpu"), chunk_size=2,
        exclude_self=True, query_reference_indices=np.arange(4),
        query_groups=groups, reference_groups=groups,
    )
    assert np.all(result.indices[:, 0] != np.arange(4))
    assert result.nearest_same_distance is not None
    assert result.nearest_other_distance is not None
    assert np.all(result.nearest_same_distance < result.nearest_other_distance)


def test_source_knn_candidate_fraction_uses_source_true_labels_only():
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    source = np.array([[1.0, 0.01], [0.99, 0.02], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    source_labels = np.array([0, 0, 1, 1])
    candidate = np.array([0])
    knn = exact_cosine_knn(
        query, source, k_max=4, device="cpu", query_groups=candidate, reference_groups=source_labels,
    )
    obs = source_knn_observables(knn, source_labels, candidate, num_classes=2, k_values=(2, 4))
    assert obs["source_knn2_candidate_fraction"][0] == 1.0
    assert obs["source_knn4_candidate_fraction"][0] == 0.5
    assert obs["source_nearest_candidate_vs_other_margin"][0] > 0


def test_target_knn_prediction_agreement_is_observable_not_true_label_vote():
    features = np.array([[1, 0], [0.99, 0.01], [0.98, 0.02], [0, 1]], dtype=np.float32)
    raw_pred = np.array([0, 0, 0, 1])
    posterior = np.array([
        [0.9, 0.1], [0.8, 0.2], [0.85, 0.15], [0.1, 0.9]
    ])
    knn = exact_cosine_knn(
        features, features, k_max=3, device="cpu", exclude_self=True,
        query_reference_indices=np.arange(4), query_groups=raw_pred, reference_groups=raw_pred,
    )
    obs = target_knn_observables(knn, raw_pred, posterior, num_classes=2, k_values=(2, 3))
    assert obs["target_knn2_candidate_fraction"][0] == 1.0
    assert obs["target_knn2_candidate_posterior_mean"][0] > 0.8
    assert obs["target_neighborhood_margin"][0] > 0


def test_fixed_4x4_and_top_intersection_do_not_search_thresholds():
    a = np.linspace(0, 1, 40)
    b = np.linspace(0, 1, 40)[::-1]
    correctness = np.array([False] * 20 + [True] * 20)
    grid = fixed_quantile_grid(a, b, correctness, a_higher_is_support=True, b_higher_is_support=True)
    assert len(grid) == 16
    assert {r["a_support_quartile"] for r in grid} == {1, 2, 3, 4}
    joint = fixed_top_intersections(a, b, correctness, a_higher_is_support=True, b_higher_is_support=True)
    assert [r["requested_each_coverage"] for r in joint] == [0.25, 0.10]


def test_complementarity_reports_support_oriented_spearman():
    a = np.arange(20, dtype=float)
    b = np.arange(20, dtype=float)
    correctness = np.array([False] * 10 + [True] * 10)
    rho, grid, joint = complementarity_diagnostics(
        a, b, correctness, a_higher_is_support=True, b_higher_is_support=True,
    )
    assert abs(rho - 1.0) < 1e-12
    assert len(grid) == 16
    assert len(joint) == 2


def test_13a2_script_contract_has_hard_oracle_boundary_and_no_training_gate():
    text = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert 'PROTOCOL = "13A2_missing_semantic_structure_and_evidence_complementarity_diagnostic"' in text
    assert '"delta_boot": "identity"' in text
    assert "target true label leaked into 13A-2 frozen feature extraction" in text
    assert "oracle field leaked into 13A-2 label-free sample artifact" in text
    assert "source-crossfit-folds" in text
    assert "protocol-fixed to 5-fold cross-fit" in text
    assert "13_metric_interpretation.md" in text
    assert "不按 AUROC 排名后自动挑 best evidence" in text
    assert '("austria/33UVP/2017", "denmark/32VNH/2017", 1, 0)' in text
    assert '(source, target, seed, fold) != ("AT1", "DK1", 1, 0)' not in text
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called.add(node.func.id)
    assert "backward" not in called
    assert "step" not in called
    assert "fit" not in called
    assert "LogisticRegression" not in called
    assert "KMeans" not in called
    assert "RandomForestClassifier" not in called
    assert "XGBClassifier" not in called


def test_13a2_launcher_is_identity_fixed_and_reuses_13a1_output():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "delta_boot=identity" in text
    assert "EXPERIMENT13A1_DIR" in text
    assert "--source-crossfit-folds 5" in text
    assert "training_updates=false" in text
    assert "trainable_gate=false" in text
