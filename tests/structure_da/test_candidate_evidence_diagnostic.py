from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from methods.structure_da.candidate_evidence_diagnostic import (
    auprc,
    auroc,
    candidate_margin,
    confidence_matched_indices,
    jensen_shannon_rows,
    precision_at_coverages,
    summarize_evidence,
    wrong_reject_at_correct_retention,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "diagnose_stage2_trainable_seed_evidence_13a.py"
LAUNCHER = REPO_ROOT / "scripts" / "run_stage2_trainable_seed_evidence_13a_at1_dk1_seed1.sh"


def test_js_rows_zero_for_identical_and_positive_for_different_posteriors():
    p = np.array([[0.9, 0.1], [0.5, 0.5]], dtype=float)
    same = jensen_shannon_rows(p, p)
    assert np.allclose(same, 0.0, atol=1e-12)
    q = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=float)
    diff = jensen_shannon_rows(p, q)
    assert np.all(diff > 0.0)


def test_candidate_margin_is_candidate_minus_best_competitor():
    values = np.array([[0.7, 0.2, 0.1], [0.4, 0.5, 0.1]], dtype=float)
    candidate = np.array([0, 0])
    got = candidate_margin(values, candidate)
    assert np.allclose(got, [0.5, -0.1])


def test_auc_and_average_precision_are_perfect_for_perfect_ranking():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(auroc(labels, scores) - 1.0) < 1e-12
    assert abs(auprc(labels, scores) - 1.0) < 1e-12


def test_precision_coverage_rewards_high_purity_top_region():
    labels = np.array([1, 1, 0, 0, 0], dtype=bool)
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    rows = precision_at_coverages(labels, scores, coverages=(0.2, 0.4, 1.0))
    assert rows[0]["precision"] == 1.0
    assert rows[1]["precision"] == 1.0
    assert abs(rows[2]["precision"] - 0.4) < 1e-12


def test_wrong_reject_is_defined_at_correct_retention_without_gate_search():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.2, 0.1])
    result = wrong_reject_at_correct_retention(labels, scores, 0.90)
    assert result["correct_retention"] >= 0.90
    assert result["wrong_reject_rate"] >= 0.5


def test_confidence_matching_balances_correct_and_wrong_counts_within_cells():
    rng = np.random.default_rng(4)
    n = 200
    correct = np.array([True] * 100 + [False] * 100)
    prob = np.concatenate([rng.uniform(0.5, 1.0, 100), rng.uniform(0.5, 1.0, 100)])
    margin = np.concatenate([rng.uniform(0.0, 0.8, 100), rng.uniform(0.0, 0.8, 100)])
    mask, diag = confidence_matched_indices(correct, prob, margin, seed=9, bins=5)
    assert diag["n_correct_matched"] == diag["n_wrong_matched"]
    assert diag["n_correct_matched"] > 0
    assert int(np.sum(correct & mask)) == int(np.sum((~correct) & mask))


def test_summarize_evidence_respects_lower_is_more_reliable_orientation():
    correct = np.array([True, True, False, False])
    raw_error = np.array([0.1, 0.2, 0.8, 0.9])
    row = summarize_evidence(raw_error, correct, higher_is_reliable=False, seed=1, bootstrap_reps=50)
    assert row["auroc"] == 1.0
    assert row["correct_mean"] < row["wrong_mean"]


def test_13a_script_uses_hard_label_stripping_boundary_and_disallows_oracle_bootstrap():
    text = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert "class LabelStrippedLoader" in text
    assert 'FORMAL_BOOTSTRAP_STATES = ("identity", "timematch_scalar")' in text
    assert "oracle_shared" not in text.split("FORMAL_BOOTSTRAP_STATES", 1)[1].split("\n", 1)[0]
    # Observable generation must reject an accidental label-bearing batch.
    assert "label leaked into 13A unlabeled observable generation" in text
    assert "oracle field leaked into label-free observable artifact" in text
    assert isinstance(tree, ast.Module)


def test_13a_launcher_requires_explicit_post_experiment12_bootstrap_state():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'if [[ -z "${BOOTSTRAP_STATE:-}" ]]' in text
    assert "SEED13A_BOOTSTRAP_STATE_REQUIRED" in text
    assert "identity,timematch_scalar" in text


def test_13a_contract_contains_no_training_or_learned_gate_calls():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    called_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
    assert "backward" not in called_names
    assert "step" not in called_names
    assert "fit" not in called_names
    assert "LogisticRegression" not in called_names
    assert "KMeans" not in called_names
