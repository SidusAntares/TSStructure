from pathlib import Path

import numpy as np

from methods.structure_da.stage2_bootstrap_diagnostic import (
    add_identity_deltas,
    classwise_semantic_distribution,
    confusion_matrix,
    hard_transition_rows,
    overall_metrics,
    per_class_metrics,
    row_normalize_confusion,
)


def _example():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probs = np.asarray([
        [0.9, 0.1],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.7, 0.3],
    ], dtype=np.float64)
    return labels, probs, ["a", "b"]


def test_per_class_metrics_include_prediction_count_and_support_ratio():
    labels, probs, classes = _example()
    rows = per_class_metrics(labels, probs, classes)
    assert rows[0]["true_support"] == 2
    assert rows[0]["predicted_count"] == 2
    assert rows[0]["predicted_to_true_support_ratio"] == 1.0
    assert rows[0]["correct_count"] == 1
    assert rows[1]["correct_count"] == 1


def test_overall_metrics_report_macro_and_collapse_diagnostics():
    labels, probs, classes = _example()
    row = overall_metrics(labels, probs, classes)
    assert row["accuracy"] == 0.5
    assert 0.0 <= row["macro_f1"] <= 1.0
    assert 0.0 <= row["predicted_class_entropy"] <= np.log(2.0) + 1e-12
    assert row["largest_predicted_class_fraction"] == 0.5


def test_identity_deltas_are_classwise():
    labels, probs, classes = _example()
    identity = per_class_metrics(labels, probs, classes)
    shifted = [dict(row) for row in identity]
    shifted[0]["recall"] += 0.1
    rows = add_identity_deltas(shifted, identity)
    assert abs(rows[0]["delta_recall_vs_identity"] - 0.1) < 1e-12
    assert rows[1]["delta_recall_vs_identity"] == 0.0


def test_hard_transition_rows_do_not_use_thresholds():
    labels = np.asarray([0, 0, 1, 1])
    before = np.asarray([[0.6, 0.4], [0.4, 0.6], [0.4, 0.6], [0.7, 0.3]])
    after = np.asarray([[0.7, 0.3], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
    rows = hard_transition_rows(labels, before, after, ["a", "b"])
    all_row = rows[0]
    assert all_row["wrong_to_correct"] == 2
    assert all_row["correct_to_wrong"] == 1
    assert all_row["net_correct_gain"] == 1


def test_confusion_row_normalization():
    labels = [0, 0, 1]
    pred = [0, 1, 1]
    matrix = confusion_matrix(labels, pred, 2)
    norm = row_normalize_confusion(matrix)
    np.testing.assert_allclose(norm.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(norm[0], [0.5, 0.5])


def test_classwise_semantic_distribution_has_required_quantiles():
    labels, probs, classes = _example()
    rows = classwise_semantic_distribution(labels, probs, classes)
    assert len(rows) == 2 * 4
    assert {row["metric"] for row in rows} == {
        "max_probability", "prediction_margin", "prediction_entropy", "true_class_probability"
    }
    for row in rows:
        for key in ("mean", "median", "q10", "q25", "q75", "q90"):
            assert key in row


def test_experiment12_script_keeps_frozen_protocol_and_target_train_oracle_phase():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "diagnose_stage2_bootstrap_temporal_state.py").read_text(encoding="utf-8")
    assert "target_train_oracle_true_class_registrations.pt" in text
    assert "class_balanced_weights" in text
    assert "production_legality_filter" in text
    assert "automatic_bootstrap_choice" in text
    assert "Teacher" not in text or "Teacher/Student" in text
    assert "pseudo-label training" not in text.lower()


def test_experiment12_launcher_is_no_training_and_resumable():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "run_stage2_bootstrap_temporal_state_at1_dk1_seed1.sh").read_text(encoding="utf-8")
    assert "stage2_training=false" in text
    assert "target_train_oracle_true_class_registrations.pt" in text
    assert "REGISTRATION_WORKERS" in text
