import numpy as np
import pytest


def test_teacher_continuous_features_match_probability_definition():
    from analysis.structure_evidence_increment_audit import teacher_continuous_features

    logits = np.array([[3., 2., 0.], [0., 1., 4.]])
    packet = teacher_continuous_features(logits)
    assert packet["probability"].shape == logits.shape
    np.testing.assert_allclose(packet["probability"].sum(1), 1.)
    np.testing.assert_array_equal(packet["ranking"][:, 0], np.array([0, 2]))
    np.testing.assert_allclose(
        packet["margin"],
        np.sort(packet["probability"], axis=1)[:, -1]
        - np.sort(packet["probability"], axis=1)[:, -2],
    )


def test_teacher_and_combined_models_use_identical_rows_and_source_only_labels():
    from analysis.structure_evidence_increment_audit import fit_reliability_pair

    rng = np.random.default_rng(4)
    teacher = rng.normal(size=(40, 3))
    structure = rng.normal(size=(40, 4))
    correct = np.array([0, 1] * 20)
    result = fit_reliability_pair(teacher, structure, correct, min_errors=5)
    assert result.status == "ok"
    assert result.sample_count == 40
    assert result.teacher_feature_dim == 3
    assert result.combined_feature_dim == 7
    with pytest.raises(ValueError, match="same samples"):
        fit_reliability_pair(teacher, structure[:-1], correct, min_errors=5)


def test_reliability_is_unestimable_when_source_errors_are_insufficient():
    from analysis.structure_evidence_increment_audit import fit_reliability_pair

    teacher = np.zeros((20, 3)); structure = np.zeros((20, 4))
    result = fit_reliability_pair(
        teacher, structure, np.ones(20, dtype=int), min_errors=2,
    )
    assert result.status == "unestimable_source_errors"
    assert result.teacher_model is None and result.combined_model is None


def test_target_train_export_never_accepts_truth():
    from analysis.structure_evidence_increment_audit import sample_evidence_rows

    packet = {
        "logits": np.array([[2., 1.]]), "parcel_index": np.array([8]),
        "domain": np.array(["target"]),
    }
    rows = sample_evidence_rows(packet, "target_train", structure=[{}])
    assert "true_label" not in rows[0]
    packet["label"] = np.array([0])
    with pytest.raises(ValueError, match="Target Train"):
        sample_evidence_rows(packet, "target_train", structure=[{}])

