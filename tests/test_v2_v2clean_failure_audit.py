from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.v2_v2clean_failure_audit import (
    audit_representation,
    extract_representations,
    leave_one_out_centroid_predictions,
    integer_day_shift,
    pseudo_quality,
    parse_training_timeline,
    source_zscore,
)


def test_source_zscore_uses_only_source_train_statistics():
    source = np.array([[0., 2.], [2., 4.]])
    target = np.array([[100., 200.]])
    values, mean, std = source_zscore(source, {"source": source, "target": target})
    np.testing.assert_allclose(mean, [1., 3.])
    np.testing.assert_allclose(std, [1., 1.])
    np.testing.assert_allclose(values["source"], [[-1., -1.], [1., 1.]])
    np.testing.assert_allclose(values["target"], [[99., 197.]])


def test_target_oracle_centroid_is_leave_one_out():
    features = np.array([[0.], [2.], [8.], [10.]])
    labels = np.array([0, 0, 1, 1])
    prediction = leave_one_out_centroid_predictions(features, labels, 2)
    np.testing.assert_array_equal(prediction, labels)


def test_geometry_detects_wrong_source_center_and_larger_target_radius():
    source_train = np.array([[-1.], [1.], [9.], [11.]])
    source_labels = np.array([0, 0, 1, 1])
    source_val = source_train.copy()
    target_val = np.array([[7.], [11.], [8.], [12.]])
    target_labels = np.array([0, 0, 1, 1])
    result = audit_representation(
        source_train, source_labels, source_val, source_labels,
        target_val, target_labels, ["a", "b"],
    )
    class_a = next(row for row in result["class_rows"] if row["class"] == "a")
    assert class_a["cross_domain_margin"] < 0
    assert class_a["intra_ratio"] > 1
    assert class_a["nearest_source_class_for_target_c"] == "b"
    assert result["summary"]["negative_margin_class_count"] >= 1


def test_response_and_ordered_representation_shapes_are_exact():
    output = {
        "shapelet_similarity": torch.randn(3, 8, 16),
        "shapelet_strength": torch.randn(3, 16),
        "shapelet_concentration": torch.randn(3, 16),
        "shapelet_response": torch.randn(3, 32),
        "instance_feature": torch.randn(3, 128),
    }
    output["shapelet_response"] = torch.cat((
        output["shapelet_strength"], output["shapelet_concentration"],
    ), dim=-1)
    representations = extract_representations(output)
    assert representations["ordered128"].shape == (3, 128)
    assert representations["response32"].shape == (3, 32)
    torch.testing.assert_close(
        representations["response32"],
        torch.cat((representations["strength16"], representations["concentration16"]), -1),
    )


def test_source_to_target_centroid_classifier_and_pseudo_quality():
    source_train = np.array([[0.], [.2], [10.], [10.2]])
    labels = np.array([0, 0, 1, 1])
    result = audit_representation(
        source_train, labels, source_train, labels,
        np.array([[.1], [10.1]]), np.array([0, 1]), ["a", "b"],
    )
    assert result["summary"]["source_to_target_centroid_f1"] == 1.

    logits = torch.tensor([[5., 0.], [4., 0.], [0., 5.], [0., 5.]])
    truth = torch.tensor([0, 1, 1, 1])
    pseudo = pseudo_quality(logits, truth, ["a", "b"], threshold=.9)
    assert pseudo["summary"]["pseudo_coverage"] == pytest.approx(1.)
    assert pseudo["summary"]["accepted_pseudo_accuracy"] == pytest.approx(.75)
    assert pseudo["true_rows"][1]["accepted_count"] == 3
    assert pseudo["pred_rows"][0]["precision"] == pytest.approx(.5)


def test_timeline_labels_shape_agreement_without_claiming_gt_accuracy(tmp_path):
    log = tmp_path / "task.log"
    log.write_text(
        "[UDA START] FR1 -> FR2 GPU1\n"
        "INITIAL_SHIFT|shift_days=-5\n"
        "EPOCH_SHIFT|epoch=0|target_to_source_days=-4\n"
        "STRUCTURE_PROTO_DA|loss_pseudo_target=0.4\n"
        "SHAPE_AUX_EPOCH|epoch=0|source_accuracy=0.8|target_pseudo_accuracy=0.7\n"
        "Validation result: loss=1, acc=1, f1=0.61\n"
        "EPOCH_SHIFT|epoch=1|target_to_source_days=-3\n"
        "Validation result: loss=1, acc=1, f1=0.72\n",
        encoding="utf-8",
    )
    rows = parse_training_timeline(log, "v2")
    assert rows[0]["initial_shift"] == pytest.approx(-5.)
    assert rows[0]["shape_vs_pseudo_accuracy"] == pytest.approx(.7)
    assert "target_pseudo_accuracy" not in rows[0]
    assert rows[1]["best_validation_epoch"] is True


def test_logged_shift_is_restored_as_integer_embedding_offset():
    positions = torch.tensor([[10, 20]], dtype=torch.long)
    shift = integer_day_shift(-5.0)
    shifted = positions + shift
    assert isinstance(shift, int)
    assert shifted.dtype == torch.long
    with pytest.raises(ValueError, match="integer day"):
        integer_day_shift(-5.25)


def test_launcher_has_exact_four_jobs_and_no_training_commands():
    launcher = Path("scripts/run_v2_v2clean_failure_audit_4gpu_seed1.sh")
    source = launcher.read_text(encoding="utf-8")
    assert source.count("FAILURE_AUDIT_PLAN|") == 1
    assert 'run_job "$GPU0" v2 FR1_FR2' in source
    assert 'run_job "$GPU1" v2 DK1_AT1' in source
    assert 'run_job "$GPU2" v2clean FR1_FR2' in source
    assert 'run_job "$GPU3" v2clean DK1_AT1' in source
    assert "DRY_RUN" in source
    assert "train.py" not in source


def test_audit_source_contains_no_training_operations():
    source = Path("analysis/v2_v2clean_failure_audit.py").read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "optimizer.step" not in source
