import json
from pathlib import Path

import numpy as np
import pytest
import torch

import analysis.structure_e123_audit as e123
from models.stclassifier import PseStructureProtoLTae


def test_feature_groups_use_block_major_anchor_phase_indices():
    assert e123.anchor_feature_indices(3, "SC") == [3, 19]
    assert e123.anchor_feature_indices(3, "SC_K1") == [3, 19, 35, 51]
    assert e123.anchor_feature_indices(3, "SC_K2") == [3, 19, 67, 83]
    assert e123.anchor_feature_indices(3, "SCP") == [3, 19, 35, 51, 67, 83]


def test_compact_arrays_drop_dense_similarity_and_target_train_truth():
    arrays = {
        "parcel_index": np.arange(2), "response": np.zeros((2, 96)),
        "events": np.zeros((2, 16, 6)), "logits": np.zeros((2, 3)),
        "confidence": np.ones(2), "prediction": np.zeros(2),
        "similarity": np.ones((2, 64, 16)), "pooling_weight": np.ones((2, 64, 16)),
        "label": np.asarray([1, 2]), "domain": np.asarray(["x", "x"]),
        "representative_curves": np.zeros((1, 8, 128)),
        "representative_anchor": np.asarray([0]),
    }
    compact = e123.compact_feature_arrays(arrays, "teacher", "target_train")
    assert "similarity" not in compact and "pooling_weight" not in compact
    assert "response" not in compact and "logits" not in compact
    assert "label" not in compact
    assert compact["anchor_reliability"].shape == (2,)
    assert "representative_curves" not in compact
    student_eval = e123.compact_feature_arrays(arrays, "student", "target_val")
    assert "representative_curves" in student_eval


def test_cache_alignment_rejects_student_teacher_key_mismatch(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    base = {
        "parcel_index": np.asarray([10, 11]),
        "domain": np.asarray(["source", "source"]),
        "events": np.zeros((2, 16, 6)),
        "prediction": np.zeros(2, dtype=int),
        "confidence": np.ones(2),
        "anchor_reliability": np.ones(2),
    }
    for role in ("student", "teacher"):
        for split in e123.ALLOWED_SPLITS:
            packet = dict(base)
            if split.startswith("target"):
                packet["domain"] = np.asarray(["target", "target"])
            if split != "target_train":
                packet["label"] = np.zeros(2, dtype=int)
            np.savez(feature_dir / f"{role}_{split}.npz", **packet)
    packet = dict(base)
    packet["parcel_index"] = np.asarray([10, 12])
    packet["label"] = np.zeros(2, dtype=int)
    np.savez(feature_dir / "teacher_source_val.npz", **packet)
    with pytest.raises(ValueError, match="Student/Teacher key mismatch"):
        e123.validate_cache_protocol(tmp_path)


def test_reused_abc_cache_selects_its_original_split_manifest(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for role in ("student", "teacher"):
        for split in e123.ALLOWED_SPLITS:
            (feature_dir / f"{role}_{split}.npz").touch()
    original = {
        split: [{"domain": "source" if split.startswith("source") else "target", "parcel_index": index}]
        for index, split in enumerate(e123.ALLOWED_SPLITS)
    }
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps({
        "checkpoint_sha256": "same-checkpoint",
        "split_manifest": original,
        "protocol": {"name": "official"},
    }))
    packet = e123.reused_abc_protocol(tmp_path, "same-checkpoint")
    assert packet["split_manifest"] == original
    with pytest.raises(ValueError, match="checkpoint hash"):
        e123.reused_abc_protocol(tmp_path, "different-checkpoint")


def test_reused_cache_shift_manifest_requires_one_coordinate_per_role(tmp_path):
    path = tmp_path / "feature_cache_manifest.json"
    path.write_text(json.dumps({
        "applied_shifts": {
            "student": {split: 0 for split in e123.ALLOWED_SPLITS},
            "teacher": {split: (3 if split == "target_val" else 2) for split in e123.ALLOWED_SPLITS},
        }
    }))
    with pytest.raises(ValueError, match="Teacher temporal coordinate"):
        e123.validate_shift_manifest(tmp_path, expected_teacher_shift=2)


def test_e1_branch_ablation_and_controlled_stability_use_full_pse_windows():
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation="phase_moment", shape_injection="current_query",
    ).eval()
    curves = np.random.default_rng(9).normal(size=(3, 24, 8)).astype(np.float32)
    dependency = e123.branch_dependency(model, curves, anchor_id=2)
    stability = e123.controlled_stability(model, curves, anchor_id=2, seed=1)
    assert set(dependency) == {"raw", "difference", "mean", "std"}
    assert set(stability) == {"noise_0.05std", "amplitude_x1.05", "position_roll_1"}
    assert all(np.isfinite(value) for value in (*dependency.values(), *stability.values()))


def test_probe_comparison_uses_same_samples_and_detects_phase_increment():
    rng = np.random.default_rng(4)
    labels = np.repeat([0, 1], 40)
    events = rng.normal(0, .2, (80, 16, 6))
    events[labels == 0, 2, 2] += 2
    events[labels == 1, 2, 2] -= 2
    train = np.r_[0:24, 40:64]
    source_val = np.r_[24:32, 64:72]
    target_val = np.r_[32:40, 72:80]
    rows = e123.compare_anchor_probe_groups(
        events[train], labels[train], events[source_val], labels[source_val],
        events[target_val], labels[target_val], 0, 1, min_support=4,
    )
    anchor = {row["feature_group"]: row for row in rows if row["anchor_id"] == 2}
    assert anchor["SC"]["source_val_support"] == anchor["SCP"]["source_val_support"]
    assert anchor["SCP"]["target_balanced_accuracy"] > anchor["SC"]["target_balanced_accuracy"]


def test_missing_pair_support_emits_na_probe_rows():
    events = np.zeros((8, 16, 6))
    labels = np.zeros(8, dtype=int)
    rows = e123.compare_anchor_probe_groups(
        events, labels, events, labels, events, labels, 0, 1, min_support=2,
    )
    assert len(rows) == 16 * 4
    assert all(row["status"] == "N/A" for row in rows)
    assert all(row["support_c"] == 0 for row in rows)


def test_target_missing_class_marks_target_metric_na_without_discarding_source_probe():
    rng = np.random.default_rng(8)
    source_labels = np.repeat([0, 1], 12)
    source = rng.normal(size=(24, 16, 6))
    target = rng.normal(size=(8, 16, 6))
    rows = e123.compare_anchor_probe_groups(
        source, source_labels, source, source_labels,
        target, np.zeros(8, dtype=int), 0, 1, min_support=4,
    )
    assert all(row["status"] == "ok" for row in rows)
    assert all(np.isfinite(row["source_val_balanced_accuracy"]) for row in rows)
    assert all(np.isnan(row["target_balanced_accuracy"]) for row in rows)
    assert all(row["target_status"] == "N/A_missing_class" for row in rows)


def test_within_prediction_metrics_never_mix_predicted_classes():
    labels = np.asarray([2, 2, 1, 0, 2, 1])
    prediction = np.asarray([2, 2, 2, 2, 1, 1])
    scores = {
        "confidence": np.asarray([.99, .95, .93, .91, .8, .7]),
        "structure_margin": np.asarray([2., 1., -2., -1., .5, -.5]),
        "anchor_reliability": np.asarray([.9, .8, .2, .3, .5, .4]),
    }
    rows = e123.within_prediction_metrics(labels, prediction, scores, ["a", "spring_barley", "winter_wheat"])
    winter = [row for row in rows if row["predicted_class"] == "winter_wheat"]
    assert all(row["support"] == 4 for row in winter)
    assert all(row["correct_support"] == 2 for row in winter)


def test_positive_evidence_requires_multiple_competitors():
    support = np.asarray([
        [0.1, 2.0, -1.0],
        [0.1, 2.0, np.nan],
    ])
    rows = e123.positive_evidence_rows(
        support, np.asarray([2, 2]), np.asarray([1, 1]),
        np.asarray([.7, .7]), np.asarray([10, 11]), "target", ["a", "b", "c"],
    )
    assert rows[0]["eligible_multi_competitor"] is True
    assert rows[1]["eligible_multi_competitor"] is False


def test_rejection_tradeoff_reports_weak_class_rejection_and_correct_class_retention():
    labels = np.asarray([1, 2, 2, 1])
    prediction = np.asarray([2, 2, 2, 1])
    scores = {"reliability": np.asarray([.1, .9, .8, .7])}
    rows = e123.rejection_tradeoff(
        labels, prediction, scores, ["other", "spring_barley", "winter_wheat"], points=3,
    )
    winter_full = next(
        row for row in rows
        if row["predicted_class"] == "winter_wheat" and row["accepted"] == 3
    )
    assert winter_full["spring_error_eligible"] == 1
    assert winter_full["correct_winter_eligible"] == 2
    assert winter_full["spring_error_rejection_rate"] == pytest.approx(0.)
    assert winter_full["correct_winter_retention_rate"] == pytest.approx(1.)


def test_teacher_calibration_fallback_fits_only_source_and_uses_unlabeled_target_range():
    rng = np.random.default_rng(12)
    labels = np.repeat([0, 1], 24)
    events = rng.normal(0, .1, (48, 16, 6))
    events[labels == 0, 4, 0] += 2
    events[labels == 1, 4, 0] -= 2
    cache = {
        "source_train": {"events": events, "label": labels, "prediction": labels},
        "source_val": {"events": events + .01, "label": labels, "prediction": labels},
        "target_train": {"events": events.copy(), "prediction": labels},
    }
    calibration = e123.fit_teacher_calibration_from_cache(
        cache, "FR1_FR2", ["winter_barley", "winter_wheat"], seed=1,
    )
    assert calibration["fit_split"] == "source_train"
    assert calibration["threshold_fit_split"] == "source_val"
    assert calibration["selection_uses_target_truth"] is False
    assert calibration["candidates"]
    assert all(row["role"] == "teacher" for row in calibration["candidates"])


def test_stage_contracts_and_scheduler_have_no_training_jobs(tmp_path):
    import scripts.run_structure_e123_seed1 as scheduler

    jobs = scheduler.build_job_graph()
    assert len(jobs) == 23
    assert scheduler.INITIAL_GPU_ASSIGNMENT == {
        "extract:E:FR1_FR2": "0", "extract:E:FR2_DK1": "1",
        "extract:G:FR1_FR2": "2", "extract:G:FR2_DK1": "3",
    }
    args = scheduler.build_parser().parse_args([
        "dry-run", "--experiment-root", str(tmp_path),
    ])
    assert all("train.py" not in " ".join(scheduler.job_command(job, args)) for job in jobs)
    assert set(e123.required_stage_artifacts("E3")) == {
        "E3_all_teacher_scores.csv", "E3_within_prediction_reliability.csv",
        "E3_rejection_tradeoff.csv", "E3_score_components.csv",
        "E3_negative_evidence.csv", "E3_positive_evidence.csv", "E3_report.md",
    }


def test_summary_decision_is_one_of_four_permitted_values():
    assert e123.choose_e123_decision({
        "anchor_problem": False, "phase_transfer": False,
        "negative_reliability": False, "positive_reliability": False,
    }) == "insufficient_evidence"
    assert e123.choose_e123_decision({
        "anchor_problem": False, "phase_transfer": True,
        "negative_reliability": False, "positive_reliability": False,
    }) == "anchor_phase_semantics"
