from __future__ import annotations

import json
import inspect
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.anchor_matching_calibration_audit import (
    CALIBRATION_PROTOCOL,
    P0R_PROTOCOL_VERSION,
    MonotonicCalibration,
    assess_anchor_reliability,
    assess_joint_reliability,
    build_ab_readiness,
    build_calibration_decision,
    bootstrap_metric_interval,
    class_matched_derangements,
    conditional_permute,
    circular_window_overlap,
    circular_window_iou,
    cross_view_morphology_records,
    deterministic_block_reorder,
    deterministic_view_seed,
    fit_monotonic_calibrator,
    freeze_morphology_thresholds,
    freeze_for_audit,
    morphology_proxy_labels,
    pair_views_by_parcel,
    perturb_windows,
    recompute_window_tokens_and_similarity,
    repeatability_rows,
    repeatability_baseline_rows,
    load_source_runtime,
    merge_sources,
    _joint_calibration_bootstrap,
    split_ids_are_disjoint,
)


def test_circular_overlap_handles_wrap_and_disjoint_windows():
    assert circular_window_overlap(60, 2, scale=8, grid_points=64)
    assert circular_window_overlap(0, 7, scale=8, grid_points=64)
    assert not circular_window_overlap(0, 8, scale=8, grid_points=64)


def test_circular_window_iou_uses_actual_extractor_indices():
    assert circular_window_iou(0, 0, 24, 64) == pytest.approx(1.0)
    assert circular_window_iou(56, 0, 24, 64) == pytest.approx(16 / 32)
    assert circular_window_iou(0, 24, 24, 64) == pytest.approx(0.0)


def test_same_class_derangements_never_pair_same_parcel_and_are_reproducible():
    parcels = np.arange(12)
    labels = np.repeat([0, 1, 2], 4)
    first = class_matched_derangements(parcels, labels, repeats=5, seed=7)
    second = class_matched_derangements(parcels, labels, repeats=5, seed=7)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (5, 12)
    for permutation in first:
        assert np.all(permutation != np.arange(12))
        np.testing.assert_array_equal(labels[permutation], labels)


def test_deterministic_view_seed_is_stable_and_view_specific():
    first = deterministic_view_seed(1, "france/30TXT/2017", 42, 0)
    assert first == deterministic_view_seed(1, "france/30TXT/2017", 42, 0)
    assert first != deterministic_view_seed(1, "france/30TXT/2017", 42, 1)


def test_pair_views_uses_domain_and_parcel_not_loader_order():
    view_a = {
        "domain": np.array(["A", "A"]),
        "parcel_index": np.array([2, 1]),
        "similarity": np.array([[[2.0]], [[1.0]]]),
    }
    view_b = {
        "domain": np.array(["A", "A"]),
        "parcel_index": np.array([1, 2]),
        "similarity": np.array([[[10.0]], [[20.0]]]),
    }
    left, right = pair_views_by_parcel(view_a, view_b)
    np.testing.assert_array_equal(left["parcel_index"], [1, 2])
    np.testing.assert_array_equal(right["parcel_index"], [1, 2])
    np.testing.assert_allclose(left["similarity"].ravel(), [1, 2])
    np.testing.assert_allclose(right["similarity"].ravel(), [10, 20])


def test_pair_views_preserves_non_sample_window_metadata():
    first = {
        "domain": np.array(["A", "A"]), "parcel_index": np.array([2, 1]),
        "similarity": np.ones((2, 3, 1)), "window_starts": np.array([0, 8, 16]),
    }
    second = {
        "domain": np.array(["A", "A"]), "parcel_index": np.array([1, 2]),
        "similarity": np.ones((2, 3, 1)), "window_starts": np.array([0, 8, 16]),
    }
    left, right = pair_views_by_parcel(first, second)
    np.testing.assert_array_equal(left["window_starts"], [0, 8, 16])
    np.testing.assert_array_equal(right["window_starts"], [0, 8, 16])


def test_split_ids_are_disjoint_rejects_source_leakage():
    assert split_ids_are_disjoint([1, 2], [3, 4])
    assert not split_ids_are_disjoint([1, 2], [2, 3])


def test_morphology_perturbations_preserve_shape_channels_and_value_multiset():
    windows = torch.arange(2 * 3 * 8 * 4).reshape(2, 3, 8, 4).float()
    reverse, reorder = perturb_windows(windows, seed=17)
    assert reverse.shape == reorder.shape == windows.shape
    for original, changed in ((windows, reverse), (windows, reorder)):
        np.testing.assert_allclose(
            original.sort(dim=2).values.numpy(),
            changed.sort(dim=2).values.numpy(),
        )
    torch.testing.assert_close(reverse, windows.flip(2))
    torch.testing.assert_close(reorder, deterministic_block_reorder(windows, 17))


def test_repeatability_uses_parcel_anchor_qmax_and_reports_both_directions():
    similarity_a = np.array([[[.1], [.9], [.2], [.1]]])
    similarity_b = np.array([[[.8], [.2], [.1], [.1]]])
    rows = repeatability_rows(
        similarity_a, similarity_b, parcels=np.array([7]),
        starts=np.array([0, 4, 8, 12]), scales=np.array([8, 8, 8, 8]),
        grid_points=64,
    )
    assert len(rows) == 2
    assert {row["direction"] for row in rows} == {"view0_to_view1", "view1_to_view0"}
    assert all(row["level"] == "parcel_anchor" for row in rows)
    assert rows[0]["q"] == pytest.approx(.9)
    assert rows[1]["q"] == pytest.approx(.8)
    assert all(row["label"] == 1 for row in rows)  # starts 4 and 0 overlap for W8


def test_morphology_thresholds_are_fit_on_train_normal_view_change_only():
    normal_train = np.array([
        [[.01, .02], [.03, .04]],
        [[.05, .06], [.07, .08]],
    ])
    thresholds = freeze_morphology_thresholds(normal_train, quantile=.95)
    np.testing.assert_allclose(thresholds, np.quantile(normal_train.reshape(-1, 2), .95, axis=0))
    original = np.array([[[.8, .7], [.6, .5]]])
    perturbed = original - np.array([[[.2, .01], [.3, .02]]])
    labels, delta = morphology_proxy_labels(original, perturbed, thresholds)
    np.testing.assert_allclose(delta, original - perturbed)
    assert labels.shape == original.shape
    assert labels[..., 0].all()


def test_cross_view_morphology_uses_opposite_view_label_at_same_window():
    q0 = np.array([[[.9], [.2]]])
    q1 = np.array([[[.7], [.6]]])
    perturbed0 = np.array([[[.1], [.1]]])
    perturbed1 = np.array([[[.65], [.1]]])
    records = cross_view_morphology_records(
        q0, q1, perturbed0, perturbed1, thresholds=np.array([.2]),
        parcels=np.array([11]), labels=np.array([3]), starts=np.array([0, 8]),
        perturbation="reverse",
    )
    forward = [row for row in records if row["direction"] == "view0_to_view1"]
    reverse = [row for row in records if row["direction"] == "view1_to_view0"]
    assert [row["q"] for row in forward] == [.9, .2]
    assert [row["proxy_label"] for row in forward] == [0, 1]
    assert [row["q"] for row in reverse] == [.7, .6]
    assert [row["proxy_label"] for row in reverse] == [1, 0]
    assert [row["window_index"] for row in forward] == [0, 1]


def test_conditional_permutation_stays_within_class_time_and_q_band():
    values = np.arange(24)
    strata = np.array([
        (clazz, time, band)
        for clazz in range(2) for time in range(3) for band in range(2)
        for _ in range(2)
    ], dtype=object)
    first = conditional_permute(values, strata, seed=9)
    second = conditional_permute(values, strata, seed=9)
    np.testing.assert_array_equal(first, second)
    for key in set(map(tuple, strata.tolist())):
        selected = np.array([tuple(row) == key for row in strata])
        np.testing.assert_array_equal(np.sort(first[selected]), np.sort(values[selected]))


def test_monotonic_calibrator_has_nonnegative_slope_and_train_only_fit():
    q_train = np.linspace(-1, 1, 200)
    y_train = (q_train > 0).astype(np.int64)
    groups = np.repeat(np.arange(100), 2)
    fit = fit_monotonic_calibrator(q_train, y_train, groups)
    assert isinstance(fit, MonotonicCalibration)
    assert fit.slope >= 0
    assert fit.fit_split == "source_train"
    probability = fit.predict_proba(np.array([-.5, .5]))
    assert probability[1] >= probability[0]


def test_monotonic_calibrator_refuses_single_class_or_insufficient_support():
    with pytest.raises(ValueError, match="both proxy classes"):
        fit_monotonic_calibrator(np.linspace(0, 1, 100), np.ones(100), np.arange(100))
    with pytest.raises(ValueError, match="minimum support"):
        fit_monotonic_calibrator(np.arange(18), np.tile([0, 1], 9), np.arange(18))


def test_bootstrap_is_grouped_by_parcel_and_reproducible():
    labels = np.tile([0, 1], 50)
    scores = labels * .8 + .1
    groups = np.repeat(np.arange(50), 2)
    first = bootstrap_metric_interval(labels, scores, groups, "auroc", repeats=40, seed=3)
    second = bootstrap_metric_interval(labels, scores, groups, "auroc", repeats=40, seed=3)
    assert first == second
    assert first[0] == pytest.approx(1.0)


def test_bootstrap_uses_full_parcel_population_and_stratifies_classes():
    labels = np.tile([0, 1], 60)
    scores = labels * .8 + .1
    parcels = np.repeat(np.arange(60), 2)
    classes = np.repeat(np.tile([0, 1, 2], 20), 2)
    result = bootstrap_metric_interval(
        labels, scores, parcels, "auroc", repeats=25, seed=4,
        group_classes=classes, return_metadata=True,
    )
    estimate, low, high, metadata = result
    assert estimate == pytest.approx(1.0)
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0)
    assert metadata["population_group_count"] == 60
    assert metadata["resampled_group_count"] == 60
    assert metadata["stratified"] is True
    assert metadata["invalid_replicates"] == 0


def test_shared_calibration_bootstrap_matches_sklearn_point_metrics_with_ties():
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([.1, .8, .4, .8, .6, .4])
    intervals, metadata = _joint_calibration_bootstrap(
        labels, scores, np.arange(len(labels)), seed=2,
        group_classes=np.array([0, 0, 1, 1, 2, 2]),
    )
    assert intervals["auroc"][0] == pytest.approx(roc_auc_score(labels, scores))
    assert intervals["auprc"][0] == pytest.approx(average_precision_score(labels, scores))
    assert metadata["population_group_count"] == len(labels)


@pytest.mark.parametrize(
    ("repeatability", "morphology", "expected"),
    [
        ("stable", "stable", "calibratable"),
        ("stable", "unsupported", "uncertain"),
        ("stable", "failed", "uncertain"),
        ("failed", "failed", "not_calibratable"),
        ("unsupported", "failed", "uncertain"),
    ],
)
def test_joint_status_follows_preregistered_truth_table(repeatability, morphology, expected):
    assert assess_joint_reliability(repeatability, morphology) == expected


def test_protocol_constants_are_fixed_before_source_val_results():
    assert P0R_PROTOCOL_VERSION == "p0r-v1"
    assert CALIBRATION_PROTOCOL["normal_change_quantile"] == .95
    assert CALIBRATION_PROTOCOL["minimum_total"] == 50
    assert CALIBRATION_PROTOCOL["minimum_per_class"] == 10
    assert CALIBRATION_PROTOCOL["bootstrap_repeats"] == 200
    assert CALIBRATION_PROTOCOL["bootstrap_confidence"] == .95
    assert CALIBRATION_PROTOCOL["same_class_shuffle_repeats"] == 20
    assert CALIBRATION_PROTOCOL["conditional_permutation_repeats"] == 200
    assert "bootstrap_max_groups" not in CALIBRATION_PROTOCOL


def test_reliability_requires_controls_and_emits_reason_codes():
    passed = assess_anchor_reliability(
        repeatability={
            "supported": True, "monotonic_generalizes": True,
            "baseline_increment_supported": True, "paired_shuffle_effect_supported": True,
        },
        morphology={
            "supported": True, "cross_view_generalizes": True,
            "exceeds_normal_variation": True, "conditional_control_supported": True,
            "perturbations": {"reverse": "pass", "block_reorder": "pass"},
        },
    )
    assert passed["status"] == "calibratable"
    assert passed["reason_codes"] == []
    failed = assess_anchor_reliability(
        repeatability={
            "supported": True, "monotonic_generalizes": False,
            "baseline_increment_supported": False, "paired_shuffle_effect_supported": False,
        },
        morphology={
            "supported": True, "cross_view_generalizes": False,
            "exceeds_normal_variation": False, "conditional_control_supported": False,
            "perturbations": {"reverse": "fail", "block_reorder": "fail"},
        },
    )
    assert failed["status"] == "not_calibratable"
    assert "repeatability_no_baseline_increment" in failed["reason_codes"]
    assert "morphology_not_above_normal_variation" in failed["reason_codes"]


def test_qmax_increment_is_measured_against_same_class_position_population():
    def rows(size):
        output = []
        for index in range(size):
            q = (index % 10) / 9
            output.append({
                "anchor": 0, "direction": "view0_to_view1",
                "class_id": index % 2, "source_window": index % 4,
                "parcel_index": index, "q": q, "label": int(q > .5),
            })
        return output
    result = repeatability_baseline_rows(rows(120), rows(80), 0, seed=5)
    by_name = {row["model"]: row for row in result}
    assert set(by_name) == {"constant", "position_only", "class_position", "class_position_qmax"}
    assert by_name["class_position_qmax"]["qmax_coefficient"] >= 0
    assert by_name["class_position_qmax"]["auprc"] > by_name["class_position"]["auprc"]


def test_runtime_constructs_only_source_splits_for_p0r():
    source = inspect.getsource(load_source_runtime)
    assert "build_abc_datasets" not in source
    assert "config.target" not in source
    assert '"source_train"' in source and '"source_val"' in source
    assert "return model, config" in source


def test_machine_readable_decision_and_ab_readiness_do_not_invent_defaults():
    anchors = [
        {"anchor": 0, "status": "calibratable", "reason_codes": [],
         "repeatability": {"slope": 2.0}, "morphology": {"slope": 1.0},
         "score_ranges": {"parcel_qmax": [-.2, .9], "window_q": [-.8, .8]}},
        {"anchor": 1, "status": "uncertain", "reason_codes": ["proxy_conflict"],
         "repeatability": None, "morphology": None,
         "score_ranges": {"parcel_qmax": [-.1, .7], "window_q": [-.7, .7]}},
    ]
    decision = build_calibration_decision(
        source="FR1", checkpoint_sha="abc", train_fingerprint="train",
        val_fingerprint="val", anchors=anchors, seed=1,
    )
    assert decision["protocol_version"] == "p0r-v1"
    assert decision["anchors"][1]["repeatability"] is None
    assert decision["anchors"][1]["reason_codes"] == ["proxy_conflict"]
    readiness = build_ab_readiness({"FR1": decision, "FR2": decision})
    assert readiness["status"] == "ready_with_limitations"
    assert readiness["credible_occurrence_interpretation"] is False
    blocked_decision = dict(decision, anchors=[dict(anchors[1])])
    assert build_ab_readiness({"FR1": blocked_decision})["status"] == "blocked"
    incomplete = build_ab_readiness({"FR1": decision}, expected_sources=("FR1", "FR2"))
    assert incomplete["status"] == "blocked"
    assert incomplete["missing_sources"] == ["FR2"]


def test_launcher_dry_run_has_four_sources_and_no_git_or_training(capsys):
    from scripts.run_anchor_matching_calibration_seed1 import build_parser, planned_sources, run_scheduler

    tmp_path = Path(".p0_launcher_test")
    shutil.rmtree(tmp_path, ignore_errors=True)
    args = build_parser().parse_args([
        "dry-run", "--data-root", str(tmp_path / "missing_data"),
        "--source-root", str(tmp_path / "missing_checkpoints"),
        "--output-root", str(tmp_path / "output"),
        "--gpus", "2", "3",
    ])
    plans = planned_sources(args)
    assert [plan.alias for plan in plans] == ["AT1", "FR1", "FR2", "DK1"]
    run_scheduler(args)
    output = capsys.readouterr().out
    assert output.count("P0_PLAN|") == 4
    assert "protocol=p0r-v1" in output
    assert "git " not in output.lower()
    assert "train.py" not in output
    assert "gpu=2" in output and "gpu=3" in output
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_launcher_resume_requires_matching_manifest():
    from scripts.run_anchor_matching_calibration_seed1 import artifact_complete, _mark_in_progress

    tmp_path = Path(".p0_artifact_test")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "AT1"
    output.mkdir()
    for name in (
        "matching_distribution.csv", "view_repeatability.csv",
        "morphology_counterfactual.csv", "calibration_candidates.csv",
        "anchor_reliability_summary.csv", "calibration_decision.json",
        "repeatability_controls.csv", "repeatability_baselines.csv",
        "morphology_conditional_controls.csv", "per_class_evidence.csv",
        "bootstrap_protocol.json", "checkpoint_manifest.json",
    ):
        (output / name).write_text("x\n", encoding="utf-8")
    manifest = {
        "checkpoint_sha256": __import__("hashlib").sha256(b"checkpoint").hexdigest(),
        "settings_fingerprint": "abc", "status": "complete",
        "target_data_used": False, "training_performed": False,
        "protocol_version": "p0r-v1",
    }
    (output / "checkpoint_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert artifact_complete(output, checkpoint, "abc")
    assert not artifact_complete(output, checkpoint, "different")
    _mark_in_progress(output, checkpoint, "abc")
    assert not artifact_complete(output, checkpoint, "abc")
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_summary_rejects_stale_outputs_with_in_progress_manifest():
    root = Path(".p0_summary_test")
    shutil.rmtree(root, ignore_errors=True)
    source = root / "AT1"; source.mkdir(parents=True)
    (source / "anchor_reliability_summary.csv").write_text(
        "anchor,joint_status\n0,calibratable\n", encoding="utf-8",
    )
    (source / "checkpoint_manifest.json").write_text(json.dumps({
        "status": "in_progress", "protocol_version": "p0r-v1",
        "source_alias": "AT1", "checkpoint_sha256": "sha",
    }), encoding="utf-8")
    (source / "calibration_decision.json").write_text(json.dumps({
        "protocol_version": "p0r-v1", "source": "AT1",
        "checkpoint_sha256": "sha", "anchors": [],
    }), encoding="utf-8")
    merge_sources(root)
    readiness = json.loads((root / "AB_readiness.json").read_text(encoding="utf-8"))
    assert readiness["status"] == "blocked"
    assert "AT1" in readiness["missing_sources"]
    shutil.rmtree(root, ignore_errors=True)


def test_seeded_dataset_access_restores_rng_state():
    from analysis.anchor_matching_calibration_audit import seeded_dataset_item

    class RandomDataset:
        def __getitem__(self, index):
            return random.random(), np.random.random(), torch.rand(())

    random.seed(10); np.random.seed(10); torch.manual_seed(10)
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    first = seeded_dataset_item(RandomDataset(), 0, seed=33)
    after = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    second = seeded_dataset_item(RandomDataset(), 0, seed=33)
    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])
    assert before[0] == after[0]
    assert all(np.array_equal(a, b) for a, b in zip(before[1], after[1]))
    torch.testing.assert_close(before[2], after[2])


def test_frozen_recompute_has_expected_shape_and_no_trainable_parameters():
    class TokenGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.projection = torch.nn.Linear(3, 5)

        def forward(self, windows, return_encoded_components=False):
            token = self.projection(windows.mean(2))
            return {"shape_token": token} if return_encoded_components else token

    class Dictionary(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchors = torch.nn.Parameter(torch.randn(4, 5))

        def compute_similarity(self, tokens):
            return torch.nn.functional.normalize(tokens, dim=-1) @ torch.nn.functional.normalize(self.anchors, dim=-1).T

    class Branch(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.token_generator = TokenGenerator(); self.shapelet_dictionary = Dictionary()

    branch = freeze_for_audit(Branch())
    with torch.inference_mode():
        tokens, similarity = recompute_window_tokens_and_similarity(
            branch, torch.randn(2, 8, 24, 3),
        )
    assert tokens.shape == (2, 8, 5)
    assert similarity.shape == (2, 8, 4)
    assert not branch.training
    assert all(not parameter.requires_grad for parameter in branch.parameters())


def test_missing_checkpoint_fails_without_training_fallback():
    from analysis.anchor_matching_calibration_audit import load_source_runtime

    with pytest.raises(FileNotFoundError, match="missing Source checkpoint"):
        load_source_runtime("definitely_missing_p0_checkpoint.pt", ".", "cpu")
