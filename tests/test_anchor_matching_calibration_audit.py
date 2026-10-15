from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.anchor_matching_calibration_audit import (
    CALIBRATION_PROTOCOL,
    MonotonicCalibration,
    assess_joint_reliability,
    bootstrap_metric_interval,
    circular_window_overlap,
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
    split_ids_are_disjoint,
)


def test_circular_overlap_handles_wrap_and_disjoint_windows():
    assert circular_window_overlap(60, 2, scale=8, grid_points=64)
    assert circular_window_overlap(0, 7, scale=8, grid_points=64)
    assert not circular_window_overlap(0, 8, scale=8, grid_points=64)


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
    assert CALIBRATION_PROTOCOL == {
        "normal_change_quantile": 0.95,
        "minimum_total": 50,
        "minimum_per_class": 10,
        "bootstrap_repeats": 200,
        "bootstrap_max_groups": 2000,
        "bootstrap_confidence": 0.95,
        "stable_auroc_lower_bound": 0.5,
    }


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
    assert "git " not in output.lower()
    assert "train.py" not in output
    assert "gpu=2" in output and "gpu=3" in output
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_launcher_resume_requires_matching_manifest():
    from scripts.run_anchor_matching_calibration_seed1 import artifact_complete

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
        "anchor_reliability_summary.csv", "checkpoint_manifest.json",
    ):
        (output / name).write_text("x\n", encoding="utf-8")
    manifest = {
        "checkpoint_sha256": __import__("hashlib").sha256(b"checkpoint").hexdigest(),
        "settings_fingerprint": "abc", "status": "complete",
        "target_data_used": False, "training_performed": False,
    }
    (output / "checkpoint_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert artifact_complete(output, checkpoint, "abc")
    assert not artifact_complete(output, checkpoint, "different")
    shutil.rmtree(tmp_path, ignore_errors=True)


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
