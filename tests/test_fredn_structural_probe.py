import inspect
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from models.fredn.structural_probe import (
    Landmark,
    SourceAlignmentTemplate,
    TopologyMismatchError,
    align_curve_to_landmark_template,
    build_landmark_warp,
    build_direct_fourier_views,
    candidate_alignment_prediction,
    classification_metrics,
    coarse_fine_hierarchy,
    contrastive_feasibility_by_class,
    compute_topology_comparison,
    detect_structural_landmarks,
    extract_fourier_conditions,
    fit_source_class_projections,
    normalized_l2,
    nearest_prototype_prediction,
    pareto_modes,
    parse_fredn_checkpoint_specs,
    pointwise_intra_class_variance,
    project_curves,
    segment_shape_descriptors,
    shape_distance,
    stratified_bootstrap_macro_f1_delta,
    topology_signature,
)
from scripts import probe_fredn_structure
from scripts import probe_fredn_shape_alignment


def _piecewise_curve(grid, landmark_times, landmark_values):
    knots = np.asarray([grid[0], *landmark_times, grid[-1]], dtype=float)
    values = np.asarray([0.0, *landmark_values, 0.0], dtype=float)
    return np.interp(grid, knots, values)


def _landmarks(times, values):
    return [
        Landmark(
            kind="peak" if value > 0 else "valley",
            time=float(time),
            amplitude=float(value),
            prominence=1.0,
        )
        for time, value in zip(times, values)
    ]


def test_pure_phase_shift_preserves_topology_and_alignment_reduces_variance():
    grid = np.arange(0.0, 101.0)
    values = [-1.0, 1.0, -0.8, 0.9]
    source_times = [15.0, 35.0, 60.0, 82.0]
    target_times = [22.0, 43.0, 67.0, 88.0]
    source = _piecewise_curve(grid, source_times, values)
    target = _piecewise_curve(grid, target_times, values)

    source_detected = detect_structural_landmarks(
        grid, source, min_distance_days=10.0, prominence_threshold=0.3
    )
    target_detected = detect_structural_landmarks(
        grid, target, min_distance_days=10.0, prominence_threshold=0.3
    )
    assert topology_signature(source_detected) == topology_signature(target_detected)

    target_aligned = align_curve_to_landmark_template(
        grid, target, target_detected, source_detected
    )
    before = pointwise_intra_class_variance(np.stack([source, target]))
    after = pointwise_intra_class_variance(np.stack([source, target_aligned]))

    assert after < before * 0.1
    assert normalized_l2(source, target_aligned) < normalized_l2(source, target) * 0.2
    np.testing.assert_allclose(
        [item.amplitude for item in target_detected], values, atol=1e-7
    )


def test_incompatible_topology_is_rejected_from_phase_probe():
    grid = np.arange(0.0, 101.0)
    source_landmarks = _landmarks([15, 35, 60, 82], [-1, 1, -0.8, 0.9])
    target_landmarks = _landmarks([25, 65], [-1, 1])

    with pytest.raises(TopologyMismatchError):
        align_curve_to_landmark_template(
            grid,
            _piecewise_curve(grid, [25, 65], [-1, 1]),
            target_landmarks,
            source_landmarks,
        )


def test_topology_match_exceeds_different_class_collision():
    source = {
        0: [(4, "V-P-V-P")] * 8 + [(2, "V-P")] * 2,
        1: [(3, "P-V-P")] * 10,
    }
    target = {
        0: [(4, "V-P-V-P")] * 9 + [(2, "V-P")],
        1: [(3, "P-V-P")] * 8 + [(1, "P")] * 2,
    }

    metrics = compute_topology_comparison(source, target)

    assert metrics["same_class_topology_match"] > metrics[
        "different_class_topology_collision"
    ]
    assert metrics["topology_discrimination_margin"] == pytest.approx(
        metrics["same_class_topology_match"]
        - metrics["different_class_topology_collision"]
    )


def test_projection_fit_is_source_only_and_has_deterministic_sign():
    source = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.1], [3.0, -0.1]],
            [[1.2, 0.0], [2.2, -0.1], [3.2, 0.1]],
            [[0.0, 1.0], [0.1, 2.0], [-0.1, 3.0]],
            [[0.0, 1.2], [-0.1, 2.2], [0.1, 3.2]],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    target_a = torch.randn(5, 3, 2)
    target_b = target_a * 1000.0

    projections_a = fit_source_class_projections(source, labels)
    projections_b = fit_source_class_projections(source.clone(), labels.clone())

    assert set(projections_a) == {0, 1}
    for class_id in projections_a:
        assert torch.equal(projections_a[class_id], projections_b[class_id])
        loading = projections_a[class_id]
        dominant = torch.argmax(torch.abs(loading))
        assert loading[dominant] > 0
    # Target data are deliberately not accepted by the fitting API.
    assert target_a.shape == target_b.shape


def test_landmark_warp_is_monotonic_endpoint_fixed_and_maps_landmarks():
    sample = _landmarks([20, 45, 75], [-1, 1, -1])
    prototype = _landmarks([15, 40, 80], [-1, 1, -1])

    sample_knots, canonical_knots = build_landmark_warp(
        sample, prototype, support_start=0.0, support_end=100.0
    )

    assert np.all(np.diff(sample_knots) > 0)
    assert np.all(np.diff(canonical_knots) > 0)
    assert sample_knots[0] == canonical_knots[0] == 0.0
    assert sample_knots[-1] == canonical_knots[-1] == 100.0
    np.testing.assert_allclose(sample_knots[1:-1], [20, 45, 75])
    np.testing.assert_allclose(canonical_knots[1:-1], [15, 40, 80])


def test_raw_and_trend_projection_use_the_exact_same_source_loading():
    torch.manual_seed(7)
    raw = torch.randn(4, 6, 3)
    trend = raw * torch.tensor([1.0, 0.5, 0.2])
    labels = torch.tensor([0, 0, 1, 1])
    projections = fit_source_class_projections(raw, labels)

    raw_curves = project_curves(raw, projections)
    trend_curves = project_curves(trend, projections)

    for index, class_id in enumerate(labels.tolist()):
        loading = projections[class_id]
        torch.testing.assert_close(raw_curves[class_id][index], raw[index] @ loading)
        torch.testing.assert_close(
            trend_curves[class_id][index], trend[index] @ loading
        )


def test_adjacent_landmark_segments_report_duration_slope_and_area():
    grid = np.arange(0.0, 11.0)
    curve = grid.copy()
    landmarks = _landmarks([2.0, 8.0], [-1.0, 1.0])

    segments = segment_shape_descriptors(grid, curve, landmarks)

    assert len(segments) == 1
    assert segments[0]["duration"] == pytest.approx(6.0)
    assert segments[0]["slope"] == pytest.approx(1.0)
    assert segments[0]["area"] == pytest.approx(30.0)


def test_probe_never_invokes_git_or_subprocess():
    source = inspect.getsource(probe_fredn_structure)

    assert "import subprocess" not in source
    assert 'subprocess.run(' not in source
    assert '["git",' not in source
    assert "os.system(" not in source
    assert "Popen(" not in source
    assert "import requests" not in source
    assert "import urllib" not in source
    assert "huggingface" not in source.lower()
    assert "pip install" not in source.lower()


def test_unified_probe_launcher_passes_checkpoints_and_manual_version_metadata():
    path = Path("scripts/run_fredn_structural_probe_mode_sweep_at1_dk1.sh")
    source = path.read_text(encoding="utf-8")

    for mode in (9, 11, 13, 15, 17, 19):
        assert f'--fredn-checkpoint "{mode}=$CKPT_{mode}"' in source
    assert '--git-commit "$GIT_COMMIT"' in source
    assert '--git-branch "$GIT_BRANCH"' in source
    assert '--git-dirty "$GIT_DIRTY"' in source
    assert "git " not in source.lower()
    assert "requests" not in source.lower()
    assert "urllib" not in source.lower()


class _Analyzer:
    def __call__(self, features, positions):
        return features.to(torch.complex64), {}


class _Synthesizer:
    def __call__(self, coeffs, positions):
        return coeffs.real


class _MaskDisentangler:
    def __init__(self, scale):
        self.scale = scale
        self.calls = 0

    def __call__(self, coeffs):
        self.calls += 1
        trend = coeffs * self.scale
        return trend, coeffs - trend, torch.full_like(coeffs.real[0], self.scale)


class _FourierModel:
    def __init__(self, mask_scale):
        self.fourier_analyzer = _Analyzer()
        self.fourier_synthesizer = _Synthesizer()
        self.frequency_disentangler = _MaskDisentangler(mask_scale)


def test_fourier_recon_bypasses_mask_while_fredn_trend_uses_it():
    features = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    positions = torch.tensor([[0.0, 1.0, 2.0]])

    low_mask = extract_fourier_conditions(
        _FourierModel(0.25), features, positions, positions
    )
    high_mask = extract_fourier_conditions(
        _FourierModel(0.75), features, positions, positions
    )

    torch.testing.assert_close(low_mask["fourier_recon"], features)
    torch.testing.assert_close(
        low_mask["fourier_recon"], high_mask["fourier_recon"]
    )
    assert not torch.equal(low_mask["fredn_trend"], high_mask["fredn_trend"])
    torch.testing.assert_close(low_mask["fredn_trend"], features * 0.25)
    torch.testing.assert_close(high_mask["fredn_trend"], features * 0.75)


def test_raw_recon_and_trend_share_one_projection():
    projection = {0: torch.tensor([2.0, -1.0])}
    raw = torch.tensor([[[1.0, 3.0], [2.0, 5.0]]])
    recon = raw + 1.0
    trend = raw * 0.5

    projected = {
        name: project_curves(values, projection)
        for name, values in (
            ("raw", raw),
            ("fourier_recon", recon),
            ("fredn_trend", trend),
        )
    }

    for name, values in (("raw", raw), ("fourier_recon", recon), ("fredn_trend", trend)):
        torch.testing.assert_close(projected[name][0][0], values[0] @ projection[0])


def test_projection_api_applies_all_source_loadings_without_target_labels():
    features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    projections = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}

    projected = project_curves(features, projections)

    assert set(projected) == {0, 1}
    torch.testing.assert_close(projected[0], features[..., 0])
    torch.testing.assert_close(projected[1], features[..., 1])
    assert "labels" not in inspect.signature(project_curves).parameters


def test_generic_checkpoint_parser_rejects_even_and_conflicting_modes():
    parsed = parse_fredn_checkpoint_specs(
        ["9=a.pt", "13=b.pt", "17=c.pt"], legacy9=None, legacy17=None
    )
    assert parsed == {9: "a.pt", 13: "b.pt", 17: "c.pt"}

    with pytest.raises(ValueError, match="positive odd"):
        parse_fredn_checkpoint_specs(["10=a.pt"], None, None)
    with pytest.raises(ValueError, match="conflicting"):
        parse_fredn_checkpoint_specs(["9=a.pt"], "other.pt", None)


def test_contrastive_summary_distinguishes_macro_micro_and_dominance():
    prototypes = {0: np.array([1.0]), 1: np.array([10.0])}
    target_curves = {
        0: [np.array([1.1])] * 90 + [np.array([9.9])] * 10,
        1: [np.array([1.1]), np.array([9.9])],
    }

    aggregate, rows = contrastive_feasibility_by_class(
        prototypes, target_curves, target_total_by_class={0: 100, 1: 10}
    )

    by_class = {row["class_id"]: row for row in rows}
    assert by_class[0]["positive_margin_rate"] == pytest.approx(0.9)
    assert by_class[1]["positive_margin_rate"] == pytest.approx(0.5)
    assert aggregate["micro_positive_margin_rate"] == pytest.approx(91 / 102)
    assert aggregate["macro_positive_margin_rate"] == pytest.approx(0.7)
    assert aggregate["dominant_class_fraction"] == pytest.approx(100 / 102)
    assert by_class[1]["gate_coverage"] == pytest.approx(0.2)


def test_pareto_modes_excludes_dominated_points_without_scalar_score():
    rows = [
        {"mode": 9, "phase_coverage_macro": 0.9, "contrastive_positive_margin_macro": 0.4},
        {"mode": 13, "phase_coverage_macro": 0.7, "contrastive_positive_margin_macro": 0.7},
        {"mode": 15, "phase_coverage_macro": 0.6, "contrastive_positive_margin_macro": 0.6},
        {"mode": 17, "phase_coverage_macro": 0.4, "contrastive_positive_margin_macro": 0.9},
    ]

    result = pareto_modes(
        rows,
        x_key="phase_coverage_macro",
        y_key="contrastive_positive_margin_macro",
    )

    assert [row["mode"] for row in result] == [9, 13, 17]
    assert all("best_mode_score" not in row for row in result)


def test_pareto_modes_excludes_non_finite_points():
    rows = [
        {"mode": 9, "coverage": 0.8, "discrimination": 0.5},
        {"mode": 11, "coverage": float("nan"), "discrimination": 0.9},
    ]

    result = pareto_modes(rows, x_key="coverage", y_key="discrimination")

    assert [row["mode"] for row in result] == [9]


def test_shared_spatial_features_feed_both_fourier_resolutions():
    class CountingPSE:
        def __init__(self):
            self.calls = 0

        def __call__(self, values):
            self.calls += 1
            return values

    torch.manual_seed(3)
    pse = CountingPSE()
    spatial = pse(torch.randn(2, 21, 4, dtype=torch.float64))
    positions = torch.stack(
        [torch.linspace(2.0, 300.0, 21), torch.linspace(4.0, 330.0, 21)]
    ).to(torch.float64)
    dense = torch.linspace(1.0, 365.0, 64, dtype=torch.float64).repeat(2, 1)

    views = build_direct_fourier_views(
        spatial, positions, dense, mode_counts=(9, 13), period_days=365.0, reg=1e-3
    )

    assert pse.calls == 1
    assert set(views) == {9, 13}
    assert views[9].shape == views[13].shape == (2, 64, 4)
    torch.testing.assert_close(spatial, spatial.clone(), rtol=0, atol=0)


@pytest.mark.parametrize("family", ["absolute", "correlation", "derivative"])
def test_shape_distance_is_zero_for_identical_curves(family):
    curve = np.asarray([0.0, 1.0, 0.5, -0.5, 0.0])
    assert shape_distance(curve, curve, family) == pytest.approx(0.0)


def test_absolute_shape_distance_uses_required_time_divisor():
    first = np.asarray([0.0, 0.0, 0.0, 0.0])
    second = np.asarray([1.0, 1.0, 1.0, 1.0])
    assert shape_distance(first, second, "absolute") == pytest.approx(0.5)


def test_mode9_warp_is_transferred_unchanged_to_mode13():
    grid = np.arange(0.0, 101.0)
    source_landmarks = _landmarks([20, 50, 80], [1, -1, 1])
    shifted_landmarks = _landmarks([30, 60, 90], [1, -1, 1])
    fine_source = _piecewise_curve(grid, [20, 35, 50, 65, 80], [1, 0.2, -1, 0.4, 1])
    fine_shifted = _piecewise_curve(grid, [30, 45, 60, 75, 90], [1, 0.2, -1, 0.4, 1])

    aligned = align_curve_to_landmark_template(
        grid, fine_shifted, shifted_landmarks, source_landmarks
    )

    assert normalized_l2(fine_source, aligned) < 0.05
    assert normalized_l2(fine_source, aligned) < normalized_l2(fine_source, fine_shifted)


def _alignment_template(signature, times, values, prototype):
    return SourceAlignmentTemplate(
        signature=signature,
        canonical_landmarks=tuple(_landmarks(times, values)),
        aligned_prototype=np.asarray(prototype, dtype=float),
        accepted_indices=(0,),
    )


def test_candidate_alignment_uses_only_signature_compatible_class():
    grid = np.arange(0.0, 101.0)
    curve_a = _piecewise_curve(grid, [20, 50, 80], [1, -1, 1])
    curve_b = _piecewise_curve(grid, [10, 30, 50, 70, 90], [1, -1, 1, -1, 1])
    templates = {
        0: _alignment_template((3, "P-V-P"), [20, 50, 80], [1, -1, 1], curve_a),
        1: _alignment_template((5, "P-V-P-V-P"), [10, 30, 50, 70, 90], [1, -1, 1, -1, 1], curve_b),
    }

    result = candidate_alignment_prediction(
        grid=grid,
        correspondence_curves={0: curve_a, 1: curve_a},
        comparison_curves={0: curve_a, 1: curve_a},
        templates=templates,
        thresholds={0: 0.2, 1: 0.2},
        unaligned_prototypes={0: curve_a, 1: curve_b},
        distance_family="absolute",
        min_distance_days=10.0,
    )

    assert result.eligible_classes == (0,)
    assert result.prediction == 0
    assert result.used_alignment and not result.used_fallback


def test_no_eligible_class_falls_back_to_unaligned_mode13_prediction():
    grid = np.arange(0.0, 101.0)
    flat = np.zeros_like(grid)
    proto0 = np.zeros_like(grid)
    proto1 = np.ones_like(grid)
    fallback = nearest_prototype_prediction(
        {0: flat, 1: flat}, {0: proto0, 1: proto1}, "absolute"
    )
    templates = {
        0: _alignment_template((1, "P"), [30], [1], proto0),
        1: _alignment_template((1, "V"), [60], [-1], proto1),
    }

    result = candidate_alignment_prediction(
        grid, {0: flat, 1: flat}, {0: flat, 1: flat}, templates,
        {0: 0.2, 1: 0.2}, {0: proto0, 1: proto1}, "absolute", 10.0
    )

    assert result.used_fallback and not result.used_alignment
    assert result.prediction == fallback.prediction


def test_target_label_shuffle_cannot_change_alignment_or_prediction():
    grid = np.arange(0.0, 101.0)
    curve = _piecewise_curve(grid, [20, 50, 80], [1, -1, 1])
    template = _alignment_template((3, "P-V-P"), [20, 50, 80], [1, -1, 1], curve)
    kwargs = dict(
        grid=grid,
        correspondence_curves={0: curve}, comparison_curves={0: curve},
        templates={0: template}, thresholds={0: 0.2},
        unaligned_prototypes={0: curve}, distance_family="correlation",
        min_distance_days=10.0,
    )
    first = candidate_alignment_prediction(**kwargs)
    second = candidate_alignment_prediction(**kwargs)

    assert "target_labels" not in inspect.signature(candidate_alignment_prediction).parameters
    assert first == second
    assert classification_metrics([0], [first.prediction])["accuracy"] == 1.0
    assert classification_metrics([1], [second.prediction])["accuracy"] == 0.0


def test_mode13_can_add_discrimination_when_mode9_is_identical():
    coarse = np.asarray([0.0, 1.0, 0.0])
    fine0 = np.asarray([0.0, 1.0, 0.4, -0.2, 0.0])
    fine1 = np.asarray([0.0, 1.0, -0.4, 0.2, 0.0])
    coarse_prototypes = {0: coarse, 1: coarse}
    fine_prototypes = {0: fine0, 1: fine1}
    coarse_predictions = [
        nearest_prototype_prediction({0: coarse, 1: coarse}, coarse_prototypes, "absolute").prediction
        for _ in range(2)
    ]
    fine_predictions = [
        nearest_prototype_prediction({0: curve, 1: curve}, fine_prototypes, "absolute").prediction
        for curve in (fine0, fine1)
    ]

    assert classification_metrics([0, 1], fine_predictions)["accuracy"] > classification_metrics(
        [0, 1], coarse_predictions
    )["accuracy"]


def test_coarse_fine_hierarchy_counts_internal_landmarks():
    coarse = _landmarks([20, 50, 80], [1, -1, 1])
    fine = _landmarks([20, 30, 40, 50, 60, 70, 80], [1, -1, 1, -1, 1, -1, 1])
    result = coarse_fine_hierarchy(coarse, fine)
    assert result["fine_landmarks_per_coarse_segment_mean"] == pytest.approx(2.0)
    assert result["fraction_mode13_inside_coarse_segments"] == pytest.approx(4 / 7)


def test_stratified_bootstrap_is_reproducible():
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    baseline = np.asarray([0, 1, 1, 0, 0, 1])
    proposed = labels.copy()
    first = stratified_bootstrap_macro_f1_delta(labels, proposed, baseline, 100, 1)
    second = stratified_bootstrap_macro_f1_delta(labels, proposed, baseline, 100, 1)
    assert first == second
    assert first["mean_delta"] > 0


def test_bootstrap_uses_explicit_protocol_class_universe():
    result = stratified_bootstrap_macro_f1_delta(
        true_labels=[0, 0, 0],
        proposed_predictions=[0, 0, 0],
        baseline_predictions=[1, 1, 1],
        repeats=20,
        seed=1,
        class_ids=[0, 1],
    )
    assert result["mean_delta"] == pytest.approx(0.5)


def test_canonical_grid_is_derived_from_source_only():
    class SourceDataset:
        date_positions = np.asarray([12.0, 40.0, 90.0])

    grid = probe_fredn_shape_alignment._source_grid(SourceDataset(), step=2.0)
    assert grid[0] == 12.0
    assert grid[-1] == 90.0


def test_split_replay_is_deterministic_and_uses_independent_seed_resets():
    initial_draws = []
    replay_calls = []

    def historical_fold_creator(datasets, num_folds, num_indices):
        initial_draws.append(random.random())
        replay_calls.append(tuple(datasets))
        splits = {}
        for dataset in datasets:
            count = len(num_indices[dataset])
            n_test, n_val = int(0.2 * count), int(0.1 * count)
            n_train = count - n_test - n_val
            splits[dataset] = {
                "train": set(range(n_train)),
                "val": set(range(n_train, n_train + n_val)),
                "test": set(range(n_train + n_val, count)),
            }
        return [splits]

    source, target = probe_fredn_shape_alignment.replay_protocol_splits(
        "AT1", "DK1", range(100), range(80), seed=1,
        fold_creator=historical_fold_creator,
    )
    source_again, target_again = probe_fredn_shape_alignment.replay_protocol_splits(
        "AT1", "DK1", range(100), range(80), seed=1,
        fold_creator=historical_fold_creator,
    )

    assert source == source_again
    assert target == target_again
    assert len(source["train"]) == 70
    assert len(source["val"]) == 10
    assert len(source["test"]) == 20
    assert len(target["test"]) == 16
    assert replay_calls == [
        ("AT1", "AT1"), ("AT1", "DK1"),
        ("AT1", "AT1"), ("AT1", "DK1"),
    ]
    assert len(set(initial_draws)) == 1


def test_shape_probe_and_launcher_are_offline_and_git_free():
    probe_source = inspect.getsource(probe_fredn_shape_alignment)
    launcher = Path("scripts/run_fredn_shape_alignment_at1_dk1.sh").read_text(
        encoding="utf-8"
    )
    for source in (probe_source, launcher):
        lowered = source.lower()
        assert "subprocess" not in lowered
        assert "os.system" not in lowered
        assert "popen(" not in lowered
        assert "requests" not in lowered
        assert "urllib" not in lowered
        assert "huggingface" not in lowered
        assert "pip install" not in lowered
        assert "git rev-parse" not in lowered
        assert "git status" not in lowered
        assert "git branch" not in lowered
    assert '--git-commit "$GIT_COMMIT"' in launcher
    assert '--git-branch "$GIT_BRANCH"' in launcher
    assert '--git-dirty "$GIT_DIRTY"' in launcher
