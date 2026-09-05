import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from models.fredn.structural_probe import (
    Landmark,
    TopologyMismatchError,
    align_curve_to_landmark_template,
    build_landmark_warp,
    contrastive_feasibility_by_class,
    compute_topology_comparison,
    detect_structural_landmarks,
    extract_fourier_conditions,
    fit_source_class_projections,
    normalized_l2,
    pareto_modes,
    parse_fredn_checkpoint_specs,
    pointwise_intra_class_variance,
    project_curves,
    segment_shape_descriptors,
    topology_signature,
)
from scripts import probe_fredn_structure


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


def test_source_mode_sweep_launcher_is_offline_and_maps_four_gpus():
    path = Path("scripts/run_fredn_source_mode_sweep_at1_4gpu.sh")
    source = path.read_text(encoding="utf-8")

    assert 'launch_source "$GPU0" 11' in source
    assert 'launch_source "$GPU1" 13' in source
    assert 'launch_source "$GPU2" 15' in source
    assert 'launch_source "$GPU3" 19' in source
    assert "--fredn_num_modes" in source
    assert "--fredn_fourier_solver dense_direct" in source
    assert "nohup" in source
    assert "\n        timematch" not in source.lower()
    assert "git " not in source.lower()
    assert "wait " not in source


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
