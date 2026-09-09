import numpy as np
import pytest
import torch


def test_phase_normalization_excludes_near_constant_channels():
    from analysis.phase_shape_diagnostic import normalize_phase_pair
    t = np.linspace(0, 1, 64)
    source = np.column_stack([np.sin(6*t), 1 + 1e-12*np.sin(t), np.cos(6*t)])
    result = normalize_phase_pair(source, 1.5*source)
    assert result["valid_channels"].tolist() == [True, False, True]
    assert np.isfinite(result["source"]).all()
    assert np.abs(result["source"]).max() < 10
    np.testing.assert_allclose(result["source"], result["target"], atol=1e-10)


def test_selection_uses_landmarks_excludes_zero_and_breaks_ties_conservatively():
    from analysis.phase_shape_diagnostic import select_phase_candidate
    rows = [
        dict(lambda_value=lam, candidate_admissible=True, landmark_error_mean=error,
             gamma_mean_displacement_days=displacement, registration_error=reg)
        for lam, error, displacement, reg in [
            (0, 0, 0, 0), (.01, 5, 3, .01), (.1, 2, 2, .8),
            (1, 2, 1, .9), (10, 2, 1, 1.0)]
    ]
    assert select_phase_candidate(rows, 6)["lambda_value"] == 10
    assert select_phase_candidate(rows, 2) is None
    assert select_phase_candidate(rows, np.nan) is None


def test_overwarp_rejected_and_raw_scalar_retained(monkeypatch):
    import analysis.phase_shape_diagnostic as diagnostic
    t = np.linspace(0, 1, 64)
    source = _curve(t)
    monkeypatch.setattr(diagnostic, "_solve_joint_gamma",
                        lambda *args, **kwargs: np.linspace(0, 1, 128)**5)
    result = diagnostic.constrained_residual_phase(
        source, source*1.5, np.array([1., 0.]), t*365)
    assert all(row["rejection_reason"] == "residual_warp_too_large" for row in result["candidates"])
    assert not result["nonlinear_accepted"]
    np.testing.assert_array_equal(result["raw_aligned"], source*1.5)


def test_no_valid_channels_skips_solver(monkeypatch):
    import analysis.phase_shape_diagnostic as diagnostic
    def fail(*args, **kwargs):
        pytest.fail("SRVF must not be called")
    monkeypatch.setattr(diagnostic, "_solve_joint_gamma", fail)
    result = diagnostic.constrained_residual_phase(
        np.ones((64, 2)), np.ones((64, 2))*1.5, np.array([1., 0.]), np.arange(64))
    assert result["failure_reason"] == "no_valid_phase_channels"
    assert result["selected_phase"] == "scalar"


def test_residual_solver_receives_scalar_aligned_normalized_data(monkeypatch):
    import analysis.phase_shape_diagnostic as diagnostic
    t = np.linspace(0, 1, 64, endpoint=False)
    source = _curve(t)
    target = diagnostic._periodic_shift(source, -5, 365)
    delta, scalar = diagnostic.estimate_scalar_phase(source, target)
    assert delta == 5
    expected = diagnostic.normalize_phase_pair(source, scalar)
    calls = []
    def solve(a, b, lam):
        np.testing.assert_allclose(b, diagnostic._resample(expected["target"], 128))
        calls.append(lam)
        return np.linspace(0, 1, 128)
    monkeypatch.setattr(diagnostic, "_solve_joint_gamma", solve)
    result = diagnostic.constrained_residual_phase(source, scalar, np.array([1., 0.]), t*365)
    assert calls == [0, .01, .1, 1, 10]
    assert not result["nonlinear_accepted"]
    np.testing.assert_array_equal(result["raw_aligned"], scalar)


def test_feature_cache_is_opt_in(tmp_path):
    from scripts.diagnose_mode13_phase_shape_oracle import save_feature_caches
    save_feature_caches(tmp_path, {}, {})
    assert not list(tmp_path.glob('*cache.pt'))
    save_feature_caches(tmp_path, {}, {}, True)
    assert len(list(tmp_path.glob('*cache.pt'))) == 2


def test_actual_penalized_residual_improves_small_local_warp():
    from analysis.phase_shape_diagnostic import (
        _periodic_shift, estimate_scalar_phase, constrained_residual_phase,
        registration_metrics,
    )
    t = np.linspace(0, 1, 64)
    source = _curve(t)
    target = _periodic_shift(_curve(t**1.15), -3, 365)
    delta, scalar = estimate_scalar_phase(source, target)
    assert abs(delta) <= 7
    assert registration_metrics(source, scalar)["normalized_l2"] < registration_metrics(source, target)["normalized_l2"]
    result = constrained_residual_phase(source, scalar, np.array([1., 0.]), t*365)
    assert result["nonlinear_accepted"]
    assert result["selected_lambda"] > 0
    assert result["selected_landmark_error"] < result["scalar_landmark_error"]
    assert result["phase"].max_displacement*365 <= 60


def test_amplitude_scaling_selected_shape_stays_raw():
    from analysis.phase_shape_diagnostic import constrained_residual_phase, amplitude_metrics, estimate_scalar_phase
    t = np.linspace(0, 1, 64, endpoint=False)
    source = _curve(t)
    delta, scalar = estimate_scalar_phase(source, 1.5*source)
    assert delta == 0
    result = constrained_residual_phase(source, scalar, np.array([1., 0.]), t*365)
    np.testing.assert_allclose(result["phase"].gamma, np.linspace(0, 1, 128))
    for name in ("range_ratio", "std_ratio", "iqr_ratio"):
        assert amplitude_metrics(source, result["raw_aligned"])[name] == pytest.approx(1.5)


def test_nonfinite_phase_input_is_rejected_before_srvf(monkeypatch):
    import analysis.phase_shape_diagnostic as diagnostic
    source = _curve(np.linspace(0, 1, 64))
    target = source.copy(); target[0, 0] = np.nan
    normalized = diagnostic.normalize_phase_pair(source, target)
    assert normalized["failure_reason"] == "nonfinite_normalized_phase"


def _curve(t):
    return np.stack(
        [np.sin(2 * np.pi * t) + 0.35 * np.sin(6 * np.pi * t),
         np.cos(2 * np.pi * t) - 0.2 * np.cos(4 * np.pi * t)], axis=1
    )


def test_identity_joint_gamma_and_zero_shape_difference():
    from analysis.phase_shape_diagnostic import estimate_nonlinear_phase, registration_metrics, warp_curve
    t = np.linspace(0, 1, 64)
    source = _curve(t)
    result = estimate_nonlinear_phase(source, source)
    aligned = warp_curve(source, result.gamma)
    assert result.valid
    assert result.gamma.shape == (128,)
    assert np.max(np.abs(result.gamma - np.linspace(0, 1, 128))) < 0.03
    assert registration_metrics(source, aligned)["normalized_l2"] < 1e-6


def test_scalar_and_nonlinear_improve_known_translation():
    from analysis.phase_shape_diagnostic import estimate_scalar_phase, estimate_nonlinear_phase, registration_metrics, warp_curve
    t = np.linspace(0, 1, 64)
    source = _curve(t)
    target = _curve(np.mod(t - 4 / 365.0, 1.0))
    delta, scalar = estimate_scalar_phase(source, target, period_days=365.0)
    nonlinear = warp_curve(target, estimate_nonlinear_phase(source, target).gamma)
    baseline = registration_metrics(source, target)["normalized_l2"]
    assert abs(delta) <= 7
    assert registration_metrics(source, scalar)["normalized_l2"] < baseline
    assert registration_metrics(source, nonlinear)["normalized_l2"] < baseline


def test_nonlinear_warp_beats_scalar_landmark_timing():
    from analysis.phase_shape_diagnostic import estimate_scalar_phase, estimate_nonlinear_phase, landmark_alignment_metrics, warp_curve
    t = np.linspace(0, 1, 64)
    source = _curve(t)
    target = _curve(t ** 1.45)
    _, scalar = estimate_scalar_phase(source, target)
    nonlinear = warp_curve(target, estimate_nonlinear_phase(source, target).gamma)
    scalar_error = landmark_alignment_metrics(source[:, 0], scalar[:, 0], t * 365)["mean_time_error"]
    nonlinear_error = landmark_alignment_metrics(source[:, 0], nonlinear[:, 0], t * 365)["mean_time_error"]
    assert nonlinear_error < scalar_error


def test_amplitude_scaling_is_not_mistaken_for_phase():
    from analysis.phase_shape_diagnostic import amplitude_metrics, estimate_nonlinear_phase
    t = np.linspace(0, 1, 64)
    source = _curve(t)
    target = 1.5 * source
    result = estimate_nonlinear_phase(source, target)
    assert np.mean(np.abs(result.gamma - np.linspace(0, 1, 128))) < 0.02
    assert amplitude_metrics(source[:, 0], target[:, 0])["range_ratio"] == pytest.approx(1.5)


def test_prototype_grouping_is_the_only_target_label_dependent_operation():
    from analysis.phase_shape_diagnostic import build_pointwise_median_prototypes
    values = np.arange(4 * 3 * 2).reshape(4, 3, 2)
    a = build_pointwise_median_prototypes(values, np.array([0, 0, 1, 1]))
    b = build_pointwise_median_prototypes(values, np.array([1, 1, 0, 0]))
    assert np.array_equal(a[0], b[1])
    assert np.array_equal(a[1], b[0])


def test_joint_gamma_is_single_monotonic_endpoint_fixed_vector():
    from analysis.phase_shape_diagnostic import estimate_nonlinear_phase
    t = np.linspace(0, 1, 64)
    result = estimate_nonlinear_phase(_curve(t), _curve(t ** 1.2))
    assert result.gamma.shape == (128,)
    assert np.all(np.diff(result.gamma) >= -1e-10)
    assert result.gamma[0] == pytest.approx(0.0)
    assert result.gamma[-1] == pytest.approx(1.0)


def test_shape_margin_is_positive_for_near_same_and_far_wrong_class():
    from analysis.phase_shape_diagnostic import shape_margin
    t = np.linspace(0, 1, 64)
    source = {0: _curve(t), 1: _curve(t + 0.25)}
    result = shape_margin(0, _curve(t) + 0.001, source)
    assert result["nearest_wrong_class"] == 1
    assert result["shape_margin"] > 0
    assert result["margin_positive"]


def test_shape_margin_selects_nearest_wrong_class_by_distance_not_class_id():
    from analysis.phase_shape_diagnostic import shape_margin
    t = np.linspace(0, 1, 64)
    target = _curve(t)
    source = {
        0: target + 0.001,
        1: _curve(t + 0.25),
        2: target + 0.02,
    }
    result = shape_margin(0, target, source)
    assert result["nearest_wrong_class"] == 2


def test_invalid_gamma_falls_back_to_identity_and_records_failure(monkeypatch):
    import analysis.phase_shape_diagnostic as diagnostic
    t = np.linspace(0, 1, 64)
    monkeypatch.setattr(diagnostic, "_solve_joint_gamma", lambda *_: np.array([0.0, np.nan, 1.0]))
    result = diagnostic.estimate_nonlinear_phase(_curve(t), _curve(t))
    assert not result.valid
    assert result.failure_reason
    assert np.array_equal(result.gamma, np.linspace(0, 1, 128))


def test_oracle_analysis_smoke_emits_complete_metrics_and_artifacts(tmp_path):
    from types import SimpleNamespace
    from scripts.diagnose_mode13_phase_shape_oracle import analyze

    time = np.linspace(0, 1, 64)
    class_zero = _curve(time)
    class_one = _curve(time + 0.2)
    source = np.stack(
        [class_zero + offset for offset in (-0.01, 0.0, 0.01)]
        + [class_one + offset for offset in (-0.01, 0.0, 0.01)]
    )
    target = source * 1.5
    labels = np.array([0, 0, 0, 1, 1, 1])
    cache_source = {
        "mode13_features": torch.tensor(source, dtype=torch.float32),
        "labels": labels,
    }
    cache_target = {
        "mode13_features": torch.tensor(target, dtype=torch.float32),
        "labels": labels,
    }
    args = SimpleNamespace(
        task="S_T",
        output_dir=tmp_path,
        prominence_rel=0.05,
        min_distance_days=5.0,
    )
    phase, shape, landmarks, segments, artifacts, summary = analyze(
        args, cache_source, cache_target, ["class0", "class1"]
    )
    assert {row["phase_type"] for row in phase} == {
        "global", "scalar", "nonlinear"
    }
    nonlinear = next(row for row in phase if row["phase_type"] == "nonlinear")
    assert {
        "registration_l2", "registration_normalized_l2",
        "nonlinear_improvement_vs_scalar", "gamma_p95_displacement_days",
        "phase_failure_reason",
    } <= nonlinear.keys()
    assert {
        "global_iqr_ratio", "peak_height_diff_mean",
        "valley_height_diff_mean", "prominence_diff_mean",
        "peak_valley_amplitude_ratio_mean",
    } <= shape[0].keys()
    assert summary["num_common_classes"] == 2
    assert set(artifacts["gamma_by_class"]) == {0, 1}
    assert (tmp_path / "class_0_phase_shape.png").is_file()
    assert (tmp_path / "gamma_by_class.png").is_file()
    assert isinstance(landmarks, list)
    assert isinstance(segments, list)
    for row in shape:
        assert row["selected_phase"] == "scalar"
        assert not row["nonlinear_accepted"]
        assert row["global_range_ratio"] == pytest.approx(1.5, abs=1e-5)
        assert row["global_std_ratio"] == pytest.approx(1.5, abs=1e-5)
        assert row["peak_height_ratio_mean"] == pytest.approx(1.5, abs=1e-5)
    for rows in (landmarks, segments):
        assert all(row["selected_phase"] == "scalar" for row in rows)
    assert not list(tmp_path.glob('*cache.pt'))
    for name in ('phase_channel_metrics.csv', 'phase_lambda_metrics.csv', 'phase_lambda_summary.csv',
                 'gamma_candidates_by_class.png'):
        assert (tmp_path / name).is_file()


def test_cache_labels_are_joined_by_parcel_id_not_loader_order():
    from scripts.diagnose_mode13_phase_shape_oracle import labels_for_cache

    class Dataset:
        def get_parcel_indices(self):
            return np.array([10, 20, 30])

        def get_labels(self):
            return np.array([1, 2, 3])

    cache = {"sample_id": torch.tensor([30, 10, 20])}
    assert np.array_equal(labels_for_cache(Dataset(), cache), [3, 1, 2])


def test_launcher_is_offline_and_contains_exact_four_tasks():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/run_mode13_phase_shape_oracle_4tasks.sh").read_text()
    forbidden = ("git ", "curl ", "wget ", "pip install", "conda install")
    assert not any(token in launcher for token in forbidden)
    commands = [line.strip() for line in launcher.splitlines() if line.strip().startswith("run \"")]
    assert commands == [
        'run "$GPU0" AT1 DK1 "$AT1_WEIGHTS/fold_0/model.pt" & p0=$!',
        'run "$GPU1" DK1 FR1 "$DK1_WEIGHTS/fold_0/model.pt" & p1=$!',
        'run "$GPU2" FR1 FR2 "$FR1_WEIGHTS/fold_0/model.pt" & p2=$!',
        'run "$GPU3" FR2 AT1 "$FR2_WEIGHTS/fold_0/model.pt" & p3=$!',
    ]
    assert 'local gpu="$1"\n' in launcher
    assert 'local source="$2"\n' in launcher
    assert 'local target="$3"\n' in launcher
    assert 'local checkpoint="$4"\n' in launcher
    assert 'local task="${source}_${target}"\n' in launcher
    for flag, value in (("phase-edge-points", "8"), ("phase-edge-monotonicity", "0.75"),
                        ("phase-edge-range-ratio", "0.15"), ("full-landmark-coverage", "0.80"),
                        ("partial-min-landmarks", "2"), ("partial-min-time-coverage", "0.20")):
        assert "--" + flag + " " + value in launcher


def test_order_preserving_landmark_matching_skips_spurious_early_mark():
    from analysis.phase_shape_diagnostic import _minimum_cost_ordered_pairs
    from models.fredn.structural_probe import Landmark

    source = [Landmark("peak", time, 1.0, 1.0) for time in (10.0, 30.0)]
    target = [Landmark("peak", time, 1.0, 1.0) for time in (1.0, 11.0, 31.0)]
    pairs = _minimum_cost_ordered_pairs(source, target)
    assert [(left.time, right.time) for left, right in pairs] == [
        (10.0, 11.0), (30.0, 31.0)
    ]


def test_target_dataset_is_redacted_before_getitem():
    from scripts.diagnose_mode13_phase_shape_oracle import redact_dataset_labels

    class Dataset:
        samples = [("a", 10, 7, None), ("b", 20, 8, None)]

    original = Dataset()
    redacted = redact_dataset_labels(original)
    assert [sample[2] for sample in redacted.samples] == [0, 0]
    assert [sample[2] for sample in original.samples] == [7, 8]


def _support_curves(side=None):
    grid = np.linspace(0, 365, 64)
    source = np.interp(grid, [0, 45, 120, 215, 295, 365], [0, 2, -2, 2, -2, 0])
    if side == "left":
        target = np.interp(grid, [0, 120, 215, 295, 365], [3, -2, 2, -2, 0])
    elif side == "right":
        target = np.interp(grid, [0, 45, 120, 215, 365], [0, 2, -2, 2, -3])
    else:
        target = np.interp(grid, [0, 50, 125, 220, 300, 365], [0, 2, -2, 2, -2, 0])
    return grid, source, target


@pytest.mark.parametrize("side,expected", [(None, "FULL_PHASE"), ("left", "PARTIAL_PHASE"),
                                          ("right", "PARTIAL_PHASE")])
def test_scalar_support_completeness_and_full_priority(side, expected, monkeypatch):
    import analysis.phase_shape_diagnostic as d
    grid, source, target = _support_curves(side)
    support = d.discover_phase_support(source, target, grid)
    assert support["common_chain_landmark_count"] == (4 if side is None else 3)
    assert support["partial_support_valid"]
    assert support["full_support_valid"] == (side is None)
    if side:
        assert support[side + "_truncation_evidence"] == "strong"
    monkeypatch.setattr(d, "_solve_joint_gamma", lambda *a, **kw: np.linspace(0, 1, 128))
    raw_s, raw_t = source[:, None], target[:, None]
    full = d.constrained_residual_phase(raw_s, raw_t, np.ones(1), grid)
    partial = d.constrained_partial_phase(raw_s, raw_t, grid, support)
    state = d.classify_phase_applicability(support, full, partial)
    assert state["phase_applicability"] == expected
    assert state["phase_applicability_reason"]


def test_single_shared_landmark_and_unrelated_flat_target_are_not_applicable():
    import analysis.phase_shape_diagnostic as d
    grid = np.linspace(0, 365, 64)
    source = np.exp(-((grid-150)/30)**2)
    for target in (source, source*1e-12):
        support = d.discover_phase_support(source, target, grid)
        assert support["common_chain_landmark_count"] <= 1
        assert not support["partial_support_valid"]
        assert not support["full_support_valid"]
        assert d.classify_phase_applicability(support, None, None)["phase_applicability"] == "PHASE_NOT_APPLICABLE"


def test_contiguous_chain_cannot_bridge_missing_landmark():
    from analysis.phase_shape_diagnostic import longest_contiguous_chain
    from models.fredn.structural_probe import Landmark
    source = [Landmark("peak" if i % 2 else "valley", float(i), 1., 1.) for i in range(5)]
    target = [Landmark("peak", float(i), 1., 1.) for i in range(4)]
    pairs = [(source[1], target[1]), (source[3], target[2]), (source[4], target[3])]
    assert longest_contiguous_chain(source, target, pairs) == pairs[1:]


def test_edge_activity_alone_does_not_reject_complete_support():
    from analysis.phase_shape_diagnostic import discover_phase_support
    grid, source, _ = _support_curves()
    support = discover_phase_support(source, source, grid)
    assert support["left_boundary_active_source"]
    assert support["left_truncation_evidence"] != "strong"
    assert support["full_support_valid"]
    assert support["source_left_monotonicity_ratio"] == 1


def test_partial_isolated_normalization_gamma_errors_and_raw_shape(monkeypatch):
    import analysis.phase_shape_diagnostic as d
    grid, source, target = _support_curves("left")
    support = d.discover_phase_support(source, target, grid)
    raw_s = np.column_stack((source, 1 + source*1e-12))
    raw_t = np.column_stack((target*1.5, 1 + target*1e-12))
    calls = []
    def solve(s, t, lam):
        calls.append((s.copy(), t.copy()))
        assert s.shape == t.shape == (128, 1)
        return np.linspace(0, 1, 128)
    monkeypatch.setattr(d, "_solve_joint_gamma", solve)
    a = d.constrained_partial_phase(raw_s, raw_t, grid, support)
    raw_s[grid < support["source_common_start_day"]] = 1e8
    raw_t[grid < support["target_common_start_day"]] = -1e8
    b = d.constrained_partial_phase(raw_s, raw_t, grid, support)
    assert a["solution_valid"] and b["solution_valid"]
    assert a["normalized"]["valid_channels"].tolist() == [True, False]
    for key in ("raw_source", "raw_aligned"):
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(a["phase"].gamma, b["phase"].gamma)
    assert a["selected_landmark_error"] == b["selected_landmark_error"]
    for first, second in zip(calls[:5], calls[5:]):
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


def test_partial_overwarp_uses_target_interval_days(monkeypatch):
    import analysis.phase_shape_diagnostic as d
    grid, source, target = _support_curves("left")
    support = d.discover_phase_support(source, target, grid)
    gamma = np.linspace(0, 1, 128)**5
    monkeypatch.setattr(d, "_solve_joint_gamma", lambda *a, **kw: gamma)
    result = d.constrained_partial_phase(source[:, None], target[:, None], grid, support)
    duration = support["target_common_end_day"] - support["target_common_start_day"]
    displacement = np.max(np.abs(gamma - np.linspace(0, 1, 128))) * duration
    assert displacement > 60
    assert all(r["gamma_max_displacement_days"] == pytest.approx(displacement) for r in result["candidates"])
    assert all(r["rejection_reason"] == "residual_warp_too_large" for r in result["candidates"])
    assert not result["nonlinear_accepted"]
    assert result["selected_phase"] == "partial_linear"


def test_partial_csv_observability_and_scope_separated_summary(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import csv
    import analysis.phase_shape_diagnostic as d
    from scripts.diagnose_mode13_phase_shape_oracle import analyze
    monkeypatch.setattr(d, "_solve_joint_gamma", lambda *a, **kw: np.linspace(0, 1, 128))
    # This fixture IS the scalar frame; do not introduce a second coarse shift.
    monkeypatch.setattr("scripts.diagnose_mode13_phase_shape_oracle.estimate_scalar_phase",
                        lambda s, t: (0, t))
    _, source, target = _support_curves("left")
    def cache(curve):
        return {"mode13_features": torch.tensor(curve[None, :, None]), "labels": np.array([0])}
    args = SimpleNamespace(task="S_T", output_dir=tmp_path, prominence_rel=.05, min_distance_days=14)
    _, shapes, landmarks, segments, artifacts, summary = analyze(args, cache(source), cache(target), ["crop"])
    assert shapes[0]["phase_applicability"] == "PARTIAL_PHASE"
    assert shapes[0]["partial_shape_metrics_valid"] and not shapes[0]["full_shape_metrics_valid"]
    missing = [r for r in landmarks if not r["observable_in_both_domains"]]
    assert missing
    assert all(np.isnan(r["height_ratio"]) and np.isnan(r["prominence_ratio"]) for r in missing)
    assert any(not r["observable_in_both_domains"] for r in segments)
    assert summary["partial_phase_count"] == 1 and summary["full_phase_count"] == 0
    assert artifacts["phase_applicability_by_class"][0] == "PARTIAL_PHASE"
    assert 0 in artifacts["partial_gamma_by_class"]
    with (tmp_path / "phase_applicability_metrics.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["left_truncation_evidence"] == "strong"
    assert float(row["common_time_coverage_min"]) >= .20


def test_partial_shape_values_ignore_outside_peak_with_fixed_support(monkeypatch):
    import analysis.phase_shape_diagnostic as d
    from scripts.diagnose_mode13_phase_shape_oracle import support_shape_metrics
    grid, source, target = _support_curves("left")
    source, target = source[:, None], target[:, None]
    loading = np.ones(1)
    support = d.discover_phase_support(source[:, 0], target[:, 0], grid)
    monkeypatch.setattr(d, "_solve_joint_gamma", lambda *a, **kw: np.linspace(0, 1, 128))
    def measure():
        partial = d.constrained_partial_phase(source, target, grid, support)
        state = d.classify_phase_applicability(support, None, partial)
        return support_shape_metrics("S_T", 0, "crop", source, target, grid, loading, support, partial, state)
    a = measure()
    source[0] = 1e8
    target[0] = -1e8
    b = measure()
    for rows_a, rows_b in zip(([a[0]], a[1], a[2]), ([b[0]], b[1], b[2])):
        for ra, rb in zip(rows_a, rows_b):
            assert ra.keys() == rb.keys()
            for key in ra:
                if isinstance(ra[key], (float, np.floating)):
                    np.testing.assert_allclose(ra[key], rb[key], equal_nan=True)
                else:
                    assert ra[key] == rb[key]


def test_partial_iqr_is_computed_before_resampling_unequal_crops(monkeypatch):
    import analysis.phase_shape_diagnostic as d
    from models.fredn.structural_probe import Landmark
    grid = np.linspace(0, 365, 64)
    raw = _curve(grid/365)
    chain = [(Landmark("peak", grid[10], 1., 1.), Landmark("peak", grid[15], 1., 1.)),
             (Landmark("valley", grid[50], -1., 1.), Landmark("valley", grid[45], -1., 1.))]
    support = dict(partial_support_valid=True, common_chain=chain,
                   source_common_start_day=grid[10], source_common_end_day=grid[50],
                   target_common_start_day=grid[15], target_common_end_day=grid[45])
    monkeypatch.setattr(d, "_solve_joint_gamma", lambda *a, **kw: np.linspace(0, 1, 128))
    result = d.constrained_partial_phase(raw, raw*1.5, grid, support)
    assert result["normalized"]["source"].shape[0] == 41
    assert result["normalized"]["target"].shape[0] == 31
    np.testing.assert_allclose(result["normalized"]["source_iqr"], np.percentile(raw[10:51], 75, axis=0)-np.percentile(raw[10:51], 25, axis=0))
    np.testing.assert_allclose(result["normalized"]["target_iqr"], np.percentile(raw[15:46]*1.5, 75, axis=0)-np.percentile(raw[15:46]*1.5, 25, axis=0))


def test_no_applicable_phase_emits_only_na_shape_and_no_partial_gamma(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import analysis.phase_shape_diagnostic as d
    from scripts.diagnose_mode13_phase_shape_oracle import analyze
    monkeypatch.setattr(d, "_solve_joint_gamma", lambda *a, **kw: np.linspace(0, 1, 128))
    monkeypatch.setattr("scripts.diagnose_mode13_phase_shape_oracle.estimate_scalar_phase", lambda s, t: (0, t))
    grid, source, _ = _support_curves()
    target = np.exp(-((grid-150)/30)**2)*1e-12
    def cache(x):
        return dict(mode13_features=torch.tensor(x[None, :, None]), labels=np.array([0]))
    args = SimpleNamespace(task="S_T", output_dir=tmp_path, prominence_rel=.05, min_distance_days=14)
    _, shapes, marks, segments, artifacts, summary = analyze(args, cache(source), cache(target), ["crop"])
    assert shapes[0]["phase_applicability"] == "PHASE_NOT_APPLICABLE"
    assert not shapes[0]["shape_phase_conditioned_valid"]
    for key in ("global_range_ratio", "peak_height_ratio_mean", "segment_auc_ratio_mean", "same_class_shape_distance"):
        assert np.isnan(shapes[0][key])
    assert all(not r["observable_in_both_domains"] for r in marks + segments)
    assert artifacts["partial_gamma_by_class"] == {}
    assert summary["phase_not_applicable_count"] == 1


def test_full_support_with_missing_boundary_landmark_reports_unobservable(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import analysis.phase_shape_diagnostic as d
    from scripts.diagnose_mode13_phase_shape_oracle import analyze
    monkeypatch.setattr(d, "_solve_joint_gamma", lambda *a, **kw: np.linspace(0, 1, 128))
    monkeypatch.setattr("scripts.diagnose_mode13_phase_shape_oracle.estimate_scalar_phase", lambda s, t: (0, t))
    grid = np.linspace(0, 365, 64)
    source = np.interp(grid, [0, 35, 90, 145, 200, 255, 310, 365], [0, 2, -2, 2, -2, 2, -2, 0])
    target = source.copy()
    # Missing leading peak but almost flat first eight points: no strong edge evidence.
    target[grid < 90] = np.interp(grid[grid < 90], [0, 50, 90], [0, -.1, -2])
    def cache(x):
        return dict(mode13_features=torch.tensor(x[None, :, None]), labels=np.array([0]))
    args = SimpleNamespace(task="S_T", output_dir=tmp_path, prominence_rel=.05, min_distance_days=14)
    _, shapes, marks, segments, _, _ = analyze(args, cache(source), cache(target), ["crop"])
    assert shapes[0]["phase_applicability"] == "FULL_PHASE"
    missing = [r for r in marks if not r["observable_in_both_domains"]]
    assert missing and all(np.isnan(r["height_ratio"]) for r in missing)
    assert any(not r["observable_in_both_domains"] for r in segments)


def test_target_only_boundary_segment_is_explicitly_unobservable(monkeypatch):
    import analysis.phase_shape_diagnostic as d
    from scripts.diagnose_mode13_phase_shape_oracle import support_shape_metrics
    grid, complete, truncated = _support_curves("left")
    source, target = truncated[:, None], complete[:, None]
    support = d.discover_phase_support(source[:, 0], target[:, 0], grid)
    monkeypatch.setattr(d, "_solve_joint_gamma", lambda *a, **kw: np.linspace(0, 1, 128))
    partial = d.constrained_partial_phase(source, target, grid, support)
    state = d.classify_phase_applicability(support, None, partial)
    _, _, segments = support_shape_metrics("S_T", 0, "crop", source, target, grid,
                                           np.ones(1), support, partial, state)
    missing = [r for r in segments if not r["observable_in_both_domains"]]
    assert missing and all(np.isnan(r["auc_ratio"]) for r in missing)


def test_partial_real_srvf_preserves_joint_gamma_and_amplitude():
    import analysis.phase_shape_diagnostic as d
    grid, source, target = _support_curves("left")
    support = d.discover_phase_support(source, target, grid)
    raw_s = np.column_stack((source, source**2))
    raw_t = np.column_stack((target, target**2))*1.5
    result = d.constrained_partial_phase(raw_s, raw_t, grid, support)
    assert result["solution_valid"]
    assert result["phase"].gamma.shape == (128,)
    assert np.all(np.diff(result["phase"].gamma) >= 0)
    assert result["phase"].max_displacement * (support["target_common_end_day"]-support["target_common_start_day"]) <= 60
    assert result["selected_lambda"] != 0
    np.testing.assert_allclose(result["raw_aligned"], result["raw_source"]*1.5, atol=1e-8)


def test_partial_acceptance_uses_chain_error_and_excludes_unrestricted(monkeypatch):
    import analysis.phase_shape_diagnostic as d
    from models.fredn.structural_probe import Landmark
    grid = np.linspace(0, 365, 64)
    source = _curve(grid/365)
    chain = [(Landmark(k, s, 1., 1.), Landmark(k, t, 1., 1.))
             for k, s, t in (("valley", 50, 80), ("peak", 150, 200), ("valley", 300, 280))]
    support = dict(partial_support_valid=True, common_chain=chain,
                   source_common_start_day=50., source_common_end_day=300.,
                   target_common_start_day=80., target_common_end_day=280.)
    identity = np.linspace(0, 1, 128)
    gamma = np.interp(identity, [0, .4, 1], [0, .6, 1])
    monkeypatch.setattr(d, "_solve_joint_gamma", lambda *a, **kw: gamma)
    result = d.constrained_partial_phase(source, source*1.5, grid, support)
    assert result["nonlinear_accepted"]
    assert result["selected_lambda"] == 10  # Same landmark error/displacement: conservative penalty tie-break.
    assert result["scalar_landmark_error"] == pytest.approx(40/3)
    assert result["selected_landmark_error"] < .3
    np.testing.assert_allclose(result["mapped_target_grid"], 80 + gamma*200)
    assert not result["candidates"][0]["candidate_admissible"]


def test_partial_invalid_crop_channels_skip_solver_and_not_applicable(monkeypatch):
    import analysis.phase_shape_diagnostic as d
    grid, s, t = _support_curves("left")
    support = d.discover_phase_support(s, t, grid)
    def fail(*a, **kw):
        pytest.fail("no valid cropped channels must skip SRVF")
    monkeypatch.setattr(d, "_solve_joint_gamma", fail)
    partial = d.constrained_partial_phase(np.ones((64, 2)), np.ones((64, 2)), grid, support)
    assert not partial["solution_valid"]
    assert partial["failure_reason"] == "no_valid_phase_channels"
    assert d.classify_phase_applicability(support, None, partial)["phase_applicability"] == "PHASE_NOT_APPLICABLE"


def test_partial_coverage_gate_rejects_short_common_interval():
    from analysis.phase_shape_diagnostic import discover_phase_support
    grid = np.linspace(0, 365, 64)
    curve = np.interp(grid, [0, 125, 145, 165, 185, 365], [0, 0, 2, -2, 0, 0])
    support = discover_phase_support(curve, curve, grid, prominence=.1)
    assert support["common_chain_landmark_count"] == 2
    assert support["common_time_coverage_min"] < .2
    assert not support["partial_support_valid"]


def test_lambda_summaries_do_not_mix_full_and_partial_scopes():
    from scripts.diagnose_mode13_phase_shape_oracle import summarize_lambdas
    rows = [dict(registration_scope=scope, lambda_value=.1, gamma_valid=True,
                 candidate_admissible=True, registration_error=error, landmark_error_mean=error,
                 gamma_mean_displacement_days=0., gamma_max_displacement_days=0., rejection_reason="")
            for scope, error in (("FULL_CONSTRAINED", 10.), ("PARTIAL_CONSTRAINED", 1.))]
    summary = summarize_lambdas("S_T", rows)
    assert len(summary) == 2
    assert {r["registration_scope"]: r["landmark_error_mean"] for r in summary} == {
        "FULL_CONSTRAINED": 10., "PARTIAL_CONSTRAINED": 1.}


def _viz_fixture():
    grid = np.linspace(0, 365, 64, endpoint=False)
    source = np.stack([_curve(grid/365) + [i*.1, -i*.05] for i in range(5)])
    target = source*1.5
    return grid, source, target


def test_viz_shared_pca_is_source_only_sign_fixed_and_not_class_axes():
    from analysis.phase_shape_diagnostic import fit_visualization_shared_pca
    grid = np.linspace(0, 1, 64)
    prototypes = {0: np.column_stack([grid*3, grid*0]), 1: np.column_stack([grid*0, grid*2])}
    w = fit_visualization_shared_pca(prototypes)
    assert w[np.argmax(np.abs(w))] > 0
    assert np.linalg.norm(w) == pytest.approx(1.)
    assert not np.allclose(w, [1, 0]) and not np.allclose(w, [0, 1])
    np.testing.assert_array_equal(w, fit_visualization_shared_pca(dict(reversed(list(prototypes.items())))))


def test_viz_deterministic_selection_uses_sorted_real_ids_and_no_rng():
    from analysis.phase_shape_diagnostic import select_visualization_samples
    ids = np.arange(100)[::-1]
    before = np.random.get_state()
    indices = select_visualization_samples(ids, 40)
    expected = np.linspace(0, 99, 40).astype(int)
    np.testing.assert_array_equal(ids[indices], expected)
    np.testing.assert_array_equal(indices, select_visualization_samples(ids, 40))
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
    np.testing.assert_array_equal(np.array([5, 1])[select_visualization_samples([5, 1], 40)], [1, 5])


def test_viz_group_reuses_one_axis_preserves_prototype_and_full_group_bands():
    from analysis.phase_shape_diagnostic import prepare_visualization_group
    grid, source, target = _viz_fixture()
    # Deliberately noncommuting coordinate median and projection.
    source[:3] = np.array([[0., 0.], [0., 10.], [10., 0.]])[:, None, :]
    direction = np.array([1., 1.])/np.sqrt(2)
    sp, tp = np.median(source, axis=0), np.median(target, axis=0)
    data = prepare_visualization_group(source, target, np.arange(5), np.arange(5)+10,
        sp, tp, direction, grid, 0, "FULL_PHASE", max_curves=2)
    np.testing.assert_allclose(data["source"]["prototype"], sp @ direction)
    np.testing.assert_allclose(data["target_global"]["prototype"], tp @ direction)
    np.testing.assert_allclose(data["source"]["quantiles"], np.percentile(source @ direction, [25,50,75], axis=0))
    assert not np.allclose(data["source"]["prototype"], data["source"]["quantiles"][1])
    assert data["source"]["samples"].shape == (2,64)
    np.testing.assert_allclose(data["residual"]["quantiles"],
        np.percentile(target @ direction - sp @ direction, [25,50,75], axis=0))


@pytest.mark.parametrize("state", ["FULL_PHASE", "PARTIAL_PHASE", "PHASE_NOT_APPLICABLE"])
def test_viz_applicability_nan_masks_and_projection_warp_commute(state):
    from analysis.phase_shape_diagnostic import prepare_visualization_group, warp_curve
    grid, source, target = _viz_fixture()
    sp, tp = np.median(source, axis=0), np.median(target, axis=0)
    w = np.array([.6,.8])
    gamma = np.linspace(0,1,128)**1.2
    support = dict(source_common_start_day=100., source_common_end_day=220.,
                   target_common_start_day=120., target_common_end_day=260.)
    data = prepare_visualization_group(source, target, np.arange(5), np.arange(5),
        sp, tp, w, grid, 0, state, gamma=gamma, support=support, partial_gamma=gamma)
    if state == "FULL_PHASE":
        np.testing.assert_allclose(data["target_selected"]["samples"][0], warp_curve(target[0], gamma) @ w)
        assert np.isfinite(data["residual"]["samples"]).all()
        assert data["support_mask"].all()
    elif state == "PARTIAL_PHASE":
        mask = (grid >=100)&(grid <=220)
        assert np.isnan(data["residual"]["samples"][:,~mask]).all()
        assert np.isfinite(data["residual"]["samples"][:,mask]).all()
        assert np.isnan(data["target_selected"]["prototype"][~mask]).all()
    else:
        assert np.isnan(data["residual"]["samples"]).all()
        np.testing.assert_array_equal(data["target_selected"]["samples"], data["target_scalar"]["samples"])


def test_viz_distance_matrix_multidimensional_before_scalar_and_selected_na():
    from analysis.phase_shape_diagnostic import visualization_distance_matrices, normalized_l2, _periodic_shift
    grid, source, _ = _viz_fixture()
    prototypes = {i: source[i] for i in range(3)}
    artifacts = dict(source_raw_prototypes=prototypes, target_raw_prototypes=prototypes,
        scalar_delta_by_class={0:3,1:0,2:0}, gamma_by_class={0:np.linspace(0,1,128)},
        selected_lambda={0:None}, phase_applicability_by_class={0:"FULL_PHASE",1:"PARTIAL_PHASE",2:"PHASE_NOT_APPLICABLE"})
    before, selected = visualization_distance_matrices(artifacts, [0,1,2])
    assert before[1,0] == pytest.approx(normalized_l2(prototypes[1], _periodic_shift(prototypes[0],3,365)))
    assert np.isfinite(selected[:,0]).all()
    assert np.isnan(selected[:,1:]).all()


def test_viz_spaghetti_shared_ylim_and_no_fake_not_applicable_residual():
    from analysis.phase_shape_diagnostic import prepare_visualization_group
    from scripts.diagnose_mode13_phase_shape_oracle import plot_visualization_spaghetti
    import matplotlib.pyplot as plt
    grid, s, t = _viz_fixture()
    data = prepare_visualization_group(s,t,np.arange(5),np.arange(5),np.median(s,0),np.median(t,0),
                                      np.ones(2),grid,0,"PHASE_NOT_APPLICABLE")
    figure = plot_visualization_spaghetti(data, "S_T", "crop", {"phase_applicability_reason":"no_support"})
    try:
        assert len({ax.get_ylim() for ax in figure.axes}) == 1
        assert all(ax.get_xlim() == (0,365) for ax in figure.axes)
        assert "NOT APPLICABLE" in figure.axes[2].get_title()
        assert not figure.axes[3].lines
    finally:
        plt.close(figure)


def test_viz_manifest_outputs_no_cache_and_no_numeric_mutation(tmp_path, monkeypatch):
    import copy
    import json
    import scripts.diagnose_mode13_phase_shape_oracle as script
    from scripts.diagnose_mode13_phase_shape_oracle import generate_diagnostic_visualizations
    grid, s, t = _viz_fixture()
    s, t = np.concatenate([s,s+1]), np.concatenate([t,t+1])
    labels = np.repeat([0,1],5)
    sc = dict(mode13_features=torch.from_numpy(s), labels=labels, sample_id=torch.arange(10))
    tc = dict(mode13_features=torch.from_numpy(t), labels=labels, sample_id=torch.arange(100,110))
    artifacts = dict(source_raw_prototypes={i:np.median(s[labels==i],0) for i in (0,1)},
        target_raw_prototypes={i:np.median(t[labels==i],0) for i in (0,1)},
        source_pca_direction_by_class={0:np.array([1.,0.]),1:np.array([0.,1.])},
        scalar_delta_by_class={0:0,1:0}, gamma_by_class={i:np.linspace(0,1,128) for i in (0,1)},
        selected_lambda={0:None,1:None}, phase_applicability_by_class={0:"FULL_PHASE",1:"PHASE_NOT_APPLICABLE"},
        common_support_by_class={0:{},1:{}}, partial_gamma_by_class={}, partial_selected_lambda_by_class={})
    original = copy.deepcopy(artifacts)
    directions = []
    prepare = script.prepare_visualization_group
    def capture(**kwargs):
        directions.append(kwargs["direction"].copy())
        return prepare(**kwargs)
    monkeypatch.setattr(script, "prepare_visualization_group", capture)
    sentinel = tmp_path / "shape_class_metrics.csv"
    sentinel.write_text("unchanged\n")
    manifest = generate_diagnostic_visualizations(tmp_path, "S_T", ["cropA","cropB"], sc,tc,artifacts,
                                                  [], max_curves=2, dpi=40)
    assert sentinel.read_text() == "unchanged\n"
    assert manifest["shared_pca_fit_source_only"]
    assert manifest["sample_ids"]["source"]["0"] == [0,4]
    assert manifest["sample_ids"]["target"]["1"] == [105,109]
    assert len(set(tuple(lim) for lim in manifest["gallery_ylims"].values())) == 1
    assert not list(tmp_path.rglob('*.pt'))
    assert (tmp_path / 'visualizations/visualization_manifest.json').is_file()
    assert len(list(tmp_path.rglob('*.png'))) == 13
    np.testing.assert_array_equal(artifacts["source_raw_prototypes"][0], original["source_raw_prototypes"][0])
    assert json.loads((tmp_path / 'visualizations/visualization_manifest.json').read_text())["mode"] == 13
    # Each class uses its own axis for same-class plots, but precisely the SAME
    # shared source axis for gallery/overlay, including all target panels.
    np.testing.assert_array_equal(directions[0], artifacts["source_pca_direction_by_class"][0])
    np.testing.assert_array_equal(directions[2], artifacts["source_pca_direction_by_class"][1])
    np.testing.assert_array_equal(directions[1], directions[3])
    np.testing.assert_array_equal(directions[1], manifest["shared_pca_direction"])
    altered = copy.deepcopy(artifacts)
    altered["target_raw_prototypes"] = {c: p * [-3., 5.] for c,p in altered["target_raw_prototypes"].items()}
    altered_tc = dict(tc, mode13_features=tc["mode13_features"] * torch.tensor([-3.,5.]))
    second = generate_diagnostic_visualizations(tmp_path / "altered_target", "S_T", ["cropA","cropB"],
        sc,altered_tc,altered,[],max_curves=2,dpi=40)
    np.testing.assert_array_equal(manifest["shared_pca_direction"], second["shared_pca_direction"])


def test_phase_applicability_visualizations_are_fixed_triptychs(tmp_path):
    import json
    import scripts.diagnose_mode13_phase_shape_oracle as script
    grid, s, t = _viz_fixture()
    labels = np.repeat([0, 1], 5)
    source = np.concatenate([s, s + 1])
    target = np.concatenate([t, t + 1])
    sc = dict(mode13_features=torch.from_numpy(source), labels=labels, sample_id=torch.arange(10))
    tc = dict(mode13_features=torch.from_numpy(target), labels=labels, sample_id=torch.arange(100, 110))
    artifacts = dict(
        source_raw_prototypes={i: np.median(source[labels == i], 0) for i in (0, 1)},
        target_raw_prototypes={i: np.median(target[labels == i], 0) for i in (0, 1)},
        source_pca_direction_by_class={0: np.array([1., 0.]), 1: np.array([0., 1.])},
        scalar_delta_by_class={0: 0, 1: 0},
        gamma_by_class={0: np.linspace(0, 1, 128), 1: np.linspace(0, 1, 128)},
        selected_lambda={0: 0.1, 1: None},
        phase_applicability_by_class={0: "FULL_PHASE", 1: "PHASE_NOT_APPLICABLE"},
        common_support_by_class={0: {}, 1: {}}, partial_gamma_by_class={},
        partial_selected_lambda_by_class={})
    states = [
        dict(class_index="0", phase_applicability="FULL_PHASE", common_time_coverage_min="1.0",
             full_landmark_error="2", phase_applicability_reason="full"),
        dict(class_index="1", phase_applicability="PHASE_NOT_APPLICABLE", common_time_coverage_min="0.1",
             left_truncation_evidence="strong", right_truncation_evidence="none",
             phase_applicability_reason="topology mismatch"),
    ]
    manifest = script.generate_diagnostic_visualizations(
        tmp_path, "S_T", ["crop/A", "crop B"], sc, tc, artifacts, states, max_curves=2, dpi=40)
    root = tmp_path / "visualizations" / "phase_applicability"
    assert (root / "phase_applicability_cases.png").is_file()
    assert (root / "paper_style_triptychs/0_crop_A.png").is_file()
    assert (root / "paper_style_triptychs/1_crop_B.png").is_file()
    assert manifest["registration_space"] == "multivariate Mode13"
    assert manifest["visualization_space"] == "source_class_PC1"
    assert manifest["phase_level"] == "class_level_cross_domain"
    assert manifest["phase_applicability_representatives"]["PARTIAL_PHASE"] is None
    saved = json.loads((tmp_path / "visualizations/visualization_manifest.json").read_text())
    assert saved["phase_applicability_representatives"]["FULL_PHASE"]["class_index"] == 0


def test_phase_applicability_representative_selection_uses_frozen_priority():
    from scripts.diagnose_mode13_phase_shape_oracle import select_phase_applicability_representatives
    rows = [
        dict(task="Z", class_index=0, state="FULL_PHASE", common_coverage=.8,
             landmark_error=1., warp_days=1.),
        dict(task="A", class_index=1, state="FULL_PHASE", common_coverage=.9,
             landmark_error=8., warp_days=20.),
        dict(task="A", class_index=2, state="PARTIAL_PHASE", nonlinear_accepted=False,
             landmark_gain=99., common_coverage=.9),
        dict(task="Z", class_index=3, state="PARTIAL_PHASE", nonlinear_accepted=True,
             landmark_gain=2., common_coverage=.3),
        dict(task="A", class_index=4, state="PHASE_NOT_APPLICABLE", strong_truncation=False,
             common_coverage=0., topology_mismatch=9),
        dict(task="Z", class_index=5, state="PHASE_NOT_APPLICABLE", strong_truncation=True,
             common_coverage=.7, topology_mismatch=0),
    ]
    chosen = select_phase_applicability_representatives(rows)
    assert chosen["FULL_PHASE"]["class_index"] == 1
    assert chosen["PARTIAL_PHASE"]["class_index"] == 3
    assert chosen["PHASE_NOT_APPLICABLE"]["class_index"] == 5


def test_global_phase_applicability_figure_aggregates_task_manifests(tmp_path):
    import json
    from scripts.diagnose_mode13_phase_shape_oracle import generate_global_phase_applicability_figure
    for task, state in (("Z_T", "FULL_PHASE"), ("A_T", "PHASE_NOT_APPLICABLE"),
                        ("B_T", "FULL_PHASE"), ("C_T", "PHASE_NOT_APPLICABLE")):
        directory = tmp_path / task / "visualizations"
        directory.mkdir(parents=True)
        record = dict(task=task, class_index=0, class_name="crop", state=state,
            common_coverage=1., landmark_error=0., warp_days=0., nonlinear_accepted=False,
            landmark_gain=0., strong_truncation=state == "PHASE_NOT_APPLICABLE", topology_mismatch=0,
            reason="synthetic", grid=list(np.linspace(0, 365, 64, endpoint=False)),
            source=list(np.sin(np.linspace(0, 2*np.pi, 64))),
            target_before=list(np.sin(np.linspace(0, 2*np.pi, 64))),
            source_selected=list(np.sin(np.linspace(0, 2*np.pi, 64))),
            target_selected=list(np.sin(np.linspace(0, 2*np.pi, 64))),
            support=None,
            phase_x=list(np.linspace(0, 365, 128)) if state == "FULL_PHASE" else None,
            phase_y=list(np.linspace(0, 365, 128)) if state == "FULL_PHASE" else None,
            phase_identity=list(np.linspace(0, 365, 128)) if state == "FULL_PHASE" else None)
        (directory / "visualization_manifest.json").write_text(json.dumps(dict(
            phase_applicability_representatives={state: record})))
    selected = generate_global_phase_applicability_figure(tmp_path, dpi=40)
    output = tmp_path / "visualizations/phase_applicability/phase_applicability_representative_global.png"
    assert output.is_file()
    assert selected["FULL_PHASE"]["task"] == "B_T"
    assert selected["PARTIAL_PHASE"] is None
    assert selected["PHASE_NOT_APPLICABLE"]["task"] == "A_T"


def test_global_phase_applicability_requires_all_four_task_manifests(tmp_path):
    import pytest
    from scripts.diagnose_mode13_phase_shape_oracle import generate_global_phase_applicability_figure
    with pytest.raises(ValueError, match="exactly four"):
        generate_global_phase_applicability_figure(tmp_path, dpi=40)


def test_viz_ylim_includes_unselected_outlier_not_only_iqr():
    from analysis.phase_shape_diagnostic import prepare_visualization_group
    from scripts.diagnose_mode13_phase_shape_oracle import plot_visualization_spaghetti
    import matplotlib.pyplot as plt
    grid,s,t = _viz_fixture()
    s[2,:,0] = 1000
    data = prepare_visualization_group(s,t,np.arange(5),np.arange(5),np.median(s,0),np.median(t,0),
                                      np.array([1.,0.]),grid,0,"FULL_PHASE",max_curves=2)
    figure = plot_visualization_spaghetti(data,"S_T","crop",{})
    try:
        assert figure.axes[0].get_ylim()[1] > 1000
    finally:
        plt.close(figure)


@pytest.mark.parametrize("state", ["FULL_PHASE", "PARTIAL_PHASE"])
def test_viz_full_and_partial_plot_shading(state):
    from analysis.phase_shape_diagnostic import prepare_visualization_group
    from scripts.diagnose_mode13_phase_shape_oracle import plot_visualization_spaghetti
    import matplotlib.pyplot as plt
    grid,s,t = _viz_fixture()
    support = dict(source_common_start_day=100.,source_common_end_day=220.,
                   target_common_start_day=120.,target_common_end_day=260.)
    data = prepare_visualization_group(s,t,np.arange(5),np.arange(5),np.median(s,0),np.median(t,0),
        np.array([1.,0.]),grid,0,state,support=support,partial_gamma=np.linspace(0,1,128))
    figure = plot_visualization_spaghetti(data,"S_T","crop",{})
    try:
        assert len(figure.axes[2].patches) == (2 if state == "PARTIAL_PHASE" else 0)
        assert len(figure.axes[3].patches) == (2 if state == "PARTIAL_PHASE" else 0)
        assert len({ax.get_ylim() for ax in figure.axes}) == 1
    finally:
        plt.close(figure)


def test_viz_helpers_have_no_extraction_or_phase_estimation_dependency():
    import inspect
    from analysis.phase_shape_diagnostic import prepare_visualization_group
    from scripts.diagnose_mode13_phase_shape_oracle import generate_diagnostic_visualizations, main
    for helper in (prepare_visualization_group,generate_diagnostic_visualizations):
        parameters = inspect.signature(helper).parameters
        assert not any(k in parameters for k in ('model','dataloader','spatial_encoder','analyzer','synthesizer'))
        code = inspect.getsource(helper)
        assert not any(token in code for token in ('extract_cache(', 'constrained_residual_phase(',
                      'constrained_partial_phase(', 'fit_source_class_projections(', 'build_direct_fourier_views('))
    code = inspect.getsource(main)
    assert code.index('generate_diagnostic_visualizations(') > code.index('(args.output_dir / "metadata.json").write_text(')


def test_viz_empty_nonfinite_groups_are_explicit_unavailable_and_bugs_raise():
    from analysis.phase_shape_diagnostic import prepare_visualization_group, VisualizationUnavailable
    grid,s,t = _viz_fixture()
    with pytest.raises(VisualizationUnavailable,match="empty"):
        prepare_visualization_group(s[:0],t,[],np.arange(5),s[0],t[0],np.ones(2),grid,0,"FULL_PHASE")
    with pytest.raises(ValueError,match="sample IDs"):
        prepare_visualization_group(s,t,[],np.arange(5),s[0],t[0],np.ones(2),grid,0,"FULL_PHASE")
    s[0,0,0] = np.nan
    with pytest.raises(VisualizationUnavailable,match="nonfinite"):
        prepare_visualization_group(s,t,np.arange(5),np.arange(5),s[0],t[0],np.ones(2),grid,0,"FULL_PHASE")
