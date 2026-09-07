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
