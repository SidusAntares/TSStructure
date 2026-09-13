import importlib.util
from pathlib import Path

import numpy as np


def _gaussian(grid, center, width=8.0, amplitude=1.0):
    return amplitude * np.exp(-0.5 * ((grid - center) / width) ** 2)


def test_peak_and_valley_detection_and_principal_anchor():
    from analysis.recon_anchor_diagnostic import detect_salient_extrema

    grid = np.arange(365, dtype=np.float64)
    curve = _gaussian(grid, 90, 7, 2.0) - _gaussian(grid, 230, 9, 1.5)
    result = detect_salient_extrema(curve, grid, 7, 3, 0.20)
    assert abs(result.strongest_peak.day - 90) <= 1
    assert abs(result.strongest_valley.day - 230) <= 1
    assert result.strongest_peak.kind == "peak"
    assert result.strongest_valley.kind == "valley"
    assert result.principal_anchor is not None


def test_flat_and_near_flat_curves_have_no_anchor():
    from analysis.recon_anchor_diagnostic import detect_salient_extrema

    grid = np.arange(365, dtype=np.float64)
    for curve in (np.ones(365), 1e-14 * np.sin(grid / 3.0)):
        result = detect_salient_extrema(curve, grid, 7, 3, 0.20)
        assert result.principal_anchor is None
        assert result.status == "NO_SALIENT_ANCHOR"


def test_sample_matching_is_same_type_and_limited_to_local_window():
    from analysis.recon_anchor_diagnostic import Extremum, match_sample_anchor

    grid = np.arange(365, dtype=np.float64)
    curve = _gaussian(grid, 110, 5, 1.0) + _gaussian(grid, 260, 5, 5.0)
    anchor = Extremum("peak", 100.0, 1.0, 1.0, 6.0)
    match = match_sample_anchor(curve, grid, anchor, 30, 3, 0.05)
    assert match.matched
    assert abs(match.day - 110) <= 1
    assert abs(match.day - 260) > 100


def test_known_global_shift_recovery_uses_existing_coordinate_sign():
    from analysis.recon_anchor_diagnostic import Extremum, match_sample_anchor

    grid = np.arange(365, dtype=np.float64)
    target = _gaussian(grid, 195, 6)
    anchor = Extremum("peak", 180.0, 1.0, 1.0, 6.0)
    match = match_sample_anchor(
        target, grid, anchor, 30, 3, 0.05, calendar_shift_days=-15
    )
    assert match.matched
    assert abs(match.day - 180) <= 1
    assert match.absolute_error <= 1


def test_calendar_shift_drops_out_of_support_and_never_wraps():
    from analysis.recon_anchor_diagnostic import shifted_calendar

    grid = np.arange(365, dtype=np.float64)
    shifted, valid = shifted_calendar(grid, -20, 0, 364)
    assert shifted[0] == -20
    assert not valid[:20].any()
    assert valid[20:].all()
    assert not np.any(shifted[~valid] == shifted[valid][-1])


def test_occurrence_statistics_use_known_matched_and_unmatched_counts():
    from analysis.recon_anchor_diagnostic import AnchorMatch, summarize_matches

    matches = [
        AnchorMatch(True, 99.0, 1.0, 0.7, 1.0),
        AnchorMatch(True, 104.0, 4.0, 0.8, 1.0),
        AnchorMatch(False, np.nan, np.nan, np.nan, np.nan),
        AnchorMatch(False, np.nan, np.nan, np.nan, np.nan),
    ]
    summary = summarize_matches(matches)
    assert summary["match_count"] == 2
    assert summary["occurrence_rate"] == 0.5
    assert summary["timing_error_median"] == 2.5


def test_normalized_prominence_is_amplitude_invariant():
    from analysis.recon_anchor_diagnostic import detect_salient_extrema

    grid = np.arange(365, dtype=np.float64)
    curve = _gaussian(grid, 170, 11) - 0.3 * _gaussian(grid, 250, 8)
    a = detect_salient_extrema(curve, grid, 7, 3, 0.10).strongest_peak
    b = detect_salient_extrema(17.0 * curve, grid, 7, 3, 0.10).strongest_peak
    np.testing.assert_allclose(a.normalized_prominence, b.normalized_prominence)


def test_anchor_centered_window_correlation_is_high_after_translation():
    from analysis.recon_anchor_diagnostic import local_scalar_correlation

    grid = np.arange(365, dtype=np.float64)
    source = _gaussian(grid, 150, 8) - 0.4 * _gaussian(grid, 165, 4)
    target = _gaussian(grid, 178, 8) - 0.4 * _gaussian(grid, 193, 4)
    corr = local_scalar_correlation(source, grid, 150, target, grid, 178, 30)
    assert corr > 0.999


def test_multivariate_window_correlation_is_channel_robust():
    from analysis.recon_anchor_diagnostic import local_multivariate_correlation

    u = np.arange(-30, 31, dtype=np.float64)
    source = np.column_stack((np.sin(u / 8), 4 * np.cos(u / 11), np.ones_like(u)))
    target = np.column_stack((5 * np.sin(u / 8) + 10, np.cos(u / 11) - 3, np.ones_like(u)))
    corr, valid = local_multivariate_correlation(source, target)
    assert valid == 2
    assert corr > 0.999


def test_low_mode_simplifies_high_frequency_shoulders():
    from analysis.recon_anchor_diagnostic import detect_salient_extrema

    grid = np.arange(365, dtype=np.float64)
    low = _gaussian(grid, 180, 25)
    high = low + 0.45 * np.sin(2 * np.pi * 16 * grid / 365) * _gaussian(grid, 180, 45)
    low_count = len(detect_salient_extrema(low, grid, 4, 2, 0.08).extrema)
    high_count = len(detect_salient_extrema(high, grid, 4, 2, 0.08).extrema)
    assert high_count > low_count


def test_meadow_like_broad_weak_structure_is_not_stable():
    from analysis.recon_anchor_diagnostic import classify_source_anchor, detect_salient_extrema

    grid = np.arange(365, dtype=np.float64)
    curve = 1e-14 * (_gaussian(grid, 100, 55) + _gaussian(grid, 260, 60))
    detected = detect_salient_extrema(curve, grid, 7, 3, 0.20)
    status = classify_source_anchor(detected.principal_anchor, 1.0, 0.0, 0.20)
    assert status == "NO_SALIENT_ANCHOR"


def test_modes_are_fit_independently_not_truncated_from_mode13():
    from analysis.recon_anchor_diagnostic import fit_fourier_modes_independently

    calls = []

    class FakeAnalyzer:
        def __init__(self, mode):
            self.mode = mode

        def __call__(self, features, positions, collect_diagnostics=False):
            calls.append((self.mode, id(features), id(positions)))
            return np.full((features.shape[0], self.mode, features.shape[2]), self.mode), {}

    features = np.zeros((2, 8, 3))
    positions = np.zeros((2, 8))
    result = fit_fourier_modes_independently(
        features, positions, (7, 9, 11, 13), lambda mode: FakeAnalyzer(mode)
    )
    assert [call[0] for call in calls] == [7, 9, 11, 13]
    assert [result[mode].shape[1] for mode in (7, 9, 11, 13)] == [7, 9, 11, 13]


def test_one_projection_object_is_reused_across_all_modes_and_target_is_transform_only():
    from analysis.recon_anchor_diagnostic import project_modes_with_fixed_pc1
    from analysis.shift_visualization import fit_source_class_pc1

    rng = np.random.default_rng(3)
    source_raw = rng.normal(size=(4, 12, 3))
    projection = fit_source_class_pc1(source_raw)
    target_modes = {mode: rng.normal(size=(2, 20, 3)) for mode in (7, 9, 11, 13)}
    projected = project_modes_with_fixed_pc1(target_modes, projection)
    for mode, curves in target_modes.items():
        np.testing.assert_allclose(curves @ projection.axis - projection.center @ projection.axis, projected[mode])
    shifted_target = {mode: value + 1000 for mode, value in target_modes.items()}
    project_modes_with_fixed_pc1(shifted_target, projection)
    np.testing.assert_allclose(projection.center, fit_source_class_pc1(source_raw).center)


def test_authoritative_shift_reader_requires_manifest_and_agrees_with_csv():
    from analysis.recon_anchor_diagnostic import read_authoritative_global_shift
    import shutil
    import uuid

    root = Path.cwd() / f"recon-anchor-shift-{uuid.uuid4().hex}"
    try:
        task = root / "AT1_DK1"
        (task / "03_reconshift13_shift").mkdir(parents=True)
        (task / "manifest.json").write_text(
            '{"reconshift13_shift":{"shift_days":-15,"source_path":"x.log"}}',
            encoding="utf-8",
        )
        (task / "shifts_summary.csv").write_text(
            "class_id,reconshift_shift_days\n0,-15\n1,-15\n", encoding="utf-8"
        )
        selection = read_authoritative_global_shift(root, "AT1_DK1")
        assert selection.shift_days == -15
        assert selection.source_path.endswith("manifest.json")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cli_is_offline_oracle_only_and_has_no_training_side_effects():
    path = Path("scripts/diagnose_recon_anchor_modes_4tasks_seed1.py")
    text = path.read_text(encoding="utf-8")
    forbidden = (
        "optimizer",
        ".backward(",
        "import train",
        "from train",
        "timematch",
        "requests",
        "urllib",
        "subprocess",
        "git ",
        "pip install",
    )
    assert all(token not in text for token in forbidden)
    assert "oracle_offline_grouping_only" in text
    spec = importlib.util.spec_from_file_location("recon_anchor_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_launcher_has_exact_tasks_gpus_and_no_network_or_git():
    text = Path("scripts/run_recon_anchor_diagnostic_4tasks_seed1.sh").read_text(
        encoding="utf-8"
    )
    for gpu, task in enumerate(("AT1 DK1", "DK1 FR1", "FR1 FR2", "FR2 AT1")):
        assert f'run_task {gpu} {task}' in text
    for forbidden in ("git ", "curl ", "wget ", "pip install", "conda install"):
        assert forbidden not in text


def test_domain_baseline_uses_all_domain_samples_under_one_projection():
    from analysis.recon_anchor_diagnostic import domain_projection_baseline

    values = np.array([[[0.0], [1.0]], [[10.0], [11.0]], [[100.0], [101.0]]])
    baseline = domain_projection_baseline(values, np.array([1.0]), np.array([0.0]))
    np.testing.assert_allclose(baseline.median, 10.5)
    assert baseline.iqr > 9.0


def test_domain_elevation_can_reject_internally_prominent_low_curve():
    from analysis.recon_anchor_diagnostic import DomainProjectionBaseline, Extremum, anchor_domain_elevation, select_gated_anchor

    grid = np.arange(5.0)
    curve = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    anchor = Extremum("peak", 2.0, 1.0, 1.0, 1.0)
    baseline = DomainProjectionBaseline(10.0, 4.0)
    assert anchor_domain_elevation(curve, grid, anchor, baseline) < 0
    selected, reason = select_gated_anchor((anchor,), curve, grid, baseline, .2, .75)
    assert selected is None
    assert reason == "low_domain_relative_elevation"


def test_anchor_warp_direction_and_three_fixed_points():
    from analysis.recon_anchor_diagnostic import build_anchor_time_maps

    warp = build_anchor_time_maps(130.0, 190.0, 170.0, 160.0)
    np.testing.assert_allclose(warp.forward(np.array([130.0, 170.0, 190.0])), [130, 160, 190])
    np.testing.assert_allclose(warp.query(np.array([130.0, 160.0, 190.0])), [130, 170, 190])
    assert warp.valid


def test_anchor_warp_is_strict_and_identity_outside_window():
    from analysis.recon_anchor_diagnostic import build_anchor_time_maps

    warp = build_anchor_time_maps(130, 190, 170, 160)
    grid = np.arange(365.0)
    mapped = warp.forward(grid)
    assert np.all(np.diff(mapped) > 0)
    np.testing.assert_array_equal(mapped[(grid < 130) | (grid > 190)], grid[(grid < 130) | (grid > 190)])


def test_composite_query_includes_anchor_query_and_residual_gamma():
    from analysis.recon_anchor_diagnostic import build_anchor_time_maps, compose_local_query, invert_monotone_map

    warp = build_anchor_time_maps(130, 190, 170, 160)
    y = np.linspace(130, 190, 121)
    gamma = y + 3.0 * np.sin(2 * np.pi * (y - 130) / 60.0)
    query = compose_local_query(y, warp, gamma)
    np.testing.assert_allclose(query, warp.query(gamma))
    x = invert_monotone_map(y, query, query)
    np.testing.assert_allclose(x, y, atol=1e-8)


def test_residual_gamma_fixes_window_and_anchor():
    from analysis.recon_anchor_diagnostic import stitch_half_phase_gammas

    left = np.linspace(0, 1, 128) ** 1.2
    right = np.linspace(0, 1, 128) ** .8
    days, gamma = stitch_half_phase_gammas(130, 160, 190, left, right)
    assert gamma[0] == 130 and gamma[-1] == 190
    assert gamma[np.argmin(abs(days - 160))] == 160
    assert np.all(np.diff(gamma) > 0)


def test_raw_time_preview_keeps_outside_identity_and_order():
    from analysis.recon_anchor_diagnostic import build_anchor_time_maps, apply_local_forward_to_timestamps

    warp = build_anchor_time_maps(130, 190, 170, 160)
    timestamps = np.array([20., 129., 140., 170., 180., 191., 300.])
    mapped = apply_local_forward_to_timestamps(timestamps, warp)
    np.testing.assert_array_equal(mapped[[0, 1, 5, 6]], timestamps[[0, 1, 5, 6]])
    assert np.all(np.diff(mapped) > 0)


def test_anchor_sampling_recovers_a_pure_translation():
    from analysis.recon_anchor_diagnostic import build_anchor_time_maps, sample_curve_with_query

    days = np.arange(130.0, 191.0)
    warp = build_anchor_time_maps(130, 190, 170, 160)
    source = _gaussian(days, 160, 7)[:, None]
    # Generate the target through the exact inverse of the boundary-fixed map.
    target = _gaussian(warp.forward(days), 160, 7)[:, None]
    aligned = sample_curve_with_query(target, days, warp.query(days))
    assert np.corrcoef(source[:, 0], aligned[:, 0])[0, 1] > .99


def test_local_phase_calls_two_multivariate_halves_and_fixes_anchor(monkeypatch):
    from analysis.phase_shape_diagnostic import PhaseEstimate
    import analysis.phase_shape_diagnostic as phase_module
    from analysis.recon_anchor_diagnostic import estimate_anchor_fixed_local_phase

    calls = []
    identity = np.linspace(0, 1, 128)

    def fake(source, target, k_reg=128):
        calls.append((source.shape, target.shape, k_reg))
        return PhaseEstimate(identity, True, "", 0, 0, 0, 1, 1, 0)

    monkeypatch.setattr(phase_module, "estimate_nonlinear_phase", fake)
    days = np.arange(130.0, 191.0)
    source = np.column_stack((np.sin(days / 9), np.cos(days / 13), days / 365))
    result = estimate_anchor_fixed_local_phase(source, source.copy(), days, 160)
    assert result.valid
    assert calls == [((31, 3), (31, 3), 128), ((31, 3), (31, 3), 128)]
    np.testing.assert_allclose(result.residual_query_days[[0, 30, -1]], [130, 160, 190])


def test_invalid_half_phase_falls_back_to_anchor_state(monkeypatch):
    from analysis.phase_shape_diagnostic import PhaseEstimate
    import analysis.phase_shape_diagnostic as phase_module
    from analysis.recon_anchor_diagnostic import estimate_anchor_fixed_local_phase

    identity = np.linspace(0, 1, 128)
    invalid = PhaseEstimate(identity, False, "solver_failure", 0, 0, 0, 1, 1, 0)
    monkeypatch.setattr(phase_module, "estimate_nonlinear_phase", lambda *args, **kwargs: invalid)
    days = np.arange(130.0, 191.0)
    curve = np.column_stack((np.sin(days), np.cos(days)))
    result = estimate_anchor_fixed_local_phase(curve, curve, days, 160)
    assert not result.valid
    np.testing.assert_array_equal(result.residual_query_days, days)
    assert "solver_failure" in result.failure_reason


def test_multivariate_srvf_distance_is_zero_for_identical_curves():
    from analysis.recon_anchor_diagnostic import multivariate_srvf_distance

    days = np.linspace(0, 1, 61)
    curve = np.column_stack((np.sin(2 * np.pi * days), np.cos(np.pi * days)))
    np.testing.assert_allclose(multivariate_srvf_distance(curve, curve), 0.0, atol=1e-12)


def test_prepared_srvf_reference_is_numerically_equivalent():
    from analysis.recon_anchor_diagnostic import (
        multivariate_srvf_distance,
        multivariate_srvf_distance_from_reference,
        prepare_multivariate_srvf_reference,
    )

    days = np.linspace(0, 1, 61)
    source = np.column_stack((np.sin(2 * np.pi * days), np.cos(np.pi * days)))
    target = np.column_stack((np.sin(2 * np.pi * days + .15), np.cos(np.pi * days - .1)))
    prepared = prepare_multivariate_srvf_reference(source)
    np.testing.assert_allclose(
        multivariate_srvf_distance_from_reference(prepared, target),
        multivariate_srvf_distance(source, target),
        rtol=1e-12,
        atol=1e-12,
    )


def test_batch_multivariate_reconstruction_reduces_synthesizer_calls():
    import torch
    from scripts.diagnose_recon13_local_anchor_phase_4tasks_seed1 import reconstruct_multivariate_batch

    class FakeSynthesizer:
        def __init__(self):
            self.calls = 0

        def __call__(self, coefficients, positions):
            self.calls += 1
            value = coefficients.real.sum(dim=1)[:, None, :]
            return value + positions[:, :, None]

    coefficients = np.arange(5 * 3 * 2).reshape(5, 3, 2).astype(np.complex64)
    positions = np.arange(5 * 4).reshape(5, 4).astype(np.float32)
    synthesizer = FakeSynthesizer()
    result = reconstruct_multivariate_batch(
        coefficients, positions, synthesizer, torch.device("cpu"), batch_size=2
    )
    expected = coefficients.real.sum(axis=1)[:, None, :] + positions[:, :, None]
    np.testing.assert_allclose(result, expected)
    assert synthesizer.calls == 3


def test_parallel_local_phase_batch_preserves_input_order(monkeypatch):
    import analysis.recon_anchor_diagnostic as module
    from analysis.recon_anchor_diagnostic import LocalPhaseResult, estimate_anchor_fixed_local_phases

    days = np.arange(5.0)

    prepared = object()

    def fake(_prepared, target):
        marker = float(target[0, 0])
        return LocalPhaseResult(days, days + marker, True, "", marker, 1, 1, 1, False)

    monkeypatch.setattr(
        module, "prepare_anchor_fixed_local_phase_reference", lambda *args: prepared
    )
    monkeypatch.setattr(module, "estimate_anchor_fixed_local_phase_prepared", fake)
    source = np.zeros((5, 2))
    targets = np.stack([np.full((5, 2), marker) for marker in (3.0, 1.0, 2.0)])
    results = estimate_anchor_fixed_local_phases(source, targets, days, 2.0, workers=2)
    assert [item.max_displacement_days for item in results] == [3.0, 1.0, 2.0]


def test_prepared_anchor_phase_matches_reference_path():
    from analysis.recon_anchor_diagnostic import (
        estimate_anchor_fixed_local_phase,
        estimate_anchor_fixed_local_phase_prepared,
        prepare_anchor_fixed_local_phase_reference,
    )

    days = np.arange(130.0, 191.0)
    source = np.column_stack((np.sin(days / 9), np.cos(days / 13), days / 365))
    target = source.copy()
    reference = estimate_anchor_fixed_local_phase(source, target, days, 160)
    prepared = prepare_anchor_fixed_local_phase_reference(source, days, 160)
    optimized = estimate_anchor_fixed_local_phase_prepared(prepared, target)
    assert optimized.valid == reference.valid
    np.testing.assert_allclose(optimized.residual_query_days, reference.residual_query_days)


def test_local_phase_launcher_is_fixed_mode13_offline_and_four_tasks():
    cli = Path("scripts/diagnose_recon13_local_anchor_phase_4tasks_seed1.py")
    launcher = Path("scripts/run_recon13_local_anchor_phase_diagnostic_4tasks_seed1.sh")
    assert cli.is_file() and launcher.is_file()
    cli_text = cli.read_text(encoding="utf-8")
    launch_text = launcher.read_text(encoding="utf-8")
    assert "NUM_MODES = 13" in cli_text
    assert "oracle_offline_grouping_only" in cli_text
    assert "SRVF_LAMBDA = 0" in cli_text
    for gpu, task in enumerate(("AT1 DK1", "DK1 FR1", "FR1 FR2", "FR2 AT1")):
        assert f"run_task {gpu} {task}" in launch_text
    forbidden = ("git ", "curl ", "wget ", "pip install", "optimizer", ".backward(")
    assert all(token not in cli_text + launch_text for token in forbidden)
    spec = importlib.util.spec_from_file_location("recon13_local_phase_cli", cli)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    assert callable(module.normalized_l2)
    defaults = module.build_parser().parse_args(["--data-root", "local-data"])
    assert defaults.shift_visualization_root == Path("outputs/fredn/shift_visualizations_seed1")
    assert defaults.output_root == Path("outputs/fredn/shift_visualizations_seed1")
    assert defaults.max_examples == 0
    assert defaults.max_spaghetti == 40
    assert "04_oracle_samplewise_anchor_nonlinear" in cli_text


def test_batched_monotone_alignment_recovers_internal_phase_and_preserves_contract():
    from analysis.recon_anchor_diagnostic import estimate_batched_window_phase, sample_curve_with_query

    days = np.arange(130., 191.)
    u = (days - 130) / 60
    source = np.column_stack((np.sin(2*np.pi*u), np.cos(3*np.pi*u), np.sin(5*np.pi*u)))
    # A known source-output -> target-input map fixing both endpoints and the anchor.
    query = days + 3 * np.sin(2*np.pi*u)
    target = np.column_stack([np.interp(days, query, source[:, d]) for d in range(3)])
    result, identity = estimate_batched_window_phase(
        source, np.stack((target, source)), days, 160, device="cpu", steps=100,
    )
    assert result.valid and identity.valid
    assert np.all(np.diff(result.residual_query_days) > 0)
    np.testing.assert_allclose(result.residual_query_days[[0, 30, -1]], days[[0, 30, -1]], atol=1e-5)
    aligned = sample_curve_with_query(target, days, result.residual_query_days)
    assert np.mean((aligned-source)**2) < .15 * np.mean((target-source)**2)
    assert np.max(abs(result.residual_query_days-query)) < 1.5
    np.testing.assert_allclose(identity.residual_query_days, days, atol=1e-5)


def test_batched_alignment_handles_flat_nonfinite_and_does_not_change_rng():
    import torch
    from analysis.recon_anchor_diagnostic import estimate_batched_window_phase

    days = np.arange(61.)
    source = np.column_stack((np.sin(days/8), np.cos(days/9)))
    targets = np.stack((source, np.zeros_like(source), source))
    targets[2, 5, 1] = np.nan
    state = torch.get_rng_state().clone()
    results = estimate_batched_window_phase(source, targets, days, 30, steps=5)
    assert torch.equal(state, torch.get_rng_state())
    assert results[0].valid
    for result in results[1:]:
        assert not result.valid and result.failure_reason
        np.testing.assert_array_equal(result.residual_query_days, days)


def test_fast_projected_reconstruction_preserves_whole_domain_baseline():
    import torch
    from analysis.shift_visualization import SourceClassPC1
    from scripts.diagnose_recon_anchor_modes_4tasks_seed1 import reconstruct_projected_curves
    from scripts.diagnose_recon13_local_anchor_phase_4tasks_seed1 import reconstruct_projected_fast

    rng = np.random.default_rng(3)
    coefficients = (rng.normal(size=(7,13,4)) + 1j*rng.normal(size=(7,13,4))).astype(np.complex64)
    projection = SourceClassPC1(rng.normal(size=4), rng.normal(size=4))
    expected = reconstruct_projected_curves(coefficients, projection, 13, torch.device("cpu"), 3)
    actual = reconstruct_projected_fast(coefficients, projection)
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=2e-5)


def test_vectorized_window_metrics_match_reference():
    from analysis.recon_anchor_diagnostic import window_metrics_batch, local_multivariate_correlation, multivariate_srvf_distance
    from analysis.phase_shape_diagnostic import normalized_l2
    rng = np.random.default_rng(4)
    source = rng.normal(size=(61,5))
    targets = rng.normal(size=(4,61,5))
    source[:, -1] = 2
    targets[:, :, -1] = 3
    corr, srvf, distance = window_metrics_batch(source, targets)
    for i, target in enumerate(targets):
        np.testing.assert_allclose(corr[i], local_multivariate_correlation(source, target)[0], atol=1e-12)
        np.testing.assert_allclose(srvf[i], multivariate_srvf_distance(source, target), atol=1e-12)
        np.testing.assert_allclose(distance[i], normalized_l2(source, target), atol=1e-12)


def test_pca_statistics_use_original_interpolation_without_dense_day_features():
    from scripts.diagnose_recon13_local_anchor_phase_4tasks_seed1 import interpolation_statistics
    from analysis.shift_visualization import interpolate_latent
    rng = np.random.default_rng(9)
    positions = np.array([20., 0., 20., 180., 340.])
    features = rng.normal(size=(5,4))
    grid = np.arange(365.)
    total, cross = interpolation_statistics(features, positions, grid)
    dense = interpolate_latent(positions, features, grid)
    np.testing.assert_allclose(total, dense.sum(0), atol=1e-10)
    np.testing.assert_allclose(cross, dense.T @ dense, atol=1e-10)


def test_source_pca_and_fourier_extraction_share_one_spatial_pass(monkeypatch):
    import torch
    import scripts.diagnose_recon13_local_anchor_phase_4tasks_seed1 as cli
    from scripts.visualize_shift_configs_4tasks import fit_class_projections
    import scripts.visualize_shift_configs_4tasks as original
    rng = np.random.default_rng(21)
    features = torch.tensor(rng.normal(size=(3, 15, 4)), dtype=torch.float32)
    sample = {"pixels": features, "valid_pixels": torch.ones(3,15),
              "positions": torch.arange(15)[None].expand(3,-1)*20,
              "label": torch.tensor([0,1,0]), "parcel_index": torch.arange(3)}
    calls = []
    def spatial(pixels, mask, extra):
        calls.append(1)
        return pixels
    monkeypatch.setattr(cli, "_loader", lambda *args: [sample])
    monkeypatch.setattr(original, "_loader", lambda *args: [sample])
    reference, _ = fit_class_projections(spatial, None, 2, np.arange(365), 3, "cpu", False)
    calls.clear()
    result = cli.extract_mode13_cache(spatial, None, 3, "cpu", False, fit_source_pca=True)
    assert len(calls) == 1
    for key, projection in result["projections"].items():
        np.testing.assert_allclose(projection.axis, reference[key].axis, atol=1e-10)
        np.testing.assert_allclose(projection.center, reference[key].center, atol=1e-10)


def test_fast_prototype_is_pointwise_median_not_mean():
    import torch
    from models.fourier_reconstruction import BatchedDirectFourierSynthesizer
    from scripts.visualize_shift_configs_4tasks import reconstruct_class_prototype
    from scripts.diagnose_recon13_local_anchor_phase_4tasks_seed1 import reconstruct_prototype_fast
    rng = np.random.default_rng(31)
    coefficients = (rng.normal(size=(7,13,4))+1j*rng.normal(size=(7,13,4))).astype(np.complex64)
    coefficients[0] *= 30
    expected = reconstruct_class_prototype(coefficients, BatchedDirectFourierSynthesizer(13), "cpu", 3, 16)
    actual = reconstruct_prototype_fast(coefficients, "cpu", memory_mb=1)
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=3e-5)


def test_feature_plot_shows_actual_same_channels_and_shared_scales():
    import matplotlib.pyplot as plt
    from scripts.diagnose_recon13_local_anchor_phase_4tasks_seed1 import _channel_figure
    days = np.arange(61.)
    source = np.column_stack((np.sin(days/7), 2*np.cos(days/8), .2*np.sin(days/4)))
    curves = (source+.1, source+.2, source+.3)
    fig, channels = _channel_figure(days, source, curves, "synthetic", max_channels=2)
    try:
        assert len(fig.axes) == 8
        for row, channel in enumerate(channels):
            axes = fig.axes[row*4:row*4+4]
            assert len({ax.get_ylim() for ax in axes}) == 1
            for stage in range(3):
                np.testing.assert_array_equal(axes[stage].lines[1].get_ydata(), curves[stage][:,channel])
    finally:
        plt.close(fig)


def test_batched_phase_is_independent_of_batch_partition_and_composes_correctly():
    from analysis.recon_anchor_diagnostic import (
        estimate_batched_window_phase, build_anchor_time_maps,
        compose_local_query, apply_local_forward_to_timestamps,
    )
    days = np.arange(130.,191.)
    source = np.column_stack((np.sin(days/7),np.cos(days/11)))
    targets = np.stack((np.roll(source,1,axis=0),np.roll(source,-1,axis=0)))
    together = estimate_batched_window_phase(source,targets,days,160,steps=20)
    for target, actual in zip(targets,together):
        alone = estimate_batched_window_phase(source,target[None],days,160,steps=20)[0]
        np.testing.assert_allclose(actual.residual_query_days,alone.residual_query_days,atol=2e-4)
        maps = build_anchor_time_maps(130,190,170,160)
        query = compose_local_query(days,maps,actual.residual_query_days)
        forward = apply_local_forward_to_timestamps(query,maps,days,actual.residual_query_days)
        np.testing.assert_allclose(forward,days,atol=1e-8)
        outside = np.array([20.,125.,200.,300.])
        np.testing.assert_array_equal(apply_local_forward_to_timestamps(outside,maps,days,actual.residual_query_days),outside)


def test_fast_full_task_synthetic_smoke_emits_curves_metrics_and_manifest(monkeypatch, tmp_path):
    import json
    import tempfile
    import torch
    from types import SimpleNamespace
    from analysis.recon_anchor_diagnostic import GlobalShiftSelection
    import scripts.diagnose_recon13_local_anchor_phase_4tasks_seed1 as cli
    days = torch.linspace(0,364,31)
    def sample(shift):
        phase = 2*torch.pi*(days-68.75-shift)/365
        features = torch.stack((3*torch.sin(phase),torch.cos(2*phase),.4*torch.sin(3*phase)),dim=-1)
        return {"pixels": features[None].repeat(3,1,1), "valid_pixels":torch.ones(3,31),
                "positions":days[None].repeat(3,1), "label":torch.zeros(3,dtype=torch.long),
                "parcel_index":torch.arange(3)}
    class Dataset:
        def __init__(self,shift): self.sample = sample(shift)
        def __len__(self): return 3
    source_set,target_set = Dataset(0),Dataset(8)
    calls = []
    def spatial(pixels,mask,extra):
        calls.append(len(pixels))
        return pixels
    monkeypatch.setattr(cli,"_loader",lambda dataset,*args:[dataset.sample])
    monkeypatch.setattr(cli,"load_raw_spatial_encoder",lambda *args:spatial)
    monkeypatch.setattr(cli,"resolve_source_checkpoint",lambda *args:(Path("synthetic.pt"),{}))
    monkeypatch.setattr(cli,"load_classes",lambda *args:["crop"])
    monkeypatch.setattr(cli,"build_split_datasets",lambda *args:(source_set,target_set))
    monkeypatch.setattr(cli,"read_authoritative_global_shift",lambda *args:GlobalShiftSelection(0.,"synthetic","test"))
    directory = tmp_path
    task_root = directory/"AT1_DK1"
    task_root.mkdir()
    original_manifest = json.dumps({
        "class_outputs": {"0": {"raw": {"ylim": [-4.0, 4.0]}}},
        "sentinel": "must-not-be-overwritten",
    })
    (task_root/"manifest.json").write_text(original_manifest, encoding="utf-8")
    args = cli.build_parser().parse_args(["--data-root",str(directory),"--output-root",str(directory),
                                         "--shift-visualization-root",str(directory),
                                         "--device","cpu","--phase-steps","8","--max-examples","1",
                                         "--min-domain-relative-elevation","0"])
    cli.run_task(args,"AT1","DK1",{})
    output = task_root/"04_oracle_samplewise_anchor_nonlinear"
    manifest = json.loads((output/"manifest.json").read_text())
    assert (task_root/"manifest.json").read_text(encoding="utf-8") == original_manifest
    assert calls == [3,3]
    assert manifest["phase_solver"] == "batched_monotone"
    assert manifest["source_count"] == manifest["target_count"] == 3
    import csv
    with (output/"sample_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 and all(row["anchor_found"] == "True" for row in rows)
    assert all(np.isfinite(float(row["corr_nonlinear"])) for row in rows)
    assert all(float(row["corr_nonlinear"])+1e-6 >= float(row["corr_anchor"]) for row in rows)
    assert (output/"00_crop.png").is_file()
    for stage in ("01_global_only", "02_anchor_aligned", "03_local_nonlinear_aligned"):
        assert (output/"full_year_alignment"/stage/"00_crop.png").is_file()
    assert list(output.glob("diagnostics/class_*/feature_examples/*_channels.png"))
    assert list(output.glob("diagnostics/class_*/feature_examples/*_heatmap.png"))
    assert list(output.glob("diagnostics/class_*/feature_examples/*_curves.npz"))


def test_full_year_queries_change_only_the_anchor_window():
    from analysis.recon_anchor_diagnostic import build_anchor_time_maps
    from scripts.diagnose_recon13_local_anchor_phase_4tasks_seed1 import full_year_queries
    grid = np.arange(365.)
    days = np.arange(130.,191.)
    maps = build_anchor_time_maps(130,190,170,160)
    gamma = days + 2*np.sin(2*np.pi*(days-130)/60)
    global_query, anchor_query, nonlinear_query = full_year_queries(
        grid, global_shift=-5, anchor_maps=maps, window_days=days,
        residual_query_days=gamma,
    )
    outside = (grid < 130) | (grid > 190)
    np.testing.assert_array_equal(anchor_query[outside], global_query[outside])
    np.testing.assert_array_equal(nonlinear_query[outside], global_query[outside])
    np.testing.assert_allclose(anchor_query[160], 170+5)
    np.testing.assert_allclose(nonlinear_query[130], anchor_query[130])
    np.testing.assert_allclose(nonlinear_query[160], anchor_query[160])
    np.testing.assert_allclose(nonlinear_query[190], anchor_query[190])


def test_full_year_figures_match_shift_visualization_layout_and_share_raw_ylim(tmp_path):
    import matplotlib.pyplot as plt
    from scripts.diagnose_recon13_local_anchor_phase_4tasks_seed1 import render_full_year_alignment_figures
    rng = np.random.default_rng(44)
    grid = np.arange(365.)
    source = rng.normal(size=(8,365)).cumsum(1)/10
    stages = {
        "global_only": rng.normal(size=(7,365)).cumsum(1)/10,
        "anchor_aligned": rng.normal(size=(7,365)).cumsum(1)/10,
        "local_nonlinear_aligned": rng.normal(size=(7,365)).cumsum(1)/10,
    }
    metadata = render_full_year_alignment_figures(
        tmp_path, "AT1_DK1", 0, "corn", grid, source, stages,
        global_shift=-5, raw_ylim=(-4,18), max_spaghetti=4, seed=1,
    )
    assert set(metadata) == set(stages)
    for key, record in metadata.items():
        path = Path(record["path"])
        assert path.name == "00_corn.png" and path.is_file()
        assert record["ylim"] == [-4.0,18.0]
        assert record["source_count"] == 8 and record["target_count"] == 7
        assert np.isfinite(record["median_profile_correlation"])
        assert np.isfinite(record["median_profile_normalized_l2"])
    assert "01_global_only" in metadata["global_only"]["path"]
    assert "02_anchor_aligned" in metadata["anchor_aligned"]["path"]
    assert "03_local_nonlinear_aligned" in metadata["local_nonlinear_aligned"]["path"]
    plt.close("all")


def test_projected_query_reconstruction_matches_scalar_fourier_definition():
    import torch
    from analysis.shift_visualization import SourceClassPC1
    from scripts.diagnose_recon13_local_anchor_phase_4tasks_seed1 import reconstruct_projected_queries
    rng = np.random.default_rng(45)
    coeff = (rng.normal(size=(5,13,4))+1j*rng.normal(size=(5,13,4))).astype(np.complex64)
    projection = SourceClassPC1(rng.normal(size=4),rng.normal(size=4))
    queries = rng.uniform(-20,385,size=(5,365)).astype(np.float32)
    actual = reconstruct_projected_queries(coeff,projection,queries,torch.device("cpu"),batch_size=2)
    modes = np.arange(-6,7)
    points = (2*np.pi*queries/365+np.pi)%(2*np.pi)-np.pi
    projected = np.einsum("nfd,d->nf",coeff,projection.axis)
    expected = np.einsum("ntf,nf->nt",np.exp(1j*points[...,None]*modes),projected).real-projection.center@projection.axis
    np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=2e-5)


def test_joint_anchor_window_uses_shared_calendar_and_support_gate():
    from analysis.recon_anchor_diagnostic import build_joint_anchor_window

    window = build_joint_anchor_window(
        source_anchor_day=150,
        target_anchor_day=160,
        source_support=(100, 220),
        target_support=(110, 210),
        margin_days=20,
        max_anchor_distance_days=20,
        min_anchor_side_support_days=15,
    )
    assert window.valid
    assert (window.start_day, window.end_day, window.width_days) == (130, 180, 50)

    too_far = build_joint_anchor_window(150, 171, (0, 364), (0, 364))
    assert not too_far.valid
    assert too_far.failure_reason == "LOCAL_INELIGIBLE_ANCHOR_TOO_FAR"

    incomplete = build_joint_anchor_window(12, 18, (0, 364), (5, 364))
    assert not incomplete.valid
    assert incomplete.failure_reason == "LOCAL_INELIGIBLE_INCOMPLETE_ANCHOR_SUPPORT"


def test_whole_window_phase_calls_estimator_once_without_anchor_constraint(monkeypatch):
    import pytest
    from types import SimpleNamespace
    from analysis import phase_shape_diagnostic as phase_module
    from analysis.recon_anchor_diagnostic import estimate_whole_window_local_phase

    days = np.linspace(130.0, 180.0, 51)
    source = np.column_stack((np.sin(days / 8), np.cos(days / 11)))
    target = np.column_stack((np.sin((days - 7) / 8), np.cos((days - 7) / 11)))
    source_before, target_before = source.copy(), target.copy()
    gamma = np.linspace(0, 1, 128) ** 1.12
    calls = []

    def fake(left, right, k_reg=128):
        calls.append((left.copy(), right.copy(), k_reg))
        return SimpleNamespace(gamma=gamma, valid=True, failure_reason="")

    monkeypatch.setattr(phase_module, "estimate_nonlinear_phase", fake)
    result = estimate_whole_window_local_phase(source, target, days)
    assert result.valid
    assert len(calls) == 1 and calls[0][2] == 128
    np.testing.assert_array_equal(calls[0][0], source_before)
    np.testing.assert_array_equal(calls[0][1], target_before)
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(target, target_before)
    assert result.query_days[0] == pytest.approx(130)
    assert result.query_days[-1] == pytest.approx(180)
    assert np.interp(150, result.reference_days, result.query_days) != pytest.approx(150)


def test_whole_window_query_inverse_and_no_cherry_pick(monkeypatch):
    from types import SimpleNamespace
    from analysis import phase_shape_diagnostic as phase_module
    from analysis.recon_anchor_diagnostic import (
        apply_whole_window_forward_map,
        estimate_whole_window_local_phase,
        sample_curve_with_query,
    )

    days = np.linspace(100.0, 160.0, 61)
    source = np.column_stack((np.sin(days / 7), np.cos(days / 9)))
    target = -source
    gamma = np.linspace(0, 1, 128) ** 1.2
    monkeypatch.setattr(
        phase_module,
        "estimate_nonlinear_phase",
        lambda *args, **kwargs: SimpleNamespace(gamma=gamma, valid=True, failure_reason=""),
    )
    result = estimate_whole_window_local_phase(source, target, days)
    assert result.valid
    assert sample_curve_with_query(target, days, result.query_days).shape == source.shape

    mapped = apply_whole_window_forward_map(result.query_days, result)
    np.testing.assert_allclose(mapped, result.reference_days, atol=1e-10)
    outside = np.array([20.0, 99.0, 161.0, 300.0])
    np.testing.assert_array_equal(apply_whole_window_forward_map(outside, result), outside)
    assert result.forward_min_derivative > 0


def test_whole_window_phase_invalid_gamma_falls_back_to_identity(monkeypatch):
    from types import SimpleNamespace
    from analysis import phase_shape_diagnostic as phase_module
    from analysis.recon_anchor_diagnostic import estimate_whole_window_local_phase

    days = np.linspace(100.0, 150.0, 51)
    curve = np.column_stack((np.sin(days), np.cos(days)))
    bad = np.linspace(0, 1, 128)
    bad[64] = bad[63]
    monkeypatch.setattr(
        phase_module,
        "estimate_nonlinear_phase",
        lambda *args, **kwargs: SimpleNamespace(gamma=bad, valid=True, failure_reason=""),
    )
    result = estimate_whole_window_local_phase(curve, curve, days)
    assert not result.valid
    assert result.failure_reason == "NONLINEAR_INVALID_NON_STRICT_GAMMA"
    np.testing.assert_array_equal(result.query_days, result.reference_days)


def test_whole_window_phase_uses_lam_zero_and_query_direction_moves_anchor_naturally(monkeypatch):
    from types import SimpleNamespace
    from analysis import phase_shape_diagnostic as phase_module
    from analysis.recon_anchor_diagnostic import (
        apply_whole_window_forward_map,
        estimate_whole_window_local_phase,
    )

    days = np.linspace(130.0, 180.0, 128)
    source = np.column_stack((np.sin(days / 9), np.cos(days / 13)))
    target = source.copy()
    normalized_source_anchor = (150.0 - 130.0) / 50.0
    normalized_target_anchor = (160.0 - 130.0) / 50.0
    base = np.linspace(0, 1, 128)
    gamma = np.interp(
        base,
        [0.0, normalized_source_anchor, 1.0],
        [0.0, normalized_target_anchor, 1.0],
    )
    observed_lam = []

    def fake_solver(left, right, lam=0.0):
        observed_lam.append(lam)
        return gamma

    monkeypatch.setattr(phase_module, "_solve_joint_gamma", fake_solver)
    result = estimate_whole_window_local_phase(source, target, days)
    assert result.valid
    assert observed_lam == [0.0]
    mapped_anchor = apply_whole_window_forward_map(np.array([160.0]), result)[0]
    assert abs(mapped_anchor - 150.0) < 0.2
    assert abs(mapped_anchor - 150.0) < abs(160.0 - 150.0)


def test_circular_event_detector_finds_peak_valley_and_secondary_structures():
    from analysis.recon_anchor_diagnostic import DomainProjectionBaseline
    from analysis.recon_event_diagnostic import detect_circular_events

    days = np.arange(365.0)
    curve = (
        3.0 * np.exp(-0.5 * ((days - 80) / 9) ** 2)
        - 2.5 * np.exp(-0.5 * ((days - 155) / 11) ** 2)
        + 2.2 * np.exp(-0.5 * ((days - 245) / 10) ** 2)
        - 2.0 * np.exp(-0.5 * ((days - 310) / 9) ** 2)
    )
    events = detect_circular_events(
        curve,
        days,
        DomainProjectionBaseline(0.0, 1.0),
        min_distance_days=15,
        min_width_days=5,
        min_relative_prominence=0.15,
        min_domain_prominence=0.20,
        min_domain_elevation=0.50,
    )
    accepted = [event for event in events if event.accepted]
    assert len([event for event in accepted if event.kind == "peak"]) >= 2
    assert len([event for event in accepted if event.kind == "valley"]) >= 2
    assert any(abs(event.day - 245) < 3 for event in accepted)


def test_event_salience_records_rejections_and_width_distance_gates():
    from analysis.recon_anchor_diagnostic import DomainProjectionBaseline
    from analysis.recon_event_diagnostic import detect_circular_events

    days = np.arange(365.0)
    broad = np.exp(-0.5 * ((days - 100) / 8) ** 2)
    narrow = 0.8 * np.exp(-0.5 * ((days - 180) / 0.8) ** 2)
    neighbor = 0.7 * np.exp(-0.5 * ((days - 107) / 2) ** 2)
    events = detect_circular_events(
        broad + narrow + neighbor,
        days,
        DomainProjectionBaseline(10.0, 20.0),
        min_distance_days=15,
        min_width_days=5,
        min_relative_prominence=0.05,
        min_domain_prominence=0.20,
        min_domain_elevation=0.50,
    )
    assert sum(abs(event.day - 100) < 12 for event in events if event.kind == "peak") == 1
    rejected = [event for event in events if not event.accepted]
    assert any("too_narrow" in event.rejection_reason for event in rejected)
    assert any("low_domain_prominence" in event.rejection_reason for event in events)


def test_circular_boundary_detection_and_statistics_do_not_duplicate_events():
    from analysis.recon_anchor_diagnostic import DomainProjectionBaseline
    from analysis.recon_event_diagnostic import (
        circular_day_distance,
        circular_day_residual,
        circular_mad_days,
        detect_circular_events,
    )

    assert circular_day_distance(355, 8) == 18
    assert circular_day_residual(8, 355) == 18
    assert circular_mad_days([355, 2, 5], reference_day=355) < 6
    days = np.arange(365.0)
    distance = np.minimum((days - 358) % 365, (358 - days) % 365)
    curve = np.exp(-0.5 * (distance / 8) ** 2)
    events = detect_circular_events(
        curve,
        days,
        DomainProjectionBaseline(0.0, 0.2),
        min_distance_days=15,
        min_width_days=5,
        min_relative_prominence=0.15,
        min_domain_prominence=0.20,
        min_domain_elevation=0.50,
    )
    peaks = [event for event in events if event.kind == "peak"]
    assert len(peaks) == 1
    assert abs(peaks[0].day - 358) < 2
    assert peaks[0].boundary_crossing


def test_source_event_stability_uses_circular_occurrence_and_timing():
    from analysis.recon_anchor_diagnostic import DomainProjectionBaseline
    from analysis.recon_event_diagnostic import (
        detect_circular_events,
        evaluate_source_event_stability,
    )

    days = np.arange(365.0)

    def circular_peak(day):
        distance = np.minimum((days - day) % 365, (day - days) % 365)
        return 3.0 * np.exp(-0.5 * (distance / 8) ** 2)

    prototype = circular_peak(355)
    candidate = next(
        event
        for event in detect_circular_events(
            prototype, days, DomainProjectionBaseline(0.0, 1.0), 15, 5, 0.15, 0.2, 0.5
        )
        if event.kind == "peak"
    )
    stable = evaluate_source_event_stability(
        candidate,
        np.stack([circular_peak(day) for day in (355, 2, 5)]),
        days,
        DomainProjectionBaseline(0.0, 1.0),
        occurrence_radius_days=20,
        min_occurrence=0.60,
        max_timing_mad_days=20,
        min_distance_days=15,
        min_width_days=5,
        min_relative_prominence=0.15,
        min_domain_prominence=0.20,
        min_domain_elevation=0.50,
    )
    assert stable.accepted
    assert stable.source_occurrence_rate == 1.0
    assert stable.source_timing_mad_days < 6


def test_greedy_event_matching_is_same_type_one_to_one_and_audits_boundary():
    from analysis.recon_event_diagnostic import StructuralEvent, greedy_match_events

    def event(event_id, kind, day):
        return StructuralEvent(
            event_id, kind, day, 1.0, 1.0, 0.5, 0.5, 1.0, 8.0,
            day - 4, day + 4, True, "", False, 1.0, 2.0
        )

    source = [event("S0", "peak", 355), event("S1", "valley", 100), event("S2", "peak", 200)]
    target = [event("T0", "peak", 8), event("T1", "peak", 102), event("T2", "valley", 130)]
    first = greedy_match_events(source, target, match_radius_days=25)
    second = greedy_match_events(source, target, match_radius_days=25)
    keys = lambda values: [
        (item.source_event_id, item.target_event_id, item.match_status) for item in values
    ]
    assert keys(first) == keys(second)
    matched = [item for item in first if item.match_status.startswith("MATCHED")]
    assert len(matched) == 1
    assert matched[0].source_event_id == "S0" and matched[0].target_event_id == "T0"
    assert matched[0].match_status == "MATCHED_CIRCULAR_CANDIDATE"
    assert matched[0].boundary_crossing_candidate
    assert any(item.match_status == "UNMATCHED_SOURCE" for item in first)
    assert any(item.match_status == "UNMATCHED_TARGET" for item in first)


def test_greedy_event_matching_prefers_nearest_same_type_deterministically():
    from analysis.recon_event_diagnostic import StructuralEvent, greedy_match_events

    def event(event_id, kind, day):
        return StructuralEvent(
            event_id, kind, day, 1.0, 1.0, 0.5, 0.5, 1.0, 8.0,
            day - 4, day + 4, True, "", False
        )

    source = [event("S0", "peak", 100), event("S1", "peak", 110), event("S2", "valley", 108)]
    target = [event("T0", "peak", 108)]
    results = greedy_match_events(source, target, match_radius_days=25)
    matched = [item for item in results if item.match_status == "MATCHED"]
    assert len(matched) == 1
    assert (matched[0].source_event_id, matched[0].target_event_id) == ("S1", "T0")
    assert all(item.source_event_id != "S2" for item in matched)


def _structure_event(day, kind, value, prominence=1.0, accepted=True, event_id=""):
    from analysis.recon_event_diagnostic import StructuralEvent

    return StructuralEvent(
        event_id, kind, float(day), float(value), float(prominence), float(prominence),
        float(prominence), float(abs(value)), 5.0, float(day) - 2,
        float(day) + 2, accepted, "" if accepted else "weak", False,
    )


def test_structure_chain_compresses_same_type_including_circular_endpoints():
    from analysis.recon_structure_segments import build_alternating_chain

    events = (
        _structure_event(20, "valley", -1),
        _structure_event(35, "peak", 1),
        _structure_event(40, "peak", 2),
        _structure_event(55, "valley", -2),
        _structure_event(75, "peak", 1.5),
    )
    chain = build_alternating_chain(events)
    assert [(item.day % 365, item.kind) for item in chain] == [
        (20, "valley"), (40, "peak"), (55, "valley"), (75, "peak")
    ]
    circular = build_alternating_chain(
        (
            _structure_event(20, "peak", 1, prominence=0.5),
            _structure_event(120, "valley", -2),
            _structure_event(240, "peak", 3),
            _structure_event(355, "peak", 2, prominence=0.4),
        )
    )
    assert [(item.day % 365, item.kind) for item in circular] == [
        (120, "valley"), (240, "peak")
    ]


def test_directed_segments_include_circular_last_to_first_fall():
    from analysis.recon_structure_segments import (
        build_alternating_chain,
        build_directed_segments,
    )

    events = (
        _structure_event(20, "valley", -1),
        _structure_event(180, "peak", 2),
        _structure_event(355, "peak", 3),
    )
    chain = build_alternating_chain(events)
    segments = build_directed_segments(
        chain,
        np.linspace(-1, 3, 365),
        domain_scale=1.0,
        min_duration_days=10,
        max_duration_days=200,
        min_domain_change=0.1,
        min_curve_change=0.1,
    )
    boundary = next(item for item in segments if item.crosses_year_boundary)
    assert boundary.direction == "FALL"
    assert boundary.start_day == 355
    assert boundary.end_day == 20
    assert boundary.unwrapped_start_day == 355
    assert boundary.unwrapped_end_day == 385
    assert boundary.duration_days == 30


def test_directed_segment_uses_loose_members_and_records_core_aux_roles():
    from analysis.recon_structure_segments import build_directed_segments

    chain = (
        _structure_event(100, "valley", 0.0, accepted=True),
        _structure_event(125, "peak", 1.2, prominence=0.05, accepted=True),
    )
    core = (_structure_event(100, "valley", 0.0),)
    segments = build_directed_segments(
        chain,
        np.linspace(0, 1.2, 365),
        domain_scale=1.0,
        core_events=core,
        min_duration_days=10,
        max_duration_days=120,
        min_domain_change=0.3,
        min_curve_change=0.15,
    )
    rise = next(item for item in segments if item.direction == "RISE")
    assert rise.accepted
    assert (rise.start_role, rise.end_role) == ("CORE", "AUX")
    assert rise.signed_change > 0


def test_directed_segment_gates_duration_and_change():
    from analysis.recon_structure_segments import build_directed_segments

    segments = build_directed_segments(
        (
            _structure_event(10, "valley", 0.0),
            _structure_event(14, "peak", 0.1),
        ),
        np.linspace(-50.0, 50.0, 365),
        domain_scale=10.0,
        min_duration_days=10,
        max_duration_days=120,
        min_domain_change=0.3,
        min_curve_change=0.15,
    )
    first = segments[0]
    assert not first.accepted
    assert "duration_too_short" in first.rejection_reason
    assert "low_domain_change" in first.rejection_reason
    assert "low_curve_change" in first.rejection_reason


def test_directed_segment_matching_is_one_to_one_and_deterministic():
    from analysis.recon_structure_segments import (
        build_directed_segments,
        match_directed_segments_one_to_one,
    )

    def segments(events, prefix):
        from analysis.recon_structure_segments import build_alternating_chain

        return build_directed_segments(
            build_alternating_chain(events),
            np.sin(np.arange(365) * 2 * np.pi / 365), 1.0,
            min_duration_days=1, max_duration_days=180,
            min_domain_change=0.01, min_curve_change=0.01,
            segment_prefix=prefix,
        )

    prototypes = segments(
        (_structure_event(90, "valley", -1), _structure_event(120, "peak", 2),
         _structure_event(150, "valley", -1)), "P",
    )
    candidates = segments(
        (_structure_event(92, "valley", -1), _structure_event(121, "peak", 2),
         _structure_event(149, "valley", -1)), "C",
    )
    first = match_directed_segments_one_to_one(prototypes, candidates, 30, 2.0)
    second = match_directed_segments_one_to_one(prototypes, candidates, 30, 2.0)
    assert [(a.segment_id, b.segment_id) for a, b in first] == [
        (a.segment_id, b.segment_id) for a, b in second
    ]
    assert len({candidate.segment_id for _, candidate in first}) == len(first)


def test_source_directed_segment_stability_uses_one_to_one_matches():
    from analysis.recon_structure_segments import (
        build_directed_segments,
        evaluate_source_segments_stability,
    )

    curve = np.sin(np.arange(365) * 2 * np.pi / 365)
    kwargs = dict(
        min_duration_days=1, max_duration_days=180,
        min_domain_change=0.01, min_curve_change=0.01,
    )
    base = build_directed_segments(
        (_structure_event(100, "valley", -1), _structure_event(130, "peak", 2)),
        curve, 1.0, **kwargs,
    )
    samples = [
        build_directed_segments(
            (_structure_event(100 + shift, "valley", -1),
             _structure_event(130 + shift, "peak", 2)),
            curve, 1.0, **kwargs,
        )
        for shift in (0, 2, -1)
    ] + [()]
    stable = evaluate_source_segments_stability(
        base, samples, 30, 0.5, 25, 2.0,
    )
    rise = next(item for item in stable if item.direction == "RISE")
    assert rise.accepted
    assert rise.source_occurrence_rate == 0.75
    assert rise.source_center_timing_mad_days <= 2
    assert 0.9 < rise.source_duration_ratio_median < 1.1
    assert rise.source_duration_ratio_error_p90 < 0.1


def test_isolated_extremum_cannot_form_directed_segment():
    from analysis.recon_structure_segments import build_alternating_chain, build_directed_segments

    chain = build_alternating_chain((_structure_event(120, "peak", 2),))
    assert build_directed_segments(chain, np.zeros(365), 1.0) == ()


def test_meadow_like_small_oscillation_has_no_accepted_segments():
    from analysis.recon_anchor_diagnostic import DomainProjectionBaseline
    from analysis.recon_structure_segments import detect_structure_segments

    days = np.arange(365.0)
    curve = 0.02 * np.sin(2 * np.pi * days / 30.0)
    _, _, segments = detect_structure_segments(
        curve, days, DomainProjectionBaseline(0.0, 2.0),
        min_distance_days=7,
        member_min_width_days=2,
        member_min_relative_prominence=0.01,
        member_min_domain_prominence=0.001,
        min_duration_days=10,
        max_duration_days=120,
        min_domain_change=0.30,
        min_curve_change=0.15,
    )
    assert segments
    assert not any(item.accepted for item in segments)


def _coarse_result(events, domain_scale=10.0, **overrides):
    from analysis.recon_structure_segments import (
        build_alternating_chain,
        build_coarse_structure,
        build_directed_segments,
    )

    chain = build_alternating_chain(events)
    curve = np.linspace(-10.0, 30.0, 365)
    fine = build_directed_segments(
        chain, curve, domain_scale,
        min_duration_days=1, max_duration_days=365,
        min_domain_change=0.0, min_curve_change=0.0,
        segment_prefix="F",
    )
    options = dict(
        max_reversal_ratio=0.50,
        max_reversal_domain_change=0.35,
        max_reversal_duration_days=45,
        max_merge_depth=5,
        min_duration_days=1,
        max_duration_days=240,
        min_curve_change=0.0,
        min_domain_change=0.0,
        min_monotonicity=0.0,
    )
    options.update(overrides)
    return build_coarse_structure(chain, fine, curve, domain_scale, **options)


def test_coarse_structure_removes_weak_fall_inside_overall_rise():
    result = _coarse_result((
        _structure_event(0, "valley", 0, event_id="V0"),
        _structure_event(30, "peak", 10, event_id="P0"),
        _structure_event(45, "valley", 8, event_id="V1"),
        _structure_event(80, "peak", 20, event_id="P1"),
    ))
    rise = next(item for item in result.segments if item.direction == "RISE")
    assert (rise.start_event_id, rise.end_event_id) == ("V0", "P1")
    assert rise.num_fine_segments_covered == 3
    assert rise.num_removed_reversals == 1
    assert rise.fine_segment_ids == ("F0", "F1", "F2")
    assert rise.total_path_variation == 24
    assert rise.net_change == 20
    assert np.isclose(rise.monotonicity_ratio, 20 / 24)


def test_coarse_structure_removes_weak_rise_inside_overall_fall():
    result = _coarse_result((
        _structure_event(0, "peak", 20, event_id="P0"),
        _structure_event(30, "valley", 0, event_id="V0"),
        _structure_event(45, "peak", 2, event_id="P1"),
        _structure_event(80, "valley", -10, event_id="V1"),
    ))
    fall = next(item for item in result.segments if item.direction == "FALL")
    assert (fall.start_event_id, fall.end_event_id) == ("P0", "V1")
    assert fall.signed_change == -30
    assert fall.num_removed_reversals == 1


def test_coarse_structure_keeps_strong_or_long_or_domain_large_reversal():
    strong = _coarse_result((
        _structure_event(0, "valley", 0), _structure_event(30, "peak", 10),
        _structure_event(45, "valley", 2), _structure_event(80, "peak", 20),
    ))
    domain_large = _coarse_result((
        _structure_event(0, "valley", 0), _structure_event(30, "peak", 100),
        _structure_event(45, "valley", 96), _structure_event(80, "peak", 200),
    ))
    long = _coarse_result((
        _structure_event(0, "valley", 0), _structure_event(30, "peak", 10),
        _structure_event(90, "valley", 9), _structure_event(120, "peak", 20),
    ))
    assert not strong.removed_reversals
    assert not domain_large.removed_reversals
    assert not long.removed_reversals


def test_coarse_structure_selects_weakest_then_rebuilds_deterministically():
    events = (
        _structure_event(0, "valley", 0, event_id="V0"),
        _structure_event(30, "peak", 10, event_id="P0"),
        _structure_event(40, "valley", 9, event_id="V1"),
        _structure_event(70, "peak", 20, event_id="P1"),
        _structure_event(80, "valley", 19.5, event_id="V2"),
        _structure_event(110, "peak", 30, event_id="P2"),
    )
    first = _coarse_result(events)
    second = _coarse_result(events)
    assert [item.removed_event_ids for item in first.removed_reversals] == [
        ("P1", "V2"), ("P0", "V1")
    ]
    assert repr(first) == repr(second)
    rise = next(item for item in first.segments if item.start_event_id == "V0")
    assert rise.num_fine_segments_covered == 5
    assert rise.num_removed_reversals == 2


def test_coarse_structure_merge_depth_blocks_over_simplification():
    result = _coarse_result((
        _structure_event(0, "valley", 0), _structure_event(30, "peak", 10),
        _structure_event(45, "valley", 9), _structure_event(80, "peak", 20),
    ), max_merge_depth=2)
    assert not result.removed_reversals
    assert result.stop_reason == "merge_depth_limit"


def test_coarse_structure_handles_circular_weak_reversal_and_duration():
    result = _coarse_result((
        _structure_event(10, "valley", 9, event_id="V1"),
        _structure_event(40, "peak", 20, event_id="P1"),
        _structure_event(300, "valley", 0, event_id="V0"),
        _structure_event(350, "peak", 10, event_id="P0"),
    ))
    rise = next(item for item in result.segments if item.start_event_id == "V0")
    assert rise.end_event_id == "P1"
    assert rise.crosses_year_boundary
    assert rise.start_day == 300
    assert rise.end_day == 40
    assert rise.unwrapped_end_day == 405
    assert rise.duration_days == 105


def test_source_coarse_stability_matches_coarse_to_coarse_one_to_one():
    from analysis.recon_structure_segments import evaluate_source_coarse_stability

    prototype = _coarse_result((
        _structure_event(0, "valley", 0), _structure_event(30, "peak", 10),
        _structure_event(45, "valley", 9), _structure_event(80, "peak", 20),
    )).segments
    samples = [
        _coarse_result((
            _structure_event(shift, "valley", 0),
            _structure_event(30 + shift, "peak", 10),
            _structure_event(45 + shift, "valley", 9),
            _structure_event(80 + shift, "peak", 20),
        )).segments
        for shift in (0, 2, -1)
    ] + [()]
    stable = evaluate_source_coarse_stability(
        prototype, samples, occurrence_radius_days=40,
        min_occurrence=0.40, max_center_mad_days=35,
        max_duration_ratio=2.5,
    )
    rise = next(item for item in stable if item.direction == "RISE")
    assert rise.source_occurrence_rate == 0.75
    assert rise.accepted


def test_coarse_derivation_does_not_mutate_fine_segments_or_chain():
    from analysis.recon_structure_segments import (
        build_alternating_chain,
        build_coarse_structure,
        build_directed_segments,
    )

    chain = build_alternating_chain((
        _structure_event(0, "valley", 0, event_id="V0"),
        _structure_event(30, "peak", 10, event_id="P0"),
        _structure_event(45, "valley", 9, event_id="V1"),
        _structure_event(80, "peak", 20, event_id="P1"),
    ))
    curve = np.linspace(-10.0, 30.0, 365)
    fine = build_directed_segments(
        chain, curve, 10.0, min_duration_days=1, max_duration_days=365,
        min_domain_change=0.0, min_curve_change=0.0,
    )
    before = (repr(chain), repr(fine))
    build_coarse_structure(
        chain, fine, curve, 10.0, min_duration_days=1,
        min_curve_change=0.0, min_domain_change=0.0,
        min_monotonicity=0.0,
    )
    assert (repr(chain), repr(fine)) == before
