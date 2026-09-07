import numpy as np
import pytest
import torch


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
    target = source * 1.05
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
