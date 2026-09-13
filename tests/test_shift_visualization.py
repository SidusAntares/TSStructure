import importlib.util
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def test_visualization_and_training_share_the_exact_residual_estimator():
    from analysis import shift_visualization as sv
    from class_residual_shift import estimate_class_residual_shift

    assert sv.estimate_class_residual_shift is estimate_class_residual_shift


def test_shared_residual_estimator_recovers_shift_and_has_strict_range():
    from class_residual_shift import decide_class_residual_shift, estimate_class_residual_shift

    grid = np.arange(365, dtype=np.float64)
    source = np.column_stack(
        [np.sin(2 * np.pi * grid / 365), np.cos(4 * np.pi * grid / 365)]
    )
    target = np.column_stack(
        [
            np.sin(2 * np.pi * (grid + 7) / 365),
            np.cos(4 * np.pi * (grid + 7) / 365),
        ]
    )
    result = estimate_class_residual_shift(source, target, 3, max_residual_days=20)
    assert [x.residual_shift_days for x in result.candidates] == list(range(-20, 21))
    assert result.class_residual_shift_days == 4
    decision = decide_class_residual_shift(result, pseudo_count=64, min_samples=32, min_gain=0.005)
    assert decision.accepted
    assert decision.accepted_residual_shift == 4
    assert decision.final_target_to_source_shift == 7
    assert decision.final_source_to_target_shift == -7


def test_shared_residual_estimator_gates_and_accepts_boundary():
    from dataclasses import replace
    from class_residual_shift import decide_class_residual_shift, estimate_class_residual_shift

    grid = np.arange(365, dtype=np.float64)
    source = np.column_stack([np.sin(grid / 13), np.cos(grid / 17)])
    target = np.column_stack([np.sin((grid + 20) / 13), np.cos((grid + 20) / 17)])
    result = estimate_class_residual_shift(source, target, 0, max_residual_days=20)
    boundary = decide_class_residual_shift(result, 64, 32, 0.0)
    assert boundary.accepted and boundary.boundary_hit
    assert boundary.accepted_residual_shift == 20
    insufficient = decide_class_residual_shift(result, 31, 32, 0.0)
    assert not insufficient.accepted
    assert insufficient.accepted_residual_shift == 0
    assert insufficient.fallback_reason == "insufficient_samples"
    low_gain = decide_class_residual_shift(replace(result, score_gain=0.004), 64, 32, 0.005)
    assert not low_gain.accepted
    assert low_gain.fallback_reason == "insufficient_gain"


@contextmanager
def _workspace_tmp():
    path = Path.cwd() / f"shift-viz-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_three_views_keep_values_and_only_shift_target_axis():
    from analysis.shift_visualization import build_shift_views

    grid = np.linspace(0.0, 365.0, 8)
    source = np.arange(16, dtype=np.float64).reshape(2, 8)
    target = source + 100.0
    views = build_shift_views(grid, source, target, 11.0, -7.0)

    for view in views.values():
        np.testing.assert_array_equal(view.source_x, grid)
        np.testing.assert_array_equal(view.source_y, source)
        np.testing.assert_array_equal(view.target_y, target)
    np.testing.assert_array_equal(views["raw"].target_x, grid)
    np.testing.assert_array_equal(views["timematch"].target_x, grid + 11.0)
    np.testing.assert_array_equal(views["reconshift13"].target_x, grid - 7.0)


def test_source_class_pca_is_fit_from_source_only():
    from analysis.shift_visualization import fit_source_class_pc1

    rng = np.random.default_rng(4)
    source = rng.normal(size=(5, 9, 3))
    target_a = rng.normal(size=(4, 9, 3))
    target_b = target_a + np.array([1000.0, -500.0, 800.0])

    projection_a = fit_source_class_pc1(source)
    projection_b = fit_source_class_pc1(source)
    np.testing.assert_allclose(projection_a.center, projection_b.center)
    np.testing.assert_allclose(projection_a.axis, projection_b.axis)
    np.testing.assert_allclose(
        projection_a.transform(source), projection_b.transform(source)
    )
    assert not np.allclose(
        projection_a.transform(target_a), projection_a.transform(target_b)
    )


def test_shift_reader_selects_shift_at_best_validation_epoch():
    from analysis.shift_visualization import read_best_validation_shift

    with _workspace_tmp() as tmp_path:
        log = tmp_path / "task.log"
        log.write_text(
            "SHIFT_TRAJECTORY|epoch=0|actual_training_shift=-3\n"
            "Validation result: loss=0.5, acc=60.0, f1=0.51\n"
            "SHIFT_TRAJECTORY|epoch=1|actual_training_shift=-8\n"
            "Validation result: loss=0.4, acc=70.0, f1=0.72\n"
            "SHIFT_TRAJECTORY|epoch=2|actual_training_shift=-5\n"
            "Validation result: loss=0.3, acc=65.0, f1=0.63\n",
            encoding="utf-8",
        )

        selected = read_best_validation_shift(log_path=log)
        assert selected.shift_days == -8.0
        assert selected.best_epoch == 1
        assert selected.selection == "best_validation_epoch"
        assert selected.fallback is False


def test_shift_reader_supports_legacy_am_log_at_best_epoch():
    from analysis.shift_visualization import read_best_validation_shift

    with _workspace_tmp() as tmp_path:
        log = tmp_path / "legacy.log"
        log.write_text(
            "Best AM Score shift -2 with accuracy 0.4\n"
            "Validation result: loss=0.5, acc=60.0, f1=0.51\n"
            "Best AM Score shift -9 with accuracy 0.5\n"
            "Validation result: loss=0.4, acc=70.0, f1=0.72\n",
            encoding="utf-8",
        )
        selected = read_best_validation_shift(log_path=log)
        assert selected.shift_days == -9.0
        assert selected.best_epoch == 1
        assert selected.source_format == "legacy_am_log"


def test_shift_reader_prefers_structured_trajectory():
    from analysis.shift_visualization import read_best_validation_shift

    with _workspace_tmp() as root:
        output = root / "output"
        output.mkdir()
        (output / "shift_trajectory.csv").write_text(
            "epoch,actual_training_shift,validation_macro_f1\n"
            "0,-3,0.51\n1,-12,0.81\n2,-6,0.73\n",
            encoding="utf-8",
        )
        log = root / "task.log"
        log.write_text(
            "Best AM Score shift 55 with accuracy 0.1\n"
            "Validation result: loss=1, acc=1, f1=0.01\n",
            encoding="utf-8",
        )
        selected = read_best_validation_shift(log, output)
        assert selected.shift_days == -12
        assert selected.best_epoch == 1
        assert selected.source_format == "structured_csv"


def test_shift_reader_marks_last_epoch_fallback_when_validation_is_unavailable():
    from analysis.shift_visualization import read_best_validation_shift

    with _workspace_tmp() as root:
        log = root / "fallback.log"
        log.write_text(
            "SHIFT_TRAJECTORY|epoch=0|actual_training_shift=-3\n"
            "SHIFT_TRAJECTORY|epoch=1|actual_training_shift=-6\n",
            encoding="utf-8",
        )
        selected = read_best_validation_shift(log)
        assert selected.shift_days == -6
        assert selected.best_epoch == 1
        assert selected.fallback is True
        assert selected.selection == "last_recorded_epoch_fallback"


def test_rendered_configs_have_identical_classes_and_ylimits():
    from analysis.shift_visualization import render_task_class_figures

    grid = np.linspace(0.0, 365.0, 12)
    source = np.stack([np.sin(grid / 40.0), np.sin(grid / 40.0) + 0.2])
    target = np.stack([np.cos(grid / 50.0), np.cos(grid / 50.0) - 0.1])
    with _workspace_tmp() as tmp_path:
        metadata = render_task_class_figures(
            output_dir=tmp_path,
            task_name="AT1_DK1",
            class_id=3,
            class_name="spring_oat",
            grid=grid,
            source_curves=source,
            target_curves=target,
            timematch_shift=-11.0,
            reconshift_shift=-9.0,
            max_spaghetti=40,
            seed=1,
        )

        expected = "03_spring_oat.png"
        for folder in (
            "01_raw_pse",
            "02_timematch_shift",
            "03_reconshift13_shift",
        ):
            assert (tmp_path / folder / expected).is_file()
        assert metadata["raw"]["ylim"] == metadata["timematch"]["ylim"]
        assert metadata["raw"]["ylim"] == metadata["reconshift13"]["ylim"]


def test_cli_is_offline_and_does_not_import_training_entry():
    path = Path("scripts/visualize_shift_configs_4tasks.py")
    text = path.read_text(encoding="utf-8")
    forbidden = ("import train", "from train", "train.py", "subprocess", "requests", "urllib")
    assert all(token not in text for token in forbidden)
    spec = importlib.util.spec_from_file_location("shift_visualization_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def _analytic_shift_prototypes(residual_days=12.0):
    grid = np.arange(365, dtype=np.float64)

    def signal(days):
        return np.column_stack(
            (
                np.sin(days / 19.0) + 0.2 * np.cos(days / 7.0),
                np.cos(days / 31.0) - 0.15 * np.sin(days / 11.0),
                np.sin(days / 43.0) + np.cos(days / 17.0),
            )
        )

    return signal(grid), signal(grid + residual_days)


def test_known_multivariate_residual_shift_recovers_positive_twelve_days():
    from analysis.shift_visualization import estimate_class_residual_shift

    source, target = _analytic_shift_prototypes(12.0)
    result = estimate_class_residual_shift(
        source, target, global_shift_days=0.0, max_residual_days=20
    )

    assert result.class_residual_shift_days == 12
    assert result.final_shift_days == 12.0
    assert len(result.candidates) == 41
    assert [item.residual_shift_days for item in result.candidates] == list(
        range(-20, 21)
    )
    assert result.best_score > result.score_at_residual_0
    assert result.boundary_hit is False
    assert result.num_valid_channels == 3
    assert result.common_support_days == 353

    global_five = estimate_class_residual_shift(
        source, target, global_shift_days=5.0, max_residual_days=20
    )
    assert global_five.class_residual_shift_days == 7
    assert global_five.final_shift_days == 12.0


def test_residual_estimator_uses_all_recon_channels_and_never_wraps():
    from analysis.shift_visualization import estimate_class_residual_shift

    grid = np.arange(365, dtype=np.float64)
    good_source = [
        np.sin(grid / 29.0) + 0.3 * np.cos(grid / 8.0),
        np.cos(grid / 37.0) - 0.2 * np.sin(grid / 15.0),
    ]
    good_target = [
        np.sin((grid + 9.0) / 29.0) + 0.3 * np.cos((grid + 9.0) / 8.0),
        np.cos((grid + 9.0) / 37.0) - 0.2 * np.sin((grid + 9.0) / 15.0),
    ]
    source = np.column_stack([np.sin(grid / 13.0)] + good_source * 50)
    # Channel zero (a hypothetical PC1) prefers -7; the multivariate Recon majority prefers +9.
    target = np.column_stack([np.sin((grid - 7.0) / 13.0)] + good_target * 50)
    result = estimate_class_residual_shift(
        source, target, global_shift_days=0.0, max_residual_days=20
    )
    assert result.class_residual_shift_days == 9
    assert result.common_support_days == 365 - 9

    boundary_source, boundary_target = _analytic_shift_prototypes(20.0)
    boundary = estimate_class_residual_shift(
        boundary_source,
        boundary_target,
        global_shift_days=0.0,
        max_residual_days=20,
    )
    assert boundary.class_residual_shift_days == 20
    assert boundary.boundary_hit is True
    assert boundary.common_support_days == 345


def test_fourth_view_only_adds_residual_to_target_coordinates():
    from analysis.shift_visualization import build_shift_views

    grid = np.linspace(0.0, 365.0, 8)
    source = np.arange(16, dtype=np.float64).reshape(2, 8)
    target = source + 100.0
    views = build_shift_views(
        grid,
        source,
        target,
        timematch_shift=11.0,
        reconshift_shift=-7.0,
        class_residual_shift=5.0,
    )
    fourth = views["reconshift13_class_shift20"]
    np.testing.assert_array_equal(fourth.source_y, views["reconshift13"].source_y)
    np.testing.assert_array_equal(fourth.target_y, views["reconshift13"].target_y)
    np.testing.assert_array_equal(fourth.source_x, views["reconshift13"].source_x)
    np.testing.assert_allclose(fourth.target_x, grid - 2.0)


def test_incremental_fourth_render_preserves_old_images_and_updates_manifest():
    from analysis.shift_visualization import (
        render_class_residual_figure,
        update_class_residual_outputs,
        validate_existing_task_outputs,
    )

    grid = np.linspace(0.0, 365.0, 12)
    source = np.stack([np.sin(grid / 40.0), np.sin(grid / 40.0) + 0.2])
    target = np.stack([np.cos(grid / 50.0), np.cos(grid / 50.0) - 0.1])
    filename = "03_spring_oat.png"
    with _workspace_tmp() as tmp_path:
        old_bytes = {}
        for folder in ("01_raw_pse", "02_timematch_shift", "03_reconshift13_shift"):
            path = tmp_path / folder / filename
            path.parent.mkdir()
            path.write_bytes((folder + "-sentinel").encode())
            old_bytes[path] = path.read_bytes()
        manifest = {
            "task": "AT1_DK1",
            "classes": ["a", "b", "c", "spring_oat"],
            "class_outputs": {"3": {"raw": {"ylim": [-2.0, 2.0]}}},
            "untouched": {"keep": True},
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        loaded, expected = validate_existing_task_outputs(tmp_path)
        assert expected == {filename}
        metadata = render_class_residual_figure(
            output_dir=tmp_path,
            task_name="AT1→DK1",
            class_id=3,
            class_name="spring_oat",
            grid=grid,
            source_curves=source,
            target_curves=target,
            global_shift_days=-7.0,
            class_residual_shift_days=5,
            score_at_residual_0=0.2,
            best_score=0.6,
            score_gain=0.4,
            ylim=(-2.0, 2.0),
            seed=1,
        )
        update_class_residual_outputs(
            tmp_path,
            loaded,
            [
                {
                    "class_id": 3,
                    "class_name": "spring_oat",
                    "source_count": 2,
                    "target_count": 2,
                    "global_reconshift_days": -7.0,
                    "class_residual_shift_days": 5,
                    "final_shift_days": -2.0,
                    "score_at_residual_0": 0.2,
                    "best_score": 0.6,
                    "score_gain": 0.4,
                    "boundary_hit": False,
                    "num_valid_channels": 3,
                    "common_support_days": 363,
                }
            ],
            {3: [(shift, float(shift), 365 - abs(shift)) for shift in range(-20, 21)]},
            {"3": metadata},
            max_residual_days=20,
        )

        assert (tmp_path / "04_reconshift13_class_shift20" / filename).is_file()
        for path, content in old_bytes.items():
            assert path.read_bytes() == content
        config_dir = tmp_path / "04_reconshift13_class_shift20"
        assert {path.name for path in config_dir.glob("*.png") if path.name in expected} == expected
        assert (config_dir / "class_shift20_overview.png").is_file()
        updated = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert updated["untouched"] == {"keep": True}
        assert updated["class_residual_shift"]["target_grouping"] == (
            "oracle_true_labels_offline_only"
        )
        score_rows = (config_dir / "class_shift20_scores" / filename.replace(".png", ".csv")).read_text(
            encoding="utf-8"
        ).strip().splitlines()
        assert len(score_rows) == 42


def test_launcher_extends_same_baseline_output_root():
    text = Path("scripts/run_shift_visualization_4tasks_seed1.sh").read_text(
        encoding="utf-8"
    )
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/shift_visualizations_seed1}"' in text
    assert "--add-class-residual-shift" in text
    assert '"$OUTPUT_ROOT"' in text


def test_launcher_can_run_local_nonlinear_four_task_gpu_wave_without_changing_default():
    text = Path("scripts/run_shift_visualization_4tasks_seed1.sh").read_text(
        encoding="utf-8"
    )
    assert 'ADD_RECON13_LOCAL_NONLINEAR="${ADD_RECON13_LOCAL_NONLINEAR:-0}"' in text
    assert "--add-recon13-local-nonlinear" in text
    for gpu, task in enumerate(("AT1 DK1", "DK1 FR1", "FR1 FR2", "FR2 AT1")):
        assert f"run_local_task {gpu} {task}" in text
    assert "--add-class-residual-shift" in text
    assert all(token not in text for token in ("git ", "curl ", "wget ", "pip install"))


def test_local_nonlinear_runner_uses_whole_window_path_not_anchor_fixed_helpers():
    text = Path("scripts/visualize_shift_configs_4tasks.py").read_text(encoding="utf-8")
    assert "--add-recon13-local-nonlinear" in text
    assert "estimate_whole_window_local_phase(" in text
    active = text[text.index("def run_local_nonlinear_extension(") : text.index("def run_task(")]
    assert "reconstruct_windows_batched(" in active
    assert "reconstruct_one_window(" not in active
    for forbidden in (
        "build_anchor_time_maps(",
        "stitch_half_phase_gammas(",
        "estimate_anchor_fixed_local_phase(",
        "estimate_anchor_fixed_local_phases(",
        "estimate_anchor_fixed_local_phase_prepared(",
    ):
        assert forbidden not in active


def test_sample_specific_fourier_windows_are_reconstructed_in_one_batch():
    import torch
    from scripts.visualize_shift_configs_4tasks import reconstruct_windows_batched

    class Synthesizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, coefficients, positions):
            self.calls += 1
            scale = coefficients.real.sum(dim=1)[:, None, :]
            return positions[:, :, None] + scale

    synthesizer = Synthesizer()
    coefficients = np.ones((3, 5, 2), dtype=np.complex64)
    queries = np.stack(
        (np.linspace(10, 20, 8), np.linspace(11, 21, 8), np.linspace(12, 22, 8))
    )
    actual = reconstruct_windows_batched(
        coefficients, queries, synthesizer, torch.device("cpu"), batch_size=16
    )
    expected = np.repeat(queries[:, :, None] + 5.0, 2, axis=2)
    np.testing.assert_allclose(actual, expected)
    assert synthesizer.calls == 1


def test_cli_uses_neutral_mode13_reconstruction_and_oracle_labels_only_for_grouping():
    path = Path("scripts/visualize_shift_configs_4tasks.py")
    text = path.read_text(encoding="utf-8")
    assert "from models.fourier_reconstruction import" in text
    assert "BatchedDirectFourierAnalyzer" in text
    assert "num_modes=13, period_days=365.0, reg=0.001" in text
    assert "models.fredn" not in text
    assert "target_prototype" in text
    assert "target_grouping" not in text  # Manifest wording lives in the pure analysis module.


def test_daily_prototype_is_median_of_neutral_fourier_recon13_curves():
    from models.fourier_reconstruction import (
        BatchedDirectFourierAnalyzer,
        BatchedDirectFourierSynthesizer,
    )

    path = Path("scripts/visualize_shift_configs_4tasks.py")
    spec = importlib.util.spec_from_file_location("shift_visualization_recon", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    positions = torch.tensor(
        [[3.0, 19.0, 47.0, 88.0, 131.0, 177.0, 221.0, 268.0, 319.0, 354.0]]
    ).repeat(3, 1)
    features = torch.stack(
        [
            torch.stack(
                (
                    torch.sin(2.0 * np.pi * positions[index] / 365.0) + offset,
                    torch.cos(4.0 * np.pi * positions[index] / 365.0) - offset,
                ),
                dim=-1,
            )
            for index, offset in enumerate((0.0, 0.1, -0.2))
        ]
    )
    analyzer = BatchedDirectFourierAnalyzer(13, period_days=365.0, reg=0.001)
    synthesizer = BatchedDirectFourierSynthesizer(13, period_days=365.0)
    coefficients, _ = analyzer(features, positions)
    prototype = module.reconstruct_class_prototype(
        coefficients.numpy(), synthesizer, torch.device("cpu"), sample_batch_size=2
    )
    daily = torch.arange(365, dtype=torch.float32).repeat(3, 1)
    expected = torch.median(synthesizer(coefficients, daily), dim=0).values.numpy()
    np.testing.assert_allclose(prototype, expected, atol=1e-6, rtol=1e-6)


def test_samplewise_time_mapping_preserves_raw_y_and_warps_before_aggregation():
    from analysis.shift_visualization import build_samplewise_time_mapped_curves

    display = np.linspace(0, 10, 11)
    source = np.stack((display, display + 1))
    positions = (np.array([0.0, 5.0, 10.0]), np.array([0.0, 5.0, 10.0]))
    values = (np.array([0.0, 5.0, 10.0]), np.array([0.0, 5.0, 10.0]))
    mapped = (positions[0].copy(), np.array([0.0, 7.0, 10.0]))
    result = build_samplewise_time_mapped_curves(
        display, source, positions, values, mapped
    )
    np.testing.assert_array_equal(result.raw_target_y[0], values[0])
    np.testing.assert_array_equal(result.raw_target_y[1], values[1])
    np.testing.assert_array_equal(result.mapped_target_x[0], mapped[0])
    np.testing.assert_array_equal(result.mapped_target_x[1], mapped[1])
    assert not np.array_equal(result.target_y[0], result.target_y[1])
    assert result.target_y.shape == (2, 11)


def test_samplewise_time_mapping_rejects_non_monotone_timestamps():
    import pytest
    from analysis.shift_visualization import build_samplewise_time_mapped_curves

    with pytest.raises(ValueError, match="strictly increasing"):
        build_samplewise_time_mapped_curves(
            np.arange(4.0),
            np.ones((1, 4)),
            (np.array([0.0, 1.0, 2.0]),),
            (np.array([1.0, 2.0, 3.0]),),
            (np.array([0.0, 2.0, 1.0]),),
        )


def test_local_nonlinear_output_is_parallel_04_and_old_folders_untouched(tmp_path):
    import json
    from analysis.shift_visualization import (
        build_samplewise_time_mapped_curves,
        render_local_nonlinear_figure,
        update_local_nonlinear_outputs,
    )

    old_paths = []
    for folder in (
        "01_raw_pse",
        "02_timematch_shift",
        "03_reconshift13_shift",
        "04_reconshift13_class_shift20",
    ):
        path = tmp_path / folder / "00_crop.png"
        path.parent.mkdir(parents=True)
        path.write_bytes((folder + "-sentinel").encode())
        old_paths.append(path)
    old_bytes = {path: path.read_bytes() for path in old_paths}
    manifest = {"class_outputs": {"0": {"raw": {"ylim": [-2, 4]}}}, "keep": 7}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    grid = np.linspace(0, 365, 32)
    source = np.stack((np.sin(grid / 40), np.sin(grid / 40) + 0.1))
    raw_x = (np.array([20.0, 100.0, 180.0, 260.0, 340.0]),)
    raw_y = (np.sin(raw_x[0] / 40),)
    mapped_x = (raw_x[0] + np.array([0.0, 2.0, 0.0, 0.0, 0.0]),)
    curves = build_samplewise_time_mapped_curves(grid, source, raw_x, raw_y, mapped_x)
    metadata = render_local_nonlinear_figure(
        tmp_path,
        "AT1→DK1",
        0,
        "crop",
        curves,
        global_shift_days=-5,
        ylim=(-2, 4),
        eligible_rate=1.0,
        valid_rate=1.0,
        anchor_error_before=8.0,
        anchor_error_after=3.0,
        corr_before=0.2,
        corr_after=0.5,
    )
    update_local_nonlinear_outputs(
        tmp_path,
        manifest,
        [{"sample_id": 1, "class_id": 0}],
        [{"class_id": 0, "class_name": "crop"}],
        {"0": metadata},
    )

    assert (tmp_path / "04_reconshift13_local_nonlinear" / "00_crop.png").is_file()
    folder = tmp_path / "04_reconshift13_local_nonlinear"
    assert (folder / "local_nonlinear_sample_summary.csv").is_file()
    assert (folder / "local_nonlinear_class_summary.csv").is_file()
    for path, content in old_bytes.items():
        assert path.read_bytes() == content
    updated = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert updated["keep"] == 7
    config = updated["reconshift13_local_nonlinear"]
    assert config["anchor_role"] == "correspondence_and_window_gate_only"
    assert config["anchor_hard_constraint"] is False
    assert config["raw_features_modified"] is False
    assert updated["class_outputs"]["0"]["reconshift13_local_nonlinear"] == metadata


def test_multi_event_extension_is_mode13_non_warping_and_cli_tunable():
    runner = Path("scripts/visualize_shift_configs_4tasks.py").read_text(encoding="utf-8")
    launcher = Path("scripts/run_shift_visualization_4tasks_seed1.sh").read_text(encoding="utf-8")
    assert "--add-recon13-multi-event" in runner
    defaults = {
        "--event-min-distance-days": "15",
        "--event-min-width-days": "5",
        "--event-min-relative-prominence": "0.15",
        "--event-min-domain-prominence": "0.20",
        "--event-min-domain-elevation": "0.50",
        "--event-source-occurrence-radius-days": "20",
        "--event-min-source-occurrence": "0.60",
        "--event-max-source-timing-mad-days": "20",
        "--event-match-radius-days": "25",
    }
    for option, default in defaults.items():
        assert option in runner and f"default={default}" in runner
    active = runner[runner.index("def run_multi_event_extension(") : runner.index("def run_local_nonlinear_extension(")]
    for forbidden in ("estimate_nonlinear_phase", "SRVF", "apply_whole_window_forward_map"):
        assert forbidden not in active
    assert '["positions"]' not in active
    assert 'ADD_RECON13_MULTI_EVENT="${ADD_RECON13_MULTI_EVENT:-0}"' in launcher
    for gpu, task in enumerate(("AT1 DK1", "DK1 FR1", "FR1 FR2", "FR2 AT1")):
        assert f"run_multi_event_task {gpu} {task}" in launcher


def test_multi_event_outputs_are_parallel_and_preserve_existing_views(tmp_path):
    import json
    from analysis.shift_visualization import update_multi_event_outputs

    sentinels = {}
    for folder in (
        "01_raw_pse",
        "02_timematch_shift",
        "03_reconshift13_shift",
        "04_reconshift13_class_shift20",
        "04_reconshift13_local_nonlinear",
    ):
        path = tmp_path / folder / "00_crop.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(folder.encode())
        sentinels[path] = path.read_bytes()
    manifest = {"keep": 9, "class_outputs": {"0": {"raw": {"ylim": [-1, 2]}}}}
    update_multi_event_outputs(
        tmp_path,
        manifest,
        {"min_distance_days": 15, "match_radius_days": 25},
        [{"task": "A_B", "event_id": "S0"}],
        [{"task": "A_B", "event_id": "T0"}],
        [{"task": "A_B", "match_status": "MATCHED"}],
        [{"class_id": 0, "num_source_candidates": 1}],
        {"0": {"path": "04_reconshift13_multi_event/00_crop.png"}},
    )
    for path, value in sentinels.items():
        assert path.read_bytes() == value
    folder = tmp_path / "04_reconshift13_multi_event"
    for name in ("source_events.csv", "target_events.csv", "event_matches.csv", "class_summary.csv"):
        assert (folder / name).is_file()
    updated = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert updated["keep"] == 9
    assert updated["multi_event"]["mode"] == 13
    assert updated["multi_event"]["nonlinear_registration"] is False
    assert updated["multi_event"]["raw_timestamp_modified"] is False


def test_multi_event_two_panel_and_acceptance_detail_figures_render(tmp_path):
    from analysis.recon_anchor_diagnostic import DomainProjectionBaseline
    from analysis.recon_event_diagnostic import detect_circular_events, greedy_match_events
    from scripts.visualize_shift_configs_4tasks import _plot_multi_event_class

    days = np.arange(365.0)
    source = (
        2.8 * np.exp(-0.5 * ((days - 110) / 10) ** 2)
        - 2.2 * np.exp(-0.5 * ((days - 205) / 12) ** 2)
        + 2.0 * np.exp(-0.5 * ((days - 290) / 9) ** 2)
    )
    target = np.stack([np.roll(source, shift) for shift in (3, 6)])
    baseline = DomainProjectionBaseline(0.0, 1.0)
    source_events = detect_circular_events(source, days, baseline, event_prefix="S")
    target_events = [
        detect_circular_events(curve, days, baseline, event_prefix="T")
        for curve in target
    ]
    matches = [greedy_match_events(source_events, events) for events in target_events]
    main = tmp_path / "04_reconshift13_multi_event" / "05_winter_barley.png"
    detail = tmp_path / "04_reconshift13_multi_event" / "diagnostics" / "05_winter_barley_event_detail.png"
    _plot_multi_event_class(
        main,
        detail,
        "winter_barley",
        source,
        target,
        source_events,
        target_events,
        matches,
    )
    assert main.is_file() and main.stat().st_size > 0
    assert detail.is_file() and detail.stat().st_size > 0
    assert len([event for event in source_events if event.accepted]) >= 3


def test_multi_event_extension_skips_classes_absent_from_baseline_views(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from analysis.recon_anchor_diagnostic import DomainProjectionBaseline
    from scripts.visualize_shift_configs_4tasks import (
        SourceClassPC1,
        run_multi_event_extension,
    )

    args = SimpleNamespace(
        event_min_distance_days=15,
        event_min_width_days=5,
        event_min_relative_prominence=0.15,
        event_min_domain_prominence=0.20,
        event_min_domain_elevation=0.50,
        event_source_occurrence_radius_days=20,
        event_min_source_occurrence=0.60,
        event_max_source_timing_mad_days=20,
        event_match_radius_days=25,
    )
    projection = SourceClassPC1(np.zeros(1), np.ones(1))
    coefficients = np.zeros((1, 13, 1), dtype=np.complex128)
    prototype = np.zeros((365, 1), dtype=np.float64)
    baseline = DomainProjectionBaseline(0.0, 1.0)

    def render(output_path, detail_path, *_args):
        for path in (Path(output_path), Path(detail_path)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"figure")

    monkeypatch.setattr(
        "scripts.visualize_shift_configs_4tasks._plot_multi_event_class", render
    )
    run_multi_event_extension(
        args=args,
        output_dir=tmp_path,
        task_name="FR2_AT1",
        classes=("present", "missing_target"),
        projections={0: projection, 1: projection},
        source_coefficients={0: coefficients, 1: coefficients},
        target_coefficients={0: coefficients},
        source_records={0: [], 1: []},
        target_records={0: [{"sample_id": 1}]},
        source_baselines={0: baseline, 1: baseline},
        target_baselines={0: baseline},
        source_prototypes={0: prototype, 1: prototype},
        recon_shift_days=0,
        existing_manifest={"class_outputs": {"0": {}}},
        expected_filenames={"00_present.png"},
    )

    assert (tmp_path / "04_reconshift13_multi_event" / "00_present.png").is_file()
    summary = (tmp_path / "04_reconshift13_multi_event" / "class_summary.csv").read_text(
        encoding="utf-8"
    )
    assert "present" in summary
    assert "missing_target" not in summary


def test_visualization_layout_migration_is_safe_complete_and_idempotent(tmp_path):
    from analysis.shift_visualization import (
        migrate_visualization_layout,
        visualization_config_dir,
        visualization_meta_dir,
    )

    task = tmp_path / "AT1_DK1"
    (task / "03_reconshift13_shift").mkdir(parents=True)
    (task / "03_reconshift13_shift" / "00_crop.png").write_bytes(b"recon")
    (task / "04_oracle_samplewise_anchor_nonlinear").mkdir()
    (task / "04_oracle_samplewise_anchor_nonlinear" / "obsolete.png").write_bytes(b"old")
    (task / "class_shift20_scores").mkdir()
    (task / "class_shift20_scores" / "00_crop.csv").write_text("score", encoding="utf-8")
    (task / "class_shift20_summary.csv").write_text("summary", encoding="utf-8")
    (task / "local_nonlinear_class_summary.csv").write_text("local", encoding="utf-8")
    (task / "manifest.json").write_text('{"task":"AT1_DK1"}', encoding="utf-8")
    (task / "shifts_summary.csv").write_text("shift", encoding="utf-8")

    first = migrate_visualization_layout(tmp_path)
    assert first["moved"] >= 6
    assert first["deleted_obsolete"] == 1
    assert (visualization_config_dir(tmp_path, "03_reconshift13_shift", "AT1_DK1") / "00_crop.png").is_file()
    assert (visualization_config_dir(tmp_path, "04_reconshift13_class_shift20", "AT1_DK1") / "class_shift20_scores" / "00_crop.csv").is_file()
    assert (visualization_config_dir(tmp_path, "04_reconshift13_local_nonlinear", "AT1_DK1") / "local_nonlinear_class_summary.csv").is_file()
    assert (visualization_meta_dir(tmp_path, "AT1_DK1") / "manifest.json").is_file()
    assert not task.exists()
    second = migrate_visualization_layout(tmp_path)
    assert second["moved"] == 0
    assert second["deleted_obsolete"] == 0


def test_visualization_layout_migration_refuses_conflicting_destination(tmp_path):
    import pytest
    from analysis.shift_visualization import migrate_visualization_layout

    source = tmp_path / "AT1_DK1" / "03_reconshift13_shift"
    destination = tmp_path / "03_reconshift13_shift" / "AT1_DK1"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    (source / "00_crop.png").write_bytes(b"source")
    (destination / "00_crop.png").write_bytes(b"different")
    with pytest.raises(FileExistsError, match="conflict"):
        migrate_visualization_layout(tmp_path)
    assert (source / "00_crop.png").read_bytes() == b"source"
    assert (destination / "00_crop.png").read_bytes() == b"different"


def test_visualization_reader_prefers_new_layout_then_falls_back_to_legacy(tmp_path):
    from analysis.shift_visualization import resolve_visualization_config_dir

    legacy = tmp_path / "AT1_DK1" / "01_raw_pse"
    legacy.mkdir(parents=True)
    assert resolve_visualization_config_dir(tmp_path, "01_raw_pse", "AT1_DK1") == legacy
    modern = tmp_path / "01_raw_pse" / "AT1_DK1"
    modern.mkdir(parents=True)
    assert resolve_visualization_config_dir(tmp_path, "01_raw_pse", "AT1_DK1") == modern


def test_structure_segment_outputs_are_self_cont_and_use_new_layout(tmp_path):
    from analysis.shift_visualization import (
        visualization_meta_dir,
        write_structure_segment_outputs,
    )

    meta = visualization_meta_dir(tmp_path, "AT1_DK1")
    write_structure_segment_outputs(
        meta,
        {"mode": 13, "nonlinear_registration": False},
        [{"event_id": "P0", "class_id": 0}],
        [{"segment_id": "SSEG0", "class_id": 0}],
        [{"segment_id": "TSEG0", "true_class": 0}],
        [{"class_id": 0, "num_core_events": 1}],
    )
    folder = tmp_path / "05_reconshift13_structure_segments" / "AT1_DK1"
    assert {path.name for path in folder.iterdir()} == {
        "source_core_events.csv", "source_segments.csv", "target_segments.csv",
        "class_summary.csv", "manifest.json",
    }
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == 13
    assert manifest["nonlinear_registration"] is False
    assert not (tmp_path / "AT1_DK1" / "05_reconshift13_structure_segments").exists()


def test_structure_segment_runner_exposes_all_thresholds_and_no_registration():
    runner = Path("scripts/visualize_shift_configs_4tasks.py").read_text(encoding="utf-8")
    launcher = Path("scripts/run_shift_visualization_4tasks_seed1.sh").read_text(encoding="utf-8")
    for option in (
        "--add-recon13-structure-segments",
        "--segment-event-min-relative-prominence",
        "--segment-event-min-domain-prominence",
        "--segment-event-min-width-days",
        "--segment-min-span-days", "--segment-max-span-days",
        "--segment-min-domain-variation", "--segment-min-curve-variation",
        "--segment-occurrence-radius-days", "--segment-min-source-occurrence",
        "--segment-max-source-timing-mad-days", "--segment-max-width-ratio",
    ):
        assert option in runner
    active = runner[runner.index("def run_structure_segment_extension(") : runner.index("def run_multi_event_extension(")]
    assert "estimate_nonlinear_phase" not in active
    assert "greedy_match_events" not in active
    assert 'ADD_RECON13_STRUCTURE_SEGMENTS="${ADD_RECON13_STRUCTURE_SEGMENTS:-0}"' in launcher


def test_structure_segment_three_panel_and_detail_figures_render(tmp_path):
    from analysis.recon_anchor_diagnostic import DomainProjectionBaseline
    from analysis.recon_event_diagnostic import detect_circular_events
    from analysis.recon_structure_segments import detect_structure_segments
    from scripts.visualize_shift_configs_4tasks import _plot_structure_segment_class

    days = np.arange(365.0)
    source = (
        -2.0 * np.exp(-0.5 * ((days - 100) / 9) ** 2)
        + 3.0 * np.exp(-0.5 * ((days - 130) / 10) ** 2)
        - 2.2 * np.exp(-0.5 * ((days - 165) / 9) ** 2)
    )
    targets = np.stack((np.roll(source, 3), np.roll(source, 5)))
    baseline = DomainProjectionBaseline(0.0, 1.0)
    source_candidates, _, source_segments = detect_structure_segments(
        source, days, baseline, min_span_days=15, max_span_days=100,
        min_domain_variation=0.2, min_curve_variation=0.2,
    )
    target_median = np.median(targets, axis=0)
    target_candidates, _, target_segments = detect_structure_segments(
        target_median, days, baseline, min_span_days=15, max_span_days=100,
        min_domain_variation=0.2, min_curve_variation=0.2,
    )
    source_core = detect_circular_events(source, days, baseline)
    target_core = detect_circular_events(target_median, days, baseline)
    main = tmp_path / "05_reconshift13_structure_segments" / "AT1_DK1" / "00_crop.png"
    detail = main.parent / "diagnostics" / "00_crop_segment_detail.png"
    _plot_structure_segment_class(
        main, detail, "crop", source, targets, source_core, source_candidates,
        source_segments, target_core, target_candidates, target_segments,
    )
    assert main.is_file() and main.stat().st_size > 0
    assert detail.is_file() and detail.stat().st_size > 0
