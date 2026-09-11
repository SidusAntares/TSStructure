import importlib.util
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


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
        assert {
            path.name for path in (tmp_path / "04_reconshift13_class_shift20").glob("*.png")
        } == expected
        updated = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert updated["untouched"] == {"keep": True}
        assert updated["class_residual_shift"]["target_grouping"] == (
            "oracle_true_labels_offline_only"
        )
        score_rows = (tmp_path / "class_shift20_scores" / filename.replace(".png", ".csv")).read_text(
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
