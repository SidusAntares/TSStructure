import importlib.util
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np


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
