from pathlib import Path

import numpy as np
import pytest


def test_epoch_history_keeps_validation_pseudo_and_gradient_state():
    from analysis.structure_e_g_validation import parse_epoch_history

    text = "\n".join((
        "TARGET_STRUCTURE_GRADIENT|epoch=0|open=false|detached=true|detach_epochs=5",
        "SHAPE_V2CLEAN_EPOCH|epoch=0|accepted_pseudo_count=17|accepted_pseudo_accuracy=0.75",
        "Validation result: loss=0.2, acc=0.8, f1=0.61",
        "TARGET_STRUCTURE_GRADIENT|epoch=5|open=true|detached=false|detach_epochs=5",
        "PSEUDO_ORACLE_AUDIT|epoch=5|accepted_count=23|accepted_pseudo_accuracy=0.80",
        "Validation result: loss=0.1, acc=0.9, f1=0.72",
    ))
    rows = parse_epoch_history(text, "G", "FR1_FR2")
    assert rows == [
        {
            "method": "G", "task": "FR1_FR2", "epoch": 0,
            "validation_macro_f1": pytest.approx(0.61),
            "accepted_pseudo_count": 17,
            "accepted_pseudo_accuracy": pytest.approx(0.75),
            "target_query_gradient_open": False,
        },
        {
            "method": "G", "task": "FR1_FR2", "epoch": 5,
            "validation_macro_f1": pytest.approx(0.72),
            "accepted_pseudo_count": 23,
            "accepted_pseudo_accuracy": pytest.approx(0.80),
            "target_query_gradient_open": True,
        },
    ]


def test_prototype_geometry_separates_correct_and_wrong_source_classes():
    from analysis.structure_e_g_validation import prototype_geometry_rows

    source = np.asarray([[1., 0.], [.9, .1], [0., 1.], [.1, .9]])
    source_labels = np.asarray([0, 0, 1, 1])
    target = np.asarray([[.8, .2], [.2, .8]])
    target_labels = np.asarray([0, 1])
    rows, summary = prototype_geometry_rows(
        "FR2_DK1", "E_final", "shape_response", source, source_labels,
        target, target_labels, ("a", "b"),
    )
    assert len(rows) == 2
    assert all(row["same_class_prototype_cosine"] > .9 for row in rows)
    assert all(row["mean_correct_wrong_margin"] > 0 for row in rows)
    assert summary["nearest_source_prototype_macro_f1"] == pytest.approx(1.)


def test_raw_ndvi_uses_b04_and_b08_and_keeps_real_dates():
    from analysis.structure_e_g_validation import raw_ndvi_observations

    pixels = np.zeros((2, 10, 3), dtype=np.float32)
    pixels[:, 2, :] = np.asarray([[1., 1., 1.], [2., 2., 2.]])
    pixels[:, 6, :] = np.asarray([[3., 3., 3.], [6., 6., 6.]])
    rows = raw_ndvi_observations(
        pixels, ("20170410", "20170520"), class_id=2, sample_id=9,
    )
    assert [row["date"] for row in rows] == ["2017-04-10", "2017-05-20"]
    assert [row["day_of_year"] for row in rows] == [100, 140]
    assert [row["ndvi"] for row in rows] == pytest.approx([.5, .5])


def test_focus_classes_prioritize_requested_crops_then_supported_common_classes():
    from analysis.structure_e_g_validation import select_focus_classes

    names = ("meadow", "winter_wheat", "spring_barley", "winter_barley", "corn")
    selected = select_focus_classes(
        names, np.asarray([30, 40, 25, 22, 100]),
        np.asarray([35, 41, 21, 2, 100]), min_support=20, max_classes=4,
    )
    assert selected == [2, 1, 0, 4]


def test_raw_curve_gap_interpolates_only_shared_calendar_range():
    from analysis.structure_e_g_validation import raw_curve_gap_rows

    rows = [
        {"task": "T", "domain": "source", "class_id": 0, "class_name": "crop", "day_of_year": 100, "ndvi_median": .2},
        {"task": "T", "domain": "source", "class_id": 0, "class_name": "crop", "day_of_year": 200, "ndvi_median": .8},
        {"task": "T", "domain": "target", "class_id": 0, "class_name": "crop", "day_of_year": 120, "ndvi_median": .3},
        {"task": "T", "domain": "target", "class_id": 0, "class_name": "crop", "day_of_year": 180, "ndvi_median": .7},
    ]
    result = raw_curve_gap_rows(rows)
    assert result[0]["shared_start_doy"] == 120
    assert result[0]["shared_end_doy"] == 180
    assert result[0]["mean_absolute_ndvi_gap"] < .02


def test_launcher_adds_only_missing_g_tasks_then_runs_visualization():
    source = Path("scripts/run_structure_e_cross_seed1.sh").read_text()
    wrapper = Path(
        "scripts/run_structure_g_validation_visualization_seed1.sh"
    ).read_text()
    assert "G_EXTEND" in source
    assert 'run_uda "${GPU0}" G AT1 "$AT1" DK1 "$DK1"' in source
    assert 'run_uda "${GPU1}" G DK1 "$DK1" AT1 "$AT1"' in source
    assert "RUN_ROUND=G_EXTEND" in wrapper
    assert "analysis/structure_e_g_validation.py" in wrapper
    assert "run_source" not in wrapper


def test_gradient_control_contract_remains_detach5_query_only():
    import timematch
    from types import SimpleNamespace

    config = SimpleNamespace(
        detach_target_structure=False, target_structure_detach_epochs=5,
    )
    assert [
        timematch.target_structure_detached_for_epoch(config, epoch)
        for epoch in range(7)
    ] == [True, True, True, True, True, False, False]
