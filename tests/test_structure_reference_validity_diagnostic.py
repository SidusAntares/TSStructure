import ast
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _segment(
    structure_id,
    direction="RISE",
    center=100.0,
    duration=40.0,
    change=2.0,
    monotonicity=0.8,
    occurrence=0.6,
):
    return SimpleNamespace(
        coarse_segment_id=structure_id,
        direction=direction,
        center_day=float(center % 365),
        duration_days=float(duration),
        absolute_change=float(change),
        monotonicity_ratio=float(monotonicity),
        source_occurrence_rate=float(occurrence),
        start_day=float((center - duration / 2) % 365),
        end_day=float((center + duration / 2) % 365),
        accepted=True,
    )


def test_bootstrap_subsamples_are_deterministic_without_replacement_and_exact_fraction():
    from analysis.structure_reference_validity_diagnostic import bootstrap_subsample_indices

    first = bootstrap_subsample_indices(10, repeats=4, fraction=0.7, seed=17)
    second = bootstrap_subsample_indices(10, repeats=4, fraction=0.7, seed=17)
    assert np.array_equal(first, second)
    assert first.shape == (4, 7)
    assert all(len(np.unique(row)) == 7 for row in first)


def test_reference_matching_respects_direction_circular_distance_duration_and_one_to_one():
    from analysis.structure_reference_validity_diagnostic import match_structure_sets

    references = [_segment("C0", center=359, duration=40), _segment("C1", center=100)]
    candidates = [
        _segment("B0", center=2, duration=42),
        _segment("B1", center=4, duration=41),
        _segment("B2", direction="FALL", center=100),
        _segment("B3", center=101, duration=100),
    ]
    matches = match_structure_sets(
        references, candidates, center_radius_days=30, max_duration_ratio=2.0
    )
    assert len(matches) == 1
    assert matches[0].reference.coarse_segment_id == "C0"
    assert matches[0].candidate.coarse_segment_id == "B0"
    assert matches[0].center_distance_days == 8


def test_structure_stability_statistics_include_circular_mad_and_medoid_presence():
    from analysis.structure_reference_validity_diagnostic import audit_reference_structures

    reference = [_segment("C0", center=359, duration=40, occurrence=0.25)]
    bootstrap = [
        [_segment("B0", center=1, duration=38, change=2.2, monotonicity=0.9)],
        [_segment("B1", center=357, duration=42, change=1.8, monotonicity=0.7)],
        [],
        [_segment("B3", direction="FALL", center=359)],
    ]
    medoid = [_segment("M0", center=2, duration=40)]
    row = audit_reference_structures(
        "AT1", 0, "corn", reference, bootstrap, medoid,
        center_radius_days=30, max_duration_ratio=2.0,
    )[0]
    assert row["num_bootstrap_runs"] == 4
    assert row["num_bootstrap_matches"] == 2
    assert row["bootstrap_occurrence_rate"] == 0.5
    assert row["bootstrap_stability"] == "MEDIUM_STABILITY"
    assert row["bootstrap_center_mad_days"] == pytest.approx(4.5)
    assert row["bootstrap_duration_median"] == 40
    assert row["bootstrap_duration_iqr"] == 2
    assert row["present_in_medoid"] is True
    assert row["medoid_center_distance_days"] == 8


@pytest.mark.parametrize(
    "bootstrap,individual,expected",
    [
        (0.90, 0.60, "ROBUST_REFERENCE"),
        (0.90, 0.20, "STABLE_BUT_LOW_SAMPLE_SUPPORT"),
        (0.30, 0.80, "UNSTABLE_REFERENCE"),
        (0.65, 0.40, "INTERMEDIATE_REFERENCE"),
    ],
)
def test_reference_confidence_categories(bootstrap, individual, expected):
    from analysis.structure_reference_validity_diagnostic import reference_confidence

    assert reference_confidence(bootstrap, individual) == expected


def test_medoid_is_an_actual_sample_and_uses_all_mode13_dimensions():
    from analysis.structure_reference_validity_diagnostic import select_multivariate_medoid

    curves = np.array([
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 100.0]],
        [[0.0, 0.0], [0.0, 101.0]],
    ])
    result = select_multivariate_medoid(curves, max_samples=128, seed=1)
    assert result.sample_index == 1
    assert np.array_equal(result.curve, curves[1])
    assert result.curve.base is None or np.shares_memory(result.curve, curves)


def test_class_and_total_summaries_count_stable_low_support_separately():
    from analysis.structure_reference_validity_diagnostic import (
        build_class_summary,
        build_total_summary,
    )

    rows = [
        {"source_domain": "AT1", "class_id": 0, "class_name": "corn", "reference_confidence": "ROBUST_REFERENCE", "individual_occurrence_rate": 0.7, "bootstrap_occurrence_rate": 0.9, "present_in_medoid": True},
        {"source_domain": "AT1", "class_id": 0, "class_name": "corn", "reference_confidence": "STABLE_BUT_LOW_SAMPLE_SUPPORT", "individual_occurrence_rate": 0.2, "bootstrap_occurrence_rate": 0.85, "present_in_medoid": False},
        {"source_domain": "DK1", "class_id": 1, "class_name": "wheat", "reference_confidence": "UNSTABLE_REFERENCE", "individual_occurrence_rate": 0.1, "bootstrap_occurrence_rate": 0.3, "present_in_medoid": False},
    ]
    classes = build_class_summary(rows, {("AT1", 0): 3, ("DK1", 1): 1})
    assert classes[0]["num_stable_low_sample_support"] == 1
    total = build_total_summary(rows, classes)
    assert total["num_source_domains"] == 2
    assert total["num_reference_structures"] == 3
    assert total["num_stable_low_sample_support"] == 1
    assert total["fraction_reference_bootstrap_ge_08"] == pytest.approx(2 / 3)


def test_csv_schema_manifest_and_both_diagnostic_figures(tmp_path):
    from analysis.structure_reference_validity_diagnostic import (
        plot_bootstrap_stability,
        plot_median_vs_medoid,
        write_source_outputs,
    )

    structures = [_segment("C0", center=120, occurrence=0.3)]
    rows = [{
        "source_domain": "AT1", "class_id": 0, "class_name": "corn",
        "reference_structure_id": "C0", "direction": "RISE",
        "start_day": 100.0, "end_day": 140.0, "center_day": 120.0,
        "duration_days": 40.0, "individual_occurrence_rate": 0.3,
        "bootstrap_occurrence_rate": 0.9, "bootstrap_center_mad_days": 2.0,
        "bootstrap_duration_median": 41.0, "bootstrap_duration_iqr": 3.0,
        "bootstrap_change_median": 2.0, "bootstrap_change_iqr": 0.2,
        "bootstrap_monotonicity_median": 0.8, "bootstrap_monotonicity_iqr": 0.1,
        "present_in_medoid": True, "medoid_center_distance_days": 2.0,
        "medoid_duration_ratio": 1.1, "reference_confidence": "STABLE_BUT_LOW_SAMPLE_SUPPORT",
        "num_bootstrap_runs": 100, "num_bootstrap_matches": 90,
    }]
    class_rows = [{"source_domain": "AT1", "class_id": 0, "class_name": "corn", "num_reference_structures": 1}]
    out = tmp_path / "AT1"
    write_source_outputs(out, rows, class_rows, [{"repeat": 0, "class_id": 0, "num_structures": 1}], {"source_domain": "AT1"})
    with (out / "structure_stability.csv").open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    for field in ("bootstrap_occurrence_rate", "individual_occurrence_rate", "present_in_medoid", "reference_confidence"):
        assert field in header
    assert json.loads((out / "manifest.json").read_text())["source_domain"] == "AT1"

    grid = np.arange(365.0)
    reference_curve = np.sin(2 * np.pi * grid / 365)
    bootstrap_curves = np.stack([reference_curve, reference_curve + 0.1])
    medoid_curve = reference_curve - 0.1
    first = out / "diagnostics" / "corn_bootstrap_stability.png"
    second = out / "diagnostics" / "corn_median_vs_medoid.png"
    plot_bootstrap_stability(first, grid, reference_curve, bootstrap_curves, structures, rows)
    plot_median_vs_medoid(second, grid, reference_curve, medoid_curve, structures, structures)
    assert first.is_file() and first.stat().st_size > 0
    assert second.is_file() and second.stat().st_size > 0


def test_runner_is_source_only_reuses_05_and_launcher_runs_each_source_once():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/diagnose_structure_reference_validity.py").read_text()
    launcher = (root / "scripts/run_structure_reference_validity_4sources.sh").read_text()
    assert "fit_class_projections" in runner
    assert "build_coarse_structure" in runner
    assert "source_coarse_segments.csv" in runner
    assert "target" not in runner.lower()
    for forbidden in ("train(", "optimizer", "backward(", "SRVF", "warp"):
        assert forbidden.lower() not in runner.lower()
    for alias in ("AT1", "DK1", "FR1", "FR2"):
        assert launcher.count(f" {alias} ") >= 1
    assert "--prototype-bootstrap-repeats" in launcher
    assert "--prototype-bootstrap-fraction" in launcher
    assert "--prototype-bootstrap-seed" in launcher


def test_runner_fits_raw_pse_pc1_once_and_reuses_it_for_all_mode13_structures():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/diagnose_structure_reference_validity.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    fit_calls = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "fit_class_projections"
    ]
    assert len(fit_calls) == 1

    medoid_calls = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "select_multivariate_medoid"
    ]
    assert len(medoid_calls) == 1
    assert isinstance(medoid_calls[0].args[0], ast.Name)
    assert medoid_calls[0].args[0].id == "mode13"

    transformed_names = {
        ast.unparse(node.args[0])
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "transform"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "projection"
    }
    assert transformed_names == {
        "reference_multi[None]",
        "prototype_multi[None]",
        "medoid.curve[None]",
    }
