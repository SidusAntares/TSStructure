import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _segment(
    structure_id,
    direction="RISE",
    center=100.0,
    duration=40.0,
    accepted=True,
):
    return SimpleNamespace(
        coarse_segment_id=structure_id,
        direction=direction,
        center_day=float(center % 365),
        duration_days=float(duration),
        start_day=float((center - duration / 2) % 365),
        end_day=float((center + duration / 2) % 365),
        accepted=accepted,
    )


def test_normal_and_circular_observation_support_include_window_boundaries():
    from analysis.structure_observation_support_diagnostic import (
        observation_support_metrics,
    )

    normal = observation_support_metrics(
        np.array([5, 10, 30, 50, 75, 95]), 10, 90, radius_days=15
    )
    assert normal.num_observations == 4
    assert normal.observation_density == pytest.approx(4 / 80)
    assert normal.max_observation_gap_days == 25
    assert normal.nearest_start_observation_days == 0
    assert normal.nearest_end_observation_days == 5
    assert normal.quarter_coverage == 1.0

    circular = observation_support_metrics(
        np.array([350, 5, 20, 200]), 340, 30, radius_days=15
    )
    assert circular.crosses_year_boundary
    assert circular.window_end_unwrapped == 395
    assert circular.num_observations == 3
    assert circular.max_observation_gap_days == 20
    assert circular.nearest_start_observation_days == 10
    assert circular.nearest_end_observation_days == 10


def test_zero_observation_padding_exclusion_quarters_and_radius_coverage():
    from analysis.structure_observation_support_diagnostic import (
        observation_support_metrics,
    )

    empty = observation_support_metrics([], 100, 140, radius_days=15)
    assert empty.num_observations == 0
    assert empty.max_observation_gap_days == 40
    assert np.isnan(empty.nearest_start_observation_days)
    assert empty.quarter_coverage == 0
    assert empty.support_coverage_radius == 0

    metrics = observation_support_metrics(
        np.array([100, 110, 130, 140, 999]),
        100,
        140,
        radius_days=0,
        valid_time_mask=np.array([True, True, True, True, False]),
    )
    assert metrics.num_observations == 4
    assert metrics.quarter_coverage == 0.75
    assert metrics.support_coverage_radius == pytest.approx(4 / 41)


def test_each_acquisition_timestep_counts_even_when_timestamps_repeat():
    from analysis.structure_observation_support_diagnostic import observation_support_metrics

    metrics = observation_support_metrics([100, 100, 120], 90, 130)
    assert metrics.num_observations == 3
    assert metrics.observation_density == pytest.approx(3 / 40)


def test_matching_and_failure_reasons_include_real_assignment_conflict():
    from analysis.structure_observation_support_diagnostic import audit_sample_matches

    references = [_segment("SC0", center=100), _segment("SC1", center=110)]
    candidate = _segment("IC0", center=103)
    audits = audit_sample_matches(references, [candidate], 40, 2.5, "COARSE_SEGMENT_EXISTS")
    assert audits["SC0"].matched
    assert audits["SC1"].failure_reason == "ASSIGNMENT_CONFLICT"

    assert audit_sample_matches(
        [_segment("SC0")], [], 40, 2.5, "NO_MEMBER_EXTREMA"
    )["SC0"].failure_reason == "NO_COARSE_SEGMENT"
    assert audit_sample_matches(
        [_segment("SC0", "RISE")], [_segment("I0", "FALL")], 40, 2.5,
        "COARSE_SEGMENT_EXISTS",
    )["SC0"].failure_reason == "NO_SAME_DIRECTION_SEGMENT"
    assert audit_sample_matches(
        [_segment("SC0", center=100)], [_segment("I0", center=170)], 40, 2.5,
        "COARSE_SEGMENT_EXISTS",
    )["SC0"].failure_reason == "CENTER_DISTANCE_FAIL"
    assert audit_sample_matches(
        [_segment("SC0", center=100, duration=20)],
        [_segment("I0", center=105, duration=100)], 40, 2.5,
        "COARSE_SEGMENT_EXISTS",
    )["SC0"].failure_reason == "DURATION_RATIO_FAIL"


@pytest.mark.parametrize(
    "member_count,coarse_count,accepted_count,expected",
    [
        (0, 0, 0, "NO_MEMBER_EXTREMA"),
        (1, 0, 0, "INSUFFICIENT_MEMBER_EXTREMA"),
        (2, 0, 0, "COARSE_SIMPLIFICATION_NO_SURVIVING_SEGMENT"),
        (2, 2, 0, "COARSE_GATE_REJECTED"),
        (2, 2, 1, "COARSE_SEGMENT_EXISTS"),
    ],
)
def test_detector_state(member_count, coarse_count, accepted_count, expected):
    from analysis.structure_observation_support_diagnostic import detector_state

    segments = [SimpleNamespace(accepted=index < accepted_count) for index in range(coarse_count)]
    assert detector_state(member_count, segments) == expected


def test_exact_join_filters_bootstrap_and_rejects_missing_or_duplicate_rows():
    from analysis.structure_observation_support_diagnostic import join_reliable_references

    five = [
        {"source_domain": "AT1", "class_id": "0", "coarse_segment_id": "SC0"},
        {"source_domain": "AT1", "class_id": "0", "coarse_segment_id": "SC1"},
    ]
    six = [
        {"source_domain": "AT1", "class_id": "0", "reference_structure_id": "SC0", "bootstrap_occurrence_rate": "0.9", "individual_occurrence_rate": "0.6"},
        {"source_domain": "AT1", "class_id": "0", "reference_structure_id": "SC1", "bootstrap_occurrence_rate": "0.7", "individual_occurrence_rate": "0.5"},
    ]
    joined = join_reliable_references(five, six, min_bootstrap_occurrence=0.8)
    assert [row["coarse_segment_id"] for row in joined] == ["SC0"]
    with pytest.raises(ValueError, match="missing"):
        join_reliable_references(five, six[:1] + [{**six[1], "reference_structure_id": "SC9"}])
    with pytest.raises(ValueError, match="duplicate"):
        join_reliable_references(five, six + [six[0]])


def test_cliffs_delta_and_structure_summary_are_structure_local():
    from analysis.structure_observation_support_diagnostic import (
        cliffs_delta,
        summarize_structures,
    )

    assert cliffs_delta([3, 4], [1, 2]) == 1.0
    rows = []
    for structure, matched, value in (
        ("SC0", True, 10), ("SC0", False, 2),
        ("SC1", True, 100), ("SC1", False, 90),
    ):
        rows.append({
            "task": "AT1_DK1", "source_domain": "AT1", "class_id": 0,
            "class_name": "corn", "reference_structure_id": structure,
            "direction": "RISE", "reference_start_day": 10,
            "reference_end_day": 50, "reference_duration_days": 40,
            "crosses_year_boundary": False, "bootstrap_occurrence_rate": 0.9,
            "individual_occurrence_rate": 0.5, "matched": matched,
            "num_observations": value, "max_observation_gap_days": 40 - value,
            "support_coverage_radius": value / 100,
            "nearest_start_observation_days": 1,
            "nearest_end_observation_days": 2,
            "failure_reason": "" if matched else "CENTER_DISTANCE_FAIL",
        })
    summary = summarize_structures(rows)
    assert len(summary) == 2
    first = next(row for row in summary if row["reference_structure_id"] == "SC0")
    assert first["Gplus_median_num_obs"] == 10
    assert first["Gminus_median_num_obs"] == 2
    assert first["delta_num_obs"] == -8
    assert first["cliffs_delta_num_obs"] == -1


def test_failure_and_boundary_summaries_have_total_and_boundary_groups():
    from analysis.structure_observation_support_diagnostic import (
        summarize_boundaries,
        summarize_failure_reasons,
    )

    rows = [
        {"task": "T", "class_id": 0, "class_name": "corn", "reference_structure_id": "SC0", "matched": False, "failure_reason": "NO_COARSE_SEGMENT", "crosses_year_boundary": True, "num_observations": 2, "max_observation_gap_days": 30, "support_coverage_radius": 0.4},
        {"task": "T", "class_id": 0, "class_name": "corn", "reference_structure_id": "SC0", "matched": True, "failure_reason": "", "crosses_year_boundary": True, "num_observations": 4, "max_observation_gap_days": 10, "support_coverage_radius": 0.8},
        {"task": "T", "class_id": 1, "class_name": "wheat", "reference_structure_id": "SC1", "matched": False, "failure_reason": "CENTER_DISTANCE_FAIL", "crosses_year_boundary": False, "num_observations": 3, "max_observation_gap_days": 20, "support_coverage_radius": 0.6},
    ]
    failures = summarize_failure_reasons(rows)
    assert any(row["scope"] == "TOTAL" for row in failures)
    assert next(row for row in failures if row["scope"] == "TOTAL")["num_unmatched"] == 2
    boundaries = summarize_boundaries(rows)
    assert {row["cross_boundary"] for row in boundaries} == {True, False}
    boundary = next(row for row in boundaries if row["cross_boundary"])
    assert boundary["matched_rate"] == 0.5
    assert boundary["mean_structure_matched_rate"] == 0.5
    assert boundary["median_structure_matched_rate"] == 0.5


def test_random_and_gap_masks_only_remove_window_acquisitions_and_are_deterministic():
    from analysis.structure_observation_support_diagnostic import (
        contiguous_gap_mask,
        random_window_mask,
    )

    positions = np.array([0, 90, 100, 110, 120, 130, 200])
    first = random_window_mask(positions, 100, 130, 0.25, seed=4)
    second = random_window_mask(positions, 100, 130, 0.25, seed=4)
    assert np.array_equal(first, second)
    assert first.sum() == 1
    assert set(np.flatnonzero(first)).issubset({2, 3, 4, 5})

    gap = contiguous_gap_mask(positions, 100, 130, gap_days=20, seed=7)
    assert set(np.flatnonzero(gap)).issubset({2, 3, 4, 5})
    skipped = contiguous_gap_mask(positions, 100, 130, gap_days=60, seed=7)
    assert skipped is None


def test_delete_acquisitions_removes_time_axis_from_all_temporal_inputs():
    from analysis.structure_observation_support_diagnostic import delete_acquisitions

    sample = {
        "pixels": np.arange(5 * 2 * 3).reshape(5, 2, 3),
        "valid_pixels": np.ones((5, 3)),
        "positions": np.arange(5),
        "extra": np.array([9]),
    }
    masked = delete_acquisitions(sample, np.array([False, True, False, True, False]))
    assert masked["pixels"].shape[0] == 3
    assert masked["valid_pixels"].shape[0] == 3
    assert masked["positions"].tolist() == [0, 2, 4]
    assert masked["extra"].tolist() == [9]


def test_masking_summary_recovery_rate_uses_only_valid_runs():
    from analysis.structure_observation_support_diagnostic import summarize_masking

    rows = [
        {"task": "T", "source_domain": "AT1", "class_id": 0, "class_name": "corn", "reference_structure_id": "SC0", "crosses_year_boundary": True, "sample_id": 7, "mask_type": "random", "mask_level": 0.5, "valid_mask_run": True, "masked_matched": True, "masked_num_obs": 3, "masked_max_gap": 20, "masked_support_coverage": 0.7, "failure_reason": ""},
        {"task": "T", "source_domain": "AT1", "class_id": 0, "class_name": "corn", "reference_structure_id": "SC0", "crosses_year_boundary": True, "sample_id": 7, "mask_type": "random", "mask_level": 0.5, "valid_mask_run": True, "masked_matched": False, "masked_num_obs": 2, "masked_max_gap": 30, "masked_support_coverage": 0.5, "failure_reason": "NO_COARSE_SEGMENT"},
        {"task": "T", "source_domain": "AT1", "class_id": 0, "class_name": "corn", "reference_structure_id": "SC0", "crosses_year_boundary": True, "sample_id": 7, "mask_type": "random", "mask_level": 0.5, "valid_mask_run": False, "masked_matched": False, "masked_num_obs": 0, "masked_max_gap": 40, "masked_support_coverage": 0, "failure_reason": "INVALID_MASK"},
    ]
    summary = summarize_masking(rows)
    item = next(row for row in summary if row["scope"] == "STRUCTURE")
    assert item["num_samples"] == 1
    assert item["num_runs"] == 3
    assert item["num_valid_runs"] == 2
    assert item["crosses_year_boundary"] is True
    assert item["recovery_rate"] == 0.5
    assert item["loss_rate"] == 0.5
    assert any(row["scope"] == "TOTAL" for row in summary)


def test_masking_reference_selection_includes_all_low_support_and_boundary_controls():
    from analysis.structure_observation_support_diagnostic import select_masking_references

    rows = [
        {"class_id": 0, "coarse_segment_id": "LOW", "bootstrap_occurrence_rate": 0.9, "individual_occurrence_rate": 0.4, "duration_days": 50, "crosses_year_boundary": False},
        {"class_id": 1, "coarse_segment_id": "HCROSS", "bootstrap_occurrence_rate": 0.9, "individual_occurrence_rate": 0.95, "duration_days": 55, "crosses_year_boundary": True},
        {"class_id": 2, "coarse_segment_id": "HNORMAL", "bootstrap_occurrence_rate": 0.9, "individual_occurrence_rate": 0.95, "duration_days": 52, "crosses_year_boundary": False},
        {"class_id": 3, "coarse_segment_id": "MID", "bootstrap_occurrence_rate": 0.9, "individual_occurrence_rate": 0.7, "duration_days": 50, "crosses_year_boundary": False},
    ]
    selected = select_masking_references(rows)
    assert {row["coarse_segment_id"] for row in selected} == {"LOW", "HCROSS", "HNORMAL"}


def test_masked_curve_path_reexecutes_spatial_encoder_on_deleted_sequences():
    from scripts.diagnose_structure_observation_support import _masked_curves

    class Spatial(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(0.0))
            self.lengths = []

        def forward(self, pixels, valid_pixels, extra):
            self.lengths.extend([pixels.shape[1]] * pixels.shape[0])
            return pixels.mean(dim=-1) + self.marker

    class Analyzer:
        def __call__(self, features, positions, collect_diagnostics=False):
            return torch.complex(features, torch.zeros_like(features)), {}

    class Synthesizer:
        def __call__(self, coefficients, positions):
            pooled = coefficients.real.mean(dim=1)
            return pooled[:, None, :].expand(-1, positions.shape[1], -1)

    class Projection:
        def transform(self, curves):
            return curves[..., 0]

    variants = [
        {"pixels": torch.ones(length, 2, 3), "valid_pixels": torch.ones(length, 3), "positions": torch.arange(length), "extra": torch.zeros(1)}
        for length in (3, 2, 3)
    ]
    spatial = Spatial()
    curves = _masked_curves(
        variants, spatial, Analyzer(), Synthesizer(), Projection(),
        torch.device("cpu"), with_extra=False,
    )
    assert sorted(spatial.lengths) == [2, 3, 3]
    assert len(curves) == 3
    assert all(curve.shape == (365,) for curve in curves)


def test_staged_output_preserves_old_on_failure_and_atomically_replaces_on_success(tmp_path):
    from analysis.structure_observation_support_diagnostic import staged_output

    final = tmp_path / "TASK"
    final.mkdir()
    (final / "old.txt").write_text("old")
    with pytest.raises(RuntimeError):
        with staged_output(final, required_files=("manifest.json",)) as staging:
            (staging / "partial.txt").write_text("partial")
            raise RuntimeError("fail")
    assert (final / "old.txt").read_text() == "old"

    with staged_output(final, required_files=("manifest.json",)) as staging:
        (staging / "manifest.json").write_text("{}")
    assert not (final / "old.txt").exists()
    assert json.loads((final / "manifest.json").read_text()) == {}


def test_runner_is_source_only_reruns_pse_after_real_mask_and_keeps_pc1_fixed():
    root = Path(__file__).resolve().parents[1]
    runner_path = root / "scripts/diagnose_structure_observation_support.py"
    launcher_path = root / "scripts/run_structure_observation_support_4tasks.sh"
    assert runner_path.is_file()
    assert launcher_path.is_file()
    runner = runner_path.read_text(encoding="utf-8").lower()
    launcher = launcher_path.read_text(encoding="utf-8")
    for forbidden in ("target_dataset", "optimizer", "backward(", "srvf", "warp"):
        assert forbidden not in runner
    assert "delete_acquisitions" in runner
    assert "spatial_encoder(" in runner
    assert '"label": "baseline g+ sample"' in runner
    assert "pc1_refit" in runner
    assert '"pc1_refit": false' in runner
    assert "--diagnose-structure-observation-support" in runner
    assert "--run-observation-masking-ablation" in runner
    for gpu, task in enumerate(("AT1_DK1", "DK1_FR1", "FR1_FR2", "FR2_AT1")):
        assert f"run_one {gpu} {task}" in launcher
