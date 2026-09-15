"""Time-free identity, calibrated confidence and post-match timing contracts."""
import importlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest


def api():
    return importlib.import_module("analysis.structure_identity_phase_diagnostic")


def segment(sid="S0", direction="RISE", amplitude=1., start=10., duration=30., **kwargs):
    sign = 1 if direction == "RISE" else -1
    result = dict(coarse_segment_id=sid, direction=direction,
                  curve_normalized_change=amplitude, domain_normalized_change=amplitude,
                  monotonicity_ratio=1., num_fine_segments_covered=1,
                  start_day=start, end_day=(start + duration) % 365,
                  center_day=(start + duration / 2) % 365, duration_days=duration,
                  start_value=0., end_value=sign * amplitude, signed_change=sign * amplitude,
                  absolute_change=amplitude, total_path_variation=amplitude,
                  accepted=True, crosses_year_boundary=start + duration >= 365)
    result.update(kwargs)
    return result


def calibration(threshold=1.):
    return dict(scales=[1., 1., 1., 1.], active=[True] * 4,
                positive_pairs=[20] * 4, num_active_components=4,
                threshold=threshold, calibration_level="source_direction")


def test_descriptor_excludes_all_time_and_acceptance_fields():
    m = api()
    a = segment()
    b = segment(start=180, duration=180, accepted=False)
    assert m.descriptor(a) == m.descriptor(b)
    assert set(asdict(m.descriptor(a))) == {"direction", "curve_change", "domain_change", "monotonicity", "fine_count"}
    np.testing.assert_allclose(m.differences(m.descriptor(a), m.descriptor(segment(amplitude=2))), [np.log(2), np.log(2), 0, 0], atol=1e-7)


def test_calibration_quantiles_and_all_fallback_levels():
    m = api()
    records = [dict(source_domain=domain, direction=direction, differences=[i + 1., 2., .1, .3])
               for domain in ("AT1", "DK1", "FR1", "FR2")
               for direction, count in (("RISE", 4 if domain == "AT1" else 1), ("FALL", 1))
               for i in range(count)]
    result = m.calibrate(records, min_pairs=4, min_positive_pairs=1)
    assert result["AT1:RISE"]["calibration_level"] == "source_direction"
    assert result["AT1:FALL"]["calibration_level"] == "source_pooled"
    assert result["DK1:RISE"]["calibration_level"] == "all_source_pooled"
    expected = np.quantile(np.array([[i + 1., 2, .1, .3] for i in range(4)]), .9, axis=0)
    np.testing.assert_allclose(result["AT1:RISE"]["scales"], expected)
    costs = (np.array([[i + 1., 2, .1, .3] for i in range(4)]) / expected).mean(axis=1)
    assert result["AT1:RISE"]["threshold"] == pytest.approx(np.quantile(costs, .95))
    with pytest.raises(ValueError, match="four sources"):
        m.calibrate(records[:-1], min_pairs=4, source_domains=("AT1", "DK1", "FR1"))
    with pytest.raises(ValueError, match="insufficient"):
        m.calibrate(records, min_pairs=100)


def test_circular_alignment_rotation_and_deterministic_ties():
    m = api()
    costs = np.array([[np.inf, 0, np.inf], [np.inf, np.inf, 0], [0, np.inf, np.inf]])
    result = m.circular_alignment(costs, [1, 1, 1])
    assert result.pairs == ((0, 1), (1, 2), (2, 0))
    assert result.rotation == 1
    tied = m.circular_alignment(np.zeros((2, 2)), [1, 1])
    assert tied.rotation == 0
    assert tied.pairs == ((0, 0), (1, 1))
    partial = m.circular_alignment(np.array([[0, np.inf], [np.inf, np.inf], [np.inf, 0]]), [1]*3)
    assert partial.pairs == ((0, 0), (2, 1))


def test_alignment_maximizes_count_before_cost_and_preserves_order():
    m = api()
    result = m.circular_alignment(np.array([[.9, 0, np.inf], [np.inf, .9, np.inf]]), [1, 1])
    assert len(result.pairs) == 2
    assert result.pairs == ((0, 0), (1, 1))


def test_rejected_candidate_is_secondary_plausible_despite_large_phase_duration():
    m = api()
    refs = [segment("R0")]
    candidates = [segment("S0", amplitude=1.1, start=170, duration=160, accepted=False),
                  segment("S1", amplitude=100, start=12)]
    rows, alignment = m.identify(refs, candidates, {"RISE": calibration()})
    assert alignment.pairs == ()
    assert rows[0]["identity_status"] == "STRUCTURALLY_PLAUSIBLE"
    assert rows[0]["candidate_source"] == "SECONDARY_REJECTED"
    assert rows[0]["sample_segment_accepted"] is False
    assert not rows[0]["unique_mutual_best"]
    assert rows[0]["duration_ratio"] > 2.5
    assert abs(rows[0]["center_displacement_days"]) > 40


def test_accepted_unique_mutual_best_is_likely_and_rejected_never_likely():
    m = api()
    rows, _ = m.identify([segment("R")], [segment("P", amplitude=1.1)], {"RISE": calibration()})
    assert rows[0]["identity_status"] == "LIKELY_SAME_STRUCTURE"
    assert rows[0]["candidate_source"] == "PRIMARY_ACCEPTED"
    assert rows[0]["reference_unique_best"] and rows[0]["sample_unique_best"]
    assert rows[0]["unique_mutual_best"]
    rows, _ = m.identify([segment("R")], [segment("S", amplitude=1.1, accepted=False)], {"RISE": calibration()})
    assert rows[0]["identity_status"] == "STRUCTURALLY_PLAUSIBLE"
    assert rows[0]["candidate_source"] == "SECONDARY_REJECTED"


@pytest.mark.parametrize("costs,expected", [
    ([[0., 0.]], "AMBIGUOUS"),
    ([[0.], [0.]], "STRUCTURALLY_PLAUSIBLE"),
    ([[np.inf]], "NO_STRUCTURAL_COUNTERPART"),
])
def test_confidence_ties_and_labels(costs, expected):
    m = api()
    a = np.array(costs)
    alignment = m.circular_alignment(a, np.ones(a.shape[0]))
    result = m.classify_alignment(a, np.ones(a.shape[0]), alignment)
    assert result[0]["identity_status"] == expected
    assert not result[0]["unique_mutual_best"]
    if a.shape == (2, 1):
        assert result[1]["identity_status"] == "AMBIGUOUS"
        assert result[1]["assignment_conflict"]


def test_neighbors_give_context_but_not_mutual_best_on_cost_ties():
    m = api()
    costs = np.zeros((4, 4))
    result = m.classify_alignment(costs, np.ones(4), m.circular_alignment(costs, np.ones(4)))
    assert all(row["neighbor_support"] == 2 for row in result)
    assert all(not row["unique_mutual_best"] for row in result)
    assert all(row["identity_status"] == "AMBIGUOUS" for row in result)


def test_zero_heavy_calibration_disables_component_instead_of_epsilon_scale():
    m = api()
    records = [dict(source_domain=source, direction="RISE",
                    differences=[0. if i < 23 else .2, .1 + i/100, .2 + i/100, 0.])
               for source in m.SOURCES for i in range(25)]
    row = m.calibrate(records, min_pairs=20, min_positive_pairs=20,
                      min_active_components=2)["AT1:RISE"]
    assert row["active"] == [False, True, True, False]
    assert np.isnan(row["scales"][0]) and np.isnan(row["scales"][3])
    assert row["positive_pairs"] == [2, 25, 25, 0]
    assert row["num_active_components"] == 2
    assert np.isfinite(row["threshold"]) and row["threshold"] < 1e4
    assert m.structural_cost(m.Descriptor("RISE", 1, 1, .8, 1),
                             m.Descriptor("RISE", 100, 1.2, .9, 99), row) < 10


def test_positive_only_quantile_and_active_denominator():
    m = api()
    records = [dict(source_domain=s, direction="RISE", differences=[0, 0, .1, .2])
               for s in m.SOURCES for _ in range(20)]
    for row, value in zip(records[:20], np.linspace(.1, 2, 20)):
        row["differences"][0] = value
    row = m.calibrate(records, min_pairs=20, min_positive_pairs=20,
                      min_active_components=2)["AT1:RISE"]
    assert row["scales"][0] == pytest.approx(np.quantile(np.linspace(.1, 2, 20), .9))
    assert row["active"] == [True, False, True, True]
    reference = m.Descriptor("RISE", 1, 1, .8, 1)
    sample = m.Descriptor("RISE", np.exp(row["scales"][0]), 1000,
                          .8 + row["scales"][2], 2)
    _, contribution = m.cost_components(reference, sample, row)
    assert np.isnan(contribution[1])
    assert m.structural_cost(reference, sample, row) == pytest.approx(np.nanmean(contribution))


def test_insufficient_active_components_falls_back_then_hard_fails():
    m = api()
    records = [dict(source_domain=source, direction=direction,
                    differences=[i/20 if direction == "RISE" else 0,
                                 i/20 if direction == "FALL" else 0, 0, 0])
               for source in m.SOURCES for direction in ("RISE", "FALL")
               for i in range(1, 21)]
    result = m.calibrate(records, min_pairs=20, min_positive_pairs=20,
                         min_active_components=2)
    assert result["AT1:RISE"]["calibration_level"] == "source_pooled"
    with pytest.raises(ValueError, match="active components"):
        m.calibrate(records, min_pairs=20, min_positive_pairs=20,
                    min_active_components=3)


def test_direction_mismatch_and_ambiguous_timing_are_nan():
    m = api()
    rows, _ = m.identify([segment("R")], [segment("S", "FALL")], {"RISE": calibration()})
    assert rows[0]["identity_status"] == "NO_STRUCTURAL_COUNTERPART"
    assert np.isnan(rows[0]["center_displacement_days"])
    rows, _ = m.identify([segment("R")], [segment("S0"), segment("S1")], {"RISE": calibration()})
    assert rows[0]["identity_status"] == "AMBIGUOUS"
    assert all(np.isnan(rows[0][k]) for k in m.TIMING_FIELDS)


def test_circular_timing_unwrap_and_stretch():
    m = api()
    timing = m.measure_timing(segment(start=350, duration=30), segment(start=5, duration=50))
    assert timing["start_displacement_days"] == 20
    assert timing["end_displacement_days"] == 40
    assert timing["center_displacement_days"] == 30
    assert timing["duration_ratio"] == pytest.approx(5/3)
    assert timing["stretch_difference_days"] == timing["duration_difference_days"] == 20


@pytest.mark.parametrize("direction,sign", [("RISE", 1), ("FALL", -1)])
def test_grouped_three_segments_uses_net_change_and_total_path(direction, sign):
    m = api()
    opposite = "FALL" if direction == "RISE" else "RISE"
    seq = [segment("S0", direction, 2, start_value=0, end_value=2*sign),
           segment("S1", opposite, .2, start_value=2*sign, end_value=1.8*sign),
           segment("S2", direction, 2, start_value=1.8*sign, end_value=3.8*sign)]
    groups = m.grouped_candidates(seq, .6)
    assert len(groups) == 1
    assert groups[0]["curve_normalized_change"] == pytest.approx(3.8)
    assert groups[0]["monotonicity_ratio"] == pytest.approx(3.8/4.2)
    assert groups[0]["num_fine_segments_covered"] == 3
    assert not m.grouped_candidates(seq, .99)


def test_summary_rescue_consistency_grouped_and_failure_transitions():
    m = api()
    base = dict(task="FR2_AT1", source_domain="FR2", class_id=7, class_name="barley",
                reference_structure_id="SC0", reference_cross_boundary=True,
                bootstrap_occurrence=.9, baseline_individual_occurrence=.5,
                center_displacement_days=70., duration_ratio=3., stretch_difference_days=60.,
                has_grouped_candidate=False, identity_sample_segment_id="S0", baseline_matched_segment_id="S0",
                candidate_source="PRIMARY_ACCEPTED", gplus_exact_segment_recovered=False)
    rows = [dict(base, baseline_06B_status="G+", baseline_failure_reason="", identity_status="LIKELY_SAME_STRUCTURE", gplus_exact_segment_recovered=True),
            dict(base, baseline_06B_status="G-", baseline_failure_reason="CENTER_DISTANCE_FAIL", identity_status="LIKELY_SAME_STRUCTURE"),
            dict(base, baseline_06B_status="G-", baseline_failure_reason="DURATION_RATIO_FAIL", identity_status="STRUCTURALLY_PLAUSIBLE",
                 candidate_source="SECONDARY_REJECTED", has_grouped_candidate=True)]
    summary = m.summarize_structures(rows)[0]
    assert summary["Gminus_high_confidence_rescue_rate"] == .5
    assert summary["Gminus_plausible_or_better_rate"] == 1
    assert summary["gplus_exact_segment_rate"] == 1
    assert summary["num_grouped_candidates"] == 1
    assert summary["Gminus_primary_likely"] == 1
    assert summary["Gminus_secondary_plausible"] == 1
    transitions = m.summarize_transitions(rows)
    assert any(r["scope"] == "TOTAL" and r["candidate_source"] == "SECONDARY_REJECTED"
               and r["count"] == 1 for r in transitions)
    boundaries = m.summarize_boundaries(rows)
    assert boundaries[0]["high_confidence_rescue_rate"] == .5


def test_baseline_exact_join_rejects_duplicate_missing_and_stale_segment():
    m = api()
    rows = [dict(source_domain="AT1", class_id=0, reference_structure_id="R0", sample_id=4,
                 matched="True", matched_sample_segment_id="S0", failure_reason="")]
    expected = [("AT1", 0, "R0", 4)]
    assert len(m.index_baseline(rows, expected)) == 1
    with pytest.raises(ValueError, match="duplicate"):
        m.index_baseline(rows * 2, expected)
    with pytest.raises(ValueError, match="missing|mismatch"):
        m.index_baseline([], expected)


def test_output_transaction_and_runner_contract(tmp_path):
    m = api()
    final = tmp_path / "AT1_DK1"
    final.mkdir()
    (final / "old").write_text("keep")
    with pytest.raises(Exception):
        with m.staged_output(final, ("manifest.json",)):
            raise RuntimeError("incomplete")
    assert (final / "old").read_text() == "keep"
    runner = importlib.import_module("scripts.diagnose_structure_identity_phase")
    parser = runner.build_parser()
    args = parser.parse_args(["--stage", "calibrate"])
    assert args.identity_component_scale_quantile == .9
    assert args.identity_cost_quantile == .95
    assert args.identity_calibration_min_pairs == 50
    assert args.identity_component_min_positive_pairs == 20
    assert args.identity_calibration_min_active_components == 2
    script = (Path(__file__).parents[1] / "scripts/run_structure_identity_phase_4tasks.sh").read_text()
    assert "--stage prepare" in script and "--stage calibrate" in script and "--stage audit" in script
    assert "--stage publish" in script
    assert ".tmp_06C_structure_identity_phase_revision_" in script
    assert "gplus_consistency_summary.csv" in runner.REQUIRED_OUTPUTS


def test_runner_audits_historical_labels_and_stale_gplus_fails():
    runner = importlib.import_module("scripts.diagnose_structure_identity_phase")
    payload = dict(class_id=0, class_name="corn", references=[dict(segment("SC0"), bootstrap_occurrence=.9, baseline_individual_occurrence=.5)],
                   samples=[dict(sample_id=1, segments=[segment("I0_C0", start=150, duration=120)],
                                 baseline=[dict(reference_structure_id="SC0", matched="False", failure_reason="CENTER_DISTANCE_FAIL", matched_sample_segment_id="")])])
    rows, evidence = runner.audit_prepared_class("AT1_DK1", payload, {"RISE": calibration()}, .6)
    assert rows[0]["baseline_06B_status"] == "G-"
    assert rows[0]["baseline_failure_reason"] == "CENTER_DISTANCE_FAIL"
    assert rows[0]["identity_status"] == "LIKELY_SAME_STRUCTURE"
    assert evidence[1].pairs == ((0, 0),)
    payload["samples"][0]["baseline"][0].update(matched="True", matched_sample_segment_id="missing")
    with pytest.raises(ValueError, match="G\\+.*segment"):
        runner.collect_gplus("AT1", payload)


def test_synthetic_four_source_calibrate_audit_and_pngs(tmp_path):
    import json
    runner = importlib.import_module("scripts.diagnose_structure_identity_phase")
    m = api()
    work = tmp_path / "work"
    work.mkdir()
    grid = np.arange(365.)
    for task, (source, _) in runner.TASKS.items():
        folder = work / task
        folder.mkdir()
        payload = dict(class_id=0, class_name="corn", references=[dict(segment("SC0"), bootstrap_occurrence=.9, baseline_individual_occurrence=.5)], samples=[])
        for i, reason in enumerate(("", "CENTER_DISTANCE_FAIL", "DURATION_RATIO_FAIL")):
            payload["samples"].append(dict(sample_id=i, segments=[segment(f"I{i}_C0", amplitude=1.1 + .01 * len(source), start=10 if not reason else 150, duration=30 if not reason else 120)],
                baseline=[dict(reference_structure_id="SC0", matched=str(not bool(reason)), matched_sample_segment_id="I0_C0" if not reason else "", failure_reason=reason)]))
        m.write_json(folder / "class_0.json", payload)
        np.savez_compressed(folder / "class_0.npz", prototype=np.sin(grid/40), curves=np.tile(np.sin(grid/40), (3, 1)))
        m.write_json(folder / "manifest.json", dict(task=task, source_domain=source, class_ids=[0], coarse_min_monotonicity=.6,
            gplus=runner.collect_gplus(source, payload), completed=True))
    args = runner.build_parser().parse_args(["--stage", "calibrate", "--work-root", str(work), "--output-root", str(tmp_path / "out"),
                                             "--identity-calibration-min-pairs", "1", "--identity-component-min-positive-pairs", "1"])
    runner.run_calibration(args)
    args.task = "AT1_DK1"
    runner.run_audit(args)
    output = args.output_root / args.task
    assert all((output / name).is_file() for name in runner.REQUIRED_OUTPUTS)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["gplus_identity_consistency_rate"] == 1
    assert manifest["target_used"] is False
    assert list((output / "diagnostics").glob("*identity.png"))
    assert list((output / "diagnostics").glob("*phase_distribution.png"))


def test_fragmentation_auxiliary_does_not_change_primary():
    m = api()
    ref = segment("R", amplitude=3.8, monotonicity_ratio=3.8/4.2, num_fine_segments_covered=3)
    seq = [segment("S0", amplitude=2, end_value=2), segment("S1", "FALL", .2, start_value=2, end_value=1.8),
           segment("S2", amplitude=2, start_value=1.8, end_value=3.8)]
    rows, _ = m.identify([ref], seq, {"RISE": calibration(.01)})
    assert rows[0]["identity_status"] == "NO_STRUCTURAL_COUNTERPART"
    group = m.audit_group(ref, m.grouped_candidates(seq, .6), calibration(.01))
    assert group["has_grouped_candidate"]
    assert rows[0]["identity_status"] == "NO_STRUCTURAL_COUNTERPART"


def test_alignment_agrees_with_exhaustive_small_problem():
    from itertools import combinations
    m = api()
    rng = np.random.default_rng(4)
    for _ in range(12):
        costs = rng.choice([0., .3, .7, np.inf], size=(3, 4))
        choices = []
        for rotation in range(4):
            order = [(j + rotation) % 4 for j in range(4)]
            for count in range(4):
                for left in combinations(range(3), count):
                    for right in combinations(range(4), count):
                        pairs = tuple(zip(left, [order[j] for j in right]))
                        score = sum(costs[i, j] for i, j in pairs)
                        if np.isfinite(score):
                            choices.append((-count, score, 7 - 2 * count, rotation, pairs))
        best = min(choices)
        actual = m.circular_alignment(costs, np.ones(3))
        assert (-len(actual.pairs), actual.total_cost, actual.gap_count, actual.rotation, actual.pairs) == best


def test_time_fields_perturbation_cannot_change_assignment():
    m = api()
    refs = [segment("R0", amplitude=1), segment("R1", "FALL", 2)]
    samples = [segment("S0", "FALL", 2), segment("S1", amplitude=1)]
    calibrations = {d: calibration() for d in ("RISE", "FALL")}
    before, a = m.identify(refs, samples, calibrations)
    changed = [dict(s, start_day=300., end_day=3., center_day=-9999, duration_days=9000.) for s in samples]
    after, b = m.identify(refs, changed, calibrations)
    assert a == b
    assert [r["identity_status"] for r in before] == [r["identity_status"] for r in after]


def test_gplus_consistency_requires_original_candidate_id():
    m = api()
    row = dict(task="AT1_DK1", source_domain="AT1", class_id=0, class_name="corn", reference_structure_id="R0",
               bootstrap_occurrence=1., baseline_individual_occurrence=1., baseline_06B_status="G+",
               identity_status="LIKELY_SAME_STRUCTURE", identity_sample_segment_id="S1", baseline_matched_segment_id="S0",
               candidate_source="PRIMARY_ACCEPTED", gplus_exact_segment_recovered=False)
    summary = m.summarize_structures([row])[0]
    assert summary["gplus_exact_segment_rate"] == 0
    assert summary["wrong_segment_likely_rate"] == 1
    assert np.isnan(summary["Gminus_high_confidence_rescue_rate"])


def test_gplus_any_exact_exact_likely_and_wrong_likely_are_distinct():
    m = api()
    base = dict(task="AT1_DK1", source_domain="AT1", class_id=0, class_name="corn",
                reference_structure_id="R0", baseline_06B_status="G+", baseline_failure_reason="",
                candidate_source="PRIMARY_ACCEPTED", has_grouped_candidate=False)
    rows = [dict(base, identity_status="LIKELY_SAME_STRUCTURE", identity_sample_segment_id="S0", baseline_matched_segment_id="S0", gplus_exact_segment_recovered=True),
            dict(base, identity_status="STRUCTURALLY_PLAUSIBLE", identity_sample_segment_id="S0", baseline_matched_segment_id="S0", gplus_exact_segment_recovered=True),
            dict(base, identity_status="LIKELY_SAME_STRUCTURE", identity_sample_segment_id="S1", baseline_matched_segment_id="S0", gplus_exact_segment_recovered=False),
            dict(base, identity_status="AMBIGUOUS", identity_sample_segment_id="", baseline_matched_segment_id="S0", gplus_exact_segment_recovered=False)]
    total = next(row for row in m.summarize_gplus(rows) if row["scope"] == "TOTAL")
    assert total["gplus_any_identity_rate"] == .75
    assert total["gplus_exact_segment_rate"] == .5
    assert total["gplus_exact_likely_rate"] == .25
    assert total["wrong_segment_likely_rate"] == .25


def test_secondary_timing_is_excluded_from_formal_phase_summary():
    m = api()
    base = dict(task="AT1_DK1", source_domain="AT1", class_id=0, class_name="corn", reference_structure_id="R",
                reference_cross_boundary=False, bootstrap_occurrence=.9, baseline_individual_occurrence=.5,
                baseline_06B_status="G-", baseline_failure_reason="CENTER_DISTANCE_FAIL", identity_status="STRUCTURALLY_PLAUSIBLE",
                has_grouped_candidate=False, identity_sample_segment_id="S", baseline_matched_segment_id="",
                gplus_exact_segment_recovered=False, duration_ratio=9., center_displacement_days=100., stretch_difference_days=80.)
    secondary = dict(base, candidate_source="SECONDARY_REJECTED")
    primary = dict(base, candidate_source="PRIMARY_ACCEPTED", duration_ratio=2., center_displacement_days=30., stretch_difference_days=10.)
    summary = m.summarize_structures([secondary, primary])[0]
    assert summary["median_center_displacement"] == 30
    assert summary["median_duration_ratio"] == 2


def test_nonadjacent_assignments_do_not_count_as_neighbors():
    m = api()
    costs = np.array([[0, np.inf, np.inf, np.inf], [np.inf, np.inf, 0, np.inf]])
    rows = m.classify_alignment(costs, [1, 1], m.circular_alignment(costs, [1, 1]))
    assert [r["neighbor_support"] for r in rows] == [0, 0]


def test_duplicate_expected_and_invalid_historical_state_fail():
    m = api()
    row = dict(source_domain="AT1", class_id=0, reference_structure_id="R0", sample_id=0,
               matched="True", matched_sample_segment_id="S0", failure_reason="")
    key = ("AT1", 0, "R0", 0)
    with pytest.raises(ValueError, match="duplicate"):
        m.index_baseline([row], [key, key])
    with pytest.raises(ValueError, match="historical state"):
        m.index_baseline([dict(row, matched="False")], [key])


def test_06b_packet_must_belong_to_the_task_and_all_rows_match():
    runner = importlib.import_module("scripts.diagnose_structure_identity_phase")
    manifest = dict(task="AT1_DK1", source_domain="AT1", experiment=dict(name="structure_observation_support_06B"),
                    model=dict(mode=13, frozen=True, pc1_refit=False), target_used=False, training=False)
    row = dict(task="AT1_DK1", source_domain="AT1")
    runner.validate_06b_packet("AT1_DK1", "AT1", manifest, [row])
    with pytest.raises(ValueError, match="06B manifest"):
        runner.validate_06b_packet("AT1_DK1", "AT1", dict(manifest, task="DK1_FR1"), [row])
    with pytest.raises(ValueError, match="06B row"):
        runner.validate_06b_packet("AT1_DK1", "AT1", manifest, [dict(row, task="DK1_FR1")])


def test_bootstrap_gate_is_fixed_to_point_eight():
    runner = importlib.import_module("scripts.diagnose_structure_identity_phase")
    args = runner.build_parser().parse_args(["--stage", "calibrate", "--min-bootstrap-occurrence", ".7"])
    with pytest.raises(ValueError, match="0.8"):
        runner.validate_args(args)


def test_transaction_requires_complete_files_and_can_publish(tmp_path):
    m = api()
    final = tmp_path / "FR2_AT1"
    final.mkdir()
    (final / "old").write_text("old")
    with pytest.raises(RuntimeError, match="incomplete"):
        with m.staged_output(final, ("manifest.json",)):
            pass
    assert (final / "old").exists()
    with m.staged_output(final, ("manifest.json",)) as staging:
        (staging / "manifest.json").write_text("{}")
    assert not (final / "old").exists()
    assert (final / "manifest.json").exists()


def test_whole_revision_publish_requires_four_tasks_and_replaces_without_stale_files(tmp_path):
    runner = importlib.import_module("scripts.diagnose_structure_identity_phase")
    revision, final = tmp_path / "revision", tmp_path / "06C"
    revision.mkdir()
    final.mkdir()
    (final / "stale.csv").write_text("old")
    runner.write_json(revision / "calibration.json", {"source_domains": list(runner.SOURCES)})
    for task in runner.TASKS:
        folder = revision / task
        folder.mkdir()
        for name in runner.REQUIRED_OUTPUTS:
            if name == "manifest.json":
                runner.write_json(folder / name, {"task": task, "completed": True, "plots": []})
            else:
                (folder / name).write_text("header\n")
    missing = revision / "FR2_AT1" / "gplus_consistency_summary.csv"
    missing.unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.publish_revision(revision, final)
    assert (final / "stale.csv").exists()
    missing.write_text("header\n")
    runner.publish_revision(revision, final)
    assert not (final / "stale.csv").exists()
    assert (final / "AT1_DK1" / "manifest.json").exists()


def test_gplus_misassignment_selection_is_bounded_and_prioritizes_wrong_likely_with_class_coverage():
    m = api()
    rows = []
    for index in range(30):
        rows.append(dict(baseline_06B_status="G+", gplus_exact_segment_recovered=False,
                         identity_status="LIKELY_SAME_STRUCTURE" if index in (5, 7) else "AMBIGUOUS",
                         cost_margin=float(index), class_id=index % 3,
                         reference_structure_id=f"R{index}", sample_id=index))
    selected = m.select_gplus_misassignments(rows, 20)
    assert len(selected) == 20
    assert {row["class_id"] for row in selected} == {0, 1, 2}
    assert {5, 7}.issubset({row["sample_id"] for row in selected})


def test_calibration_refuses_missing_source_or_stale_packet(tmp_path):
    m = api()
    records = [dict(source_domain=d, direction="RISE", differences=[0, 0, 0, 0]) for d in m.SOURCES[:-1]]
    with pytest.raises(ValueError, match="four sources"):
        m.calibrate(records, min_pairs=1)


def test_offline_source_only_import_and_call_boundary():
    import ast
    root = Path(__file__).parents[1]
    for filename in ("analysis/structure_identity_phase_diagnostic.py", "scripts/diagnose_structure_identity_phase.py"):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                assert not any(s in name for s in ("requests", "urllib", "fdasrsf", "subprocess") for name in modules)
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                assert name not in ("backward", "train_timematch", "train_supervised", "estimate_nonlinear_phase")
    launcher = (root / "scripts/run_structure_identity_phase_4tasks.sh").read_text(encoding="utf-8")
    assert not any(token in launcher for token in ("git ", "curl ", "wget ", "pip install", "conda activate", "nohup"))
    for task in ("AT1_DK1", "DK1_FR1", "FR1_FR2", "FR2_AT1"):
        assert f"{task}" in launcher


def test_prepare_with_real_fourier_and_detector_replays_baseline(tmp_path, monkeypatch):
    """Synthetic feature provider; real PCA, Mode13, detector, joining and runner."""
    import inspect
    import torch
    from dataclasses import replace
    from analysis.recon_structure_segments import detect_structure_segments, build_coarse_structure
    from scripts import visualize_shift_configs_4tasks as view
    from scripts import diagnose_structure_reference_validity as validity
    from scripts import diagnose_structure_observation_support as support
    from scripts import diagnose_structure_identity_phase as runner
    from models.fourier_reconstruction import BatchedDirectFourierAnalyzer, BatchedDirectFourierSynthesizer
    m = api()
    device = torch.device("cpu")
    days = torch.linspace(0, 364, 40)
    base = torch.sin(2 * torch.pi * (days - 20) / 365)
    features = torch.stack([torch.stack([base * (1 + i*.01), .4 * base], -1) for i in range(4)])
    batch = dict(pixels=features[..., None].expand(-1, -1, -1, 3), valid_pixels=torch.ones(4, 40, 3),
                 positions=days.expand(4, -1), label=torch.zeros(4, dtype=torch.long), parcel_index=torch.arange(4))

    class Spatial(torch.nn.Module):
        def forward(self, pixels, mask, extra):
            return pixels.mean(-1)

    encoder = Spatial().eval()
    monkeypatch.setattr(view, "_loader", lambda dataset, size: [batch])
    monkeypatch.setattr(view, "load_classes", lambda *a: ["corn"])
    monkeypatch.setattr(view, "load_raw_spatial_encoder", lambda *a: encoder)
    monkeypatch.setattr(validity, "build_source_split", lambda *a: list(range(4)))
    fine_names = ("min_distance_days", "member_min_width_days", "member_min_relative_prominence", "member_min_domain_prominence",
                  "min_duration_days", "max_duration_days", "min_domain_change", "min_curve_change")
    coarse_names = ("max_reversal_ratio", "max_reversal_domain_change", "max_reversal_duration_days", "max_merge_depth",
                    "min_duration_days", "max_duration_days", "min_curve_change", "min_domain_change", "min_monotonicity")
    fine = {name: inspect.signature(detect_structure_segments).parameters[name].default for name in fine_names}
    coarse = {name: inspect.signature(build_coarse_structure).parameters[name].default for name in coarse_names}
    fine["max_duration_days"] = 240
    grid = np.linspace(0, 365, 128)
    projection, _ = view.fit_class_projections(encoder, [], 1, grid, 128, device, False)
    analyzer, synthesizer = BatchedDirectFourierAnalyzer(13, period_days=365., reg=.001), BatchedDirectFourierSynthesizer(13, period_days=365.)
    _, coeffs, records, baselines = view.project_dataset(encoder, [], projection, grid, 128, device, False, analyzer, collect_samplewise=True)
    prototype = view.reconstruct_class_prototype(coeffs[0], synthesizer, device, sample_batch_size=128)
    proto = projection[0].transform(prototype[None])[0]
    _, refs = support._detect(proto, baselines[0], fine, coarse, "S")
    assert refs and all(r.accepted for r in refs)
    root05 = tmp_path / "view" / "05_reconshift13_structure_segments" / "AT1_DK1"
    m.write_json(root05 / "manifest.json", {"structure_capture": {"segment": fine, "coarse": coarse}})
    rows05 = [view._coarse_segment_row("AT1_DK1", 0, "corn", replace(r, source_occurrence_rate=1.)) for r in refs]
    m.write_csv(root05 / "source_coarse_segments.csv", rows05)
    m.write_csv(root05 / "class_summary.csv", [dict(class_id=0, class_name="corn")])
    rows06a = [dict(source_domain="AT1", class_id=0, reference_structure_id=r.coarse_segment_id,
                   bootstrap_occurrence_rate=1., individual_occurrence_rate=.123) for r in refs]
    m.write_csv(tmp_path / "validity" / "AT1" / "structure_stability.csv", rows06a)
    curves = view.reconstruct_projected_coefficients(coeffs[0], projection[0], np.arange(365.))
    history = []
    for index, curve in enumerate(curves):
        chain, segments = support._detect(curve, baselines[0], fine, coarse, f"I{index}_")
        matched = support.audit_sample_matches(refs, segments, 40, 2.5, "HAS_COARSE_SEGMENTS")
        for ref in refs:
            item = matched[ref.coarse_segment_id]
            history.append(dict(task="AT1_DK1", source_domain="AT1", class_id=0, reference_structure_id=ref.coarse_segment_id,
                sample_id=index, matched=item.matched, matched_sample_segment_id=item.matched_sample_segment_id, failure_reason=item.failure_reason))
    m.write_json(tmp_path / "obs" / "AT1_DK1" / "manifest.json", dict(
        experiment=dict(name="structure_observation_support_06B"), task="AT1_DK1", source_domain="AT1",
        source_sample_count=4, model=dict(mode=13, frozen=True, pc1_refit=False), target_used=False, training=False))
    m.write_csv(tmp_path / "obs" / "AT1_DK1" / "sample_structure_support.csv", history)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "fold_0").mkdir(parents=True)
    (checkpoint / "fold_0" / "model.pt").write_bytes(b"synthetic-provider-only")
    m.write_json(checkpoint / "train_config.json", dict(with_extra=False))
    args = runner.build_parser().parse_args(["--stage", "prepare", "--task", "AT1_DK1", "--device", "cpu", "--data-root", str(tmp_path),
        "--source-checkpoint", str(checkpoint), "--structure-view-root", str(tmp_path / "view"), "--validity-root", str(tmp_path / "validity"),
        "--observation-root", str(tmp_path / "obs"), "--work-root", str(tmp_path / "work")])
    runner.run_prepare(args)
    payload = runner.read_json(args.work_root / "AT1_DK1" / "class_0.json")
    assert len(payload["samples"]) == 4
    assert payload["references"][0]["baseline_individual_occurrence"] == 1.  # 05, not the injected 06A rate
    assert len(runner.collect_gplus("AT1", payload)) == len(history)
