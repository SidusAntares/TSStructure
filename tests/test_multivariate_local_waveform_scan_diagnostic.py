"""07A source-only local waveform identity diagnostic contracts."""
import importlib
from pathlib import Path

import numpy as np
import pytest


def api():
    return importlib.import_module("analysis.multivariate_local_waveform_scan_diagnostic")


def segment(sid, direction="RISE", start=10., end=40., accepted=True):
    return dict(coarse_segment_id=sid, direction=direction, start_day=start,
                end_day=end, accepted=accepted)


def test_relative_queries_include_endpoints_and_unwrap_cross_year():
    m = api()
    np.testing.assert_allclose(m.relative_queries(segment("S", start=10, end=40), 4), [10, 20, 30, 40])
    np.testing.assert_allclose(m.relative_queries(segment("S", start=350, end=20), 4), [350, 350 + 35/3, 350 + 70/3, 385])
    assert len(m.relative_queries(segment("S", start=10, end=40), 32)) == 32


def test_endpoint_relative_ned_is_offset_invariant_but_not_amplitude_normalized():
    m = api()
    q = np.arange(12, dtype=float).reshape(4, 3)
    assert m.endpoint_relative_ned(q, q) == pytest.approx(0)
    assert m.endpoint_relative_ned(q, q + 99) == pytest.approx(0)
    assert m.endpoint_relative_ned(q, 3 * q) > .1


def test_cosine_handles_degenerate_waveform_without_nan():
    m = api()
    score, q_energy, w_energy, degenerate = m.endpoint_relative_cosine(np.ones((4, 2)), np.ones((4, 2)))
    assert np.isfinite(score) and score == pytest.approx(0)
    assert q_energy == 0 and w_energy == 0 and degenerate


def test_candidate_pool_is_accepted_same_direction_and_requires_baseline():
    m = api()
    candidates = [segment("A"), segment("B", accepted=False), segment("C", "FALL")]
    assert [row["coarse_segment_id"] for row in m.candidate_pool(candidates, "RISE", "A")] == ["A"]
    with pytest.raises(ValueError, match="baseline"):
        m.candidate_pool(candidates, "RISE", "B")


def test_ranking_top1_rank_margin_pairwise_and_tie_confidence():
    m = api()
    result = m.rank_scores({"S2": .4, "S0": .2, "S1": .3}, "S1")
    assert result["best_id"] == "S0" and result["positive_rank"] == 2
    assert not result["exact"] and result["top2"]
    assert result["mrr"] == pytest.approx(.5)
    assert result["best_negative_distance"] == pytest.approx(.2)
    assert result["margin"] == pytest.approx(-1/3)
    assert result["pairwise_wins"] == 1 and result["pairwise_total"] == 2
    tied = m.rank_scores({"S1": .2, "S0": .2}, "S1")
    assert tied["best_id"] == "S0" and not tied["unique_best"]


def test_compare_same_pool_keeps_handcrafted_pc1_and_multivariate_separate():
    m = api()
    reference = np.array([[0., 0.], [1., 0.], [0., 1.]])
    candidates = {"good": reference.copy(), "bad": reference * 3}
    pc1 = np.array([1., 0.])
    rows = m.compare_candidate_waveforms(reference, candidates, pc1, "good", {"good": .8, "bad": .1})
    assert rows["handcrafted"]["best_id"] == "bad"
    assert rows["pc1_ned"]["best_id"] == "good"
    assert rows["multi_ned"]["best_id"] == "good"
    assert not rows["multi_cosine"]["unique_best"]  # cosine intentionally discards magnitude


def test_periodic_sampling_matches_normal_and_cross_year_interpolation():
    m = api()
    curve = np.arange(365., dtype=float)[:, None]
    normal = m.sample_periodic_curve(curve, np.array([0., 10., 20.]))[:, 0]
    cross = m.sample_periodic_curve(curve, np.array([350., 365., 370.]))[:, 0]
    np.testing.assert_allclose(normal, [0, 10, 20])
    np.testing.assert_allclose(cross, [350, 0, 5])


def test_fourier_evaluation_is_multivariate_and_independent_of_pc1():
    m = api()
    coeff = np.zeros((13, 2), complex)
    coeff[6] = [2, 3]
    values = m.evaluate_mode13(coeff, np.array([0., 100.]))
    np.testing.assert_allclose(values, [[2, 3], [2, 3]])


def test_template_pc1_consistency_uses_fixed_axis():
    m = api()
    multi = np.array([[1., 2.], [3., 4.]])
    axis = np.array([1., 0.])
    assert m.template_pc1_consistency_error(multi, axis, np.array([1., 3.])) == pytest.approx(0)


def test_metric_summary_separates_multi_candidate_headline():
    m = api()
    base = dict(task="AT1_DK1", class_id=0, reference_structure_id="R", sample_id=1)
    rows = [dict(base, multi_candidate=False, handcrafted_local_exact=True, handcrafted_positive_rank=1,
                 handcrafted_pairwise_wins=0, handcrafted_pairwise_total=0,
                 pc1_ned_exact=True, pc1_ned_positive_rank=1, pc1_ned_pairwise_wins=0, pc1_ned_pairwise_total=0,
                 multi_ned_exact=True, multi_ned_positive_rank=1, multi_ned_pairwise_wins=0, multi_ned_pairwise_total=0,
                 multi_ned_unique_best=True, multi_ned_margin=np.nan, multi_cosine_exact=True),
            dict(base, sample_id=2, multi_candidate=True, handcrafted_local_exact=False, handcrafted_positive_rank=2,
                 handcrafted_pairwise_wins=0, handcrafted_pairwise_total=1,
                 pc1_ned_exact=False, pc1_ned_positive_rank=2, pc1_ned_pairwise_wins=0, pc1_ned_pairwise_total=1,
                 multi_ned_exact=True, multi_ned_positive_rank=1, multi_ned_pairwise_wins=1, multi_ned_pairwise_total=1,
                 multi_ned_unique_best=True, multi_ned_margin=.5, multi_cosine_exact=True)]
    summary = m.metric_summary(rows, "AT1_DK1")
    multi = next(row for row in summary if row["scope"] == "MULTI_CANDIDATE_ONLY")
    assert multi["handcrafted_top1"] == 0 and multi["multi_top1"] == 1
    assert multi["multi_minus_handcrafted"] == 1


def test_structure_summary_includes_reference_and_task_total_rows():
    m = api()
    row = dict(
        task="AT1_DK1", class_id=0, class_name="corn",
        reference_structure_id="R0", multi_candidate=True,
        handcrafted_local_exact=True, handcrafted_positive_rank=1,
        handcrafted_pairwise_wins=1, handcrafted_pairwise_total=1,
        pc1_ned_exact=True, pc1_ned_positive_rank=1,
        pc1_ned_pairwise_wins=1, pc1_ned_pairwise_total=1,
        multi_ned_exact=True, multi_ned_positive_rank=1,
        multi_ned_pairwise_wins=1, multi_ned_pairwise_total=1,
        multi_ned_unique_best=True, multi_ned_margin=.25,
    )
    summary = m.structure_summary([row])
    keys = {(item["scope"], item["reference_structure_id"]) for item in summary}
    assert ("ALL_GPLUS", "R0") in keys
    assert ("MULTI_CANDIDATE_ONLY", "R0") in keys
    assert ("ALL_GPLUS", "TOTAL") in keys
    assert ("MULTI_CANDIDATE_ONLY", "TOTAL") in keys


def test_transitions_are_posthoc_only():
    m = api()
    rows = [dict(task="T", **{"06c_final_exact": False, "handcrafted_local_exact": False, "multi_ned_exact": True}),
            dict(task="T", **{"06c_final_exact": True, "handcrafted_local_exact": True, "multi_ned_exact": False})]
    transitions = m.transition_summary(rows, "T")
    assert {row["transition"] for row in transitions} == {
        "06C_WRONG_TO_MULTI_CORRECT", "06C_CORRECT_TO_MULTI_WRONG",
        "HANDCRAFTED_WRONG_TO_MULTI_CORRECT", "HANDCRAFTED_CORRECT_TO_MULTI_WRONG"}


def test_cluster_bootstrap_resamples_sample_clusters_and_is_reproducible():
    m = api()
    rows = [dict(sample_id=1, multi_candidate=True, multi_ned_exact=True, handcrafted_local_exact=False, pc1_ned_exact=False),
            dict(sample_id=1, multi_candidate=True, multi_ned_exact=True, handcrafted_local_exact=False, pc1_ned_exact=True),
            dict(sample_id=2, multi_candidate=True, multi_ned_exact=False, handcrafted_local_exact=True, pc1_ned_exact=False)]
    a = m.cluster_bootstrap(rows, repeats=100, seed=7)
    b = m.cluster_bootstrap(rows, repeats=100, seed=7)
    assert a == b and a["num_clusters"] == 2


def test_reference_and_gplus_filters_are_strict():
    m = api()
    refs = [dict(reference_structure_id="R0", bootstrap_occurrence_rate=.8),
            dict(reference_structure_id="R1", bootstrap_occurrence_rate=.799)]
    assert [r["reference_structure_id"] for r in m.reliable_references(refs)] == ["R0"]
    rows = [dict(matched="True"), dict(matched="False")]
    assert len(m.gplus_rows(rows)) == 1


def test_atomic_revision_publish_requires_all_tasks_and_preserves_old(tmp_path):
    m = api()
    staging, final = tmp_path / "new", tmp_path / "07A"
    staging.mkdir(); final.mkdir(); (final / "old.csv").write_text("old")
    for name in m.REQUIRED_ROOT_FILES:
        (staging / name).write_text("{}" if name == "manifest.json" else "header\n")
    for task in m.TASKS:
        folder = staging / task; folder.mkdir()
        for name in m.REQUIRED_TASK_FILES:
            (folder / name).write_text("{}" if name == "manifest.json" else "header\n")
    (staging / "AT1_DK1" / "manifest.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        m.publish_revision(staging, final)
    assert (final / "old.csv").exists()
    (staging / "AT1_DK1" / "manifest.json").write_text("{}")
    m.publish_revision(staging, final)
    assert not (final / "old.csv").exists()


def test_source_only_static_boundary_and_launcher_contract():
    root = Path(__file__).parents[1]
    runner = (root / "scripts/diagnose_multivariate_local_waveform_scan.py").read_text(encoding="utf-8")
    launcher = (root / "scripts/run_multivariate_local_waveform_scan_4tasks.sh").read_text(encoding="utf-8")
    assert not any(token in runner for token in ("target pseudo", "backward(", "optimizer", "train_timematch"))
    assert "--local-waveform-points" in runner
    assert "32" in runner
    assert "07A_multivariate_local_waveform_scan" in launcher
    assert not any(token in launcher for token in ("git ", "curl ", "wget ", "pip install", "conda activate", "nohup"))
