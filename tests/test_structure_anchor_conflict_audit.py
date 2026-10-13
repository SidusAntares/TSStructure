import numpy as np
import pytest


def test_anchor_conflict_distinguishes_weak_opposition_from_strong_disagreement():
    from analysis.structure_anchor_conflict_audit import anchor_conflict_statistics

    weak = anchor_conflict_statistics(np.array([1., .8, -.01]), weak_threshold=.1)
    strong = anchor_conflict_statistics(np.array([1., .8, -.9]), weak_threshold=.1)
    assert weak["strong_conflict"] is False
    assert strong["strong_conflict"] is True
    assert strong["negative_strength"] > weak["negative_strength"]
    assert strong["cancellation_ratio"] > weak["cancellation_ratio"]


def test_source_reference_and_abstention_never_use_target_labels():
    from analysis.structure_anchor_conflict_audit import (
        fit_source_conflict_reference, aggregate_anchor_evidence,
    )

    source = np.array([[1., .8, .7], [.9, .6, .5], [-1., -.8, -.6]])
    reference = fit_source_conflict_reference(source)
    assert set(reference) >= {"weak_threshold", "conflict_threshold", "dispersion_threshold"}
    packet = aggregate_anchor_evidence(
        np.array([1., -.95, .8]), np.ones(3), method="abstain_conflict",
        reference=reference,
    )
    assert packet["available"] is False
    assert packet["status"] == "abstain_conflict"


def test_weighted_median_is_robust_and_coverage_counts_only_available_rows():
    from analysis.structure_anchor_conflict_audit import (
        aggregate_anchor_evidence, evidence_coverage,
    )

    scores = np.array([1., 1.1, -8.])
    mean = aggregate_anchor_evidence(scores, np.ones(3), method="weighted_mean")
    robust = aggregate_anchor_evidence(scores, np.ones(3), method="weighted_median")
    assert mean["score"] < 0 < robust["score"]
    assert evidence_coverage([mean, robust, {"available": False}]) == pytest.approx(2 / 3)


def test_pair_direction_is_antisymmetric_for_each_anchor():
    from analysis.structure_anchor_conflict_audit import oriented_anchor_scores

    scores = np.array([.2, -.4, 1.])
    np.testing.assert_allclose(
        oriented_anchor_scores(scores, 1, 3, 1, 3),
        -oriented_anchor_scores(scores, 1, 3, 3, 1),
    )


def test_stability_weights_follow_retained_anchor_ids_after_range_filtering():
    from analysis.structure_anchor_conflict_audit import aggregate_anchor_evidence

    reference = {
        "weak_threshold": .1, "conflict_threshold": 1., "dispersion_threshold": 10.,
        "anchor_ids": np.array([2, 5, 9]),
        "anchor_stability": np.array([1., 100., 1.]),
    }
    packet = aggregate_anchor_evidence(
        np.array([1., -1.]), np.ones(2), method="stability_weighted",
        reference=reference, anchor_ids=np.array([2, 9]),
    )
    assert packet["available"] is True
    assert packet["score"] == pytest.approx(0.)

