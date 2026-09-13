"""Side-effect-free structural segment discovery on circular Recon13 curves."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Tuple

import numpy as np

from analysis.recon_anchor_diagnostic import DomainProjectionBaseline, EPS
from analysis.recon_event_diagnostic import (
    StructuralEvent,
    circular_day_distance,
    circular_mad_days,
    detect_circular_events,
)


@dataclass(frozen=True)
class StructuralSegment:
    segment_id: str
    pattern: str
    events: Tuple[StructuralEvent, StructuralEvent, StructuralEvent]
    unwrapped_days: Tuple[float, float, float]
    span_days: float
    total_variation: float
    curve_normalized_variation: float
    domain_normalized_variation: float
    amplitude_range: float
    accepted: bool
    rejection_reason: str
    crosses_year_boundary: bool
    source_occurrence_rate: float = float("nan")
    source_center_timing_mad_days: float = float("nan")
    source_span_ratio_median: float = float("nan")

    @property
    def center_day(self) -> float:
        return float(self.unwrapped_days[1] % 365.0)


def _preferred_same_type(left: StructuralEvent, right: StructuralEvent) -> StructuralEvent:
    if left.kind != right.kind:
        raise ValueError("same-type compression requires matching event kinds")
    if left.kind == "peak":
        key = lambda item: (item.value, item.prominence, -item.day)
    else:
        key = lambda item: (-item.value, item.prominence, -item.day)
    return max((left, right), key=key)


def build_alternating_chain(
    events: Sequence[StructuralEvent], period_days: float = 365.0
) -> Tuple[StructuralEvent, ...]:
    """Cut at the largest circular gap, unwrap, and compress adjacent equal kinds."""
    eligible = sorted((item for item in events if item.kind in {"peak", "valley"}), key=lambda x: x.day)
    if not eligible:
        return ()
    days = np.asarray([item.day % period_days for item in eligible], dtype=np.float64)
    gaps = np.diff(np.r_[days, days[0] + period_days])
    start = (int(np.argmax(gaps)) + 1) % len(eligible)
    rotated = eligible[start:] + eligible[:start]
    unwrapped = []
    previous = None
    for item in rotated:
        day = float(item.day % period_days)
        if previous is not None:
            while day <= previous:
                day += period_days
        unwrapped.append(replace(item, day=day))
        previous = day
    chain = []
    for item in unwrapped:
        if chain and chain[-1].kind == item.kind:
            chain[-1] = _preferred_same_type(chain[-1], item)
        else:
            chain.append(item)
    return tuple(chain)


def build_segments(
    chain: Sequence[StructuralEvent],
    curve,
    domain_scale: float,
    min_span_days: float,
    max_span_days: float,
    min_domain_variation: float,
    min_curve_variation: float,
    segment_prefix: str = "SEG",
) -> Tuple[StructuralSegment, ...]:
    values = np.asarray(curve, dtype=np.float64)
    curve_scale = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    results = []
    for index in range(max(0, len(chain) - 2)):
        events = tuple(chain[index : index + 3])
        kinds = tuple(item.kind for item in events)
        if kinds == ("valley", "peak", "valley"):
            pattern = "valley_peak_valley"
        elif kinds == ("peak", "valley", "peak"):
            pattern = "peak_valley_peak"
        else:
            continue
        days = tuple(float(item.day) for item in events)
        event_values = np.asarray([item.value for item in events], dtype=np.float64)
        span = float(days[2] - days[0])
        total = float(abs(event_values[1] - event_values[0]) + abs(event_values[2] - event_values[1]))
        domain_variation = float(total / (float(domain_scale) + EPS))
        curve_variation = float(total / (curve_scale + EPS))
        reasons = []
        if span < float(min_span_days):
            reasons.append("span_too_short")
        if span > float(max_span_days):
            reasons.append("span_too_long")
        if domain_variation < float(min_domain_variation):
            reasons.append("low_domain_variation")
        if curve_variation < float(min_curve_variation):
            reasons.append("low_curve_variation")
        results.append(
            StructuralSegment(
                f"{segment_prefix}{len(results)}",
                pattern,
                events,
                days,
                span,
                total,
                curve_variation,
                domain_variation,
                float(np.ptp(event_values)),
                not reasons,
                ";".join(reasons),
                bool(days[0] < 0 or days[2] >= 365.0),
            )
        )
    return tuple(results)


def detect_structure_segments(
    curve,
    days,
    baseline: DomainProjectionBaseline,
    min_distance_days: float = 15,
    member_min_width_days: float = 3,
    member_min_relative_prominence: float = 0.05,
    member_min_domain_prominence: float = 0.05,
    min_span_days: float = 15,
    max_span_days: float = 100,
    min_domain_variation: float = 0.50,
    min_curve_variation: float = 0.25,
    event_prefix: str = "E",
    segment_prefix: str = "SEG",
):
    candidates = detect_circular_events(
        curve,
        days,
        baseline,
        min_distance_days=min_distance_days,
        min_width_days=member_min_width_days,
        min_relative_prominence=member_min_relative_prominence,
        min_domain_prominence=member_min_domain_prominence,
        min_domain_elevation=-float("inf"),
        event_prefix=event_prefix,
    )
    members = tuple(item for item in candidates if item.accepted)
    chain = build_alternating_chain(members)
    segments = build_segments(
        chain,
        curve,
        baseline.iqr,
        min_span_days,
        max_span_days,
        min_domain_variation,
        min_curve_variation,
        segment_prefix,
    )
    return candidates, chain, segments


def evaluate_source_segment_stability(
    segment: StructuralSegment,
    sample_segments: Sequence[StructuralSegment],
    occurrence_radius_days: float = 30,
    min_occurrence: float = 0.50,
    max_timing_mad_days: float = 25,
    max_width_ratio: float = 2.0,
) -> StructuralSegment:
    eligible = []
    for sample in sample_segments:
        candidates = (sample,) if isinstance(sample, StructuralSegment) else tuple(sample)
        matches = []
        for candidate in candidates:
            ratio = candidate.span_days / max(segment.span_days, EPS)
            distance = circular_day_distance(candidate.center_day, segment.center_day)
            if (
                candidate.accepted
                and candidate.pattern == segment.pattern
                and distance <= float(occurrence_radius_days)
                and 1.0 / float(max_width_ratio) <= ratio <= float(max_width_ratio)
            ):
                matches.append((distance, candidate.center_day, ratio, candidate))
        if matches:
            eligible.append(min(matches, key=lambda item: (item[0], item[1])))
    total = len(sample_segments)
    occurrence = len(eligible) / total if total else 0.0
    centers = [item[1] for item in eligible]
    timing_mad = circular_mad_days(centers, reference_day=segment.center_day)
    ratio_median = float(np.median([item[2] for item in eligible])) if eligible else float("nan")
    reasons = [value for value in segment.rejection_reason.split(";") if value]
    if occurrence < float(min_occurrence):
        reasons.append("source_occurrence_low")
    if not np.isfinite(timing_mad) or timing_mad > float(max_timing_mad_days):
        reasons.append("source_timing_unstable")
    return replace(
        segment,
        accepted=not reasons,
        rejection_reason=";".join(reasons),
        source_occurrence_rate=float(occurrence),
        source_center_timing_mad_days=float(timing_mad),
        source_span_ratio_median=ratio_median,
    )


def match_sample_segment(
    prototype: StructuralSegment,
    candidates: Sequence[StructuralSegment],
    occurrence_radius_days: float,
    max_width_ratio: float,
):
    eligible = []
    for candidate in candidates:
        ratio = candidate.span_days / max(prototype.span_days, EPS)
        distance = circular_day_distance(candidate.center_day, prototype.center_day)
        if (
            candidate.accepted
            and candidate.pattern == prototype.pattern
            and distance <= float(occurrence_radius_days)
            and 1.0 / float(max_width_ratio) <= ratio <= float(max_width_ratio)
        ):
            eligible.append((distance, abs(1.0 - ratio), candidate.center_day, candidate))
    return min(eligible, default=(None, None, None, None))[-1]
