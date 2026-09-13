"""Side-effect-free circular Mode13 event detection and diagnostic matching."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

from analysis.recon_anchor_diagnostic import DomainProjectionBaseline, EPS


@dataclass(frozen=True)
class StructuralEvent:
    event_id: str
    kind: str
    day: float
    value: float
    prominence: float
    relative_prominence: float
    domain_relative_prominence: float
    domain_relative_elevation: float
    width_days: float
    left_base_day: float
    right_base_day: float
    accepted: bool
    rejection_reason: str
    boundary_crossing: bool
    source_occurrence_rate: float = float("nan")
    source_timing_mad_days: float = float("nan")


@dataclass(frozen=True)
class EventMatch:
    source_event_id: str
    source_kind: str
    source_day: float
    target_event_id: str
    target_kind: str
    target_day: float
    linear_day_distance: float
    circular_day_distance: float
    width_ratio: float
    relative_prominence_difference: float
    domain_elevation_difference: float
    match_status: str
    boundary_crossing_candidate: bool


def circular_day_residual(day, reference_day, period_days=365.0):
    period = float(period_days)
    return float((float(day) - float(reference_day) + period / 2) % period - period / 2)


def circular_day_distance(left, right, period_days=365.0):
    return abs(circular_day_residual(left, right, period_days))


def circular_median_day(days, reference_day=None, period_days=365.0):
    values = np.asarray(days, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    reference = float(values[0] if reference_day is None else reference_day)
    residuals = np.asarray(
        [circular_day_residual(value, reference, period_days) for value in values]
    )
    return float((reference + np.median(residuals)) % float(period_days))


def circular_mad_days(days, reference_day=None, period_days=365.0):
    values = np.asarray(days, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    reference = float(values[0] if reference_day is None else reference_day)
    residuals = np.asarray(
        [circular_day_residual(value, reference, period_days) for value in values]
    )
    center = np.median(residuals)
    return float(np.median(np.abs(residuals - center)))


def _rejection_reason(width, relative, domain_prominence, elevation, thresholds):
    reasons = []
    if width < thresholds[0]:
        reasons.append("too_narrow")
    if relative < thresholds[1]:
        reasons.append("low_relative_prominence")
    if domain_prominence < thresholds[2]:
        reasons.append("low_domain_prominence")
    if elevation < thresholds[3]:
        reasons.append("low_domain_elevation")
    return ";".join(reasons)


def detect_circular_events(
    curve,
    days,
    baseline,
    min_distance_days=15,
    min_width_days=5,
    min_relative_prominence=0.15,
    min_domain_prominence=0.20,
    min_domain_elevation=0.50,
    period_days=365.0,
    event_prefix="E",
):
    """Detect peak/valley candidates on a three-period calendar extension.

    The extension is detection-only. Only events whose extrema lie in the center
    period are returned, so no feature or timestamp is duplicated downstream.
    """
    values = np.asarray(curve, dtype=np.float64)
    grid = np.asarray(days, dtype=np.float64)
    if values.ndim != 1 or grid.shape != values.shape or len(values) < 5:
        raise ValueError("circular event detection requires equal-length 1D inputs")
    if not np.isfinite(values).all() or not np.all(np.diff(grid) > 0):
        raise ValueError("event curve must be finite on an increasing grid")
    step = float(np.median(np.diff(grid)))
    distance_samples = max(1, int(np.ceil(float(min_distance_days) / step)))
    extended = np.tile(values, 3)
    extended_days = np.concatenate(
        (grid - float(period_days), grid, grid + float(period_days))
    )
    curve_scale = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    domain_scale = float(baseline.iqr)
    candidates = []
    count = len(values)
    for kind, signal in (("peak", extended), ("valley", -extended)):
        indices, _ = find_peaks(signal, distance=distance_samples)
        if not len(indices):
            continue
        prominences, left_bases, right_bases = peak_prominences(signal, indices)
        widths, _, left_ips, right_ips = peak_widths(
            signal, indices, rel_height=0.5, prominence_data=(prominences, left_bases, right_bases)
        )
        for index, prominence, width, left_ip, right_ip, left_base, right_base in zip(
            indices, prominences, widths, left_ips, right_ips, left_bases, right_bases
        ):
            if index < count or index >= 2 * count:
                continue
            center_index = int(index - count)
            day = float(grid[center_index] % float(period_days))
            value = float(values[center_index])
            relative = float(prominence / (curve_scale + EPS))
            domain_prominence = float(prominence / (domain_scale + EPS))
            elevation = float(
                ((value - baseline.median) if kind == "peak" else (baseline.median - value))
                / (domain_scale + EPS)
            )
            width_days = float(width * step)
            left_day = float(extended_days[int(left_base)])
            right_day = float(extended_days[int(right_base)])
            left_width_day = float(
                np.interp(left_ip, np.arange(len(extended_days)), extended_days)
            )
            right_width_day = float(
                np.interp(right_ip, np.arange(len(extended_days)), extended_days)
            )
            reason = _rejection_reason(
                width_days,
                relative,
                domain_prominence,
                elevation,
                (
                    float(min_width_days),
                    float(min_relative_prominence),
                    float(min_domain_prominence),
                    float(min_domain_elevation),
                ),
            )
            candidates.append(
                StructuralEvent(
                    "",
                    kind,
                    day,
                    value,
                    float(prominence),
                    relative,
                    domain_prominence,
                    elevation,
                    width_days,
                    left_day,
                    right_day,
                    not reason,
                    reason,
                    bool(
                        left_width_day < 0.0
                        or right_width_day >= float(period_days)
                    ),
                )
            )
    candidates.sort(key=lambda item: (item.day, item.kind))
    return tuple(
        replace(event, event_id=f"{event_prefix}{index}")
        for index, event in enumerate(candidates)
    )


def evaluate_source_event_stability(
    event,
    sample_curves,
    days,
    baseline,
    occurrence_radius_days=20,
    min_occurrence=0.60,
    max_timing_mad_days=20,
    **detector_kwargs,
):
    matches = []
    for sample in np.asarray(sample_curves, dtype=np.float64):
        candidates = detect_circular_events(
            sample, days, baseline, event_prefix="I", **detector_kwargs
        )
        eligible = [
            item
            for item in candidates
            if item.accepted
            and item.kind == event.kind
            and circular_day_distance(item.day, event.day)
            <= float(occurrence_radius_days)
        ]
        if eligible:
            matches.append(
                min(eligible, key=lambda item: (circular_day_distance(item.day, event.day), item.day))
            )
    occurrence = len(matches) / len(sample_curves) if len(sample_curves) else 0.0
    timing_mad = circular_mad_days(
        [item.day for item in matches], reference_day=event.day
    )
    reasons = [value for value in event.rejection_reason.split(";") if value]
    if occurrence < float(min_occurrence):
        reasons.append("source_occurrence_low")
    if not np.isfinite(timing_mad) or timing_mad > float(max_timing_mad_days):
        reasons.append("source_timing_unstable")
    return replace(
        event,
        accepted=not reasons,
        rejection_reason=";".join(reasons),
        source_occurrence_rate=float(occurrence),
        source_timing_mad_days=float(timing_mad),
    )


def greedy_match_events(
    source_events: Sequence[StructuralEvent],
    target_events: Sequence[StructuralEvent],
    match_radius_days=25,
) -> Tuple[EventMatch, ...]:
    source = [item for item in source_events if item.accepted]
    target = [item for item in target_events if item.accepted]
    edges = []
    for source_index, left in enumerate(source):
        for target_index, right in enumerate(target):
            distance = circular_day_distance(left.day, right.day)
            if left.kind == right.kind and distance <= float(match_radius_days):
                edges.append((distance, left.event_id, right.event_id, source_index, target_index))
    used_source, used_target, results = set(), set(), []
    for distance, _, _, source_index, target_index in sorted(edges):
        if source_index in used_source or target_index in used_target:
            continue
        used_source.add(source_index)
        used_target.add(target_index)
        left, right = source[source_index], target[target_index]
        linear = abs(left.day - right.day)
        boundary = bool(linear > float(match_radius_days) and distance <= float(match_radius_days))
        results.append(
            EventMatch(
                left.event_id,
                left.kind,
                left.day,
                right.event_id,
                right.kind,
                right.day,
                float(linear),
                float(distance),
                float(right.width_days / max(left.width_days, EPS)),
                float(abs(left.relative_prominence - right.relative_prominence)),
                float(abs(left.domain_relative_elevation - right.domain_relative_elevation)),
                "MATCHED_CIRCULAR_CANDIDATE" if boundary else "MATCHED",
                boundary,
            )
        )
    nan = float("nan")
    for index, event in enumerate(source):
        if index not in used_source:
            results.append(
                EventMatch(event.event_id, event.kind, event.day, "", "", nan, nan, nan, nan, nan, nan, "UNMATCHED_SOURCE", False)
            )
    for index, event in enumerate(target):
        if index not in used_target:
            results.append(
                EventMatch("", "", nan, event.event_id, event.kind, event.day, nan, nan, nan, nan, nan, "UNMATCHED_TARGET", False)
            )
    return tuple(results)
