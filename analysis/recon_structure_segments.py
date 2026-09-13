"""Side-effect-free directed two-extrema segments on circular Recon13 curves."""

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
class DirectedStructureSegment:
    segment_id: str
    direction: str
    start_event: StructuralEvent
    end_event: StructuralEvent
    unwrapped_start_day: float
    unwrapped_end_day: float
    duration_days: float
    signed_change: float
    absolute_change: float
    slope: float
    curve_normalized_change: float
    domain_normalized_change: float
    start_role: str
    end_role: str
    crosses_year_boundary: bool
    accepted: bool
    rejection_reason: str
    left_context_pattern: str
    right_context_pattern: str
    source_occurrence_rate: float = float("nan")
    source_center_timing_mad_days: float = float("nan")
    source_duration_ratio_median: float = float("nan")
    source_duration_ratio_error_p90: float = float("nan")
    source_domain_change_median: float = float("nan")
    source_domain_change_iqr: float = float("nan")

    @property
    def start_day(self) -> float:
        return float(self.unwrapped_start_day % 365.0)

    @property
    def end_day(self) -> float:
        return float(self.unwrapped_end_day % 365.0)

    @property
    def center_day(self) -> float:
        return float((self.unwrapped_start_day + self.duration_days / 2.0) % 365.0)


@dataclass(frozen=True)
class RemovedReversal:
    removal_id: str
    removed_event_ids: Tuple[str, str]
    reversal_ratio: float
    domain_normalized_reversal: float
    reversal_duration_days: float
    circular_start_day: float


@dataclass(frozen=True)
class CoarseStructureSegment:
    coarse_segment_id: str
    direction: str
    start_event_id: str
    end_event_id: str
    start_day: float
    end_day: float
    center_day: float
    unwrapped_end_day: float
    duration_days: float
    start_value: float
    end_value: float
    signed_change: float
    absolute_change: float
    curve_normalized_change: float
    domain_normalized_change: float
    fine_segment_ids: Tuple[str, ...]
    removed_reversal_ids: Tuple[str, ...]
    num_fine_segments_covered: int
    num_removed_reversals: int
    total_path_variation: float
    net_change: float
    monotonicity_ratio: float
    max_removed_reversal_ratio: float
    max_removed_reversal_domain_change: float
    max_removed_reversal_duration_days: float
    crosses_year_boundary: bool
    accepted: bool
    rejection_reason: str
    source_occurrence_rate: float = float("nan")
    source_center_timing_mad_days: float = float("nan")
    source_duration_ratio_median: float = float("nan")


@dataclass(frozen=True)
class CoarseStructureResult:
    segments: Tuple[CoarseStructureSegment, ...]
    removed_reversals: Tuple[RemovedReversal, ...]
    surviving_event_ids: Tuple[str, ...]
    stop_reason: str


def _preferred_same_type(left: StructuralEvent, right: StructuralEvent) -> StructuralEvent:
    if left.kind != right.kind:
        raise ValueError("same-type compression requires matching event kinds")
    if left.kind == "peak":
        key = lambda item: (item.prominence, item.value, -float(item.day % 365.0))
    else:
        key = lambda item: (item.prominence, -item.value, -float(item.day % 365.0))
    return max((left, right), key=key)


def build_alternating_chain(
    events: Sequence[StructuralEvent], period_days: float = 365.0
) -> Tuple[StructuralEvent, ...]:
    """Compress equal-kind circular runs into a deterministic alternating chain."""
    chain = sorted(
        (item for item in events if item.kind in {"peak", "valley"}),
        key=lambda item: (float(item.day % period_days), item.event_id),
    )
    compressed = []
    for item in chain:
        normalized = replace(item, day=float(item.day % period_days))
        if compressed and compressed[-1].kind == normalized.kind:
            compressed[-1] = _preferred_same_type(compressed[-1], normalized)
        else:
            compressed.append(normalized)
    while len(compressed) > 1 and compressed[0].kind == compressed[-1].kind:
        winner = _preferred_same_type(compressed[-1], compressed[0])
        compressed = compressed[1:-1] + [winner]
        compressed.sort(key=lambda item: (item.day, item.event_id))
    return tuple(compressed)


def _event_role(event: StructuralEvent, core_events: Sequence[StructuralEvent]) -> str:
    for core in core_events:
        if (
            core.accepted
            and core.kind == event.kind
            and circular_day_distance(core.day, event.day) < 1e-6
        ):
            return "CORE"
    return "AUX"


def _context_pattern(*events: StructuralEvent) -> str:
    symbols = {"valley": "V", "peak": "P"}
    return "-".join(symbols[item.kind] for item in events)


def build_directed_segments(
    chain: Sequence[StructuralEvent],
    curve,
    domain_scale: float,
    min_duration_days: float = 10,
    max_duration_days: float = 120,
    min_domain_change: float = 0.30,
    min_curve_change: float = 0.15,
    core_events: Sequence[StructuralEvent] = (),
    segment_prefix: str = "SEG",
    period_days: float = 365.0,
) -> Tuple[DirectedStructureSegment, ...]:
    """Build every adjacent directed segment, including the circular closing pair."""
    if len(chain) < 2:
        return ()
    ordered = tuple(sorted(chain, key=lambda item: float(item.day % period_days)))
    values = np.asarray(curve, dtype=np.float64)
    curve_scale = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    results = []
    for index, start in enumerate(ordered):
        end = ordered[(index + 1) % len(ordered)]
        if start.kind == end.kind:
            raise ValueError("directed segments require an alternating event chain")
        start_day = float(start.day % period_days)
        end_day = float(end.day % period_days)
        unwrapped_end = end_day
        if unwrapped_end <= start_day:
            unwrapped_end += float(period_days)
        duration = float(unwrapped_end - start_day)
        signed_change = float(end.value - start.value)
        absolute_change = abs(signed_change)
        direction = "RISE" if start.kind == "valley" else "FALL"
        domain_change = float(absolute_change / (float(domain_scale) + EPS))
        curve_change = float(absolute_change / (curve_scale + EPS))
        reasons = []
        if duration < float(min_duration_days):
            reasons.append("duration_too_short")
        if duration > float(max_duration_days):
            reasons.append("duration_too_long")
        if domain_change < float(min_domain_change):
            reasons.append("low_domain_change")
        if curve_change < float(min_curve_change):
            reasons.append("low_curve_change")
        previous = ordered[(index - 1) % len(ordered)]
        following = ordered[(index + 2) % len(ordered)]
        results.append(
            DirectedStructureSegment(
                segment_id=f"{segment_prefix}{index}",
                direction=direction,
                start_event=start,
                end_event=end,
                unwrapped_start_day=start_day,
                unwrapped_end_day=unwrapped_end,
                duration_days=duration,
                signed_change=signed_change,
                absolute_change=absolute_change,
                slope=float(signed_change / max(duration, EPS)),
                curve_normalized_change=curve_change,
                domain_normalized_change=domain_change,
                start_role=_event_role(start, core_events),
                end_role=_event_role(end, core_events),
                crosses_year_boundary=bool(unwrapped_end >= period_days),
                accepted=not reasons,
                rejection_reason=";".join(reasons),
                left_context_pattern=_context_pattern(previous, start, end),
                right_context_pattern=_context_pattern(start, end, following),
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
    min_duration_days: float = 10,
    max_duration_days: float = 120,
    min_domain_change: float = 0.30,
    min_curve_change: float = 0.15,
    core_events: Sequence[StructuralEvent] = (),
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
    segments = build_directed_segments(
        chain,
        curve,
        baseline.iqr,
        min_duration_days,
        max_duration_days,
        min_domain_change,
        min_curve_change,
        core_events,
        segment_prefix,
    )
    return candidates, chain, segments


def match_directed_segments_one_to_one(
    prototypes: Sequence[DirectedStructureSegment],
    candidates: Sequence[DirectedStructureSegment],
    occurrence_radius_days: float = 30,
    max_duration_ratio: float = 2.0,
):
    """Greedily match same-direction segments without reusing either segment."""
    possible = []
    for prototype in prototypes:
        for candidate in candidates:
            ratio = candidate.duration_days / max(prototype.duration_days, EPS)
            distance = circular_day_distance(candidate.center_day, prototype.center_day)
            if (
                prototype.accepted
                and candidate.accepted
                and prototype.direction == candidate.direction
                and distance <= float(occurrence_radius_days)
                and 1.0 / float(max_duration_ratio) <= ratio <= float(max_duration_ratio)
            ):
                possible.append(
                    (
                        distance,
                        abs(1.0 - ratio),
                        prototype.segment_id,
                        candidate.segment_id,
                        prototype,
                        candidate,
                    )
                )
    used_prototypes, used_candidates, matches = set(), set(), []
    for _, _, prototype_id, candidate_id, prototype, candidate in sorted(possible):
        if prototype_id in used_prototypes or candidate_id in used_candidates:
            continue
        used_prototypes.add(prototype_id)
        used_candidates.add(candidate_id)
        matches.append((prototype, candidate))
    return tuple(matches)


def evaluate_source_segments_stability(
    prototype_segments: Sequence[DirectedStructureSegment],
    sample_segment_sets: Sequence[Sequence[DirectedStructureSegment]],
    occurrence_radius_days: float = 30,
    min_occurrence: float = 0.50,
    max_timing_mad_days: float = 25,
    max_duration_ratio: float = 2.0,
) -> Tuple[DirectedStructureSegment, ...]:
    """Attach source stability using one-to-one matches in every sample."""
    by_prototype = {item.segment_id: [] for item in prototype_segments}
    for sample_segments in sample_segment_sets:
        for prototype, candidate in match_directed_segments_one_to_one(
            prototype_segments,
            sample_segments,
            occurrence_radius_days,
            max_duration_ratio,
        ):
            by_prototype[prototype.segment_id].append(candidate)
    total = len(sample_segment_sets)
    results = []
    for prototype in prototype_segments:
        matched = by_prototype[prototype.segment_id]
        occurrence = len(matched) / total if total else 0.0
        centers = [item.center_day for item in matched]
        timing_mad = circular_mad_days(centers, reference_day=prototype.center_day)
        ratios = np.asarray(
            [item.duration_days / max(prototype.duration_days, EPS) for item in matched],
            dtype=np.float64,
        )
        changes = np.asarray(
            [item.domain_normalized_change for item in matched], dtype=np.float64
        )
        reasons = [value for value in prototype.rejection_reason.split(";") if value]
        if occurrence < float(min_occurrence):
            reasons.append("source_occurrence_low")
        if not np.isfinite(timing_mad) or timing_mad > float(max_timing_mad_days):
            reasons.append("source_timing_unstable")
        results.append(
            replace(
                prototype,
                accepted=not reasons,
                rejection_reason=";".join(reasons),
                source_occurrence_rate=float(occurrence),
                source_center_timing_mad_days=float(timing_mad),
                source_duration_ratio_median=float(np.median(ratios)) if ratios.size else float("nan"),
                source_duration_ratio_error_p90=float(np.quantile(np.abs(ratios - 1.0), 0.9)) if ratios.size else float("nan"),
                source_domain_change_median=float(np.median(changes)) if changes.size else float("nan"),
                source_domain_change_iqr=float(np.quantile(changes, 0.75) - np.quantile(changes, 0.25)) if changes.size else float("nan"),
            )
        )
    return tuple(results)


def _forward_indices(start: int, end: int, size: int) -> Tuple[int, ...]:
    indices = []
    current = int(start)
    while current != int(end):
        indices.append(current)
        current = (current + 1) % int(size)
    return tuple(indices)


def _forward_duration(start_day: float, end_day: float, period_days: float) -> float:
    duration = (float(end_day) - float(start_day)) % float(period_days)
    return float(period_days) if duration == 0 else float(duration)


def build_coarse_structure(
    chain: Sequence[StructuralEvent],
    fine_segments: Sequence[DirectedStructureSegment],
    curve,
    domain_scale: float,
    max_reversal_ratio: float = 0.50,
    max_reversal_domain_change: float = 0.35,
    max_reversal_duration_days: float = 45,
    max_merge_depth: int = 5,
    min_duration_days: float = 20,
    max_duration_days: float = 240,
    min_curve_change: float = 0.20,
    min_domain_change: float = 0.40,
    min_monotonicity: float = 0.60,
    period_days: float = 365.0,
    segment_prefix: str = "C",
) -> CoarseStructureResult:
    """Iteratively remove the weakest local reversal from one circular chain."""
    ordered = tuple(sorted(chain, key=lambda item: float(item.day % period_days)))
    size = len(ordered)
    if size < 2:
        return CoarseStructureResult((), (), tuple(item.event_id for item in ordered), "no_segments")
    if len(fine_segments) != size:
        raise ValueError("coarse structure requires one original fine edge per extrema")

    surviving = list(range(size))
    removed = []
    depth_blocked = False
    while len(surviving) >= 4:
        removable = []
        for position in range(len(surviving)):
            indices = tuple(surviving[(position + offset) % len(surviving)] for offset in range(4))
            e0, e1, e2, e3 = (ordered[index] for index in indices)
            if e0.kind != e2.kind or e1.kind != e3.kind or e0.kind == e1.kind:
                continue
            left = abs(float(e1.value) - float(e0.value))
            reversal = abs(float(e2.value) - float(e1.value))
            right = abs(float(e3.value) - float(e2.value))
            ratio = float(reversal / (min(left, right) + EPS))
            domain_change = float(reversal / (float(domain_scale) + EPS))
            reversal_duration = _forward_duration(e1.day, e2.day, period_days)
            covered_edges = _forward_indices(indices[0], indices[3], size)
            weak = (
                ratio <= float(max_reversal_ratio)
                and domain_change <= float(max_reversal_domain_change)
                and reversal_duration <= float(max_reversal_duration_days)
            )
            if not weak:
                continue
            if len(covered_edges) > int(max_merge_depth):
                depth_blocked = True
                continue
            event_key = e1.event_id or f"event_{indices[1]}"
            removable.append(
                (
                    ratio,
                    domain_change,
                    reversal_duration,
                    float(e0.day % period_days),
                    event_key,
                    position,
                    indices,
                )
            )
        if not removable:
            break
        ratio, domain_change, duration, start_day, _, position, indices = min(removable)
        e1, e2 = ordered[indices[1]], ordered[indices[2]]
        removed.append(
            RemovedReversal(
                removal_id=f"R{len(removed)}",
                removed_event_ids=(e1.event_id, e2.event_id),
                reversal_ratio=float(ratio),
                domain_normalized_reversal=float(domain_change),
                reversal_duration_days=float(duration),
                circular_start_day=float(start_day),
            )
        )
        remove_indices = {indices[1], indices[2]}
        surviving = [index for index in surviving if index not in remove_indices]

    values = np.asarray(curve, dtype=np.float64)
    curve_scale = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    segments = []
    for segment_index, start_index in enumerate(surviving):
        end_index = surviving[(segment_index + 1) % len(surviving)]
        start, end = ordered[start_index], ordered[end_index]
        edge_indices = _forward_indices(start_index, end_index, size)
        path_fine = tuple(fine_segments[index] for index in edge_indices)
        interior_indices = set(edge_indices[1:])
        path_removed = tuple(
            item for item in removed
            if all(
                next(
                    index for index, event in enumerate(ordered)
                    if event.event_id == event_id
                ) in interior_indices
                for event_id in item.removed_event_ids
            )
        ) if all(event.event_id for event in ordered) else tuple(
            item for item in removed
            if item.circular_start_day in {
                float(ordered[index].day % period_days) for index in edge_indices
            }
        )
        duration = sum(item.duration_days for item in path_fine)
        signed_change = float(end.value - start.value)
        absolute_change = abs(signed_change)
        path_variation = float(sum(item.absolute_change for item in path_fine))
        monotonicity = float(absolute_change / (path_variation + EPS))
        reasons = []
        if duration < float(min_duration_days):
            reasons.append("duration_too_short")
        if duration > float(max_duration_days):
            reasons.append("duration_too_long")
        curve_change = float(absolute_change / (curve_scale + EPS))
        domain_change = float(absolute_change / (float(domain_scale) + EPS))
        if curve_change < float(min_curve_change):
            reasons.append("low_curve_change")
        if domain_change < float(min_domain_change):
            reasons.append("low_domain_change")
        if monotonicity < float(min_monotonicity):
            reasons.append("low_monotonicity")
        unwrapped_end = float(start.day % period_days) + float(duration)
        segments.append(
            CoarseStructureSegment(
                coarse_segment_id=f"{segment_prefix}{segment_index}",
                direction="RISE" if start.kind == "valley" else "FALL",
                start_event_id=start.event_id,
                end_event_id=end.event_id,
                start_day=float(start.day % period_days),
                end_day=float(end.day % period_days),
                center_day=float((start.day + duration / 2.0) % period_days),
                unwrapped_end_day=unwrapped_end,
                duration_days=float(duration),
                start_value=float(start.value),
                end_value=float(end.value),
                signed_change=signed_change,
                absolute_change=absolute_change,
                curve_normalized_change=curve_change,
                domain_normalized_change=domain_change,
                fine_segment_ids=tuple(item.segment_id for item in path_fine),
                removed_reversal_ids=tuple(item.removal_id for item in path_removed),
                num_fine_segments_covered=len(path_fine),
                num_removed_reversals=len(path_removed),
                total_path_variation=path_variation,
                net_change=absolute_change,
                monotonicity_ratio=monotonicity,
                max_removed_reversal_ratio=max((item.reversal_ratio for item in path_removed), default=float("nan")),
                max_removed_reversal_domain_change=max((item.domain_normalized_reversal for item in path_removed), default=float("nan")),
                max_removed_reversal_duration_days=max((item.reversal_duration_days for item in path_removed), default=float("nan")),
                crosses_year_boundary=bool(unwrapped_end >= period_days),
                accepted=not reasons,
                rejection_reason=";".join(reasons),
            )
        )
    return CoarseStructureResult(
        segments=tuple(segments),
        removed_reversals=tuple(removed),
        surviving_event_ids=tuple(ordered[index].event_id for index in surviving),
        stop_reason="merge_depth_limit" if depth_blocked else "no_weak_reversal",
    )


def match_coarse_segments_one_to_one(
    prototypes: Sequence[CoarseStructureSegment],
    candidates: Sequence[CoarseStructureSegment],
    occurrence_radius_days: float = 40,
    max_duration_ratio: float = 2.5,
):
    possible = []
    for prototype in prototypes:
        for candidate in candidates:
            ratio = candidate.duration_days / max(prototype.duration_days, EPS)
            distance = circular_day_distance(candidate.center_day, prototype.center_day)
            if (
                prototype.accepted and candidate.accepted
                and prototype.direction == candidate.direction
                and distance <= float(occurrence_radius_days)
                and 1.0 / float(max_duration_ratio) <= ratio <= float(max_duration_ratio)
            ):
                possible.append((
                    distance, abs(1.0 - ratio), prototype.coarse_segment_id,
                    candidate.coarse_segment_id, prototype, candidate,
                ))
    used_prototypes, used_candidates, matches = set(), set(), []
    for _, _, prototype_id, candidate_id, prototype, candidate in sorted(possible):
        if prototype_id in used_prototypes or candidate_id in used_candidates:
            continue
        used_prototypes.add(prototype_id)
        used_candidates.add(candidate_id)
        matches.append((prototype, candidate))
    return tuple(matches)


def evaluate_source_coarse_stability(
    prototype_segments: Sequence[CoarseStructureSegment],
    sample_segment_sets: Sequence[Sequence[CoarseStructureSegment]],
    occurrence_radius_days: float = 40,
    min_occurrence: float = 0.40,
    max_center_mad_days: float = 35,
    max_duration_ratio: float = 2.5,
) -> Tuple[CoarseStructureSegment, ...]:
    by_prototype = {item.coarse_segment_id: [] for item in prototype_segments}
    for sample_segments in sample_segment_sets:
        for prototype, candidate in match_coarse_segments_one_to_one(
            prototype_segments, sample_segments,
            occurrence_radius_days, max_duration_ratio,
        ):
            by_prototype[prototype.coarse_segment_id].append(candidate)
    total = len(sample_segment_sets)
    results = []
    for prototype in prototype_segments:
        matched = by_prototype[prototype.coarse_segment_id]
        occurrence = len(matched) / total if total else 0.0
        timing_mad = circular_mad_days(
            [item.center_day for item in matched], reference_day=prototype.center_day
        )
        ratios = [item.duration_days / max(prototype.duration_days, EPS) for item in matched]
        reasons = [item for item in prototype.rejection_reason.split(";") if item]
        if occurrence < float(min_occurrence):
            reasons.append("source_occurrence_low")
        if not np.isfinite(timing_mad) or timing_mad > float(max_center_mad_days):
            reasons.append("source_timing_unstable")
        results.append(replace(
            prototype,
            accepted=not reasons,
            rejection_reason=";".join(reasons),
            source_occurrence_rate=float(occurrence),
            source_center_timing_mad_days=float(timing_mad),
            source_duration_ratio_median=float(np.median(ratios)) if ratios else float("nan"),
        ))
    return tuple(results)
