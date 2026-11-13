"""Full-observation matching and ordered Shapelet-guided merge proposals."""

from typing import NamedTuple

import numpy as np
import torch

from utils.source_segment_refinement import cut_gap, dtw, merge_costs, prototype_windows, ranges_for


class ShapeletMatch(NamedTuple):
    prototype: int
    start: int
    stop: int
    distance: float


def match_shapelet(dates, shape, prototype, min_size, settings):
    size = len(prototype['shape'])
    best = min(((dtw(shape[a:b], prototype['shape']), a, b)
                for a, b in prototype_windows(dates, 0, len(dates), prototype, settings)
                if max(min_size, .5*size) <= b-a <= 2*size), default=None)
    if best is None or best[0] > settings['dtw_limit']:
        return None
    distance, start, stop = best
    return ShapeletMatch(prototype['id'], start, stop, distance)


def match_shapelets(dates, shape, prototypes, min_size, settings):
    candidates = []
    for p in prototypes:
        match = match_shapelet(dates, shape, p, min_size, settings)
        if match is not None:
            candidates.append((match.distance, -p['frequency'], match))
    matches = []
    occupied = np.zeros(len(dates), dtype=bool)
    for _, _, match in sorted(candidates):
        span = slice(match.start, match.stop)
        if not occupied[span].any():
            occupied[span] = True
            matches.append(match)
    return matches


def split_matches(dates, cuts, matches, min_size, settings):
    """Insert only missing outer boundaries of cached matches."""
    inserted = []
    current = cuts
    for match in matches:
        for index in (match.start, match.stop):
            if len(inserted) == settings['split_budget']:
                return inserted
            if index == 0 or index == len(dates) or dates[index-1] == dates[index]:
                continue
            cut = cut_gap(dates, index)
            if np.any(np.abs(current-cut) <= settings['boundary_slack']):
                continue
            proposal = np.sort(np.r_[current, cut])
            if all(b-a >= min_size for a, b in ranges_for(dates, proposal)):
                inserted.append(cut)
                current = proposal
    return inserted


def guided_merge(model, features, positions, y, cuts, gates, protected, matches, gamma, settings):
    """Accept the first valid guided triple, otherwise the first valid adjacent pair."""
    dates = positions.cpu().numpy()
    spans = ranges_for(dates, cuts)
    kept = torch.as_tensor(np.repeat(gates, [b-a for a, b in spans]), device=features.device)
    protected = set(protected)

    def accepted(first, stop):
        a, b = spans[first][0], spans[stop-1][1]
        inner = cuts[first:stop-1].tolist()
        if (not gates[first:stop].all() or protected.intersection(inner)
                or b-a > settings['merge_max_points']
                or dates[b-1]-dates[a] > settings['merge_max_span']
                or np.any(np.diff(dates[a:b]) > settings['merge_max_gap'])):
            return []
        masks = [kept]
        for left, right in spans[first:stop]:
            dropped = kept.clone()
            dropped[left:right] = False
            masks.append(dropped)
        costs = merge_costs(model, features, positions, y, torch.stack(masks), gamma,
                            states=len(masks), coverage_cost=settings['merge_coverage_cost'])[0]
        return inner if costs[0]-costs.min() <= settings['merge_tolerance'] else []

    checked = set()
    for match in matches:
        first = next(i for i, (_, b) in enumerate(spans) if b > match.start)
        stop = next(i+1 for i, (_, b) in enumerate(spans) if b >= match.stop)
        if stop-first != 3 or (first, stop) in checked:
            continue
        if (abs(dates[spans[first][0]]-dates[match.start]) > settings['boundary_slack']
                or abs(dates[spans[stop-1][1]-1]-dates[match.stop-1]) > settings['boundary_slack']):
            continue
        checked.add((first, stop))
        removed = accepted(first, stop)
        if removed:
            return removed, 'guided'
    for first in range(len(spans)-1):
        removed = accepted(first, first+2)
        if removed:
            return removed, 'ordinary'
    return [], None
