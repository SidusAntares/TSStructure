"""Offline NumPy RBF segmentation diagnostics; no model or training dependencies.

Objectives charge a penalty per internal boundary. Calendar lengths are the
last minus first observed date of each segment, without imputing observations.
"""

import itertools

import numpy as np


def make_cost(features, normalization='feature', gamma_multiplier=1.):
    """Return an O(1) kernel SSE cost and diagnostics for a finite [T,D] array.

    ``feature`` exactly follows production population standardization. ``global``
    uses one mean/std for all entries; ``none`` retains original feature units.
    Bandwidth uses positive (>1e-12) upper-triangle squared distances. The IQR
    ratio is (Q75-Q25)/median over those same positive distances (zero if none).
    """
    x = np.array(features, dtype=np.float64, copy=True)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite [T,D] feature array')
    if normalization not in ('feature', 'global', 'none'):
        raise ValueError('Unknown normalization')
    if not np.isfinite(gamma_multiplier) or gamma_multiplier <= 0:
        raise ValueError('gamma_multiplier must be finite and positive')
    feature_scale = x.std(0)
    constant_fraction = float(np.mean(feature_scale <= 1e-12))
    if normalization == 'feature':
        x = (x - x.mean(0)) / np.where(feature_scale > 1e-12, feature_scale, 1.)
    elif normalization == 'global':
        scale = x.std()
        x = (x - x.mean()) / (scale if scale > 1e-12 else 1.)
    distances = np.maximum(((x[:, None] - x[None]) ** 2).sum(-1), 0.)
    if not np.isfinite(distances).all():
        raise ValueError('Feature magnitudes overflow pairwise distances')
    upper = distances[np.triu_indices(len(x), 1)]
    positive = upper[upper > 1e-12]
    median = float(np.median(positive)) if len(positive) else 0.
    gamma = float(gamma_multiplier) / median if median else float(gamma_multiplier)
    kernel = np.exp(-gamma * distances)
    integral = np.pad(kernel.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def cost(start, end):
        total = integral[end, end] - integral[start, end] - integral[end, start] + integral[start, start]
        return np.maximum(0., end - start - total / (end - start))

    stats = dict(median_distance=median, gamma=gamma,
                 kernel_offdiag_mean=float(kernel[np.triu_indices(len(x), 1)].mean()) if len(x) > 1 else 1.,
                 distance_iqr_ratio=float(np.diff(np.percentile(positive, [25, 75]))[0] / median) if median else 0.,
                 constant_feature_fraction=constant_fraction,
                 single_cost=float(cost(0, len(x))))
    return cost, stats


def pelt_boundaries(cost, n, penalty, min_size=3):
    """Exact PELT with delayed pruning for minimum observed segment length.

    The recurrence and 1e-10 pruning tolerance match production. A sequence
    shorter than min_size remains one segment. Cost may be any scalar closure.
    """
    if int(n) != n or n < 1 or int(min_size) != min_size or min_size < 1:
        raise ValueError('n and min_size must be positive integers')
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError('penalty must be finite and nonnegative')
    n, min_size = int(n), int(min_size)
    if n < 2 * min_size:
        return [0, n]
    objective = np.full(n + 1, np.inf)
    previous = np.zeros(n + 1, dtype=int)
    objective[0] = -penalty
    active = []
    expires = {}
    for end in range(min_size, n + 1):
        active = [start for start in active if expires.get(start, n + 1) > end]
        start = end - min_size
        if np.isfinite(objective[start]):
            active.append(start)
        costs = [cost(start, end) for start in active]
        values = [objective[start] + segment_cost + penalty for start, segment_cost in zip(active, costs)]
        winner = int(np.argmin(values))
        objective[end] = values[winner]
        previous[end] = active[winner]
        for start, segment_cost in zip(active, costs):
            if objective[start] + segment_cost > objective[end] + 1e-10:
                expires[start] = min(expires.get(start, n + 1), end + min_size)
    boundaries = [n]
    while boundaries[-1] > 0:
        boundaries.append(int(previous[boundaries[-1]]))
    return boundaries[::-1]


def _date_days(dates, n):
    dates = np.asarray(dates)
    if dates.ndim != 1 or len(dates) != n:
        raise ValueError('dates must have one timestamp per observation')
    if dates.dtype.kind in 'USO':
        try:
            dates = dates.astype('datetime64[ns]')
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid timestamps') from exc
    if dates.dtype.kind == 'M':
        if np.isnat(dates).any():
            raise ValueError('Dates must be finite')
        dates = (dates - dates[0]) / np.timedelta64(1, 'D')
    dates = np.asarray(dates, dtype=float)
    if not np.isfinite(dates).all() or np.any(np.diff(dates) < 0):
        raise ValueError('Dates must be finite and sorted')
    return dates


def sweep_sequence(features, dates, penalties=(.003, .01, .03, .1, .3),
                   normalizations=('feature', 'global', 'none'),
                   gamma_multipliers=(.25, 1., 4.),
                   penalty_modes=('absolute', 'per_observation'), min_size=3):
    """Evaluate the full factorial grid without labels or segment-count targets.

    Kernel statistics and costs are cached for each normalization/bandwidth.
    best_single_split_gain compares the full cost with the best valid two-piece
    split, before penalty; it does not describe the global optimum.
    """
    x = np.asarray(features)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite [T,D] feature array')
    n = len(x)
    days = _date_days(dates, n)
    penalties, penalty_modes = tuple(penalties), tuple(penalty_modes)
    if any(mode not in ('absolute', 'per_observation') for mode in penalty_modes):
        raise ValueError('Unknown penalty mode')
    if any(not np.isfinite(p) or p < 0 for p in penalties):
        raise ValueError('Penalties must be finite and nonnegative')
    if int(min_size) != min_size or min_size < 1:
        raise ValueError('min_size must be a positive integer')
    min_size = int(min_size)
    rows = []
    for normalization, multiplier in itertools.product(normalizations, gamma_multipliers):
        cost, stats = make_cost(x, normalization, multiplier)
        # Precompute all scalar segment costs once across the penalty grid.
        cached = np.zeros((n + 1, n + 1))
        for start in range(n):
            ends = np.arange(start + 1, n + 1)
            cached[start, ends] = cost(start, ends)
        lookup = lambda start, end: cached[start, end]
        cuts = range(min_size, n - min_size + 1)
        split_costs = [lookup(0, cut) + lookup(cut, n) for cut in cuts]
        gain = float(stats['single_cost'] - min(split_costs)) if split_costs else None
        for mode, coefficient in itertools.product(penalty_modes, penalties):
            penalty = float(coefficient * n if mode == 'per_observation' else coefficient)
            bounds = pelt_boundaries(lookup, n, penalty, min_size)
            lengths = np.diff(bounds)
            count = len(lengths)
            rows.append(dict(normalization=normalization, gamma_multiplier=float(multiplier),
                             penalty_mode=mode, penalty_coefficient=float(coefficient), actual_penalty=penalty,
                             min_size=min_size, n_observations=n, boundaries=bounds, segment_count=count,
                             single_segment=count == 1, longest_fraction=float(lengths.max() / n),
                             segment_lengths_observations=lengths.tolist(),
                             segment_lengths_days=[float(days[b - 1] - days[a]) for a, b in zip(bounds, bounds[1:])],
                             penalized_objective=float(sum(lookup(a, b) for a, b in zip(bounds, bounds[1:])) + penalty * (count - 1)),
                             single_segment_cost=stats['single_cost'], best_single_split_gain=gain, **stats))
    return rows


def summarize_records(rows):
    """Aggregate sequences within task/view and each fixed scan condition.

    Observation/day length quantiles pool all segments in the group. No labels
    enter grouping, ranking, filtering, or selection.
    """
    condition = ('normalization', 'gamma_multiplier', 'penalty_mode', 'penalty_coefficient', 'min_size')
    groups = {}
    for row in rows:
        keys = tuple(k for k in ('task', 'view') + condition if k in row)
        key = tuple((k, row[k]) for k in keys)
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in groups.items():
        out = dict(key)
        k = np.asarray([r['segment_count'] for r in group])
        longest = np.asarray([r['longest_fraction'] for r in group])
        out.update(sequence_count=len(group), mean_K=float(k.mean()), median_K=float(np.median(k)),
                   p10_K=float(np.percentile(k, 10)), p90_K=float(np.percentile(k, 90)),
                   single_segment_rate=float(np.mean([r['single_segment'] for r in group])),
                   mean_longest_fraction=float(longest.mean()),
                   p10_longest_fraction=float(np.percentile(longest, 10)),
                   median_longest_fraction=float(np.median(longest)),
                   p90_longest_fraction=float(np.percentile(longest, 90)))
        for field, suffix in [('segment_lengths_observations', 'segment_length_observations'), ('segment_lengths_days', 'segment_length_days')]:
            values = np.asarray([value for r in group for value in r[field]])
            for prefix, value in [('min', values.min()), ('median', np.median(values)),
                                  ('p10', np.percentile(values, 10)), ('p90', np.percentile(values, 90))]:
                out[prefix + '_' + suffix] = float(value)
        gains = [r['best_single_split_gain'] for r in group if r['best_single_split_gain'] is not None]
        out['median_best_single_split_gain'] = float(np.median(gains)) if gains else None
        for field in ('median_distance', 'kernel_offdiag_mean', 'distance_iqr_ratio', 'constant_feature_fraction', 'single_segment_cost'):
            out['median_' + field] = float(np.median([r[field] for r in group]))
        result.append(out)
    return result
