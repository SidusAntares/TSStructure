"""Pure numerical helpers for the offline oracle Mode-13 phase/shape audit."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np


EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class PhaseEstimate:
    gamma: np.ndarray
    valid: bool
    failure_reason: str
    mean_displacement: float
    max_displacement: float
    p95_displacement: float
    min_derivative: float
    max_derivative: float
    roughness: float


def build_pointwise_median_prototypes(features, labels) -> Dict[int, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if values.ndim != 3 or labels.shape != (len(values),):
        raise ValueError("features must be [N,K,D] and labels must be [N]")
    return {
        int(class_id): np.median(values[labels == class_id], axis=0)
        for class_id in np.unique(labels)
    }


def robust_normalize(prototype):
    values = np.asarray(prototype, dtype=np.float64)
    median = np.median(values, axis=0, keepdims=True)
    q25, q75 = np.percentile(values, (25, 75), axis=0, keepdims=True)
    iqr = q75 - q25
    positive = iqr[np.isfinite(iqr) & (iqr > EPS)]
    floor = max(EPS, 1e-3 * np.median(positive)) if positive.size else EPS
    valid = np.isfinite(iqr) & (iqr > floor)
    result = np.zeros_like(values)
    np.divide(values - median, np.maximum(iqr, floor), out=result,
              where=np.broadcast_to(valid, values.shape))
    return result


def normalize_phase_pair(source, target, floor_ratio=1e-3):
    """Joint validity mask; separate source/target scales, invalid channels zero."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError("phase prototypes must have matching [K,D] shapes")
    if not np.isfinite(floor_ratio) or floor_ratio <= 0:
        raise ValueError("phase IQR floor ratio must be positive and finite")
    sq = np.percentile(source, 75, axis=0) - np.percentile(source, 25, axis=0)
    tq = np.percentile(target, 75, axis=0) - np.percentile(target, 25, axis=0)
    scales = np.r_[sq, tq]
    positive = scales[np.isfinite(scales) & (scales > EPS)]
    floor = max(EPS, floor_ratio * np.median(positive)) if positive.size else EPS
    valid = (sq > floor) & (tq > floor) & np.isfinite(sq) & np.isfinite(tq)
    sn, tn = np.zeros_like(source), np.zeros_like(target)
    sn[:, valid] = (source[:, valid] - np.median(source[:, valid], axis=0)) / np.maximum(sq[valid], floor)
    tn[:, valid] = (target[:, valid] - np.median(target[:, valid], axis=0)) / np.maximum(tq[valid], floor)
    finite = np.isfinite(source).all() and np.isfinite(target).all() and np.isfinite(sn).all() and np.isfinite(tn).all()
    reason = "" if finite and valid.any() else (
        "nonfinite_normalized_phase" if not finite else "no_valid_phase_channels")
    return dict(source=sn, target=tn, valid_channels=valid, source_iqr=sq,
                target_iqr=tq, iqr_floor=floor, failure_reason=reason,
                normalized_global_max_abs=float(max(np.max(np.abs(sn)), np.max(np.abs(tn)))))


def _resample(curve, count):
    curve = np.asarray(curve, dtype=np.float64)
    old = np.linspace(0.0, 1.0, len(curve))
    new = np.linspace(0.0, 1.0, count)
    return np.stack([np.interp(new, old, curve[:, d]) for d in range(curve.shape[1])], axis=1)


def _solve_joint_gamma(source, target, lam=0.0):
    from fdasrsf import curve_functions

    q_source = curve_functions.curve_to_q(source.T, mode="O", scale=False)[0]
    q_target = curve_functions.curve_to_q(target.T, mode="O", scale=False)[0]
    return curve_functions.optimum_reparam_curve(
        q_source, q_target, lam=lam, method="DP"
    )


def _phase_result(gamma, valid=True, reason=""):
    identity = np.linspace(0.0, 1.0, 128)
    gamma = np.asarray(gamma, dtype=np.float64)
    failure = ""
    if gamma.shape != (128,):
        failure = f"invalid_shape:{gamma.shape}"
    elif not np.all(np.isfinite(gamma)):
        failure = "non_finite"
    elif np.any(np.diff(gamma) < -1e-10):
        failure = "non_monotonic"
    elif not np.isclose(gamma[0], 0.0, atol=1e-5):
        failure = "invalid_start"
    elif not np.isclose(gamma[-1], 1.0, atol=1e-5):
        failure = "invalid_end"
    if failure or not valid:
        gamma = identity
        valid = False
        reason = failure or reason or "solver_failure"
    displacement = np.abs(gamma - identity)
    derivative = np.diff(gamma) * (len(gamma) - 1)
    return PhaseEstimate(
        gamma=gamma,
        valid=bool(valid),
        failure_reason=reason,
        mean_displacement=float(displacement.mean()),
        max_displacement=float(displacement.max()),
        p95_displacement=float(np.percentile(displacement, 95)),
        min_derivative=float(derivative.min()),
        max_derivative=float(derivative.max()),
        roughness=float(np.mean((derivative - 1.0) ** 2)),
    )


def estimate_nonlinear_phase(source, target, k_reg=128):
    if k_reg != 128:
        raise ValueError("the frozen diagnostic protocol requires k_reg=128")
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target prototypes must be [K,D]")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same feature dimension")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        return _phase_result(np.linspace(0, 1, 128), False, "non_finite_input")
    source_norm = _resample(robust_normalize(source), k_reg)
    target_norm = _resample(robust_normalize(target), k_reg)
    try:
        gamma = _solve_joint_gamma(source_norm, target_norm)
    except Exception as error:
        return _phase_result(np.linspace(0, 1, 128), False, type(error).__name__)
    return _phase_result(gamma)


def warp_curve(curve, gamma, output_points=None):
    curve = np.asarray(curve, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    count = len(curve) if output_points is None else int(output_points)
    query = np.interp(np.linspace(0, 1, count), np.linspace(0, 1, len(gamma)), gamma)
    base = np.linspace(0, 1, len(curve))
    return np.stack([np.interp(query, base, curve[:, d]) for d in range(curve.shape[1])], axis=1)


def _periodic_shift(curve, delta_days, period_days):
    curve = np.asarray(curve, dtype=np.float64)
    base = np.linspace(0.0, 1.0, len(curve), endpoint=False)
    query = np.mod(base + float(delta_days) / period_days, 1.0)
    xp = np.r_[base, 1.0]
    return np.stack(
        [np.interp(query, xp, np.r_[curve[:, d], curve[0, d]]) for d in range(curve.shape[1])],
        axis=1,
    )


def _flat_correlation(a, b):
    x, y = robust_normalize(a).ravel(), robust_normalize(b).ravel()
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > EPS else 0.0


def estimate_scalar_phase(source, target, period_days=365.0, radius_days=7):
    candidates = range(-int(radius_days), int(radius_days) + 1)
    aligned = [_periodic_shift(target, delta, period_days) for delta in candidates]
    scores = [_flat_correlation(source, value) for value in aligned]
    best = int(np.argmax(scores))
    return list(candidates)[best], aligned[best]


def registration_metrics(source, target):
    source, target = np.asarray(source), np.asarray(target)
    residual = target - source
    return {
        "l2": float(np.linalg.norm(residual)),
        "normalized_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(source), EPS)),
        "correlation": _flat_correlation(source, target),
    }


def amplitude_metrics(source, target):
    source, target = np.asarray(source), np.asarray(target)
    sr, tr = np.ptp(source), np.ptp(target)
    ss, ts = np.std(source), np.std(target)
    si = np.percentile(source, 75) - np.percentile(source, 25)
    ti = np.percentile(target, 75) - np.percentile(target, 25)
    return {
        "range_source": float(sr), "range_target": float(tr), "range_ratio": float(tr / max(sr, EPS)),
        "std_source": float(ss), "std_target": float(ts), "std_ratio": float(ts / max(ss, EPS)),
        "iqr_source": float(si), "iqr_target": float(ti), "iqr_ratio": float(ti / max(si, EPS)),
    }


def normalized_l2(a, b):
    return registration_metrics(robust_normalize(a), robust_normalize(b))["normalized_l2"]


def shape_margin(class_id, target, source_prototypes: Mapping[int, np.ndarray]):
    distances = {int(key): normalized_l2(value, target) for key, value in source_prototypes.items()}
    same = distances[int(class_id)]
    wrong_candidates = [
        (key, value) for key, value in distances.items() if key != int(class_id)
    ]
    if not wrong_candidates:
        raise ValueError("shape margin requires at least two common classes")
    wrong_class, wrong = min(wrong_candidates, key=lambda item: item[1])
    margin = wrong - same
    return {"same_class_shape_distance": same, "nearest_wrong_class": wrong_class,
            "nearest_wrong_distance": wrong, "shape_margin": margin, "margin_positive": margin > 0}


def landmark_alignment_metrics(source, target, times, prominence=0.0, min_distance_days=0.0):
    from models.structural_probe import detect_structural_landmarks

    source_marks = detect_structural_landmarks(times, source, min_distance_days, prominence)
    target_marks = detect_structural_landmarks(times, target, min_distance_days, prominence)
    errors = []
    matched_pairs = []
    matched = {"peak": 0, "valley": 0}
    for kind in ("peak", "valley"):
        left = [x for x in source_marks if x.kind == kind]
        right = [x for x in target_marks if x.kind == kind]
        pairs = _minimum_cost_ordered_pairs(left, right)
        matched[kind] = len(pairs)
        matched_pairs.extend(pairs)
        errors.extend(abs(a.time - b.time) for a, b in pairs)
    values = np.asarray(errors, dtype=np.float64)
    return {
        "matched_peak_count": matched["peak"], "matched_valley_count": matched["valley"],
        "unmatched_source_count": len(source_marks) - sum(matched.values()),
        "unmatched_target_count": len(target_marks) - sum(matched.values()),
        "mean_time_error": float(values.mean()) if len(values) else np.nan,
        "median_time_error": float(np.median(values)) if len(values) else np.nan,
        "p95_time_error": float(np.percentile(values, 95)) if len(values) else np.nan,
        "source_landmarks": source_marks, "target_landmarks": target_marks,
        "matched_pairs": tuple(sorted(matched_pairs, key=lambda pair: pair[0].time)),
    }


def _minimum_cost_ordered_pairs(left, right):
    """Match the smaller ordered sequence to a subset of the larger one."""
    if not left or not right:
        return []
    swapped = len(left) > len(right)
    short, long = (right, left) if swapped else (left, right)
    n, m = len(short), len(long)
    costs = np.full((n + 1, m + 1), np.inf)
    choices = np.zeros((n + 1, m + 1), dtype=np.int8)
    costs[0, :] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            skip = costs[i, j - 1]
            match = costs[i - 1, j - 1] + abs(
                short[i - 1].time - long[j - 1].time
            )
            if match <= skip:
                costs[i, j], choices[i, j] = match, 1
            else:
                costs[i, j], choices[i, j] = skip, 0
    pairs = []
    i, j = n, m
    while i and j:
        if choices[i, j]:
            pair = (short[i - 1], long[j - 1])
            pairs.append((pair[1], pair[0]) if swapped else pair)
            i -= 1
            j -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def select_phase_candidate(rows, scalar_error, tolerance=1e-9):
    """Select by landmark error, mean displacement, then larger penalty."""
    if not np.isfinite(scalar_error):
        return None
    eligible = [r for r in rows if r["lambda_value"] > 0
                and r["candidate_admissible"]
                and np.isfinite(r["landmark_error_mean"])]
    if not eligible:
        return None
    minimum = min(r["landmark_error_mean"] for r in eligible)
    tied = [r for r in eligible if r["landmark_error_mean"] <= minimum + tolerance]
    best = min(tied, key=lambda r: (r["gamma_mean_displacement_days"], -r["lambda_value"]))
    return best if best["landmark_error_mean"] < scalar_error else None


def constrained_residual_phase(source, target_scalar, loading, grid,
                               lambdas=(0, .01, .1, 1, 10), floor_ratio=1e-3,
                               max_warp_days=60, prominence=0, min_distance_days=14):
    """All candidates act on scalar-aligned raw data; only accepted warp escapes."""
    if not lambdas or any(not np.isfinite(lam) or lam < 0 for lam in lambdas):
        raise ValueError("phase lambdas must be finite and nonnegative")
    if not np.isfinite(max_warp_days) or max_warp_days < 0:
        raise ValueError("maximum residual warp days must be finite and nonnegative")
    normalized = normalize_phase_pair(source, target_scalar, floor_ratio)
    mask = normalized["valid_channels"]
    sn, tn = normalized["source"], normalized["target"]
    identity = np.linspace(0, 1, 128)
    scalar_marks = landmark_alignment_metrics(source @ loading, target_scalar @ loading,
                                             grid, prominence, min_distance_days)
    rows, gammas = [], {}
    for lam in lambdas:
        if normalized["failure_reason"]:
            phase = _phase_result(identity, False, normalized["failure_reason"])
        else:
            try:
                phase = _phase_result(_solve_joint_gamma(
                    _resample(sn[:, mask], 128), _resample(tn[:, mask], 128), lam=lam))
            except Exception as error:
                phase = _phase_result(identity, False, f"solver_failure:{type(error).__name__}")
        gammas[lam] = phase.gamma
        warped = warp_curve(target_scalar, phase.gamma)
        marks = landmark_alignment_metrics(source @ loading, warped @ loading,
                                           grid, prominence, min_distance_days)
        count = marks["matched_peak_count"] + marks["matched_valley_count"]
        metric_valid = count > 0 and np.isfinite(marks["mean_time_error"])
        reg = registration_metrics(sn, warp_curve(tn, phase.gamma))
        reason = phase.failure_reason
        if phase.valid:
            if phase.max_displacement * 365 > max_warp_days:
                reason = "residual_warp_too_large"
            elif not metric_valid:
                reason = "no_matched_landmarks"
            elif lam == 0:
                reason = "unrestricted_reference"
        rows.append(dict(
            lambda_value=float(lam), unrestricted_reference=lam == 0,
            registration_error=reg["normalized_l2"], registration_corr=reg["correlation"],
            landmark_error_mean=marks["mean_time_error"],
            landmark_error_median=marks["median_time_error"],
            landmark_error_p95=marks["p95_time_error"], matched_landmark_count=count,
            landmark_metric_valid=metric_valid,
            gamma_mean_displacement_days=phase.mean_displacement*365,
            gamma_max_displacement_days=phase.max_displacement*365,
            gamma_p95_displacement_days=phase.p95_displacement*365,
            gamma_roughness=phase.roughness, gamma_min_derivative=phase.min_derivative,
            gamma_max_derivative=phase.max_derivative, gamma_valid=phase.valid,
            candidate_admissible=bool(phase.valid and not reason), rejection_reason=reason))
    best = select_phase_candidate(rows, scalar_marks["mean_time_error"])
    accepted = best is not None
    if accepted:
        selected = _phase_result(gammas[best["lambda_value"]])
        reason = ""
    else:
        reason = normalized["failure_reason"] or (
            "no_matched_landmarks" if not np.isfinite(scalar_marks["mean_time_error"])
            else "no_landmark_improvement" if any(r["candidate_admissible"] for r in rows)
            else "no_admissible_candidate")
        selected = _phase_result(identity, False, reason)
    return dict(phase=selected, raw_aligned=warp_curve(target_scalar, selected.gamma)
                if accepted else target_scalar.copy(), normalized=normalized,
                candidates=rows, candidate_gammas=gammas, selected_lambda=best["lambda_value"] if accepted else None,
                nonlinear_accepted=accepted, selected_phase="scalar_plus_nonlinear" if accepted else "scalar",
                scalar_landmark_error=scalar_marks["mean_time_error"],
                selected_landmark_error=best["landmark_error_mean"] if accepted else scalar_marks["mean_time_error"],
                failure_reason=reason)


def edge_activity(curve, edge_points=8, monotonicity=.75, range_ratio=.15):
    """Boundary activity is evidence only, never a standalone rejection gate."""
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 1 or not 2 <= edge_points <= len(curve):
        raise ValueError("edge_points must be between 2 and curve length")
    result = {}
    for side, window in (("left", curve[:edge_points]), ("right", curve[-edge_points:])):
        diff = np.diff(window)
        ratio = float(max(np.count_nonzero(diff > 0), np.count_nonzero(diff < 0)) / len(diff))
        relative_range = float(np.ptp(window) / max(float(np.ptp(curve)), EPS))
        result[side] = dict(net_change=float(window[-1]-window[0]),
                            absolute_total_change=float(np.abs(diff).sum()),
                            monotonicity_ratio=ratio, edge_range=float(np.ptp(window)),
                            edge_range_ratio=relative_range, mean_slope=float(diff.mean()),
                            active=bool(ratio >= monotonicity and relative_range >= range_ratio))
    return result


def longest_contiguous_chain(source_marks, target_marks, pairs):
    """Both full-sequence indices must increment by one; ties choose earliest."""
    si = {(m.kind, m.time): i for i, m in enumerate(source_marks)}
    ti = {(m.kind, m.time): i for i, m in enumerate(target_marks)}
    best, current, previous = [], [], None
    for pair in sorted(pairs, key=lambda p: p[0].time):
        index = (si[(pair[0].kind, pair[0].time)], ti[(pair[1].kind, pair[1].time)])
        if previous is None or index != (previous[0]+1, previous[1]+1):
            current = []
        current.append(pair)
        if len(current) > len(best):
            best = current.copy()
        previous = index
    return best


def discover_phase_support(source, target_scalar, grid, prominence=0., min_distance_days=14.,
                           edge_points=8, edge_monotonicity=.75, edge_range_ratio=.15,
                           full_landmark_coverage=.80, partial_min_landmarks=2,
                           partial_min_time_coverage=.20):
    """Discover structure only in the Global+Scalar frame. No padding or extension."""
    if partial_min_landmarks < 2 or not 0 <= partial_min_time_coverage <= 1:
        raise ValueError("partial support needs >=2 landmarks and coverage in [0,1]")
    if not all(0 <= x <= 1 for x in (edge_monotonicity, edge_range_ratio, full_landmark_coverage)):
        raise ValueError("support ratios must be in [0,1]")
    marks = landmark_alignment_metrics(source, target_scalar, grid, prominence, min_distance_days)
    sm, tm, pairs = marks["source_landmarks"], marks["target_landmarks"], marks["matched_pairs"]
    chain = longest_contiguous_chain(sm, tm, pairs)
    edges = {"source": edge_activity(source, edge_points, edge_monotonicity, edge_range_ratio),
             "target": edge_activity(target_scalar, edge_points, edge_monotonicity, edge_range_ratio)}
    result = dict(source_landmark_count=len(sm), target_landmark_count=len(tm),
                  matched_landmark_count=len(pairs), common_chain_landmark_count=len(chain),
                  source_landmark_coverage=len(chain)/len(sm) if sm else 0.,
                  target_landmark_coverage=len(chain)/len(tm) if tm else 0.,
                  source_landmarks=sm, target_landmarks=tm, matched_pairs=pairs, common_chain=chain)
    for domain, edge in edges.items():
        for side, metrics in edge.items():
            result[side + "_boundary_active_" + domain] = metrics["active"]
            for key, value in metrics.items():
                result[domain + "_" + side + "_" + key] = value
    for side in ("left", "right"):
        strong = False
        for domain, own, other, oriented in (("source", sm, tm, pairs),
                                             ("target", tm, sm, [(b, a) for a, b in pairs])):
            if not own or not oriented or not edges[domain][side]["active"]:
                continue
            anchor = own[0 if side == "left" else -1]
            mate = next((b for a, b in oriented if a == anchor), None)
            if mate is None:
                continue
            slope = edges[domain][side]["mean_slope"]
            # Left declining into valley => missing peak; right declining out of peak => missing valley.
            expected_anchor = ("valley" if slope < 0 else "peak") if side == "left" else (
                "peak" if slope < 0 else "valley")
            missing_kind = "peak" if expected_anchor == "valley" else "valley"
            matched_other = {(b.kind, b.time) for _, b in oriented}
            unmatched = [m for m in other if (m.kind, m.time) not in matched_other
                         and (m.time < mate.time if side == "left" else m.time > mate.time)]
            strong |= anchor.kind == expected_anchor and any(m.kind == missing_kind for m in unmatched)
        result[side + "_truncation_evidence"] = "strong" if strong else (
            "candidate" if any(edges[d][side]["active"] for d in edges) else "none")
    for domain, column in (("source", 0), ("target", 1)):
        start, end = (chain[0][column].time, chain[-1][column].time) if chain else (np.nan, np.nan)
        result[domain + "_common_start_day"] = start
        result[domain + "_common_end_day"] = end
        result[domain + "_common_time_coverage"] = (end-start)/365 if chain else 0.
    coverage = min(result["source_common_time_coverage"], result["target_common_time_coverage"])
    result["common_time_coverage_min"] = coverage
    result["partial_support_valid"] = len(chain) >= partial_min_landmarks and coverage >= partial_min_time_coverage and coverage > 0
    result["full_support_valid"] = (len(chain) >= 2 and coverage > 0
        and min(result["source_landmark_coverage"], result["target_landmark_coverage"]) >= full_landmark_coverage
        and all(result[s + "_truncation_evidence"] != "strong" for s in ("left", "right")))
    return result


def _crop_support(values, grid, start, end):
    grid, values = np.asarray(grid, dtype=float), np.asarray(values, dtype=float)
    if not grid[0] <= start < end <= grid[-1]:
        raise ValueError("common support must be a nonempty interval within the observed grid")
    times = np.r_[start, grid[(grid > start) & (grid < end)], end]
    return times, np.column_stack([np.interp(times, grid, v) for v in values.T])


def _normalize_partial_pair(source, target, floor_ratio):
    """Native cropped samples, BEFORE resampling; supports unequal crop lengths."""
    if not np.isfinite(floor_ratio) or floor_ratio <= 0:
        raise ValueError("phase IQR floor ratio must be positive and finite")
    sq = np.percentile(source, 75, axis=0)-np.percentile(source, 25, axis=0)
    tq = np.percentile(target, 75, axis=0)-np.percentile(target, 25, axis=0)
    scales = np.r_[sq, tq]
    positive = scales[np.isfinite(scales) & (scales > EPS)]
    floor = max(EPS, floor_ratio*np.median(positive)) if positive.size else EPS
    mask = np.isfinite(sq) & np.isfinite(tq) & (sq > floor) & (tq > floor)
    sn, tn = np.zeros_like(source), np.zeros_like(target)
    sn[:, mask] = (source[:, mask]-np.median(source[:, mask], axis=0))/sq[mask]
    tn[:, mask] = (target[:, mask]-np.median(target[:, mask], axis=0))/tq[mask]
    finite = all(np.isfinite(v).all() for v in (source, target, sn, tn))
    return dict(source=sn, target=tn, valid_channels=mask, source_iqr=sq, target_iqr=tq,
                iqr_floor=floor, failure_reason="" if finite and mask.any() else (
                    "nonfinite_normalized_phase" if not finite else "no_valid_phase_channels"))


def constrained_partial_phase(source, target_scalar, grid, support,
                              lambdas=(0, .01, .1, 1, 10), floor_ratio=1e-3, max_warp_days=60):
    """Same joint SRVF and selection gates as full; operates ONLY inside support."""
    if not lambdas or any(not np.isfinite(lam) or lam < 0 for lam in lambdas):
        raise ValueError("phase lambdas must be finite and nonnegative")
    if not np.isfinite(max_warp_days) or max_warp_days < 0:
        raise ValueError("maximum residual warp days must be finite and nonnegative")
    if not support["partial_support_valid"]:
        return dict(solution_valid=False, selected_phase="none", failure_reason="invalid_partial_support",
                    selected_lambda=None, nonlinear_accepted=False, candidates=[], candidate_gammas={},
                    scalar_landmark_error=np.nan, selected_landmark_error=np.nan)
    a_s, b_s = (support["source_common_" + k + "_day"] for k in ("start", "end"))
    a_t, b_t = (support["target_common_" + k + "_day"] for k in ("start", "end"))
    st, sr = _crop_support(source, grid, a_s, b_s)
    tt, tr = _crop_support(target_scalar, grid, a_t, b_t)
    normalized = _normalize_partial_pair(sr, tr, floor_ratio)
    identity = np.linspace(0., 1., 128)
    source_grid = a_s + identity*(b_s-a_s)
    target_grid = a_t + identity*(b_t-a_t)
    def resample(values, times, query):
        return np.column_stack([np.interp(query, times, v) for v in values.T])
    sn = resample(normalized["source"], st, source_grid)
    tn = resample(normalized["target"], tt, target_grid)
    chain = support["common_chain"]
    u_s = (np.array([a.time for a, _ in chain])-a_s)/(b_s-a_s)
    target_times = np.array([b.time for _, b in chain])
    def errors(gamma):
        return np.abs(a_t + np.interp(u_s, identity, gamma)*(b_t-a_t)-target_times)
    baseline = float(errors(identity).mean())
    rows, gammas = [], {}
    mask = normalized["valid_channels"]
    for lam in lambdas:
        if normalized["failure_reason"]:
            phase = _phase_result(identity, False, normalized["failure_reason"])
        else:
            try:
                phase = _phase_result(_solve_joint_gamma(sn[:, mask], tn[:, mask], lam=lam))
            except Exception as error:
                phase = _phase_result(identity, False, f"solver_failure:{type(error).__name__}")
        gammas[lam] = phase.gamma
        error = errors(phase.gamma)
        metric_valid = len(chain) >= 2 and np.isfinite(error).all()
        reg = registration_metrics(sn, warp_curve(tn, phase.gamma))
        reason = phase.failure_reason
        if phase.valid:
            if phase.max_displacement*(b_t-a_t) > max_warp_days:
                reason = "residual_warp_too_large"
            elif not metric_valid:
                reason = "no_matched_landmarks"
            elif lam == 0:
                reason = "unrestricted_reference"
        rows.append(dict(lambda_value=float(lam), unrestricted_reference=lam == 0,
                         registration_error=reg["normalized_l2"], registration_corr=reg["correlation"],
                         landmark_error_mean=float(error.mean()), landmark_error_median=float(np.median(error)),
                         landmark_error_p95=float(np.percentile(error, 95)), matched_landmark_count=len(chain),
                         landmark_metric_valid=bool(metric_valid), gamma_valid=phase.valid,
                         gamma_mean_displacement_days=phase.mean_displacement*(b_t-a_t),
                         gamma_max_displacement_days=phase.max_displacement*(b_t-a_t),
                         gamma_p95_displacement_days=phase.p95_displacement*(b_t-a_t),
                         gamma_roughness=phase.roughness, gamma_min_derivative=phase.min_derivative,
                         gamma_max_derivative=phase.max_derivative,
                         candidate_admissible=bool(phase.valid and not reason), rejection_reason=reason))
    best = select_phase_candidate(rows, baseline)
    selected = _phase_result(gammas[best["lambda_value"]] if best else identity)
    valid = not normalized["failure_reason"] and np.isfinite(baseline)
    return dict(phase=selected, solution_valid=bool(valid), normalized=normalized, candidates=rows,
                candidate_gammas=gammas, selected_lambda=best["lambda_value"] if best else None,
                nonlinear_accepted=best is not None, selected_phase=("partial_nonlinear" if best else "partial_linear") if valid else "none",
                scalar_landmark_error=baseline, selected_landmark_error=best["landmark_error_mean"] if best else baseline,
                failure_reason=normalized["failure_reason"] or ("" if best else "no_landmark_improvement"),
                raw_source=resample(sr, st, source_grid),
                raw_aligned=resample(tr, tt, a_t + selected.gamma*(b_t-a_t)),
                source_grid=source_grid, mapped_target_grid=a_t + selected.gamma*(b_t-a_t))


def classify_phase_applicability(support, full, partial):
    full_valid = (full is not None and not full["normalized"]["failure_reason"]
                  and np.isfinite(full["selected_landmark_error"])
                  and (not full["nonlinear_accepted"] or full["phase"].valid))
    if support["full_support_valid"] and full_valid:
        return dict(phase_applicability="FULL_PHASE", phase_applicability_reason=
                    "full_landmark_coverage_high;no_boundary_truncation;full_warp_valid:" + full["selected_phase"])
    reasons = [side + "_boundary_truncation" for side in ("left", "right")
               if support[side + "_truncation_evidence"] == "strong"]
    if support["partial_support_valid"] and partial is not None and partial["solution_valid"]:
        reasons += [f'{support["common_chain_landmark_count"]}_contiguous_landmarks',
                    f'common_support={support["common_time_coverage_min"]:.6f}', partial["selected_phase"],
                    f'partial_landmark_gain={partial["scalar_landmark_error"]-partial["selected_landmark_error"]:.6f}d']
        return dict(phase_applicability="PARTIAL_PHASE", phase_applicability_reason=";".join(reasons))
    reasons.append("insufficient_common_structure" if not support["partial_support_valid"] else
                   "invalid_partial_alignment:" + (partial["failure_reason"] if partial else "not_attempted"))
    return dict(phase_applicability="PHASE_NOT_APPLICABLE", phase_applicability_reason=";".join(reasons))


class VisualizationUnavailable(ValueError):
    """Expected missing/invalid plotting data; numerical diagnostics stay intact."""


def fit_visualization_shared_pca(source_prototypes):
    """Source-only equal-class prototype PCA, centered for fitting, raw projection."""
    if not source_prototypes:
        raise VisualizationUnavailable("no source prototypes for shared PCA")
    values = np.concatenate([source_prototypes[c] for c in sorted(source_prototypes)], axis=0)
    if not np.isfinite(values).all():
        raise VisualizationUnavailable("nonfinite source prototypes for shared PCA")
    centered = values - values.mean(axis=0)
    if np.linalg.norm(centered) <= EPS:
        raise VisualizationUnavailable("no variation for shared PCA")
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0].copy()
    if direction[np.argmax(np.abs(direction))] < 0:
        direction *= -1
    return direction


def select_visualization_samples(sample_ids, maximum=40):
    ids = np.asarray(sample_ids)
    if ids.ndim != 1 or maximum < 1:
        raise ValueError("sample IDs must be a vector and maximum must be positive")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("visualization sample IDs must be unique within each group")
    order = np.argsort(ids, kind="stable")
    return order[np.linspace(0, len(order)-1, min(maximum, len(order))).astype(int)] if len(order) else order


def _visualization_bundle(curves, prototype, indices):
    """Summaries use the whole group; only thin curves are subsampled."""
    curves, prototype = np.asarray(curves), np.asarray(prototype)
    quantiles = np.full((3, curves.shape[1]), np.nan)
    for column in range(curves.shape[1]):
        finite = curves[:, column][np.isfinite(curves[:, column])]
        if finite.size:
            quantiles[:, column] = np.percentile(finite, [25, 50, 75])
    finite = curves[np.isfinite(curves)]
    bounds = (float(finite.min()), float(finite.max())) if finite.size else (0., 0.)
    return dict(samples=curves[indices], prototype=prototype, quantiles=quantiles, bounds=bounds)


def prepare_visualization_group(source, target, source_ids, target_ids, source_prototype,
                                target_prototype, direction, grid, scalar_delta, state,
                                gamma=None, support=None, partial_gamma=None, max_curves=40):
    """Read-only post-diagnostic projection/warp. No model, loader or Fourier API."""
    source, target, grid = np.asarray(source), np.asarray(target), np.asarray(grid)
    if source.ndim != 3 or target.ndim != 3 or source.shape[1:] != target.shape[1:]:
        raise ValueError("visualization representations must have matching [N,K,D] dimensions")
    if len(source_ids) != len(source) or len(target_ids) != len(target):
        raise ValueError("sample IDs must correspond to representations")
    if not len(source) or not len(target):
        raise VisualizationUnavailable("empty sample group")
    if state not in ("FULL_PHASE", "PARTIAL_PHASE", "PHASE_NOT_APPLICABLE"):
        raise ValueError("unknown phase applicability")
    sc, tc = source @ direction, target @ direction
    sp, tp = source_prototype @ direction, target_prototype @ direction
    if not all(np.isfinite(x).all() for x in (sc, tc, sp, tp)):
        raise VisualizationUnavailable("nonfinite projected group")
    si, ti = select_visualization_samples(source_ids, max_curves), select_visualization_samples(target_ids, max_curves)
    scalar = _periodic_shift(tc.T, scalar_delta, 365).T
    scalar_proto = _periodic_shift(tp[:, None], scalar_delta, 365)[:, 0]
    selected, selected_proto = scalar.copy(), scalar_proto.copy()
    selected_sc, selected_sp = sc.copy(), sp.copy()
    support_mask = np.ones(len(grid), dtype=bool)
    if state == "FULL_PHASE" and gamma is not None:
        selected = warp_curve(scalar.T, gamma).T
        selected_proto = warp_curve(scalar_proto[:, None], gamma)[:, 0]
    elif state == "PARTIAL_PHASE":
        if support is None or partial_gamma is None:
            raise ValueError("partial visualization needs existing support and selected gamma")
        a_s, b_s = (support["source_common_" + k + "_day"] for k in ("start", "end"))
        a_t, b_t = (support["target_common_" + k + "_day"] for k in ("start", "end"))
        support_mask = (grid >= a_s) & (grid <= b_s)
        if not support_mask.any():
            raise VisualizationUnavailable("no grid points inside common support")
        u = (grid[support_mask]-a_s)/(b_s-a_s)
        query = a_t + np.interp(u, np.linspace(0, 1, len(partial_gamma)), partial_gamma)*(b_t-a_t)
        times, cropped = _crop_support(scalar.T, grid, a_t, b_t)
        _, cropped_proto = _crop_support(scalar_proto[:, None], grid, a_t, b_t)
        selected[:] = np.nan
        selected[:, support_mask] = np.stack([np.interp(query, times, c) for c in cropped.T])
        selected_proto[:] = np.nan
        selected_proto[support_mask] = np.interp(query, times, cropped_proto[:, 0])
        selected_sc[:, ~support_mask] = np.nan
        selected_sp[~support_mask] = np.nan
    residual = selected - sp
    residual_proto = selected_proto - sp
    if state == "PHASE_NOT_APPLICABLE":
        residual[:] = np.nan
        residual_proto[:] = np.nan
    return dict(grid=grid, state=state, support=support or {}, support_mask=support_mask,
                source_ids=np.asarray(source_ids)[si], target_ids=np.asarray(target_ids)[ti],
                source=_visualization_bundle(sc, sp, si),
                target_global=_visualization_bundle(tc, tp, ti),
                target_scalar=_visualization_bundle(scalar, scalar_proto, ti),
                source_selected=_visualization_bundle(selected_sc, selected_sp, si),
                target_selected=_visualization_bundle(selected, selected_proto, ti),
                residual=_visualization_bundle(residual, residual_proto, ti))


def visualization_distance_matrices(artifacts, class_ids):
    """Use the unchanged multidimensional metric; never construct cross-class partial warps."""
    before = np.full((len(class_ids), len(class_ids)), np.nan)
    selected = before.copy()
    for column, target_id in enumerate(class_ids):
        target = artifacts["target_raw_prototypes"][target_id]
        scalar = _periodic_shift(target, artifacts["scalar_delta_by_class"][target_id], 365)
        full = artifacts["phase_applicability_by_class"][target_id] == "FULL_PHASE"
        aligned = (warp_curve(scalar, artifacts["gamma_by_class"][target_id])
                   if full and artifacts["selected_lambda"][target_id] is not None else scalar)
        for row, source_id in enumerate(class_ids):
            source = artifacts["source_raw_prototypes"][source_id]
            if np.isfinite(source).all() and np.isfinite(scalar).all():
                before[row, column] = normalized_l2(source, scalar)
                if full:
                    selected[row, column] = normalized_l2(source, aligned)
    return before, selected
