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
    from models.fredn.structural_probe import detect_structural_landmarks

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
