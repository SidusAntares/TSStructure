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
    scale = np.maximum(q75 - q25, EPS)
    return (values - median) / scale


def _resample(curve, count):
    curve = np.asarray(curve, dtype=np.float64)
    old = np.linspace(0.0, 1.0, len(curve))
    new = np.linspace(0.0, 1.0, count)
    return np.stack([np.interp(new, old, curve[:, d]) for d in range(curve.shape[1])], axis=1)


def _solve_joint_gamma(source, target):
    from fdasrsf import curve_functions

    q_source = curve_functions.curve_to_q(source.T, mode="O", scale=False)[0]
    q_target = curve_functions.curve_to_q(target.T, mode="O", scale=False)[0]
    return curve_functions.optimum_reparam_curve(
        q_source, q_target, lam=0.0, method="DP"
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
