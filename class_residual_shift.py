"""Shared, side-effect-free class residual temporal-shift estimator."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ResidualShiftCandidate:
    residual_shift_days: int
    score: float
    common_support_days: int


@dataclass(frozen=True)
class ClassResidualShiftResult:
    global_shift_days: float
    class_residual_shift_days: int
    final_shift_days: float
    score_at_residual_0: float
    best_score: float
    score_gain: float
    boundary_hit: bool
    num_valid_channels: int
    common_support_days: int
    candidates: Tuple[ResidualShiftCandidate, ...]


@dataclass(frozen=True)
class ClassResidualShiftDecision:
    accepted: bool
    accepted_residual_shift: int
    final_target_to_source_shift: float
    final_source_to_target_shift: float
    boundary_hit: bool
    fallback_reason: str


def _robust_temporal_normalize(prototype: np.ndarray, eps: float = 1e-8):
    prototype = np.asarray(prototype, dtype=np.float64)
    if prototype.ndim != 2 or prototype.shape[0] < 2:
        raise ValueError("prototype must have shape [T,D] with T >= 2")
    median = np.median(prototype, axis=0)
    q25, q75 = np.quantile(prototype, (0.25, 0.75), axis=0)
    iqr = q75 - q25
    return (prototype - median) / (iqr + eps), iqr


def _common_support_score(source, target, total_shift_days, valid_channels):
    length = source.shape[0]
    source_days = np.arange(length, dtype=np.float64)
    target_days = source_days - float(total_shift_days)
    support = (target_days >= 0.0) & (target_days <= float(length - 1))
    source_index = np.flatnonzero(support)
    if source_index.size < 2 or not np.any(valid_channels):
        return float("nan"), int(source_index.size)
    query = target_days[support]
    source_part = source[source_index][:, valid_channels]
    target_part = np.column_stack([
        np.interp(query, source_days, target[:, channel])
        for channel in np.flatnonzero(valid_channels)
    ])
    source_centered = source_part - source_part.mean(axis=0, keepdims=True)
    target_centered = target_part - target_part.mean(axis=0, keepdims=True)
    numerator = np.sum(source_centered * target_centered, axis=0)
    denominator = np.sqrt(
        np.sum(source_centered**2, axis=0) * np.sum(target_centered**2, axis=0)
    )
    correlations = np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan),
        where=denominator > np.finfo(np.float64).eps,
    )
    finite = correlations[np.isfinite(correlations)]
    return (float(finite.mean()) if finite.size else float("nan"), int(source_index.size))


def estimate_class_residual_shift(
    source_prototype: np.ndarray,
    target_prototype: np.ndarray,
    global_shift_days: float,
    max_residual_days: int = 20,
    iqr_floor: float = 1e-6,
) -> ClassResidualShiftResult:
    source, source_iqr = _robust_temporal_normalize(source_prototype)
    target, target_iqr = _robust_temporal_normalize(target_prototype)
    if source.shape != target.shape:
        raise ValueError("source and target prototypes must have identical [T,D] shape")
    if max_residual_days < 0:
        raise ValueError("max_residual_days must be non-negative")
    valid = (
        np.isfinite(source).all(axis=0) & np.isfinite(target).all(axis=0)
        & np.isfinite(source_iqr) & np.isfinite(target_iqr)
        & (source_iqr > iqr_floor) & (target_iqr > iqr_floor)
    )
    if not np.any(valid):
        raise ValueError("no non-flat finite channels for class residual estimation")
    candidates = []
    for residual in range(-int(max_residual_days), int(max_residual_days) + 1):
        score, support_days = _common_support_score(
            source, target, float(global_shift_days) + residual, valid
        )
        candidates.append(ResidualShiftCandidate(residual, score, support_days))
    finite = [candidate for candidate in candidates if np.isfinite(candidate.score)]
    if not finite:
        raise ValueError("no finite class residual candidate scores")
    best = min(
        finite,
        key=lambda item: (-item.score, abs(item.residual_shift_days), item.residual_shift_days),
    )
    zero = candidates[int(max_residual_days)]
    return ClassResidualShiftResult(
        global_shift_days=float(global_shift_days),
        class_residual_shift_days=best.residual_shift_days,
        final_shift_days=float(global_shift_days) + best.residual_shift_days,
        score_at_residual_0=zero.score,
        best_score=best.score,
        score_gain=best.score - zero.score,
        boundary_hit=abs(best.residual_shift_days) == int(max_residual_days),
        num_valid_channels=int(valid.sum()),
        common_support_days=best.common_support_days,
        candidates=tuple(candidates),
    )


def decide_class_residual_shift(result, pseudo_count, min_samples=32, min_gain=0.005):
    reason = ""
    if int(pseudo_count) < int(min_samples):
        reason = "insufficient_samples"
    elif not np.isfinite(result.score_gain) or result.score_gain < float(min_gain):
        reason = "insufficient_gain"
    residual = 0 if reason else int(result.class_residual_shift_days)
    final = float(result.global_shift_days) + residual
    return ClassResidualShiftDecision(
        accepted=not reason,
        accepted_residual_shift=residual,
        final_target_to_source_shift=final,
        final_source_to_target_shift=-final,
        boundary_hit=(not reason and abs(residual) == abs(result.class_residual_shift_days)
                      and result.boundary_hit),
        fallback_reason=reason,
    )
