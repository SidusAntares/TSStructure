"""ORACLE-only raw-NDVI class discrepancy and shared domain-style diagnostics.

This module deliberately uses true target class labels.  Its outputs are
post-hoc diagnostics and are not deployable unsupervised domain adaptation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .decomposition_diagnostics import (
    COMPONENT_LABELS,
    TAU_FAST,
    TAU_SLOW,
    decompose_ndvi_series,
)
from .raw_timeseries import DOMAIN_COLORS, collect_ndvi_diagnostic_parcels


EPS = 1e-8
HUBER_C = 1.345
HUBER_MAX_ITER = 20
HUBER_TOL = 1e-8
PHASE_MAD_REFERENCE_DAYS = 14.0
MIN_STYLE_CLASSES = 5
ORACLE_NOTICE = "ORACLE DIAGNOSTIC — uses target labels; not deployable UDA"


@dataclass(frozen=True)
class CanonicalParcelRecord:
    domain: str
    class_name: str
    parcel_index: object
    canonical_doys: np.ndarray
    original_h: np.ndarray
    trend_t: np.ndarray
    structure_s: np.ndarray
    valid_h: np.ndarray
    valid_t: np.ndarray
    valid_s: np.ndarray
    original_doys: np.ndarray
    original_valid: np.ndarray


@dataclass(frozen=True)
class DomainStyleConfig:
    source_domain: str
    target_domain: str
    samples_per_group: int = 100
    sample_seed: int = 1
    bootstrap_repeats: int = 200
    canonical_grid_size: int = 128
    min_class_samples: int = 20
    min_common_support: float = 0.65
    min_bootstrap_valid_rate: float = 0.80
    peak_search_start: float = 45.0
    peak_search_end: float = 330.0
    min_peak_prominence_ratio: float = 0.15
    max_shift_days: float = 90.0
    shift_refine_radius_days: float = 14.0
    max_interpolation_gap_days: float = 60.0
    min_relative_phase_gain: float = 0.02
    style_lambdas: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)

    def __post_init__(self) -> None:
        if self.source_domain == self.target_domain:
            raise ValueError("source_domain and target_domain must be different")
        for name in ("samples_per_group", "bootstrap_repeats", "canonical_grid_size", "min_class_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "min_common_support", "min_bootstrap_valid_rate",
            "min_peak_prominence_ratio", "min_relative_phase_gain",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if not (1.0 <= self.peak_search_start < self.peak_search_end <= 365.0):
            raise ValueError("peak search range must satisfy 1 <= start < end <= 365")
        for name in ("max_shift_days", "shift_refine_radius_days", "max_interpolation_gap_days"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        lambdas = tuple(sorted({0.0, *(float(value) for value in self.style_lambdas)}))
        if not lambdas or any(not math.isfinite(value) or value < 0 for value in lambdas):
            raise ValueError("style_lambdas must be finite nonnegative values")
        object.__setattr__(self, "style_lambdas", lambdas)


def stable_class_seed(source: str, target: str, class_name: str, seed: int) -> int:
    payload = f"{source}|{target}|{class_name}|{int(seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def interpolate_canonical_curve(
    doys: np.ndarray,
    values: np.ndarray,
    valid_mask: np.ndarray,
    canonical_doys: np.ndarray,
    max_gap_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate only within observed hull and acceptable gaps."""

    doys = np.asarray(doys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    grid = np.asarray(canonical_doys, dtype=np.float64)
    if doys.ndim != 1 or values.shape != doys.shape or valid_mask.shape != doys.shape:
        raise ValueError("doys, values, and valid_mask must be equal-length 1-D arrays")
    keep = valid_mask & np.isfinite(doys) & np.isfinite(values)
    observed_doys = doys[keep]
    observed_values = values[keep]
    order = np.argsort(observed_doys, kind="stable")
    observed_doys, observed_values = observed_doys[order], observed_values[order]
    unique, unique_indices = np.unique(observed_doys, return_index=True)
    observed_doys, observed_values = unique, observed_values[unique_indices]
    result = np.full(grid.shape, np.nan, dtype=np.float64)
    result_valid = np.zeros(grid.shape, dtype=bool)
    if observed_doys.size < 2:
        return result, result_valid
    inside = (grid >= observed_doys[0]) & (grid <= observed_doys[-1]) & np.isfinite(grid)
    right = np.searchsorted(observed_doys, grid, side="left")
    exact = inside & (right < observed_doys.size) & np.isclose(observed_doys[np.minimum(right, observed_doys.size - 1)], grid)
    bracket_right = np.clip(right, 1, observed_doys.size - 1)
    gaps = observed_doys[bracket_right] - observed_doys[bracket_right - 1]
    allowed = inside & ((gaps <= float(max_gap_days)) | exact)
    result[allowed] = np.interp(grid[allowed], observed_doys, observed_values)
    result_valid[allowed] = True
    return result, result_valid


def canonicalize_parcel(
    parcel: Mapping[str, object], canonical_doys: np.ndarray, max_gap_days: float,
    decomposition_fn: Callable[..., Mapping[str, object]] = decompose_ndvi_series,
) -> CanonicalParcelRecord:
    decomposition = decomposition_fn(parcel["ndvi"], parcel["doys"], parcel["valid"])
    canonical: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for source_name, short_name in (("original", "h"), ("trend", "t"), ("structure", "s")):
        canonical[short_name], masks[short_name] = interpolate_canonical_curve(
            np.asarray(decomposition["doys"]), np.asarray(decomposition[source_name]),
            np.asarray(decomposition["valid"]), canonical_doys, max_gap_days,
        )
    return CanonicalParcelRecord(
        str(parcel["domain"]), str(parcel["class_name"]), parcel["parcel_index"],
        np.asarray(canonical_doys, dtype=np.float64).copy(),
        canonical["h"], canonical["t"], canonical["s"],
        masks["h"], masks["t"], masks["s"],
        np.asarray(decomposition["doys"], dtype=np.float64).copy(),
        np.asarray(decomposition["valid"], dtype=bool).copy(),
    )


def robust_pointwise_location(
    curves: np.ndarray, valid_masks: np.ndarray, c: float = HUBER_C,
    max_iter: int = HUBER_MAX_ITER, tol: float = HUBER_TOL,
) -> dict[str, np.ndarray]:
    curves = np.asarray(curves, dtype=np.float64)
    valid_masks = np.asarray(valid_masks, dtype=bool)
    if curves.ndim != 2 or valid_masks.shape != curves.shape:
        raise ValueError("curves and valid_masks must have shape [N,K]")
    valid_masks = valid_masks & np.isfinite(curves)
    shape = (curves.shape[1],)
    outputs = {name: np.full(shape, np.nan) for name in ("mean", "robust", "median", "q25", "q75")}
    outputs["n_valid"] = valid_masks.sum(axis=0).astype(np.int64)
    for index in range(curves.shape[1]):
        values = curves[valid_masks[:, index], index]
        if values.size < 3:
            continue
        median = float(np.median(values))
        outputs["mean"][index] = float(np.mean(values))
        outputs["median"][index] = median
        outputs["q25"][index], outputs["q75"][index] = np.quantile(values, [0.25, 0.75])
        mad = float(np.median(np.abs(values - median)))
        if mad <= EPS:
            outputs["robust"][index] = median
            continue
        location, scale = median, 1.4826 * mad + EPS
        for _ in range(max_iter):
            residual = np.abs(values - location) / scale
            weights = np.where(residual <= c, 1.0, c / np.maximum(residual, EPS))
            updated = float(np.sum(weights * values) / np.sum(weights))
            if abs(updated - location) <= tol:
                location = updated
                break
            location = updated
        outputs["robust"][index] = location
    return outputs


def detect_main_peak(
    center: np.ndarray, canonical_doys: np.ndarray,
    peak_search_start: float = 45.0, peak_search_end: float = 330.0,
    min_peak_prominence_ratio: float = 0.15,
) -> dict[str, object]:
    values = np.asarray(center, dtype=np.float64)
    doys = np.asarray(canonical_doys, dtype=np.float64)
    search = np.isfinite(values) & np.isfinite(doys) & (doys >= peak_search_start) & (doys <= peak_search_end)
    observed = values[search]
    if observed.size < 3:
        return {"valid": False, "reason": "insufficient_peak_support", "peak_doy": np.nan, "prominence_ratio": np.nan}
    dynamic_range = float(np.quantile(observed, 0.95) - np.quantile(observed, 0.05))
    if dynamic_range < 0.03:
        return {"valid": False, "reason": "insufficient_dynamic_range", "peak_doy": np.nan, "prominence_ratio": 0.0}
    candidates: list[tuple[float, float, int]] = []
    valid_indices = np.flatnonzero(search)
    for indices in np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) > 1) + 1):
        position = 0
        while position < len(indices):
            start = position
            while (
                position + 1 < len(indices)
                and indices[position + 1] == indices[position] + 1
                and np.isclose(values[indices[position + 1]], values[indices[start]], atol=1e-12)
            ):
                position += 1
            end = position
            if start > 0 and end + 1 < len(indices):
                left_index, right_index = indices[start - 1], indices[end + 1]
                height = values[indices[start]]
                if height >= values[left_index] and height >= values[right_index]:
                    plateau_midpoint = float((doys[indices[start]] + doys[indices[end]]) / 2.0)
                    left_min = float(np.min(values[indices[:start + 1]]))
                    right_min = float(np.min(values[indices[end:]]))
                    prominence = float(height - max(left_min, right_min))
                    candidates.append((prominence, -abs(plateau_midpoint - 182.5), plateau_midpoint))
            position += 1
    if not candidates:
        return {"valid": False, "reason": "no_local_peak", "peak_doy": np.nan, "prominence_ratio": 0.0}
    prominence, _, peak_doy = max(candidates)
    ratio = prominence / (dynamic_range + EPS)
    if min(peak_doy - peak_search_start, peak_search_end - peak_doy) <= 15.0:
        reason = "peak_near_search_boundary"
    elif ratio < min_peak_prominence_ratio:
        reason = "insufficient_peak_prominence"
    else:
        reason = ""
    return {"valid": not reason, "reason": reason, "peak_doy": peak_doy, "prominence": prominence, "prominence_ratio": ratio, "dynamic_range": dynamic_range}


# Descriptive alias retained for callers that refer to the T-only role.
detect_trend_peak = detect_main_peak


def _shift_curve(values: np.ndarray, doys: np.ndarray, shift_days: float) -> tuple[np.ndarray, np.ndarray]:
    values, doys = np.asarray(values, dtype=np.float64), np.asarray(doys, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(doys)
    result = np.full(values.shape, np.nan)
    if not finite.any():
        return result, np.zeros(values.shape, bool)
    query = doys - shift_days
    support = np.zeros(values.shape, bool)
    finite_indices = np.flatnonzero(finite)
    splits = np.flatnonzero(np.diff(finite_indices) > 1) + 1
    for segment in np.split(finite_indices, splits):
        if segment.size == 1:
            exact = np.isclose(query, doys[segment[0]])
            result[exact] = values[segment[0]]
            support[exact] = True
            continue
        inside = (query >= doys[segment[0]]) & (query <= doys[segment[-1]])
        result[inside] = np.interp(query[inside], doys[segment], values[segment])
        support[inside] = True
    return result, support


def _shape_objective(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 3:
        return np.inf
    first, second = source[mask], target[mask]
    first = first - first.mean()
    second = second - second.mean()
    first /= np.sqrt(np.mean(first * first)) + EPS
    second /= np.sqrt(np.mean(second * second)) + EPS
    return float(np.mean(np.square(first - second)))


def _search_trend_shift(
    source_trend: np.ndarray,
    target_trend: np.ndarray,
    canonical_doys: np.ndarray,
    candidate_shifts: Sequence[float],
    min_common_support: float,
) -> dict[str, object]:
    """Evaluate the complete T-only objective curve over candidate shifts."""

    source = np.asarray(source_trend, dtype=np.float64)
    shifts = np.asarray(candidate_shifts, dtype=np.float64)
    objectives = np.full(shifts.shape, np.nan, dtype=np.float64)
    supports = np.full(shifts.shape, np.nan, dtype=np.float64)
    aligned_curves: list[np.ndarray] = []
    common_masks: list[np.ndarray] = []
    for index, shift in enumerate(shifts):
        aligned, aligned_valid = _shift_curve(
            target_trend, canonical_doys, float(shift)
        )
        common = np.isfinite(source) & aligned_valid & np.isfinite(aligned)
        support = float(np.mean(common))
        supports[index] = support
        if support >= min_common_support:
            objective = _shape_objective(source, aligned, common)
            if np.isfinite(objective):
                objectives[index] = objective
        aligned_curves.append(aligned)
        common_masks.append(common)
    return {
        "candidate_shifts": shifts,
        "objectives": objectives,
        "common_supports": supports,
        "aligned_curves": aligned_curves,
        "common_masks": common_masks,
    }


def _unavailable_phase(
    status: str, reason: str, source_peak: Mapping[str, object],
    target_peak: Mapping[str, object], search_mode: str,
    candidate_shifts: np.ndarray | None = None,
    objectives: np.ndarray | None = None,
) -> dict[str, object]:
    return {
        "phase_available": False,
        "phase_status": status,
        "search_mode": search_mode,
        "valid": False,
        "reason": reason,
        "shift_days": 0.0,
        "initial_shift_days": np.nan,
        "identity_objective": np.nan,
        "best_objective": np.nan,
        "relative_gain": np.nan,
        "second_best_objective": np.nan,
        "objective_margin": np.nan,
        "shift_at_boundary": False,
        "common_support": np.nan,
        "target_aligned": None,
        "common_mask": None,
        "source_peak": source_peak,
        "target_peak": target_peak,
        "source_peak_valid": bool(source_peak.get("valid", False)),
        "target_peak_valid": bool(target_peak.get("valid", False)),
        "source_peak_reason": str(source_peak.get("reason", "")),
        "target_peak_reason": str(target_peak.get("reason", "")),
        "candidate_shifts": (
            np.asarray(candidate_shifts, dtype=np.float64)
            if candidate_shifts is not None else np.empty(0, dtype=np.float64)
        ),
        "objective_curve": (
            np.asarray(objectives, dtype=np.float64)
            if objectives is not None else np.empty(0, dtype=np.float64)
        ),
    }


def estimate_phase_shift(
    source_trend: np.ndarray, target_trend: np.ndarray, canonical_doys: np.ndarray,
    peak_search_start: float = 45.0, peak_search_end: float = 330.0,
    min_peak_prominence_ratio: float = 0.15, max_shift_days: float = 90.0,
    shift_refine_radius_days: float = 14.0, min_common_support: float = 0.65,
    min_relative_phase_gain: float = 0.02,
) -> dict[str, object]:
    source = np.asarray(source_trend, dtype=np.float64)
    target = np.asarray(target_trend, dtype=np.float64)
    doys = np.asarray(canonical_doys, dtype=np.float64)
    if source.shape != target.shape or source.shape != doys.shape or source.ndim != 1:
        invalid_peak = {
            "valid": False, "reason": "invalid_samples",
            "peak_doy": np.nan, "prominence_ratio": np.nan,
        }
        return _unavailable_phase(
            "invalid_samples", "invalid_samples", invalid_peak,
            invalid_peak.copy(), "not_searched",
        )
    source_peak = detect_main_peak(
        source, doys, peak_search_start, peak_search_end,
        min_peak_prominence_ratio,
    )
    target_peak = detect_main_peak(
        target, doys, peak_search_start, peak_search_end,
        min_peak_prominence_ratio,
    )
    differences = np.diff(doys)
    if (
        doys.size < 2 or not np.isfinite(doys).all()
        or not np.isfinite(differences).all() or np.any(differences <= 0)
    ):
        return _unavailable_phase(
            "invalid_grid", "invalid_canonical_grid", source_peak, target_peak,
            "not_searched",
        )
    if min(np.isfinite(source).sum(), np.isfinite(target).sum()) < 3:
        return _unavailable_phase(
            "invalid_samples", "insufficient_trend_samples",
            source_peak, target_peak, "not_searched",
        )
    source_range = float(np.quantile(source[np.isfinite(source)], 0.95) - np.quantile(source[np.isfinite(source)], 0.05))
    target_range = float(np.quantile(target[np.isfinite(target)], 0.95) - np.quantile(target[np.isfinite(target)], 0.05))
    if min(source_range, target_range) < 0.03:
        return _unavailable_phase(
            "invalid_flat_trend", "flat_trend", source_peak, target_peak,
            "not_searched",
        )

    grid_step = float(np.median(differences))
    peak_guided = bool(source_peak["valid"] and target_peak["valid"])
    initial = (
        float(source_peak["peak_doy"] - target_peak["peak_doy"])
        if peak_guided else np.nan
    )
    if peak_guided:
        search_center = float(np.clip(initial, -max_shift_days, max_shift_days))
        lower = max(-max_shift_days, search_center - shift_refine_radius_days)
        upper = min(max_shift_days, search_center + shift_refine_radius_days)
        offsets = np.arange(
            math.ceil((lower - search_center) / grid_step),
            math.floor((upper - search_center) / grid_step) + 1,
            dtype=np.float64,
        )
        candidates = search_center + offsets * grid_step
        search_mode = "peak_guided"
    else:
        steps = np.arange(
            math.ceil(-max_shift_days / grid_step),
            math.floor(max_shift_days / grid_step) + 1,
            dtype=np.float64,
        )
        candidates = steps * grid_step
        search_mode = "full_trend_search"
    candidates = np.unique(np.concatenate([candidates, np.array([0.0])]))
    search = _search_trend_shift(
        source, target, doys, candidates, min_common_support,
    )
    objectives = np.asarray(search["objectives"], dtype=np.float64)
    available_indices = np.flatnonzero(np.isfinite(objectives))
    if available_indices.size == 0:
        return _unavailable_phase(
            "invalid_common_support", "insufficient_common_support",
            source_peak, target_peak, search_mode, candidates, objectives,
        )
    best_index = min(
        available_indices,
        key=lambda index: (
            objectives[index], abs(float(candidates[index])),
            float(candidates[index]),
        ),
    )
    best_shift = float(candidates[best_index])
    best_objective = float(objectives[best_index])
    zero_index = int(np.flatnonzero(np.isclose(candidates, 0.0))[0])
    identity_objective = float(objectives[zero_index]) if np.isfinite(objectives[zero_index]) else np.nan
    relative_gain = (
        (identity_objective - best_objective) / (identity_objective + EPS)
        if np.isfinite(identity_objective) else np.nan
    )
    if np.isfinite(relative_gain) and relative_gain < min_relative_phase_gain:
        final_index = zero_index
        phase_status = "valid_identity"
    else:
        final_index = best_index
        phase_status = (
            "valid_identity" if np.isclose(candidates[best_index], 0.0)
            else "valid_nonidentity"
        )
    final_shift = float(candidates[final_index])
    separated = [
        index for index in available_indices
        if abs(float(candidates[index]) - best_shift) >= shift_refine_radius_days
    ]
    second_best = min((float(objectives[index]) for index in separated), default=np.nan)
    denominator = identity_objective - best_objective
    objective_margin = (
        (second_best - best_objective) / (denominator + EPS)
        if np.isfinite(second_best) and np.isfinite(identity_objective) else np.nan
    )
    return {
        "phase_available": True,
        "phase_status": phase_status,
        "search_mode": search_mode,
        "valid": True,
        "reason": "",
        "shift_days": final_shift,
        "initial_shift_days": initial,
        "identity_objective": identity_objective,
        "best_objective": best_objective,
        "objective": float(objectives[final_index]),
        "relative_gain": relative_gain,
        "second_best_objective": second_best,
        "objective_margin": objective_margin,
        "shift_at_boundary": bool(np.isclose(abs(final_shift), max_shift_days, atol=grid_step / 2.0)),
        "common_support": float(search["common_supports"][final_index]),
        "target_aligned": search["aligned_curves"][final_index],
        "common_mask": search["common_masks"][final_index],
        "source_peak": source_peak,
        "target_peak": target_peak,
        "source_peak_valid": bool(source_peak.get("valid", False)),
        "target_peak_valid": bool(target_peak.get("valid", False)),
        "source_peak_reason": str(source_peak.get("reason", "")),
        "target_peak_reason": str(target_peak.get("reason", "")),
        "candidate_shifts": candidates,
        "objective_curve": objectives,
    }


def estimate_class_phase(
    source_trend: np.ndarray, target_trend: np.ndarray,
    source_structure: np.ndarray, target_structure: np.ndarray,
    canonical_doys: np.ndarray, **kwargs,
) -> dict[str, object]:
    result = estimate_phase_shift(source_trend, target_trend, canonical_doys, **kwargs)
    source_s = detect_main_peak(source_structure, canonical_doys, kwargs.get("peak_search_start", 45.0), kwargs.get("peak_search_end", 330.0), kwargs.get("min_peak_prominence_ratio", 0.15))
    target_s = detect_main_peak(target_structure, canonical_doys, kwargs.get("peak_search_start", 45.0), kwargs.get("peak_search_end", 330.0), kwargs.get("min_peak_prominence_ratio", 0.15))
    result.update({
        "structure_source_peak_valid": source_s["valid"],
        "structure_source_peak_doy": source_s["peak_doy"],
        "structure_source_peak_prominence_ratio": source_s.get("prominence_ratio", np.nan),
        "structure_target_peak_valid": target_s["valid"],
        "structure_target_peak_doy": target_s["peak_doy"],
        "structure_target_peak_prominence_ratio": target_s.get("prominence_ratio", np.nan),
    })
    return result


def support_weighted_rmse(first: np.ndarray, second: np.ndarray, valid: np.ndarray | None = None) -> float:
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    mask = np.isfinite(first) & np.isfinite(second)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    return float(np.sqrt(np.mean(np.square(first[mask] - second[mask])))) if mask.any() else np.nan


def style_explained_fraction(delta: np.ndarray, style: np.ndarray, valid: np.ndarray | None = None) -> float:
    delta, style = np.asarray(delta, dtype=np.float64), np.asarray(style, dtype=np.float64)
    mask = np.isfinite(delta) & np.isfinite(style)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    if not mask.any():
        return np.nan
    return float(1.0 - np.sum(np.square(delta[mask] - style[mask])) / (np.sum(np.square(delta[mask])) + EPS))


def fit_robust_domain_style(
    classes: Sequence[str], deltas: Mapping[str, np.ndarray],
    valid_masks: Mapping[str, np.ndarray], base_reliability: Mapping[str, float],
    min_classes: int = MIN_STYLE_CLASSES,
) -> dict[str, object]:
    classes = [name for name in classes if name in deltas and name in valid_masks]
    if len(classes) < min_classes:
        size = len(next(iter(deltas.values()))) if deltas else 0
        return {"valid": False, "reason": "insufficient_style_classes", "style": np.full(size, np.nan), "classes_used": classes, "consensus": {name: 0.0 for name in classes}, "final_weights": {name: 0.0 for name in classes}}
    arrays = {name: np.asarray(deltas[name], dtype=np.float64) for name in classes}
    masks = {name: np.asarray(valid_masks[name], dtype=bool) & np.isfinite(arrays[name]) for name in classes}
    base = {name: max(0.0, float(base_reliability[name])) for name in classes}

    def pointwise(weights: Mapping[str, float]) -> np.ndarray:
        style = np.full_like(next(iter(arrays.values())), np.nan)
        for index in range(style.size):
            present = [name for name in classes if masks[name][index] and weights[name] > 0]
            if len(present) >= min_classes:
                denominator = sum(weights[name] for name in present)
                style[index] = sum(weights[name] * arrays[name][index] for name in present) / denominator
        return style

    weights = base.copy()
    consensus = {name: 1.0 for name in classes}
    style = pointwise(weights)
    for _ in range(HUBER_MAX_ITER):
        distances = {}
        for name in classes:
            common = masks[name] & np.isfinite(style)
            distances[name] = support_weighted_rmse(arrays[name], style, common)
        finite = np.array([value for value in distances.values() if np.isfinite(value)])
        if finite.size == 0:
            break
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        scale = 1.4826 * mad + EPS
        for name in classes:
            z = max(0.0, (distances[name] - median) / scale) if np.isfinite(distances[name]) else np.inf
            consensus[name] = 1.0 if z <= HUBER_C else HUBER_C / z
        updated_weights = {name: base[name] * consensus[name] for name in classes}
        updated = pointwise(updated_weights)
        common = np.isfinite(style) & np.isfinite(updated)
        weights = updated_weights
        relative_change = (
            np.max(np.abs(updated[common] - style[common]))
            / (np.max(np.abs(style[common])) + EPS)
            if common.any() else np.inf
        )
        if relative_change <= HUBER_TOL:
            style = updated
            break
        style = updated
    total = sum(weights.values())
    normalized = {name: weights[name] / total if total > 0 else 0.0 for name in classes}
    return {"valid": bool(np.isfinite(style).any()), "reason": "", "style": style, "classes_used": classes, "consensus": consensus, "final_weights": normalized}


def compute_loco_domain_styles(
    evaluation_classes: Sequence[str], contributor_classes: Sequence[str],
    deltas: Mapping[str, np.ndarray],
    valid_masks: Mapping[str, np.ndarray], base_reliability: Mapping[str, float],
    min_classes: int = MIN_STYLE_CLASSES,
    base_reliability_by_held: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict[str, object]]:
    contributors = list(dict.fromkeys(contributor_classes))
    result: dict[str, dict[str, object]] = {}
    template = np.asarray(next(iter(deltas.values())), dtype=np.float64)
    for held in dict.fromkeys(evaluation_classes):
        remaining = [name for name in contributors if name != held]
        if len(remaining) < min_classes:
            result[held] = {
                "valid": False, "reason": "insufficient_loco_classes",
                "style": np.full_like(template, np.nan),
                "classes_used": remaining, "consensus": {},
                "final_weights": {},
            }
            continue
        held_reliability = (
            base_reliability_by_held.get(held, base_reliability)
            if base_reliability_by_held is not None else base_reliability
        )
        result[held] = fit_robust_domain_style(
            remaining, deltas, valid_masks, held_reliability,
            min_classes=min_classes,
        )
    return result


def apply_shared_style(components: Mapping[str, np.ndarray], style: np.ndarray, style_lambda: float) -> dict[str, np.ndarray]:
    adjustment = (
        np.zeros_like(np.asarray(style, dtype=np.float64))
        if float(style_lambda) == 0.0
        else float(style_lambda) * np.asarray(style, dtype=np.float64)
    )
    return {name: np.asarray(components[name], dtype=np.float64) + adjustment for name in ("original", "trend", "structure")}


def physical_violation_fraction(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    return float(np.mean((values[finite] < -1.0) | (values[finite] > 1.0))) if finite.any() else np.nan


def hierarchy_max_errors(original: Mapping[str, np.ndarray], styled: Mapping[str, np.ndarray]) -> dict[str, float]:
    dynamics_before = np.asarray(original["structure"]) - np.asarray(original["trend"])
    dynamics_after = np.asarray(styled["structure"]) - np.asarray(styled["trend"])
    residual_before = np.asarray(original["original"]) - np.asarray(original["structure"])
    residual_after = np.asarray(styled["original"]) - np.asarray(styled["structure"])
    return {"max_dynamics_error": float(np.nanmax(np.abs(dynamics_after - dynamics_before))), "max_residual_error": float(np.nanmax(np.abs(residual_after - residual_before)))}


def _pair_distance(first: np.ndarray, first_valid: np.ndarray, second: np.ndarray, second_valid: np.ndarray, min_common_support: float) -> float:
    common = first_valid & second_valid & np.isfinite(first) & np.isfinite(second)
    if np.mean(common) < min_common_support:
        return np.nan
    return float(np.sqrt(np.mean(np.square(first[common] - second[common]))))


def support_weighted_energy_distance(
    source: np.ndarray, source_valid: np.ndarray, target: np.ndarray,
    target_valid: np.ndarray, min_common_support: float,
) -> float:
    source, target = np.asarray(source, float), np.asarray(target, float)
    source_valid, target_valid = np.asarray(source_valid, bool), np.asarray(target_valid, bool)
    cross = [_pair_distance(a, am, b, bm, min_common_support) for a, am in zip(source, source_valid) for b, bm in zip(target, target_valid)]
    within_s = [_pair_distance(a, am, b, bm, min_common_support) for a, am in zip(source, source_valid) for b, bm in zip(source, source_valid)]
    within_t = [_pair_distance(a, am, b, bm, min_common_support) for a, am in zip(target, target_valid) for b, bm in zip(target, target_valid)]
    means = [np.mean([x for x in group if np.isfinite(x)]) if any(np.isfinite(x) for x in group) else np.nan for group in (cross, within_s, within_t)]
    if not np.isfinite(means).all():
        return np.nan
    return float(max(0.0, 2.0 * means[0] - means[1] - means[2]))


def _component_arrays(
    records: Sequence[CanonicalParcelRecord], component: str,
) -> tuple[np.ndarray, np.ndarray]:
    value_name = {"original": "original_h", "trend": "trend_t", "structure": "structure_s"}[component]
    valid_name = {"original": "valid_h", "trend": "valid_t", "structure": "valid_s"}[component]
    return (
        np.stack([getattr(record, value_name) for record in records]),
        np.stack([getattr(record, valid_name) for record in records]),
    )


def _record_center(records: Sequence[CanonicalParcelRecord], component: str) -> dict[str, np.ndarray]:
    values, valid = _component_arrays(records, component)
    return robust_pointwise_location(values, valid)


def _align_array(values: np.ndarray, valid: np.ndarray, doys: np.ndarray, shift: float) -> tuple[np.ndarray, np.ndarray]:
    safe = np.where(np.asarray(valid, bool), np.asarray(values, float), np.nan)
    return _shift_curve(safe, doys, shift)


def _align_records(
    records: Sequence[CanonicalParcelRecord], component: str, shift: float,
) -> tuple[np.ndarray, np.ndarray]:
    arrays, masks = _component_arrays(records, component)
    aligned, valid = [], []
    for array, mask, record in zip(arrays, masks, records):
        shifted, shifted_valid = _align_array(array, mask, record.canonical_doys, shift)
        aligned.append(shifted)
        valid.append(shifted_valid)
    return np.stack(aligned), np.stack(valid)


def bootstrap_phase_discrepancy(
    source_records: Sequence[CanonicalParcelRecord],
    target_records: Sequence[CanonicalParcelRecord], config: DomainStyleConfig,
    class_name: str,
) -> dict[str, object]:
    """Bootstrap cached canonical records without rereading or decomposing."""

    rng = np.random.default_rng(stable_class_seed(
        config.source_domain, config.target_domain, class_name, config.sample_seed
    ))
    grid = source_records[0].canonical_doys
    shifts: list[float] = []
    deltas: list[np.ndarray] = []
    relative_gains: list[float] = []
    objective_margins: list[float] = []
    identity_count = 0
    nonidentity_count = 0
    failure_counts = {"flat": 0, "support": 0, "grid": 0, "samples": 0}
    for _ in range(config.bootstrap_repeats):
        source_sample = [source_records[index] for index in rng.integers(0, len(source_records), len(source_records))]
        target_sample = [target_records[index] for index in rng.integers(0, len(target_records), len(target_records))]
        source_t = _record_center(source_sample, "trend")["robust"]
        target_t = _record_center(target_sample, "trend")["robust"]
        phase = estimate_phase_shift(
            source_t, target_t, grid,
            config.peak_search_start, config.peak_search_end,
            config.min_peak_prominence_ratio, config.max_shift_days,
            config.shift_refine_radius_days, config.min_common_support,
            config.min_relative_phase_gain,
        )
        if not phase["phase_available"]:
            category = {
                "invalid_flat_trend": "flat",
                "invalid_common_support": "support",
                "invalid_grid": "grid",
                "invalid_samples": "samples",
            }.get(str(phase["phase_status"]), "samples")
            failure_counts[category] += 1
            continue
        aligned = np.asarray(phase["target_aligned"], float)
        delta = aligned - source_t
        delta[~np.asarray(phase["common_mask"], bool)] = np.nan
        shifts.append(float(phase["shift_days"]))
        deltas.append(delta)
        identity_count += int(phase["phase_status"] == "valid_identity")
        nonidentity_count += int(phase["phase_status"] == "valid_nonidentity")
        if np.isfinite(phase["relative_gain"]):
            relative_gains.append(float(phase["relative_gain"]))
        if np.isfinite(phase["objective_margin"]):
            objective_margins.append(float(phase["objective_margin"]))
    success_rate = len(shifts) / config.bootstrap_repeats
    common_statistics = {
        "valid_rate": success_rate,
        "bootstrap_search_success_rate": success_rate,
        "bootstrap_identity_rate": identity_count / config.bootstrap_repeats,
        "bootstrap_nonidentity_rate": nonidentity_count / config.bootstrap_repeats,
        "bootstrap_failure_flat_count": failure_counts["flat"],
        "bootstrap_failure_support_count": failure_counts["support"],
        "bootstrap_failure_grid_count": failure_counts["grid"],
        "bootstrap_failure_samples_count": failure_counts["samples"],
        "bootstrap_relative_gain_median": (
            float(np.median(relative_gains)) if relative_gains else np.nan
        ),
        "bootstrap_objective_margin_median": (
            float(np.median(objective_margins)) if objective_margins else np.nan
        ),
    }
    if not shifts:
        return {**common_statistics,
            "shift_median": np.nan, "shift_mad": np.nan,
            "shift_q25": np.nan, "shift_q75": np.nan,
            "delta_variance_integral": np.nan, "delta_finite_rate": 0.0,
            "deltas": np.empty((0, len(grid))),
        }
    shift_array = np.asarray(shifts)
    delta_array = np.stack(deltas)
    variance_integral = _bootstrap_variance_integral(delta_array, grid)
    return {**common_statistics,
        "shift_median": float(np.median(shift_array)),
        "shift_mad": float(np.median(np.abs(shift_array - np.median(shift_array)))),
        "shift_q25": float(np.quantile(shift_array, 0.25)),
        "shift_q75": float(np.quantile(shift_array, 0.75)),
        "delta_variance_integral": variance_integral,
        "delta_finite_rate": float(np.mean(np.isfinite(delta_array), axis=0).mean()),
        "deltas": delta_array,
    }


def _empty_bootstrap(grid_size: int) -> dict[str, object]:
    return {
        "valid_rate": 0.0,
        "bootstrap_search_success_rate": 0.0,
        "bootstrap_identity_rate": 0.0,
        "bootstrap_nonidentity_rate": 0.0,
        "bootstrap_failure_flat_count": 0,
        "bootstrap_failure_support_count": 0,
        "bootstrap_failure_grid_count": 0,
        "bootstrap_failure_samples_count": 0,
        "bootstrap_relative_gain_median": np.nan,
        "bootstrap_objective_margin_median": np.nan,
        "shift_median": np.nan, "shift_mad": np.nan,
        "shift_q25": np.nan, "shift_q75": np.nan,
        "delta_variance_integral": np.nan, "delta_finite_rate": np.nan,
        "deltas": np.empty((0, grid_size)),
    }


def _bootstrap_variance_integral(
    delta_array: np.ndarray, grid: np.ndarray,
    support: np.ndarray | None = None,
) -> float:
    delta_array = np.asarray(delta_array, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    variance = np.full(delta_array.shape[1], np.nan)
    for index in range(delta_array.shape[1]):
        if support is not None and not np.asarray(support, dtype=bool)[index]:
            continue
        finite = delta_array[:, index][np.isfinite(delta_array[:, index])]
        if finite.size:
            variance[index] = float(np.var(finite))
    variance_integral = 0.0
    finite_indices = np.flatnonzero(np.isfinite(variance))
    for segment in np.split(
        finite_indices, np.flatnonzero(np.diff(finite_indices) > 1) + 1
    ):
        if segment.size >= 2:
            variance_integral += float(np.trapz(
                variance[segment], grid[segment]
            ))
    return variance_integral if np.isfinite(variance).any() else np.nan


def _weighted_style(
    classes: Sequence[str], deltas: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray], weights: Mapping[str, float],
    min_classes: int = MIN_STYLE_CLASSES,
) -> np.ndarray:
    result = np.full_like(np.asarray(deltas[classes[0]], float), np.nan)
    for index in range(result.size):
        present = [name for name in classes if masks[name][index] and np.isfinite(deltas[name][index]) and weights[name] > 0]
        if len(present) >= min_classes:
            denominator = sum(weights[name] for name in present)
            result[index] = sum(weights[name] * deltas[name][index] for name in present) / denominator
    return result


def iqr_distance(source: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray]) -> float:
    mask = np.isfinite(source["q25"]) & np.isfinite(source["q75"]) & np.isfinite(target["q25"]) & np.isfinite(target["q75"])
    if not mask.any():
        return np.nan
    return float(np.mean((np.abs(source["q25"][mask] - target["q25"][mask]) + np.abs(source["q75"][mask] - target["q75"][mask])) / 2.0))


def nearest_class_margin(
    source_center: np.ndarray, own_target: np.ndarray,
    other_targets: Sequence[np.ndarray], valid: np.ndarray,
) -> float:
    own = support_weighted_rmse(source_center, own_target, valid)
    others = [support_weighted_rmse(source_center, center, valid) for center in other_targets]
    finite = [value for value in others if np.isfinite(value)]
    return float(min(finite) - own) if np.isfinite(own) and finite else np.nan


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _draw_distribution(
    axis: plt.Axes, grid: np.ndarray, curves: np.ndarray, masks: np.ndarray,
    summary: Mapping[str, np.ndarray], color: str, label: str,
) -> None:
    for curve, valid in zip(curves[:50], masks[:50]):
        axis.plot(grid[valid], curve[valid], color=color, alpha=0.10, linewidth=0.7)
    axis.fill_between(grid, summary["q25"], summary["q75"], color=color, alpha=0.18)
    axis.plot(grid, summary["robust"], color=color, linewidth=2.4, label=f"{label} robust")
    axis.plot(grid, summary["mean"], color=color, linewidth=1.8, linestyle="--", label=f"{label} mean")


def _phase_target_label(target: str, phase_available: bool) -> str:
    return (
        f"{target} T-phase aligned" if phase_available
        else f"{target} identity fallback — not aligned"
    )


def _plot_phase_class(
    class_name: str, source: str, target: str, grid: np.ndarray,
    source_data: Mapping[str, tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray]]],
    target_data: Mapping[str, tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray]]],
    phase: Mapping[str, object], bootstrap: Mapping[str, object], n_source: int,
    n_target: int, path: Path,
) -> None:
    phase_available = bool(phase.get("phase_available", phase.get("valid", False)))
    target_label = _phase_target_label(target, phase_available)
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    for axis, component in zip(axes, ("original", "trend", "structure")):
        _draw_distribution(axis, grid, *source_data[component], DOMAIN_COLORS[source], source)
        _draw_distribution(
            axis, grid, *target_data[component], DOMAIN_COLORS[target],
            target_label,
        )
        axis.set_ylabel(COMPONENT_LABELS[component])
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Canonical DOY")
    status = str(phase.get("phase_status", "unavailable"))
    source_t_peak = phase.get("source_peak", {}).get("peak_doy", np.nan)
    target_t_peak = phase.get("target_peak", {}).get("peak_doy", np.nan)
    fig.suptitle(
        f"{ORACLE_NOTICE}\n{source}→{target} | {class_name} | phase_status={status} | "
        f"search={phase.get('search_mode', 'not_searched')} | shift={phase.get('shift_days', 0.0):.1f} d | "
        f"gain={phase.get('relative_gain', np.nan):.3f} | margin={phase.get('objective_margin', np.nan):.3f}\n"
        f"bootstrap success={bootstrap.get('bootstrap_search_success_rate', bootstrap.get('valid_rate', 0.0)):.2f} | "
        f"shift MAD={bootstrap.get('shift_mad', np.nan):.1f} | n={n_source}/{n_target} | "
        f"T peak valid={phase.get('source_peak', {}).get('valid', False)}/"
        f"{phase.get('target_peak', {}).get('valid', False)} | T peaks={source_t_peak:.1f}/{target_t_peak:.1f}; "
        f"S peaks={phase.get('structure_source_peak_doy', np.nan):.1f}/"
        f"{phase.get('structure_target_peak_doy', np.nan):.1f} diagnostic only; never used for phase"
    )
    _save_figure(fig, path)


def _placeholder(path: Path, title: str, reason: str, rows: int = 1) -> None:
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3.5 * rows))
    for axis in np.atleast_1d(axes):
        axis.text(0.5, 0.5, f"ORACLE DIAGNOSTIC\nUnavailable: {reason}", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    fig.suptitle(title)
    _save_figure(fig, path)


def _plot_style_sweep(
    class_name: str, source: str, target: str, grid: np.ndarray,
    source_data: Mapping[str, tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray]]],
    target_data: Mapping[str, tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray]]],
    style: np.ndarray, lambdas: Sequence[float], metric_rows: Sequence[Mapping[str, object]], path: Path,
) -> None:
    fig, axes = plt.subplots(3, len(lambdas), figsize=(5 * len(lambdas), 11), squeeze=False, sharex=True)
    components = ("original", "trend", "structure")
    for column, style_lambda in enumerate(lambdas):
        for row, component in enumerate(components):
            axis = axes[row, column]
            source_values, source_valid, source_summary = source_data[component]
            target_values, target_valid, target_summary = target_data[component]
            adjustment = (
                np.zeros_like(style) if style_lambda == 0.0
                else style_lambda * style
            )
            styled_values = source_values + adjustment[None, :]
            styled_summary = {
                name: (value + adjustment if name != "n_valid" else value)
                for name, value in source_summary.items()
            }
            _draw_distribution(
                axis, grid, styled_values, source_valid, styled_summary,
                DOMAIN_COLORS[source], "styled source",
            )
            _draw_distribution(
                axis, grid, target_values, target_valid, target_summary,
                DOMAIN_COLORS[target], "aligned target",
            )
            axis.set_ylabel(COMPONENT_LABELS[component])
            axis.grid(alpha=0.2)
            matching = [
                item for item in metric_rows
                if item["lambda"] == style_lambda and item["component"] == component
            ]
            prefix = (
                "no domain style (lambda=0)" if style_lambda == 0.0
                else f"with domain style (lambda={style_lambda:g})"
            )
            annotation = f"{prefix} | {COMPONENT_LABELS[component]}"
            if matching:
                metric = matching[0]
                annotation += (
                    f"\nRMSE {metric['center_rmse_before']:.3f}→{metric['center_rmse_after']:.3f}"
                    f" | ED {metric['energy_distance_before']:.3f}→{metric['energy_distance_after']:.3f}"
                    f"\nIQR {metric['iqr_distance_before']:.3f}→{metric['iqr_distance_after']:.3f}"
                    f" | margin {metric['nearest_class_margin_before']:.3f}→"
                    f"{metric['nearest_class_margin_after']:.3f}"
                )
            axis.set_title(annotation, fontsize=9)
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("Canonical DOY")
    fig.suptitle(f"{ORACLE_NOTICE}\n{source}→{target} | {class_name} | LOCO shared T-style sweep")
    _save_figure(fig, path)


def _plot_discrepancy_style(
    class_name: str, source: str, target: str, grid: np.ndarray,
    delta_t: np.ndarray, delta_s: np.ndarray, style: np.ndarray, support: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(grid, delta_t, label="ΔT", linewidth=2)
    axes[0].plot(grid, style, label="LOCO style", linewidth=2)
    axes[1].plot(grid, delta_s, label="ΔS", linewidth=2)
    axes[1].plot(grid, style, label="same LOCO style", linewidth=2)
    axes[2].plot(grid, delta_t - style, label="ΔT-style")
    axes[2].plot(grid, delta_s - style, label="ΔS-style")
    axes[2].fill_between(grid, 0, support.astype(float), alpha=0.12, label="common support")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    axes[-1].set_xlabel("Canonical DOY")
    fig.suptitle(f"{ORACLE_NOTICE}\n{source}→{target} | {class_name} | class discrepancy vs LOCO style")
    _save_figure(fig, path)


def _summarize_aligned(
    records: Sequence[CanonicalParcelRecord], component: str, shift: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    values, valid = _align_records(records, component, shift)
    return values, valid, robust_pointwise_location(values, valid)


def _phase_row(
    class_name: str, source_records: Sequence[CanonicalParcelRecord],
    target_records: Sequence[CanonicalParcelRecord], phase: Mapping[str, object],
    bootstrap: Mapping[str, object], style_contributor: bool,
    style_evaluable: bool, reason: str,
) -> dict[str, object]:
    source_peak = phase.get("source_peak", {})
    target_peak = phase.get("target_peak", {})
    return {
        "source_domain": source_records[0].domain if source_records else "",
        "target_domain": target_records[0].domain if target_records else "",
        "class_name": class_name,
        "n_source": len(source_records), "n_target": len(target_records),
        "source_t_peak_doy": source_peak.get("peak_doy", np.nan),
        "target_t_peak_doy": target_peak.get("peak_doy", np.nan),
        "source_t_peak_prominence_ratio": source_peak.get("prominence_ratio", np.nan),
        "target_t_peak_prominence_ratio": target_peak.get("prominence_ratio", np.nan),
        "source_s_peak_doy_diagnostic": phase.get("structure_source_peak_doy", np.nan),
        "target_s_peak_doy_diagnostic": phase.get("structure_target_peak_doy", np.nan),
        "source_s_peak_prominence_ratio_diagnostic": phase.get(
            "structure_source_peak_prominence_ratio", np.nan
        ),
        "target_s_peak_prominence_ratio_diagnostic": phase.get(
            "structure_target_peak_prominence_ratio", np.nan
        ),
        "trend_source_peak_valid": source_peak.get("valid", False),
        "trend_target_peak_valid": target_peak.get("valid", False),
        "trend_source_peak_reason": source_peak.get("reason", ""),
        "trend_target_peak_reason": target_peak.get("reason", ""),
        "source_peak_valid": phase.get("source_peak_valid", False),
        "target_peak_valid": phase.get("target_peak_valid", False),
        "source_peak_reason": phase.get("source_peak_reason", ""),
        "target_peak_reason": phase.get("target_peak_reason", ""),
        "structure_source_peak_valid": phase.get("structure_source_peak_valid", False),
        "structure_target_peak_valid": phase.get("structure_target_peak_valid", False),
        "initial_shift_days": phase.get("initial_shift_days", np.nan),
        "refined_shift_days": phase.get("shift_days", np.nan),
        "phase_objective": phase.get("objective", np.nan),
        "phase_available": phase.get("phase_available", False),
        "phase_status": phase.get("phase_status", "invalid_samples"),
        "search_mode": phase.get("search_mode", "not_searched"),
        "identity_objective": phase.get("identity_objective", np.nan),
        "best_objective": phase.get("best_objective", np.nan),
        "relative_gain": phase.get("relative_gain", np.nan),
        "second_best_objective": phase.get("second_best_objective", np.nan),
        "objective_margin": phase.get("objective_margin", np.nan),
        "shift_at_boundary": phase.get("shift_at_boundary", False),
        "common_support_fraction": phase.get("common_support", np.nan),
        "bootstrap_valid_rate": bootstrap["valid_rate"],
        "bootstrap_search_success_rate": bootstrap["bootstrap_search_success_rate"],
        "bootstrap_identity_rate": bootstrap["bootstrap_identity_rate"],
        "bootstrap_nonidentity_rate": bootstrap["bootstrap_nonidentity_rate"],
        "bootstrap_failure_flat_count": bootstrap["bootstrap_failure_flat_count"],
        "bootstrap_failure_support_count": bootstrap["bootstrap_failure_support_count"],
        "bootstrap_failure_grid_count": bootstrap["bootstrap_failure_grid_count"],
        "bootstrap_failure_samples_count": bootstrap["bootstrap_failure_samples_count"],
        "bootstrap_relative_gain_median": bootstrap["bootstrap_relative_gain_median"],
        "bootstrap_objective_margin_median": bootstrap["bootstrap_objective_margin_median"],
        "shift_median_days": bootstrap["shift_median"],
        "shift_mad_days": bootstrap["shift_mad"],
        "shift_q25_days": bootstrap["shift_q25"],
        "shift_q75_days": bootstrap["shift_q75"],
        "delta_bootstrap_variance_integral": bootstrap["delta_variance_integral"],
        "delta_bootstrap_finite_rate": bootstrap["delta_finite_rate"],
        "phase_valid": phase.get("phase_available", False),
        "style_contributor": style_contributor,
        "style_evaluable": style_evaluable,
        "eligible": style_contributor,
        "exclusion_reason": reason,
        "phase_anchor": "trend_t_only",
        "structure_peak_role": "diagnostic_only_no_fallback",
    }


def _metric_row(
    class_name: str, component: str, scheme: str, style_lambda: float,
    source_values: np.ndarray, source_valid: np.ndarray,
    target_values: np.ndarray, target_valid: np.ndarray,
    source_summary: Mapping[str, np.ndarray], target_summary: Mapping[str, np.ndarray],
    style: np.ndarray, target_other_centers: Sequence[np.ndarray],
    min_common_support: float, physical_fraction: float,
    hierarchy: Mapping[str, float], valid: bool, reason: str,
) -> dict[str, object]:
    adjustment = (
        np.zeros_like(np.asarray(style, dtype=np.float64))
        if style_lambda == 0.0 else style_lambda * style
    )
    adjusted_values = source_values + adjustment[None, :]
    adjusted_summary = {
        name: (np.asarray(value) + adjustment if name != "n_valid" else value)
        for name, value in source_summary.items()
    }
    support = np.isfinite(source_summary["robust"]) & np.isfinite(target_summary["robust"])
    before_rmse = support_weighted_rmse(source_summary["robust"], target_summary["robust"], support)
    after_rmse = support_weighted_rmse(adjusted_summary["robust"], target_summary["robust"], support)
    before_energy = support_weighted_energy_distance(
        source_values, source_valid, target_values, target_valid, min_common_support,
    )
    after_energy = support_weighted_energy_distance(
        adjusted_values, source_valid, target_values, target_valid, min_common_support,
    )
    before_margin = nearest_class_margin(source_summary["robust"], target_summary["robust"], target_other_centers, support)
    after_margin = nearest_class_margin(adjusted_summary["robust"], target_summary["robust"], target_other_centers, support)
    return {
        "class_name": class_name, "component": component,
        "weighting_scheme": scheme, "lambda": style_lambda,
        "center_rmse_before": before_rmse, "center_rmse_after": after_rmse,
        "center_rmse_relative_change": (after_rmse - before_rmse) / (before_rmse + EPS),
        "iqr_distance_before": iqr_distance(source_summary, target_summary),
        "iqr_distance_after": iqr_distance(adjusted_summary, target_summary),
        "energy_distance_before": before_energy, "energy_distance_after": after_energy,
        "energy_relative_change": (after_energy - before_energy) / (before_energy + EPS),
        "nearest_class_margin_before": before_margin,
        "nearest_class_margin_after": after_margin,
        "margin_change": after_margin - before_margin,
        "physical_violation_fraction": physical_fraction,
        "style_explained_fraction": style_explained_fraction(
            target_summary["robust"] - source_summary["robust"], adjustment, support
        ),
        "hierarchy_error": max(hierarchy["max_dynamics_error"], hierarchy["max_residual_error"]),
        "valid": bool(
            valid
            and support.any()
            and np.isfinite(before_rmse)
            and np.isfinite(after_rmse)
            and np.isfinite(before_energy)
            and np.isfinite(after_energy)
            and np.isfinite(before_margin)
            and np.isfinite(after_margin)
        ),
        "invalid_reason": reason if reason else (
            "insufficient_metric_support"
            if not (
                support.any()
                and np.isfinite(before_rmse)
                and np.isfinite(after_rmse)
                and np.isfinite(before_energy)
                and np.isfinite(after_energy)
                and np.isfinite(before_margin)
                and np.isfinite(after_margin)
            )
            else ""
        ),
    }


def _write_summary_plots(
    figure_dir: Path, grid: np.ndarray, class_info: Mapping[str, Mapping[str, object]],
    phase_table: pd.DataFrame, weights_table: pd.DataFrame,
    metrics_table: pd.DataFrame, styles: Mapping[str, np.ndarray],
) -> None:
    summary_dir = figure_dir / "task_summary"
    contributors = [
        name for name, info in class_info.items()
        if info["style_contributor"]
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for name in contributors:
        axes[0].plot(grid, class_info[name]["delta_t"], alpha=0.7, label=name)
    line_styles = {
        "robust_reliability": ("black", "-", "robust reliability"),
        "uniform": ("tab:blue", "--", "uniform"),
        "source_frequency": ("tab:orange", "-.", "source frequency"),
    }
    for scheme, style in styles.items():
        color, linestyle, label = line_styles[scheme]
        axes[1].plot(grid, style, color=color, linestyle=linestyle, linewidth=2.5, label=label)
    for axis in axes:
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(ncol=3, fontsize=7)
    axes[0].set_ylabel("class ΔT"); axes[1].set_ylabel("shared style"); axes[1].set_xlabel("Canonical DOY")
    fig.suptitle(f"{ORACLE_NOTICE}\nClass trend discrepancies and shared domain style")
    _save_figure(fig, summary_dir / "class_trend_discrepancies_and_style.png")

    columns = ["q_valid", "q_phase", "q_precision", "q_coverage", "consensus_weight", "final_weight"]
    fig, axis = plt.subplots(figsize=(11, max(5, 0.45 * len(weights_table))))
    bottom = np.arange(len(weights_table), dtype=float)
    height = 0.11
    for index, column in enumerate(columns):
        axis.barh(bottom + (index - 2.5) * height, weights_table[column], height=height, label=column)
    axis.set_yticks(bottom, weights_table["class_name"])
    axis.set_xlim(0, max(1.0, float(weights_table[columns].max().max()) * 1.05 if not weights_table.empty else 1.0))
    axis.legend(ncol=3, fontsize=8); axis.grid(axis="x", alpha=0.2)
    axis.set_title(f"{ORACLE_NOTICE}\nClass style reliability and final weights")
    _save_figure(fig, summary_dir / "class_style_weights.png")

    if not contributors:
        _placeholder(
            summary_dir / "discrepancy_cosine_matrix.png",
            f"{ORACLE_NOTICE}\nClass discrepancy cosine matrix",
            "no classes used to estimate shared style",
        )
    else:
        matrix = np.full((len(contributors), len(contributors)), np.nan)
        for row, first in enumerate(contributors):
            for column, second in enumerate(contributors):
                a, b = class_info[first]["delta_t"], class_info[second]["delta_t"]
                valid = np.isfinite(a) & np.isfinite(b)
                if valid.sum() >= 3:
                    denominator = np.linalg.norm(a[valid]) * np.linalg.norm(b[valid])
                    matrix[row, column] = float(np.dot(a[valid], b[valid]) / denominator) if denominator > 0 else np.nan
        fig, axis = plt.subplots(figsize=(max(6, len(contributors) * 0.6), max(5, len(contributors) * 0.55)))
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(contributors)), contributors, rotation=45, ha="right")
        axis.set_yticks(range(len(contributors)), contributors)
        fig.colorbar(image, ax=axis, label="cosine")
        axis.set_title(f"{ORACLE_NOTICE}\nClass discrepancy cosine matrix")
        _save_figure(fig, summary_dir / "discrepancy_cosine_matrix.png")

    valid_metrics = metrics_table[(metrics_table["weighting_scheme"] == "robust_reliability") & metrics_table["valid"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    if valid_metrics.empty:
        for axis in axes: axis.text(0.5, 0.5, "No valid LOCO metrics", ha="center", va="center")
    else:
        for component in ("original", "trend", "structure"):
            grouped = valid_metrics[valid_metrics["component"] == component].groupby("lambda", sort=True)
            if not grouped.ngroups:
                continue
            axes[0].plot(grouped["center_rmse_relative_change"].median().index, grouped["center_rmse_relative_change"].median().values, marker="o", label=component)
            axes[1].plot(grouped["energy_relative_change"].median().index, grouped["energy_relative_change"].median().values, marker="o", label=component)
            axes[2].plot(grouped["center_rmse_relative_change"].apply(lambda values: np.mean(values < 0)).index, grouped["center_rmse_relative_change"].apply(lambda values: np.mean(values < 0)).values, marker="o", label=component)
            axes[3].plot(grouped["margin_change"].median().index, grouped["margin_change"].median().values, marker="o", label=component)
    axes[0].set_title("Median center RMSE relative change")
    axes[1].set_title("Median energy relative change")
    axes[2].set_title("Fraction of classes with RMSE improvement")
    axes[3].set_title("Median nearest-margin change")
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    for axis in axes: axis.set_xlabel("λ"); axis.grid(alpha=0.2)
    fig.suptitle(f"{ORACLE_NOTICE}\nLambda sensitivity (not target-label model selection)")
    _save_figure(fig, summary_dir / "lambda_summary.png")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    positions = np.arange(len(phase_table))
    axes[0].errorbar(positions, phase_table["shift_median_days"], yerr=[phase_table["shift_median_days"] - phase_table["shift_q25_days"], phase_table["shift_q75_days"] - phase_table["shift_median_days"]], fmt="o", label="bootstrap shift median/IQR")
    axes[0].plot(positions, phase_table["refined_shift_days"], color="black", alpha=0.35, label="final shift")
    status_markers = {
        "valid_identity": ("o", "valid identity"),
        "valid_nonidentity": ("^", "valid nonidentity"),
    }
    for status, (marker, label) in status_markers.items():
        selected = phase_table["phase_status"] == status
        axes[0].scatter(positions[selected], phase_table.loc[selected, "refined_shift_days"], marker=marker, label=label)
    unavailable = ~phase_table["phase_available"].astype(bool)
    axes[0].scatter(positions[unavailable], np.zeros(int(unavailable.sum())), marker="x", color="red", label="unavailable")
    axes[0].plot(positions, phase_table["shift_mad_days"], "+", label="bootstrap shift MAD")
    axes[1].plot(positions, phase_table["relative_gain"], "o-", label="relative gain")
    axes[1].plot(positions, phase_table["objective_margin"], "s-", label="objective margin")
    axes[1].plot(positions, phase_table["source_t_peak_prominence_ratio"], ".-", label="source T prominence")
    axes[1].plot(positions, phase_table["target_t_peak_prominence_ratio"], ".-", label="target T prominence")
    axes[2].bar(positions, phase_table["bootstrap_search_success_rate"], label="bootstrap search success rate", alpha=0.6)
    axes[2].plot(positions, phase_table["common_support_fraction"], "o-", label="common support")
    axes[2].scatter(positions, phase_table["style_contributor"].astype(float), marker="x", color="black", label="used to estimate shared style")
    axes[2].set_xticks(positions, phase_table["class_name"], rotation=45, ha="right")
    for axis in axes: axis.legend(); axis.grid(alpha=0.2)
    axes[0].set_ylabel("T-only shift (days)"); axes[1].set_ylabel("prominence ratio"); axes[2].set_ylabel("rate")
    fig.suptitle(f"{ORACLE_NOTICE}\nT-only phase diagnostics; S peaks are diagnostic only")
    _save_figure(fig, summary_dir / "phase_diagnostics.png")


def run_ndvi_domain_style_diagnostic(
    data_root: Path | str, output_dir: Path | str, config: DomainStyleConfig,
    classes: Iterable[str] | None = None,
    dataset_factory: Callable[[Path, str, tuple[str, ...]], object] | None = None,
) -> dict[str, object]:
    """Run the target-label ORACLE diagnostic on two selected domains."""

    if not isinstance(config, DomainStyleConfig):
        raise TypeError("config must be a DomainStyleConfig")
    requested_classes = None if classes is None else tuple(name for name in dict.fromkeys(classes) if name != "unknown")
    _, sampled, common_classes = collect_ndvi_diagnostic_parcels(
        data_root, config.samples_per_group, config.sample_seed,
        requested_classes, dataset_factory,
        domains=(config.source_domain, config.target_domain),
    )
    common_classes = [name for name in common_classes if name != "unknown"]
    if not common_classes:
        raise ValueError("no shared output classes remain after excluding 'unknown'")
    grid = np.linspace(1.0, 365.0, config.canonical_grid_size)
    records = [canonicalize_parcel(parcel, grid, config.max_interpolation_gap_days) for parcel in sampled if parcel["class_name"] in common_classes]
    grouped: dict[tuple[str, str], list[CanonicalParcelRecord]] = {}
    for record in records:
        grouped.setdefault((record.domain, record.class_name), []).append(record)

    class_info: dict[str, dict[str, object]] = {}
    phase_rows: list[dict[str, object]] = []
    discrepancy_rows: list[dict[str, object]] = []
    table_dir = Path(output_dir) / "tables" / "domain_style_oracle" / f"{config.source_domain}_to_{config.target_domain}"
    figure_dir = Path(output_dir) / "figures" / "raw_timeseries" / "domain_style_oracle" / f"{config.source_domain}_to_{config.target_domain}"
    table_dir.mkdir(parents=True, exist_ok=True)
    legacy_per_class = figure_dir / "per_class"
    if legacy_per_class.exists():
        resolved_figure_dir = figure_dir.resolve()
        resolved_legacy = legacy_per_class.resolve()
        if (
            resolved_legacy.parent != resolved_figure_dir
            or resolved_legacy.name != "per_class"
        ):
            raise RuntimeError("refusing to remove legacy figures outside task directory")
        shutil.rmtree(legacy_per_class)
    phase_figure_dir = figure_dir / "01_phase_aligned_before_style"
    sweep_figure_dir = figure_dir / "02_style_compensation_lambda_sweep"
    discrepancy_figure_dir = figure_dir / "03_class_discrepancy_vs_style"

    for class_name in common_classes:
        source_records = grouped.get((config.source_domain, class_name), [])
        target_records = grouped.get((config.target_domain, class_name), [])
        if min(len(source_records), len(target_records)) < 3:
            phase = _unavailable_phase(
                "invalid_samples", "insufficient_robust_center_samples",
                {}, {}, "not_searched",
            )
            phase.update({
                "structure_source_peak_valid": False,
                "structure_target_peak_valid": False,
            })
            bootstrap = _empty_bootstrap(len(grid))
            source_centers = target_centers = None
        else:
            source_centers = {component: _record_center(source_records, component) for component in ("original", "trend", "structure")}
            target_unaligned = {component: _record_center(target_records, component) for component in ("original", "trend", "structure")}
            phase = estimate_class_phase(
                source_centers["trend"]["robust"], target_unaligned["trend"]["robust"],
                source_centers["structure"]["robust"], target_unaligned["structure"]["robust"], grid,
                peak_search_start=config.peak_search_start, peak_search_end=config.peak_search_end,
                min_peak_prominence_ratio=config.min_peak_prominence_ratio,
                max_shift_days=config.max_shift_days,
                shift_refine_radius_days=config.shift_refine_radius_days,
                min_common_support=config.min_common_support,
                min_relative_phase_gain=config.min_relative_phase_gain,
            )
            bootstrap = bootstrap_phase_discrepancy(source_records, target_records, config, class_name)
            target_centers = None

        phase_available = bool(phase.get("phase_available", False))
        shift = float(phase["shift_days"]) if phase_available else 0.0
        if source_records and target_records:
            source_data = {}
            target_data = {}
            for component in ("original", "trend", "structure"):
                source_values, source_valid = _component_arrays(source_records, component)
                source_summary = robust_pointwise_location(source_values, source_valid)
                target_values, target_valid, target_summary = _summarize_aligned(target_records, component, shift)
                source_data[component] = (source_values, source_valid, source_summary)
                target_data[component] = (target_values, target_valid, target_summary)
            target_centers = {component: target_data[component][2] for component in target_data}
            common = np.isfinite(source_data["trend"][2]["robust"]) & np.isfinite(target_data["trend"][2]["robust"])
            bootstrap_delta_available = False
            if bootstrap["deltas"].size and common.any():
                rates = np.mean(np.isfinite(bootstrap["deltas"][:, common]), axis=0)
                bootstrap["delta_finite_rate"] = float(np.mean(rates >= 0.80))
                bootstrap_delta_available = bool(
                    np.isfinite(bootstrap["deltas"][:, common]).any()
                )
                bootstrap["delta_variance_integral"] = _bootstrap_variance_integral(
                    bootstrap["deltas"], grid, common,
                )
            delta_t = target_data["trend"][2]["robust"] - source_data["trend"][2]["robust"]
            delta_s = target_data["structure"][2]["robust"] - source_data["structure"][2]["robust"]
            delta_h = target_data["original"][2]["robust"] - source_data["original"][2]["robust"]
            for component, delta in (("original", delta_h), ("trend", delta_t), ("structure", delta_s)):
                for index, doy in enumerate(grid):
                    source_summary, target_summary = source_data[component][2], target_data[component][2]
                    discrepancy_rows.append({
                        "class_name": class_name, "component": component, "day_of_year": doy,
                        "source_center": source_summary["robust"][index],
                        "target_center_before_phase": (_record_center(target_records, component)["robust"])[index],
                        "target_center_after_phase": target_summary["robust"][index],
                        "delta": delta[index], "source_mean": source_summary["mean"][index],
                        "target_mean_after_phase": target_summary["mean"][index],
                        "source_q25": source_summary["q25"][index], "source_q75": source_summary["q75"][index],
                        "target_q25": target_summary["q25"][index], "target_q75": target_summary["q75"][index],
                        "valid": bool(np.isfinite(delta[index])),
                    })
        else:
            source_data = target_data = {}
            common = np.zeros_like(grid, bool)
            delta_t = delta_s = delta_h = np.full_like(grid, np.nan)

            bootstrap_delta_available = False

        reason_parts: list[str] = []
        if (
            len(source_records) < config.min_class_samples
            or len(target_records) < config.min_class_samples
        ):
            reason_parts.append("insufficient_class_samples")
        if not phase_available:
            reason_parts.append({
                "invalid_flat_trend": "invalid_flat_trend",
                "invalid_common_support": "phase_search_support_failure",
                "invalid_grid": "invalid_grid",
                "invalid_samples": "invalid_samples",
            }.get(str(phase.get("phase_status")), "invalid_samples"))
        if (
            phase.get("phase_status") == "invalid_common_support"
            or (
                phase_available
                and float(phase.get("common_support", np.nan))
                < config.min_common_support
            )
        ):
            reason_parts.append("phase_search_support_failure")
        if (
            bootstrap["bootstrap_search_success_rate"] <= 0.0
            or not bootstrap_delta_available
        ):
            reason_parts.append("no_bootstrap_delta")
        style_contributor = not reason_parts
        reason = ";".join(dict.fromkeys(reason_parts))
        class_info[class_name] = {
            "source_records": source_records, "target_records": target_records,
            "source_data": source_data, "target_data": target_data,
            "phase": phase, "bootstrap": bootstrap, "delta_t": delta_t,
            "delta_s": delta_s, "delta_h": delta_h, "common": common,
            "phase_available": phase_available,
            "style_contributor": style_contributor,
            "style_evaluable": False,
            "eligible": style_contributor, "reason": reason,
        }
        phase_path = phase_figure_dir / f"{_safe_name(class_name)}.png"
        if source_data and target_data:
            _plot_phase_class(
                class_name, config.source_domain, config.target_domain, grid,
                source_data, target_data, phase, bootstrap, len(source_records), len(target_records),
                phase_path,
            )
        else:
            _placeholder(
                phase_path, f"{ORACLE_NOTICE} | {class_name}",
                reason or "no sampled parcels", 3,
            )

    style_contributor_classes = [
        name for name in common_classes
        if class_info[name]["style_contributor"]
    ]
    eligible_classes = style_contributor_classes
    uncertainty_values = [class_info[name]["bootstrap"]["delta_variance_integral"] for name in style_contributor_classes if np.isfinite(class_info[name]["bootstrap"]["delta_variance_integral"])]
    uncertainty_reference = float(np.median(uncertainty_values)) if uncertainty_values else 0.0
    q_values: dict[str, dict[str, float]] = {}
    for name in common_classes:
        info = class_info[name]
        bootstrap = info["bootstrap"]
        q_valid = float(bootstrap["bootstrap_search_success_rate"])
        q_phase = math.exp(-0.5 * (float(bootstrap["shift_mad"]) / PHASE_MAD_REFERENCE_DAYS) ** 2) if np.isfinite(bootstrap["shift_mad"]) else 0.0
        uncertainty = float(bootstrap["delta_variance_integral"])
        q_precision = 1.0 if uncertainty_reference <= 0 and np.isfinite(uncertainty) and uncertainty == 0 else (1.0 / (1.0 + uncertainty / uncertainty_reference) if uncertainty_reference > 0 and np.isfinite(uncertainty) else 0.0)
        q_coverage = float(info["phase"].get("common_support", np.nan))
        base_weight = (
            q_valid * q_phase * q_precision * q_coverage
            if info["style_contributor"] and np.isfinite(q_coverage) else 0.0
        )
        q_values[name] = {"q_valid": q_valid, "q_phase": q_phase, "q_precision": q_precision, "q_coverage": q_coverage, "base": base_weight}

    deltas = {name: class_info[name]["delta_t"] for name in common_classes}
    masks = {name: class_info[name]["common"] for name in common_classes}
    base = {name: q_values[name]["base"] for name in style_contributor_classes}
    robust = fit_robust_domain_style(style_contributor_classes, deltas, masks, base)
    uniform_weights = {name: 1.0 for name in style_contributor_classes}
    source_frequency_weights = {name: float(len(class_info[name]["source_records"])) for name in style_contributor_classes}
    empty_style = np.full_like(grid, np.nan)
    schemes = {
        "robust_reliability": robust["style"] if robust["valid"] else empty_style,
        "uniform": _weighted_style(style_contributor_classes, deltas, masks, uniform_weights) if len(style_contributor_classes) >= MIN_STYLE_CLASSES else empty_style,
        "source_frequency": _weighted_style(style_contributor_classes, deltas, masks, source_frequency_weights) if len(style_contributor_classes) >= MIN_STYLE_CLASSES else empty_style,
    }
    base_by_held: dict[str, dict[str, float]] = {}
    for held_out in common_classes:
        remaining = [
            name for name in style_contributor_classes if name != held_out
        ]
        remaining_uncertainty = [
            float(class_info[name]["bootstrap"]["delta_variance_integral"])
            for name in remaining
            if np.isfinite(class_info[name]["bootstrap"]["delta_variance_integral"])
        ]
        remaining_reference = float(np.median(remaining_uncertainty)) if remaining_uncertainty else 0.0
        remaining_base = {}
        for name in remaining:
            uncertainty = float(class_info[name]["bootstrap"]["delta_variance_integral"])
            if remaining_reference > 0 and np.isfinite(uncertainty):
                q_precision = 1.0 / (1.0 + uncertainty / remaining_reference)
            elif remaining_reference == 0 and uncertainty == 0:
                q_precision = 1.0
            else:
                q_precision = 0.0
            remaining_base[name] = (
                q_values[name]["q_valid"] * q_values[name]["q_phase"]
                * q_precision * q_values[name]["q_coverage"]
            )
        base_by_held[held_out] = remaining_base
    loco_robust = compute_loco_domain_styles(
        common_classes, style_contributor_classes, deltas, masks, base,
        base_reliability_by_held=base_by_held,
    )
    for name in common_classes:
        class_info[name]["style_evaluable"] = bool(
            class_info[name]["phase_available"]
            and loco_robust[name]["valid"]
        )
    phase_rows = []
    for name in common_classes:
        evaluation_reason = ""
        if not class_info[name]["style_evaluable"]:
            evaluation_reason = (
                class_info[name]["reason"]
                if not class_info[name]["phase_available"]
                else "insufficient_style_contributors"
            )
        class_info[name]["style_evaluation_reason"] = evaluation_reason
        row = _phase_row(
            name, class_info[name]["source_records"],
            class_info[name]["target_records"], class_info[name]["phase"],
            class_info[name]["bootstrap"],
            class_info[name]["style_contributor"],
            class_info[name]["style_evaluable"], class_info[name]["reason"],
        )
        row["style_evaluation_reason"] = evaluation_reason
        phase_rows.append(row)

    weights_rows = []
    source_total = sum(source_frequency_weights.values()) or 1.0
    for name in common_classes:
        weights_rows.append({
            "class_name": name,
            "q_valid": q_values[name]["q_valid"],
            "q_phase": q_values[name]["q_phase"],
            "q_precision": q_values[name]["q_precision"],
            "q_coverage": q_values[name]["q_coverage"],
            "base_weight": q_values[name]["base"],
            "consensus_weight": robust.get("consensus", {}).get(name, 0.0),
            "final_weight": robust.get("final_weights", {}).get(name, 0.0),
            "uniform_weight": 1.0 / len(eligible_classes) if name in eligible_classes and eligible_classes else 0.0,
            "source_frequency_weight": source_frequency_weights.get(name, 0.0) / source_total,
            "style_contributor": class_info[name]["style_contributor"],
            "style_evaluable": class_info[name]["style_evaluable"],
            "eligible": class_info[name]["style_contributor"],
            "exclusion_reason": class_info[name]["reason"],
        })
    weights_table = pd.DataFrame(weights_rows)

    style_rows, loco_rows = [], []
    scheme_class_weights = {
        "robust_reliability": robust.get("final_weights", {}),
        "uniform": {
            name: 1.0 / len(eligible_classes) for name in eligible_classes
        } if eligible_classes else {},
        "source_frequency": {
            name: source_frequency_weights[name] / source_total
            for name in eligible_classes
        },
    }
    for scheme, style in schemes.items():
        for index, doy in enumerate(grid):
            contributing = [
                name for name in eligible_classes
                if masks[name][index] and np.isfinite(deltas[name][index])
                and scheme_class_weights[scheme].get(name, 0.0) > 0
            ]
            style_rows.append({"weighting_scheme": scheme, "day_of_year": doy, "style_value": style[index], "n_classes_contributing": len(contributing), "total_weight": float(sum(scheme_class_weights[scheme].get(name, 0.0) for name in contributing)), "valid": bool(np.isfinite(style[index])), "oracle_target_labels": True})
    for name in common_classes:
        loco = loco_robust.get(name, {
            "valid": False, "reason": "insufficient_style_contributors",
            "style": empty_style, "classes_used": [],
        })
        for index, doy in enumerate(grid):
            contributing = [
                other for other in loco.get("classes_used", [])
                if masks[other][index] and np.isfinite(deltas[other][index])
                and loco.get("final_weights", {}).get(other, 0.0) > 0
            ]
            loco_rows.append({"held_out_class": name, "day_of_year": doy, "style_value": loco["style"][index], "n_classes_contributing": len(contributing), "total_weight": float(sum(loco.get("final_weights", {}).get(other, 0.0) for other in contributing)), "valid": bool(loco["valid"] and np.isfinite(loco["style"][index])), "invalid_reason": loco.get("reason", ""), "oracle_target_labels": True})

    metric_rows: list[dict[str, object]] = []
    for class_name in common_classes:
        info = class_info[class_name]
        safe_filename = f"{_safe_name(class_name)}.png"
        sweep_path = sweep_figure_dir / safe_filename
        discrepancy_path = discrepancy_figure_dir / safe_filename
        loco = loco_robust.get(class_name)
        if not info["style_evaluable"] or loco is None or not loco["valid"]:
            reason = info["style_evaluation_reason"]
            for scheme in schemes:
                for style_lambda in config.style_lambdas:
                    if style_lambda == 0.0 and info["phase_available"]:
                        identity_style = np.zeros_like(grid)
                        source_center_components = {
                            component: info["source_data"][component][2]["robust"]
                            for component in ("original", "trend", "structure")
                        }
                        styled_centers = apply_shared_style(
                            source_center_components, identity_style, style_lambda,
                        )
                        hierarchy = hierarchy_max_errors(
                            source_center_components, styled_centers,
                        )
                        physical = physical_violation_fraction(
                            info["source_data"]["original"][0]
                        )
                        for component in ("original", "trend", "structure"):
                            other_targets = [
                                class_info[other]["target_data"][component][2]["robust"]
                                for other in common_classes
                                if other != class_name
                                and class_info[other]["phase_available"]
                                and class_info[other]["target_data"]
                            ]
                            row = _metric_row(
                                class_name, component, scheme, style_lambda,
                                info["source_data"][component][0],
                                info["source_data"][component][1],
                                info["target_data"][component][0],
                                info["target_data"][component][1],
                                info["source_data"][component][2],
                                info["target_data"][component][2],
                                identity_style, other_targets,
                                config.min_common_support,
                                physical if component == "original" else np.nan,
                                hierarchy, True, "",
                            )
                            row["style_explained_t"] = style_explained_fraction(
                                info["delta_t"], identity_style, info["common"],
                            )
                            row["style_transfer_explained_s"] = style_explained_fraction(
                                info["delta_s"], identity_style, info["common"],
                            )
                            metric_rows.append(row)
                        continue
                    for component in ("original", "trend", "structure"):
                        metric_rows.append({
                            "class_name": class_name, "component": component,
                            "weighting_scheme": scheme, "lambda": style_lambda,
                            "center_rmse_before": np.nan, "center_rmse_after": np.nan,
                            "center_rmse_relative_change": np.nan, "iqr_distance_before": np.nan,
                            "iqr_distance_after": np.nan, "energy_distance_before": np.nan,
                            "energy_distance_after": np.nan, "energy_relative_change": np.nan,
                            "nearest_class_margin_before": np.nan,
                            "nearest_class_margin_after": np.nan,
                            "margin_change": np.nan, "physical_violation_fraction": np.nan,
                            "style_explained_fraction": np.nan, "style_explained_t": np.nan,
                            "style_transfer_explained_s": np.nan, "hierarchy_error": np.nan,
                            "valid": False, "invalid_reason": reason,
                        })
            _placeholder(sweep_path, f"{ORACLE_NOTICE} | {class_name}", reason, 3)
            _placeholder(discrepancy_path, f"{ORACLE_NOTICE} | {class_name}", reason, 3)
            continue
        for scheme in schemes:
            if scheme == "robust_reliability":
                style = loco["style"]
            else:
                remaining = [name for name in style_contributor_classes if name != class_name]
                weights = uniform_weights if scheme == "uniform" else source_frequency_weights
                style = _weighted_style(remaining, deltas, masks, weights) if len(remaining) >= MIN_STYLE_CLASSES else empty_style
            for style_lambda in config.style_lambdas:
                source_center_components = {component: info["source_data"][component][2]["robust"] for component in ("original", "trend", "structure")}
                styled_centers = apply_shared_style(source_center_components, style, style_lambda)
                hierarchy = hierarchy_max_errors(source_center_components, styled_centers)
                if max(hierarchy.values()) >= 1e-10:
                    raise RuntimeError("shared H/T/S style application changed D or R")
                adjustment = (
                    np.zeros_like(style) if style_lambda == 0.0
                    else style_lambda * style
                )
                physical = physical_violation_fraction(
                    info["source_data"]["original"][0] + adjustment[None, :]
                )
                for component in ("original", "trend", "structure"):
                    other_targets = [
                        class_info[other]["target_data"][component][2]["robust"]
                        for other in common_classes
                        if other != class_name
                        and class_info[other]["phase_available"]
                        and class_info[other]["target_data"]
                    ]
                    row = _metric_row(
                        class_name, component, scheme, style_lambda,
                        info["source_data"][component][0], info["source_data"][component][1],
                        info["target_data"][component][0], info["target_data"][component][1],
                        info["source_data"][component][2], info["target_data"][component][2],
                        style, other_targets, config.min_common_support,
                        physical if component == "original" else np.nan,
                        hierarchy, bool(np.isfinite(style).any()), "",
                    )
                    row["style_explained_t"] = style_explained_fraction(info["delta_t"], adjustment, info["common"])
                    row["style_transfer_explained_s"] = style_explained_fraction(info["delta_s"], adjustment, info["common"])
                    metric_rows.append(row)
        robust_rows = [row for row in metric_rows if row["class_name"] == class_name and row["weighting_scheme"] == "robust_reliability"]
        _plot_style_sweep(
            class_name, config.source_domain, config.target_domain, grid,
            info["source_data"], info["target_data"],
            loco["style"], config.style_lambdas, robust_rows,
            sweep_path,
        )
        _plot_discrepancy_style(
            class_name, config.source_domain, config.target_domain, grid,
            info["delta_t"], info["delta_s"], loco["style"], info["common"],
            discrepancy_path,
        )

    phase_table = pd.DataFrame(phase_rows)
    discrepancy_table = pd.DataFrame(discrepancy_rows)
    style_table = pd.DataFrame(style_rows)
    loco_table = pd.DataFrame(loco_rows)
    metrics_table = pd.DataFrame(metric_rows)
    if metrics_table.empty:
        metrics_table = pd.DataFrame(columns=("class_name", "component", "weighting_scheme", "lambda", "valid", "invalid_reason"))
    summary_table = (
        metrics_table.groupby(["weighting_scheme", "lambda", "component"], dropna=False)
        .agg(
            n_valid_classes=("valid", "sum"),
            median_center_rmse_relative_change=("center_rmse_relative_change", "median"),
            fraction_center_rmse_improved=("center_rmse_relative_change", lambda values: float(np.mean(np.asarray(values.dropna()) < 0)) if len(values.dropna()) else np.nan),
            median_energy_relative_change=("energy_relative_change", "median"),
            fraction_energy_improved=("energy_relative_change", lambda values: float(np.mean(np.asarray(values.dropna()) < 0)) if len(values.dropna()) else np.nan),
            median_margin_change=("margin_change", "median"),
            median_style_explained_fraction=("style_explained_fraction", "median"),
        ).reset_index()
        if "center_rmse_relative_change" in metrics_table else pd.DataFrame()
    )
    outputs = {
        "phase_alignment.csv": phase_table,
        "class_domain_discrepancy_long.csv": discrepancy_table,
        "class_style_weights.csv": weights_table,
        "domain_style_curves.csv": style_table,
        "loco_domain_style_curves.csv": loco_table,
        "style_compensation_metrics.csv": metrics_table,
        "task_style_summary.csv": summary_table,
    }
    for filename, frame in outputs.items():
        frame["oracle_target_labels"] = True
        frame["not_for_training"] = True
        frame["deployable_uda"] = False
        frame.to_csv(table_dir / filename, index=False)

    _write_summary_plots(
        figure_dir, grid, class_info, phase_table, weights_table,
        metrics_table, schemes,
    )
    manifest = {
        "analysis": "raw_ndvi_oracle_domain_style",
        "oracle_target_labels": True,
        "not_for_training": True,
        "deployable_uda": False,
        "warning": ORACLE_NOTICE,
        "source_domain": config.source_domain, "target_domain": config.target_domain,
        "classes": common_classes, "eligible_classes": eligible_classes,
        "style_contributor_classes": style_contributor_classes,
        "parameters": asdict(config), "canonical_doys": grid.tolist(),
        "phase_anchor": "trend_t_only", "structure_peak": "diagnostic_only_no_fallback",
        "phase_search": {
            "peak_is_hard_requirement": False,
            "peak_role": "initialization_and_diagnostic_only",
            "fallback": "full_T_only_shift_search",
            "structure_s_defines_phase": False,
        },
        "target_alignment_formula": "target_aligned(u)=target(u-shift)",
        "style_definition": "robust equal-class-prior Huber IRLS over target-minus-source DeltaT",
        "style_application": "same un-clipped lambda*g_LOCO added to source H, T, and S",
        "lambda_selection": False,
        "lambda_role": "fixed sensitivity sweep only; never selected by target labels",
        "lambda_zero_baseline_mandatory": True,
        "figure_layout": "figure_type_then_class",
        "sensitivity_weighting": ["uniform", "source_frequency_not_recommended"],
        "weighting_formulas": {
            "q_valid": "bootstrap_search_success_rate",
            "q_phase": "exp(-0.5*(shift_mad_days/14)^2)",
            "q_precision": "1/(1+U_c/U_ref)",
            "q_coverage": "common_support_fraction",
            "base_weight": "q_valid*q_phase*q_precision*q_coverage",
            "final_weight": "normalize(base_weight*one_sided_huber_consensus)",
        },
        "constants": {"huber_c": HUBER_C, "huber_max_iter": HUBER_MAX_ITER, "huber_tol": HUBER_TOL, "phase_mad_reference_days": PHASE_MAD_REFERENCE_DAYS, "min_style_classes": MIN_STYLE_CLASSES, "eps": EPS},
        "decomposition": {"tau_fast": TAU_FAST, "tau_slow": TAU_SLOW, "hierarchy": "H=S+R, S=T+D"},
        "git_head": _git_head(), "created_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "tables": list(outputs) + ["manifest.json"],
            "figures": sorted(
                str(path.relative_to(output_dir)).replace("\\", "/")
                for path in figure_dir.rglob("*.png")
            ),
        },
    }
    (table_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "classes": common_classes, "eligible_classes": eligible_classes,
        "style_contributor_classes": style_contributor_classes,
        "records": records, "phase": phase_table, "weights": weights_table,
        "styles": style_table, "loco_styles": loco_table, "metrics": metrics_table,
        "summary": summary_table, "manifest": manifest,
    }
