"""Side-effect-free helpers for offline Fourier reconstruction anchor audits."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths


EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class Extremum:
    kind: str
    day: float
    prominence: float
    normalized_prominence: float
    width_days: float


@dataclass(frozen=True)
class ExtremaDetection:
    extrema: Tuple[Extremum, ...]
    strongest_peak: Optional[Extremum]
    strongest_valley: Optional[Extremum]
    principal_anchor: Optional[Extremum]
    status: str


@dataclass(frozen=True)
class AnchorMatch:
    matched: bool
    day: float
    absolute_error: float
    normalized_prominence: float
    prominence: float


@dataclass(frozen=True)
class GlobalShiftSelection:
    shift_days: float
    source_path: str
    semantics: str


@dataclass(frozen=True)
class DomainProjectionBaseline:
    median: float
    iqr: float


@dataclass(frozen=True)
class AnchorTimeMaps:
    start_day: float
    end_day: float
    target_anchor_day: float
    source_anchor_day: float
    valid: bool
    failure_reason: str
    left_scale: float
    right_scale: float
    max_displacement: float
    extreme: bool

    def forward(self, target_days):
        """Target-calendar input day -> source/reference output day (F_B)."""
        values = np.asarray(target_days, dtype=np.float64)
        result = values.copy()
        inside = (values >= self.start_day) & (values <= self.end_day)
        result[inside] = np.interp(
            values[inside],
            [self.start_day, self.target_anchor_day, self.end_day],
            [self.start_day, self.source_anchor_day, self.end_day],
        )
        return result

    def query(self, source_days):
        """Source/reference output day -> target-calendar input query day (Q_B)."""
        values = np.asarray(source_days, dtype=np.float64)
        result = values.copy()
        inside = (values >= self.start_day) & (values <= self.end_day)
        result[inside] = np.interp(
            values[inside],
            [self.start_day, self.source_anchor_day, self.end_day],
            [self.start_day, self.target_anchor_day, self.end_day],
        )
        return result


@dataclass(frozen=True)
class LocalPhaseResult:
    days: np.ndarray
    residual_query_days: np.ndarray
    valid: bool
    failure_reason: str
    max_displacement_days: float
    min_derivative: float
    median_derivative: float
    max_derivative: float
    extreme: bool


@dataclass(frozen=True)
class JointAnchorWindow:
    start_day: float
    end_day: float
    width_days: float
    valid: bool
    failure_reason: str


@dataclass(frozen=True)
class WholeWindowPhaseResult:
    reference_days: np.ndarray
    query_days: np.ndarray
    valid: bool
    failure_reason: str
    max_displacement_days: float
    forward_min_derivative: float
    forward_median_derivative: float
    forward_max_derivative: float
    extreme: bool


@dataclass(frozen=True)
class PreparedAnchorPhaseReference:
    days: np.ndarray
    anchor_day: float
    split_index: int
    left_source_q: np.ndarray
    right_source_q: np.ndarray


def robust_normalize_curve(curve: np.ndarray, eps: float = 1e-8):
    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return np.full_like(values, np.nan), 0.0
    q25, q75 = np.quantile(finite, (0.25, 0.75))
    scale = float(q75 - q25)
    dynamic = float(np.quantile(finite, 0.95) - np.quantile(finite, 0.05))
    if dynamic <= eps:
        return np.zeros_like(values), 0.0
    return (values - np.median(finite)) / (scale + eps), scale


def build_joint_anchor_window(
    source_anchor_day,
    target_anchor_day,
    source_support,
    target_support,
    margin_days=20,
    max_anchor_distance_days=20,
    min_anchor_side_support_days=15,
):
    """Build one unpadded source/target calendar window around corresponding anchors."""
    source_anchor = float(source_anchor_day)
    target_anchor = float(target_anchor_day)
    margin = float(margin_days)
    start = min(source_anchor, target_anchor) - margin
    end = max(source_anchor, target_anchor) + margin
    if abs(target_anchor - source_anchor) > float(max_anchor_distance_days):
        return JointAnchorWindow(
            start, end, end - start, False, "LOCAL_INELIGIBLE_ANCHOR_TOO_FAR"
        )
    source_start, source_end = map(float, source_support)
    target_start, target_end = map(float, target_support)
    common_start = max(source_start, target_start)
    common_end = min(source_end, target_end)
    side = float(min_anchor_side_support_days)
    complete = (
        start >= common_start
        and end <= common_end
        and source_anchor - common_start >= side
        and common_end - source_anchor >= side
        and target_anchor - common_start >= side
        and common_end - target_anchor >= side
    )
    if not complete:
        return JointAnchorWindow(
            start,
            end,
            end - start,
            False,
            "LOCAL_INELIGIBLE_INCOMPLETE_ANCHOR_SUPPORT",
        )
    return JointAnchorWindow(start, end, end - start, True, "")


def _invalid_whole_window(reference_days, reason):
    reference = np.asarray(reference_days, dtype=np.float64).copy()
    return WholeWindowPhaseResult(
        reference,
        reference.copy(),
        False,
        reason,
        0.0,
        1.0,
        1.0,
        1.0,
        False,
    )


def estimate_whole_window_local_phase(source_window, target_window, days):
    """Estimate one endpoint-fixed query map over the complete joint window.

    The returned query map is always source/reference output day -> target input day.
    No anchor is moved or fixed inside this adapter, and validity is purely numerical.
    """
    from analysis import phase_shape_diagnostic as phase_module

    source = np.asarray(source_window, dtype=np.float64)
    target = np.asarray(target_window, dtype=np.float64)
    reference = np.asarray(days, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape != target.shape
        or len(source) != len(reference)
        or len(reference) < 3
        or not np.all(np.diff(reference) > 0)
    ):
        raise ValueError("whole-window inputs must share [T,D] on an increasing grid")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        return _invalid_whole_window(reference, "NONLINEAR_NONFINITE_INPUT")
    phase = phase_module.estimate_nonlinear_phase(source, target, k_reg=128)
    if not phase.valid:
        return _invalid_whole_window(
            reference, "NONLINEAR_SOLVER_FAILURE:" + str(phase.failure_reason)
        )
    gamma = np.asarray(phase.gamma, dtype=np.float64)
    if (
        gamma.shape != (128,)
        or not np.isfinite(gamma).all()
        or np.any(np.diff(gamma) <= 0)
        or not np.isclose(gamma[0], 0.0, atol=1e-5)
        or not np.isclose(gamma[-1], 1.0, atol=1e-5)
    ):
        return _invalid_whole_window(reference, "NONLINEAR_INVALID_NON_STRICT_GAMMA")
    normalized = np.interp(
        np.linspace(0.0, 1.0, len(reference)),
        np.linspace(0.0, 1.0, len(gamma)),
        gamma,
    )
    query = reference[0] + (reference[-1] - reference[0]) * normalized
    query[0], query[-1] = reference[0], reference[-1]
    if np.any(np.diff(query) <= 0):
        return _invalid_whole_window(reference, "NONLINEAR_INVALID_NON_STRICT_GAMMA")
    forward_derivative = np.diff(reference) / np.diff(query)
    low, median, high = map(
        float,
        (
            forward_derivative.min(),
            np.median(forward_derivative),
            forward_derivative.max(),
        ),
    )
    displacement = float(np.max(np.abs(query - reference)))
    return WholeWindowPhaseResult(
        reference.copy(),
        query,
        True,
        "",
        displacement,
        low,
        median,
        high,
        bool(displacement > 15.0 or low < 0.5 or high > 2.0),
    )


def apply_whole_window_forward_map(target_global_days, phase):
    """Apply F=Q^-1 inside the joint window and identity outside it."""
    values = np.asarray(target_global_days, dtype=np.float64)
    if not phase.valid:
        return values.copy()
    result = values.copy()
    start, end = phase.reference_days[0], phase.reference_days[-1]
    inside = (values >= start) & (values <= end)
    result[inside] = invert_monotone_map(
        phase.reference_days, phase.query_days, values[inside]
    )
    return result


def domain_projection_baseline(features, axis, center):
    """Project an entire domain under one source-class PC1 and summarize it."""
    values = np.asarray(features)
    axis = np.asarray(axis, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != axis.size or center.shape != axis.shape:
        raise ValueError("features must be [N,T,D], and axis/center must be [D]")
    projected = np.einsum("ntd,d->nt", values, axis) - float(center @ axis)
    finite = projected[np.isfinite(projected)]
    if finite.size == 0:
        return DomainProjectionBaseline(float("nan"), float("nan"))
    q25, q75 = np.quantile(finite, (.25, .75))
    return DomainProjectionBaseline(float(np.median(finite)), float(q75 - q25))


def anchor_domain_elevation(curve, grid, anchor, baseline):
    value = float(np.interp(anchor.day, np.asarray(grid), np.asarray(curve)))
    signed = value - baseline.median if anchor.kind == "peak" else baseline.median - value
    return float(signed / (baseline.iqr + EPS))


def select_gated_anchor(
    extrema, curve, grid, baseline, min_normalized_prominence=.20,
    min_domain_relative_elevation=.75,
):
    """Choose the strongest peak/valley satisfying both independent gates."""
    prominence_ok = [
        item for item in extrema
        if item.normalized_prominence >= float(min_normalized_prominence)
    ]
    if not prominence_ok:
        return None, "low_relative_prominence"
    eligible = [
        item for item in prominence_ok
        if anchor_domain_elevation(curve, grid, item, baseline)
        >= float(min_domain_relative_elevation)
    ]
    if not eligible:
        return None, "low_domain_relative_elevation"
    return max(eligible, key=lambda item: item.normalized_prominence), ""


def build_anchor_time_maps(start_day, end_day, target_anchor_day, source_anchor_day):
    """Build endpoint-fixed B maps with explicit forward/query conventions."""
    a, b = float(start_day), float(end_day)
    target, source = float(target_anchor_day), float(source_anchor_day)
    valid = bool(a < target < b and a < source < b)
    reason = "" if valid else "anchor_outside_open_window"
    if not valid:
        return AnchorTimeMaps(a, b, target, source, False, reason, np.nan, np.nan,
                              np.nan, False)
    left = (source - a) / (target - a)
    right = (b - source) / (b - target)
    displacement = abs(source - target)
    return AnchorTimeMaps(
        a, b, target, source, True, "", left, right, displacement,
        bool(left < .5 or left > 2 or right < .5 or right > 2),
    )


def stitch_half_phase_gammas(start_day, anchor_day, end_day, left_gamma, right_gamma):
    """Join normalized half-window query maps while fixing a, anchor and b."""
    a, anchor, b = float(start_day), float(anchor_day), float(end_day)
    left_gamma = np.asarray(left_gamma, dtype=np.float64)
    right_gamma = np.asarray(right_gamma, dtype=np.float64)
    if left_gamma.ndim != 1 or right_gamma.ndim != 1 or min(len(left_gamma), len(right_gamma)) < 2:
        raise ValueError("half phase maps must be nonempty 1D arrays")
    left_days = np.linspace(a, anchor, len(left_gamma))
    right_days = np.linspace(anchor, b, len(right_gamma))
    left_query = a + (anchor - a) * left_gamma
    right_query = anchor + (b - anchor) * right_gamma
    days = np.r_[left_days, right_days[1:]]
    query = np.r_[left_query, right_query[1:]]
    query[0], query[len(left_days) - 1], query[-1] = a, anchor, b
    if not np.isfinite(query).all() or np.any(np.diff(query) <= 0):
        raise ValueError("stitched residual query map is not strictly monotone")
    return days, query


def estimate_anchor_fixed_local_phase(source_window, target_anchor_aligned, days, anchor_day):
    """Estimate independent left/right Mode13 SRVF query maps with lam=0."""
    from analysis.phase_shape_diagnostic import estimate_nonlinear_phase

    source = np.asarray(source_window, dtype=np.float64)
    target = np.asarray(target_anchor_aligned, dtype=np.float64)
    days = np.asarray(days, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or len(days) != len(source):
        raise ValueError("local phase inputs must share [T,D] and an equal-length day grid")
    split = int(np.argmin(abs(days - float(anchor_day))))
    if split < 2 or len(days) - split < 3:
        identity = days.copy()
        return LocalPhaseResult(days, identity, False, "insufficient_half_support", 0, 1, 1, 1, False)
    left = estimate_nonlinear_phase(source[:split + 1], target[:split + 1], k_reg=128)
    right = estimate_nonlinear_phase(source[split:], target[split:], k_reg=128)
    if not left.valid or not right.valid:
        reason = "left:" + left.failure_reason if not left.valid else "right:" + right.failure_reason
        return LocalPhaseResult(days, days.copy(), False, reason, 0, 1, 1, 1, False)
    try:
        phase_days, query = stitch_half_phase_gammas(
            days[0], days[split], days[-1], left.gamma, right.gamma
        )
        query = np.interp(days, phase_days, query)
    except ValueError as error:
        return LocalPhaseResult(days, days.copy(), False, str(error), 0, 1, 1, 1, False)
    derivative = np.diff(query) / np.diff(days)
    displacement = float(np.max(np.abs(query - days)))
    low, median, high = map(float, (derivative.min(), np.median(derivative), derivative.max()))
    return LocalPhaseResult(
        days, query, True, "", displacement, low, median, high,
        bool(displacement > 15 or low < .5 or high > 2),
    )


def estimate_anchor_fixed_local_phases(
    source_window, target_anchor_aligned, days, anchor_day, workers=1,
):
    """Estimate a class batch in order; threads share the immutable source reference."""
    targets = np.asarray(target_anchor_aligned, dtype=np.float64)
    if targets.ndim != 3:
        raise ValueError("target anchor-aligned curves must be [N,T,D]")
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be positive")

    prepared = prepare_anchor_fixed_local_phase_reference(source_window, days, anchor_day)

    def estimate(target):
        return estimate_anchor_fixed_local_phase_prepared(prepared, target)

    if workers == 1 or len(targets) < 2:
        return [estimate(target) for target in targets]
    with ThreadPoolExecutor(max_workers=min(workers, len(targets))) as pool:
        return list(pool.map(estimate, targets))


def prepare_anchor_fixed_local_phase_reference(source_window, days, anchor_day):
    """Cache the two immutable source SRVFs used by every sample in a class."""
    from fdasrsf import curve_functions
    from analysis import phase_shape_diagnostic as phase_module

    source = np.asarray(source_window, dtype=np.float64)
    days = np.asarray(days, dtype=np.float64)
    if source.ndim != 2 or len(source) != len(days):
        raise ValueError("source reference must be [T,D] on the supplied day grid")
    split = int(np.argmin(abs(days - float(anchor_day))))
    if split < 2 or len(days) - split < 3:
        raise ValueError("source reference has insufficient half-window support")

    def source_q(half):
        normalized = phase_module._resample(phase_module.robust_normalize(half), 128)
        return curve_functions.curve_to_q(normalized.T, mode="O", scale=False)[0]

    return PreparedAnchorPhaseReference(
        days.copy(), float(anchor_day), split,
        source_q(source[:split + 1]), source_q(source[split:]),
    )


def estimate_anchor_fixed_local_phase_prepared(prepared, target_anchor_aligned):
    """Reference-equivalent phase estimate with cached source half-window SRVFs."""
    from fdasrsf import curve_functions
    from analysis import phase_shape_diagnostic as phase_module

    target = np.asarray(target_anchor_aligned, dtype=np.float64)
    days, split = prepared.days, prepared.split_index
    if target.ndim != 2 or len(target) != len(days):
        raise ValueError("target curve must be [T,D] on the prepared day grid")
    if not np.isfinite(target).all():
        return LocalPhaseResult(days, days.copy(), False, "non_finite_input", 0, 1, 1, 1, False)

    def estimate_half(source_q, target_half):
        try:
            normalized = phase_module._resample(
                phase_module.robust_normalize(target_half), 128
            )
            target_q = curve_functions.curve_to_q(
                normalized.T, mode="O", scale=False
            )[0]
            gamma = curve_functions.optimum_reparam_curve(
                source_q, target_q, lam=0.0, method="DP"
            )
            return phase_module._phase_result(gamma)
        except Exception as error:
            return phase_module._phase_result(
                np.linspace(0, 1, 128), False, type(error).__name__
            )

    left = estimate_half(prepared.left_source_q, target[:split + 1])
    right = estimate_half(prepared.right_source_q, target[split:])
    if not left.valid or not right.valid:
        reason = "left:" + left.failure_reason if not left.valid else "right:" + right.failure_reason
        return LocalPhaseResult(days, days.copy(), False, reason, 0, 1, 1, 1, False)
    try:
        phase_days, query = stitch_half_phase_gammas(
            days[0], days[split], days[-1], left.gamma, right.gamma
        )
        query = np.interp(days, phase_days, query)
    except ValueError as error:
        return LocalPhaseResult(days, days.copy(), False, str(error), 0, 1, 1, 1, False)
    derivative = np.diff(query) / np.diff(days)
    displacement = float(np.max(np.abs(query - days)))
    low, median, high = map(float, (derivative.min(), np.median(derivative), derivative.max()))
    return LocalPhaseResult(
        days, query, True, "", displacement, low, median, high,
        bool(displacement > 15 or low < .5 or high > 2),
    )


def compose_local_query(source_days, anchor_maps, residual_query_days):
    """C inverse query map: Q_C = Q_B composed with residual gamma."""
    source_days = np.asarray(source_days, dtype=np.float64)
    gamma = np.asarray(residual_query_days, dtype=np.float64)
    if gamma.shape != source_days.shape:
        raise ValueError("residual query days must match source day grid")
    return anchor_maps.query(gamma)


def estimate_batched_window_phase(
    source_window, targets, days, anchor_day, device="cpu", steps=80,
    segments=8, learning_rate=.05,
):
    """Fit multivariate correlation with batched, endpoint/anchor-fixed query maps.

    This is a finite-knot shape registration solver, NOT the SRVF-DP objective.
    Each half has positive segment durations (5% uniform + 95% softmax).
    The identity and best iterate are retained independently for every sample.
    Only temporary time-map parameters receive gradients; no model is involved.
    """
    import torch
    import torch.nn.functional as functional

    source = np.asarray(source_window, dtype=np.float32)
    values = np.asarray(targets, dtype=np.float32)
    days = np.asarray(days, dtype=np.float64)
    if source.ndim != 2 or values.ndim != 3 or values.shape[1:] != source.shape:
        raise ValueError("expected source [T,D] and targets [N,T,D]")
    if len(days) != len(source) or len(days) < 5 or not np.all(np.diff(days) > 0):
        raise ValueError("expected increasing window days matching curves")
    if not np.allclose(np.diff(days), np.diff(days)[0]):
        raise ValueError("batched phase requires a uniform reconstruction grid")
    split = int(np.argmin(abs(days - anchor_day)))
    if split < 2 or len(days)-split < 3 or not np.isclose(days[split], anchor_day):
        raise ValueError("anchor must lie on an interior grid point")
    if steps < 1 or segments < 2 or learning_rate <= 0:
        raise ValueError("steps, segments and learning rate must be positive")
    results = [LocalPhaseResult(days, days.copy(), False, "non_finite_input", 0, 1, 1, 1, False)
               for _ in values]
    if not np.isfinite(source).all():
        return results
    indices = np.flatnonzero(np.isfinite(values).all(axis=(1, 2)))
    if not len(indices):
        return results
    # Input copies keep the routine usable even under a caller's inference_mode.
    with torch.inference_mode(False), torch.enable_grad():
        reference = torch.tensor(source, device=device).unsqueeze(0)
        curves = torch.tensor(values[indices], device=device)
        source_iqr = torch.quantile(reference, .75, dim=1) - torch.quantile(reference, .25, dim=1)
        target_iqr = torch.quantile(curves, .75, dim=1) - torch.quantile(curves, .25, dim=1)
        active = (source_iqr > 1e-6) & (target_iqr > 1e-6)
        active_count = active.sum(dim=-1)
        # Scaling is fixed before fitting. Pearson correlation is invariant to it.
        reference = (reference-reference.mean(dim=1, keepdim=True))/source_iqr.clamp_min(1e-6)[:, None]
        curves = (curves-curves.mean(dim=1, keepdim=True))/target_iqr.clamp_min(1e-6)[:, None]
        reference = torch.where(active[:, None], reference, torch.zeros_like(curves))
        curves = torch.where(active[:, None], curves, torch.zeros_like(curves))
        reference_norm = reference.square().sum(dim=1).sqrt().clamp_min(1e-8)
        parameters = torch.zeros((len(indices), 2, segments), device=device, requires_grad=True)
        first_moment, second_moment = torch.zeros_like(parameters), torch.zeros_like(parameters)
        identity = torch.arange(len(days), device=device, dtype=torch.float32)[None].expand(len(indices), -1)
        best_query = identity.clone()

        def score(query):
            # gather provides differentiable piecewise-linear sampling in time.
            left = query.floor().long().clamp(0, len(days)-2)
            weight = (query-left).unsqueeze(-1)
            left_values = curves.gather(1, left[..., None].expand(-1, -1, curves.shape[-1]))
            right_values = curves.gather(1, (left+1)[..., None].expand(-1, -1, curves.shape[-1]))
            warped = left_values + weight*(right_values-left_values)
            centered = warped-warped.mean(dim=1, keepdim=True)
            correlation = (reference*centered).sum(dim=1) / (
                reference_norm * centered.square().sum(dim=1).clamp_min(1e-16).sqrt()
            )
            return torch.where(active, correlation, torch.zeros_like(correlation)).sum(-1)/active_count.clamp_min(1)

        def query_from_parameters():
            increments = .05/segments + .95*parameters.softmax(dim=-1)
            knots = functional.pad(increments.cumsum(dim=-1), (1, 0))
            # Explicit constants fix both endpoints exactly despite float rounding.
            knots = torch.cat((torch.zeros_like(knots[..., :1]), knots[..., 1:-1], torch.ones_like(knots[..., :1])), dim=-1)
            left = functional.interpolate(knots[:, :1], size=split+1, mode="linear", align_corners=True)[:, 0]*split
            right = split + functional.interpolate(knots[:, 1:], size=len(days)-split, mode="linear", align_corners=True)[:, 0]*(len(days)-1-split)
            return torch.cat((left, right[:, 1:]), dim=1)

        best_score = score(identity).detach()
        for iteration in range(steps+1):
            query = query_from_parameters()
            scores = score(query)
            improved = torch.isfinite(scores) & (scores > best_score + 1e-7)
            best_query = torch.where(improved[:, None], query.detach(), best_query)
            best_score = torch.where(improved, scores.detach(), best_score)
            if iteration == steps:
                break
            gradient, = torch.autograd.grad(-scores.sum(), parameters)
            gradient = torch.nan_to_num(gradient)
            with torch.no_grad():
                first_moment.mul_(.9).add_(gradient, alpha=.1)
                second_moment.mul_(.999).addcmul_(gradient, gradient, value=.001)
                update = first_moment/(1-.9**(iteration+1))
                denominator = (second_moment/(1-.999**(iteration+1))).sqrt()+1e-8
                parameters.addcdiv_(update, denominator, value=-learning_rate)
        queries = best_query.detach().cpu().numpy().astype(np.float64)*np.diff(days)[0]+days[0]
        counts = active_count.detach().cpu().numpy()
    for index, query, count in zip(indices, queries, counts):
        query[[0, split, -1]] = days[[0, split, -1]]
        if count == 0:
            results[index] = LocalPhaseResult(days, days.copy(), False, "no_variable_channels", 0, 1, 1, 1, False)
            continue
        derivative = np.diff(query)/np.diff(days)
        if not np.isfinite(query).all() or np.any(derivative <= 0):
            results[index] = LocalPhaseResult(days, days.copy(), False, "invalid_query", 0, 1, 1, 1, False)
            continue
        displacement = float(np.max(abs(query-days)))
        low, median, high = map(float, (derivative.min(), np.median(derivative), derivative.max()))
        results[index] = LocalPhaseResult(days, query, True, "", displacement, low, median, high,
                                          bool(displacement > 15 or low < .5 or high > 2))
    return results


def window_metrics_batch(source, targets):
    """Vectorized historical correlation, open-curve SRVF distance and robust L2."""
    source = np.asarray(source, dtype=np.float64)[None]
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 3 or targets.shape[1:] != source.shape[1:]:
        raise ValueError("expected source [T,D] and targets [N,T,D]")
    sq, tq = np.quantile(source, (.25, .75), axis=1), np.quantile(targets, (.25, .75), axis=1)
    si, ti = sq[1]-sq[0], tq[1]-tq[0]
    valid = (si > 1e-6) & (ti > 1e-6) & np.isfinite(targets).all(axis=1) & np.isfinite(source).all(axis=1)
    sc, tc = source-source.mean(axis=1, keepdims=True), targets-targets.mean(axis=1, keepdims=True)
    denominator = np.sqrt((sc**2).sum(axis=1)*(tc**2).sum(axis=1))
    channel_corr = np.divide((sc*tc).sum(axis=1), denominator, out=np.zeros_like(denominator), where=denominator>EPS)
    correlation = np.divide(np.where(valid, channel_corr, 0).sum(-1), valid.sum(-1),
                            out=np.full(len(targets), np.nan), where=valid.sum(-1)>0)

    def q(values):
        velocity = np.gradient(values, 1./values.shape[1], axis=1)
        norm = np.sqrt(np.linalg.norm(velocity, axis=2, keepdims=True))
        return np.where(norm > .0001, velocity/np.maximum(norm, EPS), velocity*.0001)

    source_q, target_q = q(source), q(targets)
    srvf = np.linalg.norm((target_q-source_q).reshape(len(targets), -1), axis=1)/max(np.linalg.norm(source_q), EPS)

    def normalize(values, iqr):
        positive = np.where(iqr > EPS, iqr, np.nan)
        # A channel loop is unnecessary; all-flat samples have a defined EPS floor.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            floor = np.maximum(EPS, .001*np.nanmedian(positive, axis=1))
        floor = np.nan_to_num(floor, nan=EPS)
        good = np.isfinite(iqr) & (iqr > floor[:, None])
        return np.where(good[:, None], (values-np.median(values, axis=1, keepdims=True))/np.maximum(iqr, floor[:, None])[:, None], 0)

    sn, tn = normalize(source, si), normalize(targets, ti)
    distance = np.linalg.norm((tn-sn).reshape(len(targets), -1), axis=1)/max(np.linalg.norm(sn), EPS)
    return correlation, srvf, distance


def invert_monotone_map(domain, mapped, query):
    domain = np.asarray(domain, dtype=np.float64)
    mapped = np.asarray(mapped, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if domain.ndim != 1 or mapped.shape != domain.shape or np.any(np.diff(mapped) <= 0):
        raise ValueError("map inversion requires a strictly increasing 1D map")
    return np.interp(query, mapped, domain)


def apply_local_forward_to_timestamps(target_days, anchor_maps, residual_days=None, residual_query_days=None):
    """Apply F_B or F_C to target-calendar timestamps; outside the window is identity."""
    target_days = np.asarray(target_days, dtype=np.float64)
    if residual_days is None:
        result = anchor_maps.forward(target_days)
    else:
        residual_days = np.asarray(residual_days, dtype=np.float64)
        composite_query = compose_local_query(residual_days, anchor_maps, residual_query_days)
        result = target_days.copy()
        inside = (target_days >= anchor_maps.start_day) & (target_days <= anchor_maps.end_day)
        result[inside] = invert_monotone_map(residual_days, composite_query, target_days[inside])
    return result


def sample_curve_with_query(curve, days, query_days):
    curve = np.asarray(curve, dtype=np.float64)
    days = np.asarray(days, dtype=np.float64)
    query = np.asarray(query_days, dtype=np.float64)
    if curve.ndim != 2 or len(curve) != len(days):
        raise ValueError("curve must be [T,D] on the supplied grid")
    return np.column_stack([np.interp(query, days, curve[:, channel])
                            for channel in range(curve.shape[1])])


def multivariate_srvf_distance(source, target):
    """Normalized L2 distance between open-curve multivariate SRVFs."""
    from fdasrsf import curve_functions

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("SRVF inputs must share [T,D]")
    q_source = curve_functions.curve_to_q(source.T, mode="O", scale=False)[0]
    q_target = curve_functions.curve_to_q(target.T, mode="O", scale=False)[0]
    denominator = max(float(np.linalg.norm(q_source)), EPS)
    return float(np.linalg.norm(q_source - q_target) / denominator)


def prepare_multivariate_srvf_reference(source):
    """Compute the source SRVF once for repeated same-class distance audits."""
    from fdasrsf import curve_functions

    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("SRVF source must be [T,D]")
    return curve_functions.curve_to_q(source.T, mode="O", scale=False)[0]


def multivariate_srvf_distance_from_reference(source_q, target):
    from fdasrsf import curve_functions

    source_q = np.asarray(source_q, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 2:
        raise ValueError("SRVF target must be [T,D]")
    target_q = curve_functions.curve_to_q(target.T, mode="O", scale=False)[0]
    if source_q.shape != target_q.shape:
        raise ValueError("prepared source and target SRVFs must have the same shape")
    return float(np.linalg.norm(source_q - target_q) / max(float(np.linalg.norm(source_q)), EPS))


def _detected_kind(
    raw: np.ndarray,
    normalized: np.ndarray,
    grid: np.ndarray,
    kind: str,
    min_distance_days: float,
    min_width_days: float,
    min_normalized_prominence: float,
) -> Sequence[Extremum]:
    sign = 1.0 if kind == "peak" else -1.0
    step = float(np.median(np.diff(grid)))
    distance = max(1, int(np.ceil(float(min_distance_days) / step)))
    width = max(EPS, float(min_width_days) / step)
    dynamic = float(np.quantile(raw, 0.95) - np.quantile(raw, 0.05))
    if not np.isfinite(dynamic) or dynamic <= 1e-8:
        return ()
    indices, _ = find_peaks(sign * normalized, distance=distance, width=width)
    if indices.size == 0:
        return ()
    raw_prominence = peak_prominences(sign * raw, indices)[0]
    widths = peak_widths(sign * normalized, indices, rel_height=0.5)[0] * step
    result = []
    for index, prominence, width_days in zip(indices, raw_prominence, widths):
        normalized_prominence = float(prominence / (dynamic + EPS))
        if normalized_prominence >= float(min_normalized_prominence):
            result.append(
                Extremum(
                    kind=kind,
                    day=float(grid[index]),
                    prominence=float(prominence),
                    normalized_prominence=normalized_prominence,
                    width_days=float(width_days),
                )
            )
    return result


def detect_salient_extrema(
    curve: np.ndarray,
    grid: np.ndarray,
    min_distance_days: float = 7,
    min_width_days: float = 3,
    min_normalized_prominence: float = 0.20,
) -> ExtremaDetection:
    raw = np.asarray(curve, dtype=np.float64).reshape(-1)
    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    if raw.shape != grid.shape or len(grid) < 3 or not np.all(np.diff(grid) > 0):
        raise ValueError("curve/grid must be equal-length 1D arrays on an increasing grid")
    normalized, scale = robust_normalize_curve(raw)
    dynamic = float(np.quantile(raw, 0.95) - np.quantile(raw, 0.05))
    if dynamic <= 1e-8 or not np.isfinite(normalized).all():
        return ExtremaDetection((), None, None, None, "NO_SALIENT_ANCHOR")
    peaks = tuple(
        _detected_kind(
            raw,
            normalized,
            grid,
            "peak",
            min_distance_days,
            min_width_days,
            min_normalized_prominence,
        )
    )
    valleys = tuple(
        _detected_kind(
            raw,
            normalized,
            grid,
            "valley",
            min_distance_days,
            min_width_days,
            min_normalized_prominence,
        )
    )
    strongest_peak = max(peaks, key=lambda item: item.normalized_prominence, default=None)
    strongest_valley = max(valleys, key=lambda item: item.normalized_prominence, default=None)
    principal = max(
        (item for item in (strongest_peak, strongest_valley) if item is not None),
        key=lambda item: item.normalized_prominence,
        default=None,
    )
    return ExtremaDetection(
        peaks + valleys,
        strongest_peak,
        strongest_valley,
        principal,
        "ANCHOR_FOUND" if principal is not None else "NO_SALIENT_ANCHOR",
    )


def shifted_calendar(grid, shift_days, support_start=0.0, support_end=364.0):
    shifted = np.asarray(grid, dtype=np.float64) + float(shift_days)
    valid = (shifted >= float(support_start)) & (shifted <= float(support_end))
    return shifted, valid


def match_sample_anchor(
    curve: np.ndarray,
    grid: np.ndarray,
    prototype_anchor: Extremum,
    search_radius_days: float = 30,
    min_width_days: float = 3,
    min_normalized_prominence: float = 0.20,
    calendar_shift_days: float = 0.0,
    min_distance_days: float = 7,
) -> AnchorMatch:
    shifted, valid = shifted_calendar(grid, calendar_shift_days)
    if valid.sum() < 3:
        return AnchorMatch(False, np.nan, np.nan, np.nan, np.nan)
    detection = detect_salient_extrema(
        np.asarray(curve)[valid],
        shifted[valid],
        min_distance_days,
        min_width_days,
        min_normalized_prominence,
    )
    candidates = [
        item
        for item in detection.extrema
        if item.kind == prototype_anchor.kind
        and abs(item.day - prototype_anchor.day) <= float(search_radius_days)
    ]
    if not candidates:
        return AnchorMatch(False, np.nan, np.nan, np.nan, np.nan)
    best = min(
        candidates,
        key=lambda item: (
            -item.normalized_prominence,
            abs(item.day - prototype_anchor.day),
            item.day,
        ),
    )
    error = abs(best.day - prototype_anchor.day)
    return AnchorMatch(True, best.day, error, best.normalized_prominence, best.prominence)


def summarize_matches(matches: Sequence[AnchorMatch]) -> Dict[str, float]:
    matched = [item for item in matches if item.matched]
    errors = np.asarray([item.absolute_error for item in matched], dtype=np.float64)
    days = np.asarray([item.day for item in matched], dtype=np.float64)
    prominence = np.asarray(
        [item.normalized_prominence for item in matched], dtype=np.float64
    )
    nan = float("nan")
    return {
        "match_count": len(matched),
        "occurrence_rate": len(matched) / len(matches) if matches else nan,
        "timing_median": float(np.median(days)) if days.size else nan,
        "timing_std": float(np.std(days)) if days.size else nan,
        "timing_mad": float(np.median(np.abs(days - np.median(days)))) if days.size else nan,
        "timing_error_median": float(np.median(errors)) if errors.size else nan,
        "timing_error_p90": float(np.quantile(errors, 0.90)) if errors.size else nan,
        "prominence_median": float(np.median(prominence)) if prominence.size else nan,
        "prominence_iqr": (
            float(np.quantile(prominence, 0.75) - np.quantile(prominence, 0.25))
            if prominence.size
            else nan
        ),
    }


def classify_source_anchor(anchor, occurrence_rate, timing_mad, min_prominence=0.20):
    if anchor is None:
        return "NO_SALIENT_ANCHOR"
    if (
        float(occurrence_rate) >= 0.70
        and np.isfinite(timing_mad)
        and float(timing_mad) <= 20.0
        and anchor.normalized_prominence >= float(min_prominence)
    ):
        return "SOURCE_ANCHOR_STABLE"
    return "SOURCE_ANCHOR_UNSTABLE"


def diagnostic_eligibility(source_status, target_match_rate, target_error_median):
    if (
        source_status == "SOURCE_ANCHOR_STABLE"
        and float(target_match_rate) >= 0.60
        and np.isfinite(target_error_median)
        and float(target_error_median) <= 20.0
    ):
        return "LOCAL_DIAGNOSTIC_ELIGIBLE"
    return "GLOBAL_ONLY_DIAGNOSTIC"


def _window(curve, grid, anchor_day, radius):
    values = np.asarray(curve, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    query = float(anchor_day) + np.arange(-int(radius), int(radius) + 1)
    if query[0] < grid[0] or query[-1] > grid[-1]:
        return None
    if values.ndim == 1:
        return np.interp(query, grid, values)
    return np.column_stack(
        [np.interp(query, grid, values[:, channel]) for channel in range(values.shape[1])]
    )


def _finite_corr(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > EPS else float("nan")


def local_scalar_correlation(
    source_curve,
    source_grid,
    source_anchor_day,
    target_curve,
    target_grid,
    target_anchor_day,
    radius_days=30,
):
    source = _window(source_curve, source_grid, source_anchor_day, radius_days)
    target = _window(target_curve, target_grid, target_anchor_day, radius_days)
    if source is None or target is None:
        return float("nan")
    source, source_scale = robust_normalize_curve(source)
    target, target_scale = robust_normalize_curve(target)
    if source_scale <= 1e-8 or target_scale <= 1e-8:
        return float("nan")
    return _finite_corr(source, target)


def local_multivariate_correlation(source_window, target_window, iqr_floor=1e-6):
    source = np.asarray(source_window, dtype=np.float64)
    target = np.asarray(target_window, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source/target windows must have the same [T,D] shape")
    source_q = np.quantile(source, (0.25, 0.75), axis=0)
    target_q = np.quantile(target, (0.25, 0.75), axis=0)
    source_iqr = source_q[1] - source_q[0]
    target_iqr = target_q[1] - target_q[0]
    valid = (
        np.isfinite(source).all(axis=0)
        & np.isfinite(target).all(axis=0)
        & (source_iqr > iqr_floor)
        & (target_iqr > iqr_floor)
    )
    correlations = []
    for channel in np.flatnonzero(valid):
        left = (source[:, channel] - np.median(source[:, channel])) / source_iqr[channel]
        right = (target[:, channel] - np.median(target[:, channel])) / target_iqr[channel]
        value = _finite_corr(left, right)
        if np.isfinite(value):
            correlations.append(value)
    return (
        float(np.mean(correlations)) if correlations else float("nan"),
        len(correlations),
    )


def fit_fourier_modes_independently(features, positions, modes, analyzer_factory):
    result = {}
    for mode in modes:
        coefficients, _ = analyzer_factory(int(mode))(
            features, positions, collect_diagnostics=False
        )
        result[int(mode)] = coefficients
    return result


def project_modes_with_fixed_pc1(
    reconstructed_by_mode: Mapping[int, np.ndarray], projection
):
    return {
        int(mode): projection.transform(np.asarray(curves))
        for mode, curves in reconstructed_by_mode.items()
    }


def project_fourier_coefficients(coefficients, projection):
    coefficients = np.asarray(coefficients)
    projected = np.einsum("nfd,d->nf", coefficients, projection.axis)
    offset = float(np.dot(projection.center, projection.axis))
    return projected, offset


def read_authoritative_global_shift(root: Path, task_name: str) -> GlobalShiftSelection:
    task_dir = Path(root) / task_name
    manifest_path = task_dir / "manifest.json"
    summary_path = task_dir / "shifts_summary.csv"
    view_path = task_dir / "03_reconshift13_shift"
    if not manifest_path.is_file() or not summary_path.is_file() or not view_path.is_dir():
        raise FileNotFoundError(
            f"authoritative ReconShift13 artifacts missing below {task_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest.get("reconshift13_shift", {})
    if "shift_days" not in record:
        raise ValueError(f"reconshift13_shift.shift_days missing in {manifest_path}")
    shift = float(record["shift_days"])
    with summary_path.open(newline="", encoding="utf-8") as stream:
        csv_values = {
            float(row["reconshift_shift_days"])
            for row in csv.DictReader(stream)
            if row.get("reconshift_shift_days") not in (None, "")
        }
    if csv_values != {shift}:
        raise ValueError(
            f"ReconShift13 shift disagreement: manifest={shift}, csv={sorted(csv_values)}"
        )
    return GlobalShiftSelection(
        shift,
        str(manifest_path),
        "target calendar is interpreted as t + delta_g; no circular wrap",
    )


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_median(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")
