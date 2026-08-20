"""TimeMatch-anchored nonlinear residual Phase helpers.

The module deliberately owns only the small mathematical layer added on top of
TimeMatch.  Scalar shift estimation, pseudo-label/self-training, optimizer,
EMA and augmentation stay in the TimeMatch-compatible trainer.

The target-to-source candidate is

    gamma_{delta,alpha}(u) = u + delta / time_scale
                           + alpha * (rho_dom(u) - u)

where ``rho_dom`` is a frozen, monotone residual direction estimated once at
Stage-2 bootstrap.  ``alpha=0`` is therefore exactly the scalar TimeMatch view.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .phase_geometry import sqrt_mean_gamma

ALPHA_BANK: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)


def requires_nonlinear_phase(alphas: Sequence[float], *, atol: float = 1e-12) -> bool:
    """Return whether an alpha candidate set can use nonlinear residual Phase."""
    if not alphas:
        raise ValueError("alpha candidates must not be empty")
    values = tuple(float(value) for value in alphas)
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("alpha candidates must be finite values in [0, 1]")
    return any(abs(value) > atol for value in values)


@dataclass(frozen=True)
class AlphaSelection:
    alpha: float
    best_am: float
    second_best_am: float
    margin: float
    rows: tuple[dict, ...]


def canonical_grid(size: int, *, dtype: torch.dtype = torch.float64) -> Tensor:
    if int(size) < 2:
        raise ValueError("grid size must be >= 2")
    return torch.linspace(0.0, 1.0, int(size), dtype=dtype)


def validate_monotone_grid(values: Tensor, *, strict: bool = True) -> Tensor:
    if not isinstance(values, Tensor) or values.ndim != 1 or values.numel() < 2:
        raise ValueError("phase grid must have shape [K], K>=2")
    out = values.detach().cpu().double().contiguous()
    if not torch.isfinite(out).all().item():
        raise ValueError("phase grid must be finite")
    diff = out[1:] - out[:-1]
    good = torch.all(diff > 0) if strict else torch.all(diff >= 0)
    if not bool(good.item()):
        raise ValueError("phase grid must be strictly increasing" if strict else "phase grid must be monotone")
    return out


def validate_residual_phase(rho_dom: Tensor, *, endpoint_tolerance: float = 1e-6) -> Tensor:
    rho = validate_monotone_grid(rho_dom)
    if abs(float(rho[0])) > float(endpoint_tolerance) or abs(float(rho[-1]) - 1.0) > float(endpoint_tolerance):
        raise ValueError("residual rho_dom must map [0,1] to [0,1]")
    return rho


def aggregate_class_residual_phases(class_rhos: Sequence[Tensor]) -> Tensor:
    """Equal-class Fisher--Rao/Karcher mean using existing Phase geometry."""
    if not class_rhos:
        raise ValueError("at least one valid class residual is required")
    validated = [validate_residual_phase(x) for x in class_rhos]
    shape = validated[0].shape
    if any(x.shape != shape for x in validated):
        raise ValueError("all class residual phases must share the same grid")
    return validate_residual_phase(sqrt_mean_gamma(torch.stack(validated, dim=0)))


def candidate_phase_grid(
    *,
    delta_days: float,
    alpha: float,
    rho_dom: Tensor,
    time_scale_days: float = 365.0,
) -> Tensor:
    if not np.isfinite(float(delta_days)):
        raise ValueError("delta_days must be finite")
    if not np.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    if not np.isfinite(float(time_scale_days)) or float(time_scale_days) <= 0:
        raise ValueError("time_scale_days must be positive")
    rho = validate_residual_phase(rho_dom)
    grid = canonical_grid(rho.numel())
    gamma = grid + float(delta_days) / float(time_scale_days) + float(alpha) * (rho - grid)
    return validate_monotone_grid(gamma)


def _interp_extrapolate_1d(query: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Piecewise-linear interpolation with linear endpoint extrapolation."""
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.numel() < 2:
        raise ValueError("x/y must be matching one-dimensional grids")
    x = validate_monotone_grid(x).to(device=query.device, dtype=query.dtype)
    y = y.to(device=query.device, dtype=query.dtype)
    q = query
    idx = torch.searchsorted(x, q, right=True) - 1
    idx = idx.clamp(0, x.numel() - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    w = (q - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


def evaluate_phase_grid(
    positions: Tensor,
    phase_grid: Tensor,
    *,
    time_mask: Tensor | None = None,
) -> Tensor:
    """Evaluate a monotone phase on normalized observation positions.

    Values outside the phase input range use linear extrapolation.  This is
    needed by the inverse source-to-target phase when the TimeMatch translation
    moves endpoints outside [0,1].  Padding positions remain zero.
    """
    if positions.ndim not in (1, 2):
        raise ValueError("positions must have shape [L] or [B,L]")
    if not positions.is_floating_point() or not torch.isfinite(positions).all().item():
        raise ValueError("positions must be finite floating-point values")
    gamma = validate_monotone_grid(phase_grid).to(device=positions.device, dtype=positions.dtype)
    grid = canonical_grid(gamma.numel(), dtype=positions.dtype).to(positions.device)
    out = _interp_extrapolate_1d(positions, grid, gamma)
    if time_mask is not None:
        mask = time_mask.to(device=positions.device, dtype=torch.bool)
        if mask.shape != positions.shape:
            raise ValueError("time_mask must match positions")
        out = torch.where(mask, out, torch.zeros_like(out))
    return out


def inverse_phase_on_canonical_grid(phase_grid: Tensor) -> Tensor:
    """Sample the inverse monotone phase on the canonical [0,1] input grid."""
    gamma = validate_monotone_grid(phase_grid)
    u = canonical_grid(gamma.numel())
    return validate_monotone_grid(_interp_extrapolate_1d(u, gamma, u))


def evaluate_candidate_positions(
    normalized_positions: Tensor,
    *,
    delta_days: float,
    alpha: float,
    rho_dom: Tensor,
    time_scale_days: float,
    time_mask: Tensor | None = None,
) -> Tensor:
    gamma = candidate_phase_grid(
        delta_days=delta_days,
        alpha=alpha,
        rho_dom=rho_dom,
        time_scale_days=time_scale_days,
    )
    return evaluate_phase_grid(normalized_positions, gamma, time_mask=time_mask)


def translate_grid_function(
    values: Tensor,
    *,
    delta_days: float,
    time_scale_days: float,
    support: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Express a target grid function after the TimeMatch scalar correction.

    If corrected coordinate is ``v = u + delta``, then
    ``f_corrected(v) = f_raw(v-delta)``.  Outside the raw [0,1] support the
    returned curve/support are zero.  A translation has unit derivative, so an
    SRVF receives no additional scale factor.
    """
    if values.ndim not in (1, 2):
        raise ValueError("values must have shape [K] or [K,D]")
    k = values.shape[0]
    grid = canonical_grid(k, dtype=values.dtype).to(values.device)
    query = grid - float(delta_days) / float(time_scale_days)
    inside = (query >= 0.0) & (query <= 1.0)
    q = query.clamp(0.0, 1.0)

    def interp(v: Tensor) -> Tensor:
        idx = torch.searchsorted(grid, q, right=True) - 1
        idx = idx.clamp(0, k - 2)
        x0, x1 = grid[idx], grid[idx + 1]
        w = ((q - x0) / (x1 - x0))
        if v.ndim == 2:
            w = w[:, None]
        out = v[idx] + w * (v[idx + 1] - v[idx])
        if v.ndim == 2:
            return torch.where(inside[:, None], out, torch.zeros_like(out))
        return torch.where(inside, out, torch.zeros_like(out))

    shifted_values = interp(values)
    shifted_support = None if support is None else interp(support)
    return shifted_values, shifted_support


def am_rows_from_probabilities(
    probabilities: np.ndarray,
    alphas: Sequence[float],
    class_distribution_target: np.ndarray,
) -> list[dict]:
    """Official-TimeMatch AM scoring, with alpha replacing the scalar index."""
    p = np.asarray(probabilities, dtype=np.float64)
    a = [float(v) for v in alphas]
    if p.ndim != 3 or p.shape[1] != len(a) or p.shape[2] < 2 or p.shape[0] == 0:
        raise ValueError("probabilities must have shape [N,A,C]")
    if not np.isfinite(p).all():
        raise ValueError("probabilities must be finite")
    c = np.asarray(class_distribution_target, dtype=np.float64)
    if c.shape != (p.shape[2],):
        raise ValueError("class distribution shape mismatch")
    eps = 1e-5
    pred = p.argmax(axis=2)
    one_hot_py = np.zeros((len(a), p.shape[2]), dtype=np.float64)
    for j in range(len(a)):
        one_hot_py[j] = np.bincount(pred[:, j], minlength=p.shape[2]) / float(p.shape[0])
    kl = np.sum(c[None] * (np.log(c[None] + eps) - np.log(one_hot_py + eps)), axis=1)
    entropy = np.mean(np.sum(-p * np.log(p + eps), axis=2), axis=0)
    am = kl + entropy
    inception_py = p.mean(axis=0)
    inception = np.mean(
        np.sum(p * (np.log(p + eps) - np.log(inception_py[None] + eps)), axis=2),
        axis=0,
    )
    rows = []
    best = int(np.argmin(am))
    for j, alpha in enumerate(a):
        rows.append({
            "alpha": float(alpha),
            "am_score": float(am[j]),
            "kl_class_distribution": float(kl[j]),
            "entropy": float(entropy[j]),
            "inception_score": float(inception[j]),
            "selected": bool(j == best),
        })
    return rows


def select_alpha_from_probabilities(
    probabilities: np.ndarray,
    alphas: Sequence[float],
    class_distribution_target: np.ndarray,
) -> AlphaSelection:
    rows = am_rows_from_probabilities(probabilities, alphas, class_distribution_target)
    ordered = sorted(rows, key=lambda r: (float(r["am_score"]), float(r["alpha"])))
    best, second = ordered[0], ordered[1] if len(ordered) > 1 else ordered[0]
    return AlphaSelection(
        alpha=float(best["alpha"]),
        best_am=float(best["am_score"]),
        second_best_am=float(second["am_score"]),
        margin=float(second["am_score"] - best["am_score"]),
        rows=tuple(rows),
    )
