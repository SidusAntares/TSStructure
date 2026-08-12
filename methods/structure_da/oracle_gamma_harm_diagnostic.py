"""Pure helpers for experiment 08 oracle-gamma harm diagnostics.

This module is diagnostic-only.  It never solves registrations, selects Phase
hypotheses, updates model parameters, or modifies Domain-Phase state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


ALPHA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def shrink_gamma_toward_identity(gamma: Tensor, alpha: float) -> Tensor:
    """Convexly shrink one monotone gamma toward identity.

    ``gamma`` is source->target on a uniform [0,1] grid.  The result preserves
    endpoints and strict monotonicity whenever ``gamma`` is strictly monotone
    and ``alpha`` lies in [0,1].
    """
    if not isinstance(gamma, Tensor) or gamma.ndim != 1 or gamma.numel() < 2:
        raise ValueError("gamma must have shape [K] with K>=2")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    values = gamma.detach().double()
    if not torch.isfinite(values).all().item():
        raise ValueError("gamma must be finite")
    if not torch.all(values[1:] > values[:-1]).item():
        raise ValueError("gamma must be strictly increasing")
    identity = torch.linspace(0.0, 1.0, values.numel(), dtype=values.dtype, device=values.device)
    out = (1.0 - alpha) * identity + alpha * values
    if not torch.all(out[1:] > out[:-1]).item():
        raise RuntimeError("convex gamma shrinkage unexpectedly lost monotonicity")
    return out


def classifier_margin_and_competitor(logits: Tensor, true_class: int) -> tuple[float, int]:
    """Return true-logit minus strongest non-true logit and its class id."""
    if not isinstance(logits, Tensor) or logits.ndim != 1 or logits.numel() < 2:
        raise ValueError("logits must have shape [C] with C>=2")
    true_class = int(true_class)
    if not 0 <= true_class < logits.numel():
        raise ValueError("true_class is outside logits")
    values = logits.detach().float()
    competitor_values = values.clone()
    competitor_values[true_class] = -torch.inf
    competitor = int(torch.argmax(competitor_values).item())
    margin = float((values[true_class] - values[competitor]).item())
    return margin, competitor


def hard_transition(no_correct: bool, gamma_correct: bool) -> str:
    if not bool(no_correct) and bool(gamma_correct):
        return "beneficial_hard"
    if bool(no_correct) and not bool(gamma_correct):
        return "harmful_hard"
    if bool(no_correct) and bool(gamma_correct):
        return "stable_correct"
    return "stable_wrong"


def _linear_interp_uniform(values: Tensor, query: Tensor) -> Tensor:
    if not isinstance(values, Tensor) or values.ndim not in (1, 2) or values.shape[0] < 2:
        raise ValueError("values must have shape [K] or [K,D] with K>=2")
    if not isinstance(query, Tensor) or query.ndim != 1:
        raise ValueError("query must have shape [Q]")
    q = query.to(device=values.device, dtype=values.dtype).clamp(0.0, 1.0)
    scaled = q * (values.shape[0] - 1)
    lower = torch.floor(scaled).long().clamp(0, values.shape[0] - 2)
    upper = lower + 1
    frac = scaled - lower.to(dtype=scaled.dtype)
    if values.ndim == 1:
        out = values[lower] + frac * (values[upper] - values[lower])
    else:
        out = values[lower] + frac.unsqueeze(-1) * (values[upper] - values[lower])
    if torch.any(q == 1.0).item():
        out = out.clone()
        out[q == 1.0] = values[-1]
    return out


def warp_value_function_gamma(values: Tensor, gamma_on_grid: Tensor) -> Tensor:
    """Evaluate target value-space function at source->target gamma positions."""
    if values.ndim != 2:
        raise ValueError("values must have shape [K,D]")
    if gamma_on_grid.ndim != 1 or gamma_on_grid.numel() != values.shape[0]:
        raise ValueError("gamma_on_grid must have shape [K]")
    return _linear_interp_uniform(values, gamma_on_grid)


@dataclass(frozen=True)
class ValueResidualDiagnostic:
    value_residual: float
    level_difference: float
    centered_energy_ratio: float
    affine_residual: float
    affine_residual_ratio: float
    affine_scale: float
    common_support: float


def value_space_residual_diagnostic(
    target_values: Tensor,
    source_values: Tensor,
    target_support: Tensor,
    source_support: Tensor,
    integration_weights: Tensor,
    *,
    eps: float = 1e-8,
) -> ValueResidualDiagnostic:
    """Compare phase-corrected target S values to one source-class reference.

    The affine diagnostic fits one positive global scalar ``a`` and one
    channel-wise constant offset vector ``b``.  It is analysis-only and never
    feeds the classifier.
    """
    if target_values.shape != source_values.shape or target_values.ndim != 2:
        raise ValueError("target_values and source_values must share [K,D]")
    if target_support.shape != target_values.shape[:1] or source_support.shape != target_values.shape[:1]:
        raise ValueError("supports must have shape [K]")
    if integration_weights.shape != target_values.shape[:1]:
        raise ValueError("integration_weights must have shape [K]")
    common = torch.minimum(target_support.double(), source_support.double())
    weights = integration_weights.double() * common
    common_support = float(weights.sum().item())
    if common_support <= eps:
        return ValueResidualDiagnostic(*(float("nan"),) * 6, common_support=common_support)
    weights = weights / weights.sum().clamp_min(eps)
    x = target_values.double()
    y = source_values.double()
    diff = x - y
    value_residual = float((weights.unsqueeze(-1) * diff.square()).sum().item())

    mean_x = (weights.unsqueeze(-1) * x).sum(dim=0)
    mean_y = (weights.unsqueeze(-1) * y).sum(dim=0)
    level_difference = float((mean_x - mean_y).square().sum().item())

    xc = x - mean_x
    yc = y - mean_y
    energy_x = torch.sqrt((weights.unsqueeze(-1) * xc.square()).sum().clamp_min(0.0))
    energy_y = torch.sqrt((weights.unsqueeze(-1) * yc.square()).sum().clamp_min(0.0))
    centered_energy_ratio = float((energy_x / (energy_y + eps)).item())

    numerator = (weights.unsqueeze(-1) * xc * yc).sum()
    denominator = (weights.unsqueeze(-1) * xc.square()).sum().clamp_min(eps)
    a = torch.clamp(numerator / denominator, min=eps)
    b = mean_y - a * mean_x
    affine_diff = a * x + b - y
    affine_residual = float((weights.unsqueeze(-1) * affine_diff.square()).sum().item())
    ratio = float(affine_residual / (value_residual + eps))
    return ValueResidualDiagnostic(
        value_residual=value_residual,
        level_difference=level_difference,
        centered_energy_ratio=centered_energy_ratio,
        affine_residual=affine_residual,
        affine_residual_ratio=ratio,
        affine_scale=float(a.item()),
        common_support=common_support,
    )


def geometry_margin(distances: Tensor, true_class: int) -> tuple[float, int]:
    """Return nearest-competitor distance minus true-class distance."""
    if not isinstance(distances, Tensor) or distances.ndim != 1 or distances.numel() < 2:
        raise ValueError("distances must have shape [C]")
    true_class = int(true_class)
    if not 0 <= true_class < distances.numel():
        raise ValueError("true_class is outside distances")
    values = distances.detach().float()
    competitor_values = values.clone()
    competitor_values[true_class] = torch.inf
    competitor = int(torch.argmin(competitor_values).item())
    if not torch.isfinite(values[true_class]) or not torch.isfinite(values[competitor]):
        return float("nan"), competitor
    return float((values[competitor] - values[true_class]).item()), competitor
