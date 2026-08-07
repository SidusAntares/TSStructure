"""Evidence computation for class-conditioned phase hypotheses.

This module holds the per-sample-per-class candidate record, the empirical CDF
over the Stage-1 source intra-class Shape distances, and the pure helpers that
turn a DP2 gamma plus Shape geometry into hypothesis evidence. Everything here
is a statistical diagnostic: no gradient, no trainable parameters, no
classifier and no target labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .prototype_bank import support_aware_q_distance


@dataclass(frozen=True)
class PairwisePhaseCandidate:
    """One ``sample_id x class_id`` DP2 registration candidate.

    A rejected candidate carries ``legal=False`` and a ``reject_reason`` and
    is never promoted to a Shape/CDF-filtered hypothesis.
    """

    sample_id: int
    class_id: int

    gamma: Tensor

    t_identity_error: float
    t_registered_error: float
    t_gain_ratio: float

    common_support: float

    roughness: float
    min_increment: float
    max_local_speed: float
    phase_deviation: float

    legal: bool
    reject_reason: str | None


@dataclass(frozen=True)
class GammaDiagnostics:
    """Pre-computed gamma statistics for the fair identity/registered compare."""

    gamma: Tensor
    registered_trend: Tensor
    aligned_support: Tensor
    common_support: float
    e_id: float
    e_reg: float
    gain_ratio: float


def _common_support_integral(
    support_a: Tensor,
    support_b: Tensor,
    integration_weights: Tensor,
) -> float:
    """Common support integral ``sum_k w_k * min(support_a[k], support_b[k])``."""
    common = integration_weights * torch.minimum(support_a, support_b)
    return float(common.sum().item())


def empirical_cdf(sorted_source_distances: Tensor, value: Tensor) -> Tensor:
    """Empirical CDF of ``value`` against sorted source distances.

    ``u = count(d_source <= value) / N`` using ``searchsorted``.
    """
    if not isinstance(sorted_source_distances, Tensor) or sorted_source_distances.ndim != 1:
        raise ValueError("sorted_source_distances must be a one-dimensional tensor")
    if not isinstance(value, Tensor):
        raise ValueError("value must be a torch.Tensor")
    n = sorted_source_distances.numel()
    if n == 0:
        return torch.zeros_like(value)
    counts = torch.searchsorted(sorted_source_distances, value, right=True)
    return counts.to(value.dtype) / n


def compute_gamma_diagnostics(
    *,
    sample_id: int,
    class_id: int,
    gamma: Tensor,
    source_trend_srvf: Tensor,
    target_trend_srvf: Tensor,
    source_support: Tensor,
    target_support: Tensor,
    integration_weights: Tensor,
    registration_grid: Tensor,
    adapter,
    eps: float = 1e-8,
) -> GammaDiagnostics:
    """Fairly compare identity vs registered T error on one common support.

    The final common support is established after warping the target support,
    and the *same* support is used for both errors, so a gamma cannot fake an
    improvement by moving support mass.
    """
    from .phase_registration import warp_q_gamma, warp_support_gamma

    aligned_support = warp_support_gamma(
        target_support, gamma, registration_grid
    )
    final_common_support = _common_support_integral(
        source_support, aligned_support, integration_weights
    )
    support_mask = torch.minimum(source_support, aligned_support)  # [K]
    weights = integration_weights * support_mask
    norm = weights.sum() + eps

    def weighted_sq_error(a: Tensor, b: Tensor) -> float:
        diff = (a - b).square().sum(dim=-1)  # [K]
        return float((weights * diff).sum().item() / norm)

    e_id = weighted_sq_error(target_trend_srvf, source_trend_srvf)
    aligned_target = warp_q_gamma(target_trend_srvf.unsqueeze(0), gamma).squeeze(0)
    e_reg = weighted_sq_error(aligned_target, source_trend_srvf)
    gain_ratio = e_reg / (e_id + eps)
    return GammaDiagnostics(
        gamma=gamma,
        registered_trend=aligned_target,
        aligned_support=aligned_support,
        common_support=final_common_support,
        e_id=e_id,
        e_reg=e_reg,
        gain_ratio=gain_ratio,
    )


def shape_distance_to_prototype(
    target_structure_srvf: Tensor,
    target_structure_support: Tensor,
    source_shape_prototype: Tensor,
    source_shape_support: Tensor,
    integration_weights: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[float, bool]:
    """Support-aware Shape distance from an aligned target S-SRVF to a prototype."""
    distance = support_aware_q_distance(
        target_structure_srvf.unsqueeze(0),
        source_shape_prototype.unsqueeze(0),
        target_structure_support.unsqueeze(0),
        source_shape_support.unsqueeze(0),
        integration_weights,
        eps=eps,
    )
    valid = bool(distance.valid[0, 0].item())
    value = float(distance.distance[0, 0].item())
    return value, valid
