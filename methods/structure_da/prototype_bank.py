"""Source class prototypes for Stage 1 prototype-relative supervision.

The prototype bank is a pure statistical summary of the labelled source data.
It is deliberately *not* a ``nn.Parameter`` bank: the prototypes are class
averages of trend SRVFs, Shape SRVFs, supports and fused raw features. Nothing
here is trainable and nothing here participates in an optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

QUANTILE_LEVELS = (0.50, 0.75, 0.95)


@dataclass(frozen=True)
class SourcePrototypeBank:
    """Per-class source prototypes and their distance statistics."""

    trend_srvf: Tensor        # [C, K, D]
    shape_srvf: Tensor        # [C, K, D]
    trend_support: Tensor     # [C, K]
    shape_support: Tensor     # [C, K]
    fused: Tensor             # [C, 2 * d_L]
    class_counts: Tensor      # [C]
    ready: Tensor             # [C] bool
    q_distance_samples: tuple[Tensor, ...]  # per-class sorted intra-class q distances
    f_distance_samples: tuple[Tensor, ...]  # per-class sorted intra-class fused distances
    q_quantiles: Tensor       # [C, 3]
    f_quantiles: Tensor       # [C, 3]
    version: int

    def __post_init__(self) -> None:
        num_classes = self.trend_srvf.shape[0]
        for name, value in (
            ("shape_srvf", self.shape_srvf),
            ("trend_support", self.trend_support),
            ("shape_support", self.shape_support),
            ("fused", self.fused),
            ("class_counts", self.class_counts),
            ("ready", self.ready),
            ("q_quantiles", self.q_quantiles),
            ("f_quantiles", self.f_quantiles),
        ):
            if value.shape[0] != num_classes:
                raise ValueError(
                    f"{name} batch/class dimension must match trend_srvf"
                )
        if not isinstance(self.ready, Tensor) or self.ready.dtype != torch.bool:
            raise ValueError("ready must be a boolean tensor")
        if self.ready.shape != (num_classes,):
            raise ValueError("ready must have shape [C]")
        if self.q_quantiles.shape != (num_classes, len(QUANTILE_LEVELS)):
            raise ValueError("q_quantiles must have shape [C, 3]")
        if self.f_quantiles.shape != (num_classes, len(QUANTILE_LEVELS)):
            raise ValueError("f_quantiles must have shape [C, 3]")
        if len(self.q_distance_samples) != num_classes:
            raise ValueError("q_distance_samples must have one entry per class")
        if len(self.f_distance_samples) != num_classes:
            raise ValueError("f_distance_samples must have one entry per class")
        if isinstance(self.version, bool) or self.version < 0:
            raise ValueError("version must be a nonnegative integer")

    def ready_classes(self) -> list[int]:
        return [int(index) for index, is_ready in enumerate(self.ready.tolist()) if is_ready]


@dataclass(frozen=True)
class SupportAwareDistanceOutput:
    distance_sq: Tensor   # [B, C]
    distance: Tensor      # [B, C]
    valid: Tensor         # [B, C] bool
    common_support: Tensor  # [B, C]


def support_aware_q_distance(
    q_a: Tensor,
    q_b: Tensor,
    support_a: Tensor,
    support_b: Tensor,
    integration_weights: Tensor,
    *,
    eps: float = 1e-8,
    min_common_support: float = 0.0,
) -> SupportAwareDistanceOutput:
    """Vector-valued support-aware SRVF distance.

    Args:
        q_a: Query SRVFs with shape ``[B, K, D]``.
        q_b: Prototype SRVFs with shape ``[C, K, D]``.
        support_a: Query support with shape ``[B, K]``.
        support_b: Prototype support with shape ``[C, K]``.
        integration_weights: Canonical-grid integration weights ``[K]``.
        eps: Denominator floor.
        min_common_support: Minimum shared-integration weight for a pair to be
            considered valid. Defaults to 0.0, i.e. any positive overlap.

    Returns:
        A :class:`SupportAwareDistanceOutput` with invalid pairs flagged by a
        boolean mask instead of a large fake distance.
    """
    if q_a.ndim != 3 or q_b.ndim != 3:
        raise ValueError("q_a and q_b must have shape [B, K, D] and [C, K, D]")
    if q_a.shape[-1] != q_b.shape[-1]:
        raise ValueError("q_a and q_b must share the feature dimension")
    if q_a.shape[1] != q_b.shape[1]:
        raise ValueError("q_a and q_b must share the canonical grid size")
    if support_a.shape != q_a.shape[:2]:
        raise ValueError("support_a must have shape [B, K]")
    if support_b.shape != q_b.shape[:2]:
        raise ValueError("support_b must have shape [C, K]")
    if integration_weights.shape != (q_a.shape[1],):
        raise ValueError("integration_weights must have shape [K]")
    for tensor in (q_a, q_b, support_a, support_b, integration_weights):
        if not tensor.is_floating_point():
            raise ValueError("all distance inputs must use floating-point dtypes")
    if not torch.isfinite(q_a).all().item() or not torch.isfinite(q_b).all().item():
        raise ValueError("q values must be finite")
    if torch.any((support_a < 0) | (support_a > 1)).item():
        raise ValueError("support_a must lie in [0, 1]")
    if torch.any((support_b < 0) | (support_b > 1)).item():
        raise ValueError("support_b must lie in [0, 1]")

    weights = integration_weights.to(device=q_a.device, dtype=q_a.dtype)
    common = weights * torch.minimum(
        support_a.unsqueeze(1), support_b.unsqueeze(0)
    )  # [B, C, K]
    diff_sq = (q_a.unsqueeze(1) - q_b.unsqueeze(0)).square().sum(dim=-1)  # [B, C, K]
    common_support = common.sum(dim=-1)  # [B, C]
    numerator = (common * diff_sq).sum(dim=-1)
    distance_sq = numerator / (common_support + eps)
    distance = torch.sqrt(distance_sq)
    valid = common_support > min_common_support
    return SupportAwareDistanceOutput(
        distance_sq=distance_sq,
        distance=distance,
        valid=valid,
        common_support=common_support,
    )
