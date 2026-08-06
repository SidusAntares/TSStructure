"""Shared data structures for the two-stage structure model.

The module holds only unambiguous dataclasses used by the single forward
chain: the raw classification representation, the functional geometry output
and the top-level model output. No trainable module lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _require_floating(name: str, tensor: Tensor) -> None:
    if not isinstance(tensor, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")


@dataclass(frozen=True)
class RawTemporalRepresentation:
    """Raw per-component and fused classification embeddings."""

    trend_repr: Tensor
    structure_repr: Tensor
    fused_repr: Tensor
    positions_used: Tensor

    def __post_init__(self) -> None:
        for name in ("trend_repr", "structure_repr"):
            _require_floating(name, getattr(self, name))
        if self.trend_repr.shape != self.structure_repr.shape:
            raise ValueError(
                "trend_repr and structure_repr must have identical shapes"
            )
        if self.fused_repr.shape != (
            self.trend_repr.shape[0],
            2 * self.trend_repr.shape[-1],
        ):
            raise ValueError(
                "fused_repr must have shape [B, 2 * component_dim]"
            )
        _require_floating("fused_repr", self.fused_repr)
        if not isinstance(self.positions_used, Tensor) or not self.positions_used.is_floating_point():
            raise ValueError("positions_used must be a floating-point tensor")
        if self.positions_used.shape[0] != self.trend_repr.shape[0]:
            raise ValueError("positions_used batch must match representation batch")


@dataclass(frozen=True)
class FunctionalGeometryOutput:
    """Deterministic vector-valued SRVF geometry on a canonical grid."""

    trend_srvf: Tensor
    structure_srvf: Tensor
    trend_support: Tensor
    structure_support: Tensor
    canonical_grid: Tensor
    trend_valid: Tensor
    structure_valid: Tensor

    def __post_init__(self) -> None:
        if self.trend_srvf.shape != self.structure_srvf.shape:
            raise ValueError("trend_srvf and structure_srvf must share shape")
        if self.trend_srvf.ndim != 3:
            raise ValueError("srvf tensors must have shape [B, K, D]")
        batch_size = self.trend_srvf.shape[0]
        grid_size = self.trend_srvf.shape[1]
        _require_floating("trend_srvf", self.trend_srvf)
        _require_floating("structure_srvf", self.structure_srvf)
        for name, support in (
            ("trend_support", self.trend_support),
            ("structure_support", self.structure_support),
        ):
            _require_floating(name, support)
            if support.shape != (batch_size, grid_size):
                raise ValueError(f"{name} must have shape [B, K]")
        _require_floating("canonical_grid", self.canonical_grid)
        if self.canonical_grid.shape != (grid_size,):
            raise ValueError("canonical_grid must have shape [K]")
        for name, valid in (
            ("trend_valid", self.trend_valid),
            ("structure_valid", self.structure_valid),
        ):
            if (
                not isinstance(valid, Tensor)
                or valid.dtype != torch.bool
                or valid.shape != (batch_size,)
            ):
                raise ValueError(f"{name} must be a boolean tensor with shape [B]")


@dataclass(frozen=True)
class TSStructureForwardOutput:
    """Everything the single forward chain produces for one batch."""

    logits: Tensor
    fused_repr: Tensor
    trend_repr: Tensor
    structure_repr: Tensor
    latent: Tensor
    trend: Tensor
    structure: Tensor
    dynamics: Tensor | None
    residual: Tensor | None
    positions: Tensor
    mask: Tensor
    geometry: FunctionalGeometryOutput | None
