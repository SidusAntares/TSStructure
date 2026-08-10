"""Data structures for the Phase-only TSStructure model."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


def _require_floating(name: str, tensor: Tensor) -> None:
    if not isinstance(tensor, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")


@dataclass(frozen=True)
class RawTemporalRepresentation:
    """Single-stream classification embedding from the full PSE latent process."""

    fused_repr: Tensor
    positions_used: Tensor

    def __post_init__(self) -> None:
        _require_floating("fused_repr", self.fused_repr)
        if self.fused_repr.ndim != 2:
            raise ValueError("fused_repr must have shape [B, d]")
        _require_floating("positions_used", self.positions_used)
        if self.positions_used.ndim != 2:
            raise ValueError("positions_used must have shape [B, L]")
        if self.positions_used.shape[0] != self.fused_repr.shape[0]:
            raise ValueError("positions_used batch must match representation batch")


@dataclass(frozen=True)
class FunctionalGeometryOutput:
    """Deterministic T/S SRVF geometry on a canonical grid."""

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
            if valid.dtype != __import__("torch").bool or valid.shape != (batch_size,):
                raise ValueError(f"{name} must be a boolean tensor with shape [B]")


@dataclass(frozen=True)
class TSStructureForwardOutput:
    """Outputs of the Phase-only classification and geometry paths."""

    logits: Tensor
    fused_repr: Tensor
    latent: Tensor
    trend: Tensor | None
    structure: Tensor | None
    dynamics: Tensor | None
    residual: Tensor | None
    positions: Tensor
    mask: Tensor
    geometry: FunctionalGeometryOutput | None
