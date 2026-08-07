"""K_reg functional-geometry view for class-conditioned T registration.

Round 1/2 established a K_shape=64 canonical grid for the Shape geometry (SRVF
prototypes, support-aware distance, source intra-class distance distributions).
Round 3 introduces a separate K_reg=128 grid used *only* for T-SRVF numerical
registration with fdasrsf DP2 and the resulting gamma. The two grids never mix:

- K_shape=64 stays the sole space for Shape prototype comparison and the
  Stage-1 empirical CDF.
- K_reg=128 is used only for T-SRVF registration and gamma estimation.

The K_reg T-SRVF is produced by re-evaluating the same penalised cubic
B-spline functional fit on a 128-point grid, not by linearly interpolating the
K_shape values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .temporal_srvf import TemporalSRVFExtractor


@dataclass(frozen=True)
class RegistrationGeometryOutput:
    """Trend geometry on the K_reg registration grid for one batch."""

    trend_srvf: Tensor      # [B, K_reg, D]
    trend_support: Tensor   # [B, K_reg]
    trend_valid: Tensor     # [B]
    registration_grid: Tensor  # [K_reg]

    def __post_init__(self) -> None:
        if self.trend_srvf.ndim != 3:
            raise ValueError("trend_srvf must have shape [B, K_reg, D]")
        batch_size = self.trend_srvf.shape[0]
        grid_size = self.trend_srvf.shape[1]
        if self.trend_support.shape != (batch_size, grid_size):
            raise ValueError("trend_support must have shape [B, K_reg]")
        if self.trend_valid.shape != (batch_size,):
            raise ValueError("trend_valid must have shape [B]")
        if self.registration_grid.shape != (grid_size,):
            raise ValueError("registration_grid must have shape [K_reg]")


@dataclass(frozen=True)
class TargetGeometryCache:
    """Per-target-sample cached geometry for the class-conditioned scan.

    Holds only geometry. It never stores target labels, classifier logits,
    pseudo-labels or any derived phase-group state.
    """

    sample_ids: Tensor  # [N]

    trend_srvf_reg: Tensor      # [N, K_reg, D]
    trend_support_reg: Tensor   # [N, K_reg]
    trend_valid: Tensor         # [N]

    structure_srvf_shape: Tensor   # [N, K_shape, D]
    structure_support_shape: Tensor  # [N, K_shape]
    structure_valid: Tensor         # [N]

    registration_grid: Tensor  # [K_reg]
    shape_grid: Tensor         # [K_shape]


@dataclass(frozen=True)
class SourceRegistrationPrototypeBank:
    """Per-class source T-SRVF prototypes on the K_reg registration grid.

    Built once from a deterministic full-source scan with the Stage-1 model.
    This is a runtime statistical object for Round 3 only; it is never written
    back into the Stage-1 checkpoint.
    """

    trend_srvf: Tensor     # [C, K_reg, D]
    trend_support: Tensor  # [C, K_reg]
    class_counts: Tensor   # [C]
    ready: Tensor          # [C] bool
    registration_grid: Tensor  # [K_reg]

    def __post_init__(self) -> None:
        num_classes = self.trend_srvf.shape[0]
        grid_size = self.trend_srvf.shape[1]
        if self.trend_support.shape != (num_classes, grid_size):
            raise ValueError("trend_support must have shape [C, K_reg]")
        if self.class_counts.shape != (num_classes,):
            raise ValueError("class_counts must have shape [C]")
        if not isinstance(self.ready, Tensor) or self.ready.dtype != torch.bool:
            raise ValueError("ready must be a boolean tensor")
        if self.ready.shape != (num_classes,):
            raise ValueError("ready must have shape [C]")
        if self.registration_grid.shape != (grid_size,):
            raise ValueError("registration_grid must have shape [K_reg]")

    def ready_classes(self) -> list[int]:
        return [int(i) for i, r in enumerate(self.ready.tolist()) if r]


def evaluate_registration_geometry(
    trend: Tensor,
    positions: Tensor,
    mask: Tensor,
    reg_extractor: TemporalSRVFExtractor,
) -> RegistrationGeometryOutput:
    """Re-evaluate the trend functional lift on the K_reg grid.

    The extractor re-runs the same B-spline functional fit (same knots and
    smoothing) and evaluates the resulting spline on a 128-point canonical
    grid, so the registration SRVF is not a linear interpolation of K_shape.
    """
    output = reg_extractor(trend, positions, mask)
    registration_grid = reg_extractor.functional_lift.canonical_grid.to(
        device=trend.device, dtype=trend.dtype
    )
    return RegistrationGeometryOutput(
        trend_srvf=output.srvf,
        trend_support=output.support_confidence,
        trend_valid=output.structure_valid,
        registration_grid=registration_grid,
    )
