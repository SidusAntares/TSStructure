"""Pure temporal module combining raw encoding and functional geometry.

This module keeps no source/target state: no running references, no accepted
warps, no EMA buffers and no geometry loss. Forward only computes the raw
shared-LTAE representation and, optionally, the deterministic functional
geometry of the trend and structure components.
"""

from __future__ import annotations

from torch import Tensor, nn

from .representation import FunctionalGeometryOutput, RawTemporalRepresentation
from .temporal_head import SharedTrendStructureLTAE
from .temporal_srvf import TemporalSRVFExtractor


class TrendStructureTemporalModule(nn.Module):
    """Encode raw T/S representations and extract functional geometry."""

    def __init__(
        self,
        raw_encoder: SharedTrendStructureLTAE,
        trend_geometry: TemporalSRVFExtractor,
        structure_geometry: TemporalSRVFExtractor,
    ) -> None:
        super().__init__()
        self.raw_encoder = raw_encoder
        self.trend_geometry = trend_geometry
        self.structure_geometry = structure_geometry

    @staticmethod
    def _geometry(
        trend: Tensor,
        structure: Tensor,
        positions: Tensor,
        mask: Tensor,
        trend_geometry: TemporalSRVFExtractor,
        structure_geometry: TemporalSRVFExtractor,
    ) -> FunctionalGeometryOutput:
        trend_functional = trend_geometry(trend, positions, mask)
        structure_functional = structure_geometry(structure, positions, mask)
        canonical_grid = trend_geometry.functional_lift.canonical_grid.to(
            device=trend.device, dtype=trend.dtype
        )
        return FunctionalGeometryOutput(
            trend_srvf=trend_functional.srvf,
            structure_srvf=structure_functional.srvf,
            trend_support=trend_functional.support_confidence,
            structure_support=structure_functional.support_confidence,
            canonical_grid=canonical_grid,
            trend_valid=trend_functional.structure_valid,
            structure_valid=structure_functional.structure_valid,
        )

    def forward(
        self,
        trend: Tensor,
        structure: Tensor,
        positions: Tensor,
        mask: Tensor,
        *,
        raw_positions: Tensor | None = None,
        return_geometry: bool = True,
    ) -> tuple[RawTemporalRepresentation, FunctionalGeometryOutput | None]:
        """Return ``(raw, geometry)``; geometry is ``None`` when disabled.

        Args:
            trend: Temporal trend tokens with shape ``[B, L, D]``.
            structure: Temporal structure tokens with shape ``[B, L, D]``.
            positions: Backbone-normalized shared physical positions ``[B, L]``.
            mask: Boolean validity mask with shape ``[B, L]``.
            return_geometry: Whether to run the functional-geometry path.
        """
        raw = self.raw_encoder(
            trend=trend,
            structure=structure,
            positions=positions if raw_positions is None else raw_positions,
            mask=mask,
        )
        if not return_geometry:
            return raw, None
        geometry = self._geometry(
            trend,
            structure,
            positions,
            mask,
            self.trend_geometry,
            self.structure_geometry,
        )
        return raw, geometry
