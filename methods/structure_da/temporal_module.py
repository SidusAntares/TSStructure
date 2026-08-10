"""Phase-only temporal module with separated task and functional-geometry paths."""

from __future__ import annotations

from torch import Tensor, nn

from .representation import FunctionalGeometryOutput, RawTemporalRepresentation
from .temporal_head import LatentTemporalLTAE
from .temporal_srvf import TemporalSRVFExtractor


class PhaseOnlyTemporalModule(nn.Module):
    """Classify complete PSE latents while T/S decomposition serves geometry only."""

    def __init__(
        self,
        raw_encoder: LatentTemporalLTAE,
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
        latent: Tensor,
        trend: Tensor,
        structure: Tensor,
        positions: Tensor,
        mask: Tensor,
        *,
        raw_positions: Tensor | None = None,
        return_geometry: bool = True,
    ) -> tuple[RawTemporalRepresentation, FunctionalGeometryOutput | None]:
        raw = self.raw_encoder(
            latent=latent,
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
