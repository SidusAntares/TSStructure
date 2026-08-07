"""End-to-end two-stage structure model: source-only CE classification path."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from models.decoder import get_decoder

from .backbone import StructureBackbone, StructureBackboneOutput
from .representation import (
    FunctionalGeometryOutput,
    TSStructureForwardOutput,
)
from .temporal_head import SharedTrendStructureLTAE
from .temporal_module import TrendStructureTemporalModule
from .temporal_srvf import TemporalSRVFExtractor


class TSStructureModel(nn.Module):
    """Backbone -> decomposition -> shared LTAE raw path -> classifier.

    The forward chain is single-path and domain-free: it never accepts a
    source/target pair, domain labels, warp candidates or quality flags.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 10,
        mlp1: Sequence[int] | None = None,
        pooling: str = "mean_std",
        mlp2: Sequence[int] | None = None,
        with_extra: bool = False,
        extra_size: int = 4,
        time_reference: float = 0.0,
        time_scale: float = 365.0,
        tau_fast_init: float = 0.05,
        tau_slow_init: float = 0.20,
        tau_min: float = 1e-4,
        delta_tau_min: float = 1e-4,
        trend_num_basis: int = 12,
        structure_num_basis: int = 12,
        canonical_grid_size: int = 64,
        roughness_grid_size: int = 256,
        trend_smoothing: float = 1e-2,
        structure_smoothing: float = 1e-3,
        n_head: int = 16,
        d_k: int = 8,
        d_model: int = 256,
        ltae_mlp: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        classifier_hidden: Sequence[int] = (64, 32),
        max_initial_frequency: float = 16.0,
    ) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.backbone = StructureBackbone(
            input_dim=input_dim,
            mlp1=None if mlp1 is None else list(mlp1),
            pooling=pooling,
            mlp2=None if mlp2 is None else list(mlp2),
            with_extra=with_extra,
            extra_size=extra_size,
            tau_fast_init=tau_fast_init,
            tau_slow_init=tau_slow_init,
            tau_min=tau_min,
            delta_tau_min=delta_tau_min,
            time_reference=time_reference,
            time_scale=time_scale,
        )
        feature_dim = self.backbone.feature_dim
        raw_encoder = SharedTrendStructureLTAE(
            in_channels=feature_dim,
            n_head=n_head,
            d_k=d_k,
            n_neurons=ltae_mlp,
            dropout=dropout,
            d_model=d_model,
            time_reference=0.0,
            time_scale=1.0,
            max_initial_frequency=max_initial_frequency,
        )
        trend_geometry = TemporalSRVFExtractor(
            feature_dim=feature_dim,
            num_basis=trend_num_basis,
            canonical_grid_size=canonical_grid_size,
            roughness_grid_size=roughness_grid_size,
            smoothing_weight=trend_smoothing,
            time_reference=0.0,
            time_scale=1.0,
        )
        structure_geometry = TemporalSRVFExtractor(
            feature_dim=feature_dim,
            num_basis=structure_num_basis,
            canonical_grid_size=canonical_grid_size,
            roughness_grid_size=roughness_grid_size,
            smoothing_weight=structure_smoothing,
            time_reference=0.0,
            time_scale=1.0,
        )
        self.temporal_module = TrendStructureTemporalModule(
            raw_encoder=raw_encoder,
            trend_geometry=trend_geometry,
            structure_geometry=structure_geometry,
        )
        self.classifier = get_decoder(
            [2 * raw_encoder.component_dim, *classifier_hidden], num_classes
        )

    def forward_backbone(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
    ) -> StructureBackboneOutput:
        return self.backbone(pixels, valid_pixels, positions, extra, time_mask)

    @staticmethod
    def _trend_and_structure(
        backbone: StructureBackboneOutput,
    ) -> tuple[Tensor, Tensor]:
        trend = backbone.decomposition.trend
        structure = trend + backbone.decomposition.dynamics
        mask = backbone.time_mask[:, :, None]
        return (
            torch.where(mask, trend, torch.zeros_like(trend)),
            torch.where(mask, structure, torch.zeros_like(structure)),
        )

    def forward_from_backbone(
        self,
        backbone: StructureBackboneOutput,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        temporal_positions_override: Tensor | None = None,
        return_geometry: bool = True,
    ) -> TSStructureForwardOutput:
        del extra, positions
        trend, structure = self._trend_and_structure(backbone)
        mask = backbone.time_mask
        raw, geometry = self.temporal_module(
            trend=trend,
            structure=structure,
            positions=backbone.normalized_positions,
            mask=mask,
            raw_positions=temporal_positions_override,
            return_geometry=return_geometry,
        )
        logits = self.classifier(raw.fused_repr)
        dynamics = backbone.decomposition.dynamics
        residual = backbone.decomposition.residual
        return TSStructureForwardOutput(
            logits=logits,
            fused_repr=raw.fused_repr,
            trend_repr=raw.trend_repr,
            structure_repr=raw.structure_repr,
            latent=backbone.tokens,
            trend=trend,
            structure=structure,
            dynamics=dynamics,
            residual=residual,
            positions=backbone.normalized_positions,
            mask=mask,
            geometry=geometry,
        )

    def forward(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
        temporal_positions_override: Tensor | None = None,
        return_geometry: bool = True,
    ) -> TSStructureForwardOutput:
        backbone = self.forward_backbone(
            pixels, valid_pixels, positions, extra, time_mask=time_mask
        )
        return self.forward_from_backbone(
            backbone,
            positions,
            temporal_positions_override=temporal_positions_override,
            return_geometry=return_geometry,
        )

    def encode_geometry(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
    ) -> FunctionalGeometryOutput:
        """Return only the functional geometry of one batch.

        The backbone runs once; the geometry extractors reuse the same
        decomposition without a second PSE pass.
        """
        output = self.forward(
            pixels,
            valid_pixels,
            positions,
            extra,
            time_mask=time_mask,
            return_geometry=True,
        )
        if output.geometry is None:
            raise RuntimeError("geometry was not computed")
        return output.geometry
