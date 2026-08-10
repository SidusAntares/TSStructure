"""Phase-only TSStructure: TimeMatch-like task path plus frozen functional geometry."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from models.decoder import get_decoder

from .backbone import StructureBackbone, StructureBackboneOutput
from .representation import FunctionalGeometryOutput, TSStructureForwardOutput
from .temporal_head import LatentTemporalLTAE
from .temporal_module import PhaseOnlyTemporalModule
from .temporal_srvf import TemporalSRVFExtractor


class TSStructureModel(nn.Module):
    """PSE -> single LTAE -> classifier, with parallel T/S SRVF geometry.

    The decomposition never feeds the classifier.  It exists only to construct
    trend/structure functional geometry used by Domain Phase and Stable Labels.
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
        time_encoder_type: str = "continuous_time2vec",
        timematch_pe_period: float = 1000.0,
        timematch_pe_max_shift: float = 100.0,
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
        # Phase-only contract: decomposition is a fixed statistical operator.
        # It never receives task-classification gradients.
        for parameter in self.backbone.decomposition.parameters():
            parameter.requires_grad_(False)

        feature_dim = self.backbone.feature_dim
        raw_encoder = LatentTemporalLTAE(
            in_channels=feature_dim,
            n_head=n_head,
            d_k=d_k,
            n_neurons=ltae_mlp,
            dropout=dropout,
            d_model=d_model,
            time_reference=0.0,
            time_scale=1.0,
            max_initial_frequency=max_initial_frequency,
            time_encoder_type=time_encoder_type,
            timematch_pe_period=timematch_pe_period,
            timematch_pe_max_shift=timematch_pe_max_shift,
            calendar_scale_days=time_scale,
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
        self.temporal_module = PhaseOnlyTemporalModule(
            raw_encoder=raw_encoder,
            trend_geometry=trend_geometry,
            structure_geometry=structure_geometry,
        )
        self.classifier = get_decoder(
            [raw_encoder.output_dim, *classifier_hidden], num_classes
        )

    def forward_backbone(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
        compute_decomposition: bool = True,
    ) -> StructureBackboneOutput:
        return self.backbone(
            pixels, valid_pixels, positions, extra, time_mask,
            compute_decomposition=compute_decomposition,
        )

    @staticmethod
    def _trend_and_structure(
        backbone: StructureBackboneOutput,
    ) -> tuple[Tensor, Tensor]:
        if backbone.decomposition is None:
            raise RuntimeError("functional geometry requires decomposition")
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
        return_geometry: bool = False,
    ) -> TSStructureForwardOutput:
        del extra, positions
        mask = backbone.time_mask
        raw_positions = (
            backbone.normalized_positions
            if temporal_positions_override is None
            else temporal_positions_override
        )
        if not return_geometry:
            raw = self.temporal_module.raw_encoder(
                latent=backbone.tokens, positions=raw_positions, mask=mask
            )
            logits = self.classifier(raw.fused_repr)
            return TSStructureForwardOutput(
                logits=logits,
                fused_repr=raw.fused_repr,
                latent=backbone.tokens,
                trend=None,
                structure=None,
                dynamics=None,
                residual=None,
                positions=backbone.normalized_positions,
                mask=mask,
                geometry=None,
            )

        trend, structure = self._trend_and_structure(backbone)
        raw, geometry = self.temporal_module(
            latent=backbone.tokens,
            trend=trend,
            structure=structure,
            positions=backbone.normalized_positions,
            mask=mask,
            raw_positions=temporal_positions_override,
            return_geometry=True,
        )
        logits = self.classifier(raw.fused_repr)
        assert backbone.decomposition is not None
        return TSStructureForwardOutput(
            logits=logits,
            fused_repr=raw.fused_repr,
            latent=backbone.tokens,
            trend=trend,
            structure=structure,
            dynamics=backbone.decomposition.dynamics,
            residual=backbone.decomposition.residual,
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
        return_geometry: bool = False,
    ) -> TSStructureForwardOutput:
        backbone = self.forward_backbone(
            pixels, valid_pixels, positions, extra, time_mask=time_mask,
            compute_decomposition=return_geometry,
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
