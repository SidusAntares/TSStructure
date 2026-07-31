"""End-to-end structure-aware model assembled from the current building blocks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .backbone import StructureBackbone, StructureBackboneOutput
from .channel_module import (
    ChannelStructurePairOutput,
    MultiScaleChannelRelationStructure,
    SharedChannelStructureOperator,
)
from .eden_alignment import EDENDomainAlignmentOutput, EDENFusedFeatureAlignment
from .representation import (
    PairedStructureFeatures,
    QualityAwareClassifierOutput,
    QualityAwareComponentClassifier,
)
from .temporal_module import (
    SharedTemporalStructureOperator,
    TemporalGeometryPairOutput,
    TemporalStructureExtractor,
    TemporalStructurePairOutput,
)


@dataclass(frozen=True)
class StructureAwareForwardOutput:
    backbone: StructureBackboneOutput
    temporal: TemporalStructurePairOutput
    channel: ChannelStructurePairOutput
    representation: QualityAwareClassifierOutput


@dataclass(frozen=True)
class StructureAwareGeometryOutput:
    temporal: TemporalGeometryPairOutput

    @property
    def total_loss(self) -> Tensor:
        return self.temporal.total_loss


def _options_with_fixed(
    name: str,
    options: Mapping[str, Any] | None,
    **fixed: Any,
) -> dict[str, Any]:
    if options is None:
        resolved: dict[str, Any] = {}
    elif isinstance(options, Mapping):
        resolved = dict(options)
    else:
        raise ValueError(f"{name} must be a mapping or None")
    for key, value in fixed.items():
        if key in resolved and resolved[key] != value:
            raise ValueError(f"{name}[{key!r}] conflicts with the shared model setting")
        resolved[key] = value
    return resolved


class StructureAwareDomainAdaptationModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_channels: int = 10,
        channel_feature_dim: int = 16,
        pixel_hidden_dim: int = 16,
        structure_dim: int = 128,
        time_scale: float = 366.0,
        tau_fast_init: float = 0.05,
        tau_slow_init: float = 0.20,
        tau_min: float = 1e-4,
        delta_tau_min: float = 1e-4,
        temporal_options: Mapping[str, Any] | None = None,
        channel_options: Mapping[str, Any] | None = None,
        representation_options: Mapping[str, Any] | None = None,
        alignment_hidden_dim: int = 128,
        grl_alpha: float = 1.0,
        grl_max_iters: int = 250,
        grl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if isinstance(num_channels, bool) or not isinstance(num_channels, int) or num_channels < 2:
            raise ValueError("num_channels must be at least 2")
        self.num_channels = num_channels
        self.structure_dim = structure_dim
        self.backbone = StructureBackbone(
            num_channels=num_channels,
            channel_feature_dim=channel_feature_dim,
            pixel_hidden_dim=pixel_hidden_dim,
            tau_fast_init=tau_fast_init,
            tau_slow_init=tau_slow_init,
            tau_min=tau_min,
            delta_tau_min=delta_tau_min,
            time_scale=time_scale,
        )
        temporal_kwargs = _options_with_fixed(
            "temporal_options",
            temporal_options,
            num_channels=num_channels,
            channel_feature_dim=channel_feature_dim,
            structure_dim=structure_dim,
            time_scale=time_scale,
        )
        self.temporal_operator = SharedTemporalStructureOperator(
            TemporalStructureExtractor(**temporal_kwargs)
        )
        channel_kwargs = _options_with_fixed(
            "channel_options",
            channel_options,
            num_channels=num_channels,
            token_dim=channel_feature_dim,
            structure_dim=structure_dim,
            time_scale=time_scale,
        )
        self.channel_operator = SharedChannelStructureOperator(
            MultiScaleChannelRelationStructure(**channel_kwargs)
        )
        representation_kwargs = _options_with_fixed(
            "representation_options",
            representation_options,
            num_channels=num_channels,
            channel_feature_dim=channel_feature_dim,
            structure_dim=structure_dim,
            num_classes=num_classes,
        )
        self.representation = QualityAwareComponentClassifier(
            **representation_kwargs
        )
        self.alignment = EDENFusedFeatureAlignment(
            feature_dim=self.representation.component_dim + 2 * structure_dim,
            hidden_dim=alignment_hidden_dim,
            grl_alpha=grl_alpha,
            grl_max_iters=grl_max_iters,
            grl_weight=grl_weight,
        )

    def _resolve_channel_mask(
        self,
        channel_mask: Tensor | None,
        time_mask: Tensor,
    ) -> Tensor:
        batch_size, sequence_length = time_mask.shape
        if channel_mask is None:
            return time_mask.unsqueeze(-1).expand(-1, -1, self.num_channels)
        if not isinstance(channel_mask, Tensor):
            raise ValueError("channel_mask must be a torch.Tensor or None")
        if channel_mask.ndim == 2:
            if channel_mask.shape != (sequence_length, self.num_channels):
                raise ValueError("channel_mask must have shape [L, C] or [B, L, C]")
            channel_mask = channel_mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif channel_mask.ndim == 3:
            if channel_mask.shape != (batch_size, sequence_length, self.num_channels):
                raise ValueError("channel_mask must have shape [L, C] or [B, L, C]")
        else:
            raise ValueError("channel_mask must have shape [L, C] or [B, L, C]")
        if channel_mask.is_complex() or (
            channel_mask.dtype != torch.bool
            and (
                not torch.isfinite(channel_mask).all().item()
                or not torch.all((channel_mask == 0) | (channel_mask == 1)).item()
            )
        ):
            raise ValueError("channel_mask must contain only finite 0/1 values")
        resolved = channel_mask.to(device=time_mask.device, dtype=torch.bool)
        return resolved & time_mask.unsqueeze(-1)

    def _backbone_and_mask(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        time_mask: Tensor | None,
        channel_mask: Tensor | None,
    ) -> tuple[StructureBackboneOutput, Tensor]:
        backbone = self.backbone(pixels, valid_pixels, positions, time_mask)
        return backbone, self._resolve_channel_mask(channel_mask, backbone.time_mask)

    def forward_details(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
        channel_mask: Tensor | None = None,
    ) -> StructureAwareForwardOutput:
        del extra
        backbone, resolved_channel_mask = self._backbone_and_mask(
            pixels, valid_pixels, positions, time_mask, channel_mask
        )
        temporal = self.temporal_operator.forward_task(
            backbone.decomposition.trend,
            backbone.decomposition.dynamics,
            positions,
            backbone.time_mask,
        )
        channel = self.channel_operator(
            backbone.decomposition.trend,
            backbone.decomposition.dynamics,
            positions,
            backbone.time_mask,
            resolved_channel_mask,
        )
        representation = self.representation(
            backbone.decomposition,
            PairedStructureFeatures.from_temporal(temporal),
            PairedStructureFeatures.from_channel(channel),
            positions,
            backbone.time_mask,
        )
        return StructureAwareForwardOutput(
            backbone=backbone,
            temporal=temporal,
            channel=channel,
            representation=representation,
        )

    def forward(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        return self.forward_details(
            pixels,
            valid_pixels,
            positions,
            extra,
            time_mask=time_mask,
            channel_mask=channel_mask,
        ).representation.logits

    @torch.no_grad()
    def update_source_state(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
        channel_mask: Tensor | None = None,
    ) -> None:
        del extra
        backbone, resolved_channel_mask = self._backbone_and_mask(
            pixels, valid_pixels, positions, time_mask, channel_mask
        )
        self.temporal_operator.update_source_state(
            backbone.decomposition.trend,
            backbone.decomposition.dynamics,
            positions,
            backbone.time_mask,
        )
        self.channel_operator.update_source_state(
            backbone.decomposition.trend,
            backbone.decomposition.dynamics,
            positions,
            backbone.time_mask,
            resolved_channel_mask,
        )

    def forward_source_geometry(
        self,
        source_output: StructureAwareForwardOutput,
        positions: Tensor,
    ) -> StructureAwareGeometryOutput:
        if not isinstance(source_output, StructureAwareForwardOutput):
            raise ValueError("source_output must be StructureAwareForwardOutput")
        trend = source_output.backbone.decomposition.trend
        temporal = self.temporal_operator.forward_geometry(
            trend,
            source_output.backbone.decomposition.dynamics,
            positions,
            source_output.backbone.time_mask,
            source_mask=torch.ones(
                trend.shape[0], dtype=torch.bool, device=trend.device
            ),
        )
        return StructureAwareGeometryOutput(temporal=temporal)

    def align(
        self,
        source_output: StructureAwareForwardOutput,
        target_output: StructureAwareForwardOutput,
    ) -> EDENDomainAlignmentOutput:
        if not isinstance(source_output, StructureAwareForwardOutput) or not isinstance(target_output, StructureAwareForwardOutput):
            raise ValueError("source_output and target_output must be StructureAwareForwardOutput")
        return self.alignment(
            source_output.representation.fused_feature,
            target_output.representation.fused_feature,
        )
