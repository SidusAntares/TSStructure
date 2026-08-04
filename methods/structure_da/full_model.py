"""End-to-end phase-aware T/S/Shape domain-adaptation model."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn

from .backbone import StructureBackbone, StructureBackboneOutput
from .eden_alignment import EDENDomainAlignmentOutput, EDENFusedFeatureAlignment
from .phase_aware_objective import (
    PhaseAwarePrototypeAlignment,
    PhaseAwareSemanticFeatures,
    PrototypeAlignmentConfig,
    PrototypeAlignmentLossOutput,
    PrototypeAlignmentWeights,
    TrendLedGeometryLossOutput,
    TrendLedGeometryObjective,
)
from .representation import (
    PhaseAwareTwoScaleClassifier,
    PhaseAwareTwoScaleClassifierOutput,
)
from .temporal_module import (
    TrendStructureTaskFeatureModule,
    TrendStructureTaskFeatureOutput,
    TrendStructureTemporalCoreOutput,
)


@dataclass(frozen=True)
class StructureAwareForwardOutput:
    backbone: StructureBackboneOutput
    temporal: TrendStructureTaskFeatureOutput
    representation: PhaseAwareTwoScaleClassifierOutput
    semantic: PhaseAwareSemanticFeatures


@dataclass(frozen=True)
class StructureAwareGeometryOutput:
    source_core: TrendStructureTemporalCoreOutput
    target_core: TrendStructureTemporalCoreOutput
    loss: TrendLedGeometryLossOutput

    @property
    def total_loss(self) -> Tensor:
        return self.loss.total_loss


def _options_with_fixed(
    name: str, options: Mapping[str, Any] | None, **fixed: Any
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
    """Use hierarchical T/S geometry and Shape coordinates for adaptation."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 10,
        mlp1: Sequence[int] | None = None,
        pooling: str = "mean_std",
        mlp2: Sequence[int] | None = None,
        with_extra: bool = False,
        extra_size: int = 4,
        shape_dim: int = 128,
        time_reference: float = 0.0,
        time_scale: float = 365.0,
        tau_fast_init: float = 0.05,
        tau_slow_init: float = 0.20,
        tau_min: float = 1e-4,
        delta_tau_min: float = 1e-4,
        temporal_options: Mapping[str, Any] | None = None,
        representation_options: Mapping[str, Any] | None = None,
        prototype_options: Mapping[str, Any] | None = None,
        prototype_weight_options: Mapping[str, Any] | None = None,
        geometry_objective_options: Mapping[str, Any] | None = None,
        alignment_hidden_dim: int = 128,
        grl_alpha: float = 1.0,
        grl_max_iters: int = 250,
        grl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if isinstance(shape_dim, bool) or not isinstance(shape_dim, int) or shape_dim <= 0:
            raise ValueError("shape_dim must be a positive integer")
        self.shape_dim = shape_dim
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
        temporal_kwargs = _options_with_fixed(
            "temporal_options",
            temporal_options,
            feature_dim=self.backbone.feature_dim,
            shape_output_dim=shape_dim,
            time_reference=0.0,
            time_scale=1.0,
        )
        self.temporal_features = TrendStructureTaskFeatureModule(**temporal_kwargs)
        representation_kwargs = _options_with_fixed(
            "representation_options",
            representation_options,
            component_input_dim=self.backbone.feature_dim,
            shape_dim=shape_dim,
            num_classes=num_classes,
            time_reference=0.0,
            time_scale=1.0,
        )
        self.representation = PhaseAwareTwoScaleClassifier(**representation_kwargs)
        prototype_kwargs = _options_with_fixed(
            "prototype_options",
            prototype_options,
            num_classes=num_classes,
            canonical_grid_size=self.temporal_features.core.canonical_grid_size,
            srvf_dim=self.backbone.feature_dim,
            shape_dim=shape_dim,
            raw_dim=self.representation.component_dim,
        )
        prototype_weights = (
            PrototypeAlignmentWeights()
            if prototype_weight_options is None
            else PrototypeAlignmentWeights(**dict(prototype_weight_options))
        )
        self.prototype_alignment = PhaseAwarePrototypeAlignment(
            PrototypeAlignmentConfig(**prototype_kwargs), prototype_weights
        )
        self.geometry_objective = TrendLedGeometryObjective(
            **({} if geometry_objective_options is None else dict(geometry_objective_options))
        )
        expected_fused_dim = 2 * self.representation.component_dim + shape_dim
        if self.representation.fused_dim != expected_fused_dim:
            raise RuntimeError("phase-aware fused dimension is inconsistent")
        self.alignment = EDENFusedFeatureAlignment(
            feature_dim=self.representation.fused_dim,
            hidden_dim=alignment_hidden_dim,
            grl_alpha=grl_alpha,
            grl_max_iters=grl_max_iters,
            grl_weight=grl_weight,
        )

    def _parameter_partition(self) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...]]:
        geometry = tuple(
            parameter
            for parameter in self.temporal_features.warp_parameters()
            if parameter.requires_grad
        )
        geometry_ids = {id(parameter) for parameter in geometry}
        trainable = tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
        task = tuple(parameter for parameter in trainable if id(parameter) not in geometry_ids)
        task_ids = {id(parameter) for parameter in task}
        trainable_ids = {id(parameter) for parameter in trainable}
        if geometry_ids & task_ids or geometry_ids | task_ids != trainable_ids:
            raise RuntimeError("geometry and task parameters must be disjoint and exhaustive")
        return geometry, task

    def geometry_parameters(self) -> Iterator[nn.Parameter]:
        yield from self._parameter_partition()[0]

    def task_parameters(self) -> Iterator[nn.Parameter]:
        yield from self._parameter_partition()[1]

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
        if not isinstance(backbone, StructureBackboneOutput):
            raise ValueError("backbone must be a StructureBackboneOutput")
        trend = backbone.decomposition.trend
        structure = trend + backbone.decomposition.dynamics
        mask = backbone.time_mask[:, :, None]
        return (
            torch.where(mask, trend, torch.zeros_like(trend)),
            torch.where(mask, structure, torch.zeros_like(structure)),
        )

    def _task_positions(
        self, positions: Tensor, backbone: StructureBackboneOutput
    ) -> Tensor:
        if not isinstance(positions, Tensor) or positions.is_complex() or positions.dtype == torch.bool:
            raise ValueError("positions must be a real tensor")
        batch, length = backbone.time_mask.shape
        if positions.ndim == 1:
            if positions.shape != (length,):
                raise ValueError("positions must have shape [L] or [B,L]")
            resolved = positions.unsqueeze(0).expand(batch, -1)
        elif positions.ndim == 2 and positions.shape == (batch, length):
            resolved = positions
        else:
            raise ValueError("positions must have shape [L] or [B,L]")
        del resolved
        return backbone.normalized_positions

    def forward_from_backbone(
        self,
        backbone: StructureBackboneOutput,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        domain_score_weight: float = 1.0,
    ) -> StructureAwareForwardOutput:
        del extra
        trend, structure = self._trend_and_structure(backbone)
        task_positions = self._task_positions(positions, backbone)
        temporal = self.temporal_features(
            trend, structure, task_positions, backbone.time_mask
        )
        representation = self.representation(
            trend,
            structure,
            temporal.shape.feature,
            temporal.aligned_positions,
            time_mask=backbone.time_mask,
            shape_valid=temporal.shape.valid,
            domain_score_weight=domain_score_weight,
        )
        semantic_dtype = temporal.aligned_structure_srvf.dtype
        semantic = PhaseAwareSemanticFeatures(
            aligned_structure_srvf=temporal.aligned_structure_srvf,
            aligned_structure_support=temporal.aligned_structure_support,
            shape_feature=representation.shape_feature.to(dtype=semantic_dtype),
            trend_embedding=representation.trend_embedding.to(dtype=semantic_dtype),
            structure_embedding=representation.structure_embedding.to(
                dtype=semantic_dtype
            ),
            shape_valid=representation.shape_valid,
            component_valid=representation.component_valid,
        )
        return StructureAwareForwardOutput(
            backbone=backbone,
            temporal=temporal,
            representation=representation,
            semantic=semantic,
        )

    def forward_details(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
        domain_score_weight: float = 1.0,
    ) -> StructureAwareForwardOutput:
        backbone = self.forward_backbone(
            pixels, valid_pixels, positions, extra, time_mask=time_mask
        )
        return self.forward_from_backbone(
            backbone, positions, domain_score_weight=domain_score_weight
        )

    def forward(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
        domain_score_weight: float = 1.0,
    ) -> Tensor:
        return self.forward_details(
            pixels,
            valid_pixels,
            positions,
            extra,
            time_mask=time_mask,
            domain_score_weight=domain_score_weight,
        ).representation.logits

    def forward_target_shape_feature_da(
        self, target_output: StructureAwareForwardOutput
    ) -> Tensor:
        if not isinstance(target_output, StructureAwareForwardOutput):
            raise ValueError("target_output must be StructureAwareForwardOutput")
        coordinates = target_output.temporal.coordinates.shape_coordinates.detach()
        valid = target_output.temporal.coordinates.shape_valid
        return self.temporal_features.shape_encoder(
            coordinates, valid, deterministic=False
        ).feature.to(
            dtype=target_output.semantic.shape_feature.dtype
        )

    @torch.no_grad()
    def forward_target_shape_teacher_feature(
        self, target_output: StructureAwareForwardOutput
    ) -> Tensor:
        """Return the dropout-free target Shape confirmation feature."""

        if not isinstance(target_output, StructureAwareForwardOutput):
            raise ValueError("target_output must be StructureAwareForwardOutput")
        coordinates = target_output.temporal.coordinates.shape_coordinates.detach()
        valid = target_output.temporal.coordinates.shape_valid
        return self.temporal_features.shape_encoder(
            coordinates, valid, deterministic=True
        ).feature.to(dtype=target_output.semantic.shape_feature.dtype)

    def forward_geometry_from_backbones(
        self,
        source_backbone: StructureBackboneOutput,
        source_positions: Tensor,
        target_backbone: StructureBackboneOutput,
        target_positions: Tensor,
    ) -> StructureAwareGeometryOutput:
        source_trend, source_structure = self._trend_and_structure(source_backbone)
        target_trend, target_structure = self._trend_and_structure(target_backbone)
        source_positions = self._task_positions(source_positions, source_backbone).float()
        target_positions = self._task_positions(target_positions, target_backbone).float()
        device_type = source_trend.device.type
        if target_trend.device.type != device_type:
            raise ValueError("source and target backbones must share a device")
        with torch.autocast(device_type=device_type, enabled=False):
            source_core = self.temporal_features.core(
                source_trend.detach().float(),
                source_structure.detach().float(),
                source_positions,
                source_backbone.time_mask,
            )
            target_core = self.temporal_features.core(
                target_trend.detach().float(),
                target_structure.detach().float(),
                target_positions,
                target_backbone.time_mask,
            )
            loss = self.geometry_objective.forward_pair(
                source_core.selection, target_core.selection
            )
        return StructureAwareGeometryOutput(source_core, target_core, loss)

    def prototype_losses(
        self,
        source_output: StructureAwareForwardOutput,
        source_labels: Tensor,
        target_output: StructureAwareForwardOutput,
        target_shape_feature_da: Tensor,
        target_shape_teacher_feature: Tensor | None = None,
    ) -> PrototypeAlignmentLossOutput:
        target_semantic = target_output.semantic
        if target_shape_teacher_feature is not None:
            target_semantic = replace(
                target_semantic, shape_feature=target_shape_teacher_feature
            )
        return self.prototype_alignment(
            source_output.semantic,
            source_labels,
            target_semantic,
            target_shape_feature_da,
        )

    @torch.no_grad()
    def update_source_state_from_output(
        self,
        source_output: StructureAwareForwardOutput,
        source_positions: Tensor,
        source_labels: Tensor,
    ) -> None:
        if not isinstance(source_output, StructureAwareForwardOutput):
            raise ValueError("source_output must be StructureAwareForwardOutput")
        positions = self._task_positions(
            source_positions, source_output.backbone
        ).float()
        self.prototype_alignment.update_source_state(
            source_output.semantic, source_labels
        )
        trend, structure = self._trend_and_structure(source_output.backbone)
        self.temporal_features.update_source_state(
            trend.detach().float(),
            structure.detach().float(),
            positions,
            source_output.backbone.time_mask,
        )

    @torch.no_grad()
    def update_source_state(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        source_labels: Tensor,
        extra: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
    ) -> None:
        output = self.forward_details(
            pixels, valid_pixels, positions, extra, time_mask=time_mask
        )
        self.update_source_state_from_output(output, positions, source_labels)

    def align(
        self,
        source_output: StructureAwareForwardOutput,
        target_output: StructureAwareForwardOutput,
    ) -> EDENDomainAlignmentOutput:
        if not isinstance(source_output, StructureAwareForwardOutput) or not isinstance(
            target_output, StructureAwareForwardOutput
        ):
            raise ValueError("source_output and target_output must be StructureAwareForwardOutput")
        return self.alignment(
            source_output.representation.fused_feature,
            target_output.representation.fused_feature,
        )
