"""End-to-end structure-aware model assembled from the current building blocks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .backbone import StructureBackbone, StructureBackboneOutput
from .decomposition import DecompositionOutput
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
    representation: QualityAwareClassifierOutput


@dataclass(frozen=True)
class StructureAwareGeometryOutput:
    temporal: TemporalGeometryPairOutput

    @property
    def total_loss(self) -> Tensor:
        return self.temporal.total_loss


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
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 10,
        mlp1: Sequence[int] | None = None,
        pooling: str = "mean_std",
        mlp2: Sequence[int] | None = None,
        with_extra: bool = False,
        extra_size: int = 4,
        structure_dim: int = 128,
        time_scale: float = 366.0,
        tau_fast_init: float = 0.05,
        tau_slow_init: float = 0.20,
        tau_min: float = 1e-4,
        delta_tau_min: float = 1e-4,
        temporal_options: Mapping[str, Any] | None = None,
        representation_options: Mapping[str, Any] | None = None,
        alignment_hidden_dim: int = 128,
        grl_alpha: float = 1.0,
        grl_max_iters: int = 250,
        grl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.structure_dim = structure_dim
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
            time_scale=time_scale,
        )
        temporal_kwargs = _options_with_fixed(
            "temporal_options",
            temporal_options,
            feature_dim=self.backbone.feature_dim,
            structure_dim=structure_dim,
            time_scale=time_scale,
        )
        self.temporal_operator = SharedTemporalStructureOperator(
            TemporalStructureExtractor(**temporal_kwargs)
        )
        representation_kwargs = _options_with_fixed(
            "representation_options",
            representation_options,
            component_input_dim=self.backbone.feature_dim,
            structure_dim=structure_dim,
            num_classes=num_classes,
        )
        self.representation = QualityAwareComponentClassifier(**representation_kwargs)
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
            for parameter in self.temporal_operator.extractor.warp_parameters()
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
    def detach_backbone_for_state(backbone: StructureBackboneOutput) -> StructureBackboneOutput:
        if not isinstance(backbone, StructureBackboneOutput):
            raise ValueError("backbone must be a StructureBackboneOutput")
        return StructureBackboneOutput(
            tokens=backbone.tokens.detach(),
            time_mask=backbone.time_mask,
            decomposition=DecompositionOutput(
                trend=backbone.decomposition.trend.detach(),
                dynamics=backbone.decomposition.dynamics.detach(),
                residual=backbone.decomposition.residual.detach(),
            ),
        )

    def forward_from_backbone(
        self,
        backbone: StructureBackboneOutput,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        domain_score_weight: float = 1.0,
    ) -> StructureAwareForwardOutput:
        del extra
        if not isinstance(backbone, StructureBackboneOutput):
            raise ValueError("backbone must be a StructureBackboneOutput")
        temporal = self.temporal_operator.forward_task(
            backbone.decomposition.trend,
            backbone.decomposition.dynamics,
            positions,
            backbone.time_mask,
        )
        representation = self.representation(
            backbone.decomposition,
            PairedStructureFeatures.from_temporal(temporal),
            positions,
            backbone.time_mask,
            domain_score_weight=domain_score_weight,
        )
        return StructureAwareForwardOutput(
            backbone=backbone, temporal=temporal, representation=representation
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

    @torch.no_grad()
    def update_source_state_from_backbone(
        self, backbone: StructureBackboneOutput, positions: Tensor
    ) -> None:
        if not isinstance(backbone, StructureBackboneOutput):
            raise ValueError("backbone must be a StructureBackboneOutput")
        self.temporal_operator.update_source_state(
            backbone.decomposition.trend,
            backbone.decomposition.dynamics,
            positions,
            backbone.time_mask,
        )

    @torch.no_grad()
    def update_source_state(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        extra: Tensor | None = None,
        *,
        time_mask: Tensor | None = None,
    ) -> None:
        backbone = self.forward_backbone(
            pixels, valid_pixels, positions, extra, time_mask=time_mask
        )
        self.update_source_state_from_backbone(
            self.detach_backbone_for_state(backbone), positions
        )

    def forward_source_geometry(
        self, source_output: StructureAwareForwardOutput, positions: Tensor
    ) -> StructureAwareGeometryOutput:
        if not isinstance(source_output, StructureAwareForwardOutput):
            raise ValueError("source_output must be StructureAwareForwardOutput")
        del positions
        trend = source_output.backbone.decomposition.trend
        temporal = self.temporal_operator.forward_geometry_from_task(
            source_output.temporal,
            source_mask=torch.ones(trend.shape[0], dtype=torch.bool, device=trend.device),
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
