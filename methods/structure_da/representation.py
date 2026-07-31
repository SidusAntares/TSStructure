"""Shared-LTAE component representation with hierarchical quality fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from models.decoder import get_decoder
from models.ltae import ComponentAwareSharedLTAE

from .channel_module import ChannelStructurePairOutput
from .decomposition import DecompositionOutput
from .quality_fusion import HierarchicalQualityFusion, HierarchicalQualityOutput
from .temporal_module import TemporalStructurePairOutput


@dataclass(frozen=True)
class PairedStructureFeatures:
    trend: Tensor
    dynamics: Tensor
    trend_valid: Tensor
    dynamics_valid: Tensor

    @classmethod
    def from_temporal(
        cls, output: TemporalStructurePairOutput
    ) -> "PairedStructureFeatures":
        if not isinstance(output, TemporalStructurePairOutput):
            raise ValueError("output must be a TemporalStructurePairOutput")
        return cls(
            trend=output.trend.encoded.feature,
            dynamics=output.dynamics.encoded.feature,
            trend_valid=output.trend.encoded.valid,
            dynamics_valid=output.dynamics.encoded.valid,
        )

    @classmethod
    def from_channel(
        cls, output: ChannelStructurePairOutput
    ) -> "PairedStructureFeatures":
        if not isinstance(output, ChannelStructurePairOutput):
            raise ValueError("output must be a ChannelStructurePairOutput")
        return cls(
            trend=output.trend.feature,
            dynamics=output.dynamics.feature,
            trend_valid=output.trend.valid,
            dynamics_valid=output.dynamics.valid,
        )


@dataclass(frozen=True)
class QualityAwareClassifierOutput:
    logits: Tensor
    fused_feature: Tensor
    trend_embedding: Tensor
    dynamics_embedding: Tensor
    residual_embedding: Tensor
    temporal_features: PairedStructureFeatures
    channel_features: PairedStructureFeatures
    quality: HierarchicalQualityOutput
    component_valid: Tensor
    ltae_positions: Tensor
    time_mask: Tensor


def _positive_int(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _resolve_time_mask(
    time_mask: Tensor | None,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> Tensor:
    if time_mask is None:
        return torch.ones(
            batch_size, sequence_length, dtype=torch.bool, device=device
        )
    if not isinstance(time_mask, Tensor):
        raise ValueError("time_mask must be a torch.Tensor or None")
    if time_mask.ndim == 1:
        if time_mask.shape != (sequence_length,):
            raise ValueError("time_mask must have shape [L] or [B, L]")
        time_mask = time_mask.unsqueeze(0).expand(batch_size, -1)
    elif time_mask.ndim == 2:
        if time_mask.shape != (batch_size, sequence_length):
            raise ValueError("time_mask must have shape [L] or [B, L]")
    else:
        raise ValueError("time_mask must have shape [L] or [B, L]")
    if time_mask.is_complex() or (
        time_mask.dtype != torch.bool
        and (
            not torch.isfinite(time_mask).all().item()
            or not torch.all((time_mask == 0) | (time_mask == 1)).item()
        )
    ):
        raise ValueError("time_mask must contain only finite 0/1 values")
    return time_mask.to(device=device, dtype=torch.bool)


class QualityAwareComponentClassifier(nn.Module):
    """Encode T/D/R with component-aware stems and one shared attention body."""

    def __init__(
        self,
        num_channels: int,
        channel_feature_dim: int,
        structure_dim: int,
        num_classes: int,
        n_head: int = 16,
        d_k: int = 8,
        d_model: int = 256,
        ltae_mlp: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        positional_period: int = 1000,
        max_position: int = 365,
        max_temporal_shift: int = 100,
        classifier_hidden: Sequence[int] = (64, 32),
        quality_domain_hidden_dim: int = 128,
        quality_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_channels = _positive_int("num_channels", num_channels)
        self.channel_feature_dim = _positive_int(
            "channel_feature_dim", channel_feature_dim
        )
        self.structure_dim = _positive_int("structure_dim", structure_dim)
        self.num_classes = _positive_int("num_classes", num_classes, minimum=2)
        n_head = _positive_int("n_head", n_head)
        d_k = _positive_int("d_k", d_k)
        d_model = _positive_int("d_model", d_model)
        positional_period = _positive_int("positional_period", positional_period)
        self.max_position = _positive_int("max_position", max_position)
        if (
            isinstance(max_temporal_shift, bool)
            or not isinstance(max_temporal_shift, int)
            or max_temporal_shift < 0
        ):
            raise ValueError("max_temporal_shift must be a nonnegative integer")
        self.max_temporal_shift = max_temporal_shift
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        try:
            ltae_mlp = tuple(ltae_mlp)
            classifier_hidden = tuple(classifier_hidden)
        except TypeError as error:
            raise ValueError("ltae_mlp and classifier_hidden must be sequences") from error
        if (
            not ltae_mlp
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in ltae_mlp)
            or ltae_mlp[0] != d_model
        ):
            raise ValueError(
                "ltae_mlp must be nonempty, positive, and start with d_model"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in classifier_hidden
        ):
            raise ValueError("classifier_hidden must contain positive integers")
        try:
            dropout = float(dropout)
        except (TypeError, ValueError) as error:
            raise ValueError("dropout must lie in [0, 1)") from error
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        raw_component_dim = num_channels * channel_feature_dim
        self.component_ltae = ComponentAwareSharedLTAE(
            in_channels=raw_component_dim,
            n_head=n_head,
            d_k=d_k,
            n_neurons=list(ltae_mlp),
            dropout=dropout,
            d_model=d_model,
            T=positional_period,
            max_temporal_shift=max_temporal_shift,
            max_position=max_position,
        )
        self.component_dim = ltae_mlp[-1]
        self.quality_fusion = HierarchicalQualityFusion(
            component_dim=self.component_dim,
            structure_dim=structure_dim,
            num_classes=num_classes,
            domain_hidden_dim=quality_domain_hidden_dim,
            eps=quality_eps,
        )
        self.classifier = get_decoder(
            [
                self.component_dim + 2 * structure_dim,
                *classifier_hidden,
            ],
            num_classes,
        )

    def _validate_decomposition(
        self,
        decomposition: DecompositionOutput,
        time_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(decomposition, DecompositionOutput):
            raise ValueError("decomposition must be a DecompositionOutput")
        components = (
            decomposition.trend,
            decomposition.dynamics,
            decomposition.residual,
        )
        expected_shape = (
            time_mask.shape[0],
            time_mask.shape[1],
            self.num_channels,
            self.channel_feature_dim,
        )
        reference_parameter = next(self.component_ltae.parameters())
        safe_components = []
        for component in components:
            if (
                not isinstance(component, Tensor)
                or component.shape != expected_shape
                or component.ndim != 4
            ):
                raise ValueError(
                    "decomposition components must have identical shape [B, L, C, P]"
                )
            if not component.is_floating_point():
                raise ValueError("decomposition components must be floating point")
            if component.dtype != reference_parameter.dtype or component.device != reference_parameter.device:
                raise ValueError(
                    "decomposition and classifier must use the same dtype and device"
                )
            safe = torch.where(
                time_mask[:, :, None, None],
                component,
                torch.zeros_like(component),
            )
            if not torch.isfinite(safe).all().item():
                raise ValueError("valid decomposition values must be finite")
            safe_components.append(safe)
        return tuple(safe_components)

    def _validate_structure_features(
        self,
        features: PairedStructureFeatures,
        name: str,
        batch_size: int,
        reference: Tensor,
    ) -> None:
        if not isinstance(features, PairedStructureFeatures):
            raise ValueError(f"{name} must be PairedStructureFeatures")
        for feature in (features.trend, features.dynamics):
            if (
                not isinstance(feature, Tensor)
                or feature.shape != (batch_size, self.structure_dim)
                or not feature.is_floating_point()
            ):
                raise ValueError(
                    f"{name} features must have shape [B, structure_dim]"
                )
            if feature.dtype != reference.dtype or feature.device != reference.device:
                raise ValueError(
                    f"{name} features must match decomposition dtype and device"
                )
            if not torch.isfinite(feature).all().item():
                raise ValueError(f"{name} features must be finite")
        for valid in (features.trend_valid, features.dynamics_valid):
            if (
                not isinstance(valid, Tensor)
                or valid.dtype != torch.bool
                or valid.shape != (batch_size,)
                or valid.device != reference.device
            ):
                raise ValueError(
                    f"{name} valid masks must be boolean tensors with shape [B]"
                )

    def _resolve_positions(
        self,
        positions: Tensor,
        time_mask: Tensor,
        device: torch.device,
    ) -> Tensor:
        batch_size, sequence_length = time_mask.shape
        if not isinstance(positions, Tensor):
            raise ValueError("positions must be a torch.Tensor")
        if positions.is_complex() or positions.dtype == torch.bool:
            raise ValueError("positions must contain real numeric values")
        if positions.ndim == 1:
            if positions.shape != (sequence_length,):
                raise ValueError("positions must have shape [L] or [B, L]")
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        elif positions.ndim == 2:
            if positions.shape != (batch_size, sequence_length):
                raise ValueError("positions must have shape [L] or [B, L]")
        else:
            raise ValueError("positions must have shape [L] or [B, L]")
        positions = positions.to(device=device)
        valid_positions = positions[time_mask]
        if not torch.isfinite(valid_positions).all().item():
            raise ValueError("valid positions must be finite")
        if positions.is_floating_point() and not torch.equal(
            valid_positions, valid_positions.round()
        ):
            raise ValueError("valid positions must be exact integer values")
        lower = -self.max_temporal_shift
        upper = self.max_position + self.max_temporal_shift
        if valid_positions.numel() and (
            (valid_positions < lower).any().item()
            or (valid_positions >= upper).any().item()
        ):
            raise ValueError("valid positions are outside the supported range")
        safe_positions = torch.where(
            time_mask, positions, torch.zeros_like(positions)
        )
        return safe_positions.to(dtype=torch.long)

    def forward(
        self,
        decomposition: DecompositionOutput,
        temporal_features: PairedStructureFeatures,
        channel_features: PairedStructureFeatures,
        positions: Tensor,
        time_mask: Tensor | None = None,
        domain_score_weight: float = 1.0,
    ) -> QualityAwareClassifierOutput:
        if not isinstance(decomposition, DecompositionOutput):
            raise ValueError("decomposition must be a DecompositionOutput")
        if not isinstance(decomposition.trend, Tensor) or decomposition.trend.ndim != 4:
            raise ValueError("decomposition components must have shape [B, L, C, P]")
        batch_size, sequence_length = decomposition.trend.shape[:2]
        resolved_mask = _resolve_time_mask(
            time_mask,
            batch_size,
            sequence_length,
            decomposition.trend.device,
        )
        trend, dynamics, residual = self._validate_decomposition(
            decomposition, resolved_mask
        )
        self._validate_structure_features(
            temporal_features, "temporal_features", batch_size, trend
        )
        self._validate_structure_features(
            channel_features, "channel_features", batch_size, trend
        )
        ltae_positions = self._resolve_positions(
            positions, resolved_mask, trend.device
        )
        trend_sequence = trend.flatten(start_dim=2)
        dynamics_sequence = dynamics.flatten(start_dim=2)
        residual_sequence = residual.flatten(start_dim=2)
        trend_embedding, dynamics_embedding, residual_embedding = (
            self.component_ltae(
                trend_sequence,
                dynamics_sequence,
                residual_sequence,
                ltae_positions,
                time_mask=resolved_mask,
            )
        )
        component_valid = resolved_mask.any(dim=-1)
        quality = self.quality_fusion(
            trend_embedding,
            dynamics_embedding,
            residual_embedding,
            temporal_features.trend,
            temporal_features.dynamics,
            channel_features.trend,
            channel_features.dynamics,
            component_valid,
            temporal_features.trend_valid,
            temporal_features.dynamics_valid,
            channel_features.trend_valid,
            channel_features.dynamics_valid,
            domain_score_weight,
        )
        logits = self.classifier(quality.fused_feature)
        return QualityAwareClassifierOutput(
            logits=logits,
            fused_feature=quality.fused_feature,
            trend_embedding=trend_embedding,
            dynamics_embedding=dynamics_embedding,
            residual_embedding=residual_embedding,
            temporal_features=temporal_features,
            channel_features=channel_features,
            quality=quality,
            component_valid=component_valid,
            ltae_positions=ltae_positions,
            time_mask=resolved_mask,
        )
