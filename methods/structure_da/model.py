"""Component-specific structure assembly with one shared LTAE encoder."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch
from torch import nn

from models.decoder import get_decoder
from models.ltae import LTAE

from .decomposition import DecompositionOutput, SymmetricTimeKernelDecomposition
from .structure_ops import (
    ChannelRelationOperator,
    StructureOutput,
    TemporalRelationOperator,
)


@dataclass(frozen=True)
class ComponentLTAEInputs:
    """The assembled trend, dynamics, and residual LTAE inputs."""

    trend: torch.Tensor
    dynamics: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True)
class ComponentStructureOutput:
    """Classification output and all inspectable component intermediates."""

    logits: torch.Tensor
    fused_embedding: torch.Tensor
    trend_embedding: torch.Tensor
    dynamics_embedding: torch.Tensor
    residual_embedding: torch.Tensor
    decomposition: DecompositionOutput
    trend_temporal: StructureOutput
    dynamics_temporal: StructureOutput
    dynamics_channel: StructureOutput
    ltae_inputs: ComponentLTAEInputs


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _integer_sequence(name: str, values: Sequence[int]) -> tuple:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of positive integers")
    try:
        converted = tuple(values)
    except TypeError as error:
        raise ValueError(
            f"{name} must be a sequence of positive integers"
        ) from error
    for value in converted:
        _positive_integer(name, value)
    return converted


class ComponentStructureClassifier(nn.Module):
    """Assemble component-specific structure and classify shared-LTAE features.

    Trend receives temporal structure, dynamics receives temporal and channel
    structure, and residual receives content only. The three assembled streams
    are encoded by repeated calls to one shared :class:`LTAE` instance.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        time_scale: float = 365.0,
        tau_fast_init: float = 0.05,
        tau_slow_init: float = 0.20,
        tau_min: float = 1e-4,
        delta_tau_min: float = 1e-4,
        n_head: int = 16,
        d_k: int = 8,
        d_model: int = 256,
        ltae_mlp: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        positional_period: int = 1000,
        max_position: int = 365,
        max_temporal_shift: int = 100,
        classifier_hidden: Sequence[int] = (64, 32),
    ) -> None:
        super().__init__()

        self.feature_dim = _positive_integer("feature_dim", feature_dim)
        self.num_classes = _positive_integer("num_classes", num_classes)
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        n_head = _positive_integer("n_head", n_head)
        d_k = _positive_integer("d_k", d_k)
        d_model = _positive_integer("d_model", d_model)
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        ltae_mlp = _integer_sequence("ltae_mlp", ltae_mlp)
        if not ltae_mlp:
            raise ValueError("ltae_mlp must not be empty")
        if ltae_mlp[0] != d_model:
            raise ValueError("ltae_mlp[0] must equal d_model")
        classifier_hidden = _integer_sequence(
            "classifier_hidden", classifier_hidden
        )

        positional_period = _positive_integer(
            "positional_period", positional_period
        )
        self.max_position = _positive_integer("max_position", max_position)
        if (
            isinstance(max_temporal_shift, bool)
            or not isinstance(max_temporal_shift, int)
            or max_temporal_shift < 0
        ):
            raise ValueError("max_temporal_shift must be a non-negative integer")
        self.max_temporal_shift = max_temporal_shift
        self.positional_embedding_size = (
            self.max_position + 2 * self.max_temporal_shift
        )

        try:
            dropout = float(dropout)
        except (TypeError, ValueError) as error:
            raise ValueError("dropout must be a finite number in [0, 1]") from error
        if not math.isfinite(dropout) or not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be a finite number in [0, 1]")

        self.decomposition = SymmetricTimeKernelDecomposition(
            tau_fast_init=tau_fast_init,
            tau_slow_init=tau_slow_init,
            tau_min=tau_min,
            delta_tau_min=delta_tau_min,
            time_scale=time_scale,
        )
        self.temporal_operator = TemporalRelationOperator(time_scale=time_scale)
        self.channel_operator = ChannelRelationOperator(time_scale=time_scale)

        self.content_norm = nn.LayerNorm(
            self.feature_dim, elementwise_affine=False
        )
        self.temporal_norm = nn.LayerNorm(
            2 * self.feature_dim, elementwise_affine=False
        )
        self.channel_norm = nn.LayerNorm(
            self.feature_dim, elementwise_affine=False
        )

        self.shared_ltae = LTAE(
            in_channels=4 * self.feature_dim,
            n_head=n_head,
            d_k=d_k,
            n_neurons=list(ltae_mlp),
            dropout=dropout,
            d_model=d_model,
            T=positional_period,
            max_temporal_shift=self.max_temporal_shift,
            max_position=self.max_position,
        )
        embedding_dim = ltae_mlp[-1]
        self.classifier = get_decoder(
            [3 * embedding_dim, *classifier_hidden], self.num_classes
        )

    def forward(
        self,
        H: torch.Tensor,
        positions: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
    ) -> ComponentStructureOutput:
        """Classify PSE features with component-specific structural inputs."""

        self._validate_features(H)
        self._validate_supported_mask(time_mask, H)
        ltae_positions = self._prepare_ltae_positions(positions, H)

        decomposition = self.decomposition(H, positions, time_mask)
        tau_fast = self.decomposition.tau_fast
        tau_slow = self.decomposition.tau_slow
        trend_temporal = self.temporal_operator(
            decomposition.trend,
            positions,
            tau_fast,
            tau_slow,
            time_mask,
        )
        dynamics_temporal = self.temporal_operator(
            decomposition.dynamics,
            positions,
            tau_fast,
            tau_slow,
            time_mask,
        )
        dynamics_channel = self.channel_operator(
            decomposition.dynamics, positions, time_mask
        )

        zero_channel = torch.zeros_like(decomposition.trend)
        zero_temporal = H.new_zeros(
            H.shape[0], H.shape[1], 2 * self.feature_dim
        )
        ltae_inputs = ComponentLTAEInputs(
            trend=torch.cat(
                [
                    self.content_norm(decomposition.trend),
                    self.temporal_norm(trend_temporal.local),
                    zero_channel,
                ],
                dim=-1,
            ),
            dynamics=torch.cat(
                [
                    self.content_norm(decomposition.dynamics),
                    self.temporal_norm(dynamics_temporal.local),
                    self.channel_norm(dynamics_channel.local),
                ],
                dim=-1,
            ),
            residual=torch.cat(
                [
                    self.content_norm(decomposition.residual),
                    zero_temporal,
                    zero_channel,
                ],
                dim=-1,
            ),
        )

        trend_embedding = self.shared_ltae(ltae_inputs.trend, ltae_positions)
        dynamics_embedding = self.shared_ltae(
            ltae_inputs.dynamics, ltae_positions
        )
        residual_embedding = self.shared_ltae(
            ltae_inputs.residual, ltae_positions
        )
        fused_embedding = torch.cat(
            [trend_embedding, dynamics_embedding, residual_embedding], dim=-1
        )
        logits = self.classifier(fused_embedding)

        return ComponentStructureOutput(
            logits=logits,
            fused_embedding=fused_embedding,
            trend_embedding=trend_embedding,
            dynamics_embedding=dynamics_embedding,
            residual_embedding=residual_embedding,
            decomposition=decomposition,
            trend_temporal=trend_temporal,
            dynamics_temporal=dynamics_temporal,
            dynamics_channel=dynamics_channel,
            ltae_inputs=ltae_inputs,
        )

    def _validate_features(self, H: torch.Tensor) -> None:
        if not isinstance(H, torch.Tensor):
            raise ValueError("H must be a torch.Tensor")
        if H.ndim != 3:
            raise ValueError("H must have shape [B, L, D]")
        if H.shape[-1] != self.feature_dim:
            raise ValueError(
                f"H feature_dim must equal configured feature_dim={self.feature_dim}"
            )
        if H.shape[1] < 1:
            raise ValueError("H must contain at least one time point")
        if not H.is_floating_point():
            raise ValueError("H must use a floating-point dtype")

    @staticmethod
    def _validate_supported_mask(
        time_mask: Optional[torch.Tensor], H: torch.Tensor
    ) -> None:
        if time_mask is None:
            return
        if not isinstance(time_mask, torch.Tensor):
            raise ValueError("time_mask must be a torch.Tensor")
        batch_size, sequence_length, _ = H.shape
        if time_mask.ndim == 1:
            if time_mask.shape[0] != sequence_length:
                raise ValueError("time_mask length must match H")
        elif time_mask.ndim == 2:
            if time_mask.shape != (batch_size, sequence_length):
                raise ValueError("time_mask must have shape [B, L] or [L]")
        else:
            raise ValueError("time_mask must have shape [B, L] or [L]")
        if time_mask.is_complex() or not torch.all(
            (time_mask == 0) | (time_mask == 1)
        ).item():
            raise ValueError("time_mask must contain only boolean or 0/1 values")
        if not torch.all(time_mask == 1).item():
            raise NotImplementedError(
                "partial time_mask values are unsupported because LTAE has no "
                "attention-mask interface"
            )

    def _prepare_ltae_positions(
        self, positions: torch.Tensor, H: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(positions, torch.Tensor):
            raise ValueError("positions must be a torch.Tensor")
        batch_size, sequence_length, _ = H.shape
        if positions.ndim == 1:
            if positions.shape[0] != sequence_length:
                raise ValueError("positions length must match H")
        elif positions.ndim == 2:
            if positions.shape != (batch_size, sequence_length):
                raise ValueError("positions must have shape [B, L] or [L]")
        else:
            raise ValueError("positions must have shape [B, L] or [L]")
        if positions.is_complex() or positions.dtype == torch.bool:
            raise ValueError("positions must contain real numeric timestamps")
        if not torch.isfinite(positions).all().item():
            raise ValueError("positions must contain only finite values")
        if positions.is_floating_point() and not torch.equal(
            positions, torch.round(positions)
        ):
            raise ValueError("LTAE positions must contain exact integer values")

        minimum = -self.max_temporal_shift
        maximum_exclusive = self.max_position + self.max_temporal_shift
        if torch.any(positions < minimum).item() or torch.any(
            positions >= maximum_exclusive
        ).item():
            raise ValueError(
                "positions exceed the LTAE positional embedding range after "
                "max_temporal_shift"
            )
        return positions.to(device=H.device, dtype=torch.long)
