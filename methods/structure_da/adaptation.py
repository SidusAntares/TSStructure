"""Explicit structural-space adversarial adaptation building blocks."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .structure_ops import vectorize_channel_statistic


def _positive_integer(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _coefficient(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("coefficient must be a finite real number in [0, 1]")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "coefficient must be a finite real number in [0, 1]"
        ) from error
    if not math.isfinite(converted) or not 0 <= converted <= 1:
        raise ValueError("coefficient must be a finite real number in [0, 1]")
    return converted


def _floating_finite_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _structural_matrix(
    name: str,
    value: torch.Tensor,
    batch_size: int,
    width: int,
) -> torch.Tensor:
    value = _floating_finite_tensor(name, value)
    if value.ndim != 2 or value.shape != (batch_size, width):
        raise ValueError(f"{name} must have shape [{batch_size}, {width}]")
    return value


def _quality_gate(
    name: str,
    value: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    value = _floating_finite_tensor(name, value)
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [{batch_size}]")
    if value.device != device:
        raise ValueError(f"{name} must be on the same device as its statistic")
    if not torch.all((value >= 0) & (value <= 1)).item():
        raise ValueError(f"{name} must contain values in [0, 1]")
    return value


@dataclass(frozen=True)
class JointStructuralOutput:
    """Normalized, quality-gated branches and their joint representation."""

    joint: torch.Tensor
    trend_temporal: torch.Tensor
    dynamics_temporal: torch.Tensor
    dynamics_channel: torch.Tensor


class JointStructuralSpaceBuilder(nn.Module):
    """Build the ACON joint space from three explicit structural statistics."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = _positive_integer("feature_dim", feature_dim, minimum=2)
        self.temporal_dim = 4 * self.feature_dim
        self.channel_dim = self.feature_dim * (self.feature_dim - 1) // 2
        self.joint_dim = 2 * self.temporal_dim + self.channel_dim
        self.trend_temporal_norm = nn.LayerNorm(
            self.temporal_dim, elementwise_affine=False
        )
        self.dynamics_temporal_norm = nn.LayerNorm(
            self.temporal_dim, elementwise_affine=False
        )
        self.dynamics_channel_norm = nn.LayerNorm(
            self.channel_dim, elementwise_affine=False
        )

    def forward(
        self,
        trend_temporal_statistic: torch.Tensor,
        dynamics_temporal_statistic: torch.Tensor,
        dynamics_channel_statistic: torch.Tensor,
        beta_trend_temporal: torch.Tensor,
        beta_dynamics_temporal: torch.Tensor,
        beta_dynamics_channel: torch.Tensor,
    ) -> JointStructuralOutput:
        trend_temporal_statistic = _floating_finite_tensor(
            "trend_temporal_statistic", trend_temporal_statistic
        )
        if trend_temporal_statistic.ndim != 2:
            raise ValueError(
                "trend_temporal_statistic must have shape [B, temporal_dim]"
            )
        batch_size = trend_temporal_statistic.shape[0]
        trend_temporal_statistic = _structural_matrix(
            "trend_temporal_statistic",
            trend_temporal_statistic,
            batch_size,
            self.temporal_dim,
        )
        dynamics_temporal_statistic = _structural_matrix(
            "dynamics_temporal_statistic",
            dynamics_temporal_statistic,
            batch_size,
            self.temporal_dim,
        )
        dynamics_channel_statistic = _floating_finite_tensor(
            "dynamics_channel_statistic", dynamics_channel_statistic
        )
        if dynamics_channel_statistic.ndim != 3 or dynamics_channel_statistic.shape != (
            batch_size,
            self.feature_dim,
            self.feature_dim,
        ):
            raise ValueError(
                "dynamics_channel_statistic must have shape [B, feature_dim, feature_dim]"
            )
        statistics = (
            dynamics_temporal_statistic,
            dynamics_channel_statistic,
        )
        if any(
            statistic.device != trend_temporal_statistic.device
            for statistic in statistics
        ):
            raise ValueError("all structural statistics must share a device")

        beta_trend_temporal = _quality_gate(
            "beta_trend_temporal",
            beta_trend_temporal,
            batch_size,
            trend_temporal_statistic.device,
        )
        beta_dynamics_temporal = _quality_gate(
            "beta_dynamics_temporal",
            beta_dynamics_temporal,
            batch_size,
            dynamics_temporal_statistic.device,
        )
        beta_dynamics_channel = _quality_gate(
            "beta_dynamics_channel",
            beta_dynamics_channel,
            batch_size,
            dynamics_channel_statistic.device,
        )

        channel_vector = vectorize_channel_statistic(
            dynamics_channel_statistic
        )
        trend_temporal = beta_trend_temporal.detach()[:, None] * (
            self.trend_temporal_norm(trend_temporal_statistic)
        )
        dynamics_temporal = beta_dynamics_temporal.detach()[:, None] * (
            self.dynamics_temporal_norm(dynamics_temporal_statistic)
        )
        dynamics_channel = beta_dynamics_channel.detach()[:, None] * (
            self.dynamics_channel_norm(channel_vector)
        )
        joint = torch.cat(
            (trend_temporal, dynamics_temporal, dynamics_channel), dim=-1
        )
        return JointStructuralOutput(
            joint=joint,
            trend_temporal=trend_temporal,
            dynamics_temporal=dynamics_temporal,
            dynamics_channel=dynamics_channel,
        )


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return value.clone()

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


def gradient_reverse(value: torch.Tensor, coefficient: float) -> torch.Tensor:
    """Return ``value`` unchanged while reversing its backward gradient."""

    value = _floating_finite_tensor("value", value)
    return _GradientReversal.apply(value, _coefficient(coefficient))


class SDADiscriminator(nn.Module):
    """The shared two-layer binary-logit discriminator for the joint space."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = _positive_integer("input_dim", input_dim)
        self.hidden_dim = _positive_integer("hidden_dim", hidden_dim)
        self.hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(self.hidden_dim, 1)

    def forward(self, joint: torch.Tensor) -> torch.Tensor:
        joint = _floating_finite_tensor("joint", joint)
        if joint.ndim != 2 or joint.shape[1] != self.input_dim:
            raise ValueError(f"joint must have shape [B, {self.input_dim}]")
        return self.output(self.activation(self.hidden(joint))).squeeze(-1)


@dataclass(frozen=True)
class StructuralAdversarialOutput:
    """Source/target logits together with their pre-GRL joint features."""

    source_logits: torch.Tensor
    target_logits: torch.Tensor
    source_joint: torch.Tensor
    target_joint: torch.Tensor


class StructuralAdversarialAdapter(nn.Module):
    """Apply one shared SDA discriminator to source and target joint spaces."""

    def __init__(self, joint_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.joint_dim = _positive_integer("joint_dim", joint_dim)
        self.discriminator = SDADiscriminator(self.joint_dim, hidden_dim)

    def forward(
        self,
        source_joint: torch.Tensor,
        target_joint: torch.Tensor,
        grl_coefficient: float,
    ) -> StructuralAdversarialOutput:
        source_joint = self._validate_joint("source_joint", source_joint)
        target_joint = self._validate_joint("target_joint", target_joint)
        grl_coefficient = _coefficient(grl_coefficient)
        source_logits = self.discriminator(
            gradient_reverse(source_joint, grl_coefficient)
        )
        target_logits = self.discriminator(
            gradient_reverse(target_joint, grl_coefficient)
        )
        return StructuralAdversarialOutput(
            source_logits=source_logits,
            target_logits=target_logits,
            source_joint=source_joint,
            target_joint=target_joint,
        )

    def _validate_joint(self, name: str, value: torch.Tensor) -> torch.Tensor:
        value = _floating_finite_tensor(name, value)
        if value.ndim != 2 or value.shape[1] != self.joint_dim:
            raise ValueError(f"{name} must have shape [B, {self.joint_dim}]")
        return value
