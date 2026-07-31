"""EDEN warm-start gradient reversal on fused representations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _finite_float(name: str, value: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class _GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature: Tensor, coefficient: float) -> Tensor:
        ctx.coefficient = coefficient
        return feature.view_as(feature)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return -ctx.coefficient * gradient, None


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        low: float = 0.0,
        high: float = 1.0,
        max_iters: int = 250,
        weight: float = 1.0,
        auto_step: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = _finite_float("alpha", alpha)
        self.low = _finite_float("low", low)
        self.high = _finite_float("high", high)
        self.max_iters = _positive_int("max_iters", max_iters)
        self.weight = _finite_float("weight", weight)
        if self.alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if self.high < self.low:
            raise ValueError("high must be at least low")
        if self.weight < 0:
            raise ValueError("weight must be nonnegative")
        if not isinstance(auto_step, bool):
            raise ValueError("auto_step must be boolean")
        self.auto_step = auto_step
        self.register_buffer("iteration", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "last_coefficient", torch.tensor(self.low, dtype=torch.float32)
        )

    def get_coefficient(self) -> float:
        iteration = int(self.iteration.item())
        return (
            2.0
            * (self.high - self.low)
            / (1.0 + math.exp(-self.alpha * iteration / self.max_iters))
            - (self.high - self.low)
            + self.low
        )

    def forward(self, feature: Tensor) -> Tensor:
        if not isinstance(feature, Tensor) or not feature.is_floating_point():
            raise ValueError("feature must be a floating-point tensor")
        coefficient = self.get_coefficient()
        self.last_coefficient.fill_(coefficient)
        output = _GradientReverseFunction.apply(
            feature, coefficient * self.weight
        )
        if self.auto_step:
            self.iteration.add_(1)
        return output

    @torch.no_grad()
    def reset(self) -> None:
        self.iteration.zero_()
        self.last_coefficient.fill_(self.low)


class EDENDomainDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        input_dim = _positive_int("input_dim", input_dim)
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, feature: Tensor) -> Tensor:
        if (
            not isinstance(feature, Tensor)
            or feature.ndim != 2
            or feature.shape[1] != self.input_dim
            or not feature.is_floating_point()
        ):
            raise ValueError("feature must have shape [B, input_dim]")
        reference = self.network[0].weight
        if feature.dtype != reference.dtype or feature.device != reference.device:
            raise ValueError("feature and discriminator must share dtype and device")
        if not torch.isfinite(feature).all().item():
            raise ValueError("feature must contain only finite values")
        return self.network(feature)


@dataclass(frozen=True)
class EDENDomainAlignmentOutput:
    loss: Tensor
    logits: Tensor
    labels: Tensor
    accuracy: Tensor
    coefficient: Tensor
    source_batch_size: int
    target_batch_size: int


class EDENFusedFeatureAlignment(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        grl_alpha: float = 1.0,
        grl_low: float = 0.0,
        grl_high: float = 1.0,
        grl_max_iters: int = 250,
        grl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.grl = WarmStartGradientReverseLayer(
            alpha=grl_alpha,
            low=grl_low,
            high=grl_high,
            max_iters=grl_max_iters,
            weight=grl_weight,
        )
        self.discriminator = EDENDomainDiscriminator(feature_dim, hidden_dim)

    def _validate(self, name: str, feature: Tensor) -> None:
        if (
            not isinstance(feature, Tensor)
            or feature.ndim != 2
            or feature.shape[0] == 0
            or feature.shape[1] != self.feature_dim
            or not feature.is_floating_point()
        ):
            raise ValueError(f"{name} must have nonempty shape [B, feature_dim]")

    def forward(
        self, source_feature: Tensor, target_feature: Tensor
    ) -> EDENDomainAlignmentOutput:
        self._validate("source_feature", source_feature)
        self._validate("target_feature", target_feature)
        if source_feature.dtype != target_feature.dtype or source_feature.device != target_feature.device:
            raise ValueError("source and target features must share dtype and device")
        source_batch_size = source_feature.shape[0]
        target_batch_size = target_feature.shape[0]
        feature = torch.cat([source_feature, target_feature], dim=0)
        reversed_feature = self.grl(feature)
        logits = self.discriminator(reversed_feature)
        labels = torch.cat(
            [
                torch.ones(source_batch_size, dtype=torch.long, device=feature.device),
                torch.zeros(target_batch_size, dtype=torch.long, device=feature.device),
            ],
            dim=0,
        )
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        return EDENDomainAlignmentOutput(
            loss=loss,
            logits=logits,
            labels=labels,
            accuracy=accuracy,
            coefficient=self.grl.last_coefficient.to(
                device=feature.device, dtype=feature.dtype
            ).clone(),
            source_batch_size=source_batch_size,
            target_batch_size=target_batch_size,
        )
