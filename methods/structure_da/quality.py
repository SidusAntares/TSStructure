"""Standalone quality measurement modules with stopped input gradients."""

from dataclasses import dataclass
import math
from typing import Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class QualityScores:
    """Transferability and discriminability measurements for one input."""

    transferability: torch.Tensor
    entropy: torch.Tensor
    confidence: torch.Tensor
    domain_logits: torch.Tensor
    class_logits: torch.Tensor


@dataclass(frozen=True)
class StructuralQualityOutput:
    """Raw and classification-safe quality gates for one structure."""

    scores: QualityScores
    raw_gate: torch.Tensor
    gate: torch.Tensor


@dataclass(frozen=True)
class ComponentQualityOutput:
    """Raw and classification-safe quality gates for one component."""

    scores: QualityScores
    diversity: torch.Tensor
    raw_base_quality: torch.Tensor
    raw_gate: torch.Tensor
    gate: torch.Tensor


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_features(
    features: torch.Tensor, input_dim: int
) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        raise ValueError("features must be a torch.Tensor")
    if features.ndim != 2 or features.shape[-1] != input_dim:
        raise ValueError(f"features must have shape [B, {input_dim}]")
    if not features.is_floating_point():
        raise ValueError("features must use a floating-point dtype")
    if not torch.isfinite(features).all().item():
        raise ValueError("features must contain only finite values")
    return features.detach()


class TransferabilityScorer(nn.Module):
    """Measure domain ambiguity using a two-class domain prediction head."""

    def __init__(self, input_dim: int, hidden_cap: int = 128) -> None:
        super().__init__()
        self.input_dim = _positive_integer("input_dim", input_dim)
        hidden_cap = _positive_integer("hidden_cap", hidden_cap)
        hidden_dim = min(hidden_cap, self.input_dim)

        self.normalization = nn.LayerNorm(self.input_dim)
        self.hidden = nn.Linear(self.input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.domain_head = nn.Linear(hidden_dim, 2)

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(domain_logits, transferability)`` for ``[B,F]`` input."""

        quality_input = _validate_features(features, self.input_dim)
        hidden = self.activation(self.hidden(self.normalization(quality_input)))
        domain_logits = self.domain_head(hidden)
        domain_probability = torch.softmax(domain_logits, dim=-1)
        p_source = domain_probability[..., 1]
        transferability = 1 - 2 * torch.abs(p_source - 0.5)
        return domain_logits, transferability.clamp(0, 1)


class DiscriminabilityScorer(nn.Module):
    """Measure normalized predictive certainty with a linear class head."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.input_dim = _positive_integer("input_dim", input_dim)
        self.num_classes = _positive_integer("num_classes", num_classes)
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")

        self.normalization = nn.LayerNorm(self.input_dim)
        self.class_head = nn.Linear(self.input_dim, self.num_classes)

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(class_logits, entropy_quality, confidence)``."""

        quality_input = _validate_features(features, self.input_dim)
        class_logits = self.class_head(self.normalization(quality_input))
        probabilities = torch.softmax(class_logits, dim=-1)
        safe_probabilities = probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
        entropy = -torch.sum(
            probabilities * torch.log(safe_probabilities), dim=-1
        )
        entropy_quality = 1 - entropy / math.log(self.num_classes)
        confidence = probabilities.amax(dim=-1)
        return (
            class_logits,
            entropy_quality.clamp(0, 1),
            confidence.clamp(0, 1),
        )


class DiversityScorer(nn.Module):
    """Measure component diversity with an independent sigmoid head."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = _positive_integer("input_dim", input_dim)
        self.normalization = nn.LayerNorm(self.input_dim)
        self.diversity_head = nn.Linear(self.input_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one diversity score per sample."""

        quality_input = _validate_features(features, self.input_dim)
        diversity = self.activation(
            self.diversity_head(self.normalization(quality_input))
        )
        return diversity.squeeze(-1)


class StructuralQualityPerception(nn.Module):
    """Combine independent domain and class measurements for one structure."""

    def __init__(
        self, input_dim: int, num_classes: int, hidden_cap: int = 128
    ) -> None:
        super().__init__()
        self.transferability = TransferabilityScorer(input_dim, hidden_cap)
        self.discriminability = DiscriminabilityScorer(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> StructuralQualityOutput:
        domain_logits, transferability = self.transferability(features)
        class_logits, entropy, confidence = self.discriminability(features)
        scores = QualityScores(
            transferability=transferability,
            entropy=entropy,
            confidence=confidence,
            domain_logits=domain_logits,
            class_logits=class_logits,
        )
        raw_gate = (transferability + entropy + confidence) / 3
        return StructuralQualityOutput(
            scores=scores,
            raw_gate=raw_gate,
            gate=raw_gate.detach(),
        )


class ComponentQualityPerception(nn.Module):
    """Add an independent diversity measurement to component quality."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        eta: float = 0.1,
        hidden_cap: int = 128,
    ) -> None:
        super().__init__()
        if isinstance(eta, bool):
            raise ValueError("eta must be a finite non-negative real number")
        try:
            eta = float(eta)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "eta must be a finite non-negative real number"
            ) from error
        if not math.isfinite(eta) or eta < 0:
            raise ValueError("eta must be a finite non-negative real number")
        self.eta = eta

        self.transferability = TransferabilityScorer(input_dim, hidden_cap)
        self.discriminability = DiscriminabilityScorer(input_dim, num_classes)
        self.diversity = DiversityScorer(input_dim)

    def forward(self, features: torch.Tensor) -> ComponentQualityOutput:
        domain_logits, transferability = self.transferability(features)
        class_logits, entropy, confidence = self.discriminability(features)
        diversity = self.diversity(features)
        scores = QualityScores(
            transferability=transferability,
            entropy=entropy,
            confidence=confidence,
            domain_logits=domain_logits,
            class_logits=class_logits,
        )
        raw_base_quality = (transferability + entropy + confidence) / 3
        raw_gate = (raw_base_quality + self.eta * diversity) / (1 + self.eta)
        return ComponentQualityOutput(
            scores=scores,
            diversity=diversity,
            raw_base_quality=raw_base_quality,
            raw_gate=raw_gate,
            gate=raw_gate.detach(),
        )
