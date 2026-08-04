"""Two-scale quality scoring, semantic fusion, and auxiliary objective."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QualityScoreOutput:
    coefficient: Tensor
    domain_invariance: Tensor
    entropy_score: Tensor
    confidence_score: Tensor
    discriminability: Tensor
    domain_logits: Tensor
    class_logits: Tensor
    valid: Tensor


@dataclass(frozen=True)
class TwoScaleQualityOutput:
    trend: QualityScoreOutput
    structure: QualityScoreOutput

    alpha_trend: Tensor
    alpha_structure: Tensor

    weighted_trend: Tensor
    weighted_structure: Tensor
    shape_feature: Tensor

    fused_feature: Tensor


@dataclass(frozen=True)
class TwoScaleQualityLossOutput:
    total_loss: Tensor
    classification_loss: Tensor
    domain_loss: Tensor
    classification_count: Tensor
    domain_count: Tensor


def _concatenate_batch_tensors(name: str, source: Tensor, target: Tensor) -> Tensor:
    if not isinstance(source, Tensor) or not isinstance(target, Tensor):
        raise ValueError(f"{name} values must be tensors")
    if source.ndim == 0 or target.ndim == 0 or source.shape[1:] != target.shape[1:]:
        raise ValueError(f"{name} non-batch dimensions must match")
    if source.dtype != target.dtype or source.device != target.device:
        raise ValueError(f"{name} dtype and device must match")
    return torch.cat([source, target], dim=0)


def _concatenate_score(
    name: str, source: QualityScoreOutput, target: QualityScoreOutput
) -> QualityScoreOutput:
    if not isinstance(source, QualityScoreOutput) or not isinstance(target, QualityScoreOutput):
        raise ValueError(f"{name} values must be QualityScoreOutput")
    return QualityScoreOutput(
        coefficient=_concatenate_batch_tensors(f"{name}.coefficient", source.coefficient, target.coefficient),
        domain_invariance=_concatenate_batch_tensors(f"{name}.domain_invariance", source.domain_invariance, target.domain_invariance),
        entropy_score=_concatenate_batch_tensors(f"{name}.entropy_score", source.entropy_score, target.entropy_score),
        confidence_score=_concatenate_batch_tensors(f"{name}.confidence_score", source.confidence_score, target.confidence_score),
        discriminability=_concatenate_batch_tensors(f"{name}.discriminability", source.discriminability, target.discriminability),
        domain_logits=_concatenate_batch_tensors(f"{name}.domain_logits", source.domain_logits, target.domain_logits),
        class_logits=_concatenate_batch_tensors(f"{name}.class_logits", source.class_logits, target.class_logits),
        valid=_concatenate_batch_tensors(f"{name}.valid", source.valid, target.valid),
    )


def concatenate_two_scale_quality_outputs(
    source: TwoScaleQualityOutput,
    target: TwoScaleQualityOutput,
) -> TwoScaleQualityOutput:
    """Concatenate source-first two-scale outputs while preserving autograd."""
    if not isinstance(source, TwoScaleQualityOutput) or not isinstance(
        target, TwoScaleQualityOutput
    ):
        raise ValueError("source and target must be TwoScaleQualityOutput")
    return TwoScaleQualityOutput(
        trend=_concatenate_score("trend", source.trend, target.trend),
        structure=_concatenate_score(
            "structure", source.structure, target.structure
        ),
        alpha_trend=_concatenate_batch_tensors(
            "alpha_trend", source.alpha_trend, target.alpha_trend
        ),
        alpha_structure=_concatenate_batch_tensors(
            "alpha_structure", source.alpha_structure, target.alpha_structure
        ),
        weighted_trend=_concatenate_batch_tensors(
            "weighted_trend", source.weighted_trend, target.weighted_trend
        ),
        weighted_structure=_concatenate_batch_tensors(
            "weighted_structure",
            source.weighted_structure,
            target.weighted_structure,
        ),
        shape_feature=_concatenate_batch_tensors(
            "shape_feature", source.shape_feature, target.shape_feature
        ),
        fused_feature=_concatenate_batch_tensors(
            "fused_feature", source.fused_feature, target.fused_feature
        ),
    )


def _positive_int(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _positive_float(name: str, value: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return converted


def _unit_interval_float(name: str, value: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must lie in [0, 1]") from error
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return converted


def _resolve_valid(valid: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
    if valid is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    if not isinstance(valid, Tensor) or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean tensor with shape [B]")
    if valid.shape != (batch_size,):
        raise ValueError("valid must be a boolean tensor with shape [B]")
    if valid.device != device:
        raise ValueError("valid and feature must use the same device")
    return valid


def _autocast_enabled_for(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.is_autocast_enabled()
    if device.type == "cpu":
        return torch.is_autocast_cpu_enabled()
    return False


def _to_quality_master_dtype(
    name: str,
    feature: Tensor,
    reference: Tensor,
) -> Tensor:
    if feature.device != reference.device:
        raise ValueError(f"{name} device must match quality scorer device")
    if feature.dtype == reference.dtype:
        return feature
    if not _autocast_enabled_for(feature.device):
        raise ValueError(
            f"{name} dtype must match quality scorer dtype when autocast is disabled"
        )
    if reference.dtype != torch.float32:
        raise ValueError(
            f"{name} mixed-precision input requires float32 master parameters"
        )
    if feature.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"{name} uses an unsupported autocast dtype")
    return feature.float()


class QualityScorer(nn.Module):
    """Predict class/domain logits and derive one bounded quality coefficient."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        domain_hidden_dim: int = 128,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int("input_dim", input_dim)
        self.num_classes = _positive_int("num_classes", num_classes, minimum=2)
        domain_hidden_dim = _positive_int(
            "domain_hidden_dim", domain_hidden_dim
        )
        self.eps = _positive_float("eps", eps)
        self.class_classifier = nn.Linear(input_dim, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, domain_hidden_dim),
            nn.ReLU(),
            nn.Linear(domain_hidden_dim, domain_hidden_dim),
            nn.ReLU(),
            nn.Linear(domain_hidden_dim, 2),
        )

    def forward(
        self,
        feature: Tensor,
        valid: Tensor | None = None,
        domain_score_weight: float = 1.0,
    ) -> QualityScoreOutput:
        if not isinstance(feature, Tensor) or feature.ndim != 2:
            raise ValueError("feature must have shape [B, input_dim]")
        if not feature.is_floating_point() or feature.shape[1] != self.input_dim:
            raise ValueError("feature must be floating point with input_dim columns")
        reference = self.class_classifier.weight
        if feature.device != reference.device or feature.dtype != reference.dtype:
            raise ValueError("feature and scorer must use the same dtype and device")
        if not torch.isfinite(feature).all().item():
            raise ValueError("feature must contain only finite values")
        resolved_valid = _resolve_valid(valid, feature.shape[0], feature.device)
        domain_score_weight = _unit_interval_float(
            "domain_score_weight", domain_score_weight
        )

        probe_feature = feature.detach()
        class_logits = self.class_classifier(probe_feature)
        domain_logits = self.domain_classifier(probe_feature)
        class_probability = torch.softmax(class_logits, dim=-1)
        source_probability = torch.softmax(domain_logits, dim=-1)[..., 1]
        domain_invariance = (
            1.0 - 2.0 * torch.abs(source_probability - 0.5)
        ).clamp(0.0, 1.0)
        entropy = -(
            class_probability * torch.log(class_probability + self.eps)
        ).sum(dim=-1)
        entropy_score = (
            1.0 - entropy / math.log(self.num_classes)
        ).clamp(0.0, 1.0)
        maximum_probability = class_probability.max(dim=-1).values
        uniform_probability = 1.0 / self.num_classes
        confidence_score = (
            (maximum_probability - uniform_probability)
            / (1.0 - uniform_probability)
        ).clamp(0.0, 1.0)
        discriminability = (
            0.5 * (entropy_score + confidence_score)
        ).clamp(0.0, 1.0)
        domain_weight = 0.50 * domain_score_weight
        classification_weight = 0.50
        raw_coefficient = (
            domain_weight * domain_invariance
            + 0.25 * entropy_score
            + 0.25 * confidence_score
        ) / (domain_weight + classification_weight)
        raw_coefficient = raw_coefficient.clamp(0.0, 1.0)
        raw_coefficient = torch.where(
            resolved_valid,
            raw_coefficient,
            torch.zeros_like(raw_coefficient),
        )
        return QualityScoreOutput(
            coefficient=raw_coefficient.detach(),
            domain_invariance=domain_invariance.detach(),
            entropy_score=entropy_score.detach(),
            confidence_score=confidence_score.detach(),
            discriminability=discriminability.detach(),
            domain_logits=domain_logits,
            class_logits=class_logits,
            valid=resolved_valid,
        )


class TwoScaleQualityFusion(nn.Module):
    """Independently score T/S and concatenate them with unscored Shape."""

    def __init__(
        self,
        component_dim: int,
        shape_dim: int,
        num_classes: int,
        domain_hidden_dim: int = 128,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.component_dim = _positive_int("component_dim", component_dim)
        self.shape_dim = _positive_int("shape_dim", shape_dim)
        num_classes = _positive_int("num_classes", num_classes, minimum=2)
        domain_hidden_dim = _positive_int(
            "domain_hidden_dim", domain_hidden_dim
        )
        eps = _positive_float("eps", eps)
        self.trend_quality = QualityScorer(
            self.component_dim,
            num_classes,
            domain_hidden_dim,
            eps,
        )
        self.structure_quality = QualityScorer(
            self.component_dim,
            num_classes,
            domain_hidden_dim,
            eps,
        )
        self.fused_dim = 2 * self.component_dim + self.shape_dim

    def forward(
        self,
        trend_embedding: Tensor,
        structure_embedding: Tensor,
        shape_feature: Tensor,
        component_valid: Tensor,
        shape_valid: Tensor,
        domain_score_weight: float = 1.0,
    ) -> TwoScaleQualityOutput:
        if (
            not isinstance(trend_embedding, Tensor)
            or trend_embedding.ndim != 2
            or trend_embedding.shape[1] != self.component_dim
            or not trend_embedding.is_floating_point()
        ):
            raise ValueError(
                "trend_embedding must have floating shape [B, component_dim]"
            )
        batch_size = trend_embedding.shape[0]
        if (
            not isinstance(structure_embedding, Tensor)
            or structure_embedding.shape != trend_embedding.shape
            or structure_embedding.device != trend_embedding.device
            or not structure_embedding.is_floating_point()
        ):
            raise ValueError(
                "structure_embedding must match trend_embedding shape, dtype, and device"
            )
        if (
            not isinstance(shape_feature, Tensor)
            or shape_feature.shape != (batch_size, self.shape_dim)
            or not shape_feature.is_floating_point()
            or shape_feature.device != trend_embedding.device
        ):
            raise ValueError("shape_feature must have shape [B, shape_dim]")
        for feature in (trend_embedding, structure_embedding, shape_feature):
            if not torch.isfinite(feature).all().item():
                raise ValueError("quality fusion features must be finite")
        component_valid = _resolve_valid(
            component_valid, batch_size, trend_embedding.device
        )
        shape_valid = _resolve_valid(
            shape_valid, batch_size, trend_embedding.device
        )
        reference = self.trend_quality.class_classifier.weight
        trend_embedding = _to_quality_master_dtype(
            "trend_embedding", trend_embedding, reference
        )
        structure_embedding = _to_quality_master_dtype(
            "structure_embedding", structure_embedding, reference
        )
        shape_feature = _to_quality_master_dtype(
            "shape_feature", shape_feature, reference
        )
        with torch.autocast(device_type=reference.device.type, enabled=False):
            trend = self.trend_quality(
                trend_embedding, component_valid, domain_score_weight
            )
            structure = self.structure_quality(
                structure_embedding, component_valid, domain_score_weight
            )
            alpha_trend = trend.coefficient
            alpha_structure = structure.coefficient
            weighted_trend = alpha_trend.unsqueeze(-1) * trend_embedding
            weighted_structure = alpha_structure.unsqueeze(-1) * structure_embedding
            safe_shape = torch.where(
                shape_valid.unsqueeze(-1),
                shape_feature,
                torch.zeros_like(shape_feature),
            )
            fused_feature = torch.cat(
                [weighted_trend, weighted_structure, safe_shape], dim=-1
            )
        return TwoScaleQualityOutput(
            trend=trend,
            structure=structure,
            alpha_trend=alpha_trend,
            alpha_structure=alpha_structure,
            weighted_trend=weighted_trend,
            weighted_structure=weighted_structure,
            shape_feature=safe_shape,
            fused_feature=fused_feature,
        )


def _aggregate_quality_branches(
    branches: tuple[QualityScoreOutput, ...],
    labels: Tensor,
    selector: Tensor | None,
    *,
    domain: bool,
) -> tuple[Tensor, Tensor]:
    selected_logits = []
    selected_labels = []
    count = 0
    graph_zero = None
    for branch in branches:
        logits = branch.domain_logits if domain else branch.class_logits
        graph_term = logits.sum() * 0.0
        graph_zero = graph_term if graph_zero is None else graph_zero + graph_term
        active = branch.valid
        if selector is not None:
            active = active & selector
        active_count = int(active.sum().item())
        count += active_count
        if active_count:
            selected_logits.append(logits[active])
            selected_labels.append(labels[active])
    count_tensor = labels.new_tensor(count)
    if count == 0:
        return graph_zero, count_tensor
    return (
        F.cross_entropy(
            torch.cat(selected_logits, dim=0),
            torch.cat(selected_labels, dim=0),
        ),
        count_tensor,
    )


class TwoScaleQualityObjective(nn.Module):
    """Aggregate auxiliary classification/domain losses over T and S only."""

    def __init__(
        self,
        classification_weight: float = 1.0,
        domain_weight: float = 1.0,
    ) -> None:
        super().__init__()
        converted = []
        for name, value in (
            ("classification_weight", classification_weight),
            ("domain_weight", domain_weight),
        ):
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be finite and nonnegative") from error
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            converted.append(number)
        self.classification_weight, self.domain_weight = converted

    def forward(
        self,
        quality: TwoScaleQualityOutput,
        class_labels: Tensor,
        domain_labels: Tensor,
        source_mask: Tensor,
    ) -> TwoScaleQualityLossOutput:
        if not isinstance(quality, TwoScaleQualityOutput):
            raise ValueError("quality must be a TwoScaleQualityOutput")
        batch_size = quality.alpha_trend.shape[0]
        for labels, name in (
            (class_labels, "class_labels"),
            (domain_labels, "domain_labels"),
        ):
            if (
                not isinstance(labels, Tensor)
                or labels.dtype != torch.long
                or labels.shape != (batch_size,)
            ):
                raise ValueError(f"{name} must be a long tensor with shape [B]")
            if labels.device != quality.alpha_trend.device:
                raise ValueError("labels and quality must use the same device")
        source_mask = _resolve_valid(
            source_mask, batch_size, quality.alpha_trend.device
        )
        if not torch.all((domain_labels == 0) | (domain_labels == 1)).item():
            raise ValueError("domain_labels must contain only target=0 or source=1")
        if not torch.equal(source_mask, domain_labels == 1):
            raise ValueError("source_mask must equal domain_labels == 1")
        branches = (quality.trend, quality.structure)
        classification_loss, classification_count = (
            _aggregate_quality_branches(
                branches,
                class_labels,
                source_mask,
                domain=False,
            )
        )
        domain_loss, domain_count = _aggregate_quality_branches(
            branches,
            domain_labels,
            None,
            domain=True,
        )
        total_loss = (
            self.classification_weight * classification_loss
            + self.domain_weight * domain_loss
        )
        return TwoScaleQualityLossOutput(
            total_loss=total_loss,
            classification_loss=classification_loss,
            domain_loss=domain_loss,
            classification_count=classification_count,
            domain_count=domain_count,
        )
