"""Transparent loss calculations for structure-aware domain adaptation."""

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F

from .adaptation import StructuralAdversarialOutput
from .model import ComponentStructureOutput


def _finite_non_negative_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative real number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a finite non-negative real number"
        ) from error
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be a finite non-negative real number")
    return converted


def _floating_finite_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _matrix_logits(
    name: str,
    logits: torch.Tensor,
    batch_size: Optional[int] = None,
    width: Optional[int] = None,
) -> torch.Tensor:
    logits = _floating_finite_tensor(name, logits)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError(f"{name} must have shape [B, C] with B,C > 0")
    if batch_size is not None and logits.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must equal {batch_size}")
    if width is not None and logits.shape[1] != width:
        raise ValueError(f"{name} width must equal {width}")
    return logits


def _validate_labels(
    source_labels: torch.Tensor,
    batch_size: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(source_labels, torch.Tensor):
        raise ValueError("source_labels must be a torch.Tensor")
    if source_labels.dtype != torch.long:
        raise ValueError("source_labels must use torch.long dtype")
    if source_labels.ndim != 1 or source_labels.shape[0] != batch_size:
        raise ValueError(f"source_labels must have shape [{batch_size}]")
    if source_labels.device != device:
        raise ValueError("source_labels must be on the same device as logits")
    if torch.any(source_labels < 0).item() or torch.any(
        source_labels >= num_classes
    ).item():
        raise ValueError(f"source_labels must contain values in [0, {num_classes})")
    return source_labels


def _valid_mask(
    name: str, valid: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    if not isinstance(valid, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if valid.dtype != torch.bool or valid.ndim != 1 or valid.shape[0] != batch_size:
        raise ValueError(f"{name} must be a boolean tensor with shape [{batch_size}]")
    if valid.device != device:
        raise ValueError(f"{name} must be on the same device as logits")
    return valid


def _structural_branches(output: ComponentStructureOutput):
    return (
        (
            output.structural_quality.trend_temporal,
            output.trend_temporal.valid,
        ),
        (
            output.structural_quality.dynamics_temporal,
            output.dynamics_temporal.valid,
        ),
        (
            output.structural_quality.dynamics_channel,
            output.dynamics_channel.valid,
        ),
    )


def _component_branches(output: ComponentStructureOutput):
    return (
        output.component_quality.trend,
        output.component_quality.dynamics,
        output.component_quality.residual,
    )


def classification_loss(
    source_output: ComponentStructureOutput,
    source_labels: torch.Tensor,
) -> torch.Tensor:
    """Return mean source-only classification cross entropy."""

    logits = _matrix_logits("source_output.logits", source_output.logits)
    labels = _validate_labels(
        source_labels,
        logits.shape[0],
        logits.shape[1],
        logits.device,
    )
    return F.cross_entropy(logits, labels)


def _domain_logits(name: str, quality, batch_size: int) -> torch.Tensor:
    return _matrix_logits(
        name, quality.scores.domain_logits, batch_size=batch_size, width=2
    )


def quality_domain_loss(
    source_output: ComponentStructureOutput,
    target_output: ComponentStructureOutput,
) -> torch.Tensor:
    """Sum six source=1/target=0 quality-domain branch losses."""

    source_first = source_output.structural_quality.trend_temporal
    target_first = target_output.structural_quality.trend_temporal
    source_batch = _matrix_logits(
        "source structural domain logits 0",
        source_first.scores.domain_logits,
        width=2,
    ).shape[0]
    target_batch = _matrix_logits(
        "target structural domain logits 0",
        target_first.scores.domain_logits,
        width=2,
    ).shape[0]
    contributions = []
    for index, ((source_quality, source_valid), (target_quality, target_valid)) in enumerate(
        zip(_structural_branches(source_output), _structural_branches(target_output))
    ):
        source_logits = _domain_logits(
            f"source structural domain logits {index}", source_quality, source_batch
        )
        target_logits = _domain_logits(
            f"target structural domain logits {index}", target_quality, target_batch
        )
        source_valid = _valid_mask(
            f"source structural valid {index}",
            source_valid,
            source_batch,
            source_logits.device,
        )
        target_valid = _valid_mask(
            f"target structural valid {index}",
            target_valid,
            target_batch,
            target_logits.device,
        )
        if source_valid.any().item() and target_valid.any().item():
            contributions.append(
                F.cross_entropy(
                    source_logits[source_valid],
                    torch.ones(
                        int(source_valid.sum().item()),
                        dtype=torch.long,
                        device=source_logits.device,
                    ),
                )
                + F.cross_entropy(
                    target_logits[target_valid],
                    torch.zeros(
                        int(target_valid.sum().item()),
                        dtype=torch.long,
                        device=target_logits.device,
                    ),
                )
            )
        else:
            contributions.append(
                source_logits.sum() * 0.0 + target_logits.sum() * 0.0
            )

    for index, (source_quality, target_quality) in enumerate(
        zip(_component_branches(source_output), _component_branches(target_output))
    ):
        source_logits = _domain_logits(
            f"source component domain logits {index}", source_quality, source_batch
        )
        target_logits = _domain_logits(
            f"target component domain logits {index}", target_quality, target_batch
        )
        contributions.append(
            F.cross_entropy(
                source_logits,
                torch.ones(source_batch, dtype=torch.long, device=source_logits.device),
            )
            + F.cross_entropy(
                target_logits,
                torch.zeros(target_batch, dtype=torch.long, device=target_logits.device),
            )
        )
    return sum(contributions[1:], contributions[0])


def quality_classification_loss(
    source_output: ComponentStructureOutput,
    source_labels: torch.Tensor,
) -> torch.Tensor:
    """Sum six source-only quality class-head cross entropies."""

    first_logits = _matrix_logits(
        "structural class logits 0",
        source_output.structural_quality.trend_temporal.scores.class_logits,
    )
    batch_size, num_classes = first_logits.shape
    labels = _validate_labels(
        source_labels, batch_size, num_classes, first_logits.device
    )
    contributions = []
    for index, (quality, valid) in enumerate(_structural_branches(source_output)):
        logits = _matrix_logits(
            f"structural class logits {index}",
            quality.scores.class_logits,
            batch_size=batch_size,
            width=num_classes,
        )
        valid = _valid_mask(
            f"structural valid {index}", valid, batch_size, logits.device
        )
        if valid.any().item():
            contributions.append(F.cross_entropy(logits[valid], labels[valid]))
        else:
            contributions.append(logits.sum() * 0.0)
    for index, quality in enumerate(_component_branches(source_output)):
        logits = _matrix_logits(
            f"component class logits {index}",
            quality.scores.class_logits,
            batch_size=batch_size,
            width=num_classes,
        )
        contributions.append(F.cross_entropy(logits, labels))
    return sum(contributions[1:], contributions[0])


def _class_cv(
    scores: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    eps: float,
) -> torch.Tensor:
    scores = _floating_finite_tensor("diversity scores", scores)
    if scores.ndim != 1 or scores.shape[0] != labels.shape[0]:
        raise ValueError("diversity scores must have shape [B]")
    membership = F.one_hot(labels, num_classes=num_classes).to(scores.dtype)
    counts = membership.sum(dim=0)
    present = counts > 0
    if int(present.sum().item()) <= 1:
        return scores.sum() * 0.0
    class_means = (membership.transpose(0, 1) @ scores) / counts.clamp_min(1)
    class_means = class_means[present]
    return class_means.std(unbiased=False) / (class_means.mean() + eps)


def component_diversity_loss(
    source_output: ComponentStructureOutput,
    source_labels: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return negative summed class-level CV for T/D/R diversity scores."""

    eps = _finite_non_negative_float("eps", eps)
    if eps == 0:
        raise ValueError("eps must be greater than zero")
    final_logits = _matrix_logits("source_output.logits", source_output.logits)
    batch_size, num_classes = final_logits.shape
    labels = _validate_labels(
        source_labels, batch_size, num_classes, final_logits.device
    )
    cvs = [
        _class_cv(quality.diversity, labels, num_classes, eps)
        for quality in _component_branches(source_output)
    ]
    return -sum(cvs[1:], cvs[0])


def structural_adversarial_loss(
    adaptation_output: StructuralAdversarialOutput,
) -> torch.Tensor:
    """Return ordinary source=1 plus target=0 binary domain BCE."""

    source_logits = _floating_finite_tensor(
        "source_logits", adaptation_output.source_logits
    )
    target_logits = _floating_finite_tensor(
        "target_logits", adaptation_output.target_logits
    )
    if source_logits.ndim != 1 or source_logits.shape[0] == 0:
        raise ValueError("source_logits must have non-empty shape [B_S]")
    if target_logits.ndim != 1 or target_logits.shape[0] == 0:
        raise ValueError("target_logits must have non-empty shape [B_T]")
    if source_logits.device != target_logits.device:
        raise ValueError("source_logits and target_logits must share a device")
    if source_logits.dtype != target_logits.dtype:
        raise ValueError("source_logits and target_logits must share a dtype")
    return F.binary_cross_entropy_with_logits(
        source_logits, torch.ones_like(source_logits)
    ) + F.binary_cross_entropy_with_logits(
        target_logits, torch.zeros_like(target_logits)
    )


@dataclass(frozen=True)
class LossWeights:
    """Non-negative engineering weights for the four auxiliary losses."""

    qdom: float = 1.0
    qcls: float = 1.0
    diversity: float = 1.0
    sda: float = 1.0

    def __post_init__(self) -> None:
        for name in ("qdom", "qcls", "diversity", "sda"):
            object.__setattr__(
                self, name, _finite_non_negative_float(name, getattr(self, name))
            )


@dataclass(frozen=True)
class StructureDALosses:
    """The five scalar losses and their exact weighted total."""

    classification: torch.Tensor
    quality_domain: torch.Tensor
    quality_classification: torch.Tensor
    diversity: torch.Tensor
    structural_adversarial: torch.Tensor
    total: torch.Tensor


def compose_total_loss(
    classification: torch.Tensor,
    quality_domain: torch.Tensor,
    quality_classification: torch.Tensor,
    diversity: torch.Tensor,
    structural_adversarial: torch.Tensor,
    weights: LossWeights,
) -> StructureDALosses:
    """Compose already-calculated scalar losses without another forward pass."""

    if not isinstance(weights, LossWeights):
        raise ValueError("weights must be a LossWeights instance")
    named_losses = (
        ("classification", classification),
        ("quality_domain", quality_domain),
        ("quality_classification", quality_classification),
        ("diversity", diversity),
        ("structural_adversarial", structural_adversarial),
    )
    for name, loss in named_losses:
        loss = _floating_finite_tensor(name, loss)
        if loss.ndim != 0:
            raise ValueError(f"{name} must be a scalar tensor")
    reference = classification
    for name, loss in named_losses[1:]:
        if loss.device != reference.device:
            raise ValueError(f"{name} must share the classification device")
        if loss.dtype != reference.dtype:
            raise ValueError(f"{name} must share the classification dtype")
    total = (
        classification
        + weights.qdom * quality_domain
        + weights.qcls * quality_classification
        + weights.diversity * diversity
        + weights.sda * structural_adversarial
    )
    return StructureDALosses(
        classification=classification,
        quality_domain=quality_domain,
        quality_classification=quality_classification,
        diversity=diversity,
        structural_adversarial=structural_adversarial,
        total=total,
    )
