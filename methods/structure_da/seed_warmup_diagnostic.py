"""Pure helpers for experiment 13B fixed-seed semantic warmup diagnostics.

Experiment 13B deliberately does not reuse the production Stage2Trainer because
that trainer freezes PSE and lets EMA Teacher construct Stable Labels.  Here the
Student semantic path (PSE + raw LTAE/time encoder + classifier) is trainable,
while decomposition/SRVF geometry stays frozen.  Teacher is observation-only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .stage2_parameter_policy import Stage2ParameterPolicy


FORBIDDEN_SEED_FIELDS = {
    "true_label",
    "true_class_name",
    "candidate_correct",
    "seed_correct",
    "confusion_flow",
    "oracle_precision",
    "oracle_f1",
}


@dataclass(frozen=True)
class FixedSeedRecord:
    sample_id: int
    pseudo_label: int
    selection_evidence: str


def configure_13b_semantic_student(model: nn.Module) -> Stage2ParameterPolicy:
    """Train PSE + complete raw temporal encoder + classifier; freeze geometry.

    The decomposition and T/S SRVF extractors are not part of the classification
    forward used by 13B and remain frozen.  This policy intentionally differs
    from the production Phase-only Stage-2 policy, which freezes PSE.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    try:
        pse = model.backbone.pixel_set_encoder
        raw = model.temporal_module.raw_encoder
        classifier = model.classifier
        model.backbone.decomposition
        model.temporal_module.trend_geometry
        model.temporal_module.structure_geometry
    except AttributeError as error:
        raise ValueError("model does not expose the 13B semantic/geometry modules") from error

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (pse, raw, classifier):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        (trainable if parameter.requires_grad else frozen).append(name)

    all_names = {name for name, _ in model.named_parameters()}
    if set(trainable) & set(frozen) or set(trainable) | set(frozen) != all_names:
        raise RuntimeError("13B parameter policy does not partition model parameters")
    if not any(name.startswith("backbone.pixel_set_encoder.") for name in trainable):
        raise RuntimeError("13B requires trainable PSE parameters")
    forbidden_prefixes = (
        "backbone.decomposition.",
        "temporal_module.trend_geometry.",
        "temporal_module.structure_geometry.",
    )
    if any(name.startswith(forbidden_prefixes) for name in trainable):
        raise RuntimeError("13B geometry/decomposition parameters must remain frozen")
    return Stage2ParameterPolicy(tuple(trainable), tuple(frozen))


def class_balanced_seed_cross_entropy(logits: Tensor, pseudo_labels: Tensor) -> Tensor:
    """Mean CE per present seed class, then mean across present classes."""
    if logits.ndim != 2 or logits.shape[0] < 1:
        raise ValueError("logits must have shape [B,C] with B>0")
    if pseudo_labels.shape != (logits.shape[0],) or pseudo_labels.dtype != torch.long:
        raise ValueError("pseudo_labels must have shape [B] and dtype torch.long")
    if torch.any(pseudo_labels < 0).item() or torch.any(pseudo_labels >= logits.shape[1]).item():
        raise ValueError("pseudo_labels contain an invalid class id")
    per_sample = F.cross_entropy(logits, pseudo_labels, reduction="none")
    class_losses = []
    for class_id in torch.unique(pseudo_labels, sorted=True):
        mask = pseudo_labels == class_id
        class_losses.append(per_sample[mask].mean())
    if not class_losses:
        raise ValueError("target seed batch is empty")
    return torch.stack(class_losses).mean()


def normalize_seed_manifest_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    valid_sample_ids: Iterable[int],
    initial_candidates: Mapping[int, int],
    num_classes: int,
) -> tuple[FixedSeedRecord, ...]:
    """Validate a precomputed, label-free T0 manifest without changing it."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer >=2")
    valid = {int(v) for v in valid_sample_ids}
    seen: set[int] = set()
    normalized: list[FixedSeedRecord] = []
    for row in rows:
        leaked = FORBIDDEN_SEED_FIELDS.intersection(str(key) for key in row.keys())
        if leaked:
            raise ValueError("seed manifest contains forbidden oracle fields: " + ",".join(sorted(leaked)))
        if "sample_id" not in row or "pseudo_label" not in row:
            raise ValueError("seed manifest requires sample_id and pseudo_label")
        sample_id = int(row["sample_id"])
        pseudo_label = int(row["pseudo_label"])
        if sample_id in seen:
            raise ValueError(f"duplicate seed sample_id {sample_id}")
        seen.add(sample_id)
        if sample_id not in valid:
            raise ValueError(f"seed sample_id {sample_id} is not in target-train")
        if not 0 <= pseudo_label < num_classes:
            raise ValueError(f"seed pseudo_label {pseudo_label} is outside class range")
        if sample_id not in initial_candidates:
            raise ValueError(f"initial Stage-1 candidate missing for seed {sample_id}")
        if int(initial_candidates[sample_id]) != pseudo_label:
            raise ValueError(
                f"seed {sample_id} pseudo_label={pseudo_label} relabels Stage-1 candidate "
                f"{initial_candidates[sample_id]}"
            )
        selected = row.get("seed_selected", True)
        if isinstance(selected, str):
            selected = selected.strip().lower() not in {"0", "false", "no", ""}
        if not bool(selected):
            continue
        evidence = str(row.get("selection_evidence", row.get("selection_metadata", "upstream_fixed_T0")))
        normalized.append(FixedSeedRecord(sample_id, pseudo_label, evidence))
    if not normalized:
        raise ValueError("T0 seed manifest contains no selected samples")
    return tuple(sorted(normalized, key=lambda item: item.sample_id))


def seed_manifest_fingerprint(records: Sequence[FixedSeedRecord]) -> str:
    h = hashlib.sha256()
    for item in sorted(records, key=lambda row: row.sample_id):
        h.update(f"{item.sample_id}:{item.pseudo_label}:{item.selection_evidence}\n".encode("utf-8"))
    return h.hexdigest()


def validate_checkpoint_epochs(total_epochs: int, checkpoints: Sequence[int]) -> tuple[int, ...]:
    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int) or total_epochs < 1:
        raise ValueError("total_epochs must be a positive integer")
    values = tuple(int(v) for v in checkpoints)
    if not values or values[0] != 0 or values[-1] != total_epochs:
        raise ValueError("checkpoint schedule must start at 0 and end at total_epochs")
    if tuple(sorted(set(values))) != values:
        raise ValueError("checkpoint epochs must be strictly increasing and unique")
    if any(v < 0 or v > total_epochs for v in values):
        raise ValueError("checkpoint epoch outside training window")
    if len(values) < 4:
        raise ValueError("13B requires initial, early, middle and end checkpoints")
    return values


def posterior_margin(probabilities: Tensor) -> Tensor:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [N,C], C>=2")
    top2 = probabilities.topk(k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def posterior_entropy(probabilities: Tensor, eps: float = 1e-12) -> Tensor:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N,C]")
    q = probabilities.clamp_min(float(eps))
    return -(q * q.log()).sum(dim=1)


def cosine_feature_displacement(current: Tensor, initial: Tensor, eps: float = 1e-12) -> Tensor:
    if current.shape != initial.shape or current.ndim != 2:
        raise ValueError("current and initial features must share shape [N,D]")
    a = current / current.norm(dim=1, keepdim=True).clamp_min(float(eps))
    b = initial / initial.norm(dim=1, keepdim=True).clamp_min(float(eps))
    return 1.0 - (a * b).sum(dim=1)


def finite_hyperparameters(*, lr: float, weight_decay: float, lambda_target: float, ema_decay: float) -> None:
    for name, value in (("lr", lr), ("weight_decay", weight_decay), ("lambda_target", lambda_target)):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(lr) <= 0.0:
        raise ValueError("lr must be greater than zero")
    if not math.isfinite(float(ema_decay)) or not 0.0 <= float(ema_decay) < 1.0:
        raise ValueError("ema_decay must satisfy 0 <= decay < 1")
