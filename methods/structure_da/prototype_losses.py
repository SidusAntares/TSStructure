"""Instance-prototype and shared-shapelet losses for structure-aware training."""

import math
from typing import NamedTuple

import torch
from torch.nn import functional as F


class SourceLosses(NamedTuple):
    total: torch.Tensor
    prototype_total: torch.Tensor


def prototype_contrastive_loss(features, labels, prototypes, temperature=.1):
    if features.shape[0] == 0:
        return features.sum() * 0
    logits = F.normalize(features, dim=-1) @ F.normalize(prototypes.detach(), dim=-1).T
    return F.cross_entropy(logits / temperature, labels.long())


def instance_prototype_loss(features, labels, instance_bank, temperature=.1):
    return prototype_contrastive_loss(features, labels, instance_bank.prototypes, temperature)


def shapelet_diversity_loss(anchors, margin=.5):
    if anchors.shape[0] < 2:
        return anchors.sum() * 0
    normalized = F.normalize(anchors, dim=-1)
    similarity = normalized @ normalized.T
    upper = torch.triu_indices(anchors.shape[0], anchors.shape[0], offset=1, device=anchors.device)
    return F.relu(similarity[upper[0], upper[1]] - margin).square().mean()


def shapelet_data_support_loss(source_shape_tokens, anchors, temperature=.1):
    if temperature <= 0:
        raise ValueError("shapelet shaping temperature must be positive")
    tokens = source_shape_tokens.detach().reshape(-1, source_shape_tokens.shape[-1])
    if tokens.shape[0] == 0:
        raise ValueError("shapelet shaping requires at least one source token")
    tokens = F.normalize(tokens, dim=-1)
    anchors = F.normalize(anchors, dim=-1)
    distance = 1. - tokens @ anchors.T
    log_mean_exp = (
        torch.logsumexp(-distance / temperature, dim=0)
        - math.log(distance.shape[0])
    )
    return (-temperature * log_mean_exp).mean()


@torch.no_grad()
def update_instance_bank(features, labels, instance_bank):
    instance_bank.update_source(features, labels)


@torch.no_grad()
def initialize_instance_bank(batches, instance_bank):
    feature_sum = torch.zeros_like(instance_bank.prototypes)
    counts = torch.zeros(instance_bank.num_classes, device=feature_sum.device)
    for features, labels in batches:
        normalized = F.normalize(features, dim=-1)
        feature_sum.index_add_(0, labels.long(), normalized)
        counts.index_add_(0, labels.long(), torch.ones_like(labels, dtype=counts.dtype))
    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise RuntimeError(f"missing source classes during prototype initialization: {missing}")
    labels = torch.arange(instance_bank.num_classes, device=feature_sum.device)
    instance_bank.initialize_source(feature_sum / counts[:, None], labels)


@torch.no_grad()
def accumulate_class_feature_sums(class_sums, class_counts, features, labels):
    normalized = F.normalize(features.detach(), dim=-1)
    labels = labels.detach().long()
    class_sums.index_add_(0, labels, normalized)
    class_counts.index_add_(0, labels, torch.ones(labels.shape[0], device=class_counts.device, dtype=class_counts.dtype))


@torch.no_grad()
def centroid_alignment_summary(source_sums, source_counts, target_sums, target_counts):
    valid = (source_counts > 0) & (target_counts > 0)
    valid_classes = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_classes.numel() == 0:
        return {"macro_cos": source_sums.new_tensor(float("nan")), "valid_classes": valid_classes, "per_class": source_sums.new_empty(0)}
    source_centroids = F.normalize(source_sums[valid] / source_counts[valid, None], dim=-1)
    target_centroids = F.normalize(target_sums[valid] / target_counts[valid, None], dim=-1)
    per_class = (source_centroids * target_centroids).sum(-1)
    return {"macro_cos": per_class.mean(), "valid_classes": valid_classes, "per_class": per_class}


def prototype_ramp(step, ramp_epochs, start=.1):
    if not 0 <= start <= 1:
        raise ValueError("prototype ramp start must be in [0, 1]")
    if ramp_epochs < 0:
        raise ValueError("prototype ramp epochs must be non-negative")
    if ramp_epochs == 0 or step >= ramp_epochs:
        return 1.
    if step <= 0:
        return float(start)
    return float(start + (1. - start) * step / ramp_epochs)


def compose_source_loss(
    classification, instance, diversity, shaping, ramp=1.,
    instance_weight=.1, diversity_weight=.01, shaping_weight=.01,
):
    prototype = instance_weight * instance
    return SourceLosses(
        classification + ramp * prototype + diversity_weight * diversity
        + shaping_weight * shaping,
        prototype,
    )


def compose_da_loss(
    cls_source, pseudo_target, trade_off, source_instance, target_instance,
    diversity, shaping, target_ramp=1., instance_weight=.1,
    diversity_weight=.01, shaping_weight=.01,
):
    return (cls_source + trade_off * pseudo_target + instance_weight * source_instance
            + target_ramp * instance_weight * target_instance
            + diversity_weight * diversity + shaping_weight * shaping)


def ensure_finite_structure_loss(loss):
    if not torch.isfinite(loss).all():
        raise FloatingPointError("non-finite loss in structure-shapelet training")


@torch.no_grad()
def instance_batch_statistics(outputs, labels, instance_bank):
    instance = F.normalize(outputs["instance_feature"], dim=-1)
    prototype = F.normalize(instance_bank.prototypes[labels], dim=-1)
    return {"instance_cos": (instance * prototype).sum(-1)}
