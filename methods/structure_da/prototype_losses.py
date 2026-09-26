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


def compose_structure_v4_source_loss(
    classification, source_shape, diversity,
    shape_weight=.1, diversity_weight=.01,
):
    return classification + shape_weight * source_shape + diversity_weight * diversity


def compose_structure_v4_da_loss(
    classification, pseudo_target, source_shape, diversity, domain,
    trade_off=2., shape_weight=.1, diversity_weight=.01,
):
    return (
        classification + trade_off * pseudo_target
        + shape_weight * source_shape + diversity_weight * diversity + domain
    )


def compose_structure_v5_da_loss(
    classification, pseudo_target, source_shape, diversity,
    shared_adversarial, private_domain, separation,
    trade_off=2., shape_weight=.1, diversity_weight=.01,
    shared_adv_weight=.1, private_domain_weight=.1, separation_weight=.01,
):
    return (
        classification + trade_off * pseudo_target
        + shape_weight * source_shape + diversity_weight * diversity
        + shared_adv_weight * shared_adversarial
        + private_domain_weight * private_domain
        + separation_weight * separation
    )


def masked_pseudo_classification_loss(logits, pseudo_labels, pseudo_mask, criterion):
    if not (logits.shape[0] == pseudo_labels.shape[0] == pseudo_mask.shape[0]):
        raise ValueError("target logits, pseudo labels, and mask must align")
    if not pseudo_mask.any():
        return logits.sum() * 0
    return criterion(logits[pseudo_mask], pseudo_labels[pseudo_mask])


def structure_domain_adversarial_loss(
    classifier, source_features, target_features, alpha,
):
    from models.structure_da.discriminative_structure import gradient_reverse

    source_logits = classifier(gradient_reverse(source_features, alpha))
    target_logits = classifier(gradient_reverse(target_features, alpha))
    source_labels = torch.zeros(
        source_logits.shape[0], dtype=torch.long, device=source_logits.device,
    )
    target_labels = torch.ones(
        target_logits.shape[0], dtype=torch.long, device=target_logits.device,
    )
    source_loss = F.cross_entropy(source_logits, source_labels)
    target_loss = F.cross_entropy(target_logits, target_labels)
    return {
        "loss": .5 * (source_loss + target_loss),
        "source_loss": source_loss,
        "target_loss": target_loss,
        "source_accuracy": (source_logits.detach().argmax(1) == source_labels).float().mean(),
        "target_accuracy": (target_logits.detach().argmax(1) == target_labels).float().mean(),
        "source_count": int(source_logits.shape[0]),
        "target_count": int(target_logits.shape[0]),
    }


def structure_private_domain_loss(classifier, source_features, target_features):
    source_logits = classifier(source_features)
    target_logits = classifier(target_features)
    source_labels = torch.zeros(
        source_logits.shape[0], dtype=torch.long, device=source_logits.device,
    )
    target_labels = torch.ones(
        target_logits.shape[0], dtype=torch.long, device=target_logits.device,
    )
    source_loss = F.cross_entropy(source_logits, source_labels)
    target_loss = F.cross_entropy(target_logits, target_labels)
    return {
        "loss": .5 * (source_loss + target_loss),
        "source_loss": source_loss,
        "target_loss": target_loss,
        "source_accuracy": (source_logits.detach().argmax(1) == source_labels).float().mean(),
        "target_accuracy": (target_logits.detach().argmax(1) == target_labels).float().mean(),
        "source_count": int(source_logits.shape[0]),
        "target_count": int(target_logits.shape[0]),
    }


def shared_private_separation_loss(shared_features, private_features):
    if shared_features.ndim != 2 or private_features.ndim != 2:
        raise ValueError("shared/private features must both be rank-2")
    if shared_features.shape[0] != private_features.shape[0]:
        raise ValueError("shared/private features must contain the same samples")
    shared_centered = shared_features - shared_features.mean(dim=0, keepdim=True)
    private_centered = private_features - private_features.mean(dim=0, keepdim=True)
    covariance = shared_centered.T @ private_centered / max(shared_features.shape[0], 1)
    return covariance.square().mean()


def selected_shape_pseudo_loss(logits, pseudo_labels, pseudo_mask, criterion, minimum=2):
    """Apply the existing TimeMatch pseudo-label selection to shape logits."""
    selected_labels = pseudo_labels[pseudo_mask]
    if logits.shape[0] != selected_labels.shape[0]:
        raise ValueError("shape logits must contain exactly the pseudo-mask-selected samples")
    if logits.shape[0] < minimum:
        return logits.sum() * 0, 0.
    loss = criterion(logits, selected_labels)
    accuracy = float((logits.detach().argmax(1) == selected_labels).float().mean())
    return loss, accuracy


def _confidence_weight(confidence, threshold):
    if not 0 <= threshold < 1:
        raise ValueError("pseudo threshold must be in [0, 1)")
    return ((confidence - threshold) / (1. - threshold)).clamp(0., 1.)


def _per_sample_focal_loss(logits, labels, gamma):
    cross_entropy = F.cross_entropy(logits, labels.long(), reduction="none")
    if gamma == 0:
        return cross_entropy
    probability = torch.exp(-cross_entropy)
    return (1. - probability).pow(gamma) * cross_entropy


def class_balanced_shape_pseudo_loss(
    shape_logits, pseudo_labels, teacher_confidence, pseudo_threshold,
    focal_gamma=1., min_support=1, support_saturation=4, eps=1e-12,
):
    """Confidence- and support-reliable mean of pseudo-class shape losses."""
    if not (shape_logits.shape[0] == pseudo_labels.shape[0]
            == teacher_confidence.shape[0]):
        raise ValueError("shape logits, pseudo labels, and confidence must align")
    zero = shape_logits.sum() * 0
    if shape_logits.shape[0] == 0:
        return {
            "loss": zero, "valid_classes": 0,
            "mean_class_reliability": zero.detach(),
            "class_ids": pseudo_labels.new_empty(0),
            "per_class_support": pseudo_labels.new_empty(0),
            "per_class_reliability": teacher_confidence.new_empty(0),
        }
    weights = _confidence_weight(teacher_confidence, pseudo_threshold)
    losses = _per_sample_focal_loss(shape_logits, pseudo_labels, focal_gamma)
    class_ids, class_losses, supports, reliabilities = [], [], [], []
    for class_id in torch.unique(pseudo_labels, sorted=True):
        selected = pseudo_labels == class_id
        support = int(selected.sum())
        if support < int(min_support):
            continue
        class_weights = weights[selected]
        reliability = min(1., support / float(support_saturation)) * class_weights.mean()
        class_loss = (class_weights * losses[selected]).sum() / class_weights.sum().clamp_min(eps)
        class_ids.append(class_id)
        class_losses.append(class_loss)
        supports.append(support)
        reliabilities.append(reliability)
    if not class_ids:
        return {
            "loss": zero, "valid_classes": 0,
            "mean_class_reliability": zero.detach(),
            "class_ids": pseudo_labels.new_empty(0),
            "per_class_support": pseudo_labels.new_empty(0),
            "per_class_reliability": teacher_confidence.new_empty(0),
        }
    reliability = torch.stack(reliabilities)
    loss = (reliability * torch.stack(class_losses)).sum() / reliability.sum().clamp_min(eps)
    return {
        "loss": loss,
        "valid_classes": len(class_ids),
        "mean_class_reliability": reliability.detach().mean(),
        "class_ids": torch.stack(class_ids).detach(),
        "per_class_support": pseudo_labels.new_tensor(supports).detach(),
        "per_class_reliability": reliability.detach(),
    }


def memory_class_balanced_shape_pseudo_loss(
    shape_logits, pseudo_labels, teacher_confidence, target_memory,
    pseudo_threshold, focal_gamma=1., support_saturation=4, eps=1e-12,
):
    """Class-balanced target shape loss weighted by cross-batch reliability."""
    if not (shape_logits.shape[0] == pseudo_labels.shape[0]
            == teacher_confidence.shape[0]):
        raise ValueError("shape logits, pseudo labels, and confidence must align")
    zero = shape_logits.sum() * 0
    if shape_logits.shape[0] == 0:
        return {
            "loss": zero, "valid_classes": 0,
            "mean_class_reliability": zero.detach(),
            "class_ids": pseudo_labels.new_empty(0),
            "per_class_support": pseudo_labels.new_empty(0),
            "per_class_reliability": teacher_confidence.new_empty(0),
        }
    sample_weights = _confidence_weight(teacher_confidence, pseudo_threshold)
    sample_losses = _per_sample_focal_loss(shape_logits, pseudo_labels, focal_gamma)
    memory_reliability = target_memory.reliability(support_saturation)
    class_ids, class_losses, supports, reliabilities = [], [], [], []
    for class_id in torch.unique(pseudo_labels, sorted=True):
        selected = pseudo_labels == class_id
        reliability = memory_reliability[class_id]
        if not target_memory.initialized[class_id] or reliability <= 0:
            continue
        weights = sample_weights[selected]
        class_losses.append(
            (weights * sample_losses[selected]).sum() / weights.sum().clamp_min(eps)
        )
        class_ids.append(class_id)
        supports.append(int(selected.sum()))
        reliabilities.append(reliability)
    if not class_ids:
        return {
            "loss": zero, "valid_classes": 0,
            "mean_class_reliability": zero.detach(),
            "class_ids": pseudo_labels.new_empty(0),
            "per_class_support": pseudo_labels.new_empty(0),
            "per_class_reliability": teacher_confidence.new_empty(0),
        }
    reliability = torch.stack(reliabilities).detach()
    loss = (reliability * torch.stack(class_losses)).sum() / reliability.sum().clamp_min(eps)
    return {
        "loss": loss, "valid_classes": len(class_ids),
        "mean_class_reliability": reliability.mean(),
        "class_ids": torch.stack(class_ids).detach(),
        "per_class_support": pseudo_labels.new_tensor(supports).detach(),
        "per_class_reliability": reliability,
    }


def class_relative_domain_alignment(
    source_features, source_labels, target_features, target_pseudo,
    target_confidence, pseudo_threshold, distance="mse",
    min_target_support=2, support_saturation=4, eps=1e-12,
):
    """Align class-balanced global centers and class-relative residuals."""
    if distance not in {"mse", "smooth_l1"}:
        raise ValueError("distance must be mse or smooth_l1")
    if not (target_features.shape[0] == target_pseudo.shape[0]
            == target_confidence.shape[0]):
        raise ValueError("target features, pseudo labels, and confidence must align")
    zero = target_features.sum() * 0
    target_weights = _confidence_weight(target_confidence, pseudo_threshold)
    source_centers, target_centers, reliabilities, class_ids = [], [], [], []
    for class_id in torch.unique(target_pseudo, sorted=True):
        source_selected = source_labels == class_id
        target_selected = target_pseudo == class_id
        support = int(target_selected.sum())
        if not source_selected.any() or support < int(min_target_support):
            continue
        weights = target_weights[target_selected]
        target_center = (
            weights[:, None] * target_features[target_selected]
        ).sum(0) / weights.sum().clamp_min(eps)
        source_centers.append(source_features[source_selected].mean(0).detach())
        target_centers.append(target_center)
        reliabilities.append(
            min(1., support / float(support_saturation)) * weights.mean()
        )
        class_ids.append(class_id)
    if not class_ids:
        return {
            "total_loss": zero, "global_loss": zero, "relative_loss": zero,
            "center_gap": zero.detach(), "valid_classes": 0,
            "mean_class_reliability": zero.detach(),
            "class_ids": target_pseudo.new_empty(0),
        }
    source_centers = torch.stack(source_centers)
    target_centers = torch.stack(target_centers)
    reliability = torch.stack(reliabilities)
    normalized_reliability = reliability / reliability.sum().clamp_min(eps)
    source_mean = (normalized_reliability[:, None] * source_centers).sum(0).detach()
    target_mean = (normalized_reliability[:, None] * target_centers).sum(0)

    def metric(left, right):
        if distance == "mse":
            return F.mse_loss(left, right, reduction="none").mean(-1)
        return F.smooth_l1_loss(left, right, reduction="none").mean(-1)

    global_loss = metric(target_mean, source_mean)
    if len(class_ids) < 2:
        relative_loss = zero
    else:
        relative = metric(
            target_centers - target_mean,
            source_centers - source_mean,
        )
        relative_loss = (reliability * relative).sum() / reliability.sum().clamp_min(eps)
    return {
        "total_loss": global_loss + relative_loss,
        "global_loss": global_loss,
        "relative_loss": relative_loss,
        "center_gap": (target_mean - source_mean).norm().detach(),
        "valid_classes": len(class_ids),
        "mean_class_reliability": reliability.detach().mean(),
        "class_ids": torch.stack(class_ids).detach(),
    }


def memory_class_relative_domain_alignment(
    source_memory, target_memory, target_features, target_pseudo,
    target_confidence, pseudo_threshold, distance="mse",
    support_saturation=4, eps=1e-12,
):
    """Align current target centers to detached cross-batch class geometry."""
    if distance not in {"mse", "smooth_l1"}:
        raise ValueError("distance must be mse or smooth_l1")
    if not (target_features.shape[0] == target_pseudo.shape[0]
            == target_confidence.shape[0]):
        raise ValueError("target features, pseudo labels, and confidence must align")
    zero = target_features.sum() * 0
    reliability = target_memory.reliability(support_saturation)
    common = (
        source_memory.initialized & target_memory.initialized
        & (reliability > 0)
    )
    memory_ids = torch.nonzero(common, as_tuple=False).flatten()
    if memory_ids.numel() == 0 or target_features.shape[0] == 0:
        return {
            "total_loss": zero, "global_loss": zero, "relative_loss": zero,
            "center_gap": zero.detach(), "valid_classes": 0,
            "valid_current_classes": 0, "valid_memory_classes": int(memory_ids.numel()),
            "mean_class_reliability": zero.detach(),
            "class_ids": target_pseudo.new_empty(0), "relative_active": False,
        }

    memory_weights = reliability[memory_ids]
    memory_weights = memory_weights / memory_weights.sum().clamp_min(eps)
    source_memory_mean = (
        memory_weights[:, None] * source_memory.prototypes[memory_ids].detach()
    ).sum(0).detach()
    target_memory_mean = (
        memory_weights[:, None] * target_memory.prototypes[memory_ids].detach()
    ).sum(0).detach()
    sample_weights = _confidence_weight(target_confidence, pseudo_threshold)
    current_ids, current_centers, current_source, current_reliability = [], [], [], []
    for class_id in torch.unique(target_pseudo, sorted=True):
        if not common[class_id]:
            continue
        selected = target_pseudo == class_id
        weights = sample_weights[selected]
        center = (
            weights[:, None] * target_features[selected]
        ).sum(0) / weights.sum().clamp_min(eps)
        current_ids.append(class_id)
        current_centers.append(center)
        current_source.append(source_memory.prototypes[class_id].detach())
        current_reliability.append(reliability[class_id])
    if not current_ids:
        return {
            "total_loss": zero, "global_loss": zero, "relative_loss": zero,
            "center_gap": (target_memory_mean - source_memory_mean).norm().detach(),
            "valid_classes": 0, "valid_current_classes": 0,
            "valid_memory_classes": int(memory_ids.numel()),
            "mean_class_reliability": zero.detach(),
            "class_ids": target_pseudo.new_empty(0), "relative_active": False,
        }
    current_centers = torch.stack(current_centers)
    current_source = torch.stack(current_source)
    current_reliability = torch.stack(current_reliability).detach()
    normalized = current_reliability / current_reliability.sum().clamp_min(eps)

    def metric(left, right):
        if distance == "mse":
            return F.mse_loss(left, right, reduction="none").mean(-1)
        return F.smooth_l1_loss(left, right, reduction="none").mean(-1)

    target_batch_mean = (normalized[:, None] * current_centers).sum(0)
    source_reference_mean = (normalized[:, None] * current_source).sum(0).detach()
    global_loss = metric(target_batch_mean, source_reference_mean)
    relative_active = memory_ids.numel() >= 2
    if relative_active:
        relative = metric(
            current_centers - target_memory_mean,
            current_source - source_memory_mean,
        )
        relative_loss = (
            current_reliability * relative
        ).sum() / current_reliability.sum().clamp_min(eps)
    else:
        relative_loss = zero
    return {
        "total_loss": global_loss + relative_loss,
        "global_loss": global_loss,
        "relative_loss": relative_loss,
        "center_gap": (target_memory_mean - source_memory_mean).norm().detach(),
        "valid_classes": len(current_ids),
        "valid_current_classes": len(current_ids),
        "valid_memory_classes": int(memory_ids.numel()),
        "mean_class_reliability": current_reliability.mean(),
        "class_ids": torch.stack(current_ids).detach(),
        "relative_active": bool(relative_active),
    }


def _parameter_l2_norm(module):
    values = [parameter.detach().float().square().sum() for parameter in module.parameters()]
    if not values:
        return 0.
    return float(torch.sqrt(torch.stack(values).sum()))


def _gradient_l2_norm(module):
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters() if parameter.grad is not None
    ]
    if not values:
        return 0.
    return float(torch.sqrt(torch.stack(values).sum()))


@torch.no_grad()
def shape_health_snapshot(model, outputs):
    response = outputs["shapelet_response"].detach().float()
    tokens = outputs["shape_tokens"].detach().float()
    response_std = response.std(dim=0, unbiased=False)
    singular_values = torch.linalg.svdvals(response)
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    branch = model.structure_branch
    query_projection = model.temporal_encoder.attention_heads.external_query_projection
    values = {
        "shape_response_std_mean": float(response_std.mean()),
        "shape_response_std_min": float(response_std.min()),
        "shape_response_effective_rank": float(effective_rank),
        "shape_token_std": float(tokens.std(unbiased=False)),
        "shape_token_norm": float(tokens.norm(dim=-1).mean()),
        "shape_raw_encoder_param_norm": _parameter_l2_norm(branch.token_generator.raw_encoder),
        "shape_diff_encoder_param_norm": _parameter_l2_norm(branch.token_generator.diff_encoder),
        "shape_fusion_param_norm": _parameter_l2_norm(branch.token_generator.fusion),
        "shape_anchor_param_norm": float(branch.shapelet_dictionary.anchors.detach().float().norm()),
        "shape_query_projection_norm": _parameter_l2_norm(query_projection),
    }
    if "shapelet_strength" in outputs:
        values["shape_strength_std"] = float(
            outputs["shapelet_strength"].detach().float()
            .std(dim=0, unbiased=False).mean()
        )
    if "shapelet_concentration" in outputs:
        concentration = outputs["shapelet_concentration"].detach().float()
        values["shape_concentration_mean"] = float(concentration.mean())
        values["shape_concentration_std"] = float(concentration.std(unbiased=False))
    if "shapelet_phase_response" in outputs:
        values["phase_response_norm"] = float(
            outputs["shapelet_phase_response"].detach().float().norm(dim=-1).mean()
        )
        values["phase_feature_norm"] = float(
            outputs["shape_phase_feature"].detach().float().norm(dim=-1).mean()
        )
        values["semantic_feature_norm"] = float(
            outputs["shape_semantic_feature"].detach().float().norm(dim=-1).mean()
        )
    return values


def shape_gradient_snapshot(model):
    return {
        "grad_norm_shape_token_generator": _gradient_l2_norm(
            model.structure_branch.token_generator
        ),
        "grad_norm_shapelet_anchors": _gradient_l2_norm(
            model.structure_branch.shapelet_dictionary
        ),
        "grad_norm_shape_classifier": _gradient_l2_norm(model.shape_classifier),
        "grad_norm_query_projection": _gradient_l2_norm(
            model.temporal_encoder.attention_heads.external_query_projection
        ),
    }


def log_shape_health(writer, epoch, model, outputs):
    """Log the already-computed first training batch without another forward/backward."""
    values = {**shape_health_snapshot(model, outputs), **shape_gradient_snapshot(model)}
    for name, value in values.items():
        writer.add_scalar(f"shape_health/{name}", value, epoch)
    print("SHAPE_HEALTH|epoch=" + str(epoch) + "|" + "|".join(
        f"{name}={value:.6f}" for name, value in values.items()
    ))
    warnings = (
        ("shape_response_std_mean", 1e-5),
        ("shape_response_effective_rank", 1.1),
        ("shape_token_std", 1e-5),
    )
    for name, threshold in warnings:
        if values[name] < threshold:
            print(
                f"SHAPE_COLLAPSE_WARNING|epoch={epoch}|metric={name}|"
                f"value={values[name]:.6g}"
            )
    return values


@torch.no_grad()
def instance_batch_statistics(outputs, labels, instance_bank):
    instance = F.normalize(outputs["instance_feature"], dim=-1)
    prototype = F.normalize(instance_bank.prototypes[labels], dim=-1)
    return {"instance_cos": (instance * prototype).sum(-1)}
