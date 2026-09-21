"""Prototype selection and loss composition for structure-aware training."""

import math
from typing import NamedTuple

import torch
from torch.nn import functional as F


class SourceLosses(NamedTuple):
    total: torch.Tensor
    prototype_total: torch.Tensor


def select_top_shape_tokens(attention, valid_mask, ratio=.6):
    selected = torch.zeros_like(valid_mask)
    for index in range(attention.shape[0]):
        valid = torch.nonzero(valid_mask[index], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        count = max(1, math.ceil(valid.numel() * ratio))
        top = valid[torch.topk(attention[index, valid], count).indices]
        selected[index, top] = True
    return selected


def prototype_contrastive_loss(features, labels, prototypes, temperature=.1):
    if features.shape[0] == 0:
        return features.sum() * 0
    logits = F.normalize(features, dim=-1) @ F.normalize(prototypes.detach(), dim=-1).T
    return F.cross_entropy(logits / temperature, labels.long())


def shape_prototype_loss(tokens, selected, labels, prototypes, temperature=.1):
    sample_losses = []
    for index in range(tokens.shape[0]):
        chosen = tokens[index, selected[index]]
        if chosen.shape[0] == 0:
            continue
        repeated = labels[index].expand(chosen.shape[0])
        sample_losses.append(prototype_contrastive_loss(chosen, repeated, prototypes, temperature))
    if not sample_losses:
        return tokens.sum() * 0
    return torch.stack(sample_losses).mean()


def sample_shape_centers(tokens, selected):
    centers = []
    for index in range(tokens.shape[0]):
        chosen = F.normalize(tokens[index, selected[index]], dim=-1)
        if chosen.shape[0] == 0:
            raise ValueError("every source sample must select at least one shape token")
        centers.append(F.normalize(chosen.mean(0), dim=0))
    return torch.stack(centers)


def compose_source_loss(classification, instance, shape, shape_mix=.01, weight=1.):
    prototype = (1 - shape_mix) * instance + shape_mix * shape
    return SourceLosses(classification + weight * prototype, prototype)


def compose_da_loss(cls_source, pseudo_target, trade_off, proto_source, proto_target,
                    source_weight=1., target_weight=1.):
    return cls_source + trade_off * pseudo_target + source_weight * proto_source + target_weight * proto_target


def two_level_prototype_losses(
    outputs, labels, shape_bank, instance_bank, ratio=.6, temperature=.1,
    shape_mix=.01, selected_tokens=None,
):
    if selected_tokens is None:
        selected_tokens = select_top_shape_tokens(
            outputs["shape_attention"], outputs["shape_mask"], ratio,
        )
    instance = prototype_contrastive_loss(
        outputs["instance_feature"], labels, instance_bank.prototypes, temperature,
    )
    shape = shape_prototype_loss(
        outputs["shape_tokens"], selected_tokens, labels,
        shape_bank.prototypes, temperature,
    )
    return instance, shape, (1 - shape_mix) * instance + shape_mix * shape, selected_tokens


@torch.no_grad()
def update_source_banks(outputs, labels, shape_bank, instance_bank, ratio=.6):
    selected = select_top_shape_tokens(outputs["shape_attention"], outputs["shape_mask"], ratio)
    shape_bank.update_source(sample_shape_centers(outputs["shape_tokens"], selected), labels)
    instance_bank.update_source(outputs["instance_feature"], labels)


@torch.no_grad()
def initialize_source_banks(batches, shape_bank, instance_bank, ratio=.6):
    shape_sum = torch.zeros_like(shape_bank.prototypes)
    instance_sum = torch.zeros_like(instance_bank.prototypes)
    counts = torch.zeros(shape_bank.num_classes, device=shape_sum.device)
    for outputs, batch_labels in batches:
        selected = select_top_shape_tokens(outputs["shape_attention"], outputs["shape_mask"], ratio)
        shapes = sample_shape_centers(outputs["shape_tokens"], selected)
        instances = F.normalize(outputs["instance_feature"], dim=-1)
        for class_id in batch_labels.unique().tolist():
            class_mask = batch_labels == class_id
            shape_sum[class_id] += shapes[class_mask].sum(0)
            instance_sum[class_id] += instances[class_mask].sum(0)
            counts[class_id] += class_mask.sum()
    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise RuntimeError(f"missing source classes during prototype initialization: {missing}")
    labels = torch.arange(shape_bank.num_classes, device=shape_sum.device)
    shape_bank.initialize_source(shape_sum / counts[:, None], labels)
    instance_bank.initialize_source(instance_sum / counts[:, None], labels)


@torch.no_grad()
def structure_batch_statistics(outputs, labels, shape_bank, instance_bank, selected):
    """Return detached GPU aggregates for epoch-boundary diagnostics."""
    instance = F.normalize(outputs["instance_feature"], dim=-1)
    instance_proto = F.normalize(instance_bank.prototypes[labels], dim=-1)
    shape = sample_shape_centers(outputs["shape_tokens"], selected)
    shape_proto = F.normalize(shape_bank.prototypes[labels], dim=-1)
    attention = outputs["shape_attention"].clamp_min(1e-12)
    entropy = -(attention.log() * attention).sum(-1)
    top_counts = selected.sum(-1).to(attention.dtype)
    by_scale = {}
    for scale in outputs["shape_scales"].unique().tolist():
        scale_mask = outputs["shape_scales"] == scale
        by_scale[int(scale)] = (
            selected[:, scale_mask].sum().to(attention.dtype),
            outputs["shape_mask"][:, scale_mask].sum().to(attention.dtype),
        )
    return {
        "instance_cos": (instance * instance_proto).sum(-1),
        "shape_cos": (shape * shape_proto).sum(-1),
        "attention_entropy": entropy,
        "top_counts": top_counts,
        "selected_by_scale": by_scale,
    }
