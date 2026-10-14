"""Source-calibrated class-conditional organization loss for Response96.

The reference is fitted only from labelled Source Train responses.  It is a
buffer-only module: no optimizer can update it, and Target data are never used
to rebuild or adapt its statistics.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


REFERENCE_SCHEMA_VERSION = 1


def response96_to_organization(response, anchor_count=16):
    """Decode block-major SC/K1/K2 response into [B, anchor, 4]."""
    if response.ndim != 2:
        raise ValueError(f"response must be [B,D], got {tuple(response.shape)}")
    anchor_count = int(anchor_count)
    expected = 6 * anchor_count
    if response.shape[1] != expected:
        raise ValueError(
            f"phase-moment response must have {expected} dimensions, "
            f"got {response.shape[1]}"
        )
    strength = response[:, 0:anchor_count]
    concentration = response[:, anchor_count:2 * anchor_count]
    k1_cos = response[:, 2 * anchor_count:3 * anchor_count]
    k1_sin = response[:, 3 * anchor_count:4 * anchor_count]
    k2_cos = response[:, 4 * anchor_count:5 * anchor_count]
    k2_sin = response[:, 5 * anchor_count:6 * anchor_count]
    return torch.stack((
        strength,
        concentration,
        torch.hypot(k1_cos, k1_sin),
        torch.hypot(k2_cos, k2_sin),
    ), dim=-1)


class ClassOrganizationReference(nn.Module):
    """Frozen class-by-anchor robust centers, tolerances, and weights."""

    def __init__(self, mu, tau, weight, support, metadata=None):
        super().__init__()
        mu = torch.as_tensor(mu).detach().float()
        tau = torch.as_tensor(tau).detach().float()
        weight = torch.as_tensor(weight).detach().float()
        support = torch.as_tensor(support).detach().long()
        if mu.ndim != 3 or mu.shape[-1] != 4:
            raise ValueError("mu must be [class,anchor,4]")
        if tau.shape != mu.shape or weight.shape != mu.shape:
            raise ValueError("tau and weight must match mu")
        if support.shape != (mu.shape[0],):
            raise ValueError("support must be [class]")
        if (tau <= 0).any():
            raise ValueError("all tolerances must be positive")
        if (weight < 0).any() or (weight > 1).any():
            raise ValueError("organization weights must be in [0,1]")
        self.register_buffer("mu", mu)
        self.register_buffer("tau", tau)
        self.register_buffer("weight", weight)
        self.register_buffer("support", support)
        self.metadata = dict(metadata or {})

    @property
    def num_classes(self):
        return int(self.mu.shape[0])

    @property
    def anchor_count(self):
        return int(self.mu.shape[1])


@torch.no_grad()
def build_organization_reference(
    responses,
    labels,
    num_classes,
    anchor_count=16,
    min_support=8,
    tolerance_quantile=.90,
    min_tau=1e-4,
    metadata=None,
):
    """Fit a deterministic robust Source-only organization reference.

    A statistic receives non-zero weight only when it has enough Source
    support, its nearest-class center gap exceeds half the joint tolerance,
    and its within-class tolerance is no larger than the global robust scale.
    No fixed number of anchors is selected.
    """
    responses = torch.as_tensor(responses).detach().float().cpu()
    labels = torch.as_tensor(labels).detach().long().cpu()
    if responses.shape[0] != labels.shape[0]:
        raise ValueError("responses and labels must contain the same samples")
    if labels.numel() == 0:
        raise ValueError("cannot fit organization reference from no samples")
    if labels.min() < 0 or labels.max() >= int(num_classes):
        raise ValueError("source labels are outside the configured class order")
    events = response96_to_organization(responses, anchor_count)
    classes, anchors, components = int(num_classes), int(anchor_count), 4
    mu = torch.zeros(classes, anchors, components)
    tau = torch.full_like(mu, float(min_tau))
    support = torch.bincount(labels, minlength=classes)
    for class_id in range(classes):
        selected = events[labels == class_id]
        if selected.numel() == 0:
            continue
        center = torch.quantile(selected, .5, dim=0)
        deviation = (selected - center).abs()
        mu[class_id] = center
        tau[class_id] = torch.quantile(
            deviation, float(tolerance_quantile), dim=0,
        ).clamp_min(float(min_tau))

    global_center = torch.quantile(events, .5, dim=0)
    global_scale = torch.quantile(
        (events - global_center).abs(), float(tolerance_quantile), dim=0,
    ).clamp_min(float(min_tau))
    weight = torch.zeros_like(mu)
    valid_classes = support >= int(min_support)
    maximum_support = support[valid_classes].max().float() if valid_classes.any() else 1.
    for class_id in range(classes):
        if not valid_classes[class_id]:
            continue
        competitors = torch.where(valid_classes)[0]
        competitors = competitors[competitors != class_id]
        if competitors.numel() == 0:
            continue
        gap = (mu[class_id] - mu[competitors]).abs()
        joint_tolerance = tau[class_id] + tau[competitors]
        separation = (gap / joint_tolerance.clamp_min(float(min_tau))).min(0).values
        stability_ratio = tau[class_id] / global_scale
        reliable = (separation >= .5) & (stability_ratio <= 1.)
        separation_score = ((separation - .5) / 1.5).clamp(0., 1.)
        stability_score = (1. - stability_ratio).clamp(0., 1.)
        support_score = torch.sqrt(support[class_id].float() / maximum_support)
        weight[class_id] = (
            reliable.float() * separation_score * stability_score * support_score
        )

    details = dict(metadata or {})
    details.setdefault("schema_version", REFERENCE_SCHEMA_VERSION)
    details.setdefault("class_ids", list(range(classes)))
    details.setdefault("anchor_count", anchors)
    details.setdefault("response_dim", 6 * anchors)
    details.setdefault("component_order", [
        "strength", "concentration", "phase_k1_magnitude", "phase_k2_magnitude",
    ])
    details.setdefault("center_rule", "source_train_median")
    details.setdefault("tolerance_rule", f"source_train_abs_deviation_q{tolerance_quantile:g}")
    details.setdefault("weight_rule", "support_x_stability_x_nearest_class_separation")
    details.setdefault("min_support", int(min_support))
    details.setdefault("target_updates", False)
    return ClassOrganizationReference(mu, tau, weight, support, details)


def class_organization_loss(response, labels, reference, sample_mask=None, eps=1e-8):
    """Penalize only deviations outside the frozen Source tolerance band."""
    if not isinstance(reference, ClassOrganizationReference):
        raise TypeError("reference must be ClassOrganizationReference")
    labels = torch.as_tensor(labels, device=response.device, dtype=torch.long)
    if labels.shape != (response.shape[0],):
        raise ValueError("labels must be [B]")
    if labels.numel() and (labels.min() < 0 or labels.max() >= reference.num_classes):
        raise ValueError("organization labels are outside the reference class order")
    if sample_mask is None:
        sample_mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        sample_mask = torch.as_tensor(sample_mask, device=response.device, dtype=torch.bool)
    if sample_mask.shape != labels.shape:
        raise ValueError("sample_mask must be [B]")
    organization = response96_to_organization(response, reference.anchor_count)
    selected_labels = labels[sample_mask]
    selected = organization[sample_mask]
    mu = reference.mu[selected_labels].to(selected)
    tau = reference.tau[selected_labels].to(selected)
    weight = reference.weight[selected_labels].to(selected)
    residual = torch.relu((selected - mu).abs() / tau.clamp_min(eps) - 1.)
    denominator = weight.sum()
    loss = (weight * residual.square()).sum() / denominator.clamp_min(eps)
    return {
        "loss": loss,
        "valid_samples": sample_mask.sum().detach(),
        "valid_statistics": (weight > 0).sum().detach(),
    }


def target_organization_objective(
    response,
    pseudo_labels,
    trusted_mask,
    reference,
    epoch,
    configured_weight,
    start_epoch=5,
):
    """Apply the Source reference only to already-trusted pseudo-labelled samples."""
    configured_weight = float(configured_weight)
    active = configured_weight > 0 and int(epoch) >= int(start_epoch)
    if not active:
        zero = response.sum() * 0.
        return {
            "loss": zero,
            "weighted_loss": zero,
            "effective_weight": 0.,
            "valid_samples": 0,
            "valid_statistics": 0,
        }
    if reference is None:
        raise ValueError("positive Target organization weight requires a Source reference")
    result = class_organization_loss(
        response, pseudo_labels, reference, sample_mask=trusted_mask,
    )
    return {
        **result,
        "weighted_loss": configured_weight * result["loss"],
        "effective_weight": configured_weight,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def collect_source_responses(model, data_loader, device):
    """Extract deterministic Source Train Response96 and GT labels once."""
    was_training = model.training
    model.eval()
    responses, labels = [], []
    try:
        for sample in data_loader:
            pixels = sample["pixels"].to(device)
            valid_pixels = sample["valid_pixels"].to(device)
            positions = sample["positions"].to(device)
            extra = sample.get("extra")
            if extra is not None:
                extra = extra.to(device)
            output = model(
                pixels, valid_pixels, positions, extra, return_dict=True,
            )
            responses.append(output["shapelet_response"].detach().cpu())
            labels.append(sample["label"].detach().long().cpu())
    finally:
        model.train(was_training)
    if not responses:
        raise RuntimeError("Source Train reference loader produced no samples")
    return torch.cat(responses), torch.cat(labels)


def build_source_organization_reference(
    model,
    data_loader,
    device,
    num_classes,
    anchor_count=16,
    min_support=8,
    metadata=None,
):
    responses, labels = collect_source_responses(model, data_loader, device)
    return build_organization_reference(
        responses, labels, num_classes=num_classes,
        anchor_count=anchor_count, min_support=min_support,
        metadata=metadata,
    )


def save_organization_reference(reference, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "state_dict": reference.state_dict(),
        "metadata": dict(reference.metadata),
    }, path)


def load_organization_reference(
    path,
    source_checkpoint=None,
    expected_class_ids=None,
    expected_anchor_count=None,
    expected_response_dim=None,
):
    packet = torch.load(path, map_location="cpu", weights_only=False)
    if int(packet.get("schema_version", -1)) != REFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported organization reference schema")
    state = packet.get("state_dict", {})
    metadata = dict(packet.get("metadata", {}))
    required = ("mu", "tau", "weight", "support")
    if any(name not in state for name in required):
        raise ValueError("organization reference tensors are incomplete")
    reference = ClassOrganizationReference(
        state["mu"], state["tau"], state["weight"], state["support"], metadata,
    )
    if expected_class_ids is not None and list(expected_class_ids) != list(
        metadata.get("class_ids", [])
    ):
        raise ValueError("organization reference class order mismatch")
    if expected_anchor_count is not None and reference.anchor_count != int(
        expected_anchor_count
    ):
        raise ValueError("organization reference Anchor dimension mismatch")
    if expected_response_dim is not None and int(metadata.get("response_dim", -1)) != int(
        expected_response_dim
    ):
        raise ValueError("organization reference Response dimension mismatch")
    if source_checkpoint is not None:
        actual = sha256_file(source_checkpoint)
        expected = metadata.get("source_checkpoint_sha256")
        if actual != expected:
            raise ValueError(
                f"organization reference Source checkpoint SHA256 mismatch: "
                f"expected={expected}, actual={actual}"
            )
    return reference
