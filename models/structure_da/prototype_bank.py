"""Source-only class prototype banks."""

import torch
from torch import nn
from torch.nn import functional as F


class ClassPrototypeBank(nn.Module):
    def __init__(self, num_classes, feature_dim, momentum=.9):
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("update_count", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def update_source(self, features, labels):
        features = F.normalize(features.detach(), dim=-1)
        labels = labels.detach().long()
        for class_id in labels.unique().tolist():
            center = F.normalize(features[labels == class_id].mean(0), dim=0)
            if self.initialized[class_id]:
                center = F.normalize(
                    self.momentum * self.prototypes[class_id]
                    + (1 - self.momentum) * center,
                    dim=0,
                )
            self.prototypes[class_id].copy_(center)
            self.initialized[class_id] = True
            self.update_count[class_id] += 1

    @torch.no_grad()
    def initialize_source(self, features, labels):
        self.prototypes.zero_()
        self.initialized.zero_()
        self.update_count.zero_()
        self.update_source(features, labels)
        missing = torch.nonzero(~self.initialized, as_tuple=False).flatten().tolist()
        if missing:
            raise RuntimeError(f"missing source classes during prototype initialization: {missing}")


class ClassFeatureMemory(nn.Module):
    """Detached cross-batch EMA class centers used only by the UDA trainer."""

    def __init__(self, num_classes, feature_dim, momentum=.9):
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("sample_count", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("confidence_ema", torch.zeros(num_classes))
        self.register_buffer("update_count", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def _ema_update(self, class_id, center, count, confidence):
        center = center.detach()
        confidence = torch.as_tensor(
            confidence, device=self.confidence_ema.device,
            dtype=self.confidence_ema.dtype,
        ).detach()
        if self.initialized[class_id]:
            self.prototypes[class_id].mul_(self.momentum).add_(
                center, alpha=1. - self.momentum,
            )
            self.confidence_ema[class_id].mul_(self.momentum).add_(
                confidence, alpha=1. - self.momentum,
            )
        else:
            self.prototypes[class_id].copy_(center)
            self.confidence_ema[class_id].copy_(confidence)
            self.initialized[class_id] = True
        self.sample_count[class_id] += int(count)
        self.update_count[class_id] += 1

    @torch.no_grad()
    def update_source(self, features, labels):
        features = features.detach()
        labels = labels.detach().long()
        for class_id in torch.unique(labels, sorted=True).tolist():
            selected = labels == class_id
            self._ema_update(
                class_id, features[selected].mean(0), int(selected.sum()), 1.,
            )

    @torch.no_grad()
    def update_target(
        self, features, labels, confidence, pseudo_threshold, eps=1e-12,
    ):
        features = features.detach()
        labels = labels.detach().long()
        confidence = confidence.detach()
        weights = ((confidence - pseudo_threshold) / (1. - pseudo_threshold)).clamp(0., 1.)
        for class_id in torch.unique(labels, sorted=True).tolist():
            selected = labels == class_id
            class_weights = weights[selected]
            center = (
                class_weights[:, None] * features[selected]
            ).sum(0) / class_weights.sum().clamp_min(eps)
            self._ema_update(
                class_id, center, int(selected.sum()), class_weights.mean(),
            )

    def reliability(self, support_saturation=4):
        support = self.sample_count.to(self.confidence_ema.dtype)
        return (
            (support / float(support_saturation)).clamp(max=1.)
            * self.confidence_ema
            * self.initialized.to(self.confidence_ema.dtype)
        ).detach()
