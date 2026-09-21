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
