"""Stage-1 source supervision: CE + prototype margins + q-to-cls consistency.

The Stage-1 objective is the sum

    L = L_cls
      + lambda_q * L_proto^q
      + lambda_f * L_proto^f
      + lambda_q_to_cls * L_q_to_cls

where

- L_cls is the fused-feature cross entropy;
- L_proto^q is a Triplet-Center-style relative margin on Shape SRVF distance;
- L_proto^f is a relative margin on fused-feature cosine distance;
- L_q_to_cls aligns the (stop-gradient) Shape geometry class distribution with
  the classifier distribution.

During warmup the objective reduces to the classification term alone. All
loss components must return a well-typed zero (never NaN) when no valid sample
or prototype is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .prototype_bank import SourcePrototypeBank, support_aware_q_distance


@dataclass(frozen=True)
class Stage1LossOutput:
    total: Tensor
    classification: Tensor
    q_prototype: Tensor
    fused_prototype: Tensor
    q_to_classifier: Tensor
    q_valid_count: int
    fused_valid_count: int
    consistency_valid_count: int


def _zero_like(tensor: Tensor) -> Tensor:
    return tensor.sum() * 0.0


def _relative_margin_loss(
    positive: Tensor,
    negative: Tensor,
    positive_valid: Tensor,
    negative_valid: Tensor,
    margin: float,
) -> tuple[Tensor, int]:
    """Triplet-Center relative margin ``relu(margin + d_plus - d_minus)``.

    Args:
        positive: Per-sample positive distance ``[B]``.
        negative: Per-sample best negative distance ``[B]``.
        positive_valid: ``[B]`` bool mask.
        negative_valid: ``[B]`` bool mask.
        margin: Relative margin.

    Returns:
        ``(loss, valid_count)``. The loss is a differentiable scalar; when no
        sample is valid it is a typed zero on the same device as ``positive``.
    """
    eligible = positive_valid & negative_valid
    if not torch.any(eligible).item():
        return _zero_like(positive), 0
    violation = torch.relu(margin + positive[eligible] - negative[eligible])
    return violation.mean(), int(eligible.sum().item())


def _cosine_distance(features: Tensor, prototypes: Tensor) -> Tensor:
    """Pairwise 1 - cosine similarity with L2 normalization on both sides."""
    features = F.normalize(features, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    return 1.0 - (features @ prototypes.T)


class Stage1Objective(nn.Module):
    """Compute the complete Stage-1 source loss against a prototype bank."""

    def __init__(
        self,
        *,
        num_classes: int,
        lambda_q: float = 0.1,
        lambda_f: float = 0.1,
        lambda_q_to_cls: float = 0.1,
        margin_q: float = 0.1,
        margin_f: float = 0.1,
        tau_q: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        for name, value in (
            ("lambda_q", lambda_q),
            ("lambda_f", lambda_f),
            ("lambda_q_to_cls", lambda_q_to_cls),
            ("margin_q", margin_q),
            ("margin_f", margin_f),
            ("tau_q", tau_q),
        ):
            if not torch.isfinite(torch.tensor(float(value))).item() or float(value) < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.num_classes = num_classes
        self.lambda_q = float(lambda_q)
        self.lambda_f = float(lambda_f)
        self.lambda_q_to_cls = float(lambda_q_to_cls)
        self.margin_q = float(margin_q)
        self.margin_f = float(margin_f)
        self.tau_q = float(tau_q)
        self.eps = float(eps)

    def _prototype_q_loss(
        self,
        q: Tensor,
        support: Tensor,
        valid: Tensor,
        labels: Tensor,
        bank: SourcePrototypeBank,
        integration_weights: Tensor,
    ) -> tuple[Tensor, int]:
        """Relative-margin loss between a sample's Shape SRVF and class prototypes."""
        if bank is None or not torch.any(bank.ready).item():
            return _zero_like(q), 0
        distances = support_aware_q_distance(
            q,
            bank.shape_srvf,
            support,
            bank.shape_support,
            integration_weights,
            eps=self.eps,
        )
        positive = torch.gather(distances.distance, 1, labels.unsqueeze(-1)).squeeze(-1)
        positive_valid = valid & torch.gather(
            bank.ready.unsqueeze(0).expand(distances.valid.shape[0], -1),
            1,
            labels.unsqueeze(-1),
        ).squeeze(-1)
        valid_true = distances.valid.gather(1, labels.unsqueeze(-1)).squeeze(-1)
        positive_valid = positive_valid & valid_true

        # Best valid negative among ready classes not equal to the true class.
        ready = bank.ready.to(device=q.device)
        not_true = ~F.one_hot(labels, num_classes=self.num_classes).bool()
        candidate_valid = distances.valid & ready.unsqueeze(0) & not_true
        candidate_distances = torch.where(
            candidate_valid,
            distances.distance,
            torch.full_like(distances.distance, torch.inf),
        )
        negative = candidate_distances.min(dim=-1).values
        negative_valid = candidate_valid.any(dim=-1)
        return _relative_margin_loss(
            positive, negative, positive_valid, negative_valid, self.margin_q
        )

    def _prototype_fused_loss(
        self,
        fused: Tensor,
        labels: Tensor,
        bank: SourcePrototypeBank,
    ) -> tuple[Tensor, int]:
        """Relative-margin loss between fused features and fused prototypes."""
        if bank is None or not torch.any(bank.ready).item():
            return _zero_like(fused), 0
        distances = _cosine_distance(fused, bank.fused)
        positive = torch.gather(distances, 1, labels.unsqueeze(-1)).squeeze(-1)
        ready = bank.ready.to(device=fused.device)
        not_true = ~F.one_hot(labels, num_classes=self.num_classes).bool()
        candidate_valid = ready.unsqueeze(0) & not_true
        candidate_distances = torch.where(
            candidate_valid,
            distances,
            torch.full_like(distances, torch.inf),
        )
        negative = candidate_distances.min(dim=-1).values
        negative_valid = candidate_valid.any(dim=-1)
        positive_valid = torch.gather(
            ready.unsqueeze(0).expand(labels.shape[0], -1),
            1,
            labels.unsqueeze(-1),
        ).squeeze(-1)
        return _relative_margin_loss(
            positive, negative, positive_valid, negative_valid, self.margin_f
        )

    def _q_to_classifier_loss(
        self,
        q: Tensor,
        support: Tensor,
        valid: Tensor,
        logits: Tensor,
        bank: SourcePrototypeBank,
        integration_weights: Tensor,
    ) -> tuple[Tensor, int]:
        """KL from a stop-gradient Shape geometry distribution to logits."""
        if bank is None or not torch.all(bank.ready).item():
            return _zero_like(logits), 0
        distances = support_aware_q_distance(
            q,
            bank.shape_srvf,
            support,
            bank.shape_support,
            integration_weights,
            eps=self.eps,
        )
        # Only samples whose distance to *every* class prototype is valid enter
        # the consistency loss; invalid classes are never given a huge distance.
        all_valid = distances.valid.all(dim=-1) & valid
        if not torch.any(all_valid).item():
            return _zero_like(logits), 0
        log_p_q = F.log_softmax(-distances.distance_sq[all_valid] / self.tau_q, dim=-1)
        # stop-gradient teacher: the q distribution itself receives no gradient
        p_q = log_p_q.detach().exp()
        logits_valid = logits[all_valid]
        log_p_cls = F.log_softmax(logits_valid, dim=-1)
        # F.kl_div(input=log_p_cls, target=p_q) with batchmean reduction
        kl = F.kl_div(log_p_cls, p_q, reduction="batchmean")
        return kl, int(all_valid.sum().item())

    def forward(
        self,
        *,
        logits: Tensor,
        fused_repr: Tensor,
        labels: Tensor,
        q: Tensor | None,
        q_support: Tensor | None,
        q_valid: Tensor | None,
        bank: SourcePrototypeBank | None,
        integration_weights: Tensor | None,
        warmup: bool = False,
    ) -> Stage1LossOutput:
        """Compute the Stage-1 objective for one source batch.

        Args:
            logits: Classifier logits ``[B, C]``.
            fused_repr: Fused features ``[B, 2*d_L]``.
            labels: Source labels ``[B]``.
            q: Structure SRVFs ``[B, K, D]`` or ``None`` in warmup.
            q_support: Structure supports ``[B, K]`` or ``None``.
            q_valid: Structure validity ``[B]`` or ``None``.
            bank: Current source prototype bank, or ``None`` before first scan.
            integration_weights: Canonical-grid weights ``[K]``.
            warmup: When True only the classification term contributes.
        """
        classification = F.cross_entropy(logits, labels)
        if warmup:
            return Stage1LossOutput(
                total=classification,
                classification=classification,
                q_prototype=_zero_like(classification),
                fused_prototype=_zero_like(classification),
                q_to_classifier=_zero_like(classification),
                q_valid_count=0,
                fused_valid_count=0,
                consistency_valid_count=0,
            )

        missing_inputs = [
            name
            for name, value in (
                ("q", q),
                ("q_support", q_support),
                ("q_valid", q_valid),
                ("integration_weights", integration_weights),
                ("bank", bank),
            )
            if value is None
        ]
        if missing_inputs:
            raise ValueError(
                "missing Stage-1 non-warmup inputs: " + ", ".join(missing_inputs)
            )

        q_proto, q_count = self._prototype_q_loss(
            q, q_support, q_valid, labels, bank, integration_weights
        )
        f_proto, f_count = self._prototype_fused_loss(fused_repr, labels, bank)
        q_to_cls, consistency_count = self._q_to_classifier_loss(
            q, q_support, q_valid, logits, bank, integration_weights
        )
        total = (
            classification
            + self.lambda_q * q_proto
            + self.lambda_f * f_proto
            + self.lambda_q_to_cls * q_to_cls
        )
        return Stage1LossOutput(
            total=total,
            classification=classification,
            q_prototype=q_proto,
            fused_prototype=f_proto,
            q_to_classifier=q_to_cls,
            q_valid_count=q_count,
            fused_valid_count=f_count,
            consistency_valid_count=consistency_count,
        )
