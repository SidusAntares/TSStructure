"""Stage-2 trainable objective over source and synthetic target-style source."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .domain_shape_state import DomainShapeState, DomainShapeStatus
from .prototype_bank import SourcePrototypeBank
from .stage1_objective import Stage1Objective


@dataclass(frozen=True)
class Stage2ObjectiveConfig:
    lambda_src_proto: float
    lambda_src_cons: float
    lambda_syn: float
    lambda_syn_cons: float
    tau_q: float
    fused_margin: float

    def __post_init__(self) -> None:
        for name in (
            "lambda_src_proto",
            "lambda_src_cons",
            "lambda_syn",
            "lambda_syn_cons",
            "fused_margin",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(float(self.tau_q)) or float(self.tau_q) <= 0.0:
            raise ValueError("tau_q must be finite and greater than zero")


@dataclass(frozen=True)
class Stage2ObjectiveOutput:
    total: Tensor
    source_cls: Tensor
    source_proto: Tensor
    source_consistency: Tensor
    synthetic_cls: Tensor
    synthetic_consistency: Tensor
    source_count: int
    synthetic_count: int
    source_consistency_count: int
    synthetic_consistency_count: int


def _zero_like(tensor: Tensor) -> Tensor:
    return tensor.sum() * 0.0


def _detached_bank(bank: SourcePrototypeBank) -> SourcePrototypeBank:
    return SourcePrototypeBank(
        trend_srvf=bank.trend_srvf.detach(),
        shape_srvf=bank.shape_srvf.detach(),
        trend_support=bank.trend_support.detach(),
        shape_support=bank.shape_support.detach(),
        fused=bank.fused.detach(),
        class_counts=bank.class_counts.detach(),
        ready=bank.ready.detach(),
        q_distance_samples=tuple(item.detach() for item in bank.q_distance_samples),
        f_distance_samples=tuple(item.detach() for item in bank.f_distance_samples),
        q_quantiles=bank.q_quantiles.detach(),
        f_quantiles=bank.f_quantiles.detach(),
        version=bank.version,
    )


def _target_style_bank(
    bank: SourcePrototypeBank,
    state: DomainShapeState,
    lambda_delta: float,
) -> SourcePrototypeBank:
    if state.status is not DomainShapeStatus.CONFIRMED or state.delta is None:
        raise ValueError("synthetic consistency requires a confirmed Domain Shape state")
    if not math.isfinite(float(lambda_delta)) or not 0.0 <= float(lambda_delta) <= 1.0:
        raise ValueError("lambda_delta must lie in [0,1]")
    detached = _detached_bank(bank)
    delta = state.delta.detach().to(
        device=detached.shape_srvf.device,
        dtype=detached.shape_srvf.dtype,
    )
    if delta.shape != detached.shape_srvf.shape[1:]:
        raise ValueError("Domain Shape delta must match source Shape prototypes")
    return SourcePrototypeBank(
        trend_srvf=detached.trend_srvf,
        shape_srvf=detached.shape_srvf + float(lambda_delta) * delta.unsqueeze(0),
        trend_support=detached.trend_support,
        shape_support=detached.shape_support,
        fused=detached.fused,
        class_counts=detached.class_counts,
        ready=detached.ready,
        q_distance_samples=detached.q_distance_samples,
        f_distance_samples=detached.f_distance_samples,
        q_quantiles=detached.q_quantiles,
        f_quantiles=detached.f_quantiles,
        version=detached.version,
    )


class Stage2Objective(nn.Module):
    """Compute the frozen Round-6 Stage-2 objective.

    Target stable labels are intentionally absent from this API. Synthetic CE
    consumes only true source labels carried by accepted synthetic source
    examples. Shape distributions are stop-gradient teachers, so only logits
    receive gradient through q-to-classifier consistency.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        config: Stage2ObjectiveConfig,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        if not isinstance(config, Stage2ObjectiveConfig):
            raise TypeError("config must be a Stage2ObjectiveConfig")
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError("eps must be finite and positive")
        self.num_classes = num_classes
        self.config = config
        self._stage1_geometry = Stage1Objective(
            num_classes=num_classes,
            lambda_q=0.0,
            lambda_f=1.0,
            lambda_q_to_cls=1.0,
            margin_q=0.0,
            margin_f=config.fused_margin,
            tau_q=config.tau_q,
            eps=eps,
        )

    def forward(
        self,
        *,
        source_logits: Tensor,
        source_fused_repr: Tensor,
        source_labels: Tensor,
        source_q: Tensor,
        source_q_support: Tensor,
        source_q_valid: Tensor,
        source_prototype_bank: SourcePrototypeBank,
        integration_weights: Tensor,
        synthetic_logits: Tensor | None = None,
        synthetic_labels: Tensor | None = None,
        synthetic_q: Tensor | None = None,
        synthetic_q_support: Tensor | None = None,
        synthetic_q_valid: Tensor | None = None,
        domain_shape_state: DomainShapeState | None = None,
        lambda_delta: float | None = None,
    ) -> Stage2ObjectiveOutput:
        if source_logits.ndim != 2 or source_logits.shape[1] != self.num_classes:
            raise ValueError("source_logits must have shape [B,C]")
        if source_labels.shape != (source_logits.shape[0],):
            raise ValueError("source_labels must have shape [B]")
        if source_labels.dtype != torch.long:
            raise ValueError("source_labels must use torch.long dtype")
        if source_fused_repr.shape[0] != source_logits.shape[0]:
            raise ValueError("source_fused_repr batch must match source_logits")
        if not isinstance(source_prototype_bank, SourcePrototypeBank):
            raise TypeError("source_prototype_bank must be a SourcePrototypeBank")

        bank = _detached_bank(source_prototype_bank)
        source_cls = F.cross_entropy(source_logits, source_labels)
        source_proto, _source_proto_count = self._stage1_geometry._prototype_fused_loss(
            source_fused_repr,
            source_labels,
            bank,
        )
        source_consistency, source_consistency_count = (
            self._stage1_geometry._q_to_classifier_loss(
                source_q,
                source_q_support,
                source_q_valid,
                source_logits,
                bank,
                integration_weights,
            )
        )

        synthetic_present = synthetic_logits is not None and synthetic_logits.shape[0] > 0
        if not synthetic_present:
            synthetic_cls = _zero_like(source_logits)
            synthetic_consistency = _zero_like(source_logits)
            synthetic_count = 0
            synthetic_consistency_count = 0
        else:
            if synthetic_logits.ndim != 2 or synthetic_logits.shape[1] != self.num_classes:
                raise ValueError("synthetic_logits must have shape [B_syn,C]")
            synthetic_count = int(synthetic_logits.shape[0])
            required = {
                "synthetic_labels": synthetic_labels,
                "synthetic_q": synthetic_q,
                "synthetic_q_support": synthetic_q_support,
                "synthetic_q_valid": synthetic_q_valid,
                "domain_shape_state": domain_shape_state,
                "lambda_delta": lambda_delta,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "synthetic inputs require " + ", ".join(missing)
                )
            assert synthetic_labels is not None
            assert synthetic_q is not None
            assert synthetic_q_support is not None
            assert synthetic_q_valid is not None
            assert domain_shape_state is not None
            assert lambda_delta is not None
            if synthetic_labels.shape != (synthetic_count,) or synthetic_labels.dtype != torch.long:
                raise ValueError("synthetic_labels must be torch.long with shape [B_syn]")
            # These are source true labels. No target-label or StableTargetLabel
            # object is accepted anywhere in this objective.
            synthetic_cls = F.cross_entropy(synthetic_logits, synthetic_labels)
            target_bank = _target_style_bank(
                source_prototype_bank,
                domain_shape_state,
                lambda_delta,
            )
            synthetic_consistency, synthetic_consistency_count = (
                self._stage1_geometry._q_to_classifier_loss(
                    synthetic_q,
                    synthetic_q_support,
                    synthetic_q_valid,
                    synthetic_logits,
                    target_bank,
                    integration_weights,
                )
            )

        total = (
            source_cls
            + self.config.lambda_src_proto * source_proto
            + self.config.lambda_src_cons * source_consistency
            + self.config.lambda_syn * synthetic_cls
            + self.config.lambda_syn_cons * synthetic_consistency
        )
        return Stage2ObjectiveOutput(
            total=total,
            source_cls=source_cls,
            source_proto=source_proto,
            source_consistency=source_consistency,
            synthetic_cls=synthetic_cls,
            synthetic_consistency=synthetic_consistency,
            source_count=int(source_logits.shape[0]),
            synthetic_count=synthetic_count,
            source_consistency_count=source_consistency_count,
            synthetic_consistency_count=synthetic_consistency_count,
        )
