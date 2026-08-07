from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from methods.structure_da.domain_shape_state import DomainShapeState, DomainShapeStatus
from methods.structure_da.prototype_bank import SourcePrototypeBank
from methods.structure_da.stage2_objective import (
    Stage2Objective,
    Stage2ObjectiveConfig,
)


def _weights(grid: int = 5) -> torch.Tensor:
    weights = torch.full((grid,), 1.0 / (grid - 1))
    weights[[0, -1]] *= 0.5
    return weights


def _config(**overrides):
    values = dict(
        lambda_src_proto=0.2,
        lambda_src_cons=0.3,
        lambda_syn=0.7,
        lambda_syn_cons=0.4,
        tau_q=0.2,
        fused_margin=0.1,
    )
    values.update(overrides)
    return Stage2ObjectiveConfig(**values)


def _bank(*, requires_grad: bool = False) -> SourcePrototypeBank:
    shape = torch.zeros(3, 5, 2)
    shape[1, :, 0] = 2.0
    shape[2, :, 1] = 2.0
    fused = torch.eye(3, 4)
    if requires_grad:
        shape.requires_grad_()
        fused.requires_grad_()
    return SourcePrototypeBank(
        trend_srvf=torch.zeros(3, 5, 2),
        shape_srvf=shape,
        trend_support=torch.ones(3, 5),
        shape_support=torch.ones(3, 5),
        fused=fused,
        class_counts=torch.full((3,), 10),
        ready=torch.ones(3, dtype=torch.bool),
        q_distance_samples=(torch.zeros(0),) * 3,
        f_distance_samples=(torch.zeros(0),) * 3,
        q_quantiles=torch.zeros(3, 3),
        f_quantiles=torch.zeros(3, 3),
        version=1,
    )


def _state(delta: torch.Tensor) -> DomainShapeState:
    return DomainShapeState(
        scan_index=1,
        status=DomainShapeStatus.CONFIRMED,
        class_centers=(),
        valid_classes=(0, 1, 2),
        delta=delta,
        interactions=(),
        rho_shape=1.0,
        leave_one_out_drift=0.0,
        center_drift=0.0,
        confirmation_age=2,
    )


def _source(batch: int = 3):
    labels = torch.arange(batch) % 3
    q = torch.zeros(batch, 5, 2)
    q[labels == 1, :, 0] = 2.0
    q[labels == 2, :, 1] = 2.0
    fused = torch.zeros(batch, 4)
    fused[torch.arange(batch), labels] = 1.0
    logits = torch.randn(batch, 3, requires_grad=True)
    return logits, fused.requires_grad_(), labels, q, torch.ones(batch, 5), torch.ones(batch, dtype=torch.bool)


def _call(obj, *, synthetic=False, bank=None, state=None, synthetic_q_requires_grad=False):
    source_logits, source_fused, labels, source_q, support, valid = _source()
    kwargs = dict(
        source_logits=source_logits,
        source_fused_repr=source_fused,
        source_labels=labels,
        source_q=source_q,
        source_q_support=support,
        source_q_valid=valid,
        source_prototype_bank=bank or _bank(),
        integration_weights=_weights(),
    )
    if synthetic:
        syn_q = source_q.clone().requires_grad_(synthetic_q_requires_grad)
        kwargs.update(
            synthetic_logits=torch.randn(3, 3, requires_grad=True),
            synthetic_labels=labels.clone(),
            synthetic_q=syn_q,
            synthetic_q_support=support.clone(),
            synthetic_q_valid=valid.clone(),
            domain_shape_state=state or _state(torch.zeros(5, 2)),
            lambda_delta=1.0,
        )
    return obj(**kwargs), kwargs


def test_only_source_available_produces_valid_total_and_safe_synthetic_zero() -> None:
    obj = Stage2Objective(num_classes=3, config=_config())
    out, kwargs = _call(obj)
    assert torch.isfinite(out.total)
    assert out.source_count == 3
    assert out.synthetic_count == 0
    assert out.synthetic_cls.item() == 0.0
    assert out.synthetic_consistency.item() == 0.0
    assert out.synthetic_cls.requires_grad
    out.total.backward()
    assert kwargs["source_logits"].grad is not None


def test_synthetic_ce_contributes_gradient_to_synthetic_logits() -> None:
    obj = Stage2Objective(
        num_classes=3,
        config=_config(
            lambda_src_proto=0.0,
            lambda_src_cons=0.0,
            lambda_syn=1.0,
            lambda_syn_cons=0.0,
        ),
    )
    out, kwargs = _call(obj, synthetic=True)
    out.total.backward()
    syn_logits = kwargs["synthetic_logits"]
    assert syn_logits.grad is not None and syn_logits.grad.abs().sum().item() > 0


def test_target_stable_labels_are_not_an_objective_input_or_ce_target() -> None:
    signature = inspect.signature(Stage2Objective.forward)
    assert "target_labels" not in signature.parameters
    assert "stable_target_labels" not in signature.parameters
    obj = Stage2Objective(num_classes=3, config=_config())
    _, kwargs = _call(obj)
    with pytest.raises(TypeError):
        obj(**kwargs, target_labels=torch.tensor([0, 1, 2]))


def test_source_shape_prototypes_and_delta_never_receive_gradient() -> None:
    bank = _bank(requires_grad=True)
    delta = torch.randn(5, 2, requires_grad=True)
    obj = Stage2Objective(num_classes=3, config=_config())
    out, _ = _call(obj, synthetic=True, bank=bank, state=_state(delta))
    out.total.backward()
    assert bank.shape_srvf.grad is None
    assert bank.fused.grad is None
    assert delta.grad is None


def test_synthetic_q_distribution_is_stop_gradient_teacher() -> None:
    obj = Stage2Objective(
        num_classes=3,
        config=_config(
            lambda_src_proto=0.0,
            lambda_src_cons=0.0,
            lambda_syn=0.0,
            lambda_syn_cons=1.0,
        ),
    )
    out, kwargs = _call(obj, synthetic=True, synthetic_q_requires_grad=True)
    out.synthetic_consistency.backward()
    assert kwargs["synthetic_logits"].grad is not None
    assert kwargs["synthetic_q"].grad is None


def test_empty_synthetic_tensor_is_treated_as_no_synthetic_sample() -> None:
    obj = Stage2Objective(num_classes=3, config=_config())
    source_logits, source_fused, labels, q, support, valid = _source()
    out = obj(
        source_logits=source_logits,
        source_fused_repr=source_fused,
        source_labels=labels,
        source_q=q,
        source_q_support=support,
        source_q_valid=valid,
        source_prototype_bank=_bank(),
        integration_weights=_weights(),
        synthetic_logits=torch.empty(0, 3, requires_grad=True),
    )
    assert out.synthetic_count == 0
    assert out.synthetic_cls.item() == 0.0
    assert out.synthetic_consistency_count == 0


def test_gradients_reach_only_stage2_representation_and_classifier_surrogates() -> None:
    torch.manual_seed(3)
    encoder = nn.Linear(4, 4)
    classifier = nn.Linear(4, 3)
    frozen_upstream = nn.Linear(4, 4)
    for parameter in frozen_upstream.parameters():
        parameter.requires_grad_(False)

    x = frozen_upstream(torch.randn(3, 4)).detach()
    source_fused = encoder(x)
    source_logits = classifier(source_fused)
    labels = torch.tensor([0, 1, 2])
    q = torch.zeros(3, 5, 2, requires_grad=True)
    support = torch.ones(3, 5)
    valid = torch.ones(3, dtype=torch.bool)
    bank = _bank(requires_grad=True)
    delta = torch.zeros(5, 2, requires_grad=True)

    synthetic_fused = encoder(x + 0.1)
    synthetic_logits = classifier(synthetic_fused)
    synthetic_q = torch.zeros(3, 5, 2, requires_grad=True)
    obj = Stage2Objective(num_classes=3, config=_config())
    out = obj(
        source_logits=source_logits,
        source_fused_repr=source_fused,
        source_labels=labels,
        source_q=q,
        source_q_support=support,
        source_q_valid=valid,
        source_prototype_bank=bank,
        integration_weights=_weights(),
        synthetic_logits=synthetic_logits,
        synthetic_labels=labels,
        synthetic_q=synthetic_q,
        synthetic_q_support=support,
        synthetic_q_valid=valid,
        domain_shape_state=_state(delta),
        lambda_delta=1.0,
    )
    out.total.backward()

    assert all(parameter.grad is not None for parameter in encoder.parameters())
    assert all(parameter.grad is not None for parameter in classifier.parameters())
    assert all(parameter.grad is None for parameter in frozen_upstream.parameters())
    assert q.grad is None
    assert synthetic_q.grad is None
    assert bank.shape_srvf.grad is None
    assert bank.fused.grad is None
    assert delta.grad is None
