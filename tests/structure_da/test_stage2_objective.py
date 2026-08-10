from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from methods.structure_da.stage2_objective import (
    Stage2Objective,
    Stage2ObjectiveConfig,
)


def _config(**overrides):
    values = dict(lambda_target=1.0, focal_gamma=1.0)
    values.update(overrides)
    return Stage2ObjectiveConfig(**values)


def test_objective_has_exactly_two_training_branches() -> None:
    signature = inspect.signature(Stage2Objective.forward)
    assert tuple(signature.parameters) == (
        "self",
        "source_to_target_logits",
        "source_labels",
        "native_target_logits",
        "stable_target_labels",
    )
    assert "source_prototype_bank" not in signature.parameters
    assert "domain_shape_state" not in signature.parameters


def test_source_to_target_focal_loss_backpropagates() -> None:
    obj = Stage2Objective(num_classes=3, config=_config(lambda_target=0.0))
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    out = obj(source_to_target_logits=logits, source_labels=labels)
    assert out.source_count == 4
    assert out.target_count == 0
    assert out.native_target.item() == 0.0
    out.total.backward()
    assert logits.grad is not None and logits.grad.abs().sum().item() > 0


def test_native_target_stable_labels_contribute_gradient() -> None:
    obj = Stage2Objective(num_classes=3, config=_config(lambda_target=0.7))
    source_logits = torch.randn(3, 3, requires_grad=True)
    target_logits = torch.randn(2, 3, requires_grad=True)
    source_labels = torch.tensor([0, 1, 2], dtype=torch.long)
    stable_labels = torch.tensor([2, 1], dtype=torch.long)
    out = obj(
        source_to_target_logits=source_logits,
        source_labels=source_labels,
        native_target_logits=target_logits,
        stable_target_labels=stable_labels,
    )
    assert out.source_count == 3
    assert out.target_count == 2
    out.total.backward()
    assert source_logits.grad is not None
    assert target_logits.grad is not None and target_logits.grad.abs().sum().item() > 0


def test_stable_target_labels_are_stop_gradient_integer_supervision() -> None:
    obj = Stage2Objective(num_classes=3, config=_config())
    with pytest.raises(ValueError, match="torch.long"):
        obj(
            source_to_target_logits=torch.randn(3, 3, requires_grad=True),
            source_labels=torch.tensor([0, 1, 2], dtype=torch.long),
            native_target_logits=torch.randn(2, 3, requires_grad=True),
            stable_target_labels=torch.tensor([0.0, 1.0]),
        )


def test_target_logits_require_stable_labels() -> None:
    obj = Stage2Objective(num_classes=3, config=_config())
    with pytest.raises(ValueError, match="require Stable Labels"):
        obj(
            source_to_target_logits=torch.randn(3, 3),
            source_labels=torch.tensor([0, 1, 2], dtype=torch.long),
            native_target_logits=torch.randn(2, 3),
        )


def test_gradients_reach_student_representation_and_classifier_only() -> None:
    torch.manual_seed(3)
    encoder = nn.Linear(4, 4)
    classifier = nn.Linear(4, 3)
    frozen_upstream = nn.Linear(4, 4)
    for parameter in frozen_upstream.parameters():
        parameter.requires_grad_(False)

    x = frozen_upstream(torch.randn(4, 4)).detach()
    source_repr = encoder(x[:2])
    target_repr = encoder(x[2:])
    source_logits = classifier(source_repr)
    target_logits = classifier(target_repr)
    obj = Stage2Objective(num_classes=3, config=_config())
    out = obj(
        source_to_target_logits=source_logits,
        source_labels=torch.tensor([0, 1], dtype=torch.long),
        native_target_logits=target_logits,
        stable_target_labels=torch.tensor([1, 2], dtype=torch.long),
    )
    out.total.backward()
    assert all(parameter.grad is not None for parameter in encoder.parameters())
    assert all(parameter.grad is not None for parameter in classifier.parameters())
    assert all(parameter.grad is None for parameter in frozen_upstream.parameters())
