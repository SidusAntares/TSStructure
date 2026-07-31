import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from methods.structure_da.eden_alignment import (
    EDENDomainDiscriminator,
    EDENFusedFeatureAlignment,
    WarmStartGradientReverseLayer,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grl_preserves_values_and_reverses_gradient(dtype) -> None:
    layer = WarmStartGradientReverseLayer(
        alpha=1.0, low=0.0, high=1.0, max_iters=10, weight=2.0
    )
    layer.iteration.fill_(5)
    value = torch.randn(3, 4, dtype=dtype, requires_grad=True)

    output = layer(value)
    coefficient = 2.0 * (2.0 / (1.0 + math.exp(-0.5)) - 1.0)
    output.sum().backward()

    torch.testing.assert_close(output, value.detach())
    torch.testing.assert_close(
        value.grad, torch.full_like(value, -coefficient)
    )
    assert layer.iteration.item() == 6


def test_grl_schedule_reset_and_state_dict() -> None:
    layer = WarmStartGradientReverseLayer(max_iters=4)
    assert layer.get_coefficient() == pytest.approx(0.0)
    layer(torch.ones(1, 2))
    assert layer.iteration.item() == 1
    layer.iteration.fill_(4)
    expected = 2.0 / (1.0 + math.exp(-1.0)) - 1.0
    assert layer.get_coefficient() == pytest.approx(expected)
    assert layer.get_coefficient() != pytest.approx(1.0)
    layer(torch.ones(1, 2))
    assert layer.last_coefficient.item() == pytest.approx(expected)
    assert {"iteration", "last_coefficient"} <= set(layer.state_dict())
    layer.reset()
    assert layer.iteration.item() == 0
    assert layer.last_coefficient.item() == pytest.approx(0.0)


def test_domain_discriminator_has_exact_architecture() -> None:
    discriminator = EDENDomainDiscriminator(7, hidden_dim=5)
    assert [type(module) for module in discriminator.network] == [
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
    ]
    assert len([m for m in discriminator.modules() if isinstance(m, nn.Linear)]) == 3
    assert discriminator.network[0].in_features == 7
    assert discriminator.network[-1].out_features == 2
    assert not any(
        isinstance(module, (nn.Sigmoid, nn.Softmax, nn.BatchNorm1d, nn.LayerNorm, nn.Dropout))
        for module in discriminator.modules()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_alignment_uses_source_one_target_zero_and_keeps_gradients(dtype) -> None:
    torch.manual_seed(801)
    alignment = EDENFusedFeatureAlignment(
        feature_dim=4, hidden_dim=6, grl_max_iters=10
    ).to(dtype=dtype)
    alignment.grl.iteration.fill_(3)
    source = torch.randn(2, 4, dtype=dtype, requires_grad=True)
    target = torch.randn(3, 4, dtype=dtype, requires_grad=True)

    output = alignment(source, target)
    expected_labels = torch.tensor([1, 1, 0, 0, 0])

    assert torch.equal(output.labels, expected_labels)
    assert output.source_batch_size == 2
    assert output.target_batch_size == 3
    assert output.logits.shape == (5, 2)
    torch.testing.assert_close(output.loss, F.cross_entropy(output.logits, output.labels))
    torch.testing.assert_close(
        output.accuracy,
        (output.logits.argmax(-1) == output.labels).float().mean(),
    )
    output.loss.backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert target.grad is not None and torch.isfinite(target.grad).all()
    assert source.grad.abs().sum() > 0
    assert target.grad.abs().sum() > 0
    assert alignment.grl.iteration.item() == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -1.0},
        {"low": 1.0, "high": 0.0},
        {"max_iters": 0},
        {"weight": -1.0},
    ],
)
def test_grl_rejects_invalid_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        WarmStartGradientReverseLayer(**kwargs)


def test_alignment_rejects_invalid_inputs() -> None:
    alignment = EDENFusedFeatureAlignment(4)
    with pytest.raises(ValueError):
        alignment(torch.randn(2, 4, 1), torch.randn(2, 4))
    with pytest.raises(ValueError):
        alignment(torch.randn(2, 4), torch.randn(2, 5))
    with pytest.raises(ValueError):
        alignment(torch.empty(0, 4), torch.randn(2, 4))
