"""Tests for the standalone two-level quality measurement modules."""

import pytest
import torch
from torch import nn

from methods.structure_da import (
    ComponentQualityPerception,
    DiscriminabilityScorer,
    DiversityScorer,
    StructuralQualityPerception,
    TransferabilityScorer,
)


def _set_linear_bias(linear, values):
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.copy_(torch.tensor(values, dtype=linear.bias.dtype))


def test_transferability_shapes_and_range():
    scorer = TransferabilityScorer(input_dim=5, hidden_cap=3)

    domain_logits, transferability = scorer(torch.randn(4, 5))

    assert scorer.hidden.out_features == 3
    assert domain_logits.shape == (4, 2)
    assert transferability.shape == (4,)
    assert torch.all((transferability >= 0) & (transferability <= 1))


@pytest.mark.parametrize(
    "bias, expected",
    [
        ([0.0, 0.0], 1.0),
        ([-10.0, 10.0], 0.0),
        ([10.0, -10.0], 0.0),
    ],
)
def test_transferability_formula_rewards_domain_ambiguity(bias, expected):
    scorer = TransferabilityScorer(input_dim=4)
    _set_linear_bias(scorer.domain_head, bias)

    _, transferability = scorer(torch.randn(3, 4))

    torch.testing.assert_close(
        transferability,
        torch.full_like(transferability, expected),
        atol=1e-7 if expected == 1.0 else 1e-6,
        rtol=0,
    )


def test_discriminability_shapes_and_ranges():
    scorer = DiscriminabilityScorer(input_dim=5, num_classes=3)

    class_logits, entropy, confidence = scorer(torch.randn(4, 5))

    assert class_logits.shape == (4, 3)
    assert entropy.shape == (4,)
    assert confidence.shape == (4,)
    assert torch.all((entropy >= 0) & (entropy <= 1))
    assert torch.all((confidence >= 0) & (confidence <= 1))


def test_uniform_class_logits_have_zero_entropy_quality():
    scorer = DiscriminabilityScorer(input_dim=4, num_classes=4)
    _set_linear_bias(scorer.class_head, [0.0, 0.0, 0.0, 0.0])

    _, entropy, confidence = scorer(torch.randn(2, 4))

    torch.testing.assert_close(entropy, torch.zeros_like(entropy), atol=1e-7, rtol=0)
    torch.testing.assert_close(
        confidence, torch.full_like(confidence, 0.25), atol=1e-7, rtol=0
    )


def test_confident_class_logits_have_near_one_entropy_and_confidence():
    scorer = DiscriminabilityScorer(input_dim=4, num_classes=3)
    _set_linear_bias(scorer.class_head, [20.0, -20.0, -20.0])

    _, entropy, confidence = scorer(torch.randn(2, 4))

    assert torch.all(entropy > 0.999)
    assert torch.all(confidence > 0.999)


def test_diversity_shape_and_range():
    scorer = DiversityScorer(input_dim=5)

    diversity = scorer(torch.randn(4, 5))

    assert diversity.shape == (4,)
    assert torch.all((diversity >= 0) & (diversity <= 1))


def test_structural_quality_uses_exact_three_score_mean():
    quality = StructuralQualityPerception(input_dim=4, num_classes=3)
    features = torch.randn(3, 4)

    output = quality(features)

    expected = (
        output.scores.transferability
        + output.scores.entropy
        + output.scores.confidence
    ) / 3
    torch.testing.assert_close(output.raw_gate, expected)
    assert torch.all((output.raw_gate >= 0) & (output.raw_gate <= 1))


def test_component_quality_uses_exact_base_and_eta_weighted_formula():
    quality = ComponentQualityPerception(
        input_dim=4, num_classes=3, eta=0.25
    )
    features = torch.randn(3, 4)

    output = quality(features)

    expected_base = (
        output.scores.transferability
        + output.scores.entropy
        + output.scores.confidence
    ) / 3
    expected_gate = (expected_base + 0.25 * output.diversity) / 1.25
    torch.testing.assert_close(output.raw_base_quality, expected_base)
    torch.testing.assert_close(output.raw_gate, expected_gate)
    assert torch.all((output.raw_gate >= 0) & (output.raw_gate <= 1))


def test_component_eta_zero_makes_gate_equal_base_quality():
    output = ComponentQualityPerception(
        input_dim=4, num_classes=3, eta=0
    )(torch.randn(3, 4))

    torch.testing.assert_close(output.raw_gate, output.raw_base_quality)


@pytest.mark.parametrize("eta", [-0.1, float("nan"), float("inf"), True])
def test_component_quality_rejects_invalid_eta(eta):
    with pytest.raises(ValueError, match="eta"):
        ComponentQualityPerception(input_dim=4, num_classes=3, eta=eta)


@pytest.mark.parametrize(
    "quality",
    [
        StructuralQualityPerception(input_dim=4, num_classes=3),
        ComponentQualityPerception(input_dim=4, num_classes=3),
    ],
)
def test_raw_gate_remains_trainable_while_public_gate_is_detached(quality):
    output = quality(torch.randn(3, 4))

    assert output.raw_gate.requires_grad
    assert not output.gate.requires_grad
    torch.testing.assert_close(output.gate, output.raw_gate)


@pytest.mark.parametrize(
    "quality",
    [
        StructuralQualityPerception(input_dim=4, num_classes=3),
        ComponentQualityPerception(input_dim=4, num_classes=3),
    ],
)
def test_quality_objective_stops_input_gradient_but_trains_all_parameters(quality):
    features = torch.randn(4, 4, requires_grad=True)
    output = quality(features)
    loss = (
        output.scores.domain_logits.sum()
        + output.scores.class_logits.sum()
        + output.raw_gate.sum()
    )
    if hasattr(output, "diversity"):
        loss = loss + output.diversity.sum()

    loss.backward()

    assert features.grad is None
    for parameter in quality.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: StructuralQualityPerception(input_dim=4, num_classes=3),
        lambda: ComponentQualityPerception(input_dim=4, num_classes=3),
    ],
)
def test_distinct_quality_instances_do_not_share_linear_parameters(factory):
    first = factory()
    second = factory()
    first_linear_parameters = [
        parameter
        for module in first.modules()
        if isinstance(module, nn.Linear)
        for parameter in module.parameters()
    ]
    second_linear_parameters = [
        parameter
        for module in second.modules()
        if isinstance(module, nn.Linear)
        for parameter in module.parameters()
    ]

    assert len(first_linear_parameters) == len(second_linear_parameters)
    assert all(
        first_parameter is not second_parameter
        for first_parameter, second_parameter in zip(
            first_linear_parameters, second_linear_parameters
        )
    )


def test_component_gates_have_no_cross_component_softmax_normalization():
    qualities = [
        ComponentQualityPerception(input_dim=4, num_classes=3)
        for _ in range(3)
    ]
    for quality in qualities:
        for parameter in quality.parameters():
            nn.init.zeros_(parameter)
    features = torch.randn(2, 4)

    gates = torch.stack([quality(features).gate for quality in qualities])

    assert not any(
        isinstance(module, nn.Softmax)
        for quality in qualities
        for module in quality.modules()
    )
    assert not torch.allclose(gates.sum(dim=0), torch.ones(2))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TransferabilityScorer(input_dim=4),
        lambda: DiscriminabilityScorer(input_dim=4, num_classes=3),
        lambda: DiversityScorer(input_dim=4),
    ],
)
@pytest.mark.parametrize(
    "features",
    [torch.randn(2, 3, 4), torch.randn(2, 4, 4), torch.randn(2, 5)],
)
def test_scorers_reject_invalid_input_shapes(factory, features):
    with pytest.raises(ValueError, match="shape"):
        factory()(features)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TransferabilityScorer(input_dim=4),
        lambda: DiscriminabilityScorer(input_dim=4, num_classes=3),
        lambda: DiversityScorer(input_dim=4),
    ],
)
def test_scorers_reject_non_floating_inputs(factory):
    with pytest.raises(ValueError, match="floating"):
        factory()(torch.ones(2, 4, dtype=torch.long))


def test_quality_outputs_are_finite_for_ordinary_fp32_features():
    features = torch.randn(5, 4)
    structural = StructuralQualityPerception(4, 3)(features)
    component = ComponentQualityPerception(4, 3)(features)
    tensors = [
        structural.scores.transferability,
        structural.scores.entropy,
        structural.scores.confidence,
        structural.scores.domain_logits,
        structural.scores.class_logits,
        structural.raw_gate,
        structural.gate,
        component.diversity,
        component.raw_base_quality,
        component.raw_gate,
        component.gate,
    ]

    assert all(torch.isfinite(value).all() for value in tensors)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TransferabilityScorer(input_dim=0),
        lambda: TransferabilityScorer(input_dim=4, hidden_cap=0),
        lambda: DiscriminabilityScorer(input_dim=4, num_classes=1),
        lambda: DiversityScorer(input_dim=-1),
    ],
)
def test_scorer_constructor_validation(factory):
    with pytest.raises(ValueError):
        factory()
