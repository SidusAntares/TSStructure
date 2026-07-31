from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from methods.structure_da.quality_fusion import (
    ComponentQualityBundle,
    HierarchicalQualityFusion,
    HierarchicalQualityObjective,
    HierarchicalQualityOutput,
    QualityLossOutput,
    QualityScoreOutput,
    QualityScorer,
    StructuralQualityBundle,
)


def _scorer(dtype: torch.dtype = torch.float32) -> QualityScorer:
    torch.manual_seed(401)
    return QualityScorer(4, 3, domain_hidden_dim=5).to(dtype=dtype)


def _fusion(dtype: torch.dtype = torch.float32) -> HierarchicalQualityFusion:
    torch.manual_seed(402)
    return HierarchicalQualityFusion(
        component_dim=4,
        structure_dim=3,
        num_classes=3,
        domain_hidden_dim=5,
    ).to(dtype=dtype)


def _fusion_inputs(dtype: torch.dtype = torch.float32):
    torch.manual_seed(403)
    embeddings = [torch.randn(4, 4, dtype=dtype) for _ in range(3)]
    structures = [torch.randn(4, 3, dtype=dtype) for _ in range(4)]
    component_valid = torch.tensor([True, True, False, True])
    structure_valid = [
        torch.tensor([True, True, False, False]),
        torch.tensor([True, False, False, True]),
        torch.tensor([True, True, False, True]),
        torch.tensor([False, True, False, True]),
    ]
    return (*embeddings, *structures, component_valid, *structure_valid)


def test_quality_scorer_has_exact_classifier_architecture() -> None:
    scorer = _scorer()

    assert isinstance(scorer.class_classifier, nn.Linear)
    assert scorer.class_classifier.in_features == 4
    assert scorer.class_classifier.out_features == 3
    assert [type(module) for module in scorer.domain_classifier] == [
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
    ]
    assert scorer.domain_classifier[0].in_features == 4
    assert scorer.domain_classifier[-1].out_features == 2


def test_quality_scorer_contains_no_forbidden_modules() -> None:
    scorer = _scorer()

    assert not any(
        isinstance(module, (nn.Sigmoid, nn.LayerNorm, nn.Dropout))
        for module in scorer.modules()
    )


def test_quality_output_shapes_and_score_bounds() -> None:
    scorer = _scorer()
    feature = torch.randn(6, 4)

    output = scorer(feature)

    assert isinstance(output, QualityScoreOutput)
    assert output.domain_logits.shape == (6, 2)
    assert output.class_logits.shape == (6, 3)
    assert output.valid.shape == (6,)
    for score in (
        output.coefficient,
        output.domain_invariance,
        output.entropy_score,
        output.confidence_score,
        output.discriminability,
    ):
        assert score.shape == (6,)
        assert score.min().item() >= 0.0
        assert score.max().item() <= 1.0


def test_uniform_class_probabilities_have_zero_entropy_and_confidence_scores() -> None:
    scorer = _scorer()
    with torch.no_grad():
        scorer.class_classifier.weight.zero_()
        scorer.class_classifier.bias.zero_()

    output = scorer(torch.zeros(2, 4))

    torch.testing.assert_close(output.entropy_score, torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(output.confidence_score, torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(output.discriminability, torch.zeros(2), atol=1e-6, rtol=0)


def test_high_confidence_class_probability_scores_approach_one() -> None:
    scorer = _scorer()
    with torch.no_grad():
        scorer.class_classifier.weight.zero_()
        scorer.class_classifier.bias.copy_(torch.tensor([20.0, -20.0, -20.0]))

    output = scorer(torch.zeros(1, 4))

    assert output.entropy_score.item() > 0.999
    assert output.confidence_score.item() > 0.999
    assert output.discriminability.item() > 0.999


def test_half_source_probability_has_unit_domain_invariance() -> None:
    scorer = _scorer()
    with torch.no_grad():
        for module in scorer.domain_classifier:
            if isinstance(module, nn.Linear):
                module.weight.zero_()
                module.bias.zero_()

    output = scorer(torch.zeros(2, 4))

    torch.testing.assert_close(output.domain_invariance, torch.ones(2))


@pytest.mark.parametrize("bias", [(-20.0, 20.0), (20.0, -20.0)])
def test_extreme_domain_probability_has_near_zero_invariance(bias) -> None:
    scorer = _scorer()
    with torch.no_grad():
        for module in scorer.domain_classifier:
            if isinstance(module, nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        scorer.domain_classifier[-1].bias.copy_(torch.tensor(bias))

    output = scorer(torch.zeros(1, 4))

    assert output.domain_invariance.item() < 1e-6


def test_quality_coefficient_matches_exact_formula() -> None:
    scorer = _scorer()
    feature = torch.randn(5, 4)

    output = scorer(feature)
    expected = (
        0.50 * output.domain_invariance
        + 0.25 * output.entropy_score
        + 0.25 * output.confidence_score
    ).clamp(0.0, 1.0)

    torch.testing.assert_close(output.coefficient, expected)
    torch.testing.assert_close(
        output.discriminability,
        0.5 * (output.entropy_score + output.confidence_score),
    )


def test_invalid_only_zeros_final_coefficient() -> None:
    scorer = _scorer()
    feature = torch.randn(3, 4)
    valid = torch.tensor([True, False, True])

    masked = scorer(feature, valid)
    unmasked = scorer(feature)

    assert masked.coefficient[1].item() == 0
    torch.testing.assert_close(masked.coefficient[valid], unmasked.coefficient[valid])
    for field in (
        "domain_invariance",
        "entropy_score",
        "confidence_score",
        "discriminability",
        "domain_logits",
        "class_logits",
    ):
        torch.testing.assert_close(getattr(masked, field), getattr(unmasked, field))


def test_quality_scorer_gradients_reach_feature_and_both_heads() -> None:
    scorer = _scorer()
    feature = torch.randn(5, 4, requires_grad=True)

    output = scorer(feature)
    (output.coefficient.sum() + output.class_logits.square().mean() + output.domain_logits.square().mean()).backward()

    assert feature.grad is not None and feature.grad.abs().sum() > 0
    for parameter in scorer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_hierarchical_scorer_sharing_and_independence_are_exact() -> None:
    fusion = _fusion()

    assert isinstance(fusion.temporal_quality, QualityScorer)
    assert isinstance(fusion.channel_quality, QualityScorer)
    assert set(fusion.component_quality) == {"trend", "dynamics", "residual"}
    groups = [
        fusion.temporal_quality,
        fusion.channel_quality,
        fusion.component_quality["trend"],
        fusion.component_quality["dynamics"],
        fusion.component_quality["residual"],
    ]
    parameter_sets = [{id(parameter) for parameter in group.parameters()} for group in groups]
    assert all(
        parameter_sets[left].isdisjoint(parameter_sets[right])
        for left in range(len(groups))
        for right in range(left + 1, len(groups))
    )
    assert not hasattr(fusion, "diversity")
    assert not hasattr(fusion, "warmup")


class _FixedScorer(nn.Module):
    def __init__(self, coefficient: float, num_classes: int = 3) -> None:
        super().__init__()
        self.coefficient = coefficient
        self.num_classes = num_classes

    def forward(self, feature: torch.Tensor, valid: torch.Tensor | None = None):
        batch = feature.shape[0]
        if valid is None:
            valid = torch.ones(batch, dtype=torch.bool, device=feature.device)
        coefficient = feature.new_full((batch,), self.coefficient)
        coefficient = torch.where(valid, coefficient, torch.zeros_like(coefficient))
        score = feature.new_full((batch,), 0.5)
        return QualityScoreOutput(
            coefficient=coefficient,
            domain_invariance=score,
            entropy_score=score,
            confidence_score=score,
            discriminability=score,
            domain_logits=feature.new_zeros(batch, 2),
            class_logits=feature.new_zeros(batch, self.num_classes),
            valid=valid,
        )


def test_fusion_matches_exact_same_semantic_weighted_sum() -> None:
    fusion = _fusion()
    fusion.temporal_quality = _FixedScorer(0.2)
    fusion.channel_quality = _FixedScorer(0.4)
    fusion.component_quality = nn.ModuleDict(
        {
            "trend": _FixedScorer(0.3),
            "dynamics": _FixedScorer(0.5),
            "residual": _FixedScorer(0.7),
        }
    )
    inputs = list(_fusion_inputs())
    inputs[7] = torch.ones(4, dtype=torch.bool)
    inputs[8:] = [torch.ones(4, dtype=torch.bool) for _ in range(4)]

    output = fusion(*inputs)
    trend, dynamics, residual = inputs[:3]
    trend_temporal, dynamics_temporal, trend_channel, dynamics_channel = inputs[3:7]

    torch.testing.assert_close(output.weighted_trend_temporal, 0.2 * trend_temporal)
    torch.testing.assert_close(output.weighted_dynamics_temporal, 0.2 * dynamics_temporal)
    torch.testing.assert_close(output.weighted_trend_channel, 0.4 * trend_channel)
    torch.testing.assert_close(output.weighted_dynamics_channel, 0.4 * dynamics_channel)
    torch.testing.assert_close(output.raw_fusion, 0.3 * trend + 0.5 * dynamics + 0.7 * residual)
    torch.testing.assert_close(
        output.temporal_fusion,
        0.3 * 0.2 * trend_temporal + 0.5 * 0.2 * dynamics_temporal,
    )
    torch.testing.assert_close(
        output.channel_fusion,
        0.3 * 0.4 * trend_channel + 0.5 * 0.4 * dynamics_channel,
    )
    torch.testing.assert_close(
        output.fused_feature,
        torch.cat([output.raw_fusion, output.temporal_fusion, output.channel_fusion], dim=-1),
    )


def test_residual_component_input_has_two_exact_zero_structure_slots() -> None:
    output = _fusion()(*_fusion_inputs())

    assert output.residual_component_input.shape == (4, 10)
    torch.testing.assert_close(
        output.residual_component_input[:, 4:],
        torch.zeros(4, 6),
        atol=0,
        rtol=0,
    )


def test_invalid_structure_has_zero_beta_and_weighted_feature() -> None:
    output = _fusion()(*_fusion_inputs())
    valids = _fusion_inputs()[8:]
    pairs = [
        (output.beta_trend_temporal, output.weighted_trend_temporal),
        (output.beta_dynamics_temporal, output.weighted_dynamics_temporal),
        (output.beta_trend_channel, output.weighted_trend_channel),
        (output.beta_dynamics_channel, output.weighted_dynamics_channel),
    ]

    for valid, (beta, weighted) in zip(valids, pairs):
        assert torch.count_nonzero(beta[~valid]) == 0
        assert torch.count_nonzero(weighted[~valid]) == 0


def test_alpha_and_beta_are_not_softmax_competitions() -> None:
    fusion = _fusion()
    fusion.temporal_quality = _FixedScorer(0.8)
    fusion.channel_quality = _FixedScorer(0.8)
    fusion.component_quality = nn.ModuleDict(
        {name: _FixedScorer(0.8) for name in ("trend", "dynamics", "residual")}
    )
    inputs = list(_fusion_inputs())
    inputs[7:] = [torch.ones(4, dtype=torch.bool) for _ in range(5)]

    output = fusion(*inputs)

    torch.testing.assert_close(
        output.alpha_trend + output.alpha_dynamics + output.alpha_residual,
        torch.full((4,), 2.4),
    )
    torch.testing.assert_close(
        output.beta_trend_temporal + output.beta_dynamics_temporal,
        torch.full((4,), 1.6),
    )
    assert not hasattr(fusion, "fusion_projection")


def test_hierarchical_output_shapes_are_complete() -> None:
    output = _fusion()(*_fusion_inputs())

    assert isinstance(output, HierarchicalQualityOutput)
    assert isinstance(output.structural, StructuralQualityBundle)
    assert isinstance(output.component, ComponentQualityBundle)
    assert output.trend_component_input.shape == (4, 10)
    assert output.dynamics_component_input.shape == (4, 10)
    assert output.raw_fusion.shape == (4, 4)
    assert output.temporal_fusion.shape == (4, 3)
    assert output.channel_fusion.shape == (4, 3)
    assert output.fused_feature.shape == (4, 10)


def _manual_branch_mean(branches, labels, masks, *, domain: bool):
    logits = []
    targets = []
    for branch, mask in zip(branches, masks):
        selected = branch.domain_logits if domain else branch.class_logits
        logits.append(selected[mask])
        targets.append(labels[mask])
    return F.cross_entropy(torch.cat(logits), torch.cat(targets))


def test_quality_objective_uses_exact_branch_sample_global_means() -> None:
    quality = _fusion()(*_fusion_inputs())
    class_labels = torch.tensor([0, 1, 2, 1])
    domain_labels = torch.tensor([1, 1, 0, 0])
    source_mask = domain_labels == 1
    objective = HierarchicalQualityObjective()

    output = objective(quality, class_labels, domain_labels, source_mask)
    structural = [
        quality.structural.trend_temporal,
        quality.structural.dynamics_temporal,
        quality.structural.trend_channel,
        quality.structural.dynamics_channel,
    ]
    components = [quality.component.trend, quality.component.dynamics, quality.component.residual]
    torch.testing.assert_close(
        output.structural_classification_loss,
        _manual_branch_mean(structural, class_labels, [source_mask & branch.valid for branch in structural], domain=False),
    )
    torch.testing.assert_close(
        output.structural_domain_loss,
        _manual_branch_mean(structural, domain_labels, [branch.valid for branch in structural], domain=True),
    )
    torch.testing.assert_close(
        output.component_classification_loss,
        _manual_branch_mean(components, class_labels, [source_mask & branch.valid for branch in components], domain=False),
    )
    torch.testing.assert_close(
        output.component_domain_loss,
        _manual_branch_mean(components, domain_labels, [branch.valid for branch in components], domain=True),
    )


def test_quality_objective_counts_active_branch_samples() -> None:
    quality = _fusion()(*_fusion_inputs())
    class_labels = torch.tensor([0, 1, 2, 1])
    domain_labels = torch.tensor([1, 1, 0, 0])

    output = HierarchicalQualityObjective()(quality, class_labels, domain_labels, domain_labels == 1)

    assert output.structural_classification_count.item() == 6
    assert output.structural_domain_count.item() == 9
    assert output.component_classification_count.item() == 6
    assert output.component_domain_count.item() == 9


def test_quality_objective_four_weights_are_independent() -> None:
    quality = _fusion()(*_fusion_inputs())
    labels = torch.tensor([0, 1, 2, 1])
    domains = torch.tensor([1, 1, 0, 0])
    objective = HierarchicalQualityObjective(2.0, 3.0, 4.0, 5.0)

    output = objective(quality, labels, domains, domains == 1)
    expected = (
        2.0 * output.structural_classification_loss
        + 3.0 * output.structural_domain_loss
        + 4.0 * output.component_classification_loss
        + 5.0 * output.component_domain_loss
    )

    assert isinstance(output, QualityLossOutput)
    torch.testing.assert_close(output.total_loss, expected)


def test_no_active_quality_samples_return_graph_connected_zero() -> None:
    inputs = list(_fusion_inputs())
    inputs[7:] = [torch.zeros(4, dtype=torch.bool) for _ in range(5)]
    fusion = _fusion()
    quality = fusion(*inputs)
    labels = torch.tensor([0, 1, 2, 1])
    domains = torch.tensor([0, 0, 0, 0])

    output = HierarchicalQualityObjective()(quality, labels, domains, domains == 1)
    output.total_loss.backward()

    assert output.total_loss.item() == 0
    for count in (
        output.structural_classification_count,
        output.structural_domain_count,
        output.component_classification_count,
        output.component_domain_count,
    ):
        assert count.item() == 0
    assert all(parameter.grad is not None for parameter in fusion.parameters())


def test_source_mask_must_match_domain_labels() -> None:
    quality = _fusion()(*_fusion_inputs())
    labels = torch.tensor([0, 1, 2, 1])
    domains = torch.tensor([1, 1, 0, 0])

    with pytest.raises(ValueError, match="source_mask"):
        HierarchicalQualityObjective()(quality, labels, domains, torch.ones(4, dtype=torch.bool))


def test_task_fusion_gradients_reach_all_inputs_and_quality_scorers() -> None:
    fusion = _fusion()
    inputs = list(_fusion_inputs())
    for index in range(7):
        inputs[index] = inputs[index].requires_grad_()
    inputs[7:] = [torch.ones(4, dtype=torch.bool) for _ in range(5)]

    output = fusion(*inputs)
    weights = torch.linspace(0.2, 1.1, output.fused_feature.shape[-1])
    (output.fused_feature * weights).sum().backward()

    for tensor in inputs[:7]:
        assert tensor.grad is not None and tensor.grad.abs().sum() > 0
    for parameter in fusion.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_quality_auxiliary_losses_update_heads_and_feature_inputs() -> None:
    fusion = _fusion()
    inputs = list(_fusion_inputs())
    for index in range(7):
        inputs[index] = inputs[index].requires_grad_()
    inputs[7:] = [torch.ones(4, dtype=torch.bool) for _ in range(5)]
    quality = fusion(*inputs)
    class_labels = torch.tensor([0, 1, 2, 1])
    domain_labels = torch.tensor([1, 1, 0, 0])

    losses = HierarchicalQualityObjective()(
        quality,
        class_labels,
        domain_labels,
        domain_labels == 1,
    )
    losses.total_loss.backward()

    for tensor in inputs[:7]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    for parameter in fusion.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quality_fusion_preserves_cpu_dtype(dtype: torch.dtype) -> None:
    output = _fusion(dtype)(*_fusion_inputs(dtype))

    assert output.fused_feature.dtype == dtype
    assert output.alpha_trend.dtype == dtype
    assert output.beta_trend_temporal.dtype == dtype


@pytest.mark.parametrize(
    "factory,match",
    [
        (lambda: QualityScorer(0, 3), "input_dim"),
        (lambda: QualityScorer(4, 1), "num_classes"),
        (lambda: QualityScorer(4, 3, domain_hidden_dim=0), "domain_hidden_dim"),
        (lambda: QualityScorer(4, 3, eps=0.0), "eps"),
        (lambda: HierarchicalQualityFusion(0, 3, 3), "component_dim"),
        (lambda: HierarchicalQualityFusion(4, 0, 3), "structure_dim"),
        (lambda: HierarchicalQualityObjective(-1.0), "weight"),
    ],
)
def test_invalid_constructor_arguments_raise_value_error(factory, match) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_quality_scorer_rejects_invalid_feature_or_valid_mask() -> None:
    scorer = _scorer()
    with pytest.raises(ValueError, match="feature"):
        scorer(torch.ones(2, 3))
    with pytest.raises(ValueError, match="finite"):
        scorer(torch.full((2, 4), float("nan")))
    with pytest.raises(ValueError, match="valid"):
        scorer(torch.ones(2, 4), torch.ones(2))
