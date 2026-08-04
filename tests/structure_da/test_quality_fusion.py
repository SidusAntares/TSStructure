from __future__ import annotations

import torch

from methods.structure_da.quality_fusion import (
    QualityScorer,
    TwoScaleQualityFusion,
    TwoScaleQualityObjective,
    concatenate_two_scale_quality_outputs,
)

def test_quality_scorer_coefficients_are_bounded_and_invalid_is_zero() -> None:
    scorer = QualityScorer(4, 3, domain_hidden_dim=5)
    output = scorer(torch.randn(3, 4), torch.tensor([True, False, True]))
    assert torch.all((0 <= output.coefficient) & (output.coefficient <= 1))
    assert output.coefficient[1] == 0


def _two_scale_fusion() -> TwoScaleQualityFusion:
    torch.manual_seed(501)
    return TwoScaleQualityFusion(
        component_dim=4,
        shape_dim=3,
        num_classes=3,
        domain_hidden_dim=5,
    )


def _two_scale_inputs(batch: int = 4):
    torch.manual_seed(502 + batch)
    return (
        torch.randn(batch, 4, requires_grad=True),
        torch.randn(batch, 4, requires_grad=True),
        torch.randn(batch, 3, requires_grad=True),
        torch.tensor(([True, False, True, True] * batch)[:batch]),
        torch.tensor(([True, True, False, True] * batch)[:batch]),
    )


def test_two_scale_fusion_is_exact_three_block_concatenation() -> None:
    fusion = _two_scale_fusion()
    trend, structure, shape, component_valid, shape_valid = _two_scale_inputs()
    output = fusion(trend, structure, shape, component_valid, shape_valid)

    assert fusion.trend_quality is not fusion.structure_quality
    assert fusion.fused_dim == 11
    assert torch.all((output.alpha_trend >= 0) & (output.alpha_trend <= 1))
    assert torch.all((output.alpha_structure >= 0) & (output.alpha_structure <= 1))
    assert output.alpha_trend[1] == output.alpha_structure[1] == 0
    safe_shape = torch.where(shape_valid[:, None], shape, torch.zeros_like(shape))
    expected = torch.cat(
        [
            output.alpha_trend[:, None] * trend,
            output.alpha_structure[:, None] * structure,
            safe_shape,
        ],
        dim=-1,
    )
    torch.testing.assert_close(output.fused_feature, expected)
    torch.testing.assert_close(output.shape_feature, safe_shape)
    assert not hasattr(fusion, "shape_quality")


def test_two_scale_task_fusion_detaches_quality_but_not_task_features() -> None:
    fusion = _two_scale_fusion()
    inputs = _two_scale_inputs()
    output = fusion(*inputs)

    output.fused_feature.sum().backward()

    for feature in inputs[:3]:
        assert feature.grad is not None and torch.isfinite(feature.grad).all()
    for scorer in (fusion.trend_quality, fusion.structure_quality):
        for parameter in scorer.parameters():
            assert parameter.grad is None
    assert not output.alpha_trend.requires_grad
    assert not output.trend.coefficient.requires_grad


def test_two_scale_concatenate_preserves_source_first_order_and_autograd() -> None:
    fusion = _two_scale_fusion()
    source_inputs = _two_scale_inputs(2)
    target_inputs = _two_scale_inputs(3)
    source = fusion(*source_inputs)
    target = fusion(*target_inputs)

    merged = concatenate_two_scale_quality_outputs(source, target)

    for name in (
        "alpha_trend",
        "alpha_structure",
        "weighted_trend",
        "weighted_structure",
        "shape_feature",
        "fused_feature",
    ):
        torch.testing.assert_close(
            getattr(merged, name),
            torch.cat([getattr(source, name), getattr(target, name)], dim=0),
        )
    merged.fused_feature.sum().backward()
    assert source_inputs[0].grad is not None
    assert target_inputs[0].grad is not None


def test_two_scale_objective_uses_source_classification_and_both_domains() -> None:
    quality = _two_scale_fusion()(*_two_scale_inputs())
    objective = TwoScaleQualityObjective(1.0, 1.0)
    labels = torch.tensor([0, 1, 2, 0])
    domains = torch.tensor([1, 1, 0, 0])

    output = objective(quality, labels, domains, domains == 1)

    assert torch.isfinite(output.total_loss)
    assert output.classification_count == 2
    assert output.domain_count == 6

    invalid_quality = _two_scale_fusion()(
        torch.randn(2, 4),
        torch.randn(2, 4),
        torch.randn(2, 3),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    empty = objective(
        invalid_quality,
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
        torch.tensor([True, False]),
    )
    assert empty.total_loss.item() == 0.0
    assert empty.total_loss.requires_grad
