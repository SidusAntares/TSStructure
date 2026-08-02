from __future__ import annotations

import torch

from methods.structure_da.quality_fusion import (
    HierarchicalQualityFusion,
    HierarchicalQualityObjective,
    QualityScorer,
    concatenate_hierarchical_quality_outputs,
)


def _fusion() -> HierarchicalQualityFusion:
    torch.manual_seed(402)
    return HierarchicalQualityFusion(4, 3, 3, domain_hidden_dim=5)


def _inputs(batch: int = 4):
    torch.manual_seed(403 + batch)
    embeddings = [torch.randn(batch, 4, requires_grad=True) for _ in range(3)]
    structures = [torch.randn(batch, 3, requires_grad=True) for _ in range(2)]
    component_valid = torch.ones(batch, dtype=torch.bool)
    structural_valid = [torch.ones(batch, dtype=torch.bool) for _ in range(2)]
    return (*embeddings, *structures, component_valid, *structural_valid)


def test_quality_scorer_coefficients_are_bounded_and_invalid_is_zero() -> None:
    scorer = QualityScorer(4, 3, domain_hidden_dim=5)
    output = scorer(torch.randn(3, 4), torch.tensor([True, False, True]))
    assert torch.all((0 <= output.coefficient) & (output.coefficient <= 1))
    assert output.coefficient[1] == 0


def test_quality_fusion_has_two_semantic_blocks_and_expected_dimensions() -> None:
    output = _fusion()(*_inputs())
    assert output.trend_component_input.shape == (4, 7)
    assert output.residual_component_input.shape == (4, 7)
    assert output.raw_fusion.shape == (4, 4)
    assert output.temporal_fusion.shape == (4, 3)
    assert output.fused_feature.shape == (4, 7)
    torch.testing.assert_close(
        output.fused_feature,
        torch.cat([output.raw_fusion, output.temporal_fusion], dim=-1),
    )


def test_residual_branch_is_preserved_with_zero_temporal_structure() -> None:
    output = _fusion()(*_inputs())
    torch.testing.assert_close(output.residual_component_input[:, 4:], torch.zeros(4, 3))
    assert output.component.residual.class_logits.shape == (4, 3)


def test_concatenate_preserves_source_target_order_and_autograd() -> None:
    fusion = _fusion()
    source_inputs = _inputs(2)
    target_inputs = _inputs(3)
    source = fusion(*source_inputs)
    target = fusion(*target_inputs)
    merged = concatenate_hierarchical_quality_outputs(source, target)
    for name in ("alpha_trend", "alpha_dynamics", "alpha_residual", "beta_trend_temporal", "beta_dynamics_temporal", "fused_feature"):
        torch.testing.assert_close(
            getattr(merged, name), torch.cat([getattr(source, name), getattr(target, name)])
        )
    merged.fused_feature.sum().backward()
    assert source_inputs[0].grad is not None
    assert target_inputs[0].grad is not None


def test_quality_objective_keeps_structural_and_component_losses() -> None:
    quality = _fusion()(*_inputs())
    objective = HierarchicalQualityObjective(1.0, 1.0, 1.0, 1.0)
    labels = torch.tensor([0, 1, 2, 0])
    domains = torch.tensor([1, 1, 0, 0])
    output = objective(quality, labels, domains, domains == 1)
    assert torch.isfinite(output.total_loss)
    assert output.structural_classification_count == 4
    assert output.structural_domain_count == 8
    assert output.component_classification_count == 6
    assert output.component_domain_count == 12
