from __future__ import annotations

import pytest
import torch

from methods.structure_da.decomposition import DecompositionOutput
from methods.structure_da.representation import (
    PairedStructureFeatures,
    QualityAwareClassifierOutput,
    QualityAwareComponentClassifier,
)


def _classifier(**overrides):
    options = dict(
        component_input_dim=4,
        structure_dim=3,
        num_classes=3,
        n_head=1,
        d_k=2,
        d_model=8,
        ltae_mlp=(8, 4),
        dropout=0.0,
        positional_period=100,
        max_position=365,
        max_temporal_shift=10,
        classifier_hidden=(6,),
        quality_domain_hidden_dim=5,
    )
    options.update(overrides)
    return QualityAwareComponentClassifier(**options)


def _inputs():
    torch.manual_seed(502)
    shape = (3, 5, 4)
    decomposition = DecompositionOutput(
        trend=torch.randn(shape),
        dynamics=torch.randn(shape),
        residual=torch.randn(shape),
    )
    temporal = PairedStructureFeatures(
        trend=torch.randn(3, 3),
        dynamics=torch.randn(3, 3),
        trend_valid=torch.tensor([True, True, False]),
        dynamics_valid=torch.tensor([True, False, False]),
    )
    positions = torch.tensor(
        [[0, 30, 80, 170, 300], [1, 31, 81, 171, 301], [2, 32, 82, 172, 302]]
    )
    mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, False], [False] * 5]
    )
    return decomposition, temporal, positions, mask


def test_representation_accepts_three_dimensional_components() -> None:
    classifier = _classifier()
    output = classifier(*_inputs())
    assert isinstance(output, QualityAwareClassifierOutput)
    assert output.logits.shape == (3, 3)
    assert output.trend_embedding.shape == (3, 4)
    assert output.fused_feature.shape == (3, 7)
    assert classifier.fused_dim == classifier.component_dim + classifier.structure_dim
    assert set(output.__dataclass_fields__) == {
        "logits", "fused_feature", "trend_embedding", "dynamics_embedding",
        "residual_embedding", "temporal_features", "quality", "component_valid",
        "ltae_positions", "time_mask",
    }


def test_representation_does_not_flatten_a_four_dimensional_input() -> None:
    decomposition, temporal, positions, mask = _inputs()
    invalid = DecompositionOutput(
        trend=decomposition.trend.unsqueeze(2),
        dynamics=decomposition.dynamics.unsqueeze(2),
        residual=decomposition.residual.unsqueeze(2),
    )
    with pytest.raises(ValueError, match=r"\[B, L, D\]"):
        _classifier()(invalid, temporal, positions, mask)


def test_masked_values_do_not_affect_representation() -> None:
    classifier = _classifier().eval()
    decomposition, temporal, positions, mask = _inputs()
    changed = DecompositionOutput(
        trend=decomposition.trend.masked_fill(~mask.unsqueeze(-1), 1e6),
        dynamics=decomposition.dynamics.masked_fill(~mask.unsqueeze(-1), -1e6),
        residual=decomposition.residual.masked_fill(~mask.unsqueeze(-1), 3e6),
    )
    expected = classifier(decomposition, temporal, positions, mask)
    actual = classifier(changed, temporal, positions, mask)
    torch.testing.assert_close(actual.logits, expected.logits)


def test_representation_backpropagates_through_all_components() -> None:
    classifier = _classifier()
    decomposition, temporal, positions, mask = _inputs()
    components = [value.detach().requires_grad_() for value in (
        decomposition.trend, decomposition.dynamics, decomposition.residual
    )]
    output = classifier(DecompositionOutput(*components), temporal, positions, mask)
    output.logits.sum().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in components)
