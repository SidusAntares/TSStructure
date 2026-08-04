from __future__ import annotations

import pytest
import torch

from methods.structure_da.decomposition import DecompositionOutput
from methods.structure_da.representation import (
    PhaseAwareTwoScaleClassifier,
    PhaseAwareTwoScaleClassifierOutput,
    PairedStructureFeatures,
    QualityAwareClassifierOutput,
    QualityAwareComponentClassifier,
)


def _phase_aware_classifier(**overrides):
    options = dict(
        component_input_dim=4,
        shape_dim=5,
        num_classes=3,
        n_head=1,
        d_k=2,
        d_model=8,
        ltae_mlp=(8, 4),
        dropout=0.0,
        time_reference=0.0,
        time_scale=10.0,
        max_initial_frequency=4.0,
        classifier_hidden=(6,),
        quality_domain_hidden_dim=5,
    )
    options.update(overrides)
    return PhaseAwareTwoScaleClassifier(**options)


def _phase_aware_inputs():
    torch.manual_seed(503)
    trend = torch.randn(3, 5, 4)
    structure = torch.randn(3, 5, 4)
    shape = torch.randn(3, 5)
    positions = torch.tensor(
        [[0.0, 1.25, 3.5, 7.75, 10.0]] * 3,
    )
    mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, False], [False] * 5]
    )
    shape_valid = torch.tensor([True, False, True])
    return trend, structure, shape, positions, mask, shape_valid


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


def test_phase_aware_classifier_uses_exact_two_scale_fusion() -> None:
    classifier = _phase_aware_classifier().eval()
    trend, structure, shape, positions, mask, shape_valid = _phase_aware_inputs()

    output = classifier(
        trend,
        structure,
        shape,
        positions,
        time_mask=mask,
        shape_valid=shape_valid,
    )

    assert isinstance(output, PhaseAwareTwoScaleClassifierOutput)
    assert output.logits.shape == (3, 3)
    assert output.trend_embedding.shape == (3, 4)
    assert output.structure_embedding.shape == (3, 4)
    assert output.shape_feature.shape == (3, 5)
    assert classifier.fused_dim == 2 * classifier.component_dim + classifier.shape_dim
    torch.testing.assert_close(
        output.fused_feature,
        torch.cat(
            [
                output.quality.alpha_trend[:, None] * output.trend_embedding,
                output.quality.alpha_structure[:, None] * output.structure_embedding,
                output.shape_feature,
            ],
            dim=-1,
        ),
    )
    assert output.shape_feature[1].count_nonzero().item() == 0
    assert set(output.__dataclass_fields__) == {
        "logits",
        "fused_feature",
        "trend_embedding",
        "structure_embedding",
        "shape_feature",
        "quality",
        "component_valid",
        "shape_valid",
        "aligned_positions",
        "time_mask",
    }


def test_phase_aware_classifier_passes_one_continuous_position_tensor_to_shared_ltae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifier = _phase_aware_classifier().eval()
    trend, structure, shape, positions, mask, shape_valid = _phase_aware_inputs()
    seen: list[torch.Tensor] = []
    original_forward = classifier.component_ltae.forward

    def capture_positions(*args, **kwargs):
        seen.append(args[2])
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(classifier.component_ltae, "forward", capture_positions)
    output = classifier(
        trend,
        structure,
        shape,
        positions,
        time_mask=mask,
        shape_valid=shape_valid,
    )

    assert len(seen) == 1
    assert seen[0] is positions
    assert output.aligned_positions is positions
    assert not torch.equal(positions, positions.round())


def test_phase_aware_final_ce_updates_raw_and_shape_but_not_quality_scorers() -> None:
    classifier = _phase_aware_classifier()
    trend, structure, shape, positions, mask, shape_valid = _phase_aware_inputs()
    trend = trend.requires_grad_()
    structure = structure.requires_grad_()
    shape = shape.requires_grad_()

    output = classifier(
        trend,
        structure,
        shape,
        positions,
        time_mask=mask,
        shape_valid=shape_valid,
    )
    torch.nn.functional.cross_entropy(
        output.logits, torch.tensor([0, 1, 2])
    ).backward()

    assert trend.grad is not None and trend.grad.abs().sum().item() > 0
    assert structure.grad is not None and structure.grad.abs().sum().item() > 0
    assert shape.grad is not None and shape.grad[shape_valid].abs().sum().item() > 0
    assert shape.grad[~shape_valid].count_nonzero().item() == 0
    ltae_gradients = [
        parameter.grad for parameter in classifier.component_ltae.parameters()
    ]
    assert any(gradient is not None and gradient.abs().sum().item() > 0 for gradient in ltae_gradients)
    assert all(
        parameter.grad is None
        for parameter in classifier.quality_fusion.parameters()
    )


def test_phase_aware_invalid_shape_keeps_raw_classification_path() -> None:
    classifier = _phase_aware_classifier().eval()
    trend, structure, shape, positions, mask, _ = _phase_aware_inputs()
    shape_valid = torch.zeros(3, dtype=torch.bool)

    output = classifier(
        trend,
        structure,
        shape,
        positions,
        time_mask=mask,
        shape_valid=shape_valid,
    )

    assert output.shape_feature.count_nonzero().item() == 0
    assert output.fused_feature[:, : 2 * classifier.component_dim].abs().sum().item() > 0
    assert torch.isfinite(output.logits).all()


def test_phase_aware_classifier_supports_fp32_master_autocast() -> None:
    classifier = _phase_aware_classifier()
    trend, structure, shape, positions, mask, shape_valid = _phase_aware_inputs()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = classifier(
            trend,
            structure,
            shape,
            positions,
            time_mask=mask,
            shape_valid=shape_valid,
        )

    assert output.trend_embedding.dtype == torch.bfloat16
    assert output.structure_embedding.dtype == torch.bfloat16
    assert output.quality.fused_feature.dtype == torch.float32
    assert torch.isfinite(output.logits).all()
