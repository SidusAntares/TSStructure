"""Regression tests for the component-specific shared-LTAE backbone."""

import pytest
import torch

from methods.structure_da import ComponentStructureClassifier
from models.ltae import LTAE


def _make_model(**overrides):
    parameters = {
        "feature_dim": 4,
        "num_classes": 3,
        "time_scale": 365.0,
        "n_head": 2,
        "d_k": 2,
        "d_model": 8,
        "ltae_mlp": (8, 6),
        "dropout": 0.0,
        "positional_period": 1000,
        "max_position": 64,
        "max_temporal_shift": 4,
        "classifier_hidden": (5,),
    }
    parameters.update(overrides)
    return ComponentStructureClassifier(**parameters)


def _sample(batch_size=2):
    torch.manual_seed(7)
    features = torch.randn(batch_size, 4, 4)
    positions = torch.tensor([0, 2, 7, 12])
    return features, positions


def test_forward_returns_all_component_outputs_with_expected_shapes():
    model = _make_model().eval()
    features, positions = _sample()

    output = model(features, positions)

    assert output.logits.shape == (2, 3)
    assert output.fused_embedding.shape == (2, 18)
    assert output.trend_embedding.shape == (2, 6)
    assert output.dynamics_embedding.shape == (2, 6)
    assert output.residual_embedding.shape == (2, 6)
    assert output.decomposition.trend.shape == (2, 4, 4)
    assert output.trend_temporal.local.shape == (2, 4, 8)
    assert output.dynamics_temporal.local.shape == (2, 4, 8)
    assert output.dynamics_channel.local.shape == (2, 4, 4)
    assert output.ltae_inputs.trend.shape == (2, 4, 16)
    assert output.ltae_inputs.dynamics.shape == (2, 4, 16)
    assert output.ltae_inputs.residual.shape == (2, 4, 16)


def test_component_slots_follow_the_exact_structure_assignment():
    model = _make_model().eval()
    features, positions = _sample()

    output = model(features, positions)
    width = model.feature_dim

    torch.testing.assert_close(
        output.ltae_inputs.trend[..., :width],
        model.content_norm(output.decomposition.trend),
    )
    torch.testing.assert_close(
        output.ltae_inputs.trend[..., width : 3 * width],
        model.temporal_norm(output.trend_temporal.local),
    )
    assert torch.count_nonzero(output.ltae_inputs.trend[..., 3 * width :]) == 0

    torch.testing.assert_close(
        output.ltae_inputs.dynamics[..., :width],
        model.content_norm(output.decomposition.dynamics),
    )
    torch.testing.assert_close(
        output.ltae_inputs.dynamics[..., width : 3 * width],
        model.temporal_norm(output.dynamics_temporal.local),
    )
    torch.testing.assert_close(
        output.ltae_inputs.dynamics[..., 3 * width :],
        model.channel_norm(output.dynamics_channel.local),
    )

    torch.testing.assert_close(
        output.ltae_inputs.residual[..., :width],
        model.content_norm(output.decomposition.residual),
    )
    assert torch.count_nonzero(output.ltae_inputs.residual[..., width:]) == 0


def test_operators_are_called_only_for_the_assigned_components_and_share_tau():
    model = _make_model().eval()
    features, positions = _sample()
    temporal_calls = []
    channel_calls = []
    temporal_hook = model.temporal_operator.register_forward_pre_hook(
        lambda _module, arguments: temporal_calls.append(arguments)
    )
    channel_hook = model.channel_operator.register_forward_pre_hook(
        lambda _module, arguments: channel_calls.append(arguments)
    )

    try:
        output = model(features, positions)
    finally:
        temporal_hook.remove()
        channel_hook.remove()

    assert len(temporal_calls) == 2
    assert len(channel_calls) == 1
    torch.testing.assert_close(temporal_calls[0][0], output.decomposition.trend)
    torch.testing.assert_close(temporal_calls[1][0], output.decomposition.dynamics)
    torch.testing.assert_close(channel_calls[0][0], output.decomposition.dynamics)
    assert temporal_calls[0][2] is temporal_calls[1][2]
    assert temporal_calls[0][3] is temporal_calls[1][3]
    assert temporal_calls[0][2].requires_grad
    assert temporal_calls[0][3].requires_grad


def test_exactly_one_ltae_instance_is_reused_three_times():
    model = _make_model().eval()
    features, positions = _sample()
    calls = []
    hook = model.shared_ltae.register_forward_pre_hook(
        lambda _module, arguments: calls.append(arguments)
    )

    try:
        output = model(features, positions)
    finally:
        hook.remove()

    assert sum(isinstance(module, LTAE) for module in model.modules()) == 1
    assert len(calls) == 3
    torch.testing.assert_close(calls[0][0], output.ltae_inputs.trend)
    torch.testing.assert_close(calls[1][0], output.ltae_inputs.dynamics)
    torch.testing.assert_close(calls[2][0], output.ltae_inputs.residual)
    assert not hasattr(model, "trend_ltae")
    assert not hasattr(model, "dynamics_ltae")
    assert not hasattr(model, "residual_ltae")


def test_classifier_gradient_reaches_input_tau_shared_ltae_and_classifier():
    model = _make_model()
    features, positions = _sample()
    features.requires_grad_()

    model(features, positions).logits.square().mean().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    for parameter in model.decomposition.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for module in (model.shared_ltae, model.classifier):
        for parameter in module.parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
    assert list(model.temporal_operator.parameters()) == []
    assert list(model.channel_operator.parameters()) == []


def test_fused_embedding_is_the_exact_ordered_concatenation():
    model = _make_model().eval()
    features, positions = _sample()

    output = model(features, positions)

    torch.testing.assert_close(
        output.fused_embedding,
        torch.cat(
            [
                output.trend_embedding,
                output.dynamics_embedding,
                output.residual_embedding,
            ],
            dim=-1,
        ),
    )


def test_normalizers_are_parameter_free_and_assembled_inputs_are_finite():
    model = _make_model().eval()
    features, positions = _sample()

    output = model(features, positions)

    assert list(model.content_norm.parameters()) == []
    assert list(model.temporal_norm.parameters()) == []
    assert list(model.channel_norm.parameters()) == []
    for value in (
        output.ltae_inputs.trend,
        output.ltae_inputs.dynamics,
        output.ltae_inputs.residual,
    ):
        assert torch.isfinite(value).all()
    assert torch.count_nonzero(output.ltae_inputs.trend[..., 12:]) == 0
    assert torch.count_nonzero(output.ltae_inputs.residual[..., 4:]) == 0


def test_integer_valued_float_positions_match_long_positions():
    model = _make_model().eval()
    features, positions = _sample()

    with torch.no_grad():
        integer_output = model(features, positions)
        float_output = model(features, positions.to(torch.float32))

    torch.testing.assert_close(integer_output.logits, float_output.logits)
    torch.testing.assert_close(
        integer_output.fused_embedding, float_output.fused_embedding
    )


@pytest.mark.parametrize(
    "positions, message",
    [
        (torch.tensor([0.0, 1.5, 2.0, 3.0]), "integer"),
        (torch.tensor([-5, 0, 1, 2]), "positional embedding"),
        (torch.tensor([0, 1, 2, 68]), "positional embedding"),
    ],
)
def test_invalid_ltae_positions_are_rejected_before_attention(positions, message):
    model = _make_model().eval()
    features, _ = _sample()

    with pytest.raises(ValueError, match=message):
        model(features, positions)


def test_none_and_all_true_masks_match_but_partial_masks_are_explicitly_unsupported():
    model = _make_model().eval()
    features, positions = _sample()
    all_true = torch.ones(2, 4, dtype=torch.bool)

    with torch.no_grad():
        without_mask = model(features, positions)
        with_mask = model(features, positions, all_true)

    torch.testing.assert_close(without_mask.logits, with_mask.logits)
    partial = all_true.clone()
    partial[0, -1] = False
    with pytest.raises(NotImplementedError, match="mask"):
        model(features, positions, partial)


def test_eval_forward_is_deterministic_even_with_configured_dropout():
    model = _make_model(dropout=0.3).eval()
    features, positions = _sample()

    with torch.no_grad():
        first = model(features, positions)
        second = model(features, positions)

    torch.testing.assert_close(first.logits, second.logits)
    torch.testing.assert_close(first.fused_embedding, second.fused_embedding)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"feature_dim": 0}, "feature_dim"),
        ({"num_classes": 1}, "num_classes"),
        ({"n_head": 0}, "n_head"),
        ({"d_model": 7}, "divisible"),
        ({"ltae_mlp": ()}, "ltae_mlp"),
        ({"ltae_mlp": (7, 6)}, "ltae_mlp"),
        ({"classifier_hidden": (5, 0)}, "classifier_hidden"),
    ],
)
def test_constructor_rejects_invalid_backbone_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_model(**overrides)


@pytest.mark.parametrize(
    "features, positions, message",
    [
        (torch.randn(2, 4), torch.arange(4), "shape"),
        (torch.randn(2, 4, 3), torch.arange(4), "feature_dim"),
        (torch.randn(2, 4, 4), torch.arange(3), "positions"),
    ],
)
def test_forward_validates_feature_and_position_shapes(features, positions, message):
    with pytest.raises(ValueError, match=message):
        _make_model()(features, positions)
