"""Tests for explicit structural-space adversarial adaptation."""

import copy
import inspect

import pytest
import torch
from torch import nn

import methods.structure_da.adaptation as adaptation_module
from methods.structure_da import (
    JointStructuralSpaceBuilder,
    SDADiscriminator,
    StructuralAdversarialAdapter,
    gradient_reverse,
    vectorize_channel_statistic,
)


def _inputs(batch_size=3, feature_dim=4, requires_grad=False):
    torch.manual_seed(7)
    temporal_dim = 4 * feature_dim
    values = (
        torch.randn(batch_size, temporal_dim, requires_grad=requires_grad),
        torch.randn(batch_size, temporal_dim, requires_grad=requires_grad),
        torch.randn(
            batch_size, feature_dim, feature_dim, requires_grad=requires_grad
        ),
    )
    gates = tuple(
        torch.linspace(0.2 + offset, 0.6 + offset, batch_size).requires_grad_(
            requires_grad
        )
        for offset in (0.0, 0.1, 0.2)
    )
    return values + gates


def test_builder_dimensions_and_distinct_affine_free_normalizers():
    builder = JointStructuralSpaceBuilder(feature_dim=4)

    assert builder.temporal_dim == 16
    assert builder.channel_dim == 6
    assert builder.joint_dim == 38
    normalizers = (
        builder.trend_temporal_norm,
        builder.dynamics_temporal_norm,
        builder.dynamics_channel_norm,
    )
    assert all(isinstance(normalizer, nn.LayerNorm) for normalizer in normalizers)
    assert len({id(normalizer) for normalizer in normalizers}) == 3
    assert all(normalizer.elementwise_affine is False for normalizer in normalizers)
    assert list(builder.parameters()) == []


def test_builder_matches_normalize_then_detached_gate_formula_exactly():
    builder = JointStructuralSpaceBuilder(feature_dim=4)
    trend, dynamics_t, dynamics_c, beta_t, beta_dt, beta_dc = _inputs()

    output = builder(trend, dynamics_t, dynamics_c, beta_t, beta_dt, beta_dc)
    expected_t = beta_t.detach()[:, None] * builder.trend_temporal_norm(trend)
    expected_dt = beta_dt.detach()[:, None] * builder.dynamics_temporal_norm(
        dynamics_t
    )
    expected_dc = beta_dc.detach()[:, None] * builder.dynamics_channel_norm(
        vectorize_channel_statistic(dynamics_c)
    )

    torch.testing.assert_close(output.trend_temporal, expected_t)
    torch.testing.assert_close(output.dynamics_temporal, expected_dt)
    torch.testing.assert_close(output.dynamics_channel, expected_dc)
    torch.testing.assert_close(
        output.joint, torch.cat((expected_t, expected_dt, expected_dc), dim=-1)
    )
    assert output.joint.shape == (3, 38)


def test_builder_uses_only_strict_upper_triangle_of_channel_statistic():
    builder = JointStructuralSpaceBuilder(feature_dim=4)
    inputs = list(_inputs())
    changed = inputs[2].clone()
    lower_and_diagonal = torch.tril_indices(4, 4, offset=0)
    changed[:, lower_and_diagonal[0], lower_and_diagonal[1]] += 1000

    original = builder(*inputs)
    inputs[2] = changed
    modified = builder(*inputs)

    torch.testing.assert_close(original.dynamics_channel, modified.dynamics_channel)
    torch.testing.assert_close(original.joint, modified.joint)


def test_builder_normalizes_before_applying_each_independent_gate():
    builder = JointStructuralSpaceBuilder(feature_dim=4)
    trend, dynamics_t, dynamics_c, *_ = _inputs()
    gates = (torch.full((3,), 0.25), torch.full((3,), 0.5), torch.ones(3))

    output = builder(trend, dynamics_t, dynamics_c, *gates)

    torch.testing.assert_close(
        output.trend_temporal,
        0.25 * builder.trend_temporal_norm(trend),
    )
    assert not torch.allclose(
        output.trend_temporal,
        builder.trend_temporal_norm(0.25 * trend),
    )
    torch.testing.assert_close(
        output.dynamics_temporal,
        0.5 * builder.dynamics_temporal_norm(dynamics_t),
    )


def test_builder_stops_gate_gradients_but_preserves_all_statistic_gradients():
    builder = JointStructuralSpaceBuilder(feature_dim=4)
    inputs = _inputs(requires_grad=True)

    builder(*inputs).joint.square().mean().backward()

    assert all(value.grad is not None for value in inputs[:3])
    assert all(torch.isfinite(value.grad).all() for value in inputs[:3])
    assert all(gate.grad is None for gate in inputs[3:])


def test_builder_api_has_no_residual_component():
    parameters = inspect.signature(JointStructuralSpaceBuilder.forward).parameters

    assert "residual" not in parameters
    assert tuple(parameters) == (
        "self",
        "trend_temporal_statistic",
        "dynamics_temporal_statistic",
        "dynamics_channel_statistic",
        "beta_trend_temporal",
        "beta_dynamics_temporal",
        "beta_dynamics_channel",
    )


@pytest.mark.parametrize("feature_dim", [0, 1, -1, True])
def test_builder_rejects_invalid_feature_dimension(feature_dim):
    with pytest.raises(ValueError, match="feature_dim"):
        JointStructuralSpaceBuilder(feature_dim)


@pytest.mark.parametrize(
    "index,replacement",
    [
        (0, torch.randn(3, 15)),
        (1, torch.randn(3, 2, 8)),
        (2, torch.randn(3, 4, 3)),
        (0, torch.ones(3, 16, dtype=torch.long)),
        (1, torch.full((3, 16), float("nan"))),
        (2, torch.randn(2, 4, 4)),
    ],
)
def test_builder_rejects_malformed_structural_statistics(index, replacement):
    inputs = list(_inputs())
    inputs[index] = replacement

    with pytest.raises(ValueError):
        JointStructuralSpaceBuilder(4)(*inputs)


@pytest.mark.parametrize(
    "replacement",
    [
        torch.ones(3, 1),
        torch.ones(3, dtype=torch.long),
        torch.tensor([0.2, float("nan"), 0.4]),
        torch.tensor([-0.1, 0.2, 0.3]),
        torch.tensor([0.1, 0.2, 1.1]),
        torch.ones(2),
        [0.2, 0.3, 0.4],
    ],
)
def test_builder_rejects_malformed_quality_gates(replacement):
    inputs = list(_inputs())
    inputs[3] = replacement

    with pytest.raises(ValueError):
        JointStructuralSpaceBuilder(4)(*inputs)


@pytest.mark.parametrize("coefficient", [0.0, 0.5, 1.0])
def test_gradient_reversal_is_identity_forward_and_exact_backward(coefficient):
    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    upstream = torch.tensor([2.0, 4.0, -1.0])

    reversed_x = gradient_reverse(x, coefficient)
    torch.testing.assert_close(reversed_x, x)
    (reversed_x * upstream).sum().backward()

    torch.testing.assert_close(x.grad, -coefficient * upstream)


@pytest.mark.parametrize("coefficient", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_gradient_reversal_rejects_invalid_coefficient(coefficient):
    with pytest.raises(ValueError, match="coefficient"):
        gradient_reverse(torch.ones(2), coefficient)


def test_discriminator_has_exact_two_linear_shape_and_finite_logits():
    discriminator = SDADiscriminator(input_dim=7, hidden_dim=5)
    logits = discriminator(torch.randn(4, 7))

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    assert [type(module) for module in discriminator.modules()] == [
        SDADiscriminator,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
    ]


@pytest.mark.parametrize("input_dim,hidden_dim", [(0, 2), (2, 0), (True, 2)])
def test_discriminator_rejects_invalid_dimensions(input_dim, hidden_dim):
    with pytest.raises(ValueError):
        SDADiscriminator(input_dim, hidden_dim)


def test_adapter_uses_one_shared_discriminator_twice_for_unequal_batches():
    adapter = StructuralAdversarialAdapter(joint_dim=7, hidden_dim=5)
    calls = []
    handle = adapter.discriminator.register_forward_hook(
        lambda _module, args, _output: calls.append(args[0].shape)
    )
    source = torch.randn(3, 7)
    target = torch.randn(5, 7)

    output = adapter(source, target, grl_coefficient=0.4)
    handle.remove()

    assert output.source_logits.shape == (3,)
    assert output.target_logits.shape == (5,)
    assert calls == [torch.Size([3, 7]), torch.Size([5, 7])]
    assert len([m for m in adapter.modules() if isinstance(m, SDADiscriminator)]) == 1
    torch.testing.assert_close(output.source_joint, source)
    torch.testing.assert_close(output.target_joint, target)


def test_grl_reverses_input_gradients_without_reversing_discriminator_gradients():
    plain = SDADiscriminator(6, 4)
    reversed_discriminator = copy.deepcopy(plain)
    plain_input = torch.randn(5, 6, requires_grad=True)
    reversed_input = plain_input.detach().clone().requires_grad_(True)

    plain(plain_input).sum().backward()
    reversed_discriminator(gradient_reverse(reversed_input, 1.0)).sum().backward()

    torch.testing.assert_close(reversed_input.grad, -plain_input.grad)
    for plain_parameter, reversed_parameter in zip(
        plain.parameters(), reversed_discriminator.parameters()
    ):
        torch.testing.assert_close(reversed_parameter.grad, plain_parameter.grad)


def test_zero_coefficient_blocks_backbone_gradients_but_trains_discriminator():
    adapter = StructuralAdversarialAdapter(joint_dim=7, hidden_dim=5)
    source = torch.randn(3, 7, requires_grad=True)
    target = torch.randn(4, 7, requires_grad=True)

    output = adapter(source, target, grl_coefficient=0.0)
    (output.source_logits.sum() + output.target_logits.sum()).backward()

    torch.testing.assert_close(source.grad, torch.zeros_like(source))
    torch.testing.assert_close(target.grad, torch.zeros_like(target))
    assert all(parameter.grad is not None for parameter in adapter.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in adapter.parameters())


def test_full_cpu_float32_path_is_finite_and_preserves_builder_gradients():
    builder = JointStructuralSpaceBuilder(4)
    adapter = StructuralAdversarialAdapter(builder.joint_dim, hidden_dim=8)
    source_inputs = _inputs(batch_size=2, requires_grad=True)
    target_inputs = _inputs(batch_size=3, requires_grad=True)

    source_joint = builder(*source_inputs).joint
    target_joint = builder(*target_inputs).joint
    output = adapter(source_joint, target_joint, grl_coefficient=0.7)
    loss = output.source_logits.square().mean() + output.target_logits.square().mean()
    loss.backward()

    assert output.source_logits.dtype == torch.float32
    assert torch.isfinite(output.source_logits).all()
    assert torch.isfinite(output.target_logits).all()
    assert all(value.grad is not None for value in source_inputs[:3])
    assert all(value.grad is not None for value in target_inputs[:3])
    assert all(gate.grad is None for gate in source_inputs[3:] + target_inputs[3:])


def test_full_builder_upstream_gradients_are_reversed_by_grl():
    plain_builder = JointStructuralSpaceBuilder(4)
    reversed_builder = JointStructuralSpaceBuilder(4)
    plain_discriminator = SDADiscriminator(plain_builder.joint_dim, hidden_dim=8)
    adapter = StructuralAdversarialAdapter(
        reversed_builder.joint_dim, hidden_dim=8
    )
    adapter.discriminator.load_state_dict(plain_discriminator.state_dict())
    plain_source = _inputs(batch_size=2, requires_grad=True)
    plain_target = _inputs(batch_size=3, requires_grad=True)
    reversed_source = tuple(
        value.detach().clone().requires_grad_(True) for value in plain_source
    )
    reversed_target = tuple(
        value.detach().clone().requires_grad_(True) for value in plain_target
    )

    plain_loss = plain_discriminator(plain_builder(*plain_source).joint).sum()
    plain_loss = plain_loss + plain_discriminator(
        plain_builder(*plain_target).joint
    ).sum()
    plain_loss.backward()
    reversed_output = adapter(
        reversed_builder(*reversed_source).joint,
        reversed_builder(*reversed_target).joint,
        grl_coefficient=1.0,
    )
    (reversed_output.source_logits.sum() + reversed_output.target_logits.sum()).backward()

    for plain_value, reversed_value in zip(
        plain_source[:3] + plain_target[:3],
        reversed_source[:3] + reversed_target[:3],
    ):
        torch.testing.assert_close(reversed_value.grad, -plain_value.grad)
    assert all(
        gate.grad is None
        for gate in reversed_source[3:] + reversed_target[3:]
    )


def test_adaptation_module_does_not_import_or_reuse_quality_modules_or_losses():
    source = inspect.getsource(adaptation_module)

    assert "from .quality" not in source
    assert "ComponentQuality" not in source
    assert "StructuralQuality" not in source
    assert "binary_cross_entropy" not in source
    assert "BCEWithLogitsLoss" not in source
