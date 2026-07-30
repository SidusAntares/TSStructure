from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from methods.structure_da import (
    MonotoneWarpEstimator,
    MonotoneWarpOutput,
    SourceRunningSRVFTemplate,
    SourceSRVFTemplateOutput,
    TemporalRegistrationOutput,
    TemporalSRVFRegistration,
)
from methods.structure_da.temporal_registration import (
    _apply_srvf_group_action,
    _warp_sequence,
)


def _make_registration(**kwargs) -> TemporalSRVFRegistration:
    parameters = {
        "num_channels": 1,
        "channel_feature_dim": 2,
        "num_basis": 6,
        "canonical_grid_size": 8,
        "roughness_grid_size": 64,
        "min_mean_support": 0.0,
        "min_dynamic_energy": 0.0,
        "min_template_mean_support": 0.0,
        "warp_hidden_dim": 8,
        "warp_kernel_size": 3,
    }
    parameters.update(kwargs)
    return TemporalSRVFRegistration(**parameters)


def _sample_batch(dtype: torch.dtype = torch.float32):
    torch.manual_seed(61)
    tokens = torch.randn(2, 6, 1, 2, dtype=dtype)
    positions = torch.tensor(
        [0.0, 39.0, 92.0, 157.0, 244.0, 345.0], dtype=dtype
    )
    mask = torch.tensor(
        [[True, True, True, True, True, True],
         [True, False, True, True, False, True]]
    )
    return tokens, positions, mask


def test_template_first_update_is_support_weighted_valid_source_mean() -> None:
    template = SourceRunningSRVFTemplate(3, 2)
    srvf = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]],
        ]
    )
    support = torch.tensor(
        [[1.0, 0.5, 0.25], [0.5, 1.0, 0.75], [1.0, 1.0, 1.0]]
    )
    valid = torch.tensor([True, True, False])

    template.update(srvf, support, valid)

    weights = support[:2]
    expected_srvf = (
        weights.unsqueeze(-1) * srvf[:2]
    ).sum(dim=0) / weights.sum(dim=0).unsqueeze(-1)
    torch.testing.assert_close(template.running_srvf, expected_srvf)
    torch.testing.assert_close(template.running_support, support[:2].mean(dim=0))
    assert template.num_updates.item() == 1


def test_template_low_weight_grid_is_not_initialized_or_updated() -> None:
    template = SourceRunningSRVFTemplate(3, 1, min_grid_weight=0.5)
    srvf = torch.tensor([[[2.0], [4.0], [8.0]]])
    support = torch.tensor([[1.0, 0.1, 0.0]])

    template.update(srvf, support, torch.tensor([True]))

    torch.testing.assert_close(
        template.running_srvf, torch.tensor([[2.0], [0.0], [0.0]])
    )
    torch.testing.assert_close(
        template.running_support, torch.tensor([1.0, 0.0, 0.0])
    )


def test_template_second_update_uses_gridwise_ema() -> None:
    template = SourceRunningSRVFTemplate(3, 1, momentum=0.75)
    valid = torch.tensor([True])
    template.update(
        torch.tensor([[[2.0], [4.0], [6.0]]]),
        torch.tensor([[1.0, 1.0, 1.0]]),
        valid,
    )
    previous_srvf = template.running_srvf.clone()
    previous_support = template.running_support.clone()

    template.update(
        torch.tensor([[[10.0], [12.0], [14.0]]]),
        torch.tensor([[0.5, 0.75, 1.0]]),
        valid,
    )

    torch.testing.assert_close(
        template.running_srvf,
        0.75 * previous_srvf
        + 0.25 * torch.tensor([[10.0], [12.0], [14.0]]),
    )
    torch.testing.assert_close(
        template.running_support,
        0.75 * previous_support + 0.25 * torch.tensor([0.5, 0.75, 1.0]),
    )
    assert template.num_updates.item() == 2


def test_template_no_valid_sample_is_read_only() -> None:
    template = SourceRunningSRVFTemplate(4, 2)
    before = {name: value.clone() for name, value in template.named_buffers()}

    template.update(
        torch.randn(2, 4, 2),
        torch.rand(2, 4),
        torch.tensor([False, False]),
    )

    for name, value in template.named_buffers():
        torch.testing.assert_close(value, before[name])


def test_template_forward_before_update_returns_zeros_and_is_read_only() -> None:
    template = SourceRunningSRVFTemplate(4, 2)
    before = {name: value.clone() for name, value in template.named_buffers()}

    output = template(device=torch.device("cpu"), dtype=torch.float64)

    assert isinstance(output, SourceSRVFTemplateOutput)
    assert output.srvf.shape == (4, 2)
    assert output.support.shape == (4,)
    assert output.srvf.dtype == torch.float64
    assert output.support.dtype == torch.float64
    assert output.initialized.shape == ()
    assert output.initialized.dtype == torch.bool
    assert not output.initialized.item()
    torch.testing.assert_close(output.srvf, torch.zeros_like(output.srvf))
    torch.testing.assert_close(output.support, torch.zeros_like(output.support))
    for name, value in template.named_buffers():
        torch.testing.assert_close(value, before[name])


def test_template_buffers_are_not_parameters_and_initialized_after_update() -> None:
    template = SourceRunningSRVFTemplate(4, 2)
    template.update(
        torch.ones(1, 4, 2), torch.ones(1, 4), torch.tensor([True])
    )

    assert set(dict(template.named_buffers())) == {
        "running_srvf", "running_support", "num_updates"
    }
    assert not dict(template.named_parameters())
    assert template(device=torch.device("cpu"), dtype=torch.float32).initialized.item()


def test_warp_estimator_zero_head_initializes_exact_identity() -> None:
    estimator = MonotoneWarpEstimator(3, 9, hidden_dim=8, kernel_size=3)
    sample = torch.randn(2, 9, 3)
    template = torch.randn(2, 9, 3)
    support = torch.rand(2, 9)

    output = estimator(sample, template, support, support, torch.ones(2, dtype=torch.bool))

    assert isinstance(output, MonotoneWarpOutput)
    identity = torch.linspace(0.0, 1.0, 9).expand(2, -1)
    torch.testing.assert_close(output.interval_logits, torch.zeros(2, 8))
    torch.testing.assert_close(output.interval_widths, torch.full((2, 8), 1 / 8))
    torch.testing.assert_close(output.warp, identity)
    torch.testing.assert_close(output.warp_derivative, torch.ones(2, 9))
    last = estimator.network[-1]
    torch.testing.assert_close(last.weight, torch.zeros_like(last.weight))
    torch.testing.assert_close(last.bias, torch.zeros_like(last.bias))


def test_warp_network_has_only_convolution_and_gelu_layers() -> None:
    estimator = MonotoneWarpEstimator(2, 8, hidden_dim=7, kernel_size=5)

    assert [type(layer) for layer in estimator.network] == [
        nn.Conv1d, nn.GELU, nn.Conv1d, nn.GELU, nn.Conv1d
    ]
    assert not any(
        isinstance(layer, (nn.BatchNorm1d, nn.LayerNorm, nn.Dropout))
        for layer in estimator.modules()
    )


def test_random_warp_parameters_remain_strictly_monotone() -> None:
    torch.manual_seed(62)
    estimator = MonotoneWarpEstimator(2, 11, hidden_dim=6, kernel_size=3)
    with torch.no_grad():
        estimator.network[-1].weight.normal_()
        estimator.network[-1].bias.normal_()
    sample = torch.randn(3, 11, 2)
    template = torch.randn(3, 11, 2)
    support = torch.rand(3, 11)

    output = estimator(sample, template, support, support, torch.ones(3, dtype=torch.bool))

    torch.testing.assert_close(output.warp[:, 0], torch.zeros(3))
    torch.testing.assert_close(output.warp[:, -1], torch.ones(3))
    assert torch.all(output.warp[:, 1:] > output.warp[:, :-1])
    assert torch.all(output.warp_derivative > 0)
    torch.testing.assert_close(output.interval_widths.sum(dim=-1), torch.ones(3))


def test_invalid_registration_rows_force_identity_warp() -> None:
    estimator = MonotoneWarpEstimator(2, 7, hidden_dim=6, kernel_size=3)
    with torch.no_grad():
        estimator.network[-1].weight.normal_()
    sample = torch.randn(2, 7, 2)
    template = torch.randn(2, 7, 2)
    support = torch.rand(2, 7)

    output = estimator(
        sample, template, support, support, torch.tensor([True, False])
    )

    identity = torch.linspace(0.0, 1.0, 7)
    torch.testing.assert_close(output.interval_logits[1], torch.zeros(6))
    torch.testing.assert_close(output.interval_widths[1], torch.full((6,), 1 / 6))
    torch.testing.assert_close(output.warp[1], identity)
    torch.testing.assert_close(output.warp_derivative[1], torch.ones(7))


def test_warp_sequence_identity_and_constant_sequence() -> None:
    sequence = torch.randn(2, 9, 3)
    identity = torch.linspace(0.0, 1.0, 9).expand(2, -1)
    nonlinear = identity.square()
    constant = torch.full((2, 9, 3), 2.75)

    torch.testing.assert_close(_warp_sequence(sequence, identity), sequence)
    torch.testing.assert_close(
        _warp_sequence(constant, nonlinear), constant, atol=1e-6, rtol=0
    )


def test_warp_sequence_is_differentiable_in_sequence_and_warp() -> None:
    sequence = torch.randn(1, 8, 2, requires_grad=True)
    warp = torch.tensor(
        [[0.0, 0.08, 0.20, 0.36, 0.53, 0.71, 0.88, 1.0]],
        requires_grad=True,
    )

    _warp_sequence(sequence, warp).square().sum().backward()

    assert sequence.grad is not None and torch.isfinite(sequence.grad).all()
    assert warp.grad is not None and torch.isfinite(warp.grad).all()


def test_srvf_group_action_identity_and_nonidentity_formula() -> None:
    srvf = torch.randn(1, 6, 2)
    identity = torch.linspace(0.0, 1.0, 6).unsqueeze(0)
    identity_derivative = torch.ones(1, 6)
    torch.testing.assert_close(
        _apply_srvf_group_action(srvf, identity, identity_derivative, 1e-6),
        srvf,
    )

    warp = torch.tensor([[0.0, 0.08, 0.25, 0.48, 0.76, 1.0]])
    derivative = torch.tensor([[0.4, 0.7, 1.0, 1.3, 1.5, 1.2]])
    expected = _warp_sequence(srvf, warp) * torch.sqrt(derivative).unsqueeze(-1)
    actual = _apply_srvf_group_action(srvf, warp, derivative, 1e-6)
    torch.testing.assert_close(actual, expected)


def test_support_is_only_resampled_without_group_action_scale() -> None:
    support = torch.tensor([[0.0, 0.2, 0.5, 0.8, 1.0]])
    warp = torch.tensor([[0.0, 0.1, 0.35, 0.7, 1.0]])
    derivative = torch.tensor([[0.5, 0.8, 1.2, 1.6, 1.0]])

    registered_support = _warp_sequence(support.unsqueeze(-1), warp).squeeze(-1)
    incorrectly_scaled = registered_support * torch.sqrt(derivative)

    assert not torch.allclose(registered_support, incorrectly_scaled)


def test_first_forward_is_unregistered_identity_then_bootstraps_template() -> None:
    registration = _make_registration()
    tokens, positions, mask = _sample_batch()

    output = registration(tokens, positions, mask)

    assert isinstance(output, TemporalRegistrationOutput)
    assert not output.template_initialized.item()
    assert not output.registration_valid.any().item()
    torch.testing.assert_close(output.registered_srvf, output.srvf_output.srvf)
    torch.testing.assert_close(
        output.registered_support, output.srvf_output.support_confidence
    )
    registration.update_source_template(output)
    template = registration.source_template(
        device=torch.device("cpu"), dtype=tokens.dtype
    )
    assert template.initialized.item()

    weights = (
        output.srvf_output.support_confidence
        * output.srvf_output.structure_valid.unsqueeze(-1)
    )
    expected = (
        weights.unsqueeze(-1) * output.srvf_output.srvf
    ).sum(dim=0) / weights.sum(dim=0).clamp_min(1e-6).unsqueeze(-1)
    torch.testing.assert_close(template.srvf, expected)


def test_initialized_template_enables_registration_and_registered_update() -> None:
    registration = _make_registration(template_momentum=0.5)
    tokens, positions, mask = _sample_batch()
    first = registration(tokens, positions, mask)
    registration.update_source_template(first)
    old_srvf = registration.source_template.running_srvf.clone()
    old_support = registration.source_template.running_support.clone()

    second = registration(tokens + 0.25, positions, mask)
    assert second.registration_valid.all().item()
    forced_srvf = torch.full_like(second.registered_srvf, 4.0)
    forced_support = torch.full_like(second.registered_support, 0.7)
    forced = replace(
        second,
        registered_srvf=forced_srvf,
        registered_support=forced_support,
        registration_valid=torch.ones_like(second.registration_valid),
    )
    registration.update_source_template(forced)

    torch.testing.assert_close(
        registration.source_template.running_srvf,
        0.5 * old_srvf + 0.5 * torch.full_like(old_srvf, 4.0),
    )
    torch.testing.assert_close(
        registration.source_template.running_support,
        0.5 * old_support + 0.5 * torch.full_like(old_support, 0.7),
    )


def test_registration_forward_does_not_update_any_source_state() -> None:
    registration = _make_registration()
    tokens, positions, mask = _sample_batch()
    before = {name: value.clone() for name, value in registration.named_buffers()}

    registration(tokens, positions, mask)

    for name, value in registration.named_buffers():
        torch.testing.assert_close(value, before[name])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_registration_shapes_and_dtype(dtype: torch.dtype) -> None:
    registration = _make_registration().to(dtype=dtype)
    tokens, positions, mask = _sample_batch(dtype)

    output = registration(tokens, positions, mask)

    assert output.template_srvf.shape == (2, 8, 2)
    assert output.template_support.shape == (2, 8)
    assert output.template_initialized.shape == ()
    assert output.template_mean_support.shape == ()
    assert output.interval_logits.shape == (2, 7)
    assert output.interval_widths.shape == (2, 7)
    assert output.warp.shape == (2, 8)
    assert output.warp_derivative.shape == (2, 8)
    assert output.registered_srvf.shape == (2, 8, 2)
    assert output.registered_support.shape == (2, 8)
    assert output.registration_valid.shape == (2,)
    for value in (
        output.template_srvf,
        output.template_support,
        output.template_mean_support,
        output.interval_logits,
        output.interval_widths,
        output.warp,
        output.warp_derivative,
        output.registered_srvf,
        output.registered_support,
    ):
        assert value.dtype == dtype
        assert torch.isfinite(value).all()


def test_registered_path_preserves_token_and_warp_head_gradients() -> None:
    registration = _make_registration()
    tokens, positions, mask = _sample_batch()
    bootstrap = registration(tokens, positions, mask)
    registration.update_source_template(bootstrap)
    differentiable = tokens.clone().requires_grad_()

    output = registration(differentiable, positions, mask)
    (output.registered_srvf.square().mean() + output.warp.square().mean()).backward()

    assert output.registration_valid.all().item()
    assert differentiable.grad is not None
    assert torch.isfinite(differentiable.grad).all()
    assert differentiable.grad[mask].abs().sum().item() > 0
    torch.testing.assert_close(
        differentiable.grad[~mask],
        torch.zeros_like(differentiable.grad[~mask]),
        atol=0,
        rtol=0,
    )
    last = registration.warp_estimator.network[-1]
    assert last.weight.grad is not None and torch.isfinite(last.weight.grad).all()
    assert last.bias.grad is not None and torch.isfinite(last.bias.grad).all()
    assert not dict(registration.source_template.named_parameters())
    for value in registration.source_template.buffers():
        assert value.grad is None


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (SourceRunningSRVFTemplate, {"canonical_grid_size": 1, "feature_dim": 2}),
        (SourceRunningSRVFTemplate, {"canonical_grid_size": 4, "feature_dim": 0}),
        (SourceRunningSRVFTemplate, {"canonical_grid_size": 4, "feature_dim": 2, "momentum": -0.1}),
        (SourceRunningSRVFTemplate, {"canonical_grid_size": 4, "feature_dim": 2, "momentum": 1.0}),
        (MonotoneWarpEstimator, {"feature_dim": 0, "canonical_grid_size": 4}),
        (MonotoneWarpEstimator, {"feature_dim": 2, "canonical_grid_size": 1}),
        (MonotoneWarpEstimator, {"feature_dim": 2, "canonical_grid_size": 4, "hidden_dim": 0}),
        (MonotoneWarpEstimator, {"feature_dim": 2, "canonical_grid_size": 4, "kernel_size": 0}),
        (MonotoneWarpEstimator, {"feature_dim": 2, "canonical_grid_size": 4, "kernel_size": 4}),
        (MonotoneWarpEstimator, {"feature_dim": 2, "canonical_grid_size": 4, "min_increment": 0.0}),
    ],
)
def test_invalid_constructor_arguments_raise_value_error(factory, kwargs) -> None:
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize(
    "srvf,support,valid,match",
    [
        (torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, dtype=torch.bool), "srvf"),
        (torch.ones(2, 4, 2), torch.ones(2, 3), torch.ones(2, dtype=torch.bool), "support"),
        (torch.ones(2, 4, 2), torch.tensor([[1.1] * 4] * 2), torch.ones(2, dtype=torch.bool), r"\[0, 1\]"),
        (torch.full((2, 4, 2), float("nan")), torch.ones(2, 4), torch.ones(2, dtype=torch.bool), "finite"),
        (torch.ones(2, 4, 2), torch.full((2, 4), float("inf")), torch.ones(2, dtype=torch.bool), "finite"),
        (torch.ones(2, 4, 2), torch.ones(2, 4), torch.ones(2), "boolean"),
    ],
)
def test_template_rejects_invalid_update_inputs(srvf, support, valid, match) -> None:
    with pytest.raises(ValueError, match=match):
        SourceRunningSRVFTemplate(4, 2).update(srvf, support, valid)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("sample_srvf", torch.ones(2, 4), "sample_srvf"),
        ("template_srvf", torch.ones(2, 3, 2), "template_srvf"),
        ("sample_support", torch.ones(2, 3), "sample_support"),
        ("template_support", torch.full((2, 4), 1.1), r"\[0, 1\]"),
        ("sample_srvf", torch.full((2, 4, 2), float("inf")), "finite"),
        ("registration_valid", torch.ones(2), "boolean"),
    ],
)
def test_warp_estimator_rejects_invalid_inputs(field, value, match) -> None:
    inputs = {
        "sample_srvf": torch.ones(2, 4, 2),
        "template_srvf": torch.ones(2, 4, 2),
        "sample_support": torch.ones(2, 4),
        "template_support": torch.ones(2, 4),
        "registration_valid": torch.ones(2, dtype=torch.bool),
    }
    inputs[field] = value
    with pytest.raises(ValueError, match=match):
        MonotoneWarpEstimator(2, 4)(**inputs)


@pytest.mark.parametrize(
    "warp,match",
    [
        (torch.tensor([[0.0, 0.5, 1.1]]), r"\[0, 1\]"),
        (torch.tensor([[0.0, 0.7, 0.6]]), "strictly increasing"),
        (torch.tensor([[0.0, float("nan"), 1.0]]), "finite"),
    ],
)
def test_warp_sequence_rejects_invalid_warp(warp, match) -> None:
    with pytest.raises(ValueError, match=match):
        _warp_sequence(torch.ones(1, 3, 2), warp)


def test_group_action_rejects_nonpositive_warp_derivative() -> None:
    with pytest.raises(ValueError, match="positive"):
        _apply_srvf_group_action(
            torch.ones(1, 3, 2),
            torch.tensor([[0.0, 0.5, 1.0]]),
            torch.tensor([[1.0, 0.0, 1.0]]),
            1e-6,
        )


def test_template_update_rejects_wrong_registration_output_type() -> None:
    with pytest.raises(ValueError, match="TemporalRegistrationOutput"):
        _make_registration().update_source_template(object())
