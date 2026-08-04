from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

import methods.structure_da as structure_da_api
from methods.structure_da import (
    MonotoneWarpEstimator,
    MonotoneWarpOutput,
    SourceRunningSRVFTemplate,
    SourceSRVFTemplateOutput,
    TemporalRegistrationOutput,
    TemporalSRVFRegistration,
)
from methods.structure_da import temporal_registration as registration_module
from methods.structure_da.temporal_registration import (
    _apply_srvf_group_action,
    _warp_sequence,
)


def _make_registration(**kwargs) -> TemporalSRVFRegistration:
    parameters = {
        "feature_dim": 2,
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
    tokens = torch.randn(2, 6, 2, dtype=dtype)
    positions = torch.tensor(
        [0.0, 39.0, 92.0, 157.0, 244.0, 345.0], dtype=dtype
    )
    mask = torch.tensor(
        [[True, True, True, True, True, True],
         [True, False, True, True, False, True]]
    )
    return tokens, positions, mask


def _warp_inputs(
    batch_size: int = 2,
    grid_size: int = 9,
    feature_dim: int = 3,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
):
    torch.manual_seed(67)
    sample = torch.randn(
        batch_size, grid_size, feature_dim, dtype=dtype, device=device
    )
    template = torch.randn_like(sample)
    support = torch.rand(batch_size, grid_size, dtype=dtype, device=device)
    valid = torch.ones(batch_size, dtype=torch.bool, device=device)
    return sample, template, support, valid


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


def test_multi_candidate_initialization_is_symmetric_and_noncollapsed() -> None:
    estimator = MonotoneWarpEstimator(
        3,
        9,
        hidden_dim=8,
        kernel_size=3,
        num_candidates=3,
        candidate_init_warp_amplitude=0.015,
    )
    sample, template, support, valid = _warp_inputs()

    output = estimator.forward_candidates(
        sample, template, support, support, valid
    )

    assert isinstance(
        output, registration_module.MonotoneWarpCandidatesOutput
    )
    assert output.interval_logits.shape == (2, 3, 8)
    assert output.interval_widths.shape == (2, 3, 8)
    assert output.warp.shape == (2, 3, 9)
    assert output.warp_derivative.shape == (2, 3, 9)
    assert output.inverse_warp.shape == (2, 3, 9)
    identity = torch.linspace(0.0, 1.0, 9)
    assert isinstance(estimator.candidate_base_logits, nn.Parameter)
    assert estimator.candidate_base_logits.shape == (3, 8)
    torch.testing.assert_close(output.warp[:, 0], identity.expand(2, -1))
    assert not torch.allclose(output.warp[:, 1], output.warp[:, 0])
    assert not torch.allclose(output.warp[:, 2], output.warp[:, 0])
    assert not torch.allclose(output.warp[:, 1], output.warp[:, 2])
    deviations = (output.warp[0, 1:] - identity).abs().amax(dim=-1)
    torch.testing.assert_close(deviations, torch.full_like(deviations, 0.015), atol=2e-4, rtol=0)
    torch.testing.assert_close(
        output.warp[0, 1] - identity,
        -(output.warp[0, 2] - identity),
        atol=5e-4,
        rtol=0,
    )
    assert torch.all(output.warp[..., 1:] > output.warp[..., :-1])
    assert torch.all(output.warp_derivative > 0)
    interval_speed = output.interval_widths[0] * 8
    assert interval_speed.min() > 0.9
    assert interval_speed.max() < 1.1
    roughness = torch.diff(torch.log(interval_speed), dim=-1).square().mean(-1)
    assert roughness.max() < 1e-3


def test_single_candidate_initializes_as_identity() -> None:
    estimator = MonotoneWarpEstimator(
        3, 9, hidden_dim=8, kernel_size=3, num_candidates=1
    )
    sample, template, support, valid = _warp_inputs()
    output = estimator.forward_candidates(sample, template, support, support, valid)
    identity = torch.linspace(0.0, 1.0, 9).expand(2, 1, -1)
    torch.testing.assert_close(output.warp, identity)
    torch.testing.assert_close(estimator.candidate_base_logits, torch.zeros(1, 8))


def test_scalar_candidate_bias_does_not_create_warp_diversity() -> None:
    estimator = MonotoneWarpEstimator(
        3,
        9,
        hidden_dim=8,
        kernel_size=3,
        num_candidates=3,
        candidate_init_warp_amplitude=0.0,
    )
    with torch.no_grad():
        estimator.network[-1].bias.copy_(torch.tensor([-2.0, 0.0, 2.0]))
    sample, template, support, valid = _warp_inputs()
    output = estimator.forward_candidates(sample, template, support, support, valid)
    identity = torch.linspace(0.0, 1.0, 9).expand(2, 3, -1)
    torch.testing.assert_close(output.warp, identity)


def test_extra_candidates_use_distinct_paired_low_frequency_profiles() -> None:
    estimator = MonotoneWarpEstimator(
        3, 9, hidden_dim=8, kernel_size=3, num_candidates=5
    )
    profiles = estimator.candidate_base_logits.detach()
    torch.testing.assert_close(profiles[0], torch.zeros_like(profiles[0]))
    assert torch.allclose(profiles[1], -profiles[2])
    assert torch.allclose(profiles[3], -profiles[4])
    assert not torch.allclose(profiles[1], profiles[3])


def test_multi_candidate_warps_are_strictly_monotone() -> None:
    estimator = MonotoneWarpEstimator(
        3, 9, hidden_dim=8, kernel_size=3, num_candidates=3
    )
    with torch.no_grad():
        for candidate in range(3):
            estimator.network[-1].weight[candidate].normal_(
                mean=0.2 * candidate, std=0.3 + 0.1 * candidate
            )
            estimator.network[-1].bias[candidate] = 0.1 * candidate
    sample, template, support, valid = _warp_inputs()

    output = estimator.forward_candidates(
        sample, template, support, support, valid
    )

    assert torch.all(output.warp[..., 1:] > output.warp[..., :-1])
    torch.testing.assert_close(output.warp[..., 0], torch.zeros(2, 3))
    torch.testing.assert_close(output.warp[..., -1], torch.ones(2, 3))
    assert torch.all(output.interval_widths > 0)
    torch.testing.assert_close(
        output.interval_widths.sum(dim=-1), torch.ones(2, 3)
    )
    assert torch.all(output.warp_derivative > 0)


def test_invalid_rows_make_all_candidates_identity() -> None:
    estimator = MonotoneWarpEstimator(
        3, 9, hidden_dim=8, kernel_size=3, num_candidates=3
    )
    with torch.no_grad():
        estimator.network[-1].weight.normal_()
        estimator.network[-1].bias.normal_()
    sample, template, support, _ = _warp_inputs()

    output = estimator.forward_candidates(
        sample, template, support, support, torch.tensor([True, False])
    )

    identity = torch.linspace(0.0, 1.0, 9).expand(3, -1)
    torch.testing.assert_close(output.interval_logits[1], torch.zeros(3, 8))
    torch.testing.assert_close(
        output.interval_widths[1], torch.full((3, 8), 1.0 / 8)
    )
    torch.testing.assert_close(output.warp[1], identity)
    torch.testing.assert_close(output.warp_derivative[1], torch.ones(3, 9))
    torch.testing.assert_close(output.inverse_warp[1], identity)


def test_legacy_forward_returns_candidate_zero() -> None:
    estimator = MonotoneWarpEstimator(
        3, 9, hidden_dim=8, kernel_size=3, num_candidates=3
    )
    with torch.no_grad():
        estimator.network[-1].weight.normal_()
    sample, template, support, valid = _warp_inputs()

    legacy = estimator(sample, template, support, support, valid)
    multi = estimator.forward_candidates(
        sample, template, support, support, valid
    )

    torch.testing.assert_close(legacy.interval_logits, multi.interval_logits[:, 0])
    torch.testing.assert_close(legacy.interval_widths, multi.interval_widths[:, 0])
    torch.testing.assert_close(legacy.warp, multi.warp[:, 0])
    torch.testing.assert_close(legacy.warp_derivative, multi.warp_derivative[:, 0])
    registration = _make_registration(warp_num_candidates=4)
    assert registration.warp_estimator.num_candidates == 4


def test_select_warp_candidate_uses_per_sample_indices() -> None:
    batch_size, candidates_count, grid_size = 4, 3, 6
    logits = torch.arange(
        batch_size * candidates_count * (grid_size - 1), dtype=torch.float32
    ).reshape(batch_size, candidates_count, grid_size - 1)
    widths = logits + 100.0
    warp = torch.arange(
        batch_size * candidates_count * grid_size, dtype=torch.float32
    ).reshape(batch_size, candidates_count, grid_size)
    derivative = warp + 200.0
    candidates = registration_module.MonotoneWarpCandidatesOutput(
        interval_logits=logits,
        interval_widths=widths,
        warp=warp,
        warp_derivative=derivative,
        inverse_warp=warp + 300.0,
    )
    indices = torch.tensor([2, 0, 1, 2], dtype=torch.long)

    selected = registration_module.select_warp_candidate(candidates, indices)

    rows = torch.arange(batch_size)
    torch.testing.assert_close(selected.interval_logits, logits[rows, indices])
    torch.testing.assert_close(selected.interval_widths, widths[rows, indices])
    torch.testing.assert_close(selected.warp, warp[rows, indices])
    torch.testing.assert_close(selected.warp_derivative, derivative[rows, indices])


@pytest.mark.parametrize("shape", [(2, 7), (2, 3, 7)])
def test_invert_identity_warp(shape) -> None:
    identity = torch.linspace(0.0, 1.0, shape[-1])
    warp = identity.expand(*shape[:-1], -1).clone()

    inverse = registration_module.invert_monotone_warp(warp)

    torch.testing.assert_close(inverse, warp)


def test_invert_nonlinear_warp_composition() -> None:
    warp = torch.tensor([[0.0, 0.08, 0.31, 0.67, 1.0]], dtype=torch.float64)
    query = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)

    inverse = registration_module.invert_monotone_warp(warp, query)
    reference_grid = torch.linspace(0.0, 1.0, warp.shape[-1], dtype=warp.dtype)
    upper = torch.searchsorted(reference_grid, inverse, right=True).clamp(1, 4)
    lower = upper - 1
    lower_t = reference_grid[lower]
    upper_t = reference_grid[upper]
    fraction = (inverse - lower_t) / (upper_t - lower_t)
    composed = torch.gather(warp, 1, lower)
    composed += fraction * (
        torch.gather(warp, 1, upper)
        - composed
    )

    torch.testing.assert_close(composed.squeeze(0), query, atol=1e-10, rtol=1e-10)


def test_invert_monotone_warp_supports_irregular_queries() -> None:
    warp = torch.tensor(
        [[[0.0, 0.05, 0.25, 0.62, 1.0],
          [0.0, 0.18, 0.40, 0.81, 1.0]]]
    )
    query = torch.tensor([0.0, 0.03, 0.27, 0.58, 0.91, 1.0])

    inverse = registration_module.invert_monotone_warp(warp, query)

    assert inverse.shape == (1, 2, 6)
    assert torch.all((inverse >= 0) & (inverse <= 1))
    assert torch.all(inverse[..., 1:] >= inverse[..., :-1])
    torch.testing.assert_close(inverse[..., 0], torch.zeros(1, 2))
    torch.testing.assert_close(inverse[..., -1], torch.ones(1, 2))


def test_invert_monotone_warp_supports_per_warp_queries() -> None:
    warp = torch.tensor(
        [[0.0, 0.15, 0.50, 1.0], [0.0, 0.30, 0.72, 1.0]]
    )
    query = torch.tensor([[0.0, 0.20, 1.0], [0.0, 0.80, 1.0]])

    inverse = registration_module.invert_monotone_warp(warp, query)

    assert inverse.shape == query.shape
    torch.testing.assert_close(inverse[:, 0], torch.zeros(2))
    torch.testing.assert_close(inverse[:, -1], torch.ones(2))


def test_invert_monotone_warp_rejects_complex_query() -> None:
    warp = torch.tensor([[0.0, 0.4, 1.0]])
    query = torch.tensor([0.0 + 0.0j, 1.0 + 0.0j])

    with pytest.raises(ValueError, match="real"):
        registration_module.invert_monotone_warp(warp, query)


def test_candidate_output_preserves_dtype_and_device() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        estimator = MonotoneWarpEstimator(
            3, 9, hidden_dim=8, kernel_size=3, num_candidates=3
        ).to(device=device, dtype=torch.float32)
        sample, template, support, valid = _warp_inputs(device=device)

        output = estimator.forward_candidates(
            sample, template, support, support, valid
        )

        for value in (
            output.interval_logits, output.interval_widths, output.warp,
            output.warp_derivative, output.inverse_warp,
        ):
            assert value.dtype == torch.float32
            assert value.device == device


def test_candidate_gradients_reach_all_candidate_heads() -> None:
    estimator = MonotoneWarpEstimator(
        3, 9, hidden_dim=8, kernel_size=3, num_candidates=3
    )
    sample, template, support, valid = _warp_inputs()

    output = estimator.forward_candidates(
        sample, template, support, support, valid
    )
    weights = torch.arange(1, output.warp.numel() + 1).reshape_as(output.warp)
    (output.warp * weights).sum().backward()

    last = estimator.network[-1]
    assert last.weight.grad is not None
    assert torch.isfinite(last.weight.grad).all()
    assert torch.all(last.weight.grad.flatten(1).abs().sum(dim=1) > 0)
    assert last.bias.grad is not None and torch.isfinite(last.bias.grad).all()


def test_multi_candidate_public_api_is_exported() -> None:
    for name in (
        "MonotoneWarpCandidatesOutput",
        "invert_monotone_warp",
        "select_warp_candidate",
    ):
        assert name in structure_da_api.__all__
        assert hasattr(structure_da_api, name)


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
        (MonotoneWarpEstimator, {"feature_dim": 2, "canonical_grid_size": 4, "num_candidates": 0}),
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
    "indices,match",
    [
        (torch.tensor([0.0, 1.0]), "torch.long"),
        (torch.tensor([0, 3], dtype=torch.long), "candidate range"),
    ],
)
def test_select_warp_candidate_rejects_invalid_indices(indices, match) -> None:
    values = torch.zeros(2, 3, 4)
    candidates = registration_module.MonotoneWarpCandidatesOutput(
        interval_logits=values[..., :-1],
        interval_widths=values[..., :-1],
        warp=values,
        warp_derivative=values,
        inverse_warp=values,
    )
    with pytest.raises(ValueError, match=match):
        registration_module.select_warp_candidate(candidates, indices)


@pytest.mark.parametrize(
    "warp,query,match",
    [
        (torch.tensor([[0.0, 0.7, 0.6, 1.0]]), None, "strictly increasing"),
        (torch.tensor([[0.0, 0.3, 0.7, 1.0]]), torch.tensor([-0.1, 0.5]), r"\[0, 1\]"),
    ],
)
def test_invert_monotone_warp_rejects_invalid_inputs(warp, query, match) -> None:
    with pytest.raises(ValueError, match=match):
        registration_module.invert_monotone_warp(warp, query)


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
