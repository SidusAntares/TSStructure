from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

import methods.structure_da as structure_da
from methods.structure_da.backbone import StructureBackbone
from methods.structure_da.channel_module import (
    ChannelStructureOutput,
    ChannelStructurePairOutput,
    MultiScaleChannelRelationStructure,
    SharedChannelStructureOperator,
    SourceRunningAttributeStandardizer,
    SourceRunningRelationEnergyScale,
)


def _extractor(
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 101,
    **overrides,
) -> MultiScaleChannelRelationStructure:
    kwargs = dict(
        num_channels=3,
        token_dim=2,
        lag_centers=(-0.2, 0.0, 0.2),
        lag_widths=(0.1, 0.1, 0.1),
        velocity_bandwidth=0.25,
        edge_hidden_dim=5,
        structure_dim=7,
        min_velocity_effective_count=2.0,
        min_velocity_time_spread=1e-4,
        min_effective_pairs=1.0,
        min_relation_mass=0.0,
        time_scale=1.0,
        dropout=0.0,
    )
    kwargs.update(overrides)
    torch.manual_seed(seed)
    return MultiScaleChannelRelationStructure(**kwargs).to(dtype=dtype)


def _inputs(
    *,
    batch_size: int = 2,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.tensor(
        [0.0, 0.16, 0.37, 0.68, 1.0], dtype=dtype
    )
    u = positions.view(1, 5, 1, 1)
    channel_scale = torch.tensor([1.0, -0.8, 0.45], dtype=dtype).view(
        1, 1, 3, 1
    )
    attributes = torch.tensor([1.0, 0.6], dtype=dtype).view(1, 1, 1, 2)
    tokens = channel_scale * attributes * (u + 0.15 * u.square())
    tokens = tokens.expand(batch_size, -1, -1, -1).clone()
    if batch_size > 1:
        tokens[1] += 0.07 * torch.sin(5.0 * u[0])
    mask = torch.ones(batch_size, 5, 3, dtype=torch.bool)
    return tokens, positions, mask


def _full_output(
    extractor: MultiScaleChannelRelationStructure | None = None,
) -> ChannelStructureOutput:
    extractor = extractor or _extractor()
    tokens, positions, mask = _inputs()
    return extractor(tokens, positions, channel_mask=mask)


def test_channel_module_public_symbols_are_exported() -> None:
    expected = {
        "ChannelStructureOutput",
        "ChannelStructurePairOutput",
        "MultiScaleChannelRelationStructure",
        "SharedChannelStructureOperator",
        "SourceRunningAttributeStandardizer",
        "SourceRunningRelationEnergyScale",
    }

    assert expected <= set(structure_da.__all__)
    assert all(hasattr(structure_da, name) for name in expected)


def test_coverage_weights_are_zero_when_masked_and_sum_per_channel() -> None:
    extractor = _extractor()
    _, positions, mask = _inputs(batch_size=1)
    mask[0, 1, 0] = False
    mask[0, :, 2] = False

    weights = extractor.compute_channel_coverage_weights(
        positions.unsqueeze(0), mask
    )

    assert weights.shape == (1, 5, 3)
    assert torch.count_nonzero(weights[~mask]) == 0
    torch.testing.assert_close(weights.sum(dim=1)[0, :2], torch.ones(2))
    assert weights[:, :, 2].sum().item() == 0


def test_single_observation_coverage_weight_is_one() -> None:
    extractor = _extractor()
    _, positions, mask = _inputs(batch_size=1)
    mask[:, :, 1] = False
    mask[:, 3, 1] = True

    weights = extractor.compute_channel_coverage_weights(
        positions.unsqueeze(0), mask
    )

    assert weights[0, 3, 1].item() == pytest.approx(1.0)
    assert torch.count_nonzero(weights[0, :, 1]).item() == 1


def test_irregular_coverage_matches_voronoi_formula() -> None:
    extractor = _extractor()
    positions = torch.tensor([[0.0, 0.2, 0.7, 1.0]])
    mask = torch.ones(1, 4, 3, dtype=torch.bool)

    weights = extractor.compute_channel_coverage_weights(positions, mask)

    raw = torch.tensor([0.1, 0.35, 0.4, 0.15])
    expected = raw / raw.sum()
    torch.testing.assert_close(weights[0, :, 0], expected)


def test_broadcast_time_mask_matches_equivalent_channel_mask() -> None:
    extractor = _extractor().eval()
    tokens, positions, _ = _inputs()
    time_mask = torch.tensor([True, False, True, True, True])
    channel_mask = time_mask[:, None].expand(-1, 3)

    by_time = extractor(tokens, positions, time_mask=time_mask)
    by_channel = extractor(tokens, positions, channel_mask=channel_mask)

    torch.testing.assert_close(by_time.feature, by_channel.feature)
    torch.testing.assert_close(by_time.state_relation, by_channel.state_relation)
    torch.testing.assert_close(by_time.local_velocity, by_channel.local_velocity)


def test_time_and_channel_masks_are_combined_with_logical_and() -> None:
    extractor = _extractor().eval()
    tokens, positions, channel_mask = _inputs()
    time_mask = torch.tensor([True, False, True, True, True])
    channel_mask[0, 2, 1] = False
    combined = channel_mask & time_mask.view(1, 5, 1)

    explicit = extractor(tokens, positions, channel_mask=combined)
    composed = extractor(
        tokens,
        positions,
        time_mask=time_mask,
        channel_mask=channel_mask,
    )

    torch.testing.assert_close(explicit.feature, composed.feature)
    torch.testing.assert_close(explicit.state_relation, composed.state_relation)


def test_local_linear_velocity_recovers_exact_slope() -> None:
    extractor = _extractor()
    positions = torch.tensor([[0.0, 0.2, 0.45, 0.72, 1.0]])
    slopes = torch.tensor([[[2.0, -1.5], [0.5, 3.0], [-2.0, 0.25]]])
    intercept = torch.tensor([[[0.3, 1.0], [-0.7, 0.2], [1.5, -2.0]]])
    tokens = intercept.unsqueeze(1) + positions[:, :, None, None] * slopes.unsqueeze(1)
    mask = torch.ones(1, 5, 3, dtype=torch.bool)

    velocity, support, _, _ = extractor.compute_local_velocity(
        tokens, positions, mask
    )

    torch.testing.assert_close(
        velocity, slopes.unsqueeze(1).expand_as(velocity), atol=2e-5, rtol=2e-5
    )
    assert (support > 0).all()


def test_constant_trajectory_has_zero_velocity_but_positive_support() -> None:
    extractor = _extractor()
    positions = torch.tensor([[0.0, 0.2, 0.45, 0.72, 1.0]])
    tokens = torch.ones(1, 5, 3, 2)
    mask = torch.ones(1, 5, 3, dtype=torch.bool)

    velocity, support, effective_count, spread = (
        extractor.compute_local_velocity(tokens, positions, mask)
    )
    expected = (effective_count / 2.0).clamp(0, 1) * (
        spread / (spread + 1e-4)
    )

    torch.testing.assert_close(velocity, torch.zeros_like(velocity), atol=1e-6, rtol=0)
    torch.testing.assert_close(support, expected)
    assert (support > 0).all()


def test_velocity_support_is_zero_for_invalid_center() -> None:
    extractor = _extractor()
    tokens, positions, mask = _inputs(batch_size=1)
    mask[0, 2, 0] = False

    velocity, support, _, _ = extractor.compute_local_velocity(
        tokens, positions.unsqueeze(0), mask
    )

    assert support[0, 2, 0].item() == 0
    assert torch.count_nonzero(velocity[0, 2, 0]).item() == 0


def test_velocity_support_is_zero_without_time_spread() -> None:
    extractor = _extractor()
    tokens = torch.ones(1, 1, 3, 2)
    positions = torch.zeros(1, 1)
    mask = torch.ones(1, 1, 3, dtype=torch.bool)

    velocity, support, _, spread = extractor.compute_local_velocity(
        tokens, positions, mask
    )

    assert torch.count_nonzero(velocity) == 0
    assert torch.count_nonzero(support) == 0
    assert torch.count_nonzero(spread) == 0


def test_identical_channel_trajectories_have_positive_zero_lag_relations() -> None:
    extractor = _extractor().eval()
    tokens, positions, mask = _inputs(batch_size=1)
    tokens[:, :, 1] = tokens[:, :, 0]

    output = extractor(tokens, positions, channel_mask=mask)

    assert output.state_relation[0, 0, 1, 1].item() > 0.9
    assert output.evolution_relation[0, 0, 1, 1].item() > 0.9


def test_opposite_channel_trajectories_have_negative_relations() -> None:
    extractor = _extractor().eval()
    tokens, positions, mask = _inputs(batch_size=1)
    tokens[:, :, 1] = -tokens[:, :, 0]

    output = extractor(tokens, positions, channel_mask=mask)

    assert output.state_relation[0, 0, 1, 1].item() < -0.9
    assert output.evolution_relation[0, 0, 1, 1].item() < -0.9


def test_directed_lag_reversal_is_symmetric() -> None:
    extractor = _extractor().eval()
    tokens, positions, mask = _inputs(batch_size=1)

    output = extractor(tokens, positions, channel_mask=mask)

    torch.testing.assert_close(
        output.state_relation[:, 0, 1, 2],
        output.state_relation[:, 1, 0, 0],
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        output.evolution_relation[:, 0, 1, 2],
        output.evolution_relation[:, 1, 0, 0],
        atol=2e-6,
        rtol=2e-6,
    )


def test_relative_strength_changes_sign_when_edge_direction_reverses() -> None:
    extractor = _extractor().eval()
    tokens, positions, mask = _inputs(batch_size=1)

    output = extractor(tokens, positions, channel_mask=mask)

    torch.testing.assert_close(
        output.relative_strength[:, 0, 1, 2],
        -output.relative_strength[:, 1, 0, 0],
        atol=2e-6,
        rtol=2e-6,
    )


def test_relation_and_reliability_bounds() -> None:
    output = _full_output()

    for relation in (output.state_relation, output.evolution_relation):
        assert relation.min().item() >= -1.0
        assert relation.max().item() <= 1.0
    for reliability in (output.state_reliability, output.evolution_reliability):
        assert reliability.min().item() >= 0.0
        assert reliability.max().item() <= 1.0


def test_all_relation_diagonals_are_zero() -> None:
    output = _full_output()
    diagonal = torch.arange(3)

    for tensor in (
        output.state_relation,
        output.evolution_relation,
        output.relative_strength,
        output.state_reliability,
        output.evolution_reliability,
        output.state_effective_pair_count,
        output.evolution_effective_pair_count,
    ):
        assert torch.count_nonzero(tensor[:, diagonal, diagonal]) == 0


def test_masked_extreme_and_nonfinite_values_do_not_change_outputs() -> None:
    extractor = _extractor().eval()
    tokens, positions, mask = _inputs()
    mask[:, 1, 0] = False
    changed = tokens.clone()
    changed[0, 1, 0] = torch.tensor([float("nan"), float("inf")])
    changed[1, 1, 0] = torch.tensor([1e20, -1e20])

    expected = extractor(tokens, positions, channel_mask=mask)
    actual = extractor(changed, positions, channel_mask=mask)

    for field in ChannelStructureOutput.__dataclass_fields__:
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field))


def test_masked_tokens_have_exactly_zero_gradient() -> None:
    extractor = _extractor()
    tokens, positions, mask = _inputs()
    mask[:, 1, 0] = False
    tokens.requires_grad_()

    output = extractor(tokens, positions, channel_mask=mask)
    output.feature.square().sum().backward()

    assert tokens.grad is not None
    assert torch.count_nonzero(tokens.grad[:, 1, 0]) == 0
    assert torch.isfinite(tokens.grad[mask]).all()


def test_asynchronous_channel_missingness_is_supported() -> None:
    extractor = _extractor().eval()
    tokens, positions, mask = _inputs()
    mask[0, [1, 3], 0] = False
    mask[0, [0, 4], 1] = False
    mask[1, [2], 2] = False

    output = extractor(tokens, positions, channel_mask=mask)

    assert torch.isfinite(output.feature).all()
    assert torch.isfinite(output.state_relation).all()
    assert torch.isfinite(output.local_velocity).all()


def test_no_reliable_pairs_yields_invalid_and_exact_zero_feature() -> None:
    extractor = _extractor(min_relation_mass=0.1).eval()
    tokens, positions, mask = _inputs()
    mask.zero_()

    output = extractor(tokens, positions, channel_mask=mask)

    assert not output.valid.any()
    assert torch.count_nonzero(output.feature) == 0


def test_sparse_support_attenuates_reliability() -> None:
    extractor = _extractor(min_effective_pairs=8.0).eval()
    tokens, positions, full_mask = _inputs(batch_size=1)
    sparse_mask = full_mask.clone()
    sparse_mask[:, 1:-1] = False

    full = extractor(tokens, positions, channel_mask=full_mask)
    sparse = extractor(tokens, positions, channel_mask=sparse_mask)

    assert sparse.state_reliability.sum() < full.state_reliability.sum()
    assert sparse.evolution_reliability.sum() < full.evolution_reliability.sum()


def test_edge_raw_and_embedding_have_exact_directed_shapes() -> None:
    output = _full_output()

    assert output.reliable_edge_raw.shape == (2, 6, 9)
    assert output.edge_embedding.shape == (2, 6, 5)


def test_directed_edge_buffers_use_fixed_source_major_order() -> None:
    extractor = _extractor()

    assert extractor.source_channel_indices.tolist() == [0, 0, 1, 1, 2, 2]
    assert extractor.target_channel_indices.tolist() == [1, 2, 0, 2, 0, 1]
    assert not extractor.source_channel_indices.requires_grad
    assert not extractor.target_channel_indices.requires_grad


def test_edge_encoder_is_exact_bias_free_two_linear_gelu() -> None:
    extractor = _extractor()

    assert list(type(module) for module in extractor.edge_encoder) == [
        nn.Linear,
        nn.GELU,
        nn.Linear,
    ]
    assert extractor.edge_encoder[0].bias is None
    assert extractor.edge_encoder[2].bias is None
    assert extractor.edge_encoder[0].in_features == 9
    assert extractor.edge_encoder[2].out_features == 5


def test_zero_edge_raw_maps_to_zero_edge_embedding() -> None:
    extractor = _extractor()
    raw = torch.zeros(2, 6, 9)

    encoded = extractor.edge_encoder(raw)

    torch.testing.assert_close(encoded, torch.zeros_like(encoded), atol=0, rtol=0)


def test_output_head_is_exact_projection_gelu_dropout_layer_norm() -> None:
    extractor = _extractor()

    assert extractor.output_projection.bias is None
    assert list(type(module) for module in extractor.output_head) == [
        nn.Linear,
        nn.GELU,
        nn.Dropout,
        nn.LayerNorm,
    ]
    assert extractor.output_head[0] is extractor.output_projection


def test_invalid_mask_is_applied_after_layer_norm() -> None:
    extractor = _extractor(min_relation_mass=1e9).eval()
    tokens, positions, mask = _inputs()
    captured: list[torch.Tensor] = []
    hook = extractor.output_head[-1].register_forward_hook(
        lambda _module, _args, value: captured.append(value)
    )

    output = extractor(tokens, positions, channel_mask=mask)
    hook.remove()

    assert captured and torch.count_nonzero(captured[0]) > 0
    assert torch.count_nonzero(output.feature) == 0


def test_extractor_contains_no_forbidden_channel_modules() -> None:
    extractor = _extractor()
    names = " ".join(name.lower() for name, _ in extractor.named_modules())

    assert "attention" not in names
    assert "gate" not in names
    assert "embedding" not in names
    assert not any(isinstance(module, nn.Sigmoid) for module in extractor.modules())


def test_output_dataclass_shapes_are_complete() -> None:
    output = _full_output()

    assert output.feature.shape == (2, 7)
    assert output.valid.shape == (2,)
    assert output.valid.dtype == torch.bool
    assert output.state_relation.shape == (2, 3, 3, 3)
    assert output.evolution_relation.shape == (2, 3, 3, 3)
    assert output.relative_strength.shape == (2, 3, 3, 3)
    assert output.local_velocity.shape == (2, 5, 3, 2)
    assert output.velocity_support.shape == (2, 5, 3)
    assert output.relation_mass.shape == (2,)


def test_attribute_standardizer_first_update_and_ema() -> None:
    standardizer = SourceRunningAttributeStandardizer(2, momentum=0.5)
    first = torch.tensor([[[[1.0, 3.0], [3.0, 5.0]]]])
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    standardizer.update(first, mask)

    torch.testing.assert_close(standardizer.running_mean, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(
        standardizer.running_second_moment, torch.tensor([5.0, 17.0])
    )
    standardizer.update(torch.full_like(first, 6.0), mask)
    torch.testing.assert_close(standardizer.running_mean, torch.tensor([4.0, 5.0]))
    assert standardizer.num_updates.item() == 2


def test_attribute_statistics_are_shared_over_physical_channels() -> None:
    standardizer = SourceRunningAttributeStandardizer(1)
    tokens = torch.tensor([[[[1.0], [5.0], [9.0]]]])
    mask = torch.ones(1, 1, 3, dtype=torch.bool)

    standardizer.update(tokens, mask)

    torch.testing.assert_close(standardizer.running_mean, torch.tensor([5.0]))
    assert standardizer.running_mean.shape == (1,)


def test_uninitialized_standardizer_is_identity_and_forward_is_read_only() -> None:
    standardizer = SourceRunningAttributeStandardizer(2)
    tokens = torch.randn(2, 3, 4, 2)
    before = {name: value.clone() for name, value in standardizer.named_buffers()}

    output = standardizer(tokens)

    torch.testing.assert_close(output, tokens)
    for name, value in standardizer.named_buffers():
        torch.testing.assert_close(value, before[name])


def test_energy_scale_first_update_ema_and_independent_types() -> None:
    scale = SourceRunningRelationEnergyScale(momentum=0.5)
    valid = torch.tensor([True, True, False])
    scale.update(
        torch.tensor([2.0, 4.0, 100.0]),
        torch.tensor([6.0, 10.0, 100.0]),
        valid,
        valid,
    )
    assert scale.running_state_scale.item() == pytest.approx(3.0)
    assert scale.running_evolution_scale.item() == pytest.approx(8.0)
    scale.update(
        torch.tensor([5.0]),
        torch.tensor([20.0]),
        torch.tensor([True]),
        torch.tensor([False]),
    )
    assert scale.running_state_scale.item() == pytest.approx(4.0)
    assert scale.running_evolution_scale.item() == pytest.approx(8.0)
    assert scale.num_updates.item() == 2


def test_energy_scale_ignores_invalid_nonpositive_or_nonfinite_values() -> None:
    scale = SourceRunningRelationEnergyScale()
    before = {name: value.clone() for name, value in scale.named_buffers()}

    scale.update(
        torch.tensor([0.0, -1.0, float("nan")]),
        torch.tensor([float("inf"), 0.0, -2.0]),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
    )

    for name, value in scale.named_buffers():
        torch.testing.assert_close(value, before[name])


def test_running_states_are_buffers_not_parameters() -> None:
    extractor = _extractor()
    parameter_ids = {id(value) for value in extractor.parameters()}

    for state in (extractor.attribute_standardizer, extractor.energy_scale):
        assert list(state.named_buffers())
        assert not any(id(value) in parameter_ids for value in state.buffers())


def test_normal_forward_does_not_update_running_state() -> None:
    extractor = _extractor()
    tokens, positions, mask = _inputs()
    before = {name: value.clone() for name, value in extractor.named_buffers()}

    extractor(tokens, positions, channel_mask=mask)

    after = dict(extractor.named_buffers())
    for name in before:
        torch.testing.assert_close(after[name], before[name])


def test_explicit_source_update_updates_both_state_counters_once() -> None:
    extractor = _extractor()
    tokens, positions, mask = _inputs()

    result = extractor.update_source_state(tokens, positions, channel_mask=mask)

    assert result is None
    assert extractor.attribute_standardizer.num_updates.item() == 1
    assert extractor.energy_scale.num_updates.item() == 1


def test_pair_operator_stores_exactly_one_extractor_and_has_no_residual() -> None:
    extractor = _extractor()
    operator = SharedChannelStructureOperator(extractor)

    assert operator.extractor is extractor
    assert list(dict(operator.named_children())) == ["extractor"]
    assert not hasattr(operator, "residual")
    assert set(ChannelStructurePairOutput.__dataclass_fields__) == {"trend", "dynamics"}


def test_pair_forward_uses_one_parameter_set_for_trend_and_dynamics() -> None:
    extractor = _extractor()
    operator = SharedChannelStructureOperator(extractor).eval()
    trend, positions, mask = _inputs()
    dynamics = trend * 0.7 + 0.1

    before = operator(trend, dynamics, positions, channel_mask=mask)
    with torch.no_grad():
        extractor.output_projection.weight.add_(0.2)
    after = operator(trend, dynamics, positions, channel_mask=mask)

    assert isinstance(before, ChannelStructurePairOutput)
    assert not torch.allclose(before.trend.feature, after.trend.feature)
    assert not torch.allclose(before.dynamics.feature, after.dynamics.feature)


def test_pair_source_update_increments_each_counter_only_once() -> None:
    operator = SharedChannelStructureOperator(_extractor())
    trend, positions, mask = _inputs()

    operator.update_source_state(
        trend, trend * 0.5, positions, channel_mask=mask
    )

    assert operator.extractor.attribute_standardizer.num_updates.item() == 1
    assert operator.extractor.energy_scale.num_updates.item() == 1


def test_pair_source_update_is_order_invariant() -> None:
    first = SharedChannelStructureOperator(_extractor(seed=123))
    second = copy.deepcopy(first)
    trend, positions, mask = _inputs()
    dynamics = -0.4 * trend + 0.2

    first.update_source_state(trend, dynamics, positions, channel_mask=mask)
    second.update_source_state(dynamics, trend, positions, channel_mask=mask)

    first_buffers = dict(first.extractor.named_buffers())
    second_buffers = dict(second.extractor.named_buffers())
    assert first_buffers.keys() == second_buffers.keys()
    for name in first_buffers:
        torch.testing.assert_close(first_buffers[name], second_buffers[name])


def test_feature_loss_reaches_tokens_edge_encoder_and_output_head() -> None:
    extractor = _extractor()
    tokens, positions, mask = _inputs()
    tokens.requires_grad_()

    output = extractor(tokens, positions, channel_mask=mask)
    output.feature.square().mean().backward()

    assert tokens.grad is not None and tokens.grad.abs().sum().item() > 0
    assert torch.isfinite(tokens.grad).all()
    for parameter in extractor.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_pair_loss_accumulates_into_shared_parameters() -> None:
    extractor = _extractor()
    operator = SharedChannelStructureOperator(extractor)
    trend, positions, mask = _inputs()
    dynamics = (trend * -0.6 + 0.1).requires_grad_()
    trend = trend.requires_grad_()

    output = operator(trend, dynamics, positions, channel_mask=mask)
    (output.trend.feature.square().mean() + output.dynamics.feature.square().mean()).backward()

    assert trend.grad is not None and trend.grad.abs().sum() > 0
    assert dynamics.grad is not None and dynamics.grad.abs().sum() > 0
    for parameter in extractor.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_dtype_is_preserved(dtype: torch.dtype) -> None:
    extractor = _extractor(dtype=dtype)
    tokens, positions, mask = _inputs(dtype=dtype)

    output = extractor(tokens, positions, channel_mask=mask)

    assert output.feature.dtype == dtype
    assert output.state_relation.dtype == dtype
    assert output.local_velocity.dtype == dtype
    assert all(parameter.dtype == dtype for parameter in extractor.parameters())
    assert extractor.lag_centers.dtype == dtype


def test_backbone_integration_reaches_pse_and_decomposition_parameters() -> None:
    torch.manual_seed(211)
    backbone = StructureBackbone(
        num_channels=3, channel_feature_dim=2, pixel_hidden_dim=3
    )
    extractor = _extractor(token_dim=2)
    pixels = torch.randn(2, 5, 3, 4)
    valid_pixels = torch.ones(2, 5, 4, dtype=torch.bool)
    positions = torch.tensor([0.0, 0.16, 0.37, 0.68, 1.0])

    backbone_output = backbone(pixels, valid_pixels, positions)
    pair = SharedChannelStructureOperator(extractor)(
        backbone_output.decomposition.trend,
        backbone_output.decomposition.dynamics,
        positions,
        time_mask=backbone_output.time_mask,
    )
    loss = (
        pair.trend.feature[:, 0].sum()
        + pair.dynamics.feature[:, 1].sum()
    )
    loss.backward()

    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in backbone.pixel_set_encoder.parameters()
    )
    assert backbone.decomposition._tau_fast_unconstrained.grad is not None
    assert backbone.decomposition._tau_gap_unconstrained.grad is not None


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"num_channels": 1}, "num_channels"),
        ({"token_dim": 0}, "token_dim"),
        ({"lag_centers": (-0.1, 0.1), "lag_widths": (0.1, 0.1)}, "zero"),
        ({"lag_centers": (-0.1, 0.0, 0.2), "lag_widths": (0.1, 0.1, 0.1)}, "symmetric"),
        ({"lag_widths": (0.1, 0.0, 0.1)}, "lag_widths"),
        ({"velocity_bandwidth": 0.0}, "velocity_bandwidth"),
        ({"edge_hidden_dim": 0}, "edge_hidden_dim"),
        ({"structure_dim": 0}, "structure_dim"),
        ({"dropout": 1.0}, "dropout"),
        ({"eps": 0.0}, "eps"),
    ],
)
def test_invalid_constructor_arguments_raise_value_error(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _extractor(**overrides)


@pytest.mark.parametrize(
    "tokens,positions,time_mask,channel_mask,match",
    [
        (torch.ones(2, 5, 3), torch.linspace(0, 1, 5), None, None, "four-dimensional"),
        (torch.ones(2, 5, 2, 2), torch.linspace(0, 1, 5), None, None, "num_channels"),
        (torch.ones(2, 5, 3, 1), torch.linspace(0, 1, 5), None, None, "token_dim"),
        (torch.ones(2, 5, 3, 2, dtype=torch.long), torch.linspace(0, 1, 5), None, None, "floating"),
        (torch.ones(2, 5, 3, 2), torch.ones(4), None, None, "positions"),
        (torch.ones(2, 5, 3, 2), torch.tensor([0.0, 0.2, 0.1, 0.8, 1.0]), None, None, "increasing"),
        (torch.ones(2, 5, 3, 2), torch.tensor([0.0, 0.2, 0.4, 0.8, 1.1]), None, None, r"\[0, 1\]"),
        (torch.ones(2, 5, 3, 2), torch.linspace(0, 1, 5), torch.ones(4), None, "time_mask"),
        (torch.ones(2, 5, 3, 2), torch.linspace(0, 1, 5), None, torch.ones(5, 2), "channel_mask"),
        (torch.ones(2, 5, 3, 2), torch.linspace(0, 1, 5), torch.tensor([1, 1, 2, 1, 1]), None, "0/1"),
    ],
)
def test_invalid_forward_inputs_raise_value_error(
    tokens, positions, time_mask, channel_mask, match
) -> None:
    with pytest.raises(ValueError, match=match):
        _extractor()(tokens, positions, time_mask, channel_mask)


def test_valid_nonfinite_token_raises_but_masked_nonfinite_is_allowed() -> None:
    extractor = _extractor()
    tokens, positions, mask = _inputs()
    tokens[0, 1, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="valid component token"):
        extractor(tokens, positions, channel_mask=mask)

    mask[0, 1, 0] = False
    output = extractor(tokens, positions, channel_mask=mask)
    assert torch.isfinite(output.feature).all()


def test_shared_operator_rejects_component_mismatch() -> None:
    operator = SharedChannelStructureOperator(_extractor())
    trend, positions, mask = _inputs()

    with pytest.raises(ValueError, match="shape"):
        operator(trend, trend[:, :-1], positions, channel_mask=mask)
    with pytest.raises(ValueError, match="dtype"):
        operator(trend, trend.double(), positions, channel_mask=mask)
