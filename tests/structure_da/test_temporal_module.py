from __future__ import annotations

import copy

import pytest
import torch

from methods.structure_da import (
    SharedTemporalStructureOperator,
    TemporalGeometryForwardOutput,
    TemporalGeometryObjective,
    TemporalGeometryPairOutput,
    TemporalShapePhaseCoordinates,
    TemporalSRVFRegistration,
    TemporalStructureEncoder,
    TemporalStructureExtractor,
    TemporalStructureOutput,
    TemporalStructurePairOutput,
)


def _extractor(
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 71,
    **overrides,
) -> TemporalStructureExtractor:
    kwargs = dict(
        num_channels=1,
        channel_feature_dim=2,
        num_basis=6,
        canonical_grid_size=7,
        roughness_grid_size=64,
        smoothing_weight=1e-3,
        min_mean_support=0.0,
        min_dynamic_energy=0.0,
        min_template_mean_support=0.0,
        warp_hidden_dim=8,
        warp_kernel_size=3,
        num_shape_basis=4,
        num_phase_basis=3,
        attribute_projection_dim=3,
        coordinate_hidden_dim=8,
        structure_dim=6,
        dropout=0.0,
    )
    kwargs.update(overrides)
    torch.manual_seed(seed)
    return TemporalStructureExtractor(**kwargs).to(dtype=dtype)


def _inputs(
    *,
    batch_size: int = 2,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(72)
    tokens = torch.randn(batch_size, 6, 1, 2, dtype=dtype)
    positions = torch.tensor(
        [0.0, 39.0, 92.0, 157.0, 244.0, 345.0], dtype=dtype
    )
    mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, False, True, True, False, True],
        ][:batch_size]
    )
    return tokens, positions, mask


def _initialize(
    extractor: TemporalStructureExtractor,
    tokens: torch.Tensor,
    positions: torch.Tensor,
    time_mask: torch.Tensor,
) -> None:
    extractor.update_source_state(tokens, positions, time_mask)


def _source_update_counters(
    extractor: TemporalStructureExtractor,
) -> tuple[int, int, int]:
    registration = extractor.registration
    standardizer = registration.srvf_extractor.functional_lift.standardizer
    support_scale = registration.srvf_extractor.support_scale
    template = registration.source_template
    return (
        int(standardizer.num_updates.item()),
        int(support_scale.num_updates.item()),
        int(template.num_updates.item()),
    )


def test_extractor_assembles_exact_existing_temporal_submodules() -> None:
    extractor = _extractor()

    assert isinstance(extractor.registration, TemporalSRVFRegistration)
    assert isinstance(extractor.coordinates, TemporalShapePhaseCoordinates)
    assert isinstance(extractor.encoder, TemporalStructureEncoder)
    assert isinstance(extractor.geometry_objective, TemporalGeometryObjective)


def test_single_component_task_forward_has_complete_output() -> None:
    extractor = _extractor()
    tokens, positions, time_mask = _inputs()
    _initialize(extractor, tokens, positions, time_mask)

    output = extractor.forward_task(tokens, positions, time_mask)

    assert isinstance(output, TemporalStructureOutput)
    assert output.registration.registered_srvf.shape == (2, 7, 2)
    assert output.coordinates.shape_coordinates.shape == (2, 4, 3)
    assert output.coordinates.phase_coordinates.shape == (2, 4)
    assert output.encoded.feature.shape == (2, 6)
    assert output.encoded.valid is output.coordinates.valid
    assert output.coordinates.valid is output.registration.registration_valid
    assert output.registration.registration_valid.all().item()


def test_standard_forward_is_equivalent_to_task_forward() -> None:
    extractor = _extractor().eval()
    tokens, positions, time_mask = _inputs()
    _initialize(extractor, tokens, positions, time_mask)

    direct = extractor.forward_task(tokens, positions, time_mask)
    standard = extractor(tokens, positions, time_mask)

    torch.testing.assert_close(
        standard.encoded.feature, direct.encoded.feature
    )
    torch.testing.assert_close(
        standard.coordinates.shape_coordinates,
        direct.coordinates.shape_coordinates,
    )
    torch.testing.assert_close(
        standard.coordinates.phase_coordinates,
        direct.coordinates.phase_coordinates,
    )


def test_task_registration_detaches_only_warp_related_tensors() -> None:
    extractor = _extractor()
    tokens, positions, time_mask = _inputs()
    _initialize(extractor, tokens, positions, time_mask)
    differentiable = tokens.clone().requires_grad_()

    registration = extractor._forward_registration(
        differentiable, positions, time_mask
    )
    task_registration = extractor._make_task_registration(registration)

    for tensor in (
        task_registration.interval_logits,
        task_registration.interval_widths,
        task_registration.warp,
        task_registration.warp_derivative,
    ):
        assert not tensor.requires_grad
    assert task_registration.srvf_output.srvf.requires_grad
    assert task_registration.registered_srvf.requires_grad
    assert task_registration.registered_support.shape == (2, 7)
    assert torch.isfinite(task_registration.registered_support).all()


def test_task_loss_updates_upstream_and_head_but_not_warp_estimator() -> None:
    extractor = _extractor()
    source, positions, time_mask = _inputs()
    _initialize(extractor, source, positions, time_mask)
    tokens = (source + 0.1 * torch.randn_like(source)).requires_grad_()

    output = extractor.forward_task(tokens, positions, time_mask)
    output.encoded.feature.square().mean().backward()

    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert tokens.grad.abs().sum().item() > 0
    projection = extractor.coordinates.attribute_projection.weight
    assert projection.grad is not None and torch.isfinite(projection.grad).all()
    for parameter in extractor.encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for parameter in extractor.registration.warp_estimator.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_geometry_loss_updates_warp_estimator_without_detach() -> None:
    extractor = _extractor()
    source, positions, time_mask = _inputs()
    _initialize(extractor, source, positions, time_mask)
    tokens = (source + 0.2 * torch.randn_like(source)).requires_grad_()

    output = extractor.forward_geometry(
        tokens,
        positions,
        time_mask,
        torch.ones(2, dtype=torch.bool),
    )
    output.geometry.total_loss.backward()

    assert isinstance(output, TemporalGeometryForwardOutput)
    assert torch.isfinite(output.geometry.total_loss)
    assert output.structure.registration.registered_srvf.requires_grad
    last = extractor.registration.warp_estimator.network[-1]
    assert last.weight.grad is not None and torch.isfinite(last.weight.grad).all()
    assert last.bias.grad is not None and torch.isfinite(last.bias.grad).all()


def test_normal_forward_never_updates_source_buffers() -> None:
    extractor = _extractor()
    tokens, positions, time_mask = _inputs()
    before = {
        name: buffer.clone()
        for name, buffer in extractor.registration.named_buffers()
    }

    extractor(tokens, positions, time_mask)
    extractor.forward_task(tokens, positions, time_mask)
    extractor.forward_geometry(
        tokens,
        positions,
        time_mask,
        torch.ones(2, dtype=torch.bool),
    )

    after = dict(extractor.registration.named_buffers())
    assert before.keys() == after.keys()
    for name, expected in before.items():
        torch.testing.assert_close(after[name], expected)


def test_explicit_source_update_updates_each_state_once() -> None:
    extractor = _extractor()
    tokens, positions, time_mask = _inputs()

    assert _source_update_counters(extractor) == (0, 0, 0)
    result = extractor.update_source_state(tokens, positions, time_mask)

    assert result is None
    assert _source_update_counters(extractor) == (1, 1, 1)
    assert extractor.registration.source_template.num_updates.item() == 1
    output = extractor.forward_task(tokens, positions, time_mask)
    assert output.registration.registration_valid.all().item()


def test_parameter_groups_are_disjoint_complete_and_fresh() -> None:
    extractor = _extractor()

    first_warp = list(extractor.warp_parameters())
    second_warp = list(extractor.warp_parameters())
    non_warp = list(extractor.non_warp_parameters())
    all_parameters = list(extractor.parameters())
    warp_ids = {id(parameter) for parameter in first_warp}
    non_warp_ids = {id(parameter) for parameter in non_warp}

    assert first_warp is not second_warp
    assert [id(parameter) for parameter in first_warp] == [
        id(parameter) for parameter in second_warp
    ]
    assert len(warp_ids) == len(first_warp)
    assert len(non_warp_ids) == len(non_warp)
    assert warp_ids.isdisjoint(non_warp_ids)
    assert warp_ids | non_warp_ids == {
        id(parameter) for parameter in all_parameters
    }
    assert not ({id(buffer) for buffer in extractor.buffers()} & warp_ids)
    assert not ({id(buffer) for buffer in extractor.buffers()} & non_warp_ids)


def test_shared_operator_stores_exactly_one_extractor_instance() -> None:
    extractor = _extractor()
    operator = SharedTemporalStructureOperator(extractor)

    assert operator.extractor is extractor
    assert list(dict(operator.named_children())) == ["extractor"]
    assert not hasattr(operator, "trend_extractor")
    assert not hasattr(operator, "dynamics_extractor")


def test_shared_task_forward_uses_one_parameter_set_for_both_branches() -> None:
    extractor = _extractor()
    operator = SharedTemporalStructureOperator(extractor).eval()
    trend, positions, time_mask = _inputs()
    dynamics = trend * 0.6 + 0.2
    operator.update_source_state(trend, dynamics, positions, time_mask)

    baseline = operator.forward_task(
        trend, dynamics, positions, time_mask
    )
    with torch.no_grad():
        extractor.encoder.output_head.projection.weight.add_(0.25)
    changed = operator.forward_task(trend, dynamics, positions, time_mask)

    assert isinstance(baseline, TemporalStructurePairOutput)
    assert baseline.trend.encoded.feature.shape == (2, 6)
    assert baseline.dynamics.encoded.feature.shape == (2, 6)
    assert not torch.allclose(
        baseline.trend.encoded.feature, changed.trend.encoded.feature
    )
    assert not torch.allclose(
        baseline.dynamics.encoded.feature,
        changed.dynamics.encoded.feature,
    )


def test_shared_standard_forward_is_task_forward() -> None:
    extractor = _extractor()
    operator = SharedTemporalStructureOperator(extractor).eval()
    trend, positions, time_mask = _inputs()
    dynamics = trend.roll(1, dims=1)
    operator.update_source_state(trend, dynamics, positions, time_mask)

    direct = operator.forward_task(trend, dynamics, positions, time_mask)
    standard = operator(trend, dynamics, positions, time_mask)

    torch.testing.assert_close(
        direct.trend.encoded.feature, standard.trend.encoded.feature
    )
    torch.testing.assert_close(
        direct.dynamics.encoded.feature,
        standard.dynamics.encoded.feature,
    )


def test_pair_source_update_is_order_independent_and_increments_once() -> None:
    first_extractor = _extractor(seed=81)
    second_extractor = copy.deepcopy(first_extractor)
    first = SharedTemporalStructureOperator(first_extractor)
    second = SharedTemporalStructureOperator(second_extractor)
    trend, positions, time_mask = _inputs()
    dynamics = trend * -0.4 + 0.3

    first.update_source_state(trend, dynamics, positions, time_mask)
    second.update_source_state(dynamics, trend, positions, time_mask)

    assert _source_update_counters(first_extractor) == (1, 1, 1)
    assert _source_update_counters(second_extractor) == (1, 1, 1)
    first_buffers = dict(first_extractor.registration.named_buffers())
    second_buffers = dict(second_extractor.registration.named_buffers())
    assert first_buffers.keys() == second_buffers.keys()
    for name in first_buffers:
        torch.testing.assert_close(
            first_buffers[name], second_buffers[name], atol=2e-6, rtol=2e-6
        )


def test_pair_update_resolves_shared_vector_positions_and_mask() -> None:
    operator = SharedTemporalStructureOperator(_extractor())
    trend, positions, _ = _inputs()
    dynamics = trend + 0.1
    one_dimensional_mask = torch.ones(6, dtype=torch.bool)

    operator.update_source_state(
        trend, dynamics, positions, one_dimensional_mask
    )

    assert _source_update_counters(operator.extractor) == (1, 1, 1)


def test_pair_geometry_total_is_mean_of_branch_losses() -> None:
    extractor = _extractor()
    operator = SharedTemporalStructureOperator(extractor)
    trend, positions, time_mask = _inputs()
    dynamics = trend * 0.5 + 0.1
    operator.update_source_state(trend, dynamics, positions, time_mask)

    output = operator.forward_geometry(
        trend,
        dynamics,
        positions,
        time_mask,
        torch.ones(2, dtype=torch.bool),
    )

    assert isinstance(output, TemporalGeometryPairOutput)
    torch.testing.assert_close(
        output.total_loss,
        0.5
        * (
            output.trend.geometry.total_loss
            + output.dynamics.geometry.total_loss
        ),
    )


def test_shared_task_gradients_accumulate_without_warp_gradients() -> None:
    extractor = _extractor()
    operator = SharedTemporalStructureOperator(extractor)
    trend, positions, time_mask = _inputs()
    dynamics = trend * 0.7 - 0.2
    operator.update_source_state(trend, dynamics, positions, time_mask)
    trend = trend.clone().requires_grad_()
    dynamics = dynamics.clone().requires_grad_()

    output = operator.forward_task(trend, dynamics, positions, time_mask)
    loss = (
        output.trend.encoded.feature.square().mean()
        + output.dynamics.encoded.feature.square().mean()
    )
    loss.backward()

    assert trend.grad is not None and trend.grad.abs().sum().item() > 0
    assert dynamics.grad is not None and dynamics.grad.abs().sum().item() > 0
    for parameter in extractor.encoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for parameter in extractor.registration.warp_estimator.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_invalid_registration_sample_has_zero_coordinates_and_feature() -> None:
    extractor = _extractor()
    source, positions, source_mask = _inputs()
    _initialize(extractor, source, positions, source_mask)
    evaluation_mask = source_mask.clone()
    evaluation_mask[1] = torch.tensor(
        [True, False, False, False, False, False]
    )

    output = extractor.forward_task(source, positions, evaluation_mask)

    assert output.registration.registration_valid.tolist() == [True, False]
    torch.testing.assert_close(
        output.coordinates.shape_coordinates[1],
        torch.zeros_like(output.coordinates.shape_coordinates[1]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        output.coordinates.phase_coordinates[1],
        torch.zeros_like(output.coordinates.phase_coordinates[1]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        output.encoded.feature[1],
        torch.zeros_like(output.encoded.feature[1]),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_dtype_is_preserved(dtype: torch.dtype) -> None:
    extractor = _extractor(dtype=dtype)
    tokens, positions, time_mask = _inputs(dtype=dtype)
    _initialize(extractor, tokens, positions, time_mask)

    output = extractor.forward_task(tokens, positions, time_mask)

    assert output.registration.registered_srvf.dtype == dtype
    assert output.coordinates.shape_coordinates.dtype == dtype
    assert output.coordinates.phase_coordinates.dtype == dtype
    assert output.encoded.feature.dtype == dtype
    assert all(parameter.dtype == dtype for parameter in extractor.parameters())


def test_shared_operator_rejects_wrong_extractor_type() -> None:
    with pytest.raises(ValueError, match="TemporalStructureExtractor"):
        SharedTemporalStructureOperator(object())


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("trend", torch.ones(2, 6, 2), "four-dimensional"),
        ("dynamics", torch.ones(2, 6, 2), "four-dimensional"),
        ("dynamics", torch.ones(2, 5, 1, 2), "shape"),
        ("dynamics", torch.ones(2, 6, 1, 2, dtype=torch.float64), "dtype"),
        ("trend", torch.full((2, 6, 1, 2), float("nan")), "finite"),
        ("dynamics", torch.full((2, 6, 1, 2), float("inf")), "finite"),
    ],
)
def test_shared_operator_rejects_invalid_components(field, value, match) -> None:
    operator = SharedTemporalStructureOperator(_extractor())
    trend, positions, time_mask = _inputs()
    dynamics = trend.clone()
    if field == "trend":
        trend = value
    else:
        dynamics = value

    with pytest.raises(ValueError, match=match):
        operator.forward_task(trend, dynamics, positions, time_mask)


def test_shared_operator_rejects_component_device_mismatch() -> None:
    operator = SharedTemporalStructureOperator(_extractor())
    trend, positions, time_mask = _inputs()
    dynamics = torch.ones(2, 6, 1, 2, device="meta")
    with pytest.raises(ValueError, match="device"):
        operator.forward_task(trend, dynamics, positions, time_mask)


@pytest.mark.parametrize(
    "positions,time_mask,match",
    [
        (torch.ones(5), torch.ones(2, 6, dtype=torch.bool), "positions"),
        (torch.ones(3, 6), torch.ones(2, 6, dtype=torch.bool), "positions"),
        (torch.ones(6), torch.ones(5, dtype=torch.bool), "time_mask"),
        (torch.ones(6), torch.ones(3, 6, dtype=torch.bool), "time_mask"),
    ],
)
def test_shared_operator_rejects_invalid_time_inputs(
    positions, time_mask, match
) -> None:
    operator = SharedTemporalStructureOperator(_extractor())
    trend, _, _ = _inputs()
    with pytest.raises(ValueError, match=match):
        operator.forward_task(trend, trend, positions, time_mask)


@pytest.mark.parametrize(
    "source_mask",
    [torch.ones(2), torch.ones(3, dtype=torch.bool)],
)
def test_geometry_rejects_invalid_source_mask(source_mask) -> None:
    extractor = _extractor()
    tokens, positions, time_mask = _inputs()
    with pytest.raises(ValueError, match="source_mask"):
        extractor.forward_geometry(tokens, positions, time_mask, source_mask)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"num_channels": 0}, "num_channels"),
        ({"channel_feature_dim": 0}, "channel_feature_dim"),
        ({"canonical_grid_size": 2}, "canonical_grid_size"),
        ({"num_shape_basis": 0}, "num_shape_basis"),
        ({"num_phase_basis": 6}, "num_phase_basis"),
        ({"attribute_projection_dim": 0}, "attribute_projection_dim"),
        ({"coordinate_hidden_dim": 0}, "coordinate_hidden_dim"),
        ({"structure_dim": 0}, "structure_dim"),
        ({"dropout": 1.0}, "dropout"),
        ({"geometry_alignment_weight": -1.0}, "alignment_weight"),
        ({"eps": 0.0}, "eps"),
    ],
)
def test_extractor_rejects_invalid_constructor_arguments(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _extractor(**overrides)
