from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import pytest
import torch

import methods.structure_da.temporal_module as temporal_module

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
    TrendStructureTemporalCore,
    TrendStructureTemporalCoreOutput,
)
from methods.structure_da.temporal_module import (
    TrendStructureTaskFeatureModule,
    TrendStructureTaskFeatureOutput,
)
from methods.structure_da.temporal_registration import invert_monotone_warp


def _extractor(
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 71,
    **overrides,
) -> TemporalStructureExtractor:
    kwargs = dict(
        feature_dim=2,
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
    tokens = torch.randn(batch_size, 6, 2, dtype=dtype)
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


def _joint_core(**overrides) -> TrendStructureTemporalCore:
    kwargs = dict(
        feature_dim=2,
        trend_num_basis=6,
        structure_num_basis=6,
        canonical_grid_size=7,
        roughness_grid_size=64,
        min_mean_support=0.0,
        min_dynamic_energy=0.0,
        min_template_mean_support=0.0,
        warp_hidden_dim=8,
        warp_kernel_size=3,
        warp_num_candidates=3,
    )
    kwargs.update(overrides)
    torch.manual_seed(73)
    return TrendStructureTemporalCore(**kwargs)


def _joint_inputs():
    trend, positions, mask = _inputs()
    structure = trend + 0.15 * torch.sin(trend)
    return trend, structure, positions, mask


def _joint_state_counters(core: TrendStructureTemporalCore):
    return (
        int(core.trend_srvf_extractor.functional_lift.standardizer.num_updates.item()),
        int(core.structure_srvf_extractor.functional_lift.standardizer.num_updates.item()),
        int(core.trend_srvf_extractor.support_scale.num_updates.item()),
        int(core.structure_srvf_extractor.support_scale.num_updates.item()),
        int(core.trend_template.num_updates.item()),
        int(core.structure_diagnostic_template.num_updates.item()),
    )


def _task_module(**overrides) -> TrendStructureTaskFeatureModule:
    values = dict(
        feature_dim=2,
        shape_output_dim=6,
        trend_num_basis=6,
        structure_num_basis=6,
        canonical_grid_size=7,
        roughness_grid_size=64,
        min_mean_support=0.0,
        min_dynamic_energy=0.0,
        min_template_mean_support=0.0,
        warp_hidden_dim=8,
        warp_kernel_size=3,
        warp_num_candidates=3,
        num_shape_basis=4,
        num_phase_basis=3,
        attribute_projection_dim=3,
        shape_hidden_dim=8,
        shape_dropout=0.0,
    )
    values.update(overrides)
    torch.manual_seed(174)
    return TrendStructureTaskFeatureModule(**values)


def test_joint_core_forward_shapes_dtype_and_single_warp_estimator() -> None:
    core = _joint_core()
    trend, structure, positions, mask = _joint_inputs()
    core.update_source_state(trend, structure, positions, mask)

    output = core(trend, structure, positions, mask)

    assert isinstance(output, TrendStructureTemporalCoreOutput)
    assert output.trend_srvf.srvf.shape == (2, 7, 2)
    assert output.structure_srvf.srvf.shape == (2, 7, 2)
    assert output.selection.candidates.warp.shape == (2, 3, 7)
    assert output.selection.accepted_inverse_warp.shape == (2, 7)
    assert output.selection.accepted_warp.warp.dtype == trend.dtype
    assert sum(1 for module in core.modules() if module is core.warp_estimator) == 1


def test_joint_core_uses_independent_srvf_extractors_and_state() -> None:
    core = _joint_core()

    assert core.trend_srvf_extractor is not core.structure_srvf_extractor
    assert (
        core.trend_srvf_extractor.functional_lift.standardizer
        is not core.structure_srvf_extractor.functional_lift.standardizer
    )
    assert core.trend_template is not core.structure_diagnostic_template


def test_joint_core_structure_never_changes_trend_candidates() -> None:
    core = _joint_core().eval()
    trend, structure, positions, mask = _joint_inputs()
    core.update_source_state(trend, structure, positions, mask)

    first = core(trend, structure, positions, mask)
    second = core(trend, structure * 100.0 - 50.0, positions, mask)

    torch.testing.assert_close(
        first.selection.candidates.warp,
        second.selection.candidates.warp,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.selection.candidate_trend_score,
        second.selection.candidate_trend_score,
        rtol=0.0,
        atol=0.0,
    )


def test_joint_core_accepted_warp_gathers_both_scale_outputs() -> None:
    core = _joint_core().eval()
    trend, structure, positions, mask = _joint_inputs()
    core.update_source_state(trend, structure, positions, mask)

    output = core(trend, structure, positions, mask)
    selection = output.selection
    selected = selection.selected_candidate_index.clamp_min(0)
    batch = torch.arange(trend.shape[0])
    nonidentity = selection.selected_candidate_index >= 0

    torch.testing.assert_close(
        selection.accepted_trend_registered_srvf[nonidentity],
        selection.candidate_trend_registered_srvf[
            batch[nonidentity], selected[nonidentity]
        ],
    )
    torch.testing.assert_close(
        selection.accepted_structure_registered_srvf[nonidentity],
        selection.candidate_structure_registered_srvf[
            batch[nonidentity], selected[nonidentity]
        ],
    )
def test_joint_core_uninitialized_structure_reference_disables_disambiguation() -> None:
    core = _joint_core()
    trend, structure, positions, mask = _joint_inputs()
    trend_output = core.trend_srvf_extractor(trend, positions, mask)
    core.trend_template.update(
        trend_output.srvf,
        trend_output.support_confidence,
        trend_output.structure_valid,
    )

    output = core(trend, structure, positions, mask)

    assert not output.structure_diagnostic_template.initialized.item()
    assert not output.selection.structure_disambiguation_used.any().item()


def test_joint_core_source_update_initializes_both_references_once() -> None:
    core = _joint_core()
    trend, structure, positions, mask = _joint_inputs()

    core.update_source_state(trend, structure, positions, mask)

    assert _joint_state_counters(core) == (1, 1, 1, 1, 1, 1)


def test_joint_core_each_source_update_advances_each_state_at_most_once() -> None:
    core = _joint_core()
    trend, structure, positions, mask = _joint_inputs()
    core.update_source_state(trend, structure, positions, mask)
    before = _joint_state_counters(core)

    core.update_source_state(trend + 0.01, structure + 0.02, positions, mask)

    after = _joint_state_counters(core)
    assert after == tuple(value + 1 for value in before)


def test_joint_core_second_source_update_uses_accepted_representations(monkeypatch) -> None:
    core = _joint_core()
    trend, structure, positions, mask = _joint_inputs()
    core.update_source_state(trend, structure, positions, mask)
    captured = {}
    original_select = core._select
    original_trend_update = core.trend_template.update
    original_structure_update = core.structure_diagnostic_template.update

    def capture_select(*args, **kwargs):
        output = original_select(*args, **kwargs)
        captured["selection"] = output
        return output

    def capture_trend_update(srvf, support, valid):
        captured["trend"] = (srvf.clone(), support.clone(), valid.clone())
        return original_trend_update(srvf, support, valid)

    def capture_structure_update(srvf, support, valid):
        captured["structure"] = (srvf.clone(), support.clone(), valid.clone())
        return original_structure_update(srvf, support, valid)

    monkeypatch.setattr(core, "_select", capture_select)
    monkeypatch.setattr(core.trend_template, "update", capture_trend_update)
    monkeypatch.setattr(
        core.structure_diagnostic_template, "update", capture_structure_update
    )

    core.update_source_state(trend + 0.01, structure + 0.02, positions, mask)

    selection = captured["selection"]
    torch.testing.assert_close(
        captured["trend"][0], selection.accepted_trend_registered_srvf
    )
    torch.testing.assert_close(
        captured["trend"][1], selection.accepted_trend_registered_support
    )
    torch.testing.assert_close(captured["trend"][2], selection.phase_valid)
    torch.testing.assert_close(
        captured["structure"][0], selection.accepted_structure_registered_srvf
    )
    torch.testing.assert_close(
        captured["structure"][1], selection.accepted_structure_registered_support
    )
    torch.testing.assert_close(
        captured["structure"][2], selection.structure_shape_valid
    )


def test_joint_core_forward_is_read_only_for_all_source_state() -> None:
    core = _joint_core()
    trend, structure, positions, mask = _joint_inputs()
    core.update_source_state(trend, structure, positions, mask)
    before = {name: value.clone() for name, value in core.named_buffers()}

    core(trend + 2.0, structure - 3.0, positions, mask)

    for name, value in core.named_buffers():
        torch.testing.assert_close(value, before[name])


def test_joint_core_warp_and_non_warp_parameter_sets_partition_parameters() -> None:
    core = _joint_core()
    warp = {id(parameter) for parameter in core.warp_parameters()}
    non_warp = {id(parameter) for parameter in core.non_warp_parameters()}
    all_parameters = {id(parameter) for parameter in core.parameters()}

    assert warp
    assert warp.isdisjoint(non_warp)
    assert warp | non_warp == all_parameters


def test_joint_core_state_dict_restores_both_scales_templates_and_warp() -> None:
    core = _joint_core()
    trend, structure, positions, mask = _joint_inputs()
    core.update_source_state(trend, structure, positions, mask)
    with torch.no_grad():
        core.warp_estimator.network[-1].weight.add_(0.25)
    expected = copy.deepcopy(core.state_dict())
    restored = _joint_core()

    restored.load_state_dict(expected)

    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_joint_core_cpu_autocast_geometry_is_finite() -> None:
    core = _joint_core()
    trend, structure, positions, mask = _joint_inputs()
    core.update_source_state(trend, structure, positions, mask)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = core(trend, structure, positions, mask)

    for value in (
        output.trend_srvf.srvf,
        output.structure_srvf.srvf,
        output.selection.candidate_trend_score,
        output.selection.accepted_warp.warp,
    ):
        assert torch.isfinite(value).all().item()


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


def test_pair_task_batches_registration_and_cached_geometry_does_not_reregister() -> None:
    extractor = _extractor()
    operator = SharedTemporalStructureOperator(extractor)
    trend, positions, time_mask = _inputs()
    dynamics = trend * 0.5 + 0.1
    operator.update_source_state(trend, dynamics, positions, time_mask)
    calls = []
    handle = extractor.registration.register_forward_hook(
        lambda *args: calls.append(1)
    )

    task = operator.forward_task(trend, dynamics, positions, time_mask)
    assert len(calls) == 1
    geometry = operator.forward_geometry_from_task(
        task, torch.ones(2, dtype=torch.bool)
    )
    handle.remove()

    assert len(calls) == 1
    torch.testing.assert_close(
        geometry.total_loss,
        0.5 * (geometry.trend.geometry.total_loss + geometry.dynamics.geometry.total_loss),
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


@dataclass(frozen=True)
class _NestedFloatingValues:
    half: torch.Tensor
    bfloat: torch.Tensor
    single: torch.Tensor
    double: torch.Tensor
    integer: torch.Tensor
    boolean: torch.Tensor


@dataclass(frozen=True)
class _FloatingValues:
    nested: _NestedFloatingValues
    label: str


def test_floating_dataclass_conversion_is_recursive_and_differentiable() -> None:
    half = torch.randn(3, dtype=torch.float16, requires_grad=True)
    bfloat = torch.randn(3, dtype=torch.bfloat16, requires_grad=True)
    single = torch.randn(3, dtype=torch.float32, requires_grad=True)
    double = torch.randn(3, dtype=torch.float64, requires_grad=True)
    integer = torch.arange(3)
    boolean = torch.tensor([True, False, True])
    original = _FloatingValues(
        nested=_NestedFloatingValues(
            half=half,
            bfloat=bfloat,
            single=single,
            double=double,
            integer=integer,
            boolean=boolean,
        ),
        label="registration",
    )

    converted = temporal_module._floating_dataclass_to_float32(original)

    assert converted is not original
    assert converted.nested is not original.nested
    assert converted.nested.half.dtype == torch.float32
    assert converted.nested.bfloat.dtype == torch.float32
    assert converted.nested.single is single
    assert converted.nested.double is double
    assert converted.nested.integer is integer
    assert converted.nested.boolean is boolean
    assert converted.label == original.label
    assert original.nested.half.dtype == torch.float16
    assert original.nested.bfloat.dtype == torch.bfloat16

    (converted.nested.half.sum() + converted.nested.bfloat.sum()).backward()
    assert half.grad is not None and torch.isfinite(half.grad).all()
    assert bfloat.grad is not None and torch.isfinite(bfloat.grad).all()


def test_coordinate_construction_stays_float32_inside_cpu_autocast() -> None:
    extractor = _extractor()
    tokens, positions, time_mask = _inputs()
    _initialize(extractor, tokens, positions, time_mask)
    tokens = tokens.clone().requires_grad_()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = extractor.forward_task(tokens, positions, time_mask)
        loss = output.encoded.feature.square().mean()
    loss.backward()

    assert output.coordinates.shape_coordinates.dtype == torch.float32
    assert output.coordinates.phase_coordinates.dtype == torch.float32
    assert torch.isfinite(output.encoded.feature).all()
    projection = extractor.coordinates.attribute_projection.weight
    assert projection.grad is not None and torch.isfinite(projection.grad).all()
    for parameter in extractor.encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_float16_autocast_temporal_pipeline_is_finite() -> None:
    extractor = _extractor().cuda()
    tokens, positions, time_mask = _inputs()
    tokens = tokens.cuda()
    positions = positions.cuda()
    time_mask = time_mask.cuda()
    _initialize(extractor, tokens, positions, time_mask)
    tokens = tokens.clone().requires_grad_()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = extractor.forward_task(tokens, positions, time_mask)
        loss = output.encoded.feature.square().mean()
    loss.backward()

    assert output.coordinates.shape_coordinates.dtype == torch.float32
    assert output.coordinates.phase_coordinates.dtype == torch.float32
    assert torch.isfinite(output.encoded.feature).all()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    for parameter in extractor.encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_shared_operator_rejects_wrong_extractor_type() -> None:
    with pytest.raises(ValueError, match="TemporalStructureExtractor"):
        SharedTemporalStructureOperator(object())


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("trend", torch.ones(2, 6, 1, 2), "three-dimensional"),
        ("dynamics", torch.ones(2, 6, 1, 2), "three-dimensional"),
        ("dynamics", torch.ones(2, 5, 2), "shape"),
        ("dynamics", torch.ones(2, 6, 2, dtype=torch.float64), "dtype"),
        ("trend", torch.full((2, 6, 2), float("nan")), "finite"),
        ("dynamics", torch.full((2, 6, 2), float("inf")), "finite"),
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
    dynamics = torch.ones(2, 6, 2, device="meta")
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
        ({"feature_dim": 0}, "feature_dim"),
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


def test_task_feature_module_outputs_shapes_and_calls_core_once(monkeypatch) -> None:
    module = _task_module()
    trend, structure, positions, mask = _joint_inputs()
    module.update_source_state(trend, structure, positions, mask)
    calls = 0
    original = module.core.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module.core, "forward", counted)
    output = module(trend, structure, positions, mask)

    assert isinstance(output, TrendStructureTaskFeatureOutput)
    assert calls == 1
    assert output.aligned_structure_srvf.shape == (2, 7, 2)
    assert output.aligned_structure_support.shape == (2, 7)
    assert output.coordinates.shape_coordinates_fixed.shape == (2, 4, 2)
    assert output.coordinates.shape_coordinates.shape == (2, 4, 3)
    assert output.shape.feature.shape == (2, 6)
    assert output.aligned_positions.shape == (2, 6)
    assert output.aligned_positions.dtype == positions.dtype
    assert not hasattr(module, "phase_encoder")


def test_task_rebuilt_structure_matches_selection_but_preserves_only_s_gradient() -> None:
    module = _task_module()
    trend, structure, positions, mask = _joint_inputs()
    module.update_source_state(trend, structure, positions, mask)
    trend = trend.clone().requires_grad_()
    structure = structure.clone().requires_grad_()

    output = module(trend, structure, positions, mask)

    torch.testing.assert_close(
        output.aligned_structure_srvf,
        output.core.selection.accepted_structure_registered_srvf,
    )
    output.shape.feature.square().sum().backward()
    assert structure.grad is not None and torch.isfinite(structure.grad).all()
    assert structure.grad[mask].abs().sum().item() > 0
    for parameter in module.warp_parameters():
        assert parameter.grad is None


def test_aligned_positions_use_continuous_inverse_warp_and_mask_reference() -> None:
    module = _task_module(time_reference=0.0, time_scale=366.0)
    trend, structure, positions, mask = _joint_inputs()
    module.update_source_state(trend, structure, positions, mask)
    core = module.core(trend, structure, positions, mask)
    warp = torch.tensor(
        [0.0, 0.04, 0.16, 0.42, 0.69, 0.89, 1.0]
    ).expand(2, -1).clone()
    widths = warp[:, 1:] - warp[:, :-1]
    speed = widths * 6
    derivative = torch.cat(
        [speed[:, :1], 0.5 * (speed[:, :-1] + speed[:, 1:]), speed[:, -1:]],
        dim=-1,
    )
    accepted = replace(
        core.selection.accepted_warp,
        interval_widths=widths,
        interval_logits=widths.log(),
        warp=warp,
        warp_derivative=derivative,
    )
    selection = replace(
        core.selection,
        accepted_warp=accepted,
        phase_valid=torch.tensor([True, True]),
    )
    core = replace(core, selection=selection)

    aligned, valid = module._align_positions(positions, mask, core)

    resolved_positions = positions.unsqueeze(0).expand(2, -1)
    safe_u = torch.where(mask, resolved_positions / 366.0, torch.zeros_like(resolved_positions))
    expected = 366.0 * invert_monotone_warp(warp, query=safe_u, eps=module.eps)
    expected = torch.where(mask, expected, torch.zeros_like(expected))
    torch.testing.assert_close(aligned, expected)
    assert not torch.allclose(aligned[0, 1:-1], positions[1:-1])
    assert aligned.dtype.is_floating_point
    assert torch.all(aligned[0, 1:] >= aligned[0, :-1])
    assert torch.all(aligned[~mask] == module.time_reference)
    assert valid.tolist() == [True, True]


def test_identity_and_failure_aligned_position_semantics() -> None:
    module = _task_module()
    trend, structure, positions, mask = _joint_inputs()
    module.update_source_state(trend, structure, positions, mask)
    core = module.core(trend, structure, positions, mask)
    identity = torch.linspace(0.0, 1.0, 7).expand(2, -1)
    widths = torch.full((2, 6), 1.0 / 6)
    accepted = replace(
        core.selection.accepted_warp,
        interval_logits=torch.zeros_like(widths),
        interval_widths=widths,
        warp=identity,
        warp_derivative=torch.ones_like(identity),
    )
    valid_core = replace(
        core,
        selection=replace(core.selection, accepted_warp=accepted, phase_valid=torch.ones(2, dtype=torch.bool)),
    )
    failed_core = replace(
        core,
        selection=replace(core.selection, accepted_warp=accepted, phase_valid=torch.zeros(2, dtype=torch.bool)),
    )

    valid_positions, valid = module._align_positions(positions, mask, valid_core)
    failed_positions, failed_valid = module._align_positions(positions, mask, failed_core)
    expected = positions.unsqueeze(0).expand(2, -1).clone()
    expected = torch.where(mask, expected, torch.zeros_like(expected))

    torch.testing.assert_close(valid_positions, expected)
    torch.testing.assert_close(failed_positions, expected)
    assert valid.tolist() == [True, True]
    assert failed_valid.tolist() == [False, False]


def test_task_parameter_partition_state_dict_and_source_update_delegation(
    monkeypatch,
) -> None:
    module = _task_module()
    warp_ids = {id(parameter) for parameter in module.warp_parameters()}
    non_warp_ids = {id(parameter) for parameter in module.non_warp_parameters()}
    all_ids = {id(parameter) for parameter in module.parameters()}
    assert not warp_ids & non_warp_ids
    assert warp_ids | non_warp_ids == all_ids

    trend, structure, positions, mask = _joint_inputs()
    calls = []
    monkeypatch.setattr(
        module.core,
        "update_source_state",
        lambda *arguments: calls.append(arguments),
    )
    module.update_source_state(trend, structure, positions, mask)
    assert calls == [(trend, structure, positions, mask)]

    restored = _task_module()
    restored.load_state_dict(module.state_dict())
    for name, value in module.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
