from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from methods.structure_da import (
    TrendStructureTemporalCore,
    TrendStructureTemporalCoreOutput,
)
from methods.structure_da.temporal_module import (
    TrendStructureTaskFeatureModule,
    TrendStructureTaskFeatureOutput,
)
from methods.structure_da.temporal_registration import invert_monotone_warp


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
    expected = invert_monotone_warp(warp, query=safe_u, eps=module.eps)
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
    expected = positions.unsqueeze(0).expand(2, -1).clone() / module.time_scale
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
