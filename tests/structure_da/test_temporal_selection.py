from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import methods.structure_da as structure_da_api
import methods.structure_da.temporal_selection as selection_module
from methods.structure_da.temporal_registration import (
    MonotoneWarpCandidatesOutput,
    MonotoneWarpEstimator,
)
from methods.structure_da.temporal_selection import (
    TrendStructureSelectionConfig,
    select_trend_structure_phase,
)


def _identity_candidates(
    *, batch_size: int = 1, num_candidates: int = 3, grid_size: int = 5
) -> MonotoneWarpCandidatesOutput:
    identity = torch.linspace(0.0, 1.0, grid_size)
    widths = torch.full((batch_size, num_candidates, grid_size - 1), 1.0 / (grid_size - 1))
    warp = identity.expand(batch_size, num_candidates, -1).clone()
    return MonotoneWarpCandidatesOutput(
        interval_logits=torch.zeros_like(widths),
        interval_widths=widths,
        warp=warp,
        warp_derivative=torch.ones_like(warp),
        inverse_warp=warp.clone(),
    )


def _selection_inputs(
    *,
    candidates: MonotoneWarpCandidatesOutput | None = None,
    trend_valid: bool = True,
    structure_valid: bool = True,
    trend_initialized: bool = True,
    structure_initialized: bool = True,
):
    candidates = candidates or _identity_candidates()
    batch_size, _, grid_size = candidates.warp.shape
    grid = torch.linspace(-1.0, 1.0, grid_size).reshape(1, grid_size, 1)
    trend = grid.expand(batch_size, -1, -1).clone()
    structure = (0.5 * grid).expand(batch_size, -1, -1).clone()
    support = torch.ones(batch_size, grid_size)
    weights = torch.full((grid_size,), 1.0 / (grid_size - 1))
    weights[[0, -1]] *= 0.5
    return dict(
        trend_srvf=trend,
        trend_support=support,
        trend_valid=torch.full((batch_size,), trend_valid, dtype=torch.bool),
        trend_template_srvf=trend.clone(),
        trend_template_support=support.clone(),
        trend_template_initialized=torch.tensor(trend_initialized),
        structure_srvf=structure,
        structure_support=support.clone(),
        structure_valid=torch.full((batch_size,), structure_valid, dtype=torch.bool),
        structure_template_srvf=structure.clone(),
        structure_template_support=support.clone(),
        structure_template_initialized=torch.tensor(structure_initialized),
        candidates=candidates,
        integration_weights=weights,
        config=TrendStructureSelectionConfig(),
    )


def _controlled_selection(
    monkeypatch, trend_errors, structure_errors, *, candidates=None, **config
):
    calls = {"group": 0}

    def fake_group_action(srvf, warp, warp_derivative, eps):
        del warp, warp_derivative, eps
        values = trend_errors if calls["group"] == 0 else structure_errors
        calls["group"] += 1
        return torch.tensor(values, dtype=srvf.dtype).sqrt().reshape(-1, 1, 1).expand_as(srvf)

    def fake_warp_sequence(sequence, warp):
        del warp
        return torch.ones_like(sequence)

    monkeypatch.setattr(selection_module, "_apply_srvf_group_action", fake_group_action)
    monkeypatch.setattr(selection_module, "_warp_sequence", fake_warp_sequence)
    inputs = _selection_inputs(candidates=candidates)
    inputs["trend_srvf"] = torch.ones_like(inputs["trend_srvf"])
    inputs["trend_template_srvf"] = torch.zeros_like(inputs["trend_srvf"])
    inputs["structure_srvf"] = torch.ones_like(inputs["structure_srvf"])
    inputs["structure_template_srvf"] = torch.zeros_like(inputs["structure_srvf"])
    config_values = dict(
        identity_weight=0.0,
        roughness_weight=0.0,
        unsupported_weight=0.0,
        ambiguity_relative_tolerance=0.0,
        ambiguity_absolute_tolerance=0.0,
        max_gain_ratio=1.0,
    )
    config_values.update(config)
    inputs["config"] = TrendStructureSelectionConfig(**config_values)
    return select_trend_structure_phase(**inputs)


def test_public_selection_api_is_exported() -> None:
    assert structure_da_api.TrendStructureSelectionConfig is TrendStructureSelectionConfig
    assert structure_da_api.select_trend_structure_phase is select_trend_structure_phase
    assert hasattr(structure_da_api, "TrendStructurePhaseSelectionOutput")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gain_temperature": 0.0}, "gain_temperature"),
        ({"candidate_temperature": -1.0}, "candidate_temperature"),
        ({"min_common_support": 1.1}, "min_common_support"),
        ({"min_interval_speed": 2.0, "max_interval_speed": 1.0}, "min_interval_speed"),
        ({"structure_veto_ratio": 0.0}, "structure_veto_ratio"),
        ({"ambiguity_absolute_tolerance": -1.0}, "ambiguity_absolute_tolerance"),
    ],
)
def test_selection_config_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        TrendStructureSelectionConfig(**kwargs)


def test_identical_candidates_choose_lowest_index_deterministically() -> None:
    output = select_trend_structure_phase(**_selection_inputs())

    assert output.trend_candidate_ambiguous.item()
    assert output.structure_disambiguation_used.item()
    assert output.selected_candidate_index.item() == 0
    assert output.phase_status.item() == 2


def test_trend_candidates_do_not_depend_on_structure() -> None:
    inputs = _selection_inputs(structure_initialized=False)
    first = select_trend_structure_phase(**inputs)
    inputs["structure_srvf"] = inputs["structure_srvf"] * 1000.0 + 17.0
    second = select_trend_structure_phase(**inputs)

    assert first.candidates is second.candidates
    torch.testing.assert_close(
        first.candidate_trend_score,
        second.candidate_trend_score,
        rtol=0.0,
        atol=0.0,
    )


def test_unambiguous_trend_ignores_opposite_structure_preference(monkeypatch) -> None:
    output = _controlled_selection(
        monkeypatch,
        trend_errors=[0.1, 0.8, 0.9],
        structure_errors=[0.9, 0.8, 0.1],
    )

    assert output.candidate_near_optimal_mask.tolist() == [[True, False, False]]
    assert output.selected_candidate_index.item() == 0
    assert not output.structure_disambiguation_used.item()


def test_structure_selects_only_inside_trend_near_set(monkeypatch) -> None:
    output = _controlled_selection(
        monkeypatch,
        trend_errors=[0.1, 0.1, 0.9],
        structure_errors=[0.8, 0.2, 0.01],
        ambiguity_absolute_tolerance=1e-4,
    )

    assert output.candidate_near_optimal_mask.tolist() == [[True, True, False]]
    assert output.selected_candidate_index.item() == 1


def test_structure_breaks_trend_ambiguity(monkeypatch) -> None:
    output = _controlled_selection(
        monkeypatch,
        trend_errors=[0.1, 0.1, 0.9],
        structure_errors=[0.7, 0.2, 0.1],
        ambiguity_absolute_tolerance=1e-4,
    )

    assert output.selected_candidate_index.item() == 1
    assert output.structure_disambiguation_used.item()


def test_invalid_structure_uses_trend_best_without_disambiguation() -> None:
    output = select_trend_structure_phase(
        **_selection_inputs(structure_valid=False)
    )

    assert output.selected_candidate_index.item() == 0
    assert not output.structure_disambiguation_used.item()
    assert not output.structure_candidate_vetoed.item()
    assert output.phase_valid.item()


def test_invalid_trend_produces_failure_identity_even_when_structure_valid() -> None:
    output = select_trend_structure_phase(
        **_selection_inputs(trend_valid=False, structure_valid=True)
    )

    assert output.selected_candidate_index.item() == -1
    assert output.phase_status.item() == 0
    assert not output.phase_valid.item()
    assert output.identity_fallback.item()
    assert not output.structure_shape_valid.item()
    torch.testing.assert_close(
        output.accepted_warp.warp,
        torch.linspace(0.0, 1.0, 5).reshape(1, 5),
    )


def test_no_legal_candidate_is_valid_identity_and_softmin_is_finite() -> None:
    inputs = _selection_inputs()
    inputs["trend_srvf"] = inputs["trend_srvf"].requires_grad_()
    inputs["config"] = replace(inputs["config"], max_gain_ratio=1e-6)
    inputs["trend_template_srvf"] = torch.zeros_like(inputs["trend_srvf"])
    output = select_trend_structure_phase(**inputs)

    assert not output.candidate_legal_mask.any().item()
    assert output.selected_candidate_index.item() == -1
    assert output.phase_status.item() == 1
    assert output.phase_valid.item()
    assert output.identity_accepted.item()
    assert torch.isfinite(output.candidate_softmin_score).all().item()
    assert output.candidate_softmin_score.item() == 0.0
    output.candidate_softmin_score.sum().backward()
    assert torch.isfinite(inputs["trend_srvf"].grad).all().item()


def test_finite_candidate_score_is_preserved_when_legality_threshold_rejects_it() -> None:
    inputs = _selection_inputs()
    inputs["trend_template_srvf"] = torch.zeros_like(inputs["trend_srvf"])
    inputs["config"] = replace(inputs["config"], max_gain_ratio=0.5)

    output = select_trend_structure_phase(**inputs)

    assert not output.candidate_legal_mask.any().item()
    assert torch.isfinite(output.candidate_trend_score).all().item()


def test_trend_preference_and_near_set_exclude_lower_scoring_illegal_candidate(
    monkeypatch,
) -> None:
    candidates = _identity_candidates(num_candidates=3)
    candidates.interval_widths[0, 0] = torch.tensor([0.7, 0.1, 0.1, 0.1])
    candidates.warp[0, 0] = torch.tensor([0.0, 0.7, 0.8, 0.9, 1.0])

    output = _controlled_selection(
        monkeypatch,
        trend_errors=[0.01, 0.2, 0.3],
        structure_errors=[0.01, 0.2, 0.3],
        candidates=candidates,
        max_interval_speed=1.1,
        ambiguity_absolute_tolerance=0.15,
    )

    assert output.candidate_legal_mask.tolist() == [[False, True, True]]
    assert output.trend_preferred_candidate_index.item() == 1
    assert output.candidate_near_optimal_mask.tolist() == [[False, True, True]]


def test_structure_vetoes_all_ambiguous_candidates_to_valid_identity() -> None:
    inputs = _selection_inputs()
    inputs["structure_template_srvf"] = torch.zeros_like(inputs["structure_srvf"])
    inputs["config"] = replace(inputs["config"], structure_veto_ratio=1e-4)
    output = select_trend_structure_phase(**inputs)

    assert output.trend_candidate_ambiguous.item()
    assert output.structure_disambiguation_used.item()
    assert output.structure_candidate_vetoed.item()
    assert output.selected_candidate_index.item() == -1
    assert output.phase_status.item() == 1
    assert output.identity_accepted.item()


def test_same_selected_candidate_is_gathered_for_both_scales() -> None:
    estimator = MonotoneWarpEstimator(1, 5, hidden_dim=4, kernel_size=3, num_candidates=3)
    with torch.no_grad():
        estimator.network[-1].bias.copy_(torch.tensor([-1.0, 0.4, 1.2]))
    trend = torch.linspace(-1.0, 1.0, 5).reshape(1, 5, 1)
    support = torch.ones(1, 5)
    candidates = estimator.forward_candidates(
        trend, trend, support, support, torch.tensor([True])
    )
    inputs = _selection_inputs(candidates=candidates, structure_initialized=False)
    output = select_trend_structure_phase(**inputs)
    selected = output.selected_candidate_index.item()

    assert selected >= 0
    torch.testing.assert_close(output.accepted_warp.warp, candidates.warp[:, selected])
    torch.testing.assert_close(
        output.accepted_trend_registered_srvf,
        output.candidate_trend_registered_srvf[:, selected],
    )
    torch.testing.assert_close(
        output.accepted_structure_registered_srvf,
        output.candidate_structure_registered_srvf[:, selected],
    )


def test_structure_choice_is_stop_gradient() -> None:
    inputs = _selection_inputs()
    inputs["structure_srvf"] = inputs["structure_srvf"].requires_grad_()
    warp_parameter = inputs["candidates"].warp.detach().clone().requires_grad_()
    inputs["candidates"] = replace(inputs["candidates"], warp=warp_parameter)
    output = select_trend_structure_phase(**inputs)

    assert output.candidate_structure_registered_srvf.requires_grad
    assert not output.candidate_structure_ratio.requires_grad
    assert not output.selected_candidate_index.requires_grad
    assert not output.accepted_structure_registered_srvf.requires_grad
    assert inputs["structure_srvf"].grad is None
    assert warp_parameter.grad is None


def test_candidate_softmin_gradients_reach_every_candidate_head() -> None:
    estimator = MonotoneWarpEstimator(1, 5, hidden_dim=4, kernel_size=3, num_candidates=3)
    with torch.no_grad():
        torch.manual_seed(91)
        estimator.network[-1].weight.normal_(mean=0.0, std=0.2)
    trend = torch.linspace(-1.0, 1.0, 5).reshape(1, 5, 1)
    template = torch.flip(trend, dims=[1])
    support = torch.ones(1, 5)
    candidates = estimator.forward_candidates(
        trend, template, support, support, torch.tensor([True])
    )
    inputs = _selection_inputs(candidates=candidates, structure_initialized=False)
    inputs["trend_srvf"] = trend
    inputs["trend_template_srvf"] = template
    inputs["config"] = replace(inputs["config"], max_gain_ratio=10.0)
    output = select_trend_structure_phase(**inputs)
    output.candidate_softmin_score.sum().backward()

    gradient = estimator.network[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all().item()
    assert torch.all(gradient.reshape(3, -1).norm(dim=1) > 0).item()
