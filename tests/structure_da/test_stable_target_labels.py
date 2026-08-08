from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from methods.structure_da.confirmed_phase_view import ConfirmedPhaseView
from methods.structure_da.domain_phase_state import PhaseGroupStatus
from methods.structure_da.prototype_bank import SourcePrototypeBank
from methods.structure_da.target_hypothesis_scan import (
    TargetClassPhaseHypothesis,
    TargetHypothesisScanResult,
)
from tests.structure_da.test_confirmed_phase_view import _group, _state


def _config(**overrides):
    from methods.structure_da.stable_target_labels import StableLabelConfig

    values = dict(
        tau_f=0.2,
        tau_q=0.2,
        cls_confidence_min=0.5,
        cls_margin_min=0.1,
        fused_confidence_min=0.5,
        fused_margin_min=0.1,
        q_confidence_min=0.5,
        q_margin_min=0.1,
    )
    values.update(overrides)
    return StableLabelConfig(**values)


def _bank(**overrides) -> SourcePrototypeBank:
    shape = torch.stack(
        (torch.ones(4, 2), torch.zeros(4, 2), -torch.ones(4, 2))
    )
    values = dict(
        trend_srvf=torch.zeros(3, 4, 2),
        shape_srvf=shape,
        trend_support=torch.ones(3, 4),
        shape_support=torch.ones(3, 4),
        fused=torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        class_counts=torch.tensor([4, 4, 4]),
        ready=torch.ones(3, dtype=torch.bool),
        q_distance_samples=(torch.tensor([0.1]),) * 3,
        f_distance_samples=(torch.tensor([0.1]),) * 3,
        q_quantiles=torch.tensor([[0.5, 1.0, 2.0]] * 3),
        f_quantiles=torch.tensor([[0.5, 1.0, 2.0]] * 3),
        version=1,
    )
    values.update(overrides)
    return SourcePrototypeBank(**values)


def _view(
    *,
    sample_ids: tuple[int, ...] = (0,),
    group_id: int = 0,
    member_classes: tuple[int, ...] = (1,),
    winning_class: int = 1,
) -> ConfirmedPhaseView:
    batch = len(sample_ids)
    logits = torch.zeros(batch, 3)
    logits[:, winning_class] = 5.0
    fused = _bank().fused[winning_class].expand(batch, -1).clone()
    q = _bank().shape_srvf[winning_class].expand(batch, -1, -1).clone()
    return ConfirmedPhaseView(
        sample_ids=torch.tensor(sample_ids),
        group_id=group_id,
        member_classes=member_classes,
        center_gamma=torch.linspace(0.0, 1.0, 128),
        aligned_positions=torch.linspace(0.0, 1.0, 5).expand(batch, -1),
        logits=logits,
        probabilities=torch.softmax(logits, dim=-1),
        fused_repr=fused,
        trend_repr=torch.zeros(batch, 1),
        structure_repr=torch.zeros(batch, 1),
        aligned_q_shape=q,
        aligned_q_support=torch.ones(batch, 4),
        q_valid=torch.ones(batch, dtype=torch.bool),
    )


def _hypothesis(sample_id: int, class_id: int) -> TargetClassPhaseHypothesis:
    return TargetClassPhaseHypothesis(
        sample_id=sample_id,
        class_id=class_id,
        gamma=torch.linspace(0.0, 1.0, 128),
        t_identity_error=1.0,
        t_registered_error=0.5,
        t_gain_ratio=0.5,
        q_shape_distance=0.1,
        q_distance_percentile=0.1,
        common_support_t=1.0,
        common_support_shape=1.0,
        roughness=0.0,
        phase_deviation=0.0,
        preferred=True,
        ambiguous_class=False,
        evidence_weight=1.0,
    )


def _hypothesis_result(*hypotheses: TargetClassPhaseHypothesis, num_samples: int = 1):
    return TargetHypothesisScanResult(
        hypotheses=hypotheses,
        num_samples=num_samples,
        num_pairwise_attempted=0,
        num_pre_support_rejected=0,
        num_solver_failed=0,
        num_gamma_rejected=0,
        num_gain_rejected=0,
        num_shape_support_rejected=0,
        num_outer_rejected=0,
        samples_with_zero_hypothesis=0,
        samples_with_one_hypothesis=0,
        samples_with_two_hypotheses=0,
    )


def _evaluate(view=None, bank=None, config=None):
    from methods.structure_da.stable_target_labels import evaluate_stable_target_candidate

    return evaluate_stable_target_candidate(
        view=view or _view(),
        view_index=0,
        class_id=1,
        source_prototype_bank=bank or _bank(),
        config=config or _config(),
    )


def test_all_three_evidence_branches_must_pass() -> None:
    accepted = _evaluate()
    assert accepted.accepted
    assert accepted.passed_classifier and accepted.passed_fused and accepted.passed_q

    classifier_wrong = _view(winning_class=0)
    assert not _evaluate(view=classifier_wrong).accepted

    fused_wrong = replace(_view(), fused_repr=torch.tensor([[1.0, 0.0]]))
    assert not _evaluate(view=fused_wrong).accepted

    q_wrong = replace(_view(), aligned_q_shape=torch.ones(1, 4, 2))
    assert not _evaluate(view=q_wrong).accepted


def test_outer_support_and_branch_threshold_gates_reject_candidates() -> None:
    fused_query = torch.tensor([[0.2, 1.0]])
    fused_view = replace(_view(), fused_repr=fused_query)
    f_quantiles = _bank().f_quantiles.clone()
    f_quantiles[1, 2] = 1e-4
    assert not _evaluate(view=fused_view, bank=_bank(f_quantiles=f_quantiles)).passed_fused

    q_query = torch.full((1, 4, 2), 0.1)
    q_view = replace(_view(), aligned_q_shape=q_query)
    q_quantiles = _bank().q_quantiles.clone()
    q_quantiles[1, 2] = 1e-4
    assert not _evaluate(view=q_view, bank=_bank(q_quantiles=q_quantiles)).passed_q

    no_support = replace(_view(), aligned_q_support=torch.zeros(1, 4))
    assert not _evaluate(view=no_support).passed_q

    assert not _evaluate(config=_config(cls_confidence_min=0.9999)).passed_classifier
    assert not _evaluate(config=_config(fused_confidence_min=0.9999)).passed_fused
    assert not _evaluate(
        config=_config(q_confidence_min=0.9999, tau_q=100.0)
    ).passed_q


@pytest.mark.parametrize(
    "overrides",
    [
        {"tau_f": 0.0},
        {"tau_q": -1.0},
        {"cls_confidence_min": None, "cls_margin_min": None},
        {"fused_confidence_min": None, "fused_margin_min": None},
        {"q_confidence_min": None, "q_margin_min": None},
    ],
)
def test_stable_label_config_requires_temperature_and_one_gate_per_branch(overrides) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)


def _loader(labels: tuple[int, ...] = (0,)):
    batch = len(labels)
    return [
        {
            "pixels": torch.zeros(batch, 5, 2, 4),
            "valid_pixels": torch.ones(batch, 5, 4, dtype=torch.bool),
            "positions": torch.linspace(0.0, 365.0, 5),
            "label": torch.tensor(labels),
        }
    ]


def _fake_builder_factory(*, fail_group: int | None = None):
    calls: list[tuple[int, tuple[int, ...]]] = []

    def build(*, model, batch, sample_ids, group):
        ids = tuple(int(value) for value in sample_ids.tolist())
        calls.append((group.group_id, ids))
        winning = 0 if group.group_id == fail_group else group.member_classes[0]
        return _view(
            sample_ids=ids,
            group_id=group.group_id,
            member_classes=group.member_classes,
            winning_class=winning,
        )

    return calls, build


def _scan(monkeypatch, *, hypotheses, state, fail_group=None, labels=(0,)):
    import methods.structure_da.stable_target_labels as module

    calls, builder = _fake_builder_factory(fail_group=fail_group)
    monkeypatch.setattr(module, "build_confirmed_phase_view", builder)
    ema = SimpleNamespace(model=lambda: torch.nn.Linear(1, 1).eval())
    result = module.scan_stable_target_labels(
        ema_teacher=ema,
        target_loader=_loader(labels),
        hypothesis_result=_hypothesis_result(*hypotheses, num_samples=len(labels)),
        phase_state=state,
        source_prototype_bank=_bank(),
        config=_config(),
    )
    return result, calls


def test_one_accepted_candidate_becomes_stable_and_two_accepted_remain_ambiguous(monkeypatch) -> None:
    groups = (
        _group(0, (1,), PhaseGroupStatus.CONFIRMED),
        _group(1, (2,), PhaseGroupStatus.CONFIRMED, power=0.7),
    )
    hypotheses = (_hypothesis(0, 1), _hypothesis(0, 2))
    one, _ = _scan(
        monkeypatch,
        hypotheses=hypotheses,
        state=_state(*groups),
        fail_group=1,
    )
    assert [(label.sample_id, label.class_id) for label in one.stable_labels] == [(0, 1)]

    both, _ = _scan(monkeypatch, hypotheses=hypotheses, state=_state(*groups))
    assert both.stable_labels == ()
    assert both.num_ambiguous_rejected == 1


def test_two_classes_in_same_confirmed_group_reuse_one_teacher_forward(monkeypatch) -> None:
    group = _group(0, (1, 2), PhaseGroupStatus.CONFIRMED)
    result, calls = _scan(
        monkeypatch,
        hypotheses=(_hypothesis(0, 1), _hypothesis(0, 2)),
        state=_state(group),
    )
    assert calls == [(0, (0,))]
    assert result.num_candidate_views == 1


def test_provisional_groups_produce_no_views_or_labels(monkeypatch) -> None:
    provisional = _group(0, (1,), PhaseGroupStatus.PROVISIONAL)
    result, calls = _scan(
        monkeypatch,
        hypotheses=(_hypothesis(0, 1),),
        state=_state(provisional),
    )
    assert calls == []
    assert result.candidates == ()
    assert result.stable_labels == ()
    assert result.num_without_confirmed_phase == 1


def test_target_true_labels_are_ignored_and_outputs_are_no_grad(monkeypatch) -> None:
    group = _group(0, (1,), PhaseGroupStatus.CONFIRMED)
    hypotheses = (_hypothesis(0, 1),)
    first, _ = _scan(
        monkeypatch, hypotheses=hypotheses, state=_state(group), labels=(0,)
    )
    second, _ = _scan(
        monkeypatch, hypotheses=hypotheses, state=_state(group), labels=(2,)
    )
    assert [(c.sample_id, c.class_id, c.accepted) for c in first.candidates] == [
        (c.sample_id, c.class_id, c.accepted) for c in second.candidates
    ]
    assert [(x.sample_id, x.class_id) for x in first.stable_labels] == [
        (x.sample_id, x.class_id) for x in second.stable_labels
    ]
    for label in first.stable_labels:
        assert label.aligned_q_shape.requires_grad is False
        assert label.aligned_q_support.requires_grad is False
        assert label.fused_repr.requires_grad is False


def test_confirmed_phase_scan_no_longer_requires_individual_dp_hypotheses(monkeypatch) -> None:
    import methods.structure_da.stable_target_labels as module

    calls, builder = _fake_builder_factory()
    monkeypatch.setattr(module, "build_confirmed_phase_view", builder)
    ema = SimpleNamespace(model=lambda: torch.nn.Linear(1, 1).eval())
    loader = [
        {
            "index": torch.tensor([10, 20]),
            "pixels": torch.zeros(2, 5, 2, 4),
            "valid_pixels": torch.ones(2, 5, 4, dtype=torch.bool),
            "positions": torch.linspace(0.0, 365.0, 5),
            "label": torch.tensor([0, 0]),
        }
    ]
    group = _group(0, (1, 2), PhaseGroupStatus.CONFIRMED)
    result = module.scan_stable_target_labels_from_confirmed_phase(
        ema_teacher=ema,
        target_loader=loader,
        phase_state=_state(group),
        source_prototype_bank=_bank(),
        config=_config(),
        sample_ids=(20,),
    )

    assert calls == [(0, (20,))]
    assert result.num_samples == 1
    assert [(item.sample_id, item.class_id) for item in result.stable_labels] == [(20, 1)]
    assert result.num_candidate_views == 1


def _roundc_phase_config(**overrides):
    from methods.structure_da.domain_phase_state import DomainPhaseConfig

    values = dict(
        phase_min_samples_per_class=1.0,
        phase_class_dispersion_max=1.0,
        phase_class_diameter_max=1.0,
        phase_group_dispersion_max=1.0,
        phase_group_diameter_max=0.5,
        phase_group_core_separation=0.1,
        phase_global_radius=2.0,
        phase_confirmation_patience=2,
        phase_center_drift_max=1.0,
    )
    values.update(overrides)
    return DomainPhaseConfig(**values)


def _roundc_pairwise(sample_id: int, class_id: int, gamma: torch.Tensor, *, eligible=True):
    from methods.structure_da.target_hypothesis_scan import PairwiseClassAlignment

    return PairwiseClassAlignment(
        sample_id=sample_id,
        class_id=class_id,
        gamma=gamma.detach().clone(),
        t_identity_error=1.0,
        t_registered_error=0.5,
        t_gain_ratio=0.5,
        pre_common_support_t=1.0,
        common_support_t=1.0,
        gamma_finite=True,
        gamma_endpoint_error=0.0,
        gamma_strictly_increasing=True,
        gamma_min_increment=0.01,
        gamma_max_local_speed=2.0,
        gamma_roughness=0.0,
        phase_deviation=0.1,
        q_shape_distance=0.1,
        q_distance_percentile=0.1,
        common_support_shape=1.0,
        numerically_valid=True,
        phase_evidence_eligible=eligible,
        reject_reasons=(),
    )


def _roundc_candidate(sample_id: int, class_id: int, *, ambiguous=False):
    from methods.structure_da.target_hypothesis_scan import CandidatePseudoLabel

    return CandidatePseudoLabel(
        sample_id=sample_id,
        class_id=class_id,
        q_shape_distance=0.1,
        q_distance_percentile=0.1,
        phase_evidence_eligible=True,
        ambiguous=ambiguous,
    )


def _roundc_scan_result(*, sample_id=20, class_id=1, gamma=None):
    gamma = torch.linspace(0.0, 1.0, 128).pow(1.5) if gamma is None else gamma
    return replace(
        _hypothesis_result(num_samples=1),
        pairwise_alignments=(_roundc_pairwise(sample_id, class_id, gamma),),
        candidate_pseudo_labels=(_roundc_candidate(sample_id, class_id),),
        scanned_sample_ids=(sample_id,),
    )


def _roundc_nonidentity_state(group):
    from methods.structure_da.domain_phase_state import PhaseDecisionStatus

    return replace(
        _state(group),
        decision_status=PhaseDecisionStatus.NONIDENTITY_CONFIRMED,
        decision_stability_age=2,
    )


def _roundc_loader(label=0):
    return [
        {
            "index": torch.tensor([20]),
            "pixels": torch.zeros(1, 5, 2, 4),
            "valid_pixels": torch.ones(1, 5, 4, dtype=torch.bool),
            "positions": torch.linspace(0.0, 365.0, 5),
            "label": torch.tensor([label]),
        }
    ]


def test_roundc_formal_scan_preserves_primary_candidate_identity(monkeypatch) -> None:
    import methods.structure_da.stable_target_labels as module

    group = _group(0, (1, 2), PhaseGroupStatus.CONFIRMED, power=1.5)
    state = _roundc_nonidentity_state(group)
    scan = _roundc_scan_result(gamma=group.center_gamma.float())
    seen_classes = []
    original_evaluate = module.evaluate_stable_target_candidate

    def fake_evaluate(*, view, view_index, class_id, source_prototype_bank, config):
        seen_classes.append(class_id)
        # The classifier/fused/q validators reject the original candidate.  The
        # formal scan must abstain rather than invent class 2 from the same group.
        return replace(
            original_evaluate(
                view=_view(sample_ids=(20,), member_classes=(1, 2)),
                view_index=0,
                class_id=1,
                source_prototype_bank=_bank(),
                config=_config(),
            ),
            sample_id=20,
            class_id=class_id,
            group_id=view.group_id,
            passed_classifier=False,
            accepted=False,
            reject_reason="classifier",
        )

    def fake_view(**kwargs):
        return _view(
            sample_ids=tuple(int(x) for x in kwargs["sample_ids"].tolist()),
            group_id=kwargs["group_id"],
            member_classes=kwargs["member_classes"],
            winning_class=1,
        )

    monkeypatch.setattr(module, "evaluate_stable_target_candidate", fake_evaluate)
    monkeypatch.setattr(module, "build_phase_calibrated_view", fake_view)
    ema = SimpleNamespace(model=lambda: torch.nn.Linear(1, 1).eval())
    result = module.scan_stable_target_labels_from_candidates(
        ema_teacher=ema,
        target_loader=_roundc_loader(),
        hypothesis_result=scan,
        phase_state=state,
        phase_config=_roundc_phase_config(),
        source_prototype_bank=_bank(),
        config=_config(),
        sample_ids=(20,),
    )
    assert seen_classes == [1]
    assert [(item.sample_id, item.class_id) for item in result.candidates] == [(20, 1)]
    assert result.stable_labels == ()


def test_roundc_formal_scan_uses_group_center_not_individual_candidate_gamma(monkeypatch) -> None:
    import methods.structure_da.stable_target_labels as module

    group = _group(0, (1, 2), PhaseGroupStatus.CONFIRMED, power=1.5)
    candidate_gamma = group.center_gamma.float().pow(1.01)
    scan = _roundc_scan_result(gamma=candidate_gamma)
    state = _roundc_nonidentity_state(group)
    seen_gamma = []

    def fake_view(**kwargs):
        seen_gamma.append(kwargs["center_gamma"].detach().cpu().double())
        return _view(
            sample_ids=(20,),
            group_id=kwargs["group_id"],
            member_classes=kwargs["member_classes"],
            winning_class=1,
        )

    monkeypatch.setattr(module, "build_phase_calibrated_view", fake_view)
    ema = SimpleNamespace(model=lambda: torch.nn.Linear(1, 1).eval())
    result = module.scan_stable_target_labels_from_candidates(
        ema_teacher=ema,
        target_loader=_roundc_loader(),
        hypothesis_result=scan,
        phase_state=state,
        phase_config=_roundc_phase_config(phase_group_diameter_max=1.0),
        source_prototype_bank=_bank(),
        config=_config(),
        sample_ids=(20,),
    )
    assert len(result.stable_labels) == 1
    torch.testing.assert_close(seen_gamma[0], group.center_gamma)
    assert not torch.equal(seen_gamma[0].float(), candidate_gamma)
    torch.testing.assert_close(
        result.stable_labels[0].aligned_q_shape,
        _bank().shape_srvf[1],
    )


def test_roundc_compatible_nonmember_class_can_use_confirmed_group_center(monkeypatch) -> None:
    import methods.structure_da.stable_target_labels as module

    group = _group(0, (1, 2), PhaseGroupStatus.CONFIRMED, power=1.5)
    state = _roundc_nonidentity_state(group)
    scan = _roundc_scan_result(
        class_id=0,
        gamma=group.center_gamma.float(),
    )
    seen = []

    def fake_view(**kwargs):
        seen.append(
            (
                kwargs["group_id"],
                kwargs["member_classes"],
                kwargs["center_gamma"].detach().clone(),
            )
        )
        return _view(
            sample_ids=(20,),
            group_id=kwargs["group_id"],
            member_classes=kwargs["member_classes"],
            winning_class=0,
        )

    monkeypatch.setattr(module, "build_phase_calibrated_view", fake_view)
    ema = SimpleNamespace(model=lambda: torch.nn.Linear(1, 1).eval())
    result = module.scan_stable_target_labels_from_candidates(
        ema_teacher=ema,
        target_loader=_roundc_loader(),
        hypothesis_result=scan,
        phase_state=state,
        phase_config=_roundc_phase_config(phase_group_diameter_max=0.05),
        source_prototype_bank=_bank(),
        config=_config(),
        sample_ids=(20,),
    )

    assert group.member_classes == (1, 2)
    assert seen[0][0] == 0
    assert seen[0][1] == (1, 2)
    torch.testing.assert_close(seen[0][2], group.center_gamma)
    assert result.num_phase_compatible == 1
    assert [(item.sample_id, item.class_id, item.group_id) for item in result.stable_labels] == [
        (20, 0, 0)
    ]


def test_roundc_phase_incompatible_primary_candidate_is_rejected_before_class_gates(monkeypatch) -> None:
    import methods.structure_da.stable_target_labels as module

    group = _group(0, (1, 2), PhaseGroupStatus.CONFIRMED, power=2.0)
    scan = _roundc_scan_result(gamma=torch.linspace(0.0, 1.0, 128).pow(0.3))
    state = _roundc_nonidentity_state(group)
    calls = []
    monkeypatch.setattr(
        module,
        "build_phase_calibrated_view",
        lambda **kwargs: calls.append(kwargs) or _view(),
    )
    ema = SimpleNamespace(model=lambda: torch.nn.Linear(1, 1).eval())
    result = module.scan_stable_target_labels_from_candidates(
        ema_teacher=ema,
        target_loader=_roundc_loader(),
        hypothesis_result=scan,
        phase_state=state,
        phase_config=_roundc_phase_config(phase_group_diameter_max=0.01),
        source_prototype_bank=_bank(),
        config=_config(),
        sample_ids=(20,),
    )
    assert calls == []
    assert result.num_phase_compatible == 0
    assert result.num_phase_incompatible == 1
    assert result.stable_labels == ()
    assert result.candidates[0].reject_reason == "phase_residual"


def test_roundc_identity_confirmed_uses_identity_group_center(monkeypatch) -> None:
    import methods.structure_da.stable_target_labels as module
    from methods.structure_da.domain_phase_state import DomainPhaseState, PhaseDecisionStatus

    identity = torch.linspace(0.0, 1.0, 128)
    scan = _roundc_scan_result(gamma=identity)
    state = DomainPhaseState(
        scan_index=1,
        m=0,
        class_centers=(),
        valid_phase_classes=(1, 2),
        groups=(),
        rejected_classes=(),
        decision_status=PhaseDecisionStatus.IDENTITY_CONFIRMED,
        decision_stability_age=2,
        identity_evidence_classes=(1, 2),
        identity_evidence_count=4.0,
    )
    seen = []

    def fake_view(**kwargs):
        seen.append((kwargs["group_id"], kwargs["center_gamma"].detach().clone()))
        return _view(
            sample_ids=(20,),
            group_id=kwargs["group_id"],
            member_classes=kwargs["member_classes"],
            winning_class=1,
        )

    monkeypatch.setattr(module, "build_phase_calibrated_view", fake_view)
    ema = SimpleNamespace(model=lambda: torch.nn.Linear(1, 1).eval())
    result = module.scan_stable_target_labels_from_candidates(
        ema_teacher=ema,
        target_loader=_roundc_loader(label=2),
        hypothesis_result=scan,
        phase_state=state,
        phase_config=_roundc_phase_config(
            phase_identity_radius=0.05,
            phase_identity_gain_ratio_min=0.95,
        ),
        source_prototype_bank=_bank(),
        config=_config(),
        sample_ids=(20,),
    )
    assert [(item.sample_id, item.class_id, item.group_id) for item in result.stable_labels] == [
        (20, 1, -1)
    ]
    assert seen[0][0] == -1
    torch.testing.assert_close(seen[0][1], identity.double())


def test_roundc_confirmed_phase_can_resolve_existing_primary_secondary_ambiguity(monkeypatch) -> None:
    import methods.structure_da.stable_target_labels as module
    from methods.structure_da.target_hypothesis_scan import CandidatePseudoLabel

    group = _group(0, (1, 2), PhaseGroupStatus.CONFIRMED, power=1.5)
    state = _roundc_nonidentity_state(group)
    primary_gamma = torch.linspace(0.0, 1.0, 128).pow(0.3)
    secondary_gamma = group.center_gamma.float()
    candidate = CandidatePseudoLabel(
        sample_id=20,
        class_id=1,
        q_shape_distance=0.10,
        q_distance_percentile=0.1,
        phase_evidence_eligible=True,
        ambiguous=True,
        secondary_class_id=2,
        secondary_q_shape_distance=0.11,
        secondary_phase_evidence_eligible=True,
    )
    scan = replace(
        _hypothesis_result(num_samples=1),
        pairwise_alignments=(
            _roundc_pairwise(20, 1, primary_gamma),
            _roundc_pairwise(20, 2, secondary_gamma),
        ),
        candidate_pseudo_labels=(candidate,),
        scanned_sample_ids=(20,),
    )

    def fake_view(**kwargs):
        return _view(
            sample_ids=(20,),
            group_id=kwargs["group_id"],
            member_classes=kwargs["member_classes"],
            winning_class=2,
        )

    monkeypatch.setattr(module, "build_phase_calibrated_view", fake_view)
    ema = SimpleNamespace(model=lambda: torch.nn.Linear(1, 1).eval())
    result = module.scan_stable_target_labels_from_candidates(
        ema_teacher=ema,
        target_loader=_roundc_loader(),
        hypothesis_result=scan,
        phase_state=state,
        phase_config=_roundc_phase_config(phase_group_diameter_max=0.05),
        source_prototype_bank=_bank(),
        config=_config(),
        sample_ids=(20,),
    )
    assert [(item.sample_id, item.class_id) for item in result.stable_labels] == [(20, 2)]
    primary = next(item for item in result.candidates if item.class_id == 1)
    secondary = next(item for item in result.candidates if item.class_id == 2)
    assert primary.phase_compatible is False
    assert secondary.phase_compatible is True


def test_roundc_postconfirmation_phase_check_does_not_require_estimation_eligibility(monkeypatch) -> None:
    import methods.structure_da.stable_target_labels as module

    group = _group(0, (1, 2), PhaseGroupStatus.CONFIRMED, power=1.5)
    state = _roundc_nonidentity_state(group)
    scan = replace(
        _roundc_scan_result(gamma=group.center_gamma.float()),
        pairwise_alignments=(
            _roundc_pairwise(20, 1, group.center_gamma.float(), eligible=False),
        ),
    )

    def fake_view(**kwargs):
        return _view(
            sample_ids=(20,),
            group_id=kwargs["group_id"],
            member_classes=kwargs["member_classes"],
            winning_class=1,
        )

    monkeypatch.setattr(module, "build_phase_calibrated_view", fake_view)
    ema = SimpleNamespace(model=lambda: torch.nn.Linear(1, 1).eval())
    result = module.scan_stable_target_labels_from_candidates(
        ema_teacher=ema,
        target_loader=_roundc_loader(),
        hypothesis_result=scan,
        phase_state=state,
        phase_config=_roundc_phase_config(),
        source_prototype_bank=_bank(),
        config=_config(),
        sample_ids=(20,),
    )
    assert result.num_phase_compatible == 1
    assert [(item.sample_id, item.class_id) for item in result.stable_labels] == [(20, 1)]
