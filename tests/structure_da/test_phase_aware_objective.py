from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from methods.structure_da.phase_aware_objective import (
    PhaseAwarePrototypeAlignment,
    PhaseAwareSemanticFeatures,
    PhaseAwareTaskLossWeights,
    PhaseAwareTaskObjective,
    PrototypeAlignmentConfig,
    PrototypeAlignmentLossOutput,
    PrototypeAlignmentWeights,
    TrendLedGeometryObjective,
    cosine_prototype_distance,
    support_weighted_srvf_distance,
)
from methods.structure_da.quality_fusion import TwoScaleQualityLossOutput


def _config(**overrides) -> PrototypeAlignmentConfig:
    values = dict(
        num_classes=3,
        canonical_grid_size=4,
        srvf_dim=2,
        shape_dim=2,
        raw_dim=2,
        prototype_momentum=0.5,
        radius_buffer_size=3,
        min_radius_samples=2,
        min_source_class_samples_for_separation=1,
        target_q_margin=0.05,
        raw_pull_confidence=0.4,
    )
    values.update(overrides)
    return PrototypeAlignmentConfig(**values)


def _features(batch: int = 3, *, requires_grad: bool = False) -> PhaseAwareSemanticFeatures:
    torch.manual_seed(41)
    q = torch.randn(batch, 4, 2, requires_grad=requires_grad)
    z = torch.randn(batch, 2, requires_grad=requires_grad)
    trend = torch.randn(batch, 2, requires_grad=requires_grad)
    structure = torch.randn(batch, 2, requires_grad=requires_grad)
    return PhaseAwareSemanticFeatures(
        aligned_structure_srvf=q,
        aligned_structure_support=torch.ones(batch, 4),
        shape_feature=z,
        trend_embedding=trend,
        structure_embedding=structure,
        shape_valid=torch.ones(batch, dtype=torch.bool),
        component_valid=torch.ones(batch, dtype=torch.bool),
    )


def _ready_alignment(**config_overrides) -> PhaseAwarePrototypeAlignment:
    module = PhaseAwarePrototypeAlignment(_config(**config_overrides))
    directions = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    with torch.no_grad():
        module.q_prototype.copy_(directions[:, None, :].expand(-1, 4, -1))
        module.q_support.fill_(1.0)
        module.z_prototype.copy_(directions)
        module.trend_prototype.copy_(directions)
        module.structure_prototype.copy_(directions)
        for name in (
            "q_prototype_ready", "z_prototype_ready",
            "trend_prototype_ready", "structure_prototype_ready",
            "q_radius_ready", "z_radius_ready",
            "trend_radius_ready", "structure_radius_ready",
        ):
            getattr(module, name).fill_(True)
        module.q_radius_inner.fill_(0.5)
        module.q_radius_outer.fill_(1.5)
        module.z_radius_inner.fill_(0.2)
        module.trend_radius_inner.fill_(0.2)
        module.structure_radius_inner.fill_(0.2)
    return module


def _semantic_from_vectors(
    q_vectors: torch.Tensor,
    z: torch.Tensor,
    trend: torch.Tensor,
    structure: torch.Tensor,
    *,
    requires_grad: bool = False,
) -> PhaseAwareSemanticFeatures:
    q = q_vectors[:, None, :].expand(-1, 4, -1).clone().requires_grad_(requires_grad)
    z = z.clone().requires_grad_(requires_grad)
    trend = trend.clone().requires_grad_(requires_grad)
    structure = structure.clone().requires_grad_(requires_grad)
    batch = q.shape[0]
    return PhaseAwareSemanticFeatures(
        q, torch.ones(batch, 4), z, trend, structure,
        torch.ones(batch, dtype=torch.bool),
        torch.ones(batch, dtype=torch.bool),
    )


def test_distance_functions_are_support_aware_finite_and_differentiable() -> None:
    eps = 1e-8
    sample = torch.zeros(2, 4, 2, requires_grad=True)
    sample.data[1, -1] = 1000.0
    sample_support = torch.tensor([[1.0] * 4, [1.0, 1.0, 1.0, 1e-8]])
    prototype = torch.zeros(2, 4, 2)
    prototype_support = torch.stack([torch.ones(4), torch.zeros(4)])
    weights = torch.tensor([1 / 6, 1 / 3, 1 / 3, 1 / 6])

    distance, mass, valid = support_weighted_srvf_distance(
        sample, sample_support, prototype, prototype_support, weights,
        min_common_support=0.05, eps=eps,
    )

    assert distance.shape == mass.shape == valid.shape == (2, 2)
    assert distance[0, 0].item() == pytest.approx(eps**0.5)
    assert distance[1, 0].item() < 1.0
    assert not valid[:, 1].any() and distance[:, 1].count_nonzero().item() == 0
    distance[:, 0].sum().backward()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()

    cosine_feature = torch.tensor([[0.0, 0.0], [1.0, 0.0]], requires_grad=True)
    normalized, cosine, ready = cosine_prototype_distance(
        cosine_feature,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([True, False]), eps=eps,
    )
    assert torch.isfinite(normalized).all() and torch.isfinite(cosine).all()
    assert torch.all((cosine >= 0) & (cosine <= 2))
    assert ready.tolist() == [[True, False], [True, False]]
    assert cosine[:, 1].count_nonzero().item() == 0
    cosine[:, 0].sum().backward()
    assert cosine_feature.grad is not None and torch.isfinite(cosine_feature.grad).all()


def test_configuration_and_weights_validate_boundaries() -> None:
    with pytest.raises(ValueError):
        _config(num_classes=1)
    with pytest.raises(ValueError):
        _config(q_inner_quantile=0.95, q_outer_quantile=0.75)
    with pytest.raises(ValueError):
        _config(raw_pull_confidence=1.1)
    with pytest.raises(ValueError):
        PrototypeAlignmentWeights(raw_pull=-1.0)
    with pytest.raises(ValueError):
        PhaseAwareTaskLossWeights(classification=float("nan"))


def test_source_state_uses_old_prototypes_ring_buffers_radii_and_state_dict() -> None:
    module = PhaseAwarePrototypeAlignment(_config())
    source = _features(requires_grad=True)
    labels = torch.tensor([0, 1, 2])

    module.update_source_state(source, labels)
    assert module.q_prototype_ready.all()
    assert module.z_prototype_ready.all()
    assert module.trend_prototype_ready.all()
    assert module.structure_prototype_ready.all()
    assert module.q_distance_count.sum().item() == 0
    assert all(parameter.numel() == 0 for parameter in module.parameters())
    assert source.aligned_structure_srvf.grad_fn is None

    for _ in range(4):
        module.update_source_state(source, labels)
    for name in ("q", "z", "trend", "structure"):
        count = getattr(module, f"{name}_distance_count")
        index = getattr(module, f"{name}_distance_index")
        assert torch.equal(count, torch.full_like(count, 3))
        assert torch.equal(index, torch.full_like(index, 1))
        assert getattr(module, f"{name}_radius_ready").all()
    assert torch.all(module.q_radius_outer >= module.q_radius_inner)

    restored = PhaseAwarePrototypeAlignment(_config())
    restored.load_state_dict(module.state_dict())
    for key, value in module.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_source_state_measures_against_old_prototype_before_ema_update() -> None:
    module = PhaseAwarePrototypeAlignment(_config(min_radius_samples=1))
    first = _semantic_from_vectors(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    second = _semantic_from_vectors(
        torch.tensor([[3.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[0.0, 1.0]]),
    )
    labels = torch.tensor([0])

    module.update_source_state(first, labels)
    module.update_source_state(second, labels)

    assert module.q_distance_buffer[0, 0].item() == pytest.approx(2.0, abs=1e-6)
    assert module.q_prototype[0, 0, 0].item() == pytest.approx(2.0)
    assert first.aligned_structure_srvf.grad is None
    assert all(not value.requires_grad for value in module.buffers())


def test_global_radius_fallback_resolves_missing_class_radius() -> None:
    module = _ready_alignment()
    with torch.no_grad():
        module.q_radius_ready[2] = False
        module.q_global_radius_inner.fill_(0.7)
        module.q_global_radius_inner_ready.fill_(True)
    radius, ready = module._resolved_radius("q", outer=False)
    assert radius[2].item() == pytest.approx(0.7)
    assert ready[2]
    assert not hasattr(module, "update_target_state")


def test_source_losses_update_q_z_trend_and_structure_without_state_gradients() -> None:
    module = _ready_alignment(q_separation_margin=10.0)
    directions = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]])
    source = _semantic_from_vectors(
        directions, directions, directions, directions, requires_grad=True
    )
    target = _semantic_from_vectors(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    output = module(source, torch.tensor([0, 1, 2]), target, target.shape_feature)

    assert output.q_compact_loss > 0
    assert output.q_separate_loss > 0
    assert output.source_z_proto_count == 3
    assert output.source_trend_proto_count == 3
    assert output.source_structure_proto_count == 3
    output.source_shape_loss.backward(retain_graph=True)
    assert source.aligned_structure_srvf.grad is not None
    assert source.shape_feature.grad is not None
    output.source_raw_loss.backward()
    assert source.trend_embedding.grad is not None
    assert source.structure_embedding.grad is not None
    assert all(value.grad is None for value in module.buffers())


def test_source_compact_separation_and_relation_activation_regimes() -> None:
    module = _ready_alignment(q_separation_margin=1.0)
    directions = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    source = _semantic_from_vectors(directions, directions, directions, directions)
    target = _semantic_from_vectors(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
    )

    matched = module(source, torch.tensor([0, 1, 2]), target, target.shape_feature)
    assert matched.q_compact_loss.item() == pytest.approx(0.0)
    assert matched.q_separate_loss.item() == pytest.approx(0.0)
    assert matched.source_relation_count.item() == 3

    mismatched = module(source, torch.tensor([1, 2, 0]), target, target.shape_feature)
    assert mismatched.source_relation_count.item() == 0


def test_target_teacher_shape_da_and_raw_relation_gradient_boundaries() -> None:
    module = _ready_alignment()
    source = _semantic_from_vectors(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
    )
    target = _semantic_from_vectors(
        torch.tensor([[1.0, 0.0], [1.8, 0.0], [3.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0], [1.0, 0.3], [1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0], [1.0, 0.3], [1.0, 0.0]]),
        requires_grad=True,
    )
    shape_da = torch.tensor(
        [[-1.0, 0.2], [0.6, 0.8], [1.0, 0.0]], requires_grad=True
    )
    output = module(source, torch.tensor([0, 1, 2]), target, shape_da)

    assert output.target_teacher_mask.tolist() == [True, True, False]
    assert output.target_inner_mask.tolist() == [True, False, False]
    assert output.target_middle_mask.tolist() == [False, True, False]
    assert output.target_outer_mask.tolist() == [False, False, True]
    assert output.target_pseudo_label.tolist() == [0, 0, -1]
    assert output.q_to_z_target_loss > 0
    assert output.z_pull_loss >= 0
    assert output.q_to_trend_target_loss > 0
    assert output.q_to_structure_target_loss > 0
    output.target_semantic_loss.backward()
    assert shape_da.grad is not None and shape_da.grad.abs().sum() > 0
    assert target.trend_embedding.grad is not None
    assert target.structure_embedding.grad is not None
    assert target.aligned_structure_srvf.grad is None
    assert target.shape_feature.grad is None


def test_target_teacher_rejects_disagreement_small_margin_outer_and_invalid() -> None:
    module = _ready_alignment()
    source = _features()
    target = _semantic_from_vectors(
        torch.tensor([[1.0, 0.0], [0.5, 0.5], [3.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]] * 4),
        torch.tensor([[1.0, 0.0]] * 4),
    )
    target.shape_valid[-1] = False
    output = module(source, torch.tensor([0, 1, 2]), target, target.shape_feature.clone())

    assert not output.target_teacher_mask.any()
    assert output.target_pseudo_label.tolist() == [-1, -1, -1, -1]
    assert output.target_outer_mask.tolist() == [False, False, True, False]


@pytest.mark.parametrize(
    ("raw", "confidence", "relation_positive", "pull_count", "pull_positive"),
    [
        ([-1.0, 0.0], 0.4, True, 0, False),
        ([0.6, -0.8], 0.4, True, 1, True),
        ([0.6, -0.8], 1.0, True, 0, False),
        ([1.0, 0.0], 0.4, True, 1, False),
    ],
)
def test_target_raw_relation_and_pull_use_distinct_gates(
    raw, confidence, relation_positive, pull_count, pull_positive
) -> None:
    module = _ready_alignment(raw_pull_confidence=confidence)
    source = _features()
    target = _semantic_from_vectors(
        torch.tensor([[1.8, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([raw]),
        torch.tensor([raw]),
        requires_grad=True,
    )
    output = module(source, torch.tensor([0, 1, 2]), target, target.shape_feature.detach())

    assert (output.q_to_trend_target_loss.item() > 0) is relation_positive
    assert output.target_trend_relation_count.item() == 1
    assert output.target_trend_pull_count.item() == pull_count
    assert (output.trend_pull_loss.item() > 0) is pull_positive
    output.target_semantic_loss.backward()
    assert target.trend_embedding.grad is not None
    assert target.aligned_structure_srvf.grad is None


def test_raw_pull_does_not_require_two_branch_prototypes() -> None:
    module = _ready_alignment()
    with torch.no_grad():
        module.trend_prototype_ready[1:] = False
        module.structure_prototype_ready[1:] = False
        module.trend_radius_ready[1:] = False
        module.structure_radius_ready[1:] = False
    source = _features()
    target = _semantic_from_vectors(
        torch.tensor([[1.8, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.6, -0.8]]),
        torch.tensor([[0.6, -0.8]]),
    )

    output = module(source, torch.tensor([0, 1, 2]), target, target.shape_feature)

    assert output.target_trend_relation_count.item() == 0
    assert output.target_structure_relation_count.item() == 0
    assert output.target_trend_pull_count.item() == 1
    assert output.target_structure_pull_count.item() == 1
    assert output.trend_pull_loss > 0
    assert output.structure_pull_loss > 0


def test_raw_huber_formula_has_small_and_large_branches() -> None:
    module = _ready_alignment(raw_huber_delta=0.1)
    error = torch.tensor([0.05, 0.2])
    actual = module._raw_huber(error)
    expected = torch.tensor([0.05**2 / 0.2, 0.2 - 0.05])
    torch.testing.assert_close(actual, expected)


def _geometry_selection(interval_widths, scores, trainable, selected, phase_valid):
    candidates = SimpleNamespace(interval_widths=interval_widths)
    return SimpleNamespace(
        candidates=candidates,
        candidate_softmin_score=scores,
        candidate_trainable_mask=trainable,
        candidate_acceptable_mask=torch.zeros_like(trainable),
        selected_candidate_index=selected,
        phase_valid=phase_valid,
    )


def test_trend_led_geometry_uses_differentiable_candidates_and_graph_zeros() -> None:
    logits = torch.randn(3, 2, 3, requires_grad=True)
    widths = torch.softmax(logits, dim=-1)
    scores = (widths.square().sum(dim=-1)).mean(dim=-1)
    selection = _geometry_selection(
        widths, scores,
        torch.tensor([[True, True], [False, False], [True, False]]),
        torch.tensor([0, -1, 1]),
        torch.tensor([True, False, True]),
    )
    output = TrendLedGeometryObjective()(selection, torch.tensor([True, True, False]))
    assert all(getattr(output, field).ndim == 0 for field in (
        "total_loss", "candidate_loss", "center_loss",
        "valid_candidate_sample_count", "source_center_count",
    ))
    output.total_loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    empty = _geometry_selection(
        widths, scores, torch.zeros_like(selection.candidate_trainable_mask),
        torch.full((3,), -1), torch.zeros(3, dtype=torch.bool),
    )
    zero = TrendLedGeometryObjective()(empty, torch.zeros(3, dtype=torch.bool))
    assert zero.candidate_loss.item() == zero.center_loss.item() == 0.0
    assert zero.total_loss.requires_grad


def test_geometry_center_updates_selected_candidate_and_identity_is_exact_zero() -> None:
    logits = torch.randn(2, 2, 3, requires_grad=True)
    widths = torch.softmax(logits, dim=-1)
    selection = _geometry_selection(
        widths,
        widths.sum((-1, -2)) * 0,
        torch.ones(2, 2, dtype=torch.bool),
        torch.tensor([1, -1]),
        torch.tensor([True, True]),
    )
    output = TrendLedGeometryObjective(candidate_weight=0.0)(
        selection, torch.tensor([True, False])
    )
    output.center_loss.backward()

    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 0].count_nonzero() == 0
    assert logits.grad[1].count_nonzero() == 0

    identity_only = _geometry_selection(
        widths.detach(),
        torch.zeros(2),
        torch.ones(2, 2, dtype=torch.bool),
        torch.full((2,), -1),
        torch.ones(2, dtype=torch.bool),
    )
    identity_output = TrendLedGeometryObjective()(identity_only, torch.ones(2, dtype=torch.bool))
    assert identity_output.center_loss.item() == pytest.approx(0.0, abs=1e-12)


def test_geometry_forward_pair_weights_candidates_and_centers_source_only() -> None:
    source_logits = torch.randn(2, 2, 3, requires_grad=True)
    target_logits = torch.randn(3, 2, 3, requires_grad=True)
    source_widths = torch.softmax(source_logits, dim=-1)
    target_widths = torch.softmax(target_logits, dim=-1)
    source_scores = source_widths.square().sum((-1, -2))
    target_scores = target_widths.square().sum((-1, -2))
    source = _geometry_selection(
        source_widths,
        source_scores,
        torch.tensor([[True, False], [False, False]]),
        torch.tensor([0, -1]),
        torch.tensor([True, False]),
    )
    target = _geometry_selection(
        target_widths,
        target_scores,
        torch.tensor([[True, True], [False, True], [False, False]]),
        torch.tensor([1, 1, -1]),
        torch.tensor([True, True, False]),
    )
    objective = TrendLedGeometryObjective(candidate_weight=2.0, center_weight=3.0)

    paired = objective.forward_pair(source, target)
    source_only = objective(source, torch.ones(2, dtype=torch.bool))
    expected_candidate = (source_scores[0] + target_scores[:2].sum()) / 3

    torch.testing.assert_close(paired.candidate_loss, expected_candidate)
    torch.testing.assert_close(paired.center_loss, source_only.center_loss)
    assert paired.valid_candidate_sample_count.item() == 3
    assert paired.source_center_count.item() == 1
    torch.testing.assert_close(
        paired.total_loss,
        2.0 * paired.candidate_loss + 3.0 * paired.center_loss,
    )
    paired.total_loss.backward()
    assert source_logits.grad is not None and torch.isfinite(source_logits.grad).all()
    assert target_logits.grad is not None and torch.isfinite(target_logits.grad).all()
    assert target_logits.grad.abs().sum() > 0


def test_geometry_forward_pair_handles_empty_candidates_and_rejects_mismatch() -> None:
    source_logits = torch.randn(1, 2, 3, requires_grad=True)
    target_logits = torch.randn(2, 2, 3, requires_grad=True)
    source_widths = torch.softmax(source_logits, dim=-1)
    target_widths = torch.softmax(target_logits, dim=-1)
    source = _geometry_selection(
        source_widths,
        torch.full((1,), float("inf"), device=source_widths.device),
        torch.zeros(1, 2, dtype=torch.bool),
        torch.tensor([-1]),
        torch.tensor([False]),
    )
    target = _geometry_selection(
        target_widths,
        torch.full((2,), float("inf"), device=target_widths.device),
        torch.zeros(2, 2, dtype=torch.bool),
        torch.full((2,), -1),
        torch.zeros(2, dtype=torch.bool),
    )

    output = TrendLedGeometryObjective().forward_pair(source, target)
    assert output.candidate_loss.item() == 0
    assert output.center_loss.item() == 0
    assert output.total_loss.requires_grad
    output.total_loss.backward()
    assert source_logits.grad is not None and target_logits.grad is not None

    mismatched = _geometry_selection(
        target_widths[:, :1],
        target_widths.sum((-1, -2)) * 0,
        torch.zeros(2, 1, dtype=torch.bool),
        torch.full((2,), -1),
        torch.zeros(2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="candidate count"):
        TrendLedGeometryObjective().forward_pair(source, mismatched)


def _prototype_loss(**overrides) -> PrototypeAlignmentLossOutput:
    scalar = torch.tensor(1.0, requires_grad=True)
    values = {}
    for field in fields(PrototypeAlignmentLossOutput):
        if field.name == "target_pseudo_label":
            values[field.name] = torch.tensor([-1], dtype=torch.long)
        elif field.name.endswith("_mask"):
            values[field.name] = torch.tensor([False])
        else:
            values[field.name] = scalar
    values.update(overrides)
    return PrototypeAlignmentLossOutput(**values)


def test_task_objective_applies_five_weights_and_rejects_nonfinite_losses() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    qloss = TwoScaleQualityLossOutput(*[torch.tensor(2.0, requires_grad=True)] * 5)
    prototype = _prototype_loss(
        source_shape_loss=torch.tensor(4.0, requires_grad=True),
        source_raw_loss=torch.tensor(5.0, requires_grad=True),
        target_semantic_loss=torch.tensor(6.0, requires_grad=True),
    )
    weights = PhaseAwareTaskLossWeights(1, 2, 3, 4, 6)
    output = PhaseAwareTaskObjective(weights)(logits, labels, qloss, prototype)
    expected = F.cross_entropy(logits, labels) + 4 + 12 + 20 + 36
    torch.testing.assert_close(output.total_loss, expected)
    output.total_loss.backward()
    assert logits.grad is not None
    assert qloss.total_loss.grad is not None
    assert prototype.target_semantic_loss.grad is not None

    bad = _prototype_loss(target_semantic_loss=torch.tensor(float("nan")))
    with pytest.raises(FloatingPointError):
        PhaseAwareTaskObjective()(logits.detach(), labels, qloss, bad)
