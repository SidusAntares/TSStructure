from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from methods.structure_da.backbone import StructureBackbone, StructureBackboneOutput
from methods.structure_da.full_model import (
    StructureAwareDomainAdaptationModel,
    StructureAwareForwardOutput,
)
from methods.structure_da.phase_aware_objective import (
    PhaseAwarePrototypeAlignment,
    TrendLedGeometryObjective,
)
from methods.structure_da.representation import (
    PhaseAwareTwoScaleClassifier,
    QualityAwareComponentClassifier,
)
from methods.structure_da.temporal_module import (
    SharedTemporalStructureOperator,
    TrendStructureTaskFeatureModule,
)
from models.ltae import ComponentAwareSharedLTAE, TrendStructureSharedLTAE


def _model(dtype: torch.dtype = torch.float32, **overrides):
    warp_num_candidates = overrides.pop("warp_num_candidates", 3)
    options = dict(
        num_classes=3,
        input_dim=2,
        mlp1=(2, 4, 4),
        mlp2=(8, 4),
        shape_dim=4,
        time_reference=0.0,
        time_scale=365.0,
        temporal_options={
            "trend_num_basis": 4,
            "structure_num_basis": 4,
            "canonical_grid_size": 5,
            "roughness_grid_size": 64,
            "min_mean_support": 0.0,
            "min_dynamic_energy": 0.0,
            "min_template_mean_support": 0.0,
            "warp_hidden_dim": 6,
            "warp_kernel_size": 3,
            "warp_num_candidates": warp_num_candidates,
            "num_shape_basis": 3,
            "num_phase_basis": 2,
            "attribute_projection_dim": 2,
            "shape_hidden_dim": 6,
            "shape_dropout": 0.0,
        },
        representation_options={
            "n_head": 1,
            "d_k": 2,
            "d_model": 8,
            "ltae_mlp": (8, 4),
            "dropout": 0.0,
            "classifier_hidden": (4,),
            "quality_domain_hidden_dim": 5,
        },
        prototype_options={
            "radius_buffer_size": 8,
            "min_radius_samples": 2,
            "min_common_support": 0.0,
        },
        alignment_hidden_dim=5,
        grl_max_iters=10,
    )
    options.update(overrides)
    return StructureAwareDomainAdaptationModel(**options).to(dtype=dtype)


def _inputs(batch: int = 3, length: int = 5, dtype=torch.float32):
    torch.manual_seed(901 + length)
    pixels = torch.randn(batch, length, 2, 4, dtype=dtype)
    valid = torch.ones(batch, length, 4, dtype=torch.bool)
    positions = torch.linspace(0, 300, length).round().long()
    return pixels, valid, positions


@torch.no_grad()
def _initialize_temporal_source_state(model, inputs) -> None:
    backbone = model.forward_backbone(*inputs)
    trend, structure = model._trend_and_structure(backbone)
    model.temporal_features.update_source_state(
        trend.detach().float(),
        structure.detach().float(),
        model._task_positions(inputs[2], backbone).float(),
        backbone.time_mask,
    )


def test_model_contains_only_phase_aware_high_level_modules() -> None:
    model = _model()
    assert isinstance(model.backbone, StructureBackbone)
    assert isinstance(model.temporal_features, TrendStructureTaskFeatureModule)
    assert isinstance(model.representation, PhaseAwareTwoScaleClassifier)
    assert isinstance(model.prototype_alignment, PhaseAwarePrototypeAlignment)
    assert isinstance(model.geometry_objective, TrendLedGeometryObjective)
    estimator = model.temporal_features.core.warp_estimator
    assert estimator.candidate_init_warp_amplitude == pytest.approx(0.015)
    assert estimator.candidate_base_logits.shape == (3, 4)
    assert sum(isinstance(module, TrendStructureSharedLTAE) for module in model.modules()) == 1
    assert not any(isinstance(module, SharedTemporalStructureOperator) for module in model.modules())
    assert not any(isinstance(module, QualityAwareComponentClassifier) for module in model.modules())
    assert not any(isinstance(module, ComponentAwareSharedLTAE) for module in model.modules())


def test_trend_structure_and_forward_output_have_exact_final_semantics() -> None:
    model = _model().eval()
    pixels, valid, positions = _inputs()
    backbone = model.forward_backbone(pixels, valid, positions)
    trend, structure = model._trend_and_structure(backbone)
    torch.testing.assert_close(trend, backbone.decomposition.trend)
    torch.testing.assert_close(
        structure,
        backbone.decomposition.trend + backbone.decomposition.dynamics,
    )

    output = model.forward_from_backbone(backbone, positions)
    assert isinstance(output, StructureAwareForwardOutput)
    assert set(output.__dataclass_fields__) == {
        "backbone", "temporal", "representation", "semantic"
    }
    assert model.representation.fused_dim == 2 * model.representation.component_dim + 4
    expected_fused = torch.cat(
        [
            output.representation.quality.weighted_trend,
            output.representation.quality.weighted_structure,
            output.representation.shape_feature,
        ],
        dim=-1,
    )
    torch.testing.assert_close(output.representation.fused_feature, expected_fused)
    torch.testing.assert_close(
        output.semantic.aligned_structure_srvf,
        output.temporal.aligned_structure_srvf,
    )
    torch.testing.assert_close(
        output.semantic.trend_embedding, output.representation.trend_embedding
    )
    torch.testing.assert_close(
        output.semantic.structure_embedding,
        output.representation.structure_embedding,
    )
    assert output.representation.aligned_positions.is_floating_point()
    assert model(pixels, valid, positions).shape == (3, 3)


def test_backbone_normalizes_physical_positions_exactly_once() -> None:
    model = _model().eval()
    pixels, valid, _ = _inputs(length=3)
    physical_positions = torch.tensor([0.0, 182.5, 365.0])

    backbone = model.forward_backbone(pixels, valid, physical_positions)

    expected = torch.tensor([0.0, 0.5, 1.0]).expand(3, -1)
    torch.testing.assert_close(backbone.normalized_positions, expected)
    torch.testing.assert_close(
        model._task_positions(physical_positions, backbone), expected
    )


def test_backbone_rejects_nonincreasing_and_out_of_contract_physical_time() -> None:
    model = _model().eval()
    pixels, valid, _ = _inputs(length=3)

    with pytest.raises(ValueError, match="strictly increasing"):
        model.forward_backbone(
            pixels, valid, torch.tensor([0.0, 182.5, 182.5])
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.forward_backbone(
            pixels, valid, torch.tensor([-1.0, 182.5, 365.0])
        )


def test_task_loss_updates_time_encoder_and_shared_ltae_but_not_warp() -> None:
    model = _model()
    inputs = _inputs()
    _initialize_temporal_source_state(model, inputs)

    output = model.forward_details(*inputs)
    output.representation.logits.square().mean().backward()

    ltae = model.representation.component_ltae
    for module in (
        ltae.shared_time_encoder,
        ltae.shared_input_projection,
        ltae.attention_heads,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum() > 0
            for gradient in gradients
        )
    assert all(parameter.grad is None for parameter in model.geometry_parameters())


def test_residual_is_not_an_independent_task_branch() -> None:
    model = _model().eval()
    pixels, valid, positions = _inputs()
    backbone = model.forward_backbone(pixels, valid, positions)
    changed = StructureBackboneOutput(
        tokens=backbone.tokens,
        time_mask=backbone.time_mask,
        normalized_positions=backbone.normalized_positions,
        decomposition=replace(
            backbone.decomposition,
            residual=backbone.decomposition.residual + 1000,
        ),
    )
    original_output = model.forward_from_backbone(backbone, positions)
    changed_output = model.forward_from_backbone(changed, positions)
    torch.testing.assert_close(
        original_output.representation.logits,
        changed_output.representation.logits,
    )


def test_target_shape_da_detaches_coordinates_but_updates_shape_encoder() -> None:
    model = _model()
    inputs = _inputs()
    _initialize_temporal_source_state(model, inputs)
    output = model.forward_details(*inputs)
    coordinates = output.temporal.coordinates.shape_coordinates
    coordinates.retain_grad()
    feature = model.forward_target_shape_feature_da(output)
    feature.sum().backward()
    assert feature.shape == (3, 4)
    assert coordinates.grad is None
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.temporal_features.shape_encoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.geometry_parameters())
    assert torch.equal(
        feature[~output.temporal.coordinates.shape_valid],
        torch.zeros_like(feature[~output.temporal.coordinates.shape_valid]),
    )


def test_target_shape_teacher_is_deterministic_no_grad_and_used_for_gating() -> None:
    model = _model().train()
    inputs = _inputs()
    _initialize_temporal_source_state(model, inputs)
    source = model.forward_details(*inputs)
    target = model.forward_details(*inputs)

    torch.manual_seed(1)
    first = model.forward_target_shape_teacher_feature(target)
    torch.manual_seed(999)
    second = model.forward_target_shape_teacher_feature(target)
    torch.testing.assert_close(first, second)
    assert not first.requires_grad
    assert model.training

    captured = {}
    def capture_target(_module, args):
        captured["target"] = args[2]

    handle = model.prototype_alignment.register_forward_pre_hook(capture_target)
    student = model.forward_target_shape_feature_da(target)
    model.prototype_losses(source, torch.tensor([0, 1, 2]), target, student, first)
    handle.remove()
    torch.testing.assert_close(captured["target"].shape_feature, first)
    torch.testing.assert_close(
        target.semantic.shape_feature, target.representation.shape_feature
    )


def test_alignment_uses_final_fused_feature() -> None:
    model = _model()
    source = model.forward_details(*_inputs(length=5))
    target = model.forward_details(*_inputs(length=7))
    captured = {}

    def capture(_module, args):
        captured["source"], captured["target"] = args

    handle = model.alignment.register_forward_pre_hook(capture)
    try:
        result = model.align(source, target)
    finally:
        handle.remove()
    assert model.alignment.feature_dim == model.representation.fused_dim == 12
    assert captured["source"] is source.representation.fused_feature
    assert captured["target"] is target.representation.fused_feature
    assert torch.isfinite(result.loss)


def test_geometry_uses_core_only_and_detaches_backbones() -> None:
    model = _model()
    source_inputs = _inputs(length=5)
    target_inputs = _inputs(length=7)
    source_backbone = model.forward_backbone(*source_inputs)
    target_backbone = model.forward_backbone(*target_inputs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("task encoder must not run during geometry forward")

    model.representation.component_ltae.forward = forbidden
    model.temporal_features.shape_encoder.forward = forbidden
    geometry = model.forward_geometry_from_backbones(
        source_backbone,
        source_inputs[2],
        target_backbone,
        target_inputs[2],
    )
    assert torch.isfinite(geometry.total_loss)
    geometry.total_loss.backward()
    assert any(parameter.grad is not None for parameter in model.geometry_parameters())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_parameter_partition_is_disjoint_exhaustive_and_excludes_state_modules() -> None:
    model = _model()
    geometry = tuple(model.geometry_parameters())
    task = tuple(model.task_parameters())
    assert geometry
    assert {id(parameter) for parameter in geometry}.isdisjoint(
        {id(parameter) for parameter in task}
    )
    assert {id(parameter) for parameter in (*geometry, *task)} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert not tuple(model.prototype_alignment.parameters())
    assert not tuple(model.geometry_objective.parameters())


def test_source_state_update_updates_temporal_and_semantic_state_once() -> None:
    model = _model()
    inputs = _inputs()
    labels = torch.tensor([0, 1, 2])
    _initialize_temporal_source_state(model, inputs)
    output = model.forward_details(*inputs)
    trend_before = model.temporal_features.core.trend_template.num_updates.clone()
    structure_before = (
        model.temporal_features.core.structure_diagnostic_template.num_updates.clone()
    )
    model.update_source_state_from_output(output, inputs[2], labels)
    assert model.prototype_alignment.q_update_count.sum() == 3
    assert model.prototype_alignment.z_update_count.sum() == 3
    assert model.prototype_alignment.trend_update_count.sum() == 3
    assert model.prototype_alignment.structure_update_count.sum() == 3
    assert model.temporal_features.core.trend_template.num_updates > trend_before
    assert (
        model.temporal_features.core.structure_diagnostic_template.num_updates
        > structure_before
    )
    assert not hasattr(model, "update_target_state")
    state_keys = model.state_dict()
    assert "prototype_alignment.q_prototype" in state_keys
    assert "prototype_alignment.q_distance_index" in state_keys
    assert "prototype_alignment.q_radius_inner" in state_keys


def test_convenience_state_update_requires_labels_and_with_extra_reaches_pse() -> None:
    model = _model(with_extra=True)
    pixels, valid, positions = _inputs()
    labels = torch.tensor([0, 1, 2])
    extra = torch.randn(3, 4)
    output = model.forward_details(pixels, valid, positions, extra)
    changed_extra = extra.clone()
    changed_extra[0, 0] += 2
    changed = model.forward_details(pixels, valid, positions, changed_extra)
    assert not torch.allclose(output.backbone.tokens, changed.backbone.tokens)
    model.update_source_state(pixels, valid, positions, labels, extra)


@pytest.mark.parametrize(
    ("option_name", "conflict"),
    [
        ("temporal_options", {"feature_dim": 9}),
        ("representation_options", {"shape_dim": 9}),
        ("prototype_options", {"raw_dim": 9}),
    ],
)
def test_fixed_dimensions_cannot_be_overridden(option_name, conflict) -> None:
    with pytest.raises(ValueError, match="conflicts"):
        _model(**{option_name: conflict})
