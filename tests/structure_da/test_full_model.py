from __future__ import annotations

import torch
import importlib.util

import methods.structure_da as structure_da

from methods.structure_da.backbone import StructureBackbone
from methods.structure_da.full_model import (
    StructureAwareDomainAdaptationModel,
    StructureAwareForwardOutput,
)
from methods.structure_da.temporal_module import SharedTemporalStructureOperator
from models.ltae import ComponentAwareSharedLTAE


def _model(dtype: torch.dtype = torch.float32, **overrides):
    options = dict(
        num_classes=3,
        input_dim=2,
        mlp1=(2, 4, 4),
        mlp2=(8, 4),
        structure_dim=3,
        time_scale=366.0,
        temporal_options={
            "num_basis": 4,
            "canonical_grid_size": 5,
            "roughness_grid_size": 64,
            "min_mean_support": 0.0,
            "min_dynamic_energy": 0.0,
            "min_template_mean_support": 0.0,
            "warp_hidden_dim": 6,
            "warp_kernel_size": 3,
            "num_shape_basis": 3,
            "num_phase_basis": 2,
            "attribute_projection_dim": 2,
            "coordinate_hidden_dim": 6,
            "dropout": 0.0,
        },
        representation_options={
            "n_head": 1,
            "d_k": 2,
            "d_model": 8,
            "ltae_mlp": (8, 4),
            "dropout": 0.0,
            "max_position": 366,
            "max_temporal_shift": 0,
            "classifier_hidden": (4,),
            "quality_domain_hidden_dim": 5,
        },
        alignment_hidden_dim=5,
        grl_max_iters=10,
    )
    options.update(overrides)
    return StructureAwareDomainAdaptationModel(**options).to(dtype=dtype)


def _inputs(batch: int = 2, length: int = 5, dtype=torch.float32):
    torch.manual_seed(901 + length)
    pixels = torch.randn(batch, length, 2, 4, dtype=dtype)
    valid = torch.ones(batch, length, 4, dtype=torch.bool)
    positions = torch.linspace(0, 300, length, dtype=dtype).round().long()
    return pixels, valid, positions


def test_model_contains_backbone_temporal_and_shared_component_ltae() -> None:
    model = _model()
    assert isinstance(model.backbone, StructureBackbone)
    assert isinstance(model.temporal_operator, SharedTemporalStructureOperator)
    assert sum(isinstance(m, ComponentAwareSharedLTAE) for m in model.modules()) == 1


def test_removed_public_api_and_module_are_absent() -> None:
    removed_module = "methods.structure_da." + "channel" + "_module"
    assert importlib.util.find_spec(removed_module) is None
    for name in ("Shared" + "ChannelStructureOperator", "Channel" + "StructurePairOutput"):
        assert not hasattr(structure_da, name)


def test_forward_interfaces_have_no_removed_branch() -> None:
    model = _model()
    pixels, valid, positions = _inputs()
    backbone = model.forward_backbone(pixels, valid, positions)
    assert backbone.tokens.shape == (2, 5, 4)
    output = model.forward_from_backbone(backbone, positions)
    assert isinstance(output, StructureAwareForwardOutput)
    assert set(output.__dataclass_fields__) == {"backbone", "temporal", "representation"}
    assert output.representation.fused_feature.shape == (2, 7)
    assert model(*_inputs()).shape == (2, 3)


def test_source_state_update_and_parameter_partition() -> None:
    model = _model()
    before = model.temporal_operator.extractor.registration.source_template.num_updates.clone()
    model.update_source_state(*_inputs())
    after = model.temporal_operator.extractor.registration.source_template.num_updates
    assert after > before
    geometry = tuple(model.geometry_parameters())
    task = tuple(model.task_parameters())
    assert {id(p) for p in geometry}.isdisjoint({id(p) for p in task})
    assert {id(p) for p in (*geometry, *task)} == {
        id(p) for p in model.parameters() if p.requires_grad
    }


def test_alignment_uses_declared_fused_dimension() -> None:
    model = _model()
    source = model.forward_details(*_inputs(length=5))
    target = model.forward_details(*_inputs(length=7))
    aligned = model.align(source, target)
    assert model.alignment.feature_dim == model.representation.fused_dim == 7
    assert torch.isfinite(aligned.loss)


def test_with_extra_reaches_original_pse() -> None:
    model = _model(with_extra=True)
    pixels, valid, positions = _inputs()
    extra = torch.randn(2, 4)
    output = model.forward_details(pixels, valid, positions, extra)
    changed_extra = extra.clone()
    changed_extra[0, 0] += 2
    changed = model.forward_details(pixels, valid, positions, changed_extra)
    assert not torch.allclose(output.backbone.tokens, changed.backbone.tokens)
