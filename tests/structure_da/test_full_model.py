from __future__ import annotations

import pytest
import torch

from methods.structure_da.backbone import StructureBackbone
from methods.structure_da.channel_module import (
    MultiScaleChannelRelationStructure,
    SharedChannelStructureOperator,
)
from methods.structure_da.full_model import (
    StructureAwareDomainAdaptationModel,
    StructureAwareForwardOutput,
)
from methods.structure_da.temporal_module import (
    SharedTemporalStructureOperator,
    TemporalStructureExtractor,
)
from models.ltae import LTAE


def _model(dtype: torch.dtype = torch.float32, **overrides):
    options = dict(
        num_classes=3,
        num_channels=2,
        channel_feature_dim=2,
        pixel_hidden_dim=3,
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
        channel_options={
            "lag_centers": (-0.1, 0.0, 0.1),
            "lag_widths": (0.08, 0.08, 0.08),
            "velocity_bandwidth": 0.15,
            "edge_hidden_dim": 4,
            "min_effective_pairs": 1.0,
            "min_relation_mass": 0.0,
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


def _state_counts(model):
    temporal = model.temporal_operator.extractor.registration
    channel = model.channel_operator.extractor
    return (
        int(temporal.srvf_extractor.functional_lift.standardizer.num_updates.item()),
        int(temporal.srvf_extractor.support_scale.num_updates.item()),
        int(temporal.source_template.num_updates.item()),
        int(channel.attribute_standardizer.num_updates.item()),
        int(channel.energy_scale.num_updates.item()),
    )


def test_model_contains_exactly_one_shared_module_of_each_kind() -> None:
    model = _model()
    assert isinstance(model.backbone, StructureBackbone)
    assert isinstance(model.temporal_operator, SharedTemporalStructureOperator)
    assert isinstance(model.channel_operator, SharedChannelStructureOperator)
    assert sum(isinstance(m, StructureBackbone) for m in model.modules()) == 1
    assert sum(isinstance(m, TemporalStructureExtractor) for m in model.modules()) == 1
    assert sum(isinstance(m, MultiScaleChannelRelationStructure) for m in model.modules()) == 1
    assert sum(isinstance(m, LTAE) for m in model.modules()) == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_details_shapes_extra_compatibility_and_read_only_state(dtype) -> None:
    model = _model(dtype).eval()
    pixels, valid, positions = _inputs(dtype=dtype)
    before = _state_counts(model)
    first = model.forward_details(pixels, valid, positions, torch.randn(2, 4, dtype=dtype))
    second = model.forward_details(pixels, valid, positions, torch.randn(2, 9, dtype=dtype))

    assert isinstance(first, StructureAwareForwardOutput)
    assert first.representation.logits.shape == (2, 3)
    assert first.representation.fused_feature.shape == (2, 10)
    torch.testing.assert_close(first.representation.logits, second.representation.logits)
    torch.testing.assert_close(model(pixels, valid, positions, None), first.representation.logits)
    assert _state_counts(model) == before == (0, 0, 0, 0, 0)
    for coefficient in (
        first.representation.quality.alpha_trend,
        first.representation.quality.alpha_dynamics,
        first.representation.quality.alpha_residual,
        first.representation.quality.beta_trend_temporal,
        first.representation.quality.beta_dynamics_temporal,
        first.representation.quality.beta_trend_channel,
        first.representation.quality.beta_dynamics_channel,
    ):
        assert coefficient.min() >= 0 and coefficient.max() <= 1


def test_channel_mask_broadcasts_and_is_combined_with_time_mask() -> None:
    model = _model().eval()
    pixels, valid, positions = _inputs()
    time_mask = torch.tensor([[1, 1, 0, 1, 1], [1, 0, 1, 1, 1]], dtype=torch.bool)
    channel_mask = torch.tensor(
        [[1, 0], [1, 1], [1, 1], [0, 1], [1, 1]], dtype=torch.bool
    )
    output = model.forward_details(
        pixels, valid, positions, time_mask=time_mask, channel_mask=channel_mask
    )
    assert output.channel.trend.valid.shape == (2,)
    with pytest.raises(ValueError):
        model.forward_details(pixels, valid, positions, channel_mask=torch.ones(5, 3))


def test_source_state_update_is_explicit_and_combined_once() -> None:
    model = _model()
    pixels, valid, positions = _inputs()
    model.update_source_state(pixels, valid, positions)
    assert _state_counts(model) == (1, 1, 1, 1, 1)


def test_source_geometry_reuses_backbone_and_routes_only_geometry_to_warp() -> None:
    model = _model()
    pixels, valid, positions = _inputs()
    model.update_source_state(pixels, valid, positions)
    calls = []
    handle = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    output = model.forward_details(pixels, valid, positions)
    assert len(calls) == 1
    geometry = model.forward_source_geometry(output, positions)
    assert len(calls) == 1
    handle.remove()

    warp_parameters = list(model.temporal_operator.extractor.warp_parameters())
    model.zero_grad(set_to_none=True)
    output.representation.logits.sum().backward(retain_graph=True)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in warp_parameters)
    model.zero_grad(set_to_none=True)
    geometry.total_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in warp_parameters)


def test_alignment_routes_gradients_through_both_fused_features() -> None:
    model = _model()
    model.alignment.grl.iteration.fill_(3)
    source = model.forward_details(*_inputs(length=5))
    target = model.forward_details(*_inputs(length=6))
    source.representation.fused_feature.retain_grad()
    target.representation.fused_feature.retain_grad()
    aligned = model.align(source, target)
    aligned.loss.backward()
    assert source.representation.fused_feature.grad.abs().sum() > 0
    assert target.representation.fused_feature.grad.abs().sum() > 0
    assert model.alignment.grl.iteration.item() == 4


def test_invalid_constructor_and_masks_are_rejected() -> None:
    with pytest.raises(ValueError):
        StructureAwareDomainAdaptationModel(num_classes=1)
    model = _model()
    pixels, valid, positions = _inputs()
    with pytest.raises(ValueError):
        model.forward_details(pixels, valid, positions, channel_mask=torch.ones(2, 5, 2, 1))


def test_tau_constructor_values_reach_stkd_initial_scales() -> None:
    baseline = _model()
    changed = _model(
        tau_fast_init=0.08,
        tau_slow_init=0.31,
        tau_min=1e-3,
        delta_tau_min=2e-3,
    )

    assert baseline.backbone.decomposition.tau_fast.item() == pytest.approx(0.05)
    assert baseline.backbone.decomposition.tau_slow.item() == pytest.approx(0.20)
    assert changed.backbone.decomposition.tau_fast.item() == pytest.approx(0.08)
    assert changed.backbone.decomposition.tau_slow.item() == pytest.approx(0.31)
    assert changed.backbone.decomposition.tau_fast.item() != pytest.approx(
        baseline.backbone.decomposition.tau_fast.item()
    )
