from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import methods.structure_da as structure_da
from methods.structure_da.backbone import StructureBackbone
from methods.structure_da.channel_module import (
    ChannelStructurePairOutput,
    MultiScaleChannelRelationStructure,
    SharedChannelStructureOperator,
)
from methods.structure_da.decomposition import DecompositionOutput
from methods.structure_da.representation import (
    PairedStructureFeatures,
    QualityAwareClassifierOutput,
    QualityAwareComponentClassifier,
)
from methods.structure_da.temporal_module import (
    SharedTemporalStructureOperator,
    TemporalStructureExtractor,
    TemporalStructurePairOutput,
)
from models.decoder import get_decoder
from models.layers import LinearLayer
from models.ltae import LTAE


def _classifier(dtype: torch.dtype = torch.float32, **overrides):
    kwargs = dict(
        num_channels=2,
        channel_feature_dim=2,
        structure_dim=3,
        num_classes=3,
        n_head=2,
        d_k=2,
        d_model=8,
        ltae_mlp=(8, 4),
        dropout=0.0,
        positional_period=100,
        max_position=365,
        max_temporal_shift=10,
        classifier_hidden=(6,),
        quality_domain_hidden_dim=5,
    )
    kwargs.update(overrides)
    torch.manual_seed(501)
    return QualityAwareComponentClassifier(**kwargs).to(dtype=dtype)


def _inputs(dtype: torch.dtype = torch.float32):
    torch.manual_seed(502)
    shape = (3, 5, 2, 2)
    decomposition = DecompositionOutput(
        trend=torch.randn(*shape, dtype=dtype),
        dynamics=torch.randn(*shape, dtype=dtype),
        residual=torch.randn(*shape, dtype=dtype),
    )
    temporal = PairedStructureFeatures(
        trend=torch.randn(3, 3, dtype=dtype),
        dynamics=torch.randn(3, 3, dtype=dtype),
        trend_valid=torch.tensor([True, True, False]),
        dynamics_valid=torch.tensor([True, False, False]),
    )
    channel = PairedStructureFeatures(
        trend=torch.randn(3, 3, dtype=dtype),
        dynamics=torch.randn(3, 3, dtype=dtype),
        trend_valid=torch.tensor([True, True, False]),
        dynamics_valid=torch.tensor([False, True, False]),
    )
    positions = torch.tensor(
        [[0, 30, 80, 170, 300], [1, 31, 81, 171, 301], [2, 32, 82, 172, 302]]
    )
    mask = torch.tensor(
        [
            [True, True, False, True, True],
            [True, False, True, True, False],
            [False, False, False, False, False],
        ]
    )
    return decomposition, temporal, channel, positions, mask


def test_quality_representation_public_symbols_are_exported() -> None:
    expected = {
        "QualityScoreOutput",
        "QualityScorer",
        "StructuralQualityBundle",
        "ComponentQualityBundle",
        "HierarchicalQualityOutput",
        "HierarchicalQualityFusion",
        "QualityLossOutput",
        "HierarchicalQualityObjective",
        "PairedStructureFeatures",
        "QualityAwareClassifierOutput",
        "QualityAwareComponentClassifier",
    }

    assert expected <= set(structure_da.__all__)
    assert all(hasattr(structure_da, name) for name in expected)


def test_temporal_and_channel_pair_adapters_read_exact_public_features() -> None:
    temporal_feature = torch.randn(2, 3)
    channel_feature = torch.randn(2, 3)
    valid = torch.tensor([True, False])
    temporal_output = TemporalStructurePairOutput(
        trend=SimpleNamespace(encoded=SimpleNamespace(feature=temporal_feature, valid=valid)),
        dynamics=SimpleNamespace(encoded=SimpleNamespace(feature=temporal_feature + 1, valid=~valid)),
    )
    channel_output = ChannelStructurePairOutput(
        trend=SimpleNamespace(feature=channel_feature, valid=valid),
        dynamics=SimpleNamespace(feature=channel_feature + 1, valid=~valid),
    )

    temporal = PairedStructureFeatures.from_temporal(temporal_output)
    channel = PairedStructureFeatures.from_channel(channel_output)

    assert temporal.trend is temporal_feature
    assert temporal.trend_valid is valid
    assert channel.trend is channel_feature
    assert channel.trend_valid is valid


def test_classifier_contains_exactly_one_shared_ltae() -> None:
    model = _classifier()

    ltae_modules = [module for module in model.modules() if isinstance(module, LTAE)]
    assert ltae_modules == [model.shared_ltae]
    assert not hasattr(model, "trend_ltae")
    assert not hasattr(model, "dynamics_ltae")
    assert not hasattr(model, "residual_ltae")


def test_shared_ltae_is_called_three_times_with_raw_c_times_p_features() -> None:
    model = _classifier().eval()
    decomposition, temporal, channel, positions, mask = _inputs()
    captured = []
    hook = model.shared_ltae.register_forward_pre_hook(
        lambda _module, args, _kwargs: captured.append(args[0].shape),
        with_kwargs=True,
    )

    model(decomposition, temporal, channel, positions, mask)
    hook.remove()

    assert captured == [(3, 5, 4), (3, 5, 4), (3, 5, 4)]


def test_representation_output_shapes_are_complete() -> None:
    model = _classifier().eval()
    decomposition, temporal, channel, positions, mask = _inputs()

    output = model(decomposition, temporal, channel, positions, mask)

    assert isinstance(output, QualityAwareClassifierOutput)
    assert output.trend_embedding.shape == (3, 4)
    assert output.dynamics_embedding.shape == (3, 4)
    assert output.residual_embedding.shape == (3, 4)
    assert output.quality.raw_fusion.shape == (3, 4)
    assert output.quality.temporal_fusion.shape == (3, 3)
    assert output.quality.channel_fusion.shape == (3, 3)
    assert output.fused_feature.shape == (3, 10)
    assert output.logits.shape == (3, 3)
    assert output.component_valid.tolist() == [True, True, False]
    assert output.time_mask.shape == (3, 5)
    assert output.ltae_positions.dtype == torch.long


def test_classifier_uses_timematch_decoder_style_and_correct_input_dim() -> None:
    model = _classifier()
    reference = get_decoder([10, 6], 3)

    assert [type(module) for module in model.classifier] == [
        type(module) for module in reference
    ]
    assert isinstance(model.classifier[0], LinearLayer)
    assert model.classifier[0].linear.in_features == 10
    assert isinstance(model.classifier[-1], nn.Linear)
    assert model.classifier[-1].out_features == 3


def test_vector_time_mask_matches_batched_mask() -> None:
    model = _classifier().eval()
    decomposition, temporal, channel, positions, _ = _inputs()
    vector = torch.tensor([True, False, True, True, False])

    vector_output = model(decomposition, temporal, channel, positions, vector)
    batch_output = model(
        decomposition, temporal, channel, positions, vector.expand(3, -1)
    )

    torch.testing.assert_close(vector_output.logits, batch_output.logits)
    torch.testing.assert_close(vector_output.fused_feature, batch_output.fused_feature)


def test_none_time_mask_is_all_valid() -> None:
    model = _classifier().eval()
    decomposition, temporal, channel, positions, _ = _inputs()

    implicit = model(decomposition, temporal, channel, positions)
    explicit = model(
        decomposition,
        temporal,
        channel,
        positions,
        torch.ones(3, 5, dtype=torch.bool),
    )

    torch.testing.assert_close(implicit.logits, explicit.logits)
    assert implicit.component_valid.all()


def test_masked_raw_tokens_do_not_change_any_embedding() -> None:
    model = _classifier().eval()
    decomposition, temporal, channel, positions, mask = _inputs()
    changed = DecompositionOutput(
        trend=decomposition.trend.clone(),
        dynamics=decomposition.dynamics.clone(),
        residual=decomposition.residual.clone(),
    )
    for component in (changed.trend, changed.dynamics, changed.residual):
        component[~mask] = 1e20
        component[2] = float("nan")

    expected = model(decomposition, temporal, channel, positions, mask)
    actual = model(changed, temporal, channel, positions, mask)

    torch.testing.assert_close(actual.trend_embedding, expected.trend_embedding)
    torch.testing.assert_close(actual.dynamics_embedding, expected.dynamics_embedding)
    torch.testing.assert_close(actual.residual_embedding, expected.residual_embedding)


def test_masked_positions_are_replaced_before_integer_and_range_validation() -> None:
    model = _classifier().eval()
    decomposition, temporal, channel, positions, mask = _inputs()
    changed = positions.double()
    changed[~mask] = float("nan")
    changed[0, 2] = 1e9 + 0.5

    output = model(decomposition, temporal, channel, changed, mask)

    assert torch.count_nonzero(output.ltae_positions[~mask]) == 0
    assert torch.isfinite(output.fused_feature).all()


def test_all_invalid_sample_has_zero_embeddings_alpha_and_fused_feature() -> None:
    model = _classifier().eval()
    decomposition, temporal, channel, positions, mask = _inputs()

    output = model(decomposition, temporal, channel, positions, mask)

    for embedding in (
        output.trend_embedding,
        output.dynamics_embedding,
        output.residual_embedding,
    ):
        assert torch.count_nonzero(embedding[2]) == 0
    assert output.quality.alpha_trend[2].item() == 0
    assert output.quality.alpha_dynamics[2].item() == 0
    assert output.quality.alpha_residual[2].item() == 0
    assert torch.count_nonzero(output.fused_feature[2]) == 0


def test_task_loss_updates_ltae_classifier_quality_and_all_feature_inputs() -> None:
    model = _classifier()
    decomposition, temporal, channel, positions, mask = _inputs()
    components = [
        decomposition.trend.clone().requires_grad_(),
        decomposition.dynamics.clone().requires_grad_(),
        decomposition.residual.clone().requires_grad_(),
    ]
    decomposition = DecompositionOutput(*components)
    structure_tensors = [
        temporal.trend.clone().requires_grad_(),
        temporal.dynamics.clone().requires_grad_(),
        channel.trend.clone().requires_grad_(),
        channel.dynamics.clone().requires_grad_(),
    ]
    temporal = PairedStructureFeatures(
        structure_tensors[0], structure_tensors[1], temporal.trend_valid, temporal.dynamics_valid
    )
    channel = PairedStructureFeatures(
        structure_tensors[2], structure_tensors[3], channel.trend_valid, channel.dynamics_valid
    )

    output = model(decomposition, temporal, channel, positions, mask)
    F.cross_entropy(output.logits[:2], torch.tensor([0, 1])).backward()

    for tensor in (*components, *structure_tensors):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    for group in (model.shared_ltae, model.quality_fusion, model.classifier):
        assert all(
            parameter.grad is not None
            for parameter in group.parameters()
            if parameter.requires_grad
        )


def test_residual_branch_has_no_structure_interface_or_slot_values() -> None:
    model = _classifier().eval()
    decomposition, temporal, channel, positions, mask = _inputs()

    output = model(decomposition, temporal, channel, positions, mask)

    torch.testing.assert_close(
        output.quality.residual_component_input[:, 4:],
        torch.zeros(3, 6),
        atol=0,
        rtol=0,
    )
    assert not hasattr(output, "residual_temporal")
    assert not hasattr(output, "residual_channel")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_representation_preserves_cpu_dtype(dtype: torch.dtype) -> None:
    model = _classifier(dtype).eval()
    decomposition, temporal, channel, positions, mask = _inputs(dtype)

    output = model(decomposition, temporal, channel, positions, mask)

    assert output.logits.dtype == dtype
    assert output.fused_feature.dtype == dtype
    assert output.trend_embedding.dtype == dtype


@pytest.mark.parametrize(
    "positions,mask,match",
    [
        (torch.ones(4), None, "positions"),
        (torch.ones(2, 5), None, "positions"),
        (torch.tensor([0.0, 1.5, 2.0, 3.0, 4.0]), None, "integer"),
        (torch.tensor([-11, 0, 1, 2, 3]), None, "range"),
        (torch.tensor([0, 1, 2, 3, 375]), None, "range"),
        (torch.arange(5), torch.ones(4), "time_mask"),
        (torch.arange(5), torch.tensor([1, 1, 2, 1, 1]), "0/1"),
    ],
)
def test_invalid_positions_or_time_masks_raise_value_error(positions, mask, match) -> None:
    model = _classifier().eval()
    decomposition, temporal, channel, _, _ = _inputs()

    with pytest.raises(ValueError, match=match):
        model(decomposition, temporal, channel, positions, mask)


@pytest.mark.parametrize(
    "factory,match",
    [
        (lambda: _classifier(num_channels=0), "num_channels"),
        (lambda: _classifier(channel_feature_dim=0), "channel_feature_dim"),
        (lambda: _classifier(structure_dim=0), "structure_dim"),
        (lambda: _classifier(num_classes=1), "num_classes"),
        (lambda: _classifier(n_head=0), "n_head"),
        (lambda: _classifier(d_model=7), "d_model"),
        (lambda: _classifier(ltae_mlp=()), "ltae_mlp"),
        (lambda: _classifier(ltae_mlp=(7, 4)), "ltae_mlp"),
    ],
)
def test_invalid_classifier_constructor_arguments_raise_value_error(factory, match) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_invalid_decomposition_shape_or_valid_nonfinite_value_is_rejected() -> None:
    model = _classifier().eval()
    decomposition, temporal, channel, positions, mask = _inputs()
    wrong = DecompositionOutput(
        decomposition.trend[:, :-1], decomposition.dynamics, decomposition.residual
    )
    with pytest.raises(ValueError, match="shape"):
        model(wrong, temporal, channel, positions, mask)

    invalid = decomposition.trend.clone()
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(
            DecompositionOutput(invalid, decomposition.dynamics, decomposition.residual),
            temporal,
            channel,
            positions,
            mask,
        )


def _temporal_extractor() -> TemporalStructureExtractor:
    return TemporalStructureExtractor(
        num_channels=2,
        channel_feature_dim=2,
        num_basis=4,
        canonical_grid_size=5,
        roughness_grid_size=64,
        smoothing_weight=1e-3,
        min_mean_support=0.0,
        min_dynamic_energy=0.0,
        min_template_mean_support=0.0,
        warp_hidden_dim=6,
        warp_kernel_size=3,
        num_shape_basis=3,
        num_phase_basis=2,
        attribute_projection_dim=2,
        coordinate_hidden_dim=6,
        structure_dim=3,
        dropout=0.0,
    )


def test_real_backbone_structure_operators_and_classifier_integrate_end_to_end() -> None:
    torch.manual_seed(503)
    backbone = StructureBackbone(
        num_channels=2, channel_feature_dim=2, pixel_hidden_dim=3
    )
    temporal_extractor = _temporal_extractor()
    temporal_operator = SharedTemporalStructureOperator(temporal_extractor)
    channel_extractor = MultiScaleChannelRelationStructure(
        num_channels=2,
        token_dim=2,
        lag_centers=(-0.1, 0.0, 0.1),
        lag_widths=(0.08, 0.08, 0.08),
        velocity_bandwidth=0.15,
        edge_hidden_dim=4,
        structure_dim=3,
        min_effective_pairs=1.0,
        min_relation_mass=0.0,
        time_scale=366.0,
        dropout=0.0,
    )
    channel_operator = SharedChannelStructureOperator(channel_extractor)
    classifier = _classifier()
    pixels = torch.randn(2, 5, 2, 4)
    valid_pixels = torch.ones(2, 5, 4, dtype=torch.bool)
    positions = torch.tensor([0, 30, 80, 170, 300])
    time_mask = torch.ones(2, 5, dtype=torch.bool)

    backbone_output = backbone(pixels, valid_pixels, positions, time_mask)
    trend = backbone_output.decomposition.trend
    dynamics = backbone_output.decomposition.dynamics
    temporal_operator.update_source_state(trend, dynamics, positions, time_mask)
    temporal_output = temporal_operator(trend, dynamics, positions, time_mask)
    channel_output = channel_operator(trend, dynamics, positions, time_mask=time_mask)
    output = classifier(
        backbone_output.decomposition,
        PairedStructureFeatures.from_temporal(temporal_output),
        PairedStructureFeatures.from_channel(channel_output),
        positions,
        time_mask,
    )
    F.cross_entropy(output.logits, torch.tensor([0, 1])).backward()

    assert output.logits.shape == (2, 3)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in backbone.pixel_set_encoder.parameters())
    assert backbone.decomposition._tau_fast_unconstrained.grad is not None
    assert backbone.decomposition._tau_gap_unconstrained.grad is not None
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in classifier.shared_ltae.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in classifier.quality_fusion.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in classifier.classifier.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in channel_extractor.parameters())
    for parameter in temporal_extractor.registration.warp_estimator.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
