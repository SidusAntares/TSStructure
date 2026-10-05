from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from methods.structure_da.prototype_losses import (
    compose_structure_v2clean_da_loss,
    masked_pseudo_classification_loss,
    shape_health_snapshot,
)
from models.stclassifier import PseStructureProtoLTae
from models.structure_da.discriminative_structure import (
    DiscriminativeStructureBranch,
    StateShapeTokenGenerator,
)


def _model(representation="state_org", injection="direct_response_query"):
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=128, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, shape_resample_length=7,
        fourier_num_modes=5, shape_representation=representation,
        shape_injection=injection, structure_shift_mode="none", dropout=0.,
    )


def _batch(batch=3):
    pixels = torch.randn(batch, 10, 3, 4)
    mask = torch.ones(batch, 10, 4)
    positions = torch.arange(10).repeat(batch, 1) * 30
    return pixels, mask, positions, torch.zeros(batch, 4)


def test_state_token_is_8_by_128_without_interpolation(monkeypatch):
    generator = StateShapeTokenGenerator(
        3, shape_dim=128, period_days=365., grid_points=64,
    ).eval()
    windows = torch.randn(2, 8, 24, 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("state_org must not interpolate windows")

    monkeypatch.setattr(F, "interpolate", forbidden)
    tokens = generator(windows)
    assert tokens.shape == (2, 8, 128)
    assert not hasattr(generator, "raw_encoder")
    assert not hasattr(generator, "diff_encoder")
    assert not hasattr(generator, "mean_encoder")
    assert not hasattr(generator, "std_encoder")
    assert not hasattr(generator, "fusion")


def test_state_token_is_invariant_to_positive_scale_and_offset_in_eval():
    torch.manual_seed(401)
    generator = StateShapeTokenGenerator(
        3, shape_dim=128, period_days=365., grid_points=64,
    ).eval()
    windows = torch.randn(2, 8, 24, 3)
    scale = torch.tensor([.5, 2., 3.])[None, None, None]
    offset = torch.tensor([-4., 7., 11.])[None, None, None]
    with torch.no_grad():
        original = generator(windows)
        transformed = generator(windows * scale + offset)
    torch.testing.assert_close(original, transformed, atol=2e-5, rtol=2e-5)


def test_state_org_response_dimensions_distribution_and_contents():
    torch.manual_seed(403)
    branch = DiscriminativeStructureBranch(
        3, shape_dim=128, shapelet_count=16, shape_representation="state_org",
    ).eval()
    similarity = torch.randn(2, 8, 16)
    result = branch.compose_state_org_response(similarity)
    assert result["presence"].shape == (2, 16)
    assert result["state_distribution"].shape == (2, 8, 16)
    assert result["organization"].shape == (2, 32)
    assert result["shapelet_response"].shape == (2, 48)
    torch.testing.assert_close(
        result["state_distribution"].sum(-1), torch.ones(2, 8),
    )
    torch.testing.assert_close(
        result["shapelet_response"],
        torch.cat((result["presence"], result["organization"]), dim=-1),
    )


def test_presence_and_organization_are_circular_window_roll_invariant():
    torch.manual_seed(409)
    branch = DiscriminativeStructureBranch(
        3, shape_dim=128, shapelet_count=16, shape_representation="state_org",
    ).eval()
    similarity = torch.randn(3, 8, 16)
    base = branch.compose_state_org_response(similarity)
    shifted = branch.compose_state_org_response(torch.roll(similarity, 3, dims=1))
    torch.testing.assert_close(base["presence"], shifted["presence"])
    torch.testing.assert_close(
        base["organization"], shifted["organization"], atol=2e-6, rtol=2e-6,
    )


def test_state_org_model_uses_normalized_48d_direct_query_and_zero_projection():
    torch.manual_seed(419)
    model = _model().eval()
    assert model.shape_evidence_dim == 48
    assert model.shape_classifier.in_features == 48
    assert model.structure_branch.response_to_query is None
    projection = model.temporal_encoder.attention_heads.external_query_projection
    assert projection.in_features == 48
    assert projection.out_features == 8
    assert torch.count_nonzero(projection.weight) == 0

    pixels, mask, positions, extra = _batch()
    with torch.no_grad():
        spatial = model.spatial_encoder(pixels, mask, extra)
        structure = model.prepare_structure(spatial, positions)
        evidence = model._shape_evidence(structure)
        injected = model._encode_instance(spatial, positions, structure)
        baseline = model.temporal_encoder(spatial, positions, external_query=None)
        output = model._output_from_prepared_structure(spatial, positions, structure)
    torch.testing.assert_close(
        evidence, model.shape_response_norm(structure["shapelet_response"]),
    )
    torch.testing.assert_close(injected, baseline)
    torch.testing.assert_close(output["shape_logits"], model.shape_classifier(evidence))
    assert "shapelet_concentration" not in output
    assert "shape_stats_feature" not in output
    assert "shapelet_phase_moments" not in output


def test_target_main_loss_reaches_state_org_after_zero_init_query_projection_learns():
    torch.manual_seed(421)
    model = _model().train()
    projection = model.temporal_encoder.attention_heads.external_query_projection
    labels = torch.tensor([0, 1, 2])
    trusted = torch.tensor([True, False, True])
    output = model(*_batch(), return_dict=True)
    masked_pseudo_classification_loss(
        output["logits"], labels, trusted, F.cross_entropy,
    ).backward()
    assert projection.weight.grad is not None
    assert projection.weight.grad.abs().sum() > 0
    with torch.no_grad():
        projection.weight.add_(-.1 * projection.weight.grad)
    model.zero_grad(set_to_none=True)

    output = model(*_batch(), return_dict=True)
    masked_pseudo_classification_loss(
        output["logits"], labels, trusted, F.cross_entropy,
    ).backward()
    organization_conv = model.structure_branch.organization_encoder[0]
    state_projection = model.structure_branch.token_generator.state_encoder.input_projection
    assert organization_conv.weight.grad is not None
    assert organization_conv.weight.grad.abs().sum() > 0
    assert model.structure_branch.shapelet_dictionary.anchors.grad is not None
    assert model.structure_branch.shapelet_dictionary.anchors.grad.abs().sum() > 0
    assert state_projection.weight.grad is not None
    assert state_projection.weight.grad.abs().sum() > 0
    assert projection.weight.grad is not None
    assert projection.weight.grad.abs().sum() > 0


def test_state_org_uda_loss_contains_only_main_shape_and_diversity_terms():
    classification = torch.tensor(2.)
    pseudo = torch.tensor(3.)
    source_shape = torch.tensor(5.)
    diversity = torch.tensor(7.)
    disabled_alignment = torch.tensor(11.)
    disabled_equivariance = torch.tensor(13.)
    loss = compose_structure_v2clean_da_loss(
        classification, pseudo, source_shape, diversity, disabled_alignment,
        ramp=.8, trade_off=2., shape_weight=.1, diversity_weight=.01,
        shape_align_weight=0., shape_equivariance=disabled_equivariance,
        shape_equivariance_weight=0.,
    )
    torch.testing.assert_close(
        loss, classification + 2. * pseudo + .1 * source_shape + .01 * diversity,
    )


def test_state_org_health_and_legacy_current_phase_forward():
    state = _model().eval()
    with torch.no_grad():
        state_output = state(*_batch(2), return_dict=True)
    health = shape_health_snapshot(state, state_output)
    assert "shape_state_encoder_param_norm" in health
    assert "shape_concentration_mean" not in health
    for representation in ("current", "phase_moment"):
        legacy = _model(representation, "current_query").eval()
        with torch.no_grad():
            output = legacy(*_batch(2), return_dict=True)
        assert output["logits"].shape == (2, 3)


def test_state_org_rejects_other_injections_and_launcher_disables_alignment():
    for injection in ("current_query", "direct_query", "late_fusion"):
        with pytest.raises(ValueError, match="state_org"):
            _model("state_org", injection)
    launcher = Path(
        "scripts/run_structure_state_org_4tasks_4gpu_seed1.sh"
    ).read_text(encoding="utf-8")
    assert launcher.count("--shape-representation state_org") == 2
    assert launcher.count("--shape-injection direct_response_query") == 2
    assert launcher.count("--shape-resample-length 24") == 2
    assert "--shape-da-mode batch_align" in launcher
    assert "--shape-alignment-view none" in launcher
    assert "--shape-align-weight 0" in launcher
    assert "--shape-equivariance-weight 0" in launcher
    assert "--adaptive-pseudo-selection false" in launcher
    assert "--oracle-pseudo-labels false" in launcher
    assert launcher.count('run_task "$GPU') == 4
