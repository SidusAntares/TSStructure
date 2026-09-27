from pathlib import Path
import argparse
from types import SimpleNamespace

import pytest
import torch

from models.stclassifier import PseStructureProtoLTae
from models.structure_da.discriminative_structure import (
    DiscriminativeStructureBranch,
    sorted_anchor_profile,
)


def _model(representation="current", injection="current_query"):
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation=representation, shape_injection=injection,
    )


def _batch():
    pixels = torch.randn(2, 8, 3, 4)
    mask = torch.ones(2, 8, 4)
    positions = torch.arange(8).repeat(2, 1) * 40
    extra = torch.zeros(2, 4)
    return pixels, mask, positions, extra


def test_sorted_anchor_profile_shape_and_window_permutation_invariance():
    similarity = torch.randn(3, 8, 16)
    profile = sorted_anchor_profile(similarity)
    assert profile.shape == (3, 128)
    permutation = torch.tensor([4, 1, 7, 0, 6, 2, 5, 3])
    torch.testing.assert_close(profile, sorted_anchor_profile(similarity[:, permutation]))


def test_sorted_anchor_profile_preserves_anchor_identity():
    similarity = torch.randn(2, 8, 16)
    permutation = torch.tensor([3, 0, 1, 2] + list(range(4, 16)))
    original = sorted_anchor_profile(similarity).reshape(2, 16, 8)
    permuted = sorted_anchor_profile(similarity[:, :, permutation]).reshape(2, 16, 8)
    torch.testing.assert_close(permuted, original[:, permutation])
    assert not torch.equal(original, permuted)


def test_branch_returns_similarity_and_sorted_profile_for_eight_by_sixteen():
    branch = DiscriminativeStructureBranch(
        6, shape_dim=8, shapelet_count=16,
        window_scales=(24,), window_stride=8, num_modes=5,
    ).eval()
    features = torch.randn(2, 8, 6)
    positions = torch.arange(8).repeat(2, 1) * 40
    with torch.no_grad():
        output = branch(features, positions)
    assert output["shapelet_similarity"].shape == (2, 8, 16)
    assert output["sorted_anchor_profile"].shape == (2, 128)


def test_sorted_direct_query_bypasses_response_mlp_and_uses_zero_projection():
    model = _model("sorted_profile", "direct_query")
    model.structure_branch.response_to_query.forward = lambda _: (_ for _ in ()).throw(
        AssertionError("legacy response_to_query must be bypassed")
    )
    projection = model.temporal_encoder.attention_heads.external_query_projection
    assert projection.in_features == 128
    assert torch.count_nonzero(projection.weight) == 0
    output = model(*_batch(), return_dict=True)
    assert output["shape_logits"].shape == (2, 3)
    assert model.shape_classifier.in_features == 128
    output["logits"].sum().backward()
    assert torch.isfinite(output["logits"]).all()


def test_sorted_late_fusion_omits_external_query_and_zero_initializes_projection(monkeypatch):
    model = _model("sorted_profile", "late_fusion")
    assert model.temporal_encoder.attention_heads.external_query_projection is None
    assert model.late_fusion_projection.in_features == 128
    assert model.late_fusion_projection.out_features == 6
    assert torch.count_nonzero(model.late_fusion_projection.weight) == 0
    calls = []
    original = model.temporal_encoder.forward

    def capture(*args, **kwargs):
        calls.append(kwargs.get("external_query", "missing"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model.temporal_encoder, "forward", capture)
    output = model(*_batch(), return_dict=True)
    assert calls == [None]
    assert model.shape_classifier.in_features == 128
    output["logits"].sum().backward()
    assert torch.isfinite(output["logits"]).all()


def test_late_fusion_shape_health_uses_late_projection_without_external_query():
    from methods.structure_da.prototype_losses import (
        shape_gradient_snapshot,
        shape_health_snapshot,
    )

    model = _model("sorted_profile", "late_fusion")
    output = model(*_batch(), return_dict=True)
    output["logits"].sum().backward()
    health = shape_health_snapshot(model, output)
    gradients = shape_gradient_snapshot(model)
    assert health["shape_query_projection_norm"] == pytest.approx(
        float(model.late_fusion_projection.weight.detach().norm())
    )
    assert gradients["grad_norm_query_projection"] >= 0.


def test_current_defaults_preserve_legacy_modules_and_strict_checkpoint_load():
    implicit = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
    )
    explicit = _model("current", "current_query")
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    assert explicit.shape_classifier.in_features == 32
    assert explicit.structure_branch.response_to_query[0].in_features == 32
    assert explicit.temporal_encoder.attention_heads.external_query_projection.in_features == 8
    assert not hasattr(explicit, "late_fusion_projection")


def test_invalid_representation_injection_combinations_fail_fast():
    with pytest.raises(ValueError, match="current representation"):
        _model("current", "late_fusion")
    with pytest.raises(ValueError, match="sorted_profile representation"):
        _model("sorted_profile", "current_query")


def test_cli_defaults_and_create_model_propagate_sorted_usage():
    import train

    parser = argparse.ArgumentParser()
    train.add_model_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.shape_representation == "current"
    assert defaults.shape_injection == "current_query"
    selected = parser.parse_args([
        "--model", "psestructureprotoltae",
        "--shape-representation", "sorted_profile",
        "--shape-injection", "late_fusion",
    ])
    config = SimpleNamespace(
        **vars(selected), input_dim=10, num_classes=3, with_extra=False,
    )
    model = train.create_model(config)
    assert model.shape_representation == "sorted_profile"
    assert model.shape_injection == "late_fusion"
    assert model.shape_evidence_dim == 128


def test_usage_manifest_and_diagnostics_describe_only_selected_injection():
    import train

    config = SimpleNamespace(
        shape_representation="sorted_profile", shape_injection="direct_query",
        shapelet_count=16, shape_window_scales=[24], shape_window_stride=8,
        shape_align_weight=0.,
    )
    manifest = train.structure_usage_manifest(config)
    assert manifest == {
        "shape_representation": "sorted_profile",
        "shape_injection": "direct_query",
        "shape_evidence_dim": 128,
        "shape_align_weight": 0.,
    }
    model = _model("sorted_profile", "direct_query")
    output = model(*_batch(), return_dict=True)
    diagnostics = model.structure_usage_diagnostics(output)
    assert set(diagnostics) == {
        "sorted_profile_norm_mean", "query_projection_weight_norm",
    }
    assert diagnostics["sorted_profile_norm_mean"].ndim == 0


def test_launcher_dry_run_contract_is_four_fixed_workers():
    source = Path("scripts/run_structure_sorted_usage_2tasks_4gpu_seed1.sh").read_text()
    assert source.count("run_task \"$GPU") == 4
    assert 'run_task "$GPU0" direct_query FR2 "$FR2" DK1 "$DK1"' in source
    assert 'run_task "$GPU1" direct_query AT1 "$AT1" DK1 "$DK1"' in source
    assert 'run_task "$GPU2" late_fusion FR2 "$FR2" DK1 "$DK1"' in source
    assert 'run_task "$GPU3" late_fusion AT1 "$AT1" DK1 "$DK1"' in source
    assert "--shape-align-weight 0.0" in source
    assert "--shape-representation sorted_profile" in source
    assert 'STRUCTURE_USAGE_CONFIG|' in source
