from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.ltae import LTAE
from models.stclassifier import PseStructureProtoLTae
from models.structure_da.discriminative_structure import DiscriminativeStructureBranch


def _model(shift_mode="none"):
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation="current", shape_injection="local_query",
        structure_shift_mode=shift_mode,
    )


def _batch():
    pixels = torch.randn(2, 8, 3, 4)
    mask = torch.ones(2, 8, 4)
    positions = torch.arange(8).repeat(2, 1) * 40
    extra = torch.zeros(2, 4)
    return pixels, mask, positions, extra


def test_ltae_local_queries_share_memory_attention_and_mlp_and_zero_is_base():
    torch.manual_seed(3)
    encoder = LTAE(
        in_channels=8, n_head=2, d_k=4, d_model=8,
        n_neurons=[8, 6], dropout=0,
    ).eval()
    x = torch.randn(2, 7, 8)
    positions = torch.arange(7).repeat(2, 1)
    corrections = torch.zeros(2, 8, 2, 4)
    base, local = encoder.forward_with_local_queries(x, positions, corrections)
    assert base.shape == (2, 6)
    assert local.shape == (2, 8, 6)
    torch.testing.assert_close(local, base[:, None].expand_as(local), rtol=0, atol=0)
    assert not hasattr(encoder, "local_key")
    assert not hasattr(encoder, "local_value")
    assert not hasattr(encoder, "local_mlp")


def test_zero_local_query_is_base_in_training_with_shared_dropout_masks():
    torch.manual_seed(31)
    encoder = LTAE(
        in_channels=8, n_head=2, d_k=4, d_model=8,
        n_neurons=[8, 6], dropout=.2,
    ).train()
    x = torch.randn(4, 7, 8)
    positions = torch.arange(7).repeat(4, 1)
    base, local = encoder.forward_with_local_queries(
        x, positions, torch.zeros(4, 8, 2, 4),
    )
    torch.testing.assert_close(
        local, base[:, None].expand_as(local), rtol=1e-6, atol=1e-6,
    )


def test_local_query_shapes_zero_init_residual_aggregation_and_gamma():
    torch.manual_seed(5)
    model = _model().eval()
    output = model(*_batch(), return_dict=True)
    assert output["shapelet_similarity"].shape == (2, 8, 16)
    assert output["local_query_correction"].shape == (2, 8, 2, 4)
    assert output["local_structure_readout"].shape == (2, 8, 6)
    assert output["structure_residual"].shape == (2, 8, 6)
    assert output["structure_attention"].shape == (2, 8)
    torch.testing.assert_close(
        output["structure_attention"].sum(1), torch.ones(2),
    )
    torch.testing.assert_close(
        output["structure_residual"], torch.zeros_like(output["structure_residual"]),
        rtol=0, atol=0,
    )
    expected = (
        output["structure_attention"].unsqueeze(-1)
        * output["structure_residual"]
    ).sum(1)
    torch.testing.assert_close(output["structure_feature"], expected)
    torch.testing.assert_close(output["instance_feature"], output["base_instance_feature"])
    assert float(output["structure_gamma"]) == pytest.approx(.1, abs=1e-6)
    assert 0. <= float(output["structure_gamma"]) <= .5
    assert torch.count_nonzero(model.local_query_projection.weight) == 0
    output["logits"].sum().backward()
    assert model.local_query_projection.weight.grad is not None


def test_fourier_shift_sign_and_per_sample_shift():
    torch.manual_seed(7)
    branch = DiscriminativeStructureBranch(
        6, shape_dim=8, shapelet_count=16,
        window_scales=(24,), window_stride=8, num_modes=5,
    ).eval()
    features = torch.randn(2, 12, 6)
    positions = torch.arange(12).repeat(2, 1) * 25
    context = branch.prepare_context(features, positions)
    base = branch.forward_from_context(context, temporal_shift=0)
    spacing = 365. / 64
    shifted = branch.forward_from_context(context, temporal_shift=spacing)
    torch.testing.assert_close(
        shifted["exposed_curve"], torch.roll(base["exposed_curve"], 1, dims=1),
        atol=2e-4, rtol=2e-4,
    )
    per_sample = branch.forward_from_context(
        context, temporal_shift=torch.tensor([[0.], [spacing]]),
    )
    torch.testing.assert_close(per_sample["exposed_curve"][0], base["exposed_curve"][0])
    torch.testing.assert_close(
        per_sample["exposed_curve"][1],
        torch.roll(base["exposed_curve"][1], 1, dims=0),
        atol=2e-4, rtol=2e-4,
    )


def test_uq_ignores_structure_shift_sq_applies_it_and_zero_shift_matches():
    torch.manual_seed(11)
    uq, sq = _model("none").eval(), _model("timematch").eval()
    sq.load_state_dict(uq.state_dict(), strict=True)
    batch = _batch()
    uq0 = uq.forward_with_temporal_shift(*batch, temporal_shift=0, return_dict=True)
    uq8 = uq.forward_with_temporal_shift(*batch, temporal_shift=8, return_dict=True)
    sq0 = sq.forward_with_temporal_shift(*batch, temporal_shift=0, return_dict=True)
    sq8 = sq.forward_with_temporal_shift(*batch, temporal_shift=8, return_dict=True)
    torch.testing.assert_close(uq0["shape_tokens"], uq8["shape_tokens"])
    assert not torch.allclose(sq0["shape_tokens"], sq8["shape_tokens"])
    torch.testing.assert_close(uq0["logits"], sq0["logits"])


def test_sq_shift_grid_reuses_one_fourier_analysis(monkeypatch):
    from timematch import _classify_shift_grid

    model = _model("timematch").eval()
    prepared = torch.randn(2, 10, 8)
    positions = torch.arange(10).repeat(2, 1) * 20
    calls = 0
    original = model.structure_branch.exposer.analyzer.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.structure_branch.exposer.analyzer, "forward", counted)
    result = _classify_shift_grid(model, prepared, positions, [-5, 0, 5])
    assert result.shape == (2, 3, 3)
    assert calls == 1


def test_launcher_dry_run_has_four_uda_jobs_and_shared_source_paths():
    source = Path("scripts/run_structure_local_query_shift_2tasks_4gpu_seed1.sh").read_text()
    assert source.count("LOCAL_QUERY_UDA_PLAN|") == 1
    assert 'run_uda "$GPU0" uq none AT1 "$AT1" DK1 "$DK1"' in source
    assert 'run_uda "$GPU1" uq none FR2 "$FR2" DK1 "$DK1"' in source
    assert 'run_uda "$GPU2" sq timematch AT1 "$AT1" DK1 "$DK1"' in source
    assert 'run_uda "$GPU3" sq timematch FR2 "$FR2" DK1 "$DK1"' in source
    assert '--shape-align-weight 0.0' in source
    assert source.count('source_weights="$EXP_ROOT/source/${src_name}/source_${src_name}_seed1"') == 1


def test_local_query_manifest_records_reusable_architecture_contract():
    from train import structure_usage_manifest

    manifest = structure_usage_manifest(SimpleNamespace(
        shape_representation="current", shape_injection="local_query",
        structure_shift_mode="none", shapelet_count=16,
        shape_window_scales=[24], shape_window_stride=8,
        shape_align_weight=0.,
    ))
    assert manifest["structure_shift_mode"] == "none"
    assert manifest["local_structure_token_dim"] == 16
    assert manifest["local_structure_windows"] == 8
    assert manifest["local_query"] == "base_plus_local_delta"
    assert manifest["structure_gamma_max"] == .5
