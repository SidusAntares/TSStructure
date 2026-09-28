from pathlib import Path

import pytest
import torch

from models.stclassifier import PseStructureProtoLTae


def _model(injection="local_query_only"):
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation="current", shape_injection=injection,
        structure_shift_mode="none", dropout=0.,
    )


def _batch():
    pixels = torch.randn(2, 8, 3, 4)
    mask = torch.ones(2, 8, 4)
    positions = torch.arange(8).repeat(2, 1) * 40
    extra = torch.zeros(2, 4)
    return pixels, mask, positions, extra


def test_structure_query_only_shapes_queries_and_exact_aggregation():
    torch.manual_seed(41)
    model = _model().eval()
    output = model(*_batch(), return_dict=True)

    assert output["shapelet_similarity"].shape == (2, 8, 16)
    assert output["local_query"].shape == (2, 8, 2, 4)
    assert output["local_structure_readout"].shape == (2, 8, 6)
    assert output["structure_attention"].shape == (2, 8)
    torch.testing.assert_close(output["structure_attention"].sum(1), torch.ones(2))
    expected = (
        output["structure_attention"].unsqueeze(-1)
        * output["local_structure_readout"]
    ).sum(1)
    torch.testing.assert_close(output["instance_feature"], expected)
    assert "base_instance_feature" not in output
    assert "structure_residual" not in output
    assert "structure_gamma" not in output


def test_structure_query_only_projection_is_xavier_and_excludes_base_query():
    torch.manual_seed(43)
    model = _model().eval()
    assert torch.count_nonzero(model.local_query_projection.weight) > 0
    similarity = torch.zeros(1, 8, 16)
    similarity[:, 1] = 1.
    projected = model._project_local_queries(similarity)
    expected = model.local_query_projection(similarity).view(1, 8, 2, 4)
    torch.testing.assert_close(projected, expected)
    assert not torch.equal(projected[:, 0], projected[:, 1])


def test_structure_query_only_backward_skips_base_query_but_trains_shared_readout():
    torch.manual_seed(47)
    model = _model().train()
    output = model(*_batch(), return_dict=True)
    output["logits"].sum().backward()

    base_grad = model.temporal_encoder.attention_heads.query.grad
    assert base_grad is None or torch.count_nonzero(base_grad) == 0
    assert model.temporal_encoder.attention_heads.key.weight.grad is not None
    assert torch.count_nonzero(model.temporal_encoder.attention_heads.key.weight.grad) > 0
    mlp_parameter = next(model.temporal_encoder.mlp.parameters())
    assert mlp_parameter.grad is not None
    assert torch.count_nonzero(mlp_parameter.grad) > 0
    assert model.local_query_projection.weight.grad is not None


def test_structure_query_only_shift_changes_readout_not_structure_tokens():
    torch.manual_seed(53)
    model = _model().eval()
    batch = _batch()
    zero = model.forward_with_temporal_shift(
        *batch, temporal_shift=0, return_dict=True,
    )
    shifted = model.forward_with_temporal_shift(
        *batch, temporal_shift=8, return_dict=True,
    )
    torch.testing.assert_close(zero["shape_tokens"], shifted["shape_tokens"])
    assert not torch.allclose(
        zero["local_structure_readout"], shifted["local_structure_readout"],
    )


def test_structure_query_only_launcher_contract():
    source = Path(
        "scripts/run_structure_query_only_and_audit_2tasks_4gpu_seed1.sh"
    ).read_text(encoding="utf-8")
    assert "DRY_RUN" in source
    assert "shape-injection local_query_only" in source
    assert "--structure-shift-mode none" in source
    assert "--shape-align-weight 0.0" in source
    assert "uq_AT1_DK1_seed1" in source
    assert "uq_FR2_DK1_seed1" in source
    assert 'run_source "$GPU0" AT1' in source
    assert 'run_source "$GPU1" FR2' in source
    assert 'run_audit "$GPU2" AT1_DK1' in source
    assert 'run_audit "$GPU3" FR2_DK1' in source
    assert 'run_uda "$GPU0" AT1' in source
    assert 'run_uda "$GPU1" FR2' in source
