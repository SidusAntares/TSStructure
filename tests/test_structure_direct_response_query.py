from pathlib import Path

import pytest
import torch
from torch import nn

from models.stclassifier import PseStructureProtoLTae


def _model(representation="current", injection="direct_response_query"):
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=16, d_k=8, d_model=128, mlp3=[128, 16], mlp4=[16],
        num_classes=3, shape_dim=128, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation=representation, shape_injection=injection,
        structure_shift_mode="none", dropout=0.,
    )


def _batch():
    pixels = torch.randn(2, 8, 3, 4)
    mask = torch.ones(2, 8, 4)
    positions = torch.arange(8).repeat(2, 1) * 40
    return pixels, mask, positions, torch.zeros(2, 4)


def test_direct_response_query_accepts_only_current_response():
    model = _model()
    assert model.shape_evidence_dim == 32
    for representation in ("set_response", "residual_response", "sorted_profile"):
        with pytest.raises(ValueError, match="direct_response_query"):
            _model(representation)


def test_direct_response_projection_is_32_to_128_and_zero_initialized():
    projection = _model().temporal_encoder.attention_heads.external_query_projection
    assert projection.in_features == 32
    assert projection.out_features == 128
    assert torch.count_nonzero(projection.weight) == 0


class _ForbiddenResponseToQuery(nn.Module):
    def forward(self, value):
        raise AssertionError("response_to_query must be bypassed")


def test_direct_response_query_passes_raw_evidence_without_adapter_or_layer_norm():
    torch.manual_seed(71)
    model = _model().eval()
    model.structure_branch.response_to_query = _ForbiddenResponseToQuery()
    captured = {}

    def capture_external_query(module, args, kwargs):
        captured["external_query"] = kwargs["external_query"].detach().clone()

    handle = model.temporal_encoder.register_forward_pre_hook(
        capture_external_query, with_kwargs=True,
    )
    output = model(*_batch(), return_dict=True)
    handle.remove()

    assert output["shape_class_token"] is None
    torch.testing.assert_close(captured["external_query"], output["shape_evidence"])
    assert not torch.allclose(
        captured["external_query"],
        torch.nn.functional.layer_norm(
            output["shape_evidence"], output["shape_evidence"].shape[-1:],
        ),
    )
    torch.testing.assert_close(
        output["shape_logits"], model.shape_classifier(output["shapelet_response"]),
    )


def test_existing_representation_checkpoints_remain_strict_loadable():
    for representation in ("current", "set_response", "residual_response"):
        original = _model(representation, "current_query")
        restored = _model(representation, "current_query")
        restored.load_state_dict(original.state_dict(), strict=True)


def test_direct_response_launcher_contract():
    source = Path(
        "scripts/run_structure_direct_response_query_4tasks_4gpu_seed1.sh"
    ).read_text(encoding="utf-8")
    assert source.count('run_task "$GPU') == 4
    assert 'run_task "$GPU0" AT1 "$AT1" DK1 "$DK1"' in source
    assert 'run_task "$GPU1" FR1 "$FR1" FR2 "$FR2"' in source
    assert 'run_task "$GPU2" FR2 "$FR2" DK1 "$DK1"' in source
    assert 'run_task "$GPU3" DK1 "$DK1" AT1 "$AT1"' in source
    assert source.count("--shape-representation current") == 2
    assert source.count("--shape-injection direct_response_query") == 2
    assert source.count("--source-minority-mode base") == 2
    assert "external_query_dim=32" in source
    assert "response_to_query_bypassed=true" in source
