from pathlib import Path

import pytest
import torch

from models.stclassifier import PseStructureProtoLTae
from models.structure_da.discriminative_structure import DiscriminativeStructureBranch


def _branch(representation="set_response"):
    return DiscriminativeStructureBranch(
        6, shape_dim=8, shapelet_count=16,
        window_scales=(24,), window_stride=8, num_modes=5,
        shape_representation=representation,
    )


def _model(representation="set_response", injection="current_query"):
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation=representation, shape_injection=injection,
    )


def test_set_response_is_32d_and_window_permutation_invariant():
    branch = _branch()
    similarity = torch.randn(3, 8, 16)
    mask = torch.ones(3, 8, dtype=torch.bool)
    response = branch.compose_set_response(similarity, mask)
    permutation = torch.tensor([4, 1, 7, 0, 6, 2, 5, 3])
    permuted = branch.compose_set_response(
        similarity[:, permutation], mask[:, permutation],
    )
    assert response.shape == (3, 32)
    torch.testing.assert_close(response, permuted)


def test_set_response_uses_masked_mean_and_backpropagates_to_similarity():
    branch = _branch()
    similarity = torch.randn(2, 8, 16, requires_grad=True)
    mask = torch.tensor([
        [True, True, False, False, False, False, False, False],
        [True, False, True, False, True, False, False, False],
    ])
    response = branch.compose_set_response(similarity, mask)
    encoded = branch.window_set_encoder(similarity)
    expected = (
        encoded * mask.unsqueeze(-1)
    ).sum(1) / mask.sum(1, keepdim=True)
    torch.testing.assert_close(response, expected)
    response.square().mean().backward()
    assert similarity.grad is not None
    assert torch.isfinite(similarity.grad).all()
    assert branch.window_set_encoder[0].weight.grad is not None


def test_set_model_uses_response_for_shape_head_and_existing_query_mlp():
    model = _model()
    assert model.shape_classifier.in_features == 32
    assert model.structure_branch.response_to_query[0].in_features == 32
    pixels = torch.randn(2, 8, 3, 4)
    mask = torch.ones(2, 8, 4)
    positions = torch.arange(8).repeat(2, 1) * 40
    output = model(pixels, mask, positions, torch.zeros(2, 4), return_dict=True)
    assert output["shapelet_response"].shape == (2, 32)
    torch.testing.assert_close(
        output["shape_logits"], model.shape_classifier(output["shapelet_response"]),
    )
    torch.testing.assert_close(
        output["shape_class_token"],
        model.structure_branch.response_to_query(output["shapelet_response"]),
    )
    output["logits"].sum().backward()
    assert model.structure_branch.window_set_encoder[0].weight.grad is not None


def test_set_response_only_allows_current_query():
    with pytest.raises(ValueError, match="set_response representation"):
        _model("set_response", "direct_query")


def test_default_current_has_no_set_parameters_and_strict_loads():
    implicit = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
    )
    explicit = _model("current", "current_query")
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    assert not hasattr(implicit.structure_branch, "window_set_encoder")
    assert not hasattr(explicit.structure_branch, "window_set_encoder")


def test_set_response_launcher_contract():
    source = Path("scripts/run_structure_set_response_4tasks_4gpu_seed1.sh").read_text()
    assert source.count('run_task "$GPU') == 4
    assert 'run_task "$GPU0" AT1 "$AT1" DK1 "$DK1"' in source
    assert 'run_task "$GPU1" FR1 "$FR1" FR2 "$FR2"' in source
    assert 'run_task "$GPU2" FR2 "$FR2" DK1 "$DK1"' in source
    assert 'run_task "$GPU3" DK1 "$DK1" AT1 "$AT1"' in source
    assert source.count("--shape-representation set_response") == 2
    assert source.count("--shape-injection current_query") == 2
    assert source.count("--source-minority-mode base") == 2
    assert "--shapelet-count 16" in source
    assert "--shape-window-scales 24 --shape-window-stride 8" in source

