from pathlib import Path

import pytest
import torch

from models.stclassifier import PseStructureProtoLTae
from models.structure_da.discriminative_structure import DiscriminativeStructureBranch


def _branch(representation="residual_response", count=16):
    return DiscriminativeStructureBranch(
        6, shape_dim=8, shapelet_count=count,
        window_scales=(24,), window_stride=8, num_modes=5,
        shape_representation=representation,
    )


def _model(representation="residual_response", injection="current_query"):
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
    return pixels, mask, positions, torch.zeros(2, 4)


def test_zero_context_strictly_matches_current_response():
    current = _branch("current").eval()
    residual = _branch("residual_response").eval()
    incompatibility = residual.load_state_dict(current.state_dict(), strict=False)
    assert incompatibility.missing_keys == ["anchor_context_weight"]
    assert incompatibility.unexpected_keys == []
    features = torch.randn(3, 8, 6)
    positions = torch.arange(8).repeat(3, 1) * 40
    with torch.no_grad():
        expected = current(features, positions)
        actual = residual(features, positions)
    torch.testing.assert_close(actual["shapelet_response"], expected["shapelet_response"])
    torch.testing.assert_close(
        actual["shapelet_context_score"], actual["shapelet_similarity"],
    )
    assert torch.count_nonzero(actual["shapelet_modulation"]) == 0


def test_context_diagonal_is_never_used_for_self_modulation():
    branch = _branch(count=4)
    with torch.no_grad():
        branch.anchor_context_weight.zero_()
        branch.anchor_context_weight.diagonal().fill_(1000.)
    similarity = torch.randn(2, 5, 4)
    contextual, modulation = branch.contextualize_similarity(similarity)
    torch.testing.assert_close(contextual, similarity)
    assert torch.count_nonzero(modulation) == 0


def test_modulation_is_bounded_sign_preserving_and_trainable():
    branch = _branch(count=4)
    with torch.no_grad():
        branch.anchor_context_weight.copy_(torch.tensor([
            [0., 2., -1., .5], [-2., 0., 1., .2],
            [.7, -.3, 0., 1.], [1., 1., -1., 0.],
        ]))
    similarity = torch.randn(3, 6, 4, requires_grad=True)
    contextual, modulation = branch.contextualize_similarity(similarity)
    assert float(modulation.min()) >= -1.
    assert float(modulation.max()) <= 1.
    nonzero = similarity != 0
    assert torch.equal(torch.sign(contextual[nonzero]), torch.sign(similarity[nonzero]))
    contextual.square().mean().backward()
    assert branch.anchor_context_weight.grad is not None
    off_diagonal = ~torch.eye(4, dtype=torch.bool)
    assert torch.count_nonzero(branch.anchor_context_weight.grad[off_diagonal]) > 0
    assert torch.count_nonzero(branch.anchor_context_weight.grad.diagonal()) == 0


def test_residual_output_pools_contextual_similarity_and_uses_existing_query():
    model = _model().eval()
    output = model(*_batch(), return_dict=True)
    assert output["shapelet_response"].shape == (2, 32)
    torch.testing.assert_close(
        output["shapelet_response"],
        torch.cat((output["shapelet_strength"], output["shapelet_concentration"]), -1),
    )
    torch.testing.assert_close(
        output["shape_class_token"],
        model.structure_branch.response_to_query(output["shapelet_response"]),
    )
    output["logits"].sum().backward()
    assert model.structure_branch.anchor_context_weight.grad is not None


def test_residual_only_allows_current_query_and_preserves_old_checkpoint_keys():
    with pytest.raises(ValueError, match="residual_response representation"):
        _model("residual_response", "direct_query")
    for representation in ("current", "set_response"):
        implicit = _model(representation, "current_query")
        explicit = _model(representation, "current_query")
        explicit.load_state_dict(implicit.state_dict(), strict=True)
        assert not hasattr(implicit.structure_branch, "anchor_context_weight")


def test_residual_usage_diagnostics_are_exact():
    model = _model().eval()
    output = model(*_batch(), return_dict=True)
    diagnostics = model.structure_usage_diagnostics(output)
    assert set(diagnostics) == {
        "mean_abs_modulation", "max_abs_modulation",
        "relative_context_correction",
    }
    assert all(float(value) == pytest.approx(0.) for value in diagnostics.values())


def test_residual_launcher_contract():
    source = Path(
        "scripts/run_structure_residual_response_4tasks_4gpu_seed1.sh"
    ).read_text()
    assert source.count('run_task "$GPU') == 4
    assert 'run_task "$GPU0" AT1 "$AT1" DK1 "$DK1"' in source
    assert 'run_task "$GPU1" FR1 "$FR1" FR2 "$FR2"' in source
    assert 'run_task "$GPU2" FR2 "$FR2" DK1 "$DK1"' in source
    assert 'run_task "$GPU3" DK1 "$DK1" AT1 "$AT1"' in source
    assert source.count("--shape-representation residual_response") == 2
    assert source.count("--shape-injection current_query") == 2
    assert source.count("--source-minority-mode base") == 2
    assert "--shapelet-count 16" in source
    assert "--shape-window-scales 24 --shape-window-stride 8" in source

