import argparse
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.stclassifier import PseStructureProtoLTae
from models.structure_da.discriminative_structure import ShapeTokenAttentionPool
from timematch import target_structure_detached_for_epoch


def _model():
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=2, fourier_num_modes=5,
        shape_representation="direct_shape_token", shape_injection="current_query",
    )


def _batch(batch=2, timesteps=12):
    pixels = torch.randn(batch, timesteps, 3, 4)
    mask = torch.ones(batch, timesteps, 4, dtype=torch.bool)
    positions = torch.arange(timesteps).repeat(batch, 1) * 25
    return pixels, mask, positions, torch.zeros(batch, 4)


def test_attention_pool_shape_mask_and_calendar_shift():
    torch.manual_seed(2)
    pool = ShapeTokenAttentionPool(128, hidden_dim=64, period_days=365., grid_points=64)
    tokens = torch.randn(2, 8, 128, requires_grad=True)
    centers = torch.arange(8, dtype=torch.float32) * 8 + 12
    mask = torch.tensor([[True] * 8, [True] * 7 + [False]])
    result = pool(tokens, centers, candidate_mask=mask, phase_shift=torch.tensor([0., 15.]))
    assert result["shape_token"].shape == (2, 128)
    assert result["attention"].shape == (2, 8)
    torch.testing.assert_close(result["attention"].sum(1), torch.ones(2))
    assert result["attention"][1, -1].item() == 0
    expected_theta = 2 * torch.pi * (
        centers * 365. / 64. + torch.tensor([[0.], [15.]])
    ) / 365.
    expected = torch.stack((
        expected_theta.cos(), expected_theta.sin(),
        (2 * expected_theta).cos(), (2 * expected_theta).sin(),
    ), dim=-1)
    torch.testing.assert_close(result["time_encoding"], expected)
    result["shape_token"].square().mean().backward()
    assert tokens.grad is not None and tokens.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pool.parameters())


def test_direct_shape_token_bypasses_anchor_and_strict_loads():
    model = _model().eval()
    assert not hasattr(model.structure_branch, "shapelet_dictionary")
    assert model.shape_evidence_dim == 8
    clone = _model().eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    output = model(*_batch(), return_dict=True)
    assert output["shapelet_response"].shape == (2, 8)
    assert output["shape_class_token"].shape == (2, 8)
    assert output["shape_response_semantics"] == "direct_token_128"
    assert "shapelet_similarity" not in output
    assert "shapelet_concentration" not in output


def test_shape_aux_and_query_paths_receive_gradients_and_detach_is_value_preserving():
    torch.manual_seed(7)
    model = _model().eval()
    projection = model.temporal_encoder.attention_heads.external_query_projection
    torch.nn.init.normal_(projection.weight, std=.1)
    batch = _batch()
    normal = model.forward_with_temporal_shift(
        *batch, return_dict=True, detach_structure_query=False,
    )
    assert projection(normal["shape_class_token"]).abs().sum() > 0
    detached = model.forward_with_temporal_shift(
        *batch, return_dict=True, detach_structure_query=True,
    )
    torch.testing.assert_close(normal["logits"], detached["logits"])
    (normal["logits"].square().mean() + normal["shape_logits"].square().mean()).backward()
    assert projection.weight.grad is not None and projection.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.structure_branch.token_generator.parameters()
    )
    assert model.shape_classifier.weight.grad is not None

    detached_model = _model().eval()
    detached_model.load_state_dict(model.state_dict(), strict=True)
    detached_projection = (
        detached_model.temporal_encoder.attention_heads.external_query_projection
    )
    detached_output = detached_model.forward_with_temporal_shift(
        *batch, return_dict=True, detach_structure_query=True,
    )
    detached_output["logits"].square().mean().backward()
    assert detached_projection.weight.grad is None
    key = detached_model.temporal_encoder.attention_heads.key.weight
    assert key.grad is not None and key.grad.abs().sum() > 0


def test_direct_shape_token_cli_contract_and_detach_schedule():
    sys.modules.setdefault("ot", types.ModuleType("ot"))
    from train import add_model_arguments, validate_structure_configuration

    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parsed = parser.parse_args([
        "--shape-representation", "direct_shape_token",
        "--shape-injection", "current_query",
    ])
    assert parsed.shape_representation == "direct_shape_token"
    config = SimpleNamespace(
        shape_representation="direct_shape_token", shape_injection="current_query",
        shape_equivariance_weight=0., shape_align_weight=0.,
        shapelet_diversity_weight=0., shapelet_shaping_weight=0.,
        class_org_source_weight=0., class_org_target_weight=0.,
        source_minority_mode="base",
    )
    validate_structure_configuration(config)
    config.target_structure_detach_epochs = 5
    assert [target_structure_detached_for_epoch(config, i) for i in range(7)] == [
        True, True, True, True, True, False, False,
    ]
    config.shape_equivariance_weight = .05
    with pytest.raises(ValueError, match="shape_equivariance_weight=0"):
        validate_structure_configuration(config)
    config.shape_equivariance_weight = 0.
    config.source_minority_mode = "weighted_proto"
    with pytest.raises(ValueError, match="source_minority_mode=base"):
        validate_structure_configuration(config)


def test_direct_shape_token_validation_reports_shape_head_metrics(monkeypatch):
    import evaluation as evaluation_module

    class FixedDirectModel(torch.nn.Module):
        shape_representation = "direct_shape_token"

        def forward(self, pixels, mask, positions, extra, return_dict=False):
            output = {
                "logits": torch.tensor([[4., 0.], [0., 4.]]),
                "shape_logits": torch.tensor([[3., 0.], [2., 1.]]),
            }
            return output if return_dict else output["logits"]

    sample = {
        "label": torch.tensor([0, 1]),
        "pixels": torch.zeros(2, 1),
    }
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    monkeypatch.setattr(
        evaluation_module, "to_cuda",
        lambda current, device: (
            current["pixels"], None, torch.zeros(2, 1, dtype=torch.long), None,
        ),
    )
    metrics = evaluation_module.evaluation(
        FixedDirectModel(), [sample], "cpu", ["zero", "one"],
        mode="val", progress_bar="off",
    )
    assert metrics["shape_head_accuracy"] == pytest.approx(.5)
    assert metrics["shape_head_macro_f1"] == pytest.approx(1 / 3)
    assert metrics["shape_head_fixed_class_macro_f1"] == pytest.approx(1 / 3)


def test_scheduler_has_four_source_to_uda_dependencies():
    from scripts.run_structure_direct_shape_token_seed1 import (
        planned_jobs, source_command, uda_command,
    )

    jobs = planned_jobs()
    assert {job.task for job in jobs} == {
        "AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1",
    }
    assert all(job.representation == "direct_shape_token" for job in jobs)
    assert len({job.source_experiment for job in jobs}) == 4
    args = SimpleNamespace(
        python="python", data_root="/data", root=__import__("pathlib").Path("/out"),
    )
    source = source_command(args, jobs[0])
    uda = uda_command(args, jobs[0])
    for command in (source, uda):
        joined = " ".join(map(str, command))
        assert "--shapelet-diversity-weight 0" in joined
        assert "--shapelet-shaping-weight 0" in joined
        assert "--shape-align-weight 0" in joined
    joined = " ".join(map(str, uda))
    assert "--shape-equivariance-weight 0" in joined
    assert "--shape-alignment-view none" in joined
    assert "--target-structure-detach-epochs 5" in joined
    assert "--structure-basis-mode adaptive" in joined
    assert "--uda-anchor-update none" in joined
    assert "--anchor-geometric-mode none" in joined


def test_resume_accepts_only_complete_compatible_source_and_uda(monkeypatch):
    from scripts.run_structure_direct_shape_token_seed1 import artifact_status, planned_jobs

    job = planned_jobs()[0]
    config = {
        "seed": 1, "shape_representation": "direct_shape_token",
        "shape_injection": "current_query", "source": job.source,
        "target": job.target, "target_structure_detach_epochs": 5,
        "steps_per_epoch": 500,
    }
    state = {"structure_branch.shape_token_pool.score.weight": torch.zeros(1, 64)}
    packets = {
        "checkpoint_last.pt": {"epoch": 99, "config": config, "state_dict": state},
        "model.pt": {"epoch": 31, "config": config, "state_dict": state},
    }
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: packets[Path(path).name])
    source = Path("virtual/source")
    assert artifact_status(source, "source", job) == "complete"
    packets["checkpoint_last.pt"] = {
        "epoch": 19, "config": config, "state_dict": state,
    }
    uda = Path("virtual/uda")
    assert artifact_status(uda, "uda", job) == "complete"
    config["shape_representation"] = "phase_moment"
    assert artifact_status(uda, "uda", job) == "incompatible"
