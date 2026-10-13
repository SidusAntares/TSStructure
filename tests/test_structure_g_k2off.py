from pathlib import Path
import argparse
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.modules.setdefault("ot", types.ModuleType("ot"))


def _model(k2=True):
    from models.stclassifier import PseStructureProtoLTae

    return PseStructureProtoLTae(
        input_dim=10, mlp1=[10, 16], pooling="mean_std", mlp2=[32, 128],
        with_extra=False, n_head=2, d_k=4, d_model=16, mlp3=[16, 8],
        dropout=0., T=1000, mlp4=[8, 3], num_classes=3,
        max_temporal_shift=60, shape_dim=128, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, shapelet_beta=5.,
        shape_resample_length=16, fourier_num_modes=13,
        shape_representation="phase_moment", shape_injection="current_query",
        structure_shift_mode="none", shape_phase_k2_enabled=k2,
    )


def _input(batch=3):
    return (
        torch.randn(batch, 10, 10, 4), torch.ones(batch, 10, 4),
        torch.arange(10).repeat(batch, 1) * 30, torch.zeros(batch, 4),
    )


def test_k2off_response_and_classifier_evidence_end_in_zero32_while_k1_remains():
    model = _model(False).eval()
    output = model(*_input(), return_dict=True)
    assert output["shapelet_response"].shape == (3, 96)
    torch.testing.assert_close(output["shapelet_response"][:, 64:], torch.zeros(3, 32))
    assert float(output["shapelet_response"][:, 32:64].abs().sum()) > 0
    torch.testing.assert_close(output["shape_evidence"], output["shapelet_response"])


def test_k2off_query_and_shape_loss_cannot_backpropagate_to_k2():
    model = _model(False).train()
    seen = {}

    def capture(name):
        def hook(_module, inputs):
            seen[name] = inputs[0].detach().clone()
        return hook

    handles = [
        model.structure_branch.response_to_query[0].register_forward_pre_hook(capture("query")),
        model.shape_classifier.register_forward_pre_hook(capture("shape")),
    ]
    output = model(*_input(), return_dict=True)
    (output["logits"].square().mean() + output["shape_logits"].square().mean()).backward()
    for handle in handles:
        handle.remove()
    assert set(seen) == {"query", "shape"}
    for evidence in seen.values():
        torch.testing.assert_close(evidence[:, 64:], torch.zeros_like(evidence[:, 64:]))
        assert float(evidence[:, 32:64].abs().sum()) > 0
    assert any(p.grad is not None for p in model.structure_branch.token_generator.parameters())


def test_k2off_equivariance_uses_exact_k1_slice_and_keeps_k1_gradient(monkeypatch):
    import timematch

    model = _model(False).train(); seen = {}
    original = timematch.phase_equivariance_loss

    def capture(base, shifted, delta, **kwargs):
        seen["base_dim"] = base["shapelet_phase_moments"].shape[1]
        seen["shifted_dim"] = shifted["shapelet_phase_moments"].shape[1]
        seen["harmonics"] = kwargs["harmonics"]
        return original(base, shifted, delta, **kwargs)

    monkeypatch.setattr(timematch, "phase_equivariance_loss", capture)
    _, losses, _ = timematch.forward_target_phase_equivariance(
        model, *_input(), max_shift=60,
    )
    assert seen == {"base_dim": 32, "shifted_dim": 32, "harmonics": (1,)}
    losses["total_loss"].backward()
    assert any(p.grad is not None for p in model.structure_branch.token_generator.parameters())


def test_default_switch_preserves_original_forward_exactly():
    torch.manual_seed(9); implicit = _model()
    torch.manual_seed(9); explicit = _model(True)
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    inputs = _input()
    implicit.eval(); explicit.eval()
    with torch.no_grad():
        left = implicit(*inputs, return_dict=True)
        right = explicit(*inputs, return_dict=True)
    torch.testing.assert_close(left["logits"], right["logits"], rtol=0, atol=0)
    torch.testing.assert_close(left["shapelet_response"], right["shapelet_response"], rtol=0, atol=0)


def test_cli_and_scheduler_freeze_k2off_contract():
    from train import add_model_arguments

    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    config = parser.parse_args(["--shape-phase-k2-enabled", "false"])
    assert config.shape_phase_k2_enabled is False
    launcher = Path("scripts/run_structure_g_k2off_seed1.py").read_text()
    assert "--shape-phase-k2-enabled" in launcher
    assert "false" in launcher
    assert "--target-structure-detach-epochs" in launcher and "5" in launcher
    assert "gpus" in launcher and "resume" in launcher and "dry-run" in launcher


def test_checkpoint_status_requires_k2off_and_detach5():
    from scripts.run_structure_g_k2off_seed1 import artifact_status

    source_packet = {
        "epoch": 99,
        "config": SimpleNamespace(epochs=100, shape_phase_k2_enabled=False),
    }
    uda_packet = {
        "epoch": 19,
        "config": SimpleNamespace(
            epochs=20, shape_phase_k2_enabled=False,
            target_structure_detach_epochs=5,
        ),
    }
    incompatible = {
        **uda_packet,
        "config": SimpleNamespace(
            epochs=20, shape_phase_k2_enabled=True,
            target_structure_detach_epochs=5,
        ),
    }
    partial_source = {
        "epoch": 1,
        "config": SimpleNamespace(epochs=100, shape_phase_k2_enabled=False),
    }
    with patch.object(Path, "exists", return_value=True), patch.object(
        Path, "is_file", return_value=True,
    ), patch("torch.load", side_effect=[source_packet, uda_packet, incompatible, partial_source]):
        assert artifact_status(Path("virtual/source"), "source") == "complete"
        assert artifact_status(Path("virtual/uda"), "uda") == "complete"
        assert artifact_status(Path("virtual/uda"), "uda") == "incompatible"
        assert artifact_status(Path("virtual/source"), "source") == "partial"


def test_two_gpu_scheduler_has_all_tasks_and_no_baseline_output_overlap():
    from scripts.run_structure_g_k2off_seed1 import jobs, build_parser

    assert {job.task for job in jobs()} == {"AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1"}
    args = build_parser().parse_args(["dry-run"])
    assert len(args.gpus) == 2
    assert "g_k2off" in args.root.lower()


def test_frozen_audit_missing_cache_is_a_formal_failure():
    from scripts.run_structure_g_k2off_seed1 import run_frozen_audits

    args = SimpleNamespace(
        cache_root="virtual/missing", mode="run", root=Path("virtual/root"),
    )
    with patch.object(Path, "is_dir", return_value=False), patch.object(
        Path, "glob", return_value=[],
    ), patch.object(Path, "write_text", return_value=1):
        with pytest.raises(RuntimeError, match="frozen audits incomplete"):
            run_frozen_audits(args)

