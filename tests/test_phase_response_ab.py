from pathlib import Path
import shutil
import argparse
from types import SimpleNamespace

import pytest
import torch

from methods.structure_da.phase_equivariance import (
    phase_equivariance_loss,
    rotate_phase_moments,
)
from models.stclassifier import PseStructureProtoLTae
from models.structure_da.discriminative_structure import (
    DiscriminativeStructureBranch,
    normalized_candidate_concentration,
)


def _branch(mode="softmax"):
    return DiscriminativeStructureBranch(
        channels=3, shape_dim=8, num_modes=5, grid_points=64,
        window_scales=(24,), window_stride=8, shapelet_count=2,
        shapelet_beta=5., shape_resample_length=16,
        shape_representation="phase_moment", phase_response_mode=mode,
    )


def _details(similarity):
    similarity = torch.as_tensor(similarity, dtype=torch.float32)
    mask = torch.ones(similarity.shape[:2], dtype=torch.bool)
    weights = torch.softmax(5. * similarity, dim=1)
    return {
        "similarity": similarity,
        "candidate_mask": mask,
        "weights": weights,
        "response": (weights * similarity).sum(1),
    }


def _centers(count=8):
    return torch.arange(count, dtype=torch.float32) * (64 / count) + 12.


def test_softmax_mode_is_numerically_identical_to_legacy_formula():
    torch.manual_seed(3)
    branch = _branch("softmax")
    details = _details(torch.rand(2, 8, 2) * 2 - 1)
    result = branch.compose_phase_response(details, _centers(), phase_shift=0)
    rich = branch.compose_rich_response(details)
    moments = branch.compose_phase_moments(details["weights"], _centers(), 0)
    torch.testing.assert_close(result["response"], torch.cat((rich, moments), -1))
    torch.testing.assert_close(result["strength"], details["response"])
    torch.testing.assert_close(
        result["concentration"],
        normalized_candidate_concentration(details["weights"], details["candidate_mask"]),
    )


@pytest.mark.parametrize("mode", ["support", "independent", "b2"])
def test_ab_fixed_six_block_response_is_96d(mode):
    branch = DiscriminativeStructureBranch(
        channels=3, shape_dim=8, num_modes=5, grid_points=64,
        window_scales=(24,), window_stride=8, shapelet_count=16,
        shapelet_beta=5., shape_resample_length=16,
        shape_representation="phase_moment", phase_response_mode=mode,
    )
    details = _details(torch.rand(2, 8, 16) * 2 - 1)
    result = branch.compose_phase_response(details, _centers(), 0)
    assert result["response"].shape == (2, 96)
    torch.testing.assert_close(
        result["response"], torch.cat((
            result["strength"], result["concentration"], result["phase_moments"],
        ), -1),
    )


def test_support_mode_reuses_legacy_statistics_then_multiplies_by_absolute_support():
    branch = _branch("support")
    details = _details([[[-1., .2], [.5, -.4], [.1, .8]]])
    centers = torch.tensor([0., 8., 16.])
    result = branch.compose_phase_response(details, centers, phase_shift=0)
    support = ((details["similarity"] + 1.) / 2.).amax(1)
    legacy_rich = branch.compose_rich_response(details)
    legacy_moments = branch.compose_phase_moments(details["weights"], centers, 0)
    expected = torch.cat((
        support * legacy_rich[:, :2], support * legacy_rich[:, 2:],
        torch.cat((support, support, support, support), -1) * legacy_moments,
    ), -1)
    torch.testing.assert_close(result["support"], support)
    torch.testing.assert_close(result["response"], expected)
    assert result["response"].shape == (1, 12)


def test_independent_mode_uses_presence_coverage_and_no_window_softmax(monkeypatch):
    branch = _branch("independent")
    similarity = torch.tensor([[[-1., 0.], [0., 1.], [1., -1.], [0., 0.]]])
    details = _details(similarity)
    centers = torch.tensor([0., 16., 32., 48.])
    monkeypatch.setattr(torch, "softmax", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("independent response must not call window softmax")
    ))
    result = branch.compose_phase_response(details, centers, phase_shift=0)
    gate = (similarity + 1.) / 2.
    torch.testing.assert_close(result["strength"], gate.amax(1))
    torch.testing.assert_close(result["concentration"], gate.mean(1))
    assert result["concentration_semantics"] == "window_coverage"
    assert result["response"].shape == (1, 12)


@pytest.mark.parametrize("mode", ["independent", "b2"])
def test_nonsoftmax_full_forward_skips_shapelet_softmax(monkeypatch, mode):
    branch = _branch(mode).eval()
    monkeypatch.setattr(
        branch.shapelet_dictionary, "compute_response",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("independent full forward must bypass compute_response")
        ),
    )
    features = torch.randn(2, 12, 3)
    positions = torch.arange(12).repeat(2, 1) * 25
    output = branch(features, positions)
    assert output["shapelet_response"].shape == (2, 12)


def test_b2_matches_manual_masked_formula_and_block_order():
    branch = _branch("b2")
    similarity = torch.tensor([[[-1., .2], [.5, -.4], [.1, .8], [.9, .0]]])
    mask = torch.tensor([[True, True, False, True]])
    details = {"similarity": similarity, "candidate_mask": mask}
    centers = torch.tensor([0., 16., 32., 48.])
    result = branch.compose_phase_response(details, centers, phase_shift=0)

    valid = mask.unsqueeze(-1)
    count = valid.sum(1).to(similarity.dtype)
    mean = (similarity.masked_fill(~valid, 0.).sum(1) / count)
    evidence = torch.relu(similarity - mean.unsqueeze(1)) / 2.
    evidence = evidence * valid
    presence = (
        similarity.masked_fill(~valid, -torch.inf).amax(1) + 1.
    ) / 2.
    coverage = evidence.sum(1) / count
    peak = evidence.amax(1)
    mass = evidence.sum(1)
    weights = evidence / mass.clamp_min(1e-8).unsqueeze(1)
    moments = branch.compose_phase_moments(weights, centers, 0)
    moments = moments * torch.cat((peak,) * 4, dim=-1)
    expected = torch.cat((presence, coverage, moments), dim=-1)

    torch.testing.assert_close(result["relative_evidence"], evidence)
    torch.testing.assert_close(result["peak_evidence"], peak)
    torch.testing.assert_close(result["response"], expected)
    assert result["concentration_semantics"] == "relative_evidence_coverage"


def test_b2_constant_similarity_has_zero_coverage_and_phase():
    branch = _branch("b2")
    details = {"similarity": torch.full((2, 8, 2), .4),
               "candidate_mask": torch.ones(2, 8, dtype=torch.bool)}
    result = branch.compose_phase_response(details, _centers(), 0)
    torch.testing.assert_close(result["strength"], torch.full((2, 2), .7))
    torch.testing.assert_close(result["concentration"], torch.zeros(2, 2))
    torch.testing.assert_close(result["phase_moments"], torch.zeros(2, 8), atol=0, rtol=0)
    assert torch.isfinite(result["response"]).all()


def test_b2_single_and_opposite_double_peaks_have_analytic_moments():
    branch = _branch("b2")
    centers = torch.tensor([0., 16., 32., 48.])
    single = torch.tensor([[[1.], [-1.], [-1.], [-1.]]])
    single_result = branch.compose_phase_response(
        {"similarity": single, "candidate_mask": torch.ones(1, 4, dtype=torch.bool)},
        centers, 0,
    )
    peak = single_result["peak_evidence"]
    expected_single = torch.cat((peak, torch.zeros_like(peak), peak, torch.zeros_like(peak)), -1)
    torch.testing.assert_close(single_result["phase_moments"], expected_single, atol=2e-6, rtol=0)

    double = torch.tensor([[[1.], [-1.], [1.], [-1.]]])
    double_result = branch.compose_phase_response(
        {"similarity": double, "candidate_mask": torch.ones(1, 4, dtype=torch.bool)},
        centers, 0,
    )
    blocks = double_result["phase_moments"].reshape(1, 2, 2, 1)
    torch.testing.assert_close(blocks[:, 0], torch.zeros_like(blocks[:, 0]), atol=2e-6, rtol=0)
    torch.testing.assert_close(blocks[:, 1, 0], double_result["peak_evidence"], atol=2e-6, rtol=0)
    torch.testing.assert_close(blocks[:, 1, 1], torch.zeros_like(peak), atol=2e-6, rtol=0)


def test_b2_uses_no_window_softmax_and_zero_evidence_has_finite_gradients(monkeypatch):
    branch = _branch("b2")
    similarity = torch.zeros(2, 8, 2, requires_grad=True)
    details = {"similarity": similarity, "candidate_mask": torch.ones(2, 8, dtype=torch.bool)}
    monkeypatch.setattr(torch, "softmax", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("b2 response must not call window softmax")
    ))
    result = branch.compose_phase_response(details, _centers(), 0)
    assert torch.isfinite(result["response"]).all()
    result["response"].sum().backward()
    assert similarity.grad is not None
    assert torch.isfinite(similarity.grad).all()


def test_independent_constant_activation_cancels_on_uniform_circle():
    branch = _branch("independent")
    details = _details(torch.zeros(2, 8, 2))
    result = branch.compose_phase_response(details, _centers(), phase_shift=0)
    torch.testing.assert_close(
        result["phase_moments"], torch.zeros_like(result["phase_moments"]),
        atol=2e-6, rtol=0,
    )


@pytest.mark.parametrize("mode", ["support", "independent", "b2"])
def test_ab_phase_shift_rotates_moments_and_preserves_occurrence(mode):
    torch.manual_seed(4)
    branch = _branch(mode)
    details = _details(torch.rand(2, 8, 2) * 2 - 1)
    base = branch.compose_phase_response(details, _centers(), phase_shift=0)
    shifted = branch.compose_phase_response(details, _centers(), phase_shift=31)
    torch.testing.assert_close(shifted["strength"], base["strength"])
    torch.testing.assert_close(shifted["concentration"], base["concentration"])
    expected = rotate_phase_moments(
        base["phase_moments"], torch.full((2,), 31.),
        shapelet_count=2, harmonics=(1, 2), period_days=365.,
    )
    torch.testing.assert_close(shifted["phase_moments"], expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("mode", ["support", "independent", "b2"])
def test_ab_response_and_equivariance_backpropagate_to_similarity(mode):
    torch.manual_seed(5)
    branch = _branch(mode)
    similarity = (torch.rand(2, 8, 2) * 2 - 1).requires_grad_()
    details = _details(similarity)
    base = branch.compose_phase_response(details, _centers(), phase_shift=0)
    shifted = branch.compose_phase_response(details, _centers(), phase_shift=17)
    loss = phase_equivariance_loss(
        {
            "shapelet_strength": base["strength"],
            "shapelet_concentration": base["concentration"],
            "shapelet_phase_moments": base["phase_moments"],
        },
        {
            "shapelet_strength": shifted["strength"],
            "shapelet_concentration": shifted["concentration"],
            "shapelet_phase_moments": shifted["phase_moments"],
        },
        torch.full((2,), 17.), shapelet_count=2,
    )["total_loss"] + shifted["response"].square().mean()
    loss.backward()
    assert similarity.grad is not None
    assert float(similarity.grad.abs().sum()) > 0


@pytest.mark.parametrize("mode", ["support", "independent", "b2"])
def test_full_ab_response_backpropagates_to_token_encoder_and_anchors(mode):
    torch.manual_seed(9)
    branch = _branch(mode)
    features = torch.randn(2, 12, 3)
    positions = torch.arange(12).repeat(2, 1) * 25
    output = branch(features, positions)
    output["shapelet_response"].square().mean().backward()
    assert branch.shapelet_dictionary.anchors.grad is not None
    assert float(branch.shapelet_dictionary.anchors.grad.abs().sum()) > 0
    token_grads = [
        parameter.grad for parameter in branch.token_generator.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None and float(gradient.abs().sum()) > 0 for gradient in token_grads)


@pytest.mark.parametrize("mode", ["support", "independent", "b2"])
def test_ab_usage_diagnostics_are_tensor_values_for_epoch_accumulation(mode):
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6], num_classes=3,
        shape_dim=8, shape_window_scales=(24,), shape_window_stride=8,
        shapelet_count=2, fourier_num_modes=5,
        shape_representation="phase_moment", shape_injection="current_query",
        phase_response_mode=mode,
    ).eval()
    pixels = torch.randn(2, 10, 3, 4)
    mask = torch.ones(2, 10, 4, dtype=torch.bool)
    positions = torch.arange(10).repeat(2, 1) * 25
    with torch.no_grad():
        output = model(
            pixels, mask, positions, torch.zeros(2, 4), return_dict=True,
        )
    diagnostics = model.structure_usage_diagnostics(output)
    assert diagnostics
    assert all(torch.is_tensor(value) for value in diagnostics.values())
    assert all(torch.isfinite(value).all() for value in diagnostics.values())


def test_default_model_has_no_new_parameters_and_old_checkpoint_loads_strictly():
    kwargs = dict(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6], num_classes=3,
        shape_dim=8, shape_window_scales=(24,), shape_window_stride=8,
        shapelet_count=2, fourier_num_modes=5,
        shape_representation="phase_moment", shape_injection="current_query",
    )
    legacy = PseStructureProtoLTae(**kwargs)
    explicit = PseStructureProtoLTae(**kwargs, phase_response_mode="softmax")
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    assert legacy.structure_branch.phase_response_mode == "softmax"
    assert set(legacy.state_dict()) == set(explicit.state_dict())


def test_default_and_explicit_softmax_models_have_identical_forward_values():
    kwargs = dict(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6], num_classes=3,
        shape_dim=8, shape_window_scales=(24,), shape_window_stride=8,
        shapelet_count=2, fourier_num_modes=5,
        shape_representation="phase_moment", shape_injection="current_query",
    )
    legacy = PseStructureProtoLTae(**kwargs).eval()
    explicit = PseStructureProtoLTae(
        **kwargs, phase_response_mode="softmax",
    ).eval()
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    pixels = torch.randn(2, 10, 3, 4)
    mask = torch.ones(2, 10, 4, dtype=torch.bool)
    positions = torch.arange(10).repeat(2, 1) * 25
    extra = torch.zeros(2, 4)
    with torch.no_grad():
        legacy_output = legacy(
            pixels, mask, positions, extra, return_dict=True,
        )
        explicit_output = explicit(
            pixels, mask, positions, extra, return_dict=True,
        )
    torch.testing.assert_close(
        legacy_output["shapelet_response"],
        explicit_output["shapelet_response"],
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        legacy_output["logits"], explicit_output["logits"], rtol=0, atol=0,
    )


def test_phase_response_cli_defaults_to_legacy_and_accepts_ab_modes():
    from train import add_model_arguments

    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    assert parser.parse_args([]).phase_response_mode == "softmax"
    assert parser.parse_args(["--phase-response-mode", "support"]).phase_response_mode == "support"
    assert parser.parse_args(["--phase-response-mode", "independent"]).phase_response_mode == "independent"
    assert parser.parse_args(["--phase-response-mode", "b2"]).phase_response_mode == "b2"
    with pytest.raises(ValueError, match="requires phase_moment"):
        DiscriminativeStructureBranch(
            channels=3, shape_dim=8, shapelet_count=2,
            shape_representation="current", phase_response_mode="support",
        )


def test_source_checkpoint_mode_must_match_uda_mode():
    from timematch import validate_phase_response_source_checkpoint

    validate_phase_response_source_checkpoint(
        {"config": {"shape_representation": "phase_moment", "phase_response_mode": "support"}},
        SimpleNamespace(shape_representation="phase_moment", phase_response_mode="support"),
    )
    with pytest.raises(ValueError, match="phase_response_mode"):
        validate_phase_response_source_checkpoint(
            {"config": {"shape_representation": "phase_moment", "phase_response_mode": "support"}},
            SimpleNamespace(shape_representation="phase_moment", phase_response_mode="independent"),
        )
    validate_phase_response_source_checkpoint(
        {"config": {"shape_representation": "phase_moment"}},
        SimpleNamespace(shape_representation="phase_moment", phase_response_mode="softmax"),
    )
    validate_phase_response_source_checkpoint(
        {"config": {"shape_representation": "phase_moment", "phase_response_mode": "b2"}},
        SimpleNamespace(shape_representation="phase_moment", phase_response_mode="b2"),
    )
    with pytest.raises(ValueError, match="phase_response_mode"):
        validate_phase_response_source_checkpoint(
            {"config": {"shape_representation": "phase_moment", "phase_response_mode": "b2"}},
            SimpleNamespace(shape_representation="phase_moment", phase_response_mode="independent"),
        )


def test_response_ab_scheduler_has_four_tasks_two_modes_and_resume_checks(capsys):
    from scripts.run_structure_g_response_ab_seed1 import (
        artifact_status, build_parser, planned_jobs, run_scheduler,
    )

    tmp_path = Path(".phase_response_ab_test")
    shutil.rmtree(tmp_path, ignore_errors=True)
    args = build_parser().parse_args([
        "dry-run", "--data-root", str(tmp_path / "missing"),
        "--output-root", str(tmp_path / "output"), "--gpus", "0", "1", "2", "3",
    ])
    jobs = planned_jobs()
    assert len(jobs) == 8
    assert {job.mode for job in jobs} == {"support", "independent"}
    assert {job.task for job in jobs} == {"AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1"}
    run_scheduler(args)
    output = capsys.readouterr().out
    assert output.count("SOURCE_PLAN|") == 8
    assert output.count("UDA_PLAN|") == 8
    assert "--phase-response-mode support" in output
    assert "--phase-response-mode independent" in output
    assert artifact_status(tmp_path / "missing_fold", "source", "support", "AT1") == "missing"
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_b2_only_scheduler_and_fixed_class_macro_f1(capsys):
    from scripts.run_structure_g_response_ab_seed1 import (
        build_parser, confusion_metrics, planned_jobs, run_scheduler,
    )

    args = build_parser().parse_args([
        "dry-run", "--variants", "B2", "--gpus", "0", "1", "2", "3",
    ])
    jobs = planned_jobs(args.variants)
    assert len(jobs) == 4
    assert {job.variant for job in jobs} == {"B2"}
    assert {job.mode for job in jobs} == {"b2"}
    run_scheduler(args)
    output = capsys.readouterr().out
    assert output.count("SOURCE_PLAN|") == 4
    assert output.count("UDA_PLAN|") == 4
    assert "--phase-response-mode b2" in output
    assert str(args.output_root).replace("\\", "/").endswith(
        "experiments/structure_b2_seed1"
    )

    metrics = confusion_metrics(torch.tensor([
        [2, 0, 0], [0, 2, 0], [0, 0, 0],
    ]).numpy())
    assert metrics["observed_macro_f1"] == pytest.approx(1.)
    assert metrics["fixed_class_macro_f1"] == pytest.approx(2 / 3)


def test_b2_frozen_check_detects_only_numerical_blockers():
    from analysis.structure_b2_frozen_check import (
        evidence_profile, frozen_check_blockers, response_summary,
    )

    response = torch.tensor([
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
    ])
    summary = response_summary(response, block_size=1)
    assert summary[0]["block"] == "presence"
    assert all(torch.isfinite(torch.tensor(row["effective_rank_raw"])) for row in summary)

    evidence = torch.tensor([[[0.], [0.], [0.], [0.]],
                             [[1.], [0.], [0.], [0.]],
                             [[1.], [0.], [1.], [0.]]])
    profile = evidence_profile(evidence)
    assert profile["zero_evidence_fraction"] == pytest.approx(1 / 3)
    assert profile["single_peak_fraction"] == pytest.approx(1 / 3)
    assert profile["multi_peak_fraction"] == pytest.approx(1 / 3)
    assert frozen_check_blockers(response, rotation_error=1e-7) == []
    assert "non_finite_response" in frozen_check_blockers(
        torch.tensor([[float("nan")]]), rotation_error=1e-7,
    )
    assert "phase_rotation_error" in frozen_check_blockers(
        response, rotation_error=1e-3,
    )


def test_b2_training_gate_requires_successful_frozen_manifest():
    import hashlib
    import json
    from scripts.run_structure_g_response_ab_seed1 import (
        TASKS, validate_frozen_manifest,
    )

    root = Path(".phase_response_b2_gate_test")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    path = root / "manifest.json"
    with pytest.raises(FileNotFoundError):
        validate_frozen_manifest(path)
    path.write_text(json.dumps([{"training_allowed": False, "blockers": ["nan"]}]))
    with pytest.raises(RuntimeError, match="blocked"):
        validate_frozen_manifest(path)
    path.write_text(json.dumps([{
        "label": "G_AT1_DK1", "task": "AT1_DK1",
        "trained_response_mode": "softmax",
        "training_allowed": True, "blockers": [],
    }]))
    with pytest.raises(RuntimeError, match="coverage"):
        validate_frozen_manifest(path)
    complete = []
    for task in TASKS:
        for variant, mode in (("G", "softmax"), ("B", "independent")):
            checkpoint = root / f"{variant}_{task}.pt"
            checkpoint.write_bytes(f"{variant}:{task}".encode())
            complete.append({
                "label": f"{variant}_{task}", "task": task,
                "trained_response_mode": mode,
                "training_allowed": True, "blockers": [],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            })
    path.write_text(json.dumps(complete))
    validate_frozen_manifest(path)
    path.write_text(json.dumps(complete + [complete[0]]))
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_frozen_manifest(path)
    complete[0]["checkpoint_sha256"] = "0" * 64
    path.write_text(json.dumps(complete))
    with pytest.raises(RuntimeError, match="SHA256"):
        validate_frozen_manifest(path)
    shutil.rmtree(root, ignore_errors=True)


def test_confusion_metrics_rejects_non_square_matrix_cleanly():
    from scripts.run_structure_g_response_ab_seed1 import confusion_metrics

    with pytest.raises(ValueError, match="square"):
        confusion_metrics(torch.zeros(2, 3, dtype=torch.int64).numpy())


def test_artifact_status_checks_mode_epoch_seed_and_source_identity():
    from scripts.run_structure_g_response_ab_seed1 import artifact_status

    root = Path(".phase_response_artifact_test")
    shutil.rmtree(root, ignore_errors=True)
    fold = root / "fold_0"; fold.mkdir(parents=True)
    packet = {
        "epoch": 99, "state_dict": {},
        "config": {
            "seed": 1, "epochs": 100, "source": "austria/33UVP/2017",
            "shape_representation": "phase_moment", "phase_response_mode": "support",
        },
    }
    torch.save(packet, fold / "checkpoint_last.pt")
    torch.save(packet, fold / "model.pt")
    assert artifact_status(fold, "source", "support", "AT1") == "complete"
    assert artifact_status(fold, "source", "independent", "AT1") == "incompatible"
    packet["epoch"] = 98
    torch.save(packet, fold / "checkpoint_last.pt")
    assert artifact_status(fold, "source", "support", "AT1") == "incompatible"
    shutil.rmtree(root, ignore_errors=True)
