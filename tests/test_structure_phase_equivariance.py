from pathlib import Path
from types import SimpleNamespace
import inspect

import pytest
import torch

from models.stclassifier import PseStructureProtoLTae
from models.structure_da.discriminative_structure import DiscriminativeStructureBranch


def _branch(representation="phase_moment"):
    return DiscriminativeStructureBranch(
        3, shape_dim=8, shapelet_count=16,
        window_scales=(24,), window_stride=8, num_modes=5,
        shape_representation=representation,
    )


def _model(representation="phase_moment"):
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation=representation, shape_injection="current_query",
        structure_shift_mode="none",
    )


def _features(batch=2):
    positions = torch.arange(16).repeat(batch, 1) * 20
    return torch.randn(batch, 16, 3), positions


def test_phase_moment_response_has_frozen_order_and_dimensions():
    branch = _branch().eval()
    features, positions = _features()
    output = branch(features, positions, phase_shift=0)
    assert output["shapelet_strength"].shape == (2, 16)
    assert output["shapelet_concentration"].shape == (2, 16)
    assert output["shapelet_phase_moments"].shape == (2, 64)
    assert output["shapelet_response"].shape == (2, 96)
    torch.testing.assert_close(
        output["shapelet_response"],
        torch.cat((
            output["shapelet_strength"], output["shapelet_concentration"],
            output["shapelet_phase_moments"],
        ), dim=-1),
    )


def test_phase_moment_uses_real_window_centers(monkeypatch):
    branch = _branch().eval()
    features, positions = _features()
    original = branch.window_extractor.forward
    calls = []

    def recording(curve, return_centers=False):
        calls.append(return_centers)
        return original(curve, return_centers=return_centers)

    monkeypatch.setattr(branch.window_extractor, "forward", recording)
    branch(features, positions, phase_shift=0)
    assert calls == [True]


def test_phase_coordinate_shift_rotates_only_moments():
    from methods.structure_da.phase_equivariance import rotate_phase_moments

    torch.manual_seed(31)
    branch = _branch().eval()
    features, positions = _features()
    base = branch(features, positions, phase_shift=0)
    shifted = branch(features, positions, phase_shift=37)
    annual = branch(features, positions, phase_shift=365)
    for key in ("shapelet_similarity", "shapelet_strength", "shapelet_concentration"):
        torch.testing.assert_close(shifted[key], base[key])
    assert not torch.allclose(shifted["shapelet_phase_moments"], base["shapelet_phase_moments"])
    expected = rotate_phase_moments(
        base["shapelet_phase_moments"], torch.full((2,), 37.),
        shapelet_count=16, harmonics=(1, 2), period_days=365.,
    )
    torch.testing.assert_close(
        shifted["shapelet_phase_moments"], expected, atol=1e-5, rtol=1e-5,
    )
    torch.testing.assert_close(
        annual["shapelet_phase_moments"], base["shapelet_phase_moments"],
        atol=1e-5, rtol=1e-5,
    )


def test_phase_model_uses_96d_classifier_and_query_while_current_is_unchanged():
    phase = _model()
    assert phase.shape_classifier.in_features == 96
    assert phase.structure_branch.response_to_query[0].in_features == 96
    current = _model("current")
    assert current.shape_classifier.in_features == 32
    assert current.structure_branch.response_to_query[0].in_features == 32
    clone = _model("current")
    clone.load_state_dict(current.state_dict(), strict=True)


def test_phase_time_match_shift_does_not_move_morphology():
    torch.manual_seed(37)
    model = _model().eval()
    pixels = torch.randn(2, 10, 3, 4)
    mask = torch.ones(2, 10, 4)
    positions = torch.arange(10).repeat(2, 1) * 30
    extra = torch.zeros(2, 4)
    base = model.forward_with_temporal_shift(
        pixels, mask, positions, extra, temporal_shift=0, return_dict=True,
    )
    shifted = model.forward_with_temporal_shift(
        pixels, mask, positions, extra, temporal_shift=25, return_dict=True,
    )
    for key in ("shapelet_similarity", "shapelet_strength", "shapelet_concentration"):
        torch.testing.assert_close(shifted[key], base[key])
    assert not torch.allclose(shifted["shapelet_phase_moments"], base["shapelet_phase_moments"])


def test_real_fourier_shift_fixes_positive_phase_rotation_sign():
    from methods.structure_da.phase_equivariance import rotate_phase_moments

    torch.manual_seed(41)
    branch = _branch().eval()
    positions = (torch.arange(64, dtype=torch.float32) * 365 / 64)[None]
    time = torch.arange(64, dtype=torch.float32) * 2 * torch.pi / 64
    features = torch.stack((
        torch.cos(time), torch.sin(time), torch.cos(2 * time + .3),
    ), dim=-1)[None]
    context = branch.prepare_context(features, positions)
    delta = 8 * 365 / 64
    with torch.no_grad():
        base = branch.forward_from_context(
            context, structure_aug_shift=0, phase_shift=0,
        )
        shifted = branch.forward_from_context(
            context, structure_aug_shift=delta, phase_shift=0,
        )
    torch.testing.assert_close(
        shifted["shapelet_similarity"],
        torch.roll(base["shapelet_similarity"], 1, dims=1),
        atol=1e-5, rtol=1e-5,
    )
    expected = rotate_phase_moments(
        base["shapelet_phase_moments"],
        torch.tensor([delta]), shapelet_count=16,
        harmonics=(1, 2), period_days=365.,
    )
    torch.testing.assert_close(
        shifted["shapelet_phase_moments"], expected, atol=2e-5, rtol=2e-5,
    )


def test_equivariance_loss_detaches_base_and_backpropagates_shifted_only():
    from methods.structure_da.phase_equivariance import phase_equivariance_loss

    base = {
        "shapelet_strength": torch.randn(3, 16, requires_grad=True),
        "shapelet_concentration": torch.rand(3, 16, requires_grad=True),
        "shapelet_phase_moments": torch.randn(3, 64, requires_grad=True),
    }
    shifted = {
        key: value.detach().clone().requires_grad_(True)
        for key, value in base.items()
    }
    losses = phase_equivariance_loss(
        base, shifted, torch.zeros(3), shapelet_count=16,
        harmonics=(1, 2), period_days=365.,
    )
    assert set(losses) == {"total_loss", "occurrence_loss", "phase_loss"}
    assert float(losses["total_loss"]) == pytest.approx(0.)
    losses["total_loss"].backward()
    assert all(value.grad is None for value in base.values())
    assert all(value.grad is not None for value in shifted.values())


def test_nonzero_artificial_shifts_are_seeded_and_bounded():
    from methods.structure_da.phase_equivariance import sample_structure_aug_shifts

    first = sample_structure_aug_shifts(
        128, 60, torch.device("cpu"), generator=torch.Generator().manual_seed(7),
    )
    second = sample_structure_aug_shifts(
        128, 60, torch.device("cpu"), generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(first, second)
    assert first.dtype == torch.long
    assert bool((first != 0).all())
    assert int(first.min()) >= -60 and int(first.max()) <= 60


def test_target_main_base_and_shifted_share_one_spatial_and_fourier_context(monkeypatch):
    torch.manual_seed(43)
    model = _model().eval()
    pixels = torch.randn(3, 10, 3, 4)
    mask = torch.ones(3, 10, 4)
    positions = torch.arange(10).repeat(3, 1) * 30
    extra = torch.zeros(3, 4)
    spatial_calls = analyzer_calls = 0
    spatial_forward = model.spatial_encoder.forward
    analyzer_forward = model.structure_branch.exposer.analyzer.forward

    def counted_spatial(*args, **kwargs):
        nonlocal spatial_calls
        spatial_calls += 1
        return spatial_forward(*args, **kwargs)

    def counted_analyzer(*args, **kwargs):
        nonlocal analyzer_calls
        analyzer_calls += 1
        return analyzer_forward(*args, **kwargs)

    monkeypatch.setattr(model.spatial_encoder, "forward", counted_spatial)
    monkeypatch.setattr(
        model.structure_branch.exposer.analyzer, "forward", counted_analyzer,
    )
    output, base, shifted = model.forward_phase_equivariance_target(
        pixels, mask, positions, extra,
        structure_aug_shift=torch.tensor([8., -8., 16.]),
    )
    assert spatial_calls == 1
    assert analyzer_calls == 1
    assert output["shapelet_response"] is base["shapelet_response"]
    assert shifted["shapelet_response"].shape == (3, 96)
    assert output["logits"].shape == (3, 3)


def test_e_target_helper_uses_full_batch_without_label_or_pseudo_arguments(monkeypatch):
    import timematch

    signature = inspect.signature(timematch.forward_target_phase_equivariance)
    forbidden = {"labels", "target_gt", "pseudo", "confidence", "pseudo_mask"}
    assert forbidden.isdisjoint(signature.parameters)
    model = _model().eval()
    pixels = torch.randn(8, 10, 3, 4)
    mask = torch.ones(8, 10, 4)
    positions = torch.arange(10).repeat(8, 1) * 30
    extra = torch.zeros(8, 4)
    fixed_shift = torch.tensor([-4, -3, -2, -1, 1, 2, 3, 4])
    monkeypatch.setattr(
        timematch, "sample_structure_aug_shifts",
        lambda batch, maximum, device: fixed_shift.to(device),
    )
    output, losses, delta = timematch.forward_target_phase_equivariance(
        model, pixels, mask, positions, extra, max_shift=60,
    )
    assert output["logits"].shape[0] == 8
    assert torch.equal(delta.cpu(), fixed_shift)
    assert torch.isfinite(losses["total_loss"])
    assert float(losses["total_loss"]) > 0


def test_synthetic_p_and_e_backward_are_finite():
    import timematch

    torch.manual_seed(47)
    model = _model().train()
    pixels = torch.randn(4, 10, 3, 4)
    mask = torch.ones(4, 10, 4)
    positions = torch.arange(10).repeat(4, 1) * 30
    extra = torch.zeros(4, 4)

    p_output = model.forward_with_temporal_shift(
        pixels, mask, positions, extra, temporal_shift=13, return_dict=True,
    )
    p_loss = p_output["logits"].square().mean() + p_output["shape_logits"].square().mean()
    assert torch.isfinite(p_loss)
    p_loss.backward()
    model.zero_grad(set_to_none=True)

    e_output, e_losses, _ = timematch.forward_target_phase_equivariance(
        model, pixels, mask, positions, extra, max_shift=60,
    )
    e_loss = e_output["logits"].square().mean() + .05 * e_losses["total_loss"]
    assert torch.isfinite(e_loss) and float(e_losses["total_loss"]) > 0
    e_loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.structure_branch.parameters()
    )


def test_v2clean_loss_adds_only_weighted_ramped_equivariance():
    from methods.structure_da.prototype_losses import compose_structure_v2clean_da_loss

    values = [torch.tensor(value) for value in (1., 2., 3., 4., 5.)]
    equivariance = torch.tensor(7.)
    p = compose_structure_v2clean_da_loss(
        *values, ramp=.4, shape_align_weight=0.,
        shape_equivariance=equivariance, shape_equivariance_weight=0.,
    )
    e = compose_structure_v2clean_da_loss(
        *values, ramp=.4, shape_align_weight=0.,
        shape_equivariance=equivariance, shape_equivariance_weight=.05,
    )
    assert float(e - p) == pytest.approx(.4 * .05 * 7.)


def test_phase_source_checkpoint_is_shared_by_p_and_e_but_current_is_incompatible():
    source = _model()
    p = _model()
    e = _model()
    p.load_state_dict(source.state_dict(), strict=True)
    e.load_state_dict(source.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        _model().load_state_dict(_model("current").state_dict(), strict=True)


def test_phase_manifest_and_launcher_freeze_p_and_e_contract():
    from train import structure_usage_manifest

    manifest = structure_usage_manifest(SimpleNamespace(
        shape_representation="phase_moment", shape_injection="current_query",
        structure_shift_mode="none", shapelet_count=16,
        shape_window_scales=[24], shape_window_stride=8,
        shape_align_weight=0., shape_equivariance_weight=.05,
        shape_equivariance_max_shift=60,
    ))
    assert manifest["shape_evidence_dim"] == 96
    assert manifest["phase_harmonics"] == [1, 2]
    assert manifest["phase_period_days"] == 365
    assert manifest["occurrence_phase"] is True
    assert manifest["structure_shift_mode"] == "none"
    launcher = Path(
        "scripts/run_structure_phase_equivariance_4tasks_4gpu_seed1.sh"
    ).read_text()
    assert launcher.count("--shape-representation phase_moment") == 3
    assert launcher.count("--shape-alignment-view none") == 2
    assert launcher.count("--shape-align-weight 0") == 2
    assert "--shape-equivariance-weight 0" in launcher
    assert "--shape-equivariance-weight 0.05" in launcher
    assert "--shape-equivariance-max-shift 60" in launcher
    assert 'P_SOURCE_ROOT="$P_ROOT/source"' in launcher


def test_phase_usage_diagnostics_report_both_harmonic_magnitudes():
    model = _model().eval()
    pixels = torch.randn(4, 10, 3, 4)
    mask = torch.ones(4, 10, 4)
    positions = torch.arange(10).repeat(4, 1) * 30
    output = model(
        pixels, mask, positions, torch.zeros(4, 4), return_dict=True,
    )
    diagnostics = model.structure_usage_diagnostics(output)
    assert set(diagnostics) == {
        "phase_k1_mean_magnitude", "phase_k2_mean_magnitude",
    }
    assert all(torch.isfinite(value) and float(value) >= 0 for value in diagnostics.values())


def test_phase_uda_config_rejects_explicit_alignment_and_invalid_e_settings():
    import timematch

    base = dict(
        shape_representation="phase_moment", shape_alignment_view="none",
        shape_align_weight=0., shape_equivariance_weight=0.,
        shape_equivariance_max_shift=60,
    )
    timematch.validate_phase_equivariance_config(SimpleNamespace(**base))
    with pytest.raises(ValueError, match="explicit shape alignment"):
        timematch.validate_phase_equivariance_config(SimpleNamespace(
            **{**base, "shape_align_weight": .05},
        ))
    with pytest.raises(ValueError, match="max shift"):
        timematch.validate_phase_equivariance_config(SimpleNamespace(
            **{**base, "shape_equivariance_weight": .05,
               "shape_equivariance_max_shift": 0},
        ))


def test_formal_v2clean_loop_calls_e_only_when_weight_is_positive():
    import timematch

    source = inspect.getsource(timematch._train_structure_proto_timematch)
    assert "forward_target_phase_equivariance(" in source
    assert "shape_equivariance_weight > 0" in source
    assert "shape_equivariance=equivariance[\"total_loss\"]" in source
    assert "shape_equivariance_weight=shape_equivariance_weight" in source
    for metric in (
        "loss_shape_equivariance", "loss_shape_equiv_occ",
        "loss_shape_equiv_phase", "equiv_shift_abs_mean",
    ):
        assert metric in source


def test_launcher_has_strict_three_round_barriers_and_four_task_mapping():
    source = Path(
        "scripts/run_structure_phase_equivariance_4tasks_4gpu_seed1.sh"
    ).read_text()
    assert source.index("run_source \"$GPU0\"") < source.index("run_p \"$GPU0\"")
    assert source.index("run_p \"$GPU0\"") < source.index("run_e \"$GPU0\"")
    for command in ("run_source", "run_p", "run_e"):
        assert f'{command} "$GPU0" AT1' in source
        assert f'{command} "$GPU1" FR1' in source
        assert f'{command} "$GPU2" FR2' in source
        assert f'{command} "$GPU3" DK1' in source
    assert source.count("wait \"$pid\" ||") == 3

