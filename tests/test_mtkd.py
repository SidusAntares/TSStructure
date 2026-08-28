import copy
import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.mtkd import (
    MTKDEarlyConcatLTAE,
    MTKDMidConcatLTAE,
    MTKDSOnlyLTAE,
    MTKDTDMidConcatLTAE,
    MTKDTQMidConcatLTAE,
    MTKDTQSingleLTAE,
    MTKDTQLOOLTAE,
    MultiScaleTemporalKernelDecomposition,
    LearnableSingleScaleTemporalKernel,
    SingleScaleTemporalKernel,
    compute_feature_curve_diagnostics,
    leave_one_out_smooth_l1,
    log_mtkd_diagnostics,
)
from models.ltae import LTAE
from models.decoder import MTKDLateLogitDecoder
from models.stclassifier import (
    PseLTae,
    PseMTKDLateLtae,
    PseMTKDLtae,
    PseMTKDMidLtae,
    PseMTKDSOnlyLtae,
    PseMTKDTDMidLtae,
    PseMTKDTQMidLtae,
    PseMTKDTQSingleLtae,
    PseMTKDTQLOOLtae,
)
from train import _add_model_arguments, _build_model, _optimizer_parameter_groups
from timematch import _domain_balanced_loo


def _make_mtkd(**kwargs):
    defaults = dict(
        time_scale_days=365.0,
        tau_fast_init_days=30.0,
        tau_slow_init_days=90.0,
        tau_min_days=1.0,
        delta_tau_min_days=1.0,
        learnable_tau=True,
    )
    defaults.update(kwargs)
    return MultiScaleTemporalKernelDecomposition(**defaults)


def _model_inputs(batch_size=2, seq_len=5, input_dim=10, num_pixels=7):
    pixels = torch.randn(batch_size, seq_len, input_dim, num_pixels)
    valid_pixels = torch.ones(batch_size, seq_len, num_pixels)
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1) * 17
    extra = torch.randn(batch_size, 4)
    return pixels, valid_pixels, positions, extra


def test_mtkd_outputs_match_input_shape_and_preserve_constant_signal():
    module = _make_mtkd().double()
    h = torch.full((2, 6, 4), 3.25, dtype=torch.float64)
    positions = torch.tensor(
        [[1.0, 4.0, 20.0, 55.0, 130.0, 300.0]] * 2,
        dtype=torch.float64,
    )

    t, s = module(h, positions)

    assert t.shape == h.shape
    assert s.shape == h.shape
    torch.testing.assert_close(t, h)
    torch.testing.assert_close(s, h)
    assert t.dtype == h.dtype
    assert t.device == h.device


def test_debug_components_satisfy_decomposition_identities():
    module = _make_mtkd()
    h = torch.randn(3, 7, 5)
    positions = torch.randint(0, 365, (3, 7))

    components = module.debug_components(h, positions)

    t, s, d, r = (components[name] for name in ("T", "S", "D", "R"))
    torch.testing.assert_close(d, s - t)
    torch.testing.assert_close(h, s + r)
    torch.testing.assert_close(h, t + d + r)


def test_tau_initialization_is_exact_and_ordered():
    module = _make_mtkd()

    tau_fast, tau_slow = module.get_tau_days()

    torch.testing.assert_close(tau_fast, torch.tensor(30.0))
    torch.testing.assert_close(tau_slow, torch.tensor(90.0))
    assert tau_fast > module.tau_min_days > 0
    assert tau_slow > tau_fast + module.delta_tau_min_days


def test_learnable_tau_receives_finite_gradients():
    module = _make_mtkd()
    h = torch.randn(2, 8, 3, requires_grad=True)
    positions = torch.tensor(
        [[0, 4, 11, 29, 61, 120, 210, 350]] * 2,
        dtype=torch.float32,
    )

    t, s = module(h, positions)
    (t.square().mean() + s.abs().mean()).backward()

    assert module.a.grad is not None and torch.isfinite(module.a.grad).all()
    assert module.b.grad is not None and torch.isfinite(module.b.grad).all()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_global_time_shift_does_not_change_decomposition():
    module = _make_mtkd()
    h = torch.randn(2, 6, 9)
    positions = torch.tensor([[2, 7, 19, 80, 150, 330]] * 2)

    original = module(h, positions)
    shifted = module(h, positions + 47)

    torch.testing.assert_close(original[0], shifted[0])
    torch.testing.assert_close(original[1], shifted[1])


def test_temporal_encoder_concatenates_t_then_s_before_single_ltae():
    encoder = MTKDEarlyConcatLTAE(in_channels=128).eval()
    spatial_feats = torch.randn(2, 5, 128)
    positions = torch.arange(5).unsqueeze(0).repeat(2, 1)
    captured = {}

    def capture_ltae_input(_module, args):
        captured["input"] = args[0].detach().clone()

    handle = encoder.ltae.register_forward_pre_hook(capture_ltae_input)
    try:
        temporal_feats = encoder(spatial_feats, positions)
    finally:
        handle.remove()

    t, s = encoder.mtkd(spatial_feats, positions)
    expected = torch.cat([t, s], dim=-1)
    assert captured["input"].shape == (2, 5, 256)
    torch.testing.assert_close(captured["input"], expected)
    assert temporal_feats.shape == (2, 128)
    assert encoder.ltae.in_channels == 256


def test_temporal_encoder_forwards_timematch_properties_without_reregistering_embedding():
    encoder = MTKDEarlyConcatLTAE(in_channels=128)

    assert encoder.max_temporal_shift == encoder.ltae.max_temporal_shift
    assert encoder.positional_enc is encoder.ltae.positional_enc
    embedding_keys = [
        key for key in encoder.state_dict() if key.endswith("positional_enc.weight")
    ]
    assert embedding_keys == ["ltae.positional_enc.weight"]


def test_pse_mtkd_ltae_logits_return_feats_and_manual_timematch_path_match():
    model = PseMTKDLtae(input_dim=10, with_extra=False, num_classes=6).eval()
    pixels, valid_pixels, positions, extra = _model_inputs()

    with torch.no_grad():
        logits = model(pixels, valid_pixels, positions, extra)
        returned_logits, temporal_feats = model(
            pixels, valid_pixels, positions, extra, return_feats=True
        )
        spatial = model.spatial_encoder(pixels, valid_pixels, extra)
        manual_temporal = model.temporal_encoder(spatial, positions)
        manual_logits = model.decoder(manual_temporal)

    assert spatial.shape == (2, 5, 128)
    assert logits.shape == (2, 6)
    assert temporal_feats.shape == (2, 128)
    torch.testing.assert_close(logits, returned_logits)
    torch.testing.assert_close(logits, manual_logits)
    torch.testing.assert_close(temporal_feats, manual_temporal)


def test_pse_mtkd_ltae_state_dict_round_trip_preserves_output():
    model = PseMTKDLtae(input_dim=10, with_extra=False, num_classes=6).eval()
    restored = PseMTKDLtae(input_dim=10, with_extra=False, num_classes=6).eval()
    inputs = _model_inputs()

    with torch.no_grad():
        expected = model(*inputs)
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        actual = restored(*inputs)

    torch.testing.assert_close(actual, expected)


def test_existing_pseltae_state_dict_and_parameter_count_are_unchanged_by_new_model():
    before = PseLTae(input_dim=10, with_extra=False, num_classes=6)
    after = PseLTae(input_dim=10, with_extra=False, num_classes=6)

    assert before.state_dict().keys() == after.state_dict().keys()
    assert sum(p.numel() for p in before.parameters()) == sum(
        p.numel() for p in after.parameters()
    )


def test_train_build_model_keeps_pseltae_independent_of_mtkd_arguments():
    config = SimpleNamespace(
        model="pseltae", input_dim=10, num_classes=6, with_extra=False
    )

    model = _build_model(config)

    assert isinstance(model, PseLTae)


def test_train_build_model_passes_mtkd_configuration():
    config = SimpleNamespace(
        model="psemtkdltae",
        input_dim=10,
        num_classes=6,
        with_extra=False,
        mtkd_time_scale_days=400.0,
        mtkd_tau_fast_init_days=20.0,
        mtkd_tau_slow_init_days=75.0,
        mtkd_tau_min_days=2.0,
        mtkd_delta_tau_min_days=3.0,
        mtkd_learnable_tau=False,
    )

    model = _build_model(config)
    tau_fast, tau_slow = model.temporal_encoder.mtkd.get_tau_days()

    assert isinstance(model, PseMTKDLtae)
    torch.testing.assert_close(tau_fast, torch.tensor(20.0))
    torch.testing.assert_close(tau_slow, torch.tensor(75.0))
    assert model.temporal_encoder.mtkd.time_scale_days == 400.0
    assert not isinstance(model.temporal_encoder.mtkd.a, torch.nn.Parameter)
    assert not isinstance(model.temporal_encoder.mtkd.b, torch.nn.Parameter)


def test_mtkd_cli_arguments_have_pilot_defaults_and_boolean_parsing():
    parser = argparse.ArgumentParser()
    _add_model_arguments(parser)

    defaults = parser.parse_args([])
    fixed_tau = parser.parse_args(
        ["--model", "psemtkdltae", "--mtkd_learnable_tau", "false"]
    )

    assert "psemtkdltae" in parser._option_string_actions["--model"].choices
    assert defaults.mtkd_time_scale_days == 365.0
    assert defaults.mtkd_tau_fast_init_days == 30.0
    assert defaults.mtkd_tau_slow_init_days == 90.0
    assert defaults.mtkd_tau_min_days == 1.0
    assert defaults.mtkd_delta_tau_min_days == 1.0
    assert defaults.mtkd_learnable_tau is True
    assert fixed_tau.mtkd_learnable_tau is False


def test_mtkd_diagnostics_report_finite_ordered_tau_and_component_metrics():
    encoder = MTKDEarlyConcatLTAE(in_channels=128)
    t = torch.randn(3, 5, 128)
    s = t + 0.25 * torch.randn(3, 5, 128)

    encoder.reset_diagnostics()
    encoder.record_diagnostics(t, s)
    diagnostics = encoder.get_diagnostics()

    assert math.isfinite(diagnostics["tau_fast_days"])
    assert math.isfinite(diagnostics["tau_slow_days"])
    assert diagnostics["delta_tau_days"] > 0
    assert math.isfinite(diagnostics["ts_relative_difference"])
    assert math.isfinite(diagnostics["ts_cosine_similarity"])
    assert math.isfinite(diagnostics["t_to_s_norm_ratio"])


def test_identical_mtkd_components_have_zero_difference_and_unit_cosine():
    encoder = MTKDEarlyConcatLTAE(in_channels=128)
    component = torch.full((2, 4, 128), 3.0)

    encoder.reset_diagnostics()
    encoder.record_diagnostics(component, component)
    diagnostics = encoder.get_diagnostics()

    assert diagnostics["ts_relative_difference"] == pytest.approx(0.0, abs=1e-7)
    assert diagnostics["ts_cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert diagnostics["t_to_s_norm_ratio"] == pytest.approx(1.0, abs=1e-6)


def test_mtkd_diagnostics_do_not_change_logits_or_checkpoint_state():
    model = PseMTKDLtae(input_dim=10, with_extra=False, num_classes=6).eval()
    inputs = _model_inputs()
    original_keys = set(model.state_dict())

    with torch.no_grad():
        expected = model(*inputs)
        component = torch.randn(2, 5, 128)
        model.temporal_encoder.record_diagnostics(component, component)
        actual = model(*inputs)

    torch.testing.assert_close(actual, expected)
    assert set(model.state_dict()) == original_keys
    assert not any("diagnostic" in key for key in model.state_dict())


def test_mtkd_diagnostics_print_epoch_block_and_write_tensorboard(capsys):
    model = PseMTKDLtae(input_dim=10, with_extra=False, num_classes=6)
    component = torch.ones(2, 3, 128)
    model.temporal_encoder.record_diagnostics(component, component)
    scalar_calls = []
    writer = SimpleNamespace(
        add_scalar=lambda name, value, step: scalar_calls.append((name, value, step))
    )

    diagnostics = log_mtkd_diagnostics(model, "source", 1, writer)
    output = capsys.readouterr().out

    assert output.startswith("\n\n")
    assert "MTKD DIAGNOSTICS" in output
    assert "stage                  : source" in output
    assert "epoch                  : 1" in output
    assert output.endswith("\n\n\n")
    assert len(scalar_calls) == 6
    assert {name for name, _value, _step in scalar_calls} == {
        "mtkd/tau_fast_days",
        "mtkd/tau_slow_days",
        "mtkd/delta_tau_days",
        "mtkd/ts_relative_difference",
        "mtkd/ts_cosine_similarity",
        "mtkd/t_to_s_norm_ratio",
    }
    assert diagnostics["delta_tau_days"] > 0


def test_mtkd_launcher_uses_only_gpu_1_2_3_and_disables_progress_bars():
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "run_mtkd_ts_9tasks_3gpu.sh"
    text = launcher.read_text(encoding="utf-8")

    assert "GPU_IDS=(1 2 3)" in text
    assert "SOURCE_DOMAINS=(AT1 FR1 FR2)" in text
    assert "--progress_bar off" in text
    assert "--model psemtkdltae" in text
    assert "CUDA_VISIBLE_DEVICES=\"$gpu_id\"" in text
    assert "DRY_RUN" in text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_scale_days": 0.0},
        {"tau_min_days": 0.0},
        {"delta_tau_min_days": 0.0},
        {"tau_fast_init_days": 1.0},
        {"tau_slow_init_days": 31.0},
    ],
)
def test_mtkd_rejects_invalid_time_scales(kwargs):
    with pytest.raises(ValueError):
        _make_mtkd(**kwargs)



def test_mid_temporal_encoder_uses_independent_ltaes_then_concatenates_features():
    encoder = MTKDMidConcatLTAE(in_channels=128).eval()
    spatial_feats = torch.randn(2, 5, 128)
    positions = torch.arange(5).unsqueeze(0).repeat(2, 1) * 17

    with torch.no_grad():
        t, s = encoder.mtkd(spatial_feats, positions)
        expected_t = encoder.ltae_t(t, positions)
        expected_s = encoder.ltae_s(s, positions)
        actual = encoder(spatial_feats, positions)

    assert encoder.ltae_t is not encoder.ltae_s
    assert encoder.ltae_t.in_channels == 128
    assert encoder.ltae_s.in_channels == 128
    assert expected_t.shape == (2, 128)
    assert expected_s.shape == (2, 128)
    assert actual.shape == (2, 256)
    torch.testing.assert_close(actual, torch.cat([expected_t, expected_s], dim=-1))

    t_param = next(encoder.ltae_t.parameters())
    s_param = next(encoder.ltae_s.parameters())
    assert t_param is not s_param
    assert t_param.data_ptr() != s_param.data_ptr()


def test_mid_temporal_encoder_forwards_timematch_properties_and_keeps_embeddings_independent():
    encoder = MTKDMidConcatLTAE(in_channels=128)

    assert encoder.max_temporal_shift == encoder.ltae_t.max_temporal_shift
    assert encoder.positional_enc is encoder.ltae_t.positional_enc
    assert encoder.ltae_t.positional_enc is not encoder.ltae_s.positional_enc
    assert (
        encoder.ltae_t.positional_enc.num_embeddings
        == encoder.ltae_s.positional_enc.num_embeddings
    )
    embedding_keys = [
        key for key in encoder.state_dict() if key.endswith("positional_enc.weight")
    ]
    assert embedding_keys == [
        "ltae_t.positional_enc.weight",
        "ltae_s.positional_enc.weight",
    ]


def test_pse_mtkd_mid_ltae_logits_return_feats_and_manual_timematch_path_match():
    model = PseMTKDMidLtae(input_dim=10, with_extra=False, num_classes=6).eval()
    pixels, valid_pixels, positions, extra = _model_inputs()

    with torch.no_grad():
        logits = model(pixels, valid_pixels, positions, extra)
        returned_logits, temporal_feats = model(
            pixels, valid_pixels, positions, extra, return_feats=True
        )
        spatial = model.spatial_encoder(pixels, valid_pixels, extra)
        manual_temporal = model.temporal_encoder(spatial, positions)
        manual_logits = model.decoder(manual_temporal)

    assert spatial.shape == (2, 5, 128)
    assert temporal_feats.shape == (2, 256)
    assert logits.shape == (2, 6)
    assert model.decoder[0].linear.in_features == 256
    torch.testing.assert_close(logits, returned_logits)
    torch.testing.assert_close(logits, manual_logits)
    torch.testing.assert_close(temporal_feats, manual_temporal)


def test_train_build_model_passes_mtkd_configuration_to_mid_model():
    config = SimpleNamespace(
        model="psemtkdmidltae",
        input_dim=10,
        num_classes=6,
        with_extra=False,
        mtkd_time_scale_days=400.0,
        mtkd_tau_fast_init_days=20.0,
        mtkd_tau_slow_init_days=75.0,
        mtkd_tau_min_days=2.0,
        mtkd_delta_tau_min_days=3.0,
        mtkd_learnable_tau=False,
    )

    model = _build_model(config)
    tau_fast, tau_slow = model.temporal_encoder.mtkd.get_tau_days()

    assert isinstance(model, PseMTKDMidLtae)
    torch.testing.assert_close(tau_fast, torch.tensor(20.0))
    torch.testing.assert_close(tau_slow, torch.tensor(75.0))
    assert model.temporal_encoder.mtkd.time_scale_days == 400.0
    assert not isinstance(model.temporal_encoder.mtkd.a, torch.nn.Parameter)
    assert not isinstance(model.temporal_encoder.mtkd.b, torch.nn.Parameter)


def test_mid_model_is_registered_and_launcher_uses_expected_configuration():
    parser = argparse.ArgumentParser()
    _add_model_arguments(parser)
    choices = parser._option_string_actions["--model"].choices
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_mtkd_ts_mid_9tasks_3gpu.sh"
    )
    text = launcher.read_text(encoding="utf-8")

    assert "psemtkdmidltae" in choices
    assert "GPU_IDS=(1 2 3)" in text
    assert "SOURCE_DOMAINS=(AT1 FR1 FR2)" in text
    assert "--model psemtkdmidltae" in text
    assert "--progress_bar off" in text
    assert "mtkd_ts_mid_9tasks" in text
    assert "run_mtkd_ts_mid_9tasks_3gpu.sh" in text


def test_mid_model_uses_existing_mtkd_diagnostics_helpers(capsys):
    model = PseMTKDMidLtae(input_dim=10, with_extra=False, num_classes=6)
    component = torch.ones(2, 3, 128)
    model.temporal_encoder.record_diagnostics(component, component)

    diagnostics = log_mtkd_diagnostics(model, "source", 1)
    output = capsys.readouterr().out

    assert diagnostics is not None
    assert diagnostics["ts_relative_difference"] == pytest.approx(0.0, abs=1e-7)
    assert "MTKD DIAGNOSTICS" in output


def test_late_decoder_uses_independent_classifiers_and_sums_raw_logits():
    decoder = MTKDLateLogitDecoder([128, 64, 32], num_classes=6).eval()
    temporal_feats = torch.randn(3, 256)
    z_t = temporal_feats[:, :128]
    z_s = temporal_feats[:, 128:]

    with torch.no_grad():
        logits_t = decoder.classifier_t(z_t)
        logits_s = decoder.classifier_s(z_s)
        actual = decoder(temporal_feats)

    assert decoder.classifier_t is not decoder.classifier_s
    assert next(decoder.classifier_t.parameters()).data_ptr() != next(
        decoder.classifier_s.parameters()
    ).data_ptr()
    assert logits_t.shape == (3, 6)
    assert logits_s.shape == (3, 6)
    torch.testing.assert_close(actual, logits_t + logits_s)


def test_late_model_reuses_mid_encoder_and_passes_identical_positions_to_both_ltaes():
    model = PseMTKDLateLtae(input_dim=10, with_extra=False, num_classes=6).eval()
    spatial_feats = torch.randn(2, 5, 128)
    positions = torch.arange(5).unsqueeze(0).repeat(2, 1) * 17
    captured = {}

    def capture_positions(name):
        def hook(_module, args):
            captured[name] = args[1]

        return hook

    handle_t = model.temporal_encoder.ltae_t.register_forward_pre_hook(
        capture_positions("t")
    )
    handle_s = model.temporal_encoder.ltae_s.register_forward_pre_hook(
        capture_positions("s")
    )
    try:
        with torch.no_grad():
            temporal_feats = model.temporal_encoder(spatial_feats, positions)
    finally:
        handle_t.remove()
        handle_s.remove()

    assert isinstance(model.temporal_encoder, MTKDMidConcatLTAE)
    assert captured["t"] is positions
    assert captured["s"] is positions
    assert captured["t"] is captured["s"]
    assert temporal_feats.shape == (2, 256)
    assert isinstance(model.temporal_encoder.mtkd, MultiScaleTemporalKernelDecomposition)
    assert sum(
        isinstance(module, MultiScaleTemporalKernelDecomposition)
        for module in model.modules()
    ) == 1


def test_pse_mtkd_late_ltae_return_feats_and_manual_timematch_path_match():
    model = PseMTKDLateLtae(input_dim=10, with_extra=False, num_classes=6).eval()
    pixels, valid_pixels, positions, extra = _model_inputs()

    with torch.no_grad():
        full_logits = model(pixels, valid_pixels, positions, extra)
        returned_logits, temporal_feats = model(
            pixels, valid_pixels, positions, extra, return_feats=True
        )
        spatial = model.spatial_encoder(pixels, valid_pixels, extra)
        temporal = model.temporal_encoder(spatial, positions)
        manual_logits = model.decoder(temporal)
        logits_t = model.decoder.classifier_t(temporal[:, :128])
        logits_s = model.decoder.classifier_s(temporal[:, 128:])

    assert temporal_feats.shape == (2, 256)
    assert full_logits.shape == (2, 6)
    torch.testing.assert_close(full_logits, returned_logits)
    torch.testing.assert_close(full_logits, manual_logits)
    torch.testing.assert_close(full_logits, logits_t + logits_s)
    torch.testing.assert_close(temporal_feats, temporal)


def test_train_build_model_passes_mtkd_configuration_to_late_model():
    config = SimpleNamespace(
        model="psemtkdlateltae",
        input_dim=10,
        num_classes=6,
        with_extra=False,
        mtkd_time_scale_days=400.0,
        mtkd_tau_fast_init_days=20.0,
        mtkd_tau_slow_init_days=75.0,
        mtkd_tau_min_days=2.0,
        mtkd_delta_tau_min_days=3.0,
        mtkd_learnable_tau=False,
    )

    model = _build_model(config)
    tau_fast, tau_slow = model.temporal_encoder.mtkd.get_tau_days()

    assert isinstance(model, PseMTKDLateLtae)
    torch.testing.assert_close(tau_fast, torch.tensor(20.0))
    torch.testing.assert_close(tau_slow, torch.tensor(75.0))
    assert not isinstance(model.temporal_encoder.mtkd.a, torch.nn.Parameter)
    assert not isinstance(model.temporal_encoder.mtkd.b, torch.nn.Parameter)


def test_late_model_is_registered_and_launcher_uses_expected_configuration():
    parser = argparse.ArgumentParser()
    _add_model_arguments(parser)
    choices = parser._option_string_actions["--model"].choices
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_mtkd_ts_late_9tasks_3gpu.sh"
    )
    text = launcher.read_text(encoding="utf-8")

    assert "psemtkdlateltae" in choices
    assert "GPU_IDS=(1 2 3)" in text
    assert "SOURCE_DOMAINS=(AT1 FR1 FR2)" in text
    assert "--model psemtkdlateltae" in text
    assert "--progress_bar off" in text
    assert "mtkd_ts_late_9tasks" in text
    assert "run_mtkd_ts_late_9tasks_3gpu.sh" in text
    assert "psemtkdmidltae" not in text


def test_s_only_temporal_encoder_matches_direct_ltae_of_s():
    encoder = MTKDSOnlyLTAE(in_channels=128).eval()
    spatial_feats = torch.randn(2, 5, 128)
    positions = torch.arange(5).unsqueeze(0).repeat(2, 1) * 17

    with torch.no_grad():
        t, s = encoder.mtkd(spatial_feats, positions)
        expected = encoder.ltae(s, positions)
        actual = encoder(spatial_feats, positions)

    assert t.shape == (2, 5, 128)
    assert s.shape == (2, 5, 128)
    assert expected.shape == (2, 128)
    assert actual.shape == (2, 128)
    torch.testing.assert_close(actual, expected)


def test_s_only_temporal_encoder_downstream_output_does_not_depend_on_t(monkeypatch):
    encoder = MTKDSOnlyLTAE(in_channels=128).eval()
    spatial_feats = torch.randn(2, 5, 128)
    positions = torch.arange(5).unsqueeze(0).repeat(2, 1) * 17
    s = torch.randn_like(spatial_feats)

    monkeypatch.setattr(
        encoder.mtkd,
        "forward",
        lambda _spatial, _positions: (torch.zeros_like(s), s),
    )
    with torch.no_grad():
        first = encoder(spatial_feats, positions)

    monkeypatch.setattr(
        encoder.mtkd,
        "forward",
        lambda _spatial, _positions: (torch.randn_like(s) * 1000.0, s),
    )
    with torch.no_grad():
        second = encoder(spatial_feats, positions)

    torch.testing.assert_close(first, second)


def test_pse_mtkd_s_only_ltae_shapes_return_feats_and_manual_path_match():
    model = PseMTKDSOnlyLtae(input_dim=10, with_extra=False, num_classes=6).eval()
    pixels, valid_pixels, positions, extra = _model_inputs()

    with torch.no_grad():
        full_logits = model(pixels, valid_pixels, positions, extra)
        returned_logits, temporal_feats = model(
            pixels, valid_pixels, positions, extra, return_feats=True
        )
        spatial = model.spatial_encoder(pixels, valid_pixels, extra)
        t, s = model.temporal_encoder.mtkd(spatial, positions)
        direct_s_feats = model.temporal_encoder.ltae(s, positions)
        manual_temporal = model.temporal_encoder(spatial, positions)
        manual_logits = model.decoder(manual_temporal)

    assert spatial.shape == (2, 5, 128)
    assert t.shape == (2, 5, 128)
    assert s.shape == (2, 5, 128)
    assert temporal_feats.shape == (2, 128)
    assert full_logits.shape == (2, 6)
    assert model.decoder[0].linear.in_features == 128
    assert sum(isinstance(module, LTAE) for module in model.modules()) == 1
    assert sum(
        isinstance(module, MultiScaleTemporalKernelDecomposition)
        for module in model.modules()
    ) == 1
    torch.testing.assert_close(full_logits, returned_logits)
    torch.testing.assert_close(full_logits, manual_logits)
    torch.testing.assert_close(manual_temporal, direct_s_feats)


def test_train_build_model_passes_mtkd_configuration_to_s_only_model():
    config = SimpleNamespace(
        model="psemtkdsltae",
        input_dim=10,
        num_classes=6,
        with_extra=False,
        mtkd_time_scale_days=400.0,
        mtkd_tau_fast_init_days=20.0,
        mtkd_tau_slow_init_days=75.0,
        mtkd_tau_min_days=2.0,
        mtkd_delta_tau_min_days=3.0,
        mtkd_learnable_tau=False,
    )

    model = _build_model(config)
    tau_fast, tau_slow = model.temporal_encoder.mtkd.get_tau_days()

    assert isinstance(model, PseMTKDSOnlyLtae)
    torch.testing.assert_close(tau_fast, torch.tensor(20.0))
    torch.testing.assert_close(tau_slow, torch.tensor(75.0))
    assert not isinstance(model.temporal_encoder.mtkd.a, torch.nn.Parameter)
    assert not isinstance(model.temporal_encoder.mtkd.b, torch.nn.Parameter)


def test_s_only_model_keeps_existing_mtkd_diagnostics(capsys):
    model = PseMTKDSOnlyLtae(input_dim=10, with_extra=False, num_classes=6)
    component = torch.ones(2, 3, 128)
    model.temporal_encoder.record_diagnostics(component, component)

    diagnostics = log_mtkd_diagnostics(model, "source", 1)
    output = capsys.readouterr().out

    assert diagnostics is not None
    assert set(diagnostics) == {
        "tau_fast_days",
        "tau_slow_days",
        "delta_tau_days",
        "ts_relative_difference",
        "ts_cosine_similarity",
        "t_to_s_norm_ratio",
    }
    assert "MTKD DIAGNOSTICS" in output


def test_s_only_model_is_registered_and_launcher_uses_expected_configuration():
    parser = argparse.ArgumentParser()
    _add_model_arguments(parser)
    choices = parser._option_string_actions["--model"].choices
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_mtkd_s_only_9tasks_3gpu.sh"
    )
    text = launcher.read_text(encoding="utf-8")

    assert "psemtkdsltae" in choices
    assert "GPU_IDS=(1 2 3)" in text
    assert "SOURCE_DOMAINS=(AT1 FR1 FR2)" in text
    assert "--model psemtkdsltae" in text
    assert "--progress_bar off" in text
    assert "mtkd_s_only_9tasks" in text
    assert "run_mtkd_s_only_9tasks_3gpu.sh" in text
    assert "psemtkdltae" not in text
    assert "psemtkdmidltae" not in text
    assert "psemtkdlateltae" not in text


def test_td_mid_temporal_encoder_uses_explicit_d_and_independent_ltaes():
    encoder = MTKDTDMidConcatLTAE(in_channels=128).eval()
    spatial_feats = torch.randn(2, 5, 128)
    positions = torch.arange(5).unsqueeze(0).repeat(2, 1) * 17

    with torch.no_grad():
        t, s = encoder.mtkd(spatial_feats, positions)
        d = s - t
        expected_t = encoder.ltae_t(t, positions)
        expected_d = encoder.ltae_d(d, positions)
        actual = encoder(spatial_feats, positions)

    assert t.shape == (2, 5, 128)
    assert s.shape == (2, 5, 128)
    assert d.shape == (2, 5, 128)
    torch.testing.assert_close(t + d, s)
    assert expected_t.shape == (2, 128)
    assert expected_d.shape == (2, 128)
    assert actual.shape == (2, 256)
    torch.testing.assert_close(actual, torch.cat([expected_t, expected_d], dim=-1))

    assert encoder.ltae_t is not encoder.ltae_d
    t_param = next(encoder.ltae_t.parameters())
    d_param = next(encoder.ltae_d.parameters())
    assert t_param is not d_param
    assert t_param.data_ptr() != d_param.data_ptr()
    assert sum(
        isinstance(module, MultiScaleTemporalKernelDecomposition)
        for module in encoder.modules()
    ) == 1


def test_td_mid_temporal_encoder_passes_identical_positions_to_both_branches():
    encoder = MTKDTDMidConcatLTAE(in_channels=128).eval()
    spatial_feats = torch.randn(2, 5, 128)
    positions = torch.arange(5).unsqueeze(0).repeat(2, 1) * 17
    captured = {}

    def capture_positions(name):
        def hook(_module, args):
            captured[name] = args[1]

        return hook

    handle_t = encoder.ltae_t.register_forward_pre_hook(capture_positions("t"))
    handle_d = encoder.ltae_d.register_forward_pre_hook(capture_positions("d"))
    try:
        with torch.no_grad():
            encoder(spatial_feats, positions)
    finally:
        handle_t.remove()
        handle_d.remove()

    assert captured["t"] is positions
    assert captured["d"] is positions
    assert captured["t"] is captured["d"]
    assert encoder.max_temporal_shift == encoder.ltae_t.max_temporal_shift
    assert encoder.positional_enc is encoder.ltae_t.positional_enc
    assert encoder.ltae_t.positional_enc is not encoder.ltae_d.positional_enc


def test_pse_mtkd_td_mid_shapes_return_feats_and_manual_path_match():
    model = PseMTKDTDMidLtae(input_dim=10, with_extra=False, num_classes=6).eval()
    pixels, valid_pixels, positions, extra = _model_inputs()

    with torch.no_grad():
        full_logits = model(pixels, valid_pixels, positions, extra)
        returned_logits, temporal_feats = model(
            pixels, valid_pixels, positions, extra, return_feats=True
        )
        spatial = model.spatial_encoder(pixels, valid_pixels, extra)
        t, s = model.temporal_encoder.mtkd(spatial, positions)
        d = s - t
        expected = torch.cat(
            [
                model.temporal_encoder.ltae_t(t, positions),
                model.temporal_encoder.ltae_d(d, positions),
            ],
            dim=-1,
        )
        manual_temporal = model.temporal_encoder(spatial, positions)
        manual_logits = model.decoder(manual_temporal)

    assert spatial.shape == (2, 5, 128)
    assert t.shape == s.shape == d.shape == (2, 5, 128)
    assert temporal_feats.shape == (2, 256)
    assert full_logits.shape == (2, 6)
    assert model.decoder[0].linear.in_features == 256
    torch.testing.assert_close(full_logits, returned_logits)
    torch.testing.assert_close(full_logits, manual_logits)
    torch.testing.assert_close(manual_temporal, expected)


def test_train_build_model_passes_mtkd_configuration_to_td_mid_model():
    config = SimpleNamespace(
        model="psemtkdtdmidltae",
        input_dim=10,
        num_classes=6,
        with_extra=False,
        mtkd_time_scale_days=400.0,
        mtkd_tau_fast_init_days=20.0,
        mtkd_tau_slow_init_days=75.0,
        mtkd_tau_min_days=2.0,
        mtkd_delta_tau_min_days=3.0,
        mtkd_learnable_tau=False,
    )

    model = _build_model(config)
    tau_fast, tau_slow = model.temporal_encoder.mtkd.get_tau_days()

    assert isinstance(model, PseMTKDTDMidLtae)
    torch.testing.assert_close(tau_fast, torch.tensor(20.0))
    torch.testing.assert_close(tau_slow, torch.tensor(75.0))
    assert model.temporal_encoder.mtkd.time_scale_days == 400.0
    assert not isinstance(model.temporal_encoder.mtkd.a, torch.nn.Parameter)
    assert not isinstance(model.temporal_encoder.mtkd.b, torch.nn.Parameter)


def test_td_mid_diagnostics_add_detached_d_metrics_without_state_dict_entries(capsys):
    model = PseMTKDTDMidLtae(input_dim=10, with_extra=False, num_classes=6)
    t = torch.ones(2, 3, 128, requires_grad=True)
    s = torch.full((2, 3, 128), 3.0, requires_grad=True)
    model.temporal_encoder.record_diagnostics(t, s)

    diagnostics = log_mtkd_diagnostics(model, "source", 1)
    output = capsys.readouterr().out

    assert diagnostics["d_to_s_norm_ratio"] == pytest.approx(2.0 / 3.0)
    assert diagnostics["td_cosine_similarity"] == pytest.approx(1.0)
    assert "d_to_s_norm_ratio" in output
    assert "td_cosine_similarity" in output
    assert not any("diagnostic" in key for key in model.state_dict())
    assert all(not value.requires_grad for value in model.temporal_encoder._diagnostic_totals.values())


def test_td_mid_model_is_registered_and_launcher_uses_expected_configuration():
    parser = argparse.ArgumentParser()
    _add_model_arguments(parser)
    choices = parser._option_string_actions["--model"].choices
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_mtkd_td_mid_9tasks_3gpu.sh"
    )
    text = launcher.read_text(encoding="utf-8")

    assert "psemtkdtdmidltae" in choices
    assert "GPU_IDS=(1 2 3)" in text
    assert "SOURCE_DOMAINS=(AT1 FR1 FR2)" in text
    assert "--model psemtkdtdmidltae" in text
    assert "--progress_bar off" in text
    assert "mtkd_td_mid_9tasks" in text
    assert "run_mtkd_td_mid_9tasks_3gpu.sh" in text
    assert "psemtkdmidltae" not in text


def test_tq_mid_temporal_encoder_conserves_h_and_matches_explicit_tq_path():
    encoder = MTKDTQMidConcatLTAE(in_channels=128).eval()
    h = torch.randn(2, 5, 128)
    positions = torch.arange(5).unsqueeze(0).repeat(2, 1) * 17

    with torch.no_grad():
        t, s = encoder.mtkd(h, positions)
        q = h - t
        d = s - t
        r = h - s
        expected_t = encoder.ltae_t(t, positions)
        expected_q = encoder.ltae_q(q, positions)
        actual = encoder(h, positions)

    assert h.shape == t.shape == s.shape == q.shape == (2, 5, 128)
    torch.testing.assert_close(t + q, h)
    torch.testing.assert_close(q, d + r)
    assert expected_t.shape == (2, 128)
    assert expected_q.shape == (2, 128)
    assert actual.shape == (2, 256)
    torch.testing.assert_close(actual, torch.cat([expected_t, expected_q], dim=-1))

    assert encoder.ltae_t is not encoder.ltae_q
    t_param = next(encoder.ltae_t.parameters())
    q_param = next(encoder.ltae_q.parameters())
    assert t_param is not q_param
    assert t_param.data_ptr() != q_param.data_ptr()
    assert sum(
        isinstance(module, MultiScaleTemporalKernelDecomposition)
        for module in encoder.modules()
    ) == 1


def test_tq_mid_temporal_encoder_passes_identical_positions_to_both_branches():
    encoder = MTKDTQMidConcatLTAE(in_channels=128).eval()
    h = torch.randn(2, 5, 128)
    positions = torch.arange(5).unsqueeze(0).repeat(2, 1) * 17
    captured = {}

    def capture_positions(name):
        def hook(_module, args):
            captured[name] = args[1]

        return hook

    handle_t = encoder.ltae_t.register_forward_pre_hook(capture_positions("t"))
    handle_q = encoder.ltae_q.register_forward_pre_hook(capture_positions("q"))
    try:
        with torch.no_grad():
            encoder(h, positions)
    finally:
        handle_t.remove()
        handle_q.remove()

    assert captured["t"] is positions
    assert captured["q"] is positions
    assert captured["t"] is captured["q"]
    assert encoder.max_temporal_shift == encoder.ltae_t.max_temporal_shift
    assert encoder.positional_enc is encoder.ltae_t.positional_enc
    assert encoder.ltae_t.positional_enc is not encoder.ltae_q.positional_enc


def test_pse_mtkd_tq_mid_shapes_return_feats_and_manual_path_match():
    model = PseMTKDTQMidLtae(input_dim=10, with_extra=False, num_classes=6).eval()
    pixels, valid_pixels, positions, extra = _model_inputs()

    with torch.no_grad():
        full_logits = model(pixels, valid_pixels, positions, extra)
        returned_logits, temporal_feats = model(
            pixels, valid_pixels, positions, extra, return_feats=True
        )
        h = model.spatial_encoder(pixels, valid_pixels, extra)
        t, s = model.temporal_encoder.mtkd(h, positions)
        q = h - t
        expected = torch.cat(
            [
                model.temporal_encoder.ltae_t(t, positions),
                model.temporal_encoder.ltae_q(q, positions),
            ],
            dim=-1,
        )
        manual_temporal = model.temporal_encoder(h, positions)
        manual_logits = model.decoder(manual_temporal)

    assert h.shape == t.shape == s.shape == q.shape == (2, 5, 128)
    assert temporal_feats.shape == (2, 256)
    assert full_logits.shape == (2, 6)
    assert model.decoder[0].linear.in_features == 256
    torch.testing.assert_close(full_logits, returned_logits)
    torch.testing.assert_close(full_logits, manual_logits)
    torch.testing.assert_close(manual_temporal, expected)


def test_train_build_model_passes_mtkd_configuration_to_tq_mid_model():
    config = SimpleNamespace(
        model="psemtkdtqmidltae",
        input_dim=10,
        num_classes=6,
        with_extra=False,
        mtkd_time_scale_days=400.0,
        mtkd_tau_fast_init_days=20.0,
        mtkd_tau_slow_init_days=75.0,
        mtkd_tau_min_days=2.0,
        mtkd_delta_tau_min_days=3.0,
        mtkd_learnable_tau=False,
    )

    model = _build_model(config)
    tau_fast, tau_slow = model.temporal_encoder.mtkd.get_tau_days()

    assert isinstance(model, PseMTKDTQMidLtae)
    torch.testing.assert_close(tau_fast, torch.tensor(20.0))
    torch.testing.assert_close(tau_slow, torch.tensor(75.0))
    assert model.temporal_encoder.mtkd.time_scale_days == 400.0
    assert not isinstance(model.temporal_encoder.mtkd.a, torch.nn.Parameter)
    assert not isinstance(model.temporal_encoder.mtkd.b, torch.nn.Parameter)


def test_tq_mid_diagnostics_add_detached_q_metrics_without_state_dict_entries(capsys):
    model = PseMTKDTQMidLtae(input_dim=10, with_extra=False, num_classes=6)
    t = torch.ones(2, 3, 128, requires_grad=True)
    s = torch.full((2, 3, 128), 2.0, requires_grad=True)
    h = torch.full((2, 3, 128), 3.0, requires_grad=True)
    model.temporal_encoder.record_diagnostics(t, s, h)

    diagnostics = log_mtkd_diagnostics(model, "source", 1)
    output = capsys.readouterr().out

    assert diagnostics["q_to_h_norm_ratio"] == pytest.approx(2.0 / 3.0)
    assert diagnostics["tq_cosine_similarity"] == pytest.approx(1.0)
    assert "q_to_h_norm_ratio" in output
    assert "tq_cosine_similarity" in output
    assert not any("diagnostic" in key for key in model.state_dict())
    assert all(not value.requires_grad for value in model.temporal_encoder._diagnostic_totals.values())


def test_tq_mid_model_is_registered_and_launcher_uses_expected_configuration():
    parser = argparse.ArgumentParser()
    _add_model_arguments(parser)
    choices = parser._option_string_actions["--model"].choices
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_mtkd_tq_mid_12tasks_4gpu.sh"
    )
    text = launcher.read_text(encoding="utf-8")

    assert "psemtkdtqmidltae" in choices
    assert "GPU_IDS=(0 1 2 3)" in text
    assert "SOURCE_DOMAINS=(AT1 DK1 FR1 FR2)" in text
    assert "DK1) TARGET_DOMAINS=(AT1 FR1 FR2)" in text
    assert "--model psemtkdtqmidltae" in text
    assert "--progress_bar off" in text
    assert "mtkd_tq_mid_12tasks" in text
    assert "run_mtkd_tq_mid_12tasks_4gpu.sh" in text
    assert "psemtkdmidltae" not in text
    assert "psemtkdtdmidltae" not in text


def test_single_scale_smoother_matches_existing_laplace_smoothing_with_mask():
    h = torch.randn(2, 5, 8, dtype=torch.float64)
    positions = torch.tensor([[0, 10, 30, 80, 150], [2, 15, 44, 90, 170]])
    time_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    tau_days = 60.0
    smoother = SingleScaleTemporalKernel(tau_days=tau_days).double()
    existing = _make_mtkd().double()

    normalized_positions = positions.to(dtype=h.dtype) / existing.time_scale_days
    pairwise_distance = torch.abs(
        normalized_positions.unsqueeze(-1) - normalized_positions.unsqueeze(-2)
    )
    expected = existing._smooth(
        h, pairwise_distance, torch.tensor(tau_days, dtype=h.dtype), time_mask
    )
    actual = smoother(h, positions, time_mask=time_mask)

    assert actual.shape == h.shape
    torch.testing.assert_close(actual, expected)


def test_single_scale_tau_is_fixed_and_sweep_values_produce_different_trends():
    h = torch.randn(2, 6, 8)
    positions = torch.tensor([[0, 8, 30, 75, 140, 260]] * 2)
    trends = [SingleScaleTemporalKernel(tau)(h, positions) for tau in (45, 60, 75)]

    assert not torch.allclose(trends[0], trends[1])
    assert not torch.allclose(trends[1], trends[2])
    for tau, trend in zip((45, 60, 75), trends):
        smoother = SingleScaleTemporalKernel(tau)
        assert smoother.tau_days.item() == pytest.approx(tau)
        assert list(smoother.named_parameters()) == []
        assert trend.shape == h.shape


def test_tq_single_encoder_conserves_h_uses_explicit_path_and_independent_ltaes():
    encoder = MTKDTQSingleLTAE(in_channels=128, tau_days=60).eval()
    h = torch.randn(2, 5, 128)
    positions = torch.arange(5).unsqueeze(0).repeat(2, 1) * 17
    captured = {}

    def capture_positions(name):
        def hook(_module, args):
            captured[name] = args[1]

        return hook

    handle_t = encoder.ltae_t.register_forward_pre_hook(capture_positions("t"))
    handle_q = encoder.ltae_q.register_forward_pre_hook(capture_positions("q"))
    try:
        with torch.no_grad():
            t = encoder.smoother(h, positions)
            q = h - t
            expected_t = encoder.ltae_t(t, positions)
            expected_q = encoder.ltae_q(q, positions)
            actual = encoder(h, positions)
    finally:
        handle_t.remove()
        handle_q.remove()

    torch.testing.assert_close(t + q, h)
    assert expected_t.shape == expected_q.shape == (2, 128)
    assert actual.shape == (2, 256)
    torch.testing.assert_close(actual, torch.cat([expected_t, expected_q], dim=-1))
    assert encoder.ltae_t is not encoder.ltae_q
    assert next(encoder.ltae_t.parameters()).data_ptr() != next(
        encoder.ltae_q.parameters()
    ).data_ptr()
    assert encoder.ltae_t.positional_enc is not encoder.ltae_q.positional_enc
    assert captured["t"] is positions
    assert captured["q"] is positions
    assert captured["t"] is captured["q"]
    assert not any("tau" in name for name, _parameter in encoder.named_parameters())


def test_pse_mtkd_tq_single_shapes_return_feats_and_manual_path_match():
    model = PseMTKDTQSingleLtae(
        input_dim=10, with_extra=False, num_classes=6, tq_tau_days=60
    ).eval()
    pixels, valid_pixels, positions, extra = _model_inputs()

    with torch.no_grad():
        full_logits = model(pixels, valid_pixels, positions, extra)
        returned_logits, temporal_feats = model(
            pixels, valid_pixels, positions, extra, return_feats=True
        )
        h = model.spatial_encoder(pixels, valid_pixels, extra)
        t = model.temporal_encoder.smoother(h, positions)
        q = h - t
        explicit = torch.cat(
            [
                model.temporal_encoder.ltae_t(t, positions),
                model.temporal_encoder.ltae_q(q, positions),
            ],
            dim=-1,
        )
        manual_temporal = model.temporal_encoder(h, positions)
        manual_logits = model.decoder(manual_temporal)

    assert h.shape == t.shape == q.shape == (2, 5, 128)
    assert temporal_feats.shape == (2, 256)
    assert full_logits.shape == (2, 6)
    assert model.decoder[0].linear.in_features == 256
    torch.testing.assert_close(full_logits, returned_logits)
    torch.testing.assert_close(full_logits, manual_logits)
    torch.testing.assert_close(manual_temporal, explicit)


def test_tq_single_diagnostics_only_report_fixed_tau_and_tq_metrics(capsys):
    model = PseMTKDTQSingleLtae(
        input_dim=10, with_extra=False, num_classes=6, tq_tau_days=60
    )
    t = torch.ones(2, 3, 128, requires_grad=True)
    h = torch.full((2, 3, 128), 3.0, requires_grad=True)
    model.temporal_encoder.record_diagnostics(t, h)

    diagnostics = log_mtkd_diagnostics(model, "source", 1)
    output = capsys.readouterr().out

    assert set(diagnostics) == {
        "tau_days",
        "q_to_h_norm_ratio",
        "tq_cosine_similarity",
    }
    assert diagnostics["tau_days"] == pytest.approx(60.0)
    assert diagnostics["q_to_h_norm_ratio"] == pytest.approx(2.0 / 3.0)
    assert diagnostics["tq_cosine_similarity"] == pytest.approx(1.0)
    assert "tau_days" in output
    assert "tau_fast_days" not in output
    assert not any("diagnostic" in key for key in model.state_dict())
    assert all(not value.requires_grad for value in model.temporal_encoder._diagnostic_totals.values())


def test_train_builds_tq_single_model_with_cli_fixed_tau():
    parser = argparse.ArgumentParser()
    _add_model_arguments(parser)
    parsed = parser.parse_args(
        ["--model", "psemtkdtqsingleltae", "--tq_tau_days", "75"]
    )
    config = SimpleNamespace(
        model=parsed.model,
        input_dim=10,
        num_classes=6,
        with_extra=False,
        tq_tau_days=parsed.tq_tau_days,
    )

    model = _build_model(config)

    assert isinstance(model, PseMTKDTQSingleLtae)
    assert model.temporal_encoder.smoother.tau_days.item() == pytest.approx(75.0)
    assert not any("tau" in name for name, _parameter in model.named_parameters())


def test_tq_single_sweep_launcher_has_expected_tau_source_gpu_matrix():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_mtkd_tq_single_tau_sweep_fr1_fr2.sh"
    )
    text = launcher.read_text(encoding="utf-8")

    assert "TAU_VALUES=(45 60 75)" in text
    assert "GPU_IDS=(0 1 2)" in text
    assert "SOURCE_DOMAINS=(FR1 FR2)" in text
    assert "--model psemtkdtqsingleltae" in text
    assert '--tq_tau_days "$tau_days"' in text
    assert "--progress_bar off" in text
    assert "source commands:       6" in text
    assert "TimeMatch commands:    18" in text
    assert "total training commands: 24" in text
    assert "GPU3:                  unused" in text


def test_vectorized_loo_removes_diagonal_and_skips_invalid_samples():
    h = torch.tensor([[[0.0], [10.0], [20.0]]], requires_grad=True)
    positions = torch.tensor([[0, 1, 2]])
    loss, count = leave_one_out_smooth_l1(
        h, positions, tau_days=torch.tensor(1.0), time_scale_days=1.0
    )
    kernel = torch.exp(-torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]))
    kernel.fill_diagonal_(0.0)
    expected_h = torch.matmul(kernel / kernel.sum(dim=-1, keepdim=True), h[0])
    expected = torch.nn.functional.smooth_l1_loss(expected_h, h[0], beta=1.0)

    assert count == 3
    torch.testing.assert_close(loss, expected)

    invalid_mask = torch.tensor([[1, 0, 0]], dtype=torch.bool)
    zero, invalid_count = leave_one_out_smooth_l1(
        h, positions, tau_days=torch.tensor(1.0), time_scale_days=1.0,
        time_mask=invalid_mask,
    )
    assert invalid_count == 0
    assert zero.requires_grad
    assert zero.item() == 0.0
    zero.backward()
    torch.testing.assert_close(h.grad, torch.zeros_like(h))


def _loo_model(variant):
    return PseMTKDTQLOOLtae(
        input_dim=10,
        with_extra=False,
        num_classes=6,
        tq_loo_variant=variant,
        tq_tau_init_days=75.0,
        tq_tau_min_days=1.0,
        tq_loo_pse_weight=0.1,
    )


def _has_grad(module):
    return any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in module.parameters()
    )


def _clear_grads(model):
    for parameter in model.parameters():
        parameter.grad = None


def test_loo_tau_only_gradient_boundaries():
    model = _loo_model("loo_tau_only")
    pixels, mask, positions, extra = _model_inputs()
    tau_parameter = model.temporal_encoder.smoother.a

    h = model.spatial_encoder(pixels, mask, extra)
    model.forward_from_spatial(h, positions).sum().backward()
    assert tau_parameter.grad is None or tau_parameter.grad.item() == 0.0
    assert _has_grad(model.spatial_encoder)

    _clear_grads(model)
    h = model.spatial_encoder(pixels, mask, extra)
    losses = model.compute_loo_losses(h, positions)
    losses["loo_tau"].backward()
    assert tau_parameter.grad is not None and tau_parameter.grad.abs().item() > 0
    assert not _has_grad(model.spatial_encoder)
    assert losses["loo_pse"].item() == 0.0


def test_loo_tq_path_keeps_conservation_independent_ltaes_and_same_positions():
    model = _loo_model("split_loo_tau_pse").eval()
    pixels, mask, positions, extra = _model_inputs()
    h = model.spatial_encoder(pixels, mask, extra)
    encoder = model.temporal_encoder
    captured = []

    handles = [
        branch.register_forward_pre_hook(
            lambda _module, args: captured.append(args[1])
        )
        for branch in (encoder.ltae_t, encoder.ltae_q)
    ]
    try:
        actual = encoder(h, positions)
    finally:
        for handle in handles:
            handle.remove()

    t = encoder._smooth_task(h, positions)
    q = h - t
    expected = torch.cat(
        [encoder.ltae_t(t, positions), encoder.ltae_q(q, positions)], dim=-1
    )
    torch.testing.assert_close(t + q, h)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 256)
    assert encoder.ltae_t is not encoder.ltae_q
    assert next(encoder.ltae_t.parameters()).data_ptr() != next(
        encoder.ltae_q.parameters()
    ).data_ptr()
    assert captured[0] is positions and captured[1] is positions


def test_pse_loo_fixed75_gradient_boundaries():
    model = _loo_model("pse_loo_fixed75")
    pixels, mask, positions, extra = _model_inputs()
    h = model.spatial_encoder(pixels, mask, extra)
    losses = model.compute_loo_losses(h, positions)
    losses["loo_pse"].backward()

    assert model.temporal_encoder.get_tau_days().item() == pytest.approx(75.0)
    assert not any("smoother.a" in name for name, _ in model.named_parameters())
    assert _has_grad(model.spatial_encoder)
    assert not _has_grad(model.temporal_encoder.ltae_t)
    assert not _has_grad(model.temporal_encoder.ltae_q)
    assert not _has_grad(model.decoder)
    assert losses["loo_tau"].item() == 0.0

    nondefault_request = PseMTKDTQLOOLtae(
        input_dim=10,
        with_extra=False,
        num_classes=6,
        tq_loo_variant="pse_loo_fixed75",
        tq_tau_init_days=60.0,
    )
    assert nondefault_request.temporal_encoder.get_tau_days().item() == pytest.approx(75.0)


def test_split_loo_has_three_strictly_separated_gradient_graphs():
    model = _loo_model("split_loo_tau_pse")
    pixels, mask, positions, extra = _model_inputs()
    tau_parameter = model.temporal_encoder.smoother.a

    h = model.spatial_encoder(pixels, mask, extra)
    model.forward_from_spatial(h, positions).sum().backward()
    assert tau_parameter.grad is None or tau_parameter.grad.item() == 0.0
    assert _has_grad(model.spatial_encoder)
    assert _has_grad(model.temporal_encoder.ltae_t)
    assert _has_grad(model.temporal_encoder.ltae_q)
    assert _has_grad(model.decoder)

    _clear_grads(model)
    h = model.spatial_encoder(pixels, mask, extra)
    model.compute_loo_losses(h, positions)["loo_tau"].backward()
    assert tau_parameter.grad is not None and tau_parameter.grad.abs().item() > 0
    assert not _has_grad(model.spatial_encoder)

    _clear_grads(model)
    h = model.spatial_encoder(pixels, mask, extra)
    model.compute_loo_losses(h, positions)["loo_pse"].backward()
    assert tau_parameter.grad is None or tau_parameter.grad.item() == 0.0
    assert _has_grad(model.spatial_encoder)
    assert not _has_grad(model.temporal_encoder.ltae_t)
    assert not _has_grad(model.temporal_encoder.ltae_q)
    assert not _has_grad(model.decoder)


def test_loo_diagnostics_and_tau_optimizer_group_are_auditable():
    model = _loo_model("split_loo_tau_pse")
    h = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 2.0], [3.0, 2.0]],
            [[2.0, 0.0], [2.0, 1.0], [4.0, 3.0]],
            [[8.0, 7.0], [9.0, 8.0], [10.0, 8.0]],
        ]
    )
    positions = torch.tensor([[0, 2, 7], [0, 3, 9], [1, 4, 10]])
    labels = torch.tensor([0, 0, 1])
    diagnostics = compute_feature_curve_diagnostics(h, positions, labels=labels)

    assert set(diagnostics) == {
        "h_temporal_var",
        "h_cross_sample_var",
        "h_adjacent_l2",
        "h_adjacent_slope_per_day",
        "h_latent_rms",
        "h_source_within_class_scatter",
        "h_source_between_class_scatter",
        "h_source_fisher",
    }
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert all(not value.requires_grad for value in diagnostics.values())

    groups = _optimizer_parameter_groups(model, weight_decay=1e-4)
    tau_parameter = model.temporal_encoder.smoother.a
    tau_groups = [group for group in groups if any(p is tau_parameter for p in group["params"])]
    assert len(tau_groups) == 1
    assert tau_groups[0]["weight_decay"] == 0.0
    assert sum(any(p is tau_parameter for p in group["params"]) for group in groups) == 1


def test_timematch_loo_is_meaned_per_domain_before_equal_weighting():
    model = _loo_model("split_loo_tau_pse")
    h_s = torch.randn(1, 3, 128, requires_grad=True)
    h_t = torch.randn(4, 5, 128, requires_grad=True)
    positions_s = torch.tensor([[0, 10, 30]])
    positions_t = torch.arange(5).repeat(4, 1) * 11

    source, target, balanced = _domain_balanced_loo(
        model, h_s, positions_s, h_t, positions_t
    )

    for name in ("loo_tau", "loo_pse"):
        torch.testing.assert_close(
            balanced[name], 0.5 * (source[name] + target[name])
        )


def test_loo_cli_variants_and_launcher_matrix():
    parser = argparse.ArgumentParser()
    _add_model_arguments(parser)
    parsed = parser.parse_args(
        [
            "--model", "psemtkdtqlooltae",
            "--tq_loo_variant", "pse_loo_fixed75",
            "--tq_loo_pse_weight", "0.1",
        ]
    )
    model = _build_model(
        SimpleNamespace(
            model=parsed.model,
            input_dim=10,
            num_classes=6,
            with_extra=False,
            tq_loo_variant=parsed.tq_loo_variant,
            tq_tau_init_days=parsed.tq_tau_init_days,
            tq_tau_min_days=parsed.tq_tau_min_days,
            tq_loo_pse_weight=parsed.tq_loo_pse_weight,
        )
    )
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_mtkd_tq_loo_3variants_fr1_fr2.sh"
    )
    text = launcher.read_text(encoding="utf-8")

    assert isinstance(model, PseMTKDTQLOOLtae)
    assert model.loo_variant == "pse_loo_fixed75"
    assert model.loo_pse_weight == pytest.approx(0.1)
    assert "VARIANTS=(loo_tau_only pse_loo_fixed75 split_loo_tau_pse)" in text
    assert "GPU_IDS=(0 1 2)" in text
    assert "SOURCE_DOMAINS=(FR1 FR2)" in text
    assert "source commands:       0" in text
    assert "reused source checkpoints: 6" in text
    assert "TimeMatch commands:    18" in text
    assert "total training commands: 18" in text
    assert "GPU3:                  unused" in text
    assert "verify_source_checkpoints" in text
    assert "clear_da_outputs" in text
    assert 'rm -rf -- "$output_path" "$tensorboard_path"' in text
    assert "build_timematch_command" in text
    assert "run_task source" not in text
    assert text.index("verify_source_checkpoints || exit 1") < text.index(
        "clear_da_outputs || exit 1"
    )
    assert text.index("clear_da_outputs || exit 1") < text.index(
        'launch_workers "$LOG_ROOT/$RUN_TIMESTAMP"'
    )
