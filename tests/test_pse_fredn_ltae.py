from copy import deepcopy

import pytest
import torch

from models.fredn.model import PseFreDNLTae


def _tiny_model(num_classes=3):
    return PseFreDNLTae(
        input_dim=2,
        mlp1=[2, 4],
        pooling="mean_std",
        mlp2=[8, 8],
        with_extra=False,
        n_head=2,
        d_k=2,
        d_model=8,
        mlp3=[8, 4],
        dropout=0.0,
        mlp4=[4, 3],
        num_classes=num_classes,
        fredn_num_modes=3,
        fredn_period_days=365.0,
        fredn_nufft_reg=1e-3,
        fredn_nufft_tol=1e-6,
        fredn_nufft_max_iter=12,
        fredn_fourier_solver="dense_direct",
    )


def _tiny_batch(batch_size=2):
    torch.manual_seed(11)
    pixels = torch.randn(batch_size, 5, 2, 3)
    mask = torch.ones(batch_size, 5, 3)
    base_positions = torch.tensor(
        [
            [5, 40, 90, 160, 260],
            [8, 44, 95, 165, 265],
            [12, 52, 102, 172, 272],
        ],
        dtype=torch.long,
    )
    positions = base_positions[:batch_size]
    extra = torch.zeros(batch_size, 4)
    return pixels, mask, positions, extra


def test_model_uses_trend_ltae_and_shared_reim_seasonal_encoder():
    model = _tiny_model()

    assert hasattr(model, "trend_temporal_encoder")
    assert model.get_temporal_encoders() == (model.trend_temporal_encoder,)
    assert hasattr(model, "seasonal_spectral_encoder")
    assert hasattr(model.seasonal_spectral_encoder, "shared_reim_mlp")
    assert hasattr(model, "trend_classifier")
    assert hasattr(model, "seasonal_classifier")
    for removed_name in (
        "seasonal_temporal_encoder",
        "seasonal_norm",
        "fusion",
        "decoder",
    ):
        assert not hasattr(model, removed_name)
    assert not hasattr(model.seasonal_spectral_encoder, "real_mlp")
    assert not hasattr(model.seasonal_spectral_encoder, "imag_mlp")


def test_real_and_imaginary_parts_use_the_same_reim_module():
    model = _tiny_model().eval()
    coeffs = torch.randn(2, 3, 8, dtype=torch.complex64)
    calls = []
    shared = model.seasonal_spectral_encoder.shared_reim_mlp
    hook = shared.register_forward_pre_hook(
        lambda module, inputs: calls.append(inputs[0].detach().clone())
    )
    try:
        features = model.seasonal_spectral_encoder(coeffs)
    finally:
        hook.remove()

    transposed = coeffs.transpose(1, 2)
    assert features.shape == (2, 4)
    assert len(calls) == 2
    assert torch.equal(calls[0], transposed.real)
    assert torch.equal(calls[1], transposed.imag)


def test_prepare_reports_distinct_additivity_and_reconstruction_errors():
    model = _tiny_model().eval()
    pixels, mask, positions, extra = _tiny_batch()
    spatial_features = model.spatial_encoder(pixels, mask, extra)

    prepared = model.prepare_temporal_features(
        spatial_features,
        positions,
        collect_diagnostics=True,
    )

    assert prepared.trend.shape == spatial_features.shape
    assert prepared.seasonal_coeffs.shape == (2, 3, 8)
    assert torch.is_complex(prepared.seasonal_coeffs)
    assert prepared.diagnostics["additivity_error"] < 1e-6
    assert prepared.diagnostics["reconstruction_error"] >= prepared.diagnostics["additivity_error"]
    assert prepared.diagnostics["imaginary_residual"] < 1e-4
    for key in (
        "fourier_condition_mean",
        "fourier_condition_median",
        "fourier_condition_p95",
        "fourier_condition_max",
    ):
        assert torch.isfinite(prepared.diagnostics[key])


def test_prepare_uses_one_synthesis_normally_and_three_for_diagnostics(monkeypatch):
    model = _tiny_model().eval()
    pixels, mask, positions, extra = _tiny_batch()
    spatial_features = model.spatial_encoder(pixels, mask, extra)
    original = model.fourier_synthesizer.synthesize_complex
    calls = []

    def recording_synthesis(coeffs, timestamps):
        calls.append(coeffs)
        return original(coeffs, timestamps)

    monkeypatch.setattr(
        model.fourier_synthesizer,
        "synthesize_complex",
        recording_synthesis,
    )

    prepared = model.prepare_temporal_features(spatial_features, positions)
    assert len(calls) == 1
    assert prepared.diagnostics == {}
    assert model.last_diagnostics == {}

    calls.clear()
    prepared = model.prepare_temporal_features(
        spatial_features,
        positions,
        collect_diagnostics=True,
    )
    assert len(calls) == 3
    assert "reconstruction_error" in prepared.diagnostics


def test_final_logits_are_exact_branch_sum_and_features_remain_protocol_width():
    model = _tiny_model().eval()
    pixels, mask, positions, extra = _tiny_batch()
    spatial_features = model.spatial_encoder(pixels, mask, extra)
    prepared = model.prepare_temporal_features(spatial_features, positions)
    captured = {}

    def capture(name):
        return lambda module, inputs, output: captured.setdefault(name, output.detach().clone())

    hooks = [
        model.trend_norm.register_forward_hook(capture("trend_features")),
        model.seasonal_spectral_encoder.register_forward_hook(capture("seasonal_features")),
        model.trend_classifier.register_forward_hook(capture("trend_logits")),
        model.seasonal_classifier.register_forward_hook(capture("seasonal_logits")),
    ]
    try:
        logits, features = model.classify_prepared(
            prepared,
            positions,
            return_feats=True,
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert torch.equal(logits, captured["trend_logits"] + captured["seasonal_logits"])
    assert torch.equal(features, captured["trend_features"] + captured["seasonal_features"])
    assert features.shape == (2, 4)


def test_trend_ltae_receives_shift_and_seasonal_receives_phase_rotation():
    model = _tiny_model().eval()
    pixels, mask, positions, extra = _tiny_batch()
    spatial_features = model.spatial_encoder(pixels, mask, extra)
    prepared = model.prepare_temporal_features(spatial_features, positions)
    captured = {}

    def capture_trend_positions(module, inputs):
        captured["positions"] = inputs[1].detach().clone()

    def capture_seasonal_coeffs(module, inputs):
        captured["coeffs"] = inputs[0].detach().clone()

    trend_hook = model.trend_temporal_encoder.register_forward_pre_hook(
        capture_trend_positions
    )
    seasonal_hook = model.seasonal_spectral_encoder.register_forward_pre_hook(
        capture_seasonal_coeffs
    )
    try:
        logits = model.classify_prepared(prepared, positions, temporal_shift=7)
    finally:
        trend_hook.remove()
        seasonal_hook.remove()

    expected_coeffs = model._shift_seasonal_coefficients(
        prepared.seasonal_coeffs,
        temporal_shift=7,
    )
    assert logits.shape == (2, 3)
    assert torch.equal(captured["positions"], positions + 7)
    assert torch.allclose(captured["coeffs"], expected_coeffs)


def test_batch_specific_seasonal_shift_supports_positive_zero_and_negative_values():
    model = _tiny_model().eval()
    coeffs = torch.randn(3, 3, 8, dtype=torch.complex64)
    shifts = torch.tensor([[10.0], [0.0], [-5.0]])

    shifted = model._shift_seasonal_coefficients(coeffs, shifts)

    assert shifted.shape == coeffs.shape
    assert torch.equal(shifted[1], coeffs[1])
    for sample_index, shift in enumerate((10.0, 0.0, -5.0)):
        expected = model._shift_seasonal_coefficients(
            coeffs[sample_index : sample_index + 1],
            shift,
        )
        assert torch.allclose(shifted[sample_index : sample_index + 1], expected)


@pytest.mark.parametrize("delta", [13.0, -17.0])
def test_phase_rotation_matches_reanalysis_at_shifted_positions(delta):
    model = _tiny_model().eval()
    torch.manual_seed(23)
    features = torch.randn(2, 7, 8, dtype=torch.float64)
    positions = torch.tensor(
        [
            [3.0, 31.0, 79.0, 128.0, 191.0, 257.0, 331.0],
            [7.0, 42.0, 86.0, 141.0, 203.0, 271.0, 349.0],
        ],
        dtype=torch.float64,
    )

    original, _ = model.fourier_analyzer(features, positions)
    directly_shifted, _ = model.fourier_analyzer(features, positions + delta)
    phase_shifted = model._shift_seasonal_coefficients(original, delta)

    relative_error = torch.linalg.vector_norm(phase_shifted - directly_shifted) / torch.linalg.vector_norm(
        directly_shifted
    )
    assert relative_error < 1e-10


def test_branch_diagnostics_are_detached_tensors():
    model = _tiny_model().eval()
    pixels, mask, positions, extra = _tiny_batch()

    model(
        pixels,
        mask,
        positions,
        extra,
        collect_diagnostics=True,
    )

    for key in (
        "trend_logit_rms",
        "seasonal_logit_rms",
        "trend_feature_rms",
        "seasonal_feature_rms",
    ):
        value = model.last_diagnostics[key]
        assert isinstance(value, torch.Tensor)
        assert not value.requires_grad


def test_classification_gradient_reaches_every_required_component():
    model = _tiny_model().train()
    pixels, mask, positions, extra = _tiny_batch()

    logits = model(pixels, mask, positions, extra)
    logits.square().mean().backward()

    components = {
        "PSE": model.spatial_encoder,
        "Frequency Disentangler": model.frequency_disentangler,
        "trend LTAE": model.trend_temporal_encoder,
        "ReIm shared MLP": model.seasonal_spectral_encoder.shared_reim_mlp,
        "spectral readout": model.seasonal_spectral_encoder.spectral_readout,
        "seasonal output projection": model.seasonal_spectral_encoder.output_projection,
        "trend classifier": model.trend_classifier,
        "seasonal classifier": model.seasonal_classifier,
    }
    for name, component in components.items():
        gradients = [
            parameter.grad
            for parameter in component.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients, name
        assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients), name


def test_collecting_diagnostics_does_not_change_logits_or_parameter_gradients():
    normal_model = _tiny_model().eval()
    diagnostic_model = deepcopy(normal_model).eval()
    pixels, mask, positions, extra = _tiny_batch()

    normal_logits = normal_model(
        pixels,
        mask,
        positions,
        extra,
        collect_diagnostics=False,
    )
    diagnostic_logits = diagnostic_model(
        pixels,
        mask,
        positions,
        extra,
        collect_diagnostics=True,
    )
    normal_logits.square().mean().backward()
    diagnostic_logits.square().mean().backward()

    assert torch.equal(normal_logits, diagnostic_logits)
    for (normal_name, normal_parameter), (diagnostic_name, diagnostic_parameter) in zip(
        normal_model.named_parameters(),
        diagnostic_model.named_parameters(),
    ):
        assert normal_name == diagnostic_name
        if normal_parameter.grad is None or diagnostic_parameter.grad is None:
            assert normal_parameter.grad is diagnostic_parameter.grad, normal_name
        else:
            assert torch.equal(
                normal_parameter.grad,
                diagnostic_parameter.grad,
            ), normal_name
