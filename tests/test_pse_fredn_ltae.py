import torch

from models.fredn.nufft import DenseFourierBackend
from models.stclassifier import PseFreDNLTae


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
        nufft_backend=DenseFourierBackend(),
    )


def _tiny_batch():
    torch.manual_seed(11)
    pixels = torch.randn(2, 5, 2, 3)
    mask = torch.ones(2, 5, 3)
    positions = torch.tensor(
        [[5, 40, 90, 160, 260], [8, 44, 95, 165, 265]],
        dtype=torch.long,
    )
    extra = torch.zeros(2, 4)
    return pixels, mask, positions, extra


def test_model_uses_two_independent_ltaes_and_minimal_fusion():
    model = _tiny_model()

    assert model.trend_temporal_encoder is not model.seasonal_temporal_encoder
    trend_parameters = list(model.trend_temporal_encoder.parameters())
    seasonal_parameters = list(model.seasonal_temporal_encoder.parameters())
    assert all(left is not right for left, right in zip(trend_parameters, seasonal_parameters))
    assert model.trend_norm.normalized_shape == (4,)
    assert model.seasonal_norm.normalized_shape == (4,)
    assert model.fusion[0].in_features == 8
    assert model.fusion[0].out_features == 4
    for forbidden_name in ("raw_temporal_encoder", "gate", "trend_classifier", "seasonal_classifier"):
        assert not hasattr(model, forbidden_name)


def test_prepare_reports_distinct_additivity_and_reconstruction_errors():
    model = _tiny_model().eval()
    pixels, mask, positions, extra = _tiny_batch()
    spatial_features = model.spatial_encoder(pixels, mask, extra)

    prepared = model.prepare_temporal_features(spatial_features, positions)

    assert prepared.trend.shape == spatial_features.shape
    assert prepared.seasonal.shape == spatial_features.shape
    assert prepared.diagnostics["additivity_error"] < 1e-6
    assert prepared.diagnostics["reconstruction_error"] >= prepared.diagnostics["additivity_error"]
    assert prepared.diagnostics["imaginary_residual"] < 1e-4


def test_both_ltaes_receive_exactly_the_same_shifted_positions():
    model = _tiny_model().eval()
    pixels, mask, positions, extra = _tiny_batch()
    captured = []

    def capture_positions(module, inputs):
        captured.append(inputs[1].detach().clone())

    trend_hook = model.trend_temporal_encoder.register_forward_pre_hook(capture_positions)
    seasonal_hook = model.seasonal_temporal_encoder.register_forward_pre_hook(capture_positions)
    try:
        logits = model.forward_with_temporal_shift(
            pixels,
            mask,
            positions,
            extra,
            temporal_shift=7,
        )
    finally:
        trend_hook.remove()
        seasonal_hook.remove()

    assert logits.shape == (2, 3)
    assert len(captured) == 2
    assert torch.equal(captured[0], positions + 7)
    assert torch.equal(captured[1], positions + 7)


def test_classification_gradient_reaches_every_required_component():
    model = _tiny_model().train()
    pixels, mask, positions, extra = _tiny_batch()

    logits = model(pixels, mask, positions, extra)
    logits.square().mean().backward()

    components = {
        "PSE": model.spatial_encoder,
        "Frequency Disentangler": model.frequency_disentangler,
        "trend LTAE": model.trend_temporal_encoder,
        "seasonal LTAE": model.seasonal_temporal_encoder,
        "fusion MLP": model.fusion,
    }
    for name, component in components.items():
        gradients = [
            parameter.grad
            for parameter in component.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients, name
        assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients), name
