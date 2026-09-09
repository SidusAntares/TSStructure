import argparse
import inspect
import sys
import types

import pytest
import torch


def test_neutral_fourier_matches_historical_dense_reference():
    from models.fourier_reconstruction import (
        BatchedDirectFourierAnalyzer,
        BatchedDirectFourierSynthesizer,
    )
    from models.fredn.nufft import (
        BatchedDirectFourierAnalyzer as HistoricalAnalyzer,
        BatchedDirectFourierSynthesizer as HistoricalSynthesizer,
    )

    torch.manual_seed(7)
    features = torch.randn(3, 11, 5)
    positions = torch.tensor(
        [
            [2, 19, 41, 67, 93, 121, 154, 188, 226, 271, 319],
            [4, 23, 46, 73, 101, 132, 167, 203, 242, 286, 337],
            [8, 28, 52, 79, 108, 141, 176, 211, 249, 294, 348],
        ],
        dtype=torch.long,
    )
    old_analyzer = HistoricalAnalyzer(13, period_days=365.0, reg=1e-3)
    new_analyzer = BatchedDirectFourierAnalyzer(13, period_days=365.0, reg=1e-3)
    old_coeffs, _ = old_analyzer(features, positions)
    new_coeffs, _ = new_analyzer(features, positions)
    assert torch.allclose(new_coeffs, old_coeffs, atol=1e-6, rtol=1e-5)

    old_recon = HistoricalSynthesizer(13, period_days=365.0)(old_coeffs, positions)
    new_recon = BatchedDirectFourierSynthesizer(13, period_days=365.0)(
        new_coeffs, positions
    )
    assert new_recon.shape == features.shape
    assert torch.allclose(new_recon, old_recon, atol=1e-6, rtol=1e-5)


def test_centered_thirteen_modes_and_deterministic_reconstruction():
    from models.fourier_reconstruction import (
        BatchedDirectFourierAnalyzer,
        BatchedDirectFourierSynthesizer,
        centered_modes,
    )

    assert centered_modes(13).tolist() == list(range(-6, 7))
    features = torch.randn(2, 15, 4)
    positions = torch.arange(15, dtype=torch.float32).repeat(2, 1) * 17
    analyzer = BatchedDirectFourierAnalyzer(13, reg=1e-3)
    synthesizer = BatchedDirectFourierSynthesizer(13)
    coefficients, _ = analyzer(features, positions)
    assert coefficients.shape == (2, 13, 4)
    first = synthesizer(coefficients, positions)
    second = synthesizer(coefficients, positions)
    assert first.shape == features.shape
    assert torch.equal(first, second)


def test_neutral_module_has_no_historical_fredn_dependency():
    import models.fourier_reconstruction as module

    source = inspect.getsource(module)
    assert "models.fredn" not in source
    assert "FrequencyDisentangler" not in source


def _small_fourier_model():
    from models.stclassifier import PseFourierReconLTae

    return PseFourierReconLTae(
        input_dim=3,
        mlp1=[3, 4],
        pooling="mean_std",
        mlp2=[8, 8],
        with_extra=False,
        n_head=1,
        d_k=2,
        d_model=8,
        mlp3=[8, 4],
        mlp4=[4, 3],
        num_classes=2,
        dropout=0.0,
        fourier_num_modes=5,
    )


def test_model_is_single_branch_and_has_no_decomposition_parameters():
    model = _small_fourier_model()
    names = tuple(name.lower() for name, _ in model.named_parameters())
    forbidden = ("mask", "disentangler", "trend", "seasonal", "reim", "fusion")
    assert all(token not in name for name in names for token in forbidden)
    assert hasattr(model, "temporal_encoder")
    assert hasattr(model, "decoder")
    assert len(model.get_temporal_encoders()) == 1
    assert not hasattr(model, "trend_temporal_encoder")
    assert not hasattr(model, "seasonal_temporal_encoder")


def test_model_reconstruction_keeps_pse_gradient():
    model = _small_fourier_model()
    model.train()
    pixels = torch.randn(2, 7, 3, 6)
    mask = torch.ones(2, 7, 6)
    positions = torch.tensor(
        [[2, 31, 68, 112, 171, 239, 321], [7, 38, 75, 124, 184, 251, 339]],
        dtype=torch.long,
    )
    logits = model(pixels, mask, positions, None)
    assert logits.shape == (2, 2)
    logits.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.spatial_encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_factory_exposes_only_neutral_fourier_model(monkeypatch):
    tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
    tensorboard_stub.SummaryWriter = object
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tensorboard_stub)
    torchvision_stub = types.ModuleType("torchvision")
    transforms_stub = types.ModuleType("torchvision.transforms")
    transforms_stub.transforms = transforms_stub
    transforms_stub.Compose = lambda values: values
    torchvision_stub.transforms = transforms_stub
    monkeypatch.setitem(sys.modules, "torchvision", torchvision_stub)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms_stub)
    for module_name, function_name in (
        ("competitors.dann.dann", "train_dann"),
        ("competitors.jumbot.jumbot", "train_jumbot"),
        ("competitors.mmd.train_mmd", "train_mmd"),
        ("competitors.alda.train_alda", "train_alda"),
    ):
        module_stub = types.ModuleType(module_name)
        setattr(module_stub, function_name, lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, module_name, module_stub)
    import train
    from models.stclassifier import PseFourierReconLTae, PseLTae

    parser = argparse.ArgumentParser()
    train.add_model_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.model == "pseltae"
    assert isinstance(
        train.create_model(
            argparse.Namespace(
                model="pseltae", input_dim=10, num_classes=6, with_extra=True
            )
        ),
        PseLTae,
    )

    configured = parser.parse_args(
        [
            "--model",
            "psefourierreconltae",
            "--fourier_num_modes",
            "13",
            "--fourier_reg",
            "0.001",
            "--fourier_period_days",
            "365",
        ]
    )
    configured.input_dim = 10
    configured.num_classes = 6
    configured.with_extra = True
    assert isinstance(train.create_model(configured), PseFourierReconLTae)

    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "psefrednltae"])
