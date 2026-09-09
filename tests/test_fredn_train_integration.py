import argparse
import inspect
from pathlib import Path
import sys
import types

import torch


class _Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


torchvision_stub = types.ModuleType("torchvision")
transforms_stub = types.ModuleType("torchvision.transforms")
transforms_stub.Compose = _Compose
transforms_stub.transforms = transforms_stub
torchvision_stub.transforms = transforms_stub
sys.modules.setdefault("torchvision", torchvision_stub)
sys.modules.setdefault("torchvision.transforms", transforms_stub)

tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
tensorboard_stub.SummaryWriter = object
sys.modules.setdefault("torch.utils.tensorboard", tensorboard_stub)
for module_name, function_name in (
    ("competitors.dann.dann", "train_dann"),
    ("competitors.jumbot.jumbot", "train_jumbot"),
    ("competitors.mmd.train_mmd", "train_mmd"),
    ("competitors.alda.train_alda", "train_alda"),
):
    module_stub = types.ModuleType(module_name)
    setattr(module_stub, function_name, lambda *args, **kwargs: None)
    sys.modules.setdefault(module_name, module_stub)

import timematch
import train
from models.stclassifier import PseFourierReconLTae, PseLTae


def _small_model():
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


def test_model_arguments_keep_baseline_default_and_expose_neutral_fourier_flags():
    parser = argparse.ArgumentParser()
    train.add_model_arguments(parser)
    defaults = parser.parse_args([])
    configured = parser.parse_args(
        [
            "--model",
            "psefourierreconltae",
            "--fourier_num_modes",
            "13",
            "--fourier_period_days",
            "365",
        ]
    )
    assert defaults.model == "pseltae"
    assert configured.model == "psefourierreconltae"
    assert configured.fourier_num_modes == 13
    assert configured.fourier_period_days == 365.0


def test_factory_keeps_pseltae_and_builds_fourier_reconstruction():
    common = dict(input_dim=10, num_classes=6, with_extra=True)
    baseline = train.create_model(argparse.Namespace(model="pseltae", **common))
    reconstructed = train.create_model(
        argparse.Namespace(
            model="psefourierreconltae",
            fourier_num_modes=13,
            fourier_reg=1e-3,
            fourier_period_days=365.0,
            fourier_solver="dense_direct",
            **common,
        )
    )
    assert isinstance(baseline, PseLTae)
    assert isinstance(reconstructed, PseFourierReconLTae)
    assert not hasattr(baseline, "fourier_analyzer")


def test_shift_sweep_runs_fourier_analysis_once(monkeypatch):
    model = _small_model().eval()
    calls = 0
    original_forward = model.fourier_analyzer.forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.fourier_analyzer, "forward", counted_forward)
    sample = {
        "pixels": torch.randn(2, 7, 3, 6),
        "valid_pixels": torch.ones(2, 7, 6),
        "positions": torch.tensor(
            [[2, 31, 68, 112, 171, 239, 321], [7, 38, 75, 124, 184, 251, 339]],
            dtype=torch.long,
        ),
        "extra": None,
        "label": torch.tensor([0, 1]),
    }
    monkeypatch.setattr(
        timematch,
        "to_cuda",
        lambda value, device: (
            value["pixels"],
            value["valid_pixels"],
            value["positions"],
            value["extra"],
        ),
    )
    timematch.estimate_temporal_shift(
        model,
        [sample],
        "cpu",
        min_shift=-2,
        max_shift=2,
        sample_size=1,
        shift_estimator="IS",
        progress_bar="off",
    )
    assert calls == 1


def test_active_runtime_has_no_historical_fredn_entrypoints():
    train_source = inspect.getsource(train)
    timematch_source = inspect.getsource(timematch)
    for source in (train_source, timematch_source):
        assert "psefrednltae" not in source.lower()
        assert "models.fredn" not in source
        assert "FrequencyDisentangler" not in source
        assert "FREDN_MASK" not in source


def test_fourier_reconstruction_launcher_maps_four_workers_and_is_offline():
    source = Path(
        "scripts/run_fourier_recon13_timematch_4tasks_seed1.sh"
    ).read_text(encoding="utf-8")
    for command in (
        'launch_worker "$GPU0" AT1 "$AT1" DK1 "$DK1"',
        'launch_worker "$GPU1" DK1 "$DK1" FR1 "$FR1"',
        'launch_worker "$GPU2" FR1 "$FR1" FR2 "$FR2"',
        'launch_worker "$GPU3" FR2 "$FR2" AT1 "$AT1"',
    ):
        assert command in source
    assert source.count("--fourier_num_modes 13") == 2
    assert source.count("--model psefourierreconltae") == 2
    assert "nohup bash" in source
    assert "git " not in source.lower()
    assert "pip install" not in source.lower()
    assert "conda" not in source.lower()
