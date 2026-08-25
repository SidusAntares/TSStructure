import argparse
from copy import deepcopy
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
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

import train
from models.fredn.diagnostics import log_fredn_diagnostics
from models.fredn.nufft import (
    BatchedDirectFourierAnalyzer,
    DenseFourierBackend,
    IrregularFourierAnalyzer,
)
from models.stclassifier import PseFreDNLTae, PseLTae
from scripts.report_temporal_positions import summarize_temporal_positions
import timematch


def test_model_arguments_keep_pseltae_default_and_expose_fredn_flags():
    parser = argparse.ArgumentParser()
    train.add_model_arguments(parser)

    defaults = parser.parse_args([])
    configured = parser.parse_args(
        [
            "--model",
            "psefrednltae",
            "--fredn_num_modes",
            "11",
            "--fredn_period_days",
            "730",
        ]
    )

    assert defaults.model == "pseltae"
    assert defaults.fredn_fourier_solver == "dense_direct"
    assert configured.model == "psefrednltae"
    assert configured.fredn_num_modes == 11
    assert configured.fredn_period_days == 730.0


def test_model_factory_builds_baseline_and_fredn_without_changing_baseline():
    baseline = train.create_model(
        SimpleNamespace(
            model="pseltae",
            input_dim=10,
            num_classes=3,
            with_extra=False,
        )
    )
    fredn = train.create_model(
        SimpleNamespace(
            model="psefrednltae",
            input_dim=10,
            num_classes=3,
            with_extra=False,
            fredn_num_modes=5,
            fredn_nufft_reg=1e-3,
            fredn_nufft_tol=1e-5,
            fredn_nufft_max_iter=10,
            fredn_period_days=365.0,
        ),
        nufft_backend=DenseFourierBackend(),
    )

    assert isinstance(baseline, PseLTae)
    assert isinstance(fredn, PseFreDNLTae)
    assert isinstance(fredn.fourier_analyzer, BatchedDirectFourierAnalyzer)


def test_model_factory_keeps_nufft_cg_as_explicit_reference_backend():
    fredn = train.create_model(
        SimpleNamespace(
            model="psefrednltae",
            input_dim=10,
            num_classes=3,
            with_extra=False,
            fredn_num_modes=5,
            fredn_nufft_reg=1e-3,
            fredn_nufft_tol=1e-5,
            fredn_nufft_max_iter=10,
            fredn_period_days=365.0,
            fredn_fourier_solver="nufft_cg",
        ),
        nufft_backend=DenseFourierBackend(),
    )

    assert isinstance(fredn.fourier_analyzer, IrregularFourierAnalyzer)


def test_model_factory_rejects_even_mode_count():
    config = SimpleNamespace(
        model="psefrednltae",
        input_dim=10,
        num_classes=3,
        with_extra=False,
        fredn_num_modes=4,
        fredn_nufft_reg=1e-3,
        fredn_nufft_tol=1e-5,
        fredn_nufft_max_iter=10,
        fredn_period_days=365.0,
    )

    with pytest.raises(ValueError, match="positive odd"):
        train.create_model(config, nufft_backend=DenseFourierBackend())


class _Writer:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, step):
        self.scalars[name] = (value, step)


def test_required_fredn_diagnostics_are_logged(capsys):
    model = SimpleNamespace(
        last_diagnostics={
            "mask_mean": 0.4,
            "mask_std": 0.1,
            "mask_lt_0.1": 0.0,
            "mask_gt_0.9": 0.0,
            "trend_energy_ratio": 0.2,
            "seasonal_energy_ratio": 0.5,
            "reconstruction_error": 0.03,
            "additivity_error": 1e-7,
            "solver_iterations": 4,
            "shared_points_rate": 0.75,
            "analysis_time": 0.01,
            "synthesis_time": 0.02,
            "imaginary_residual": 1e-8,
            "frequency_mask_mean": [0.2, 0.4, 0.2],
        }
    )
    writer = _Writer()

    log_fredn_diagnostics(model, writer, step=10, frequency_log_interval=10)

    required = {
        "fredn/mask_mean",
        "fredn/mask_std",
        "fredn/mask_lt_0.1",
        "fredn/mask_gt_0.9",
        "fredn/trend_energy_ratio",
        "fredn/seasonal_energy_ratio",
        "fredn/reconstruction_error",
        "fredn/additivity_error",
        "fredn/nufft_solver_iterations",
    }
    assert required.issubset(writer.scalars)
    assert "FREDN_MASK_BY_FREQUENCY" in capsys.readouterr().out


def _write_metadata(root: Path, domain: str, start_date: int, dates):
    metadata_path = root / Path(domain) / "meta" / "metadata.pkl"
    metadata_path.parent.mkdir(parents=True)
    with metadata_path.open("wb") as handle:
        pickle.dump(
            {"start_date": start_date, "dates": dates, "parcels": []},
            handle,
        )


def test_metadata_report_detects_non_shared_calendar_origins(tmp_path):
    domains = ("country/a/2017", "country/b/2017")
    _write_metadata(tmp_path, domains[0], 20170101, [20170111, 20170201])
    _write_metadata(tmp_path, domains[1], 20170105, [20170111, 20170210])

    report = summarize_temporal_positions(str(tmp_path), domains=domains)

    assert report["origins_consistent"] is False
    assert report["domains"][0]["position_min"] == 10
    assert report["domains"][0]["position_max"] == 31
    assert report["domains"][1]["position_min"] == 6


class _Loader:
    def __init__(self, batches, labels):
        self.batches = batches
        self.dataset = SimpleNamespace(get_labels=lambda: np.asarray(labels))

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _model_batch(pixel_offset=0.0):
    torch.manual_seed(31)
    return {
        "pixels": torch.randn(2, 5, 2, 3) + pixel_offset,
        "valid_pixels": torch.ones(2, 5, 3),
        "positions": torch.tensor(
            [[5, 40, 90, 160, 260], [8, 44, 95, 165, 265]],
            dtype=torch.long,
        ),
        "extra": torch.zeros(2, 4),
        "label": torch.tensor([0, 1], dtype=torch.long),
    }


def _tiny_fredn_for_timematch():
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
        num_classes=2,
        fredn_num_modes=3,
        fredn_nufft_reg=1e-3,
        fredn_nufft_tol=1e-6,
        fredn_nufft_max_iter=12,
        fredn_period_days=365.0,
        nufft_backend=DenseFourierBackend(),
    )


def test_psefrednltae_timematch_one_step_smoke(monkeypatch):
    model = _tiny_fredn_for_timematch()
    source = _model_batch(0.0)
    target_weak = _model_batch(1.0)
    target_strong = _model_batch(1.5)
    source_loader = _Loader([source], [0, 1])
    target_no_aug = _Loader([], [0, 1])
    target_loader = _Loader([(target_weak, target_strong)], [0, 1])
    monkeypatch.setattr(
        timematch,
        "get_data_loaders",
        lambda *args, **kwargs: (
            source_loader,
            target_no_aug,
            target_loader,
        ),
    )
    monkeypatch.setattr(
        timematch,
        "to_cuda",
        lambda sample, device: (
            sample["pixels"],
            sample["valid_pixels"],
            sample["positions"],
            sample["extra"],
        ),
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"state_dict": deepcopy(model.state_dict())},
    )
    saved = []
    monkeypatch.setattr(torch, "save", lambda value, path: saved.append(value))
    writer = _Writer()
    config = SimpleNamespace(
        balance_source=False,
        weights="weights",
        use_focal_loss=False,
        focal_loss_gamma=1.0,
        steps_per_epoch=1,
        lr=1e-3,
        weight_decay=0.0,
        epochs=1,
        max_temporal_shift=60,
        num_classes=2,
        estimate_shift=False,
        pseudo_threshold=0.0,
        domain_specific_bn=True,
        batch_size=2,
        trade_off=1.0,
        ema_decay=0.99,
        log_step=1,
        run_validation=False,
        output_student=True,
        progress_bar="off",
    )

    timematch.train_timematch(
        model,
        config,
        writer,
        val_loader=None,
        device="cpu",
        best_model_path="unused.pt",
        fold_num=0,
        splits={},
    )

    assert saved
    assert all(torch.isfinite(value).all() for value in saved[0]["state_dict"].values())
    assert "fredn/additivity_error" in writer.scalars


def test_fredn_shift_sweep_runs_fourier_analysis_once(monkeypatch):
    model = _tiny_fredn_for_timematch().eval()
    sample = _model_batch()
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
    calls = []
    hook = model.fourier_analyzer.register_forward_hook(
        lambda module, inputs, output: calls.append(1)
    )
    try:
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
    finally:
        hook.remove()

    assert calls == [1]
