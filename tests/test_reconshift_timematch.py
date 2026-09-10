import sys
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
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

import timematch


class _Spatial(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixels, mask, extra):
        self.calls += 1
        return pixels


class _ShiftModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_encoder = _Spatial()
        self.prepared_ids = []
        self.positions_seen = []

    def prepare_temporal_features(self, spatial_feats, positions):
        return spatial_feats

    def classify_prepared(self, prepared, positions, temporal_shift=0):
        self.prepared_ids.append(id(prepared))
        self.positions_seen.append((positions.clone(), temporal_shift))
        score = prepared.mean(dim=(1, 2)) + float(temporal_shift) * 0.01
        return torch.stack([score, -score], dim=1)


class _CountingAnalyzer(torch.nn.Module):
    calls = 0
    positions = []

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, features, positions):
        type(self).calls += 1
        type(self).positions.append(positions.clone())
        return torch.complex(features, torch.zeros_like(features)), {}


class _CountingSynthesizer(torch.nn.Module):
    calls = 0
    positions = []

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, coeffs, positions):
        type(self).calls += 1
        type(self).positions.append(positions.clone())
        return coeffs.real + 1.0


def _sample(labels=(0, 1)):
    return {
        "pixels": torch.ones(2, 3, 4),
        "valid_pixels": torch.ones(2, 3),
        "positions": torch.tensor([[10, 30, 80], [12, 35, 90]]),
        "extra": torch.zeros(2, 4),
        "label": torch.tensor(labels),
    }


@pytest.fixture(autouse=True)
def _patch_cuda(monkeypatch):
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
    _CountingAnalyzer.calls = 0
    _CountingAnalyzer.positions = []
    _CountingSynthesizer.calls = 0
    _CountingSynthesizer.positions = []


def test_timematch_cli_defaults_to_raw_shift_view():
    import argparse

    parser = argparse.ArgumentParser()
    timematch.add_shift_estimation_arguments(parser)
    cfg = parser.parse_args([])
    assert cfg.shift_estimation_view == "raw"
    assert cfg.shift_fourier_num_modes == 13
    configured = parser.parse_args(
        [
            "--shift-estimation-view",
            "fourier_recon",
            "--shift-fourier-num-modes",
            "13",
            "--shift-fourier-reg",
            "0.001",
            "--shift-fourier-period-days",
            "365.0",
            "--shift-fourier-solver",
            "dense_direct",
        ]
    )
    assert configured.shift_estimation_view == "fourier_recon"


def test_recon_shift_view_fits_once_and_shifts_only_ltae_positions(monkeypatch):
    monkeypatch.setattr(timematch, "BatchedDirectFourierAnalyzer", _CountingAnalyzer)
    monkeypatch.setattr(timematch, "BatchedDirectFourierSynthesizer", _CountingSynthesizer)
    model = _ShiftModel()
    positions = _sample()["positions"]

    timematch.estimate_temporal_shift(
        model,
        [_sample()],
        "cpu",
        min_shift=-2,
        max_shift=2,
        sample_size=1,
        shift_estimator="IS",
        progress_bar="off",
        shift_estimation_view="fourier_recon",
        shift_fourier_num_modes=13,
        shift_fourier_reg=1e-3,
        shift_fourier_period_days=365.0,
        shift_fourier_solver="dense_direct",
    )

    assert model.spatial_encoder.calls == 1
    assert _CountingAnalyzer.calls == 1
    assert _CountingSynthesizer.calls == 1
    assert torch.equal(_CountingAnalyzer.positions[0], positions)
    assert torch.equal(_CountingSynthesizer.positions[0], positions)
    assert len(set(model.prepared_ids)) == 1
    assert [shift for _, shift in model.positions_seen] == [-2, -1, 0, 1, 2]
    assert all(torch.equal(seen, positions) for seen, _ in model.positions_seen)


def test_raw_shift_view_does_not_construct_fourier_operators(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("raw shift view constructed a Fourier operator")

    monkeypatch.setattr(timematch, "BatchedDirectFourierAnalyzer", forbidden)
    monkeypatch.setattr(timematch, "BatchedDirectFourierSynthesizer", forbidden)
    model = _ShiftModel()
    timematch.estimate_temporal_shift(
        model,
        [_sample()],
        "cpu",
        min_shift=0,
        max_shift=0,
        sample_size=1,
        shift_estimator="IS",
        progress_bar="off",
    )
    assert model.spatial_encoder.calls == 1


def test_shift_selection_does_not_depend_on_target_labels(monkeypatch):
    monkeypatch.setattr(timematch, "BatchedDirectFourierAnalyzer", _CountingAnalyzer)
    monkeypatch.setattr(timematch, "BatchedDirectFourierSynthesizer", _CountingSynthesizer)
    kwargs = dict(
        device="cpu",
        min_shift=-2,
        max_shift=2,
        sample_size=1,
        shift_estimator="IS",
        progress_bar="off",
        shift_estimation_view="fourier_recon",
    )
    first = timematch.estimate_temporal_shift(_ShiftModel(), [_sample((0, 0))], **kwargs)
    second = timematch.estimate_temporal_shift(_ShiftModel(), [_sample((1, 1))], **kwargs)
    assert first == second


def test_recon_shift_comparison_reuses_the_same_spatial_features(monkeypatch):
    monkeypatch.setattr(timematch, "BatchedDirectFourierAnalyzer", _CountingAnalyzer)
    monkeypatch.setattr(timematch, "BatchedDirectFourierSynthesizer", _CountingSynthesizer)
    model = _ShiftModel()

    selected, diagnostics = timematch.estimate_temporal_shift(
        model,
        [_sample()],
        "cpu",
        min_shift=-2,
        max_shift=2,
        sample_size=1,
        shift_estimator="IS",
        progress_bar="off",
        shift_estimation_view="fourier_recon",
        compare_raw=True,
        return_diagnostics=True,
    )

    assert selected == diagnostics["selected_shift"]
    assert diagnostics["raw_selected_shift"] in range(-2, 3)
    assert model.spatial_encoder.calls == 1
    assert _CountingAnalyzer.calls == 1
    assert _CountingSynthesizer.calls == 1
    assert len(model.positions_seen) == 10
    for key in (
        "mean_confidence",
        "prediction_entropy",
        "score_range",
    ):
        assert np.isfinite(diagnostics[key])
    assert 1 <= diagnostics["num_predicted_classes"] <= 2


def test_shift_configuration_is_forwarded_without_affecting_raw_defaults():
    from types import SimpleNamespace

    raw = timematch._shift_estimation_kwargs(SimpleNamespace())
    assert raw == {"shift_estimation_view": "raw"}

    configured = timematch._shift_estimation_kwargs(
        SimpleNamespace(
            shift_estimation_view="fourier_recon",
            shift_fourier_num_modes=13,
            shift_fourier_reg=0.001,
            shift_fourier_period_days=365.0,
            shift_fourier_solver="dense_direct",
        )
    )
    assert configured == {
        "shift_estimation_view": "fourier_recon",
        "shift_fourier_num_modes": 13,
        "shift_fourier_reg": 0.001,
        "shift_fourier_period_days": 365.0,
        "shift_fourier_solver": "dense_direct",
    }


def test_configured_shift_estimator_forwards_recon_view(monkeypatch):
    from types import SimpleNamespace

    calls = []

    def fake_estimator(*args, **kwargs):
        calls.append(kwargs)
        return (7, {"selected_shift": 7})

    monkeypatch.setattr(timematch, "estimate_temporal_shift", fake_estimator)
    config = SimpleNamespace(
        shift_estimation_view="fourier_recon",
        shift_fourier_num_modes=13,
        shift_fourier_reg=0.001,
        shift_fourier_period_days=365.0,
        shift_fourier_solver="dense_direct",
    )
    result = timematch._estimate_temporal_shift_for_config(
        object(),
        object(),
        "cpu",
        config,
        shift_estimator="IS",
        compare_raw=True,
        return_diagnostics=True,
    )

    assert result == (7, {"selected_shift": 7})
    assert calls == [
        {
            "shift_estimator": "IS",
            "compare_raw": True,
            "return_diagnostics": True,
            "shift_estimation_view": "fourier_recon",
            "shift_fourier_num_modes": 13,
            "shift_fourier_reg": 0.001,
            "shift_fourier_period_days": 365.0,
            "shift_fourier_solver": "dense_direct",
        }
    ]


def test_recon_shift_epoch_zero_logs_raw_and_recon_diagnostics(capsys):
    from types import SimpleNamespace

    timematch._log_shift_view_config(
        SimpleNamespace(
            shift_estimation_view="fourier_recon",
            shift_fourier_num_modes=13,
            shift_fourier_reg=0.001,
            shift_fourier_period_days=365.0,
            shift_fourier_solver="dense_direct",
        )
    )
    timematch._log_shift_view_compare(
        epoch=0,
        initial_diagnostics={
            "selected_shift": 8,
            "raw_selected_shift": 5,
            "score_range": 0.4,
        },
        epoch_diagnostics={
            "selected_shift": 7,
            "raw_selected_shift": 4,
            "score_range": 0.3,
            "mean_confidence": 0.8,
            "prediction_entropy": 0.6,
            "num_predicted_classes": 8,
        },
    )
    output = capsys.readouterr().out
    assert "SHIFT_ESTIMATION_VIEW|fourier_recon" in output
    assert "SHIFT_FOURIER_CONFIG|modes=13|reg=0.001|period_days=365.0|solver=dense_direct" in output
    assert "raw_is_shift=5|recon_is_shift=8" in output
    assert "raw_am_shift=4|recon_am_shift=7" in output
    assert "RECON_SHIFT_VIEW_DIAG|mean_confidence=0.800000" in output
    assert "am_score_range=0.300000|is_score_range=0.400000" in output


def test_four_task_launcher_uses_raw_model_and_recon_shift_only():
    script = Path("scripts/run_reconshift13_raw_timematch_4tasks_seed1.sh")
    text = script.read_text(encoding="utf-8")
    assert "--model pseltae" in text
    assert "--shift-estimation-view fourier_recon" in text
    assert "--shift-fourier-num-modes 13" in text
    assert "--shift-fourier-reg 0.001" in text
    assert "--shift-fourier-period-days 365.0" in text
    assert "--shift-fourier-solver dense_direct" in text
    assert 'launch "$GPU0" AT1 "$AT1" DK1 "$DK1" "$AT1_WEIGHTS"' in text
    assert 'launch "$GPU1" DK1 "$DK1" FR1 "$FR1" "$DK1_WEIGHTS"' in text
    assert 'launch "$GPU2" FR1 "$FR1" FR2 "$FR2" "$FR1_WEIGHTS"' in text
    assert 'launch "$GPU3" FR2 "$FR2" AT1 "$AT1" "$FR2_WEIGHTS"' in text
    assert "psefourierreconltae" not in text
    assert "models.fredn" not in text


class _TrainingModel(torch.nn.Module):
    calls = []

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.temporal_encoder = types.SimpleNamespace(
            max_temporal_shift=100,
            positional_enc=torch.nn.Embedding(565, 1),
        )

    def forward_with_temporal_shift(
        self, pixels, mask, positions, extra, temporal_shift=0
    ):
        type(self).calls.append((pixels.shape[0], temporal_shift))
        score = pixels.flatten(1).mean(dim=1) * self.scale
        return torch.stack([score, -score], dim=1)


class _Loader:
    def __init__(self, batches, labels):
        self.batches = batches
        self.dataset = types.SimpleNamespace(get_labels=lambda: np.asarray(labels))

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def _training_batch(value):
    return {
        "pixels": torch.full((2, 1, 1, 1), value),
        "valid_pixels": torch.ones(2, 1, 1),
        "positions": torch.tensor([[50], [60]]),
        "extra": torch.zeros(2, 4),
        "label": torch.tensor([0, 1]),
    }


def test_training_reestimates_with_recon_but_semantic_forwards_stay_raw(
    monkeypatch, tmp_path
):
    source = _training_batch(1.0)
    target_weak = _training_batch(2.0)
    target_strong = _training_batch(3.0)
    source_loader = _Loader([source], [0, 1])
    target_no_aug = _Loader([target_weak], [0, 1])
    target_loader = _Loader([(target_weak, target_strong)], [0, 1])
    monkeypatch.setattr(
        timematch,
        "get_data_loaders",
        lambda *args, **kwargs: (source_loader, target_no_aug, target_loader),
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    model = _TrainingModel()
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"state_dict": deepcopy(model.state_dict())},
    )
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        timematch,
        "get_pseudo_labels",
        lambda *args, **kwargs: torch.tensor([[0.99, 0.01], [0.01, 0.99]]),
    )
    estimator_calls = []

    def fake_estimator(*args, **kwargs):
        estimator_calls.append(kwargs.copy())
        selected = 5 if len(estimator_calls) == 1 else 4
        if kwargs.get("return_diagnostics"):
            return selected, {
                "selected_shift": selected,
                "raw_selected_shift": selected - 1,
                "score_range": 0.2,
                "mean_confidence": 0.8,
                "prediction_entropy": 0.5,
                "num_predicted_classes": 2,
            }
        return selected

    monkeypatch.setattr(timematch, "estimate_temporal_shift", fake_estimator)
    config = types.SimpleNamespace(
        balance_source=False,
        weights="weights",
        use_focal_loss=False,
        steps_per_epoch=1,
        lr=0.01,
        weight_decay=0.0,
        epochs=1,
        max_temporal_shift=60,
        num_classes=2,
        estimate_shift=True,
        shift_estimator="AM",
        sample_size=1,
        shift_source=True,
        pseudo_threshold=0.9,
        domain_specific_bn=True,
        batch_size=2,
        trade_off=2.0,
        ema_decay=0.99,
        log_step=1,
        run_validation=False,
        output_student=True,
        progress_bar="off",
        model="pseltae",
        source="source",
        target="target",
        classes=["a", "b"],
        fold_dir=str(tmp_path),
        shape_align=False,
        class_residual_phase=False,
        shift_estimation_view="fourier_recon",
        shift_fourier_num_modes=13,
        shift_fourier_reg=0.001,
        shift_fourier_period_days=365.0,
        shift_fourier_solver="dense_direct",
    )
    _TrainingModel.calls = []
    timematch.train_timematch(
        model,
        config,
        _Writer(),
        val_loader=None,
        device="cpu",
        best_model_path="unused.pt",
        fold_num=0,
        splits={},
    )

    assert len(estimator_calls) == 2
    assert all(
        call["shift_estimation_view"] == "fourier_recon"
        for call in estimator_calls
    )
    assert all(call["shift_fourier_num_modes"] == 13 for call in estimator_calls)
    shifts = [shift for _, shift in _TrainingModel.calls]
    assert 4 in shifts  # current teacher target-to-source shift
    assert -4 in shifts  # fixed source-to-target shift from epoch zero
    assert 0 in shifts  # target student remains on raw, unshifted positions
