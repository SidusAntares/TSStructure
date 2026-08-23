from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import timematch
from models.reimts_classifier import ReIMTSClassificationOutput


class _Baseline(nn.Module):
    def forward(self, pixels, mask, positions, extra):
        scores = positions.float().mean(dim=1)
        return torch.stack([scores, -scores], dim=1)


class _Capable(nn.Module):
    def __init__(self):
        super().__init__()
        self.split_positions = None
        self.encoder_positions = None

    def forward_with_shift(self, pixels, mask, positions, extra, shift):
        self.split_positions = positions.clone()
        self.encoder_positions = positions + shift
        scores = self.encoder_positions.float().mean(dim=1)
        return torch.stack([scores, -scores], dim=1)

    def forward_for_loss(self, *args, **kwargs):
        raise AssertionError("teacher/pseudo path must use sample-level forward")


def test_forward_model_with_shift_preserves_exact_baseline_path():
    model = _Baseline()
    pixels = torch.randn(2, 1, 1, 1)
    mask = torch.ones(2, 1, 1)
    positions = torch.tensor([[10, 20], [30, 40]])

    expected = model(pixels, mask, positions + 7, None)
    actual = timematch.forward_model_with_shift(
        model, pixels, mask, positions, None, shift=7
    )

    assert torch.equal(actual, expected)


def test_forward_model_with_shift_separates_reimts_split_and_encoder_time():
    model = _Capable()
    positions = torch.tensor([[80, 100, 180, 280]])

    logits = timematch.forward_model_with_shift(
        model,
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 1, 1),
        positions,
        None,
        shift=30,
    )

    assert logits.shape == (1, 2)
    assert torch.equal(model.split_positions, positions)
    assert torch.equal(model.encoder_positions, positions + 30)


class _CountingSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixels, valid_pixels, extra):
        self.calls += 1
        return pixels.flatten(2).mean(dim=2, keepdim=True)


class _ShiftEstimatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_encoder = _CountingSpatial()
        self.seen = []

    def forward_from_spatial_with_shift(self, spatial, positions, shift):
        self.seen.append((positions.clone(), shift))
        score = spatial.mean(dim=(1, 2)) + float(shift)
        return torch.stack([score, -score], dim=1)


class _TiedShiftEstimatorModel(_ShiftEstimatorModel):
    def forward_from_spatial_with_shift(self, spatial, positions, shift):
        self.seen.append((positions.clone(), shift))
        score = torch.ones(spatial.shape[0])
        return torch.stack([score, -score], dim=1)


def test_estimate_shift_runs_pse_once_changes_only_encoder_shift_and_logs_block(monkeypatch, capsys):
    model = _ShiftEstimatorModel()
    positions = torch.tensor([[80, 180, 280]])
    sample = {
        "pixels": torch.ones(1, 3, 1, 1),
        "valid_pixels": torch.ones(1, 3, 1),
        "positions": positions,
        "extra": torch.zeros(1, 4),
        "label": torch.tensor([0]),
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

    best = timematch.estimate_temporal_shift(
        model,
        [sample],
        "cpu",
        min_shift=-1,
        max_shift=1,
        sample_size=1,
        shift_estimator="ACC",
        progress_bar="off",
    )

    assert best in (-1, 0, 1)
    assert model.spatial_encoder.calls == 1
    assert [entry[1] for entry in model.seen] == [-1, 0, 1]
    assert all(torch.equal(entry[0], positions) for entry in model.seen)
    logged = capsys.readouterr().out
    assert "[SHIFT ESTIMATION]" in logged
    assert "estimator: ACC" in logged
    assert "candidate_count: 3" in logged
    assert "top5:" in logged
    assert "debug oracle:" in logged


def test_acc_shift_tie_preserves_original_first_candidate_selection(monkeypatch):
    model = _TiedShiftEstimatorModel()
    sample = {
        "pixels": torch.ones(1, 3, 1, 1),
        "valid_pixels": torch.ones(1, 3, 1),
        "positions": torch.tensor([[80, 180, 280]]),
        "extra": torch.zeros(1, 4),
        "label": torch.tensor([0]),
    }
    monkeypatch.setattr(
        timematch,
        "to_cuda",
        lambda value, device: (
            value["pixels"], value["valid_pixels"],
            value["positions"], value["extra"],
        ),
    )

    best = timematch.estimate_temporal_shift(
        model, [sample], "cpu", min_shift=-1, max_shift=1,
        sample_size=1, shift_estimator="ACC", progress_bar="off",
    )

    assert best == -1


def test_pseudo_labels_use_shift_capability_and_remain_sample_level(monkeypatch):
    model = _Capable()
    sample = {
        "pixels": torch.ones(2, 1, 1, 1),
        "valid_pixels": torch.ones(2, 1, 1),
        "positions": torch.tensor([[10], [20]]),
        "extra": torch.zeros(2, 4),
        "index": torch.tensor([1, 0]),
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

    pseudo = timematch.get_pseudo_labels(
        model, [sample], "cpu", best_shift=5, n=None, progress_bar="off"
    )

    assert pseudo.shape == (2, 2)
    assert torch.equal(model.split_positions, sample["positions"])
    assert torch.equal(model.encoder_positions, sample["positions"] + 5)


class _StudentPatchModel(nn.Module):
    def forward_for_loss(self, pixels, mask, positions, extra, shift=0):
        patch_logits = pixels
        return ReIMTSClassificationOutput(
            sample_logits=patch_logits.mean(dim=1),
            patch_logits=patch_logits,
            patch_valid=mask.bool(),
        )


def test_timematch_student_helper_preserves_patch_output_and_valid_loss():
    model = _StudentPatchModel()
    patch_logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 3.0], [2.0, 1.0], [9.0, -9.0]],
            [[0.0, 2.0], [1.0, 2.0], [-5.0, 5.0], [4.0, 0.0]],
        ]
    )
    patch_valid = torch.tensor(
        [[True, True, True, False], [True, True, False, True]]
    )
    targets = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss()

    output = timematch.forward_model_for_loss_with_shift(
        model,
        patch_logits,
        patch_valid,
        torch.zeros(2, 1, dtype=torch.long),
        None,
        shift=0,
    )
    loss = timematch.student_classification_loss(
        output, targets, criterion, loss_mode="patch"
    )
    expected_targets = targets.unsqueeze(1).expand(-1, 4)
    expected = criterion(
        patch_logits[patch_valid], expected_targets[patch_valid]
    )

    assert output.sample_logits.shape == (2, 2)
    assert output.patch_logits.shape == (2, 4, 2)
    assert torch.allclose(loss, expected)
