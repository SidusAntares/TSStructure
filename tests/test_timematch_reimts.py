from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import timematch
from models.reimts_classifier import ReIMTSClassificationOutput
from models.stclassifier import PseLTae


class _Baseline(nn.Module):
    def forward(self, pixels, mask, positions, extra):
        scores = positions.float().mean(dim=1)
        return torch.stack([scores, -scores], dim=1)


class _Capable(nn.Module):
    def __init__(self):
        super().__init__()
        self.observation_positions = None
        self.ltae_positions = None

    def forward_with_shift(self, pixels, mask, positions, extra, shift):
        self.observation_positions = positions.clone()
        self.ltae_positions = positions + shift
        scores = self.ltae_positions.float().mean(dim=1)
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


def test_actual_pseltae_shift_regression_matches_original_direct_forward():
    model = PseLTae(
        input_dim=2, mlp1=[2, 4], pooling="mean_std", mlp2=[8, 8],
        with_extra=False, n_head=1, d_k=2, d_model=8, mlp3=[8, 8],
        dropout=0.0, mlp4=[8], num_classes=2,
    ).eval()
    pixels = torch.randn(2, 4, 2, 3)
    mask = torch.ones(2, 4, 3)
    positions = torch.tensor([[10, 40, 90, 180], [20, 60, 120, 240]])

    with torch.no_grad():
        expected = model(pixels, mask, positions + 7, None)
        actual = timematch.forward_model_with_shift(
            model, pixels, mask, positions, None, shift=7
        )

    assert torch.equal(actual, expected)


def test_forward_model_with_shift_keeps_observation_time_real_and_shifts_ltae_time():
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
    assert torch.equal(model.observation_positions, positions)
    assert torch.equal(model.ltae_positions, positions + 30)


class _ShiftEstimatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pse_calls = 0
        self.reimts_calls = 0
        self.ltae_calls = 0
        self.classifier_calls = 0
        self.seen = []
        self.candidate_shifts = []

    def prepare_shift_features(self, pixels, valid_pixels, positions, extra):
        self.pse_calls += 1
        self.reimts_calls += 1
        return SimpleNamespace(
            tokens=pixels.flatten(2).mean(dim=2, keepdim=True),
            positions=positions.clone(),
            spatial_encoder_time=0.01,
            reimts_mtan_time=0.02,
            total_feature_preparation_time=0.03,
        )

    def forward_from_shift_features(self, features, shift):
        self.ltae_calls += 1
        self.classifier_calls += 1
        self.candidate_shifts.append(shift)
        shifted_positions = features.positions + shift
        self.seen.append((features.tokens.clone(), shifted_positions.clone()))
        score = features.tokens.mean(dim=(1, 2)) + float(shift)
        return torch.stack([score, -score], dim=1)


class _TiedShiftEstimatorModel(_ShiftEstimatorModel):
    def forward_from_shift_features(self, features, shift):
        self.ltae_calls += 1
        self.classifier_calls += 1
        self.seen.append((features.tokens.clone(), features.positions + shift))
        score = torch.ones(features.tokens.shape[0])
        return torch.stack([score, -score], dim=1)


def test_estimate_shift_caches_pse_and_reimts_for_all_121_candidates(monkeypatch, capsys):
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
        min_shift=-60,
        max_shift=60,
        sample_size=1,
        shift_estimator="ACC",
        progress_bar="off",
    )

    assert -60 <= best <= 60
    assert model.pse_calls == 1
    assert model.reimts_calls == 1
    assert model.ltae_calls == 121
    assert model.classifier_calls == 121
    assert all(torch.equal(entry[0], model.seen[0][0]) for entry in model.seen)
    assert torch.equal(model.seen[0][1], positions - 60)
    assert torch.equal(model.seen[-1][1], positions + 60)
    logged = capsys.readouterr().out
    assert "[SHIFT ESTIMATION]" in logged
    assert "estimator: ACC" in logged
    assert "candidate_count: 121" in logged
    assert "top5:" in logged
    assert "debug oracle:" in logged
    assert "feature preparation:" in logged
    assert "spatial_encoder_time:" in logged
    assert "reimts_mtan_time:" in logged
    assert "total_feature_preparation_time:" in logged
    assert "candidate evaluation:" in logged
    assert "ltae_classifier_total_time:" in logged
    assert "mean_time_per_candidate:" in logged
    assert "total_shift_estimation_time:" in logged


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


def test_estimator_applies_each_scalar_candidate_to_the_entire_batch(monkeypatch):
    model = _ShiftEstimatorModel()
    positions = torch.tensor(
        [[0, 100, 364], [7, 120, 300], [20, 200, 340]]
    )
    sample = {
        "pixels": torch.ones(3, 3, 1, 1),
        "valid_pixels": torch.ones(3, 3, 1),
        "positions": positions,
        "extra": torch.zeros(3, 4),
        "label": torch.tensor([0, 1, 0]),
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
        shift_estimator="ACC",
        progress_bar="off",
    )

    assert model.candidate_shifts == [-2, -1, 0, 1, 2]
    for candidate, (_, shifted_positions) in zip(
        model.candidate_shifts, model.seen
    ):
        assert torch.equal(
            shifted_positions - positions,
            torch.full_like(positions, candidate),
        )


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
    assert torch.equal(model.observation_positions, sample["positions"])
    assert torch.equal(model.ltae_positions, sample["positions"] + 5)


class _StudentSampleModel(nn.Module):
    def forward_for_loss(self, pixels, mask, positions, extra, shift=0):
        return ReIMTSClassificationOutput(
            sample_logits=pixels,
            patch_valid=mask.bool(),
        )


def test_timematch_student_loss_is_sample_level_without_label_repeat():
    model = _StudentSampleModel()
    sample_logits = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    patch_valid = torch.tensor(
        [[True, True, True, False], [True, True, False, True]]
    )
    targets = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss()

    output = timematch.forward_model_for_loss_with_shift(
        model,
        sample_logits,
        patch_valid,
        torch.zeros(2, 1, dtype=torch.long),
        None,
        shift=0,
    )
    loss = timematch.student_classification_loss(
        output, targets, criterion, loss_mode="sample"
    )
    expected = criterion(sample_logits, targets)

    assert output.sample_logits.shape == (2, 2)
    assert torch.allclose(loss, expected)
