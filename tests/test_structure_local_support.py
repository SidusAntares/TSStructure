import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import timematch
from methods.structure_da.prototype_losses import class_local_support_alignment


def _config(**overrides):
    values = dict(
        model="psestructureprotoltae",
        shape_representation="current",
        shape_injection="current_query",
        shape_da_mode="local_support",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cli_and_manifest_expose_only_the_frozen_local_support_controls():
    source = Path("train.py").read_text(encoding="utf-8")
    assert '"local_support"' in source
    assert '"--local-support-k"' in source
    assert '"--local-support-temperature"' in source
    for field in (
        "class_conditional_local_support", "source_bank_refresh",
        "source_bank_fixed_within_epoch", "source_bank_detached",
        "target_gt_used",
    ):
        assert field in source


@pytest.mark.parametrize("field,value", [
    ("model", "pseltae"),
    ("shape_representation", "residual_response"),
    ("shape_injection", "direct_response_query"),
])
def test_local_support_accepts_only_current_current_query(field, value):
    with pytest.raises(ValueError, match="local_support"):
        timematch.validate_local_support_config(_config(**{field: value}))


def _toy_alignment(features, labels, confidence, mask, bank, **kwargs):
    return class_local_support_alignment(
        features, labels, confidence, mask, bank,
        k=kwargs.get("k", 5), temperature=kwargs.get("temperature", .1),
        pseudo_threshold=kwargs.get("pseudo_threshold", .9),
        min_target_support=kwargs.get("min_target_support", 2),
        support_saturation=kwargs.get("support_saturation", 4),
    )


def test_local_support_topk_temperature_detach_and_target_gradient():
    bank = {
        0: F.normalize(torch.tensor([
            [1., 0.], [.9, .1], [.8, .2], [0., 1.], [-1., 0.],
        ]), dim=-1).detach(),
        1: F.normalize(torch.tensor([[0., 1.], [.1, .9]]), dim=-1).detach(),
    }
    target = torch.tensor([[1., .05], [.95, .1]], requires_grad=True)
    result = _toy_alignment(
        target, torch.tensor([0, 0]), torch.tensor([.95, .99]),
        torch.tensor([True, True]), bank, k=3,
    )
    expected_sim, expected_idx = torch.topk(
        F.normalize(target.detach(), dim=-1) @ bank[0].T, 3, dim=1,
    )
    expected_alpha = F.softmax(expected_sim / .1, dim=1)
    expected_proto = F.normalize(
        (expected_alpha[..., None] * bank[0][expected_idx]).sum(1), dim=-1,
    )
    torch.testing.assert_close(result["local_prototypes"], expected_proto)
    assert not result["local_prototypes"].requires_grad
    assert result["valid_targets"] == 2
    result["total_loss"].backward()
    assert target.grad is not None and torch.isfinite(target.grad).all()


def test_search_is_pseudo_class_only_and_k_is_clamped_to_class_support():
    bank = {
        0: torch.tensor([[1., 0.]]),
        1: torch.tensor([[0., 1.], [0., .9]]),
    }
    target = torch.tensor([[0., 1.], [0., 1.]], requires_grad=True)
    result = _toy_alignment(
        target, torch.tensor([0, 0]), torch.tensor([.95, .95]),
        torch.tensor([True, True]), bank, k=5,
    )
    torch.testing.assert_close(result["local_prototypes"], torch.tensor([[1., 0.], [1., 0.]]))
    assert result["k_effective"].tolist() == [1, 1]


def test_trusted_confidence_class_balancing_and_empty_case_are_exact():
    bank = {0: torch.tensor([[1., 0.]]), 1: torch.tensor([[0., 1.]])}
    target = torch.tensor([
        [1., 0.], [0., 1.], [1., 0.], [1., 0.], [1., 0.],
    ], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 1])
    confidence = torch.tensor([.95, 1., .95, 1., .95])
    mask = torch.tensor([True, True, True, True, False])
    result = _toy_alignment(target, labels, confidence, mask, bank)
    # Class 0 has losses 0 and 1 with confidence weights .5 and 1 => 2/3.
    # Class 1 has losses 1 and 1 => 1. Reliability is support/4 * mean(w).
    r0, r1 = .5 * .75, .5 * .75
    expected = (r0 * (2. / 3.) + r1) / (r0 + r1)
    assert float(result["total_loss"]) == pytest.approx(expected, abs=1e-6)
    empty = _toy_alignment(target, labels, confidence, torch.zeros(5, dtype=torch.bool), bank)
    assert float(empty["total_loss"]) == 0. and torch.isfinite(empty["total_loss"])
    assert empty["valid_classes"] == empty["valid_targets"] == 0


class _BankDataset(Dataset):
    def __init__(self):
        self.labels = torch.tensor([0, 1, 0, 1, 1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "pixels": torch.tensor([float(index)]),
            "valid_pixels": torch.ones(1),
            "positions": torch.zeros(1, dtype=torch.long),
            "label": self.labels[index],
        }


class _BankModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(1.))

    def forward_with_temporal_shift(self, pixels, *_args, **_kwargs):
        index = pixels[:, 0]
        return {"shapelet_response": torch.stack([
            index + 1., torch.ones_like(index), *[torch.zeros_like(index) for _ in range(30)]
        ], dim=1)}


def test_source_bank_replays_full_dataset_normalized_detached_and_reproducible(monkeypatch):
    monkeypatch.setattr(
        timematch, "to_cuda",
        lambda sample, _device: (
            sample["pixels"], sample["valid_pixels"], sample["positions"], None,
        ),
    )
    loader = DataLoader(_BankDataset(), batch_size=2, shuffle=True, num_workers=0)
    model = _BankModel().train()
    first, first_diag = timematch.build_source_shape_response_bank(model, loader, "cpu", seed=9)
    second, second_diag = timematch.build_source_shape_response_bank(model, loader, "cpu", seed=9)
    assert first_diag == second_diag == {
        "samples": 5, "classes": 2, "min_class_support": 2, "max_class_support": 3,
    }
    assert model.training
    assert first[0].shape == (2, 32) and first[1].shape == (3, 32)
    for class_id in first:
        assert not first[class_id].requires_grad
        torch.testing.assert_close(first[class_id].norm(dim=1), torch.ones(len(first[class_id])))
        torch.testing.assert_close(first[class_id], second[class_id])


def test_trainer_rebuilds_once_per_epoch_and_local_mode_bypasses_old_alignments():
    trainer = inspect.getsource(timematch._train_structure_proto_timematch)
    assert trainer.count("build_source_shape_response_bank(") == 1
    assert 'if shape_da_mode == "local_support"' in trainer
    assert "class_local_support_alignment(" in trainer
    assert "target_gt" not in inspect.signature(class_local_support_alignment).parameters


def test_launcher_is_uda_only_and_keeps_all_four_existing_task_mappings():
    text = Path("scripts/run_structure_local_support_4tasks_4gpu_seed1.sh").read_text()
    for value in (
        "SHAPE_DA_MODE=local_support", "shape_representation=current",
        "shape_injection=current_query", "source_retrained=false",
        "source_bank_refresh=every_epoch", "source_bank_fixed_within_epoch=true",
        "local_support_k=5", "local_support_temperature=0.1",
        "target_gt_used=false", "--shape-da-mode local_support",
        "--local-support-k 5", "--local-support-temperature 0.1",
    ):
        assert value in text
    assert "--epochs 100" not in text
    for call in (
        'run_task "$GPU0" AT1 "$AT1" DK1 "$DK1"',
        'run_task "$GPU1" FR1 "$FR1" FR2 "$FR2"',
        'run_task "$GPU2" FR2 "$FR2" DK1 "$DK1"',
        'run_task "$GPU3" DK1 "$DK1" AT1 "$AT1"',
    ):
        assert call in text
