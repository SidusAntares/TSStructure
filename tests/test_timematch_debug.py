import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

import timematch
from transforms import Identity, RandomTemporalShift, RandomSampleTimeSteps


class _Dataset:
    def __init__(
        self,
        data_root,
        dataset_name,
        classes,
        transform=None,
        indices=None,
        closed_set=False,
        combine_spring_and_winter=False,
    ):
        self.transform = transform
        self.labels = np.array([0, 1])

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels


class _Loader:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset

    def __len__(self):
        return 1


def _strong_transform(monkeypatch, with_shift_aug):
    monkeypatch.setattr(timematch, "PixelSetData", _Dataset)
    monkeypatch.setattr(timematch.data, "DataLoader", _Loader)
    config = SimpleNamespace(
        data_root="data",
        source="source",
        target="target",
        classes=["crop", "unknown"],
        num_pixels=2,
        seq_length=3,
        with_shift_aug=with_shift_aug,
        max_shift_aug=5,
        shift_aug_p=1.0,
        num_workers=0,
        batch_size=2,
        closed_set=False,
        combine_spring_and_winter=False,
    )
    splits = {
        "source": {"train": [0, 1]},
        "target": {"train": [0, 1]},
    }
    _, _, target_loader = timematch.get_data_loaders(
        splits, config, balance_source=False
    )
    return target_loader.dataset.strong.transform


def _sample():
    return {
        "pixels": np.repeat(
            np.arange(1, 6, dtype=np.float32)[:, None, None], 2, axis=2
        ),
        "valid_pixels": np.ones((5, 2), dtype=np.float32),
        "positions": np.array([100, 110, 120, 130, 140]),
        "extra": np.ones(4, dtype=np.float32),
        "label": 0,
    }


def _selected_original_positions(transformed):
    time_ids = torch.round(transformed["pixels"][:, 0, 0] * 65535).long() - 1
    original = torch.tensor([100, 110, 120, 130, 140])
    return original[time_ids]


def test_strong_augmentation_keeps_positions_unshifted_when_disabled(monkeypatch):
    monkeypatch.setitem(np.__dict__, "long", np.int64)
    transform = _strong_transform(monkeypatch, with_shift_aug=False)

    assert isinstance(transform.transforms[1], RandomSampleTimeSteps)
    assert isinstance(transform.transforms[2], Identity)

    random.seed(7)
    transformed = transform(deepcopy(_sample()))
    assert torch.equal(
        transformed["positions"], _selected_original_positions(transformed)
    )


def test_strong_augmentation_shifts_all_selected_positions_when_enabled(monkeypatch):
    monkeypatch.setitem(np.__dict__, "long", np.int64)
    transform = _strong_transform(monkeypatch, with_shift_aug=True)

    assert isinstance(transform.transforms[1], RandomSampleTimeSteps)
    assert isinstance(transform.transforms[2], RandomTemporalShift)

    random.seed(7)
    transformed = transform(deepcopy(_sample()))
    original_positions = _selected_original_positions(transformed)
    shifts = transformed["positions"] - original_positions

    assert torch.unique(shifts).numel() == 1
    assert shifts[0].item() != 0
    assert -5 <= shifts[0].item() <= 5
    assert transformed["pixels"].shape[0] == 3
    assert transformed["positions"].shape[0] == 3
    assert transformed["valid_pixels"].shape[0] == 3


def test_temporal_index_check_accepts_empty_positions():
    timematch._check_temporal_index_range(
        object(), torch.empty((0, 30), dtype=torch.long), 0, "target"
    )


class _TemporalEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_temporal_shift = 100
        self.positional_enc = torch.nn.Embedding(565, 1)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.temporal_encoder = _TemporalEncoder()
        self.forward_batch_sizes = []

    def forward(self, pixels, mask, positions, extra):
        self.forward_batch_sizes.append(pixels.shape[0])
        scores = pixels.flatten(1).mean(dim=1) * self.scale
        return torch.stack([scores, torch.zeros_like(scores)], dim=1)


class _BatchLoader:
    def __init__(self, batches, labels=None):
        self.batches = batches
        self.dataset = SimpleNamespace(
            get_labels=lambda: np.array([0, 1]) if labels is None else labels
        )

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def _batch(batch_size, pixel_value):
    return {
        "pixels": torch.full((batch_size, 1, 1, 1), pixel_value),
        "valid_pixels": torch.ones(batch_size, 1, 1),
        "positions": torch.zeros(batch_size, 1, dtype=torch.long),
        "extra": torch.ones(batch_size, 4),
        "label": torch.zeros(batch_size, dtype=torch.long),
    }


def _run_one_training_step(
    monkeypatch,
    target_weak_value,
    target_batch_size=1,
    domain_specific_bn=False,
):
    source = _batch(2, 1.0)
    target_weak = _batch(target_batch_size, target_weak_value)
    target_strong = _batch(target_batch_size, 2.0)
    source_loader = _BatchLoader([source])
    target_no_aug = _BatchLoader([], labels=np.array([0, 1]))
    target_loader = _BatchLoader([(target_weak, target_strong)])

    monkeypatch.setattr(
        timematch,
        "get_data_loaders",
        lambda *args, **kwargs: (source_loader, target_no_aug, target_loader),
    )
    monkeypatch.setattr(timematch, "to_cuda", lambda sample, device: (
        sample["pixels"],
        sample["valid_pixels"],
        sample["positions"],
        sample["extra"],
    ))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)

    model = _Model()
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"state_dict": deepcopy(model.state_dict())},
    )
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: None)

    config = SimpleNamespace(
        balance_source=False,
        weights="weights",
        use_focal_loss=False,
        steps_per_epoch=1,
        lr=0.01,
        weight_decay=0.0,
        epochs=1,
        max_temporal_shift=60,
        num_classes=2,
        estimate_shift=False,
        pseudo_threshold=0.9,
        domain_specific_bn=domain_specific_bn,
        batch_size=99,
        trade_off=2.0,
        ema_decay=0.99,
        log_step=1,
        run_validation=False,
        output_student=True,
    )
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
    return model


def test_empty_pseudo_label_batch_trains_source_only(monkeypatch):
    model = _run_one_training_step(monkeypatch, target_weak_value=0.0)

    assert model.forward_batch_sizes == [2]
    assert torch.isfinite(model.scale)


def test_nonempty_pseudo_label_batch_concatenates_and_splits_by_actual_size(
    monkeypatch,
):
    model = _run_one_training_step(monkeypatch, target_weak_value=10.0)

    assert model.forward_batch_sizes == [3]
    assert torch.isfinite(model.scale)


def test_domain_specific_bn_requires_two_pseudo_labels_for_target_forward(
    monkeypatch,
):
    one_pseudo = _run_one_training_step(
        monkeypatch,
        target_weak_value=10.0,
        target_batch_size=1,
        domain_specific_bn=True,
    )
    assert one_pseudo.forward_batch_sizes == [2]

    two_pseudo = _run_one_training_step(
        monkeypatch,
        target_weak_value=10.0,
        target_batch_size=2,
        domain_specific_bn=True,
    )
    assert two_pseudo.forward_batch_sizes == [2, 2]
