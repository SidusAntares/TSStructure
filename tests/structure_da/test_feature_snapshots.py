from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from methods.structure_da.feature_snapshots import (
    FeatureSnapshotConfig,
    FeatureSnapshotManager,
    deterministic_class_selection,
    create_feature_snapshot_manager,
    load_selected_samples,
)


def test_disabled_snapshot_path_does_not_construct_datasets() -> None:
    class Config:
        feature_snapshot_interval = 0

    assert create_feature_snapshot_manager(
        object(), Config(), {}, device=torch.device("cpu")
    ) is None
from methods.structure_da.full_model import StructureAwareDomainAdaptationModel


class TinySnapshotDataset(Dataset):
    def __init__(self, labels: list[int], *, parcel_offset: int, length: int) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.parcel_indices = np.arange(len(labels), dtype=np.int64) + parcel_offset
        self.length = length

    def __len__(self) -> int:
        return len(self.labels)

    def get_labels(self) -> np.ndarray:
        return self.labels.copy()

    def get_parcel_indices(self) -> np.ndarray:
        return self.parcel_indices.copy()

    def get_shapes(self) -> list[tuple[int, int, int]]:
        return [(self.length, 3, 5 + (index % 2)) for index in range(len(self))]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pixels = 5 + (index % 2)
        generator = torch.Generator().manual_seed(1000 + int(self.parcel_indices[index]))
        return {
            "index": torch.tensor(index),
            "parcel_index": torch.tensor(self.parcel_indices[index]),
            "pixels": torch.randn(self.length, 3, pixels, generator=generator),
            "valid_pixels": torch.ones(self.length, pixels),
            "positions": torch.linspace(0, 365, self.length),
            "extra": torch.zeros(4),
            "label": torch.tensor(self.labels[index]),
        }


def _tiny_model() -> StructureAwareDomainAdaptationModel:
    return StructureAwareDomainAdaptationModel(
        num_classes=2,
        input_dim=3,
        mlp1=[3, 4],
        pooling="mean_std",
        mlp2=[8, 6],
        with_extra=False,
        shape_dim=5,
        temporal_options={
            "canonical_grid_size": 8,
            "roughness_grid_size": 64,
            "trend_num_basis": 5,
            "structure_num_basis": 5,
            "num_shape_basis": 3,
            "num_phase_basis": 3,
            "attribute_projection_dim": 3,
            "shape_hidden_dim": 8,
            "warp_hidden_dim": 6,
            "warp_num_candidates": 2,
        },
        representation_options={
            "n_head": 2,
            "d_k": 2,
            "d_model": 8,
            "ltae_mlp": (8, 4),
            "classifier_hidden": (6,),
            "quality_domain_hidden_dim": 6,
            "dropout": 0.0,
        },
        prototype_options={"radius_buffer_size": 8, "min_radius_samples": 2},
    )


def test_snapshot_interval_triggers_only_positive_multiples() -> None:
    config = FeatureSnapshotConfig(interval=25, samples_per_class=32, dtype="float16")
    assert [epoch for epoch in range(101) if config.should_capture(epoch)] == [25, 50, 75, 100]
    assert not config.should_capture(24)
    assert not config.should_capture(26)
    disabled = FeatureSnapshotConfig(interval=0, samples_per_class=32, dtype="float16")
    assert not any(disabled.should_capture(epoch) for epoch in range(101))


def test_class_selection_is_deterministic_unique_and_bounded() -> None:
    labels = np.asarray([0] * 5 + [1] * 2)
    parcels = np.asarray([20, 10, 30, 40, 50, 70, 60])
    first = deterministic_class_selection(labels, parcels, samples_per_class=3, seed=7)
    second = deterministic_class_selection(labels, parcels, samples_per_class=3, seed=7)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 5
    assert len(np.unique(first)) == len(first)
    assert np.count_nonzero(labels[first] == 0) == 3
    assert np.count_nonzero(labels[first] == 1) == 2


def test_snapshot_persists_selection_and_per_sample_curves_without_metadata_duplication(
    tmp_path: Path,
) -> None:
    source = TinySnapshotDataset([0, 0, 0, 1, 1, 1], parcel_offset=100, length=5)
    target = TinySnapshotDataset([0, 0, 1, 1], parcel_offset=200, length=7)
    model = _tiny_model()
    config = FeatureSnapshotConfig(
        interval=25,
        samples_per_class=2,
        dtype="float16",
        snapshot_dir=tmp_path,
        selection_seed=3,
    )
    manager = FeatureSnapshotManager(
        model,
        source,
        target,
        config,
        device=torch.device("cpu"),
        batch_size=3,
        source_domain="source/domain",
        target_domain="target/domain",
    )

    result = manager.capture(25)
    assert result.status == "SUCCESS"
    selected = load_selected_samples(tmp_path / "selected_samples.npz")
    assert selected.sample_index.dtype == np.int64
    assert selected.domain.dtype == np.uint8
    assert selected.label.dtype == np.int16
    assert len(selected.sample_index) == 8

    epoch_path = tmp_path / "epoch_0025.npz"
    with np.load(epoch_path, allow_pickle=False) as data:
        assert set(data.files) == {
            "epoch", "pse_values", "pse_times", "pse_offsets",
            "aligned_shape", "shape_grid", "shape_valid",
        }
        assert data["epoch"].item() == 25
        assert data["pse_values"].dtype == np.float16
        assert data["aligned_shape"].dtype == np.float16
        assert data["pse_times"].dtype == np.float32
        assert data["pse_offsets"].shape == (9,)
        assert data["pse_offsets"][-1] == 4 * 5 + 4 * 7
        assert data["pse_values"].shape == (48, 6)
        assert data["aligned_shape"].shape == (8, 8, 6)
        assert data["shape_grid"].shape == (8,)
        assert data["shape_valid"].shape == (8,)
        assert not ({"sample_index", "domain", "label"} & set(data.files))
        assert not any(
            forbidden in name
            for name in data.files
            for forbidden in ("mean", "std", "logit", "prototype", "alpha", "loss")
        )

    order = selected.sample_index.copy()
    exists = manager.capture(25)
    assert exists.status == "EXISTS"
    np.testing.assert_array_equal(
        load_selected_samples(tmp_path / "selected_samples.npz").sample_index,
        order,
    )


def test_snapshot_forward_is_read_only_and_restores_training_state(tmp_path: Path) -> None:
    source = TinySnapshotDataset([0, 0, 1, 1], parcel_offset=10, length=5)
    target = TinySnapshotDataset([0, 0, 1, 1], parcel_offset=20, length=6)
    model = _tiny_model().train()
    first_parameter = next(model.parameters())
    first_parameter.grad = torch.full_like(first_parameter, 0.25)
    before_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before_grads = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before_optimizer = deepcopy(optimizer.state_dict())
    manager = FeatureSnapshotManager(
        model,
        source,
        target,
        FeatureSnapshotConfig(25, 2, "float16", tmp_path, 1),
        device=torch.device("cpu"),
        batch_size=2,
        source_domain="source",
        target_domain="target",
    )

    manager.capture(25)

    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        expected = before_grads[name]
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    assert optimizer.state_dict() == before_optimizer


def test_invalid_existing_epoch_snapshot_is_rejected(tmp_path: Path) -> None:
    source = TinySnapshotDataset([0, 1], parcel_offset=10, length=5)
    target = TinySnapshotDataset([0, 1], parcel_offset=20, length=6)
    manager = FeatureSnapshotManager(
        _tiny_model(),
        source,
        target,
        FeatureSnapshotConfig(25, 1, "float16", tmp_path, 1),
        device=torch.device("cpu"),
        batch_size=2,
        source_domain="source",
        target_domain="target",
    )
    tmp_path.mkdir(exist_ok=True)
    np.savez(tmp_path / "epoch_0025.npz", epoch=np.asarray(25))
    with pytest.raises(ValueError, match="snapshot fields"):
        manager.capture(25)
