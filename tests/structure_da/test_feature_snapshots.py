from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import random

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
    load_projection_basis,
)
import methods.structure_da.feature_snapshots as feature_snapshots


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
        self.classes = ["crop_0", "crop_1"]

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


def _force_valid_shape(model: StructureAwareDomainAdaptationModel) -> None:
    original_forward_details = model.forward_details

    def forward_details(*args, **kwargs):
        output = original_forward_details(*args, **kwargs)
        temporal = replace(
            output.temporal,
            shape=replace(
                output.temporal.shape,
                valid=torch.ones_like(output.temporal.shape.valid),
            ),
        )
        return replace(output, temporal=temporal)

    model.forward_details = forward_details


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


def test_snapshot_fits_fixed_projection_once_and_persists_only_pc_curves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = TinySnapshotDataset([0, 0, 0, 1, 1, 1], parcel_offset=100, length=12)
    target = TinySnapshotDataset([0, 0, 1, 1], parcel_offset=200, length=14)
    model = _tiny_model()
    _force_valid_shape(model)
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

    original_fit = feature_snapshots.fit_deterministic_pca
    fit_calls: list[tuple[int, int]] = []

    def counted_fit(values: np.ndarray, num_components: int = 2):
        fit_calls.append(values.shape)
        return original_fit(values, num_components)

    monkeypatch.setattr(feature_snapshots, "fit_deterministic_pca", counted_fit)
    result = manager.capture(25)
    assert result.status == "SUCCESS"
    selected = load_selected_samples(tmp_path / "selected_samples.npz")
    assert selected.sample_index.dtype == np.int64
    assert selected.domain.dtype == np.uint8
    assert selected.label.dtype == np.int16
    assert len(selected.sample_index) == 8

    basis_path = tmp_path / "projection_basis.npz"
    basis = load_projection_basis(basis_path)
    assert len(fit_calls) == 2
    assert fit_calls[0] == (104, 6)
    assert fit_calls[1] == (64, 6)
    assert basis.fitted_epoch == 25
    assert basis.pse_components.shape == (2, 6)
    assert basis.shape_components.shape == (2, 6)
    assert not np.array_equal(basis.pse_components, basis.shape_components)
    for components in (basis.pse_components, basis.shape_components):
        for component in components:
            pivot = int(np.argmax(np.abs(component)))
            assert component[pivot] >= 0

    epoch_path = tmp_path / "epoch_0025.npz"
    with np.load(epoch_path, allow_pickle=False) as data:
        assert set(data.files) == {
            "epoch", "projection_fitted_epoch", "schema_version",
            "pse_pc_values", "pse_times", "pse_offsets",
            "aligned_shape_pc", "shape_grid", "shape_valid",
        }
        assert data["epoch"].item() == 25
        assert data["projection_fitted_epoch"].item() == 25
        assert data["schema_version"].item() == 2
        assert data["pse_pc_values"].dtype == np.float16
        assert data["aligned_shape_pc"].dtype == np.float16
        assert data["pse_times"].dtype == np.float32
        assert data["pse_offsets"].shape == (9,)
        assert data["pse_offsets"][-1] == 4 * 12 + 4 * 14
        assert data["pse_pc_values"].shape == (104, 2)
        assert data["aligned_shape_pc"].shape == (8, 8, 2)
        assert data["shape_grid"].shape == (8,)
        assert data["shape_valid"].shape == (8,)
        assert not ({"sample_index", "domain", "label"} & set(data.files))
        assert "pse_values" not in data.files
        assert "aligned_shape" not in data.files
        assert not any(
            data[name].ndim and data[name].shape[-1] == 6
            for name in data.files
        )
        assert not any(
            forbidden in name
            for name in data.files
            for forbidden in ("mean", "std", "logit", "prototype", "alpha", "loss")
        )

    manager.capture(50)
    manager.capture(75)
    manager.capture(100)
    assert len(fit_calls) == 2
    reused_basis = load_projection_basis(basis_path)
    np.testing.assert_array_equal(reused_basis.pse_components, basis.pse_components)
    np.testing.assert_array_equal(reused_basis.shape_components, basis.shape_components)
    for epoch in (50, 75, 100):
        with np.load(tmp_path / f"epoch_{epoch:04d}.npz", allow_pickle=False) as data:
            assert data["pse_pc_values"].shape[-1] == 2
            assert data["aligned_shape_pc"].shape[-1] == 2

    order = selected.sample_index.copy()
    exists = manager.capture(25)
    assert exists.status == "EXISTS"
    np.testing.assert_array_equal(
        load_selected_samples(tmp_path / "selected_samples.npz").sample_index,
        order,
    )
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert '"schema_version": 2' in manifest
    assert '"snapshot_representation": "fixed_pca_projection"' in manifest
    assert '"class_names"' in manifest
    assert not list(tmp_path.glob("*.tmp"))


def test_snapshot_forward_is_read_only_and_restores_training_state(tmp_path: Path) -> None:
    source = TinySnapshotDataset([0, 0, 1, 1], parcel_offset=10, length=12)
    target = TinySnapshotDataset([0, 0, 1, 1], parcel_offset=20, length=13)
    model = _tiny_model().train()
    _force_valid_shape(model)
    first_parameter = next(model.parameters())
    first_parameter.grad = torch.full_like(first_parameter, 0.25)
    before_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before_grads = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before_optimizer = deepcopy(optimizer.state_dict())
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
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
    assert random.getstate() == python_rng
    after_numpy_rng = np.random.get_state()
    assert after_numpy_rng[0] == numpy_rng[0]
    np.testing.assert_array_equal(after_numpy_rng[1], numpy_rng[1])
    assert after_numpy_rng[2:] == numpy_rng[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)


def test_selection_is_sorted_and_uniform_without_random_resampling() -> None:
    labels = np.zeros(10, dtype=np.int64)
    parcels = np.asarray([90, 10, 80, 20, 70, 30, 60, 40, 50, 0])
    selected = deterministic_class_selection(labels, parcels, samples_per_class=4, seed=99)
    np.testing.assert_array_equal(parcels[selected], np.asarray([0, 30, 60, 90]))


def test_selection_keeps_at_most_eight_per_class() -> None:
    labels = np.repeat(np.arange(3), 20)
    parcels = np.arange(len(labels), dtype=np.int64)
    selected = deterministic_class_selection(labels, parcels, samples_per_class=8, seed=1)
    assert len(selected) == 24
    assert all(np.count_nonzero(labels[selected] == label) == 8 for label in range(3))


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


def test_existing_selection_recreates_missing_manifest(tmp_path: Path) -> None:
    source = TinySnapshotDataset([0, 1], parcel_offset=10, length=5)
    target = TinySnapshotDataset([0, 1], parcel_offset=20, length=6)
    model = _tiny_model()
    _force_valid_shape(model)
    manager = FeatureSnapshotManager(
        model, source, target,
        FeatureSnapshotConfig(25, 1, "float16", tmp_path, 1),
        device=torch.device("cpu"), batch_size=2,
        source_domain="source", target_domain="target",
    )
    manager.capture(25)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.unlink()

    result = manager.capture(25)

    assert result.status == "EXISTS"
    assert manifest_path.is_file()
    assert '"snapshot_representation": "fixed_pca_projection"' in manifest_path.read_text(
        encoding="utf-8"
    )


def test_existing_selection_rejects_incompatible_manifest(tmp_path: Path) -> None:
    source = TinySnapshotDataset([0, 1], parcel_offset=10, length=5)
    target = TinySnapshotDataset([0, 1], parcel_offset=20, length=6)
    model = _tiny_model()
    _force_valid_shape(model)
    manager = FeatureSnapshotManager(
        model, source, target,
        FeatureSnapshotConfig(25, 1, "float16", tmp_path, 1),
        device=torch.device("cpu"), batch_size=2,
        source_domain="source", target_domain="target",
    )
    manager.capture(25)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        manager.capture(25)


def test_projection_basis_rejects_nonfinite_statistics(tmp_path: Path) -> None:
    source = TinySnapshotDataset([0, 1], parcel_offset=10, length=5)
    target = TinySnapshotDataset([0, 1], parcel_offset=20, length=6)
    model = _tiny_model()
    _force_valid_shape(model)
    manager = FeatureSnapshotManager(
        model, source, target,
        FeatureSnapshotConfig(25, 1, "float16", tmp_path, 1),
        device=torch.device("cpu"), batch_size=2,
        source_domain="source", target_domain="target",
    )
    manager.capture(25)
    basis_path = tmp_path / "projection_basis.npz"
    with np.load(basis_path, allow_pickle=False) as data:
        fields = {name: data[name].copy() for name in data.files}
    fields["pse_explained_variance"][0] = np.nan
    np.savez_compressed(basis_path, **fields)

    with pytest.raises(ValueError, match="finite"):
        load_projection_basis(basis_path)


def test_existing_epoch_rejects_nonmonotonic_offsets(tmp_path: Path) -> None:
    source = TinySnapshotDataset([0, 1], parcel_offset=10, length=5)
    target = TinySnapshotDataset([0, 1], parcel_offset=20, length=6)
    model = _tiny_model()
    _force_valid_shape(model)
    manager = FeatureSnapshotManager(
        model, source, target,
        FeatureSnapshotConfig(25, 1, "float16", tmp_path, 1),
        device=torch.device("cpu"), batch_size=2,
        source_domain="source", target_domain="target",
    )
    manager.capture(25)
    epoch_path = tmp_path / "epoch_0025.npz"
    with np.load(epoch_path, allow_pickle=False) as data:
        fields = {name: data[name].copy() for name in data.files}
    fields["pse_offsets"][1] = fields["pse_offsets"][2] + 1
    np.savez_compressed(epoch_path, **fields)

    with pytest.raises(ValueError, match="offsets"):
        manager.capture(25)
