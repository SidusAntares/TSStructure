from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace
import weakref

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
from methods.structure_da import TSStructureModel


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


class _SnapshotDetails:
    pass


class RecordingSnapshotModel(torch.nn.Module):
    def __init__(self, *, oom_above_batch_size: int | None = None, error: Exception | None = None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.backbone = SimpleNamespace(feature_dim=8)
        self.temporal_features = SimpleNamespace(
            coordinates=SimpleNamespace(canonical_grid=torch.linspace(0, 1, 4))
        )
        self.oom_above_batch_size = oom_above_batch_size
        self.error = error
        self.batch_sizes: list[int] = []
        self.pixel_widths: list[int] = []
        self.previous_details: weakref.ReferenceType | None = None

    def forward(self, pixels, valid_pixels, positions, extra=None, *, return_geometry=True):
        if self.previous_details is not None:
            assert self.previous_details() is None, "previous batch details were retained"
        batch_size = int(pixels.shape[0])
        self.batch_sizes.append(batch_size)
        self.pixel_widths.append(int(pixels.shape[-1]))
        if self.error is not None:
            raise self.error
        if self.oom_above_batch_size is not None and batch_size > self.oom_above_batch_size:
            raise torch.cuda.OutOfMemoryError("synthetic snapshot OOM")
        time_mask = valid_pixels.bool().any(dim=-1)
        base = pixels.mean(dim=-1)
        tokens = torch.cat((base, base[..., :2], base[..., :3]), dim=-1)
        aligned = tokens[:, :4].clone()
        details = _SnapshotDetails()
        details.mask = time_mask
        details.latent = tokens
        geometry = SimpleNamespace(
            structure_srvf=aligned + 0.25,
            trend_srvf=aligned,
            structure_support=torch.ones(batch_size, 4),
            trend_support=torch.ones(batch_size, 4),
            canonical_grid=torch.linspace(0, 1, 4),
            structure_valid=torch.ones(batch_size, dtype=torch.bool),
            trend_valid=torch.ones(batch_size, dtype=torch.bool),
        )
        details.geometry = geometry
        self.previous_details = weakref.ref(details)
        return details


def _tiny_model() -> TSStructureModel:
    return TSStructureModel(
        num_classes=2,
        input_dim=3,
        mlp1=[3, 4],
        pooling="mean_std",
        mlp2=[8, 8],
        with_extra=False,
        canonical_grid_size=8,
        roughness_grid_size=64,
        trend_num_basis=5,
        structure_num_basis=5,
        n_head=2,
        d_k=2,
        d_model=8,
        ltae_mlp=(8, 4),
        classifier_hidden=(6,),
        dropout=0.0,
    )


def _force_valid_shape(model: TSStructureModel) -> None:
    original_forward_details = model.forward

    def forward_details(*args, **kwargs):
        output = original_forward_details(*args, **kwargs)
        geometry = output.geometry
        from methods.structure_da.representation import FunctionalGeometryOutput
        import dataclasses

        valid = torch.ones_like(geometry.trend_valid)
        replacement = FunctionalGeometryOutput(
            trend_srvf=geometry.trend_srvf,
            structure_srvf=geometry.structure_srvf,
            trend_support=geometry.trend_support,
            structure_support=geometry.structure_support,
            canonical_grid=geometry.canonical_grid,
            trend_valid=valid,
            structure_valid=valid,
        )
        return dataclasses.replace(output, geometry=replacement)

    model.forward = forward_details


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


def test_snapshot_manager_uses_dedicated_batch_size_not_eval_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dataset as dataset_module

    datasets = {
        "source": TinySnapshotDataset([0, 1], parcel_offset=0, length=5),
        "target": TinySnapshotDataset([0, 1], parcel_offset=10, length=5),
    }

    def fake_dataset(*, dataset_name, **kwargs):
        return datasets[dataset_name]

    monkeypatch.setattr(dataset_module, "PixelSetData", fake_dataset)
    config = SimpleNamespace(
        feature_snapshot_interval=25,
        feature_snapshot_samples_per_class=1,
        feature_snapshot_batch_size=8,
        feature_snapshot_dtype="float16",
        feature_snapshot_dir=tmp_path,
        seed=1,
        data_root=tmp_path,
        classes=["crop_0", "crop_1"],
        source="source",
        target="target",
        closed_set=True,
        combine_spring_and_winter=False,
        time_coordinate_mode="canonical_day_of_year",
        eval_batch_size=128,
        batch_size=128,
        amp=True,
        amp_dtype="float16",
    )
    splits = {"source": {"train": [0, 1]}, "target": {"train": [0, 1]}}

    manager = create_feature_snapshot_manager(
        RecordingSnapshotModel(), config, splits, device=torch.device("cpu")
    )

    assert manager is not None
    assert manager.batch_size == 8
    assert manager.amp_enabled
    assert manager.amp_dtype == "float16"


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
        pixel_selection_seed=3,
        num_pixels=4,
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

    def counted_fit(values: np.ndarray, num_components: int = 8, **kwargs):
        fit_calls.append(values.shape)
        return original_fit(values, num_components, **kwargs)

    monkeypatch.setattr(feature_snapshots, "fit_deterministic_pca", counted_fit)
    result = manager.capture(25)
    assert result.status == "SUCCESS"
    selected = load_selected_samples(tmp_path / "selected_samples.npz")
    assert selected.sample_index.dtype == np.int64
    assert selected.parcel_index.dtype == np.int64
    assert selected.domain.dtype == np.uint8
    assert selected.label.dtype == np.int16
    assert selected.pixel_indices.shape == (8, 4)
    assert selected.num_pixels == 4
    assert len(selected.sample_index) == 8

    basis_path = tmp_path / "projection_basis.npz"
    basis = load_projection_basis(basis_path)
    assert len(fit_calls) == 2
    assert fit_calls[0] == (104, 8)
    assert fit_calls[1] == (128, 8)
    assert basis.fitted_epoch == 25
    assert basis.pse_components.shape == (8, 8)
    assert basis.shape_components.shape == (8, 8)
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
            "unaligned_shape_pc", "aligned_shape_pc", "shape_grid", "shape_valid",
            "phase_status", "accepted_warp",
        }
        assert data["epoch"].item() == 25
        assert data["projection_fitted_epoch"].item() == 25
        assert data["schema_version"].item() == 3
        assert data["pse_pc_values"].dtype == np.float16
        assert data["aligned_shape_pc"].dtype == np.float16
        assert data["pse_times"].dtype == np.float32
        assert data["pse_offsets"].shape == (9,)
        assert data["pse_offsets"][-1] == 4 * 12 + 4 * 14
        assert data["pse_pc_values"].shape == (104, 8)
        assert data["unaligned_shape_pc"].shape == (8, 8, 8)
        assert data["aligned_shape_pc"].shape == (8, 8, 8)
        assert data["shape_grid"].shape == (8,)
        assert data["shape_valid"].shape == (8,)
        assert data["phase_status"].dtype == np.uint8
        assert set(np.unique(data["phase_status"])).issubset({0, 1, 2})
        assert data["accepted_warp"].shape == (8, 8)
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
            assert data["pse_pc_values"].shape[-1] == 8
            assert data["unaligned_shape_pc"].shape[-1] == 8
            assert data["aligned_shape_pc"].shape[-1] == 8

    order = selected.sample_index.copy()
    fixed_pixel_indices = selected.pixel_indices.copy()
    exists = manager.capture(25)
    assert exists.status == "EXISTS"
    np.testing.assert_array_equal(
        load_selected_samples(tmp_path / "selected_samples.npz").sample_index,
        order,
    )
    np.testing.assert_array_equal(
        load_selected_samples(tmp_path / "selected_samples.npz").pixel_indices,
        fixed_pixel_indices,
    )
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert '"schema_version": 3' in manifest
    assert '"snapshot_representation": "fixed_weighted_pca_projection"' in manifest
    assert '"class_names"' in manifest
    assert not list(tmp_path.glob("*.tmp"))


def test_projection_basis_recovers_at_next_epoch_after_initial_fit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = TinySnapshotDataset([0, 1], parcel_offset=10, length=5)
    target = TinySnapshotDataset([0, 1], parcel_offset=20, length=6)
    model = _tiny_model()
    _force_valid_shape(model)
    manager = FeatureSnapshotManager(
        model, source, target,
        FeatureSnapshotConfig(25, 1, "float16", tmp_path, 1, 4),
        device=torch.device("cpu"), batch_size=2,
        source_domain="source", target_domain="target",
    )
    original = feature_snapshots.fit_deterministic_pca
    calls = 0

    def fail_first(values, num_components=8, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic first-basis failure")
        return original(values, num_components, **kwargs)

    monkeypatch.setattr(feature_snapshots, "fit_deterministic_pca", fail_first)
    assert manager.capture(25).status == "FAILED"
    assert not (tmp_path / "epoch_0025.npz").exists()
    assert manager.capture(50).status == "SUCCESS"
    basis = load_projection_basis(tmp_path / "projection_basis.npz")
    assert basis.fitted_epoch == 50
    assert manager.capture(75).status == "SUCCESS"
    assert load_projection_basis(tmp_path / "projection_basis.npz").fitted_epoch == 50
    status = json.loads((tmp_path / "snapshot_status.json").read_text(encoding="utf-8"))
    assert status["epochs"]["25"]["status"] == "FAILED"
    assert status["epochs"]["50"]["status"] == "SUCCESS"


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
    result = manager.capture(25)
    assert result.status == "FAILED"
    assert "snapshot fields" in (result.error or "")


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
    assert '"snapshot_representation": "fixed_weighted_pca_projection"' in manifest_path.read_text(
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

    result = manager.capture(25)
    assert result.status == "FAILED"
    assert "manifest" in (result.error or "")


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

    result = manager.capture(25)
    assert result.status == "FAILED"
    assert "offsets" in (result.error or "")


def test_640_selected_samples_are_forwarded_in_bounded_cpu_batches(tmp_path: Path) -> None:
    source = TinySnapshotDataset([0] * 320, parcel_offset=0, length=5)
    target = TinySnapshotDataset([0] * 320, parcel_offset=1000, length=5)
    model = RecordingSnapshotModel()
    manager = FeatureSnapshotManager(
        model, source, target,
        FeatureSnapshotConfig(25, 320, "float16", tmp_path, 1),
        device=torch.device("cpu"), batch_size=8,
        source_domain="source", target_domain="target",
    )

    collected = manager._collect_features(
        np.arange(320), np.arange(320),
        np.tile(np.arange(64), (320, 1)) % 5,
        np.tile(np.arange(64), (320, 1)) % 5,
        batch_size=8,
    )
    values, times, unaligned, aligned, valid, status, warp, grid = collected

    assert len(values) == len(times) == len(unaligned) == len(aligned) == len(valid) == 640
    assert len(model.batch_sizes) == 80
    assert max(model.batch_sizes) == 8
    assert set(model.pixel_widths) == {64}
    assert all(isinstance(array, np.ndarray) for array in (*values, *times, *aligned, grid))
    assert not any(torch.is_tensor(array) for array in (*values, *times, *aligned, grid))


def test_cuda_oom_retries_8_to_4_to_2_to_1_and_then_writes_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = TinySnapshotDataset([0] * 8, parcel_offset=0, length=5)
    target = TinySnapshotDataset([0] * 8, parcel_offset=100, length=5)
    model = RecordingSnapshotModel(oom_above_batch_size=1)
    manager = FeatureSnapshotManager(
        model, source, target,
        FeatureSnapshotConfig(25, 8, "float16", tmp_path, 1),
        device=torch.device("cpu"), batch_size=8,
        source_domain="source", target_domain="target",
    )
    fit_devices: list[str] = []
    original_fit = feature_snapshots.fit_deterministic_pca

    def assert_cpu_fit(values: np.ndarray, num_components: int = 8, **kwargs):
        assert isinstance(values, np.ndarray)
        fit_devices.append("cpu")
        return original_fit(values, num_components, **kwargs)

    monkeypatch.setattr(feature_snapshots, "fit_deterministic_pca", assert_cpu_fit)
    result = manager.capture(25)

    assert result.status == "SUCCESS"
    assert model.batch_sizes[:4] == [8, 4, 2, 1]
    assert fit_devices == ["cpu", "cpu"]
    assert (tmp_path / "epoch_0025.npz").is_file()
    assert not (tmp_path / "SNAPSHOT_FAILED").exists()
    status = json.loads((tmp_path / "snapshot_status.json").read_text(encoding="utf-8"))
    assert status["has_failures"] is False
    output = capsys.readouterr().out
    for old, new in ((8, 4), (4, 2), (2, 1)):
        assert f"old_batch_size={old}|new_batch_size={new}|reason=CUDA_OOM" in output


def test_exhausted_cuda_oom_records_failure_without_partial_epoch(tmp_path: Path) -> None:
    source = TinySnapshotDataset([0] * 8, parcel_offset=0, length=5)
    target = TinySnapshotDataset([0] * 8, parcel_offset=100, length=5)
    model = RecordingSnapshotModel(oom_above_batch_size=0)
    manager = FeatureSnapshotManager(
        model, source, target,
        FeatureSnapshotConfig(25, 8, "float16", tmp_path, 1),
        device=torch.device("cpu"), batch_size=8,
        source_domain="source", target_domain="target",
    )

    result = manager.capture(25)

    assert result.status == "FAILED"
    assert "OutOfMemoryError" in (result.error or "")
    assert model.batch_sizes == [8, 4, 2, 1]
    status = json.loads((tmp_path / "snapshot_status.json").read_text(encoding="utf-8"))
    assert status["epochs"]["25"]["status"] == "FAILED"
    assert status["has_failures"] is True
    assert not (tmp_path / "SNAPSHOT_FAILED").exists()
    assert not (tmp_path / "epoch_0025.npz").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_non_oom_snapshot_error_is_not_retried(tmp_path: Path, capsys) -> None:
    source = TinySnapshotDataset([0] * 8, parcel_offset=0, length=5)
    target = TinySnapshotDataset([0] * 8, parcel_offset=100, length=5)
    model = RecordingSnapshotModel(error=RuntimeError("not an OOM"))
    manager = FeatureSnapshotManager(
        model, source, target,
        FeatureSnapshotConfig(25, 8, "float16", tmp_path, 1),
        device=torch.device("cpu"), batch_size=8,
        source_domain="source", target_domain="target",
    )

    result = manager.capture(25)

    assert result.status == "FAILED"
    assert "RuntimeError:not an OOM" in (result.error or "")
    assert model.batch_sizes == [8]
    assert "FEATURE_SNAPSHOT_RETRY" not in capsys.readouterr().out
