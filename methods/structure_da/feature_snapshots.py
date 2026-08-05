"""Compact, deterministic PSE and aligned-shape training snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


_SELECTED_FIELDS = {
    "sample_index",
    "domain",
    "label",
    "source_domain",
    "target_domain",
    "selection_seed",
    "samples_per_class",
}
_EPOCH_FIELDS = {
    "epoch",
    "pse_values",
    "pse_times",
    "pse_offsets",
    "aligned_shape",
    "shape_grid",
    "shape_valid",
}


@dataclass(frozen=True)
class FeatureSnapshotConfig:
    interval: int = 0
    samples_per_class: int = 32
    dtype: str = "float16"
    snapshot_dir: Path | None = None
    selection_seed: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.interval, bool) or self.interval < 0:
            raise ValueError("feature snapshot interval must be a non-negative integer")
        if isinstance(self.samples_per_class, bool) or self.samples_per_class <= 0:
            raise ValueError("feature snapshot samples per class must be positive")
        if self.dtype not in {"float16", "float32"}:
            raise ValueError("feature snapshot dtype must be float16 or float32")
        if self.snapshot_dir is not None:
            object.__setattr__(self, "snapshot_dir", Path(self.snapshot_dir))

    def should_capture(self, epoch: int) -> bool:
        return self.interval > 0 and epoch > 0 and epoch % self.interval == 0


@dataclass(frozen=True)
class SelectedSamples:
    sample_index: np.ndarray
    domain: np.ndarray
    label: np.ndarray
    source_domain: str
    target_domain: str
    selection_seed: int
    samples_per_class: int


@dataclass(frozen=True)
class SnapshotCaptureResult:
    status: str
    path: Path | None
    samples: int


def deterministic_class_selection(
    labels: np.ndarray,
    parcel_indices: np.ndarray,
    *,
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    """Return dataset positions sampled independently and reproducibly per class."""

    labels = np.asarray(labels)
    parcel_indices = np.asarray(parcel_indices)
    if labels.ndim != 1 or parcel_indices.shape != labels.shape:
        raise ValueError("labels and parcel_indices must be matching one-dimensional arrays")
    selected: list[np.ndarray] = []
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        positions = positions[np.argsort(parcel_indices[positions], kind="stable")]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(label)]))
        count = min(samples_per_class, len(positions))
        choice = rng.choice(len(positions), size=count, replace=False)
        selected.append(positions[np.sort(choice)])
    return np.concatenate(selected).astype(np.int64, copy=False) if selected else np.empty(0, np.int64)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_selected_samples(path: Path) -> SelectedSamples:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != _SELECTED_FIELDS:
            raise ValueError("selected sample fields are invalid")
        sample_index = data["sample_index"]
        domain = data["domain"]
        label = data["label"]
        if sample_index.dtype != np.int64 or domain.dtype != np.uint8 or label.dtype != np.int16:
            raise ValueError("selected sample dtypes are invalid")
        if not (sample_index.shape == domain.shape == label.shape) or sample_index.ndim != 1:
            raise ValueError("selected sample arrays must be matching vectors")
        if np.unique(np.stack([domain, sample_index], axis=1), axis=0).shape[0] != len(sample_index):
            raise ValueError("selected samples must be unique within each domain")
        return SelectedSamples(
            sample_index=sample_index.copy(),
            domain=domain.copy(),
            label=label.copy(),
            source_domain=str(data["source_domain"].item()),
            target_domain=str(data["target_domain"].item()),
            selection_seed=int(data["selection_seed"].item()),
            samples_per_class=int(data["samples_per_class"].item()),
        )


def _collate_snapshot_batch(
    samples: list[dict[str, Tensor]], device: torch.device
) -> dict[str, Tensor]:
    """Pad only the parcel-pixel axis; date and channel axes stay unchanged."""

    if not samples:
        raise ValueError("snapshot batch must be nonempty")
    first = samples[0]["pixels"]
    length, channels = first.shape[:2]
    max_pixels = max(sample["pixels"].shape[-1] for sample in samples)
    pixels = first.new_zeros(len(samples), length, channels, max_pixels)
    valid = samples[0]["valid_pixels"].new_zeros(len(samples), length, max_pixels)
    for index, sample in enumerate(samples):
        value = sample["pixels"]
        if value.shape[:2] != (length, channels):
            raise ValueError("snapshot batch samples must share date and channel axes")
        count = value.shape[-1]
        pixels[index, :, :, :count] = value
        valid[index, :, :count] = sample["valid_pixels"]
    batch = {
        "pixels": pixels.to(device),
        "valid_pixels": valid.to(device),
        "positions": torch.stack([sample["positions"] for sample in samples]).to(device),
    }
    if all("extra" in sample for sample in samples):
        batch["extra"] = torch.stack([sample["extra"] for sample in samples]).to(device)
    return batch


class FeatureSnapshotManager:
    """Capture selected training parcels without mutating model or optimizer state."""

    def __init__(
        self,
        model: nn.Module,
        source_dataset: Any,
        target_dataset: Any,
        config: FeatureSnapshotConfig,
        *,
        device: torch.device,
        batch_size: int,
        source_domain: str,
        target_domain: str,
    ) -> None:
        self.model = model
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.config = config
        self.device = device
        self.batch_size = int(batch_size)
        self.source_domain = source_domain
        self.target_domain = target_domain
        if self.batch_size <= 0:
            raise ValueError("feature snapshot batch size must be positive")

    @property
    def snapshot_dir(self) -> Path:
        if self.config.snapshot_dir is None:
            raise ValueError("feature snapshot directory is not configured")
        return self.config.snapshot_dir

    def _create_selection(self) -> tuple[SelectedSamples, np.ndarray, np.ndarray]:
        source_labels = np.asarray(self.source_dataset.get_labels())
        target_labels = np.asarray(self.target_dataset.get_labels())
        source_parcels = np.asarray(self.source_dataset.get_parcel_indices())
        target_parcels = np.asarray(self.target_dataset.get_parcel_indices())
        source_positions = deterministic_class_selection(
            source_labels,
            source_parcels,
            samples_per_class=self.config.samples_per_class,
            seed=self.config.selection_seed,
        )
        target_positions = deterministic_class_selection(
            target_labels,
            target_parcels,
            samples_per_class=self.config.samples_per_class,
            seed=self.config.selection_seed,
        )
        selected = SelectedSamples(
            sample_index=np.concatenate((source_parcels[source_positions], target_parcels[target_positions])).astype(np.int64),
            domain=np.concatenate((np.zeros(len(source_positions)), np.ones(len(target_positions)))).astype(np.uint8),
            label=np.concatenate((source_labels[source_positions], target_labels[target_positions])).astype(np.int16),
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            selection_seed=self.config.selection_seed,
            samples_per_class=self.config.samples_per_class,
        )
        _atomic_npz(
            self.snapshot_dir / "selected_samples.npz",
            sample_index=selected.sample_index,
            domain=selected.domain,
            label=selected.label,
            source_domain=np.asarray(selected.source_domain),
            target_domain=np.asarray(selected.target_domain),
            selection_seed=np.asarray(selected.selection_seed, dtype=np.int64),
            samples_per_class=np.asarray(selected.samples_per_class, dtype=np.int64),
        )
        _atomic_json(
            self.snapshot_dir / "manifest.json",
            {
                "aligned_shape_semantics": "aligned_structure_srvf",
                "dtype": self.config.dtype,
                "pse_storage": "ragged",
                "source_domain": self.source_domain,
                "target_domain": self.target_domain,
                "target_label_usage": "diagnostic sample stratification and visualization only; never used by training or checkpoint selection",
            },
        )
        return selected, source_positions, target_positions

    def _load_or_create_selection(self) -> tuple[SelectedSamples, np.ndarray, np.ndarray]:
        selected_path = self.snapshot_dir / "selected_samples.npz"
        if not selected_path.exists():
            return self._create_selection()
        selected = load_selected_samples(selected_path)
        if (
            selected.source_domain != self.source_domain
            or selected.target_domain != self.target_domain
            or selected.selection_seed != self.config.selection_seed
            or selected.samples_per_class != self.config.samples_per_class
        ):
            raise ValueError("existing selected samples do not match snapshot configuration")
        positions: list[np.ndarray] = []
        for domain, dataset in ((0, self.source_dataset), (1, self.target_dataset)):
            lookup = {int(parcel): index for index, parcel in enumerate(dataset.get_parcel_indices())}
            try:
                positions.append(np.asarray([lookup[int(parcel)] for parcel in selected.sample_index[selected.domain == domain]], dtype=np.int64))
            except KeyError as error:
                raise ValueError("selected sample is absent from the current dataset") from error
        return selected, positions[0], positions[1]

    @staticmethod
    def _validate_epoch(path: Path, expected_epoch: int, expected_samples: int) -> None:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != _EPOCH_FIELDS:
                raise ValueError("snapshot fields are invalid")
            if int(data["epoch"].item()) != expected_epoch:
                raise ValueError("snapshot epoch is invalid")
            offsets = data["pse_offsets"]
            aligned = data["aligned_shape"]
            valid = data["shape_valid"]
            grid = data["shape_grid"]
            values = data["pse_values"]
            times = data["pse_times"]
            if offsets.shape != (expected_samples + 1,) or offsets[0] != 0 or offsets[-1] != len(values):
                raise ValueError("snapshot ragged offsets are invalid")
            if len(times) != len(values) or aligned.shape[0] != expected_samples or valid.shape != (expected_samples,):
                raise ValueError("snapshot sample dimensions are invalid")
            if aligned.ndim != 3 or grid.shape != (aligned.shape[1],):
                raise ValueError("snapshot aligned-shape dimensions are invalid")

    def capture(self, epoch: int) -> SnapshotCaptureResult:
        if not self.config.should_capture(epoch):
            return SnapshotCaptureResult("DISABLED", None, 0)
        selected, source_positions, target_positions = self._load_or_create_selection()
        epoch_path = self.snapshot_dir / f"epoch_{epoch:04d}.npz"
        if epoch_path.exists():
            self._validate_epoch(epoch_path, epoch, len(selected.sample_index))
            print(f"FEATURE_SNAPSHOT|epoch={epoch}|status=EXISTS|path={epoch_path}", flush=True)
            return SnapshotCaptureResult("EXISTS", epoch_path, len(selected.sample_index))

        output_dtype = np.float16 if self.config.dtype == "float16" else np.float32
        values: list[np.ndarray] = []
        times: list[np.ndarray] = []
        aligned: list[np.ndarray] = []
        valid: list[bool] = []
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.inference_mode():
                for dataset, positions in (
                    (self.source_dataset, source_positions),
                    (self.target_dataset, target_positions),
                ):
                    for start in range(0, len(positions), self.batch_size):
                        batch_positions = positions[start:start + self.batch_size]
                        batch = _collate_snapshot_batch(
                            [dataset[int(position)] for position in batch_positions],
                            self.device,
                        )
                        details = self.model.forward_details(
                            batch["pixels"],
                            batch["valid_pixels"],
                            batch["positions"],
                            batch.get("extra"),
                        )
                        for sample_index in range(len(batch_positions)):
                            mask = details.backbone.time_mask[sample_index]
                            values.append(details.backbone.tokens[sample_index, mask].float().cpu().numpy().astype(output_dtype))
                            times.append(batch["positions"][sample_index, mask].float().cpu().numpy().astype(np.float32))
                            shape_valid = bool(details.temporal.shape.valid[sample_index].item())
                            shape = details.temporal.aligned_structure_srvf[sample_index].float().cpu().numpy()
                            aligned.append((shape if shape_valid else np.zeros_like(shape)).astype(output_dtype))
                            valid.append(shape_valid)
            offsets = np.concatenate(([0], np.cumsum([len(value) for value in values]))).astype(np.int64)
            shape_grid = self.model.temporal_features.coordinates.canonical_grid.detach().float().cpu().numpy().astype(np.float32)
            _atomic_npz(
                epoch_path,
                epoch=np.asarray(epoch, dtype=np.int64),
                pse_values=np.concatenate(values, axis=0),
                pse_times=np.concatenate(times, axis=0),
                pse_offsets=offsets,
                aligned_shape=np.stack(aligned),
                shape_grid=shape_grid,
                shape_valid=np.asarray(valid, dtype=np.bool_),
            )
            self._validate_epoch(epoch_path, epoch, len(selected.sample_index))
        except Exception as error:
            print(f"FEATURE_SNAPSHOT|epoch={epoch}|status=FAILED|error={type(error).__name__}:{error}", flush=True)
            raise
        finally:
            self.model.train(was_training)
        print(f"FEATURE_SNAPSHOT|epoch={epoch}|status=SUCCESS|samples={len(selected.sample_index)}|path={epoch_path}", flush=True)
        return SnapshotCaptureResult("SUCCESS", epoch_path, len(selected.sample_index))


def create_feature_snapshot_manager(
    model: nn.Module,
    config: Any,
    splits: dict[str, Any],
    *,
    device: torch.device,
) -> FeatureSnapshotManager | None:
    """Build deterministic train-split datasets only when snapshots are enabled."""

    snapshot_config = FeatureSnapshotConfig(
        interval=int(getattr(config, "feature_snapshot_interval", 0)),
        samples_per_class=int(
            getattr(config, "feature_snapshot_samples_per_class", 32)
        ),
        dtype=str(getattr(config, "feature_snapshot_dtype", "float16")),
        snapshot_dir=getattr(config, "feature_snapshot_dir", None),
        selection_seed=int(getattr(config, "seed", 1)),
    )
    if snapshot_config.interval == 0:
        return None
    if snapshot_config.snapshot_dir is None:
        raise ValueError("--feature_snapshot_dir is required when snapshots are enabled")

    from dataset import PixelSetData
    from torchvision.transforms import transforms
    from transforms import Normalize, ToTensor

    deterministic_transform = transforms.Compose([Normalize(), ToTensor()])
    common = dict(
        data_root=config.data_root,
        classes=config.classes,
        transform=deterministic_transform,
        with_extra=False,
        closed_set=config.closed_set,
        combine_spring_and_winter=config.combine_spring_and_winter,
        time_coordinate_mode=getattr(
            config, "time_coordinate_mode", "canonical_day_of_year"
        ),
    )
    source_dataset = PixelSetData(
        dataset_name=config.source,
        indices=splits[config.source]["train"],
        **common,
    )
    target_dataset = PixelSetData(
        dataset_name=config.target,
        indices=splits[config.target]["train"],
        **common,
    )
    return FeatureSnapshotManager(
        model,
        source_dataset,
        target_dataset,
        snapshot_config,
        device=device,
        batch_size=getattr(config, "eval_batch_size", None) or config.batch_size,
        source_domain=config.source,
        target_domain=config.target,
    )
