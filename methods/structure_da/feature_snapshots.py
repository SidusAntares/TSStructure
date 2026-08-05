"""Compact, deterministic PSE and aligned-shape training snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


SNAPSHOT_SCHEMA_VERSION = 3
NUM_PROJECTION_COMPONENTS = 8
_LEGACY_SELECTED_FIELDS = {
    "sample_index",
    "domain",
    "label",
    "source_domain",
    "target_domain",
    "selection_seed",
    "samples_per_class",
}
_SELECTED_FIELDS = {
    "sample_index",
    "parcel_index",
    "domain",
    "label",
    "pixel_indices",
    "source_domain",
    "target_domain",
    "samples_per_class",
    "num_pixels",
    "pixel_selection_seed",
    "parcel_selection_policy",
    "pixel_selection_policy",
}
_PROJECTION_FIELDS = {
    "schema_version",
    "fitted_epoch",
    "num_components",
    "pse_mean",
    "pse_components",
    "pse_explained_variance",
    "pse_explained_variance_ratio",
    "shape_mean",
    "shape_components",
    "shape_explained_variance",
    "shape_explained_variance_ratio",
}
_EPOCH_FIELDS = {
    "epoch",
    "projection_fitted_epoch",
    "schema_version",
    "pse_pc_values",
    "pse_times",
    "pse_offsets",
    "unaligned_shape_pc", "aligned_shape_pc",
    "shape_grid",
    "shape_valid",
    "phase_status", "accepted_warp",
}


@dataclass(frozen=True)
class FeatureSnapshotConfig:
    interval: int = 0
    samples_per_class: int = 32
    dtype: str = "float16"
    snapshot_dir: Path | None = None
    pixel_selection_seed: int = 1
    num_pixels: int = 64

    def __post_init__(self) -> None:
        if isinstance(self.interval, bool) or self.interval < 0:
            raise ValueError("feature snapshot interval must be a non-negative integer")
        if isinstance(self.samples_per_class, bool) or self.samples_per_class <= 0:
            raise ValueError("feature snapshot samples per class must be positive")
        if self.dtype not in {"float16", "float32"}:
            raise ValueError("feature snapshot dtype must be float16 or float32")
        if isinstance(self.num_pixels, bool) or self.num_pixels <= 0:
            raise ValueError("feature snapshot num_pixels must be positive")
        if self.snapshot_dir is not None:
            object.__setattr__(self, "snapshot_dir", Path(self.snapshot_dir))

    def should_capture(self, epoch: int) -> bool:
        return self.interval > 0 and epoch > 0 and epoch % self.interval == 0


@dataclass(frozen=True)
class SelectedSamples:
    sample_index: np.ndarray
    parcel_index: np.ndarray
    domain: np.ndarray
    label: np.ndarray
    pixel_indices: np.ndarray
    source_domain: str
    target_domain: str
    samples_per_class: int
    num_pixels: int
    pixel_selection_seed: int
    parcel_selection_policy: str
    pixel_selection_policy: str


@dataclass(frozen=True)
class PCAFit:
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray


@dataclass(frozen=True)
class ProjectionBasis:
    schema_version: int
    fitted_epoch: int
    num_components: int
    pse_mean: np.ndarray
    pse_components: np.ndarray
    pse_explained_variance: np.ndarray
    pse_explained_variance_ratio: np.ndarray
    shape_mean: np.ndarray
    shape_components: np.ndarray
    shape_explained_variance: np.ndarray
    shape_explained_variance_ratio: np.ndarray


@dataclass(frozen=True)
class SnapshotCaptureResult:
    status: str
    path: Path | None
    samples: int
    error: str | None = None


def deterministic_class_selection(
    labels: np.ndarray,
    parcel_indices: np.ndarray,
    *,
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    """Select uniformly spaced parcels after stable parcel-index sorting."""

    del seed  # Retained in the public interface and metadata for schema stability.
    labels = np.asarray(labels)
    parcel_indices = np.asarray(parcel_indices)
    if labels.ndim != 1 or parcel_indices.shape != labels.shape:
        raise ValueError("labels and parcel_indices must be matching one-dimensional arrays")
    selected: list[np.ndarray] = []
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        positions = positions[np.argsort(parcel_indices[positions], kind="stable")]
        count = min(samples_per_class, len(positions))
        uniform = np.linspace(0, len(positions) - 1, num=count).round().astype(np.int64)
        selected.append(positions[uniform])
    return (
        np.concatenate(selected).astype(np.int64, copy=False)
        if selected
        else np.empty(0, np.int64)
    )


def _stable_pixel_seed(pixel_seed: int, domain: str, parcel_index: int) -> int:
    payload = f"{int(pixel_seed)}\0{domain}\0{int(parcel_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def deterministic_pixel_indices(
    available_pixels: int,
    num_pixels: int,
    *,
    pixel_seed: int,
    domain: str,
    parcel_index: int,
) -> np.ndarray:
    """Return stable indices with exactly RandomSamplePixels' sampling semantics."""

    if available_pixels <= 0:
        raise ValueError("snapshot parcels must contain at least one pixel")
    if num_pixels <= 0:
        raise ValueError("num_pixels must be positive")
    if available_pixels > num_pixels:
        generator = random.Random(
            _stable_pixel_seed(pixel_seed, domain, parcel_index)
        )
        return np.asarray(
            generator.sample(range(available_pixels), num_pixels), dtype=np.int64
        )
    if available_pixels < num_pixels:
        return np.asarray(
            list(range(available_pixels)) + [0] * (num_pixels - available_pixels),
            dtype=np.int64,
        )
    return np.arange(num_pixels, dtype=np.int64)


def prepare_snapshot_sample(
    sample: dict[str, Any], pixel_indices: np.ndarray, *, num_pixels: int
) -> dict[str, Tensor]:
    """Apply saved pixel indices, then the same Normalize/ToTensor tail as training."""

    from transforms import Normalize

    pixels = sample["pixels"]
    if torch.is_tensor(pixels):
        pixels = pixels.detach().cpu().numpy()
    pixels = np.asarray(pixels)
    indices = np.asarray(pixel_indices, dtype=np.int64)
    if indices.shape != (num_pixels,):
        raise ValueError("saved pixel indices must have shape [num_pixels]")
    if pixels.ndim != 3 or pixels.shape[-1] <= 0:
        raise ValueError("snapshot pixels must have shape [T, C, S] with S > 0")
    if np.any(indices < 0) or np.any(indices >= pixels.shape[-1]):
        raise ValueError("saved pixel index is outside the parcel pixel axis")
    available = pixels.shape[-1]
    valid_count = min(available, num_pixels)
    valid_pixels = np.asarray(
        [1.0] * valid_count + [0.0] * (num_pixels - valid_count),
        dtype=np.float32,
    )
    prepared = dict(sample)
    prepared["pixels"] = pixels[..., indices].copy()
    prepared["valid_pixels"] = np.repeat(
        valid_pixels[None, :], pixels.shape[0], axis=0
    )
    for key in ("positions", "extra"):
        if key in prepared and torch.is_tensor(prepared[key]):
            prepared[key] = prepared[key].detach().cpu().numpy()
    if torch.is_tensor(prepared.get("label")):
        prepared["label"] = int(prepared["label"].item())
    prepared = Normalize()(prepared)
    prepared["pixels"] = torch.from_numpy(prepared["pixels"].astype(np.float32))
    prepared["valid_pixels"] = torch.from_numpy(
        prepared["valid_pixels"].astype(np.float32)
    )
    prepared["positions"] = torch.from_numpy(
        np.asarray(prepared["positions"]).astype(np.int64)
    )
    if "extra" in prepared:
        prepared["extra"] = torch.from_numpy(
            np.asarray(prepared["extra"]).astype(np.float32)
        )
    if isinstance(prepared.get("label"), int):
        prepared["label"] = torch.tensor(prepared["label"]).long()
    return prepared


def fit_deterministic_pca(
    values: np.ndarray,
    num_components: int = NUM_PROJECTION_COMPONENTS,
    *,
    weights: np.ndarray | None = None,
) -> PCAFit:
    """Fit PCA with deterministic eigenvalue ordering and component signs."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA requires at least two feature rows")
    if values.shape[1] < num_components:
        raise ValueError("feature dimension is smaller than requested PCA components")
    if not np.isfinite(values).all():
        raise ValueError("PCA input must be finite")
    if weights is None:
        weights = np.ones(values.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (values.shape[0],):
            raise ValueError("PCA weights must match the number of feature rows")
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("PCA weights must be finite and positive")
    weight_sum = float(weights.sum())
    mean = np.sum(values * weights[:, None], axis=0) / weight_sum
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered / weight_sum
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    explained = np.maximum(eigenvalues[order[:num_components]], 0.0)
    components = eigenvectors[:, order[:num_components]].T
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= -1.0
    total = np.maximum(eigenvalues, 0.0).sum()
    ratio = explained / total if total > 0 else np.zeros_like(explained)
    return PCAFit(
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        explained_variance=explained.astype(np.float32),
        explained_variance_ratio=ratio.astype(np.float32),
    )


def project_features(values: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != mean.shape[0]:
        raise ValueError("projection input feature dimension is invalid")
    return (values - mean) @ components.T


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def update_snapshot_status(
    snapshot_dir: Path,
    *,
    epoch: int | None = None,
    epoch_status: dict[str, Any] | None = None,
    projection_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically merge one epoch/projection result and recompute failures."""

    path = Path(snapshot_dir) / "snapshot_status.json"
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing snapshot status is unreadable") from error
        if value.get("schema_version") != 1 or not isinstance(value.get("epochs"), dict):
            raise ValueError("existing snapshot status schema is invalid")
    else:
        value = {
            "schema_version": 1,
            "projection_basis": {"status": "MISSING", "fitted_epoch": None},
            "epochs": {},
        }
    if projection_status is not None:
        value["projection_basis"] = dict(projection_status)
    if epoch is not None:
        if epoch_status is None:
            raise ValueError("epoch_status is required when epoch is provided")
        value["epochs"][str(int(epoch))] = dict(epoch_status)
    value["has_failures"] = any(
        item.get("status") == "FAILED" for item in value["epochs"].values()
    )
    value["last_updated"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(path, value)
    return value


def normalize_accepted_warp(
    phase_status: np.ndarray, accepted_warp: np.ndarray, shape_grid: np.ndarray
) -> np.ndarray:
    status = np.asarray(phase_status, dtype=np.uint8)
    warp = np.asarray(accepted_warp, dtype=np.float32).copy()
    grid = np.asarray(shape_grid, dtype=np.float32)
    if status.ndim != 1 or warp.shape != (len(status), len(grid)):
        raise ValueError("phase status and accepted warp dimensions are invalid")
    if np.any(status > 2):
        raise ValueError("phase status must use encoding 0, 1, or 2")
    warp[status != 2] = grid
    if not np.isfinite(warp).all():
        raise ValueError("accepted warp must be finite")
    return warp


def load_selected_samples(path: Path) -> SelectedSamples:
    with np.load(path, allow_pickle=False) as data:
        fields = set(data.files)
        if fields not in (_SELECTED_FIELDS, _LEGACY_SELECTED_FIELDS):
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
        if fields == _SELECTED_FIELDS:
            parcel_index = data["parcel_index"]
            pixel_indices = data["pixel_indices"]
            num_pixels = int(data["num_pixels"].item())
            if (
                parcel_index.dtype != np.int64
                or parcel_index.shape != sample_index.shape
                or pixel_indices.dtype != np.int64
                or pixel_indices.shape != (len(sample_index), num_pixels)
            ):
                raise ValueError("selected parcel or pixel indices are invalid")
            pixel_selection_seed = int(data["pixel_selection_seed"].item())
            parcel_policy = str(data["parcel_selection_policy"].item())
            pixel_policy = str(data["pixel_selection_policy"].item())
        else:
            parcel_index = sample_index.copy()
            pixel_indices = np.empty((len(sample_index), 0), dtype=np.int64)
            num_pixels = 0
            pixel_selection_seed = int(data["selection_seed"].item())
            parcel_policy = "legacy"
            pixel_policy = "legacy_unspecified"
        return SelectedSamples(
            sample_index=sample_index.copy(),
            parcel_index=parcel_index.copy(),
            domain=domain.copy(),
            label=label.copy(),
            pixel_indices=pixel_indices.copy(),
            source_domain=str(data["source_domain"].item()),
            target_domain=str(data["target_domain"].item()),
            samples_per_class=int(data["samples_per_class"].item()),
            num_pixels=num_pixels,
            pixel_selection_seed=pixel_selection_seed,
            parcel_selection_policy=parcel_policy,
            pixel_selection_policy=pixel_policy,
        )


def load_projection_basis(path: Path) -> ProjectionBasis:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != _PROJECTION_FIELDS:
            raise ValueError("projection basis fields are invalid")
        schema_version = int(data["schema_version"].item())
        fitted_epoch = int(data["fitted_epoch"].item())
        num_components = int(data["num_components"].item())
        if (
            (schema_version, num_components)
            not in {(2, 2), (SNAPSHOT_SCHEMA_VERSION, NUM_PROJECTION_COMPONENTS)}
            or fitted_epoch <= 0
        ):
            raise ValueError("projection basis schema is unsupported")
        arrays = {name: data[name].copy() for name in _PROJECTION_FIELDS if name not in {
            "schema_version", "fitted_epoch", "num_components"
        }}
    for prefix in ("pse", "shape"):
        mean = arrays[f"{prefix}_mean"]
        components = arrays[f"{prefix}_components"]
        variance = arrays[f"{prefix}_explained_variance"]
        ratio = arrays[f"{prefix}_explained_variance_ratio"]
        if mean.dtype != np.float32 or mean.ndim != 1:
            raise ValueError("projection mean is invalid")
        if components.dtype != np.float32 or components.shape != (num_components, len(mean)):
            raise ValueError("projection components are invalid")
        if (
            variance.dtype != np.float32
            or ratio.dtype != np.float32
            or variance.shape != (num_components,)
            or ratio.shape != (num_components,)
        ):
            raise ValueError("projection variance fields are invalid")
        if not all(np.all(np.isfinite(value)) for value in (mean, components, variance, ratio)):
            raise ValueError("projection basis values must be finite")
        if np.any(variance < 0) or np.any(ratio < 0) or np.any(ratio > 1):
            raise ValueError("projection variance fields are outside their valid range")
        if float(ratio.sum()) > 1.0 + 1e-5:
            raise ValueError("projection explained variance ratios exceed one")
        if not np.allclose(
            components @ components.T,
            np.eye(num_components, dtype=np.float32),
            rtol=1e-4,
            atol=1e-4,
        ):
            raise ValueError("projection components must be orthonormal")
    return ProjectionBasis(
        schema_version=schema_version,
        fitted_epoch=fitted_epoch,
        num_components=num_components,
        **arrays,
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
    """Capture fixed parcels and store fixed-basis PC1-PC8 diagnostics."""

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
        amp_enabled: bool = False,
        amp_dtype: str = "float16",
    ) -> None:
        self.model = model
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.config = config
        self.device = device
        self.batch_size = int(batch_size)
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.amp_enabled = bool(amp_enabled)
        self.amp_dtype = str(amp_dtype)
        if self.batch_size <= 0:
            raise ValueError("feature snapshot batch size must be positive")
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("feature snapshot AMP dtype must be float16 or bfloat16")

    @property
    def snapshot_dir(self) -> Path:
        if self.config.snapshot_dir is None:
            raise ValueError("feature snapshot directory is not configured")
        return self.config.snapshot_dir

    def _manifest(self, selected: SelectedSamples) -> dict[str, Any]:
        source_classes = getattr(self.source_dataset, "classes", None)
        target_classes = getattr(self.target_dataset, "classes", None)
        class_names = (
            [str(value) for value in source_classes]
            if (
                source_classes is not None
                and target_classes is not None
                and list(source_classes) == list(target_classes)
            )
            else None
        )
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_representation": "fixed_weighted_pca_projection",
            "projection_fitted_epoch": None,
            "num_projection_components": NUM_PROJECTION_COMPONENTS,
            "parcel_selection_policy": "stable_parcel_uniform",
            "pixel_selection_policy": "stable_per_parcel_training_equivalent",
            "pixel_selection_seed": self.config.pixel_selection_seed,
            "num_pixels": self.config.num_pixels,
            "pixel_indices_saved": True,
            "pse_saved_semantics": "pixel_set_encoder_tokens_projected_to_fixed_pc1_pc8",
            "unaligned_shape_saved_semantics": "structure_srvf_before_accepted_phase_projected_to_fixed_pc1_pc8",
            "aligned_shape_saved_semantics": "structure_srvf_after_accepted_phase_projected_to_fixed_pc1_pc8",
            "shape_projection_fit_scope": "joint_unaligned_aligned_source_target_first_successful_snapshot",
            "phase_status_encoding": {
                "0": "failure",
                "1": "valid_identity",
                "2": "valid_nonidentity",
            },
            "raw_feature_dimension": int(self.model.backbone.feature_dim),
            "raw_full_features_saved": False,
            "pse_storage": "ragged",
            "dtype": self.config.dtype,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "class_ids": [int(value) for value in np.unique(selected.label)],
            "class_names": class_names,
            "target_label_usage": "diagnostic_snapshot_selection_and_offline_visualization_only",
        }

    def _update_manifest_fitted_epoch(self, fitted_epoch: int) -> None:
        path = self.snapshot_dir / "manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["projection_fitted_epoch"] = int(fitted_epoch)
        _atomic_json(path, value)

    def _create_selection(self) -> tuple[SelectedSamples, np.ndarray, np.ndarray]:
        source_labels = np.asarray(self.source_dataset.get_labels())
        target_labels = np.asarray(self.target_dataset.get_labels())
        source_parcels = np.asarray(self.source_dataset.get_parcel_indices())
        target_parcels = np.asarray(self.target_dataset.get_parcel_indices())
        source_positions = deterministic_class_selection(
            source_labels, source_parcels,
            samples_per_class=self.config.samples_per_class,
            seed=0,
        )
        target_positions = deterministic_class_selection(
            target_labels, target_parcels,
            samples_per_class=self.config.samples_per_class,
            seed=0,
        )
        sample_positions = np.concatenate((source_positions, target_positions)).astype(np.int64)
        parcel_indices = np.concatenate((
            source_parcels[source_positions], target_parcels[target_positions]
        )).astype(np.int64)
        domains = np.concatenate((
            np.zeros(len(source_positions)), np.ones(len(target_positions))
        )).astype(np.uint8)
        pixel_indices: list[np.ndarray] = []
        for domain_id, domain_name, dataset, positions in (
            (0, self.source_domain, self.source_dataset, source_positions),
            (1, self.target_domain, self.target_dataset, target_positions),
        ):
            del domain_id
            for position in positions:
                sample = dataset[int(position)]
                pixels = sample["pixels"]
                available = int(pixels.shape[-1])
                pixel_indices.append(deterministic_pixel_indices(
                    available,
                    self.config.num_pixels,
                    pixel_seed=self.config.pixel_selection_seed,
                    domain=domain_name,
                    parcel_index=int(dataset.get_parcel_indices()[int(position)]),
                ))
        selected = SelectedSamples(
            sample_index=sample_positions,
            parcel_index=parcel_indices,
            domain=domains,
            label=np.concatenate((source_labels[source_positions], target_labels[target_positions])).astype(np.int16),
            pixel_indices=np.stack(pixel_indices).astype(np.int64),
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            samples_per_class=self.config.samples_per_class,
            num_pixels=self.config.num_pixels,
            pixel_selection_seed=self.config.pixel_selection_seed,
            parcel_selection_policy="stable_parcel_uniform",
            pixel_selection_policy="stable_per_parcel_training_equivalent",
        )
        _atomic_npz(
            self.snapshot_dir / "selected_samples.npz",
            sample_index=selected.sample_index,
            parcel_index=selected.parcel_index,
            domain=selected.domain,
            label=selected.label,
            pixel_indices=selected.pixel_indices,
            source_domain=np.asarray(selected.source_domain),
            target_domain=np.asarray(selected.target_domain),
            samples_per_class=np.asarray(selected.samples_per_class, dtype=np.int64),
            num_pixels=np.asarray(selected.num_pixels, dtype=np.int64),
            pixel_selection_seed=np.asarray(selected.pixel_selection_seed, dtype=np.int64),
            parcel_selection_policy=np.asarray(selected.parcel_selection_policy),
            pixel_selection_policy=np.asarray(selected.pixel_selection_policy),
        )
        _atomic_json(self.snapshot_dir / "manifest.json", self._manifest(selected))
        return selected, source_positions, target_positions

    def _load_or_create_selection(self) -> tuple[SelectedSamples, np.ndarray, np.ndarray]:
        selected_path = self.snapshot_dir / "selected_samples.npz"
        if not selected_path.exists():
            return self._create_selection()
        selected = load_selected_samples(selected_path)
        if (
            selected.source_domain != self.source_domain
            or selected.target_domain != self.target_domain
            or selected.samples_per_class != self.config.samples_per_class
            or selected.num_pixels != self.config.num_pixels
            or selected.pixel_selection_seed != self.config.pixel_selection_seed
            or selected.pixel_selection_policy
            != "stable_per_parcel_training_equivalent"
        ):
            raise ValueError("existing selected samples do not match snapshot configuration")
        manifest_path = self.snapshot_dir / "manifest.json"
        expected_manifest = self._manifest(selected)
        basis_path = self.snapshot_dir / "projection_basis.npz"
        if basis_path.exists():
            expected_manifest["projection_fitted_epoch"] = load_projection_basis(
                basis_path
            ).fitted_epoch
        if not manifest_path.exists():
            _atomic_json(manifest_path, expected_manifest)
        else:
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("existing snapshot manifest is unreadable") from error
            if existing_manifest != expected_manifest:
                raise ValueError("existing snapshot manifest does not match selected samples")
        positions: list[np.ndarray] = []
        for domain, dataset in ((0, self.source_dataset), (1, self.target_dataset)):
            lookup = {int(parcel): index for index, parcel in enumerate(dataset.get_parcel_indices())}
            try:
                positions.append(np.asarray([
                    lookup[int(parcel)] for parcel in selected.parcel_index[selected.domain == domain]
                ], dtype=np.int64))
            except KeyError as error:
                raise ValueError("selected sample is absent from the current dataset") from error
        return selected, positions[0], positions[1]

    def _validate_epoch(self, path: Path, expected_epoch: int, expected_samples: int) -> None:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != _EPOCH_FIELDS:
                raise ValueError("snapshot fields are invalid")
            if int(data["epoch"].item()) != expected_epoch:
                raise ValueError("snapshot epoch is invalid")
            if int(data["schema_version"].item()) != SNAPSHOT_SCHEMA_VERSION:
                raise ValueError("snapshot schema version is invalid")
            basis = load_projection_basis(self.snapshot_dir / "projection_basis.npz")
            if int(data["projection_fitted_epoch"].item()) != basis.fitted_epoch:
                raise ValueError("snapshot projection fitted epoch is invalid")
            offsets = data["pse_offsets"]
            aligned = data["aligned_shape_pc"]
            unaligned = data["unaligned_shape_pc"]
            valid = data["shape_valid"]
            status = data["phase_status"]
            warp = data["accepted_warp"]
            grid = data["shape_grid"]
            values = data["pse_pc_values"]
            times = data["pse_times"]
            if (
                offsets.dtype != np.int64
                or offsets.shape != (expected_samples + 1,)
                or offsets[0] != 0
                or offsets[-1] != len(values)
                or np.any(np.diff(offsets) < 0)
            ):
                raise ValueError("snapshot ragged offsets are invalid")
            if (
                len(times) != len(values)
                or aligned.shape[0] != expected_samples
                or unaligned.shape != aligned.shape
                or valid.shape != (expected_samples,)
                or status.shape != (expected_samples,)
                or warp.shape != (expected_samples, len(grid))
            ):
                raise ValueError("snapshot sample dimensions are invalid")
            if values.ndim != 2 or values.shape[1] != NUM_PROJECTION_COMPONENTS:
                raise ValueError("snapshot PSE projection dimensions are invalid")
            if aligned.ndim != 3 or aligned.shape[-1] != NUM_PROJECTION_COMPONENTS or grid.shape != (aligned.shape[1],):
                raise ValueError("snapshot aligned-shape dimensions are invalid")
            output_dtype = np.dtype(self.config.dtype)
            if (
                values.dtype != output_dtype
                or aligned.dtype != output_dtype
                or unaligned.dtype != output_dtype
                or warp.dtype != output_dtype
            ):
                raise ValueError("snapshot projection dtype is invalid")
            if (
                times.dtype != np.float32
                or grid.dtype != np.float32
                or valid.dtype != np.bool_
                or status.dtype != np.uint8
                or np.any(status > 2)
            ):
                raise ValueError("snapshot metadata dtypes are invalid")
            if not all(np.all(np.isfinite(value)) for value in (
                values, times, unaligned, aligned, grid, warp
            )):
                raise ValueError("snapshot arrays must be finite")
            if not np.allclose(warp[status != 2], grid, rtol=0, atol=5e-4):
                raise ValueError("identity/failure accepted warps must equal shape_grid")

    def _collect_features(
        self,
        source_positions: np.ndarray,
        target_positions: np.ndarray,
        source_pixel_indices: np.ndarray,
        target_pixel_indices: np.ndarray,
        *,
        batch_size: int,
    ) -> tuple[
        list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray],
        list[bool], list[int], list[np.ndarray], np.ndarray,
    ]:
        values: list[np.ndarray] = []
        times: list[np.ndarray] = []
        unaligned: list[np.ndarray] = []
        aligned: list[np.ndarray] = []
        valid: list[bool] = []
        phase_status: list[int] = []
        accepted_warp: list[np.ndarray] = []
        module_training = [(module, module.training) for module in self.model.modules()]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            self.model.eval()
            with torch.inference_mode():
                for dataset, positions, saved_indices in (
                    (self.source_dataset, source_positions, source_pixel_indices),
                    (self.target_dataset, target_positions, target_pixel_indices),
                ):
                    for start in range(0, len(positions), batch_size):
                        batch_positions = positions[start:start + batch_size]
                        batch_indices = saved_indices[start:start + batch_size]
                        samples = [
                            prepare_snapshot_sample(
                                dataset[int(position)], indices,
                                num_pixels=self.config.num_pixels,
                            )
                            for position, indices in zip(batch_positions, batch_indices)
                        ]
                        batch = _collate_snapshot_batch(
                            samples, self.device
                        )
                        amp_dtype = getattr(torch, self.amp_dtype)
                        use_autocast = self.amp_enabled and (
                            self.device.type == "cuda" or amp_dtype == torch.bfloat16
                        )
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=amp_dtype,
                            enabled=use_autocast,
                        ):
                            details = self.model.forward_details(
                                batch["pixels"],
                                batch["valid_pixels"],
                                batch["positions"],
                                batch.get("extra"),
                            )
                        batch_mask = details.backbone.time_mask.detach().cpu().numpy().astype(bool)
                        batch_values = details.backbone.tokens.detach().float().cpu().numpy()
                        batch_times = batch["positions"].detach().float().cpu().numpy().astype(np.float32)
                        batch_valid = details.temporal.shape.valid.detach().cpu().numpy().astype(bool)
                        batch_unaligned = (
                            details.temporal.core.structure_srvf.srvf.detach().float().cpu().numpy()
                        )
                        batch_aligned = (
                            details.temporal.aligned_structure_srvf.detach().float().cpu().numpy()
                        )
                        batch_status = (
                            details.temporal.core.selection.phase_status.detach().cpu().numpy().astype(np.uint8)
                        )
                        batch_warp = (
                            details.temporal.core.selection.accepted_warp.warp.detach().float().cpu().numpy()
                        )
                        del details, batch, samples
                        for sample_index in range(len(batch_positions)):
                            mask = batch_mask[sample_index]
                            values.append(batch_values[sample_index, mask].copy())
                            times.append(batch_times[sample_index, mask].copy())
                            shape_valid = bool(batch_valid[sample_index])
                            before = batch_unaligned[sample_index].copy()
                            after = batch_aligned[sample_index].copy()
                            unaligned.append(before if shape_valid else np.zeros_like(before))
                            aligned.append(after if shape_valid else np.zeros_like(after))
                            valid.append(shape_valid)
                            phase_status.append(int(batch_status[sample_index]))
                            accepted_warp.append(batch_warp[sample_index].copy())
                        del (
                            batch_mask, batch_values, batch_times, batch_valid,
                            batch_unaligned, batch_aligned, batch_status, batch_warp,
                        )
            shape_grid = self.model.temporal_features.coordinates.canonical_grid.detach().float().cpu().numpy().astype(np.float32)
            accepted = normalize_accepted_warp(
                np.asarray(phase_status, dtype=np.uint8),
                np.stack(accepted_warp),
                shape_grid,
            )
            return values, times, unaligned, aligned, valid, phase_status, list(accepted), shape_grid
        finally:
            for module, training in module_training:
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    def _collect_features_with_oom_retry(
        self,
        epoch: int,
        source_positions: np.ndarray,
        target_positions: np.ndarray,
        source_pixel_indices: np.ndarray,
        target_pixel_indices: np.ndarray,
    ) -> tuple[tuple[Any, ...], int]:
        batch_size = self.batch_size
        while True:
            oom_message: str | None = None
            try:
                result = self._collect_features(
                    source_positions,
                    target_positions,
                    source_pixel_indices,
                    target_pixel_indices,
                    batch_size=batch_size,
                )
                return result, batch_size
            except torch.cuda.OutOfMemoryError as error:
                oom_message = str(error)
            gc.collect()
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise torch.cuda.OutOfMemoryError(oom_message or "feature snapshot CUDA OOM")
            next_batch_size = max(1, batch_size // 2)
            print(
                f"FEATURE_SNAPSHOT_RETRY|epoch={epoch}|old_batch_size={batch_size}"
                f"|new_batch_size={next_batch_size}|reason=CUDA_OOM",
                flush=True,
            )
            batch_size = next_batch_size

    def _record_failure(self, epoch: int, error: Exception) -> None:
        try:
            update_snapshot_status(
                self.snapshot_dir,
                epoch=epoch,
                epoch_status={
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        except Exception as status_error:
            print(
                f"FEATURE_SNAPSHOT_STATUS|epoch={epoch}|status=FAILED"
                f"|error={type(status_error).__name__}:{status_error}",
                flush=True,
            )

    def _fit_or_load_basis(
        self,
        epoch: int,
        values: list[np.ndarray],
        unaligned: list[np.ndarray],
        aligned: list[np.ndarray],
        valid: list[bool],
    ) -> ProjectionBasis:
        basis_path = self.snapshot_dir / "projection_basis.npz"
        if basis_path.exists():
            basis = load_projection_basis(basis_path)
            update_snapshot_status(
                self.snapshot_dir,
                projection_status={"status": "FITTED", "fitted_epoch": basis.fitted_epoch},
            )
            print(
                f"FEATURE_PROJECTION|epoch={epoch}|components={basis.num_components}"
                f"|status=REUSED|fitted_epoch={basis.fitted_epoch}|path={basis_path}",
                flush=True,
            )
            return basis
        try:
            if any(len(rows) == 0 for rows in values):
                raise ValueError("PCA requires at least one valid PSE token per parcel")
            pse_rows = np.concatenate(values, axis=0)
            pse_weights = np.concatenate([
                np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
                for rows in values
            ])
            pse_fit = fit_deterministic_pca(
                pse_rows, NUM_PROJECTION_COMPONENTS, weights=pse_weights
            )
            valid_mask = np.asarray(valid, dtype=np.bool_)
            before = np.stack(unaligned)[valid_mask]
            after = np.stack(aligned)[valid_mask]
            if not len(before):
                raise ValueError("PCA requires at least one valid Shape sample")
            shape_rows = np.concatenate((before, after), axis=0)
            grid_size = shape_rows.shape[1]
            shape_weights = np.full(
                shape_rows.shape[0] * grid_size,
                1.0 / grid_size,
                dtype=np.float64,
            )
            shape_fit = fit_deterministic_pca(
                shape_rows.reshape(-1, shape_rows.shape[-1]),
                NUM_PROJECTION_COMPONENTS,
                weights=shape_weights,
            )
            _atomic_npz(
                basis_path,
                schema_version=np.asarray(SNAPSHOT_SCHEMA_VERSION, dtype=np.int64),
                fitted_epoch=np.asarray(epoch, dtype=np.int64),
                num_components=np.asarray(NUM_PROJECTION_COMPONENTS, dtype=np.int64),
                pse_mean=pse_fit.mean,
                pse_components=pse_fit.components,
                pse_explained_variance=pse_fit.explained_variance,
                pse_explained_variance_ratio=pse_fit.explained_variance_ratio,
                shape_mean=shape_fit.mean,
                shape_components=shape_fit.components,
                shape_explained_variance=shape_fit.explained_variance,
                shape_explained_variance_ratio=shape_fit.explained_variance_ratio,
            )
        except Exception as error:
            update_snapshot_status(
                self.snapshot_dir,
                projection_status={
                    "status": "FAILED",
                    "fitted_epoch": None,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            print(
                f"FEATURE_PROJECTION|epoch={epoch}|status=FAILED"
                f"|error={type(error).__name__}:{error}",
                flush=True,
            )
            raise
        basis = load_projection_basis(basis_path)
        self._update_manifest_fitted_epoch(epoch)
        update_snapshot_status(
            self.snapshot_dir,
            projection_status={"status": "FITTED", "fitted_epoch": epoch},
        )
        print(
            f"FEATURE_PROJECTION|epoch={epoch}|components={basis.num_components}"
            f"|status=FITTED|path={basis_path}",
            flush=True,
        )
        return basis

    def capture(self, epoch: int) -> SnapshotCaptureResult:
        if not self.config.should_capture(epoch):
            return SnapshotCaptureResult("DISABLED", None, 0)
        selected: SelectedSamples | None = None
        try:
            selected, source_positions, target_positions = self._load_or_create_selection()
            epoch_path = self.snapshot_dir / f"epoch_{epoch:04d}.npz"
            if epoch_path.exists():
                self._validate_epoch(epoch_path, epoch, len(selected.sample_index))
                update_snapshot_status(
                    self.snapshot_dir,
                    epoch=epoch,
                    epoch_status={
                        "status": "SUCCESS",
                        "batch_size": self.batch_size,
                        "path": epoch_path.name,
                    },
                )
                print(
                    f"FEATURE_SNAPSHOT|epoch={epoch}|status=EXISTS|path={epoch_path}",
                    flush=True,
                )
                return SnapshotCaptureResult("EXISTS", epoch_path, len(selected.sample_index))

            source_pixel_indices = selected.pixel_indices[selected.domain == 0]
            target_pixel_indices = selected.pixel_indices[selected.domain == 1]
            collected, used_batch_size = self._collect_features_with_oom_retry(
                epoch,
                source_positions,
                target_positions,
                source_pixel_indices,
                target_pixel_indices,
            )
            (
                values, times, unaligned, aligned, valid,
                phase_status, accepted_warp, shape_grid,
            ) = collected
            basis = self._fit_or_load_basis(
                epoch, values, unaligned, aligned, valid
            )
            pse_full = np.concatenate(values, axis=0)
            unaligned_full = np.stack(unaligned)
            aligned_full = np.stack(aligned)
            pse_pc = project_features(pse_full, basis.pse_mean, basis.pse_components)
            unaligned_pc = project_features(
                unaligned_full.reshape(-1, unaligned_full.shape[-1]),
                basis.shape_mean,
                basis.shape_components,
            ).reshape(*unaligned_full.shape[:-1], basis.num_components)
            shape_pc = project_features(
                aligned_full.reshape(-1, aligned_full.shape[-1]),
                basis.shape_mean,
                basis.shape_components,
            ).reshape(*aligned_full.shape[:-1], basis.num_components)
            valid_array = np.asarray(valid, dtype=np.bool_)
            unaligned_pc[~valid_array] = 0
            shape_pc[~valid_array] = 0
            output_dtype = np.float16 if self.config.dtype == "float16" else np.float32
            offsets = np.concatenate(([0], np.cumsum([len(value) for value in values]))).astype(np.int64)
            _atomic_npz(
                epoch_path,
                epoch=np.asarray(epoch, dtype=np.int64),
                projection_fitted_epoch=np.asarray(basis.fitted_epoch, dtype=np.int64),
                schema_version=np.asarray(SNAPSHOT_SCHEMA_VERSION, dtype=np.int64),
                pse_pc_values=pse_pc.astype(output_dtype),
                pse_times=np.concatenate(times).astype(np.float32),
                pse_offsets=offsets,
                unaligned_shape_pc=unaligned_pc.astype(output_dtype),
                aligned_shape_pc=shape_pc.astype(output_dtype),
                shape_grid=shape_grid,
                shape_valid=valid_array,
                phase_status=np.asarray(phase_status, dtype=np.uint8),
                accepted_warp=np.stack(accepted_warp).astype(output_dtype),
            )
            self._validate_epoch(epoch_path, epoch, len(selected.sample_index))
            update_snapshot_status(
                self.snapshot_dir,
                epoch=epoch,
                epoch_status={
                    "status": "SUCCESS",
                    "batch_size": used_batch_size,
                    "path": epoch_path.name,
                },
            )
        except Exception as error:
            error_text = f"{type(error).__name__}:{error}"
            print(
                f"FEATURE_SNAPSHOT|epoch={epoch}|status=FAILED"
                f"|error={error_text}",
                flush=True,
            )
            self._record_failure(epoch, error)
            return SnapshotCaptureResult(
                "FAILED",
                None,
                0 if selected is None else len(selected.sample_index),
                error_text,
            )
        print(
            f"FEATURE_SNAPSHOT|epoch={epoch}|samples={len(selected.sample_index)}"
            f"|pse_points={len(pse_pc)}|shape_valid={int(np.count_nonzero(valid))}"
            f"|components={basis.num_components}|status=SUCCESS|path={epoch_path}",
            flush=True,
        )
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
        samples_per_class=int(getattr(config, "feature_snapshot_samples_per_class", 32)),
        dtype=str(getattr(config, "feature_snapshot_dtype", "float16")),
        snapshot_dir=getattr(config, "feature_snapshot_dir", None),
        num_pixels=int(getattr(config, "num_pixels", 64)),
        pixel_selection_seed=int(getattr(config, "seed", 1)),
    )
    if snapshot_config.interval == 0:
        return None
    if snapshot_config.snapshot_dir is None:
        raise ValueError("--feature_snapshot_dir is required when snapshots are enabled")
    snapshot_batch_size = int(getattr(config, "feature_snapshot_batch_size", 8))
    if snapshot_batch_size <= 0:
        raise ValueError("--feature_snapshot_batch_size must be positive")

    from dataset import PixelSetData
    common = dict(
        data_root=config.data_root,
        classes=config.classes,
        transform=None,
        with_extra=False,
        closed_set=config.closed_set,
        combine_spring_and_winter=config.combine_spring_and_winter,
        time_coordinate_mode=getattr(config, "time_coordinate_mode", "canonical_day_of_year"),
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
        batch_size=snapshot_batch_size,
        source_domain=config.source,
        target_domain=config.target,
        amp_enabled=bool(getattr(config, "amp", False)),
        amp_dtype=str(getattr(config, "amp_dtype", "float16")),
    )
