"""Compact, deterministic PSE and aligned-shape training snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


SNAPSHOT_SCHEMA_VERSION = 2
NUM_PROJECTION_COMPONENTS = 2
_SELECTED_FIELDS = {
    "sample_index",
    "domain",
    "label",
    "source_domain",
    "target_domain",
    "selection_seed",
    "samples_per_class",
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
    "aligned_shape_pc",
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


def fit_deterministic_pca(
    values: np.ndarray, num_components: int = NUM_PROJECTION_COMPONENTS
) -> PCAFit:
    """Fit PCA with deterministic eigenvalue ordering and component signs."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA requires at least two feature rows")
    if values.shape[1] < num_components:
        raise ValueError("feature dimension is smaller than requested PCA components")
    if not np.isfinite(values).all():
        raise ValueError("PCA input must be finite")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / float(values.shape[0] - 1)
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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def load_projection_basis(path: Path) -> ProjectionBasis:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != _PROJECTION_FIELDS:
            raise ValueError("projection basis fields are invalid")
        schema_version = int(data["schema_version"].item())
        fitted_epoch = int(data["fitted_epoch"].item())
        num_components = int(data["num_components"].item())
        if (
            schema_version != SNAPSHOT_SCHEMA_VERSION
            or num_components != NUM_PROJECTION_COMPONENTS
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
    """Capture fixed parcels and store only fixed-basis PC1/PC2 curves."""

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
            "snapshot_representation": "fixed_pca_projection",
            "projection_fitted_epoch": self.config.interval,
            "num_projection_components": NUM_PROJECTION_COMPONENTS,
            "pse_saved_semantics": "pixel_set_encoder_tokens_projected_to_fixed_pc1_pc2",
            "aligned_shape_saved_semantics": "aligned_structure_srvf_projected_to_fixed_pc1_pc2",
            "raw_feature_dimension": int(self.model.backbone.feature_dim),
            "raw_full_features_saved": False,
            "pse_storage": "ragged",
            "dtype": self.config.dtype,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "class_ids": [int(value) for value in np.unique(selected.label)],
            "class_names": class_names,
            "target_label_usage": "diagnostic_snapshot_and_offline_visualization_only",
        }

    def _create_selection(self) -> tuple[SelectedSamples, np.ndarray, np.ndarray]:
        source_labels = np.asarray(self.source_dataset.get_labels())
        target_labels = np.asarray(self.target_dataset.get_labels())
        source_parcels = np.asarray(self.source_dataset.get_parcel_indices())
        target_parcels = np.asarray(self.target_dataset.get_parcel_indices())
        source_positions = deterministic_class_selection(
            source_labels, source_parcels,
            samples_per_class=self.config.samples_per_class,
            seed=self.config.selection_seed,
        )
        target_positions = deterministic_class_selection(
            target_labels, target_parcels,
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
            or selected.selection_seed != self.config.selection_seed
            or selected.samples_per_class != self.config.samples_per_class
        ):
            raise ValueError("existing selected samples do not match snapshot configuration")
        manifest_path = self.snapshot_dir / "manifest.json"
        expected_manifest = self._manifest(selected)
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
                    lookup[int(parcel)] for parcel in selected.sample_index[selected.domain == domain]
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
            if int(data["projection_fitted_epoch"].item()) != self.config.interval:
                raise ValueError("snapshot projection fitted epoch is invalid")
            offsets = data["pse_offsets"]
            aligned = data["aligned_shape_pc"]
            valid = data["shape_valid"]
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
            if len(times) != len(values) or aligned.shape[0] != expected_samples or valid.shape != (expected_samples,):
                raise ValueError("snapshot sample dimensions are invalid")
            if values.ndim != 2 or values.shape[1] != NUM_PROJECTION_COMPONENTS:
                raise ValueError("snapshot PSE projection dimensions are invalid")
            if aligned.ndim != 3 or aligned.shape[-1] != NUM_PROJECTION_COMPONENTS or grid.shape != (aligned.shape[1],):
                raise ValueError("snapshot aligned-shape dimensions are invalid")
            output_dtype = np.dtype(self.config.dtype)
            if values.dtype != output_dtype or aligned.dtype != output_dtype:
                raise ValueError("snapshot projection dtype is invalid")
            if times.dtype != np.float32 or grid.dtype != np.float32 or valid.dtype != np.bool_:
                raise ValueError("snapshot metadata dtypes are invalid")
            if not all(np.all(np.isfinite(value)) for value in (values, times, aligned, grid)):
                raise ValueError("snapshot arrays must be finite")

    def _collect_features(
        self,
        source_positions: np.ndarray,
        target_positions: np.ndarray,
        *,
        batch_size: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[bool], np.ndarray]:
        values: list[np.ndarray] = []
        times: list[np.ndarray] = []
        aligned: list[np.ndarray] = []
        valid: list[bool] = []
        module_training = [(module, module.training) for module in self.model.modules()]
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            self.model.eval()
            with torch.inference_mode():
                for dataset, positions in (
                    (self.source_dataset, source_positions),
                    (self.target_dataset, target_positions),
                ):
                    for start in range(0, len(positions), batch_size):
                        batch_positions = positions[start:start + batch_size]
                        samples = [dataset[int(position)] for position in batch_positions]
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
                        batch_aligned = (
                            details.temporal.aligned_structure_srvf.detach().float().cpu().numpy()
                        )
                        del details, batch, samples
                        for sample_index in range(len(batch_positions)):
                            mask = batch_mask[sample_index]
                            values.append(batch_values[sample_index, mask].copy())
                            times.append(batch_times[sample_index, mask].copy())
                            shape_valid = bool(batch_valid[sample_index])
                            shape = batch_aligned[sample_index].copy()
                            aligned.append(shape if shape_valid else np.zeros_like(shape))
                            valid.append(shape_valid)
                        del batch_mask, batch_values, batch_times, batch_valid, batch_aligned
            shape_grid = self.model.temporal_features.coordinates.canonical_grid.detach().float().cpu().numpy().astype(np.float32)
            return values, times, aligned, valid, shape_grid
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
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[bool], np.ndarray]:
        batch_size = self.batch_size
        while True:
            oom_message: str | None = None
            try:
                return self._collect_features(
                    source_positions,
                    target_positions,
                    batch_size=batch_size,
                )
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
            _atomic_text(
                self.snapshot_dir / "SNAPSHOT_FAILED",
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                ) + "\n",
            )
        except Exception as marker_error:
            print(
                f"FEATURE_SNAPSHOT_FAILURE_MARKER|epoch={epoch}|status=FAILED"
                f"|error={type(marker_error).__name__}:{marker_error}",
                flush=True,
            )

    def _fit_or_load_basis(
        self, epoch: int, values: list[np.ndarray], aligned: list[np.ndarray], valid: list[bool]
    ) -> ProjectionBasis:
        basis_path = self.snapshot_dir / "projection_basis.npz"
        if basis_path.exists():
            basis = load_projection_basis(basis_path)
            if basis.fitted_epoch != self.config.interval:
                raise ValueError("projection basis fitted epoch does not match snapshot interval")
            print(
                f"FEATURE_PROJECTION|epoch={epoch}|components={basis.num_components}"
                f"|status=REUSED|path={basis_path}",
                flush=True,
            )
            return basis
        if epoch != self.config.interval:
            raise ValueError("projection basis is missing before a later snapshot epoch")
        pse_fit = fit_deterministic_pca(np.concatenate(values, axis=0))
        shape_rows = np.stack(aligned)[np.asarray(valid, dtype=np.bool_)]
        if not len(shape_rows):
            raise ValueError("PCA requires at least one valid aligned Shape sample")
        shape_fit = fit_deterministic_pca(shape_rows.reshape(-1, shape_rows.shape[-1]))
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
        basis = load_projection_basis(basis_path)
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
                print(
                    f"FEATURE_SNAPSHOT|epoch={epoch}|status=EXISTS|path={epoch_path}",
                    flush=True,
                )
                return SnapshotCaptureResult("EXISTS", epoch_path, len(selected.sample_index))

            values, times, aligned, valid, shape_grid = self._collect_features_with_oom_retry(
                epoch, source_positions, target_positions
            )
            basis = self._fit_or_load_basis(epoch, values, aligned, valid)
            pse_full = np.concatenate(values, axis=0)
            aligned_full = np.stack(aligned)
            pse_pc = project_features(pse_full, basis.pse_mean, basis.pse_components)
            shape_pc = project_features(
                aligned_full.reshape(-1, aligned_full.shape[-1]),
                basis.shape_mean,
                basis.shape_components,
            ).reshape(*aligned_full.shape[:-1], basis.num_components)
            shape_pc[~np.asarray(valid, dtype=np.bool_)] = 0
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
                aligned_shape_pc=shape_pc.astype(output_dtype),
                shape_grid=shape_grid,
                shape_valid=np.asarray(valid, dtype=np.bool_),
            )
            self._validate_epoch(epoch_path, epoch, len(selected.sample_index))
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
        selection_seed=int(getattr(config, "seed", 1)),
    )
    if snapshot_config.interval == 0:
        return None
    if snapshot_config.snapshot_dir is None:
        raise ValueError("--feature_snapshot_dir is required when snapshots are enabled")
    snapshot_batch_size = int(getattr(config, "feature_snapshot_batch_size", 8))
    if snapshot_batch_size <= 0:
        raise ValueError("--feature_snapshot_batch_size must be positive")

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
