#!/usr/bin/env python3
"""Protocol-safe frozen feature extraction for the structure semantic ABC audit."""

from __future__ import annotations

import copy
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch

from analysis.audit_e_representation_chain import _dataset as make_dataset
from analysis.diagnose_eg_pair import (
    require_complete_checkpoint,
    validate_uda_checkpoint,
)
from analysis.summarize_reliable_composition_ab import (
    official_create_splits as create_official_splits,
    official_prepare_data_protocol as prepare_official_protocol,
)
from analysis.structure_representation_chain_audit import pixel_budget_batches
from torch.nn import functional as F


EVENT_COMPONENTS = (
    "strength",
    "concentration",
    "phase_k1_cos",
    "phase_k1_sin",
    "phase_k2_cos",
    "phase_k2_sin",
)
ALLOWED_SPLITS = ("source_train", "source_val", "target_train", "target_val")


@dataclass(frozen=True)
class FrozenModels:
    student: torch.nn.Module
    teacher: torch.nn.Module
    config: SimpleNamespace
    packet: dict
    shifts: dict[str, int]


@dataclass(frozen=True)
class FeatureCacheManifest:
    schema_version: int
    files: dict[str, dict[str, str]]
    extraction_passes: dict[str, dict[str, int]]
    applied_shifts: dict[str, dict[str, int]]
    sample_counts: dict[str, int]
    target_test_used: bool = False
    target_train_labels_cached: bool = False

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "files": self.files,
            "extraction_passes": self.extraction_passes,
            "applied_shifts": self.applied_shifts,
            "sample_counts": self.sample_counts,
            "target_test_used": self.target_test_used,
            "target_train_labels_cached": self.target_train_labels_cached,
        }


def sample_keys(domain: str, parcel_ids: Iterable[int]):
    keys = tuple((str(domain), int(parcel)) for parcel in parcel_ids)
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate sample key in domain={domain}")
    return keys


def response_to_anchor_events(response, anchor_count: int = 16):
    """Decode block-major Response96 into [sample, anchor, six components]."""
    values = np.asarray(response)
    expected = int(anchor_count) * len(EVENT_COMPONENTS)
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(
            f"response must be [N,{expected}] for {anchor_count} anchors; "
            f"got {values.shape}"
        )
    blocks = [
        values[:, index * anchor_count:(index + 1) * anchor_count]
        for index in range(len(EVENT_COMPONENTS))
    ]
    return np.stack(blocks, axis=-1)


def _clone_config(config, data_root, seed):
    raw = vars(config) if not isinstance(config, dict) else config
    copied = copy.deepcopy(dict(raw))
    copied["data_root"] = str(data_root)
    copied["seed"] = int(seed)
    return SimpleNamespace(**copied)


def build_abc_datasets(config, data_root, seed):
    """Replay official indices while deliberately never constructing Target Test."""
    cfg = _clone_config(config, data_root, seed)
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    eligible, protocol = prepare_official_protocol(cfg)
    splits = create_official_splits(
        [cfg.source, cfg.target], cfg.num_folds, eligible,
        cfg.val_ratio, cfg.test_ratio,
    )[0]
    requested = (
        ("source_train", cfg.source, "train"),
        ("source_val", cfg.source, "val"),
        ("target_train", cfg.target, "train"),
        ("target_val", cfg.target, "val"),
    )
    datasets = {
        name: make_dataset(data_root, domain, cfg, splits[domain][split_name])
        for name, domain, split_name in requested
    }
    if set(datasets) != set(ALLOWED_SPLITS):
        raise RuntimeError("ABC protocol constructed an unexpected split")
    return cfg, protocol, splits, datasets


def replay_abc_datasets(config, data_root, seed, split_manifest):
    """Rebuild exactly the parcel sets cached during extraction."""
    if set(split_manifest) != set(ALLOWED_SPLITS):
        raise ValueError("checkpoint split manifest must contain exactly the four ABC splits")
    cfg = _clone_config(config, data_root, seed)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    datasets = {}
    for split in ALLOWED_SPLITS:
        expected_domain = cfg.source if split.startswith("source") else cfg.target
        rows = list(split_manifest[split])
        domains = {str(row["domain"]) for row in rows}
        if domains - {str(expected_domain)}:
            raise ValueError(f"cached {split} domain does not match checkpoint config")
        parcels = [int(row["parcel_index"]) for row in rows]
        sample_keys(expected_domain, parcels)
        dataset = make_dataset(data_root, expected_domain, cfg, parcels)
        rebuilt = [int(value) for value in dataset.get_parcel_indices()]
        if len(rebuilt) != len(parcels) or set(rebuilt) != set(parcels):
            missing = sorted(set(parcels) - set(rebuilt))
            extra = sorted(set(rebuilt) - set(parcels))
            raise ValueError(
                f"cached {split} parcel replay mismatch: missing={missing[:5]}, extra={extra[:5]}"
            )
        datasets[split] = dataset
    return datasets


def create_model_from_config(config):
    from train import create_model

    return create_model(config)


def _integer_shift(value):
    shift = torch.as_tensor(value).detach().cpu().reshape(-1)
    if shift.numel() != 1:
        raise ValueError("global temporal shift must be one scalar integer day")
    number = float(shift.item())
    if not number.is_integer():
        raise ValueError(f"global temporal shift must be an integer day, got {number}")
    return int(number)


def _strict_load(model, state, role):
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        label = "EMA Teacher" if role == "teacher" else "Student"
        raise RuntimeError(f"strict {label} load failed: {error}") from error


def load_frozen_models(checkpoint, task, variant, device):
    checkpoint = Path(checkpoint)
    packet = require_complete_checkpoint(checkpoint)
    validate_uda_checkpoint(packet, task, variant)
    if "teacher_state_dict" not in packet:
        raise RuntimeError("strict EMA Teacher load failed: teacher_state_dict missing")
    raw_config = packet.get("config")
    if raw_config is None:
        raise ValueError("checkpoint config missing")
    config = SimpleNamespace(**dict(
        raw_config if isinstance(raw_config, dict) else vars(raw_config)
    ))
    student = create_model_from_config(config)
    teacher = create_model_from_config(config)
    _strict_load(student, packet.get("state_dict"), "student")
    _strict_load(teacher, packet.get("teacher_state_dict"), "teacher")
    shift = _integer_shift(packet.get("global_temporal_shift", 0))
    target = torch.device(device)
    student = student.to(target).eval()
    teacher = teacher.to(target).eval()
    for model in (student, teacher):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return FrozenModels(
        student=student,
        teacher=teacher,
        config=config,
        packet=packet,
        shifts={"student": 0, "teacher": shift},
    )


def _atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=path.stem + ".", suffix=".npz", dir=path.parent, delete=False,
    )
    handle.close()
    try:
        np.savez_compressed(handle.name, **arrays)
        os.replace(handle.name, path)
    finally:
        Path(handle.name).unlink(missing_ok=True)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=path.stem + ".", suffix=".json", dir=path.parent,
        delete=False, mode="w", encoding="utf-8",
    )
    try:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    finally:
        if not handle.closed:
            handle.close()
        Path(handle.name).unlink(missing_ok=True)


def _representative_arrays(records):
    if not records:
        return {
            "representative_anchor": np.empty(0, dtype=np.int64),
            "representative_kind": np.empty(0, dtype="U8"),
            "representative_score": np.empty(0, dtype=np.float32),
            "representative_parcel_index": np.empty(0, dtype=np.int64),
            "representative_window_index": np.empty(0, dtype=np.int64),
            "representative_curves": np.empty((0, 0, 0), dtype=np.float32),
            "representative_standardized": np.empty((0, 0, 0), dtype=np.float32),
            "representative_differences": np.empty((0, 0, 0), dtype=np.float32),
            "representative_mean": np.empty((0, 0), dtype=np.float32),
            "representative_std": np.empty((0, 0), dtype=np.float32),
        }
    ordered = sorted(records, key=lambda row: (row["anchor"], row["kind"], -row["score"]))
    return {
        "representative_anchor": np.asarray([row["anchor"] for row in ordered], dtype=np.int64),
        "representative_kind": np.asarray([row["kind"] for row in ordered]),
        "representative_score": np.asarray([row["score"] for row in ordered], dtype=np.float32),
        "representative_parcel_index": np.asarray([row["parcel"] for row in ordered], dtype=np.int64),
        "representative_window_index": np.asarray([row["window"] for row in ordered], dtype=np.int64),
        "representative_curves": np.stack([row["curve"] for row in ordered]),
        "representative_standardized": np.stack([row["standardized"] for row in ordered]),
        "representative_differences": np.stack([row["difference"] for row in ordered]),
        "representative_mean": np.stack([row["mean"] for row in ordered]),
        "representative_std": np.stack([row["std"] for row in ordered]),
    }


def _semantic_collate(samples):
    max_pixels = max(sample["pixels"].shape[-1] for sample in samples)
    pixels, masks = [], []
    for sample in samples:
        pad = max_pixels - sample["pixels"].shape[-1]
        pixels.append(F.pad(sample["pixels"], (0, pad)))
        masks.append(F.pad(sample["valid_pixels"], (0, pad)))
    return {
        "pixels": torch.stack(pixels),
        "valid_pixels": torch.stack(masks),
        "positions": torch.stack([sample["positions"] for sample in samples]),
        "extra": torch.stack([sample["extra"] for sample in samples]),
        "label": torch.stack([sample["label"] for sample in samples]),
        "parcel_index": torch.as_tensor(
            [int(sample["parcel_index"]) for sample in samples], dtype=torch.long,
        ),
    }


def semantic_loader(dataset, batch_size, pixel_budget=8192, num_workers=0):
    batches = pixel_budget_batches(
        [shape[2] for shape in dataset.get_shapes()], batch_size, pixel_budget,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=batches, collate_fn=_semantic_collate,
        num_workers=int(num_workers), pin_memory=torch.cuda.is_available(),
    )


def _update_representatives(store, similarity, windows, parcels, keep):
    if keep <= 0:
        return
    similarity = similarity.detach().float().cpu().numpy()
    windows = windows.detach().float().cpu().numpy()
    parcels = np.asarray(parcels, dtype=np.int64)
    anchors = similarity.shape[-1]
    for anchor in range(anchors):
        for kind, chooser in (("top", np.argmax), ("contrast", np.argmin)):
            key = (anchor, kind)
            current = store.setdefault(key, [])
            for sample in range(similarity.shape[0]):
                window = int(chooser(similarity[sample, :, anchor]))
                curve = windows[sample, window]
                mean = curve.mean(axis=0)
                std = curve.std(axis=0)
                standardized = (curve - mean) / np.maximum(std, 1e-6)
                current.append({
                    "anchor": anchor,
                    "kind": kind,
                    "score": float(similarity[sample, window, anchor]),
                    "parcel": int(parcels[sample]),
                    "window": window,
                    "curve": curve.astype(np.float32),
                    "standardized": standardized.astype(np.float32),
                    "difference": np.diff(curve, axis=0).astype(np.float32),
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                })
            current.sort(key=lambda row: row["score"], reverse=(kind == "top"))
            del current[keep:]


@torch.inference_mode()
def _extract_split(
    model, dataset, domain, shift, device, batch_size, pixel_budget,
    representative_count, include_labels,
):
    storage = {
        key: [] for key in (
            "parcel_index", "response", "similarity", "pooling_weight",
            "anchor_probability", "logits", "confidence", "prediction",
        )
    }
    if include_labels:
        storage["label"] = []
    representatives = {}
    centers_days = None
    target = torch.device(device)
    for raw in semantic_loader(
        dataset, batch_size=batch_size, num_workers=0, pixel_budget=pixel_budget,
    ):
        batch = {
            key: value.to(target) if torch.is_tensor(value) else value
            for key, value in raw.items()
        }
        output = model.forward_with_temporal_shift(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch["extra"], temporal_shift=shift, return_dict=True,
        )
        response = output["shapelet_response"]
        similarity = output["shapelet_similarity"]
        beta = float(model.structure_branch.shapelet_dictionary.beta)
        pooling = torch.softmax(beta * similarity, dim=1)
        probability = torch.softmax(beta * similarity, dim=-1)
        probability_logits = torch.softmax(output["logits"], dim=1)
        confidence, prediction = probability_logits.max(1)
        values = {
            "parcel_index": batch["parcel_index"].long(),
            "response": response,
            "similarity": similarity,
            "pooling_weight": pooling,
            "anchor_probability": probability,
            "logits": output["logits"],
            "confidence": confidence,
            "prediction": prediction.long(),
        }
        if include_labels:
            values["label"] = batch["label"].long()
        for key, value in values.items():
            storage[key].append(value.detach().float().cpu() if value.is_floating_point() else value.detach().cpu())

        groups, _, centers = model.structure_branch.window_extractor(
            output["exposed_curve"], return_centers=True,
        )
        windows = torch.cat(groups, dim=1)
        if centers_days is None:
            centers_days = (
                centers.detach().float().cpu()
                * model.structure_branch.exposer.period_days
                / model.structure_branch.window_extractor.grid_points
            ).numpy()
        _update_representatives(
            representatives, similarity, windows,
            batch["parcel_index"].detach().cpu().numpy(), representative_count,
        )
    arrays = {
        key: torch.cat(parts).numpy() for key, parts in storage.items()
    }
    arrays["parcel_index"] = arrays["parcel_index"].astype(np.int64)
    arrays["prediction"] = arrays["prediction"].astype(np.int64)
    if "label" in arrays:
        arrays["label"] = arrays["label"].astype(np.int64)
    arrays["domain"] = np.asarray([str(domain)] * len(arrays["parcel_index"]))
    arrays["events"] = response_to_anchor_events(arrays["response"])
    arrays["window_centers_days"] = np.asarray(centers_days, dtype=np.float32)
    records = [row for rows in representatives.values() for row in rows]
    arrays.update(_representative_arrays(records))
    sample_keys(domain, arrays["parcel_index"])
    return arrays


def extract_checkpoint_features(
    frozen, datasets, domains, output_dir, device="cuda", batch_size=64,
    pixel_budget=8192, representative_count=2,
):
    output_dir = Path(output_dir)
    files = {role: {} for role in ("student", "teacher")}
    passes = {role: {} for role in ("student", "teacher")}
    shifts = {role: {} for role in ("student", "teacher")}
    counts = {}
    for role in ("student", "teacher"):
        model = getattr(frozen, role)
        shift = int(frozen.shifts[role])
        for split in ALLOWED_SPLITS:
            if split not in datasets:
                raise ValueError(f"missing ABC split: {split}")
            domain = domains["source" if split.startswith("source") else "target"]
            arrays = _extract_split(
                model, datasets[split], domain, shift, device, batch_size,
                pixel_budget, representative_count,
                include_labels=split != "target_train",
            )
            relative = f"features/{role}_{split}.npz"
            _atomic_npz(output_dir / relative, arrays)
            files[role][split] = relative
            passes[role][split] = 1
            shifts[role][split] = shift
            counts[split] = len(arrays["parcel_index"])
    manifest = FeatureCacheManifest(
        schema_version=1, files=files, extraction_passes=passes,
        applied_shifts=shifts, sample_counts=counts,
    )
    _atomic_json(output_dir / "feature_cache_manifest.json", manifest.to_dict())
    validate_feature_cache(output_dir, manifest)
    return manifest


def load_feature_split(cache_dir, model_role, split):
    if model_role not in ("student", "teacher") or split not in ALLOWED_SPLITS:
        raise ValueError(f"invalid feature cache selection: {model_role}/{split}")
    path = Path(cache_dir) / "features" / f"{model_role}_{split}.npz"
    with np.load(path, allow_pickle=False) as packet:
        return {key: packet[key] for key in packet.files}


def validate_feature_cache(cache_dir, manifest=None):
    cache_dir = Path(cache_dir)
    if manifest is None:
        raw = json.loads((cache_dir / "feature_cache_manifest.json").read_text(encoding="utf-8"))
        manifest = FeatureCacheManifest(**raw)
    if manifest.target_test_used or manifest.target_train_labels_cached:
        raise ValueError("feature cache violates unlabeled/Target Test protocol")
    for role in ("student", "teacher"):
        expected_shift = None
        for split in ALLOWED_SPLITS:
            arrays = load_feature_split(cache_dir, role, split)
            if split == "target_train" and "label" in arrays:
                raise ValueError("Target Train labels must not be cached")
            sample_keys(str(arrays["domain"][0]), arrays["parcel_index"])
            np.testing.assert_allclose(
                arrays["events"], response_to_anchor_events(arrays["response"]),
            )
            current = int(manifest.applied_shifts[role][split])
            if expected_shift is None:
                expected_shift = current
            elif current != expected_shift:
                raise ValueError(f"inconsistent {role} temporal coordinate")
    if any(value != 0 for value in manifest.applied_shifts["student"].values()):
        raise ValueError("Student normal representation must use zero shift")
    return True
