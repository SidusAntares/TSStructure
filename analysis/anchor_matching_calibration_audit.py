#!/usr/bin/env python3
"""Frozen Source-only P0 audit for Anchor matching calibration evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.nn import functional as F

from analysis.structure_semantic_features import (
    build_abc_datasets,
    create_model_from_config,
    semantic_loader,
)


CALIBRATION_PROTOCOL = {
    "normal_change_quantile": 0.95,
    "minimum_total": 50,
    "minimum_per_class": 10,
    "bootstrap_repeats": 200,
    "bootstrap_max_groups": 2000,
    "bootstrap_confidence": 0.95,
    "stable_auroc_lower_bound": 0.5,
}
PERTURBATION_VERSION = "reverse_and_four_contiguous_blocks_v1"
VIEW_SEED_VERSION = "sha256_seed_domain_parcel_view_v1"
REQUIRED_OUTPUTS = (
    "matching_distribution.csv",
    "view_repeatability.csv",
    "morphology_counterfactual.csv",
    "calibration_candidates.csv",
    "anchor_reliability_summary.csv",
    "checkpoint_manifest.json",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=path.stem + ".", suffix=".json", delete=False,
    )
    try:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno()); handle.close()
        os.replace(handle.name, path)
    finally:
        if not handle.closed:
            handle.close()
        Path(handle.name).unlink(missing_ok=True)


def _write_csv(path, rows, fieldnames=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def deterministic_view_seed(seed, domain, parcel_index, view_id):
    payload = f"{int(seed)}|{domain}|{int(parcel_index)}|{int(view_id)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def seeded_dataset_item(dataset, index, seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1)); torch.manual_seed(int(seed))
        return dataset[index]
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


class DeterministicViewDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, domain, global_seed, view_id):
        self.dataset = dataset
        self.domain = str(domain)
        self.global_seed = int(global_seed)
        self.view_id = int(view_id)
        self.parcels = np.asarray(dataset.get_parcel_indices(), dtype=np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        seed = deterministic_view_seed(
            self.global_seed, self.domain, self.parcels[index], self.view_id,
        )
        return seeded_dataset_item(self.dataset, index, seed)

    def get_shapes(self):
        return self.dataset.get_shapes()

    def get_parcel_indices(self):
        return self.parcels.copy()

    def get_labels(self):
        return self.dataset.get_labels()


def split_ids_are_disjoint(train_ids, val_ids):
    return set(map(int, train_ids)).isdisjoint(set(map(int, val_ids)))


def pair_views_by_parcel(first, second):
    def mapping(packet):
        keys = list(zip(map(str, packet["domain"]), map(int, packet["parcel_index"])))
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate (domain, parcel_index) in a view")
        return {key: index for index, key in enumerate(keys)}
    left, right = mapping(first), mapping(second)
    if set(left) != set(right):
        raise ValueError("independent views do not contain identical parcels")
    keys = sorted(left)
    def reordered(packet, index, sample_count):
        result = {}
        identity = list(index) == list(range(sample_count))
        for key, value in packet.items():
            array = np.asarray(value)
            is_metadata = key in {"window_starts", "window_scales"}
            if array.ndim and len(array) == sample_count and not is_metadata:
                result[key] = array if identity else array[index]
            else:
                result[key] = array
        return result
    return (
        reordered(first, [left[item] for item in keys], len(left)),
        reordered(second, [right[item] for item in keys], len(right)),
    )


def _circular_indices(start, scale, grid_points):
    return {(int(start) + offset) % int(grid_points) for offset in range(int(scale))}


def circular_window_overlap(start_a, start_b, scale=24, grid_points=64, scale_b=None):
    scale_b = scale if scale_b is None else scale_b
    return bool(
        _circular_indices(start_a, scale, grid_points)
        & _circular_indices(start_b, scale_b, grid_points)
    )


def repeatability_rows(similarity_a, similarity_b, parcels, starts, scales, grid_points=64):
    left = np.asarray(similarity_a); right = np.asarray(similarity_b)
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("paired similarity must share [B,N,M]")
    starts = np.asarray(starts); scales = np.asarray(scales)
    if starts.shape != scales.shape or starts.shape != (left.shape[1],):
        raise ValueError("window starts/scales must match candidate count")
    rows = []
    for direction, source, target in (
        ("view0_to_view1", left, right), ("view1_to_view0", right, left),
    ):
        source_best = source.argmax(1); target_best = target.argmax(1)
        source_q = source.max(1)
        for sample, parcel in enumerate(np.asarray(parcels)):
            for anchor in range(left.shape[2]):
                a = int(source_best[sample, anchor]); b = int(target_best[sample, anchor])
                rows.append({
                    "parcel_index": int(parcel), "anchor": anchor,
                    "direction": direction, "level": "parcel_anchor",
                    "q": float(source_q[sample, anchor]),
                    "source_window": a, "target_window": b,
                    "exact_match": int(a == b),
                    "profile_cosine": float(np.dot(
                        source[sample, :, anchor], target[sample, :, anchor],
                    ) / max(
                        np.linalg.norm(source[sample, :, anchor])
                        * np.linalg.norm(target[sample, :, anchor]), 1e-12,
                    )),
                    "profile_mean_absolute_change": float(np.mean(np.abs(
                        source[sample, :, anchor] - target[sample, :, anchor],
                    ))),
                    "label": int(circular_window_overlap(
                        starts[a], starts[b], scales[a], grid_points, scales[b],
                    )),
                })
    return rows


def deterministic_block_reorder(windows, seed, blocks=4):
    values = torch.as_tensor(windows)
    if values.ndim != 4 or values.shape[2] < blocks:
        raise ValueError("windows must be [B,N,T,C] with enough time points")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(blocks, generator=generator).tolist()
    if order == list(range(blocks)):
        order = order[1:] + order[:1]
    pieces = torch.tensor_split(values, blocks, dim=2)
    return torch.cat([pieces[index] for index in order], dim=2)


def perturb_windows(windows, seed):
    values = torch.as_tensor(windows)
    return values.flip(2), deterministic_block_reorder(values, seed)


def freeze_morphology_thresholds(normal_absolute_change, quantile=None):
    values = np.asarray(normal_absolute_change, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("normal view changes must retain Anchor axis")
    quantile = CALIBRATION_PROTOCOL["normal_change_quantile"] if quantile is None else quantile
    return np.quantile(values.reshape(-1, values.shape[-1]), float(quantile), axis=0)


def morphology_proxy_labels(original, perturbed, thresholds):
    original = np.asarray(original, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if original.shape != perturbed.shape or thresholds.shape != (original.shape[-1],):
        raise ValueError("morphology inputs must share [...,Anchor] and Anchor thresholds")
    delta = original - perturbed
    return delta > thresholds, delta


@dataclass(frozen=True)
class MonotonicCalibration:
    slope: float
    intercept: float
    fit_split: str = "source_train"
    sample_count: int = 0
    positive_count: int = 0
    negative_count: int = 0

    def predict_proba(self, q):
        values = self.slope * np.asarray(q, dtype=np.float64) + self.intercept
        return 1. / (1. + np.exp(-np.clip(values, -40., 40.)))


def fit_monotonic_calibrator(q, labels, groups, minimum_total=None, minimum_per_class=None):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if not (len(q) == len(labels) == len(groups)):
        raise ValueError("q, labels, and parcel groups must align")
    minimum_total = CALIBRATION_PROTOCOL["minimum_total"] if minimum_total is None else minimum_total
    minimum_per_class = CALIBRATION_PROTOCOL["minimum_per_class"] if minimum_per_class is None else minimum_per_class
    counts = np.bincount(labels, minlength=2)
    if np.count_nonzero(counts) < 2:
        raise ValueError("both proxy classes are required for calibration")
    if len(labels) < minimum_total or counts.min() < minimum_per_class:
        raise ValueError("minimum support is not met for calibration")

    def objective(parameters):
        slope, intercept = parameters
        logits = np.clip(slope * q + intercept, -40., 40.)
        return float(np.mean(np.logaddexp(0., logits) - labels * logits))

    prevalence = np.clip(labels.mean(), 1e-6, 1. - 1e-6)
    result = minimize(
        objective, np.array([1., math.log(prevalence / (1. - prevalence))]),
        method="L-BFGS-B", bounds=((0., None), (None, None)),
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"monotonic calibration failed: {result.message}")
    return MonotonicCalibration(
        slope=float(result.x[0]), intercept=float(result.x[1]),
        sample_count=len(labels), positive_count=int(counts[1]), negative_count=int(counts[0]),
    )


def _ece(labels, scores, bins=10, sample_weight=None):
    labels = np.asarray(labels); scores = np.asarray(scores)
    weights = (
        np.ones(len(labels), dtype=np.float64)
        if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )
    total = weights.sum()
    edges = np.linspace(0., 1., bins + 1)
    result = 0.
    for index in range(bins):
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1]
        )
        if selected.any():
            selected_weight = weights[selected]
            mass = selected_weight.sum()
            result += mass / max(total, 1e-12) * abs(
                np.average(labels[selected], weights=selected_weight)
                - np.average(scores[selected], weights=selected_weight)
            )
    return float(result)


def _metric(labels, scores, name, sample_weight=None):
    if name == "auroc":
        return float(roc_auc_score(labels, scores, sample_weight=sample_weight))
    if name == "auprc":
        return float(average_precision_score(labels, scores, sample_weight=sample_weight))
    if name == "brier":
        return float(brier_score_loss(labels, scores, sample_weight=sample_weight))
    if name == "ece":
        return _ece(labels, scores, sample_weight=sample_weight)
    raise ValueError(f"unknown calibration metric: {name}")


def bootstrap_metric_interval(labels, scores, groups, metric, repeats=None, confidence=None, seed=1):
    labels = np.asarray(labels); scores = np.asarray(scores); groups = np.asarray(groups)
    repeats = CALIBRATION_PROTOCOL["bootstrap_repeats"] if repeats is None else int(repeats)
    confidence = CALIBRATION_PROTOCOL["bootstrap_confidence"] if confidence is None else float(confidence)
    estimate = _metric(labels, scores, metric)
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    maximum = int(CALIBRATION_PROTOCOL["bootstrap_max_groups"])
    base = (
        unique if len(unique) <= maximum
        else rng.choice(unique, size=maximum, replace=False)
    )
    group_position = {value: index for index, value in enumerate(unique.tolist())}
    base_positions = np.asarray([group_position[value] for value in base.tolist()])
    values = []
    for _ in range(repeats):
        sampled = rng.choice(base_positions, size=len(base_positions), replace=True)
        multiplicity = np.bincount(sampled, minlength=len(unique))
        weights = multiplicity[inverse]
        selected = weights > 0
        if np.unique(labels[selected]).size < 2 and metric in ("auroc", "auprc"):
            continue
        values.append(_metric(
            labels[selected], scores[selected], metric,
            sample_weight=weights[selected],
        ))
    if not values:
        return estimate, float("nan"), float("nan")
    alpha = (1. - confidence) / 2.
    return estimate, float(np.quantile(values, alpha)), float(np.quantile(values, 1. - alpha))


def assess_joint_reliability(repeatability, morphology):
    if repeatability == "stable" and morphology == "stable":
        return "calibratable"
    if repeatability == "failed" and morphology == "failed":
        return "not_calibratable"
    return "uncertain"


def freeze_for_audit(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def recompute_window_tokens_and_similarity(branch, windows):
    encoded = branch.token_generator(windows, return_encoded_components=True)
    tokens = encoded["shape_token"] if isinstance(encoded, dict) else encoded
    similarity = branch.shapelet_dictionary.compute_similarity(tokens)
    return tokens, similarity


def _calibration_metrics(calibrator, q, labels, groups, seed):
    scores = calibrator.predict_proba(q)
    row = {}
    for metric in ("auroc", "auprc", "brier", "ece"):
        estimate, low, high = bootstrap_metric_interval(
            labels, scores, groups, metric, seed=seed,
        )
        row[metric] = estimate; row[f"{metric}_ci_low"] = low; row[f"{metric}_ci_high"] = high
    row["sample_count"] = len(labels)
    row["positive_count"] = int(np.asarray(labels).sum())
    row["negative_count"] = int(len(labels) - np.asarray(labels).sum())
    return row


def _evidence_status(rows):
    supported = [row for row in rows if row.get("fit_status") == "fit"]
    if len(supported) != len(rows) or not rows:
        return "unsupported"
    stable = all(
        float(row["slope"]) > 0.
        and float(row["auroc_ci_low"]) > CALIBRATION_PROTOCOL["stable_auroc_lower_bound"]
        for row in supported
    )
    return "stable" if stable else "failed"


def _config_dict(raw):
    return dict(raw if isinstance(raw, dict) else vars(raw))


def _random_view_transform(config):
    from torchvision.transforms import transforms
    from transforms import Identity, Normalize, RandomSamplePixels, RandomSampleTimeSteps, RandomTemporalShift, ToTensor
    shift = (
        RandomTemporalShift(config.max_shift_aug, config.shift_aug_p)
        if bool(getattr(config, "with_shift_aug", False)) else Identity()
    )
    return transforms.Compose([
        RandomSamplePixels(int(config.num_pixels)),
        RandomSampleTimeSteps(int(config.seq_length)), shift, Normalize(), ToTensor(),
    ])


def load_source_runtime(checkpoint, data_root, device, seed=1):
    checkpoint = Path(checkpoint); data_root = Path(data_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing Source checkpoint: {checkpoint}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"missing data root: {data_root}")
    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "config" not in packet or "state_dict" not in packet:
        raise ValueError("Source checkpoint must contain config and state_dict")
    config = SimpleNamespace(**_config_dict(packet["config"])); config.data_root = str(data_root)
    if getattr(config, "shape_representation", None) != "phase_moment":
        raise ValueError("P0 requires original G phase_moment Source checkpoint")
    if int(getattr(config, "shapelet_count", -1)) != 16:
        raise ValueError("P0 requires 16 learned Anchors")
    model = create_model_from_config(config)
    model.load_state_dict(packet["state_dict"], strict=True)
    model = freeze_for_audit(model.to(device))
    cfg, protocol, splits, datasets = build_abc_datasets(config, data_root, seed)
    if not split_ids_are_disjoint(
        datasets["source_train"].get_parcel_indices(), datasets["source_val"].get_parcel_indices(),
    ):
        raise RuntimeError("Source Train/Val parcel leakage")
    transform = _random_view_transform(config)
    datasets["source_train"].transform = transform
    datasets["source_val"].transform = transform
    return model, cfg, protocol, splits, datasets


def _window_metadata(branch):
    starts, scales = [], []
    for index, scale in enumerate(branch.window_extractor.scales):
        indices = getattr(branch.window_extractor, f"indices_{index}")
        starts.extend(indices[:, 0].detach().cpu().tolist())
        scales.extend([int(scale)] * indices.shape[0])
    return np.asarray(starts, dtype=np.int64), np.asarray(scales, dtype=np.int64)


@torch.inference_mode()
def extract_view(model, dataset, domain, split, view_id, seed, device, batch_size, pixel_budget):
    wrapped = DeterministicViewDataset(dataset, domain, seed, view_id)
    target = torch.device(device)
    storage = {key: [] for key in (
        "parcel_index", "label", "similarity", "token_norm", "reverse_similarity",
        "reorder_similarity", "pooling_weight",
    )}
    starts, scales = _window_metadata(model.structure_branch)
    if len(set(scales.tolist())) != 1:
        raise ValueError("P0 morphology perturbation currently requires one fixed window scale")
    for raw in semantic_loader(wrapped, batch_size, pixel_budget, num_workers=0):
        batch = {key: value.to(target) if torch.is_tensor(value) else value for key, value in raw.items()}
        output = model.forward_with_temporal_shift(
            batch["pixels"], batch["valid_pixels"], batch["positions"], batch["extra"],
            temporal_shift=0, return_dict=True,
        )
        groups, _ = model.structure_branch.window_extractor(output["exposed_curve"])
        windows = torch.cat(groups, dim=1)
        recomputed, similarity = recompute_window_tokens_and_similarity(
            model.structure_branch, windows,
        )
        torch.testing.assert_close(recomputed, output["shape_tokens"], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(similarity, output["shapelet_similarity"], rtol=2e-5, atol=2e-6)
        reversed_windows, reordered_windows = perturb_windows(
            windows, deterministic_view_seed(seed, domain, int(view_id), 99),
        )
        reversed_tokens = model.structure_branch.token_generator(
            reversed_windows, return_encoded_components=True,
        )["shape_token"]
        reordered_tokens = model.structure_branch.token_generator(
            reordered_windows, return_encoded_components=True,
        )["shape_token"]
        beta = float(model.structure_branch.shapelet_dictionary.beta)
        values = {
            "parcel_index": batch["parcel_index"], "label": batch["label"],
            "similarity": similarity, "token_norm": recomputed.norm(dim=-1),
            "reverse_similarity": model.structure_branch.shapelet_dictionary.compute_similarity(reversed_tokens),
            "reorder_similarity": model.structure_branch.shapelet_dictionary.compute_similarity(reordered_tokens),
            "pooling_weight": torch.softmax(beta * similarity, dim=1),
        }
        for key, value in values.items():
            storage[key].append(value.detach().cpu())
    arrays = {key: torch.cat(value).numpy() for key, value in storage.items()}
    arrays["domain"] = np.asarray([str(domain)] * len(arrays["parcel_index"]))
    arrays["split"] = np.asarray([str(split)] * len(arrays["parcel_index"]))
    arrays["view_id"] = np.full(len(arrays["parcel_index"]), int(view_id), dtype=np.int64)
    arrays["window_starts"] = starts; arrays["window_scales"] = scales
    return arrays


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "q05": float(np.quantile(values, .05)), "q25": float(np.quantile(values, .25)),
        "q50": float(np.quantile(values, .5)), "q75": float(np.quantile(values, .75)),
        "q95": float(np.quantile(values, .95)),
    }


def matching_distribution_rows(packet, split, view_id, beta):
    similarity = packet["similarity"]; weights = packet["pooling_weight"]
    top_anchor = similarity.argmax(2)
    rows = []
    for anchor in range(similarity.shape[-1]):
        qmax = similarity[:, :, anchor].max(1)
        selected = similarity[:, :, anchor].argmax(1)
        entropy = -(weights[:, :, anchor] * np.log(np.maximum(weights[:, :, anchor], 1e-12))).sum(1)
        counts = np.bincount(selected, minlength=similarity.shape[1])
        row = {"split": split, "view_id": view_id, "anchor": anchor, "sample_count": len(qmax)}
        row.update({f"qmax_{key}": value for key, value in _quantiles(qmax).items()})
        row.update({f"window_entropy_{key}": value for key, value in _quantiles(entropy).items()})
        row["modal_window"] = int(counts.argmax())
        row["modal_window_fraction"] = float(counts.max() / counts.sum())
        row["selected_window_coverage"] = float(np.count_nonzero(counts) / len(counts))
        row["anchor_utilization_fraction"] = float(np.mean(top_anchor == anchor))
        row["shape_token_norm_mean"] = float(np.mean(packet["token_norm"]))
        row["shape_token_norm_std"] = float(np.std(packet["token_norm"]))
        rows.append(row)
    return rows


def _window_repeatability_rows(first, second, split):
    rows = []
    change = np.abs(first["similarity"] - second["similarity"])
    for anchor in range(change.shape[-1]):
        values = change[:, :, anchor].reshape(-1)
        row = {
            "split": split, "level": "window_anchor", "direction": "symmetric",
            "anchor": anchor, "sample_count": len(values),
        }
        row.update({f"absolute_change_{key}": value for key, value in _quantiles(values).items()})
        rows.append(row)
    return rows


def _repeatability_band_rows(rows, split, train_edges):
    output = []
    for anchor in sorted({row["anchor"] for row in rows}):
        for direction in ("view0_to_view1", "view1_to_view0"):
            selected = [row for row in rows if row["anchor"] == anchor and row["direction"] == direction]
            q = np.asarray([row["q"] for row in selected])
            label = np.asarray([row["label"] for row in selected])
            bands = np.digitize(q, train_edges[(anchor, direction)][1:-1], right=True)
            for band, name in enumerate(("low", "medium", "high")):
                current = bands == band
                if not current.any():
                    continue
                output.append({
                    "split": split, "level": "parcel_anchor_q_band",
                    "direction": direction, "anchor": anchor,
                    "similarity_band": name, "sample_count": int(current.sum()),
                    "overlap_repeatability": float(label[current].mean()),
                    "unstable_fraction": float(1. - label[current].mean()),
                    "qmax_mean": float(q[current].mean()),
                })
    return output


def _repeatability_aggregate(rows, starts, scales, train_mode=None, grid_points=64):
    output = []
    anchors = sorted({row["anchor"] for row in rows})
    for anchor in anchors:
        for direction in ("view0_to_view1", "view1_to_view0"):
            selected = [row for row in rows if row["anchor"] == anchor and row["direction"] == direction]
            q = np.asarray([row["q"] for row in selected]); label = np.asarray([row["label"] for row in selected])
            exact = np.asarray([row["exact_match"] for row in selected])
            row = {
                "level": "parcel_anchor",
                "anchor": anchor, "direction": direction, "sample_count": len(selected),
                "overlap_repeatability": float(label.mean()), "exact_repeatability": float(exact.mean()),
                "profile_cosine_mean": float(np.mean([item["profile_cosine"] for item in selected])),
                "profile_cosine_median": float(np.median([item["profile_cosine"] for item in selected])),
                "profile_mean_absolute_change": float(np.mean([
                    item["profile_mean_absolute_change"] for item in selected
                ])),
            }
            row.update({f"qmax_{key}": value for key, value in _quantiles(q).items()})
            if train_mode is not None:
                row["fixed_position_baseline_overlap"] = float(np.mean([
                    circular_window_overlap(
                        starts[train_mode[anchor]], starts[item["target_window"]],
                        scales[train_mode[anchor]], grid_points, scales[item["target_window"]],
                    ) for item in selected
                ]))
            output.append(row)
    return output


def _morphology_summary(split, view, thresholds, train_edges):
    rows = []
    for perturbation, changed in (
        ("reverse", view["reverse_similarity"]), ("block_reorder", view["reorder_similarity"]),
    ):
        labels, delta = morphology_proxy_labels(view["similarity"], changed, thresholds)
        for anchor in range(view["similarity"].shape[-1]):
            q = view["similarity"][:, :, anchor].reshape(-1)
            d = delta[:, :, anchor].reshape(-1); y = labels[:, :, anchor].reshape(-1)
            bands = np.digitize(q, train_edges[anchor][1:-1], right=True)
            for band in range(3):
                selected = bands == band
                if not selected.any():
                    continue
                row = {
                    "split": split, "view_id": int(view["view_id"][0]),
                    "perturbation": perturbation, "anchor": anchor,
                    "similarity_band": ("low", "medium", "high")[band],
                    "sample_count": int(selected.sum()),
                    "normal_change_threshold": float(thresholds[anchor]),
                    "proxy_positive_fraction": float(y[selected].mean()),
                }
                row.update({f"original_q_{key}": value for key, value in _quantiles(q[selected]).items()})
                row.update({f"delta_q_{key}": value for key, value in _quantiles(d[selected]).items()})
                rows.append(row)
    return rows


def _fit_candidate(evidence, anchor, train_q, train_y, train_groups, validations, seed):
    rows = []
    try:
        model = fit_monotonic_calibrator(train_q, train_y, train_groups)
    except ValueError as error:
        for name, q, y, groups in validations:
            rows.append({
                "evidence": evidence, "anchor": anchor, "validation": name,
                "fit_status": "unsupported", "reason": str(error),
                "train_sample_count": len(train_y),
                "train_positive_count": int(np.sum(train_y)),
                "train_negative_count": int(len(train_y) - np.sum(train_y)),
            })
        return rows
    for name, q, y, groups in validations:
        row = {
            "evidence": evidence, "anchor": anchor, "validation": name,
            "fit_status": "fit", **asdict(model),
        }
        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
        if (
            len(y) < CALIBRATION_PROTOCOL["minimum_total"]
            or counts.min() < CALIBRATION_PROTOCOL["minimum_per_class"]
        ):
            row.update({
                "fit_status": "unsupported",
                "reason": "Source Val proxy labels do not meet predeclared support",
                "sample_count": len(y), "positive_count": int(counts[1]),
                "negative_count": int(counts[0]),
            })
            rows.append(row)
            continue
        row.update(_calibration_metrics(model, q, y, groups, seed + anchor))
        rows.append(row)
    return rows


def _calibration_rows(train_views, val_views, thresholds, seed):
    train_left, train_right = pair_views_by_parcel(*train_views)
    val_left, val_right = pair_views_by_parcel(*val_views)
    starts = train_left["window_starts"]; scales = train_left["window_scales"]
    train_repeat = repeatability_rows(
        train_left["similarity"], train_right["similarity"], train_left["parcel_index"],
        starts, scales,
    )
    val_repeat = repeatability_rows(
        val_left["similarity"], val_right["similarity"], val_left["parcel_index"],
        starts, scales,
    )
    rows = []
    for anchor in range(train_left["similarity"].shape[-1]):
        train_direction = [row for row in train_repeat if row["anchor"] == anchor and row["direction"] == "view0_to_view1"]
        validations = []
        for direction in ("view0_to_view1", "view1_to_view0"):
            selected = [row for row in val_repeat if row["anchor"] == anchor and row["direction"] == direction]
            validations.append((
                direction, np.asarray([row["q"] for row in selected]),
                np.asarray([row["label"] for row in selected]),
                np.asarray([row["parcel_index"] for row in selected]),
            ))
        rows.extend(_fit_candidate(
            "repeatability", anchor,
            np.asarray([row["q"] for row in train_direction]),
            np.asarray([row["label"] for row in train_direction]),
            np.asarray([row["parcel_index"] for row in train_direction]),
            validations, seed,
        ))

        for perturbation, key in (("morphology_reverse", "reverse_similarity"), ("morphology_block_reorder", "reorder_similarity")):
            train_q, train_y, train_group = [], [], []
            for packet in train_views:
                labels, _ = morphology_proxy_labels(packet["similarity"], packet[key], thresholds)
                train_q.append(packet["similarity"][:, :, anchor].reshape(-1))
                train_y.append(labels[:, :, anchor].reshape(-1))
                train_group.append(np.repeat(packet["parcel_index"], packet["similarity"].shape[1]))
            val_q, val_y, val_group = [], [], []
            for packet in val_views:
                labels, _ = morphology_proxy_labels(packet["similarity"], packet[key], thresholds)
                val_q.append(packet["similarity"][:, :, anchor].reshape(-1))
                val_y.append(labels[:, :, anchor].reshape(-1))
                val_group.append(np.repeat(packet["parcel_index"], packet["similarity"].shape[1]))
            rows.extend(_fit_candidate(
                perturbation, anchor, np.concatenate(train_q), np.concatenate(train_y),
                np.concatenate(train_group), [("source_val", np.concatenate(val_q), np.concatenate(val_y), np.concatenate(val_group))],
                seed,
            ))
    return rows, train_repeat, val_repeat


def _reliability_rows(calibration_rows):
    output = []
    for anchor in sorted({int(row["anchor"]) for row in calibration_rows}):
        selected = [row for row in calibration_rows if int(row["anchor"]) == anchor]
        repeatability = _evidence_status([row for row in selected if row["evidence"] == "repeatability"])
        reverse = _evidence_status([row for row in selected if row["evidence"] == "morphology_reverse"])
        reorder = _evidence_status([row for row in selected if row["evidence"] == "morphology_block_reorder"])
        morphology = "stable" if reverse == reorder == "stable" else (
            "failed" if reverse == reorder == "failed" else "unsupported"
        )
        output.append({
            "anchor": anchor, "repeatability_status": repeatability,
            "morphology_reverse_status": reverse,
            "morphology_block_reorder_status": reorder,
            "morphology_status": morphology,
            "joint_status": assess_joint_reliability(repeatability, morphology),
        })
    return output


def _plot_calibration(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fitted = [row for row in rows if row.get("fit_status") == "fit"]
    if not fitted:
        return
    fig, axis = plt.subplots(figsize=(8, 4))
    for evidence in sorted({row["evidence"] for row in fitted}):
        selected = [row for row in fitted if row["evidence"] == evidence]
        axis.scatter(
            [int(row["anchor"]) for row in selected],
            [float(row["auroc"]) for row in selected], s=14, label=evidence, alpha=.7,
        )
    axis.axhline(.5, color="black", linewidth=1, linestyle="--")
    axis.set(xlabel="Anchor", ylabel="Source Val AUROC", ylim=(0, 1))
    axis.legend(fontsize=7); fig.tight_layout()
    fig.savefig(Path(output) / "calibration_validation.png", dpi=140)
    plt.close(fig)


def settings_fingerprint(checkpoint_sha, data_root, seed, batch_size, pixel_budget):
    payload = {
        "checkpoint_sha256": checkpoint_sha, "data_root": str(Path(data_root)),
        "seed": int(seed), "batch_size": int(batch_size), "pixel_budget": int(pixel_budget),
        "calibration_protocol": CALIBRATION_PROTOCOL,
        "view_seed_version": VIEW_SEED_VERSION,
        "perturbation_version": PERTURBATION_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_source(args):
    output = Path(args.output_root) / args.source_alias
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256_file(args.checkpoint)
    fingerprint = settings_fingerprint(
        checkpoint_sha, args.data_root, args.seed, args.batch_size, args.pixel_budget,
    )
    if args.settings_fingerprint and args.settings_fingerprint != fingerprint:
        raise ValueError("scheduler/analysis settings fingerprint mismatch")
    model, config, protocol, splits, datasets = load_source_runtime(
        args.checkpoint, args.data_root, args.device, args.seed,
    )
    views = {}
    for split in ("source_train", "source_val"):
        views[split] = tuple(
            extract_view(
                model, datasets[split], config.source, split, view_id, args.seed,
                args.device, args.batch_size, args.pixel_budget,
            ) for view_id in (0, 1)
        )
    train0, train1 = pair_views_by_parcel(*views["source_train"])
    normal_change = np.abs(train0["similarity"] - train1["similarity"])
    thresholds = freeze_morphology_thresholds(normal_change)
    train_edges = [
        np.quantile(train0["similarity"][:, :, anchor].reshape(-1), [0., 1/3, 2/3, 1.])
        for anchor in range(train0["similarity"].shape[-1])
    ]
    matching = []
    morphology = []
    beta = float(model.structure_branch.shapelet_dictionary.beta)
    for split, packets in views.items():
        for view_id, packet in enumerate(packets):
            matching.extend(matching_distribution_rows(packet, split, view_id, beta))
            morphology.extend(_morphology_summary(split, packet, thresholds, train_edges))
    calibration, train_repeat, val_repeat = _calibration_rows(
        views["source_train"], views["source_val"], thresholds, args.seed,
    )
    train_modes = {}
    for anchor in range(train0["similarity"].shape[-1]):
        selected = train0["similarity"][:, :, anchor].argmax(1)
        train_modes[anchor] = int(np.bincount(selected, minlength=train0["similarity"].shape[1]).argmax())
    repeatability = []
    repeat_edges = {}
    for anchor in range(train0["similarity"].shape[-1]):
        for direction in ("view0_to_view1", "view1_to_view0"):
            selected = [
                row for row in train_repeat
                if row["anchor"] == anchor and row["direction"] == direction
            ]
            repeat_edges[(anchor, direction)] = np.quantile(
                [row["q"] for row in selected], [0., 1/3, 2/3, 1.],
            )
    for split, rows in (("source_train", train_repeat), ("source_val", val_repeat)):
        for row in _repeatability_aggregate(
            rows, train0["window_starts"], train0["window_scales"], train_modes,
            model.structure_branch.window_extractor.grid_points,
        ):
            row["split"] = split
            repeatability.append(row)
        first, second = pair_views_by_parcel(*views[split])
        repeatability.extend(_window_repeatability_rows(first, second, split))
        repeatability.extend(_repeatability_band_rows(rows, split, repeat_edges))
    reliability = _reliability_rows(calibration)
    _write_csv(output / "matching_distribution.csv", matching)
    _write_csv(output / "view_repeatability.csv", repeatability)
    _write_csv(output / "morphology_counterfactual.csv", morphology)
    _write_csv(output / "calibration_candidates.csv", calibration)
    _write_csv(output / "anchor_reliability_summary.csv", reliability)
    _plot_calibration(output, calibration)
    counts = {status: sum(row["joint_status"] == status for row in reliability) for status in (
        "calibratable", "uncertain", "not_calibratable",
    )}
    (output / "P0_source_report.md").write_text(
        "\n".join([
            f"# P0 Anchor matching calibration: {args.source_alias}", "",
            f"- Source checkpoint: `{args.checkpoint}`", f"- Source Train parcels: {len(train0['parcel_index'])}",
            f"- Source Val parcels: {len(views['source_val'][0]['parcel_index'])}",
            f"- Joint statuses: `{json.dumps(counts, sort_keys=True)}`", "",
            "Repeatability and morphology sensitivity are separate proxy evidence streams; no unified matching probability is produced.",
            "Morphology proxy labels partly depend on original similarity, so monotonicity does not imply statistical independence or true occurrence probability.",
        ]) + "\n", encoding="utf-8",
    )
    manifest = {
        "status": "complete", "source_alias": args.source_alias,
        "source_domain": config.source, "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_sha256": checkpoint_sha, "settings_fingerprint": fingerprint,
        "classes": list(config.classes), "protocol": protocol,
        "split_parcel_sha256": {
            split: hashlib.sha256(np.asarray(datasets[split].get_parcel_indices(), dtype=np.int64).tobytes()).hexdigest()
            for split in ("source_train", "source_val")
        },
        "sample_counts": {split: len(datasets[split]) for split in ("source_train", "source_val")},
        "calibration_protocol": CALIBRATION_PROTOCOL,
        "morphology_thresholds": thresholds.tolist(),
        "view_seed_version": VIEW_SEED_VERSION, "perturbation_version": PERTURBATION_VERSION,
        "target_data_used": False, "training_performed": False,
    }
    _atomic_json(output / "checkpoint_manifest.json", manifest)
    print(f"P0_SOURCE_COMPLETE|source={args.source_alias}|output={output}", flush=True)


def merge_sources(output_root):
    output_root = Path(output_root)
    rows, missing = [], []
    for source in ("AT1", "FR1", "FR2", "DK1"):
        folder = output_root / source
        manifest = folder / "checkpoint_manifest.json"
        reliability = folder / "anchor_reliability_summary.csv"
        if not manifest.is_file() or not reliability.is_file():
            missing.append(source); continue
        with reliability.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append({"source": source, **row})
    _write_csv(output_root / "P0_4source_comparison.csv", rows)
    counts = {}
    for row in rows:
        counts[row["joint_status"]] = counts.get(row["joint_status"], 0) + 1
    lines = [
        "# P0 Anchor Matching Calibration Audit", "",
        f"Completed Sources: {4 - len(missing)}/4.",
        f"Joint Anchor statuses: `{json.dumps(counts, sort_keys=True)}`.", "",
        "Repeatability and morphology sensitivity were fitted and validated separately using Source-only labels and predeclared criteria.",
        "No unified matching probability was generated and no parameter was written back to training.",
    ]
    if missing:
        lines.extend(["", "## Missing Sources", "", *[f"- {item}" for item in missing]])
    (output_root / "P0_FINAL_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source", "summary"), required=True)
    parser.add_argument("--source-alias")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root", default="experiments/anchor_matching_calibration_seed1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--settings-fingerprint", default="")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "summary":
        merge_sources(args.output_root)
    else:
        for field in ("source_alias", "checkpoint", "data_root"):
            if not getattr(args, field):
                raise ValueError(f"--{field.replace('_', '-')} is required in source mode")
        run_source(args)


if __name__ == "__main__":
    main()
