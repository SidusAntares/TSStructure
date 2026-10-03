#!/usr/bin/env python3
"""Read-only audit of circular temporal alignment assumptions."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import random
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_representation_chain_audit import (
    DOMAINS,
    TASKS,
    _dataset,
    _per_class_f1,
    build_audit_datasets,
    deterministic_loader,
    load_source_model,
    read_csv,
    seed_all,
    write_csv,
)
from analysis.structure_relative_transition_audit import (
    occurrence_weights,
    transition_from_similarity,
)
from timematch import (
    _initialize_timematch_shift,
    _reestimate_timematch_shift,
    get_data_loaders,
)


PERIOD = 365
REPRESENTATIONS = (
    "current",
    "current_plus_phase_raw",
    "current_plus_phase_aligned",
    "current_plus_transition",
    "current_plus_phase_aligned_plus_transition",
)
RAW_DIMS = {
    "current": 32,
    "current_plus_phase_raw": 64,
    "current_plus_phase_aligned": 64,
    "current_plus_transition": 544,
    "current_plus_phase_aligned_plus_transition": 576,
}
WRAP_SUMMARY_FIELDS = (
    "task", "initial_is_shift", "am_shift", "num_observations", "num_wrapped",
    "wrapped_fraction", "num_samples", "num_samples_with_wrap",
    "samples_with_wrap_fraction",
)
WRAP_CLASS_FIELDS = (
    "task", "class", "support", "num_observations", "num_wrapped",
    "wrapped_fraction", "num_samples_with_wrap", "samples_with_wrap_fraction",
)
ENDPOINT_FIELDS = (
    "task", "domain", "split", "class", "endpoint_cosine",
    "start_observation_support", "end_observation_support",
    "start_sample_support", "end_sample_support",
)
SEMANTIC_SUMMARY_FIELDS = (
    "task", "semantics", "macro_f1", "accuracy", "mean_confidence",
    "pseudo_coverage", "accepted_pseudo_accuracy",
)
SEMANTIC_CLASS_FIELDS = (
    "task", "semantics", "class", "support", "f1", "accepted_count",
    "accepted_accuracy",
)
PHASE_SUMMARY_FIELDS = (
    "task", "representation", "raw_dim", "probe_dim",
    "source_val_macro_f1", "target_oracle_macro_f1",
    "source_to_target_macro_f1",
)
PHASE_CLASS_FIELDS = (
    "task", "representation", "class", "source_support", "target_support",
    "source_val_f1", "target_oracle_f1", "source_to_target_f1",
)


def replay_transfer_split(
    source, target, eligible, seed, val_ratio, test_ratio, fold_creator,
):
    random.seed(int(seed))
    np.random.seed(int(seed))
    split = fold_creator(
        [source, target], 1, eligible, val_ratio, test_ratio,
    )[0]
    result = {
        "source_train": split[source]["train"],
        "source_val": split[source]["val"],
        "source_test": split[source]["test"],
        "target_train": split[target]["train"],
        "target_val": split[target]["val"],
        "target_test": split[target]["test"],
        "full_split": split,
    }
    for left, right in (
        ("target_train", "target_val"),
        ("target_train", "target_test"),
        ("target_val", "target_test"),
    ):
        if not set(result[left]).isdisjoint(result[right]):
            raise RuntimeError(f"transfer split overlap: {left} and {right}")
    return result


def estimate_target_to_source_shift(
    model, shift_target_loader, device, source, target, num_classes,
    max_shift=60, sample_size=100,
    initialize_fn=_initialize_timematch_shift,
    reestimate_fn=_reestimate_timematch_shift,
):
    config = SimpleNamespace(
        estimate_shift=True,
        shift_estimator="AM",
        max_temporal_shift=int(max_shift),
        sample_size=int(sample_size),
        progress_bar="off",
        shift_estimation_view="raw",
        source=source,
        target=target,
        num_classes=int(num_classes),
    )
    initial, distribution, diagnostics = initialize_fn(
        model, shift_target_loader, device, config,
    )
    aligned = reestimate_fn(
        model, shift_target_loader, device, config, initial,
        distribution, diagnostics, 0,
    )
    return {"initial_is_shift": int(initial), "am_shift": int(aligned)}


def circular_positions(positions, shift, period=PERIOD):
    return torch.remainder(positions + int(shift), int(period))


def wrap_statistics(positions, labels, class_names, shift, period=PERIOD):
    positions = torch.as_tensor(positions)
    labels = torch.as_tensor(labels, dtype=torch.long)
    linear = positions + int(shift)
    wrapped = (linear < 0) | (linear >= int(period))
    sample_wrapped = wrapped.any(dim=1)
    total = int(wrapped.numel())
    num_wrapped = int(wrapped.sum())
    summary = {
        "num_observations": total,
        "num_wrapped": num_wrapped,
        "wrapped_fraction": num_wrapped / total if total else float("nan"),
        "num_samples": int(len(labels)),
        "num_samples_with_wrap": int(sample_wrapped.sum()),
        "samples_with_wrap_fraction": (
            float(sample_wrapped.float().mean()) if len(labels) else float("nan")
        ),
    }
    rows = []
    for class_id, class_name in enumerate(class_names):
        selected = labels == class_id
        class_wrapped = wrapped[selected]
        class_sample_wrapped = sample_wrapped[selected]
        count = int(class_wrapped.numel())
        rows.append({
            "class": class_name,
            "support": int(selected.sum()),
            "num_observations": count,
            "num_wrapped": int(class_wrapped.sum()),
            "wrapped_fraction": (
                float(class_wrapped.float().mean()) if count else float("nan")
            ),
            "num_samples_with_wrap": int(class_sample_wrapped.sum()),
            "samples_with_wrap_fraction": (
                float(class_sample_wrapped.float().mean())
                if class_sample_wrapped.numel() else float("nan")
            ),
        })
    return summary, rows


@torch.no_grad()
def linear_circular_logits(
    model, spatial, positions, structure, shift, period=PERIOD,
):
    linear = positions + int(shift)
    circular = circular_positions(positions, shift, period)
    linear_instance = model._encode_instance(spatial, linear, structure)
    circular_instance = model._encode_instance(spatial, circular, structure)
    linear_logits = model.decoder(linear_instance)
    circular_logits = model.decoder(circular_instance)
    no_wrap = ((linear >= 0) & (linear < int(period))).all(dim=1)
    if no_wrap.any():
        torch.testing.assert_close(linear[no_wrap], circular[no_wrap])
        torch.testing.assert_close(
            linear_logits[no_wrap], circular_logits[no_wrap],
            atol=1e-6, rtol=1e-6,
        )
    return {
        "linear_positions": linear,
        "circular_positions": circular,
        "linear_logits": linear_logits,
        "circular_logits": circular_logits,
    }


def endpoint_continuity_rows(
    domain, features, positions, labels, class_names,
    boundary_width=45, period=PERIOD, split="val",
):
    features = torch.as_tensor(features, dtype=torch.float32)
    positions = torch.as_tensor(positions)
    labels = torch.as_tensor(labels, dtype=torch.long)
    rows = []
    for class_id, class_name in enumerate(class_names):
        selected = labels == class_id
        class_features = features[selected]
        class_positions = positions[selected]
        start_mask = (class_positions >= 0) & (class_positions < boundary_width)
        end_mask = (
            (class_positions >= period - boundary_width)
            & (class_positions < period)
        )
        start_count = int(start_mask.sum())
        end_count = int(end_mask.sum())
        endpoint = float("nan")
        if start_count and end_count:
            start_mean = class_features[start_mask].mean(0)
            end_mean = class_features[end_mask].mean(0)
            endpoint = float(F.cosine_similarity(
                start_mean[None], end_mean[None], dim=-1,
            ))
        rows.append({
            "domain": domain, "split": split, "class": class_name,
            "endpoint_cosine": endpoint,
            "start_observation_support": start_count,
            "end_observation_support": end_count,
            "start_sample_support": int(start_mask.any(dim=1).sum()),
            "end_sample_support": int(end_mask.any(dim=1).sum()),
        })
    return rows


def circular_moment(occurrence, theta):
    theta = torch.as_tensor(
        theta, dtype=occurrence.dtype, device=occurrence.device,
    )
    if theta.ndim != 1 or theta.shape[0] != occurrence.shape[1]:
        raise ValueError("theta must align with the occurrence time dimension")
    cosine = torch.einsum("bnm,n->bm", occurrence, torch.cos(theta))
    sine = torch.einsum("bnm,n->bm", occurrence, torch.sin(theta))
    return torch.stack((cosine, sine), dim=-1).flatten(1)


def anchor_phase_feature(
    similarity, beta, center_indices, phase_shift=0,
    period=PERIOD, grid_points=64,
):
    occurrence = occurrence_weights(similarity, beta)
    centers = torch.as_tensor(
        center_indices, dtype=similarity.dtype, device=similarity.device,
    )
    center_days = centers * float(period) / int(grid_points)
    aligned_days = torch.remainder(center_days + float(phase_shift), float(period))
    theta = 2. * torch.pi * aligned_days / float(period)
    return circular_moment(occurrence, theta)


def compose_phase_representations(
    current, similarity, beta, center_indices, phase_shift,
    period=PERIOD, grid_points=64,
):
    raw_phase = anchor_phase_feature(
        similarity, beta, center_indices, 0, period, grid_points,
    )
    aligned_phase = anchor_phase_feature(
        similarity, beta, center_indices, phase_shift, period, grid_points,
    )
    first = transition_from_similarity(similarity, beta, 1).flatten(1)
    second = transition_from_similarity(similarity, beta, 2).flatten(1)
    transition = torch.cat((first, second), dim=-1)
    return {
        "current": current,
        "current_plus_phase_raw": torch.cat((current, raw_phase), dim=-1),
        "current_plus_phase_aligned": torch.cat((current, aligned_phase), dim=-1),
        "current_plus_transition": torch.cat((current, transition), dim=-1),
        "current_plus_phase_aligned_plus_transition": torch.cat(
            (current, aligned_phase, transition), dim=-1,
        ),
    }


class PhaseProbePreprocessor:
    def __init__(self, representation, pca_dim=32):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unknown representation: {representation}")
        self.representation = representation
        self.pca_dim = int(pca_dim)
        self.current_scaler = None
        self.phase_scaler = None
        self.transition_pipeline = None

    @property
    def has_phase(self):
        return "phase" in self.representation

    @property
    def has_transition(self):
        return "transition" in self.representation

    def _blocks(self, values):
        values = np.asarray(values, dtype=np.float64)
        cursor = 32
        current = values[:, :cursor]
        phase = values[:, cursor:cursor + 32] if self.has_phase else None
        if self.has_phase:
            cursor += 32
        transition = values[:, cursor:] if self.has_transition else None
        return current, phase, transition

    def fit(self, values):
        current, phase, transition = self._blocks(values)
        self.current_scaler = StandardScaler().fit(current)
        if phase is not None:
            self.phase_scaler = StandardScaler().fit(phase)
        if transition is not None:
            if len(transition) < self.pca_dim:
                raise ValueError("PCA32 requires at least 32 training samples")
            self.transition_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=self.pca_dim, random_state=0)),
            ]).fit(transition)
        return self

    def transform(self, values):
        current, phase, transition = self._blocks(values)
        blocks = [self.current_scaler.transform(current)]
        if phase is not None:
            blocks.append(self.phase_scaler.transform(phase))
        if transition is not None:
            blocks.append(self.transition_pipeline.transform(transition))
        return np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]


def _macro_f1(labels, predictions, class_ids):
    values = _per_class_f1(labels, predictions, class_ids)
    return float(values.mean()) if values.size else float("nan")


def fit_source_phase_probe(
    representation, source_train, source_train_labels,
    source_val, source_val_labels, target_val, target_val_labels, class_ids,
):
    preprocessor = PhaseProbePreprocessor(representation).fit(source_train)
    classifier = RidgeClassifier(alpha=1., class_weight="balanced")
    classifier.fit(preprocessor.transform(source_train), source_train_labels)
    source_predictions = classifier.predict(preprocessor.transform(source_val))
    target_predictions = classifier.predict(preprocessor.transform(target_val))
    return {
        "preprocessor": preprocessor,
        "source_predictions": source_predictions,
        "target_predictions": target_predictions,
        "source_val_macro_f1": _macro_f1(
            source_val_labels, source_predictions, class_ids,
        ),
        "target_macro_f1": _macro_f1(
            target_val_labels, target_predictions, class_ids,
        ),
        "source_per_class_f1": _per_class_f1(
            source_val_labels, source_predictions, class_ids,
        ),
        "target_per_class_f1": _per_class_f1(
            target_val_labels, target_predictions, class_ids,
        ),
    }


def target_oracle_phase_probe(representation, features, labels, class_ids, seed):
    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    counts = Counter(labels.tolist())
    available = np.asarray([
        class_id for class_id in class_ids if counts[int(class_id)] >= 2
    ], dtype=np.int64)
    per_class = np.full(class_ids.shape, np.nan, dtype=np.float64)
    if available.size < 2:
        return {"folds": 0, "macro_f1": float("nan"), "per_class_f1": per_class}
    selected = np.isin(labels, available)
    selected_features, selected_labels = features[selected], labels[selected]
    folds = min(5, min(counts[int(value)] for value in available))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    predictions = np.full(selected_labels.shape, -1, dtype=np.int64)
    for train_indices, val_indices in splitter.split(selected_features, selected_labels):
        preprocessor = PhaseProbePreprocessor(representation).fit(
            selected_features[train_indices]
        )
        classifier = RidgeClassifier(alpha=1., class_weight="balanced")
        classifier.fit(
            preprocessor.transform(selected_features[train_indices]),
            selected_labels[train_indices],
        )
        predictions[val_indices] = classifier.predict(
            preprocessor.transform(selected_features[val_indices])
        )
    available_f1 = _per_class_f1(selected_labels, predictions, available)
    locations = {int(value): index for index, value in enumerate(class_ids)}
    for class_id, value in zip(available, available_f1):
        per_class[locations[int(class_id)]] = value
    return {
        "folds": int(folds), "macro_f1": float(available_f1.mean()),
        "per_class_f1": per_class,
    }


class EndpointAccumulator:
    def __init__(self, domain, split, class_names, feature_dim):
        self.domain, self.split = domain, split
        self.class_names = list(class_names)
        shape = (len(class_names), int(feature_dim))
        self.start_sum = torch.zeros(shape)
        self.end_sum = torch.zeros(shape)
        self.start_count = torch.zeros(len(class_names), dtype=torch.long)
        self.end_count = torch.zeros(len(class_names), dtype=torch.long)
        self.start_samples = torch.zeros(len(class_names), dtype=torch.long)
        self.end_samples = torch.zeros(len(class_names), dtype=torch.long)

    def update(self, features, positions, labels):
        features, positions, labels = features.cpu(), positions.cpu(), labels.cpu()
        for class_id in range(len(self.class_names)):
            selected = labels == class_id
            class_features, class_positions = features[selected], positions[selected]
            start = (class_positions >= 0) & (class_positions < 45)
            end = (class_positions >= 320) & (class_positions < 365)
            if start.any():
                self.start_sum[class_id] += class_features[start].sum(0)
            if end.any():
                self.end_sum[class_id] += class_features[end].sum(0)
            self.start_count[class_id] += start.sum()
            self.end_count[class_id] += end.sum()
            self.start_samples[class_id] += start.any(1).sum()
            self.end_samples[class_id] += end.any(1).sum()

    def rows(self):
        rows = []
        for class_id, class_name in enumerate(self.class_names):
            start_count, end_count = self.start_count[class_id], self.end_count[class_id]
            cosine = float("nan")
            if start_count and end_count:
                cosine = float(F.cosine_similarity(
                    (self.start_sum[class_id] / start_count)[None],
                    (self.end_sum[class_id] / end_count)[None], dim=-1,
                ))
            rows.append({
                "domain": self.domain, "split": self.split, "class": class_name,
                "endpoint_cosine": cosine,
                "start_observation_support": int(start_count),
                "end_observation_support": int(end_count),
                "start_sample_support": int(self.start_samples[class_id]),
                "end_sample_support": int(self.end_samples[class_id]),
            })
        return rows


@torch.no_grad()
def extract_dataset(
    model, dataset, batch_size, num_workers, device, pixel_budget,
    phase_shift, collect_semantics=False, endpoint_domain=None,
    endpoint_split="val", class_names=None,
):
    representations = {name: [] for name in REPRESENTATIONS}
    positions_all, labels_all = [], []
    semantic_logits = {"linear": [], "circular": []}
    endpoint = None
    centers = None
    beta = float(model.structure_branch.shapelet_dictionary.beta)
    state_before = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    for batch in deterministic_loader(
        dataset, batch_size, num_workers, pixel_budget=pixel_budget,
    ):
        pixels = batch["pixels"].to(device, non_blocking=True)
        valid = batch["valid_pixels"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        extra = batch["extra"].to(device, non_blocking=True)
        labels = batch["label"].long()
        spatial = model.spatial_encoder(pixels, valid, extra)
        structure = model.prepare_structure(spatial, positions)
        similarity = structure["shapelet_similarity"]
        current = structure["shapelet_response"]
        if similarity.shape[1:] != (8, 16) or current.shape[-1] != 32:
            raise RuntimeError("audit requires N=8, M=16 and current response32")
        if centers is None:
            _, _, centers = model.structure_branch.window_extractor(
                structure["exposed_curve"], return_centers=True,
            )
            if tuple(centers.shape) != (8,):
                raise RuntimeError(f"expected eight real window centers, got {centers.shape}")
        composed = compose_phase_representations(
            current, similarity, beta, centers, phase_shift, PERIOD, 64,
        )
        for name, values in composed.items():
            representations[name].append(values.float().cpu())
        positions_all.append(positions.cpu())
        labels_all.append(labels)
        if collect_semantics:
            logits = linear_circular_logits(
                model, spatial, positions, structure, phase_shift, PERIOD,
            )
            semantic_logits["linear"].append(logits["linear_logits"].cpu())
            semantic_logits["circular"].append(logits["circular_logits"].cpu())
        if endpoint_domain is not None:
            if endpoint is None:
                if class_names is None:
                    raise ValueError("class_names are required for endpoint auditing")
                endpoint = EndpointAccumulator(
                    endpoint_domain, endpoint_split, class_names, spatial.shape[-1],
                )
            endpoint.update(spatial, positions, labels)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), state_before[name])
    return {
        "representations": {
            name: torch.cat(values).numpy() for name, values in representations.items()
        },
        "positions": torch.cat(positions_all),
        "labels": torch.cat(labels_all).numpy(),
        "semantic_logits": {
            name: torch.cat(values).numpy() if values else None
            for name, values in semantic_logits.items()
        },
        "endpoint_rows": endpoint.rows() if endpoint is not None else [],
    }


def classification_metrics(task, logits_by_semantics, labels, class_names, threshold=.9):
    labels = np.asarray(labels, dtype=np.int64)
    class_ids = np.arange(len(class_names), dtype=np.int64)
    summary, per_class = [], []
    for semantics, logits in logits_by_semantics.items():
        probabilities = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
        prediction = probabilities.argmax(1)
        confidence = probabilities.max(1)
        accepted = confidence > float(threshold)
        class_f1 = _per_class_f1(labels, prediction, class_ids)
        summary.append({
            "task": task, "semantics": semantics,
            "macro_f1": float(class_f1.mean()),
            "accuracy": float((prediction == labels).mean()),
            "mean_confidence": float(confidence.mean()),
            "pseudo_coverage": float(accepted.mean()),
            "accepted_pseudo_accuracy": (
                float((prediction[accepted] == labels[accepted]).mean())
                if accepted.any() else float("nan")
            ),
        })
        for class_id, class_name in enumerate(class_names):
            selected = labels == class_id
            class_accepted = selected & accepted
            per_class.append({
                "task": task, "semantics": semantics, "class": class_name,
                "support": int(selected.sum()), "f1": float(class_f1[class_id]),
                "accepted_count": int(class_accepted.sum()),
                "accepted_accuracy": (
                    float((prediction[class_accepted] == labels[class_accepted]).mean())
                    if class_accepted.any() else float("nan")
                ),
            })
    return summary, per_class


def evaluate_phase_probes(task, class_names, source_train, source_val, target_val, seed):
    class_ids = np.arange(len(class_names), dtype=np.int64)
    source_support = np.bincount(source_val["labels"], minlength=len(class_names))
    target_support = np.bincount(target_val["labels"], minlength=len(class_names))
    summary, per_class = [], []
    for representation in REPRESENTATIONS:
        source = fit_source_phase_probe(
            representation,
            source_train["representations"][representation], source_train["labels"],
            source_val["representations"][representation], source_val["labels"],
            target_val["representations"][representation], target_val["labels"],
            class_ids,
        )
        oracle = target_oracle_phase_probe(
            representation, target_val["representations"][representation],
            target_val["labels"], class_ids, seed,
        )
        probe_dim = {"current": 32, "current_plus_phase_raw": 64,
                     "current_plus_phase_aligned": 64,
                     "current_plus_transition": 64,
                     "current_plus_phase_aligned_plus_transition": 96}[representation]
        summary.append({
            "task": task, "representation": representation,
            "raw_dim": RAW_DIMS[representation], "probe_dim": probe_dim,
            "source_val_macro_f1": source["source_val_macro_f1"],
            "target_oracle_macro_f1": oracle["macro_f1"],
            "source_to_target_macro_f1": source["target_macro_f1"],
        })
        for class_id, class_name in enumerate(class_names):
            per_class.append({
                "task": task, "representation": representation,
                "class": class_name,
                "source_support": int(source_support[class_id]),
                "target_support": int(target_support[class_id]),
                "source_val_f1": source["source_per_class_f1"][class_id],
                "target_oracle_f1": oracle["per_class_f1"][class_id],
                "source_to_target_f1": source["target_per_class_f1"][class_id],
            })
    return summary, per_class


def build_datasets_and_shift_loader(config, source_name, target_name, data_root, seed):
    from train import create_train_val_test_folds

    audit_datasets, audit_split = build_audit_datasets(
        config, source_name, target_name, data_root, seed,
    )
    bare_source = _dataset(data_root, source_name, config)
    bare_target = _dataset(data_root, target_name, config)
    eligible = {
        source_name: bare_source.get_parcel_indices().tolist(),
        target_name: bare_target.get_parcel_indices().tolist(),
    }
    transfer = replay_transfer_split(
        source_name, target_name, eligible, seed,
        config.val_ratio, config.test_ratio, create_train_val_test_folds,
    )
    if set(transfer["target_val"]) != set(audit_split["target_val"]):
        raise RuntimeError("target val split replay diverged")
    loader_config = SimpleNamespace(**vars(config))
    loader_config.data_root = data_root
    loader_config.source = source_name
    loader_config.target = target_name
    loader_config.batch_size = int(config.batch_size)
    loader_config.num_workers = int(config.num_workers)
    loader_config.max_shift_aug = getattr(config, "max_shift_aug", 60)
    loader_config.shift_aug_p = getattr(config, "shift_aug_p", 1.)
    loader_config.with_shift_aug = False
    _, shift_target_loader, _ = get_data_loaders(
        transfer["full_split"], loader_config, balance_source=False,
    )
    actual_shift_indices = set(
        shift_target_loader.dataset.get_parcel_indices().tolist()
    )
    if actual_shift_indices != set(transfer["target_train"]):
        raise RuntimeError("shift loader does not use the replayed target-train split")
    return audit_datasets, transfer, shift_target_loader


def _task_paths(root, task):
    task_root = Path(root) / "tmp" / task
    return task_root, {
        "wrap_summary": task_root / "shift_wrap_summary.csv",
        "wrap_class": task_root / "shift_wrap_per_class.csv",
        "endpoint": task_root / "endpoint_continuity.csv",
        "semantic_summary": task_root / "shift_semantics_summary.csv",
        "semantic_class": task_root / "shift_semantics_per_class.csv",
        "phase_summary": task_root / "phase_representation_summary.csv",
        "phase_class": task_root / "phase_representation_per_class.csv",
    }


def run_task(args):
    source_alias, target_alias = TASKS[args.task]
    source_name, target_name = DOMAINS[source_alias], DOMAINS[target_alias]
    checkpoint = (
        Path(args.checkpoint_root) / f"source_{source_alias}_seed1" / "fold_0" / "model.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {checkpoint.resolve()}")
    seed_all(args.seed)
    device = torch.device(args.device)
    model, config = load_source_model(checkpoint, device, "current")
    config.num_workers = args.num_workers
    audit_datasets, transfer, shift_target_loader = build_datasets_and_shift_loader(
        config, source_name, target_name, args.data_root, args.seed,
    )
    shift = estimate_target_to_source_shift(
        model, shift_target_loader, device, source_name, target_name,
        len(config.classes), max_shift=60, sample_size=args.sample_size,
    )
    source_train = extract_dataset(
        model, audit_datasets["source_train"], args.batch_size, args.num_workers,
        device, args.pixel_budget, phase_shift=0,
    )
    source_val = extract_dataset(
        model, audit_datasets["source_val"], args.batch_size, args.num_workers,
        device, args.pixel_budget, phase_shift=0, endpoint_domain=source_alias,
        endpoint_split="source_val", class_names=config.classes,
    )
    target_val = extract_dataset(
        model, audit_datasets["target_val"], args.batch_size, args.num_workers,
        device, args.pixel_budget, phase_shift=shift["am_shift"],
        collect_semantics=True, endpoint_domain=target_alias,
        endpoint_split="target_val", class_names=config.classes,
    )
    wrap_summary, wrap_class = wrap_statistics(
        target_val["positions"], target_val["labels"], config.classes,
        shift["am_shift"], PERIOD,
    )
    wrap_summary = {"task": args.task, **shift, **wrap_summary}
    wrap_class = [{"task": args.task, **row} for row in wrap_class]
    endpoint = [
        {"task": args.task, **row}
        for row in source_val["endpoint_rows"] + target_val["endpoint_rows"]
    ]
    semantic_summary, semantic_class = classification_metrics(
        args.task, target_val["semantic_logits"], target_val["labels"], config.classes,
    )
    phase_summary, phase_class = evaluate_phase_probes(
        args.task, config.classes, source_train, source_val, target_val, args.seed,
    )
    task_root, paths = _task_paths(args.output_root, args.task)
    write_csv(paths["wrap_summary"], [wrap_summary], WRAP_SUMMARY_FIELDS)
    write_csv(paths["wrap_class"], wrap_class, WRAP_CLASS_FIELDS)
    write_csv(paths["endpoint"], endpoint, ENDPOINT_FIELDS)
    write_csv(paths["semantic_summary"], semantic_summary, SEMANTIC_SUMMARY_FIELDS)
    write_csv(paths["semantic_class"], semantic_class, SEMANTIC_CLASS_FIELDS)
    write_csv(paths["phase_summary"], phase_summary, PHASE_SUMMARY_FIELDS)
    write_csv(paths["phase_class"], phase_class, PHASE_CLASS_FIELDS)
    (task_root / "split_audit.txt").write_text(
        "\n".join((
            f"target_train={len(transfer['target_train'])}",
            f"target_val={len(transfer['target_val'])}",
            "target_test_dataset_instantiated=false",
            "target_train_gt_used_for_shift=false",
        )) + "\n", encoding="utf-8",
    )
    print(f"CIRCULAR_ALIGNMENT_FINISHED|task={args.task}|output={task_root}")


def merge_outputs(args):
    root = Path(args.output_root)
    specifications = (
        ("shift_wrap_summary.csv", WRAP_SUMMARY_FIELDS),
        ("shift_wrap_per_class.csv", WRAP_CLASS_FIELDS),
        ("endpoint_continuity.csv", ENDPOINT_FIELDS),
        ("shift_semantics_summary.csv", SEMANTIC_SUMMARY_FIELDS),
        ("shift_semantics_per_class.csv", SEMANTIC_CLASS_FIELDS),
        ("phase_representation_summary.csv", PHASE_SUMMARY_FIELDS),
        ("phase_representation_per_class.csv", PHASE_CLASS_FIELDS),
    )
    merged = {}
    for filename, fields in specifications:
        rows = []
        for task in TASKS:
            path = root / "tmp" / task / filename
            if not path.is_file():
                raise FileNotFoundError(f"incomplete audit output: {path}")
            rows.extend(read_csv(path))
        write_csv(root / filename, rows, fields)
        merged[filename] = rows
    lines = ["STRUCTURE_CIRCULAR_ALIGNMENT_VALIDITY_AUDIT"]
    for filename in (
        "shift_wrap_summary.csv", "shift_semantics_summary.csv",
        "phase_representation_summary.csv",
    ):
        for row in merged[filename]:
            lines.append(filename + "|" + "|".join(
                f"{key}={value}" for key, value in row.items()
            ))
    (root / "audit_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    print(f"CIRCULAR_ALIGNMENT_MERGED|output={root}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument(
        "--checkpoint-root",
        default="outputs/structure_proto_v2clean_4tasks_seed1/source",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/structure_circular_alignment_validity_audit_seed1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main():
    args = build_parser().parse_args()
    if args.merge_only:
        merge_outputs(args)
    elif args.task:
        run_task(args)
    else:
        raise SystemExit("--task or --merge-only is required")


if __name__ == "__main__":
    main()
