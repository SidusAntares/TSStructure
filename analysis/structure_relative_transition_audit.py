#!/usr/bin/env python3
"""Read-only audit of relative transitions between shapelet occurrences."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
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
    _per_class_f1,
    build_audit_datasets,
    deterministic_loader,
    effective_rank,
    load_source_model,
    read_csv,
    seed_all,
    write_csv,
)
from models.structure_da.discriminative_structure import (
    normalized_candidate_concentration,
)


REPRESENTATIONS = (
    "current",
    "transition_lag1",
    "transition_lag12",
    "current_plus_lag1",
    "current_plus_lag12",
)
RAW_DIMS = {
    "current": 32,
    "transition_lag1": 256,
    "transition_lag12": 512,
    "current_plus_lag1": 288,
    "current_plus_lag12": 544,
}
SUMMARY_FIELDS = (
    "task", "representation", "raw_dim", "probe_dim", "effective_rank",
    "source_val_macro_f1", "target_oracle_macro_f1",
    "source_to_target_macro_f1",
)
PER_CLASS_FIELDS = (
    "task", "representation", "class", "source_support", "target_support",
    "source_val_f1", "target_oracle_f1", "source_to_target_f1",
)
INVARIANCE_FIELDS = (
    "task", "representation", "shift_cos", "permutation_cos", "reverse_cos",
    "order_gap", "reverse_gap",
)
STATISTIC_FIELDS = (
    "task", "lag12_cos_mean", "lag12_cos_std",
    "T1_effective_rank", "T12_effective_rank",
)


def occurrence_weights(similarity, beta):
    if similarity.ndim != 3:
        raise ValueError("similarity must have shape [B,N,M]")
    return torch.softmax(float(beta) * similarity, dim=1)


def relative_anchor_transition(occurrence, lag, eps=1e-12):
    if occurrence.ndim != 3:
        raise ValueError("occurrence must have shape [B,N,M]")
    shifted = torch.roll(occurrence, shifts=-int(lag), dims=1)
    cooccurrence = torch.einsum("bni,bnj->bij", occurrence, shifted)
    return cooccurrence / cooccurrence.sum(dim=-1, keepdim=True).clamp_min(eps)


def transition_from_similarity(similarity, beta, lag):
    return relative_anchor_transition(occurrence_weights(similarity, beta), lag)


def current_from_similarity(similarity, beta, eps=1e-12):
    weights = occurrence_weights(similarity, beta)
    strength = (weights * similarity).sum(dim=1)
    mask = torch.ones(
        similarity.shape[:2], dtype=torch.bool, device=similarity.device,
    )
    concentration = normalized_candidate_concentration(
        weights, candidate_mask=mask, eps=eps,
    )
    return torch.cat((strength, concentration), dim=-1)


def compose_transition_representations(current, similarity, beta):
    if current.ndim != 2 or similarity.ndim != 3:
        raise ValueError("current must be [B,2M] and similarity must be [B,N,M]")
    if current.shape[0] != similarity.shape[0]:
        raise ValueError("current and similarity batch dimensions must match")
    first = transition_from_similarity(similarity, beta, lag=1).flatten(1)
    second = transition_from_similarity(similarity, beta, lag=2).flatten(1)
    lag12 = torch.cat((first, second), dim=-1)
    return {
        "current": current,
        "transition_lag1": first,
        "transition_lag12": lag12,
        "current_plus_lag1": torch.cat((current, first), dim=-1),
        "current_plus_lag12": torch.cat((current, lag12), dim=-1),
    }


class ProbePreprocessor:
    """Fit current and transition blocks independently on one training partition."""

    def __init__(self, representation, pca_dim=32):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unknown representation: {representation}")
        self.representation = representation
        self.pca_dim = int(pca_dim)
        self.current_scaler = None
        self.transition_pipeline = None

    @property
    def has_current(self):
        return self.representation == "current" or self.representation.startswith(
            "current_plus_"
        )

    @property
    def has_transition(self):
        return self.representation != "current"

    def _blocks(self, values):
        values = np.asarray(values, dtype=np.float64)
        if self.has_current and self.has_transition:
            return values[:, :32], values[:, 32:]
        if self.has_current:
            return values, None
        return None, values

    def fit(self, values):
        current, transition = self._blocks(values)
        if current is not None:
            self.current_scaler = StandardScaler().fit(current)
        if transition is not None:
            if len(transition) < self.pca_dim:
                raise ValueError(
                    f"PCA32 requires at least {self.pca_dim} training samples; "
                    f"received {len(transition)}"
                )
            self.transition_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=self.pca_dim, random_state=0)),
            ]).fit(transition)
        return self

    def transform(self, values):
        current, transition = self._blocks(values)
        blocks = []
        if current is not None:
            blocks.append(self.current_scaler.transform(current))
        if transition is not None:
            blocks.append(self.transition_pipeline.transform(transition))
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)


def _macro_f1(labels, prediction, class_ids):
    values = _per_class_f1(labels, prediction, class_ids)
    return float(values.mean()) if values.size else float("nan")


def fit_source_transition_probe(
    representation, source_train, source_train_labels,
    source_val, source_val_labels, target, target_labels, class_ids,
):
    preprocessor = ProbePreprocessor(representation).fit(source_train)
    source_train_probe = preprocessor.transform(source_train)
    estimator = RidgeClassifier(alpha=1., class_weight="balanced")
    estimator.fit(source_train_probe, source_train_labels)
    source_prediction = estimator.predict(preprocessor.transform(source_val))
    target_prediction = estimator.predict(preprocessor.transform(target))
    return {
        "preprocessor": preprocessor,
        "estimator": estimator,
        "source_val_predictions": source_prediction,
        "target_predictions": target_prediction,
        "source_val_macro_f1": _macro_f1(
            source_val_labels, source_prediction, class_ids,
        ),
        "source_val_per_class_f1": _per_class_f1(
            source_val_labels, source_prediction, class_ids,
        ),
        "source_to_target_macro_f1": _macro_f1(
            target_labels, target_prediction, class_ids,
        ),
        "source_to_target_per_class_f1": _per_class_f1(
            target_labels, target_prediction, class_ids,
        ),
    }


def target_oracle_transition_probe(
    representation, features, labels, class_ids, seed,
):
    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    counts = Counter(labels.tolist())
    available = np.asarray([
        class_id for class_id in class_ids if counts[int(class_id)] >= 2
    ], dtype=np.int64)
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    per_class = np.full(class_ids.shape, np.nan, dtype=np.float64)
    if available.size < 2:
        return {
            "folds": 0, "predictions": predictions,
            "macro_f1": float("nan"), "per_class_f1": per_class,
        }
    selected = np.isin(labels, available)
    selected_features = features[selected]
    selected_labels = labels[selected]
    folds = min(5, min(counts[int(class_id)] for class_id in available))
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=int(seed),
    )
    selected_prediction = np.full(selected_labels.shape, -1, dtype=np.int64)
    for train_indices, val_indices in splitter.split(
        selected_features, selected_labels,
    ):
        preprocessor = ProbePreprocessor(representation).fit(
            selected_features[train_indices]
        )
        estimator = RidgeClassifier(alpha=1., class_weight="balanced")
        estimator.fit(
            preprocessor.transform(selected_features[train_indices]),
            selected_labels[train_indices],
        )
        selected_prediction[val_indices] = estimator.predict(
            preprocessor.transform(selected_features[val_indices])
        )
    predictions[selected] = selected_prediction
    available_f1 = _per_class_f1(
        selected_labels, selected_prediction, available,
    )
    locations = {int(value): index for index, value in enumerate(class_ids)}
    for class_id, value in zip(available, available_f1):
        per_class[locations[int(class_id)]] = value
    return {
        "folds": int(folds), "predictions": predictions,
        "macro_f1": float(available_f1.mean()), "per_class_f1": per_class,
    }


@torch.no_grad()
def extract_transition_batch(model, pixels, valid_pixels, positions, extra):
    spatial = model.spatial_encoder(pixels, valid_pixels, extra)
    structure = model.prepare_structure(spatial, positions)
    similarity = structure["shapelet_similarity"]
    current = structure["shapelet_response"]
    beta = float(model.structure_branch.shapelet_dictionary.beta)
    return {
        "current": current.detach(),
        "similarity": similarity.detach(),
        "representations": {
            key: value.detach() for key, value in compose_transition_representations(
                current, similarity, beta,
            ).items()
        },
        "beta": beta,
    }


@torch.no_grad()
def extract_dataset(model, dataset, batch_size, num_workers, device, pixel_budget):
    representation_values = {name: [] for name in REPRESENTATIONS}
    similarities, currents, labels = [], [], []
    beta = None
    for batch in deterministic_loader(
        dataset, batch_size, num_workers, pixel_budget=pixel_budget,
    ):
        extracted = extract_transition_batch(
            model,
            batch["pixels"].to(device, non_blocking=True),
            batch["valid_pixels"].to(device, non_blocking=True),
            batch["positions"].to(device, non_blocking=True),
            batch["extra"].to(device, non_blocking=True),
        )
        beta = extracted["beta"]
        for name, values in extracted["representations"].items():
            representation_values[name].append(values.float().cpu())
        similarities.append(extracted["similarity"].float().cpu())
        currents.append(extracted["current"].float().cpu())
        labels.append(batch["label"].long().cpu())
    result = {
        "representations": {
            name: torch.cat(values).numpy()
            for name, values in representation_values.items()
        },
        "similarity": torch.cat(similarities),
        "current": torch.cat(currents),
        "labels": torch.cat(labels).numpy(),
        "beta": beta,
    }
    if result["similarity"].shape[1:] != (8, 16):
        raise RuntimeError(
            f"expected current Q24/stride8 trajectory [N=8,M=16], got "
            f"{tuple(result['similarity'].shape[1:])}"
        )
    reconstructed = current_from_similarity(result["similarity"], beta)
    torch.testing.assert_close(
        reconstructed, result["current"], atol=2e-5, rtol=2e-5,
    )
    return result


def _mean_cosine(left, right):
    left = F.normalize(left.flatten(1).float(), dim=-1)
    right = F.normalize(right.flatten(1).float(), dim=-1)
    return float((left * right).sum(-1).mean())


def invariance_rows(task, similarity, beta, seed):
    similarity = torch.as_tensor(similarity, dtype=torch.float32)
    generator = np.random.default_rng(int(seed))
    permutation = torch.as_tensor(
        generator.permutation(similarity.shape[1]), dtype=torch.long,
    )
    transformed = {
        "shift": torch.roll(similarity, shifts=3, dims=1),
        "permutation": similarity[:, permutation],
        "reverse": torch.flip(similarity, dims=(1,)),
    }

    def representations(values):
        first = transition_from_similarity(values, beta, 1).flatten(1)
        second = transition_from_similarity(values, beta, 2).flatten(1)
        return {
            "current": current_from_similarity(values, beta),
            "transition_lag1": first,
            "transition_lag12": torch.cat((first, second), dim=-1),
        }

    original = representations(similarity)
    alternatives = {name: representations(values) for name, values in transformed.items()}
    rows = []
    for name, values in original.items():
        shift_cos = _mean_cosine(values, alternatives["shift"][name])
        permutation_cos = _mean_cosine(
            values, alternatives["permutation"][name],
        )
        reverse_cos = _mean_cosine(values, alternatives["reverse"][name])
        rows.append({
            "task": task, "representation": name,
            "shift_cos": shift_cos,
            "permutation_cos": permutation_cos,
            "reverse_cos": reverse_cos,
            "order_gap": shift_cos - permutation_cos,
            "reverse_gap": shift_cos - reverse_cos,
        })
    return rows


def evaluate_task(task, class_names, source_train, source_val, target_val, seed):
    class_ids = np.arange(len(class_names), dtype=np.int64)
    source_support = np.bincount(
        source_val["labels"], minlength=len(class_names),
    )
    target_support = np.bincount(
        target_val["labels"], minlength=len(class_names),
    )
    summary_rows, class_rows = [], []
    for representation in REPRESENTATIONS:
        source = fit_source_transition_probe(
            representation,
            source_train["representations"][representation], source_train["labels"],
            source_val["representations"][representation], source_val["labels"],
            target_val["representations"][representation], target_val["labels"],
            class_ids,
        )
        oracle = target_oracle_transition_probe(
            representation, target_val["representations"][representation],
            target_val["labels"], class_ids, seed,
        )
        raw_rank_features = np.concatenate((
            source_val["representations"][representation],
            target_val["representations"][representation],
        ))
        probe_dim = 64 if representation.startswith("current_plus_") else 32
        summary_rows.append({
            "task": task, "representation": representation,
            "raw_dim": int(raw_rank_features.shape[1]),
            "probe_dim": probe_dim,
            "effective_rank": effective_rank(raw_rank_features),
            "source_val_macro_f1": source["source_val_macro_f1"],
            "target_oracle_macro_f1": oracle["macro_f1"],
            "source_to_target_macro_f1": source["source_to_target_macro_f1"],
        })
        for class_id, class_name in enumerate(class_names):
            class_rows.append({
                "task": task, "representation": representation,
                "class": class_name,
                "source_support": int(source_support[class_id]),
                "target_support": int(target_support[class_id]),
                "source_val_f1": source["source_val_per_class_f1"][class_id],
                "target_oracle_f1": oracle["per_class_f1"][class_id],
                "source_to_target_f1": source["source_to_target_per_class_f1"][class_id],
            })
    similarity = target_val["similarity"]
    first = transition_from_similarity(similarity, target_val["beta"], 1).flatten(1)
    second = transition_from_similarity(similarity, target_val["beta"], 2).flatten(1)
    lag_cosine = F.cosine_similarity(first, second, dim=-1)
    statistics = [{
        "task": task,
        "lag12_cos_mean": float(lag_cosine.mean()),
        "lag12_cos_std": float(lag_cosine.std(unbiased=False)),
        "T1_effective_rank": effective_rank(first),
        "T12_effective_rank": effective_rank(torch.cat((first, second), dim=-1)),
    }]
    return (
        summary_rows,
        class_rows,
        invariance_rows(task, similarity, target_val["beta"], seed),
        statistics,
    )


def checkpoint_for(root, source):
    return Path(root) / f"source_{source}_seed1" / "fold_0" / "model.pt"


def _delta_lines(summary_rows):
    indexed = {
        row["representation"]: float(row["source_to_target_macro_f1"])
        for row in summary_rows
    }
    lag1 = indexed["current_plus_lag1"] - max(
        indexed["current"], indexed["transition_lag1"],
    )
    lag12 = indexed["current_plus_lag12"] - max(
        indexed["current"], indexed["transition_lag12"],
    )
    task = summary_rows[0]["task"]
    return [
        f"task={task}|delta_comp_lag1={lag1}",
        f"task={task}|delta_comp_lag12={lag12}",
    ]


def run_task(args):
    source_alias, target_alias = TASKS[args.task]
    checkpoint = checkpoint_for(args.checkpoint_root, source_alias)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {checkpoint.resolve()}")
    seed_all(args.seed)
    device = torch.device(args.device)
    model, config = load_source_model(checkpoint, device, "current")
    if (
        tuple(config.shape_window_scales) != (24,)
        or int(config.shape_window_stride) != 8
        or int(config.shapelet_count) != 16
        or float(config.shapelet_beta) != 5.
    ):
        raise RuntimeError("audit requires current Q24/stride8/M16/beta5 source model")
    datasets, _ = build_audit_datasets(
        config, DOMAINS[source_alias], DOMAINS[target_alias],
        args.data_root, args.seed,
    )
    extracted = {
        name: extract_dataset(
            model, dataset, args.batch_size, args.num_workers, device,
            args.pixel_budget,
        )
        for name, dataset in datasets.items()
    }
    rows = evaluate_task(
        args.task, list(config.classes), extracted["source_train"],
        extracted["source_val"], extracted["target_val"], args.seed,
    )
    task_root = Path(args.output_root) / "tasks" / args.task
    write_csv(task_root / "representation_summary.csv", rows[0], SUMMARY_FIELDS)
    write_csv(task_root / "representation_per_class.csv", rows[1], PER_CLASS_FIELDS)
    write_csv(task_root / "invariance_summary.csv", rows[2], INVARIANCE_FIELDS)
    write_csv(task_root / "transition_statistics.csv", rows[3], STATISTIC_FIELDS)
    (task_root / "audit_summary.txt").write_text(
        "\n".join(_delta_lines(rows[0])) + "\n", encoding="utf-8",
    )
    print(
        f"RELATIVE_TRANSITION_FINISHED|task={args.task}|output={task_root}"
    )


def merge_outputs(args):
    root = Path(args.output_root)
    summary_rows, class_rows, invariance, statistics = [], [], [], []
    delta_lines = []
    for task in TASKS:
        task_root = root / "tasks" / task
        required = (
            task_root / "representation_summary.csv",
            task_root / "representation_per_class.csv",
            task_root / "invariance_summary.csv",
            task_root / "transition_statistics.csv",
            task_root / "audit_summary.txt",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("incomplete transition audit: " + ", ".join(missing))
        summary_rows.extend(read_csv(required[0]))
        class_rows.extend(read_csv(required[1]))
        invariance.extend(read_csv(required[2]))
        statistics.extend(read_csv(required[3]))
        delta_lines.extend(required[4].read_text(encoding="utf-8").splitlines())
    write_csv(root / "representation_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(root / "representation_per_class.csv", class_rows, PER_CLASS_FIELDS)
    write_csv(root / "invariance_summary.csv", invariance, INVARIANCE_FIELDS)
    write_csv(root / "transition_statistics.csv", statistics, STATISTIC_FIELDS)
    lines = ["STRUCTURE_RELATIVE_TRANSITION_AUDIT"]
    for row in summary_rows:
        lines.append(
            "|".join(f"{field}={row[field]}" for field in SUMMARY_FIELDS)
        )
    lines.extend(delta_lines)
    (root / "audit_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    print(f"RELATIVE_TRANSITION_MERGED|output={root}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument(
        "--checkpoint-root",
        default="outputs/structure_proto_v2clean_4tasks_seed1/source",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/structure_relative_transition_audit_seed1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main():
    args = build_parser().parse_args()
    if args.merge:
        merge_outputs(args)
    elif args.task:
        run_task(args)
    else:
        raise SystemExit("--task or --merge is required")


if __name__ == "__main__":
    main()
