#!/usr/bin/env python3
"""Read-only Layer-1/2 boundary and class-relative structure audit."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_representation_chain_audit import (
    DOMAINS,
    TASKS,
    _per_class_f1,
    available_target_classes,
    build_audit_datasets,
    cosine_knn_predictions,
    deterministic_loader,
    effective_rank,
    fit_source_probe,
    load_source_model,
    remap_to_classes,
    seed_all,
    target_oracle_probe,
    write_csv,
)


VARIANTS = ("current", "set_response", "residual_response")
STAGES = (
    "shape_token_ordered",
    "anchor_similarity_ordered",
    "anchor_similarity_sorted",
    "shape_response",
)
CHECKPOINT_ROOTS = {
    "current": "outputs/structure_proto_v2clean_4tasks_seed1/source",
    "set_response": "outputs/structure_set_response_4tasks_seed1/source",
    "residual_response": "outputs/structure_residual_response_4tasks_seed1/source",
}
BOUNDARY_FIELDS = (
    "variant", "task", "stage", "feature_dim", "effective_rank",
    "source_val_macro_f1", "target_oracle_macro_f1",
    "source_to_target_macro_f1", "source_to_target_knn_macro_f1",
)
BOUNDARY_CLASS_FIELDS = (
    "variant", "task", "stage", "class", "source_support", "target_support",
    "source_to_target_f1", "source_to_target_knn_f1",
)
RELATION_FIELDS = (
    "variant", "task", "stage", "num_classes", "valid_triplets",
    "triplet_order_agreement", "distance_rank_correlation",
    "nearest_class_agreement",
)
RELATION_CLASS_FIELDS = (
    "variant", "task", "stage", "class", "source_nearest_class",
    "target_nearest_class", "source_nearest_distance", "target_nearest_distance",
    "nearest_class_agrees",
)
PAIR_FIELDS = (
    "variant", "task", "stage", "class_a", "class_b",
    "source_distance", "target_distance", "absolute_difference",
)


@torch.no_grad()
def extract_boundary_stages(structure):
    tokens = structure["shape_tokens"]
    similarity = structure["shapelet_similarity"]
    return {
        "shape_token_ordered": tokens.flatten(1).detach(),
        "anchor_similarity_ordered": similarity.flatten(1).detach(),
        "anchor_similarity_sorted": (
            similarity.sort(dim=1).values.transpose(1, 2).flatten(1).detach()
        ),
        "shape_response": structure["shapelet_response"].detach(),
    }


@torch.no_grad()
def extract_dataset_once(model, dataset, batch_size, num_workers, device, pixel_budget):
    collected = {stage: [] for stage in STAGES}
    labels = []
    for batch in deterministic_loader(
        dataset, batch_size, num_workers, pixel_budget=pixel_budget,
    ):
        spatial = model.spatial_encoder(
            batch["pixels"].to(device, non_blocking=True),
            batch["valid_pixels"].to(device, non_blocking=True),
            batch["extra"].to(device, non_blocking=True),
        )
        structure = model.prepare_structure(
            spatial, batch["positions"].to(device, non_blocking=True),
        )
        for stage, values in extract_boundary_stages(structure).items():
            collected[stage].append(values.float().cpu())
        labels.append(batch["label"].long().cpu())
    return {
        "features": {
            stage: torch.cat(values).numpy() for stage, values in collected.items()
        },
        "labels": torch.cat(labels).numpy(),
    }


def class_distance_matrix(centers):
    normalized = F.normalize(
        torch.as_tensor(centers, dtype=torch.float64), dim=-1,
    )
    distance = (1. - normalized @ normalized.T).clamp_min(0.).cpu().numpy()
    distance = .5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.)
    return distance


def relation_metrics(source_distance, target_distance, tie_tolerance=1e-6):
    source_distance = np.asarray(source_distance, dtype=np.float64)
    target_distance = np.asarray(target_distance, dtype=np.float64)
    if source_distance.shape != target_distance.shape or source_distance.ndim != 2:
        raise ValueError("source and target distance matrices must align")
    classes = source_distance.shape[0]
    decisions = []
    for anchor in range(classes):
        others = [index for index in range(classes) if index != anchor]
        for left, right in combinations(others, 2):
            source_delta = source_distance[anchor, left] - source_distance[anchor, right]
            target_delta = target_distance[anchor, left] - target_distance[anchor, right]
            if (
                abs(source_delta) < tie_tolerance
                or abs(target_delta) < tie_tolerance
            ):
                continue
            decisions.append(np.sign(source_delta) == np.sign(target_delta))
    upper = np.triu_indices(classes, k=1)
    source_pairs = source_distance[upper]
    target_pairs = target_distance[upper]
    source_constant = np.ptp(source_pairs) < tie_tolerance
    target_constant = np.ptp(target_pairs) < tie_tolerance
    if source_pairs.size < 2 or source_constant or target_constant:
        rank_correlation = 1. if np.allclose(source_pairs, target_pairs) else 0.
    else:
        rank_correlation = float(spearmanr(source_pairs, target_pairs).statistic)
        if not np.isfinite(rank_correlation):
            rank_correlation = 1. if np.allclose(source_pairs, target_pairs) else 0.
    masked_source = source_distance.copy()
    masked_target = target_distance.copy()
    np.fill_diagonal(masked_source, np.inf)
    np.fill_diagonal(masked_target, np.inf)
    nearest_agreement = np.argmin(masked_source, axis=1) == np.argmin(
        masked_target, axis=1,
    )
    return {
        "valid_triplets": int(len(decisions)),
        "triplet_order_agreement": float(np.mean(decisions)) if decisions else 1.,
        "distance_rank_correlation": rank_correlation,
        "nearest_class_agreement": float(nearest_agreement.mean()),
    }


def _active_centers(features, labels, class_names):
    centers, active_names = [], []
    for class_id, class_name in enumerate(class_names):
        selected = features[labels == class_id]
        if not len(selected):
            continue
        center = F.normalize(
            torch.as_tensor(selected, dtype=torch.float64).mean(0), dim=0,
        )
        centers.append(center.numpy())
        active_names.append(class_name)
    return np.stack(centers), active_names


def class_relation_rows(variant, task, stage, class_names, source, target):
    source_centers, source_names = _active_centers(
        source["features"][stage], source["labels"], class_names,
    )
    target_centers, target_names = _active_centers(
        target["features"][stage], target["labels"], class_names,
    )
    active = [name for name in class_names if name in source_names and name in target_names]
    if len(active) < 2:
        raise RuntimeError(f"need at least two active classes for {variant} {task} {stage}")
    source_centers = np.stack([
        source_centers[source_names.index(name)] for name in active
    ])
    target_centers = np.stack([
        target_centers[target_names.index(name)] for name in active
    ])
    source_distance = class_distance_matrix(source_centers)
    target_distance = class_distance_matrix(target_centers)
    metrics = relation_metrics(source_distance, target_distance)
    summary = {
        "variant": variant, "task": task, "stage": stage,
        "num_classes": len(active), **metrics,
    }
    source_masked, target_masked = source_distance.copy(), target_distance.copy()
    np.fill_diagonal(source_masked, np.inf)
    np.fill_diagonal(target_masked, np.inf)
    source_nearest = source_masked.argmin(1)
    target_nearest = target_masked.argmin(1)
    per_class = [{
        "variant": variant, "task": task, "stage": stage, "class": name,
        "source_nearest_class": active[int(source_nearest[index])],
        "target_nearest_class": active[int(target_nearest[index])],
        "source_nearest_distance": float(source_masked[index, source_nearest[index]]),
        "target_nearest_distance": float(target_masked[index, target_nearest[index]]),
        "nearest_class_agrees": bool(source_nearest[index] == target_nearest[index]),
    } for index, name in enumerate(active)]
    pairs = [{
        "variant": variant, "task": task, "stage": stage,
        "class_a": active[left], "class_b": active[right],
        "source_distance": float(source_distance[left, right]),
        "target_distance": float(target_distance[left, right]),
        "absolute_difference": float(abs(
            source_distance[left, right] - target_distance[left, right]
        )),
    } for left, right in combinations(range(len(active)), 2)]
    return summary, per_class, pairs


def _remap_split(extracted, original_classes, selected_classes):
    labels = extracted["labels"]
    remapped_features, remapped_labels = {}, None
    for stage, features in extracted["features"].items():
        selected, stage_labels = remap_to_classes(
            features, labels, original_classes, selected_classes,
        )
        remapped_features[stage] = selected
        if remapped_labels is None:
            remapped_labels = stage_labels
        elif not np.array_equal(remapped_labels, stage_labels):
            raise RuntimeError("stage label remapping diverged")
    return {"features": remapped_features, "labels": remapped_labels}


def boundary_rows(variant, task, class_names, source_train, source_val, target_val, device):
    class_ids = np.arange(len(class_names), dtype=np.int64)
    metrics_rows, class_rows = [], []
    source_support = np.bincount(source_val["labels"], minlength=len(class_names))
    target_support = np.bincount(target_val["labels"], minlength=len(class_names))
    for stage in STAGES:
        source_probe = fit_source_probe(
            source_train["features"][stage], source_train["labels"],
            source_val["features"][stage], source_val["labels"],
            target_val["features"][stage], target_val["labels"], class_ids,
        )
        oracle = target_oracle_probe(
            target_val["features"][stage], target_val["labels"], class_ids, seed=1,
        )
        knn_prediction = cosine_knn_predictions(
            source_train["features"][stage], source_train["labels"],
            target_val["features"][stage], device=device,
        )
        knn_f1 = f1_score(
            target_val["labels"], knn_prediction, labels=class_ids,
            average="macro", zero_division=0,
        )
        source_to_target_per_class = source_probe["source_to_target_per_class_f1"]
        knn_per_class = _per_class_f1(
            target_val["labels"], knn_prediction, class_ids,
        )
        rank_features = np.concatenate((
            source_val["features"][stage], target_val["features"][stage],
        ))
        metrics_rows.append({
            "variant": variant, "task": task, "stage": stage,
            "feature_dim": int(source_train["features"][stage].shape[1]),
            "effective_rank": effective_rank(rank_features),
            "source_val_macro_f1": source_probe["source_val_macro_f1"],
            "target_oracle_macro_f1": oracle["macro_f1"],
            "source_to_target_macro_f1": source_probe["source_to_target_macro_f1"],
            "source_to_target_knn_macro_f1": float(knn_f1),
        })
        for class_id, class_name in enumerate(class_names):
            class_rows.append({
                "variant": variant, "task": task, "stage": stage,
                "class": class_name,
                "source_support": int(source_support[class_id]),
                "target_support": int(target_support[class_id]),
                "source_to_target_f1": float(source_to_target_per_class[class_id]),
                "source_to_target_knn_f1": float(knn_per_class[class_id]),
            })
    return metrics_rows, class_rows


def _checkpoint(root, source):
    return Path(root) / f"source_{source}_seed1" / "fold_0" / "model.pt"


def residual_current_deltas(relations):
    indexed = {
        (row["variant"], row["task"]): row for row in relations
        if row["stage"] == "shape_response"
        and row["variant"] in ("current", "residual_response")
    }
    rows = []
    for task in TASKS:
        current = indexed.get(("current", task))
        residual = indexed.get(("residual_response", task))
        if current is None or residual is None:
            continue
        rows.append({
            "task": task,
            "triplet_order_agreement_delta": (
                residual["triplet_order_agreement"]
                - current["triplet_order_agreement"]
            ),
            "distance_rank_correlation_delta": (
                residual["distance_rank_correlation"]
                - current["distance_rank_correlation"]
            ),
            "nearest_class_agreement_delta": (
                residual["nearest_class_agreement"]
                - current["nearest_class_agreement"]
            ),
        })
    return rows


def _summary_markdown(boundary, relations):
    lines = [
        "# Structure Layer Boundary and Class-relative Audit", "",
        "## Layer 1/2 Boundary", "",
        "| Variant | Task | Stage | Source→target F1 | kNN F1 | Effective rank |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in boundary:
        lines.append(
            f"| {row['variant']} | {row['task']} | {row['stage']} | "
            f"{row['source_to_target_macro_f1']} | "
            f"{row['source_to_target_knn_macro_f1']} | {row['effective_rank']} |"
        )
    lines.extend([
        "", "## Class-relative Structure", "",
        "| Variant | Task | Stage | Triplet agreement | Distance rank correlation | Nearest-class agreement |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in relations:
        lines.append(
            f"| {row['variant']} | {row['task']} | {row['stage']} | "
            f"{row['triplet_order_agreement']} | "
            f"{row['distance_rank_correlation']} | {row['nearest_class_agreement']} |"
        )
    lines.extend([
        "", "## Residual minus Current at shape_response", "",
        "| Task | Triplet agreement Δ | Distance rank correlation Δ | Nearest-class agreement Δ |",
        "|---|---:|---:|---:|",
    ])
    for row in residual_current_deltas(relations):
        lines.append(
            f"| {row['task']} | {row['triplet_order_agreement_delta']} | "
            f"{row['distance_rank_correlation_delta']} | "
            f"{row['nearest_class_agreement_delta']} |"
        )
    lines.extend([
        "", "## Interpretation boundaries", "",
        "- Poor shape-token or sorted-similarity transfer does not exclude Layer 1 or early Layer 2.",
        "- Adequate tokens/sorted similarity followed by a response drop localizes the loss to response construction.",
        "- Adequate response transfer with weak class-relative preservation supports, but does not prove, a Layer 3 mismatch.",
        "- Strong class-relative preservation with weak final adaptation does not support a Layer 3 class-relative explanation.",
        "- Residual-versus-current changes are correlational checks only.", "",
    ])
    return "\n".join(lines)


def run(args):
    seed_all(args.seed)
    device = torch.device(args.device)
    roots = {
        "current": args.current_root,
        "set_response": args.set_root,
        "residual_response": args.residual_root,
    }
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    boundary, boundary_classes = [], []
    relations, relation_classes, pair_rows = [], [], []
    manifest_runs = []
    for variant in VARIANTS:
        for task, (source_alias, target_alias) in TASKS.items():
            checkpoint = _checkpoint(roots[variant], source_alias)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"source checkpoint not found: {checkpoint.resolve()}")
            print(f"BOUNDARY_AUDIT_START|variant={variant}|task={task}|checkpoint={checkpoint}")
            model, config = load_source_model(checkpoint, device, variant)
            datasets, split = build_audit_datasets(
                config, DOMAINS[source_alias], DOMAINS[target_alias],
                args.data_root, args.seed,
            )
            extracted = {
                name: extract_dataset_once(
                    model, dataset, args.batch_size, args.num_workers,
                    device, args.pixel_budget,
                ) for name, dataset in datasets.items()
            }
            classes = available_target_classes(
                args.data_root, DOMAINS[target_alias], list(config.classes),
                bool(config.combine_spring_and_winter),
            )
            prepared = {
                name: _remap_split(values, list(config.classes), classes)
                for name, values in extracted.items()
            }
            task_boundary, task_boundary_classes = boundary_rows(
                variant, task, classes, prepared["source_train"],
                prepared["source_val"], prepared["target_val"], args.device,
            )
            boundary.extend(task_boundary)
            boundary_classes.extend(task_boundary_classes)
            for stage in STAGES:
                summary, per_class, pairs = class_relation_rows(
                    variant, task, stage, classes,
                    prepared["source_val"], prepared["target_val"],
                )
                relations.append(summary)
                relation_classes.extend(per_class)
                pair_rows.extend(pairs)
            manifest_runs.append({
                "variant": variant, "task": task, "checkpoint": str(checkpoint),
                "source": DOMAINS[source_alias], "target": DOMAINS[target_alias],
                "split_counts": {name: len(indices) for name, indices in split.items()},
                "test_split_accessed": False, "uda_checkpoint_used": False,
                "forward_passes_per_split": 1,
            })
            print(f"BOUNDARY_AUDIT_FINISHED|variant={variant}|task={task}")
    write_csv(output / "boundary_stage_metrics.csv", boundary, BOUNDARY_FIELDS)
    write_csv(
        output / "boundary_stage_per_class.csv", boundary_classes,
        BOUNDARY_CLASS_FIELDS,
    )
    write_csv(output / "class_relation_summary.csv", relations, RELATION_FIELDS)
    write_csv(
        output / "class_relation_per_class.csv", relation_classes,
        RELATION_CLASS_FIELDS,
    )
    write_csv(output / "class_pair_distances.csv", pair_rows, PAIR_FIELDS)
    (output / "summary.md").write_text(
        _summary_markdown(boundary, relations), encoding="utf-8",
    )
    (output / "audit_manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "variants": list(VARIANTS), "tasks": list(TASKS), "stages": list(STAGES),
        "test_split_accessed": False, "uda_checkpoint_used": False,
        "direct_response_query_checkpoint_used": False,
        "runs": manifest_runs,
    }, indent=2), encoding="utf-8")
    print(
        f"BOUNDARY_AUDIT_COMPLETE|output={output}|"
        f"boundary_rows={len(boundary)}|relation_rows={len(relations)}"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--current-root", default=CHECKPOINT_ROOTS["current"])
    parser.add_argument("--set-root", default=CHECKPOINT_ROOTS["set_response"])
    parser.add_argument("--residual-root", default=CHECKPOINT_ROOTS["residual_response"])
    parser.add_argument(
        "--output-root",
        default="outputs/structure_layer_boundary_class_relation_audit_seed1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
