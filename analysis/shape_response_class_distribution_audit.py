#!/usr/bin/env python3
"""Read-only class-conditional source/target Shape Response audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
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
    fit_source_probe,
    load_source_model,
    remap_to_classes,
    seed_all,
    target_oracle_probe,
    write_csv,
)


VARIANTS = ("current", "residual_response")
AUDIT_SPLITS = ("source_train", "source_val", "target_val")
CHECKPOINT_ROOTS = {
    "current": "outputs/structure_proto_v2clean_4tasks_seed1/source",
    "residual_response": "outputs/structure_residual_response_4tasks_seed1/source",
}
CLASS_FIELDS = (
    "variant", "task", "class", "source_support", "target_support",
    "centroid_shift", "source_radius", "target_radius", "radius_ratio",
    "mean_margin", "median_margin", "positive_margin_rate",
    "source_knn5_correct_rate", "dominant_wrong_source_class",
    "dominant_wrong_count", "dominant_wrong_rate", "target_oracle_f1",
    "source_to_target_probe_f1",
)
SUMMARY_FIELDS = (
    "variant", "task", "mean_centroid_shift", "mean_radius_ratio",
    "mean_positive_margin_rate", "mean_source_knn5_correct_rate",
    "target_oracle_macro_f1", "source_to_target_probe_macro_f1",
)
CONFUSION_FIELDS = (
    "variant", "task", "target_class", "wrong_source_class", "count", "rate",
)


@torch.no_grad()
def extract_shape_response(structure):
    """Return the real model response; never reconstruct it in the audit."""
    return structure["shapelet_response"].detach()


@torch.no_grad()
def extract_dataset(model, dataset, batch_size, num_workers, device, pixel_budget):
    responses, labels = [], []
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
        responses.append(extract_shape_response(structure).float().cpu())
        labels.append(batch["label"].long().cpu())
    return {
        "features": torch.cat(responses).numpy(),
        "labels": torch.cat(labels).numpy(),
    }


def _normalize(values):
    return F.normalize(torch.as_tensor(values, dtype=torch.float64), dim=-1).numpy()


def _centroids(features, labels, class_count):
    normalized = _normalize(features)
    centers = []
    for class_id in range(class_count):
        selected = normalized[labels == class_id]
        if not len(selected):
            raise ValueError(f"class {class_id} has no samples")
        center = selected.mean(0)
        norm = np.linalg.norm(center)
        if norm <= 1e-12:
            raise ValueError(f"class {class_id} has a zero centroid")
        centers.append(center / norm)
    return normalized, np.stack(centers)


def _radius(values, center):
    return float(np.mean(1. - values @ center))


def _safe_ratio(target, source, eps=1e-12):
    if abs(source) <= eps and abs(target) <= eps:
        return 1.
    return float(target / max(source, eps))


def _knn5(source, source_labels, target):
    source = np.asarray(source)
    target = np.asarray(target)
    k = min(5, len(source))
    similarity = target @ source.T
    # Stable sorting makes tied-neighbour selection deterministic.
    nearest = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    labels = np.asarray(source_labels, dtype=np.int64)[nearest]
    return np.asarray([np.bincount(row).argmax() for row in labels], dtype=np.int64)


def class_distribution_rows(
    variant, task, class_names, source_features, source_labels,
    target_features, target_labels,
):
    source_labels = np.asarray(source_labels, dtype=np.int64)
    target_labels = np.asarray(target_labels, dtype=np.int64)
    class_count = len(class_names)
    source, source_centers = _centroids(
        source_features, source_labels, class_count,
    )
    target, target_centers = _centroids(
        target_features, target_labels, class_count,
    )
    distances = 1. - target @ source_centers.T
    knn_prediction = _knn5(source, source_labels, target)
    rows, confusion = [], []
    for class_id, class_name in enumerate(class_names):
        source_mask = source_labels == class_id
        target_mask = target_labels == class_id
        class_distances = distances[target_mask]
        correct = class_distances[:, class_id]
        wrong_distances = class_distances.copy()
        wrong_distances[:, class_id] = np.inf
        nearest_wrong = wrong_distances.argmin(1)
        wrong = wrong_distances[np.arange(len(wrong_distances)), nearest_wrong]
        margin = wrong - correct
        negative = margin < 0.
        wrong_counts = Counter(nearest_wrong[negative].tolist())
        if wrong_counts:
            dominant_id, dominant_count = sorted(
                wrong_counts.items(), key=lambda item: (-item[1], item[0]),
            )[0]
            dominant_name = class_names[int(dominant_id)]
            dominant_rate = dominant_count / int(negative.sum())
        else:
            dominant_name, dominant_count, dominant_rate = "", 0, 0.
        for wrong_id, count in sorted(wrong_counts.items()):
            confusion.append({
                "variant": variant, "task": task, "target_class": class_name,
                "wrong_source_class": class_names[int(wrong_id)],
                "count": int(count), "rate": float(count / int(negative.sum())),
            })
        source_radius = _radius(source[source_mask], source_centers[class_id])
        target_radius = _radius(target[target_mask], target_centers[class_id])
        rows.append({
            "variant": variant, "task": task, "class": class_name,
            "source_support": int(source_mask.sum()),
            "target_support": int(target_mask.sum()),
            "centroid_shift": float(1. - source_centers[class_id] @ target_centers[class_id]),
            "source_radius": source_radius, "target_radius": target_radius,
            "radius_ratio": _safe_ratio(target_radius, source_radius),
            "mean_margin": float(margin.mean()),
            "median_margin": float(np.median(margin)),
            "positive_margin_rate": float((margin > 0.).mean()),
            "source_knn5_correct_rate": float(
                (knn_prediction[target_mask] == class_id).mean()
            ),
            "dominant_wrong_source_class": dominant_name,
            "dominant_wrong_count": int(dominant_count),
            "dominant_wrong_rate": float(dominant_rate),
        })
    return rows, confusion


def _remap(extracted, original_classes, selected_classes):
    features, labels = remap_to_classes(
        extracted["features"], extracted["labels"],
        original_classes, selected_classes,
    )
    return {"features": features, "labels": labels}


def active_audit_classes(extracted, original_classes):
    """Select classes observed in every permitted split, never consulting test."""
    active_ids = set(range(len(original_classes)))
    for split_name in AUDIT_SPLITS:
        active_ids &= set(np.unique(extracted[split_name]["labels"]).tolist())
    active = [
        name for class_id, name in enumerate(original_classes)
        if class_id in active_ids
    ]
    excluded = [name for name in original_classes if name not in set(active)]
    if len(active) < 2:
        raise RuntimeError(f"fewer than two classes have support in all audit splits: {active}")
    return active, excluded


def evaluate_task(variant, task, class_names, source_train, source_val, target_val, seed):
    class_ids = np.arange(len(class_names), dtype=np.int64)
    source_probe = fit_source_probe(
        source_train["features"], source_train["labels"],
        source_val["features"], source_val["labels"],
        target_val["features"], target_val["labels"], class_ids,
    )
    oracle = target_oracle_probe(
        target_val["features"], target_val["labels"], class_ids, seed,
    )
    rows, confusion = class_distribution_rows(
        variant, task, class_names,
        source_val["features"], source_val["labels"],
        target_val["features"], target_val["labels"],
    )
    for class_id, row in enumerate(rows):
        row["target_oracle_f1"] = float(oracle["per_class_f1"][class_id])
        row["source_to_target_probe_f1"] = float(
            source_probe["source_to_target_per_class_f1"][class_id]
        )
    summary = {
        "variant": variant, "task": task,
        "mean_centroid_shift": float(np.mean([r["centroid_shift"] for r in rows])),
        "mean_radius_ratio": float(np.mean([r["radius_ratio"] for r in rows])),
        "mean_positive_margin_rate": float(np.mean([
            r["positive_margin_rate"] for r in rows
        ])),
        "mean_source_knn5_correct_rate": float(np.mean([
            r["source_knn5_correct_rate"] for r in rows
        ])),
        "target_oracle_macro_f1": float(oracle["macro_f1"]),
        "source_to_target_probe_macro_f1": float(
            source_probe["source_to_target_macro_f1"]
        ),
    }
    return summary, rows, confusion


def manifest_payload(runs, seed):
    return {
        "seed": int(seed), "variants": list(VARIANTS), "tasks": list(TASKS),
        "splits": list(AUDIT_SPLITS), "shape_response_source": "real_model_forward",
        "knn": "cosine,k=5,source_val_reference",
        "target_labels_use": "offline_diagnostic_only",
        "test_split_accessed": False, "uda_checkpoint_used": False,
        "runs": list(runs),
    }


def validate_outputs(summary_rows, class_rows):
    expected = {(variant, task) for variant in VARIANTS for task in TASKS}
    actual = {(row["variant"], row["task"]) for row in summary_rows}
    if actual != expected or len(summary_rows) != len(expected):
        raise ValueError(f"incomplete variant/task grid: expected={expected}, actual={actual}")
    for row in list(summary_rows) + list(class_rows):
        for key, value in row.items():
            if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
                raise ValueError(f"non-finite output: {row.get('variant')} {row.get('task')} {key}")


def _summary_markdown(summary_rows, class_rows):
    lines = [
        "# Shape Response Class Distribution Audit", "",
        "| Variant | Task | Centroid shift | Radius ratio | Positive margin | Source 5-NN | Target oracle F1 | Source→target F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['task']} | {row['mean_centroid_shift']:.6f} | "
            f"{row['mean_radius_ratio']:.6f} | {row['mean_positive_margin_rate']:.6f} | "
            f"{row['mean_source_knn5_correct_rate']:.6f} | "
            f"{row['target_oracle_macro_f1']:.6f} | "
            f"{row['source_to_target_probe_macro_f1']:.6f} |"
        )
    lines.extend(["", "## Three most visible classes per run", ""])
    for variant in VARIANTS:
        for task in TASKS:
            selected = [r for r in class_rows if r["variant"] == variant and r["task"] == task]
            selected.sort(key=lambda r: (
                r["positive_margin_rate"] + r["source_knn5_correct_rate"],
                -r["centroid_shift"], r["class"],
            ))
            lines.append(f"### {variant} · {task}")
            lines.append("")
            lines.append("| Class | Shift | Radius ratio | Positive margin | Source 5-NN | Dominant wrong source |")
            lines.append("|---|---:|---:|---:|---:|---|")
            for row in selected[:3]:
                lines.append(
                    f"| {row['class']} | {row['centroid_shift']:.6f} | "
                    f"{row['radius_ratio']:.6f} | {row['positive_margin_rate']:.6f} | "
                    f"{row['source_knn5_correct_rate']:.6f} | "
                    f"{row['dominant_wrong_source_class']} |"
                )
            lines.append("")
    lines.extend([
        "## Interpretation boundary", "",
        "Layer 3 mismatch is supported only when target-oracle discrimination remains useful while source→target transfer, positive-margin rate, or source 5-NN support is weak, or when centroid shift, target expansion, or wrong-source overlap is evident.",
        "If same-class source/target distribution diagnostics remain normal while transfer is poor, these results do not support attributing the main failure to this semantic mismatch.",
        "Residual-versus-current differences are associations, not causal effects.", "",
    ])
    return "\n".join(lines)


def _checkpoint(root, source):
    return Path(root) / f"source_{source}_seed1" / "fold_0" / "model.pt"


def run(args):
    seed_all(args.seed)
    device = torch.device(args.device)
    roots = {"current": args.current_root, "residual_response": args.residual_root}
    summaries, classes, confusion, manifest_runs = [], [], [], []
    for variant in VARIANTS:
        for task, (source_alias, target_alias) in TASKS.items():
            checkpoint = _checkpoint(roots[variant], source_alias)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"source checkpoint not found: {checkpoint.resolve()}")
            print(f"SHAPE_DISTRIBUTION_START|variant={variant}|task={task}|checkpoint={checkpoint}")
            model, config = load_source_model(checkpoint, device, variant)
            datasets, split = build_audit_datasets(
                config, DOMAINS[source_alias], DOMAINS[target_alias],
                args.data_root, args.seed,
            )
            extracted = {name: extract_dataset(
                model, dataset, args.batch_size, args.num_workers,
                device, args.pixel_budget,
            ) for name, dataset in datasets.items()}
            class_names, excluded_classes = active_audit_classes(
                extracted, list(config.classes),
            )
            prepared = {
                name: _remap(values, list(config.classes), class_names)
                for name, values in extracted.items()
            }
            summary, per_class, task_confusion = evaluate_task(
                variant, task, class_names, prepared["source_train"],
                prepared["source_val"], prepared["target_val"], args.seed,
            )
            summaries.append(summary)
            classes.extend(per_class)
            confusion.extend(task_confusion)
            manifest_runs.append({
                "variant": variant, "task": task, "checkpoint": str(checkpoint),
                "source": DOMAINS[source_alias], "target": DOMAINS[target_alias],
                "split_counts": {name: len(indices) for name, indices in split.items()},
                "audited_classes": class_names,
                "excluded_zero_support_classes": excluded_classes,
                "strict_checkpoint_load": True, "test_split_accessed": False,
                "uda_checkpoint_used": False,
            })
            print(f"SHAPE_DISTRIBUTION_FINISHED|variant={variant}|task={task}")
    validate_outputs(summaries, classes)
    output = Path(args.output_root)
    write_csv(output / "class_distribution_per_class.csv", classes, CLASS_FIELDS)
    write_csv(output / "class_distribution_summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(output / "target_source_confusion.csv", confusion, CONFUSION_FIELDS)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.md").write_text(
        _summary_markdown(summaries, classes), encoding="utf-8",
    )
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest_payload(manifest_runs, args.seed), indent=2),
        encoding="utf-8",
    )
    print(f"SHAPE_DISTRIBUTION_COMPLETE|output={output}|rows={len(classes)}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--current-root", default=CHECKPOINT_ROOTS["current"])
    parser.add_argument("--residual-root", default=CHECKPOINT_ROOTS["residual_response"])
    parser.add_argument(
        "--output-root",
        default="outputs/shape_response_class_distribution_audit_seed1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
