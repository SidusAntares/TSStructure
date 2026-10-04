#!/usr/bin/env python3
"""Read-only audit of cross-domain Fourier curve path correspondence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_representation_chain_audit import (
    DOMAINS,
    TASKS,
    build_audit_datasets,
    deterministic_loader,
    load_source_model,
    seed_all,
    write_csv,
)


METHODS = ("calendar", "global_shift", "arc_length", "dtw")
DATASET_ROLES = ("source_val", "target_val")
CHECKPOINT_ROLE = "P_source"
MANIFEST_FLAGS = {
    "training": False,
    "target_train_accessed": False,
    "test_accessed": False,
    "uses_validation_labels": True,
}
MATRIX_FIELDS = (
    "task", "method", "target_class", "source_class", "similarity",
    "source_support", "target_support", "dtw_path_length",
)
CLASS_FIELDS = (
    "task", "class", "method", "same_class_similarity",
    "nearest_wrong_class", "nearest_wrong_similarity",
    "same_vs_wrong_margin", "best_global_shift_days",
)
TASK_FIELDS = (
    "task", "method", "mean_same_class_similarity",
    "mean_nearest_wrong_similarity", "mean_margin",
    "top1_class_retrieval_accuracy", "best_global_shift_days",
    "best_global_shift_score",
)


def normalize_curve(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, float(eps))


def curve_similarity(left, right):
    left = normalize_curve(left)
    right = normalize_curve(right)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("aligned curves must have equal [K,D] shape")
    if not len(left):
        raise ValueError("aligned curve has no common time points")
    return float(np.mean(np.sum(left * right, axis=-1)))


def calendar_class_prototypes(curves, labels, class_ids):
    curves = normalize_curve(curves)
    labels = np.asarray(labels, dtype=np.int64)
    result = {}
    for class_id in class_ids:
        selected = curves[labels == int(class_id)]
        if len(selected):
            result[int(class_id)] = normalize_curve(selected.mean(axis=0))
    return result


def compare_at_shift(source, target, grid, shift):
    """Return target(t+shift) and source(t), restricted to their overlap."""
    source = normalize_curve(source)
    target = normalize_curve(target)
    grid = np.asarray(grid, dtype=np.float64)
    if source.shape != target.shape or source.shape[0] != grid.size:
        raise ValueError("source, target, and grid must share time length")
    query = grid + float(shift)
    valid = (query >= grid[0]) & (query <= grid[-1])
    if not np.any(valid):
        raise ValueError("shift leaves no common valid time range")
    shifted = np.stack([
        np.interp(query[valid], grid, target[:, dim])
        for dim in range(target.shape[1])
    ], axis=1)
    return normalize_curve(shifted), source[valid]


def shifted_similarity(source, target, grid, shift):
    shifted, source_common = compare_at_shift(source, target, grid, shift)
    return curve_similarity(source_common, shifted)


def find_global_shift(source_prototypes, target_prototypes, grid, shifts=range(-60, 61)):
    shared = sorted(set(source_prototypes) & set(target_prototypes))
    if not shared:
        raise ValueError("global shift requires at least one shared class")
    scored = []
    for shift in shifts:
        try:
            score = np.mean([
                shifted_similarity(
                    source_prototypes[class_id], target_prototypes[class_id],
                    grid, shift,
                )
                for class_id in shared
            ])
        except ValueError as error:
            if "no common valid time range" not in str(error):
                raise
            continue
        scored.append((float(score), int(shift)))
    if not scored:
        raise ValueError("global shift scan has no valid overlap")
    best_score = max(score for score, _ in scored)
    # Deterministic tie break: smallest absolute shift, then signed value.
    tied = [
        shift for score, shift in scored
        if np.isclose(score, best_score, atol=1e-12)
    ]
    best_shift = min(tied, key=lambda value: (abs(value), value))
    return int(best_shift), float(best_score)


def arc_length_coordinates(curve, eps=1e-12):
    curve = normalize_curve(curve)
    if len(curve) < 2:
        raise ValueError("arc-length curve needs at least two points")
    increments = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    total = float(cumulative[-1])
    if total < float(eps):
        return np.linspace(0.0, 1.0, len(curve))
    result = cumulative / total
    result[0], result[-1] = 0.0, 1.0
    return result


def _compress_coordinates(q, curve, eps=1e-12):
    kept_q, kept_curve = [float(q[0])], [curve[0]]
    for coordinate, value in zip(q[1:], curve[1:]):
        if float(coordinate) <= kept_q[-1] + float(eps):
            if np.isclose(coordinate, 1.0):
                kept_q[-1], kept_curve[-1] = 1.0, value
            continue
        kept_q.append(float(coordinate))
        kept_curve.append(value)
    if kept_q[-1] < 1.0:
        kept_q.append(1.0)
        kept_curve.append(curve[-1])
    return np.asarray(kept_q), np.asarray(kept_curve)


def resample_by_arc_length(curve, query_count=64):
    curve = normalize_curve(curve)
    q, compressed = _compress_coordinates(
        arc_length_coordinates(curve), curve,
    )
    query = np.linspace(0.0, 1.0, int(query_count))
    interpolated = np.stack([
        np.interp(query, q, compressed[:, dim])
        for dim in range(compressed.shape[1])
    ], axis=1)
    return normalize_curve(interpolated)


def arc_class_prototypes(curves, labels, class_ids, query_count=64):
    labels = np.asarray(labels, dtype=np.int64)
    result = {}
    for class_id in class_ids:
        selected = np.asarray(curves)[labels == int(class_id)]
        if len(selected):
            # Required ordering: sample-wise arc resampling, then class average.
            resampled = [
                resample_by_arc_length(sample, query_count) for sample in selected
            ]
            result[int(class_id)] = normalize_curve(np.mean(resampled, axis=0))
    return result


def dtw_similarity(source, target):
    source, target = normalize_curve(source), normalize_curve(target)
    local = 1.0 - source @ target.T
    rows, columns = local.shape
    cost = np.full((rows, columns), np.inf, dtype=np.float64)
    steps = np.zeros((rows, columns), dtype=np.int64)
    cost[0, 0], steps[0, 0] = local[0, 0], 1
    for row in range(rows):
        for column in range(columns):
            if row == 0 and column == 0:
                continue
            parents = []
            if row:
                parents.append((cost[row - 1, column], steps[row - 1, column]))
            if column:
                parents.append((cost[row, column - 1], steps[row, column - 1]))
            if row and column:
                parents.append((cost[row - 1, column - 1], steps[row - 1, column - 1]))
            parent_cost, parent_steps = min(parents, key=lambda item: (item[0], item[1]))
            cost[row, column] = local[row, column] + parent_cost
            steps[row, column] = parent_steps + 1
    path_length = int(steps[-1, -1])
    return float(1.0 - cost[-1, -1] / path_length), path_length


def similarity_matrix(source_prototypes, target_prototypes, method, grid=None, shift=None):
    matrix = {}
    for target_class, target in target_prototypes.items():
        for source_class, source in source_prototypes.items():
            if method == "global_shift":
                value = shifted_similarity(source, target, grid, shift)
            elif method == "dtw":
                value, _ = dtw_similarity(source, target)
            else:
                value = curve_similarity(source, target)
            matrix[(int(target_class), int(source_class))] = float(value)
    return matrix


def class_summary_row(task, method, class_id, matrix, best_shift, class_names=None):
    same = matrix[(int(class_id), int(class_id))]
    competitors = [
        (source_class, value)
        for (target_class, source_class), value in matrix.items()
        if target_class == int(class_id) and source_class != int(class_id)
    ]
    if not competitors:
        raise ValueError("class summary requires a wrong-class competitor")
    wrong_class, wrong = max(competitors, key=lambda item: item[1])
    name = lambda value: class_names[value] if class_names is not None else value
    return {
        "task": task,
        "class": name(int(class_id)),
        "method": method,
        "same_class_similarity": float(same),
        "nearest_wrong_class": name(int(wrong_class)),
        "nearest_wrong_similarity": float(wrong),
        "same_vs_wrong_margin": float(same - wrong),
        "best_global_shift_days": best_shift if method == "global_shift" else "",
    }


@torch.no_grad()
def extract_fourier_batch(model, batch):
    spatial = model.spatial_encoder(
        batch["pixels"], batch["valid_pixels"], batch["extra"],
    )
    prepared = model.prepare_temporal_features(spatial, batch["positions"])
    context = model.prepare_structure_context(prepared, batch["positions"])
    curves, grid = model.structure_branch.exposer.synthesize_shifted(
        context["coefficients"], 0,
    )
    if grid.ndim == 2:
        grid = grid[0]
    return curves, grid


@torch.no_grad()
def extract_validation_curves(model, dataset, batch_size, num_workers, device):
    curves, labels, common_grid = [], [], None
    for batch in deterministic_loader(dataset, batch_size, num_workers):
        moved = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key in ("pixels", "valid_pixels", "positions", "extra")
        }
        current, grid = extract_fourier_batch(model, moved)
        current = normalize_curve(current.detach().float().cpu().numpy())
        current_grid = grid.detach().double().cpu().numpy()
        if common_grid is None:
            common_grid = current_grid
        elif not np.allclose(common_grid, current_grid):
            raise RuntimeError("canonical Fourier grid changed between batches")
        curves.append(current)
        labels.append(batch["label"].long().cpu().numpy())
    return np.concatenate(curves), np.concatenate(labels), common_grid


def _support(labels):
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def _evaluate_task(task, source_curves, source_labels, target_curves, target_labels, grid, class_names):
    source_support, target_support = _support(source_labels), _support(target_labels)
    source_ids, target_ids = sorted(source_support), sorted(target_support)
    source_calendar = calendar_class_prototypes(source_curves, source_labels, source_ids)
    target_calendar = calendar_class_prototypes(target_curves, target_labels, target_ids)
    shared = sorted(set(source_ids) & set(target_ids))
    if len(shared) < 2:
        raise RuntimeError(f"{task} needs at least two shared validation classes")
    best_shift, best_score = find_global_shift(source_calendar, target_calendar, grid)
    source_arc = arc_class_prototypes(source_curves, source_labels, source_ids, 64)
    target_arc = arc_class_prototypes(target_curves, target_labels, target_ids, 64)
    matrices = {
        "calendar": similarity_matrix(source_calendar, target_calendar, "calendar"),
        "global_shift": similarity_matrix(
            source_calendar, target_calendar, "global_shift", grid, best_shift,
        ),
        "arc_length": similarity_matrix(source_arc, target_arc, "arc_length"),
        "dtw": similarity_matrix(source_calendar, target_calendar, "dtw"),
    }
    dtw_paths = {
        (target_class, source_class): dtw_similarity(source, target)[1]
        for target_class, target in target_calendar.items()
        for source_class, source in source_calendar.items()
    }
    matrix_rows, class_rows, task_rows = [], [], []
    for method, matrix in matrices.items():
        for (target_class, source_class), value in sorted(matrix.items()):
            matrix_rows.append({
                "task": task, "method": method,
                "target_class": class_names[target_class],
                "source_class": class_names[source_class], "similarity": value,
                "source_support": source_support[source_class],
                "target_support": target_support[target_class],
                "dtw_path_length": (
                    dtw_paths[(target_class, source_class)]
                    if method == "dtw" else ""
                ),
            })
        rows = [
            class_summary_row(
                task, method, class_id, matrix, best_shift, class_names,
            )
            for class_id in shared
        ]
        class_rows.extend(rows)
        predicted = {
            target_class: max(
                (source_class for source_class in source_ids),
                key=lambda source_class: matrix[(target_class, source_class)],
            )
            for target_class in shared
        }
        task_rows.append({
            "task": task, "method": method,
            "mean_same_class_similarity": float(np.mean([
                row["same_class_similarity"] for row in rows
            ])),
            "mean_nearest_wrong_similarity": float(np.mean([
                row["nearest_wrong_similarity"] for row in rows
            ])),
            "mean_margin": float(np.mean([
                row["same_vs_wrong_margin"] for row in rows
            ])),
            "top1_class_retrieval_accuracy": float(np.mean([
                predicted[class_id] == class_id for class_id in shared
            ])),
            "best_global_shift_days": best_shift if method == "global_shift" else "",
            "best_global_shift_score": best_score if method == "global_shift" else "",
        })
    return matrix_rows, class_rows, task_rows


def _checkpoint(root, source):
    return Path(root) / f"source_{source}_seed1" / "fold_0" / "model.pt"


def run_task(args):
    seed_all(args.seed)
    source, target = TASKS[args.task]
    checkpoint = _checkpoint(args.checkpoint_root, source)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MISSING|{checkpoint}")
    device = torch.device(args.device)
    model, config = load_source_model(checkpoint, device, "phase_moment")
    datasets, _ = build_audit_datasets(
        config, DOMAINS[source], DOMAINS[target], args.data_root, args.seed,
    )
    source_curves, source_labels, source_grid = extract_validation_curves(
        model, datasets["source_val"], args.batch_size, args.num_workers, device,
    )
    target_curves, target_labels, target_grid = extract_validation_curves(
        model, datasets["target_val"], args.batch_size, args.num_workers, device,
    )
    if not np.allclose(source_grid, target_grid):
        raise RuntimeError("source and target Fourier grids differ")
    rows = _evaluate_task(
        args.task, source_curves, source_labels, target_curves, target_labels,
        source_grid, list(config.classes),
    )
    output = Path(args.output_root) / args.task
    write_csv(output / "curve_similarity_matrix.csv", rows[0], MATRIX_FIELDS)
    write_csv(output / "curve_class_summary.csv", rows[1], CLASS_FIELDS)
    write_csv(output / "task_summary.csv", rows[2], TASK_FIELDS)
    manifest = {
        "task": args.task, "source": source, "target": target,
        "checkpoint": str(checkpoint), "checkpoint_role": CHECKPOINT_ROLE,
        "dataset_roles": list(DATASET_ROLES), "seed": int(args.seed),
        "curve_location": "post_fourier_pre_window",
        "global_shift_wrap": False, "global_shift_scope": "task",
        "arc_length_order": "sample_first_class_average_second",
        "dtw_role": "diagnostic_only", **MANIFEST_FLAGS,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"CURVE_PATH_AUDIT_FINISHED|task={args.task}|output={output}", flush=True)


def _read_csv(path):
    import csv
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def merge_outputs(args):
    root = Path(args.output_root)
    collections = [[], [], []]
    files = ("curve_similarity_matrix.csv", "curve_class_summary.csv", "task_summary.csv")
    for task in TASKS:
        for index, name in enumerate(files):
            path = root / task / name
            if not path.is_file():
                raise RuntimeError(f"incomplete audit output: {path}")
            collections[index].extend(_read_csv(path))
    for name, rows, fields in zip(files, collections, (MATRIX_FIELDS, CLASS_FIELDS, TASK_FIELDS)):
        write_csv(root / name, rows, fields)
    lines = ["Structure curve path audit (numeric summary only)"]
    for row in collections[2]:
        lines.append("|".join(f"{field}={row[field]}" for field in TASK_FIELDS))
    (root / "audit_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CURVE_PATH_AUDIT_MERGED|output={root}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument(
        "--checkpoint-root",
        default="outputs/structure_phase_moment_4tasks_seed1/source",
    )
    parser.add_argument(
        "--output-root", default="outputs/structure_curve_path_audit_seed1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    if args.merge:
        merge_outputs(args)
    elif args.task:
        run_task(args)
    else:
        raise SystemExit("one of --task or --merge is required")


if __name__ == "__main__":
    main()
