#!/usr/bin/env python3
"""Read-only audit of phase dependence in trained phase-moment source models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_representation_chain_audit import (
    DOMAINS, TASKS, build_audit_datasets, deterministic_loader,
    fit_source_probe, load_source_model, seed_all, target_oracle_probe,
    write_csv,
)
from methods.structure_da.phase_equivariance import rotate_phase_moments


AUDIT_SPLITS = ("source_train", "source_val", "target_val")
REPRESENTATIONS = ("Current", "Phase", "Current+Phase")
DELTAS = (-60, -30, 0, 30, 60)
PROBE_FIELDS = (
    "task", "representation", "source_val_macro_f1",
    "source_to_target_macro_f1", "target_oracle_macro_f1", "transfer_gap",
)
PROBE_CLASS_FIELDS = (
    "task", "representation", "class", "source_support", "target_support",
    "source_val_f1", "source_to_target_f1", "target_oracle_f1",
)
ROTATION_FIELDS = (
    "task", "domain", "delta", "macro_f1", "accuracy",
    "prediction_consistency_vs_delta0",
)
ROTATION_CLASS_FIELDS = (
    "task", "domain", "delta", "class", "support", "f1",
)


def phase_representations(structure):
    current = torch.cat((
        structure["shapelet_strength"], structure["shapelet_concentration"],
    ), dim=-1)
    phase = structure["shapelet_phase_moments"]
    combined = structure["shapelet_response"]
    if current.shape[-1] != 32 or phase.shape[-1] != 64 or combined.shape[-1] != 96:
        raise ValueError("phase audit requires Current=32, Phase=64, Current+Phase=96")
    torch.testing.assert_close(combined, torch.cat((current, phase), dim=-1))
    return {"Current": current, "Phase": phase, "Current+Phase": combined}


def rotate_structure_phase(structure, delta):
    rotated = dict(structure)
    phase = rotate_phase_moments(
        structure["shapelet_phase_moments"],
        torch.full(
            (structure["shapelet_phase_moments"].shape[0],), float(delta),
            device=structure["shapelet_phase_moments"].device,
            dtype=structure["shapelet_phase_moments"].dtype,
        ),
        shapelet_count=16, harmonics=(1, 2), period_days=365.,
    )
    current = torch.cat((
        structure["shapelet_strength"], structure["shapelet_concentration"],
    ), dim=-1)
    response = torch.cat((current, phase), dim=-1)
    rotated["shapelet_phase_moments"] = phase
    rotated["shapelet_response"] = response
    rotated["shape_class_token"] = None
    return rotated


def classify_prepared_phase(model, spatial, positions, structure, delta):
    rotated = rotate_structure_phase(structure, delta)
    rotated["shape_class_token"] = model.structure_branch.response_to_query(
        rotated["shapelet_response"],
    )
    return model._output_from_prepared_structure(spatial, positions, rotated)


@torch.no_grad()
def phase_intervention_forward(
    model, pixels, valid_pixels, positions, extra, delta,
):
    spatial = model.spatial_encoder(pixels, valid_pixels, extra)
    structure = model.prepare_structure(spatial, positions, temporal_shift=0)
    return classify_prepared_phase(
        model, spatial, positions, structure, delta,
    ), structure


def evaluate_probe(
    source_train, source_train_labels, source_val, source_val_labels,
    target_val, target_val_labels, class_ids, seed,
):
    source = fit_source_probe(
        source_train, source_train_labels, source_val, source_val_labels,
        target_val, target_val_labels, class_ids,
    )
    oracle = target_oracle_probe(target_val, target_val_labels, class_ids, seed)
    return {
        **source,
        "source_probe": source["estimator"],
        "target_oracle_macro_f1": oracle["macro_f1"],
        "target_oracle_per_class_f1": oracle["per_class_f1"],
        "target_oracle_folds": oracle["folds"],
    }


@torch.no_grad()
def extract_dataset(model, dataset, batch_size, num_workers, device, pixel_budget):
    representations = {name: [] for name in REPRESENTATIONS}
    predictions = {delta: [] for delta in DELTAS}
    labels = []
    for batch in deterministic_loader(
        dataset, batch_size, num_workers, pixel_budget=pixel_budget,
    ):
        pixels = batch["pixels"].to(device, non_blocking=True)
        valid = batch["valid_pixels"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        extra = batch["extra"].to(device, non_blocking=True)
        spatial = model.spatial_encoder(pixels, valid, extra)
        structure = model.prepare_structure(spatial, positions, temporal_shift=0)
        for name, value in phase_representations(structure).items():
            representations[name].append(value.detach().float().cpu())
        for delta in DELTAS:
            output = classify_prepared_phase(
                model, spatial, positions, structure, delta,
            )
            predictions[delta].append(output["logits"].argmax(1).cpu())
        labels.append(batch["label"].long().cpu())
    return {
        "representations": {
            name: torch.cat(values).numpy() for name, values in representations.items()
        },
        "predictions": {
            delta: torch.cat(values).numpy() for delta, values in predictions.items()
        },
        "labels": torch.cat(labels).numpy(),
    }


def _per_class(labels, predictions, class_ids):
    return f1_score(
        labels, predictions, labels=class_ids, average=None, zero_division=0,
    )


def evaluate_task(task, class_names, extracted, seed):
    class_ids = np.arange(len(class_names), dtype=np.int64)
    source_train, source_val, target_val = (
        extracted[name] for name in AUDIT_SPLITS
    )
    probe_rows, probe_class_rows = [], []
    for representation in REPRESENTATIONS:
        result = evaluate_probe(
            source_train["representations"][representation], source_train["labels"],
            source_val["representations"][representation], source_val["labels"],
            target_val["representations"][representation], target_val["labels"],
            class_ids, seed,
        )
        probe_rows.append({
            "task": task, "representation": representation,
            "source_val_macro_f1": result["source_val_macro_f1"],
            "source_to_target_macro_f1": result["source_to_target_macro_f1"],
            "target_oracle_macro_f1": result["target_oracle_macro_f1"],
            "transfer_gap": (
                result["target_oracle_macro_f1"]
                - result["source_to_target_macro_f1"]
            ),
        })
        source_support = np.bincount(source_val["labels"], minlength=len(class_names))
        target_support = np.bincount(target_val["labels"], minlength=len(class_names))
        for class_id, class_name in enumerate(class_names):
            probe_class_rows.append({
                "task": task, "representation": representation, "class": class_name,
                "source_support": int(source_support[class_id]),
                "target_support": int(target_support[class_id]),
                "source_val_f1": result["source_val_per_class_f1"][class_id],
                "source_to_target_f1": result["source_to_target_per_class_f1"][class_id],
                "target_oracle_f1": result["target_oracle_per_class_f1"][class_id],
            })
    rotation_rows, rotation_class_rows = [], []
    for domain, values in (("source", source_val), ("target", target_val)):
        baseline = values["predictions"][0]
        support = np.bincount(values["labels"], minlength=len(class_names))
        for delta in DELTAS:
            prediction = values["predictions"][delta]
            per_class = _per_class(values["labels"], prediction, class_ids)
            rotation_rows.append({
                "task": task, "domain": domain, "delta": delta,
                "macro_f1": float(per_class.mean()),
                "accuracy": accuracy_score(values["labels"], prediction),
                "prediction_consistency_vs_delta0": float((prediction == baseline).mean()),
            })
            for class_id, class_name in enumerate(class_names):
                rotation_class_rows.append({
                    "task": task, "domain": domain, "delta": delta,
                    "class": class_name, "support": int(support[class_id]),
                    "f1": per_class[class_id],
                })
    return probe_rows, probe_class_rows, rotation_rows, rotation_class_rows


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_task(args):
    source_alias, target_alias = TASKS[args.task]
    checkpoint = (
        Path(args.checkpoint_root) / f"source_{source_alias}_seed1"
        / "fold_0" / "model.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"P source checkpoint missing: {checkpoint.resolve()}")
    seed_all(args.seed)
    device = torch.device(args.device)
    model, config = load_source_model(checkpoint, device, "phase_moment")
    datasets, split = build_audit_datasets(
        config, DOMAINS[source_alias], DOMAINS[target_alias],
        args.data_root, args.seed,
    )
    if tuple(datasets) != AUDIT_SPLITS:
        raise RuntimeError(f"unexpected audit splits: {tuple(datasets)}")
    extracted = {
        name: extract_dataset(
            model, dataset, args.batch_size, args.num_workers, device,
            args.pixel_budget,
        ) for name, dataset in datasets.items()
    }
    rows = evaluate_task(args.task, list(config.classes), extracted, args.seed)
    task_root = Path(args.output_root) / args.task
    names = (
        ("phase_probe_summary.csv", PROBE_FIELDS),
        ("phase_probe_per_class.csv", PROBE_CLASS_FIELDS),
        ("phase_rotation_summary.csv", ROTATION_FIELDS),
        ("phase_rotation_per_class.csv", ROTATION_CLASS_FIELDS),
    )
    for (filename, fields), values in zip(names, rows):
        write_csv(task_root / filename, values, fields)
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "audit_manifest.json").write_text(json.dumps({
        "task": args.task, "checkpoint": str(checkpoint),
        "checkpoint_role": "P_source", "splits": list(AUDIT_SPLITS),
        "split_counts": {name: len(indices) for name, indices in split.items()},
        "audit_splits_only": list(AUDIT_SPLITS),
        "test_accessed": False, "training": False, "backward": False,
    }, indent=2), encoding="utf-8")
    print(f"PHASE_DEPENDENCY_FINISHED|task={args.task}|output={task_root}")


def merge_outputs(args):
    root = Path(args.output_root)
    specs = (
        ("phase_probe_summary.csv", PROBE_FIELDS),
        ("phase_probe_per_class.csv", PROBE_CLASS_FIELDS),
        ("phase_rotation_summary.csv", ROTATION_FIELDS),
        ("phase_rotation_per_class.csv", ROTATION_CLASS_FIELDS),
    )
    merged = {}
    for filename, fields in specs:
        rows = []
        for task in TASKS:
            path = root / task / filename
            if not path.is_file():
                raise FileNotFoundError(f"incomplete phase audit: {path.resolve()}")
            rows.extend(_read_csv(path))
        write_csv(root / filename, rows, fields)
        merged[filename] = rows
    summary = ["PHASE_DEPENDENCY_AUDIT", "training=false", "test_accessed=false"]
    for row in merged["phase_probe_summary.csv"]:
        summary.append("|".join(f"{key}={value}" for key, value in row.items()))
    for row in merged["phase_rotation_summary.csv"]:
        summary.append("|".join(f"{key}={value}" for key, value in row.items()))
    (root / "audit_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"PHASE_DEPENDENCY_MERGED|output={root}")


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
        "--output-root",
        default="outputs/structure_phase_dependency_audit_seed1",
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
        raise SystemExit("provide --task or --merge")


if __name__ == "__main__":
    main()
