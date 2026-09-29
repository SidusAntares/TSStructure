#!/usr/bin/env python3
"""Frozen-inference geometry and pseudo-quality audit for V2 and V2-Clean."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_representation_chain_audit import (
    _pad_pixel_collate,
    build_audit_datasets,
    seed_all,
)


EPS = 1e-12
METHOD_ROOTS = {
    "v2": "structure_proto_v2_4tasks_seed1",
    "v2clean": "structure_proto_v2clean_4tasks_seed1",
}
TASKS = {
    "FR1_FR2": ("FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
    "DK1_AT1": ("DK1", "denmark/32VNH/2017", "AT1", "austria/33UVP/2017"),
}
REPRESENTATIONS = ("ordered128", "strength16", "concentration16", "response32", "instance128")
SUMMARY_FIELDS = (
    "method", "task", "checkpoint", "representation",
    "source_centroid_f1", "target_oracle_centroid_f1",
    "source_to_target_centroid_f1", "mean_source_sep", "mean_target_sep",
    "mean_intra_ratio", "mean_cross_domain_margin",
    "negative_margin_class_count", "all_target_macro_f1",
    "pseudo_coverage", "accepted_pseudo_accuracy",
)


def _macro_f1(truth, prediction, classes):
    return float(f1_score(
        np.asarray(truth), np.asarray(prediction), labels=np.arange(classes),
        average="macro", zero_division=0,
    ))


def source_zscore(source_train, arrays, eps=1e-6):
    source_train = np.asarray(source_train, dtype=np.float64)
    mean = source_train.mean(axis=0)
    std = np.maximum(source_train.std(axis=0), float(eps))
    return {
        name: (np.asarray(value, dtype=np.float64) - mean) / std
        for name, value in arrays.items()
    }, mean, std


def _centers(features, labels, num_classes):
    features, labels = np.asarray(features), np.asarray(labels, dtype=np.int64)
    centers = np.full((num_classes, features.shape[1]), np.nan, dtype=np.float64)
    for class_id in range(num_classes):
        chosen = labels == class_id
        if chosen.any():
            centers[class_id] = features[chosen].mean(axis=0)
    return centers


def _nearest(features, centers):
    distance = np.linalg.norm(features[:, None] - centers[None], axis=-1)
    distance[:, ~np.isfinite(centers).all(axis=1)] = np.inf
    return distance.argmin(axis=1)


def leave_one_out_centroid_predictions(features, labels, num_classes):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    sums = np.stack([
        features[labels == class_id].sum(axis=0) for class_id in range(num_classes)
    ])
    counts = np.bincount(labels, minlength=num_classes)
    predictions = np.empty(labels.shape, dtype=np.int64)
    for index, (feature, label) in enumerate(zip(features, labels)):
        centers = np.divide(
            sums, counts[:, None],
            out=np.full_like(sums, np.nan), where=counts[:, None] > 0,
        )
        if counts[label] > 1:
            centers[label] = (sums[label] - feature) / (counts[label] - 1)
        else:
            centers[label] = np.nan
        predictions[index] = _nearest(feature[None], centers)[0]
    return predictions


def audit_representation(
    source_train, source_train_labels, source_val, source_val_labels,
    target_val, target_val_labels, class_names,
):
    num_classes = len(class_names)
    standardized, _, _ = source_zscore(source_train, {
        "source_train": source_train, "source_val": source_val,
        "target_val": target_val,
    })
    source_train = standardized["source_train"]
    source_val = standardized["source_val"]
    target_val = standardized["target_val"]
    source_train_labels = np.asarray(source_train_labels, dtype=np.int64)
    source_val_labels = np.asarray(source_val_labels, dtype=np.int64)
    target_val_labels = np.asarray(target_val_labels, dtype=np.int64)
    source_centers = _centers(source_train, source_train_labels, num_classes)
    target_centers = _centers(target_val, target_val_labels, num_classes)
    source_prediction = _nearest(source_val, source_centers)
    target_prediction = _nearest(target_val, source_centers)
    target_oracle_prediction = leave_one_out_centroid_predictions(
        target_val, target_val_labels, num_classes,
    )
    center_distance = np.linalg.norm(
        target_centers[:, None] - source_centers[None], axis=-1,
    )
    class_rows = []
    for class_id, class_name in enumerate(class_names):
        source_selected = source_train_labels == class_id
        target_selected = target_val_labels == class_id
        if not source_selected.any() or not target_selected.any():
            continue
        source_radius = float(np.linalg.norm(
            source_train[source_selected] - source_centers[class_id], axis=1,
        ).mean())
        target_radius = float(np.linalg.norm(
            target_val[target_selected] - target_centers[class_id], axis=1,
        ).mean())
        source_other = np.linalg.norm(source_centers - source_centers[class_id], axis=1)
        target_other = np.linalg.norm(target_centers - target_centers[class_id], axis=1)
        source_other[class_id] = np.inf
        target_other[class_id] = np.inf
        source_other[~np.isfinite(source_centers).all(axis=1)] = np.inf
        target_other[~np.isfinite(target_centers).all(axis=1)] = np.inf
        source_competitor = int(source_other.argmin())
        target_competitor = int(target_other.argmin())
        wrong = center_distance[class_id].copy()
        wrong[class_id] = np.inf
        wrong[~np.isfinite(source_centers).all(axis=1)] = np.inf
        wrong_class = int(wrong.argmin())
        same_distance = float(center_distance[class_id, class_id])
        wrong_distance = float(wrong[wrong_class])
        class_rows.append({
            "class_id": class_id, "class": class_name,
            "source_radius": source_radius, "target_radius": target_radius,
            "intra_ratio": target_radius / (source_radius + EPS),
            "nearest_source_competitor": class_names[source_competitor],
            "nearest_target_competitor": class_names[target_competitor],
            "source_separation": float(source_other[source_competitor]) / (source_radius + EPS),
            "target_separation": float(target_other[target_competitor]) / (target_radius + EPS),
            "same_class_distance": same_distance,
            "nearest_wrong_source_distance": wrong_distance,
            "cross_domain_margin": wrong_distance - same_distance,
            "nearest_source_class_for_target_c": class_names[int(center_distance[class_id].argmin())],
        })
    summary = {
        "source_centroid_f1": _macro_f1(
            source_val_labels, source_prediction, num_classes,
        ),
        "target_oracle_centroid_f1": _macro_f1(
            target_val_labels, target_oracle_prediction, num_classes,
        ),
        "source_to_target_centroid_f1": _macro_f1(
            target_val_labels, target_prediction, num_classes,
        ),
        "mean_source_sep": float(np.mean([row["source_separation"] for row in class_rows])),
        "mean_target_sep": float(np.mean([row["target_separation"] for row in class_rows])),
        "mean_intra_ratio": float(np.mean([row["intra_ratio"] for row in class_rows])),
        "mean_cross_domain_margin": float(np.mean([
            row["cross_domain_margin"] for row in class_rows
        ])),
        "negative_margin_class_count": int(sum(
            row["cross_domain_margin"] < 0 for row in class_rows
        )),
    }
    return {
        "summary": summary, "class_rows": class_rows,
        "distance_matrix": center_distance,
        "target_confusion": confusion_matrix(
            target_val_labels, target_prediction, labels=np.arange(num_classes),
        ),
    }


def extract_representations(output):
    response = torch.cat((
        output["shapelet_strength"], output["shapelet_concentration"],
    ), dim=-1)
    if not torch.allclose(response, output["shapelet_response"], atol=1e-6, rtol=1e-6):
        raise RuntimeError("shapelet_response is not strength+concentration")
    return {
        "ordered128": output["shapelet_similarity"].flatten(1),
        "strength16": output["shapelet_strength"],
        "concentration16": output["shapelet_concentration"],
        "response32": response,
        "instance128": output["instance_feature"],
    }


def pseudo_quality(logits, truth, class_names, threshold=.9):
    probability = torch.as_tensor(logits).float().softmax(1)
    truth = torch.as_tensor(truth).long()
    confidence, prediction = probability.max(1)
    accepted = confidence > float(threshold)
    correct = prediction == truth
    classes = len(class_names)
    selected_truth = truth[accepted].cpu().numpy()
    selected_prediction = prediction[accepted].cpu().numpy()
    summary = {
        "all_target_accuracy": float(correct.float().mean()),
        "all_target_macro_f1": _macro_f1(
            truth.cpu().numpy(), prediction.cpu().numpy(), classes,
        ),
        "pseudo_coverage": float(accepted.float().mean()),
        "accepted_pseudo_accuracy": (
            float(correct[accepted].float().mean()) if accepted.any() else float("nan")
        ),
        "accepted_pseudo_macro_f1": (
            _macro_f1(selected_truth, selected_prediction, classes)
            if accepted.any() else float("nan")
        ),
        "mean_confidence": float(confidence.mean()),
        "correct_mean_confidence": (
            float(confidence[correct].mean()) if correct.any() else float("nan")
        ),
        "wrong_mean_confidence": (
            float(confidence[~correct].mean()) if (~correct).any() else float("nan")
        ),
    }
    true_rows, pred_rows = [], []
    for class_id, class_name in enumerate(class_names):
        true_mask = truth == class_id
        true_accepted = true_mask & accepted
        pred_accepted = (prediction == class_id) & accepted
        true_rows.append({
            "class_id": class_id, "class": class_name,
            "support": int(true_mask.sum()),
            "accepted_count": int(true_accepted.sum()),
            "coverage": float(true_accepted.sum() / true_mask.sum().clamp_min(1)),
            "accepted_correct_rate": (
                float(correct[true_accepted].float().mean())
                if true_accepted.any() else float("nan")
            ),
        })
        pred_rows.append({
            "class_id": class_id, "class": class_name,
            "accepted_count": int(pred_accepted.sum()),
            "precision": (
                float((truth[pred_accepted] == class_id).float().mean())
                if pred_accepted.any() else float("nan")
            ),
        })
    wrong_accepted = accepted & ~correct
    wrong_confusion = confusion_matrix(
        truth[wrong_accepted].cpu().numpy(), prediction[wrong_accepted].cpu().numpy(),
        labels=np.arange(classes),
    )
    return {
        "summary": summary, "true_rows": true_rows, "pred_rows": pred_rows,
        "wrong_confusion": wrong_confusion, "prediction": prediction,
        "confidence": confidence, "accepted": accepted,
    }


def _pixel_budget_batches(dataset, batch_size, pixel_budget):
    counts = [int(shape[2]) for shape in dataset.get_shapes()]
    order = sorted(range(len(counts)), key=lambda index: (counts[index], index))
    batches, current = [], []
    for index in order:
        candidate = current + [index]
        if current and (
            len(candidate) > int(batch_size)
            or len(candidate) * max(counts[item] for item in candidate) > int(pixel_budget)
        ):
            batches.append(current)
            current = [index]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _loader(dataset, batch_size, pixel_budget, num_workers):
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=_pixel_budget_batches(dataset, batch_size, pixel_budget),
        collate_fn=_pad_pixel_collate, num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def _load_model(path, device):
    from train import create_model
    packet = torch.load(path, map_location=device, weights_only=False)
    raw = dict(packet.get("config") or {})
    if not raw:
        raise ValueError(f"checkpoint config missing: {path}")
    raw.setdefault("shape_representation", "current")
    raw.setdefault("shape_injection", "current_query")
    raw.setdefault("structure_shift_mode", "none")
    config = SimpleNamespace(**raw)
    model = create_model(config)
    result = model.load_state_dict(packet["state_dict"], strict=False)
    allowed_unexpected = ("instance_prototype_bank.",)
    unexpected = [
        key for key in result.unexpected_keys
        if not key.startswith(allowed_unexpected)
    ]
    if result.missing_keys or unexpected:
        raise RuntimeError(
            f"checkpoint incompatible: missing={result.missing_keys}, unexpected={unexpected}"
        )
    return model.to(device).eval(), config, packet


@torch.no_grad()
def _extract(model, dataset, device, shift, batch_size, pixel_budget, num_workers):
    collected = {name: [] for name in REPRESENTATIONS}
    logits, labels = [], []
    for batch in _loader(dataset, batch_size, pixel_budget, num_workers):
        output = model.forward_with_temporal_shift(
            batch["pixels"].to(device, non_blocking=True),
            batch["valid_pixels"].to(device, non_blocking=True),
            batch["positions"].to(device, non_blocking=True),
            batch["extra"].to(device, non_blocking=True),
            temporal_shift=shift, return_dict=True,
        )
        representations = extract_representations(output)
        for name, value in representations.items():
            collected[name].append(value.detach().float().cpu())
        logits.append(output["logits"].detach().float().cpu())
        labels.append(batch["label"].long().cpu())
    return {
        "representations": {
            name: torch.cat(parts).numpy() for name, parts in collected.items()
        },
        "logits": torch.cat(logits), "labels": torch.cat(labels).numpy(),
    }


def _write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(fields or (rows[0].keys() if rows else ()))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _matrix_rows(matrix, row_names, column_names, row_key):
    rows = []
    for row_id, row_name in enumerate(row_names):
        for column_id, column_name in enumerate(column_names):
            rows.append({
                row_key: row_name, "source_class" if row_key == "target_class" else "predicted_class": column_name,
                "value": float(matrix[row_id, column_id]),
            })
    return rows


def _checkpoint_paths(method, task, root):
    source_alias = TASKS[task][0]
    base = Path(root) / METHOD_ROOTS[method]
    return {
        "source_checkpoint": base / "source" / f"source_{source_alias}_seed1" / "fold_0" / "model.pt",
        "uda_best": base / "uda" / f"{task}_seed1" / "fold_0" / "model.pt",
    }


def _log_path(method, task, logs_root):
    return Path(logs_root) / METHOD_ROOTS[method] / f"{task}.log"


def _key_values(line):
    return {
        key: value for key, value in re.findall(r"(?:^|\|)([A-Za-z0-9_]+)=([^|\s]+)", line)
    }


def parse_training_timeline(path, method):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    marker = "[UDA START]" if method == "v2" else "[V2-CLEAN UDA START]"
    if marker in text:
        text = text[text.index(marker):]
    rows = defaultdict(dict)
    pseudo_losses = defaultdict(list)
    current_epoch = None
    initial_match = re.search(r"INITIAL_SHIFT\|[^\n]*shift_days=([-+0-9.]+)", text)
    initial_shift = float(initial_match.group(1)) if initial_match else float("nan")
    for line in text.splitlines():
        if line.startswith("EPOCH_SHIFT|"):
            values = _key_values(line)
            current_epoch = int(values["epoch"])
            rows[current_epoch]["selected_shift"] = float(values["target_to_source_days"])
        elif line.startswith(("STRUCTURE_PROTO_DA|", "STRUCTURE_V2CLEAN_DA|")):
            values = _key_values(line)
            if current_epoch is not None and "loss_pseudo_target" in values:
                pseudo_losses[current_epoch].append(float(values["loss_pseudo_target"]))
        elif line.startswith("SHAPE_AUX_EPOCH|"):
            values = _key_values(line)
            epoch = int(values["epoch"])
            if "source_accuracy" in values:
                rows[epoch]["source_shape_accuracy"] = float(values["source_accuracy"])
            if "target_pseudo_accuracy" in values:
                rows[epoch]["shape_vs_pseudo_accuracy"] = float(values["target_pseudo_accuracy"])
        elif line.startswith(("SHAPE_V2_EPOCH|", "SHAPE_V2CLEAN_EPOCH|")):
            values = _key_values(line)
            epoch = int(values["epoch"])
            for key in (
                "source_shape_accuracy", "pseudo_coverage", "shape_align_center_gap",
                "shape_response_effective_rank", "shape_align_valid_classes",
            ):
                if key in values:
                    rows[epoch][key] = float(values[key])
        elif "Validation result:" in line and current_epoch is not None:
            match = re.search(r"f1=([0-9.]+)", line)
            if match:
                rows[current_epoch]["validation_f1"] = float(match.group(1))
    for epoch, values in rows.items():
        values["initial_shift"] = initial_shift
        values.setdefault("selected_shift", initial_shift)
        values["loss_pseudo_target"] = (
            float(np.mean(pseudo_losses[epoch])) if pseudo_losses[epoch] else ""
        )
        values["epoch"] = epoch
    best_epoch = max(
        (epoch for epoch in rows if "validation_f1" in rows[epoch]),
        key=lambda epoch: rows[epoch]["validation_f1"], default=None,
    )
    for epoch in rows:
        rows[epoch]["best_validation_epoch"] = epoch == best_epoch
    return [rows[epoch] for epoch in sorted(rows)]


def _initial_shift(timeline):
    return timeline[0].get("initial_shift", 0) if timeline else 0


def _checkpoint_audit(
    method, task, stage, checkpoint, output, data_root, device,
    batch_size, pixel_budget, num_workers, timeline,
):
    model, config, packet = _load_model(checkpoint, device)
    _, source_name, _, target_name = TASKS[task]
    datasets, split = build_audit_datasets(
        config, source_name, target_name, data_root, int(config.seed),
    )
    target_shift = (
        packet.get("global_temporal_shift", _initial_shift(timeline))
        if stage == "uda_best" else _initial_shift(timeline)
    )
    extracted = {
        "source_train": _extract(
            model, datasets["source_train"], device, 0,
            batch_size, pixel_budget, num_workers,
        ),
        "source_val": _extract(
            model, datasets["source_val"], device, 0,
            batch_size, pixel_budget, num_workers,
        ),
        "target_val": _extract(
            model, datasets["target_val"], device, target_shift,
            batch_size, pixel_budget, num_workers,
        ),
    }
    representation_rows, class_rows, distance_rows, confusion_rows = [], [], [], []
    for representation in REPRESENTATIONS:
        result = audit_representation(
            extracted["source_train"]["representations"][representation],
            extracted["source_train"]["labels"],
            extracted["source_val"]["representations"][representation],
            extracted["source_val"]["labels"],
            extracted["target_val"]["representations"][representation],
            extracted["target_val"]["labels"], config.classes,
        )
        representation_rows.append({
            "method": method, "task": task, "checkpoint": stage,
            "representation": representation, **result["summary"],
        })
        class_rows.extend({
            "method": method, "task": task, "checkpoint": stage,
            "representation": representation, **row,
        } for row in result["class_rows"])
        distance_rows.extend({
            "method": method, "task": task, "checkpoint": stage,
            "representation": representation, **row,
        } for row in _matrix_rows(
            result["distance_matrix"], config.classes, config.classes, "target_class",
        ))
        confusion_rows.extend({
            "method": method, "task": task, "checkpoint": stage,
            "representation": representation, **row,
        } for row in _matrix_rows(
            result["target_confusion"], config.classes, config.classes, "true_class",
        ))
    pseudo = pseudo_quality(
        extracted["target_val"]["logits"], extracted["target_val"]["labels"],
        config.classes, threshold=.9,
    )
    for row in representation_rows:
        row.update(pseudo["summary"])
    _write_csv(output / "representation_summary.csv", representation_rows)
    _write_csv(output / "class_geometry.csv", class_rows)
    _write_csv(output / "centroid_distance_matrix.csv", distance_rows)
    _write_csv(output / "centroid_confusion.csv", confusion_rows)
    _write_csv(output / "pseudo_summary.csv", [{
        "method": method, "task": task, "checkpoint": stage,
        **pseudo["summary"], "temporal_shift": target_shift,
    }])
    _write_csv(output / "pseudo_by_true_class.csv", pseudo["true_rows"])
    _write_csv(output / "pseudo_by_pred_class.csv", pseudo["pred_rows"])
    _write_csv(
        output / "high_conf_wrong_confusion.csv",
        _matrix_rows(pseudo["wrong_confusion"], config.classes, config.classes, "true_class"),
    )
    summary = {
        "method": method, "task": task, "checkpoint": stage,
        "checkpoint_path": str(checkpoint), "target_temporal_shift": target_shift,
        "target_test_accessed": False,
        "target_labels_used_for_offline_diagnostics_only": True,
        "split_counts": {name: len(value) for name, value in split.items()},
        "representations": representation_rows, "pseudo": pseudo["summary"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return representation_rows, class_rows


def _delta_rows(method, task, source_rows, best_rows):
    source = {row["representation"]: row for row in source_rows}
    result = []
    for best in best_rows:
        base = source[best["representation"]]
        row = {"method": method, "task": task, "representation": best["representation"]}
        for key in (
            "source_to_target_centroid_f1", "target_oracle_centroid_f1",
            "mean_target_sep", "mean_intra_ratio", "mean_cross_domain_margin",
            "negative_margin_class_count", "pseudo_coverage",
            "accepted_pseudo_accuracy", "all_target_macro_f1",
        ):
            row[f"delta_{key}"] = best[key] - base[key]
        result.append(row)
    return result


def run_job(args):
    paths = _checkpoint_paths(args.method, args.task, args.checkpoint_root)
    missing = [str(path.resolve()) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoint(s): " + ", ".join(missing))
    log_path = _log_path(args.method, args.task, args.logs_root)
    if not log_path.is_file():
        raise FileNotFoundError(f"training log not found: {log_path.resolve()}")
    timeline = parse_training_timeline(log_path, args.method)
    task_root = Path(args.output_root) / args.method / args.task
    source_rows, source_classes = _checkpoint_audit(
        args.method, args.task, "source_checkpoint", paths["source_checkpoint"],
        task_root / "source_checkpoint", args.data_root, torch.device(args.device),
        args.batch_size, args.pixel_budget, args.num_workers, timeline,
    )
    best_rows, best_classes = _checkpoint_audit(
        args.method, args.task, "uda_best", paths["uda_best"],
        task_root / "uda_best", args.data_root, torch.device(args.device),
        args.batch_size, args.pixel_budget, args.num_workers, timeline,
    )
    _write_csv(
        task_root / "comparison" / "checkpoint_delta.csv",
        _delta_rows(args.method, args.task, source_rows, best_rows),
    )
    _write_csv(task_root / "training_timeline.csv", timeline)
    _write_csv(task_root / "comparison" / "class_comparison.csv", source_classes + best_classes)
    print(f"V2_FAILURE_AUDIT_FINISHED|method={args.method}|task={args.task}|output={task_root}")


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def combine(args):
    output = Path(args.output_root)
    summary_rows, class_rows = [], []
    for method in METHOD_ROOTS:
        for task in TASKS:
            for stage in ("source_checkpoint", "uda_best"):
                root = output / method / task / stage
                summary_rows.extend(_read_csv(root / "representation_summary.csv"))
                class_rows.extend(_read_csv(root / "class_geometry.csv"))
    combined = output / "combined"
    _write_csv(combined / "comparison.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(combined / "class_comparison.csv", class_rows)
    lines = ["# V2 vs V2Clean Failure Audit", "", "Measured checkpoint metrics only.", ""]
    lines.extend((
        "| Method | Task | Checkpoint | Representation | Target oracle F1 | Source-to-target F1 | Intra ratio | Cross-domain margin | Negative classes | Pseudo coverage | Accepted accuracy |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for row in summary_rows:
        if row["representation"] != "response32":
            continue
        lines.append(
            f"| {row['method']} | {row['task']} | {row['checkpoint']} | response32 | "
            f"{row['target_oracle_centroid_f1']} | {row['source_to_target_centroid_f1']} | "
            f"{row['mean_intra_ratio']} | {row['mean_cross_domain_margin']} | "
            f"{row['negative_margin_class_count']} | {row['pseudo_coverage']} | "
            f"{row['accepted_pseudo_accuracy']} |"
        )
    (combined / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V2_FAILURE_AUDIT_COMBINED|output={combined}")


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--method", choices=tuple(METHOD_ROOTS))
    value.add_argument("--task", choices=tuple(TASKS))
    value.add_argument("--combine", action="store_true")
    value.add_argument("--checkpoint-root", default="outputs")
    value.add_argument("--logs-root", default="logs")
    value.add_argument("--output-root", default="outputs/v2_v2clean_failure_audit_seed1")
    value.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    value.add_argument("--device", default="cuda")
    value.add_argument("--batch-size", type=int, default=128)
    value.add_argument("--pixel-budget", type=int, default=8192)
    value.add_argument("--num-workers", type=int, default=0)
    value.add_argument("--seed", type=int, default=1)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    seed_all(arguments.seed)
    if arguments.combine:
        combine(arguments)
    elif arguments.method and arguments.task:
        run_job(arguments)
    else:
        raise SystemExit("provide --method and --task, or --combine")
