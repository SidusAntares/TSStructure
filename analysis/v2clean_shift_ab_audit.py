#!/usr/bin/env python3
"""Frozen shift-sensitivity and failure audits for the V2-Clean minority study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.v2_v2clean_failure_audit import (
    _extract,
    _load_model,
    _loader,
    audit_representation,
    build_audit_datasets,
    integer_day_shift,
    pseudo_quality,
)


TASKS = {
    "DK1_AT1": ("denmark/32VNH/2017", "austria/33UVP/2017"),
    "FR1_FR2": ("france/30TXT/2017", "france/31TCJ/2017"),
}
DELTAS = tuple(range(-60, 61, 15))
THRESHOLDS = (.5, .7, .8, .9, .95)


def _write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _margin(logits, labels=None):
    top = logits.topk(2, dim=1).values
    top_margin = top[:, 0] - top[:, 1]
    if labels is None:
        return top_margin
    true = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], -torch.inf)
    return true - masked.max(1).values


def isolated_structure_shift_logits(model, spatial, positions, delta):
    """Use identical temporal positions and change only the structure query."""
    delta = integer_day_shift(delta)
    context = model.prepare_structure_context(spatial, positions)
    structure_zero = model.structure_branch.forward_from_context(
        context, temporal_shift=0, include_legacy_query=True,
    )
    structure_delta = model.structure_branch.forward_from_context(
        context, temporal_shift=delta, include_legacy_query=True,
    )
    shifted_positions = positions + delta
    base_instance = model._encode_instance(
        spatial, shifted_positions, structure_zero,
    )
    delta_instance = model._encode_instance(
        spatial, shifted_positions, structure_delta,
    )
    return (
        model.decoder(base_instance), model.decoder(delta_instance),
        structure_zero, structure_delta,
    )


def _mean(values):
    return float(torch.cat(values).float().mean()) if values else float("nan")


@torch.no_grad()
def run_shift(args):
    device = torch.device(args.device)
    model, config, _ = _load_model(Path(args.checkpoint), device)
    source, target = TASKS[args.task]
    datasets, _ = build_audit_datasets(config, source, target, args.data_root, args.seed)
    output = Path(args.output_root) / args.task
    global_rows, class_rows, shape_transitions, final_transitions = [], [], [], []
    for delta in DELTAS:
        records = defaultdict(list)
        shape_counter, final_counter = Counter(), Counter()
        for batch in _loader(
            datasets["target_val"], args.batch_size, args.pixel_budget, args.num_workers,
        ):
            pixels = batch["pixels"].to(device)
            valid = batch["valid_pixels"].to(device)
            positions = batch["positions"].to(device)
            extra = batch["extra"].to(device)
            labels = batch["label"].long().to(device)
            spatial = model.spatial_encoder(pixels, valid, extra)
            logits0, logitsd, structure0, structured = isolated_structure_shift_logits(
                model, spatial, positions, delta,
            )
            response0, responsed = (
                structure0["shapelet_response"], structured["shapelet_response"],
            )
            shape0 = model.shape_classifier(response0)
            shaped = model.shape_classifier(responsed)
            shape_pred0, shape_predd = shape0.argmax(1), shaped.argmax(1)
            final_pred0, final_predd = logits0.argmax(1), logitsd.argmax(1)
            for left, right, prefix in (
                (response0, responsed, "response"),
                (structure0["shapelet_strength"], structured["shapelet_strength"], "strength"),
                (structure0["shapelet_concentration"], structured["shapelet_concentration"], "concentration"),
            ):
                records[f"{prefix}_cosine"].append(F.cosine_similarity(left, right))
                records[f"{prefix}_relative_l2"].append(
                    (right - left).norm(dim=1) / left.norm(dim=1).clamp_min(1e-12)
                )
            records["labels"].append(labels)
            records["shape_consistency"].append(shape_pred0 == shape_predd)
            records["shape_true_margin_delta"].append(_margin(shaped, labels) - _margin(shape0, labels))
            records["shape_top_margin_delta"].append(_margin(shaped) - _margin(shape0))
            records["final_consistency"].append(final_pred0 == final_predd)
            records["final_true_margin_delta"].append(_margin(logitsd, labels) - _margin(logits0, labels))
            records["final_top_margin_delta"].append(_margin(logitsd) - _margin(logits0))
            records["final_pred0"].append(final_pred0)
            records["final_predd"].append(final_predd)
            for before, after in zip(shape_pred0.tolist(), shape_predd.tolist()):
                shape_counter[(before, after)] += 1
            for before, after in zip(final_pred0.tolist(), final_predd.tolist()):
                final_counter[(before, after)] += 1
        labels = torch.cat(records["labels"])
        pred0, predd = torch.cat(records["final_pred0"]), torch.cat(records["final_predd"])
        global_rows.append({
            "task": args.task, "delta": delta,
            **{key: _mean(value) for key, value in records.items()
               if key not in ("labels", "final_pred0", "final_predd")},
            "final_macro_f1_case0": f1_score(labels.cpu(), pred0.cpu(), average="macro", zero_division=0),
            "final_macro_f1_case_delta": f1_score(labels.cpu(), predd.cpu(), average="macro", zero_division=0),
            "final_macro_f1_delta": f1_score(labels.cpu(), predd.cpu(), average="macro", zero_division=0)
            - f1_score(labels.cpu(), pred0.cpu(), average="macro", zero_division=0),
        })
        for class_id, class_name in enumerate(config.classes):
            selected = labels == class_id
            if not selected.any():
                continue
            row = {"task": args.task, "delta": delta, "class": class_name, "support": int(selected.sum())}
            for key, values in records.items():
                if key in ("labels", "final_pred0", "final_predd"):
                    continue
                row[key] = float(torch.cat(values)[selected].float().mean())
            class_rows.append(row)
        for counter, rows in ((shape_counter, shape_transitions), (final_counter, final_transitions)):
            rows.extend({
                "task": args.task, "delta": delta,
                "from_class": config.classes[left], "to_class": config.classes[right],
                "count": count,
            } for (left, right), count in sorted(counter.items()))
    _write_csv(output / "shift_sensitivity_global.csv", global_rows)
    _write_csv(output / "shift_sensitivity_by_class.csv", class_rows)
    _write_csv(output / "shape_transition.csv", shape_transitions)
    _write_csv(output / "final_transition.csv", final_transitions)
    hard = sorted(class_rows, key=lambda row: row["response_cosine"])[:10]
    lines = [f"# {args.task} current-response shift sensitivity", "", "## Global", ""]
    lines += [
        f"- delta {row['delta']:+d}: response cosine={row['response_cosine']:.6f}, "
        f"final F1 delta={row['final_macro_f1_delta']:.6f}"
        for row in global_rows
    ]
    lines += ["", "## Lowest class response cosine", ""] + [
        f"- {row['class']} @ {row['delta']:+d}: {row['response_cosine']:.6f}"
        for row in hard
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"SHIFT_SENSITIVITY_FINISHED|task={args.task}|output={output}")


def _checkpoint_shift(packet, stage):
    return integer_day_shift(packet.get("global_temporal_shift", 0)) if stage == "uda_best" else 0


@torch.no_grad()
def run_failure(args):
    device = torch.device(args.device)
    model, config, packet = _load_model(Path(args.checkpoint), device)
    source, target = TASKS[args.task]
    datasets, _ = build_audit_datasets(config, source, target, args.data_root, args.seed)
    shift = _checkpoint_shift(packet, args.stage)
    source_data = _extract(model, datasets["source_train"], device, 0, args.batch_size, args.pixel_budget, args.num_workers)
    target_data = _extract(model, datasets["target_val"], device, shift, args.batch_size, args.pixel_budget, args.num_workers)
    truth = target_data["labels"]
    logits = target_data["logits"]
    prediction = logits.argmax(1).numpy()
    precision, recall, per_f1, support = precision_recall_fscore_support(
        truth, prediction, labels=np.arange(len(config.classes)), zero_division=0,
    )
    geometry = audit_representation(
        source_data["representations"]["response32"], source_data["labels"],
        source_data["representations"]["response32"], source_data["labels"],
        target_data["representations"]["response32"], truth, config.classes,
    )
    for row in geometry["class_rows"]:
        nearest_distance = row["target_separation"] * row["target_radius"]
        row["target_nearest_class_distance"] = nearest_distance
        row["target_radius_over_nearest_class_distance"] = (
            row["target_radius"] / max(nearest_distance, 1e-12)
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    geometry_by_class = {row["class"]: row for row in geometry["class_rows"]}
    class_metrics = [{
        "task": args.task, "variant": args.variant, "stage": args.stage,
        "class": name, "precision": precision[index], "recall": recall[index],
        "f1": per_f1[index], "support": int(support[index]),
        **geometry_by_class.get(name, {}),
    } for index, name in enumerate(config.classes)]
    _write_csv(output / "classification_by_class.csv", class_metrics)
    _write_csv(output / "response32_class_geometry.csv", geometry["class_rows"])
    sweep_rows = []
    for threshold in THRESHOLDS:
        result = pseudo_quality(logits, truth, config.classes, threshold)
        sweep_rows.append({
            "task": args.task, "variant": args.variant, "stage": args.stage,
            "threshold": threshold, **result["summary"],
        })
        if threshold == .9:
            _write_csv(output / "pseudo_by_true_class.csv", result["true_rows"])
            _write_csv(output / "pseudo_by_pred_class.csv", result["pred_rows"])
            wrong_rows = []
            for row_id, row_name in enumerate(config.classes):
                for col_id, col_name in enumerate(config.classes):
                    wrong_rows.append({"true_class": row_name, "predicted_class": col_name, "count": int(result["wrong_confusion"][row_id, col_id])})
            _write_csv(output / "high_conf_wrong_confusion.csv", wrong_rows)
    _write_csv(output / "pseudo_threshold_sweep.csv", sweep_rows)
    summary = {
        "task": args.task, "variant": args.variant, "stage": args.stage,
        "checkpoint": str(args.checkpoint), "temporal_shift": shift,
        "target_macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
        **geometry["summary"], **next(row for row in sweep_rows if row["threshold"] == .9),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"FAILURE_AUDIT_FINISHED|task={args.task}|variant={args.variant}|stage={args.stage}|output={output}")


def run_aggregate(args):
    root = Path(args.root)
    summaries, classes, sweeps = [], [], []
    for variant in ("Base", "A", "B"):
        for task in TASKS:
            for stage in ("source", "uda_best"):
                directory = root / variant / "audit" / task / stage
                if not (directory / "summary.json").is_file():
                    continue
                summaries.append(json.loads((directory / "summary.json").read_text()))
                with (directory / "classification_by_class.csv").open() as stream:
                    classes.extend(csv.DictReader(stream))
                with (directory / "pseudo_threshold_sweep.csv").open() as stream:
                    sweeps.extend(csv.DictReader(stream))
    summary_dir = root / "summary"
    _write_csv(summary_dir / "source_checkpoint_comparison.csv", [row for row in summaries if row["stage"] == "source"])
    _write_csv(summary_dir / "uda_comparison.csv", [row for row in summaries if row["stage"] == "uda_best"])
    _write_csv(summary_dir / "class_comparison.csv", classes)
    _write_csv(summary_dir / "pseudo_threshold_sweep.csv", sweeps)
    lines = ["# V2-Clean minority A/B", "", "Measured results only.", ""]
    lines += [
        f"- {row['variant']} {row['task']} {row['stage']}: target Macro-F1={row['target_macro_f1']:.6f}, "
        f"coverage@0.9={row['pseudo_coverage']:.6f}, accepted accuracy={row['accepted_pseudo_accuracy']:.6f}"
        for row in summaries
    ]
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_shift_summary(args):
    root = Path(args.output_root)
    lines = ["# Current-response shift sensitivity", "", "Measured results only.", ""]
    for task in TASKS:
        path = root / task / "shift_sensitivity_global.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        lines.append(f"## {task}")
        lines.append("")
        lines.extend(
            f"- delta {int(float(row['delta'])):+d}: response cosine={float(row['response_cosine']):.6f}, "
            f"final F1 delta={float(row['final_macro_f1_delta']):.6f}"
            for row in rows
        )
        lines.append("")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser():
    main = argparse.ArgumentParser()
    sub = main.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", choices=tuple(TASKS), required=True)
    common.add_argument("--checkpoint", required=True)
    common.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    common.add_argument("--device", default="cuda")
    common.add_argument("--batch-size", type=int, default=128)
    common.add_argument("--pixel-budget", type=int, default=8192)
    common.add_argument("--num-workers", type=int, default=0)
    common.add_argument("--seed", type=int, default=1)
    shift = sub.add_parser("shift", parents=[common])
    shift.add_argument("--output-root", default="outputs/v2clean_shift_sensitivity_seed1")
    shift_summary = sub.add_parser("shift-summary")
    shift_summary.add_argument("--output-root", default="outputs/v2clean_shift_sensitivity_seed1")
    failure = sub.add_parser("failure", parents=[common])
    failure.add_argument("--variant", required=True)
    failure.add_argument("--stage", choices=("source", "uda_best"), required=True)
    failure.add_argument("--output", required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--root", default="outputs/v2clean_minority_ab_seed1")
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "shift":
        run_shift(args)
    elif args.command == "shift-summary":
        run_shift_summary(args)
    elif args.command == "failure":
        run_failure(args)
    else:
        run_aggregate(args)
