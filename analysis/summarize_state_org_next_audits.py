#!/usr/bin/env python3
"""Merge shift, affinity, and geometric-anchor results without recommendations."""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.summarize_state_org_hierarchical import validation_history_stats


TASKS = ("AT1_DK1", "FR2_DK1", "DK1_AT1")
MODES = ("fixed", "target_ema", "shared_ema")


def _read(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _per_class_f1(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    tp = np.diag(confusion)
    precision = tp / np.maximum(confusion.sum(0), 1.)
    recall = tp / np.maximum(confusion.sum(1), 1.)
    return 2. * precision * recall / np.maximum(precision + recall, 1e-12)


def _training_rows(root, logs, task, mode):
    fold = root / "anchor_geometric" / mode / "uda" / f"{task}_seed1" / "fold_0"
    log = logs / task / f"anchor_geometric_{mode}.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    validation = [float(value) for value in re.findall(
        r"Validation result:.*?f1=([0-9.]+)", text,
    )]
    row = {
        "section": "C", "task": task, "table": "anchor_geometric",
        "mode": mode, **validation_history_stats(validation),
        "status": "complete" if (fold / "checkpoint_last.pt").is_file() else "missing",
    }
    tests = re.findall(r"Test result for .*?f1=([0-9.]+)", text)
    row["final_test_macro_f1"] = float(tests[-1]) if tests else float("nan")
    epochs = re.findall(r"SHAPE_V2CLEAN_EPOCH\|[^\n]+", text)
    for key in ("pseudo_coverage", "accepted_pseudo_accuracy"):
        match = re.search(rf"{key}=([0-9.]+)", epochs[-1]) if epochs else None
        row[key] = float(match.group(1)) if match else float("nan")
    oracle = re.findall(r"PSEUDO_ORACLE_AUDIT\|[^\n]+", text)
    match = re.search(r"per_true_class=([^\n]+)", oracle[-1]) if oracle else None
    row["per_class_pseudo_accuracy"] = match.group(1) if match else ""
    history = _read(fold / "anchor_geometric_dynamics.csv")
    if history:
        row.update({
            f"final_{key}": value for key, value in history[-1].items()
            if key not in ("epoch", "mode")
        })
    per_class = []
    confusion_paths = list(fold.glob("conf_mat_final_*.pkl"))
    if confusion_paths:
        with confusion_paths[0].open("rb") as stream:
            values = _per_class_f1(pickle.load(stream))
        per_class = [{
            "section": "C", "task": task, "table": "anchor_geometric_per_class",
            "mode": mode, "class_id": class_id, "test_f1": float(value),
        } for class_id, value in enumerate(values)]
    if row["status"] == "complete":
        packet = torch.load(
            fold / "checkpoint_last.pt", map_location="cpu", weights_only=False,
        )
        row["final_epoch"] = packet.get("epoch", "")
    return row, per_class


def summarize(root, log_root, foundation_root):
    root = Path(root); logs = Path(log_root); foundation = Path(foundation_root)
    rows = []
    for task in TASKS:
        audit = root / "audit" / task
        for filename in (
            "shift_sensitivity.csv", "shift_probe.csv",
            "shift_probe_per_class.csv", "classwise_oracle_shift.csv",
        ):
            rows.extend({
                "section": "A", "task": task, "table": filename[:-4], **row,
            } for row in _read(audit / filename))
        for filename in ("assignment_probe.csv", "assignment_probe_per_class.csv"):
            rows.extend({
                "section": "B", "task": task, "table": filename[:-4], **row,
            } for row in _read(audit / filename))
        for mode in MODES:
            training, per_class = _training_rows(root, logs, task, mode)
            rows.append(training); rows.extend(per_class)

    historical = _read(foundation / "state_org_foundation_summary.csv")
    rows.extend({
        "section": "C", "table": "historical_gradient_reference", **row,
    } for row in historical if row.get("section") == "B")
    _write(root / "state_org_next_audit_summary.csv", rows)
    lines = ["# State-Org Next Audit Report", ""]
    for section, title in (
        ("A", "A. Shift sensitivity"),
        ("B", "B. Softmax vs raw affinity"),
        ("C", "C. Anchor geometric adaptation"),
    ):
        lines.extend((f"## {title}", "", "```text"))
        lines.extend(
            " | ".join(f"{key}={value}" for key, value in row.items())
            for row in rows if row.get("section") == section
        )
        lines.extend(("```", ""))
    (root / "state_org_next_audit_report.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/state_org_next_audits")
    parser.add_argument("--log-root", default="logs/state_org_next_audits")
    parser.add_argument("--foundation-root", default="outputs/state_org_foundation")
    args = parser.parse_args()
    summarize(args.root, args.log_root, args.foundation_root)


if __name__ == "__main__":
    main()
