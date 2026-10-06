#!/usr/bin/env python3
"""Merge hierarchical state-org evidence in Anchor -> Organization -> Query order."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import torch

TASKS = ("AT1_DK1", "FR2_DK1", "DK1_AT1")
VARIANTS = ("adaptive_full", "adaptive_presence", "frozen_full", "frozen_presence")


def validation_history_stats(values):
    values = [float(value) for value in values]
    if not values:
        return {"best_val": float("nan"), "final_val": float("nan"), "peak_final_drop": float("nan")}
    return {"best_val": max(values), "final_val": values[-1], "peak_final_drop": max(values) - values[-1]}


def _read(path):
    if not Path(path).is_file(): return []
    with Path(path).open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def _write(path, rows):
    rows = list(rows); fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _training_row(root, logs, old_root, task, variant):
    if variant == "adaptive_full":
        fold = Path(old_root) / "uda" / f"{task}_seed1" / "fold_0"
        log = Path("logs/state_org_feasibility") / task / "no_shape_aux.log"
    else:
        fold = Path(root) / "variants" / variant / "uda" / f"{task}_seed1" / "fold_0"
        log = Path(logs) / task / f"{variant}.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    history = [float(x) for x in re.findall(r"Validation result:.*?f1=([0-9.]+)", text)]
    result = {"section": "A", "task": task, "variant": variant, **validation_history_stats(history)}
    tests = re.findall(r"Test result for .*?f1=([0-9.]+)", text)
    result["test"] = float(tests[-1]) if tests else float("nan")
    epochs = re.findall(r"SHAPE_V2CLEAN_EPOCH\|[^\n]+", text)
    for key in ("pseudo_coverage", "accepted_pseudo_accuracy"):
        match = re.search(rf"{key}=([0-9.]+)", epochs[-1]) if epochs else None
        result[key] = float(match.group(1)) if match else float("nan")
    result["status"] = "complete" if (fold / "checkpoint_last.pt").is_file() else "missing"
    if result["status"] == "complete":
        last = torch.load(fold / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        best_path = fold / "checkpoint_best.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.is_file() else {}
        result["final_epoch"] = last.get("epoch", "")
        result["best_epoch"] = best.get("epoch", "")
    return result


def summarize(root, logs, old_root):
    root = Path(root); rows = []
    for task in TASKS:
        for variant in VARIANTS: rows.append(_training_row(root, logs, old_root, task, variant))
        audit = root / "audit" / task
        for filename in ("anchor_capacity_summary.csv", "anchor_correspondence.csv"):
            rows.extend({"section": "A", "task": task, "table": filename[:-4], **row} for row in _read(audit / filename))
        for filename in ("organization_probe.csv", "organization_counterfactual_differences.csv"):
            rows.extend({"section": "B", "task": task, "table": filename[:-4], **row} for row in _read(audit / filename))
        rows.extend({"section": "C", "task": task, "table": "query_scale", **row} for row in _read(audit / "query_scale_sweep.csv"))
    _write(root / "state_org_hierarchical_summary.csv", rows)
    lines = ["# State-Org Hierarchical Report", ""]
    for key, title in (("A", "A. Anchor / structural basis"), ("B", "B. Organization semantics"), ("C", "C. Query role")):
        lines.extend([f"## {title}", "", "```text"])
        lines.extend(" | ".join(f"{name}={value}" for name, value in row.items()) for row in rows if row["section"] == key)
        lines.extend(["```", ""])
    (root / "state_org_hierarchical_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="outputs/state_org_hierarchical")
    parser.add_argument("--log-root", default="logs/state_org_hierarchical")
    parser.add_argument("--old-root", default="outputs/state_org_feasibility/variants/no_shape_aux")
    args = parser.parse_args(); summarize(args.root, args.log_root, args.old_root)


if __name__ == "__main__": main()
