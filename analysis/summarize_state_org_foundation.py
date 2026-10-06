#!/usr/bin/env python3
"""Merge the state-org foundation study in Organization -> Anchor -> Query order."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.summarize_state_org_hierarchical import validation_history_stats

TASKS = ("AT1_DK1", "FR2_DK1", "DK1_AT1")
READOUTS = ("full", "composition", "presence")
ANCHOR_MODES = ("fixed", "source", "target", "shared")


def _read(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _training_row(root, logs, task, family, variant):
    fold = root / family / variant / "uda" / f"{task}_seed1" / "fold_0"
    log = logs / task / f"{family}_{variant}.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    validation = [float(value) for value in re.findall(
        r"Validation result:.*?f1=([0-9.]+)", text,
    )]
    row = {
        "task": task, "variant": variant, **validation_history_stats(validation),
        "status": "complete" if (fold / "checkpoint_last.pt").is_file() else "missing",
    }
    tests = re.findall(r"Test result for .*?f1=([0-9.]+)", text)
    row["final_test"] = float(tests[-1]) if tests else float("nan")
    epochs = re.findall(r"SHAPE_V2CLEAN_EPOCH\|[^\n]+", text)
    for key in ("pseudo_coverage", "accepted_pseudo_accuracy"):
        match = re.search(rf"{key}=([0-9.]+)", epochs[-1]) if epochs else None
        row[key] = float(match.group(1)) if match else float("nan")
    oracle = re.findall(r"PSEUDO_ORACLE_AUDIT\|[^\n]+", text)
    match = re.search(r"per_true_class=([^\n]+)", oracle[-1]) if oracle else None
    row["per_class_pseudo_accuracy"] = match.group(1) if match else ""
    if row["status"] == "complete":
        last = torch.load(fold / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        best_path = fold / "checkpoint_best.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.is_file() else {}
        row["final_epoch"] = last.get("epoch", "")
        row["best_epoch"] = best.get("epoch", "")
    return row


def summarize(root, log_root):
    root = Path(root); logs = Path(log_root); rows = []
    dynamics, dynamics_final = [], []
    for task in TASKS:
        audit = root / "audit" / task
        source = {row["readout"]: row for row in _read(audit / "source_readout_summary.csv")}
        for readout in READOUTS:
            training = _training_row(root, logs, task, "organization", readout)
            rows.append({"section": "A", **training, **{
                f"source_{key}": value for key, value in source.get(readout, {}).items()
                if key != "readout"
            }})
        for filename in (
            "organization_semantic_probe.csv", "organization_semantic_per_class.csv",
            "anchor_capacity_summary.csv", "anchor_correspondence.csv",
        ):
            rows.extend({
                "section": "A", "task": task, "table": filename[:-4], **row,
            } for row in _read(audit / filename))

        for mode in ANCHOR_MODES:
            training = _training_row(root, logs, task, "anchor", mode)
            fold = root / "anchor" / mode / "uda" / f"{task}_seed1" / "fold_0"
            history = _read(fold / "anchor_dynamics.csv")
            tagged = [{"task": task, "mode": mode, **row} for row in history]
            dynamics.extend(tagged)
            if tagged:
                dynamics_final.append(tagged[-1])
                training.update({
                    key: value for key, value in tagged[-1].items()
                    if key not in ("task", "mode", "epoch")
                })
            rows.append({"section": "B", **training})

        rows.extend({
            "section": "C", "task": task, "table": "query_role", **row,
        } for row in _read(audit / "query_role_audit.csv"))
        rows.extend({
            "section": "C", "task": task, "table": "query_attention", **row,
        } for row in _read(audit / "query_attention_audit.csv"))
    _write(root / "anchor_dynamics.csv", dynamics)
    _write(root / "anchor_dynamics_per_task.csv", dynamics_final)
    _write(root / "state_org_foundation_summary.csv", rows)
    lines = ["# State-Org Foundation Report", ""]
    for section, title in (
        ("A", "A. Organization"), ("B", "B. Anchor"), ("C", "C. Query"),
    ):
        lines.extend([f"## {title}", "", "```text"])
        lines.extend(
            " | ".join(f"{key}={value}" for key, value in row.items())
            for row in rows if row["section"] == section
        )
        lines.extend(["```", ""])
    (root / "state_org_foundation_report.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/state_org_foundation")
    parser.add_argument("--log-root", default="logs/state_org_foundation")
    args = parser.parse_args(); summarize(args.root, args.log_root)


if __name__ == "__main__":
    main()
