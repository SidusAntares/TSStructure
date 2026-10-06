#!/usr/bin/env python3
"""Merge state-org frozen audits and short causal UDA runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.state_org_feasibility_audit import rule_labels


TASKS = ("AT1_DK1", "FR2_DK1", "DK1_AT1")
VARIANTS = ("baseline", "no_shape_aux", "detach_target_structure", "freeze_structure_specific")


def _read_csv(path):
    if not Path(path).is_file(): return []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _float(value, default=float("nan")):
    try: return float(value)
    except (TypeError, ValueError): return default


def _training_row(root, log_root, baseline_root, task, variant):
    if variant == "baseline":
        fold = Path(baseline_root) / "uda" / f"{task}_seed1" / "fold_0"
        log = Path("logs/structure_state_org_4tasks_seed1") / f"{task}.log"
    else:
        fold = Path(root) / "variants" / variant / "uda" / f"{task}_seed1" / "fold_0"
        log = Path(log_root) / task / f"{variant}.log"
    values = {"task": task, "variant": variant, "status": "missing"}
    if not (fold / "checkpoint_last.pt").is_file(): return values
    last = torch.load(fold / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    best_path = fold / "checkpoint_best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.is_file() else {}
    values.update({
        "status": "complete", "best_val": _float(best.get("best_f1")),
        "best_epoch": best.get("epoch", ""), "final_epoch": last.get("epoch", ""),
    })
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    validation = [_float(value) for value in re.findall(r"Validation result:.*?f1=([0-9.]+)", text)]
    values["final_val"] = validation[-1] if validation else float("nan")
    values["peak_final_drop"] = max(validation) - validation[-1] if validation else float("nan")
    tests = re.findall(r"Test result for .*?f1=([0-9.]+)", text)
    values["final_test"] = _float(tests[-1]) if tests else float("nan")
    epochs = re.findall(r"SHAPE_V2CLEAN_EPOCH\|[^\n]+", text)
    if epochs:
        for key in ("pseudo_coverage", "accepted_pseudo_accuracy"):
            match = re.search(rf"{key}=([0-9.]+)", epochs[-1])
            values[key] = _float(match.group(1)) if match else float("nan")
    return values


def summarize(root, log_root, baseline_root="outputs/structure_state_org_4tasks_seed1"):
    root = Path(root); rows = []
    sections = {name: [] for name in ("probe", "query", "organization", "training")}
    for task in TASKS:
        audit = root / "audit" / task
        for row in _read_csv(audit / "probe_results.csv"):
            item = {"table": "probe", "task": task, **row}; rows.append(item); sections["probe"].append(item)
        for row in _read_csv(audit / "query_counterfactual.csv"):
            item = {"table": "query", "task": task, **row}; rows.append(item); sections["query"].append(item)
        for row in _read_csv(audit / "organization_counterfactual.csv"):
            item = {"table": "organization", "task": task, **row}; rows.append(item); sections["organization"].append(item)
        for variant in VARIANTS:
            item = {"table": "training", **_training_row(
                root, log_root, baseline_root, task, variant,
            )}
            rows.append(item); sections["training"].append(item)
    _write_csv(root / "state_org_feasibility_summary.csv", rows)
    lines = ["# State-Org Feasibility Report", ""]
    for title, key in (("Table 1: Feature probes", "probe"), ("Table 2: Query counterfactuals", "query"), ("Table 3: Organization counterfactuals", "organization"), ("Table 4: Causal UDA variants", "training")):
        lines.extend([f"## {title}", "", "```text"])
        lines.extend(" | ".join(f"{name}={value}" for name, value in row.items() if name != "table") for row in sections[key])
        lines.extend(["```", ""])
    evidence = []
    for task in TASKS:
        probe = [row for row in sections["probe"] if row["task"] == task and row.get("checkpoint") == "source" and row.get("feature") == "shape_response"]
        query = [row for row in sections["query"] if row["task"] == task]
        org = [row for row in sections["organization"] if row["task"] == task and row.get("checkpoint") == "source"]
        q = {(row.get("checkpoint"), row.get("query")): _float(row.get("target_macro_f1"), 0.) for row in query}
        o = {row.get("variant"): _float(row.get("target_macro_f1"), 0.) for row in org}
        training = {
            row["variant"]: row for row in sections["training"]
            if row["task"] == task and row.get("status") == "complete"
        }
        baseline = training.get("baseline", {})
        no_shape = training.get("no_shape_aux", {})
        detached = training.get("detach_target_structure", {})
        metrics = {
            "target_oracle": _float(probe[0].get("target_oracle_macro_f1"), 0.) if probe else 0.,
            "source_to_target": _float(probe[0].get("source_to_target_macro_f1"), 0.) if probe else 0.,
            "presence_minus_master": q.get(("source", "presence"), 0.) - q.get(("source", "master"), 0.),
            "organization_minus_master": q.get(("source", "organization"), 0.) - q.get(("source", "master"), 0.),
            "mean_repeat_minus_original": o.get("mean_repeat", 0.) - o.get("original", 0.),
            "source_full_minus_master": q.get(("source", "full"), 0.) - q.get(("source", "master"), 0.),
            "last_full_minus_master": q.get(("last", "full"), 0.) - q.get(("last", "master"), 0.),
            "no_shape_aux_peak_final_gain": (
                _float(baseline.get("peak_final_drop"), 0.)
                - _float(no_shape.get("peak_final_drop"), 0.)
            ),
            "detach_target_final_gain": (
                _float(detached.get("final_test"), 0.)
                - _float(baseline.get("final_test"), 0.)
            ),
        }
        labels = rule_labels(metrics)
        evidence.append({"task": task, "labels": labels, "metrics": metrics})
    lines.extend(["## Key per-class evidence", ""])
    hard = {
        "AT1_DK1": {"spring_barley"},
        "FR2_DK1": {"spring_barley", "winter_barley"},
        "DK1_AT1": {"winter_triticale", "winter_wheat"},
    }
    for task in TASKS:
        audit = root / "audit" / task
        per_class = [
            row for row in _read_csv(audit / "probe_per_class.csv")
            if row.get("class_name") in hard[task]
        ]
        vocabulary = [
            row for row in _read_csv(audit / "anchor_vocabulary_per_class.csv")
            if row.get("class_name") in hard[task] and row.get("domain") == "target"
        ]
        lines.extend([f"### {task}", "", "```text"])
        lines.extend(" | ".join(f"{key}={value}" for key, value in row.items()) for row in per_class + vocabulary)
        lines.extend(["```", ""])
    lines.extend(["## Evidence labels", "", "```json", json.dumps(evidence, indent=2), "```", "", "> `freeze_structure_specific` freezes only structure-specific parameters; the shared PSE remains trainable, so the structure response can still drift."])
    (root / "state_org_feasibility_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/state_org_feasibility")
    parser.add_argument("--log-root", default="logs/state_org_feasibility")
    parser.add_argument(
        "--baseline-root", default="outputs/structure_state_org_4tasks_seed1",
    )
    args = parser.parse_args(); summarize(args.root, args.log_root, args.baseline_root)


if __name__ == "__main__": main()
