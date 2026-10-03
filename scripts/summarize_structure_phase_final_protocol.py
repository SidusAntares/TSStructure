#!/usr/bin/env python3
"""Collect old best-validation and new final-epoch P/E test Macro-F1."""

import argparse
import csv
import json
from pathlib import Path


TASKS = {
    "AT1_DK1": "denmark_32VNH_2017",
    "FR1_FR2": "france_31TCJ_2017",
    "FR2_DK1": "denmark_32VNH_2017",
    "DK1_AT1": "austria_33UVP_2017",
}
FIELDS = (
    "task", "P_final_test_macro_f1", "E_final_test_macro_f1",
    "P_old_bestval_test_macro_f1", "E_old_bestval_test_macro_f1",
)


def metric(root, task, target, final):
    marker = "final_" if final else ""
    path = Path(root) / f"{task}_seed1" / "fold_0" / f"test_metrics_{marker}{target}.json"
    if not path.is_file():
        raise FileNotFoundError(path.resolve())
    return json.loads(path.read_text(encoding="utf-8"))["macro_f1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-root", default="outputs/structure_phase_moment_4tasks_seed1/uda")
    parser.add_argument("--e-root", default="outputs/structure_phase_equivariance_4tasks_seed1/uda")
    parser.add_argument("--output", default="outputs/structure_phase_final_protocol_seed1.csv")
    args = parser.parse_args()
    rows = []
    for task, target in TASKS.items():
        rows.append({
            "task": task,
            "P_final_test_macro_f1": metric(args.p_root, task, target, True),
            "E_final_test_macro_f1": metric(args.e_root, task, target, True),
            "P_old_bestval_test_macro_f1": metric(args.p_root, task, target, False),
            "E_old_bestval_test_macro_f1": metric(args.e_root, task, target, False),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"FINAL_PROTOCOL_SUMMARY|output={output}")


if __name__ == "__main__":
    main()
