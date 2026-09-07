#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


TASKS = (
    ("AT1_DK1", "AT1", "DK1"),
    ("DK1_FR1", "DK1", "FR1"),
    ("FR1_FR2", "FR1", "FR2"),
    ("FR2_AT1", "FR2", "AT1"),
)
GROUPS = ("original", "local_sample")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SUMMARY_FIELDS = (
    "task",
    "source",
    "target",
    "original_test_macro_f1",
    "local_sample_test_macro_f1",
    "local_minus_original",
    "original_status",
    "local_sample_status",
    "original_log",
    "local_sample_log",
)


def _load_statuses(status_path):
    statuses = {}
    if not status_path.is_file():
        return statuses
    with status_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row.get("group", ""), row.get("task", ""))
            try:
                exit_code = int(row["exit_code"])
            except (KeyError, TypeError, ValueError):
                exit_code = None
            statuses[key] = (exit_code, row.get("status", ""))
    return statuses


def _read_test_f1(log_path, experiment):
    if not log_path.is_file():
        return None
    pattern = re.compile(
        rf"^Test result for {re.escape(experiment)}: "
        rf"accuracy=({NUMBER}), f1=({NUMBER})\s*$",
        flags=re.MULTILINE,
    )
    matches = pattern.findall(log_path.read_text(encoding="utf-8", errors="replace"))
    if len(matches) != 1:
        return None
    return float(matches[0][1])


def _group_result(log_root, statuses, group, task, seed):
    log_path = log_root / group / f"{task}.log"
    status_record = statuses.get((group, task))
    if status_record is None:
        return None, "NOT_RUN", str(log_path)
    exit_code, explicit_status = status_record
    if explicit_status and explicit_status != "SUCCESS":
        return None, explicit_status, str(log_path)
    if exit_code is None:
        return None, "INVALID_STATUS", str(log_path)
    if exit_code != 0:
        return None, f"PROCESS_FAILED({exit_code})", str(log_path)
    experiment = f"timematch_{group}_{task}_seed{seed}"
    f1 = _read_test_f1(log_path, experiment)
    if f1 is None:
        return None, "MISSING_TEST_F1", str(log_path)
    return f1, "SUCCESS", str(log_path)


def summarize(
    log_root,
    status_path,
    output_path,
    seed=1,
    required_groups=GROUPS,
):
    log_root = Path(log_root)
    status_path = Path(status_path)
    output_path = Path(output_path)
    required_groups = tuple(required_groups)
    statuses = _load_statuses(status_path)
    rows = []
    failed = False

    for task, source, target in TASKS:
        group_results = {
            group: _group_result(log_root, statuses, group, task, seed)
            for group in GROUPS
        }
        original_f1, original_status, original_log = group_results["original"]
        local_f1, local_status, local_log = group_results["local_sample"]
        for group in required_groups:
            if group_results[group][1] != "SUCCESS":
                failed = True
        difference = ""
        if original_f1 is not None and local_f1 is not None:
            difference = f"{local_f1 - original_f1:.4f}"
        rows.append(
            {
                "task": task,
                "source": source,
                "target": target,
                "original_test_macro_f1": (
                    "" if original_f1 is None else f"{original_f1:.4f}"
                ),
                "local_sample_test_macro_f1": (
                    "" if local_f1 is None else f"{local_f1:.4f}"
                ),
                "local_minus_original": difference,
                "original_status": original_status,
                "local_sample_status": local_status,
                "original_log": original_log,
                "local_sample_log": local_log,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['task']}: original={row['original_status']} "
            f"local_sample={row['local_sample_status']}"
        )
    print(f"Summary written to {output_path}")
    return int(failed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--status-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--required-groups",
        nargs="+",
        choices=GROUPS,
        default=list(GROUPS),
    )
    args = parser.parse_args()
    raise SystemExit(
        summarize(
            args.log_root,
            args.status_file,
            args.output,
            seed=args.seed,
            required_groups=args.required_groups,
        )
    )


if __name__ == "__main__":
    main()
