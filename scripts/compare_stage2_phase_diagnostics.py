#!/usr/bin/env python3
"""Compare V6 proposal evidence with exhaustive all-class evidence at one budget."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stage_line(path: Path, prefix: str, budget: int) -> dict[str, str]:
    chosen = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line.startswith(prefix + "|"):
                continue
            fields = {}
            for item in line.split("|")[1:]:
                if "=" in item:
                    key, value = item.split("=", 1)
                    fields[key] = value
            if fields.get("budget") == str(budget):
                chosen = fields
                break
    return {} if chosen is None else chosen


def _hypothesis_keys(payload: dict, allowed_ids: set[int]) -> set[tuple[int, int]]:
    rows = payload.get("phase_hypotheses") or []
    return {
        (int(row["sample_id"]), int(row["class_id"]))
        for row in rows
        if int(row["sample_id"]) in allowed_ids
    }


def _class_counts(keys: set[tuple[int, int]]) -> dict[str, int]:
    counts = Counter(class_id for _sample_id, class_id in keys)
    return {str(class_id): counts[class_id] for class_id in sorted(counts)}


def _json_number(value: str | None):
    if value is None:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-json", type=Path, required=True)
    parser.add_argument("--all-class-json", type=Path, required=True)
    parser.add_argument("--proposal-log", type=Path, required=True)
    parser.add_argument("--all-class-log", type=Path, required=True)
    parser.add_argument("--compare-budget", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    proposal = _load(args.proposal_json)
    exhaustive = _load(args.all_class_json)
    budget = args.compare_budget

    proposal_ids = [int(v) for v in (proposal.get("phase_scanned_sample_ids") or [])]
    exhaustive_ids = [int(v) for v in (exhaustive.get("phase_scanned_sample_ids") or [])]
    proposal_first = proposal_ids[:budget]
    exhaustive_first = exhaustive_ids[:budget]
    same_samples = proposal_first == exhaustive_first and len(proposal_first) == budget

    proposal_keys = _hypothesis_keys(proposal, set(proposal_first))
    exhaustive_keys = _hypothesis_keys(exhaustive, set(exhaustive_first))
    intersection = proposal_keys & exhaustive_keys
    missed = exhaustive_keys - proposal_keys
    proposal_only = proposal_keys - exhaustive_keys
    recall = None if not exhaustive_keys else len(intersection) / len(exhaustive_keys)

    prefixes = (
        "TARGET_HYPOTHESIS_SCAN_STAGE_DONE",
        "TARGET_HYPOTHESIS_SCAN_DISTRIBUTIONS",
        "STAGE2_PHASE_EVIDENCE_STAGE",
    )
    proposal_lines = {
        prefix: _stage_line(args.proposal_log, prefix, budget) for prefix in prefixes
    }
    exhaustive_lines = {
        prefix: _stage_line(args.all_class_log, prefix, budget) for prefix in prefixes
    }

    payload = {
        "compare_budget": budget,
        "same_first_budget_sample_ids": same_samples,
        "proposal": {
            "accepted_hypothesis_pairs": len(proposal_keys),
            "accepted_class_counts": _class_counts(proposal_keys),
            "stage": {
                k: {field: _json_number(value) for field, value in v.items()}
                for k, v in proposal_lines.items()
            },
        },
        "all_class": {
            "accepted_hypothesis_pairs": len(exhaustive_keys),
            "accepted_class_counts": _class_counts(exhaustive_keys),
            "stage": {
                k: {field: _json_number(value) for field, value in v.items()}
                for k, v in exhaustive_lines.items()
            },
        },
        "proposal_vs_all_class": {
            "intersection_pairs": len(intersection),
            "missed_by_proposal_pairs": len(missed),
            "proposal_only_pairs": len(proposal_only),
            "retained_hypothesis_recall": recall,
            "missed_by_proposal": [
                {"sample_id": sample_id, "class_id": class_id}
                for sample_id, class_id in sorted(missed)
            ],
            "proposal_only": [
                {"sample_id": sample_id, "class_id": class_id}
                for sample_id, class_id in sorted(proposal_only)
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    recall_text = "NA" if recall is None else f"{recall:.4f}"
    print(
        "PHASE_PROPOSAL_RECALL|"
        f"budget={budget}|same_samples={str(same_samples).lower()}"
        f"|proposal_retained={len(proposal_keys)}"
        f"|allclass_retained={len(exhaustive_keys)}"
        f"|intersection={len(intersection)}"
        f"|missed_by_proposal={len(missed)}"
        f"|proposal_only={len(proposal_only)}"
        f"|retained_recall={recall_text}"
    )
    if not same_samples:
        raise SystemExit("A/B diagnostic did not use the exact same first evidence samples")


if __name__ == "__main__":
    main()
