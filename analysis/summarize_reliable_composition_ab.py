"""Evaluate fixed Best/Final checkpoints and aggregate absolute test Macro-F1."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TASKS = ("AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1")


def aggregate_results(rows):
    grouped = {}
    for row in rows:
        key = (row["variant"], row["task"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (variant, task), values in sorted(grouped.items()):
        for stage in ("best", "final"):
            scores = np.asarray([
                float(row[f"{stage}_test_macro_f1"]) for row in values
            ])
            summary.append({
                "variant": variant, "task": task, "stage": stage,
                "seed_count": int(scores.size),
                "test_macro_f1_mean": float(scores.mean()),
                "test_macro_f1_std": float(scores.std(ddof=0)),
            })
    return summary


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(args):
    from analysis.state_org_feasibility_audit import _datasets, _loader, _move
    from analysis.v2_v2clean_failure_audit import _load_model

    checkpoint = Path(args.checkpoint)
    if checkpoint.name not in ("checkpoint_best.pt", "checkpoint_last.pt"):
        raise ValueError("evaluation checkpoint must be checkpoint_best.pt or checkpoint_last.pt")
    model, config, packet = _load_model(checkpoint, torch.device(args.device))
    if getattr(config, "state_org_readout", None) != "reliable_composition":
        raise ValueError("checkpoint is not reliable_composition")
    if checkpoint.name == "checkpoint_last.pt":
        if int(packet.get("epoch", -1)) != int(config.epochs) - 1:
            raise ValueError("checkpoint_last.pt is not the configured final epoch")
    datasets = _datasets(
        config, config.source, config.target, args.data_root, config.seed,
    )
    labels, predictions = [], []
    shift = int(round(float(packet["global_temporal_shift"])))
    for raw in _loader(datasets["target_test"], args.batch_size):
        batch = _move(raw, args.device)
        output = model.forward_with_temporal_shift(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch["extra"], temporal_shift=shift, return_dict=True,
        )
        labels.append(batch["label"].long().cpu())
        predictions.append(output["logits"].argmax(1).cpu())
    labels = torch.cat(labels).numpy()
    predictions = torch.cat(predictions).numpy()
    result = {
        "checkpoint": str(checkpoint),
        "role": "best" if checkpoint.name == "checkpoint_best.pt" else "final",
        "epoch": int(packet["epoch"]), "global_shift": shift,
        "sample_count": int(labels.size),
        "test_macro_f1": float(f1_score(
            labels, predictions, labels=np.arange(config.num_classes),
            average="macro", zero_division=0,
        )),
        "test_labels_used_for_selection": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"RELIABLE_COMPOSITION_EVAL|role={result['role']}|"
        f"epoch={result['epoch']}|test_macro_f1={result['test_macro_f1']:.6f}|"
        f"samples={result['sample_count']}"
    )


def summarize(args):
    root = Path(args.root)
    rows = []
    for variant in ("A", "B"):
        for seed in (1, 2, 3):
            for task in TASKS:
                fold = root / variant / f"seed{seed}" / "uda" / f"{task}_seed{seed}" / "fold_0"
                best_path = fold / "test_metrics_best_audit.json"
                final_path = fold / "test_metrics_final_audit.json"
                if not best_path.is_file() or not final_path.is_file():
                    print(
                        f"MISSING|variant={variant}|task={task}|seed={seed}|"
                        f"best={best_path}|final={final_path}"
                    )
                    continue
                best = json.loads(best_path.read_text(encoding="utf-8"))
                final = json.loads(final_path.read_text(encoding="utf-8"))
                rows.append({
                    "variant": variant, "task": task, "seed": seed,
                    "best_epoch": best["epoch"],
                    "best_test_macro_f1": best["test_macro_f1"],
                    "final_epoch": final["epoch"],
                    "final_test_macro_f1": final["test_macro_f1"],
                })
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_seed_test_macro_f1.csv", rows)
    _write_csv(output / "three_seed_test_macro_f1.csv", aggregate_results(rows))
    print(f"RELIABLE_COMPOSITION_SUMMARY|runs={len(rows)}|output={output}")


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True, type=Path)
    evaluation.add_argument("--data-root", required=True, type=Path)
    evaluation.add_argument("--output", required=True, type=Path)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--batch-size", type=int, default=128)
    aggregation = sub.add_parser("summarize")
    aggregation.add_argument("--root", type=Path, default=Path("outputs/reliable_composition_ab"))
    aggregation.add_argument(
        "--output", type=Path,
        default=Path("outputs/reliable_composition_ab/summary"),
    )
    return parser


def main():
    args = build_parser().parse_args()
    evaluate(args) if args.command == "evaluate" else summarize(args)


if __name__ == "__main__":
    main()
