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


def seed1_results(rows):
    if any(int(row["seed"]) != 1 for row in rows):
        raise ValueError("reliable-composition summary accepts seed1 only")
    return sorted(rows, key=lambda row: (row["task"], row["variant"]))


def checkpoint_status(fold, kind):
    fold = Path(fold)
    if kind not in ("source", "uda"):
        raise ValueError("checkpoint kind must be source or uda")
    if not fold.exists():
        return "incomplete" if fold.parent.exists() else "missing"
    required = (
        (fold / "model.pt", fold / "checkpoint_last.pt") if kind == "source"
        else (fold / "checkpoint_best.pt", fold / "checkpoint_last.pt")
    )
    if not all(path.is_file() for path in required):
        return "incomplete"
    final_path = required[1]
    try:
        packet = torch.load(final_path, map_location="cpu", weights_only=False)
        config = packet["config"]
        epochs = int(
            config["epochs"] if isinstance(config, dict) else config.epochs
        )
        epoch = int(packet["epoch"])
    except Exception:
        return "incomplete"
    return "complete" if epoch == epochs - 1 else "incomplete"


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
        seed = 1
        for task in TASKS:
            fold = root / variant / "seed1" / "uda" / f"{task}_seed1" / "fold_0"
            best_path = fold / "test_metrics_best_audit.json"
            final_path = fold / "test_metrics_final_audit.json"
            if not best_path.is_file() or not final_path.is_file():
                print(
                    f"MISSING|variant={variant}|task={task}|seed=1|"
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
    _write_csv(output / "seed1_test_macro_f1.csv", seed1_results(rows))
    print(f"RELIABLE_COMPOSITION_SUMMARY|runs={len(rows)}|output={output}")


def report_checkpoint_status(args):
    status = checkpoint_status(args.fold, args.kind)
    print(f"CHECKPOINT_STATUS|kind={args.kind}|status={status}|fold={args.fold}")
    return {"complete": 0, "missing": 3, "incomplete": 4}[status]


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
    status = sub.add_parser("checkpoint-status")
    status.add_argument("--fold", required=True, type=Path)
    status.add_argument("--kind", required=True, choices=("source", "uda"))
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "evaluate":
        evaluate(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        raise SystemExit(report_checkpoint_status(args))


if __name__ == "__main__":
    main()
