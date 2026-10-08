"""Evaluate fixed Best/Final checkpoints and aggregate absolute test Macro-F1."""

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import create_evaluation_loaders as official_create_evaluation_loaders
from evaluation import evaluation as official_evaluation
from train import (
    create_train_val_test_folds as official_create_splits,
    prepare_data_protocol as _prepare_data_protocol,
)


TASKS = ("AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1")
TASK_TARGETS = {
    "AT1_DK1": "denmark/32VNH/2017",
    "FR1_FR2": "france/31TCJ/2017",
    "FR2_DK1": "denmark/32VNH/2017",
    "DK1_AT1": "austria/33UVP/2017",
}
METHODS = ("V2", "V2clean", "E", "A", "B")


def official_prepare_data_protocol(config):
    return _prepare_data_protocol(config, write_protocol=False)


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


def checkpoint_roles(fold):
    fold = Path(fold)
    return {
        "best": fold / "checkpoint_best.pt",
        "final": fold / "checkpoint_last.pt",
    }


def verified_final_artifacts(fold, target_name):
    fold = Path(fold)
    manifest_path = fold / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        manifest.get("uda_test_checkpoint") != "checkpoint_last.pt"
        or manifest.get("target_validation_used_for_test_selection") is not False
    ):
        return None
    metrics = fold / f"test_metrics_final_{target_name}.json"
    confusion = fold / f"conf_mat_final_{target_name}.pkl"
    return (metrics, confusion) if metrics.is_file() and confusion.is_file() else None


def build_official_test_loader(config, data_root, batch_size):
    config.data_root = str(data_root)
    config.batch_size = int(batch_size)
    random.seed(int(config.seed))
    np.random.seed(int(config.seed))
    torch.manual_seed(int(config.seed))
    indices, _ = official_prepare_data_protocol(config)
    splits = official_create_splits(
        [config.source, config.target], config.num_folds, indices,
        config.val_ratio, config.test_ratio,
    )[0]
    _, test_loader = official_create_evaluation_loaders(
        config.target, splits, config,
        bool(getattr(config, "sample_pixels_val", False)),
    )
    return test_loader


def evaluate_model_on_official_test(model, test_loader, config, device):
    metrics = official_evaluation(
        model, test_loader, device, config.classes, mode="test",
        progress_bar=getattr(config, "progress_bar", "off"),
    )
    confusion_metrics = _metrics_from_confusion(metrics["confusion_matrix"])
    metrics.update({
        "fixed_class_macro_f1": confusion_metrics["fixed_class_macro_f1"],
        "per_class_precision": confusion_metrics["per_class_precision"],
        "per_class_recall": confusion_metrics["per_class_recall"],
        "per_class_f1": confusion_metrics["per_class_f1"],
        "support": confusion_metrics["support"],
    })
    metrics["inference_shift"] = 0
    return metrics


def _load_model_for_evaluation(checkpoint, device):
    from analysis.v2_v2clean_failure_audit import _load_model
    from train import restore_state_org_reference_for_evaluation

    model, config, packet = _load_model(checkpoint, torch.device(device))
    restore_state_org_reference_for_evaluation(model, packet)
    return model.eval(), config, packet


def _metric_result(checkpoint, role, config, packet, metrics, provenance):
    class_names = list(config.classes)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    return {
        "checkpoint": str(checkpoint),
        "role": role,
        "checkpoint_epoch": int(packet.get("epoch", -1)),
        "recorded_global_temporal_shift": float(
            packet.get("global_temporal_shift", 0.)
        ),
        "inference_shift": 0,
        "num_classes": len(class_names),
        "sample_count": int(np.asarray(metrics["support"]).sum()),
        "test_macro_f1": float(metrics["macro_f1"]),
        "fixed_class_macro_f1": float(metrics["fixed_class_macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "class_names": class_names,
        "per_class_precision": np.asarray(
            metrics["per_class_precision"], dtype=float,
        ).tolist(),
        "per_class_recall": np.asarray(
            metrics["per_class_recall"], dtype=float,
        ).tolist(),
        "per_class_f1": np.asarray(
            metrics["per_class_f1"], dtype=float,
        ).tolist(),
        "support": np.asarray(metrics["support"], dtype=np.int64).tolist(),
        "confusion_matrix": confusion.tolist(),
        "provenance": provenance,
        "test_labels_used_for_selection": False,
    }


def evaluate_checkpoint_data(checkpoint, data_root, device="cuda", batch_size=128):
    checkpoint = Path(checkpoint)
    if checkpoint.name not in ("checkpoint_best.pt", "checkpoint_last.pt"):
        raise ValueError(
            "evaluation checkpoint must be checkpoint_best.pt or checkpoint_last.pt"
        )
    role = "best" if checkpoint.name == "checkpoint_best.pt" else "final"
    model, config, packet = _load_model_for_evaluation(checkpoint, device)
    if role == "final" and int(packet.get("epoch", -1)) != int(config.epochs) - 1:
        raise ValueError("checkpoint_last.pt is not the configured final epoch")
    loader = build_official_test_loader(config, data_root, batch_size)
    metrics = evaluate_model_on_official_test(model, loader, config, device)
    return _metric_result(
        checkpoint, role, config, packet, metrics, "checkpoint_reinference",
    )


@torch.no_grad()
def evaluate(args):
    result = evaluate_checkpoint_data(
        args.checkpoint, args.data_root, args.device, args.batch_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"RELIABLE_COMPOSITION_EVAL|role={result['role']}|"
        f"epoch={result['checkpoint_epoch']}|"
        f"inference_shift={result['inference_shift']}|"
        f"test_macro_f1={result['test_macro_f1']:.6f}|"
        f"samples={result['sample_count']}"
    )


def _metrics_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion matrix must be square")
    support = confusion.sum(1)
    predicted = confusion.sum(0)
    true_positive = np.diag(confusion).astype(float)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive, support, out=np.zeros_like(true_positive),
        where=support != 0,
    )
    per_class_f1 = np.divide(
        2. * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) != 0,
    )
    observed = (support + predicted) > 0
    total = int(support.sum())
    return {
        "macro_f1": (
            float(per_class_f1[observed].mean()) if observed.any() else 0.
        ),
        "fixed_class_macro_f1": float(per_class_f1.mean()),
        "accuracy": float(true_positive.sum() / total) if total else 0.,
        "confusion_matrix": confusion,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": per_class_f1,
        "support": support,
    }


def _checkpoint_config(packet):
    raw = packet.get("config", {})
    if isinstance(raw, dict):
        return SimpleNamespace(**raw)
    return raw


def _direct_final_result(fold, target_name):
    artifacts = verified_final_artifacts(fold, target_name)
    if artifacts is None:
        return None
    _, confusion_path = artifacts
    with confusion_path.open("rb") as stream:
        confusion = np.asarray(pickle.load(stream), dtype=np.int64)
    checkpoint = Path(fold) / "checkpoint_last.pt"
    packet = {}
    if checkpoint.is_file():
        packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _checkpoint_config(packet)
    classes = list(getattr(config, "classes", ()))
    if len(classes) != confusion.shape[0]:
        classes = [f"class_{index}" for index in range(confusion.shape[0])]
    config = SimpleNamespace(classes=classes)
    return _metric_result(
        checkpoint, "final", config, packet,
        _metrics_from_confusion(confusion), "confusion_recomputed",
    )


def _first_existing(candidates):
    paths = [Path(path) for path in candidates if path is not None]
    return next((path for path in paths if path.exists()), paths[0])


def _method_roots(args):
    return {
        "V2": _first_existing((
            args.v2_root, "outputs/structure_proto_v2_4tasks_seed1",
        )),
        "V2clean": _first_existing((
            args.v2clean_root, "outputs/recheck_v2clean_final_seed1",
            "outputs/structure_proto_v2clean_4tasks_seed1",
        )),
        "E": _first_existing((
            args.e_root, "outputs/recheck_e_final_seed1",
            "outputs/structure_phase_equivariance_4tasks_seed1",
        )),
        "A": Path(args.reliable_root or "outputs/reliable_composition_ab"),
        "B": Path(args.reliable_root or "outputs/reliable_composition_ab"),
    }


def _fold_for(method, root, task):
    if method in ("A", "B"):
        return root / method / "seed1" / "uda" / f"{task}_seed1" / "fold_0"
    return root / "uda" / f"{task}_seed1" / "fold_0"


def _result_class_rows(method, task, result):
    if result is None:
        return []
    rows = []
    for index, class_name in enumerate(result["class_names"]):
        rows.append({
            "method": method, "task": task, "checkpoint_role": result["role"],
            "class_index": index, "class_name": class_name,
            "support": result["support"][index],
            "precision": result["per_class_precision"][index],
            "recall": result["per_class_recall"][index],
            "f1": result["per_class_f1"][index],
            "provenance": result["provenance"],
        })
    return rows


def _evaluate_role(path, data_root, device, batch_size):
    if not Path(path).is_file():
        return None, f"missing checkpoint: {path}"
    try:
        return evaluate_checkpoint_data(path, data_root, device, batch_size), ""
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def _write_comparison(path, rows):
    by_key = {(row["method"], row["task"]): row for row in rows}
    lines = [
        "# Unified UDA Final Test evaluation",
        "",
        "Protocol: checkpoint_last.pt; no test-time shift override; official "
        "Test loader; original TimeMatch observed-class Macro-F1. Fixed-class "
        "Macro-F1 is supplemental only.",
        "",
        "| Method | " + " | ".join(TASKS) + " | 4-task Mean |",
        "|---|" + "---:|" * (len(TASKS) + 1),
    ]
    for method in METHODS:
        values, cells = [], []
        for task in TASKS:
            row = by_key[(method, task)]
            value = row.get("final_test_macro_f1")
            if value in (None, ""):
                cells.append("MISSING")
            else:
                value = float(value)
                values.append(value)
                cells.append(f"{value:.4f}")
        mean = f"{np.mean(values):.4f}" if len(values) == len(TASKS) else "MISSING"
        lines.append(f"| {method} | " + " | ".join(cells) + f" | {mean} |")
    lines.extend(("", "| Method | Task | Final provenance | Status |", "|---|---|---|---|"))
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['task']} | "
            f"{row.get('final_provenance', '')} | {row['status']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def recheck(args):
    roots = _method_roots(args)
    task_rows, class_rows = [], []
    for method in METHODS:
        for task in TASKS:
            fold = _fold_for(method, roots[method], task)
            roles = checkpoint_roles(fold)
            target_name = TASK_TARGETS[task].replace("/", "_")
            final = _direct_final_result(fold, target_name)
            final_error = ""
            if final is None:
                final, final_error = _evaluate_role(
                    roles["final"], args.data_root, args.device, args.batch_size,
                )
            best, best_error = _evaluate_role(
                roles["best"], args.data_root, args.device, args.batch_size,
            )
            status = "complete" if final is not None else "unavailable"
            row = {
                "method": method, "task": task, "status": status,
                "final_checkpoint": str(roles["final"]),
                "final_checkpoint_epoch": (
                    final["checkpoint_epoch"] if final is not None else ""
                ),
                "final_test_macro_f1": (
                    final["test_macro_f1"] if final is not None else ""
                ),
                "final_fixed_class_macro_f1": (
                    final["fixed_class_macro_f1"] if final is not None else ""
                ),
                "final_inference_shift": 0,
                "final_num_classes": final["num_classes"] if final is not None else "",
                "final_provenance": final["provenance"] if final is not None else "",
                "final_confusion_matrix": json.dumps(
                    final["confusion_matrix"] if final is not None else None,
                ),
                "final_error": final_error,
                "best_checkpoint": str(roles["best"]),
                "best_checkpoint_epoch": (
                    best["checkpoint_epoch"] if best is not None else ""
                ),
                "best_test_macro_f1": best["test_macro_f1"] if best is not None else "",
                "best_fixed_class_macro_f1": (
                    best["fixed_class_macro_f1"] if best is not None else ""
                ),
                "best_inference_shift": 0,
                "best_provenance": best["provenance"] if best is not None else "",
                "best_confusion_matrix": json.dumps(
                    best["confusion_matrix"] if best is not None else None,
                ),
                "best_error": best_error,
                "test_labels_used_for_checkpoint_selection": False,
            }
            task_rows.append(row)
            class_rows.extend(_result_class_rows(method, task, final))
            class_rows.extend(_result_class_rows(method, task, best))
            print(
                f"EVALUATION_RECHECK|method={method}|task={task}|status={status}|"
                f"final={row['final_test_macro_f1']}|best={row['best_test_macro_f1']}"
            )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "final_per_task.csv", task_rows)
    _write_csv(output / "final_per_class.csv", class_rows)
    _write_comparison(output / "comparison.md", task_rows)


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
                "best_epoch": best.get("checkpoint_epoch", best.get("epoch")),
                "best_test_macro_f1": best["test_macro_f1"],
                "best_fixed_class_macro_f1": best.get("fixed_class_macro_f1", ""),
                "final_epoch": final.get("checkpoint_epoch", final.get("epoch")),
                "final_test_macro_f1": final["test_macro_f1"],
                "final_fixed_class_macro_f1": final.get("fixed_class_macro_f1", ""),
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
    unified = sub.add_parser("recheck")
    unified.add_argument("--data-root", required=True, type=Path)
    unified.add_argument("--output", type=Path, default=Path("outputs/evaluation_recheck"))
    unified.add_argument("--device", default="cuda")
    unified.add_argument("--batch-size", type=int, default=128)
    unified.add_argument("--v2-root", type=Path)
    unified.add_argument("--v2clean-root", type=Path)
    unified.add_argument("--e-root", type=Path)
    unified.add_argument("--reliable-root", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "evaluate":
        evaluate(args)
    elif args.command == "summarize":
        summarize(args)
    elif args.command == "recheck":
        recheck(args)
    else:
        raise SystemExit(report_checkpoint_status(args))


if __name__ == "__main__":
    main()
