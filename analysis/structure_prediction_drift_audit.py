"""Read-only audit of prediction transitions and structure drift across checkpoints."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DOMAINS = {
    "AT1": "austria/33UVP/2017",
    "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017",
    "FR2": "france/31TCJ/2017",
}

TASKS = {
    "AT1_DK1": ("AT1", "DK1"),
    "DK1_AT1": ("DK1", "AT1"),
    "FR1_FR2": ("FR1", "FR2"),
}

STRUCTURE_CONFIG_KEYS = (
    "shape_representation", "shape_injection", "state_org_readout",
    "shape_dim", "shapelet_count", "shape_window_scales",
    "shape_window_stride", "shape_resample_length", "fourier_num_modes",
)

PAIR_METRICS = (
    "mean_window_token_cosine", "mean_absolute_similarity_change",
    "shape_response_cosine", "window_top1_anchor_agreement", "presence_cosine",
)


def prediction_transition(best_correct, final_correct):
    if bool(best_correct) and not bool(final_correct):
        return "correct_to_wrong"
    if not bool(best_correct) and bool(final_correct):
        return "wrong_to_correct"
    return "always_correct" if bool(best_correct) else "always_wrong"


def align_stage_records(stages):
    indexed = {}
    for stage, records in stages.items():
        stage_index = {}
        for record in records:
            sample_id = int(record["sample_id"])
            if sample_id in stage_index:
                raise ValueError(f"duplicate sample_id in {stage}: {sample_id}")
            stage_index[sample_id] = record
        indexed[stage] = stage_index
    id_sets = {stage: set(values) for stage, values in indexed.items()}
    first = next(iter(id_sets.values()), set())
    if any(values != first for values in id_sets.values()):
        details = {stage: len(values) for stage, values in id_sets.items()}
        raise ValueError(f"sample ID sets differ across stages: {details}")
    return [
        {"sample_id": sample_id, **{
            stage: indexed[stage][sample_id] for stage in indexed
        }} for sample_id in sorted(first)
    ]


def true_class_margin(logits, labels):
    labels = labels.long()
    true = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], -torch.inf)
    return true - masked.max(dim=1).values


def structure_pair_metrics(left, right):
    return {
        "mean_window_token_cosine": F.cosine_similarity(
            left["tokens"], right["tokens"], dim=-1,
        ).mean(dim=1),
        "mean_absolute_similarity_change": (
            left["similarity"] - right["similarity"]
        ).abs().mean(dim=(1, 2)),
        "shape_response_cosine": F.cosine_similarity(
            left["response"], right["response"], dim=-1,
        ),
        "window_top1_anchor_agreement": (
            left["similarity"].argmax(dim=-1)
            == right["similarity"].argmax(dim=-1)
        ).float().mean(dim=1),
        "presence_cosine": F.cosine_similarity(
            left["presence"], right["presence"], dim=-1,
        ),
    }


@torch.no_grad()
def fixed_source_structure(source_model, current_model, pixels, mask, positions, extra):
    """Measure current PSE in the immutable source structure coordinate system."""
    spatial = current_model.spatial_encoder(pixels, mask, extra)
    branch = source_model.structure_branch
    structure = branch(
        spatial, positions,
        include_legacy_query=getattr(source_model, "shape_injection", "") == "current_query",
    )
    evidence = source_model._shape_evidence(structure)
    presence = structure.get("shapelet_presence")
    if presence is None:
        presence = structure.get("shapelet_strength")
    required = {
        "tokens": structure.get("shape_tokens"),
        "similarity": structure.get("shapelet_similarity"),
        "response": structure.get("shapelet_response"),
        "presence": presence,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"fixed source structure output missing: {missing}")
    return {**required, "shape_logits": source_model.shape_classifier(evidence)}


def _config_dict(packet):
    config = packet.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint config missing")
    return dict(config)


def _missing_status(role):
    return {
        "source": "MISSING_SOURCE", "best": "MISSING_BEST", "final": "MISSING_FINAL",
    }[role]


def inspect_checkpoint(path, role, expected_source, expected_target):
    path = Path(path)
    if role not in ("source", "best", "final"):
        raise ValueError(f"unknown checkpoint role: {role}")
    if not path.exists():
        return {
            "role": role, "path": str(path), "status": _missing_status(role),
            "epoch": None, "global_shift": None, "config": None,
        }
    if role == "final" and path.name != "checkpoint_last.pt":
        raise ValueError("final checkpoint must be checkpoint_last.pt")
    if role == "best" and path.name not in ("checkpoint_best.pt", "model.pt"):
        raise ValueError("best checkpoint must be checkpoint_best.pt or legacy model.pt")
    packet = torch.load(path, map_location="cpu", weights_only=False)
    config = _config_dict(packet)
    if config.get("source") != expected_source or config.get("target") != expected_target:
        raise ValueError(
            f"checkpoint domain mismatch for {role}: "
            f"{config.get('source')}->{config.get('target')}"
        )
    epoch = packet.get("epoch")
    shift = packet.get("global_temporal_shift", packet.get("initial_shift"))
    if role in ("best", "final"):
        if config.get("method") != "timematch":
            raise ValueError(f"{role} checkpoint is not TimeMatch")
        if shift is None:
            raise ValueError(f"{role} checkpoint has no saved global shift")
    if role == "final":
        epochs = config.get("epochs")
        if epoch is None or epochs is None or int(epoch) != int(epochs) - 1:
            raise ValueError(
                f"final checkpoint epoch mismatch: epoch={epoch}, configured epochs={epochs}"
            )
    return {
        "role": role, "path": str(path), "status": "OK",
        "epoch": None if epoch is None else int(epoch),
        "global_shift": None if shift is None else int(round(float(shift))),
        "config": config,
    }


def experiment_specs(output_root):
    root = Path(output_root)
    specs = []
    for method, folder in (
        ("v2", "structure_proto_v2_4tasks_seed1"),
        ("v2clean", "structure_proto_v2clean_4tasks_seed1"),
    ):
        base = root / folder
        for task in ("DK1_AT1", "FR1_FR2"):
            source, _ = TASKS[task]
            fold = base / "uda" / f"{task}_seed1" / "fold_0"
            specs.append({
                "method": method, "task": task,
                "source": base / "source" / f"source_{source}_seed1" / "fold_0" / "model.pt",
                "best": fold / "checkpoint_best.pt",
                "final": fold / "checkpoint_last.pt",
            })
    foundation = root / "state_org_foundation"
    fold = foundation / "organization" / "presence" / "uda" / "AT1_DK1_seed1" / "fold_0"
    specs.append({
        "method": "state_org_foundation_presence", "task": "AT1_DK1",
        "source": foundation / "source" / "presence" / "source_AT1_seed1" / "fold_0" / "model.pt",
        "best": fold / "checkpoint_best.pt",
        "final": fold / "checkpoint_last.pt",
    })
    return specs


def _resolve_best(path):
    path = Path(path)
    if path.exists():
        return path
    legacy = path.with_name("model.pt")
    return legacy if legacy.exists() else path


def _spec(output_root, method, task):
    for item in experiment_specs(output_root):
        if item["method"] == method and item["task"] == task:
            item = dict(item)
            item["best"] = _resolve_best(item["best"])
            return item
    raise ValueError(f"unsupported audit: {method} {task}")


def _normalized_config(config):
    config = dict(config)
    config.setdefault("shape_representation", "current")
    config.setdefault("shape_injection", "current_query")
    config.setdefault("structure_shift_mode", "none")
    return config


def _assert_structure_compatible(source, current, stage):
    left, right = _normalized_config(source), _normalized_config(current)
    differences = {
        key: (left.get(key), right.get(key)) for key in STRUCTURE_CONFIG_KEYS
        if _comparable(left.get(key)) != _comparable(right.get(key))
    }
    if differences:
        raise ValueError(f"structure configuration mismatch at {stage}: {differences}")


def _assert_split_compatible(source, current, stage):
    keys = ("seed", "val_ratio", "test_ratio", "classes", "combine_spring_and_winter")
    differences = {
        key: (source.get(key), current.get(key)) for key in keys
        if _comparable(source.get(key)) != _comparable(current.get(key))
    }
    if differences:
        raise ValueError(f"data split configuration mismatch at {stage}: {differences}")


def _comparable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_comparable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _comparable(item)) for key, item in value.items()))
    return value


def _load_model(path, device):
    from analysis.v2_v2clean_failure_audit import _load_model as load_model
    from train import restore_state_org_reference_for_evaluation

    model, config, packet = load_model(path, device)
    restore_state_org_reference_for_evaluation(model, packet)
    return model.eval(), config, packet


def _sample_ids(batch):
    for key in ("parcel_index", "index"):
        if key in batch:
            return batch[key].long().cpu()
    raise KeyError("audit batch has no unique parcel_index or index")


def _detach_item(structure, index):
    return {
        key: value[index].detach().float().cpu() for key, value in structure.items()
        if key in ("tokens", "similarity", "response", "presence")
    }


@torch.no_grad()
def _collect_stage(
    stage, model, source_model, dataset, device, shift, fixed_shift, batch_size,
):
    from analysis.state_org_feasibility_audit import _loader, _move

    records = []
    for raw in _loader(dataset, batch_size):
        ids = _sample_ids(raw)
        batch = _move(raw, device)
        pixels, mask = batch["pixels"], batch["valid_pixels"]
        positions, extra = batch["positions"], batch["extra"]
        labels = batch["label"].long()
        output = model.forward_with_temporal_shift(
            pixels, mask, positions, extra, temporal_shift=int(shift), return_dict=True,
        )
        fixed_output = model.forward_with_temporal_shift(
            pixels, mask, positions, extra, temporal_shift=int(fixed_shift), return_dict=True,
        )
        measured = fixed_source_structure(
            source_model, model, pixels, mask, positions, extra,
        )
        probabilities = output["logits"].softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)
        fixed_probabilities = fixed_output["logits"].softmax(dim=1)
        fixed_confidence, fixed_prediction = fixed_probabilities.max(dim=1)
        margins = true_class_margin(measured["shape_logits"], labels)
        for index in range(labels.shape[0]):
            label = int(labels[index])
            pred = int(prediction[index])
            fixed_pred = int(fixed_prediction[index])
            records.append({
                "sample_id": int(ids[index]), "true_label": label,
                "prediction": pred, "confidence": float(confidence[index]),
                "correct": pred == label,
                "fixed_prediction": fixed_pred,
                "fixed_confidence": float(fixed_confidence[index]),
                "fixed_correct": fixed_pred == label,
                "shape_margin": float(margins[index]),
                "structure": _detach_item(measured, index),
                "stage": stage,
            })
    return records


def _scalar_pair_metrics(left, right):
    batched_left = {key: value.unsqueeze(0) for key, value in left.items()}
    batched_right = {key: value.unsqueeze(0) for key, value in right.items()}
    return {
        key: float(value[0]) for key, value in
        structure_pair_metrics(batched_left, batched_right).items()
    }


def _per_sample_rows(method, task, aligned):
    rows = []
    for entry in aligned:
        source, best, final = entry["source"], entry["best"], entry["final"]
        labels = {source["true_label"], best["true_label"], final["true_label"]}
        if len(labels) != 1:
            raise ValueError(f"label mismatch for sample {entry['sample_id']}: {labels}")
        row = {
            "method": method, "task": task, "sample_id": entry["sample_id"],
            "true_label": source["true_label"],
            "transition": prediction_transition(best["correct"], final["correct"]),
            "fixed_shift_transition": prediction_transition(
                best["fixed_correct"], final["fixed_correct"],
            ),
        }
        for stage, record in (("source", source), ("best", best), ("final", final)):
            for key in (
                "prediction", "confidence", "correct", "fixed_prediction",
                "fixed_confidence", "fixed_correct", "shape_margin",
            ):
                row[f"{key}_{stage}"] = record[key]
        for name, left, right in (
            ("source_best", source, best),
            ("best_final", best, final),
            ("source_final", source, final),
        ):
            values = _scalar_pair_metrics(left["structure"], right["structure"])
            row.update({f"{key}_{name}": value for key, value in values.items()})
        row.update({
            "margin_change_source_best": best["shape_margin"] - source["shape_margin"],
            "margin_change_best_final": final["shape_margin"] - best["shape_margin"],
            "margin_change_source_final": final["shape_margin"] - source["shape_margin"],
            "confidence_change_best_final": final["confidence"] - best["confidence"],
            "fixed_confidence_change_best_final": (
                final["fixed_confidence"] - best["fixed_confidence"]
            ),
        })
        rows.append(row)
    return rows


def _finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    return float(values.mean()), float(np.median(values))


def _summary_row(method, task, transition, rows, class_id=None):
    row = {
        "method": method, "task": task, "transition": transition,
        "sample_count": len(rows),
    }
    if class_id is not None:
        row["true_label"] = class_id
    metrics = [f"{name}_best_final" for name in PAIR_METRICS] + [
        "margin_change_best_final", "confidence_change_best_final",
        "fixed_confidence_change_best_final",
    ]
    for metric in metrics:
        mean, median = _finite_summary([item[metric] for item in rows])
        row[f"mean_{metric}"] = mean
        row[f"median_{metric}"] = median
    row["fixed_shift_transition_agreement"] = float(np.mean([
        item["transition"] == item["fixed_shift_transition"] for item in rows
    ])) if rows else math.nan
    return row


def summarize_rows(rows):
    transitions = ("correct_to_wrong", "wrong_to_correct", "always_correct", "always_wrong")
    transition_rows, class_rows = [], []
    method, task = rows[0]["method"], rows[0]["task"]
    for transition in transitions:
        selected = [row for row in rows if row["transition"] == transition]
        transition_rows.append(_summary_row(method, task, transition, selected))
    classes = sorted({int(row["true_label"]) for row in rows})
    for class_id in classes:
        for transition in transitions:
            selected = [
                row for row in rows
                if int(row["true_label"]) == class_id and row["transition"] == transition
            ]
            class_rows.append(_summary_row(
                method, task, transition, selected, class_id=class_id,
            ))
    return transition_rows, class_rows


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def audit(args):
    spec = _spec(args.checkpoint_root, args.method, args.task)
    source_name, target_name = TASKS[args.task]
    source_domain, target_domain = DOMAINS[source_name], DOMAINS[target_name]
    metadata = {
        "method": args.method, "task": args.task,
        "checkpoints": {
            "source": inspect_checkpoint(spec["source"], "source", source_domain, source_domain),
            "best": inspect_checkpoint(spec["best"], "best", source_domain, target_domain),
            "final": inspect_checkpoint(spec["final"], "final", source_domain, target_domain),
        },
        "status": "PENDING", "actual_sample_count": 0,
        "structure_coordinate": "fixed_source_structure_with_current_checkpoint_pse",
        "structure_calendar_shift_days": 0,
    }
    missing = [
        item["status"] for item in metadata["checkpoints"].values()
        if item["status"].startswith("MISSING_")
    ]
    part = Path(args.output_part)
    part.mkdir(parents=True, exist_ok=True)
    if missing:
        metadata["status"] = missing[0]
        (part / "manifest.json").write_text(
            json.dumps(_json_safe(metadata), indent=2), encoding="utf-8",
        )
        print(f"STRUCTURE_DRIFT_MISSING|method={args.method}|task={args.task}|status={missing[0]}")
        return

    device = torch.device(args.device)
    source_model, source_config, _ = _load_model(spec["source"], device)
    best_model, best_config, _ = _load_model(spec["best"], device)
    final_model, final_config, _ = _load_model(spec["final"], device)
    configs = {
        "source": vars(source_config), "best": vars(best_config), "final": vars(final_config),
    }
    for stage in ("best", "final"):
        _assert_structure_compatible(configs["source"], configs[stage], stage)
        _assert_split_compatible(configs["source"], configs[stage], stage)
    from analysis.state_org_feasibility_audit import _datasets

    datasets = _datasets(
        SimpleNamespace(**configs["source"]), source_domain, target_domain,
        args.data_root, int(configs["source"].get("seed", 1)),
    )
    target_test = datasets["target_test"]
    final_shift = metadata["checkpoints"]["final"]["global_shift"]
    records = {
        "source": _collect_stage(
            "source", source_model, source_model, target_test, device, 0,
            final_shift, args.batch_size,
        ),
        "best": _collect_stage(
            "best", best_model, source_model, target_test, device,
            metadata["checkpoints"]["best"]["global_shift"], final_shift,
            args.batch_size,
        ),
        "final": _collect_stage(
            "final", final_model, source_model, target_test, device,
            final_shift, final_shift, args.batch_size,
        ),
    }
    rows = _per_sample_rows(args.method, args.task, align_stage_records(records))
    transition_rows, class_rows = summarize_rows(rows)
    _write_csv(part / "per_sample.csv", rows)
    _write_csv(part / "transition_summary.csv", transition_rows)
    _write_csv(part / "per_class_summary.csv", class_rows)
    metadata.update({
        "status": "OK", "actual_sample_count": len(rows),
        "fixed_prediction_shift_days": final_shift,
        "model_config": {
            stage: {key: _json_safe(config.get(key)) for key in STRUCTURE_CONFIG_KEYS}
            for stage, config in configs.items()
        },
    })
    (part / "manifest.json").write_text(
        json.dumps(_json_safe(metadata), indent=2), encoding="utf-8",
    )
    print(f"STRUCTURE_DRIFT_FINISHED|method={args.method}|task={args.task}|samples={len(rows)}")


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def merge(args):
    parts = Path(args.parts_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    per_sample, transitions, classes, manifests = [], [], [], []
    for manifest_path in sorted(parts.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        if manifest.get("status") != "OK":
            continue
        folder = manifest_path.parent
        per_sample.extend(_read_csv(folder / "per_sample.csv"))
        transitions.extend(_read_csv(folder / "transition_summary.csv"))
        classes.extend(_read_csv(folder / "per_class_summary.csv"))
    _write_csv(output / "per_sample.csv", per_sample)
    _write_csv(output / "transition_summary.csv", transitions)
    _write_csv(output / "per_class_summary.csv", classes)
    (output / "checkpoint_manifest.json").write_text(
        json.dumps(manifests, indent=2), encoding="utf-8",
    )
    lines = [
        "# Structure Drift–Prediction Transition Audit", "",
        "This report lists observed associations only; it does not infer causality.", "",
        "| Method | Task | Status | Samples |", "|---|---|---:|---:|",
    ]
    for item in manifests:
        lines.append(
            f"| {item['method']} | {item['task']} | {item['status']} | "
            f"{item.get('actual_sample_count', 0)} |"
        )
    lines.extend(["", "## Best → Final transition summaries", ""])
    for row in transitions:
        lines.append(
            f"- {row['method']} {row['task']} {row['transition']}: "
            f"n={row['sample_count']}, "
            f"median response cosine={row.get('median_shape_response_cosine_best_final', 'NA')}, "
            f"median margin change={row.get('median_margin_change_best_final', 'NA')}"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"STRUCTURE_DRIFT_MERGED|output={output}|experiments={len(manifests)}")


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--method", required=True)
    audit_parser.add_argument("--task", required=True, choices=sorted(TASKS))
    audit_parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs"))
    audit_parser.add_argument("--data-root", type=Path, default=Path("/data/user/dataset/timematch_data"))
    audit_parser.add_argument("--output-part", type=Path, required=True)
    audit_parser.add_argument("--device", default="cuda")
    audit_parser.add_argument("--batch-size", type=int, default=128)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--parts-root", type=Path, required=True)
    merge_parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/structure_prediction_drift_seed1"),
    )
    return parser


def main():
    args = build_parser().parse_args()
    audit(args) if args.command == "audit" else merge(args)


if __name__ == "__main__":
    main()
