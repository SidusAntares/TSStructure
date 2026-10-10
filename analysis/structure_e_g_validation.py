"""Evaluate E/G checkpoints and audit cross-domain phase-moment structure."""

import argparse
import csv
import datetime as dt
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_representation_chain_audit import deterministic_loader
from analysis.summarize_reliable_composition_ab import (
    _metrics_from_confusion,
    evaluate_checkpoint_data,
    official_create_splits,
    official_prepare_data_protocol,
)


TASKS = {
    "AT1_DK1": ("AT1", "austria/33UVP/2017", "DK1", "denmark/32VNH/2017"),
    "FR1_FR2": ("FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
    "FR2_DK1": ("FR2", "france/31TCJ/2017", "DK1", "denmark/32VNH/2017"),
    "DK1_AT1": ("DK1", "denmark/32VNH/2017", "AT1", "austria/33UVP/2017"),
}
VIS_TASKS = ("FR2_DK1", "FR1_FR2")
PREFERRED_CLASSES = ("spring_barley", "winter_barley", "winter_wheat")
EPS = 1e-12


def _write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(fields or (rows[0].keys() if rows else ()))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fields(line):
    return {
        key: value for key, value in re.findall(r"(?:^|\|)([^|=]+)=([^|]+)", line)
    }


def parse_epoch_history(text, method, task):
    """Combine validation, pseudo audit, and gradient state into epoch rows."""
    rows = defaultdict(lambda: {
        "method": method, "task": task, "validation_macro_f1": "",
        "accepted_pseudo_count": "", "accepted_pseudo_accuracy": "",
        "target_query_gradient_open": "",
    })
    current_epoch = None
    for line in text.splitlines():
        values = _fields(line)
        if "epoch" in values:
            try:
                current_epoch = int(values["epoch"])
            except ValueError:
                current_epoch = None
        if line.startswith("TARGET_STRUCTURE_GRADIENT|") and current_epoch is not None:
            rows[current_epoch]["target_query_gradient_open"] = (
                values.get("open", "").lower() == "true"
            )
        if current_epoch is not None and (
            line.startswith("SHAPE_V2CLEAN_EPOCH|")
            or line.startswith("PSEUDO_ORACLE_AUDIT|")
        ):
            count = values.get("accepted_pseudo_count", values.get("accepted_count"))
            accuracy = values.get("accepted_pseudo_accuracy")
            if count is not None:
                rows[current_epoch]["accepted_pseudo_count"] = int(float(count))
            if accuracy is not None:
                rows[current_epoch]["accepted_pseudo_accuracy"] = float(accuracy)
        match = re.search(r"Validation result:.*?f1=([-+0-9.eE]+)", line)
        if match and current_epoch is not None:
            rows[current_epoch]["validation_macro_f1"] = float(match.group(1))
    result = []
    for epoch in sorted(rows):
        row = rows[epoch]
        row["epoch"] = epoch
        result.append(row)
    return result


def raw_ndvi_observations(pixels, dates, class_id, sample_id):
    """Return sample-median NDVI from true B04/B08 Sentinel-2 channels."""
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 3 or pixels.shape[1] < 7:
        raise ValueError("raw pixels must be [T,10,S] with B04/B08 available")
    red = np.median(pixels[:, 2, :], axis=-1)
    nir = np.median(pixels[:, 6, :], axis=-1)
    ndvi = np.divide(nir - red, nir + red, out=np.zeros_like(red), where=np.abs(nir + red) > EPS)
    rows = []
    for date, value in zip(dates, ndvi):
        parsed = dt.datetime.strptime(str(date), "%Y%m%d").date()
        rows.append({
            "sample_id": int(sample_id), "class_id": int(class_id),
            "date": parsed.isoformat(), "day_of_year": int(parsed.strftime("%j")),
            "ndvi": float(value),
        })
    return rows


def _canonical_class(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def select_focus_classes(class_names, source_support, target_support, min_support=20, max_classes=6):
    common = [
        index for index in range(len(class_names))
        if int(source_support[index]) >= int(min_support)
        and int(target_support[index]) >= int(min_support)
    ]
    canonical = [_canonical_class(name) for name in class_names]
    selected = []
    for wanted in PREFERRED_CLASSES:
        selected.extend(index for index in common if canonical[index] == wanted)
    remaining = [index for index in common if index not in selected]
    return (selected + remaining)[:int(max_classes)]


def _normalize(array):
    array = np.asarray(array, dtype=np.float64)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), EPS)


def prototype_geometry_rows(
    task, stage, layer, source_features, source_labels, target_features,
    target_labels, class_names,
):
    source = _normalize(source_features)
    target = _normalize(target_features)
    source_labels = np.asarray(source_labels, dtype=np.int64)
    target_labels = np.asarray(target_labels, dtype=np.int64)
    class_names = tuple(class_names)
    prototypes = np.full((len(class_names), source.shape[1]), np.nan)
    for class_id in range(len(class_names)):
        selected = source_labels == class_id
        if selected.any():
            prototypes[class_id] = _normalize(source[selected].mean(0, keepdims=True))[0]
    valid = np.isfinite(prototypes).all(1)
    similarity = target @ np.nan_to_num(prototypes).T
    similarity[:, ~valid] = -np.inf
    prediction = similarity.argmax(1)
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for truth, pred in zip(target_labels, prediction):
        confusion[int(truth), int(pred)] += 1
    metric = _metrics_from_confusion(confusion)
    rows = []
    for class_id, class_name in enumerate(class_names):
        source_selected = source_labels == class_id
        target_selected = target_labels == class_id
        if not source_selected.any() or not target_selected.any() or not valid[class_id]:
            continue
        target_proto = _normalize(target[target_selected].mean(0, keepdims=True))[0]
        correct = similarity[target_selected, class_id]
        wrong = similarity[target_selected].copy()
        wrong[:, class_id] = -np.inf
        nearest_wrong = wrong.max(1)
        margin = correct - nearest_wrong
        rows.append({
            "task": task, "stage": stage, "layer": layer,
            "class_id": class_id, "class_name": class_name,
            "source_support": int(source_selected.sum()),
            "target_support": int(target_selected.sum()),
            "same_class_prototype_cosine": float(target_proto @ prototypes[class_id]),
            "mean_correct_source_cosine": float(correct.mean()),
            "mean_nearest_wrong_source_cosine": float(nearest_wrong.mean()),
            "mean_correct_wrong_margin": float(margin.mean()),
            "median_correct_wrong_margin": float(np.median(margin)),
            "nearest_source_prototype_accuracy": float((prediction[target_selected] == class_id).mean()),
        })
    summary = {
        "task": task, "stage": stage, "layer": layer,
        "nearest_source_prototype_macro_f1": float(metric["macro_f1"]),
        "nearest_source_prototype_fixed_macro_f1": float(metric["fixed_class_macro_f1"]),
        "nearest_source_prototype_accuracy": float(metric["accuracy"]),
        "mean_same_class_prototype_cosine": float(np.mean([
            row["same_class_prototype_cosine"] for row in rows
        ])) if rows else float("nan"),
        "mean_correct_wrong_margin": float(np.mean([
            row["mean_correct_wrong_margin"] for row in rows
        ])) if rows else float("nan"),
    }
    return rows, summary


def _checkpoint_paths(args, task):
    source_alias = TASKS[task][0]
    return {
        "source": Path(args.source_root) / f"source_{source_alias}_seed1/fold_0/model.pt",
        "E_final": Path(args.e_root) / "uda" / f"{task}_seed1/fold_0/checkpoint_last.pt",
        "G_final": Path(args.cross_root) / "outputs" / "uda" / f"G_{task}_seed1/fold_0/checkpoint_last.pt",
    }


def _log_paths(args, task):
    return {
        "E": Path(args.e_log_root) / f"E_{task}.log",
        "G": Path(args.cross_root) / "logs" / f"G_{task}_seed1.log",
    }


def _require(paths):
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("\n".join(f"MISSING_CHECKPOINT|path={path}" for path in missing))


def _comparison(args, output_root):
    task_rows, class_rows, history_rows = [], [], []
    for task in TASKS:
        paths = _checkpoint_paths(args, task)
        for method, checkpoint in (("E", paths["E_final"]), ("G", paths["G_final"])):
            _require((checkpoint,))
            result = evaluate_checkpoint_data(
                checkpoint, args.data_root, args.device, args.batch_size,
            )
            task_rows.append({
                "task": task, "method": method,
                "test_macro_f1": result["test_macro_f1"],
                "fixed_closed_set_macro_f1": result["fixed_class_macro_f1"],
                "accuracy": result["accuracy"], "support": result["sample_count"],
                "checkpoint_epoch": result["checkpoint_epoch"],
                "checkpoint": str(checkpoint),
            })
            for index, class_name in enumerate(result["class_names"]):
                class_rows.append({
                    "task": task, "method": method, "class_id": index,
                    "class_name": class_name, "support": result["support"][index],
                    "precision": result["per_class_precision"][index],
                    "recall": result["per_class_recall"][index],
                    "f1": result["per_class_f1"][index],
                })
        for method, log in _log_paths(args, task).items():
            if not log.is_file():
                raise FileNotFoundError(f"MISSING_LOG|path={log}")
            rows = parse_epoch_history(log.read_text(encoding="utf-8", errors="replace"), method, task)
            if method == "G":
                states = {row["epoch"]: row["target_query_gradient_open"] for row in rows}
                expected = {epoch: epoch >= 5 for epoch in range(20)}
                if any(states.get(epoch) != value for epoch, value in expected.items()):
                    raise RuntimeError(f"G gradient log does not prove Detach5 contract: {task}")
            history_rows.extend(rows)
    _write_csv(output_root / "G_E_test_comparison.csv", task_rows)
    _write_csv(output_root / "G_E_per_class.csv", class_rows)
    _write_csv(output_root / "G_E_epoch_history.csv", history_rows)
    return task_rows


def _test_datasets(config, data_root):
    from dataset import PixelSetData
    from torchvision.transforms import transforms
    from transforms import Normalize, ToTensor

    config.data_root = str(data_root)
    indices, _ = official_prepare_data_protocol(config)
    splits = official_create_splits(
        [config.source, config.target], config.num_folds, indices,
        config.val_ratio, config.test_ratio,
    )[0]
    transform = transforms.Compose([Normalize(), ToTensor()])
    options = dict(
        classes=config.classes, closed_set=True,
        combine_spring_and_winter=getattr(config, "combine_spring_and_winter", False),
    )
    normalized = {
        "source": PixelSetData(
            data_root, config.source, transform=transform,
            indices=splits[config.source]["test"], **options,
        ),
        "target": PixelSetData(
            data_root, config.target, transform=transform,
            indices=splits[config.target]["test"], **options,
        ),
    }
    raw = {
        "source": PixelSetData(
            data_root, config.source, transform=None,
            indices=splits[config.source]["test"], **options,
        ),
        "target": PixelSetData(
            data_root, config.target, transform=None,
            indices=splits[config.target]["test"], **options,
        ),
    }
    return normalized, raw


@torch.no_grad()
def _extract_structure(model, dataset, device, batch_size, num_workers, pixel_budget):
    result = defaultdict(list)
    for batch in deterministic_loader(dataset, batch_size, num_workers, pixel_budget):
        output = model.forward_with_temporal_shift(
            batch["pixels"].to(device), batch["valid_pixels"].to(device),
            batch["positions"].to(device), batch["extra"].to(device),
            temporal_shift=0, return_dict=True,
        )
        values = {
            "shape_token": output["shape_tokens"].flatten(1),
            "strength": output["shapelet_strength"],
            "concentration": output["shapelet_concentration"],
            "phase_moment": output["shapelet_phase_moments"],
            "shape_response": output["shapelet_response"],
        }
        for name, value in values.items():
            result[name].append(value.detach().float().cpu())
        result["labels"].append(batch["label"].long().cpu())
    return {name: torch.cat(parts).numpy() for name, parts in result.items()}


def _raw_curve_rows(raw, focus, class_names, task):
    values = defaultdict(list)
    for domain, dataset in raw.items():
        for index in range(len(dataset)):
            sample = dataset[index]
            class_id = int(sample["label"])
            if class_id not in focus:
                continue
            for row in raw_ndvi_observations(
                sample["pixels"], dataset.dates, class_id, sample["parcel_index"],
            ):
                values[(domain, class_id, row["date"], row["day_of_year"])].append(row["ndvi"])
    rows = []
    for (domain, class_id, date, doy), samples in sorted(values.items()):
        samples = np.asarray(samples)
        rows.append({
            "task": task, "domain": domain, "class_id": class_id,
            "class_name": class_names[class_id], "date": date, "day_of_year": doy,
            "sample_count": int(samples.size), "ndvi_q25": float(np.quantile(samples, .25)),
            "ndvi_median": float(np.median(samples)), "ndvi_q75": float(np.quantile(samples, .75)),
        })
    return rows


def raw_curve_gap_rows(rows):
    """Compare class median NDVI curves only over their shared real-date range."""
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["task"], int(row["class_id"]), row["class_name"])].setdefault(
            row["domain"], []
        ).append(row)
    result = []
    for (task, class_id, class_name), domains in sorted(grouped.items()):
        if "source" not in domains or "target" not in domains:
            continue
        source = sorted(domains["source"], key=lambda row: row["day_of_year"])
        target = sorted(domains["target"], key=lambda row: row["day_of_year"])
        start = max(source[0]["day_of_year"], target[0]["day_of_year"])
        end = min(source[-1]["day_of_year"], target[-1]["day_of_year"])
        if end < start:
            continue
        grid = np.arange(start, end + 1, dtype=float)
        source_values = np.interp(
            grid, [row["day_of_year"] for row in source],
            [row["ndvi_median"] for row in source],
        )
        target_values = np.interp(
            grid, [row["day_of_year"] for row in target],
            [row["ndvi_median"] for row in target],
        )
        correlation = (
            float(np.corrcoef(source_values, target_values)[0, 1])
            if source_values.size > 1
            and source_values.std() > EPS and target_values.std() > EPS
            else float("nan")
        )
        result.append({
            "task": task, "class_id": class_id, "class_name": class_name,
            "shared_start_doy": int(start), "shared_end_doy": int(end),
            "shared_day_count": int(grid.size),
            "mean_absolute_ndvi_gap": float(np.abs(source_values - target_values).mean()),
            "median_absolute_ndvi_gap": float(np.median(np.abs(source_values - target_values))),
            "source_target_curve_correlation": correlation,
        })
    return result


def _plot_raw_curves(rows, output, focus, class_names):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(focus), 1, figsize=(10, 3.2 * len(focus)), sharex=True)
    axes = np.atleast_1d(axes)
    colors = {"source": "#1f77b4", "target": "#d95f02"}
    for axis, class_id in zip(axes, focus):
        for domain in ("source", "target"):
            selected = sorted(
                (row for row in rows if row["class_id"] == class_id and row["domain"] == domain),
                key=lambda row: row["day_of_year"],
            )
            x = np.asarray([row["day_of_year"] for row in selected])
            median = np.asarray([row["ndvi_median"] for row in selected])
            low = np.asarray([row["ndvi_q25"] for row in selected])
            high = np.asarray([row["ndvi_q75"] for row in selected])
            count = min((row["sample_count"] for row in selected), default=0)
            axis.plot(x, median, color=colors[domain], label=f"{domain} (n≥{count})")
            axis.fill_between(x, low, high, color=colors[domain], alpha=.18)
        axis.set_title(str(class_names[class_id]))
        axis.set_ylabel("NDVI")
        axis.grid(alpha=.2)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Day of year (2017 acquisition dates)")
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_strength_heatmap(source, target, labels_s, labels_t, focus, class_names, output):
    import matplotlib.pyplot as plt

    matrices = []
    for features, labels in ((source, labels_s), (target, labels_t)):
        matrices.append(np.stack([
            features[labels == class_id].mean(0) for class_id in focus
        ]))
    low = min(matrix.min() for matrix in matrices)
    high = max(matrix.max() for matrix in matrices)
    figure, axes = plt.subplots(1, 2, figsize=(14, max(3.5, .55 * len(focus) + 2)), sharey=True)
    image = None
    for axis, matrix, domain in zip(axes, matrices, ("Source", "Target")):
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=low, vmax=high)
        axis.set_title(f"{domain} mean anchor strength")
        axis.set_xlabel("Anchor ID")
        axis.set_xticks(range(matrix.shape[1]))
    axes[0].set_yticks(range(len(focus)), [class_names[index] for index in focus])
    figure.colorbar(image, ax=axes, label="Strength", fraction=.025, pad=.03)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _visualize(args, output_root):
    from analysis.v2_v2clean_failure_audit import _load_model

    geometry_rows, geometry_summary, curve_rows = [], [], []
    manifest = []
    device = torch.device(args.device)
    for task in VIS_TASKS:
        paths = _checkpoint_paths(args, task)
        _require(paths.values())
        protocol_model, protocol_config, _ = _load_model(paths["E_final"], device)
        del protocol_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        normalized, raw = _test_datasets(protocol_config, args.data_root)
        support_s = np.bincount(normalized["source"].get_labels(), minlength=len(protocol_config.classes))
        support_t = np.bincount(normalized["target"].get_labels(), minlength=len(protocol_config.classes))
        focus = select_focus_classes(
            protocol_config.classes, support_s, support_t,
            args.min_class_support, args.max_focus_classes,
        )
        if not focus:
            raise RuntimeError(f"no common classes meet support threshold for {task}")
        task_curves = _raw_curve_rows(raw, focus, protocol_config.classes, task)
        curve_rows.extend(task_curves)
        _plot_raw_curves(
            task_curves, output_root / f"VIS_{task}_raw_ndvi.png",
            focus, protocol_config.classes,
        )
        for stage, checkpoint in paths.items():
            model, config, packet = _load_model(checkpoint, device)
            if list(config.classes) != list(protocol_config.classes):
                raise RuntimeError(f"class protocol mismatch: {task} {stage}")
            source = _extract_structure(
                model, normalized["source"], device, args.batch_size,
                args.num_workers, args.pixel_budget,
            )
            target = _extract_structure(
                model, normalized["target"], device, args.batch_size,
                args.num_workers, args.pixel_budget,
            )
            for layer in ("shape_token", "strength", "shape_response"):
                rows, summary = prototype_geometry_rows(
                    task, stage, layer, source[layer], source["labels"],
                    target[layer], target["labels"], config.classes,
                )
                geometry_rows.extend(rows)
                geometry_summary.append(summary)
            _plot_strength_heatmap(
                source["strength"], target["strength"], source["labels"],
                target["labels"], focus, config.classes,
                output_root / f"VIS_{task}_{stage}_strength_heatmap.png",
            )
            manifest.append({
                "task": task, "stage": stage, "checkpoint": str(checkpoint),
                "checkpoint_epoch": int(packet.get("epoch", -1)),
                "source_samples": int(source["labels"].size),
                "target_samples": int(target["labels"].size),
                "structure_temporal_shift": 0,
                "target_labels_audit_only": True,
                "raw_curve": "sample-median Sentinel-2 B04/B08 NDVI",
            })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _write_csv(output_root / "VIS_raw_ndvi.csv", curve_rows)
    _write_csv(output_root / "VIS_raw_ndvi_domain_gap.csv", raw_curve_gap_rows(curve_rows))
    _write_csv(output_root / "VIS_structure_geometry.csv", geometry_rows)
    _write_csv(output_root / "VIS_structure_geometry_summary.csv", geometry_summary)
    (output_root / "VIS_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )


def run(args):
    output_root = Path(args.cross_root) / "outputs" / "VIS"
    output_root.mkdir(parents=True, exist_ok=True)
    _comparison(args, output_root)
    _visualize(args, output_root)
    print(f"G_VIS_FINISHED|output={output_root}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--cross-root", default="experiments/structure_e_cross_seed1")
    parser.add_argument("--source-root", default="outputs/structure_phase_moment_4tasks_seed1/source")
    parser.add_argument("--e-root", default="outputs/structure_phase_equivariance_4tasks_seed1")
    parser.add_argument("--e-log-root", default="logs/structure_phase_equivariance_4tasks_seed1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--min-class-support", type=int, default=20)
    parser.add_argument("--max-focus-classes", type=int, default=6)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
