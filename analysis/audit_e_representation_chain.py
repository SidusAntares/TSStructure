#!/usr/bin/env python3
"""Read-only protocol, Fourier, and representation-chain audit for E/G."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_representation_chain_audit import (
    fit_source_probe,
    make_linear_probe,
    target_oracle_probe,
)
from analysis.summarize_reliable_composition_ab import (
    _load_model_for_evaluation,
    _metrics_from_confusion,
    build_official_test_loader,
    evaluate_model_on_official_test,
    official_create_splits,
    official_prepare_data_protocol,
)


TASKS = {
    "FR1_FR2": ("FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
    "FR2_DK1": ("FR2", "france/31TCJ/2017", "DK1", "denmark/32VNH/2017"),
}
CHECKPOINTS = ("Source", "E", "G")
STAGES = (
    "window_raw", "window_diff", "window_mean", "window_std", "shape_token",
    "anchor_similarity", "anchor_distribution", "anchor_match_summary",
    "strength", "concentration", "phase_moment", "shape_response", "shape_query",
)
FOCUS_CLASSES = ("spring_barley", "horsebeans", "winter_barley", "winter_wheat")


def _write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(fields or dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _canonical(name):
    return str(name).lower().replace(" ", "_").replace("-", "_")


def confusion_metrics(confusion):
    return _metrics_from_confusion(np.asarray(confusion, dtype=np.int64))


def compare_sample_manifests(official, diagnostic, split):
    """Compare (domain, parcel_index, label) rows without hiding duplicates."""
    official = [(str(d), int(i), int(y)) for d, i, y in official]
    diagnostic = [(str(d), int(i), int(y)) for d, i, y in diagnostic]
    rows = []
    for side, values in (("official", official), ("diagnostic", diagnostic)):
        counts = Counter((d, i) for d, i, _ in values)
        for (domain, parcel), count in sorted(counts.items()):
            if count > 1:
                rows.append({
                    "split": split, "domain": domain, "parcel_index": parcel,
                    "reason": f"duplicate_{side}", "official_label": "",
                    "diagnostic_label": "", "count": count, "severity": "blocking",
                })
    left = {(d, i): y for d, i, y in official}
    right = {(d, i): y for d, i, y in diagnostic}
    for key in sorted(left.keys() - right.keys()):
        rows.append({
            "split": split, "domain": key[0], "parcel_index": key[1],
            "reason": "missing_diagnostic", "official_label": left[key],
            "diagnostic_label": "", "count": 1, "severity": "blocking",
        })
    for key in sorted(right.keys() - left.keys()):
        rows.append({
            "split": split, "domain": key[0], "parcel_index": key[1],
            "reason": "extra_diagnostic", "official_label": "",
            "diagnostic_label": right[key], "count": 1, "severity": "blocking",
        })
    for key in sorted(left.keys() & right.keys()):
        if left[key] != right[key]:
            rows.append({
                "split": split, "domain": key[0], "parcel_index": key[1],
                "reason": "label_mismatch", "official_label": left[key],
                "diagnostic_label": right[key], "count": 1, "severity": "blocking",
            })
    return rows, not rows


def require_protocol_gate(passed):
    if not passed:
        raise RuntimeError("PROTOCOL_GATE_FAILED: refusing Fourier/probe audit")


def validate_checkpoint_role(filename, packet, config, role):
    if role == "final":
        if filename != "checkpoint_last.pt":
            raise ValueError("final role must use checkpoint_last.pt")
        epochs = int(config["epochs"] if isinstance(config, dict) else config.epochs)
        if int(packet.get("epoch", -1)) != epochs - 1:
            raise ValueError("checkpoint_last.pt is not the configured final epoch")
    elif role == "source" and filename != "model.pt":
        raise ValueError("source role must use model.pt")


def calendar_bin_features(values, positions, bins=24, period_days=365.0):
    """Fixed-calendar mean bins plus occupancy; fit-free and split-independent."""
    values = torch.as_tensor(values)
    positions = torch.as_tensor(positions, device=values.device)
    if values.ndim != 3 or positions.shape != values.shape[:2]:
        raise ValueError("values/positions must be [B,T,D]/[B,T]")
    index = torch.floor(torch.remainder(positions, period_days) * bins / period_days)
    index = index.long().clamp(0, bins - 1)
    output = values.new_zeros(values.shape[0], bins, values.shape[-1])
    count = values.new_zeros(values.shape[0], bins, 1)
    expanded = index.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    output.scatter_add_(1, expanded, values)
    count.scatter_add_(1, index.unsqueeze(-1), torch.ones_like(index, dtype=values.dtype).unsqueeze(-1))
    output = output / count.clamp_min(1)
    occupancy = (count > 0).to(values.dtype)
    return torch.cat((output.flatten(1), occupancy.flatten(1)), dim=1)


def fourier_fidelity_metrics(original, reconstructed, eps=1e-12):
    original = torch.as_tensor(original, dtype=torch.float64)
    reconstructed = torch.as_tensor(reconstructed, dtype=torch.float64)
    if original.shape != reconstructed.shape or original.ndim != 3:
        raise ValueError("Fourier fidelity inputs must share [B,T,D]")
    error = (reconstructed - original).square().mean((1, 2))
    energy = original.square().mean((1, 2)).clamp_min(eps)
    centered = original - original.mean(1, keepdim=True)
    total = centered.square().sum((1, 2)).clamp_min(eps)
    residual = (reconstructed - original).square().sum((1, 2))
    x = original - original.mean(1, keepdim=True)
    y = reconstructed - reconstructed.mean(1, keepdim=True)
    corr = (x * y).sum(1) / (
        x.square().sum(1).sqrt() * y.square().sum(1).sqrt()
    ).clamp_min(eps)
    valid = (x.square().sum(1) > eps) & (y.square().sum(1) > eps)
    corr = torch.where(valid, corr, torch.full_like(corr, float("nan")))
    return {
        "normalized_mse": (error / energy).cpu().numpy(),
        "r2": (1. - residual / total).cpu().numpy(),
        "channel_time_correlation": torch.nanmean(corr, 1).cpu().numpy(),
    }


def fit_probe_metrics(source_train, source_train_labels, source_val,
                      source_val_labels, target_val, target_val_labels,
                      class_ids, seed):
    source = fit_source_probe(
        source_train, source_train_labels, source_val, source_val_labels,
        target_val, target_val_labels, class_ids,
    )
    oracle = target_oracle_probe(target_val, target_val_labels, class_ids, seed)
    return {"estimator": source["estimator"], "source": source, "oracle": oracle}


def _clone_config(config):
    return SimpleNamespace(**copy.deepcopy(vars(config)))


def _seed(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))


def checkpoint_paths(args, task):
    source = TASKS[task][0]
    return {
        "Source": Path(args.source_root) / f"source_{source}_seed1/fold_0/model.pt",
        "E": Path(args.e_root) / f"{task}_seed1/fold_0/checkpoint_last.pt",
        "G": Path(args.cross_root) / f"G_{task}_seed1/fold_0/checkpoint_last.pt",
    }


def _dataset(data_root, domain, config, indices):
    from dataset import PixelSetData
    from torchvision.transforms import transforms
    from transforms import Normalize, ToTensor
    return PixelSetData(
        data_root, domain, config.classes,
        transform=transforms.Compose([Normalize(), ToTensor()]),
        indices=indices, closed_set=True,
        with_extra=getattr(config, "with_extra", False),
        combine_spring_and_winter=getattr(config, "combine_spring_and_winter", False),
    )


def official_protocol(config, data_root, seed):
    cfg = _clone_config(config); cfg.data_root = str(data_root); cfg.seed = int(seed)
    _seed(seed)
    eligible, protocol = official_prepare_data_protocol(cfg)
    splits = official_create_splits(
        [cfg.source, cfg.target], cfg.num_folds, eligible,
        cfg.val_ratio, cfg.test_ratio,
    )[0]
    datasets = {
        "source_train": _dataset(data_root, cfg.source, cfg, splits[cfg.source]["train"]),
        "source_val": _dataset(data_root, cfg.source, cfg, splits[cfg.source]["val"]),
        "target_val": _dataset(data_root, cfg.target, cfg, splits[cfg.target]["val"]),
        "target_test": _dataset(data_root, cfg.target, cfg, splits[cfg.target]["test"]),
    }
    return cfg, protocol, splits, datasets


def dataset_manifest(dataset, domain):
    return [
        (domain, int(parcel), int(label))
        for parcel, label in zip(dataset.get_parcel_indices(), dataset.get_labels())
    ]


def _collate(samples):
    max_pixels = max(sample["pixels"].shape[-1] for sample in samples)
    pixels, valid = [], []
    for sample in samples:
        pad = max_pixels - sample["pixels"].shape[-1]
        pixels.append(F.pad(sample["pixels"], (0, pad)))
        valid.append(F.pad(sample["valid_pixels"], (0, pad)))
    return {
        "pixels": torch.stack(pixels), "valid_pixels": torch.stack(valid),
        "positions": torch.stack([sample["positions"] for sample in samples]),
        "extra": torch.stack([sample["extra"] for sample in samples]),
        "label": torch.stack([sample["label"] for sample in samples]),
        "parcel_index": torch.as_tensor([int(sample["parcel_index"]) for sample in samples]),
    }


def _loader(dataset, batch_size, num_workers):
    from analysis.structure_representation_chain_audit import pixel_budget_batches
    batches = pixel_budget_batches(
        [shape[2] for shape in dataset.get_shapes()], batch_size, 8192,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=batches, collate_fn=_collate,
        num_workers=int(num_workers), pin_memory=torch.cuda.is_available(),
    )


def _load(path, device, role):
    model, config, packet = _load_model_for_evaluation(Path(path), device)
    validate_checkpoint_role(Path(path).name, packet, config, role)
    return model.eval(), config, packet


def _historical_rows(fold, task, stage, class_count):
    rows = []
    for path in sorted(Path(fold).glob("test_metrics*.json")):
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if "macro_f1" in values:
            rows.append({
                "task": task, "checkpoint": stage, "metric_source": "historical_json",
                "artifact": str(path), "macro_f1": values["macro_f1"],
                "fixed_class_macro_f1": "", "num_classes": class_count,
            })
    return rows


def _historical_log_rows(task, roots, class_count):
    rows = []
    candidates = []
    for root in roots:
        root = Path(root)
        candidates.extend((root / f"E_{task}.log", root / f"{task}.log"))
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = list(re.finditer(
            rf"Test result for .*{task}.*?f1=([-+0-9.eE]+)", text,
        ))
        if matches:
            rows.append({
                "task": task, "checkpoint": "historical_E_log",
                "metric_source": "historical_log", "artifact": str(path),
                "macro_f1": float(matches[-1].group(1)),
                "fixed_class_macro_f1": "", "num_classes": class_count,
            })
    return rows


def run_protocol(args):
    comparison, mismatches, manifests = [], [], {}
    gate = True
    for task in args.tasks:
        paths = checkpoint_paths(args, task)
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("\n".join(f"MISSING_CHECKPOINT|path={p}" for p in missing))
        models = {}
        for stage, path in paths.items():
            models[stage] = _load(path, args.device, "source" if stage == "Source" else "final")
        task_config = models["E"][1]
        expected_structure = {
            key: getattr(task_config, key, None) for key in (
                "shape_representation", "shape_injection", "fourier_num_modes",
                "shape_dim", "shape_window_scales", "shape_window_stride",
                "shapelet_count", "shape_resample_length",
            )
        }
        for stage, (_, config, _) in models.items():
            actual = {key: getattr(config, key, None) for key in expected_structure}
            if actual != expected_structure:
                gate = False
                mismatches.append({
                    "task": task, "split": "checkpoint_config", "domain": stage,
                    "parcel_index": "", "reason": "structure_config_mismatch",
                    "official_label": json.dumps(expected_structure, sort_keys=True),
                    "diagnostic_label": json.dumps(actual, sort_keys=True), "count": 1,
                    "severity": "blocking",
                })
        cfg, protocol, splits, datasets = official_protocol(task_config, args.data_root, args.seed)
        official = dataset_manifest(datasets["target_test"], cfg.target)
        # VIS is required to replay this exact official test construction.
        vis_dataset = _dataset(args.data_root, cfg.target, cfg, splits[cfg.target]["test"])
        vis = dataset_manifest(vis_dataset, cfg.target)
        rows, passed = compare_sample_manifests(official, vis, "target_test_vs_VIS")
        for row in rows: row.update({"task": task})
        mismatches.extend(rows); gate &= passed
        class_names = list(cfg.classes)
        for stage, (model, config, packet) in models.items():
            test_loader = build_official_test_loader(
                _clone_config(cfg), args.data_root, args.batch_size,
            )
            metric = evaluate_model_on_official_test(model, test_loader, cfg, args.device)
            comparison.append({
                "task": task, "checkpoint": stage, "metric_source": "checkpoint_reinference",
                "artifact": str(paths[stage]), "macro_f1": metric["macro_f1"],
                "fixed_class_macro_f1": metric["fixed_class_macro_f1"],
                "accuracy": metric["accuracy"], "sample_count": int(metric["support"].sum()),
                "num_classes": len(class_names), "checkpoint_epoch": int(packet.get("epoch", -1)),
                "recorded_global_temporal_shift": packet.get("global_temporal_shift", 0),
                "inference_shift": 0,
                "state_dict_role": "student" if getattr(config, "output_student", True) else "teacher",
                "output_student": getattr(config, "output_student", ""),
            })
            comparison.extend(_historical_rows(paths[stage].parent, task, stage, len(class_names)))
        comparison.extend(_historical_log_rows(
            task, (args.historical_log_root, Path("logs/fredn") / Path(args.historical_log_root).name),
            len(class_names),
        ))
        prior = Path(args.cross_experiment_root) / "G_E_test_comparison.csv"
        prior_candidates = (
            prior,
            Path(args.cross_experiment_root) / "outputs/VIS/G_E_test_comparison.csv",
            Path(args.cross_experiment_root) / "VIS/G_E_test_comparison.csv",
        )
        prior = next((path for path in prior_candidates if path.is_file()), prior)
        if prior.is_file():
            with prior.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    if row.get("task") == task:
                        comparison.append({
                            "task": task, "checkpoint": row.get("method", "VIS"),
                            "metric_source": "prior_VIS_csv", "artifact": str(prior),
                            "macro_f1": row.get("test_macro_f1", ""),
                            "fixed_class_macro_f1": row.get("fixed_closed_set_macro_f1", ""),
                            "accuracy": row.get("accuracy", ""), "sample_count": row.get("support", ""),
                            "num_classes": len(class_names), "inference_shift": 0,
                        })
        historical = Path(args.historical_e_root) / f"{task}_seed1/fold_0/checkpoint_last.pt"
        if historical.is_file() and historical.resolve() != paths["E"].resolve():
            historical_model, historical_config, historical_packet = _load(
                historical, args.device, "final",
            )
            historical_loader = build_official_test_loader(
                _clone_config(cfg), args.data_root, args.batch_size,
            )
            historical_metric = evaluate_model_on_official_test(
                historical_model, historical_loader, cfg, args.device,
            )
            comparison.append({
                "task": task, "checkpoint": "historical_E_recheck",
                "metric_source": "checkpoint_reinference", "artifact": str(historical),
                "macro_f1": historical_metric["macro_f1"],
                "fixed_class_macro_f1": historical_metric["fixed_class_macro_f1"],
                "accuracy": historical_metric["accuracy"],
                "sample_count": int(historical_metric["support"].sum()),
                "num_classes": len(class_names),
                "checkpoint_epoch": int(historical_packet.get("epoch", -1)),
                "recorded_global_temporal_shift": historical_packet.get("global_temporal_shift", 0),
                "inference_shift": 0,
                "state_dict_role": "student" if getattr(historical_config, "output_student", True) else "teacher",
                "output_student": getattr(historical_config, "output_student", ""),
            })
            mismatches.append({
                "task": task, "split": "historical_metric", "domain": "E",
                "parcel_index": "", "reason": "historical_E_uses_different_checkpoint",
                "official_label": str(paths["E"]), "diagnostic_label": str(historical),
                "count": 1, "severity": "informational",
            })
        vis_manifest_candidates = (
            Path(args.cross_experiment_root) / "outputs/VIS/VIS_manifest.json",
            Path(args.cross_experiment_root) / "VIS/VIS_manifest.json",
        )
        vis_manifest_path = next((p for p in vis_manifest_candidates if p.is_file()), None)
        if vis_manifest_path is not None:
            prior_manifest = json.loads(vis_manifest_path.read_text(encoding="utf-8"))
            matching = [row for row in prior_manifest if row.get("task") == task]
            for row in matching:
                if int(row.get("target_samples", -1)) != len(official):
                    gate = False
                    mismatches.append({
                        "task": task, "split": "prior_VIS_manifest", "domain": cfg.target,
                        "parcel_index": "", "reason": "prior_VIS_sample_count_mismatch",
                        "official_label": len(official), "diagnostic_label": row.get("target_samples"),
                        "count": 1, "severity": "blocking",
                    })
            mismatches.append({
                "task": task, "split": "prior_VIS_manifest", "domain": cfg.target,
                "parcel_index": "", "reason": "prior_VIS_sample_ids_not_persisted",
                "official_label": "current_replay_exact", "diagnostic_label": str(vis_manifest_path),
                "count": len(matching), "severity": "informational",
            })
        manifests[task] = {
            "classes": class_names,
            "class_to_idx": protocol["class_to_idx"],
            "split_counts": {
                domain: {name: len(ids) for name, ids in values.items()}
                for domain, values in splits.items()
            },
            "target_test_ids": sorted(int(x) for x in splits[cfg.target]["test"]),
            "checkpoints": {
                stage: {"path": str(paths[stage]), "epoch": int(models[stage][2].get("epoch", -1)),
                        "global_temporal_shift": models[stage][2].get("global_temporal_shift", 0)}
                for stage in CHECKPOINTS
            },
            "gate_passed": passed,
        }
    output = Path(args.output_root)
    _write_csv(output / "PROTOCOL_comparison.csv", comparison)
    _write_csv(output / "PROTOCOL_mismatches.csv", mismatches, (
        "task", "split", "domain", "parcel_index", "reason",
        "official_label", "diagnostic_label", "count", "severity",
    ))
    (output / "PROTOCOL_manifest.json").write_text(
        json.dumps(manifests, indent=2), encoding="utf-8",
    )
    print(f"PROTOCOL_GATE|passed={str(gate).lower()}|mismatches={len(mismatches)}")
    return gate


def _encoded_layers(model, spatial, positions):
    branch = model.structure_branch
    context = branch.prepare_context(spatial, positions)
    exposed, _ = branch.exposer.synthesize_shifted(context["coefficients"], 0)
    groups, _, centers = branch.window_extractor(exposed, return_centers=True)
    encoded = [branch.token_generator(group, return_encoded_components=True) for group in groups]
    layer = {
        "window_raw": torch.cat([x["raw_encoded"] for x in encoded], 1),
        "window_diff": torch.cat([x["diff_encoded"] for x in encoded], 1),
        "window_mean": torch.cat([x["mean_encoded"] for x in encoded], 1),
        "window_std": torch.cat([x["std_encoded"] for x in encoded], 1),
        "shape_token": torch.cat([x["shape_token"] for x in encoded], 1),
    }
    structure = branch.forward_from_context(context, structure_aug_shift=0, phase_shift=0)
    similarity = structure["shapelet_similarity"]
    distribution = torch.softmax(float(branch.shapelet_dictionary.beta) * similarity, dim=-1)
    top = similarity.topk(2, dim=-1).values
    layer.update({
        "anchor_similarity": similarity,
        "anchor_distribution": distribution,
        "anchor_match_summary": torch.stack((top[..., 0], top[..., 0] - top[..., 1]), -1),
        "strength": structure["shapelet_strength"],
        "concentration": structure["shapelet_concentration"],
        "phase_moment": structure["shapelet_phase_moments"],
        "shape_response": structure["shapelet_response"],
        "shape_query": structure["shape_class_token"],
    })
    return context["coefficients"], exposed, structure, layer


@torch.no_grad()
def collect_features(model, dataset, args, include_fourier=True, include_probe=True):
    collected = defaultdict(list)
    labels, parcels = [], []
    fidelity = defaultdict(list)
    for batch in _loader(dataset, args.batch_size, args.num_workers):
        pixels = batch["pixels"].to(args.device); valid = batch["valid_pixels"].to(args.device)
        positions = batch["positions"].to(args.device); extra = batch["extra"].to(args.device)
        spatial = model.spatial_encoder(pixels, valid, extra)
        coefficients, canonical, structure, layers = _encoded_layers(model, spatial, positions)
        if include_fourier:
            reconstructed = model.structure_branch.exposer.synthesizer(coefficients, positions)
            metrics = fourier_fidelity_metrics(spatial, reconstructed)
            for key, value in metrics.items(): fidelity[key].append(value)
            collected["pse_original"].append(calendar_bin_features(spatial, positions).cpu())
            collected["pse_reconstructed"].append(calendar_bin_features(reconstructed, positions).cpu())
        if include_probe:
            for name, value in layers.items(): collected[name].append(value.flatten(1).float().cpu())
            max_sim = layers["anchor_similarity"].max(-1).values
            max_q = layers["anchor_distribution"].max(-1).values
            collected["reliability_max_similarity"].append(max_sim.cpu())
            collected["reliability_max_probability"].append(max_q.cpu())
            collected["reliability_margin"].append(
                layers["anchor_match_summary"][..., 1].cpu()
            )
        labels.append(batch["label"].cpu()); parcels.append(batch["parcel_index"].cpu())
    return {
        "features": {k: torch.cat(v).numpy() for k, v in collected.items()},
        "labels": torch.cat(labels).numpy(), "parcel_ids": torch.cat(parcels).numpy(),
        "fidelity": {k: np.concatenate(v) for k, v in fidelity.items()},
    }


def _support(labels, count):
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=count)


def _dynamic_f1(labels, predictions):
    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def _probe_rows(task, checkpoint, representation, arrays, classes, seed):
    ids = np.arange(len(classes))
    result = fit_probe_metrics(
        arrays["source_train"]["features"][representation], arrays["source_train"]["labels"],
        arrays["source_val"]["features"][representation], arrays["source_val"]["labels"],
        arrays["target_val"]["features"][representation], arrays["target_val"]["labels"], ids, seed,
    )
    source = result["source"]; oracle = result["oracle"]
    source_pred = source["source_val_predictions"]; target_pred = source["target_predictions"]
    target_valid = oracle["predictions"] >= 0
    summary = {
        "task": task, "checkpoint": checkpoint, "representation": representation,
        "feature_dim": arrays["source_train"]["features"][representation].shape[1],
        "source_fixed_macro_f1": source["source_val_macro_f1"],
        "source_evaluable_macro_f1": _dynamic_f1(arrays["source_val"]["labels"], source_pred),
        "source_to_target_fixed_macro_f1": source["source_to_target_macro_f1"],
        "source_to_target_evaluable_macro_f1": _dynamic_f1(arrays["target_val"]["labels"], target_pred),
        "target_oracle_evaluable_macro_f1": oracle["macro_f1"],
        "target_oracle_fixed_macro_f1": float(np.nan_to_num(oracle["per_class_f1"], nan=0).mean()),
    }
    source_support = _support(arrays["source_val"]["labels"], len(classes))
    target_support = _support(arrays["target_val"]["labels"], len(classes))
    per_class = []
    for class_id, name in enumerate(classes):
        per_class.append({
            "task": task, "checkpoint": checkpoint, "representation": representation,
            "class_id": class_id, "class_name": name,
            "source_support": int(source_support[class_id]), "target_support": int(target_support[class_id]),
            "source_f1": source["source_val_per_class_f1"][class_id],
            "source_to_target_f1": source["source_to_target_per_class_f1"][class_id],
            "target_oracle_f1": oracle["per_class_f1"][class_id],
            "target_oracle_status": "OK" if np.isfinite(oracle["per_class_f1"][class_id]) else "INSUFFICIENT_SUPPORT",
            "focus_class": _canonical(name) in FOCUS_CLASSES,
        })
    return summary, per_class


def _task_arrays(args, task, include_fourier, include_probe):
    paths = checkpoint_paths(args, task)
    e_model, e_config, _ = _load(paths["E"], args.device, "final")
    cfg, _, _, datasets = official_protocol(e_config, args.data_root, args.seed)
    result = {}
    for stage, path in paths.items():
        model, _, _ = _load(path, args.device, "source" if stage == "Source" else "final")
        result[stage] = {
            name: collect_features(model, dataset, args, include_fourier, include_probe)
            for name, dataset in datasets.items() if name != "target_test"
        }
    return cfg, result


def run_fourier(args):
    fidelity_rows, probe_rows = [], []
    for task in args.tasks:
        cfg, checkpoints = _task_arrays(args, task, True, False)
        for stage, arrays in checkpoints.items():
            for split in ("source_val", "target_val"):
                values = arrays[split]; labels = values["labels"]
                for class_id, class_name in enumerate(cfg.classes):
                    selected = labels == class_id
                    if not selected.any(): continue
                    fidelity_rows.append({
                        "task": task, "checkpoint": stage, "domain_split": split,
                        "class_id": class_id, "class_name": class_name, "support": int(selected.sum()),
                        **{key: float(np.nanmean(metric[selected])) for key, metric in values["fidelity"].items()},
                    })
            for representation in ("pse_original", "pse_reconstructed"):
                row, _ = _probe_rows(task, stage, representation, arrays, cfg.classes, args.seed)
                probe_rows.append(row)
    output = Path(args.output_root)
    _write_csv(output / "FOURIER_fidelity.csv", fidelity_rows)
    _write_csv(output / "FOURIER_probe.csv", probe_rows)
    return fidelity_rows, probe_rows


def run_probe(args):
    summary_rows, class_rows, reliability_rows = [], [], []
    for task in args.tasks:
        cfg, checkpoints = _task_arrays(args, task, False, True)
        for stage, arrays in checkpoints.items():
            for representation in STAGES:
                row, rows = _probe_rows(task, stage, representation, arrays, cfg.classes, args.seed)
                summary_rows.append(row); class_rows.extend(rows)
            for split in ("source_val", "target_val"):
                values = arrays[split]; labels = values["labels"]
                max_sim = values["features"]["reliability_max_similarity"]
                max_prob = values["features"]["reliability_max_probability"]
                margin = values["features"]["reliability_margin"]
                for class_id, class_name in enumerate(cfg.classes):
                    selected = labels == class_id
                    if not selected.any(): continue
                    sim = max_sim[selected].reshape(-1); prob = max_prob[selected].reshape(-1)
                    mar = margin[selected].reshape(-1)
                    reliability_rows.append({
                        "task": task, "checkpoint": stage, "domain_split": split,
                        "class_id": class_id, "class_name": class_name, "support": int(selected.sum()),
                        "mean_window_max_similarity": float(sim.mean()),
                        "p10_window_max_similarity": float(np.quantile(sim, .1)),
                        "mean_window_top1_top2_margin": float(mar.mean()),
                        "mean_anchor_softmax_top1": float(prob.mean()),
                        "low_absolute_high_relative_rate": float(((sim <= 0.) & (prob >= .25)).mean()),
                        "low_absolute_threshold": 0., "high_relative_threshold": .25,
                    })
    output = Path(args.output_root)
    _write_csv(output / "PROBE_summary.csv", summary_rows)
    _write_csv(output / "PROBE_per_class.csv", class_rows)
    _write_csv(output / "PROBE_match_reliability.csv", reliability_rows)
    return summary_rows, class_rows, reliability_rows


def _read_rows(path):
    if not Path(path).is_file(): return []
    with Path(path).open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def write_report(args):
    root = Path(args.output_root)
    protocol = _read_rows(root / "PROTOCOL_comparison.csv")
    fidelity = _read_rows(root / "FOURIER_fidelity.csv")
    fourier_probe = _read_rows(root / "FOURIER_probe.csv")
    probe = _read_rows(root / "PROBE_summary.csv")
    per_class = _read_rows(root / "PROBE_per_class.csv")
    lines = ["# E/G Structure Representation Chain Audit", "", "All statements below are descriptive associations under fixed probes; they are not causal claims.", ""]
    mismatch_rows = _read_rows(root / "PROTOCOL_mismatches.csv")
    blocking = [row for row in mismatch_rows if row.get("severity") == "blocking"]
    lines += ["## Protocol", "", f"- comparison rows: {len(protocol)}", f"- blocking mismatch rows: {len(blocking)}", f"- informational mismatch rows: {len(mismatch_rows) - len(blocking)}", "- Evidence: `PROTOCOL_comparison.csv`, `PROTOCOL_mismatches.csv`.", ""]
    for task in args.tasks:
        relevant = [row for row in protocol if row.get("task") == task and row.get("checkpoint") in ("E", "historical_E_recheck", "historical_E_log")]
        if relevant:
            description = "; ".join(
                f"{row['checkpoint']}={float(row['macro_f1']):.4f} ({row.get('artifact', '')})"
                for row in relevant if row.get("macro_f1") not in (None, "")
            )
            lines.append(f"- {task}: {description}.")
    lines += ["## Fourier fidelity and discrimination", ""]
    for row in fidelity:
        if _canonical(row.get("class_name", "")) in FOCUS_CLASSES:
            lines.append(f"- {row['task']} {row['checkpoint']} {row['domain_split']} {row['class_name']}: NMSE={float(row['normalized_mse']):.4f}, R2={float(row['r2']):.4f}, corr={float(row['channel_time_correlation']):.4f}.")
    for row in fourier_probe:
        lines.append(f"- {row['task']} {row['checkpoint']} {row['representation']}: S→T fixed Macro-F1={float(row['source_to_target_fixed_macro_f1']):.4f}.")
    for task in args.tasks:
        for checkpoint in CHECKPOINTS:
            pair = {row["representation"]: row for row in fourier_probe if row.get("task") == task and row.get("checkpoint") == checkpoint}
            if {"pse_original", "pse_reconstructed"} <= pair.keys():
                delta = float(pair["pse_reconstructed"]["source_to_target_fixed_macro_f1"]) - float(pair["pse_original"]["source_to_target_fixed_macro_f1"])
                lines.append(f"- {task} {checkpoint}: reconstruction S→T Macro-F1 delta={delta:+.4f}; this is reported separately from reconstruction error.")
    lines += ["- Evidence: `FOURIER_fidelity.csv`, `FOURIER_probe.csv`.", "", "## Representation chain", ""]
    for row in probe:
        lines.append(f"- {row['task']} {row['checkpoint']} {row['representation']}: source={float(row['source_fixed_macro_f1']):.4f}, S→T={float(row['source_to_target_fixed_macro_f1']):.4f}, target-oracle={float(row['target_oracle_evaluable_macro_f1']):.4f}.")
    chain = ("shape_token", "anchor_similarity", "anchor_distribution", "shape_response", "shape_query")
    failure_votes = []
    for task in args.tasks:
        for checkpoint in CHECKPOINTS:
            values = {row["representation"]: float(row["source_to_target_fixed_macro_f1"]) for row in probe if row.get("task") == task and row.get("checkpoint") == checkpoint}
            if all(stage in values for stage in chain):
                drops = [(chain[index], chain[index + 1], values[chain[index + 1]] - values[chain[index]]) for index in range(len(chain) - 1)]
                before, after, delta = min(drops, key=lambda item: item[2])
                failure_votes.append(after)
                lines.append(f"- {task} {checkpoint}: largest adjacent fixed-probe drop is {before}→{after} ({delta:+.4f}).")
    lines += ["- Focus-class evidence is in `PROBE_per_class.csv`; absolute-vs-relative anchor evidence is in `PROBE_match_reliability.csv`.", "", "## Next minimal factor", ""]
    if failure_votes:
        selected = Counter(failure_votes).most_common(1)[0][0]
        lines.append(f"The most frequent largest-drop destination is `{selected}`. This is the audit-selected layer for a next single-factor experiment, conditional on manually confirming the same direction in both tasks and the focus-class rows. It is not an information-theoretic or causal conclusion.")
    else:
        lines.append("Undetermined because complete comparable probe rows were unavailable.")
    lines.append("")
    (root / "AUDIT_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("protocol", "fourier", "probe", "all"), default="all")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--source-root", default="outputs/structure_phase_moment_4tasks_seed1/source")
    parser.add_argument("--e-root", default="outputs/structure_phase_equivariance_4tasks_seed1/uda")
    parser.add_argument("--cross-root", default="experiments/structure_e_cross_seed1/outputs/uda")
    parser.add_argument("--cross-experiment-root", default="experiments/structure_e_cross_seed1")
    parser.add_argument("--historical-e-root", default="outputs/recheck_e_final_seed1/uda")
    parser.add_argument("--historical-log-root", default="logs/recheck_e_final_seed1")
    parser.add_argument("--output-root", default="experiments/structure_e_cross_seed1/audit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def run(args):
    gate = run_protocol(args)
    if args.mode != "protocol":
        require_protocol_gate(gate)
    if args.mode in ("fourier", "all"): run_fourier(args)
    if args.mode in ("probe", "all"): run_probe(args)
    write_report(args)
    print(f"E_G_REPRESENTATION_AUDIT_FINISHED|mode={args.mode}|output={args.output_root}")


if __name__ == "__main__":
    run(build_parser().parse_args())
