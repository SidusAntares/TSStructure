#!/usr/bin/env python3
"""Read-only efficiency and representation diagnostics for Shapelet-TimeMatch."""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_efficiency_metrics import (
    adjacent_cosine, candidate_effective_number, effective_rank, macro_f1,
    nearest_source_coverage, quantile_summary, within_class_dispersion,
)

DEFAULT_CHECKPOINT_ROOT = Path("outputs/structure_proto_kmeans_init_4tasks_seed1")
DEFAULT_OUTPUT_ROOT = Path("outputs/structure_efficiency_diagnostic")
DEFAULT_LOG_ROOT = Path("logs/structure_proto_kmeans_init_4tasks_seed1")
DOMAINS = {
    "AT1": "austria/33UVP/2017", "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017", "FR2": "france/31TCJ/2017",
}
TASKS = (("AT1", "DK1"), ("FR1", "FR2"), ("FR2", "DK1"), ("DK1", "AT1"))
REPRESENTATIONS = ("normalized_morphology", "first_difference", "mean", "std", "shape_token")


def expected_task_checkpoint(checkpoint_root, source, target):
    return Path(checkpoint_root) / "uda" / f"{source}_{target}_seed1" / "fold_0" / "model.pt"


def resolve_task_checkpoint(checkpoint_root, source, target):
    path = expected_task_checkpoint(checkpoint_root, source, target)
    if not path.is_file():
        print(f"MISSING|task={source}_{target}|checkpoint={path}")
        return None
    return path


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_model(path, device):
    from train import create_model
    packet = torch.load(path, map_location=device, weights_only=False)
    config = packet.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint config missing: {path}")
    config = SimpleNamespace(**config)
    if config.model != "psestructureprotoltae":
        raise ValueError(f"expected psestructureprotoltae, got {config.model}: {path}")
    model = create_model(config)
    model.load_state_dict(packet["state_dict"])
    return model.to(device).eval(), config


def _build_train_datasets(config, source, target, data_root, seed):
    from dataset import PixelSetData
    from train import create_train_val_test_folds
    from transforms import Normalize, RandomSamplePixels, ToTensor
    from torchvision.transforms import transforms
    paths = (DOMAINS[source], DOMAINS[target])
    bare = {
        name: PixelSetData(
            data_root, name, config.classes, closed_set=True,
            combine_spring_and_winter=config.combine_spring_and_winter,
        ) for name in paths
    }
    eligible = {name: value.get_parcel_indices().tolist() for name, value in bare.items()}
    random.seed(seed); np.random.seed(seed)
    split = create_train_val_test_folds(
        list(paths), 1, eligible, config.val_ratio, config.test_ratio,
    )[0]
    transform = transforms.Compose([RandomSamplePixels(config.num_pixels), Normalize(), ToTensor()])
    return {
        alias: PixelSetData(
            data_root, name, config.classes, transform=transform,
            indices=split[name]["train"], closed_set=True,
            combine_spring_and_winter=config.combine_spring_and_winter,
        ) for alias, name in zip((source, target), paths)
    }


def _fixed_indices(dataset, classes, limit, seed):
    labels, rng, result = dataset.get_labels(), np.random.default_rng(seed), []
    for label in classes:
        available = np.flatnonzero(labels == label)
        if len(available) > limit:
            available = rng.choice(available, limit, replace=False)
        result.extend(np.sort(available).tolist())
    return np.asarray(sorted(result), dtype=np.int64)


def _loader(dataset, indices, batch_size, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices.tolist()), batch_size=batch_size,
        shuffle=False, num_workers=0,
    )


def _candidate_metadata(branch, device):
    scales, starts = [], []
    for scale in branch.window_extractor.scales:
        current = list(range(0, branch.window_extractor.grid_points, branch.window_extractor.stride))
        scales.extend([scale] * len(current)); starts.extend(current)
    return torch.tensor(scales, device=device), torch.tensor(starts, device=device)


def _masks(scales, starts):
    masks = {"FULL": torch.ones_like(scales, dtype=torch.bool)}
    for scale in (8, 16, 24):
        masks[f"ONLY_Q{scale}"] = scales == scale
        masks[f"REMOVE_Q{scale}"] = scales != scale
    for stride in (4, 8, 16):
        masks[f"STRIDE_{stride}"] = starts.remainder(stride) == 0
    return masks


def _append_candidates(store, label, values):
    store[label].append(values.detach().cpu().to(torch.float16).numpy())


@torch.no_grad()
def _collect_domain(model, loader, domain, classes, max_candidates, seed=1):
    branch, device = model.structure_branch, next(model.parameters()).device
    scales, starts = _candidate_metadata(branch, device)
    masks = _masks(scales, starts)
    predictions = {name: [] for name in masks if name != "STRIDE_4"}
    labels_all, responses, full_weights, scale_mass, effective_total = [], [], [], [], []
    anchor_removed = [[] for _ in range(branch.shapelet_dictionary.anchors.shape[0])]
    representations = {name: defaultdict(list) for name in REPRESENTATIONS}
    adjacent, reconstruction, effective_scale = defaultdict(list), defaultdict(list), defaultdict(list)
    for sample in loader:
        pixels = sample["pixels"].to(device); valid = sample["valid_pixels"].to(device)
        positions = sample["positions"].to(device); extra = sample["extra"].to(device)
        labels = sample["label"].to(device)
        spatial = model.spatial_encoder(pixels, valid, extra)
        coefficients, diagnostics = branch.exposer.analyzer(spatial, positions)
        if torch.any(diagnostics["solver_info"] != 0):
            raise FloatingPointError(f"Fourier solve failed in {domain}")
        grid = branch.exposer.canonical_grid.to(spatial).unsqueeze(0).expand(len(spatial), -1)
        exposed = branch.exposer.synthesizer(coefficients, grid)
        reconstructed = branch.exposer.synthesizer(coefficients, positions.to(spatial.dtype))
        relative = (spatial - reconstructed).square().sum((1, 2)) / spatial.square().sum((1, 2)).clamp_min(1e-12)
        for label, value in zip(labels.tolist(), relative.tolist()):
            reconstruction[int(label)].append(value)
        groups, _ = branch.window_extractor(exposed)
        token_groups, component_groups = [], defaultdict(list)
        for scale, windows in zip(branch.window_extractor.scales, groups):
            components = branch.token_generator.components(windows)
            tokens = branch.token_generator(windows)
            token_groups.append(tokens)
            component_groups["normalized_morphology"].append(components["normalized"].flatten(2))
            component_groups["first_difference"].append(components["difference"].flatten(2))
            component_groups["mean"].append(components["mean"])
            component_groups["std"].append(components["std"])
            adjacent[scale].append(adjacent_cosine(tokens.cpu().numpy()))
        tokens = torch.cat(token_groups, dim=1); component_groups["shape_token"] = token_groups
        details = branch.shapelet_dictionary.compute_response(tokens, return_details=True)
        response, weights = details["response"], details["weights"]
        responses.append(response.cpu().numpy()); full_weights.append(weights.cpu().numpy())
        scale_mass.append(torch.stack([weights[:, scales == scale].sum(1) for scale in (8, 16, 24)], -1).cpu().numpy())
        effective_total.append(candidate_effective_number(weights.cpu().numpy()))
        for scale in (8, 16, 24):
            selected = weights[:, scales == scale]
            selected = selected / selected.sum(1, keepdim=True).clamp_min(1e-12)
            effective_scale[scale].append(candidate_effective_number(selected.cpu().numpy()))
        for name, mask in masks.items():
            if name == "STRIDE_4":
                continue
            current = branch.shapelet_dictionary.compute_response(tokens, candidate_mask=mask)
            query = branch.response_to_query(current)
            instance = model.temporal_encoder(spatial, positions, external_query=query)
            predictions[name].append(model.decoder(instance).argmax(1).cpu().numpy())
        for anchor in range(response.shape[1]):
            ablated = response.clone(); ablated[:, anchor] = 0
            query = branch.response_to_query(ablated)
            instance = model.temporal_encoder(spatial, positions, external_query=query)
            anchor_removed[anchor].append(model.decoder(instance).argmax(1).cpu().numpy())
        labels_all.append(labels.cpu().numpy())
        candidate_labels = labels[:, None].expand(-1, tokens.shape[1]).reshape(-1)
        for name, value in component_groups.items():
            combined = torch.cat(value, dim=1); flattened = combined.reshape(-1, combined.shape[-1])
            for label in classes:
                _append_candidates(representations[name], int(label), flattened[candidate_labels == label])
    rng = np.random.default_rng(seed)
    packed_representations = {}
    for name, groups in representations.items():
        packed_representations[name] = {}
        for label, chunks in groups.items():
            if not chunks:
                continue
            values = np.concatenate(chunks)
            if len(values) > max_candidates:
                values = values[np.sort(rng.choice(len(values), max_candidates, replace=False))]
            packed_representations[name][label] = values.astype(np.float32)
    return {
        "domain": domain, "labels": np.concatenate(labels_all),
        "predictions": {name: np.concatenate(value) for name, value in predictions.items()},
        "responses": np.concatenate(responses), "weights": np.concatenate(full_weights),
        "scale_mass": np.concatenate(scale_mass), "effective_total": np.concatenate(effective_total),
        "effective_scale": {key: np.concatenate(value) for key, value in effective_scale.items()},
        "adjacent": {key: np.concatenate(value) for key, value in adjacent.items()},
        "anchor_removed": [np.concatenate(value) for value in anchor_removed],
        "representations": packed_representations, "reconstruction": dict(reconstruction),
        "candidate_counts": {name: int(mask.sum()) for name, mask in masks.items()},
    }


def _task_rows(task, model, source_data, target_data, classes, class_names):
    scale_rows, anchor_rows, complexity_rows, fourier_rows = [], [], [], []
    class_ids, full_scores = np.asarray(classes), {}
    for data in (source_data, target_data):
        full_scores[data["domain"]] = macro_f1(data["labels"], data["predictions"]["FULL"], class_ids)
        for name, prediction in data["predictions"].items():
            score = macro_f1(data["labels"], prediction, class_ids)
            scale_rows.append({
                "task": task, "domain": data["domain"], "configuration": name,
                "macro_f1": score, "delta_vs_full": score - full_scores[data["domain"]],
                "candidate_count": data["candidate_counts"]["STRIDE_4" if name == "FULL" else name],
            })
    target_full = full_scores[target_data["domain"]]
    for anchor, prediction in enumerate(target_data["anchor_removed"]):
        score = macro_f1(target_data["labels"], prediction, class_ids)
        anchor_rows.append({"task": task, "row_type": "single_anchor_removal", "anchor": anchor, "target_f1": score, "delta_target_f1": score - target_full})
    anchors = F.normalize(model.structure_branch.shapelet_dictionary.anchors.detach(), dim=-1).cpu().numpy()
    pairwise = anchors @ anchors.T; off = pairwise[~np.eye(len(anchors), dtype=bool)]
    anchor_rows.append({"task": task, "row_type": "anchor_geometry", "cosine_mean": float(off.mean()), "cosine_max": float(off.max()), "cosine_p90": float(np.quantile(off, .9)), "effective_rank": effective_rank(anchors)})
    for data in (source_data, target_data):
        centered = data["responses"] - data["responses"].mean(0, keepdims=True)
        std = data["responses"].std(0); corr = np.nan_to_num(np.corrcoef(data["responses"], rowvar=False))
        off_corr = np.abs(corr[~np.eye(corr.shape[0], dtype=bool)])
        anchor_rows.append({"task": task, "row_type": "response_rank", "domain": data["domain"], "response_std_mean": float(std.mean()), "response_std_min": float(std.min()), "response_std_max": float(std.max()), "correlation_abs_mean": float(off_corr.mean()), "correlation_abs_max": float(off_corr.max()), "effective_rank": effective_rank(centered)})
        mass = data["scale_mass"].mean(0)
        for anchor in range(mass.shape[0]):
            probabilities = mass[anchor] / max(mass[anchor].sum(), 1e-12)
            anchor_rows.append({"task": task, "row_type": "scale_mass", "domain": data["domain"], "anchor": anchor, "q8_mass": probabilities[0], "q16_mass": probabilities[1], "q24_mass": probabilities[2], "dominant_scale": (8, 16, 24)[int(np.argmax(probabilities))], "scale_mass_entropy": float(-(probabilities * np.log(probabilities + 1e-12)).sum())})
        for scale, values in data["adjacent"].items():
            anchor_rows.append({"task": task, "row_type": "adjacent_token_cosine", "domain": data["domain"], "scale": scale, **quantile_summary(values)})
        anchor_rows.append({"task": task, "row_type": "candidate_effective_number", "domain": data["domain"], "scale": "ALL", **quantile_summary(data["effective_total"].reshape(-1))})
        for scale, values in data["effective_scale"].items():
            anchor_rows.append({"task": task, "row_type": "candidate_effective_number", "domain": data["domain"], "scale": scale, **quantile_summary(values.reshape(-1))})
    for representation in REPRESENTATIONS:
        source_bank, target_bank = source_data["representations"][representation], target_data["representations"][representation]
        for data, bank in ((source_data, source_bank), (target_data, target_bank)):
            features = np.concatenate([bank[label] for label in classes if label in bank])
            labels = np.concatenate([np.full(len(bank[label]), label) for label in classes if label in bank])
            dispersion = within_class_dispersion(features, labels)
            for label, value in dispersion.items():
                complexity_rows.append({"task": task, "metric": "within_class_dispersion", "domain": data["domain"], "class_id": label, "class_name": class_names[label], "representation": representation, "value": value})
            complexity_rows.append({"task": task, "metric": "within_class_dispersion", "domain": data["domain"], "class_id": "ALL", "class_name": "ALL", "representation": representation, "value": float(np.mean(list(dispersion.values())))})
        for label in classes:
            if label in source_bank and label in target_bank:
                coverage = nearest_source_coverage(source_bank[label], target_bank[label])
                complexity_rows.append({"task": task, "metric": "source_target_coverage", "domain": "target", "class_id": label, "class_name": class_names[label], "representation": representation, **quantile_summary(coverage)})
    anchor_coverage = defaultdict(dict)
    for data in (source_data, target_data):
        for label, tokens in data["representations"]["shape_token"].items():
            normalized = tokens / np.maximum(np.linalg.norm(tokens, axis=1, keepdims=True), 1e-12)
            coverage = (normalized @ anchors.T).max(1)
            summary = quantile_summary(coverage)
            anchor_coverage[label][data["domain"]] = summary
            complexity_rows.append({"task": task, "metric": "anchor_coverage", "domain": data["domain"], "class_id": label, "class_name": class_names[label], "representation": "shape_token", **summary})
    for label, values in anchor_coverage.items():
        if source_data["domain"] in values and target_data["domain"] in values:
            complexity_rows.append({"task": task, "metric": "anchor_coverage_gap", "domain": "target_minus_source", "class_id": label, "class_name": class_names[label], "representation": "shape_token", "value": values[target_data["domain"]]["mean"] - values[source_data["domain"]]["mean"]})
    for data in (source_data, target_data):
        all_values = []
        for label, values in data["reconstruction"].items():
            all_values.extend(values); fourier_rows.append({"task": task, "domain": data["domain"], "class_id": label, "class_name": class_names[label], **quantile_summary(values)})
        fourier_rows.append({"task": task, "domain": data["domain"], "class_id": "ALL", "class_name": "ALL", **quantile_summary(all_values)})
    summary = [{"task": task, "source_macro_f1_full": full_scores[source_data["domain"]], "target_macro_f1_full": target_full, "num_source_samples": len(source_data["labels"]), "num_target_samples": len(target_data["labels"])}]
    return summary, scale_rows, anchor_rows, complexity_rows, fourier_rows


def _signature_rows(task, domain, dataset, seed, seq_length=30):
    raw = tuple(int(value) for value in dataset.date_positions); raw_counts = Counter([raw] * len(dataset))
    rng, sampled = random.Random(seed), []
    for _ in range(len(dataset)):
        selected = range(len(raw)) if len(raw) <= seq_length else sorted(rng.sample(range(len(raw)), seq_length))
        sampled.append(tuple(raw[index] for index in selected))
    rows = []
    for kind, counter in (("raw", raw_counts), ("sampled", Counter(sampled))):
        samples = sum(counter.values())
        rows.append({"task": task, "domain": domain, "signature_type": kind, "num_samples": samples, "num_unique_signatures": len(counter), "reuse_ratio": 1 - len(counter) / max(samples, 1), "max_signature_frequency": max(counter.values(), default=0)})
    return rows


def _source_convergence(source, log_path):
    if not log_path.is_file():
        return {"source": source, "log": str(log_path), "status": "MISSING"}
    text = log_path.read_text(errors="replace").split("UDA ", 1)[0]
    values = [float(value) for value in re.findall(r"Validation result:.*?f1=([0-9.]+)", text)]
    if not values:
        return {"source": source, "log": str(log_path), "status": "NO_VALIDATION_F1"}
    best = max(values)
    first = lambda delta: next((index + 1 for index, value in enumerate(values) if value >= best - delta), "NA")
    at = lambda epoch: values[epoch - 1] if len(values) >= epoch else "NA"
    return {"source": source, "log": str(log_path), "status": "OK", "best_epoch": int(np.argmax(values)) + 1, "best_f1": best, "last_f1": values[-1], "epoch_within_0.5": first(.005), "epoch_within_1.0": first(.01), "f1_epoch30": at(30), "f1_epoch40": at(40), "f1_epoch50": at(50)}


def _report(path, completed, missing):
    sections = ("Runtime", "Scale Usage", "Stride Simulation", "Window Redundancy", "Anchor Geometry", "Response Rank", "Anchor Ablation", "Domain Complexity", "Cross-domain Structure Coverage", "Fourier Reconstruction", "Timestamp Reuse", "Source Convergence")
    lines = ["# Structure Efficiency Diagnostic", "", f"Completed tasks: {', '.join(completed) or 'none'}", f"Missing tasks: {', '.join(missing) or 'none'}", ""]
    for section in sections:
        lines.extend((f"## {section}", "See the corresponding CSV rows; no architecture decision is made here.", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    all_rows = {name: [] for name in ("summary", "scale", "anchor", "complexity", "fourier", "timestamp")}
    completed, missing = [], []
    for source, target in TASKS:
        task = f"{source}_{target}"; checkpoint = resolve_task_checkpoint(args.checkpoint_root, source, target)
        if checkpoint is None:
            missing.append(task); continue
        model, config = _load_model(checkpoint, args.device)
        datasets = _build_train_datasets(config, source, target, args.data_root, args.seed)
        common = sorted(set(datasets[source].get_labels()) & set(datasets[target].get_labels()))
        source_indices = _fixed_indices(datasets[source], common, args.samples_per_class, args.seed)
        target_indices = _fixed_indices(datasets[target], common, args.samples_per_class, args.seed)
        source_data = _collect_domain(model, _loader(datasets[source], source_indices, args.batch_size, args.seed), source, common, args.max_candidates_per_class, args.seed)
        target_data = _collect_domain(model, _loader(datasets[target], target_indices, args.batch_size, args.seed), target, common, args.max_candidates_per_class, args.seed)
        rows = _task_rows(task, model, source_data, target_data, common, config.classes)
        for key, values in zip(("summary", "scale", "anchor", "complexity", "fourier"), rows): all_rows[key].extend(values)
        all_rows["timestamp"].extend(_signature_rows(task, source, datasets[source], args.seed)); all_rows["timestamp"].extend(_signature_rows(task, target, datasets[target], args.seed))
        completed.append(task)
        print(f"DIAGNOSTIC_FINISHED|task={task}|checkpoint={checkpoint}")
    convergence = [
        _source_convergence(source, Path(args.log_root) / f"{source}_{target}.log")
        for source, target in TASKS
    ]
    _write_csv(output / "summary.csv", all_rows["summary"], ["task", "source_macro_f1_full", "target_macro_f1_full", "num_source_samples", "num_target_samples"])
    _write_csv(output / "scale_ablation.csv", all_rows["scale"], ["task", "domain", "configuration", "macro_f1", "delta_vs_full", "candidate_count"])
    anchor_fields = sorted(set().union(*(row.keys() for row in all_rows["anchor"]))) if all_rows["anchor"] else ["task", "row_type"]
    _write_csv(output / "anchor_diagnostics.csv", all_rows["anchor"], anchor_fields)
    complexity_fields = sorted(set().union(*(row.keys() for row in all_rows["complexity"]))) if all_rows["complexity"] else ["task", "metric"]
    _write_csv(output / "complexity_coverage.csv", all_rows["complexity"], complexity_fields)
    _write_csv(output / "fourier_reconstruction.csv", all_rows["fourier"], ["task", "domain", "class_id", "class_name", "mean", "median", "p10", "p50", "p90"])
    _write_csv(output / "timestamp_reuse.csv", all_rows["timestamp"], ["task", "domain", "signature_type", "num_samples", "num_unique_signatures", "reuse_ratio", "max_signature_frequency"])
    _write_csv(output / "source_convergence.csv", convergence, ["source", "log", "status", "best_epoch", "best_f1", "last_f1", "epoch_within_0.5", "epoch_within_1.0", "f1_epoch30", "f1_epoch40", "f1_epoch50"])
    _report(output / "report.md", completed, missing)
    print(f"STRUCTURE_EFFICIENCY_DONE|completed={len(completed)}|missing={len(missing)}|output={output}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--data-root", type=Path, default=Path("/data/user/dataset/timematch_data"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--samples-per-class", type=int, default=64)
    parser.add_argument("--max-candidates-per-class", type=int, default=3000)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
