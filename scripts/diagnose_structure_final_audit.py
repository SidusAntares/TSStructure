#!/usr/bin/env python3
"""Unified read-only final audit for the Shapelet structure classifier."""

from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_efficiency_metrics import (
    candidate_effective_number, effective_rank, macro_f1, within_class_dispersion,
)
from analysis.structure_final_audit import (
    collect_structure_layers, compare_forward_paths, layer_statistics,
    parameter_statistics, response_from_similarity, response_intervention_outputs,
    transfer_margin,
)
from scripts.diagnose_structure_efficiency import (
    DEFAULT_CHECKPOINT_ROOT, DOMAINS, TASKS, _build_train_datasets,
    _fixed_indices, _loader,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/structure_final_audit")
CSV_FILES = {
    "forward": "forward_equivalence.csv",
    "collapse": "collapse_layers.csv",
    "parameter": "parameter_audit.csv",
    "specific": "sample_specific_ablation.csv",
    "beta": "beta_sweep.csv",
    "transfer": "transferability.csv",
    "complexity": "domain_complexity.csv",
}
REPRESENTATIONS = (
    "raw_input", "pse", "fourier_canonical", "normalized_morphology",
    "difference_morphology", "shape_token",
)


def expected_checkpoints(root, source, target):
    root = Path(root)
    return {
        "source": root / "source" / f"source_{source}_seed1" / "fold_0" / "model.pt",
        "uda": root / "uda" / f"{source}_{target}_seed1" / "fold_0" / "model.pt",
    }


def resolve_checkpoints(root, source, target):
    result = {}
    for stage, path in expected_checkpoints(root, source, target).items():
        if not path.is_file():
            print(f"MISSING|task={source}_{target}|stage={stage}|checkpoint={path}")
            result[stage] = None
        else:
            result[stage] = path
    return result


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _strict_load(path, device):
    from train import create_model
    packet = torch.load(path, map_location=device, weights_only=False)
    raw_config = packet.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError(f"checkpoint config missing: {path}")
    config = SimpleNamespace(**raw_config)
    model = create_model(config)
    expected, actual = set(model.state_dict()), set(packet["state_dict"])
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    print(
        f"CHECKPOINT_KEYS|checkpoint={path}|missing_keys={missing}|"
        f"unexpected_keys={unexpected}"
    )
    model.load_state_dict(packet["state_dict"], strict=True)
    return model.to(device).eval(), config, packet, missing, unexpected


def _move(sample, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in sample.items()}


def _first(loader, device):
    return _move(next(iter(loader)), device)


def _parameter_groups(model):
    branch = model.structure_branch
    generator = branch.token_generator
    return {
        "raw_encoder": generator.raw_encoder.parameters(),
        "diff_encoder": generator.diff_encoder.parameters(),
        "mean_encoder": generator.mean_encoder.parameters(),
        "std_encoder": generator.std_encoder.parameters(),
        "fusion_linear": generator.fusion[0].parameters(),
        "fusion_layernorm": generator.fusion[1].parameters(),
        "layernorm_weight": (generator.fusion[1].weight,),
        "shapelet_anchors": (branch.shapelet_dictionary.anchors,),
        "response_to_query": branch.response_to_query.parameters(),
        "external_query_projection": (
            model.temporal_encoder.attention_heads.external_query_projection.weight,
        ),
    }


def _prediction_metrics(logits, labels):
    prediction = logits.argmax(-1).detach().cpu().numpy()
    truth = labels.detach().cpu().numpy()
    return macro_f1(truth, prediction), float(np.mean(prediction == truth)), prediction


def _candidate_labels(labels, values):
    count = int(np.prod(values.shape[1:-1])) if values.ndim > 2 else 1
    return labels[:, None].expand(-1, count).reshape(-1)


@torch.no_grad()
def _collect(model, loader, device, temporal_shift=0, max_per_class=3000, seed=1):
    stores = defaultdict(list); labels_store = defaultdict(list)
    logits_all, labels_all = [], []
    layer_store = defaultdict(list)
    rng = np.random.default_rng(seed)
    for sample in loader:
        sample = _move(sample, device)
        spatial = model.spatial_encoder(sample["pixels"], sample["valid_pixels"], sample["extra"])
        layers = collect_structure_layers(
            model, spatial, sample["positions"], temporal_shift,
        )
        query = layers["response_to_query"]
        instance = model.temporal_encoder(
            spatial, sample["positions"] + temporal_shift, external_query=query,
        )
        logits = model.decoder(instance)
        logits_all.append(logits.cpu()); labels_all.append(sample["label"].cpu())
        raw = sample["pixels"].mean(-1).flatten(1)
        representations = {
            "raw_input": raw,
            "pse": spatial,
            "fourier_canonical": layers["fourier_canonical"],
            "normalized_morphology": layers["normalized_morphology"].flatten(2),
            "difference_morphology": layers["difference_morphology"].flatten(2),
            "shape_token": layers["shape_token"],
        }
        for name, value in representations.items():
            flattened = value.reshape(-1, value.shape[-1])
            candidate_labels = _candidate_labels(sample["label"], value)
            stores[name].append(flattened.cpu()); labels_store[name].append(candidate_labels.cpu())
        for name, value in layers.items():
            if name != "fourier_canonical":
                layer_store[name].append(value.cpu())
    packed = {}
    for name in REPRESENTATIONS:
        values = torch.cat(stores[name]); labels = torch.cat(labels_store[name])
        selected_values, selected_labels = [], []
        for label in labels.unique().tolist():
            indices = torch.where(labels == label)[0].numpy()
            if len(indices) > max_per_class:
                indices = np.sort(rng.choice(indices, max_per_class, replace=False))
            selected_values.append(values[indices]); selected_labels.append(labels[indices])
        packed[name] = (torch.cat(selected_values), torch.cat(selected_labels))
    return {
        "logits": torch.cat(logits_all), "labels": torch.cat(labels_all),
        "representations": packed,
        "layers": {name: torch.cat(values) for name, values in layer_store.items()},
    }


def _parameter_rows(task, stage, model, checkpoint, missing, unexpected):
    rows = []
    for name, parameters in _parameter_groups(model).items():
        rows.append({
            "task": task, "stage": stage, "group": name, "checkpoint": str(checkpoint),
            "missing_keys": ";".join(missing), "unexpected_keys": ";".join(unexpected),
            **parameter_statistics(parameters),
        })
    return rows


def _collapse_rows(task, stage, domain, collected):
    rows = []
    for layer, values in collected["layers"].items():
        statistics = layer_statistics(values)
        conservative = (
            statistics["sample_variance"] <= 1e-8
            or statistics["feature_norm_mean"] <= 1e-8
            or statistics["effective_rank"] <= 1.05
        )
        rows.append({
            "task": task, "stage": stage, "domain": domain, "layer": layer,
            **statistics, "conservative_collapse_flag": conservative,
        })
    return rows


@torch.no_grad()
def _specific_rows(task, model, loader, device, shift, seed):
    cached, labels, responses = [], [], []
    for sample in loader:
        sample = _move(sample, device)
        spatial = model.spatial_encoder(sample["pixels"], sample["valid_pixels"], sample["extra"])
        structure = model.structure_branch(spatial, sample["positions"])
        cached.append((spatial, sample["positions"], structure["shapelet_response"]))
        responses.append(structure["shapelet_response"])
        labels.append(sample["label"].cpu())
    all_response = torch.cat(responses)
    mean_response = all_response.mean(0, keepdim=True)
    generator = torch.Generator(device=all_response.device).manual_seed(int(seed))
    permutation = torch.randperm(len(all_response), generator=generator, device=all_response.device)
    shuffled = all_response[permutation]
    logits = defaultdict(list); offset = 0
    for spatial, positions, response in cached:
        size = len(response)
        variants = {
            "FULL": response,
            "ZERO": None,
            "MEAN": mean_response.expand(size, -1),
            "SHUFFLE": shuffled[offset:offset + size],
        }
        offset += size
        for name, current in variants.items():
            query = None if current is None else model.structure_branch.response_to_query(current)
            instance = model.temporal_encoder(
                spatial, positions + shift, external_query=query,
            )
            logits[name].append(model.decoder(instance).cpu())
    truth = torch.cat(labels)
    rows = []
    for name in ("FULL", "ZERO", "MEAN", "SHUFFLE"):
        score, accuracy, prediction = _prediction_metrics(torch.cat(logits[name]), truth)
        if name == "FULL": full_predictions = prediction
        rows.append({
            "task": task, "intervention": name, "target_macro_f1": score,
            "accuracy": accuracy,
            "prediction_disagreement_vs_full": float(np.mean(prediction != full_predictions)) if name != "FULL" else 0.,
        })
    return rows


@torch.no_grad()
def _beta_rows(task, model, loader, device, shift):
    storage = {name: {"logits": [], "weights": [], "responses": []} for name in ("5", "10", "20", "50", "hard")}
    labels = []
    for sample in loader:
        sample = _move(sample, device)
        spatial = model.spatial_encoder(sample["pixels"], sample["valid_pixels"], sample["extra"])
        structure = model.structure_branch(spatial, sample["positions"])
        similarity = model.structure_branch.shapelet_dictionary.compute_similarity(structure["shape_tokens"])
        for name in storage:
            response, weights = response_from_similarity(
                similarity, beta=float(name) if name != "hard" else 5., hard_max=name == "hard",
            )
            query = model.structure_branch.response_to_query(response)
            instance = model.temporal_encoder(
                spatial, sample["positions"] + shift, external_query=query,
            )
            storage[name]["logits"].append(model.decoder(instance).cpu())
            storage[name]["weights"].append(weights.cpu())
            storage[name]["responses"].append(response.cpu())
        labels.append(sample["label"].cpu())
    truth = torch.cat(labels); rows = []
    for name, values in storage.items():
        response = torch.cat(values["responses"]); weights = torch.cat(values["weights"])
        score, _, _ = _prediction_metrics(torch.cat(values["logits"]), truth)
        rows.append({
            "task": task, "beta": name, "target_macro_f1": score,
            "candidate_effective_number": float(candidate_effective_number(weights.numpy()).mean()),
            "response_effective_rank": effective_rank(response.numpy()),
            "response_std": float(response.std(unbiased=False)),
        })
    return rows


def _transfer_rows(task, source, target, source_data, target_data):
    rows = []
    for representation in REPRESENTATIONS[1:]:
        source_values, source_labels = source_data["representations"][representation]
        target_values, target_labels = target_data["representations"][representation]
        for label, result in transfer_margin(
            source_values, source_labels, target_values, target_labels,
        ).items():
            rows.append({
                "task": task, "domain": f"{source}_to_{target}", "class_id": label,
                "representation": representation, **result,
            })
    return rows


def _complexity_rows(task, stage, domain, collected):
    rows = []
    for representation in REPRESENTATIONS:
        values, labels = collected["representations"][representation]
        for label, value in within_class_dispersion(values.numpy(), labels.numpy()).items():
            rows.append({
                "task": task, "stage": stage, "domain": domain,
                "class_id": label, "representation": representation,
                "metric": "within_class_dispersion", "value": value,
                "metric_provenance": "analysis.structure_efficiency_metrics.within_class_dispersion",
            })
    return rows


def _report(path, rows, blockers):
    order = (
        "Forward Equivalence", "Collapse Stage", "Parameter Audit",
        "Sample-specific Qshape Test", "Beta / Candidate Selectivity",
        "Structure Transferability", "Domain Complexity Propagation",
        "Runtime Breakdown", "Existing Window Diagnostics Validity",
    )
    lines = ["# Structure Final Audit", ""]
    for index, title in enumerate(order, 1):
        lines.extend((f"## {index}. {title}", ""))
        if title == "Forward Equivalence":
            lines.append(f"Rows: {len(rows['forward'])}; BLOCKER count: {len(blockers)}.")
        elif title == "Existing Window Diagnostics Validity":
            tasks = sorted({row["task"] for row in rows["forward"]})
            valid = []
            for task in tasks:
                equivalent = all(
                    row["verified"] for row in rows["forward"] if row["task"] == task
                )
                collapsed = any(
                    row["task"] == task and row["stage"] == "uda"
                    and row["layer"] == "shape_token"
                    and row["conservative_collapse_flag"]
                    for row in rows["collapse"]
                )
                valid.append(f"{task}: valid_for_architecture_decision={str(equivalent and not collapsed).lower()}")
            lines.extend(valid or ["No completed task."])
        else:
            key = ("specific" if title.startswith("Sample") else "beta" if title.startswith("Beta") else
                   "transfer" if title.startswith("Structure") else "complexity" if title.startswith("Domain") else
                   "parameter" if title.startswith("Parameter") else "collapse" if title.startswith("Collapse") else None)
            lines.append(f"Rows: {len(rows.get(key, []))}." if key else "See profile_source.csv and profile_shift.csv.")
        lines.append("")
    lines.extend(("## Blockers", "", *(f"- {item}" for item in blockers or ["None"])))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = {key: [] for key in CSV_FILES}; blockers = []; completed = []
    for source, target in TASKS:
        task = f"{source}_{target}"; resolved = resolve_checkpoints(args.checkpoint_root, source, target)
        if any(value is None for value in resolved.values()):
            blockers.append(f"{task}: source or UDA checkpoint missing")
            continue
        loaded, collected_by_stage = {}, defaultdict(dict)
        try:
            for stage, path in resolved.items(): loaded[stage] = _strict_load(path, args.device)
        except Exception as error:
            blockers.append(f"{task}: strict checkpoint load failed: {error}"); print(f"BLOCKER|task={task}|reason={error}"); continue
        uda_model, config, uda_packet, _, _ = loaded["uda"]
        datasets = _build_train_datasets(config, source, target, args.data_root, args.seed)
        common = sorted(set(datasets[source].get_labels()) & set(datasets[target].get_labels()))
        loaders = {}
        for domain in (source, target):
            indices = _fixed_indices(datasets[domain], common, args.samples_per_class, args.seed)
            loaders[domain] = _loader(datasets[domain], indices, args.batch_size, args.seed)
        shift = int(uda_packet.get("global_temporal_shift", 0))
        batch = _first(loaders[target], args.device)
        equivalence = compare_forward_paths(uda_model, batch, shift)
        for comparison, values in equivalence.items():
            row = {"task": task, "stage": "uda", "comparison": comparison, **{f"{key}_max_abs_diff": value for key, value in values.items()}}
            row["verified"] = max(values.values()) <= args.equivalence_tolerance
            rows["forward"].append(row)
            if not row["verified"]: blockers.append(f"{task}: {comparison} failed")
        for stage, (model, _, packet, missing, unexpected) in loaded.items():
            rows["parameter"].extend(_parameter_rows(task, stage, model, resolved[stage], missing, unexpected))
            for domain in (source, target):
                domain_shift = shift if stage == "uda" and domain == target else 0
                collected = _collect(model, loaders[domain], args.device, domain_shift, args.max_candidates_per_class, args.seed)
                rows["collapse"].extend(_collapse_rows(task, stage, domain, collected))
                rows["complexity"].extend(_complexity_rows(task, stage, domain, collected))
                collected_by_stage[stage][domain] = collected
        if all(row["verified"] for row in rows["forward"] if row["task"] == task):
            rows["specific"].extend(_specific_rows(task, uda_model, loaders[target], args.device, shift, args.seed))
            rows["beta"].extend(_beta_rows(task, uda_model, loaders[target], args.device, shift))
            source_data = collected_by_stage["uda"][source]
            target_data = collected_by_stage["uda"][target]
            rows["transfer"].extend(_transfer_rows(task, source, target, source_data, target_data))
        else:
            print(f"BLOCKER|task={task}|reason=forward_equivalence")
        completed.append(task)
    fields = {
        "forward": ["task", "stage", "comparison", "logits_max_abs_diff", "instance_feature_max_abs_diff", "shapelet_response_max_abs_diff", "shape_class_token_max_abs_diff", "verified"],
        "collapse": ["task", "stage", "domain", "layer", "sample_variance", "candidate_variance", "feature_norm_mean", "feature_norm_std", "effective_rank", "conservative_collapse_flag"],
        "parameter": ["task", "stage", "group", "checkpoint", "parameter_norm", "mean", "std", "min", "max", "sha256", "missing_keys", "unexpected_keys"],
        "specific": ["task", "intervention", "target_macro_f1", "accuracy", "prediction_disagreement_vs_full"],
        "beta": ["task", "beta", "target_macro_f1", "candidate_effective_number", "response_effective_rank", "response_std"],
        "transfer": ["task", "domain", "class_id", "representation", "same_class_coverage", "wrong_class_coverage", "transfer_margin"],
        "complexity": ["task", "stage", "domain", "class_id", "representation", "metric", "value", "metric_provenance"],
    }
    for key, filename in CSV_FILES.items(): _write_csv(output / filename, rows[key], fields[key])
    if not args.skip_profiler:
        for task in completed:
            for mode in ("source", "shift"):
                command = [
                    sys.executable, "-u", str(ROOT / "scripts" / "profile_structure_runtime.py"),
                    "--mode", mode, "--task", task,
                    "--checkpoint-root", str(args.checkpoint_root),
                    "--data-root", str(args.data_root), "--output-dir", str(output),
                    "--device", args.device, "--seed", str(args.seed),
                    "--batch-size", str(args.batch_size), "--warmup", "5", "--steps", "20",
                ]
                result = subprocess.run(command, check=False)
                if result.returncode:
                    message = f"{task}: {mode} profiler failed with exit {result.returncode}"
                    blockers.append(message); print(f"BLOCKER|reason={message}")
    for filename, header in (("profile_source.csv", "task,stage,component,mean_ms,peak_gpu_memory_bytes"), ("profile_shift.csv", "task,stage,component,mean_ms,num_shifts,structure_branch_call_count,peak_gpu_memory_bytes")):
        path = output / filename
        if not path.exists(): path.write_text(header + "\n", encoding="utf-8")
    _report(output / "report.md", rows, blockers)
    print(f"STRUCTURE_FINAL_AUDIT_DONE|output={output}|blockers={len(blockers)}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--data-root", type=Path, default=Path("/data/user/dataset/timematch_data"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--samples-per-class", type=int, default=128)
    parser.add_argument("--max-candidates-per-class", type=int, default=3000)
    parser.add_argument("--equivalence-tolerance", type=float, default=1e-6)
    parser.add_argument("--skip-profiler", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
