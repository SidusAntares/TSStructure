#!/usr/bin/env python3
"""Organization semantics and anchor-capacity audits for state-org foundations."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.state_org_anchor_basis_audit import organization_statistics
from analysis.state_org_feasibility_audit import (
    _datasets, _load_checkpoint, _loader, _macro, _move, _per_class, _probe,
)


def semantic_features(presence, state_distribution):
    """Return encoder-independent P/C/T1/T2 features from a presence basis."""
    statistics = organization_statistics(state_distribution)
    return {
        "P": presence,
        "P_C": torch.cat((presence, statistics["composition"]), dim=1),
        "P_T1": torch.cat((presence, statistics["t1"].flatten(1)), dim=1),
        "P_T1_T2": torch.cat((
            presence, statistics["t1"].flatten(1), statistics["t2"].flatten(1),
        ), dim=1),
    }


def _write(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def _collect(model, dataset, device, batch_size, token_limit=50000, seed=1):
    storage = defaultdict(list); reservoir = None
    rng = np.random.default_rng(int(seed))
    for raw in _loader(dataset, batch_size):
        batch = _move(raw, device)
        spatial = model.spatial_encoder(
            batch["pixels"], batch["valid_pixels"], batch["extra"],
        )
        structure = model.prepare_structure(spatial, batch["positions"], 0)
        evidence = model.shape_response_norm(structure["shapelet_response"])
        output = model.forward_with_external_shape_evidence(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch["extra"], evidence,
        )
        features = semantic_features(
            structure["shapelet_presence"], structure["state_distribution"],
        )
        storage["label"].append(batch["label"].cpu())
        storage["response"].append(evidence.cpu())
        storage["prediction"].append(output["logits"].argmax(1).cpu())
        for name, value in features.items():
            storage[name].append(value.cpu())
        tokens = structure["shape_tokens"].flatten(0, 1).cpu()
        reservoir = tokens if reservoir is None else torch.cat((reservoir, tokens))
        if reservoir.shape[0] > token_limit:
            keep = rng.choice(reservoir.shape[0], token_limit, replace=False)
            reservoir = reservoir[torch.from_numpy(keep)]
    result = {name: torch.cat(values).numpy() for name, values in storage.items()}
    result["tokens"] = reservoir.numpy()
    return result


def _normalized_kmeans(features, count, seed):
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    centers = KMeans(
        n_clusters=count, random_state=int(seed), n_init=10,
    ).fit(normalized).cluster_centers_
    return centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)


def _coverage(features, centers):
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    best = (normalized @ centers.T).max(1)
    return {
        "coverage_mean": float(best.mean()),
        "coverage_median": float(np.median(best)),
        "coverage_p10": float(np.quantile(best, .1)),
        "coverage_p25": float(np.quantile(best, .25)),
    }


def run(args):
    specs = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        specs[name] = Path(path)
    if set(specs) != {"full", "composition", "presence"}:
        raise ValueError("checkpoints must define full, composition, and presence")
    device = torch.device(args.device)
    models = {}
    config = None
    for name, path in specs.items():
        model, current, _ = _load_checkpoint(path, device)
        if getattr(current, "state_org_readout", "full") != name:
            raise ValueError(f"checkpoint readout mismatch: {name} {path}")
        models[name] = model
        config = current if config is None else config
    datasets = _datasets(config, args.source, args.target, args.data_root, args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    summary_rows, per_class_rows = [], []
    collected = {}
    for name, model in models.items():
        values = {
            split: _collect(model, datasets[split], device, args.batch_size, seed=args.seed)
            for split in ("source_train", "source_test", "target_val", "target_test")
        }
        collected[name] = values
        source_f1 = _macro(
            values["source_test"]["label"], values["source_test"]["prediction"],
            len(config.classes),
        )
        target_f1 = _macro(
            values["target_test"]["label"], values["target_test"]["prediction"],
            len(config.classes),
        )
        oracle, oracle_pc = _probe(
            values["target_val"]["response"], values["target_val"]["label"],
            values["target_test"]["response"], values["target_test"]["label"],
            len(config.classes), args.seed,
        )
        target_pc = _per_class(
            values["target_test"]["label"], values["target_test"]["prediction"],
            len(config.classes),
        )
        summary_rows.append({
            "readout": name, "source_test_macro_f1": source_f1,
            "target_zero_shot_macro_f1": target_f1,
            "target_oracle_probe_macro_f1": oracle,
        })
        for class_id, class_name in enumerate(config.classes):
            per_class_rows.append({
                "readout": name, "class_id": class_id, "class_name": class_name,
                "target_zero_shot_f1": target_pc[class_id],
                "target_oracle_probe_f1": oracle_pc[class_id],
            })
    _write(output / "source_readout_summary.csv", summary_rows)
    _write(output / "source_readout_per_class.csv", per_class_rows)

    presence = collected["presence"]
    semantic_rows, semantic_pc_rows = [], []
    for feature in ("P", "P_C", "P_T1", "P_T1_T2"):
        transfer, transfer_pc = _probe(
            presence["source_train"][feature], presence["source_train"]["label"],
            presence["target_test"][feature], presence["target_test"]["label"],
            len(config.classes), args.seed,
        )
        oracle, oracle_pc = _probe(
            presence["target_val"][feature], presence["target_val"]["label"],
            presence["target_test"][feature], presence["target_test"]["label"],
            len(config.classes), args.seed,
        )
        semantic_rows.append({
            "feature": feature, "source_target_macro_f1": transfer,
            "target_oracle_macro_f1": oracle,
        })
        for class_id, class_name in enumerate(config.classes):
            semantic_pc_rows.append({
                "feature": feature, "class_id": class_id, "class_name": class_name,
                "source_target_f1": transfer_pc[class_id],
                "target_oracle_f1": oracle_pc[class_id],
            })
    baseline = semantic_rows[0]
    for row in semantic_rows:
        row["source_target_delta_vs_P"] = (
            row["source_target_macro_f1"] - baseline["source_target_macro_f1"]
        )
        row["target_oracle_delta_vs_P"] = (
            row["target_oracle_macro_f1"] - baseline["target_oracle_macro_f1"]
        )
    _write(output / "organization_semantic_probe.csv", semantic_rows)
    _write(output / "organization_semantic_per_class.csv", semantic_pc_rows)

    source_tokens = presence["source_train"]["tokens"]
    target_train = _collect(
        models["presence"], datasets["target_train"], device, args.batch_size,
        seed=args.seed,
    )
    target_tokens = target_train["tokens"]
    capacity, correspondence = [], []
    for count in (8, 16, 32):
        source_centers = _normalized_kmeans(source_tokens, count, args.seed)
        target_centers = _normalized_kmeans(target_tokens, count, args.seed)
        joint_centers = _normalized_kmeans(
            np.concatenate((source_tokens, target_tokens)), count, args.seed,
        )
        for dictionary, centers in (
            ("source", source_centers), ("target", target_centers),
            ("joint", joint_centers),
        ):
            capacity.append({
                "M": count, "dictionary": dictionary,
                **{f"source_{key}": value for key, value in _coverage(source_tokens, centers).items()},
                **{f"target_{key}": value for key, value in _coverage(target_tokens, centers).items()},
            })
        cosine = source_centers @ target_centers.T
        left, right = linear_sum_assignment(-cosine)
        matched = cosine[left, right]
        correspondence.append({
            "M": count, "matched_cos_mean": float(matched.mean()),
            "matched_cos_median": float(np.median(matched)),
            "matched_cos_min": float(matched.min()),
            "matched_cos_p10": float(np.quantile(matched, .1)),
        })
    _write(output / "anchor_capacity_summary.csv", capacity)
    _write(output / "anchor_correspondence.csv", correspondence)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--source", required=True); parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
