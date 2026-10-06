#!/usr/bin/env python3
"""Frozen-source Anchor -> Organization -> Query audits for state_org."""

from __future__ import annotations

import argparse
import csv
import json
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

from analysis.state_org_feasibility_audit import (
    _checkpoint_shift, _datasets, _load_checkpoint, _loader, _macro, _move,
    _per_class, _probe, organization_counterfactuals,
)
from models.stclassifier import FrozenStateOrgReference


FEATURE_NAMES = (
    "presence", "presence_composition", "presence_t1",
    "presence_t1_t2", "presence_current_org",
)


def organization_statistics(distribution):
    if distribution.ndim != 3:
        raise ValueError("state distribution must be [B,N,M]")
    composition = distribution.mean(1)
    t1 = torch.einsum(
        "bnm,bnk->bmk", distribution, torch.roll(distribution, -1, 1),
    ) / distribution.shape[1]
    t2 = torch.einsum(
        "bnm,bnk->bmk", distribution, torch.roll(distribution, -2, 1),
    ) / distribution.shape[1]
    return {"composition": composition, "t1": t1, "t2": t2}


def counterfactual_differences(original, changed):
    return {
        "organization_feature_l2_diff": float(
            (changed["organization"] - original["organization"]).norm(dim=1).mean()
        ),
        "query_correction_l2_diff": float(
            (changed["query"] - original["query"]).norm(dim=1).mean()
        ),
        "logit_l2_diff": float(
            (changed["logits"] - original["logits"]).norm(dim=1).mean()
        ),
        "prediction_change_rate": float(
            (changed["logits"].argmax(1) != original["logits"].argmax(1)).float().mean()
        ),
    }


def _write(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def _extract(model, dataset, device, batch_size, shift=0, token_limit=50000, seed=1):
    values = defaultdict(list)
    token_reservoir = None
    rng = np.random.default_rng(seed)
    projection = model.temporal_encoder.attention_heads.external_query_projection
    for raw in _loader(dataset, batch_size):
        batch = _move(raw, device)
        spatial = model.spatial_encoder(batch["pixels"], batch["valid_pixels"], batch["extra"])
        structure = model.prepare_structure(spatial, batch["positions"], 0)
        q = structure["state_distribution"]
        stats = organization_statistics(q)
        presence = structure["shapelet_presence"]
        current_org = structure["shape_organization"]
        feature = {
            "presence": presence,
            "presence_composition": torch.cat((presence, stats["composition"]), 1),
            "presence_t1": torch.cat((presence, stats["t1"].flatten(1)), 1),
            "presence_t1_t2": torch.cat((presence, stats["t1"].flatten(1), stats["t2"].flatten(1)), 1),
            "presence_current_org": torch.cat((presence, current_org), 1),
        }
        for key, tensor in feature.items(): values[key].append(tensor.cpu())
        tokens = structure["shape_tokens"].flatten(0, 1).cpu()
        token_reservoir = tokens if token_reservoir is None else torch.cat((token_reservoir, tokens))
        if token_reservoir.shape[0] > token_limit:
            keep = rng.choice(token_reservoir.shape[0], token_limit, replace=False)
            token_reservoir = token_reservoir[torch.from_numpy(keep)]
        values["label"].append(batch["label"].cpu())
        variants = organization_counterfactuals(model, structure)
        for name, details in variants.items():
            evidence = model.shape_response_norm(details["shapelet_response"])
            query = projection(evidence)
            instance = model.temporal_encoder(
                spatial, batch["positions"] + shift, external_query=evidence,
            )
            values[f"cf_{name}_organization"].append(details["organization"].cpu())
            values[f"cf_{name}_query"].append(query.cpu())
            values[f"cf_{name}_logits"].append(model.decoder(instance).cpu())
    result = {key: torch.cat(items).numpy() for key, items in values.items()}
    result["tokens"] = token_reservoir.numpy()
    return result


@torch.no_grad()
def _extract_tokens(model, dataset, device, batch_size, token_limit=50000, seed=1):
    reservoir = None; rng = np.random.default_rng(seed)
    for raw in _loader(dataset, batch_size):
        batch = _move(raw, device)
        spatial = model.spatial_encoder(batch["pixels"], batch["valid_pixels"], batch["extra"])
        tokens = model.prepare_structure(spatial, batch["positions"], 0)["shape_tokens"].flatten(0, 1).cpu()
        reservoir = tokens if reservoir is None else torch.cat((reservoir, tokens))
        if reservoir.shape[0] > token_limit:
            keep = rng.choice(reservoir.shape[0], token_limit, replace=False)
            reservoir = reservoir[torch.from_numpy(keep)]
    return reservoir.numpy()


def _normalized_kmeans(tokens, count, seed):
    estimator = KMeans(n_clusters=count, random_state=seed, n_init=10)
    centers = estimator.fit(tokens).cluster_centers_
    return centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)


def _coverage(tokens, anchors):
    score = tokens @ anchors.T
    best = score.max(1)
    return {
        "mean": float(best.mean()), "median": float(np.median(best)),
        "p10": float(np.quantile(best, .1)), "p25": float(np.quantile(best, .25)),
    }


def run_basis(args):
    device = torch.device(args.device)
    model, config, packet = _load_checkpoint(Path(args.source_checkpoint), device)
    datasets = _datasets(config, args.source, args.target, args.data_root, args.seed)
    _, _, shift_packet = _load_checkpoint(Path(args.shift_checkpoint), device)
    shift = int(round(float(shift_packet["global_temporal_shift"])))
    source = _extract(model, datasets["source_train"], device, args.batch_size)
    target_tokens = _extract_tokens(model, datasets["target_train"], device, args.batch_size, seed=args.seed)
    target_val = _extract(model, datasets["target_val"], device, args.batch_size, shift=shift)
    target = _extract(model, datasets["target_test"], device, args.batch_size, shift=shift)
    zs = source["tokens"]; zt = target_tokens
    zs = zs / np.maximum(np.linalg.norm(zs, axis=1, keepdims=True), 1e-12)
    zt = zt / np.maximum(np.linalg.norm(zt, axis=1, keepdims=True), 1e-12)
    rows, correspondence = [], []
    learned = F.normalize(
        model.structure_branch.shapelet_dictionary.anchors.detach(), dim=-1,
    ).cpu().numpy()
    for count in (8, 16, 32):
        source_centers = _normalized_kmeans(zs, count, args.seed)
        target_centers = _normalized_kmeans(zt, count, args.seed)
        joint_centers = _normalized_kmeans(np.concatenate((zs, zt)), count, args.seed)
        dictionaries = {
            "source_kmeans": source_centers, "target_kmeans": target_centers,
            "joint_kmeans": joint_centers,
        }
        if count == 16: dictionaries["learned_source"] = learned
        for name, anchors in dictionaries.items():
            rows.append({"M": count, "dictionary": name, **_coverage(zt, anchors)})
        cosine = source_centers @ target_centers.T
        left, right = linear_sum_assignment(-cosine)
        matched = cosine[left, right]
        correspondence.append({
            "M": count, "matched_cosine_mean": matched.mean(),
            "matched_cosine_median": np.median(matched),
            "matched_cosine_min": matched.min(),
            "matched_cosine_p10": np.quantile(matched, .1),
        })
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    _write(output / "anchor_capacity_summary.csv", rows)
    _write(output / "anchor_correspondence.csv", correspondence)
    probe_rows, class_rows = [], []
    for name in FEATURE_NAMES:
        transfer, transfer_pc = _probe(
            source[name], source["label"], target[name], target["label"],
            len(config.classes), args.seed,
        )
        oracle, oracle_pc = _probe(
            target_val[name], target_val["label"], target[name], target["label"],
            len(config.classes), args.seed,
        )
        probe_rows.append({"feature": name, "source_target_macro_f1": transfer, "target_oracle_macro_f1": oracle})
        for class_id, class_name in enumerate(config.classes):
            class_rows.append({"feature": name, "class_id": class_id, "class_name": class_name, "source_target_f1": transfer_pc[class_id], "target_oracle_f1": oracle_pc[class_id]})
    base = next(row for row in probe_rows if row["feature"] == "presence")
    for row in probe_rows:
        row["source_target_delta_vs_presence"] = row["source_target_macro_f1"] - base["source_target_macro_f1"]
        row["oracle_delta_vs_presence"] = row["target_oracle_macro_f1"] - base["target_oracle_macro_f1"]
    _write(output / "organization_probe.csv", probe_rows)
    _write(output / "organization_probe_per_class.csv", class_rows)
    cf_rows = []
    original = {key: torch.from_numpy(target[f"cf_original_{key}"]) for key in ("organization", "query", "logits")}
    for name in ("roll_1", "roll_2", "reverse", "mean_repeat"):
        changed = {key: torch.from_numpy(target[f"cf_{name}_{key}"]) for key in ("organization", "query", "logits")}
        cf_rows.append({"variant": name, **counterfactual_differences(original, changed)})
    _write(output / "organization_counterfactual_differences.csv", cf_rows)
    (output / "audit_manifest.json").write_text(json.dumps({
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "target_labels_used_for_dictionary": False, "token_limit": 50000,
        "seed": args.seed,
    }, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--shift-checkpoint", required=True)
    parser.add_argument("--source", required=True); parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--seed", type=int, default=1)
    return parser


if __name__ == "__main__":
    run_basis(build_parser().parse_args())
