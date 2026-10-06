#!/usr/bin/env python3
"""Norm-controlled read-only interventions for the state-org LTAE query."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.state_org_feasibility_audit import (
    _datasets, _load_checkpoint, _loader, _macro, _move, _per_class,
)
from models.stclassifier import FrozenStateOrgReference


def compose_query_case(master_query, correction, case, rho=None, eps=1e-12):
    """Compose [B,H,D] queries and report correction/master norm ratios."""
    if master_query.ndim != 2 or correction.ndim != 3:
        raise ValueError("master_query must be [H,D] and correction [B,H,D]")
    master = master_query.unsqueeze(0).expand(correction.shape[0], -1, -1)
    master_norm = master.norm(dim=-1).clamp_min(eps)
    normalized = correction * (
        master_norm / correction.norm(dim=-1).clamp_min(eps)
    ).unsqueeze(-1)
    if case == "raw":
        query = master + correction
        ratio = correction.norm(dim=-1) / master_norm
    elif case == "shape_only":
        query = normalized
        ratio = normalized.norm(dim=-1) / master_norm
    elif case == "rho":
        if rho is None:
            raise ValueError("rho case requires rho")
        query = master + float(rho) * normalized
        ratio = (float(rho) * normalized).norm(dim=-1) / master_norm
    else:
        raise ValueError("query case must be rho, raw, or shape_only")
    return query, ratio


def _write(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _js_divergence(left, right, eps=1e-12):
    left = left.clamp_min(eps); right = right.clamp_min(eps)
    middle = .5 * (left + right)
    return .5 * (
        (left * (left.log() - middle.log())).sum(-1)
        + (right * (right.log() - middle.log())).sum(-1)
    )


def _configured_model(checkpoint, source_checkpoint, device):
    model, config, packet = _load_checkpoint(checkpoint, device)
    source, source_config, _ = _load_checkpoint(source_checkpoint, device)
    if getattr(config, "state_org_readout", "full") != getattr(
        source_config, "state_org_readout", "full",
    ):
        raise ValueError("query checkpoint and source reference readout differ")
    reference = None
    if "state_org_reference_state_dict" in packet:
        reference = FrozenStateOrgReference.from_source_model(source).to(device).eval()
        reference.load_state_dict(packet["state_org_reference_state_dict"], strict=True)
        model.configure_state_org_query("full", 1., reference)
    return model.eval(), config, reference


@torch.no_grad()
def _evaluate(model, reference, dataset, shift, device, batch_size, class_count):
    labels, predictions = [], defaultdict(list)
    attention_values = defaultdict(list); ratio_values = defaultdict(list)
    cases = [("master", "rho", 0.), ("rho_0.25", "rho", .25),
             ("rho_0.5", "rho", .5), ("rho_1", "rho", 1.),
             ("rho_2", "rho", 2.), ("raw", "raw", None),
             ("shape_only", "shape_only", None)]
    for raw_batch in _loader(dataset, batch_size):
        batch = _move(raw_batch, device)
        spatial = model.spatial_encoder(
            batch["pixels"], batch["valid_pixels"], batch["extra"],
        )
        if reference is None:
            structure = model.prepare_structure(spatial, batch["positions"], 0)
            evidence = model.shape_response_norm(structure["shapelet_response"])
        else:
            evidence = reference(
                batch["pixels"], batch["valid_pixels"],
                batch["positions"], batch["extra"],
            )
        heads = model.temporal_encoder.attention_heads
        correction = heads.external_query_projection(evidence).reshape(
            evidence.shape[0], heads.n_head, heads.d_k,
        )
        master = heads.query
        batch_attention = {}
        for name, kind, rho in cases:
            query, ratio = compose_query_case(master, correction, kind, rho=rho)
            instance, attention = model.temporal_encoder.forward_with_explicit_queries(
                spatial, batch["positions"] + shift, query[:, None], return_att=True,
            )
            logits = model.decoder(instance[:, 0])
            predictions[name].append(logits.argmax(1).cpu())
            batch_attention[name] = attention[:, :, 0]
            ratio_values[name].append(ratio.mean((1,)).cpu())
        master_attention = batch_attention["master"]
        for name in batch_attention:
            current = batch_attention[name]
            attention_values[name].append(torch.stack((
                F.cosine_similarity(
                    current.flatten(1), master_attention.flatten(1), dim=1,
                ),
                _js_divergence(current, master_attention).mean(1),
            ), dim=1).cpu())
        labels.append(batch["label"].cpu())
    labels = torch.cat(labels).numpy()
    prediction = {name: torch.cat(parts).numpy() for name, parts in predictions.items()}
    master_prediction = prediction["master"]
    result = {}
    for name in prediction:
        result[name] = {
            "prediction": prediction[name],
            "macro_f1": _macro(labels, prediction[name], class_count),
            "prediction_change_rate": float(np.mean(prediction[name] != master_prediction)),
            "attention": torch.cat(attention_values[name]).numpy(),
            "norm_ratio": float(torch.cat(ratio_values[name]).mean()),
        }
    return labels, result


def run(args):
    device = torch.device(args.device)
    shift_packet = torch.load(args.shift_checkpoint, map_location="cpu", weights_only=False)
    shift = int(round(float(shift_packet["global_temporal_shift"])))
    rows, class_rows, attention_rows = [], [], []
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    for spec in args.checkpoint:
        name, checkpoint, source_checkpoint, readout = spec.split("::", 3)
        model, config, reference = _configured_model(
            Path(checkpoint), Path(source_checkpoint), device,
        )
        if getattr(config, "state_org_readout", "full") != readout:
            raise ValueError(f"readout mismatch for {name}")
        datasets = _datasets(config, args.source, args.target, args.data_root, args.seed)
        labels, cases = _evaluate(
            model, reference, datasets["target_test"], shift, device, args.batch_size,
            len(config.classes),
        )
        class_count = len(config.classes)
        for case, values in cases.items():
            rows.append({
                "checkpoint": name, "readout": readout, "case": case,
                "macro_f1": values["macro_f1"],
                "prediction_change_rate_vs_master": values["prediction_change_rate"],
                "correction_master_norm_ratio": values["norm_ratio"],
            })
            per_class = _per_class(labels, values["prediction"], class_count)
            for class_id, class_name in enumerate(config.classes):
                class_rows.append({
                    "checkpoint": name, "readout": readout, "case": case,
                    "class_id": class_id, "class_name": class_name,
                    "f1": per_class[class_id],
                })
            attention_rows.append({
                "checkpoint": name, "readout": readout, "case": case,
                "attention_cosine_vs_master": float(values["attention"][:, 0].mean()),
                "attention_js_vs_master": float(values["attention"][:, 1].mean()),
            })
    _write(output / "query_role_audit.csv", rows)
    _write(output / "query_role_per_class.csv", class_rows)
    _write(output / "query_attention_audit.csv", attention_rows)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--shift-checkpoint", required=True)
    parser.add_argument("--source", required=True); parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
