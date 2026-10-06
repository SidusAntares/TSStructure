#!/usr/bin/env python3
"""Read-only target inference sweep for state-org query correction scale."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.state_org_feasibility_audit import (
    _checkpoint_shift, _datasets, _load_checkpoint, _loader, _macro, _move, _per_class,
)
from models.stclassifier import FrozenStateOrgReference


def _write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def _evaluate(model, reference, dataset, shift, view, gamma, device, batch_size):
    labels, predictions, ratios = [], [], []
    projection = model.temporal_encoder.attention_heads.external_query_projection
    master_norm = model.temporal_encoder.attention_heads.query.norm().clamp_min(1e-12)
    for raw in _loader(dataset, batch_size):
        batch = _move(raw, device)
        evidence = None
        if reference is not None:
            evidence = reference(batch["pixels"], batch["valid_pixels"], batch["positions"], batch["extra"])
            output = model.forward_with_external_shape_evidence(
                batch["pixels"], batch["valid_pixels"], batch["positions"], batch["extra"],
                evidence, temporal_shift=shift, query_view=view, query_scale=gamma,
            )
        else:
            output = model.forward_with_temporal_shift(
                batch["pixels"], batch["valid_pixels"], batch["positions"], batch["extra"],
                temporal_shift=shift, return_dict=True, query_view=view, query_scale=gamma,
            )
            evidence = output["shape_evidence"]
        query = model.select_state_org_query(evidence, view)
        ratios.append((float(gamma) * projection(query).norm(dim=1) / master_norm).cpu())
        predictions.append(output["logits"].argmax(1).cpu()); labels.append(batch["label"].cpu())
    return torch.cat(labels).numpy(), torch.cat(predictions).numpy(), float(torch.cat(ratios).mean())


def run(args):
    device = torch.device(args.device); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    source_model, source_config, _ = _load_checkpoint(Path(args.source_checkpoint), device)
    _, shift_config, _ = _load_checkpoint(Path(args.shift_config_checkpoint), device)
    datasets = _datasets(source_config, args.source, args.target, args.data_root, args.seed)
    reference = FrozenStateOrgReference.from_source_model(source_model).to(device).eval()
    rows, class_rows = [], []
    for spec in args.checkpoint:
        name, path, view, basis = spec.split("::")
        model, config, packet = _load_checkpoint(Path(path), device)
        ref = reference if basis == "frozen_source" else None
        shift = _checkpoint_shift(
            "source" if name == "source" else "checkpoint", packet,
            output / "source_initial_shift.json", model, datasets["target_train"],
            shift_config if name == "source" else config, device, args.batch_size,
        )
        for gamma in (0., .25, .5, 1.):
            labels, prediction, ratio = _evaluate(
                model, ref, datasets["target_test"], shift, view, gamma,
                device, args.batch_size,
            )
            macro = _macro(labels, prediction, len(config.classes)); per = _per_class(labels, prediction, len(config.classes))
            rows.append({"checkpoint": name, "basis": basis, "query_view": view, "gamma": gamma, "target_macro_f1": macro, "query_master_norm_ratio": ratio})
            for class_id, class_name in enumerate(config.classes):
                class_rows.append({"checkpoint": name, "basis": basis, "query_view": view, "gamma": gamma, "class_id": class_id, "class_name": class_name, "f1": per[class_id]})
    _write(output / "query_scale_sweep.csv", rows); _write(output / "query_scale_per_class.csv", class_rows)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--shift-config-checkpoint", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="name::path::view::basis")
    parser.add_argument("--source", required=True); parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--seed", type=int, default=1)
    return parser


if __name__ == "__main__": run(build_parser().parse_args())
