#!/usr/bin/env python3
"""Read-only Presence-basis audit of explicit state-org temporal shifts."""

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

from analysis.state_org_anchor_basis_audit import organization_statistics
from analysis.state_org_feasibility_audit import (
    _datasets, _load_checkpoint, _loader, _macro, _move, _per_class, _probe,
)


FEATURES = ("P", "P_C", "P_T1", "P_T1_T2")


def shift_plan(global_shift, class_shifts):
    """Keep source coordinates fixed and enumerate target-only conditions."""
    return {
        "source": 0,
        "target": {
            "raw": 0,
            "timematch_global": int(global_shift),
            "oracle_class_shift": dict(class_shifts),
        },
    }


def write_rows(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def select_oracle_class_shifts(log_probabilities, labels, candidates, class_count):
    """Select shifts using only the supplied target-validation predictions."""
    values = np.asarray(log_probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    candidates = np.asarray(candidates)
    if values.shape != (labels.shape[0], candidates.shape[0], int(class_count)):
        raise ValueError("oracle log-probabilities must be [samples,shifts,classes]")
    selected = {}
    for class_id in range(int(class_count)):
        mask = labels == class_id
        if not np.any(mask):
            raise ValueError(f"target validation has no samples for class {class_id}")
        score = values[mask, :, class_id].mean(0)
        selected[class_id] = int(candidates[int(np.argmax(score))])
    return selected


def feature_views(structure):
    q = structure["state_distribution"]
    statistics = organization_statistics(q)
    presence = structure["shapelet_presence"]
    return {
        "P": presence,
        "P_C": torch.cat((presence, statistics["composition"]), dim=1),
        "P_T1": torch.cat((presence, statistics["t1"].flatten(1)), dim=1),
        "P_T1_T2": torch.cat((
            presence, statistics["t1"].flatten(1),
            statistics["t2"].flatten(1),
        ), dim=1),
        "C": statistics["composition"],
        "T1": statistics["t1"].flatten(1),
        "T2": statistics["t2"].flatten(1),
        "Q": q,
        "tokens": structure["shape_tokens"],
        "similarity": structure["shapelet_similarity"],
    }


def _batch_shift(labels, shift):
    if isinstance(shift, dict):
        return torch.as_tensor(
            [shift[int(value)] for value in labels.detach().cpu().tolist()],
            device=labels.device, dtype=torch.float32,
        )
    return float(shift)


def _shift_positions(positions, shift):
    return positions + (shift[:, None] if torch.is_tensor(shift) else shift)


def _shifted_structure(model, context, shift):
    return model.prepare_structure_from_context(context, temporal_shift=shift)


@torch.no_grad()
def collect_shift_features(model, dataset, shift, device, batch_size):
    storage = defaultdict(list)
    model.eval()
    for raw in _loader(dataset, batch_size):
        batch = _move(raw, device)
        spatial = model.spatial_encoder(
            batch["pixels"], batch["valid_pixels"], batch["extra"],
        )
        context = model.prepare_structure_context(spatial, batch["positions"])
        current_shift = _batch_shift(batch["label"], shift)
        structure = _shifted_structure(model, context, current_shift)
        views = feature_views(structure)
        instance = model._encode_instance(
            spatial, _shift_positions(batch["positions"], current_shift), structure,
        )
        logits = model.decoder(instance)
        for name, value in views.items():
            storage[name].append(value.cpu())
        storage["label"].append(batch["label"].cpu())
        storage["prediction"].append(logits.argmax(1).cpu())
    return {name: torch.cat(parts).numpy() for name, parts in storage.items()}


@torch.no_grad()
def validation_shift_grid(model, dataset, candidates, device, batch_size):
    all_labels, all_values = [], []
    model.eval()
    for raw in _loader(dataset, batch_size):
        batch = _move(raw, device)
        spatial = model.spatial_encoder(
            batch["pixels"], batch["valid_pixels"], batch["extra"],
        )
        context = model.prepare_structure_context(spatial, batch["positions"])
        values = []
        for shift in candidates:
            structure = _shifted_structure(model, context, int(shift))
            instance = model._encode_instance(
                spatial, batch["positions"] + int(shift), structure,
            )
            values.append(F.log_softmax(model.decoder(instance), dim=-1).cpu())
        all_values.append(torch.stack(values, dim=1))
        all_labels.append(batch["label"].cpu())
    return torch.cat(all_values).numpy(), torch.cat(all_labels).numpy()


def _cosine(left, right):
    left = torch.as_tensor(left).flatten(1).float()
    right = torch.as_tensor(right).flatten(1).float()
    return F.cosine_similarity(left, right, dim=1).numpy()


def sensitivity_rows(raw, shifted, classes):
    labels = raw["label"]
    metrics = {
        "P_cosine": _cosine(raw["P"], shifted["P"]),
        "C_cosine": _cosine(raw["C"], shifted["C"]),
        "T1_cosine": _cosine(raw["T1"], shifted["T1"]),
        "T2_cosine": _cosine(raw["T2"], shifted["T2"]),
        "window_top1_anchor_agreement": (
            raw["Q"].argmax(-1) == shifted["Q"].argmax(-1)
        ).mean(1),
        "mean_abs_Q_difference": np.abs(raw["Q"] - shifted["Q"]).mean((1, 2)),
        "mean_token_cosine": F.cosine_similarity(
            torch.as_tensor(raw["tokens"]).float(),
            torch.as_tensor(shifted["tokens"]).float(), dim=-1,
        ).mean(1).numpy(),
    }
    rows = []
    scopes = [("all", -1, np.ones(labels.shape[0], dtype=bool))]
    scopes.extend(
        (classes[class_id], class_id, labels == class_id)
        for class_id in range(len(classes))
    )
    for scope, class_id, mask in scopes:
        row = {"scope": scope, "class_id": class_id, "samples": int(mask.sum())}
        row.update({name: float(value[mask].mean()) for name, value in metrics.items()})
        rows.append(row)
    return rows


def run(args):
    device = torch.device(args.device)
    model, config, _ = _load_checkpoint(Path(args.source_checkpoint), device)
    if getattr(config, "state_org_readout", "full") != "presence":
        raise ValueError("shift audit requires Presence source checkpoint")
    shift_packet = torch.load(
        args.shift_checkpoint, map_location="cpu", weights_only=False,
    )
    global_shift = int(round(float(shift_packet["global_temporal_shift"])))
    datasets = _datasets(config, args.source, args.target, args.data_root, args.seed)
    candidates = np.arange(-60, 61, dtype=np.int64)
    val_grid, val_labels = validation_shift_grid(
        model, datasets["target_val"], candidates, device, args.batch_size,
    )
    class_shifts = select_oracle_class_shifts(
        val_grid, val_labels, candidates, len(config.classes),
    )
    plan = shift_plan(global_shift, class_shifts)
    source = collect_shift_features(
        model, datasets["source_train"], plan["source"], device, args.batch_size,
    )
    conditions = plan["target"]
    target_val = {
        name: collect_shift_features(
            model, datasets["target_val"], shift, device, args.batch_size,
        ) for name, shift in conditions.items()
    }
    target_test = {
        name: collect_shift_features(
            model, datasets["target_test"], shift, device, args.batch_size,
        ) for name, shift in conditions.items()
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    sensitivity = sensitivity_rows(
        target_test["raw"], target_test["timematch_global"], config.classes,
    )
    for row in sensitivity:
        row.update({"global_shift": global_shift, "comparison": "raw_vs_timematch_global"})
    write_rows(output / "shift_sensitivity.csv", sensitivity)

    summary, per_class = [], []
    for condition in conditions:
        for feature in FEATURES:
            transfer, transfer_pc = _probe(
                source[feature], source["label"],
                target_test[condition][feature], target_test[condition]["label"],
                len(config.classes), args.seed,
            )
            oracle, oracle_pc = _probe(
                target_val[condition][feature], target_val[condition]["label"],
                target_test[condition][feature], target_test[condition]["label"],
                len(config.classes), args.seed,
            )
            summary.extend((
                {"condition": condition, "feature": feature, "probe": "source_target", "macro_f1": transfer},
                {"condition": condition, "feature": feature, "probe": "target_oracle", "macro_f1": oracle},
            ))
            for class_id, class_name in enumerate(config.classes):
                per_class.extend((
                    {"condition": condition, "feature": feature, "probe": "source_target", "class_id": class_id, "class_name": class_name, "f1": transfer_pc[class_id]},
                    {"condition": condition, "feature": feature, "probe": "target_oracle", "class_id": class_id, "class_name": class_name, "f1": oracle_pc[class_id]},
                ))
    write_rows(output / "shift_probe.csv", summary)
    write_rows(output / "shift_probe_per_class.csv", per_class)
    oracle_rows = []
    for class_id, class_name in enumerate(config.classes):
        mask = val_labels == class_id
        chosen = class_shifts[class_id]
        oracle_rows.append({
            "class_id": class_id, "class_name": class_name,
            "target_val_support": int(mask.sum()), "oracle_shift": chosen,
            "selected_mean_log_probability": float(
                val_grid[mask, np.where(candidates == chosen)[0][0], class_id].mean()
            ),
            "ORACLE_DIAGNOSTIC_ONLY": True,
            "TARGET_TEST_LABEL_NOT_USED_FOR_SHIFT_SELECTION": True,
        })
    write_rows(output / "classwise_oracle_shift.csv", oracle_rows)
    print(
        "STATE_ORG_SHIFT_AUDIT_FINISHED|"
        f"global_shift={global_shift}|output={output}|"
        "oracle_diagnostic_only=true|target_test_label_used_for_search=false"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--shift-checkpoint", required=True)
    parser.add_argument("--source", required=True); parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
