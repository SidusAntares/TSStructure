#!/usr/bin/env python3
"""Read-only audit of normalized assignment versus raw anchor affinity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.state_org_feasibility_audit import _datasets, _load_checkpoint, _probe
from analysis.state_org_shift_sensitivity_audit import (
    collect_shift_features, write_rows,
)


def assignment_views(similarity, presence, beta):
    """Expose exact raw cosine and softmax assignment views without remapping."""
    soft = torch.softmax(float(beta) * similarity, dim=-1)
    return {
        "raw": similarity,
        "soft": soft,
        "presence_raw": presence,
        "presence_soft": presence,
    }


def _features(values, beta):
    similarity = torch.as_tensor(values["similarity"]).float()
    presence = torch.as_tensor(values["P"]).float()
    views = assignment_views(similarity, presence, beta)
    soft_composition = views["soft"].mean(1)
    raw_composition = views["raw"].mean(1)
    return {
        "P": presence.numpy(),
        "P_C_soft": torch.cat((presence, soft_composition), 1).numpy(),
        "P_C_raw": torch.cat((presence, raw_composition), 1).numpy(),
        "C_soft": soft_composition.numpy(),
        "C_raw": raw_composition.numpy(),
    }


def _distribution_row(values, beta, domain, condition):
    similarity = torch.as_tensor(values["similarity"]).float()
    soft = torch.softmax(float(beta) * similarity, dim=-1)
    top = torch.topk(similarity, 2, dim=-1).values
    return {
        "domain": domain, "condition": condition,
        "mean_max_cosine": float(top[..., 0].mean()),
        "mean_cosine_margin_top1_top2": float((top[..., 0] - top[..., 1]).mean()),
        "mean_max_softmax_probability": float(soft.max(-1).values.mean()),
        "softmax_entropy": float(
            -(soft * soft.clamp_min(1e-12).log()).sum(-1).mean()
        ),
    }


def run(args):
    device = torch.device(args.device)
    model, config, _ = _load_checkpoint(Path(args.source_checkpoint), device)
    if getattr(config, "state_org_readout", "full") != "presence":
        raise ValueError("assignment audit requires Presence source checkpoint")
    packet = torch.load(args.shift_checkpoint, map_location="cpu", weights_only=False)
    global_shift = int(round(float(packet["global_temporal_shift"])))
    datasets = _datasets(config, args.source, args.target, args.data_root, args.seed)
    beta = float(model.structure_branch.shapelet_dictionary.beta)
    source_values = collect_shift_features(
        model, datasets["source_train"], 0, device, args.batch_size,
    )
    source = _features(source_values, beta)
    conditions = {"raw": 0, "timematch_global": global_shift}
    target_val, target_test = {}, {}
    for name, shift in conditions.items():
        target_val[name] = collect_shift_features(
            model, datasets["target_val"], shift, device, args.batch_size,
        )
        target_test[name] = collect_shift_features(
            model, datasets["target_test"], shift, device, args.batch_size,
        )
    rows, per_class = [], []
    for condition in conditions:
        val_features = _features(target_val[condition], beta)
        test_features = _features(target_test[condition], beta)
        for feature in ("P", "P_C_soft", "P_C_raw", "C_soft", "C_raw"):
            transfer, transfer_pc = _probe(
                source[feature], source_values["label"],
                test_features[feature], target_test[condition]["label"],
                len(config.classes), args.seed,
            )
            oracle, oracle_pc = _probe(
                val_features[feature], target_val[condition]["label"],
                test_features[feature], target_test[condition]["label"],
                len(config.classes), args.seed,
            )
            rows.extend((
                {"condition": condition, "feature": feature, "probe": "source_target", "macro_f1": transfer},
                {"condition": condition, "feature": feature, "probe": "target_oracle", "macro_f1": oracle},
            ))
            for class_id, class_name in enumerate(config.classes):
                per_class.extend((
                    {"condition": condition, "feature": feature, "probe": "source_target", "class_id": class_id, "class_name": class_name, "f1": transfer_pc[class_id]},
                    {"condition": condition, "feature": feature, "probe": "target_oracle", "class_id": class_id, "class_name": class_name, "f1": oracle_pc[class_id]},
                ))
    rows.append({
        "condition": "source_shift0", "feature": "distribution",
        "probe": "statistics",
        **_distribution_row(source_values, beta, "source", "shift0"),
    })
    for condition in conditions:
        rows.append({
            "feature": "distribution", "probe": "statistics",
            **_distribution_row(target_test[condition], beta, "target", condition),
        })
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "assignment_probe.csv", rows)
    write_rows(output / "assignment_probe_per_class.csv", per_class)
    print(
        "STATE_ORG_ASSIGNMENT_AUDIT_FINISHED|"
        f"global_shift={global_shift}|beta={beta:.6f}|output={output}"
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
