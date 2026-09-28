#!/usr/bin/env python3
"""Read-only audit of the representations used by trained UQ checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_representation_chain_audit import (
    _pad_pixel_collate,
    build_audit_datasets,
    fit_source_probe,
    load_source_model,
    seed_all,
    target_oracle_probe,
)


TASKS = {
    "AT1_DK1": ("austria/33UVP/2017", "denmark/32VNH/2017"),
    "FR2_DK1": ("france/31TCJ/2017", "denmark/32VNH/2017"),
}
REPRESENTATIONS = ("E_flat", "z_Q", "z_T")
HARD_CLASSES = {
    "horsebeans", "spring_barley", "spring_peas",
    "winter_barley", "winter_triticale",
}
CSV_FIELDS = (
    "task", "scope", "representation", "class",
    "source_val_macro_f1", "target_oracle_macro_f1",
    "source_to_target_macro_f1", "source_to_target_f1",
    "source_local_local_attention_similarity",
    "source_local_base_attention_similarity",
    "target_local_local_attention_similarity",
    "target_local_base_attention_similarity",
)


def audit_checkpoint_path(root, task):
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    return Path(root) / f"uq_{task}_seed1" / "fold_0" / "model.pt"


def attention_similarity(local_attention, base_attention):
    """Average cosine across local pairs and between local/base attention maps."""
    if local_attention.ndim != 4 or base_attention.ndim != 4:
        raise ValueError("attention tensors must be [B,H,N,T] and [B,H,1,T]")
    local = F.normalize(local_attention.permute(0, 2, 1, 3).flatten(2), dim=-1)
    base = F.normalize(base_attention[:, :, 0].flatten(1), dim=-1)
    count = local.shape[1]
    pairwise = local @ local.transpose(1, 2)
    pair_mask = torch.triu(
        torch.ones(count, count, dtype=torch.bool, device=local.device), diagonal=1,
    )
    local_local = (
        pairwise[:, pair_mask].mean(dim=1)
        if pair_mask.any() else pairwise.new_full((local.shape[0],), float("nan"))
    )
    local_base = F.cosine_similarity(local, base[:, None], dim=-1).mean(dim=1)
    return {
        "local_local_attention_similarity": float(local_local.nanmean()),
        "local_base_attention_similarity": float(local_base.mean()),
    }


def _pixel_budget_batches(dataset, max_batch_size, pixel_budget):
    counts = [int(shape[2]) for shape in dataset.get_shapes()]
    order = sorted(range(len(counts)), key=lambda index: (counts[index], index))
    batches, current = [], []
    for index in order:
        candidate = current + [index]
        if current and (
            len(candidate) > int(max_batch_size)
            or len(candidate) * max(counts[item] for item in candidate) > int(pixel_budget)
        ):
            batches.append(current)
            current = [index]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _loader(dataset, batch_size, pixel_budget, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=_pixel_budget_batches(dataset, batch_size, pixel_budget),
        num_workers=int(num_workers), collate_fn=_pad_pixel_collate,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def extract_dataset(model, dataset, device, batch_size, pixel_budget, num_workers):
    values = {name: [] for name in REPRESENTATIONS}
    labels, local_local, local_base = [], [], []
    for batch in _loader(dataset, batch_size, pixel_budget, num_workers):
        pixels = batch["pixels"].to(device, non_blocking=True)
        valid = batch["valid_pixels"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        extra = batch["extra"].to(device, non_blocking=True)
        spatial = model.spatial_encoder(pixels, valid, extra)
        structure = model.prepare_structure(spatial, positions, temporal_shift=0)
        similarity = structure["shapelet_similarity"]
        correction = model._project_local_queries(similarity)
        base, local, base_attn, local_attn = (
            model.temporal_encoder.forward_with_local_queries(
                spatial, positions, correction, return_att=True,
            )
        )
        residual = local - base[:, None]
        alpha = torch.softmax(
            model.local_order_scorer(residual.transpose(1, 2)).squeeze(1), dim=1,
        )
        z_query = (alpha.unsqueeze(-1) * local).sum(1)
        values["E_flat"].append(similarity.flatten(1).float().cpu())
        values["z_Q"].append(z_query.float().cpu())
        values["z_T"].append(base.float().cpu())
        labels.append(batch["label"].long().cpu())
        sample_local = F.normalize(
            local_attn.permute(0, 2, 1, 3).flatten(2), dim=-1,
        )
        pairwise = sample_local @ sample_local.transpose(1, 2)
        mask = torch.triu(torch.ones(
            sample_local.shape[1], sample_local.shape[1], dtype=torch.bool,
            device=device,
        ), diagonal=1)
        local_local.append(pairwise[:, mask].mean(1).cpu())
        base_flat = F.normalize(base_attn[:, :, 0].flatten(1), dim=-1)
        local_base.append(F.cosine_similarity(
            sample_local, base_flat[:, None], dim=-1,
        ).mean(1).cpu())
    return {
        "representations": {
            name: torch.cat(parts).numpy() for name, parts in values.items()
        },
        "labels": torch.cat(labels).numpy(),
        "attention": {
            "local_local_attention_similarity": float(torch.cat(local_local).mean()),
            "local_base_attention_similarity": float(torch.cat(local_base).mean()),
        },
    }


def evaluate_representations(extracted, class_names, seed):
    class_ids = np.arange(len(class_names), dtype=np.int64)
    metrics = {}
    for name in REPRESENTATIONS:
        source = fit_source_probe(
            extracted["source_train"]["representations"][name],
            extracted["source_train"]["labels"],
            extracted["source_val"]["representations"][name],
            extracted["source_val"]["labels"],
            extracted["target_val"]["representations"][name],
            extracted["target_val"]["labels"], class_ids,
        )
        oracle = target_oracle_probe(
            extracted["target_val"]["representations"][name],
            extracted["target_val"]["labels"], class_ids, seed,
        )
        metrics[name] = {
            "source_val_macro_f1": source["source_val_macro_f1"],
            "target_oracle_macro_f1": oracle["macro_f1"],
            "source_to_target_macro_f1": source["source_to_target_macro_f1"],
            "source_to_target_per_class_f1": source["source_to_target_per_class_f1"],
        }
    return metrics


def build_audit_rows(task, class_names, metrics, attention):
    common = {
        "source_local_local_attention_similarity": attention["source_val"][
            "local_local_attention_similarity"
        ],
        "source_local_base_attention_similarity": attention["source_val"][
            "local_base_attention_similarity"
        ],
        "target_local_local_attention_similarity": attention["target_val"][
            "local_local_attention_similarity"
        ],
        "target_local_base_attention_similarity": attention["target_val"][
            "local_base_attention_similarity"
        ],
    }
    rows = []
    for name in REPRESENTATIONS:
        value = metrics[name]
        rows.append({
            "task": task, "scope": "macro", "representation": name,
            "class": "ALL", **common,
            "source_val_macro_f1": value["source_val_macro_f1"],
            "target_oracle_macro_f1": value["target_oracle_macro_f1"],
            "source_to_target_macro_f1": value["source_to_target_macro_f1"],
            "source_to_target_f1": "",
        })
        if task == "FR2_DK1":
            for class_id, class_name in enumerate(class_names):
                if class_name not in HARD_CLASSES:
                    continue
                rows.append({
                    "task": task, "scope": "per_class", "representation": name,
                    "class": class_name, **common,
                    "source_val_macro_f1": "", "target_oracle_macro_f1": "",
                    "source_to_target_macro_f1": "",
                    "source_to_target_f1": float(
                        value["source_to_target_per_class_f1"][class_id]
                    ),
                })
    return rows


def _json_metrics(metrics):
    return {
        name: {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in items.items()
        }
        for name, items in metrics.items()
    }


def run(args):
    checkpoint = audit_checkpoint_path(args.checkpoint_root, args.task)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"UQ best checkpoint not found: {checkpoint.resolve()}")
    source, target = TASKS[args.task]
    seed_all(args.seed)
    device = torch.device(args.device)
    model, config = load_source_model(checkpoint, device)
    if config.shape_injection != "local_query" or config.structure_shift_mode != "none":
        raise ValueError("audit requires a trained UQ local_query/none checkpoint")
    datasets, split = build_audit_datasets(
        config, source, target, args.data_root, args.seed,
    )
    extracted = {
        name: extract_dataset(
            model, dataset, device, args.batch_size,
            args.pixel_budget, args.num_workers,
        )
        for name, dataset in datasets.items()
    }
    metrics = evaluate_representations(extracted, config.classes, args.seed)
    attention = {
        split_name: extracted[split_name]["attention"]
        for split_name in ("source_val", "target_val")
    }
    rows = build_audit_rows(args.task, config.classes, metrics, attention)
    output = Path(args.output_root) / f"uq_{args.task}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    packet = {
        "task": args.task, "checkpoint": str(checkpoint), "checkpoint_role": "best",
        "splits": {name: len(indices) for name, indices in split.items()},
        "target_test_accessed": False, "representations": _json_metrics(metrics),
        "attention": attention,
        "probe": "StandardScaler+RidgeClassifier(alpha=1.0,class_weight=balanced)",
    }
    (output / "audit.json").write_text(
        json.dumps(packet, indent=2), encoding="utf-8",
    )
    print(f"LOCAL_QUERY_AUDIT_FINISHED|task={args.task}|output={output}")


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--task", required=True, choices=tuple(TASKS))
    value.add_argument("--checkpoint-root", default=(
        "outputs/structure_local_query_shift_2tasks_seed1/uda"
    ))
    value.add_argument("--output-root", default=(
        "outputs/structure_query_only_2tasks_seed1/audit"
    ))
    value.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    value.add_argument("--device", default="cuda")
    value.add_argument("--batch-size", type=int, default=128)
    value.add_argument("--pixel-budget", type=int, default=8192)
    value.add_argument("--num-workers", type=int, default=0)
    value.add_argument("--seed", type=int, default=1)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
