#!/usr/bin/env python3
"""Frozen-checkpoint feasibility audit for the state-org representation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FEATURES = ("presence", "organization", "shape_response", "instance_feature")
QUERY_CASES = ("master", "presence", "organization", "full")
ORG_CASES = ("original", "roll_1", "roll_2", "reverse", "mean_repeat")


def query_counterfactuals(model, response):
    """Split one shared LayerNorm result before the linear query projection."""
    normalized = model.shape_response_norm(response)
    presence = torch.cat((normalized[:, :16], torch.zeros_like(normalized[:, 16:])), 1)
    organization = torch.cat((torch.zeros_like(normalized[:, :16]), normalized[:, 16:]), 1)
    projection = model.temporal_encoder.attention_heads.external_query_projection
    if projection is None:
        raise ValueError("state_org audit requires external_query_projection")
    return {
        "normalized": normalized,
        "master_external_query": torch.zeros_like(normalized),
        "presence_external_query": presence,
        "organization_external_query": organization,
        "full_external_query": normalized,
        "presence_delta": projection(presence),
        "organization_delta": projection(organization),
        "full_delta": projection(normalized),
    }


def _encode_organization(branch, distribution):
    return branch.organization_encoder(distribution.transpose(1, 2)).mean(-1)


def organization_counterfactuals(model, state_details):
    """Re-evaluate organization while keeping the original presence fixed."""
    distribution = state_details["state_distribution"]
    presence = state_details["presence"]
    variants = {
        "original": distribution,
        "roll_1": torch.roll(distribution, 1, 1),
        "roll_2": torch.roll(distribution, 2, 1),
        "reverse": torch.flip(distribution, (1,)),
        "mean_repeat": distribution.mean(1, keepdim=True).expand_as(distribution),
    }
    return {
        name: {
            "presence": presence,
            "state_distribution": value,
            "organization": _encode_organization(model.structure_branch, value),
            "shapelet_response": torch.cat(
                (presence, _encode_organization(model.structure_branch, value)), 1,
            ),
        }
        for name, value in variants.items()
    }


def rule_labels(metrics, threshold=.005):
    """Return only deterministic evidence labels; never select a future model."""
    labels = []
    if metrics.get("target_oracle", 0.) < .5:
        labels.append("TARGET_REPRESENTATION_WEAK")
    if metrics.get("target_oracle", 0.) >= .7 and metrics.get("source_to_target", 1.) <= .6:
        labels.append("CROSS_DOMAIN_GEOMETRY_MISMATCH")
    if (
        metrics.get("presence_minus_master", 0.) >= 0.
        and metrics.get("organization_minus_master", 0.) < 0.
        and metrics.get("mean_repeat_minus_original", 0.) > 0.
    ):
        labels.append("ORGANIZATION_NEGATIVE_TRANSFER")
    if (
        metrics.get("source_full_minus_master", 0.) > 0.
        and metrics.get("last_full_minus_master", 0.) < 0.
    ):
        labels.append("UDA_STRUCTURE_DRIFT")
    if metrics.get("no_shape_aux_peak_final_gain", 0.) > threshold:
        labels.append("SOURCE_AUX_NEGATIVE_TRANSFER")
    if metrics.get("detach_target_final_gain", 0.) > threshold:
        labels.append("PSEUDO_STRUCTURE_CONTAMINATION")
    return labels


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _seed(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))


def _load_checkpoint(path, device, teacher=False):
    from train import create_model
    from types import SimpleNamespace

    packet = torch.load(path, map_location=device, weights_only=False)
    config = packet.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint config missing: {path}")
    if config.get("shape_representation") != "state_org":
        raise ValueError(f"checkpoint is not state_org: {path}")
    if config.get("shape_injection") != "direct_response_query":
        raise ValueError(f"checkpoint is not direct_response_query: {path}")
    namespace = SimpleNamespace(**config)
    model = create_model(namespace)
    state_key = "teacher_state_dict" if teacher and "teacher_state_dict" in packet else "state_dict"
    model.load_state_dict(packet[state_key], strict=True)
    return model.to(device).eval(), namespace, packet


def _datasets(config, source, target, data_root, seed):
    from dataset import PixelSetData
    from train import create_train_val_test_folds
    from transforms import Normalize, ToTensor
    from torchvision.transforms import transforms

    bare = {
        name: PixelSetData(
            data_root, name, config.classes, closed_set=True,
            combine_spring_and_winter=config.combine_spring_and_winter,
        ) for name in (source, target)
    }
    eligible = {name: data.get_parcel_indices().tolist() for name, data in bare.items()}
    _seed(seed)
    split = create_train_val_test_folds(
        [source, target], 1, eligible, config.val_ratio, config.test_ratio,
    )[0]
    transform = transforms.Compose([Normalize(), ToTensor()])
    datasets = {}
    for alias, name in (("source", source), ("target", target)):
        for part in ("train", "val", "test"):
            datasets[f"{alias}_{part}"] = PixelSetData(
                data_root, name, config.classes, transform=transform,
                indices=split[name][part], closed_set=True,
                combine_spring_and_winter=config.combine_spring_and_winter,
            )
    return datasets


def _loader(dataset, batch_size):
    from analysis.structure_representation_chain_audit import deterministic_loader

    return deterministic_loader(dataset, batch_size, num_workers=0)


def _move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _macro(labels, prediction, class_count):
    return float(f1_score(
        labels, prediction, labels=np.arange(class_count),
        average="macro", zero_division=0,
    ))


def _per_class(labels, prediction, class_count):
    return f1_score(
        labels, prediction, labels=np.arange(class_count),
        average=None, zero_division=0,
    )


def _probe(train_x, train_y, test_x, test_y, class_count, seed):
    estimator = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced", random_state=int(seed), max_iter=1000,
        )),
    ])
    estimator.fit(train_x, train_y)
    prediction = estimator.predict(test_x)
    return _macro(test_y, prediction, class_count), _per_class(
        test_y, prediction, class_count,
    )


@torch.no_grad()
def _collect(model, dataset, device, batch_size, shift, threshold):
    storage = defaultdict(list)
    for raw in _loader(dataset, batch_size):
        batch = _move(raw, device)
        spatial = model.spatial_encoder(
            batch["pixels"], batch["valid_pixels"], batch["extra"],
        )
        structure = model.prepare_structure(spatial, batch["positions"], shift)
        response = structure["shapelet_response"]
        query = query_counterfactuals(model, response)
        shifted_positions = batch["positions"] + shift
        variants = {
            "master": query["master_external_query"],
            "presence": query["presence_external_query"],
            "organization": query["organization_external_query"],
            "full": query["full_external_query"],
        }
        logits = {}
        for name, external in variants.items():
            instance = model.temporal_encoder(
                spatial, shifted_positions, external_query=external,
            )
            logits[name] = model.decoder(instance)
            storage[f"logits_{name}"].append(logits[name].cpu())
            if name == "full":
                storage["instance_feature"].append(instance.cpu())
        for name, value in (
            ("presence", structure["shapelet_presence"]),
            ("organization", structure["shape_organization"]),
            ("shape_response", response),
            ("similarity", structure["shapelet_similarity"]),
            ("state_distribution", structure["state_distribution"]),
        ):
            storage[name].append(value.cpu())
        org_variants = organization_counterfactuals(model, structure)
        for name, details in org_variants.items():
            external = model.shape_response_norm(details["shapelet_response"])
            instance = model.temporal_encoder(
                spatial, shifted_positions, external_query=external,
            )
            storage[f"org_logits_{name}"].append(model.decoder(instance).cpu())
            if name.startswith("roll"):
                storage[f"org_roll_error_{name}"].append(
                    (details["organization"] - structure["shape_organization"])
                    .abs().amax().cpu().reshape(1)
                )
        master = model.temporal_encoder.attention_heads.query.flatten()[None]
        master_norm = master.norm(dim=-1).clamp_min(1e-12)
        full_delta = query["full_delta"]
        storage["query_ratio"].append((full_delta.norm(dim=-1) / master_norm).cpu())
        storage["presence_query_ratio"].append((query["presence_delta"].norm(dim=-1) / master_norm).cpu())
        storage["organization_query_ratio"].append((query["organization_delta"].norm(dim=-1) / master_norm).cpu())
        repeated_master = master.expand(full_delta.shape[0], -1)
        storage["query_cosine"].append(F.cosine_similarity(repeated_master, full_delta, dim=-1).cpu())
        probabilities = logits["full"].softmax(1)
        confidence, pseudo = probabilities.max(1)
        storage["confidence"].append(confidence.cpu())
        storage["pseudo"].append(pseudo.cpu())
        storage["accepted"].append((confidence > threshold).cpu())
        storage["label"].append(batch["label"].cpu())
    return {key: torch.cat(value).numpy() for key, value in storage.items()}


@torch.no_grad()
def _teacher_pseudo(model, dataset, device, batch_size, shift, threshold):
    confidence, pseudo, labels = [], [], []
    for raw in _loader(dataset, batch_size):
        batch = _move(raw, device)
        logits = model.forward_with_temporal_shift(
            batch["pixels"], batch["valid_pixels"], batch["positions"],
            batch["extra"], temporal_shift=shift,
        )
        current_confidence, current_pseudo = logits.softmax(1).max(1)
        confidence.append(current_confidence.cpu()); pseudo.append(current_pseudo.cpu())
        labels.append(batch["label"].cpu())
    confidence = torch.cat(confidence).numpy()
    return {
        "confidence": confidence, "pseudo": torch.cat(pseudo).numpy(),
        "accepted": confidence > threshold, "label": torch.cat(labels).numpy(),
    }


def _checkpoint_shift(stage, packet, cache_path, model, target_train, config, device, batch_size):
    if stage != "source":
        if "global_temporal_shift" not in packet:
            raise ValueError(f"{stage} checkpoint lacks global_temporal_shift")
        return int(round(float(packet["global_temporal_shift"])))
    cache_path = Path(cache_path)
    if cache_path.is_file():
        return int(json.loads(cache_path.read_text(encoding="utf-8"))["shift"])
    from timematch import _initialize_timematch_shift
    shift, _, _ = _initialize_timematch_shift(
        model, _loader(target_train, batch_size), device, config,
    )
    cache_path.write_text(json.dumps({"shift": int(shift)}, indent=2), encoding="utf-8")
    return int(shift)


def _summary_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)), "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def run(args):
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "source": Path(args.source_checkpoint),
        "best": Path(args.uda_best_checkpoint),
        "last": Path(args.uda_last_checkpoint),
    }
    models, teachers = {}, {}
    for stage, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {stage} checkpoint: {path.resolve()}")
        models[stage] = _load_checkpoint(path, device)
        teachers[stage] = (
            models[stage][0] if stage == "source"
            else _load_checkpoint(path, device, teacher=True)[0]
        )
    config = models["source"][1]
    datasets = _datasets(config, args.source, args.target, args.data_root, args.seed)
    probe_rows, probe_class_rows = [], []
    query_rows, query_class_rows, org_rows = [], [], []
    vocab_rows, vocab_class_rows, strength_rows = [], [], []
    shifts = {}
    for stage, (model, stage_config, packet) in models.items():
        shift_config = models["last"][1] if stage == "source" else stage_config
        shift = _checkpoint_shift(
            stage, packet, output / "source_initial_shift.json", model,
            datasets["target_train"], shift_config, device, args.batch_size,
        )
        shifts[stage] = shift
        packed = {
            "source_train": _collect(
                model, datasets["source_train"], device, args.batch_size, 0,
                args.pseudo_threshold,
            ),
            "target_val": _collect(
                model, datasets["target_val"], device, args.batch_size, shift,
                args.pseudo_threshold,
            ),
            "target_test": _collect(
                model, datasets["target_test"], device, args.batch_size, shift,
                args.pseudo_threshold,
            ),
        }
        teacher_target = _teacher_pseudo(
            teachers[stage], datasets["target_test"], device, args.batch_size,
            shift, args.pseudo_threshold,
        )
        for key, value in teacher_target.items():
            packed["target_test"][key] = value
        class_count = len(config.classes)
        for feature in FEATURES:
            source_f1, source_per = _probe(
                packed["source_train"][feature], packed["source_train"]["label"],
                packed["target_test"][feature], packed["target_test"]["label"],
                class_count, args.seed,
            )
            oracle_f1, oracle_per = _probe(
                packed["target_val"][feature], packed["target_val"]["label"],
                packed["target_test"][feature], packed["target_test"]["label"],
                class_count, args.seed,
            )
            probe_rows.append({
                "checkpoint": stage, "feature": feature,
                "source_to_target_macro_f1": source_f1,
                "target_oracle_macro_f1": oracle_f1,
            })
            for class_id, class_name in enumerate(config.classes):
                probe_class_rows.append({
                    "checkpoint": stage, "feature": feature,
                    "class_id": class_id, "class_name": class_name,
                    "source_to_target_f1": source_per[class_id],
                    "target_oracle_f1": oracle_per[class_id],
                })
        target = packed["target_test"]
        for name in QUERY_CASES:
            prediction = target[f"logits_{name}"].argmax(1)
            query_rows.append({
                "checkpoint": stage, "query": name,
                "target_macro_f1": _macro(target["label"], prediction, class_count),
            })
            values = _per_class(target["label"], prediction, class_count)
            for class_id, class_name in enumerate(config.classes):
                query_class_rows.append({
                    "checkpoint": stage, "query": name, "class_id": class_id,
                    "class_name": class_name, "f1": values[class_id],
                })
        for name in ORG_CASES:
            prediction = target[f"org_logits_{name}"].argmax(1)
            row = {
                "checkpoint": stage, "variant": name,
                "target_macro_f1": _macro(target["label"], prediction, class_count),
            }
            if name.startswith("roll"):
                row["max_abs_organization_difference"] = float(
                    target[f"org_roll_error_{name}"].max()
                )
            org_rows.append(row)
        for domain, data in (("source", packed["source_train"]), ("target", target)):
            similarity = data["similarity"]
            top = np.sort(similarity, axis=-1)
            maximum, margin = top[..., -1], top[..., -1] - top[..., -2]
            distribution = np.clip(data["state_distribution"], 1e-12, 1.)
            entropy = -(distribution * np.log(distribution)).sum(-1)
            top_id = similarity.argmax(-1)
            groups = np.full(len(data["label"]), "source", dtype=object)
            if domain == "target":
                groups[:] = "pseudo_rejected"
                accepted = data["accepted"]
                correct = data["pseudo"] == data["label"]
                groups[accepted & correct] = "pseudo_correct"
                groups[accepted & ~correct] = "pseudo_wrong"
            for group in np.unique(groups):
                selected = groups == group
                row = {"checkpoint": stage, "domain": domain, "group": group}
                for metric, values in (("s_max", maximum[selected]), ("margin", margin[selected]), ("entropy", entropy[selected])):
                    for stat, value in _summary_stats(values).items(): row[f"{metric}_{stat}"] = value
                row["top1_anchor_mode"] = int(np.bincount(top_id[selected].reshape(-1), minlength=16).argmax())
                row["top1_anchor_histogram"] = json.dumps(
                    np.bincount(top_id[selected].reshape(-1), minlength=16).tolist()
                )
                vocab_rows.append(row)
            for class_id, class_name in enumerate(config.classes):
                class_groups = list(np.unique(groups))
                if domain == "target":
                    class_groups.append("pseudo_accepted_all")
                for group in class_groups:
                    selected = data["label"] == class_id
                    selected &= (
                        data["accepted"] if group == "pseudo_accepted_all"
                        else groups == group
                    )
                    if not selected.any(): continue
                    row = {
                        "checkpoint": stage, "domain": domain, "group": group,
                        "class_id": class_id, "class_name": class_name,
                        "samples": int(selected.sum()),
                    }
                    for metric, values in (("s_max", maximum[selected]), ("margin", margin[selected]), ("entropy", entropy[selected])):
                        for stat, value in _summary_stats(values).items(): row[f"{metric}_{stat}"] = value
                    row["top1_anchor_mode"] = int(np.bincount(
                        top_id[selected].reshape(-1), minlength=16,
                    ).argmax())
                    row["top1_anchor_histogram"] = json.dumps(
                        np.bincount(top_id[selected].reshape(-1), minlength=16).tolist()
                    )
                    if domain == "target" and group != "pseudo_rejected":
                        row["pseudo_accuracy"] = float(
                            (data["pseudo"][selected] == data["label"][selected]).mean()
                        )
                    vocab_class_rows.append(row)
        for domain, data in (("source", packed["source_train"]), ("target", target)):
            row = {"checkpoint": stage, "domain": domain}
            for name, values in (
                ("query_ratio", data["query_ratio"]),
                ("presence_ratio", data["presence_query_ratio"]),
                ("organization_ratio", data["organization_query_ratio"]),
                ("master_delta_cosine", data["query_cosine"]),
            ):
                for stat, value in _summary_stats(values).items():
                    row[f"{name}_{stat}"] = value
            strength_rows.append(row)
    for name, rows in (
        ("probe_results.csv", probe_rows), ("probe_per_class.csv", probe_class_rows),
        ("query_counterfactual.csv", query_rows),
        ("query_counterfactual_per_class.csv", query_class_rows),
        ("organization_counterfactual.csv", org_rows),
        ("anchor_vocabulary_summary.csv", vocab_rows),
        ("anchor_vocabulary_per_class.csv", vocab_class_rows),
        ("query_strength.csv", strength_rows),
    ):
        _write_csv(output / name, rows)
    (output / "audit_manifest.json").write_text(json.dumps({
        "source": args.source, "target": args.target,
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "temporal_shifts": shifts,
        "source_shift_cache": "source_initial_shift.json",
        "source_shift_label_free": True,
        "target_pseudo_model_role": "teacher",
        "target_test_labels_used_for_training": False,
        "target_labels_used_for_shift": False,
    }, indent=2), encoding="utf-8")
    print(f"STATE_ORG_AUDIT_FINISHED|output={output}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--uda-best-checkpoint", required=True)
    parser.add_argument("--uda-last-checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pseudo-threshold", type=float, default=.9)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
