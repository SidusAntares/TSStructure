#!/usr/bin/env python3
"""Unified read-only audit of Shapelet-TimeMatch transfer checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_transfer_audit import (
    candidate_mask, class_centroid_metrics, deterministic_class_indices,
    domain_gap, effective_rank, forward_prepared_intervention,
    gradient_conflict_metrics, prepare_intervention, pse_domain_gap,
    pse_temporal_metrics, query_geometry, remove_direction,
)


DOMAINS = {
    "AT1": "austria/33UVP/2017", "DK1": "denmark/32VNH/2017",
    "FR1": "france/30TXT/2017", "FR2": "france/31TCJ/2017",
}
TASKS = (("AT1", "DK1"), ("FR1", "FR2"), ("FR2", "DK1"), ("DK1", "AT1"))
QUERY_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)
COMPONENT_MODES = ("FULL", "MORPH", "STATS", "NO_MEAN", "NO_STD", "MEAN_ONLY", "STD_ONLY")
SCALE_MODES = ("FULL", "ONLY_Q8", "ONLY_Q16", "ONLY_Q24", "REMOVE_Q8", "REMOVE_Q16", "REMOVE_Q24", "STRIDE16")
DEFAULT_CHECKPOINT_ROOT = Path("outputs/structure_proto_4tasks_seed1")
DEFAULT_OUTPUT_ROOT = Path("outputs/structure_transfer_audit")


def seed_all(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def checkpoint_paths(root, source, target):
    fold = Path(root) / "uda" / f"{source}_{target}_seed1" / "fold_0"
    return {
        "source": Path(root) / "source" / f"source_{source}_seed1" / "fold_0" / "model.pt",
        "uda_best": fold / "model.pt",
        "uda_last": fold / "checkpoint_last.pt",
    }


def resolve_checkpoints(root, source, target):
    result = checkpoint_paths(root, source, target)
    for stage, path in result.items():
        if not path.is_file():
            print(f"MISSING|task={source}_{target}|stage={stage}|checkpoint={path.resolve()}")
            result[stage] = None
    return result


def load_model(path, device, teacher=False):
    from train import create_model
    packet = torch.load(path, map_location=device, weights_only=False)
    config = packet.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint config missing: {path}")
    config = SimpleNamespace(**config)
    model = create_model(config)
    key = "teacher_state_dict" if teacher and "teacher_state_dict" in packet else "state_dict"
    model.load_state_dict(packet[key], strict=True)
    return model.to(device).eval(), config, packet


def build_datasets(config, source, target, data_root, seed):
    from dataset import PixelSetData
    from train import create_train_val_test_folds
    from transforms import Normalize, RandomSamplePixels, ToTensor
    from torchvision.transforms import transforms

    names = (DOMAINS[source], DOMAINS[target])
    bare = {
        name: PixelSetData(
            data_root, name, config.classes, closed_set=True,
            combine_spring_and_winter=config.combine_spring_and_winter,
        ) for name in names
    }
    eligible = {name: data.get_parcel_indices().tolist() for name, data in bare.items()}
    seed_all(seed)
    split = create_train_val_test_folds(
        list(names), 1, eligible, config.val_ratio, config.test_ratio,
    )[0]
    transform = transforms.Compose([
        RandomSamplePixels(config.num_pixels), Normalize(), ToTensor(),
    ])
    result = {}
    for alias, name in zip((source, target), names):
        for part in ("train", "test"):
            result[(alias, part)] = PixelSetData(
                data_root, name, config.classes, transform=transform,
                indices=split[name][part], closed_set=True,
                combine_spring_and_winter=config.combine_spring_and_winter,
            )
    return result


def loader(dataset, batch_size, seed, indices=None):
    seed_all(seed)
    selected = dataset if indices is None else torch.utils.data.Subset(dataset, indices.tolist())
    return torch.utils.data.DataLoader(selected, batch_size=batch_size, shuffle=False, num_workers=0)


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def metrics_from_logits(logits, labels, class_names):
    logits = torch.cat(logits).numpy() if isinstance(logits, list) else np.asarray(logits)
    labels = torch.cat(labels).numpy() if isinstance(labels, list) else np.asarray(labels)
    prediction = logits.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=np.arange(len(class_names)), zero_division=0,
    )
    return {
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, prediction)), "prediction": prediction,
        "labels": labels, "precision": precision, "recall": recall,
        "per_class_f1": f1, "support": support,
        "confusion": confusion_matrix(labels, prediction, labels=np.arange(len(class_names))),
    }


@torch.no_grad()
def evaluate(model, dataset, device, classes, batch_size, seed, temporal_shift=0):
    logits, labels = [], []
    for batch in loader(dataset, batch_size, seed):
        batch = move_batch(batch, device)
        output = model.forward_with_temporal_shift(
            batch["pixels"], batch["valid_pixels"], batch["positions"], batch["extra"],
            temporal_shift=temporal_shift,
        )
        logits.append(output.cpu()); labels.append(batch["label"].cpu())
    return metrics_from_logits(logits, labels, classes)


@torch.no_grad()
def evaluate_interventions(model, dataset, device, classes, batch_size, seed, family):
    if family == "query":
        variants = [(str(value), {"query_alpha": value}) for value in QUERY_ALPHAS]
    elif family == "component":
        variants = [(value, {"component_mode": value}) for value in COMPONENT_MODES]
    elif family == "scale":
        variants = [(value, {"scale_mode": value}) for value in SCALE_MODES]
    else:
        raise ValueError(family)
    storage = {name: {"logits": [], "shape": [], "response": [], "qshape": [], "candidate_count": []} for name, _ in variants}
    if family == "scale":
        for current in storage.values():
            current.update({"weights": [], "tokens": []})
    labels = []
    for batch in loader(dataset, batch_size, seed):
        batch = move_batch(batch, device)
        prepared = prepare_intervention(model, batch)
        labels.append(batch["label"].cpu())
        for name, kwargs in variants:
            output = forward_prepared_intervention(model, prepared, batch["positions"], **kwargs)
            current = storage[name]
            current["logits"].append(output["logits"].cpu())
            current["shape"].append(output["shape_logits"].cpu())
            current["response"].append(output["shapelet_response"].cpu())
            current["qshape"].append(output["qshape"].cpu())
            current["candidate_count"].append(int(output["candidate_mask"].sum()))
            if family == "scale":
                current["weights"].append(output["candidate_weights"].cpu())
                if name == "FULL":
                    current["tokens"].append(output["shape_tokens"].cpu())
    rows, per_class = [], []
    reference_prediction = metrics_from_logits(storage["1.0"]["logits"], labels, classes)["prediction"] if family == "query" else None
    for name, _ in variants:
        main = metrics_from_logits(storage[name]["logits"], labels, classes)
        shape = metrics_from_logits(storage[name]["shape"], labels, classes)
        response = torch.cat(storage[name]["response"])
        qshape = torch.cat(storage[name]["qshape"])
        row = {
            "variant": name, "target_macro_f1": main["macro_f1"],
            "accuracy": main["accuracy"], "shape_head_macro_f1": shape["macro_f1"],
            "response_effective_rank": effective_rank(response),
            "qshape_norm": float(qshape.norm(dim=-1).mean()),
            "candidate_count": storage[name]["candidate_count"][0],
        }
        if reference_prediction is not None:
            row["prediction_disagreement_vs_alpha1"] = float(np.mean(main["prediction"] != reference_prediction))
        rows.append(row)
        for class_id, class_name in enumerate(classes):
            per_class.append({
                "variant": name, "class_id": class_id, "class_name": class_name,
                "f1": float(main["per_class_f1"][class_id]),
                "support": int(main["support"][class_id]),
            })
    return rows, per_class, storage


@torch.no_grad()
def pseudo_audit(stage, model, packet, dataset, device, classes, threshold, batch_size, seed):
    shift = packet.get("global_temporal_shift", 0)
    logits, labels = [], []
    for batch in loader(dataset, batch_size, seed):
        batch = move_batch(batch, device)
        output = model.forward_with_temporal_shift(
            batch["pixels"], batch["valid_pixels"], batch["positions"], batch["extra"],
            temporal_shift=shift,
        )
        logits.append(output.cpu()); labels.append(batch["label"].cpu())
    probability = torch.cat(logits).softmax(1)
    confidence, prediction = probability.max(1)
    truth, accepted = torch.cat(labels), confidence >= float(threshold)
    rows = []
    overall_accuracy = float((prediction[accepted] == truth[accepted]).float().mean()) if accepted.any() else float("nan")
    rows.append({
        "stage": stage, "scope": "overall", "class_id": "ALL", "class_name": "ALL",
        "num_samples": len(truth), "accepted_count": int(accepted.sum()),
        "coverage": float(accepted.float().mean()), "accepted_accuracy": overall_accuracy,
    })
    for class_id, class_name in enumerate(classes):
        true_mask = truth == class_id; pseudo_mask = prediction == class_id
        true_accepted = true_mask & accepted; pseudo_accepted = pseudo_mask & accepted
        rows.append({
            "stage": stage, "scope": "true_class", "class_id": class_id,
            "class_name": class_name, "num_samples": int(true_mask.sum()),
            "accepted_count": int(true_accepted.sum()),
            "coverage": float(true_accepted.sum() / true_mask.sum().clamp_min(1)),
            "accepted_accuracy": float((prediction[true_accepted] == truth[true_accepted]).float().mean()) if true_accepted.any() else float("nan"),
        })
        rows.append({
            "stage": stage, "scope": "pseudo_class", "class_id": class_id,
            "class_name": class_name, "num_samples": int(pseudo_mask.sum()),
            "accepted_count": int(pseudo_accepted.sum()), "coverage": float("nan"),
            "accepted_accuracy": float((truth[pseudo_accepted] == class_id).float().mean()) if pseudo_accepted.any() else float("nan"),
        })
    cm = confusion_matrix(truth[accepted], prediction[accepted], labels=np.arange(len(classes)))
    top_errors = []
    for class_id, class_name in enumerate(classes):
        counts = cm[class_id].copy(); counts[class_id] = 0
        for rank, wrong in enumerate(np.argsort(-counts)[:3], 1):
            top_errors.append({"stage": stage, "true_class": class_name, "rank": rank, "predicted_class": classes[wrong], "count": int(counts[wrong])})
    return rows, cm, top_errors, {
        "truth": truth, "prediction": prediction, "accepted": accepted,
        "confidence": confidence,
    }


@torch.no_grad()
def collect_fixed(model, dataset, indices, device, batch_size, seed):
    store = defaultdict(list); labels, positions = [], []
    for batch in loader(dataset, batch_size, seed, indices):
        batch = move_batch(batch, device)
        prepared = prepare_intervention(model, batch)
        output = forward_prepared_intervention(model, prepared, batch["positions"])
        components = output["component_features"]
        store["mean"].append(components["mean"].mean(1).cpu())
        store["std"].append(components["std"].mean(1).cpu())
        store["morphology"].append(torch.cat((components["raw"], components["diff"]), -1).mean(1).cpu())
        store["instance"].append(output["instance_feature"].cpu())
        store["response"].append(output["shapelet_response"].cpu())
        store["invariant"].append(output["shape_invariant_feature"].cpu())
        store["pse"].append(output["pse_feature"].cpu())
        labels.append(batch["label"].cpu()); positions.append(batch["positions"].cpu())
    packed = {key: torch.cat(value) for key, value in store.items()}
    packed["mean_std"] = torch.cat((packed["mean"], packed["std"]), -1)
    packed["labels"] = torch.cat(labels); packed["positions"] = torch.cat(positions)
    return packed


def linear_probe(features, labels, seed, domain_labels=None):
    x, y = np.asarray(features, dtype=np.float32), np.asarray(labels)
    train, test = train_test_split(
        np.arange(len(y)), test_size=.3, random_state=int(seed), stratify=y,
    )
    classifier = LogisticRegression(max_iter=500, random_state=int(seed), n_jobs=1)
    classifier.fit(x[train], y[train]); prediction = classifier.predict(x[test])
    return float(accuracy_score(y[test], prediction)), float(f1_score(y[test], prediction, average="macro", zero_division=0))


def component_probe_rows(task, source_data, target_data, seed):
    rows = []
    for name in ("mean", "std", "mean_std", "morphology"):
        source, target = source_data[name].numpy(), target_data[name].numpy()
        values = np.concatenate((source, target)); domain = np.concatenate((np.zeros(len(source)), np.ones(len(target))))
        labels = np.concatenate((source_data["labels"].numpy(), target_data["labels"].numpy()))
        domain_accuracy, _ = linear_probe(values, domain, seed)
        _, class_f1 = linear_probe(values, labels, seed)
        rows.append({"task": task, "representation": name, "domain_accuracy": domain_accuracy, "class_macro_f1": class_f1})
    return rows


def representation_rows(task, stages):
    """Summarize START/END alignment while retaining PSE time bins."""
    rows = []
    for stage, (source_data, target_data) in stages.items():
        for layer in ("instance", "response"):
            source_metrics = class_centroid_metrics(source_data[layer], source_data["labels"])
            target_metrics = class_centroid_metrics(target_data[layer], target_data["labels"])
            rows.append({
                "task": task, "stage": stage, "layer": layer,
                "domain_gap": domain_gap(source_data[layer], source_data["labels"], target_data[layer], target_data["labels"]),
                "inter_sep": np.nanmean([source_metrics["inter_sep"], target_metrics["inter_sep"]]),
                "intra_disp": np.nanmean([source_metrics["intra_disp"], target_metrics["intra_disp"]]),
            })
        source_pse = pse_temporal_metrics(source_data["pse"], source_data["positions"], source_data["labels"])
        target_pse = pse_temporal_metrics(target_data["pse"], target_data["positions"], target_data["labels"])
        rows.append({
            "task": task, "stage": stage, "layer": "pse_temporal_24bin",
            "domain_gap": pse_domain_gap(source_pse, target_pse),
            "inter_sep": np.nanmean([source_pse["inter_sep"], target_pse["inter_sep"]]),
            "intra_disp": np.nanmean([source_pse["intra_disp"], target_pse["intra_disp"]]),
        })
    keyed = {(row["stage"], row["layer"]): row for row in rows}
    for layer in ("instance", "response", "pse_temporal_24bin"):
        start, end = keyed[("START", layer)], keyed[("END", layer)]
        end["alignment_gain"] = 1.0 - end["domain_gap"] / max(start["domain_gap"], 1e-12)
        end["separation_retention"] = end["inter_sep"] / max(start["inter_sep"], 1e-12)
        start["alignment_gain"] = float("nan"); start["separation_retention"] = float("nan")
    return rows


def direction_rows(task, source_data, target_data, pseudo_packet, seed):
    source, target = source_data["mean_std"], target_data["mean_std"]
    source_labels, target_labels = source_data["labels"], target_data["labels"]
    pseudo, accepted = pseudo_packet["prediction"], pseudo_packet["accepted"]
    if len(pseudo) != len(target):
        raise ValueError("pseudo packet must be aligned to the fixed target subset")
    pseudo_available = True
    common = sorted(set(source_labels.tolist()) & set(target_labels.tolist()))
    oracle, pseudo_delta, per_class = [], [], []
    for class_id in common:
        source_mean = source[source_labels == class_id].mean(0)
        oracle_delta = target[target_labels == class_id].mean(0) - source_mean
        selected = (pseudo == class_id) & accepted
        current_pseudo = target[selected].mean(0) - source_mean if selected.any() else torch.full_like(source_mean, float("nan"))
        oracle.append(oracle_delta); pseudo_delta.append(current_pseudo)
        per_class.append({
            "task": task, "scope": "class", "class_id": class_id,
            "oracle_pseudo_cosine": float(F.cosine_similarity(oracle_delta[None], current_pseudo[None])) if torch.isfinite(current_pseudo).all() else float("nan"),
            "pseudo_subset_available": pseudo_available,
        })
    matrix = torch.stack(oracle)
    singular = torch.linalg.svdvals(matrix); ratio = singular.square() / singular.square().sum().clamp_min(1e-12)
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    valid_pseudo = [value for value in pseudo_delta if torch.isfinite(value).all()]
    pseudo_vh = None
    if valid_pseudo:
        _, _, pseudo_vh = torch.linalg.svd(torch.stack(valid_pseudo), full_matrices=False)
    def pairwise(values):
        if len(values) < 2:
            return float("nan")
        normalized = F.normalize(torch.stack(values), dim=-1)
        matrix_value = normalized @ normalized.T
        mask = ~torch.eye(len(values), dtype=torch.bool)
        return float(matrix_value[mask].mean()) if mask.any() else float("nan")
    pairwise_oracle = pairwise(oracle)
    pairwise_pseudo = pairwise(valid_pseudo)
    all_values = torch.cat((source, target)); domain = np.concatenate((np.zeros(len(source)), np.ones(len(target))))
    labels = torch.cat((source_labels, target_labels)).numpy()
    directions = [("oracle_top1", vh[0])]
    if pseudo_vh is not None:
        directions.append(("pseudo_top1", pseudo_vh[0]))
    for direction_name, direction in directions:
        cleaned = remove_direction(all_values, direction).numpy()
        domain_accuracy, _ = linear_probe(cleaned, domain, seed)
        _, class_f1 = linear_probe(cleaned, labels, seed)
        per_class.append({
            "task": task, "scope": direction_name, "class_id": "ALL",
            "top1_explained": float(ratio[0]), "top2_cumulative": float(ratio[:2].sum()),
            "top3_cumulative": float(ratio[:3].sum()), "domain_accuracy_after": domain_accuracy,
            "class_macro_f1_after": class_f1, "pseudo_subset_available": pseudo_available,
            "pairwise_oracle_delta_cosine": pairwise_oracle,
            "pairwise_pseudo_delta_cosine": pairwise_pseudo,
        })
    return per_class


def source_shape_discrimination(task, model, data, classes):
    response, labels = data["invariant"], data["labels"]
    logits = model.shape_classifier(
        response.to(next(model.parameters()).device)
    ).detach().cpu()
    prediction = logits.argmax(1); cm = confusion_matrix(labels, prediction, labels=np.arange(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(labels, prediction, labels=np.arange(len(classes)), zero_division=0)
    rows = []
    normalized = F.normalize(response.float(), dim=-1)
    centroids = {class_id: F.normalize(normalized[labels == class_id].mean(0), dim=0) for class_id in range(len(classes)) if (labels == class_id).any()}
    for class_id, class_name in enumerate(classes):
        selected = labels == class_id
        margin = logits[selected, class_id] - logits[selected].masked_fill(F.one_hot(torch.full((int(selected.sum()),), class_id), len(classes)).bool(), -torch.inf).max(1).values if selected.any() else torch.tensor([])
        competitors = [(other, float(centroids[class_id] @ value)) for other, value in centroids.items() if other != class_id] if class_id in centroids else []
        nearest = max(competitors, key=lambda value: value[1]) if competitors else (-1, float("nan"))
        rows.append({
            "task": task, "class_id": class_id, "class_name": class_name,
            "precision": float(precision[class_id]), "recall": float(recall[class_id]), "f1": float(f1[class_id]), "support": int(support[class_id]),
            "within_response_variance": float(normalized[selected].var(0, unbiased=False).mean()) if selected.any() else float("nan"),
            "mean_margin": float(margin.mean()) if margin.numel() else float("nan"),
            "p10_margin": float(torch.quantile(margin, .1)) if margin.numel() else float("nan"),
            "nearest_competing_class": classes[nearest[0]] if nearest[0] >= 0 else "NA", "nearest_competing_cosine": nearest[1],
        })
    return rows, cm


def gradient_rows(task, model, source_batch, target_batch, pseudo, pseudo_mask, config, device):
    from utils.focal_loss import FocalLoss
    groups = {
        "PSE": tuple(model.spatial_encoder.parameters()),
        "ShapeTokenGenerator": tuple(model.structure_branch.token_generator.parameters()),
        "ShapeletDictionary": tuple(model.structure_branch.shapelet_dictionary.parameters()),
        "response_to_query": tuple(model.structure_branch.response_to_query.parameters()),
        "query_projection": tuple(model.temporal_encoder.attention_heads.external_query_projection.parameters()),
    }
    criterion = FocalLoss(gamma=config.focal_loss_gamma)
    rows = []
    source_batch = move_batch(source_batch, device)
    source_output = model(
        source_batch["pixels"], source_batch["valid_pixels"], source_batch["positions"], source_batch["extra"], return_dict=True,
    )
    source_main = criterion(source_output["logits"], source_batch["label"])
    source_shape = float(config.shape_class_weight) * criterion(source_output["shape_logits"], source_batch["label"])
    for module, values in gradient_conflict_metrics(source_main, source_shape, groups).items():
        rows.append({"task": task, "domain": "source", "module": module, **values})
    if int(pseudo_mask.sum()) >= 2:
        target_batch = move_batch(target_batch, device); selected = pseudo_mask.to(device)
        target_output = model(
            target_batch["pixels"][selected], target_batch["valid_pixels"][selected], target_batch["positions"][selected], target_batch["extra"][selected], return_dict=True,
        )
        target_main = float(config.trade_off) * criterion(target_output["logits"], pseudo.to(device)[selected])
        target_shape = float(config.shape_class_weight * config.trade_off) * criterion(target_output["shape_logits"], pseudo.to(device)[selected])
        for module, values in gradient_conflict_metrics(target_main, target_shape, groups).items():
            rows.append({"task": task, "domain": "target", "module": module, **values})
    else:
        for module in groups:
            rows.append({"task": task, "domain": "target", "module": module, "gradient_cosine": float("nan"), "main_grad_norm": float("nan"), "shape_grad_norm": float("nan"), "shape_main_norm_ratio": float("nan")})
    return rows


def find_original_baseline(outputs_root, source, target):
    expected = {f"timematch_{source}_{target}_seed1", f"timematch_original_{source}_{target}_seed1"}
    matches = []
    for path in Path(outputs_root).rglob("test_metrics_*.json"):
        if any(parent.name in expected for parent in path.parents):
            try:
                matches.append((path, float(json.loads(path.read_text())["macro_f1"])))
            except (KeyError, ValueError, json.JSONDecodeError):
                pass
    unique = {value for _, value in matches}
    return unique.pop() if len(unique) == 1 else float("nan")


def report(path, task_rows, pseudo_rows, query_rows, component_rows, probe_rows, direction_rows_all, scale_rows, shape_rows, gradient_rows_all, blockers):
    def any_true(predicate, rows): return any(predicate(row) for row in rows if all(not (isinstance(value, float) and math.isnan(value)) for value in row.values()))
    flags = {
        "pseudo_issue": any_true(lambda row: row.get("scope") == "overall" and row.get("coverage", 0) > .5 and row.get("accepted_accuracy", 1) < .9, pseudo_rows),
        "query_over_injection": False,
        "stats_domain_specific": False,
        "domain_direction_low_rank": any_true(lambda row: row.get("top2_cumulative", 0) > .7, direction_rows_all),
        "q24_redundant": False, "stride16_safe": False,
        "shape_capacity_issue": False,
        "gradient_conflict": any_true(lambda row: row.get("gradient_cosine", 0) < -.2, gradient_rows_all),
    }
    by_task_query = defaultdict(dict)
    for row in query_rows: by_task_query[row["task"]][row["variant"]] = row["target_macro_f1"]
    flags["query_over_injection"] = any(max(values.get("0.25", -1), values.get("0.5", -1)) > values.get("1.0", 1) + .005 for values in by_task_query.values())
    by_task_probe = defaultdict(dict)
    for row in probe_rows: by_task_probe[row["task"]][row["representation"]] = row["domain_accuracy"]
    flags["stats_domain_specific"] = any(values.get("mean_std", 0) > values.get("morphology", 1) + .1 for values in by_task_probe.values())
    by_task_scale = defaultdict(dict)
    for row in scale_rows: by_task_scale[row["task"]][row["variant"]] = row["target_macro_f1"]
    complete = len(by_task_scale) == len(TASKS)
    flags["q24_redundant"] = complete and all(value.get("REMOVE_Q24", -1) >= value.get("FULL", 1) - .005 for value in by_task_scale.values())
    flags["stride16_safe"] = complete and all(value.get("STRIDE16", -1) >= value.get("FULL", 1) - .005 for value in by_task_scale.values())
    shape_by_task = defaultdict(list)
    for row in shape_rows: shape_by_task[row["task"]].append(row["f1"])
    rank_by_task = {row["task"]: row["response_effective_rank"] for row in component_rows if row["variant"] == "FULL"}
    flags["shape_capacity_issue"] = any(np.mean(values) < .5 and rank_by_task.get(task, 0) > 2 for task, values in shape_by_task.items()) and not flags["gradient_conflict"]
    lines = ["# Structure Transfer Audit", "", "## 1. Final Performance", ""]
    for row in task_rows: lines.append(f"- {row['task']}: target F1={row.get('uda_test_macro_f1', float('nan')):.4f}, TimeMatch={row.get('original_timematch_seed1', float('nan')):.4f}, delta={row.get('delta_vs_timematch', float('nan')):.4f}")
    sections = (("2. Pseudo Quality", pseudo_rows), ("3. Query Injection", query_rows), ("4. Shape Components", component_rows), ("5. Mean/Std Domain Direction", direction_rows_all), ("6. Multi-scale / Window", scale_rows), ("7. Shape Discrimination", shape_rows), ("8. Gradient Conflict", gradient_rows_all))
    for title, rows in sections:
        lines.extend(("", f"## {title}", "", f"Rows: {len(rows)}"))
    lines.extend(("", "## Flags", ""))
    lines.extend(f"- {key}: {str(value).lower()}" for key, value in flags.items())
    if blockers:
        lines.extend(("", "## Blockers", "")); lines.extend(f"- {value}" for value in blockers)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return flags


def run(args):
    device, output = torch.device(args.device), Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = defaultdict(list); blockers = []
    for source, target in TASKS:
        task = f"{source}_{target}"
        paths = resolve_checkpoints(args.checkpoint_root, source, target)
        if any(path is None for path in paths.values()):
            blockers.append(f"{task}: checkpoint unavailable")
            continue
        print(f"AUDIT_START|task={task}")
        source_model, config, source_packet = load_model(paths["source"], device)
        best_model, best_config, best_packet = load_model(paths["uda_best"], device)
        last_teacher, _, last_packet = load_model(paths["uda_last"], device, teacher=True)
        best_teacher, _, _ = load_model(paths["uda_best"], device, teacher=True)
        datasets = build_datasets(config, source, target, args.data_root, args.seed)
        source_start = evaluate(source_model, datasets[(source, "test")], device, config.classes, args.batch_size, args.seed)
        target_start = evaluate(source_model, datasets[(target, "test")], device, config.classes, args.batch_size, args.seed)
        source_end = evaluate(best_model, datasets[(source, "test")], device, config.classes, args.batch_size, args.seed)
        target_end = evaluate(best_model, datasets[(target, "test")], device, config.classes, args.batch_size, args.seed)
        baseline = find_original_baseline(args.baseline_root, source, target)
        final = {
            "task": task, "source_test_macro_f1": source_end["macro_f1"],
            "source_f1_start": source_start["macro_f1"], "source_f1_end": source_end["macro_f1"],
            "source_retention": source_end["macro_f1"] - source_start["macro_f1"],
            "target_f1_start": target_start["macro_f1"], "uda_test_macro_f1": target_end["macro_f1"],
            "uda_gain": target_end["macro_f1"] - target_start["macro_f1"],
            "uda_best_val_epoch": int(best_packet.get("epoch", -1)) + 1,
            "uda_best_val_macro_f1": float(best_packet.get("best_f1", float("nan"))),
            "original_timematch_seed1": baseline,
            "delta_vs_timematch": target_end["macro_f1"] - baseline,
        }
        all_rows["final"].append(final)
        for class_id, class_name in enumerate(config.classes):
            all_rows["per_class"].append({
                "task": task, "class": class_name, "precision": target_end["precision"][class_id],
                "recall": target_end["recall"][class_id], "f1": target_end["per_class_f1"][class_id],
                "support": int(target_end["support"][class_id]),
            })
        threshold = float(getattr(config, "pseudo_threshold", .9))
        task_confusions = []
        for stage, model, packet in (("source", source_model, source_packet), ("uda_best", best_teacher, best_packet), ("uda_last", last_teacher, last_packet)):
            rows, cm, errors, packet_values = pseudo_audit(stage, model, packet, datasets[(target, "train")], device, config.classes, threshold, args.batch_size, args.seed)
            for row in rows: row["task"] = task
            for row in errors: row["task"] = task
            all_rows["pseudo"].extend(rows); all_rows["pseudo_errors"].extend(errors)
            task_confusions.extend([
                dict(stage=stage, true_class=config.classes[i], **{config.classes[j]: int(cm[i, j]) for j in range(len(config.classes))})
                for i in range(len(config.classes))
            ])
            if stage == "uda_best": best_pseudo = packet_values
        write_csv(output / f"pseudo_confusion_{task}.csv", task_confusions)
        query, query_per_class, query_storage = evaluate_interventions(best_model, datasets[(target, "test")], device, config.classes, args.batch_size, args.seed, "query")
        qmaster = best_model.temporal_encoder.attention_heads.query.detach()
        full_qshape = torch.cat(query_storage["1.0"]["qshape"])
        geometry = query_geometry(qmaster, full_qshape)
        for row in query:
            row.update({"task": task, **geometry})
        for row in query_per_class: row["task"] = task
        del query_storage, full_qshape
        component, component_per_class, _ = evaluate_interventions(best_model, datasets[(target, "test")], device, config.classes, args.batch_size, args.seed, "component")
        scale, scale_per_class, scale_storage = evaluate_interventions(best_model, datasets[(target, "test")], device, config.classes, args.batch_size, args.seed, "scale")
        for row in component + scale: row["task"] = task
        for row in component_per_class + scale_per_class: row["task"] = task
        all_rows["query"].extend(query); all_rows["query_per_class"].extend(query_per_class)
        all_rows["component"].extend(component); all_rows["component_per_class"].extend(component_per_class)
        all_rows["scale"].extend(scale)
        weights = torch.cat(scale_storage["FULL"]["weights"])
        full_tokens = torch.cat(scale_storage["FULL"]["tokens"])
        scales = best_model.structure_branch.window_extractor.scales
        scale_ids = torch.cat([torch.full((best_model.structure_branch.window_extractor.grid_points // best_model.structure_branch.window_extractor.stride,), value) for value in scales])
        for anchor in range(weights.shape[-1]):
            mass = weights[:, :, anchor]
            row = {"task": task, "anchor": anchor, "candidate_effective_number": float(torch.exp(-(mass.clamp_min(1e-12) * mass.clamp_min(1e-12).log()).sum(1)).mean())}
            for value in scales: row[f"q{value}_mass"] = float(mass[:, scale_ids == value].sum(1).mean())
            all_rows["scale_mass"].append(row)
        offset = 0
        adjacent_by_scale = {}
        candidates_per_scale = best_model.structure_branch.window_extractor.grid_points // best_model.structure_branch.window_extractor.stride
        normalized_tokens = F.normalize(full_tokens.float(), dim=-1)
        for value in scales:
            current = normalized_tokens[:, offset:offset + candidates_per_scale]
            adjacent_by_scale[value] = float((current * current.roll(-1, dims=1)).sum(-1).mean())
            offset += candidates_per_scale
        for row in all_rows["scale"][-len(SCALE_MODES):]:
            if row["variant"] == "FULL":
                for value, cosine in adjacent_by_scale.items():
                    row[f"adjacent_token_cosine_q{value}"] = cosine
        source_indices = deterministic_class_indices(datasets[(source, "train")].get_labels(), args.samples_per_class, args.seed)
        target_indices = deterministic_class_indices(datasets[(target, "train")].get_labels(), args.samples_per_class, args.seed)
        source_fixed_start = collect_fixed(source_model, datasets[(source, "train")], source_indices, device, args.batch_size, args.seed)
        target_fixed_start = collect_fixed(source_model, datasets[(target, "train")], target_indices, device, args.batch_size, args.seed)
        source_fixed = collect_fixed(best_model, datasets[(source, "train")], source_indices, device, args.batch_size, args.seed)
        target_fixed = collect_fixed(best_model, datasets[(target, "train")], target_indices, device, args.batch_size, args.seed)
        all_rows["representation"].extend(representation_rows(task, {
            "START": (source_fixed_start, target_fixed_start),
            "END": (source_fixed, target_fixed),
        }))
        all_rows["probe"].extend(component_probe_rows(task, source_fixed, target_fixed, args.seed))
        fixed_pseudo = {
            "prediction": best_pseudo["prediction"][torch.as_tensor(target_indices)],
            "accepted": best_pseudo["accepted"][torch.as_tensor(target_indices)],
        }
        all_rows["direction"].extend(direction_rows(task, source_fixed, target_fixed, fixed_pseudo, args.seed))
        shape_rows, shape_cm = source_shape_discrimination(task, best_model, source_fixed, config.classes)
        all_rows["shape"].extend(shape_rows)
        write_csv(output / f"shape_confusion_{task}.csv", [dict(true_class=config.classes[i], **{config.classes[j]: int(shape_cm[i, j]) for j in range(len(config.classes))}) for i in range(len(config.classes))])
        source_batch = next(iter(loader(datasets[(source, "train")], args.gradient_batch_size, args.seed)))
        target_batch = next(iter(loader(datasets[(target, "train")], args.gradient_batch_size, args.seed)))
        target_device = move_batch(target_batch, device)
        with torch.no_grad():
            teacher_output = best_teacher.forward_with_temporal_shift(target_device["pixels"], target_device["valid_pixels"], target_device["positions"], target_device["extra"], temporal_shift=best_packet.get("global_temporal_shift", 0))
            confidence, pseudo = teacher_output.softmax(1).max(1); pseudo_mask = confidence >= threshold
        seed_all(args.seed)
        best_model.eval()
        all_rows["gradient"].extend(gradient_rows(task, best_model, source_batch, target_batch, pseudo.cpu(), pseudo_mask.cpu(), best_config, device))
        print(f"AUDIT_FINISHED|task={task}|target_macro_f1={target_end['macro_f1']:.6f}")
    write_csv(output / "final_results.csv", all_rows["final"])
    write_csv(output / "per_class_test.csv", all_rows["per_class"])
    write_csv(output / "pseudo_audit.csv", all_rows["pseudo"])
    write_csv(output / "pseudo_top_errors.csv", all_rows["pseudo_errors"])
    write_csv(output / "query_audit.csv", all_rows["query"])
    write_csv(output / "query_per_class.csv", all_rows["query_per_class"])
    write_csv(output / "component_ablation.csv", all_rows["component"] + all_rows["component_per_class"])
    write_csv(output / "component_probe.csv", all_rows["probe"])
    write_csv(output / "representation_alignment.csv", all_rows["representation"])
    write_csv(output / "domain_direction_audit.csv", all_rows["direction"])
    write_csv(output / "scale_window_audit.csv", all_rows["scale"])
    write_csv(output / "scale_mass.csv", all_rows["scale_mass"])
    write_csv(output / "shape_discrimination.csv", all_rows["shape"])
    write_csv(output / "gradient_conflict.csv", all_rows["gradient"])
    flags = report(output / "report.md", all_rows["final"], all_rows["pseudo"], all_rows["query"], all_rows["component"], all_rows["probe"], all_rows["direction"], all_rows["scale"], all_rows["shape"], all_rows["gradient"], blockers)
    print("AUDIT_FLAGS|" + "|".join(f"{key}={str(value).lower()}" for key, value in flags.items()))
    if blockers: print("AUDIT_BLOCKERS|" + "|".join(blockers))


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    result.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    result.add_argument("--baseline-root", default="outputs")
    result.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--gradient-batch-size", type=int, default=32)
    result.add_argument("--samples-per-class", type=int, default=128)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
