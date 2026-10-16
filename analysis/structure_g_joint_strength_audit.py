#!/usr/bin/env python3
"""Frozen class-conditional joint-Strength audit for final G checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from analysis.structure_semantic_features import (
    _extract_split,
    build_abc_datasets,
    load_feature_split,
)
from analysis.structure_pair_semantics_audit import load_generic_frozen, run_extract
from analysis.diagnose_eg_pair import (
    TASKS as LEGACY_EG_TASKS,
    require_complete_checkpoint, sha256_file, validate_uda_checkpoint,
)


TASKS = {
    "AT1_DK1": ("AT1", "austria/33UVP/2017", "DK1", "denmark/32VNH/2017"),
    "FR1_FR2": ("FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
    "FR2_DK1": ("FR2", "france/31TCJ/2017", "DK1", "denmark/32VNH/2017"),
    "DK1_AT1": ("DK1", "denmark/32VNH/2017", "AT1", "austria/33UVP/2017"),
}
READINESS_FIELDS = (
    "source_joint_supported", "transfer_support", "compactness_justified",
    "shape_ce_redundancy_risk", "recommended_next_step",
)


def strength16(response):
    response = np.asarray(response)
    if response.ndim != 2 or response.shape[1] != 96:
        raise ValueError(f"Response96 required, got {response.shape}")
    return response[:, :16]


def assert_cache_boundary(source_train, source_val, target_train, target_val):
    if "label" in target_train:
        raise ValueError("Target Train labels are forbidden")
    for name, values in (("source_train", source_train), ("source_val", source_val),
                         ("target_val", target_val)):
        if "label" not in values:
            raise ValueError(f"{name} labels are required for frozen evaluation")
    left = set(map(int, source_train["parcel_index"]))
    right = set(map(int, source_val["parcel_index"]))
    if left & right:
        raise ValueError("Source Train/Val sample leakage")
    return True


def assert_anchor_scope(left, right):
    if str(left) != str(right):
        raise ValueError("Anchor IDs cannot be compared across checkpoint scopes")
    return True


def validate_g_checkpoint_config(config, epoch):
    required = {
        "shape_representation": "phase_moment", "phase_response_mode": "softmax",
        "shapelet_count": 16, "target_structure_detach_epochs": 5,
        "shape_phase_k2_enabled": True, "seed": 1, "epochs": 20,
    }
    historical_defaults = {
        "phase_response_mode": "softmax",
        "shape_phase_k2_enabled": True,
    }
    mismatches = {
        key: (config.get(key), value) for key, value in required.items()
        if config.get(key, historical_defaults.get(key)) != value
    }
    if int(epoch) != 19:
        mismatches["checkpoint_epoch"] = (epoch, 19)
    if mismatches:
        raise ValueError(f"checkpoint is not final original G: {mismatches}")
    return True


def validate_response96_state(state):
    weight = state.get("shape_classifier.weight")
    if weight is None or tuple(weight.shape)[-1:] != (96,):
        shape = None if weight is None else tuple(weight.shape)
        raise ValueError(f"G audit requires a Response96 shape classifier, got {shape}")
    return True


def validate_four_task_g_checkpoint(packet, task):
    """Reuse the strict E/G validator without inheriting its two-task registry limit."""
    if task not in TASKS:
        raise ValueError(f"unsupported G audit task: {task}")
    raw = packet.get("config", {})
    config = dict(raw if isinstance(raw, dict) else vars(raw))
    expected_source, expected_target = TASKS[task][1], TASKS[task][3]
    if config.get("source") != expected_source or config.get("target") != expected_target:
        raise ValueError(
            f"checkpoint task/domain mismatch: expected={expected_source}->{expected_target}, "
            f"actual={config.get('source')}->{config.get('target')}"
        )
    if task in LEGACY_EG_TASKS:
        return validate_uda_checkpoint(packet, task, "G")

    # The shared validator's protocol is task-invariant except for source/target,
    # but its historical registry contains only FR1_FR2 and FR2_DK1. Validate all
    # invariant fields through an isolated canonical packet, then restore the
    # already-validated real domains in the returned manifest.
    canonical_task = "FR1_FR2"
    canonical = dict(config)
    canonical["source"] = LEGACY_EG_TASKS[canonical_task][1]
    canonical["target"] = LEGACY_EG_TASKS[canonical_task][3]
    proxy = dict(packet)
    proxy["config"] = canonical
    validated = dict(validate_uda_checkpoint(proxy, canonical_task, "G"))
    validated.update(source=expected_source, target=expected_target)
    return validated


def _pipeline(alpha=1.):
    return make_pipeline(StandardScaler(), RidgeClassifier(alpha=float(alpha)))


def _cv(labels, seed=1):
    counts = np.bincount(np.asarray(labels, dtype=np.int64))
    positive = counts[counts > 0]
    folds = min(5, int(positive.min())) if len(positive) else 0
    if folds < 2:
        return None
    return StratifiedKFold(folds, shuffle=True, random_state=int(seed))


def select_best_single_anchor(train, labels, random_state=1,
                              forbidden_validation_features=None):
    del forbidden_validation_features
    train, labels = np.asarray(train), np.asarray(labels)
    splitter = _cv(labels, random_state)
    scores = []
    for anchor in range(train.shape[1]):
        score = float(np.mean(cross_val_score(
            _pipeline(), train[:, [anchor]], labels, cv=splitter,
            scoring="f1_macro",
        ))) if splitter is not None else np.nan
        scores.append(score)
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    return {
        "anchor_id": int(np.argmax(finite)), "cv_macro_f1": float(np.max(finite)),
        "selection_split": "source_train_cv", "all_anchor_cv_macro_f1": scores,
    }


def fit_joint_reference(values, labels, class_count, radius_quantile=.9):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    global_center = np.median(values, axis=0)
    mad = np.median(np.abs(values - global_center), axis=0) * 1.4826
    fallback = np.std(values, axis=0)
    scale = np.where(mad > 1e-6, mad, np.maximum(fallback, 1e-6))
    centers = np.full((class_count, values.shape[1]), np.nan)
    radius = np.full(class_count, np.nan)
    support = np.zeros(class_count, dtype=np.int64)
    supported = np.zeros(class_count, dtype=bool)
    for class_id in range(class_count):
        selected = values[labels == class_id]
        support[class_id] = len(selected)
        if not len(selected):
            continue
        supported[class_id] = True
        centers[class_id] = np.median(selected, axis=0)
        distance = np.sqrt(np.mean(((selected - centers[class_id]) / scale) ** 2, axis=1))
        radius[class_id] = np.quantile(distance, radius_quantile)
    return {
        "global_center": global_center, "scale": scale, "centers": centers,
        "radius": radius, "support": support, "supported": supported,
        "radius_quantile": float(radius_quantile), "fit_split": "source_train",
    }


def class_distances(values, reference):
    values = np.asarray(values, dtype=np.float64)
    centers, scale = reference["centers"], reference["scale"]
    distances = np.sqrt(np.mean(
        ((values[:, None, :] - centers[None, :, :]) / scale[None, None, :]) ** 2,
        axis=2,
    ))
    distances[:, ~reference["supported"]] = np.inf
    return distances


def select_prototype_count(values, labels, candidates=(1, 2), seed=1):
    values, labels = np.asarray(values), np.asarray(labels)
    splitter = _cv(labels, seed)
    scores = {}
    for count in candidates:
        fold_scores = []
        if splitter is not None:
            for train_index, val_index in splitter.split(values, labels):
                packet = fit_prototypes(values[train_index], labels[train_index],
                                        int(labels.max()) + 1, count, seed)
                prediction = predict_prototypes(values[val_index], packet)
                fold_scores.append(f1_score(labels[val_index], prediction, average="macro",
                                            zero_division=0))
        scores[int(count)] = float(np.mean(fold_scores)) if fold_scores else np.nan
    finite = {key: value for key, value in scores.items() if np.isfinite(value)}
    selected = max(finite, key=finite.get) if finite else int(candidates[0])
    return {"prototype_count": int(selected), "cv_macro_f1": scores,
            "selection_split": "source_train_cv"}


def fit_prototypes(values, labels, class_count, count, seed=1):
    values, labels = np.asarray(values), np.asarray(labels)
    scale = np.std(values, axis=0)
    scale = np.maximum(scale, 1e-6)
    centers = []
    for class_id in range(class_count):
        selected = values[labels == class_id]
        if not len(selected):
            centers.append(np.empty((0, values.shape[1])))
        elif count == 1 or len(selected) < 2 * count:
            centers.append(np.median(selected, axis=0, keepdims=True))
        else:
            centers.append(KMeans(count, random_state=seed, n_init=10).fit(selected / scale).cluster_centers_ * scale)
    return {"centers": centers, "scale": scale}


def predict_prototypes(values, packet):
    columns = []
    for centers in packet["centers"]:
        if not len(centers):
            columns.append(np.full(len(values), np.inf))
        else:
            distance = np.sqrt(np.mean(
                ((np.asarray(values)[:, None] - centers[None]) / packet["scale"]) ** 2,
                axis=2,
            ))
            columns.append(distance.min(1))
    return np.argmin(np.stack(columns, axis=1), axis=1)


def predict_scaled_nearest_neighbor(train, labels, values, scale):
    """One-nearest-neighbor prediction without materializing an N_val×N_train tensor."""
    scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-6)
    classifier = KNeighborsClassifier(n_neighbors=1, algorithm="kd_tree", n_jobs=1)
    classifier.fit(np.asarray(train, dtype=np.float64) / scale, np.asarray(labels))
    return classifier.predict(np.asarray(values, dtype=np.float64) / scale)


def pair_observation_views(left, right):
    left_ids = np.asarray(left["parcel_index"], dtype=np.int64)
    right_ids = np.asarray(right["parcel_index"], dtype=np.int64)
    if len(set(left_ids)) != len(left_ids) or len(set(right_ids)) != len(right_ids):
        raise ValueError("duplicate parcel IDs in observation views")
    if set(left_ids) != set(right_ids):
        raise ValueError("observation view parcel sets differ")
    lookup = {int(parcel): index for index, parcel in enumerate(right_ids)}
    order = np.asarray([lookup[int(parcel)] for parcel in left_ids])
    packet = {"parcel_index": left_ids, "left": np.asarray(left["strength"]),
              "right": np.asarray(right["strength"])[order]}
    if "label" in left:
        packet["label"] = np.asarray(left["label"])
        if "label" in right and not np.array_equal(packet["label"], np.asarray(right["label"])[order]):
            raise ValueError("observation view labels differ")
    return packet


def _focal(logits, labels, gamma):
    log_probability = F.log_softmax(logits, dim=1)
    probability = log_probability.exp()
    index = torch.arange(len(labels), device=labels.device)
    return (-((1. - probability[index, labels]) ** float(gamma))
            * log_probability[index, labels]).mean()


def joint_constraint_gradient(strength, response_tail, labels, centers, scale, radius,
                              classifier_weight, classifier_bias, focal_gamma=1.):
    centers = torch.as_tensor(centers, dtype=strength.dtype, device=strength.device)
    scale = torch.as_tensor(scale, dtype=strength.dtype, device=strength.device)
    radius = torch.as_tensor(radius, dtype=strength.dtype, device=strength.device)
    selected_center = centers[labels]
    distance = torch.sqrt(torch.mean(((strength - selected_center) / scale) ** 2, 1) + 1e-12)
    excess = torch.relu(distance - radius[labels])
    joint_loss = excess.square().mean()
    joint_gradient = torch.autograd.grad(joint_loss, strength, retain_graph=True)[0]
    response = torch.cat((strength, response_tail.to(strength)), 1)
    logits = F.linear(response, classifier_weight.detach(), classifier_bias.detach())
    shape_loss = _focal(logits, labels, focal_gamma)
    shape_gradient = torch.autograd.grad(shape_loss, strength)[0]
    flat_joint, flat_shape = joint_gradient.reshape(len(labels), -1), shape_gradient.reshape(len(labels), -1)
    cosine = F.cosine_similarity(flat_joint, flat_shape, dim=1, eps=1e-12)
    active = flat_joint.norm(dim=1) > 1e-12
    return {
        "joint_loss": float(joint_loss.detach()),
        "joint_grad_norm": float(joint_gradient.norm().detach()),
        "joint_grad_coverage": float(active.float().mean()),
        "joint_grad_norm_per_sample": flat_joint.norm(dim=1).detach().cpu().numpy(),
        "shape_grad_norm": float(shape_gradient.norm().detach()),
        "gradient_cosine_mean": float(cosine[active].mean()) if active.any() else np.nan,
        "gradient_cosines": cosine[active].detach().cpu().numpy(),
        "distance": distance.detach().cpu().numpy(),
        "excess": excess.detach().cpu().numpy(),
    }


def build_readiness(source, target_train, constraint):
    source_supported = bool(
        source["joint_gain_ci_low"] > 0.
        and source.get("stability_supported", True)
    )
    transfer_supported = bool(target_train["ood_rate"] <= target_train["source_val_ood_ci_high"])
    compactness = bool(
        constraint["val_active_rate"] > 0.
        and constraint["view_penalty_rate"] <= constraint["val_active_rate"]
    )
    redundancy = bool(constraint["gradient_cosine_ci_low"] > 0.)
    if source_supported and transfer_supported and compactness and not redundancy:
        next_step = "train_joint_source_constraint"
    elif source_supported and not transfer_supported:
        next_step = "study_cross_domain_correspondence"
    elif source.get("relative_relation_gain_ci_low", -np.inf) > 0.:
        next_step = "study_relative_anchor_relations"
    else:
        next_step = "insufficient_evidence"
    return {
        "source_joint_supported": source_supported,
        "transfer_support": transfer_supported,
        "compactness_justified": compactness,
        "shape_ce_redundancy_risk": redundancy,
        "recommended_next_step": next_step,
        "decision_inputs": {"source": source, "target_train_unlabeled": target_train,
                            "constraint": constraint},
        "target_val_used_for_readiness": False,
    }


def _bootstrap_mean_ci(values, seed=1, iterations=1000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, len(values), replace=True).mean()
                        for _ in range(iterations)])
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _write_csv(path, rows):
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.stem + ".", suffix=".json", delete=False)
    try:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno()); handle.close(); os.replace(handle.name, path)
    finally:
        if not handle.closed: handle.close()
        Path(handle.name).unlink(missing_ok=True)


def _metric_rows(task, method, labels, prediction, classes, split):
    matrix = confusion_matrix(labels, prediction, labels=np.arange(len(classes)))
    rows = [{"task": task, "split": split, "method": method, "scope": "overall",
             "macro_f1": f1_score(labels, prediction, labels=np.arange(len(classes)),
                                  average="macro", zero_division=0),
             "support": len(labels),
             "model_features": (
                 1 if method.startswith("single_anchor") else
                 96 if method in ("response96_linear", "g_shape_head") else 16
             )}]
    for class_id, name in enumerate(classes):
        selected = np.asarray(labels) == class_id
        rows.append({"task": task, "split": split, "method": method,
                     "scope": "class", "class_id": class_id, "class_name": name,
                     "support": int(selected.sum()),
                     "f1": f1_score(selected, np.asarray(prediction) == class_id,
                                    zero_division=0)})
    for true_id in range(len(classes)):
        for pred_id in range(len(classes)):
            if matrix[true_id, pred_id]:
                rows.append({"task": task, "split": split, "method": method,
                             "scope": "confusion", "true_class": classes[true_id],
                             "predicted_class": classes[pred_id],
                             "count": int(matrix[true_id, pred_id])})
    return rows


def unlabeled_target_distribution_rows(task, model_prediction, nearest_structure,
                                       out_of_support, classes):
    model_prediction = np.asarray(model_prediction, dtype=np.int64)
    nearest_structure = np.asarray(nearest_structure, dtype=np.int64)
    out_of_support = np.asarray(out_of_support, dtype=bool)
    rows = []
    for class_id, class_name in enumerate(classes):
        model_mask = model_prediction == class_id
        structure_mask = nearest_structure == class_id
        rows.append({
            "task": task, "split": "target_train", "scope": "unlabeled_distribution",
            "class_id": class_id, "class_name": class_name,
            "model_prediction_count": int(model_mask.sum()),
            "model_prediction_fraction": float(model_mask.mean()),
            "nearest_structure_count": int(structure_mask.sum()),
            "nearest_structure_fraction": float(structure_mask.mean()),
            "structure_out_of_support_count": int((structure_mask & out_of_support).sum()),
        })
    return rows


def _class_geometry_rows(task, train, train_labels, val, val_labels, reference,
                         prototypes, classes):
    rows = []
    center_distances = class_distances(reference["centers"], reference)
    np.fill_diagonal(center_distances, np.inf)
    for class_id, name in enumerate(classes):
        for split, values, labels in (("source_train", train, train_labels),
                                      ("source_val", val, val_labels)):
            selected = np.asarray(values)[np.asarray(labels) == class_id]
            if not len(selected) or not reference["supported"][class_id]:
                rows.append({"task": task, "class_id": class_id, "class_name": name,
                             "split": split, "support": len(selected), "status": "insufficient"})
                continue
            distance = np.sqrt(np.mean(((selected - reference["centers"][class_id])
                                        / reference["scale"]) ** 2, axis=1))
            all_distances = class_distances(selected, reference)
            competing = all_distances.copy(); competing[:, class_id] = np.inf
            margin = competing.min(1) - all_distances[:, class_id]
            rows.append({"task": task, "class_id": class_id, "class_name": name,
                         "split": split, "support": len(selected), "status": "ok",
                         "distance_mean": float(distance.mean()),
                         "distance_median": float(np.median(distance)),
                         "distance_q90": float(np.quantile(distance, .9)),
                         "radius_source_train": float(reference["radius"][class_id]),
                         "outside_radius_rate": float((distance > reference["radius"][class_id]).mean()),
                         "nearest_class_center_distance": float(center_distances[class_id].min()),
                         "correct_vs_competitor_margin_mean": float(margin.mean()),
                         "correct_vs_competitor_margin_positive_rate": float((margin > 0).mean()),
                         "prototype_count_selected_source_train": prototypes})
    return rows


def _load_checkpoint(path, task):
    packet = require_complete_checkpoint(path)
    validate_four_task_g_checkpoint(packet, task)
    validate_response96_state(packet.get("state_dict", {}))
    raw = packet.get("config", {}); config = dict(raw if isinstance(raw, dict) else vars(raw))
    validate_g_checkpoint_config(config, packet.get("epoch", -1))
    source, target = TASKS[task][1], TASKS[task][3]
    if config.get("source") != source or config.get("target") != target:
        raise ValueError("checkpoint task/domain mismatch")
    classes = list(config.get("classes", ()))
    if not classes or int(config.get("num_classes", len(classes))) != len(classes):
        raise ValueError("checkpoint class mapping is invalid")
    return packet, config, classes


def load_pre_readiness_cache(cache_dir, checkpoint_sha, loader=load_feature_split):
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("cache checkpoint SHA mismatch")
    if manifest.get("target_test_used") or manifest.get("target_train_labels_cached"):
        raise ValueError("cache violates Target protocol")
    arrays = {split: loader(cache_dir, "student", split)
              for split in ("source_train", "source_val", "target_train")}
    if "label" in arrays["target_train"]:
        raise ValueError("Target Train labels are forbidden")
    if "label" not in arrays["source_train"] or "label" not in arrays["source_val"]:
        raise ValueError("Source Train/Val labels are required")
    if set(map(int, arrays["source_train"]["parcel_index"])) & set(
        map(int, arrays["source_val"]["parcel_index"])
    ):
        raise ValueError("Source Train/Val sample leakage")
    return arrays, manifest


def load_paired_views(path, checkpoint_sha):
    with np.load(path, allow_pickle=False) as packet:
        cached_sha = str(np.asarray(packet["checkpoint_sha256"]).item())
        if cached_sha != str(checkpoint_sha):
            raise ValueError("observation-view cache checkpoint SHA mismatch")
        first = {"parcel_index": packet["parcel_index_1"],
                 "strength": packet["strength_1"], "label": packet["label_1"]}
        second = {"parcel_index": packet["parcel_index_2"],
                  "strength": packet["strength_2"], "label": packet["label_2"]}
    return pair_observation_views(first, second)


def _candidate_numpy(values, labels, reference):
    distance = class_distances(values, reference)[np.arange(len(values)), labels]
    excess = np.maximum(distance - reference["radius"][labels], 0.)
    return distance, np.square(excess)


def run_audit(args):
    task, checkpoint, output = args.task, Path(args.checkpoint), Path(args.output_root)
    packet, config, classes = _load_checkpoint(checkpoint, task)
    checkpoint_sha = sha256_file(checkpoint)
    arrays, cache_manifest = load_pre_readiness_cache(args.cache_dir, checkpoint_sha)
    anchor_scope = f"G:{task}:{checkpoint_sha}"
    output.mkdir(parents=True, exist_ok=True)

    source_train, source_val, target_train = (
        arrays["source_train"], arrays["source_val"], arrays["target_train"]
    )
    train_x, val_x, target_x = map(strength16, (
        source_train["response"], source_val["response"], target_train["response"],
    ))
    train_y, val_y = source_train["label"], source_val["label"]
    class_count = len(classes)

    single = select_best_single_anchor(train_x, train_y, args.seed)
    models = {
        f"single_anchor_{single['anchor_id']}": _pipeline().fit(
            train_x[:, [single["anchor_id"]]], train_y,
        ),
        "strength16_linear": _pipeline().fit(train_x, train_y),
        "response96_linear": _pipeline().fit(source_train["response"], train_y),
    }
    reference = fit_joint_reference(train_x, train_y, class_count)
    distance_prediction = np.argmin(class_distances(val_x, reference), axis=1)
    predictions = {
        name: model.predict(
            val_x[:, [single["anchor_id"]]] if name.startswith("single_anchor")
            else source_val["response"] if name == "response96_linear" else val_x
        ) for name, model in models.items()
    }
    predictions["joint_robust_distance"] = distance_prediction
    state = packet["state_dict"]
    shape_logits = source_val["response"] @ state["shape_classifier.weight"].cpu().numpy().T
    shape_logits += state["shape_classifier.bias"].cpu().numpy()
    predictions["g_shape_head"] = shape_logits.argmax(1)
    d1_rows = []
    for name, prediction in predictions.items():
        d1_rows.extend(_metric_rows(task, name, val_y, prediction, classes, "source_val"))
    joint_correct = (predictions["strength16_linear"] == val_y).astype(float)
    single_correct = (predictions[f"single_anchor_{single['anchor_id']}"] == val_y).astype(float)
    gain, gain_low, gain_high = _bootstrap_mean_ci(joint_correct - single_correct, args.seed)
    response_correct = (predictions["response96_linear"] == val_y).astype(float)
    relation_gain, relation_low, relation_high = _bootstrap_mean_ci(
        response_correct - joint_correct, args.seed,
    )
    d1_rows.append({"task": task, "split": "source_train_cv", "method": "best_single_anchor_selection",
                    "scope": "selection", **single})
    d1_rows.append({"task": task, "split": "source_val", "method": "joint_minus_single",
                    "scope": "paired_bootstrap", "accuracy_gain": gain,
                    "gain_ci_low": gain_low, "gain_ci_high": gain_high})
    d1_rows.append({"task": task, "split": "source_val",
                    "method": "response96_minus_strength16",
                    "scope": "paired_bootstrap", "accuracy_gain": relation_gain,
                    "gain_ci_low": relation_low, "gain_ci_high": relation_high,
                    "interpretation": "relative/phase evidence signal, not high-order Strength interaction"})
    joint_wrong = predictions["strength16_linear"] != val_y
    shape_wrong = predictions["g_shape_head"] != val_y
    d1_rows.append({
        "task": task, "split": "source_val", "method": "strength16_vs_g_shape_head",
        "scope": "error_overlap", "support": len(val_y),
        "both_wrong": int((joint_wrong & shape_wrong).sum()),
        "joint_only_wrong": int((joint_wrong & ~shape_wrong).sum()),
        "shape_head_only_wrong": int((~joint_wrong & shape_wrong).sum()),
        "both_correct": int((~joint_wrong & ~shape_wrong).sum()),
    })

    prototype_selection = select_prototype_count(train_x, train_y, seed=args.seed)
    prototype_packet = fit_prototypes(train_x, train_y, class_count,
                                      prototype_selection["prototype_count"], args.seed)
    d2_rows = _class_geometry_rows(
        task, train_x, train_y, val_x, val_y, reference,
        prototype_selection["prototype_count"], classes,
    )
    for method, prediction in (
        ("multi_prototype", predict_prototypes(val_x, prototype_packet)),
        ("class_nearest_neighbor", predict_scaled_nearest_neighbor(
            train_x, train_y, val_x, reference["scale"],
        )),
    ):
        d2_rows.extend(_metric_rows(task, method, val_y, prediction, classes, "source_val"))

    views = load_paired_views(args.views, checkpoint_sha)
    view_delta = (views["left"] - views["right"]) / reference["scale"]
    view_distance = np.sqrt(np.mean(view_delta ** 2, axis=1))
    d3_rows = [{"task": task, "scope": "overall", "metric": "same_parcel_two_view_distance",
                "support": len(view_distance), "mean": float(view_distance.mean()),
                "median": float(np.median(view_distance)),
                "ci_low": _bootstrap_mean_ci(view_distance, args.seed)[1],
                "ci_high": _bootstrap_mean_ci(view_distance, args.seed)[2],
                "bootstrap_unit": "parcel"}]
    for class_id, name in enumerate(classes):
        selected = views["label"] == class_id
        if selected.any():
            d3_rows.append({"task": task, "scope": "class", "class_id": class_id,
                            "class_name": name, "metric": "same_parcel_two_view_distance",
                            "support": int(selected.sum()),
                            "mean": float(view_distance[selected].mean()),
                            "median": float(np.median(view_distance[selected]))})
    for anchor in range(16):
        d3_rows.append({"task": task, "scope": "anchor", "anchor_id": anchor,
                        "metric": "two_view_absolute_change", "support": len(views["left"]),
                        "mean": float(np.abs(view_delta[:, anchor]).mean())})
    # Deterministic parcel-level controls: circular same-class neighbor and first
    # sample from the next available class. They avoid quadratic pair materialization.
    same_distances, different_distances = [], []
    for class_id in range(class_count):
        selected = np.flatnonzero(val_y == class_id)
        if len(selected) > 1:
            rolled = np.roll(selected, 1)
            same_distances.extend(np.sqrt(np.mean(
                ((val_x[selected] - val_x[rolled]) / reference["scale"]) ** 2, axis=1,
            )).tolist())
        other = np.flatnonzero(val_y != class_id)
        if len(selected) and len(other):
            paired_other = np.resize(other, len(selected))
            different_distances.extend(np.sqrt(np.mean(
                ((val_x[selected] - val_x[paired_other]) / reference["scale"]) ** 2, axis=1,
            )).tolist())
    for metric, values in (("same_class_different_parcel_distance", same_distances),
                           ("different_class_distance", different_distances)):
        mean, low, high = _bootstrap_mean_ci(values, args.seed)
        d3_rows.append({"task": task, "scope": "overall", "metric": metric,
                        "support": len(values), "mean": mean, "ci_low": low, "ci_high": high})
    for view_name, values in (("view_1", views["left"]), ("view_2", views["right"])):
        distances = class_distances(values, reference)
        order = np.sort(distances, axis=1)
        prediction = np.argmin(distances, axis=1)
        d3_rows.append({"task": task, "scope": "overall", "metric": "two_view_class_margin",
                        "view": view_name, "support": len(values),
                        "mean": float((order[:, 1] - order[:, 0]).mean()),
                        "class_accuracy": float((prediction == views["label"]).mean())})

    target_distance = class_distances(target_x, reference)
    target_nearest = np.argmin(target_distance, axis=1)
    target_min = target_distance[np.arange(len(target_x)), target_nearest]
    target_radius = reference["radius"][target_nearest]
    target_ood = target_min > target_radius
    source_val_distance = class_distances(val_x, reference)
    val_true_distance = source_val_distance[np.arange(len(val_x)), val_y]
    val_ood = val_true_distance > reference["radius"][val_y]
    train_distance, _ = _candidate_numpy(train_x, train_y, reference)
    train_ood = train_distance > reference["radius"][train_y]
    _, _, train_ood_high = _bootstrap_mean_ci(train_ood.astype(float), args.seed)
    _, _, source_val_ood_high = _bootstrap_mean_ci(val_ood.astype(float), args.seed)
    target_summary = {
        "ood_rate": float(target_ood.mean()),
        "source_val_ood_rate": float(val_ood.mean()),
        "source_val_ood_ci_high": source_val_ood_high,
        "prediction_structure_agreement": float((target_nearest == target_train["prediction"]).mean()),
        "sample_count": len(target_x), "labels_used": False,
    }
    d4_rows = []
    for index in range(len(target_x)):
        ordered = np.sort(target_distance[index])
        d4_rows.append({"task": task, "split": "target_train", "parcel_index": int(target_train["parcel_index"][index]),
                        "predicted_class_id": int(target_train["prediction"][index]),
                        "nearest_source_class_id": int(target_nearest[index]),
                        "nearest_distance": float(target_min[index]),
                        "distance_margin": float(ordered[1] - ordered[0]),
                        "out_of_source_support": bool(target_ood[index]),
                        "prediction_structure_agree": bool(target_nearest[index] == target_train["prediction"][index]),
                        "true_label_used": False})
    d4_rows.extend(unlabeled_target_distribution_rows(
        task, target_train["prediction"], target_nearest, target_ood, classes,
    ))

    _, train_loss = _candidate_numpy(train_x, train_y, reference)
    val_distance, val_loss = _candidate_numpy(val_x, val_y, reference)
    view_label = views["label"]
    _, view_left_loss = _candidate_numpy(views["left"], view_label, reference)
    _, view_right_loss = _candidate_numpy(views["right"], view_label, reference)
    supported_val = reference["supported"][val_y]
    strength_tensor = torch.tensor(val_x[supported_val], dtype=torch.float32, requires_grad=True)
    tail = torch.tensor(source_val["response"][supported_val, 16:], dtype=torch.float32)
    gradient = joint_constraint_gradient(
        strength_tensor, tail, torch.tensor(val_y[supported_val], dtype=torch.long),
        reference["centers"], reference["scale"], reference["radius"],
        state["shape_classifier.weight"], state["shape_classifier.bias"],
        config.get("focal_loss_gamma", 1.),
    )
    cosine = gradient.pop("gradient_cosines")
    grad_norm_per_sample = gradient.pop("joint_grad_norm_per_sample")
    cosine_mean, cosine_low, cosine_high = _bootstrap_mean_ci(cosine, args.seed)
    constraint_summary = {
        "train_active_rate": float((train_loss > 0).mean()),
        "train_loss_mean": float(np.nanmean(train_loss)),
        "val_active_rate": float((val_loss > 0).mean()),
        "val_loss_mean": float(np.nanmean(val_loss)),
        "view_penalty_rate": float(((view_left_loss > 0) | (view_right_loss > 0)).mean()),
        "view_loss_mean": float(np.nanmean(np.concatenate((view_left_loss, view_right_loss)))),
        "gradient_cosine_mean": cosine_mean, "gradient_cosine_ci_low": cosine_low,
        "gradient_cosine_ci_high": cosine_high,
        **{key: value for key, value in gradient.items() if key not in ("distance", "excess")},
    }
    d5_rows = [{"task": task, "scope": "overall", **constraint_summary}]
    for class_id, name in enumerate(classes):
        selected = val_y == class_id
        selected_supported = val_y[supported_val] == class_id
        d5_rows.append({"task": task, "scope": "class", "class_id": class_id,
                        "class_name": name, "support": int(selected.sum()),
                        "active_rate": float((val_loss[selected] > 0).mean()) if selected.any() else np.nan,
                        "loss_mean": float(val_loss[selected].mean()) if selected.any() else np.nan,
                        "joint_grad_coverage": float(
                            (grad_norm_per_sample[selected_supported] > 1e-12).mean()
                        ) if selected_supported.any() else np.nan,
                        "joint_grad_norm_mean": float(
                            grad_norm_per_sample[selected_supported].mean()
                        ) if selected_supported.any() else np.nan})

    source_summary = {
        "joint_gain": gain, "joint_gain_ci_low": gain_low, "joint_gain_ci_high": gain_high,
        "strength16_source_val_macro_f1": float(next(
            row["macro_f1"] for row in d1_rows
            if row.get("method") == "strength16_linear" and row.get("scope") == "overall"
        )),
        "best_single_anchor": single["anchor_id"],
        "relative_relation_gain": relation_gain,
        "relative_relation_gain_ci_low": relation_low,
        "relative_relation_gain_ci_high": relation_high,
        "source_train_ood_rate": float(train_ood.mean()),
        "source_train_ood_ci_high": train_ood_high,
        "source_val_ood_rate": float(val_ood.mean()),
        "stability_supported": bool(float(val_ood.mean()) <= train_ood_high),
    }
    readiness = build_readiness(source_summary, target_summary, constraint_summary)
    readiness["method_parameters"] = {
        "feature": "Strength16", "distance": "global-robust-scaled-rms",
        "class_center": "source_train_median", "radius_quantile": .9,
        "single_anchor_selection": "source_train_stratified_cv",
        "prototype_count": prototype_selection["prototype_count"],
        "prototype_selection": "source_train_stratified_cv",
    }
    readiness["decision_input_sha256"] = hashlib.sha256(json.dumps(
        {"inputs": readiness["decision_inputs"], "parameters": readiness["method_parameters"]},
        sort_keys=True,
    ).encode()).hexdigest()
    readiness.update({"task": task, "checkpoint_sha256": checkpoint_sha,
                      "anchor_scope": anchor_scope, "decision_frozen_before_target_val": True})
    _atomic_json(output / "JOINT_READINESS.json", readiness)

    # Target Val is loaded only after the Source/Target-Train readiness packet is frozen.
    target_val = load_feature_split(args.cache_dir, "student", "target_val")
    if "label" not in target_val:
        raise ValueError("Target Val labels are required for post-hoc evaluation")
    assert_cache_boundary(source_train, source_val, target_train, target_val)
    target_val_x, target_val_y = strength16(target_val["response"]), target_val["label"]
    target_prediction = models["strength16_linear"].predict(target_val_x)
    d4_rows.extend(_metric_rows(task, "source_strength16_probe", target_val_y,
                                target_prediction, classes, "target_val"))
    target_val_distances = class_distances(target_val_x, reference)
    for class_id, name in enumerate(classes):
        selected = target_val_y == class_id
        if not selected.any() or not reference["supported"][class_id]:
            continue
        correct = target_val_distances[selected, class_id]
        competing = target_val_distances[selected].copy()
        competing[:, class_id] = np.inf
        wrong = competing.min(axis=1)
        d4_rows.append({"task": task, "split": "target_val", "scope": "class_geometry",
                        "class_id": class_id, "class_name": name, "support": int(selected.sum()),
                        "correct_class_distance_mean": float(correct.mean()),
                        "nearest_competitor_margin_mean": float((wrong - correct).mean()),
                        "source_target_center_shift": float(np.sqrt(np.mean(
                            ((np.median(target_val_x[selected], axis=0) - reference["centers"][class_id])
                             / reference["scale"]) ** 2)))})
    target_confidence = np.asarray(target_val.get("confidence", np.zeros(len(target_val_y))))
    target_model_prediction = np.asarray(target_val["prediction"], dtype=np.int64)
    high_confidence_wrong = (
        (target_confidence >= float(config.get("pseudo_threshold", .9)))
        & (target_model_prediction != target_val_y)
    )
    if high_confidence_wrong.any():
        structure_prediction = np.argmin(target_val_distances, axis=1)
        d4_rows.append({
            "task": task, "split": "target_val", "scope": "high_confidence_wrong",
            "support": int(high_confidence_wrong.sum()),
            "structure_agrees_true_rate": float(
                (structure_prediction[high_confidence_wrong] == target_val_y[high_confidence_wrong]).mean()
            ),
            "structure_agrees_model_wrong_rate": float(
                (structure_prediction[high_confidence_wrong]
                 == target_model_prediction[high_confidence_wrong]).mean()
            ),
        })

    _write_csv(output / "joint_discrimination.csv", d1_rows)
    _write_csv(output / "class_geometry.csv", d2_rows)
    _write_csv(output / "observation_stability.csv", d3_rows)
    _write_csv(output / "target_transfer.csv", d4_rows)
    _write_csv(output / "constraint_feasibility.csv", d5_rows)
    split_fingerprint = hashlib.sha256(json.dumps(
        cache_manifest.get("split_manifest", {}), sort_keys=True,
    ).encode()).hexdigest()
    _atomic_json(output / "checkpoint_manifest.json", {
        "task": task, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(packet.get("epoch", -1)), "classes": classes,
        "config": {key: config.get(key) for key in (
            "shape_representation", "phase_response_mode", "shapelet_count", "seed",
            "target_structure_detach_epochs", "shape_phase_k2_enabled",
        )},
        "anchor_scope": anchor_scope, "split_fingerprint": split_fingerprint,
        "cache_checkpoint_sha256": cache_manifest.get("checkpoint_sha256"),
        "cache_files_sha256": {
            path.name: sha256_file(path)
            for path in sorted((Path(args.cache_dir) / "features").glob("student_*.npz"))
            if path.name in {f"student_{split}.npz" for split in (
                "source_train", "source_val", "target_train", "target_val",
            )}
        },
        "views_sha256": sha256_file(args.views),
        "target_test_used": False, "target_train_labels_used": False,
        "target_val_used_for_readiness": False,
    })
    report = [
        "# Joint Strength audit", "", f"Task: {task}", "",
        f"- Source joint supported: {readiness['source_joint_supported']}",
        f"- Unlabeled Target support: {readiness['transfer_support']}",
        f"- Compactness justified: {readiness['compactness_justified']}",
        f"- Shape-CE redundancy risk: {readiness['shape_ce_redundancy_risk']}",
        f"- Recommended next step: {readiness['recommended_next_step']}", "",
        f"- Strength16 minus best-single paired gain: {gain:+.6f} "
        f"(95% CI {gain_low:+.6f}, {gain_high:+.6f}).",
        f"- Response96 minus Strength16 paired gain: {relation_gain:+.6f} "
        f"(95% CI {relation_low:+.6f}, {relation_high:+.6f}); this is only a "
        "relative/phase evidence signal, not proof of high-order Strength interaction.",
        f"- Source Val out-of-support rate: {float(val_ood.mean()):.6f}; "
        f"Source Train bootstrap upper reference: {train_ood_high:.6f}.",
        f"- Target Train unlabeled out-of-support rate: {target_summary['ood_rate']:.6f}.",
        f"- Candidate loss Source Val active rate: {constraint_summary['val_active_rate']:.6f}; "
        f"two-view penalty rate: {constraint_summary['view_penalty_rate']:.6f}.",
        f"- Candidate/Shape-CE gradient cosine: {constraint_summary['gradient_cosine_mean']:.6f} "
        f"(95% CI {constraint_summary['gradient_cosine_ci_low']:.6f}, "
        f"{constraint_summary['gradient_cosine_ci_high']:.6f}).", "",
        "Target Val labels were used only after JOINT_READINESS.json was frozen. "
        "Probe readability, distance geometry, and gradient alignment are diagnostic associations, not causal proof. "
        "Strength instability is not interpreted as Anchor identity drift; P0-R evidence must be considered separately.",
    ]
    (output / "TASK_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def extract_views(args):
    checkpoint, output = Path(args.checkpoint), Path(args.views)
    packet, config, _ = _load_checkpoint(checkpoint, args.task)
    frozen = load_generic_frozen(checkpoint, args.task, "G", args.device)
    cfg, _, _, datasets = build_abc_datasets(frozen.config, args.data_root, args.seed)
    packets = []
    for view_seed in (args.seed * 100 + 1, args.seed * 100 + 2):
        random.seed(view_seed); np.random.seed(view_seed); torch.manual_seed(view_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(view_seed)
        arrays = _extract_split(
            frozen.student, datasets["source_val"], cfg.source, 0, args.device,
            args.batch_size, args.pixel_budget, representative_count=0,
            include_labels=True,
        )
        packets.append(arrays)
    paired = pair_observation_views(
        {"parcel_index": packets[0]["parcel_index"], "strength": strength16(packets[0]["response"]),
         "label": packets[0]["label"]},
        {"parcel_index": packets[1]["parcel_index"], "strength": strength16(packets[1]["response"]),
         "label": packets[1]["label"]},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, parcel_index_1=paired["parcel_index"], strength_1=paired["left"],
        label_1=paired["label"], parcel_index_2=paired["parcel_index"],
        strength_2=paired["right"], label_2=paired["label"],
        checkpoint_sha256=np.asarray(sha256_file(checkpoint)),
    )


def extract_cache(args):
    run_extract(
        args.checkpoint, args.source_checkpoint, args.cache_dir, args.task, "G",
        args.data_root, args.device, args.batch_size, args.pixel_budget,
    )


def merge_outputs(root):
    root = Path(root); task_rows, readiness_rows = [], []
    for folder in sorted((root / "tasks").glob("*")):
        path = folder / "JOINT_READINESS.json"
        if not path.is_file(): continue
        packet = json.loads(path.read_text(encoding="utf-8"))
        readiness_rows.append(packet)
        task_rows.append({"task": packet["task"], **{
            key: packet[key] for key in READINESS_FIELDS
        }, **packet["decision_inputs"]["source"],
            "target_train_ood_rate": packet["decision_inputs"]["target_train_unlabeled"]["ood_rate"]})
    _write_csv(root / "JOINT_4TASKS_summary.csv", task_rows)
    complete = len(readiness_rows)
    aggregate = {
        "completed_tasks": complete,
        "source_joint_supported": complete == 4 and all(row["source_joint_supported"] for row in readiness_rows),
        "transfer_support": complete == 4 and all(row["transfer_support"] for row in readiness_rows),
        "compactness_justified": complete == 4 and all(row["compactness_justified"] for row in readiness_rows),
        "shape_ce_redundancy_risk": any(row["shape_ce_redundancy_risk"] for row in readiness_rows),
        "target_val_used_for_readiness": False,
        "per_task": readiness_rows,
    }
    if complete < 4:
        aggregate["recommended_next_step"] = "insufficient_evidence"
    elif aggregate["source_joint_supported"] and not aggregate["transfer_support"]:
        aggregate["recommended_next_step"] = "study_cross_domain_correspondence"
    elif (aggregate["source_joint_supported"] and aggregate["transfer_support"]
          and aggregate["compactness_justified"] and not aggregate["shape_ce_redundancy_risk"]):
        aggregate["recommended_next_step"] = "train_joint_source_constraint"
    elif any(row["recommended_next_step"] == "study_relative_anchor_relations"
             for row in readiness_rows):
        aggregate["recommended_next_step"] = "study_relative_anchor_relations"
    else:
        aggregate["recommended_next_step"] = "insufficient_evidence"
    _atomic_json(root / "JOINT_READINESS.json", aggregate)
    lines = [
        "# Joint Strength four-task frozen audit", "",
        f"Completed tasks: {complete}/4.", "",
        f"Recommended next step: {aggregate['recommended_next_step']}", "",
        "## 1. Full Strength16 vs single Anchor", "",
    ]
    for packet in readiness_rows:
        source = packet["decision_inputs"]["source"]
        lines.append(
            f"- {packet['task']}: paired accuracy gain "
            f"{float(source.get('joint_gain', np.nan)):+.6f} "
            f"(95% CI {float(source.get('joint_gain_ci_low', np.nan)):+.6f}, "
            f"{float(source.get('joint_gain_ci_high', np.nan)):+.6f})."
        )
    lines.extend(["", "## 2. Source joint stability", ""])
    for packet in readiness_rows:
        source = packet["decision_inputs"]["source"]
        lines.append(f"- {packet['task']}: stability_supported={source.get('stability_supported')}.")
    lines.extend(["", "## 3. Observation-view stability", ""])
    for packet in readiness_rows:
        constraint = packet["decision_inputs"]["constraint"]
        lines.append(
            f"- {packet['task']}: two-view penalty rate "
            f"{float(constraint.get('view_penalty_rate', np.nan)):.6f}."
        )
    lines.extend(["", "## 4. Source-to-Target support", ""])
    for packet in readiness_rows:
        target = packet["decision_inputs"]["target_train_unlabeled"]
        lines.append(
            f"- {packet['task']}: unlabeled Target Train OOD rate "
            f"{float(target.get('ood_rate', np.nan)):.6f}; "
            f"transfer_support={packet['transfer_support']}."
        )
    lines.extend(["", "## 5. Candidate constraint feasibility", ""])
    for packet in readiness_rows:
        constraint = packet["decision_inputs"]["constraint"]
        lines.append(
            f"- {packet['task']}: Source Val active rate "
            f"{float(constraint.get('val_active_rate', np.nan)):.6f}; "
            f"candidate/Shape-CE gradient cosine "
            f"{float(constraint.get('gradient_cosine_mean', np.nan)):.6f}; "
            f"compactness_justified={packet['compactness_justified']}; "
            f"shape_ce_redundancy_risk={packet['shape_ce_redundancy_risk']}."
        )
    lines.extend([
        "", "Readiness uses Source labels plus unlabeled Target Train only. Target Val is post-hoc. "
        "No optimizer step, Target Test access, model update, or training was performed. "
        "These are frozen diagnostic associations, not evidence that a new loss will improve training.",
    ])
    (root / "JOINT_FINAL_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("extract-cache", "views", "audit", "summary"), required=True)
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--views", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "summary": merge_outputs(args.output_root)
    elif args.mode == "extract-cache": extract_cache(args)
    elif args.mode == "views": extract_views(args)
    else: run_audit(args)


if __name__ == "__main__":
    main()
