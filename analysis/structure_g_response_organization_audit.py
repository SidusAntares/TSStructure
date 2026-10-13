#!/usr/bin/env python3
"""Frozen Source/G audit of class-conditional Response96 organization.

Target Test is never constructed.  Source Train is the only split used to fit
probes or select cross-Anchor relations; Source Val and Target Val are read-only
evaluation splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.structure_semantic_features import (
    EVENT_COMPONENTS,
    _atomic_json,
    _atomic_npz,
    _extract_split,
    build_abc_datasets,
    create_model_from_config,
    load_feature_split,
    response_to_anchor_events,
)


TASKS = {
    "AT1_DK1": ("AT1", "austria/33UVP/2017", "DK1", "denmark/32VNH/2017"),
    "FR1_FR2": ("FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
    "FR2_DK1": ("FR2", "france/31TCJ/2017", "DK1", "denmark/32VNH/2017"),
    "DK1_AT1": ("DK1", "denmark/32VNH/2017", "AT1", "austria/33UVP/2017"),
}
GROUPS = {
    "strength": (0,),
    "concentration": (1,),
    "phase_k1": (2, 3),
    "phase_k2": (4, 5),
}
MIN_SUPPORT = 5


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_response96(response):
    return response_to_anchor_events(np.asarray(response), anchor_count=16)


def circular_statistics(cosine, sine, eps=1e-12):
    cosine = np.asarray(cosine, dtype=np.float64).reshape(-1)
    sine = np.asarray(sine, dtype=np.float64).reshape(-1)
    valid = np.isfinite(cosine) & np.isfinite(sine)
    cosine, sine = cosine[valid], sine[valid]
    if not len(cosine):
        return {"support": 0, "direction_radians": np.nan,
                "resultant_length": np.nan, "circular_std": np.nan,
                "mean_magnitude": np.nan}
    magnitude = np.hypot(cosine, sine)
    unit_cos = cosine / np.maximum(magnitude, eps)
    unit_sin = sine / np.maximum(magnitude, eps)
    mean_cos, mean_sin = float(unit_cos.mean()), float(unit_sin.mean())
    resultant = float(np.hypot(mean_cos, mean_sin))
    return {
        "support": int(len(cosine)),
        "direction_radians": float(np.arctan2(mean_sin, mean_cos)),
        "resultant_length": resultant,
        "circular_std": float(np.sqrt(max(0.0, -2.0 * np.log(max(resultant, eps))))),
        "mean_magnitude": float(magnitude.mean()),
    }


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: np.nan for key in ("mean", "std", "q05", "q25", "q50", "q75", "q95",
                                         "tail_ratio", "bimodality_coefficient")}
    q05, q25, q50, q75, q95 = np.quantile(values, [.05, .25, .5, .75, .95])
    centered = values - values.mean()
    variance = float(np.mean(centered ** 2))
    if variance <= 1e-12 or len(values) < 4:
        bimodality = np.nan
    else:
        skew = float(np.mean(centered ** 3) / variance ** 1.5)
        kurtosis = float(np.mean(centered ** 4) / variance ** 2)
        bimodality = (skew ** 2 + 1.0) / max(kurtosis, 1e-12)
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "q05": float(q05), "q25": float(q25), "q50": float(q50),
        "q75": float(q75), "q95": float(q95),
        "tail_ratio": float((q95 - q50) / max(q50 - q05, 1e-12)),
        "bimodality_coefficient": bimodality,
    }


def class_anchor_distribution(events, labels, class_names, scope, split, min_support=MIN_SUPPORT):
    events = np.asarray(events, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if events.ndim != 3 or events.shape[1:] != (16, 6):
        raise ValueError(f"events must be [N,16,6], got {events.shape}")
    rows = []
    groups = [(-1, "__all__", events)] + [
        (class_id, class_name, events[labels == class_id])
        for class_id, class_name in enumerate(class_names)
    ]
    for class_id, class_name, selected in groups:
        support = int(len(selected))
        for anchor in range(16):
            for component_index, component in enumerate(EVENT_COMPONENTS):
                row = {
                    "scope": scope, "split": split, "class_id": class_id,
                    "class_name": class_name, "anchor": anchor,
                    "component": component, "support": support,
                    "estimable": support >= min_support,
                    **_quantiles(selected[:, anchor, component_index] if support else []),
                    "direction_radians": np.nan, "resultant_length": np.nan,
                    "circular_std": np.nan, "mean_magnitude": np.nan,
                }
                rows.append(row)
            for harmonic, pair in (("phase_k1", (2, 3)), ("phase_k2", (4, 5))):
                circular = circular_statistics(
                    selected[:, anchor, pair[0]] if support else [],
                    selected[:, anchor, pair[1]] if support else [],
                )
                magnitude = (
                    np.hypot(selected[:, anchor, pair[0]], selected[:, anchor, pair[1]])
                    if support else np.empty(0)
                )
                rows.append({
                    "scope": scope, "split": split, "class_id": class_id,
                    "class_name": class_name, "anchor": anchor,
                    "component": f"{harmonic}_magnitude", "support": support,
                    "estimable": support >= min_support,
                    **_quantiles(magnitude), **circular,
                })
    return rows


def _eta_squared(features, labels):
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    labels = np.asarray(labels)
    total = np.square(values - values.mean(0, keepdims=True)).sum()
    if total <= 1e-12:
        return 0.0
    between = 0.0
    for class_id in np.unique(labels):
        selected = values[labels == class_id]
        between += len(selected) * np.square(selected.mean(0) - values.mean(0)).sum()
    return float(between / total)


def _macro_f1(labels, prediction, class_count):
    return float(f1_score(
        labels, prediction, labels=np.arange(class_count), average="macro", zero_division=0,
    ))


def _nearest_centroid(train, train_labels, values, class_count):
    scale = np.maximum(np.asarray(train).std(0), 1e-8)
    center = np.asarray(train).mean(0)
    fitted = (np.asarray(train) - center) / scale
    evaluated = (np.asarray(values) - center) / scale
    centroids = []
    for class_id in range(class_count):
        selected = fitted[train_labels == class_id]
        centroids.append(selected.mean(0) if len(selected) else np.full(fitted.shape[1], np.nan))
    centroids = np.stack(centroids)
    distance = np.square(evaluated[:, None, :] - centroids[None, :, :]).sum(-1)
    distance[:, ~np.isfinite(centroids).all(1)] = np.inf
    return np.argmin(distance, axis=1), fitted, centroids


def discriminability_rows(train_events, train_labels, val_events, val_labels, class_names,
                           scope, shape_logits=None, evaluation_split="source_val"):
    rows = []
    class_count = len(class_names)
    for anchor in range(16):
        for group, indices in GROUPS.items():
            train = train_events[:, anchor, indices]
            val = val_events[:, anchor, indices]
            if train.ndim == 1:
                train, val = train[:, None], val[:, None]
            prediction, fitted, centroids = _nearest_centroid(
                train, train_labels, val, class_count,
            )
            radius = []
            for class_id in range(class_count):
                selected = fitted[train_labels == class_id]
                if len(selected):
                    radius.extend(np.linalg.norm(selected - centroids[class_id], axis=1))
            pair_distances = []
            valid = np.isfinite(centroids).all(1)
            for left in range(class_count):
                for right in range(left + 1, class_count):
                    if valid[left] and valid[right]:
                        pair_distances.append(np.linalg.norm(centroids[left] - centroids[right]))
            rows.append({
                "scope": scope, "anchor": anchor, "component_group": group,
                "evaluation_split": evaluation_split,
                "train_support": len(train), "val_support": len(val),
                "within_class_dispersion": float(np.mean(radius)) if radius else np.nan,
                "between_class_effect": _eta_squared(train, train_labels),
                "centroid_separation": float(np.mean(pair_distances)) if pair_distances else np.nan,
                "separation_to_dispersion": (
                    float(np.mean(pair_distances)) / max(float(np.mean(radius)), 1e-8)
                    if pair_distances and radius else np.nan
                ),
                "source_val_macro_f1": _macro_f1(val_labels, prediction, class_count),
                "classification_case": "pending",
            })
    dispersions = np.asarray([row["within_class_dispersion"] for row in rows])
    effects = np.asarray([row["between_class_effect"] for row in rows])
    dispersion_cut = float(np.nanmedian(dispersions))
    effect_cut = float(np.nanmedian(effects))
    for row in rows:
        stable = row["within_class_dispersion"] <= dispersion_cut
        separable = row["between_class_effect"] >= effect_cut
        row["classification_case"] = (
            "stable_separable" if stable and separable else
            "stable_weak" if stable else
            "variable_discriminative" if separable else "variable_weak"
        )
        row["stability_threshold_source_train"] = dispersion_cut
        row["effect_threshold_source_train"] = effect_cut
    if shape_logits is not None:
        shape_prediction = np.asarray(shape_logits).argmax(1)
        rows.append({
            "scope": scope, "anchor": -1, "component_group": "shape_classifier_full96",
            "evaluation_split": evaluation_split,
            "train_support": len(train_events), "val_support": len(val_events),
            "within_class_dispersion": np.nan, "between_class_effect": np.nan,
            "centroid_separation": np.nan,
            "source_val_macro_f1": _macro_f1(val_labels, shape_prediction, class_count),
            "classification_case": "reference_only",
            "stability_threshold_source_train": dispersion_cut,
            "effect_threshold_source_train": effect_cut,
        })
    return rows


def _relation_feature(events, left, right, group):
    if group == "strength":
        return events[:, left, 0] * events[:, right, 0]
    if group == "concentration":
        return events[:, left, 1] * events[:, right, 1]
    pair = GROUPS[group]
    left_value, right_value = events[:, left, pair], events[:, right, pair]
    denominator = np.maximum(
        np.linalg.norm(left_value, axis=1) * np.linalg.norm(right_value, axis=1), 1e-8,
    )
    return (left_value * right_value).sum(1) / denominator


def select_cross_anchor_relations(events, labels, max_relations=32):
    candidates = []
    for group in GROUPS:
        for left in range(16):
            for right in range(left + 1, 16):
                feature = _relation_feature(events, left, right, group)
                candidates.append((float(_eta_squared(feature, labels)), group, left, right))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    return tuple((group, left, right) for _, group, left, right in candidates[:max_relations])


def _safe_corr(left, right):
    left, right = np.asarray(left), np.asarray(right)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def shrinkage_dependence(values):
    """Return an Anchor-level dependence matrix using Ledoit-Wolf covariance.

    ``values`` may be [sample, anchor] or [sample, anchor, component].  The
    latter uses normalized Frobenius cross-covariance between Anchor blocks,
    avoiding a scalar angle average at the circular boundary.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3 or values.shape[1] != 16:
        raise ValueError(f"values must be [N,16] or [N,16,D], got {values.shape}")
    samples, anchors, width = values.shape
    if samples < 2:
        return np.eye(anchors, dtype=np.float64)
    covariance = LedoitWolf().fit(values.reshape(samples, anchors * width)).covariance_
    result = np.eye(anchors, dtype=np.float64)
    for left in range(anchors):
        left_slice = slice(left * width, (left + 1) * width)
        left_scale = np.linalg.norm(covariance[left_slice, left_slice], ord="fro")
        for right in range(left + 1, anchors):
            right_slice = slice(right * width, (right + 1) * width)
            right_scale = np.linalg.norm(covariance[right_slice, right_slice], ord="fro")
            cross = np.linalg.norm(covariance[left_slice, right_slice], ord="fro")
            score = float(cross / max(math.sqrt(left_scale * right_scale), 1e-12))
            result[left, right] = result[right, left] = score
    return result


def _residual_by_class(values, labels):
    residual = np.asarray(values, dtype=np.float64).copy()
    for class_id in np.unique(labels):
        selected = labels == class_id
        residual[selected] -= residual[selected].mean(0, keepdims=True)
    return residual


def cross_anchor_relation_rows(events, labels, class_names, scope, split, min_support=MIN_SUPPORT):
    rows = []
    labels = np.asarray(labels)
    groups = [(-1, "__all__", events)] + [
        (class_id, class_name, events[labels == class_id])
        for class_id, class_name in enumerate(class_names)
    ]
    for class_id, class_name, selected in groups:
        for group in GROUPS:
            indices = GROUPS[group]
            vector = selected[:, :, indices]
            if vector.ndim == 2:
                vector = vector[:, :, None]
            scalar = vector[:, :, 0] if len(indices) == 1 else np.arctan2(
                vector[:, :, 1], vector[:, :, 0],
            )
            shrinkage = shrinkage_dependence(vector)
            residual = _residual_by_class(scalar, labels) if class_id == -1 else None
            class_means = (
                np.stack([scalar[labels == value].mean(0) for value in np.unique(labels)])
                if class_id == -1 else None
            )
            for left in range(16):
                for right in range(left + 1, 16):
                    if group.startswith("phase"):
                        relation = np.cos(scalar[:, left] - scalar[:, right])
                        pearson = _safe_corr(
                            selected[:, left, indices[0]], selected[:, right, indices[0]],
                        )
                    else:
                        relation = scalar[:, left] * scalar[:, right]
                        pearson = _safe_corr(scalar[:, left], scalar[:, right])
                    if len(relation):
                        low, high = np.quantile(relation, [.05, .95])
                        robust = np.clip(relation, low, high)
                    else:
                        robust = relation
                    rows.append({
                        "scope": scope, "split": split, "class_id": class_id,
                        "class_name": class_name, "component_group": group,
                        "anchor_left": left, "anchor_right": right,
                        "support": int(len(selected)),
                        "estimable": len(selected) >= min_support,
                        "pearson": pearson,
                        "shrinkage_dependence": float(shrinkage[left, right]),
                        "within_class_residual_corr": (
                            _safe_corr(residual[:, left], residual[:, right])
                            if residual is not None else np.nan
                        ),
                        "between_class_mean_corr": (
                            _safe_corr(class_means[:, left], class_means[:, right])
                            if class_means is not None else np.nan
                        ),
                        "relation_mean": float(np.mean(relation)) if len(relation) else np.nan,
                        "relation_std": float(np.std(relation)) if len(relation) else np.nan,
                        "robust_relation_mean": float(np.mean(robust)) if len(robust) else np.nan,
                    })
    return rows


def _probe_features(response, events, relations):
    base = np.asarray(response, dtype=np.float64)
    if not relations:
        return base
    interaction = np.stack([
        _relation_feature(events, left, right, group)
        for group, left, right in relations
    ], axis=1)
    return np.concatenate((base, interaction), axis=1)


def fit_relation_probe(train, val, target=None, max_relations=32, class_count=None):
    relations = select_cross_anchor_relations(train["events"], train["label"], max_relations)
    if class_count is None:
        class_count = int(max(train["label"].max(), val["label"].max()) + 1)
    rows = []
    predictions = {}
    for name, selected in (("independent", ()), ("independent_plus_relations", relations)):
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1., class_weight="balanced"))
        model.fit(_probe_features(train["response"], train["events"], selected), train["label"])
        for split, arrays in (("source_val", val), ("target_val", target)):
            if arrays is None:
                continue
            prediction = model.predict(_probe_features(arrays["response"], arrays["events"], selected))
            predictions[(name, split)] = prediction
            rows.append({
                "probe": name, "split": split,
                "macro_f1": _macro_f1(arrays["label"], prediction, class_count),
                "selected_relation_count": len(selected),
                "selection_split": "source_train",
            })
    return rows, relations, predictions


def _class_f1(labels, prediction, class_count):
    from sklearn.metrics import precision_recall_fscore_support
    _, _, score, support = precision_recall_fscore_support(
        labels, prediction, labels=np.arange(class_count), zero_division=0,
    )
    return score, support


def source_target_transfer_rows(train, source_val, target_val, class_names, scope):
    probe_rows, relations, predictions = fit_relation_probe(
        train, source_val, target_val, class_count=len(class_names),
    )
    distribution = []
    for class_id, class_name in enumerate(class_names):
        source = source_val["events"][source_val["label"] == class_id]
        target = target_val["events"][target_val["label"] == class_id]
        reference = train["events"][train["label"] == class_id]
        for anchor in range(16):
            for group, indices in GROUPS.items():
                source_values = source[:, anchor, indices]
                target_values = target[:, anchor, indices]
                reference_values = reference[:, anchor, indices]
                if source_values.ndim == 1:
                    source_values, target_values, reference_values = (
                        source_values[:, None], target_values[:, None], reference_values[:, None]
                    )
                scale = np.maximum(reference_values.std(0), 1e-8) if len(reference_values) else np.ones(len(indices))
                mean_shift = (
                    np.linalg.norm((target_values.mean(0) - source_values.mean(0)) / scale)
                    if len(source_values) and len(target_values) else np.nan
                )
                source_disp = float(np.linalg.norm(source_values - source_values.mean(0), axis=1).mean()) if len(source_values) else np.nan
                target_disp = float(np.linalg.norm(target_values - target_values.mean(0), axis=1).mean()) if len(target_values) else np.nan
                distribution.append({
                    "record_type": "single_anchor_shift",
                    "scope": scope, "class_id": class_id, "class_name": class_name,
                    "anchor": anchor, "component_group": group,
                    "source_support": len(source_values), "target_support": len(target_values),
                    "standardized_mean_shift": mean_shift,
                    "source_dispersion": source_disp, "target_dispersion": target_disp,
                    "dispersion_ratio": target_disp / max(source_disp, 1e-8) if np.isfinite(source_disp) and np.isfinite(target_disp) else np.nan,
                })
        for group, left, right in relations:
            source_relation = _relation_feature(source, left, right, group)
            target_relation = _relation_feature(target, left, right, group)
            distribution.append({
                "record_type": "selected_relation_shift", "scope": scope,
                "class_id": class_id, "class_name": class_name,
                "anchor": -1, "component_group": group,
                "anchor_left": left, "anchor_right": right,
                "source_support": len(source_relation), "target_support": len(target_relation),
                "source_relation_mean": float(source_relation.mean()) if len(source_relation) else np.nan,
                "target_relation_mean": float(target_relation.mean()) if len(target_relation) else np.nan,
                "relation_mean_change": (
                    float(target_relation.mean() - source_relation.mean())
                    if len(source_relation) and len(target_relation) else np.nan
                ),
                "standardized_mean_shift": np.nan, "source_dispersion": np.nan,
                "target_dispersion": np.nan, "dispersion_ratio": np.nan,
            })
    class_count = len(class_names)
    source_prediction = predictions[("independent", "source_val")]
    target_prediction = predictions[("independent", "target_val")]
    source_f1, source_support = _class_f1(source_val["label"], source_prediction, class_count)
    target_f1, target_support = _class_f1(target_val["label"], target_prediction, class_count)
    source_macro = float(source_f1.mean())
    reliable_cut = max(.5, source_macro - .1)
    failures = []
    for class_id, class_name in enumerate(class_names):
        reliable = source_f1[class_id] >= reliable_cut and source_support[class_id] >= MIN_SUPPORT
        retained = target_f1[class_id] >= source_f1[class_id] - .15 and target_support[class_id] >= MIN_SUPPORT
        mode = (
            "insufficient_support" if min(source_support[class_id], target_support[class_id]) < MIN_SUPPORT else
            "source_organization_insufficient" if not reliable else
            "still_transferable" if retained else "transfer_failure"
        )
        failures.append({
            "scope": scope, "class_id": class_id, "class_name": class_name,
            "source_val_support": int(source_support[class_id]),
            "target_val_support": int(target_support[class_id]),
            "source_val_f1": float(source_f1[class_id]),
            "target_val_f1": float(target_f1[class_id]),
            "f1_change": float(target_f1[class_id] - source_f1[class_id]),
            "failure_mode": mode, "source_reliability_threshold": reliable_cut,
        })
    return distribution, failures, probe_rows, relations


def assert_anchor_scope_isolated(left, right):
    if left != right:
        raise ValueError("Anchor IDs from different checkpoint scopes are not comparable")
    return True


def validate_cache_protocol(cache_dir, expected_sha=None):
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target_test_used"):
        raise ValueError("Target Test access is forbidden")
    if manifest.get("target_train_labels_cached"):
        raise ValueError("Target Train labels are forbidden in the frozen cache")
    if expected_sha is not None and manifest.get("checkpoint_sha256") != expected_sha:
        raise ValueError("cache checkpoint SHA mismatch")
    if not manifest.get("anchor_scope"):
        manifest["anchor_scope"] = ":".join((
            str(manifest.get("variant", "unknown")),
            str(manifest.get("task", "unknown")),
            str(manifest.get("checkpoint_sha256", "missing")),
        ))
    feature_manifest = cache_dir / "feature_cache_manifest.json"
    if feature_manifest.is_file():
        feature = json.loads(feature_manifest.read_text(encoding="utf-8"))
        if feature.get("target_test_used"):
            raise ValueError("Target Test access is forbidden")
        if feature.get("target_train_labels_cached"):
            raise ValueError("Target Train labels are forbidden in the frozen cache")
        student_shifts = feature.get("applied_shifts", {}).get("student", {})
        if any(int(value) != 0 for value in student_shifts.values()):
            raise ValueError("Student Response cache must use unshifted calendar coordinates")
    return manifest


def _write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _write_scope_figures(output_dir, d1, d3, d4, class_names, stage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    strength = [row for row in d1 if row["split"] == "source_train"
                and row["component"] == "strength" and row["class_id"] >= 0]
    if strength and stage == "source":
        matrix = np.full((len(class_names), 16), np.nan)
        for row in strength:
            matrix[int(row["class_id"]), int(row["anchor"])] = row["mean"]
        fig, axis = plt.subplots(figsize=(9, max(3, .35 * len(class_names))))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set(xlabel="Anchor", ylabel="Source class", yticks=np.arange(len(class_names)),
                 yticklabels=class_names, title="Source Train class-conditional Strength")
        fig.colorbar(image, ax=axis, label="mean Strength"); fig.tight_layout()
        fig.savefig(figure_dir / "D1_strength_class_anchor.png", dpi=160); plt.close(fig)
    global_strength = [row for row in d3 if row["split"] == "source_train"
                       and row["class_id"] == -1 and row["component_group"] == "strength"]
    if global_strength and stage == "source":
        matrix = np.eye(16)
        for row in global_strength:
            left, right = int(row["anchor_left"]), int(row["anchor_right"])
            matrix[left, right] = matrix[right, left] = float(row["within_class_residual_corr"])
        fig, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set(xlabel="Anchor", ylabel="Anchor",
                 title="Pooled within-class Strength correlation")
        fig.colorbar(image, ax=axis); fig.tight_layout()
        fig.savefig(figure_dir / "D3_within_class_strength_correlation.png", dpi=160)
        plt.close(fig)
    single = [row for row in d4 if row.get("record_type") == "single_anchor_shift"]
    if single:
        matrix = np.full((len(class_names), 16), np.nan)
        for class_id in range(len(class_names)):
            for anchor in range(16):
                values = [row["standardized_mean_shift"] for row in single
                          if row["class_id"] == class_id and row["anchor"] == anchor]
                if values: matrix[class_id, anchor] = float(np.nanmean(values))
        fig, axis = plt.subplots(figsize=(9, max(3, .35 * len(class_names))))
        image = axis.imshow(matrix, aspect="auto", cmap="magma")
        axis.set(xlabel="Anchor", ylabel="Class", yticks=np.arange(len(class_names)),
                 yticklabels=class_names, title="Source Val to Target Val standardized shift")
        fig.colorbar(image, ax=axis); fig.tight_layout()
        fig.savefig(figure_dir / "D4_source_target_anchor_shift.png", dpi=160); plt.close(fig)


def _load_scope(cache_dir, stage):
    required = ("source_train", "source_val") if stage == "source" else (
        "source_train", "source_val", "target_val",
    )
    result = {}
    for split in required:
        arrays = load_feature_split(cache_dir, "student", split)
        if "label" not in arrays:
            raise ValueError(f"labels required for frozen evaluation split {split}")
        if "events" not in arrays:
            arrays["events"] = decode_response96(arrays["response"])
        if arrays["response"].ndim != 2 or arrays["response"].shape[1] != 96:
            raise ValueError(f"Response96 required for {split}, got {arrays['response'].shape}")
        parcel = np.asarray(arrays["parcel_index"], dtype=np.int64)
        if len(parcel) != len(np.unique(parcel)):
            raise ValueError(f"duplicate sample ID in {split}")
        result[split] = arrays
    source_train = set(map(int, result["source_train"]["parcel_index"]))
    source_val = set(map(int, result["source_val"]["parcel_index"]))
    if source_train & source_val:
        raise ValueError("Source Train/Val sample leakage")
    return result


def _checkpoint_shape_logits(checkpoint, response):
    try:
        import torch
        packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = packet["state_dict"]
        weight = state["shape_classifier.weight"].detach().cpu().numpy()
        bias = state["shape_classifier.bias"].detach().cpu().numpy()
        return np.asarray(response) @ weight.T + bias
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None


def audit_scope(cache_dir, checkpoint, output_dir, scope, stage, task):
    import torch
    checkpoint_sha = sha256_file(checkpoint)
    manifest = validate_cache_protocol(cache_dir, checkpoint_sha)
    expected_scope = manifest.get("anchor_scope")
    arrays = _load_scope(cache_dir, stage)
    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = packet["config"]
    config = config if isinstance(config, dict) else vars(config)
    if config.get("shape_representation") != "phase_moment":
        raise ValueError("audit requires phase_moment Response96")
    if not bool(config.get("shape_phase_k2_enabled", True)):
        raise ValueError("K2-off checkpoints are forbidden")
    _, expected_source, _, expected_target = TASKS[task]
    if config.get("source") != expected_source:
        raise ValueError("checkpoint Source domain does not match task")
    if stage == "uda_final":
        if config.get("target") != expected_target:
            raise ValueError("checkpoint Target domain does not match task")
        if int(config.get("target_structure_detach_epochs", -1)) != 5:
            raise ValueError("checkpoint is not the original G Detach5 configuration")
        if not np.isclose(float(config.get("shape_equivariance_weight", np.nan)), .05):
            raise ValueError("checkpoint Phase Equivariance weight is incompatible")
    if manifest.get("classes") and list(manifest["classes"]) != list(config["classes"]):
        raise ValueError("cache/checkpoint class mapping mismatch")
    raw_classes = manifest.get("classes") or config["classes"]
    classes = list(raw_classes)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    d1 = []
    for split in ("source_train", "source_val"):
        d1.extend(class_anchor_distribution(
            arrays[split]["events"], arrays[split]["label"], classes, scope, split,
        ))
    shape_logits = _checkpoint_shape_logits(checkpoint, arrays["source_val"]["response"])
    d2 = discriminability_rows(
        arrays["source_train"]["events"], arrays["source_train"]["label"],
        arrays["source_val"]["events"], arrays["source_val"]["label"],
        classes, scope, shape_logits=shape_logits,
    )
    if stage == "uda_final":
        target_shape_logits = _checkpoint_shape_logits(
            checkpoint, arrays["target_val"]["response"],
        )
        d2.extend(discriminability_rows(
            arrays["source_train"]["events"], arrays["source_train"]["label"],
            arrays["target_val"]["events"], arrays["target_val"]["label"],
            classes, scope, shape_logits=target_shape_logits,
            evaluation_split="target_val",
        ))
    d3 = []
    relation_splits = ["source_train", "source_val"]
    if stage == "uda_final": relation_splits.append("target_val")
    for split in relation_splits:
        d3.extend(cross_anchor_relation_rows(
            arrays[split]["events"], arrays[split]["label"], classes, scope, split,
        ))
    probe_rows, relations, _ = fit_relation_probe(
        arrays["source_train"], arrays["source_val"],
        arrays.get("target_val"), max_relations=32, class_count=len(classes),
    )
    for row in probe_rows:
        row.update({"scope": scope, "task": task, "stage": stage})
    d4, failures = [], []
    if stage == "uda_final":
        d4, failures, _, _ = source_target_transfer_rows(
            arrays["source_train"], arrays["source_val"], arrays["target_val"],
            classes, scope,
        )
    for rows in (d1, d2, d3, probe_rows, d4, failures):
        for row in rows:
            row.setdefault("task", task)
            row.setdefault("stage", stage)
    _write_csv(output_dir / "D1_class_anchor_distribution.csv", d1)
    _write_csv(output_dir / "D2_class_discriminability.csv", d2)
    _write_csv(output_dir / "D3_cross_anchor_relations.csv", d3)
    _write_csv(output_dir / "D3_relation_increment.csv", probe_rows)
    _write_csv(output_dir / "D4_source_target_transfer.csv", d4)
    _write_csv(output_dir / "D4_class_failure_modes.csv", failures)
    _write_scope_figures(output_dir, d1, d3, d4, classes, stage)
    _atomic_json(output_dir / "scope_manifest.json", {
        "scope": scope, "stage": stage, "task": task,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "cache": str(cache_dir), "cache_checkpoint_sha256": manifest["checkpoint_sha256"],
        "anchor_scope": expected_scope, "selected_relations": [list(row) for row in relations],
        "cache_files_sha256": {
            path.name: sha256_file(path)
            for path in sorted((Path(cache_dir) / "features").glob("student_*.npz"))
            if path.name in {
                "student_source_train.npz", "student_source_val.npz",
                "student_target_val.npz",
            }
        },
        "selection_split": "source_train", "validation_split": "source_val",
        "target_evaluation_split": "target_val" if stage == "uda_final" else None,
        "target_test_used": False, "target_train_labels_used": False,
    })


def extract_source_cache(checkpoint, cache_dir, data_root, device, batch_size, pixel_budget):
    import torch
    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = packet.get("config")
    config = SimpleNamespace(**dict(raw if isinstance(raw, dict) else vars(raw)))
    config.data_root = str(data_root)
    if config.shape_representation != "phase_moment" or config.shape_injection != "current_query":
        raise ValueError("Source checkpoint is not the original G Phase Moment representation")
    if not bool(getattr(config, "shape_phase_k2_enabled", True)):
        raise ValueError("K2-off checkpoint is forbidden for this audit")
    model = create_model_from_config(config)
    model.load_state_dict(packet["state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    cfg, protocol, _, datasets = build_abc_datasets(config, data_root, 1)
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("source_train", "source_val"):
        arrays = _extract_split(
            model, datasets[split], cfg.source, 0, device, batch_size,
            pixel_budget, representative_count=0, include_labels=True,
        )
        relative = f"features/student_{split}.npz"
        _atomic_npz(cache_dir / relative, arrays); files[split] = relative
    digest = sha256_file(checkpoint)
    _atomic_json(cache_dir / "feature_cache_manifest.json", {
        "schema_version": 2, "files": {"student": files},
        "target_test_used": False, "target_train_labels_cached": False,
        "sample_counts": {key: len(load_feature_split(cache_dir, "student", key)["label"]) for key in files},
    })
    _atomic_json(cache_dir / "checkpoint_manifest.json", {
        "stage": "source", "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "classes": list(config.classes), "protocol": protocol,
        "anchor_scope": f"SOURCE:{config.source}:{digest}",
        "target_test_used": False, "target_train_labels_cached": False,
    })


def extract_uda_cache(checkpoint, source_checkpoint, cache_dir, task, data_root,
                      device, batch_size, pixel_budget):
    from analysis.structure_pair_semantics_audit import run_extract
    run_extract(
        checkpoint, source_checkpoint, cache_dir, task, "G", data_root,
        device, batch_size, pixel_budget,
    )
    manifest_path = Path(cache_dir) / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["anchor_scope"] = f"G:{task}:{manifest['checkpoint_sha256']}"
    _atomic_json(manifest_path, manifest)


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def merge_outputs(output_root, missing=None):
    output_root = Path(output_root)
    names = (
        "D1_class_anchor_distribution.csv", "D2_class_discriminability.csv",
        "D3_cross_anchor_relations.csv", "D3_relation_increment.csv",
        "D4_source_target_transfer.csv", "D4_class_failure_modes.csv",
    )
    merged = {}
    for name in names:
        rows = []
        for path in sorted((output_root / "scopes").glob(f"*/{name}")):
            rows.extend(_read_csv(path))
        _write_csv(output_root / name, rows); merged[name] = rows
    d2, d3inc, failures = merged[names[1]], merged[names[3]], merged[names[5]]
    full = [row for row in d2 if row.get("component_group") == "shape_classifier_full96"]
    increments = []
    grouped = {}
    for row in d3inc:
        grouped.setdefault((row.get("scope"), row.get("split")), {})[row.get("probe")] = row
    for values in grouped.values():
        if "independent" in values and "independent_plus_relations" in values:
            increments.append(float(values["independent_plus_relations"]["macro_f1"]) - float(values["independent"]["macro_f1"]))
    modes = {}
    for row in failures:
        modes[row["failure_mode"]] = modes.get(row["failure_mode"], 0) + 1
    lines = [
        "# Source class-conditional Response organization audit", "",
        f"Completed scopes: {len(list((output_root / 'scopes').glob('*/scope_manifest.json')))}/8.",
        "Target Test was not constructed or accessed. Target Val labels were used only for offline evaluation.", "",
        "## Answers", "",
        f"1. Source Response96 shape-head references available: {len(full)}; detailed per-Anchor stability and discriminability are in D1/D2.",
        f"2. Stable/separable versus weak/variable Anchor statistics are explicitly labeled in D2; no blanket constraint is inferred.",
        f"3. Source-selected cross-Anchor relation probe increments: n={len(increments)}, mean={(np.mean(increments) if increments else np.nan):.6f}.",
        f"4. Offline class failure modes in G checkpoints: {json.dumps(modes, sort_keys=True)}.",
        "5. Evidence for any future constraint requires stable_separable Source Val rows and non-negative Target transfer; this audit does not implement a loss.",
        "6. Large within-class dispersion, weak Source Val effects, or preserved Target performance despite mean/covariance shift do not support class-center compression or forced distribution alignment.",
        "", "Correlation is not a state transition, probe readability is not information-theoretic information, and observed associations are not causal conclusions.",
    ]
    if missing:
        lines.extend(["", "## Missing dependencies", "", *[f"- {item}" for item in missing]])
    (output_root / "FINAL_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("extract", "audit", "summary"), required=True)
    parser.add_argument("--stage", choices=("source", "uda_final"))
    parser.add_argument("--scope"); parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--checkpoint"); parser.add_argument("--source-checkpoint")
    parser.add_argument("--cache-dir"); parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "summary":
        merge_outputs(args.output_root); return
    if not all((args.stage, args.scope, args.task, args.checkpoint, args.cache_dir)):
        raise ValueError("extract/audit require stage, scope, task, checkpoint and cache-dir")
    if args.mode == "extract":
        if args.stage == "source":
            extract_source_cache(args.checkpoint, args.cache_dir, args.data_root,
                                 args.device, args.batch_size, args.pixel_budget)
        else:
            if not args.source_checkpoint:
                raise ValueError("UDA extraction requires source-checkpoint")
            extract_uda_cache(args.checkpoint, args.source_checkpoint, args.cache_dir, args.task,
                              args.data_root, args.device, args.batch_size, args.pixel_budget)
    else:
        audit_scope(args.cache_dir, args.checkpoint,
                    Path(args.output_root) / "scopes" / args.scope,
                    args.scope, args.stage, args.task)


if __name__ == "__main__":
    main()
