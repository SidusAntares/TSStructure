#!/usr/bin/env python3
"""Frozen E1/E2/E3 diagnostics built on the protocol-safe ABC audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

from analysis.diagnose_eg_pair import sha256_file, validate_source
from analysis.structure_semantic_features import (
    ALLOWED_SPLITS, EVENT_COMPONENTS, _atomic_json, _atomic_npz, _extract_split,
    build_abc_datasets, load_frozen_models, replay_abc_datasets,
    response_to_anchor_events,
)
from analysis.structure_semantic_abc import (
    PAIR_NAMES, _batch_for_parcels, _configured_pairs, _cosine, _write_csv,
    _write_json, calibrate_source_threshold, fit_source_structure_support,
    normal_pair_margin_sensitivity, pairwise_event_statistics,
    score_structure_support, select_source_only_candidates,
)


FEATURE_GROUPS = {
    "SC": (0, 1), "SC_K1": (0, 1, 2, 3),
    "SC_K2": (0, 1, 4, 5), "SCP": (0, 1, 2, 3, 4, 5),
}
STAGE_ORDER = ("extract", "E1", "E2", "E3", "report")
STAGE_ARTIFACTS = {
    "extract": ("cache_reference.json", "checkpoint_manifest.json"),
    "E1": (
        "E1_anchor_semantics.csv", "E1_domain_correspondence.csv",
        "E1_branch_dependency.csv", "E1_representatives.png", "E1_report.md",
    ),
    "E2": (
        "E2_anchor_phase_importance.csv", "E2_phase_increment.csv",
        "E2_cross_task_comparison.csv", "E2_model_usage.csv", "E2_report.md",
    ),
    "E3": (
        "E3_all_teacher_scores.csv", "E3_within_prediction_reliability.csv",
        "E3_rejection_tradeoff.csv", "E3_score_components.csv",
        "E3_negative_evidence.csv", "E3_positive_evidence.csv", "E3_report.md",
    ),
    "report": ("checkpoint_report.md",),
}


def required_stage_artifacts(stage):
    if stage not in STAGE_ARTIFACTS:
        raise ValueError(f"unknown E123 stage: {stage}")
    return STAGE_ARTIFACTS[stage]


def anchor_feature_indices(anchor_id, group):
    if group not in FEATURE_GROUPS:
        raise ValueError(f"unknown feature group: {group}")
    anchor = int(anchor_id)
    if not 0 <= anchor < 16:
        raise ValueError("anchor id must be in [0,15]")
    return [component * 16 + anchor for component in FEATURE_GROUPS[group]]


def compact_feature_arrays(arrays, role, split):
    role, split = str(role), str(split)
    keep = {
        "parcel_index", "events", "confidence",
        "prediction", "domain", "window_centers_days",
    }
    if split != "target_train":
        keep.add("label")
    result = {key: value for key, value in arrays.items() if key in keep}
    if "events" not in result:
        result["events"] = response_to_anchor_events(arrays["response"])
    similarity = arrays.get("similarity")
    if similarity is not None:
        result["anchor_reliability"] = np.asarray(similarity).max(axis=1).mean(axis=1)
    elif "anchor_reliability" in arrays:
        result["anchor_reliability"] = arrays["anchor_reliability"]
    if role == "student" and split in ("source_train", "target_train", "target_val"):
        for key, value in arrays.items():
            if key.startswith("representative_"):
                result[key] = value
    return result


def _load_npz(path):
    with np.load(path, allow_pickle=False) as packet:
        return {key: packet[key] for key in packet.files}


def load_e123_split(cache_dir, role, split):
    path = Path(cache_dir) / "features" / f"{role}_{split}.npz"
    arrays = _load_npz(path)
    return compact_feature_arrays(arrays, role, split)


def _cache_is_complete(root):
    root = Path(root)
    return all(
        (root / "features" / f"{role}_{split}.npz").is_file()
        for role in ("student", "teacher") for split in ALLOWED_SPLITS
    )


def reused_abc_protocol(root, checkpoint_digest):
    """Return the exact protocol packet that produced a reusable ABC cache."""
    root = Path(root)
    if not _cache_is_complete(root):
        return None
    path = root / "checkpoint_manifest.json"
    if not path.is_file():
        raise ValueError(f"reusable ABC cache lacks checkpoint manifest: {path}")
    packet = json.loads(path.read_text(encoding="utf-8"))
    if packet.get("checkpoint_sha256") != str(checkpoint_digest):
        raise ValueError("reusable ABC cache checkpoint hash does not match E123 checkpoint")
    split_manifest = packet.get("split_manifest")
    if not isinstance(split_manifest, dict) or set(split_manifest) != set(ALLOWED_SPLITS):
        raise ValueError("reusable ABC cache has an invalid split manifest")
    if packet.get("target_test_used", False):
        raise ValueError("reusable ABC cache accessed Target Test")
    return packet


def validate_cache_protocol(root, datasets=None, domains=None):
    """Validate role isolation and exact (domain, parcel_index) alignment."""
    root = Path(root)
    forbidden = list((root / "features").glob("*target_test*.npz"))
    if forbidden:
        raise ValueError(f"Target Test cache is forbidden: {forbidden[0]}")
    role_keys = {}
    for role in ("student", "teacher"):
        for split in ALLOWED_SPLITS:
            path = root / "features" / f"{role}_{split}.npz"
            arrays = _load_npz(path)
            if split == "target_train" and "label" in arrays:
                raise ValueError(f"Target Train truth must not be cached: {path}")
            domains_in_cache = np.asarray(arrays["domain"]).astype(str)
            parcels_in_cache = np.asarray(arrays["parcel_index"]).astype(int)
            keys = tuple(zip(domains_in_cache.tolist(), parcels_in_cache.tolist()))
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate sample key in {path}")
            role_keys[(role, split)] = keys
            if datasets is not None:
                expected_parcels = [int(value) for value in datasets[split].get_parcel_indices()]
                expected_domain = domains["source" if split.startswith("source") else "target"]
                expected = [(str(expected_domain), parcel) for parcel in expected_parcels]
                if len(keys) != len(expected) or set(keys) != set(expected):
                    raise ValueError(f"cache/dataset key mismatch: {role}/{split}")
    for split in ALLOWED_SPLITS:
        if role_keys[("student", split)] != role_keys[("teacher", split)]:
            raise ValueError(f"Student/Teacher key mismatch: {split}")


def validate_shift_manifest(root, expected_teacher_shift):
    path = Path(root) / "feature_cache_manifest.json"
    if not path.is_file():
        raise ValueError(f"reused cache lacks temporal-coordinate manifest: {path}")
    shifts = json.loads(path.read_text(encoding="utf-8")).get("applied_shifts", {})
    student = shifts.get("student", {})
    teacher = shifts.get("teacher", {})
    if set(student) != set(ALLOWED_SPLITS) or any(int(student[split]) != 0 for split in ALLOWED_SPLITS):
        raise ValueError("Student temporal coordinate must be zero for every split")
    if set(teacher) != set(ALLOWED_SPLITS) or any(int(teacher[split]) != int(expected_teacher_shift) for split in ALLOWED_SPLITS):
        raise ValueError("Teacher temporal coordinate must match global shift for every split")
    return True


def prepare_feature_source(
    frozen, datasets, domains, output_dir, abc_checkpoint_dir=None,
    device="cuda", batch_size=64, pixel_budget=8192, representative_count=8,
):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    abc_checkpoint_dir = Path(abc_checkpoint_dir) if abc_checkpoint_dir else None
    if abc_checkpoint_dir is not None and _cache_is_complete(abc_checkpoint_dir):
        source = abc_checkpoint_dir.resolve()
        reused = True
        validate_shift_manifest(source, int(frozen.shifts["teacher"]))
    else:
        source = output_dir
        reused = False
        for role in ("student", "teacher"):
            model, shift = getattr(frozen, role), int(frozen.shifts[role])
            for split in ALLOWED_SPLITS:
                domain = domains["source" if split.startswith("source") else "target"]
                arrays = _extract_split(
                    model, datasets[split], domain, shift, device, batch_size,
                    pixel_budget, representative_count,
                    include_labels=split != "target_train",
                )
                compact = compact_feature_arrays(arrays, role, split)
                _atomic_npz(output_dir / "features" / f"{role}_{split}.npz", compact)
    validate_cache_protocol(source, datasets=datasets, domains=domains)
    reference = {
        "schema_version": 1, "cache_root": str(source), "reused_abc_cache": reused,
        "target_test_used": False, "target_train_labels_cached": False,
        "roles": {"E1_E2": "student", "E3": "teacher"},
        "applied_shifts": {"student": 0, "teacher": int(frozen.shifts["teacher"])},
    }
    _write_json(output_dir / "cache_reference.json", reference)
    return source, reference


def cache_root(output_dir):
    packet = json.loads((Path(output_dir) / "cache_reference.json").read_text(encoding="utf-8"))
    return Path(packet["cache_root"])


def _representatives(data, anchor, kind="top"):
    selected = (
        (data["representative_anchor"].astype(int) == int(anchor))
        & (data["representative_kind"].astype(str) == str(kind))
    )
    return {key: value[selected] for key, value in data.items() if key.startswith("representative_")}


def _flat_standardized(curves):
    curves = np.asarray(curves, dtype=np.float64)
    if not len(curves):
        return np.empty((0, 0))
    mean = curves.mean(1, keepdims=True)
    std = np.maximum(curves.std(1, keepdims=True), 1e-6)
    return ((curves - mean) / std).reshape(len(curves), -1)


def _mean_pairwise_rms(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan")
    distance = np.sqrt(np.square(values[:, None] - values[None, :]).mean(-1))
    return float(distance[np.triu_indices(len(values), 1)].mean())


@torch.inference_mode()
def branch_dependency(model, curves, anchor_id):
    curves = torch.as_tensor(curves, device=next(model.parameters()).device, dtype=torch.float32)
    if curves.ndim != 3 or not len(curves):
        return {name: float("nan") for name in ("raw", "difference", "mean", "std")}
    encoded = model.structure_branch.token_generator(
        curves[:, None], return_encoded_components=True,
    )
    names = ("raw_encoded", "diff_encoded", "mean_encoded", "std_encoded")
    baseline = model.structure_branch.shapelet_dictionary.compute_similarity(
        encoded["shape_token"],
    )[:, 0, int(anchor_id)]
    result = {}
    for label, removed in zip(("raw", "difference", "mean", "std"), names):
        parts = [torch.zeros_like(encoded[name]) if name == removed else encoded[name] for name in names]
        token = model.structure_branch.token_generator.fusion(torch.cat(parts, dim=-1))
        score = model.structure_branch.shapelet_dictionary.compute_similarity(token)[:, 0, int(anchor_id)]
        result[label] = float((baseline - score).abs().mean().cpu())
    return result


@torch.inference_mode()
def controlled_stability(model, curves, anchor_id, seed=1):
    curves = torch.as_tensor(curves, device=next(model.parameters()).device, dtype=torch.float32)
    if curves.ndim != 3 or not len(curves):
        return {}
    generator = torch.Generator(device=curves.device).manual_seed(int(seed))
    perturbations = {
        "noise_0.05std": curves + .05 * curves.std((1, 2), keepdim=True) * torch.randn(
            curves.shape, generator=generator, device=curves.device,
        ),
        "amplitude_x1.05": curves * 1.05,
        "position_roll_1": torch.roll(curves, 1, dims=1),
    }
    dictionary = model.structure_branch.shapelet_dictionary
    base = dictionary.compute_similarity(
        model.structure_branch.token_generator(curves[:, None]),
    )[:, 0, int(anchor_id)]
    result = {}
    for name, changed in perturbations.items():
        score = dictionary.compute_similarity(
            model.structure_branch.token_generator(changed[:, None]),
        )[:, 0, int(anchor_id)]
        result[name] = float((score - base).abs().mean().cpu())
    return result


def run_e1_stage(model, cache_dir, output_dir, classes, seed=1, abc_checkpoint_dir=None):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    source = load_e123_split(cache_dir, "student", "source_train")
    target = load_e123_split(cache_dir, "student", "target_train")
    target_val = load_e123_split(cache_dir, "student", "target_val")
    semantics, correspondence, dependencies = [], [], []
    mean_shapes = {}
    for split, data in (("source_train", source), ("target_train", target), ("target_val", target_val)):
        label_lookup = {}
        if "label" in data:
            label_lookup = dict(zip(data["parcel_index"].astype(int), data["label"].astype(int)))
        for anchor in range(16):
            reps = _representatives(data, anchor, "top")
            curves = reps.get("representative_curves", np.empty((0, 0, 0)))
            standardized = _flat_standardized(curves)
            mean_shapes[(split, anchor)] = standardized.mean(0) if len(standardized) else np.asarray([])
            labels = [label_lookup.get(int(parcel)) for parcel in reps.get("representative_parcel_index", [])]
            row = {
                "split": split, "anchor_id": anchor, "support": len(curves),
                "full_shape_dispersion": _mean_pairwise_rms(standardized),
                "nearest_real_parcel": int(reps["representative_parcel_index"][0]) if len(curves) else "",
                "nearest_real_window": int(reps["representative_window_index"][0]) if len(curves) else "",
                "nearest_similarity": float(reps["representative_score"][0]) if len(curves) else float("nan"),
                "class_coverage": len({value for value in labels if value is not None}),
            }
            for class_id, class_name in enumerate(classes):
                row[f"class_{class_name}_support"] = sum(value == class_id for value in labels)
            semantics.append(row)
            if len(curves):
                deps = branch_dependency(model, curves, anchor)
                stability = controlled_stability(model, curves[: min(4, len(curves))], anchor, seed + anchor)
                dependencies.append({"anchor_id": anchor, "split": split, **deps, **stability})
    for split in ("source_train", "target_train", "target_val"):
        for anchor in range(16):
            current = mean_shapes[(split, anchor)]
            similarities = [
                _cosine(current, mean_shapes[(split, other)])
                for other in range(16) if other != anchor and len(current) and len(mean_shapes[(split, other)])
            ]
            row = next(item for item in semantics if item["split"] == split and item["anchor_id"] == anchor)
            row["max_other_anchor_shape_cosine"] = float(np.nanmax(similarities)) if similarities else float("nan")
    centers = source.get("window_centers_days", np.asarray([]))
    target_centers = target.get("window_centers_days", np.asarray([]))
    for anchor in range(16):
        left, right = mean_shapes[("source_train", anchor)], mean_shapes[("target_train", anchor)]
        source_reps, target_reps = _representatives(source, anchor), _representatives(target, anchor)
        correspondence.append({
            "anchor_id": anchor,
            "full_shape_rms": float(np.sqrt(np.square(left - right).mean())) if len(left) and len(right) else float("nan"),
            "source_center_day_mean": float(np.mean(centers[source_reps["representative_window_index"].astype(int)])) if len(source_reps.get("representative_window_index", [])) else float("nan"),
            "target_center_day_mean": float(np.mean(target_centers[target_reps["representative_window_index"].astype(int)])) if len(target_reps.get("representative_window_index", [])) else float("nan"),
        })
    for row in correspondence:
        row["center_day_difference"] = row["target_center_day_mean"] - row["source_center_day_mean"]
    _write_csv(output_dir / "E1_anchor_semantics.csv", semantics)
    _write_csv(output_dir / "E1_domain_correspondence.csv", correspondence)
    _write_csv(output_dir / "E1_branch_dependency.csv", dependencies)
    _plot_e1(source, target, output_dir / "E1_representatives.png")
    poor = [row["anchor_id"] for row in semantics if row["split"] == "source_train" and (not np.isfinite(row["full_shape_dispersion"]) or row["full_shape_dispersion"] > 1.)]
    mismatch = [row["anchor_id"] for row in correspondence if np.isfinite(row["full_shape_rms"]) and row["full_shape_rms"] > 1.]
    target_coverage_gaps = [
        class_name for class_name in classes
        if sum(int(row.get(f"class_{class_name}_support", 0)) for row in semantics if row["split"] == "target_val") == 0
    ]
    unstable = []
    transfer_path = Path(abc_checkpoint_dir) / "A_transferability_validation.csv" if abc_checkpoint_dir else None
    if transfer_path is not None and transfer_path.is_file():
        unstable = [
            f"{row.get('pair')}:A{row.get('anchor_id')}"
            for row in _read_csv(transfer_path)
            if row.get("role") == "student" and row.get("validation_status") == "failed"
        ]
    if poor:
        conclusion = "形态代表性不足"
    elif mismatch:
        conclusion = "跨域对应不足"
    elif unstable:
        conclusion = "结构本身较可靠，但类别判别关系不稳定"
    else:
        conclusion = "暂未发现需要增加 Anchor 约束的证据"
    (output_dir / "E1_report.md").write_text(
        "# E1 — Anchor formation audit\n\n"
        f"- Morphology-representativeness concerns: {poor or ['none']}.\n"
        f"- Cross-domain morphology concerns: {mismatch or ['none']}.\n"
        f"- Morphology-supported but class-relation-unstable evidence: {unstable or ['none']}.\n"
        f"- Target Val class representative gaps: {target_coverage_gaps or ['none']}.\n"
        f"- Protocol-level conclusion: **{conclusion}**.\n"
        "- Date displacement is reported separately from full-window morphology.\n"
        "- Raw/Difference/Mean/Std ablations and noise/amplitude/position perturbations are descriptive; transformations are not assumed invariant.\n"
        "- PSE windows are latent morphology and are not interpreted as confirmed phenological stages.\n",
        encoding="utf-8",
    )


def _plot_e1(source, target, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 8, figsize=(19, 11))
    for anchor in range(16):
        row, pair_column = divmod(anchor, 4)
        source_curves = _representatives(source, anchor).get("representative_curves", np.empty((0, 0, 0)))
        target_curves = _representatives(target, anchor).get("representative_curves", np.empty((0, 0, 0)))
        combined = [values.mean(0) for values in (source_curves, target_curves) if len(values)]
        limit = max((float(np.max(np.abs(value))) for value in combined), default=1.)
        for offset, label, curves in ((0, "Source", source_curves), (1, "Target", target_curves)):
            axis = axes[row, 2 * pair_column + offset]
            if len(curves):
                axis.imshow(curves.mean(0).T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
            axis.set_title(f"A{anchor} {label}")
            axis.set_xlabel("window step")
            axis.set_ylabel("PSE channel")
    fig.suptitle("Mean full latent PSE windows (not confirmed phenology)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _probe():
    return Pipeline([("scale", StandardScaler()), ("classifier", RidgeClassifier(alpha=1., class_weight="balanced"))])


def _balanced(model, values, labels):
    return float(balanced_accuracy_score(labels, model.predict(values)))


def compare_anchor_probe_groups(
    source_train_events, source_train_labels, source_val_events, source_val_labels,
    target_events, target_labels, y_id, c_id, min_support=10,
):
    arrays = [np.asarray(value) for value in (source_train_events, source_val_events, target_events)]
    labels = [np.asarray(value, dtype=int) for value in (source_train_labels, source_val_labels, target_labels)]
    masks = [(value == int(y_id)) | (value == int(c_id)) for value in labels]
    support_y = int((labels[0] == int(y_id)).sum()); support_c = int((labels[0] == int(c_id)).sum())
    rows = []
    for anchor in range(16):
        for group, components in FEATURE_GROUPS.items():
            row = {
                "anchor_id": anchor, "feature_group": group,
                "support_y": support_y, "support_c": support_c,
                "source_train_support": int(masks[0].sum()),
                "source_val_support": int(masks[1].sum()),
                "target_support": int(masks[2].sum()),
            }
            if min(support_y, support_c, int((labels[1] == y_id).sum()), int((labels[1] == c_id).sum())) < int(min_support):
                row.update({"status": "N/A", "source_val_balanced_accuracy": float("nan"), "target_balanced_accuracy": float("nan")})
            else:
                model = _probe().fit(arrays[0][masks[0], anchor][:, components], labels[0][masks[0]])
                target_has_pair = (
                    int((labels[2] == int(y_id)).sum()) >= 2
                    and int((labels[2] == int(c_id)).sum()) >= 2
                )
                row.update({
                    "status": "ok",
                    "source_val_balanced_accuracy": _balanced(model, arrays[1][masks[1], anchor][:, components], labels[1][masks[1]]),
                    "target_status": "ok" if target_has_pair else "N/A_missing_class",
                    "target_balanced_accuracy": _balanced(model, arrays[2][masks[2], anchor][:, components], labels[2][masks[2]]) if target_has_pair else float("nan"),
                })
            rows.append(row)
    return rows


def run_e2_stage(model, datasets, cache_dir, output_dir, task, classes, seed=1):
    output_dir = Path(output_dir)
    data = {split: load_e123_split(cache_dir, "student", split) for split in ("source_train", "source_val", "target_val")}
    importance, increments, usage = [], [], []
    for pair in _configured_pairs(task, classes):
        pair_name = f"{pair['y_key']}__{pair['c_key']}"
        if pair["y_id"] is None or pair["c_id"] is None:
            for anchor in range(16):
                for group in FEATURE_GROUPS:
                    importance.append({"pair": pair_name, "anchor_id": anchor, "feature_group": group, "status": "N/A_missing_class"})
                increments.append({
                    "pair": pair_name, "anchor_id": anchor, "status": "N/A_missing_class",
                    "source_k1_increment": float("nan"), "source_k2_increment": float("nan"),
                    "source_all_phase_increment": float("nan"), "target_k1_increment": float("nan"),
                    "target_k2_increment": float("nan"), "target_all_phase_increment": float("nan"),
                })
            continue
        rows = compare_anchor_probe_groups(
            data["source_train"]["events"], data["source_train"]["label"],
            data["source_val"]["events"], data["source_val"]["label"],
            data["target_val"]["events"], data["target_val"]["label"],
            pair["y_id"], pair["c_id"],
        )
        source_stats = {row["anchor_id"]: row for row in pairwise_event_statistics(
            data["source_train"]["events"], data["source_train"]["label"],
            pair["y_id"], pair["c_id"], "source_train", "student", seed,
        )}
        target_stats = {row["anchor_id"]: row for row in pairwise_event_statistics(
            data["target_val"]["events"], data["target_val"]["label"],
            pair["y_id"], pair["c_id"], "target_val", "student", seed,
        )}
        for row in rows:
            row.update({"pair": pair_name, "y_name": pair["y_name"], "c_name": pair["c_name"]})
            source_stat, target_stat = source_stats[row["anchor_id"]], target_stats[row["anchor_id"]]
            row.update({
                "source_effect_norm": source_stat.get("effect_norm", np.nan),
                "source_projection_ci_low": source_stat.get("projection_ci_low", np.nan),
                "source_projection_ci_high": source_stat.get("projection_ci_high", np.nan),
                "target_effect_norm": target_stat.get("effect_norm", np.nan),
                "target_projection_ci_low": target_stat.get("projection_ci_low", np.nan),
                "target_projection_ci_high": target_stat.get("projection_ci_high", np.nan),
                "source_target_effect_cosine": _cosine(
                    np.asarray([source_stat.get(f"effect_{name}", np.nan) for name in EVENT_COMPONENTS]),
                    np.asarray([target_stat.get(f"effect_{name}", np.nan) for name in EVENT_COMPONENTS]),
                ),
            })
            for component in EVENT_COMPONENTS:
                row[f"source_effect_{component}"] = source_stat.get(f"effect_{component}", np.nan)
                row[f"target_effect_{component}"] = target_stat.get(f"effect_{component}", np.nan)
        importance.extend(rows)
        by_anchor = defaultdict(dict)
        for row in rows:
            by_anchor[row["anchor_id"]][row["feature_group"]] = row
        for anchor, grouped in by_anchor.items():
            sc, k1, k2, scp = grouped["SC"], grouped["SC_K1"], grouped["SC_K2"], grouped["SCP"]
            increments.append({
                "pair": pair_name, "anchor_id": anchor, "status": scp["status"],
                "source_k1_increment": k1.get("source_val_balanced_accuracy", np.nan) - sc.get("source_val_balanced_accuracy", np.nan),
                "source_k2_increment": k2.get("source_val_balanced_accuracy", np.nan) - sc.get("source_val_balanced_accuracy", np.nan),
                "source_all_phase_increment": scp.get("source_val_balanced_accuracy", np.nan) - sc.get("source_val_balanced_accuracy", np.nan),
                "target_k1_increment": k1.get("target_balanced_accuracy", np.nan) - sc.get("target_balanced_accuracy", np.nan),
                "target_k2_increment": k2.get("target_balanced_accuracy", np.nan) - sc.get("target_balanced_accuracy", np.nan),
                "target_all_phase_increment": scp.get("target_balanced_accuracy", np.nan) - sc.get("target_balanced_accuracy", np.nan),
            })
        valid = [row for row in increments if row["pair"] == pair_name and row["status"] == "ok"]
        if valid:
            chosen = max(valid, key=lambda row: row["source_all_phase_increment"])["anchor_id"]
            selected = np.concatenate([
                np.flatnonzero(data["target_val"]["label"] == class_id)[:2]
                for class_id in (pair["y_id"], pair["c_id"])
            ])
            expected = np.asarray([
                source_stats[chosen].get(f"effect_{name}", np.nan)
                for name in EVENT_COMPONENTS
            ], dtype=float)
            for index in selected:
                batch = _batch_for_parcels(datasets["target_val"], [data["target_val"]["parcel_index"][index]])
                result = normal_pair_margin_sensitivity(model, batch, pair["y_id"], pair["c_id"], temporal_shift=0)
                gradient = result["response_gradient"][0, chosen]
                usage.append({
                    "pair": pair_name, "anchor_id": chosen,
                    "parcel_index": int(data["target_val"]["parcel_index"][index]),
                    "true_class": classes[int(data["target_val"]["label"][index])],
                    "normal_margin": float(result["margin"][0]),
                    "phase_gradient_norm": float(np.linalg.norm(gradient[2:])),
                    "sc_gradient_norm": float(np.linalg.norm(gradient[:2])),
                    "source_effect_gradient_cosine": _cosine(expected, gradient),
                    "source_phase_effect_gradient_cosine": _cosine(expected[2:], gradient[2:]),
                })
    _write_csv(output_dir / "E2_anchor_phase_importance.csv", importance)
    _write_csv(output_dir / "E2_phase_increment.csv", increments)
    _write_csv(output_dir / "E2_model_usage.csv", usage)
    comparison = []
    for pair_name in sorted({row["pair"] for row in increments}):
        current = [row for row in increments if row["pair"] == pair_name]
        values = [row["target_all_phase_increment"] for row in current if np.isfinite(row["target_all_phase_increment"])]
        comparison.append({"scope": "checkpoint_task_pair", "task": task, "pair": pair_name, "anchor_support": len(values), "mean_target_phase_increment": float(np.mean(values)) if values else float("nan")})
    _write_csv(output_dir / "E2_cross_task_comparison.csv", comparison)
    positive = sum(np.isfinite(row.get("target_all_phase_increment", np.nan)) and row["target_all_phase_increment"] > 0 for row in increments)
    transferable = [f"{row['pair']}:A{row['anchor_id']}" for row in increments if row["source_all_phase_increment"] > .02 and row["target_all_phase_increment"] > .02]
    source_only = [f"{row['pair']}:A{row['anchor_id']}" for row in increments if row["source_all_phase_increment"] > .02 and np.isfinite(row["target_all_phase_increment"]) and row["target_all_phase_increment"] <= 0]
    shape_rows = _read_csv(output_dir / "E1_domain_correspondence.csv")
    finite_shape = [_float(row.get("full_shape_rms")) for row in shape_rows if np.isfinite(_float(row.get("full_shape_rms")))]
    shape_cutoff = float(np.median(finite_shape)) if finite_shape else float("nan")
    stable_no_phase = [
        f"{row['pair']}:A{row['anchor_id']}" for row in increments
        if np.isfinite(shape_cutoff)
        and _float(next((item.get("full_shape_rms") for item in shape_rows if int(item["anchor_id"]) == int(row["anchor_id"])), np.nan)) <= shape_cutoff
        and np.isfinite(row["target_all_phase_increment"]) and row["target_all_phase_increment"] <= .02
    ]
    underused = [
        f"{row['pair']}:A{row['anchor_id']}" for row in usage
        if row["phase_gradient_norm"] <= row["sc_gradient_norm"]
    ]
    (output_dir / "E2_report.md").write_text(
        "# E2 — Anchor–Phase importance\n\n"
        f"- Target-positive Anchor/pair Phase increments: {positive}/{len(increments)}.\n"
        f"- Source-and-Target transferable Phase events (>0.02): {transferable or ['none']}.\n"
        f"- Source-only Phase events: {source_only or ['none']}.\n"
        f"- Relatively stable morphology without Target Phase increment: {stable_no_phase or ['none']}.\n"
        f"- Sampled events whose Phase sensitivity does not exceed SC sensitivity: {underused or ['none']}.\n"
        "- SC, first-harmonic, second-harmonic and full SCP probes use identical samples.\n"
        "- Probe increments are predictive diagnostics, not causal model contributions; sampled normal-Query sensitivities are reported separately.\n",
        encoding="utf-8",
    )


def _safe_auc(labels, score, metric):
    labels = np.asarray(labels, dtype=int); score = np.asarray(score, dtype=float)
    valid = np.isfinite(score)
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return float("nan")
    return float(metric(labels[valid], score[valid]))


def within_prediction_metrics(labels, prediction, scores, classes):
    labels, prediction = np.asarray(labels), np.asarray(prediction)
    rows = []
    for class_id, class_name in enumerate(classes):
        selected = prediction == class_id
        if not selected.any():
            continue
        correct = labels[selected] == class_id
        error = ~correct
        for method, values in scores.items():
            values = np.asarray(values)[selected]
            reliability = values if method != "structure_conflict" else -values
            rows.append({
                "predicted_class": class_name, "predicted_class_id": class_id,
                "method": method, "support": int(selected.sum()),
                "correct_support": int(correct.sum()), "error_support": int(error.sum()),
                "correct_auroc": _safe_auc(correct, reliability, roc_auc_score),
                "error_auprc": _safe_auc(error, -reliability, average_precision_score),
            })
    return rows


def rejection_tradeoff(labels, prediction, scores, classes, points=20):
    rows = []
    labels, prediction = np.asarray(labels), np.asarray(prediction)
    lookup = {str(name).lower().replace(" ", "_"): i for i, name in enumerate(classes)}
    spring, winter = lookup.get("spring_barley"), lookup.get("winter_wheat")
    for predicted in np.unique(prediction):
        scope = prediction == predicted
        indices = np.flatnonzero(scope)
        for method, score in scores.items():
            score = np.nan_to_num(np.asarray(score, dtype=float), nan=-np.inf)
            order = indices[np.argsort(-score[indices], kind="stable")]
            for count in sorted(set(np.linspace(1, len(order), min(points, len(order)), dtype=int))):
                accepted = order[:count]
                spring_errors = (labels[indices] == spring) & (prediction[indices] == winter) if spring is not None and winter is not None else np.zeros(len(indices), dtype=bool)
                correct_winter = (labels[indices] == winter) & (prediction[indices] == winter) if winter is not None else np.zeros(len(indices), dtype=bool)
                accepted_spring = ((labels[accepted] == spring) & (prediction[accepted] == winter)).sum() if spring is not None and winter is not None else 0
                accepted_winter = ((labels[accepted] == winter) & (prediction[accepted] == winter)).sum() if winter is not None else 0
                rows.append({
                    "predicted_class": classes[int(predicted)], "method": method,
                    "accepted": int(count), "eligible": int(len(indices)),
                    "coverage": float(count / len(indices)),
                    "error_rate": float((labels[accepted] != predicted).mean()),
                    "spring_error_eligible": int(spring_errors.sum()) if spring is not None and winter is not None else "",
                    "spring_error_retained": int(accepted_spring) if spring is not None and winter is not None else "",
                    "spring_error_rejection_rate": float(1. - accepted_spring / spring_errors.sum()) if spring_errors.any() else float("nan"),
                    "correct_winter_eligible": int(correct_winter.sum()) if winter is not None else "",
                    "correct_winter_retained": int(accepted_winter) if winter is not None else "",
                    "correct_winter_retention_rate": float(accepted_winter / correct_winter.sum()) if correct_winter.any() else float("nan"),
                })
    return rows


def class_support_matrix(calibration, events, class_count):
    events = np.asarray(events, dtype=float)
    totals = np.zeros((len(events), class_count), dtype=float)
    counts = np.zeros((len(events), class_count), dtype=int)
    for candidate in calibration.get("candidates", []):
        y, c, anchor = int(candidate["pair_y"]), int(candidate["pair_c"]), int(candidate["anchor_id"])
        effect = np.asarray(candidate["effect"], dtype=float)
        unit = effect / max(np.linalg.norm(effect), 1e-12)
        midpoint = .5 * (np.asarray(candidate["mean_y"]) + np.asarray(candidate["mean_c"]))
        scale = np.maximum(np.asarray(candidate["pooled_std"]), 1e-6)
        signed = ((events[:, anchor] - midpoint) / scale) @ unit
        totals[:, y] += signed; counts[:, y] += 1
        totals[:, c] -= signed; counts[:, c] += 1
    return np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0), counts


def fit_teacher_calibration_from_cache(cache, task, classes, seed=1):
    candidates = []
    for pair in _configured_pairs(task, classes):
        if pair["y_id"] is None or pair["c_id"] is None:
            continue
        train_rows = pairwise_event_statistics(
            cache["source_train"]["events"], cache["source_train"]["label"],
            pair["y_id"], pair["c_id"], "source_train", "teacher", seed,
        )
        val_rows = pairwise_event_statistics(
            cache["source_val"]["events"], cache["source_val"]["label"],
            pair["y_id"], pair["c_id"], "source_val", "teacher", seed,
        )
        selected = select_source_only_candidates(
            train_rows, val_rows, cache["source_train"]["events"],
            cache["target_train"]["events"], min_support=10,
            min_effect=.5, min_direction_cosine=.25, min_target_range=.5,
            role="teacher", pair=(pair["y_id"], pair["c_id"]), seed=seed,
        )
        for row in selected:
            row["pair"] = f"{pair['y_key']}__{pair['c_key']}"
        candidates.extend(selected)
    calibration = fit_source_structure_support(candidates)
    source_scores = score_structure_support(
        calibration, cache["source_val"]["events"],
        cache["source_val"]["prediction"],
    )
    threshold = calibrate_source_threshold(
        source_scores,
        cache["source_val"]["prediction"] == cache["source_val"]["label"],
    )
    calibration.update(threshold)
    calibration["threshold_fit_split"] = threshold["fit_split"]
    calibration["fit_split"] = "source_train"
    calibration["selection_uses_target_truth"] = False
    return calibration


def positive_evidence_rows(support, labels, predictions, confidence, parcels, domain, classes):
    support = np.asarray(support, dtype=float)
    rows = []
    for index in range(len(support)):
        finite = np.flatnonzero(np.isfinite(support[index]))
        order = finite[np.argsort(-support[index, finite])] if len(finite) else np.asarray([], dtype=int)
        proposed = int(order[0]) if len(order) else -1
        rows.append({
            "domain": str(domain), "parcel_index": int(parcels[index]),
            "teacher_prediction": classes[int(predictions[index])],
            "teacher_confidence": float(confidence[index]),
            "true_class": classes[int(labels[index])] if labels is not None else "",
            "proposed_class": classes[proposed] if proposed >= 0 else "",
            "competitor_count": int(len(finite)),
            "eligible_multi_competitor": bool(len(finite) >= 3),
            "positive_margin": float(support[index, order[0]] - support[index, order[1]]) if len(order) >= 2 else float("nan"),
            "proposed_correct": bool(proposed == int(labels[index])) if labels is not None and proposed >= 0 else "",
            "teacher_low_confidence_or_wrong": bool(float(confidence[index]) < .9 or (labels is not None and int(predictions[index]) != int(labels[index]))),
        })
    return rows


def run_e3_stage(cache_dir, output_dir, task, classes, abc_checkpoint_dir, seed=1):
    output_dir = Path(output_dir)
    calibration_path = Path(abc_checkpoint_dir) / "B_source_calibration.json"
    candidates_path = Path(abc_checkpoint_dir) / "A_anchor_candidates.json"
    teacher_cache = {
        split: load_e123_split(cache_dir, "teacher", split)
        for split in ("source_train", "source_val", "target_train", "target_val")
    }
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    else:
        packet = json.loads(candidates_path.read_text(encoding="utf-8")) if candidates_path.is_file() else {"roles": {}}
        candidates = packet.get("roles", {}).get("teacher", [])
        calibration = (
            {"status": "uncalibrated", "candidates": candidates, "structure_threshold": float("nan")}
            if candidates else fit_teacher_calibration_from_cache(teacher_cache, task, classes, seed)
        )
    all_rows, val_scores = [], None
    datasets = {}
    for split in ("target_train", "target_val"):
        data = teacher_cache[split]; datasets[split] = data
        scored = score_structure_support(calibration, data["events"], data["prediction"])
        support, counts = class_support_matrix(calibration, data["events"], len(classes))
        if split == "target_val":
            val_scores = scored
        for index in range(len(data["parcel_index"])):
            predicted_id = int(data["prediction"][index])
            alternatives = np.delete(support[index], predicted_id)
            finite_alternatives = alternatives[np.isfinite(alternatives)]
            predicted_support = support[index, predicted_id]
            relative_margin = (
                float(predicted_support - np.max(finite_alternatives))
                if np.isfinite(predicted_support) and len(finite_alternatives) else float("nan")
            )
            row = {
                "split": split, "domain": str(data["domain"][index]),
                "parcel_index": int(data["parcel_index"][index]),
                "teacher_prediction_id": predicted_id,
                "teacher_prediction": classes[predicted_id],
                "teacher_confidence": float(data["confidence"][index]),
                "anchor_reliability": float(data["anchor_reliability"][index]),
                "original_structure_support": float(scored["margin"][index]),
                "class_relative_structure_margin": relative_margin,
                "structure_margin": float(scored["margin"][index]),
                "source_range_fraction": float(scored["range_fraction"][index]),
                "structure_available": bool(scored["available"][index]),
                "structure_conflict": bool(scored["conflict"][index]),
                "structure_out_of_range": bool(scored["out_of_range"][index]),
                "evidence_count": int(scored["evidence_count"][index]),
                "true_class": classes[int(data["label"][index])] if "label" in data else "",
                "true_class_id": int(data["label"][index]) if "label" in data else "",
            }
            all_rows.append(row)
    _write_csv(output_dir / "E3_all_teacher_scores.csv", all_rows)
    val = datasets["target_val"]
    metrics_scores = {
        "teacher_confidence": val["confidence"],
        "original_structure_support": np.asarray([
            row["original_structure_support"] for row in all_rows if row["split"] == "target_val"
        ]),
        "class_relative_structure_margin": np.asarray([
            row["class_relative_structure_margin"] for row in all_rows if row["split"] == "target_val"
        ]),
        "anchor_reliability": val["anchor_reliability"],
    }
    within = within_prediction_metrics(val["label"], val["prediction"], metrics_scores, classes)
    tradeoff = rejection_tradeoff(val["label"], val["prediction"], metrics_scores, classes)
    _write_csv(output_dir / "E3_within_prediction_reliability.csv", within)
    _write_csv(output_dir / "E3_rejection_tradeoff.csv", tradeoff)
    threshold = float(calibration.get("structure_threshold", np.nan))
    component_rows = []
    for split in ("source_val", "target_val"):
        current = teacher_cache[split]
        current_scores = score_structure_support(calibration, current["events"], current["prediction"])
        for name, mask in (
            ("available", current_scores["available"]),
            ("source_range_failure", current_scores["out_of_range"]),
            ("conflict", current_scores["conflict"]),
            ("aggregate_below_threshold", current_scores["available"] & (current_scores["margin"] < threshold) if np.isfinite(threshold) else np.zeros(len(current["prediction"]), bool)),
        ):
            component_rows.append({
                "split": split, "component": name, "support": len(current["prediction"]),
                "triggered": int(np.asarray(mask).sum()), "rate": float(np.asarray(mask).mean()),
                "source_val_threshold": threshold,
            })
    _write_csv(output_dir / "E3_score_components.csv", component_rows)
    target_failures = [
        row for row in component_rows
        if row["split"] == "target_val" and row["component"] != "available"
    ]
    dominant_failure = max(target_failures, key=lambda row: row["rate"])["component"] if target_failures else "N/A"
    negative = []
    for row in all_rows:
        if row["split"] == "target_val" and row["teacher_confidence"] >= .9:
            negative.append({**row, "negative_evidence": bool(row["structure_conflict"] or row["structure_out_of_range"] or (np.isfinite(threshold) and row["structure_margin"] < threshold))})
    _write_csv(output_dir / "E3_negative_evidence.csv", negative)
    support, _ = class_support_matrix(calibration, val["events"], len(classes))
    positive = positive_evidence_rows(
        support, val["label"], val["prediction"], val["confidence"],
        val["parcel_index"], str(val["domain"][0]), classes,
    )
    _write_csv(output_dir / "E3_positive_evidence.csv", positive)
    winter = [row for row in within if row["predicted_class"].lower().replace(" ", "_") == "winter_wheat"]
    flagged = [row for row in negative if row["negative_evidence"]]
    negative_precision = float(np.mean([row["teacher_prediction"] != row["true_class"] for row in flagged])) if flagged else float("nan")
    correct = [row for row in negative if row["teacher_prediction"] == row["true_class"]]
    correct_retention = float(np.mean([not row["negative_evidence"] for row in correct])) if correct else float("nan")
    positive_scope = [row for row in positive if row["eligible_multi_competitor"] and row["teacher_low_confidence_or_wrong"]]
    positive_precision = float(np.mean([row["proposed_correct"] for row in positive_scope])) if positive_scope else float("nan")
    (output_dir / "E3_report.md").write_text(
        "# E3 — Class-conditional Teacher reliability\n\n"
        f"- Within-Winter-Wheat metric rows: {len(winter)}.\n"
        f"- Source-calibrated candidate status: {calibration.get('status', 'unknown')}.\n"
        f"- Dominant Target Val gate component: {dominant_failure}.\n"
        f"- High-confidence negative-evidence precision: {negative_precision}; correct-pseudo-label retention: {correct_retention}.\n"
        f"- Multi-competitor positive-evidence precision within low-confidence/error scope: {positive_precision} (n={len(positive_scope)}).\n"
        "- Error identification, correct-supervision retention, and new weak-class discovery are reported separately.\n"
        "- Positive evidence requires support against multiple represented competitors; opposition to Winter Wheat alone is not treated as Spring Barley support.\n"
        "- Target Val truth is evaluation-only; Target Train rows contain no truth.\n",
        encoding="utf-8",
    )


def choose_e123_decision(evidence):
    if evidence.get("negative_reliability") and evidence.get("positive_reliability"):
        return "target_supervision_filtering"
    if evidence.get("phase_transfer"):
        return "anchor_phase_semantics"
    if evidence.get("anchor_problem"):
        return "anchor_formation"
    return "insufficient_evidence"


def stage_complete(output_dir, stage):
    return all((Path(output_dir) / name).is_file() for name in required_stage_artifacts(stage))


def _stage_state(path, stage, status, error=None, metadata=None):
    path = Path(path)
    packet = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"stages": {}}
    packet["stages"][stage] = {"status": status, "updated_at": datetime.now(timezone.utc).astimezone().isoformat(), **(metadata or {})}
    if error is not None:
        packet["stages"][stage]["error"] = error
    _write_json(path, packet)


def write_checkpoint_report(output_dir, variant, task):
    output_dir = Path(output_dir)
    bodies = []
    for stage in ("E1", "E2", "E3"):
        path = output_dir / f"{stage}_report.md"
        bodies.append(path.read_text(encoding="utf-8") if path.is_file() else f"# {stage}\n\nN/A")
    path = output_dir / "checkpoint_report.md"
    path.write_text(f"# {variant} {task} — E1/E2/E3\n\n" + "\n\n".join(bodies), encoding="utf-8")
    return path


def write_pair_report(root, task):
    root = Path(root); path = root / "pair_reports" / f"{task}.md"; path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# E/G E123 comparison — {task}", "", "Anchor IDs remain checkpoint-local.", ""]
    for variant in ("E", "G"):
        folder = root / f"{variant}_{task}"
        lines.append(f"## {variant}\n\n" + (folder / "checkpoint_report.md").read_text(encoding="utf-8"))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(root):
    root = Path(root)
    phase_values, phase_group_means = [], []
    negative_precision, correct_retention, positive_precision = [], [], []
    anchor_mismatch, anchor_issue_groups = [], []
    cross_task_rows = []
    for variant in ("E", "G"):
        for task in PAIR_NAMES:
            folder = root / f"{variant}_{task}"
            for row in _read_csv(folder / "E2_phase_increment.csv"):
                value = _float(row.get("target_all_phase_increment"));
                if np.isfinite(value): phase_values.append(value)
            for row in _read_csv(folder / "E2_cross_task_comparison.csv"):
                cross_task_rows.append({"variant": variant, "task": task, **row})
                value = _float(row.get("mean_target_phase_increment"))
                if np.isfinite(value): phase_group_means.append(value)
            negatives = _read_csv(folder / "E3_negative_evidence.csv")
            if negatives:
                flagged = [row for row in negatives if row.get("negative_evidence", "").lower() == "true"]
                correct = [row for row in negatives if row.get("teacher_prediction") == row.get("true_class")]
                if flagged:
                    negative_precision.append(np.mean([row.get("teacher_prediction") != row.get("true_class") for row in flagged]))
                if correct:
                    correct_retention.append(np.mean([row.get("negative_evidence", "").lower() != "true" for row in correct]))
            positives = [
                row for row in _read_csv(folder / "E3_positive_evidence.csv")
                if row.get("eligible_multi_competitor", "").lower() == "true"
                and row.get("teacher_low_confidence_or_wrong", "").lower() == "true"
            ]
            if positives:
                positive_precision.append(np.mean([row.get("proposed_correct", "").lower() == "true" for row in positives]))
            for row in _read_csv(folder / "E1_domain_correspondence.csv"):
                value = _float(row.get("full_shape_rms"));
                if np.isfinite(value): anchor_mismatch.append(value)
            formation = [row for row in _read_csv(folder / "E1_anchor_semantics.csv") if row.get("split") == "source_train"]
            if formation:
                concerns = [
                    row for row in formation
                    if int(float(row.get("support", 0))) == 0
                    or _float(row.get("max_other_anchor_shape_cosine")) > .98
                ]
                anchor_issue_groups.append(len(concerns) / len(formation))
    evidence = {
        "anchor_problem": len(anchor_issue_groups) >= 2 and np.mean(anchor_issue_groups) > .25,
        "phase_transfer": len(phase_group_means) >= 2 and np.mean(phase_group_means) > .02 and np.mean(np.asarray(phase_group_means) > 0.) >= .75,
        "negative_reliability": len(negative_precision) >= 2 and len(correct_retention) >= 2 and np.mean(negative_precision) > .6 and np.mean(correct_retention) > .7,
        "positive_reliability": len(positive_precision) >= 2 and np.mean(positive_precision) > .6,
    }
    decision = choose_e123_decision(evidence)
    _write_csv(root / "E2_cross_task_comparison.csv", cross_task_rows)
    labels = {
        "anchor_formation": "Anchor 形成约束值得进入训练验证",
        "anchor_phase_semantics": "Anchor–Phase 类别相对语义需要进一步优化",
        "target_supervision_filtering": "结构证据已具备 Target 监督筛选价值",
        "insufficient_evidence": "现有证据不足，暂不增加训练约束",
    }
    (root / "E123_summary.md").write_text(
        "# E123 summary\n\n"
        f"## 1. E1\n\nMean cross-domain full-shape RMS: {np.mean(anchor_mismatch) if anchor_mismatch else np.nan}; repeated formation-concern rate: {np.mean(anchor_issue_groups) if anchor_issue_groups else np.nan}.\n\n"
        f"## 2. E2\n\nMean Target Phase increment: {np.mean(phase_values) if phase_values else np.nan}; checkpoint/pair-level mean: {np.mean(phase_group_means) if phase_group_means else np.nan}.\n\n"
        f"## 3. E3\n\nNegative-evidence precision: {np.mean(negative_precision) if negative_precision else np.nan}; correct-label retention: {np.mean(correct_retention) if correct_retention else np.nan}; low-confidence/error positive-evidence precision: {np.mean(positive_precision) if positive_precision else np.nan}.\n\n"
        "## 4. Joint evidence\n\nEvidence is aggregated at pair/task level; Anchor IDs are never aligned across checkpoints.\n\n"
        f"## 5. Decision\n\n**{labels[decision]}** (`{decision}`)\n\n"
        f"Supporting/contrary flags: {evidence}. A direction is enabled only by repeated checkpoint/pair evidence plus minimum effect/retention criteria, never a single correlation. Evidence-insufficient boundaries include targeted Target Val reuse, unavailable classes, and local rather than causal sensitivity. No E4 is implemented.\n",
        encoding="utf-8",
    )


def _read_csv(path):
    if not Path(path).is_file(): return []
    with Path(path).open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def _float(value):
    try: return float(value)
    except (TypeError, ValueError): return float("nan")


def run_stage(args):
    output = Path(args.audit_root) / f"{args.variant}_{args.task}"; output.mkdir(parents=True, exist_ok=True)
    state = output / "stage_status.json"
    if stage_complete(output, args.stage): return
    _stage_state(state, args.stage, "running")
    started = time.monotonic()
    measure_memory = str(args.device).startswith("cuda") and torch.cuda.is_available()
    if measure_memory:
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    try:
        if args.stage == "extract":
            frozen = load_frozen_models(args.checkpoint, args.task, args.variant, args.device)
            source = validate_source(args.source_checkpoint, args.task, "cpu")
            checkpoint_digest = sha256_file(args.checkpoint)
            abc_dir = Path(args.abc_root) / f"{args.variant}_{args.task}"
            abc_protocol = reused_abc_protocol(abc_dir, checkpoint_digest)
            config, protocol, _, datasets = build_abc_datasets(frozen.config, args.data_root, args.seed)
            if abc_protocol is not None:
                datasets = replay_abc_datasets(
                    config, args.data_root, args.seed,
                    abc_protocol["split_manifest"],
                )
                protocol = abc_protocol.get("protocol", protocol)
            _, reference = prepare_feature_source(
                frozen, datasets, {"source": config.source, "target": config.target}, output,
                abc_checkpoint_dir=abc_dir, device=args.device, batch_size=args.batch_size,
                pixel_budget=args.pixel_budget, representative_count=args.representative_count,
            )
            split_manifest = (
                abc_protocol["split_manifest"] if abc_protocol is not None else
                {split: [{"domain": config.source if split.startswith("source") else config.target, "parcel_index": int(p)} for p in dataset.get_parcel_indices()] for split, dataset in datasets.items()}
            )
            _write_json(output / "checkpoint_manifest.json", {
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_digest,
                "source_checkpoint": source, "classes": list(config.classes), "protocol": protocol,
                "split_manifest": split_manifest, "cache_reference": reference,
                "target_test_used": False, "seed": args.seed,
            })
        else:
            frozen = load_frozen_models(args.checkpoint, args.task, args.variant, args.device if args.stage in ("E1", "E2") else "cpu") if args.stage in ("E1", "E2") else None
            manifest = json.loads((output / "checkpoint_manifest.json").read_text(encoding="utf-8"))
            classes = manifest["classes"]; root = cache_root(output)
            if args.stage == "E1": run_e1_stage(frozen.student, root, output, classes, args.seed, Path(args.abc_root) / f"{args.variant}_{args.task}")
            elif args.stage == "E2":
                datasets = replay_abc_datasets(frozen.config, args.data_root, args.seed, manifest["split_manifest"])
                run_e2_stage(frozen.student, datasets, root, output, args.task, classes, args.seed)
            elif args.stage == "E3": run_e3_stage(root, output, args.task, classes, Path(args.abc_root) / f"{args.variant}_{args.task}", args.seed)
            else: write_checkpoint_report(output, args.variant, args.task)
        if not stage_complete(output, args.stage): raise RuntimeError(f"missing {args.stage} artifacts")
        metadata = {"elapsed_seconds": time.monotonic() - started}
        if measure_memory:
            metadata["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(torch.device(args.device)))
        _stage_state(state, args.stage, "complete", metadata=metadata)
    except Exception as error:
        _stage_state(state, args.stage, "failed", str(error), {"traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started})
        raise


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("checkpoint", "pair-report", "summary"), required=True)
    parser.add_argument("--stage", choices=STAGE_ORDER); parser.add_argument("--task", choices=tuple(PAIR_NAMES))
    parser.add_argument("--variant", choices=("E", "G")); parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-checkpoint", type=Path); parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--abc-root", type=Path); parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--representative-count", type=int, default=8)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "checkpoint": run_stage(args)
    elif args.mode == "pair-report": write_pair_report(args.audit_root, args.task)
    else: write_summary(args.audit_root)


if __name__ == "__main__":
    main()
