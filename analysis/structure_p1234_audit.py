#!/usr/bin/env python3
"""Frozen P1-P4 feasibility audits built on the ABC/E123 feature protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

TASKS = {
    "FR1_FR2": ("FR1", "france/30TXT/2017", "FR2", "france/31TCJ/2017"),
    "FR2_DK1": ("FR2", "france/31TCJ/2017", "DK1", "denmark/32VNH/2017"),
}
ALLOWED_AUDIT_SPLITS = ("source_train", "source_val", "target_train", "target_val")
ALLOWED_SPLITS = ALLOWED_AUDIT_SPLITS
FEATURE_GROUPS = {
    "SC": (0, 1), "SC_K1": (0, 1, 2, 3),
    "SC_K2": (0, 1, 4, 5), "SCP": (0, 1, 2, 3, 4, 5),
}
GROUPS = ("SC", "SC_K1", "SC_K2", "SCP")
STAGES = ("P1", "P2", "P3", "P4", "report")
STAGE_ARTIFACTS = {
    "P1": ("P1_component_comparison.csv", "P1_pairwise_phase_increment.csv", "P1_sb_ww_recovery.csv", "P1_selection.json", "P1_report.md"),
    "P2": ("P2_available_checkpoints.csv", "P2_stage_reliability.csv", "P2_source_calibration.csv", "P2_report.md"),
    "P3": ("P3_decision_rules.json", "P3_target_train_decisions.csv", "P3_target_val_decisions.csv", "P3_reject_correction_comparison.csv", "P3_per_class_impact.csv", "P3_report.md"),
    "P4": ("P4_branch_sensitivity.csv", "P4_anchor_matching_changes.csv", "P4_phase_propagation.csv", "P4_e2_association.csv", "P4_report.md"),
    "report": ("checkpoint_report.md",),
}


def feature_indices(anchor_id, group):
    if group not in FEATURE_GROUPS: raise ValueError(f"unknown feature group: {group}")
    anchor = int(anchor_id)
    if not 0 <= anchor < 16: raise ValueError("anchor id must be in [0,15]")
    return [component * 16 + anchor for component in FEATURE_GROUPS[group]]


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""): digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def _digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def validate_p1_comparability(rows):
    if {row["group"] for row in rows} != set(GROUPS):
        raise ValueError("P1 requires exactly SC/SC_K1/SC_K2/SCP")
    for field in ("sample_key_digest", "candidate_digest"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"P1 groups do not share {field}")
    return True


def select_p1_group(rows):
    validate_p1_comparability(rows)
    valid = [row for row in rows if np.isfinite(float(row["source_val_auroc"]))]
    if not valid:
        return "SC"
    # Target metrics are deliberately absent from the selection key.
    return max(valid, key=lambda row: (float(row["source_val_auroc"]), -GROUPS.index(row["group"])))["group"]


def _project_candidate(candidate, components):
    row = dict(candidate)
    for name in ("effect", "mean_y", "mean_c", "pooled_std"):
        row[name] = np.asarray(candidate[name], dtype=float)[list(components)].tolist()
    joint = dict(candidate["joint_support"])
    indices = np.asarray(components, dtype=int)
    joint["center"] = np.asarray(joint["center"])[indices].tolist()
    precision = np.asarray(joint["precision"])[np.ix_(indices, indices)]
    joint["precision"] = precision.tolist()
    joint["marginal_low"] = np.asarray(joint["marginal_low"])[indices].tolist()
    joint["marginal_high"] = np.asarray(joint["marginal_high"])[indices].tolist()
    row["joint_support"] = joint
    return row


def project_calibration(calibration, group):
    components = FEATURE_GROUPS[group]
    result = {key: value for key, value in calibration.items() if key != "candidates"}
    result["feature_group"] = group
    result["candidates"] = [_project_candidate(row, components) for row in calibration.get("candidates", [])]
    return result


def _auc(correct, score):
    correct, score = np.asarray(correct, bool), np.asarray(score, float)
    mask = np.isfinite(score)
    return float(roc_auc_score(correct[mask], score[mask])) if len(np.unique(correct[mask])) == 2 else float("nan")


def _group_packet(calibration, cache, group, classes):
    from analysis.structure_e123_audit import class_support_matrix
    from analysis.structure_semantic_abc import calibrate_source_threshold, score_structure_support
    components = FEATURE_GROUPS[group]
    projected = project_calibration(calibration, group)
    data = {}
    for split, arrays in cache.items():
        events = np.asarray(arrays["events"])[..., list(components)]
        scores = score_structure_support(projected, events, arrays["prediction"])
        support, _ = class_support_matrix(projected, events, len(classes))
        data[split] = (scores, support)
    source_val = cache["source_val"]
    threshold = calibrate_source_threshold(
        data["source_val"][0], source_val["prediction"] == source_val["label"],
    )
    projected.update(threshold)
    return projected, data


def _proposal_metrics(support, labels, class_id):
    finite = np.isfinite(support).any(1)
    safe = np.where(np.isfinite(support), support, -np.inf)
    proposal = safe.argmax(1)
    selected = finite & (proposal == int(class_id))
    truth = np.asarray(labels) == int(class_id)
    tp = int((selected & truth).sum())
    return {
        "proposal_support": int(selected.sum()),
        "proposal_precision": float(tp / selected.sum()) if selected.any() else float("nan"),
        "proposal_recall": float(tp / truth.sum()) if truth.any() else float("nan"),
    }


def run_p1_stage(cache_dir, output_dir, task, classes, seed=1):
    from analysis.structure_e123_audit import fit_teacher_calibration_from_cache
    from analysis.structure_semantic_features import load_feature_split
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    cache = {split: load_feature_split(cache_dir, "teacher", split) for split in ALLOWED_SPLITS}
    calibration = fit_teacher_calibration_from_cache(cache, task, classes, seed)
    key_digest = _digest([(str(d), int(p)) for d, p in zip(cache["target_val"]["domain"], cache["target_val"]["parcel_index"])])
    candidate_digest = _digest(calibration.get("candidates", []))
    spring = classes.index("spring_barley") if "spring_barley" in classes else -1
    winter = classes.index("winter_wheat") if "winter_wheat" in classes else -1
    rows, recovery = [], []
    packets = {}
    for group in GROUPS:
        projected, data = _group_packet(calibration, cache, group, classes)
        packets[group] = (projected, data)
        source_scores = data["source_val"][0]["margin"]
        target_scores, target_support = data["target_val"]
        row = {
            "task": task, "group": group, "sample_key_digest": key_digest,
            "candidate_digest": candidate_digest,
            "source_val_auroc": _auc(cache["source_val"]["prediction"] == cache["source_val"]["label"], source_scores),
            "target_val_auroc": _auc(cache["target_val"]["prediction"] == cache["target_val"]["label"], target_scores["margin"]),
            "source_threshold": projected.get("structure_threshold", np.nan),
            "source_val_calibration_accuracy": projected.get("source_val_accuracy", np.nan),
            **(_proposal_metrics(target_support, cache["target_val"]["label"], spring) if spring >= 0 else {}),
        }
        rows.append(row)
        if spring >= 0 and winter >= 0:
            target = cache["target_val"]
            eligible = (target["confidence"] >= .9) & (target["prediction"] == winter)
            safe = np.where(np.isfinite(target_support), target_support, -np.inf)
            proposed = safe.argmax(1)
            selected = eligible & (proposed == spring)
            recovery.append({
                "task": task, "group": group, "eligible_ww_predictions": int(eligible.sum()),
                "spring_barley_to_winter_wheat_errors": int((eligible & (target["label"] == spring)).sum()),
                "spring_barley_proposals": int(selected.sum()),
                "correct_spring_barley_recovered": int((selected & (target["label"] == spring)).sum()),
                "proposal_precision": float((target["label"][selected] == spring).mean()) if selected.any() else np.nan,
            })
    validate_p1_comparability(rows)
    chosen = select_p1_group(rows)
    baseline = next(row for row in rows if row["group"] == "SC")
    increments = [{
        "task": task, "group": row["group"], "selected_by_source_val": row["group"] == chosen,
        "source_val_auroc_increment_vs_sc": row["source_val_auroc"] - baseline["source_val_auroc"],
        "target_val_auroc_increment_vs_sc_descriptive": row["target_val_auroc"] - baseline["target_val_auroc"],
        "selection_uses_target_val": False,
    } for row in rows]
    _write_csv(output_dir / "P1_component_comparison.csv", rows)
    _write_csv(output_dir / "P1_pairwise_phase_increment.csv", increments)
    _write_csv(output_dir / "P1_sb_ww_recovery.csv", recovery)
    (output_dir / "P1_report.md").write_text(
        f"# P1\n\n- Source-Val-selected evidence group: {chosen}.\n"
        "- All groups share samples and Source-only candidate anchors.\n"
        "- Target Val metrics are descriptive and did not select the group.\n", encoding="utf-8")
    _write_json(output_dir / "P1_selection.json", {"group": chosen, "selection_split": "source_val", "target_truth_used": False})
    return chosen


def available_snapshots(source, best, final):
    rows = []
    for role, path in (("source", source), ("best", best), ("final", final)):
        path = Path(path)
        rows.append({
            "role": role, "path": str(path), "available": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else "",
            "selection_role": "descriptive_only" if role == "best" else ("final_state" if role == "final" else "initial_source"),
            "substituted": False,
        })
    return rows


def _load_snapshot_model(path, role, device):
    from train import create_model
    path = Path(path); packet = torch.load(path, map_location="cpu", weights_only=False)
    raw = packet.get("config")
    if raw is None: raise ValueError(f"snapshot config missing: {path}")
    from types import SimpleNamespace
    config = SimpleNamespace(**dict(raw if isinstance(raw, dict) else vars(raw)))
    model = create_model(config)
    state_key = "teacher_state_dict" if role != "source" and "teacher_state_dict" in packet else "state_dict"
    model.load_state_dict(packet[state_key], strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    shift = 0 if role == "source" else int(torch.as_tensor(packet.get("global_temporal_shift", 0)).item())
    return model, config, packet, shift


def _snapshot_cache(snapshot, datasets, domains, output_dir, device, batch_size, pixel_budget):
    from analysis.structure_semantic_features import _extract_split, load_feature_split
    role, path = snapshot["role"], Path(snapshot["path"])
    root = Path(output_dir) / "P2_snapshot_features" / role
    complete = all((root / "features" / f"teacher_{split}.npz").is_file() for split in ALLOWED_SPLITS)
    if complete:
        return {split: load_feature_split(root, "teacher", split) for split in ALLOWED_SPLITS}, "reused_snapshot_cache"
    model, _, packet, shift = _load_snapshot_model(path, role, device)
    for split in ALLOWED_SPLITS:
        arrays = _extract_split(
            model, datasets[split], domains["source" if split.startswith("source") else "target"],
            shift, device, batch_size, pixel_budget, representative_count=0,
            include_labels=split != "target_train",
        )
        feature_dir = root / "features"; feature_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(feature_dir / f"teacher_{split}.npz", **arrays)
    _write_json(root / "snapshot_manifest.json", {
        "role": role, "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
        "epoch": int(packet.get("epoch", -1)), "applied_shift": shift,
        "own_model_forward": True, "target_test_used": False,
    })
    del model
    if str(device).startswith("cuda"): torch.cuda.empty_cache()
    return {split: load_feature_split(root, "teacher", split) for split in ALLOWED_SPLITS}, "extracted_own_snapshot"


def run_p2_stage(cache_dir, output_dir, source_checkpoint, best_checkpoint, final_checkpoint, task, variant, classes,
                 data_root=None, device="cpu", batch_size=64, pixel_budget=8192, seed=1):
    from analysis.structure_e123_audit import fit_teacher_calibration_from_cache
    from analysis.structure_semantic_features import load_feature_split, load_frozen_models, replay_abc_datasets
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = available_snapshots(source_checkpoint, best_checkpoint, final_checkpoint)
    final_cache = {split: load_feature_split(cache_dir, "teacher", split) for split in ALLOWED_SPLITS}
    datasets = domains = None
    if data_root is not None:
        checkpoint_manifest = json.loads((Path(cache_dir) / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        final_frozen = load_frozen_models(final_checkpoint, task, variant, device)
        datasets = replay_abc_datasets(final_frozen.config, data_root, seed, checkpoint_manifest["split_manifest"])
        domains = {"source": final_frozen.config.source, "target": final_frozen.config.target}
        del final_frozen
    reliability, calibration_rows = [], []
    # Final is the only snapshot guaranteed to have an existing protocol-safe cache.
    for row in snapshots:
        row["epoch"] = ""
        if row["available"]:
            packet = torch.load(row["path"], map_location="cpu", weights_only=False)
            row["epoch"] = int(packet.get("epoch", -1))
        if not row["available"]:
            row["feature_status"] = "unavailable"
            continue
        if row["role"] == "final":
            current_cache, row["feature_status"] = final_cache, "reused_verified_final_teacher_cache"
        elif datasets is not None:
            current_cache, row["feature_status"] = _snapshot_cache(row, datasets, domains, output_dir, device, batch_size, pixel_budget)
        else:
            row["feature_status"] = "requires_own_snapshot_extraction"
            continue
        calibration = fit_teacher_calibration_from_cache(current_cache, task, classes, seed)
        for group in GROUPS:
            projected, data = _group_packet(calibration, current_cache, group, classes)
            val = current_cache["target_val"]
            reliability.append({
                "task": task, "snapshot": row["role"], "epoch": row["epoch"], "group": group,
                "target_val_error_auroc": _auc(val["prediction"] != val["label"], -data["target_val"][0]["margin"]),
                "target_train_supported_fraction": float(data["target_train"][0]["available"].mean()),
                "source_calibrated_threshold": projected.get("structure_threshold", np.nan),
                "anchor_ids_cross_snapshot_comparable": False,
            })
            calibration_rows.append({"task": task, "snapshot": row["role"], "group": group, "fit_split": "source_train", "threshold_split": "source_val", "candidate_count": len(projected.get("candidates", []))})
    _write_csv(output_dir / "P2_available_checkpoints.csv", snapshots)
    _write_csv(output_dir / "P2_stage_reliability.csv", reliability)
    _write_csv(output_dir / "P2_source_calibration.csv", calibration_rows)
    missing_extraction = [r["role"] for r in snapshots if r["available"] and r.get("feature_status") == "requires_own_snapshot_extraction"]
    (output_dir / "P2_report.md").write_text(
        "# P2\n\n" +
        f"- Final Teacher score evaluated in its own representation space.\n- Snapshots requiring separate extraction: {missing_extraction}.\n" +
        "- Best is descriptive only and is never an E4 selection state.\n- Missing intermediate epochs were not fabricated or retrained.\n",
        encoding="utf-8")


def supervision_decisions(parcel_index, teacher_prediction, confidence, teacher_margin,
                          structure_proposal, structure_margin, conflict_mask,
                          strict_positive_mask, include_truth=False, true_label=None):
    arrays = [np.asarray(value) for value in (parcel_index, teacher_prediction, confidence, teacher_margin, structure_proposal, structure_margin, conflict_mask, strict_positive_mask)]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("P3 decision arrays must align")
    if include_truth and true_label is None:
        raise ValueError("truth requested without labels")
    rows = []
    for index in range(len(arrays[0])):
        conflict = bool(arrays[6][index])
        positive = bool(arrays[7][index])
        action = "KEEP" if not conflict else ("RELABEL" if positive else "REJECT")
        row = {
            "parcel_index": int(arrays[0][index]), "teacher_prediction": int(arrays[1][index]),
            "confidence": float(arrays[2][index]), "teacher_margin": float(arrays[3][index]),
            "structure_proposal": int(arrays[4][index]), "structure_margin": float(arrays[5][index]),
            "conflict": conflict, "action": action,
            "proposed_label": int(arrays[4][index]) if action == "RELABEL" else (-1 if action == "REJECT" else int(arrays[1][index])),
            "reason": "no_reliable_conflict" if action == "KEEP" else ("strict_positive_validation" if action == "RELABEL" else "conflict_without_positive_validation"),
        }
        if include_truth:
            row["true_label"] = int(np.asarray(true_label)[index])
        rows.append(row)
    return rows


def conflict_ids(rows, correction=True):
    del correction
    return {int(row["parcel_index"]) for row in rows if bool(row["conflict"])}


def summarize_decision_impact(rows):
    result = defaultdict(int)
    for row in rows:
        truth, teacher, action, proposed = int(row["true_label"]), int(row["teacher_prediction"]), row["action"], int(row["proposed_label"])
        if action == "KEEP" and teacher == truth: result["correct_supervision_kept"] += 1
        if action == "REJECT" and teacher != truth: result["wrong_supervision_rejected"] += 1
        if action == "REJECT" and teacher == truth: result["correct_supervision_rejected"] += 1
        if action == "RELABEL" and proposed == truth and teacher != truth: result["wrong_supervision_corrected"] += 1
        if action == "RELABEL" and proposed != truth: result["wrong_relabels"] += 1
    return dict(result)


def apply_conflict_policy(rows, correction):
    """Return an audit-only view of the same conflict set under Reject or Correction."""
    result = []
    for original in rows:
        row = dict(original)
        if row["conflict"] and not correction:
            row.update(action="REJECT", proposed_label=-1, reason="conflict_reject_policy")
        result.append(row)
    return result


def _teacher_margin(logits):
    logits = np.asarray(logits, float); order = np.sort(logits, axis=1)
    return order[:, -1] - order[:, -2]


def run_p3_stage(cache_dir, output_dir, task, classes):
    from analysis.structure_e123_audit import fit_teacher_calibration_from_cache
    from analysis.structure_semantic_features import load_feature_split
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    cache = {split: load_feature_split(cache_dir, "teacher", split) for split in ALLOWED_SPLITS}
    selected = json.loads((output_dir / "P1_selection.json").read_text(encoding="utf-8"))["group"]
    calibration = fit_teacher_calibration_from_cache(cache, task, classes, 1)
    projected, data = _group_packet(calibration, cache, selected, classes)
    source_support = data["source_val"][1]
    source_label = cache["source_val"]["label"]
    source_proposal = np.where(np.isfinite(source_support), source_support, -np.inf).argmax(1)
    source_top = np.sort(np.where(np.isfinite(source_support), source_support, -np.inf), axis=1)
    source_margin = source_top[:, -1] - source_top[:, -2]
    positive_values = source_margin[source_proposal == source_label]
    strict_threshold = float(np.quantile(positive_values, .25)) if len(positive_values) else float("inf")
    rule = {"feature_group": selected, "confidence_threshold": .9, "strict_positive_margin": strict_threshold, "fit_splits": ["source_train", "source_val"], "target_truth_used": False, "actions": ["KEEP", "REJECT", "RELABEL"]}
    _write_json(output_dir / "P3_decision_rules.json", rule)
    tables = {}
    for split in ("target_train", "target_val"):
        arrays = cache[split]; support = data[split][1]
        safe = np.where(np.isfinite(support), support, -np.inf)
        proposal = safe.argmax(1); ordered = np.sort(safe, axis=1); margin = ordered[:, -1] - ordered[:, -2]
        high = arrays["confidence"] >= .9
        conflict = high & np.isfinite(margin) & (proposal != arrays["prediction"])
        strict = conflict & (margin >= strict_threshold)
        rows = supervision_decisions(
            arrays["parcel_index"][high], arrays["prediction"][high], arrays["confidence"][high],
            _teacher_margin(arrays["logits"])[high], proposal[high], margin[high], conflict[high], strict[high],
            include_truth=split == "target_val",
            true_label=arrays.get("label")[high] if split == "target_val" else None,
        )
        tables[split] = rows
        _write_csv(output_dir / f"P3_{split}_decisions.csv", rows)
    reject_rows = apply_conflict_policy(tables["target_val"], correction=False)
    correction_rows = apply_conflict_policy(tables["target_val"], correction=True)
    common = {"task": task, "feature_group": selected, "conflict_count": len(conflict_ids(tables["target_val"]))}
    comparison = [
        {**common, "policy": "conflict_reject", "relabel_enabled": False, **summarize_decision_impact(reject_rows)},
        {**common, "policy": "conflict_correction", "relabel_enabled": True, **summarize_decision_impact(correction_rows)},
    ]
    per_class = []
    for class_id, name in enumerate(classes):
        for policy, rows in (("conflict_reject", reject_rows), ("conflict_correction", correction_rows)):
            subset = [row for row in rows if row["teacher_prediction"] == class_id]
            per_class.append({"task": task, "policy": policy, "teacher_class": name, "support": len(subset), **summarize_decision_impact(subset)})
    _write_csv(output_dir / "P3_reject_correction_comparison.csv", comparison)
    _write_csv(output_dir / "P3_per_class_impact.csv", per_class)
    (output_dir / "P3_report.md").write_text(
        f"# P3\n\n- Source-only evidence group: {selected}.\n- Conflict set is identical for Reject and Correction.\n"
        "- RELABEL requires stricter Source-Val-positive evidence; other conflicts remain REJECT.\n"
        "- Target Train output contains no truth field.\n", encoding="utf-8")


def rotate_phase_moments(phase, angle):
    phase = np.asarray(phase, dtype=float)
    if phase.shape[-1] % 4:
        raise ValueError("phase moments must contain k1 cos/sin and k2 cos/sin blocks")
    anchors = phase.shape[-1] // 4
    blocks = [phase[..., i * anchors:(i + 1) * anchors] for i in range(4)]
    result = []
    for harmonic, cosine, sine in ((1, blocks[0], blocks[1]), (2, blocks[2], blocks[3])):
        theta = harmonic * float(angle)
        result.extend((cosine * np.cos(theta) - sine * np.sin(theta), cosine * np.sin(theta) + sine * np.cos(theta)))
    return np.concatenate(result, axis=-1)


def controlled_window_perturbations(windows, seed=1):
    values = np.asarray(windows, dtype=float); rng = np.random.default_rng(seed)
    std = max(float(values.std()), 1e-6)
    centered = values - values.mean(axis=-2, keepdims=True)
    shape = values.copy(); shape[..., values.shape[-2] // 2, :] += .05 * std
    dynamics = values.copy(); dynamics[..., 1:, :] += .03 * np.diff(values, axis=-2)
    packets = {
        "noise": values + rng.normal(0, .02 * std, values.shape),
        "mean": values + .05 * std,
        "scale": values.mean(axis=-2, keepdims=True) + 1.05 * centered,
        "shape": shape, "dynamics": dynamics,
    }
    return {name: {"windows": changed, "magnitude": float(np.linalg.norm(changed - values) / max(np.linalg.norm(values), 1e-12)), "distribution_in_range": bool(np.max(np.abs(changed - values)) <= .25 * std + 1e-12)} for name, changed in packets.items()}


def _move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _cosine_change(left, right):
    left, right = torch.nn.functional.normalize(left, dim=-1), torch.nn.functional.normalize(right, dim=-1)
    return float((1. - (left * right).sum(-1)).mean().detach().cpu())


@torch.inference_mode()
def _p4_packet(model, batch, split, classes, seed):
    device = next(model.parameters()).device; batch = _move_batch(batch, device)
    output = model.forward_with_temporal_shift(
        batch["pixels"], batch["valid_pixels"], batch["positions"], batch["extra"],
        temporal_shift=0, return_dict=True,
    )
    groups, _, centers = model.structure_branch.window_extractor(output["exposed_curve"], return_centers=True)
    windows = torch.cat(groups, dim=1)
    generator = model.structure_branch.token_generator; dictionary = model.structure_branch.shapelet_dictionary
    baseline_components = generator(windows, return_encoded_components=True)
    baseline_tokens = baseline_components["shape_token"]
    baseline = dictionary.compute_response(baseline_tokens, return_details=True)
    baseline_phase = model.structure_branch.compose_phase_moments(baseline["weights"], centers, 0)
    perturbations = controlled_window_perturbations(windows.detach().cpu().numpy(), seed)
    branch_rows, anchor_rows, phase_rows = [], [], []
    for name, packet in perturbations.items():
        changed_windows = torch.as_tensor(packet["windows"], device=device, dtype=windows.dtype)
        changed_components = generator(changed_windows, return_encoded_components=True)
        changed = dictionary.compute_response(changed_components["shape_token"], return_details=True)
        changed_phase = model.structure_branch.compose_phase_moments(changed["weights"], centers, 0)
        for branch, key in (("raw", "raw_encoded"), ("difference", "diff_encoded"), ("mean", "mean_encoded"), ("std", "std_encoded")):
            branch_rows.append({"split": split, "perturbation": name, "branch": branch, "feature_cosine_change": _cosine_change(baseline_components[key], changed_components[key]), "magnitude": packet["magnitude"], "distribution_in_range": packet["distribution_in_range"], "sample_count": windows.shape[0]})
        anchor_rows.append({
            "split": split, "perturbation": name, "top1_anchor_change_rate": float((baseline["similarity"].argmax(-1) != changed["similarity"].argmax(-1)).float().mean().cpu()),
            "mean_absolute_similarity_change": float((baseline["similarity"] - changed["similarity"]).abs().mean().cpu()),
            "mean_absolute_weight_change": float((baseline["weights"] - changed["weights"]).abs().mean().cpu()),
            "magnitude": packet["magnitude"], "distribution_in_range": packet["distribution_in_range"],
        })
        anchors = baseline["weights"].shape[-1]
        phase_rows.append({
            "split": split, "change_source": "matching_weight", "perturbation": name,
            "phase_k1_l2": float((baseline_phase[:, :2*anchors] - changed_phase[:, :2*anchors]).norm(dim=1).mean().cpu()),
            "phase_k2_l2": float((baseline_phase[:, 2*anchors:] - changed_phase[:, 2*anchors:]).norm(dim=1).mean().cpu()),
        })
    shift_days = 15.; shifted = model.structure_branch.compose_phase_moments(baseline["weights"], centers, shift_days)
    analytic = rotate_phase_moments(baseline_phase.cpu().numpy(), 2. * np.pi * shift_days / model.structure_branch.exposer.period_days)
    anchors = baseline["weights"].shape[-1]
    phase_rows.append({
        "split": split, "change_source": "analytic_time_rotation", "perturbation": "fixed_weights_shift_15d",
        "phase_k1_l2": float(np.linalg.norm(analytic[:, :2*anchors] - baseline_phase.cpu().numpy()[:, :2*anchors], axis=1).mean()),
        "phase_k2_l2": float(np.linalg.norm(analytic[:, 2*anchors:] - baseline_phase.cpu().numpy()[:, 2*anchors:], axis=1).mean()),
        "analytic_model_max_abs_error": float(np.max(np.abs(analytic - shifted.cpu().numpy()))),
    })
    return branch_rows, anchor_rows, phase_rows


def run_p4_stage(output_dir, final_checkpoint=None, task=None, variant=None, cache_dir=None,
                 data_root=None, device="cpu", samples_per_class=1, seed=1, e123_root=None):
    from analysis.structure_semantic_abc import _batch_for_parcels
    from analysis.structure_semantic_features import load_feature_split, load_frozen_models, replay_abc_datasets
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    required = (final_checkpoint, task, variant, cache_dir, data_root)
    if any(value is None for value in required):
        raise ValueError("P4 requires checkpoint, task, variant, cache, and data root")
    frozen = load_frozen_models(final_checkpoint, task, variant, device)
    manifest = json.loads((Path(cache_dir) / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    datasets = replay_abc_datasets(frozen.config, data_root, seed, manifest["split_manifest"])
    cache = {split: load_feature_split(cache_dir, "student", split) for split in ("source_train", "target_val")}
    branch_rows, anchor_rows, phase_rows = [], [], []
    for split in ("source_train", "target_val"):
        arrays = cache[split]; indices = []
        rng = np.random.default_rng(seed)
        for class_id in np.unique(arrays["label"]):
            candidates = np.flatnonzero(arrays["label"] == class_id)
            indices.extend(rng.choice(candidates, min(samples_per_class, len(candidates)), replace=False).tolist())
        parcels = arrays["parcel_index"][np.asarray(sorted(indices))]
        batch = _batch_for_parcels(datasets[split], parcels)
        current = _p4_packet(frozen.student, batch, split, manifest["classes"], seed)
        branch_rows.extend(current[0]); anchor_rows.extend(current[1]); phase_rows.extend(current[2])
    association = []
    e2_path = Path(e123_root or "") / f"{variant}_{task}" / "E2_phase_increment.csv"
    if e2_path.is_file():
        with e2_path.open(newline="", encoding="utf-8") as stream:
            e2 = list(csv.DictReader(stream))
        for branch in ("raw", "difference", "mean", "std"):
            values = [row["feature_cosine_change"] for row in branch_rows if row["branch"] == branch]
            association.append({"task": task, "branch": branch, "mean_sensitivity": float(np.mean(values)), "e2_rows": len(e2), "interpretation": "association_only"})
    else:
        association.append({"task": task, "status": "E2_UNAVAILABLE", "path": str(e2_path)})
    _write_csv(output_dir / "P4_branch_sensitivity.csv", branch_rows)
    _write_csv(output_dir / "P4_anchor_matching_changes.csv", anchor_rows)
    _write_csv(output_dir / "P4_phase_propagation.csv", phase_rows)
    _write_csv(output_dir / "P4_e2_association.csv", association)
    (output_dir / "P4_report.md").write_text(
        "# P4\n\n- Uses stratified real full-window collections; no branch-zeroing result is reused.\n"
        "- Matching-weight changes and fixed-weight analytic phase rotation are reported separately.\n"
        "- Perturbation magnitudes and empirical in-range flags are recorded; associations do not establish causality.\n",
        encoding="utf-8")


def write_checkpoint_report(output_dir):
    output_dir = Path(output_dir)
    text = "# P1-P4 checkpoint report\n\n" + "\n".join(
        f"- {stage}: {'complete' if all((output_dir / p).is_file() for p in STAGE_ARTIFACTS[stage]) else 'incomplete'}"
        for stage in ("P1", "P2", "P3", "P4")
    ) + "\n"
    (output_dir / "checkpoint_report.md").write_text(text, encoding="utf-8")


def write_summary(audit_root):
    root = Path(audit_root); reports = sorted(root.glob("[EG]_*/checkpoint_report.md"))
    def rows(name):
        result = []
        for path in root.glob(f"[EG]_*/{name}"):
            with path.open(newline="", encoding="utf-8") as stream:
                result.extend(csv.DictReader(stream))
        return result
    p1, p2, p3, p4 = rows("P1_pairwise_phase_increment.csv"), rows("P2_stage_reliability.csv"), rows("P3_reject_correction_comparison.csv"), rows("P4_branch_sensitivity.csv")
    selected = [row for row in p1 if row.get("selected_by_source_val", "").lower() == "true"]
    phase_selected = [row for row in selected if row.get("group") != "SC"]
    usable = sorted({row.get("snapshot", "") for row in p2 if row.get("target_val_error_auroc", "") not in ("", "nan")})
    wrong_reject = sum(int(float(row.get("wrong_supervision_rejected", 0) or 0)) for row in p3 if row.get("policy") == "conflict_correction")
    corrected = sum(int(float(row.get("wrong_supervision_corrected", 0) or 0)) for row in p3 if row.get("policy") == "conflict_correction")
    wrong_relabel = sum(int(float(row.get("wrong_relabels", 0) or 0)) for row in p3 if row.get("policy") == "conflict_correction")
    branch_means = {}
    for branch in ("raw", "difference", "mean", "std"):
        values = [float(row["feature_cosine_change"]) for row in p4 if row.get("branch") == branch and row.get("feature_cosine_change") not in (None, "")]
        if values: branch_means[branch] = float(np.mean(values))
    e4_ready = bool(selected) and bool(usable) and wrong_reject > 0 and corrected > wrong_relabel
    (root / "P1234_summary.md").write_text(
        "# P1-P4 frozen audit summary\n\n"
        f"- Completed checkpoint reports: {len(reports)} / 4.\n"
        f"- P1 — Phase independent evidence: {'supported by Source-Val selection' if phase_selected else 'not established'}; selected groups: {[row.get('group') for row in selected]}.\n"
        f"- P2 — Score availability was measured for these real snapshots only: {usable or ['none']}; absent epochs remain unverified.\n"
        f"- P3 — Reliable conflict outcomes: wrong supervision rejected={wrong_reject}, corrected={corrected}, wrong relabels={wrong_relabel}. KEEP/REJECT/RELABEL thresholds are Source-only.\n"
        f"- P4 — Mean branch sensitivity (association only): {branch_means or 'unavailable'}. Matching-induced and analytic-time Phase changes are separate outputs.\n"
        f"- E4 launch condition from P1-P3: {'met for a future independent experiment' if e4_ready else 'not yet met'}; E4 was not implemented or launched.\n"
        "- Target Val results are exploratory evaluation only and were not used to select evidence groups or thresholds.\n",
        encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("checkpoint", "summary"), required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--variant", choices=("E", "G"))
    parser.add_argument("--cache-root")
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--best-checkpoint")
    parser.add_argument("--final-checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--e123-root")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "summary":
        write_summary(args.audit_root); return
    output = Path(args.audit_root) / f"{args.variant}_{args.task}"
    manifest = json.loads((Path(args.cache_root) / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    classes = manifest["classes"]
    if args.stage == "P1": run_p1_stage(args.cache_root, output, args.task, classes)
    elif args.stage == "P2": run_p2_stage(args.cache_root, output, args.source_checkpoint, args.best_checkpoint, args.final_checkpoint, args.task, args.variant, classes, args.data_root, args.device, args.batch_size, args.pixel_budget)
    elif args.stage == "P3": run_p3_stage(args.cache_root, output, args.task, classes)
    elif args.stage == "P4": run_p4_stage(output, args.final_checkpoint, args.task, args.variant, args.cache_root, args.data_root, args.device, args.samples_per_class, e123_root=args.e123_root)
    elif args.stage == "report": write_checkpoint_report(output)


if __name__ == "__main__":
    main()
