#!/usr/bin/env python3
"""Frozen Stage-1 audit of local Window-to-Anchor matching evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from analysis.anchor_matching_calibration_audit import (
    DeterministicViewDataset,
    P0R_PROTOCOL_VERSION,
    VIEW_SEED_VERSION,
    _cluster_mean_interval,
    _window_metadata,
    class_matched_derangements,
    circular_window_iou,
    freeze_for_audit,
    load_source_runtime,
    repeatability_baseline_rows,
    sha256_file,
)
from analysis.structure_pair_semantics_audit import TASKS, load_generic_frozen
from analysis.structure_semantic_features import build_abc_datasets, semantic_loader


STAGE1_PROTOCOL_VERSION = "g-anchor-stage1-v1"
PHASE_MAGNITUDE_QUANTILE = 0.25
ANCHOR_MATCH_POSITIVE_COSINE = 0.0
REQUIRED_OUTPUTS = (
    "view_integrity.csv",
    "anchor_identity_repeatability.csv",
    "anchor_strength_reliability.csv",
    "anchor_temporal_stability.csv",
    "source_uda_identity_comparison.csv",
    "checkpoint_manifest.json",
)


def _write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def pair_compact_views(first, second):
    """Strictly align view packets by the unique (domain, parcel) identity."""
    def index(packet):
        keys = list(zip(map(str, packet["domain"]), map(int, packet["parcel_index"])))
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate (domain, parcel_index) in a view")
        return {key: position for position, key in enumerate(keys)}

    left_index, right_index = index(first), index(second)
    if set(left_index) != set(right_index):
        raise ValueError("independent views do not contain identical parcels")
    keys = sorted(left_index)
    metadata = {"window_starts", "window_scales", "window_centers"}

    def reorder(packet, positions):
        output = {}
        count = len(positions)
        for key, value in packet.items():
            array = np.asarray(value)
            output[key] = (
                array[positions]
                if key not in metadata and array.ndim and len(array) == count
                else array
            )
        return output

    return (
        reorder(first, [left_index[key] for key in keys]),
        reorder(second, [right_index[key] for key in keys]),
    )


def validate_split_isolation(train_parcels, val_parcels):
    overlap = set(map(int, train_parcels)) & set(map(int, val_parcels))
    if overlap:
        raise ValueError(f"Source Train/Val parcel leakage: {sorted(overlap)[:5]}")
    return True


def validate_checkpoint_pair_metadata(source_config, uda_config, uda_epoch):
    checks = {
        "source domain": (getattr(source_config, "source", None), getattr(uda_config, "source", None)),
        "class mapping": (list(getattr(source_config, "classes", ())), list(getattr(uda_config, "classes", ()))),
        "shape representation": (getattr(source_config, "shape_representation", None), "phase_moment"),
        "Source response mode": (getattr(source_config, "phase_response_mode", "softmax"), "softmax"),
        "UDA response mode": (getattr(uda_config, "phase_response_mode", "softmax"), "softmax"),
    }
    failures = {name: values for name, values in checks.items() if values[0] != values[1]}
    if failures:
        raise ValueError(f"Source/UDA checkpoint identity mismatch: {failures}")
    if int(uda_epoch) != 19:
        raise ValueError("G UDA Final checkpoint must be epoch 19")
    return True


def _relative_l2(first, second):
    first = np.asarray(first, dtype=np.float64); second = np.asarray(second, dtype=np.float64)
    return float(np.linalg.norm(second - first) / max(np.linalg.norm(first), 1e-12))


def evaluate_view_integrity(first, second):
    first, second = pair_compact_views(first, second)
    rows = []
    for index, parcel in enumerate(first["parcel_index"]):
        hashes_differ = str(first["input_hash"][index]) != str(second["input_hash"][index])
        positions_a = np.asarray(first["positions"][index])
        positions_b = np.asarray(second["positions"][index])
        valid_a = np.asarray(first["valid_mask"][index]).astype(bool)
        valid_b = np.asarray(second["valid_mask"][index]).astype(bool)
        position_change = float(np.mean(positions_a != positions_b))
        mask_change = float(np.mean(valid_a != valid_b))
        pixel_change = _relative_l2(
            first["pixels_summary"][index], second["pixels_summary"][index],
        )
        sampling_capacity = bool(np.asarray(first.get(
            "sampling_capacity", np.ones(len(first["parcel_index"]), dtype=bool),
        ))[index])
        actual_change = bool(hashes_differ)
        invalid_reason = "" if actual_change else (
            "insufficient_observations_for_independent_view"
            if not sampling_capacity else "identical_sampled_input"
        )
        rows.append({
            "domain": str(first["domain"][index]),
            "parcel_index": int(parcel), "class_id": int(first["label"][index]),
            "input_hash_changed": hashes_differ,
            "position_change_fraction": position_change,
            "valid_mask_change_fraction": mask_change,
            "pixel_summary_relative_l2": pixel_change,
            "independent_sampling_capacity": sampling_capacity,
            "shape_token_relative_l2": _relative_l2(
                first["token_signature"][index], second["token_signature"][index],
            ),
            "similarity_relative_l2": _relative_l2(
                first["similarity"][index], second["similarity"][index],
            ),
            "strength_relative_l2": _relative_l2(
                first["response"][index, :16], second["response"][index, :16],
            ),
            "k1_relative_l2": _relative_l2(
                first["response"][index, 32:64], second["response"][index, 32:64],
            ),
            "k2_relative_l2": _relative_l2(
                first["response"][index, 64:96], second["response"][index, 64:96],
            ),
            "valid_view": actual_change,
            "invalid_reason": invalid_reason,
        })
    return rows


def identity_control_summary(paired, fixed_position, same_class):
    paired = np.asarray(paired, dtype=float)
    fixed_position = np.asarray(fixed_position, dtype=float)
    same_class = np.asarray(same_class, dtype=float)
    observed = float(paired.mean())
    time = float(fixed_position.mean())
    shuffled = float(same_class.mean())
    return {
        "paired_repeatability": observed,
        "fixed_position_repeatability": time,
        "same_class_repeatability": shuffled,
        "increment_over_fixed_position": observed - time,
        "increment_over_same_class": observed - shuffled,
        "identity_beyond_time_supported": bool(observed > time and observed > shuffled),
    }


def strength_support_summary(q, repeated, baseline_probability=0.5):
    q = np.asarray(q, dtype=float); repeated = np.asarray(repeated, dtype=float)
    uniformly_low = bool(len(q) and np.quantile(q, .95) <= 0.0)
    relation = 0.0
    if len(q) > 1 and np.std(q) > 1e-12 and np.std(repeated) > 1e-12:
        relation = float(np.corrcoef(q, repeated)[0, 1])
    supported = bool(not uniformly_low and repeated.mean() > baseline_probability and relation > 0)
    return {
        "sample_count": int(len(q)), "q_mean": float(q.mean()) if len(q) else math.nan,
        "q_q95": float(np.quantile(q, .95)) if len(q) else math.nan,
        "repeatability": float(repeated.mean()) if len(q) else math.nan,
        "q_repeatability_correlation": relation,
        "absolute_matching_support": supported,
        "failure_reason": "uniformly_low_absolute_similarity" if uniformly_low else (
            "no_incremental_monotonic_support" if not supported else ""
        ),
    }


def _angle_error(first, second, harmonic):
    angle = np.angle(second[:, 0] + 1j * second[:, 1]) - np.angle(first[:, 0] + 1j * first[:, 1])
    return np.abs(np.angle(np.exp(1j * angle))) / max(int(harmonic), 1)


def phase_stability_summary(k1_a, k1_b, k2_a, k2_b, k1_threshold, k2_threshold):
    output = {}
    for harmonic, first, second, threshold in (
        (1, np.asarray(k1_a), np.asarray(k1_b), float(k1_threshold)),
        (2, np.asarray(k2_a), np.asarray(k2_b), float(k2_threshold)),
    ):
        magnitude_a = np.linalg.norm(first, axis=1)
        magnitude_b = np.linalg.norm(second, axis=1)
        reliable = (magnitude_a >= threshold) & (magnitude_b >= threshold)
        error = _angle_error(first[reliable], second[reliable], harmonic) if reliable.any() else np.asarray([])
        output.update({
            f"k{harmonic}_magnitude_threshold": threshold,
            f"k{harmonic}_magnitude_view0_mean": float(magnitude_a.mean()),
            f"k{harmonic}_magnitude_view1_mean": float(magnitude_b.mean()),
            f"k{harmonic}_continuous_complex_l2_mean": float(np.linalg.norm(second - first, axis=1).mean()),
            f"k{harmonic}_direction_sample_count": int(reliable.sum()),
            f"k{harmonic}_direction_reliable_fraction": float(reliable.mean()),
            f"k{harmonic}_direction_error_mean": float(error.mean()) if len(error) else math.nan,
        })
    return output


def anchor_basis_assignment(source_anchors, uda_anchors):
    source = np.asarray(source_anchors, dtype=np.float64)
    uda = np.asarray(uda_anchors, dtype=np.float64)
    if source.ndim != 2 or source.shape != uda.shape:
        raise ValueError("Source/UDA Anchor matrices must have identical [M,D] shape")
    source = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
    uda = uda / np.maximum(np.linalg.norm(uda, axis=1, keepdims=True), 1e-12)
    cosine = source @ uda.T
    rows, columns = linear_sum_assignment(-cosine)
    order = np.argsort(rows); rows, columns = rows[order], columns[order]
    assigned = cosine[rows, columns]
    alternatives = cosine.copy(); alternatives[rows, columns] = -np.inf
    margins = assigned - alternatives.max(axis=1)
    interpretation_allowed = bool(
        np.all(np.isfinite(assigned))
        and np.all(assigned > ANCHOR_MATCH_POSITIVE_COSINE)
        and np.all(margins > 0)
    )
    return {
        "source_anchor": rows.astype(np.int64), "uda_anchor": columns.astype(np.int64),
        "cosine": assigned, "assignment_margin": margins,
        "interpretation_allowed": interpretation_allowed,
        "quality": "matched_unique_positive" if interpretation_allowed else "ambiguous_or_low",
    }


def build_cache_signature(checkpoint_sha, split_fingerprint, seed, sampling,
                          protocol_version=STAGE1_PROTOCOL_VERSION):
    payload = {
        "protocol_version": protocol_version, "p0r_protocol": P0R_PROTOCOL_VERSION,
        "view_seed_version": VIEW_SEED_VERSION, "checkpoint_sha256": checkpoint_sha,
        "split_fingerprint": split_fingerprint, "seed": int(seed), "sampling": sampling,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def settings_fingerprint(source_sha, uda_sha, data_root, seed, batch_size, pixel_budget):
    payload = {
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "source_checkpoint_sha256": source_sha, "uda_checkpoint_sha256": uda_sha,
        "data_root": str(Path(data_root)), "seed": int(seed),
        "batch_size": int(batch_size), "pixel_budget": int(pixel_budget),
        "phase_magnitude_quantile": PHASE_MAGNITUDE_QUANTILE,
        "independent_view_protocol": VIEW_SEED_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _sample_hash(pixels, positions, valid):
    digest = hashlib.sha256()
    for value in (pixels, positions, valid):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def extract_compact_view(model, dataset, domain, split, view_id, seed, device,
                         batch_size, pixel_budget):
    wrapped = DeterministicViewDataset(dataset, domain, seed, view_id)
    target = torch.device(device)
    storage = {key: [] for key in (
        "parcel_index", "label", "input_hash", "pixels_summary", "positions",
        "valid_mask", "similarity", "response", "token_signature", "pooling_weight",
    )}
    shape_by_parcel = {
        int(parcel): tuple(map(int, shape))
        for parcel, shape in zip(dataset.get_parcel_indices(), dataset.get_shapes())
    }
    sampling_capacity = []
    for raw in semantic_loader(wrapped, batch_size, pixel_budget, num_workers=0):
        batch = {key: value.to(target) if torch.is_tensor(value) else value for key, value in raw.items()}
        output = model.forward_with_temporal_shift(
            batch["pixels"], batch["valid_pixels"], batch["positions"], batch["extra"],
            temporal_shift=0, return_dict=True,
        )
        similarity = output["shapelet_similarity"]
        beta = float(model.structure_branch.shapelet_dictionary.beta)
        pixel_summary = torch.stack((
            batch["pixels"].mean(dim=tuple(range(1, batch["pixels"].ndim))),
            batch["pixels"].std(dim=tuple(range(1, batch["pixels"].ndim))),
            batch["pixels"].square().mean(dim=tuple(range(1, batch["pixels"].ndim))).sqrt(),
        ), dim=1)
        for index in range(len(batch["parcel_index"])):
            storage["input_hash"].append(_sample_hash(
                batch["pixels"][index], batch["positions"][index], batch["valid_pixels"][index],
            ))
            parcel = int(batch["parcel_index"][index].detach().cpu())
            original_shape = shape_by_parcel[parcel]
            sampled_times = int(batch["positions"][index].numel())
            sampled_pixels = int((batch["valid_pixels"][index, 0] > 0).sum().detach().cpu())
            sampling_capacity.append(
                original_shape[0] > sampled_times or original_shape[2] > sampled_pixels
            )
        values = {
            "parcel_index": batch["parcel_index"].long(), "label": batch["label"].long(),
            "pixels_summary": pixel_summary, "positions": batch["positions"].long(),
            "valid_mask": (batch["valid_pixels"].sum(-1) > 0).to(torch.uint8),
            "similarity": similarity, "response": output["shapelet_response"],
            "token_signature": torch.stack((
                output["shape_tokens"].float().mean(-1),
                output["shape_tokens"].float().std(-1),
                output["shape_tokens"].float().norm(dim=-1),
            ), dim=-1),
            "pooling_weight": torch.softmax(beta * similarity, dim=1),
        }
        for key, value in values.items(): storage[key].append(value.detach().cpu())
    arrays = {
        key: (np.asarray(parts) if key == "input_hash" else torch.cat(parts).numpy())
        for key, parts in storage.items()
    }
    arrays["domain"] = np.asarray([str(domain)] * len(arrays["parcel_index"]))
    arrays["split"] = np.asarray([str(split)] * len(arrays["parcel_index"]))
    arrays["view_id"] = np.full(len(arrays["parcel_index"]), int(view_id), dtype=np.int64)
    arrays["sampling_capacity"] = np.asarray(sampling_capacity, dtype=np.uint8)
    starts, scales = _window_metadata(model.structure_branch)
    arrays["window_starts"] = starts; arrays["window_scales"] = scales
    arrays["window_centers"] = (starts + (scales - 1) / 2.0) % model.structure_branch.window_extractor.grid_points
    return arrays


def _save_npz(path, packet):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream: np.savez_compressed(stream, **packet)
    os.replace(temporary, path)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as packet:
        return {key: packet[key] for key in packet.files}


def _split_fingerprint(dataset, domain):
    parcels = np.asarray(dataset.get_parcel_indices(), dtype=np.int64)
    return hashlib.sha256(str(domain).encode() + b"\0" + parcels.tobytes()).hexdigest()


def extract_or_load_pair(output, model, dataset, domain, split, checkpoint_sha,
                         seed, device, batch_size, pixel_budget):
    output = Path(output); cache = output / "cache"
    split_fingerprint = _split_fingerprint(dataset, domain)
    sampling = {
        "batch_size": int(batch_size), "pixel_budget": int(pixel_budget),
        "seq_length": int(getattr(dataset, "transform", None) is not None),
        "independent_view_ids": [0, 1], "actual_input_hash_required": True,
    }
    signature = build_cache_signature(checkpoint_sha, split_fingerprint, seed, sampling)
    manifest_path = cache / f"{split}_manifest.json"
    paths = [cache / f"{split}_view{view}.npz" for view in (0, 1)]
    reusable = False
    if manifest_path.is_file() and all(path.is_file() for path in paths):
        try: reusable = json.loads(manifest_path.read_text(encoding="utf-8")).get("signature") == signature
        except (OSError, ValueError, json.JSONDecodeError): reusable = False
    packets = []
    for view_id, path in enumerate(paths):
        packet = _load_npz(path) if reusable else extract_compact_view(
            model, dataset, domain, split, view_id, seed, device, batch_size, pixel_budget,
        )
        if not reusable: _save_npz(path, packet)
        packets.append(packet)
    if not reusable:
        _write_json(manifest_path, {
            "signature": signature, "protocol_version": STAGE1_PROTOCOL_VERSION,
            "checkpoint_sha256": checkpoint_sha, "split_fingerprint": split_fingerprint,
            "sampling": sampling, "contains_raw_pixels": False,
        })
    return pair_compact_views(*packets), split_fingerprint, signature


def _safe_interval(values, classes, seed):
    values = np.asarray(values, dtype=float); classes = np.asarray(classes)
    if not len(values): return math.nan, math.nan, math.nan
    return _cluster_mean_interval(values, classes, 500, seed)


def _identity_rows(checkpoint_role, split, pair, valid, seed):
    left, right = pair
    keep = np.asarray(valid, dtype=bool)
    sim0, sim1 = left["similarity"][keep], right["similarity"][keep]
    labels = left["label"][keep]; parcels = left["parcel_index"][keep]
    starts, scales = left["window_starts"], left["window_scales"]
    if not len(sim0): return []
    best0, best1 = sim0.argmax(1), sim1.argmax(1)
    top0, top1 = sim0.argmax(2), sim1.argmax(2)
    controls = None
    try: controls = class_matched_derangements(parcels, labels, repeats=1, seed=seed)[0]
    except ValueError: pass
    rows = []
    for anchor in range(sim0.shape[-1]):
        exact = best0[:, anchor] == best1[:, anchor]
        iou = np.asarray([
            circular_window_iou(starts[a], starts[b], scales[a], 64, scales[b])
            for a, b in zip(best0[:, anchor], best1[:, anchor])
        ])
        profile = np.sum(sim0[:, :, anchor] * sim1[:, :, anchor], axis=1) / np.maximum(
            np.linalg.norm(sim0[:, :, anchor], axis=1) * np.linalg.norm(sim1[:, :, anchor], axis=1), 1e-12,
        )
        modal = int(np.bincount(best0[:, anchor], minlength=sim0.shape[1]).argmax())
        fixed = best1[:, anchor] == modal
        shuffled = exact.astype(float) if controls is None else best0[:, anchor] == best1[controls, anchor]
        control = identity_control_summary(iou > 0, fixed, shuffled)
        exact_mean, exact_low, exact_high = _safe_interval(exact, labels, seed + anchor)
        rows.append({
            "checkpoint_role": checkpoint_role, "split": split, "anchor": anchor,
            "sample_count": int(len(labels)), "exact_match": exact_mean,
            "exact_match_ci_low": exact_low, "exact_match_ci_high": exact_high,
            "window_iou_mean": float(iou.mean()), "profile_correlation_mean": float(profile.mean()),
            "window_top_anchor_agreement": float((top0 == top1).mean()),
            **control,
        })
    return rows


def _strength_rows(checkpoint_role, split, pair, valid, train_edges, p0r_rows):
    left, right = pair; keep = np.asarray(valid, dtype=bool)
    q = left["similarity"][keep]; repeated = (q.argmax(1) == right["similarity"][keep].argmax(1))
    response = left["response"][keep]
    rows = []
    for anchor in range(q.shape[-1]):
        qmax = q[:, :, anchor].max(1)
        base = strength_support_summary(qmax, repeated[:, anchor])
        q33, q67, low_edge, high_concentration = train_edges[anchor]
        concentration = response[:, 16 + anchor]
        low_q = qmax <= low_edge; high_c = concentration >= high_concentration
        inherited = p0r_rows.get(anchor, {})
        common = {
            "checkpoint_role": checkpoint_role, "split": split, "anchor": anchor,
            "scope": "overall",
            **base, "strength_mean": float(response[:, anchor].mean()),
            "concentration_mean": float(concentration.mean()),
            "low_q_high_concentration_fraction": float(np.mean(low_q & high_c)),
            "p0r_joint_status": inherited.get("joint_status", inherited.get("status", "missing")),
            "p0r_failure_reason": inherited.get("failure_reason", "missing_p0r_result"),
        }
        rows.append(common)
        bands = np.digitize(qmax, (q33, q67), right=True)
        for band, name in enumerate(("low", "medium", "high")):
            selected = bands == band
            if selected.any():
                rows.append({
                    "checkpoint_role": checkpoint_role, "split": split,
                    "anchor": anchor, "scope": "q_band", "q_band": name,
                    "sample_count": int(selected.sum()),
                    "q_mean": float(qmax[selected].mean()),
                    "repeatability": float(repeated[selected, anchor].mean()),
                    "strength_mean": float(response[selected, anchor].mean()),
                    "concentration_mean": float(concentration[selected].mean()),
                })
        for class_id in np.unique(left["label"][keep]):
            selected = left["label"][keep] == class_id
            rows.append({
                "checkpoint_role": checkpoint_role, "split": split,
                "anchor": anchor, "scope": "class", "class_id": int(class_id),
                "sample_count": int(selected.sum()),
                "q_mean": float(qmax[selected].mean()),
                "repeatability": float(repeated[selected, anchor].mean()),
                "strength_mean": float(response[selected, anchor].mean()),
                "concentration_mean": float(concentration[selected].mean()),
            })
    return rows


def _temporal_rows(checkpoint_role, split, pair, valid, thresholds):
    left, right = pair; keep = np.asarray(valid, dtype=bool)
    response0, response1 = left["response"][keep], right["response"][keep]
    weights0, weights1 = left["pooling_weight"][keep], right["pooling_weight"][keep]
    rows = []
    for anchor in range(16):
        k1a = np.stack((response0[:, 32 + anchor], response0[:, 48 + anchor]), 1)
        k1b = np.stack((response1[:, 32 + anchor], response1[:, 48 + anchor]), 1)
        k2a = np.stack((response0[:, 64 + anchor], response0[:, 80 + anchor]), 1)
        k2b = np.stack((response1[:, 64 + anchor], response1[:, 80 + anchor]), 1)
        summary = phase_stability_summary(k1a, k1b, k2a, k2b, *thresholds[anchor])
        summary.update({
            "checkpoint_role": checkpoint_role, "split": split, "anchor": anchor,
            "sample_count": int(len(response0)),
            "pooling_weight_l1_mean": float(np.abs(weights1[:, :, anchor] - weights0[:, :, anchor]).sum(1).mean()),
            "pooling_max_weight_mean": float(weights0[:, :, anchor].max(1).mean()),
            "pooling_effective_window_count_mean": float(np.exp(
                -(weights0[:, :, anchor] * np.log(np.maximum(weights0[:, :, anchor], 1e-12))).sum(1)
            ).mean()),
        })
        centers = np.asarray(left["window_centers"], dtype=float)
        period = 365.0; delta = 15.0
        theta = 2.0 * np.pi * centers / 64.0
        for harmonic in (1, 2):
            base = np.sum(weights0[:, :, anchor] * np.exp(1j * harmonic * theta)[None, :], axis=1)
            shifted = np.sum(
                weights0[:, :, anchor]
                * np.exp(1j * harmonic * (theta + 2.0 * np.pi * delta / period))[None, :],
                axis=1,
            )
            expected = base * np.exp(1j * harmonic * 2.0 * np.pi * delta / period)
            summary[f"k{harmonic}_analytic_rotation_max_error"] = float(np.max(np.abs(shifted - expected)))
            summary[f"k{harmonic}_analytic_rotation_shift_days"] = delta
            summary[f"k{harmonic}_analytic_rotation_scope"] = "mathematical_equivariance_only"
        rows.append(summary)
    return rows


def _p0r_rows(root, alias):
    path = Path(root) / alias / "anchor_reliability_summary.csv"
    if not path.is_file(): return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(row["anchor"]): row for row in csv.DictReader(stream)}


def _anchors(model):
    return model.structure_branch.shapelet_dictionary.anchors.detach().float().cpu().numpy()


def run_checkpoint(args, role, checkpoint, model, config, datasets, output):
    checkpoint_sha = sha256_file(checkpoint)
    pairs, split_fingerprints, signatures = {}, {}, {}
    integrity = []
    for split, dataset in datasets.items():
        pair, fingerprint, signature = extract_or_load_pair(
            output / role, model, dataset, config.source, split, checkpoint_sha,
            args.seed, args.device, args.batch_size, args.pixel_budget,
        )
        pairs[split] = pair; split_fingerprints[split] = fingerprint; signatures[split] = signature
        current = evaluate_view_integrity(*pair)
        for row in current: row.update({"checkpoint_role": role, "split": split})
        integrity.extend(current)
    train_valid = np.asarray([row["valid_view"] for row in integrity if row["split"] == "source_train"])
    val_valid = np.asarray([row["valid_view"] for row in integrity if row["split"] == "source_val"])
    base_manifest = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "split_fingerprints": split_fingerprints, "cache_signatures": signatures,
        "valid_train": int(train_valid.sum()), "valid_val": int(val_valid.sum()),
        "invalid_train": int((~train_valid).sum()), "invalid_val": int((~val_valid).sum()),
    }
    if not train_valid.any() or not val_valid.any():
        return integrity, [], [], [], {
            **base_manifest, "audit_valid": False,
            "blocker": "no_actual_independent_views_in_source_train_or_source_val",
        }
    train0 = pairs["source_train"][0]
    train_qmax = train0["similarity"][train_valid].max(1)
    train_concentration = train0["response"][train_valid, 16:32]
    train_edges = [(
        float(np.quantile(train_qmax[:, a], 1 / 3)),
        float(np.quantile(train_qmax[:, a], 2 / 3)),
        float(np.quantile(train_qmax[:, a], .25)),
        float(np.quantile(train_concentration[:, a], .75)),
    ) for a in range(16)]
    thresholds = []
    for anchor in range(16):
        k1 = np.hypot(train0["response"][train_valid, 32 + anchor], train0["response"][train_valid, 48 + anchor])
        k2 = np.hypot(train0["response"][train_valid, 64 + anchor], train0["response"][train_valid, 80 + anchor])
        thresholds.append((float(np.quantile(k1, PHASE_MAGNITUDE_QUANTILE)), float(np.quantile(k2, PHASE_MAGNITUDE_QUANTILE))))
    p0r = _p0r_rows(args.p0r_root, args.source_alias)
    identity = []; strength = []; temporal = []
    for split, valid in (("source_train", train_valid), ("source_val", val_valid)):
        identity.extend(_identity_rows(role, split, pairs[split], valid, args.seed))
        strength.extend(_strength_rows(role, split, pairs[split], valid, train_edges, p0r))
        temporal.extend(_temporal_rows(role, split, pairs[split], valid, thresholds))
    # Reuse the P0-R Source-only class/time/qmax baseline on valid pairs.
    try:
        base_train = _identity_rows_to_repeatability(pairs["source_train"], train_valid)
        base_val = _identity_rows_to_repeatability(pairs["source_val"], val_valid)
        increments = []
        for anchor in range(16):
            increments.extend(repeatability_baseline_rows(base_train, base_val, anchor, args.seed))
        augmented = {int(row["anchor"]): row for row in increments if row.get("model") == "class_position_qmax"}
        for row in strength:
            if row["split"] == "source_val" and row["anchor"] in augmented:
                row.update({f"conditional_{key}": value for key, value in augmented[row["anchor"]].items() if key.endswith("increment") or key.endswith("ci_low") or key.endswith("ci_high")})
    except (ValueError, RuntimeError):
        pass
    return integrity, identity, strength, temporal, {
        **base_manifest, "audit_valid": True, "blocker": "",
    }


def _identity_rows_to_repeatability(pair, valid):
    from analysis.anchor_matching_calibration_audit import repeatability_rows
    left, right = pair; keep = np.asarray(valid, dtype=bool)
    return repeatability_rows(
        left["similarity"][keep], right["similarity"][keep], left["parcel_index"][keep],
        left["window_starts"], left["window_scales"], class_labels=left["label"][keep],
    )


def run_source(args):
    output = Path(args.output_root) / args.source_alias; output.mkdir(parents=True, exist_ok=True)
    source_runtime = load_source_runtime(
        args.source_checkpoint, args.data_root, args.device, args.seed, args.source_alias,
    )
    source_model, source_config, source_protocol, _, source_datasets = source_runtime
    frozen = load_generic_frozen(args.uda_checkpoint, args.task, "G", args.device)
    validate_checkpoint_pair_metadata(source_config, frozen.config, frozen.packet.get("epoch", -1))
    uda_config, uda_protocol, _, uda_datasets_all = build_abc_datasets(frozen.config, args.data_root, args.seed)
    uda_datasets = {key: uda_datasets_all[key] for key in ("source_train", "source_val")}
    validate_split_isolation(
        source_datasets["source_train"].get_parcel_indices(),
        source_datasets["source_val"].get_parcel_indices(),
    )
    validate_split_isolation(
        uda_datasets["source_train"].get_parcel_indices(),
        uda_datasets["source_val"].get_parcel_indices(),
    )
    collected = [[], [], [], []]; manifests = {}
    for role, checkpoint, model, config, datasets in (
        ("source_best", args.source_checkpoint, source_model, source_config, source_datasets),
        ("uda_final", args.uda_checkpoint, freeze_for_audit(frozen.student), uda_config, uda_datasets),
    ):
        result = run_checkpoint(args, role, Path(checkpoint), model, config, datasets, output)
        for target, rows in zip(collected, result[:4]):
            for row in rows:
                row.update({"source_alias": args.source_alias, "task": args.task})
            target.extend(rows)
        manifests[role] = result[4]
    if manifests["source_best"]["split_fingerprints"] != manifests["uda_final"]["split_fingerprints"]:
        raise ValueError("Source Best and UDA Final did not reconstruct identical Source splits")
    assignment = anchor_basis_assignment(_anchors(source_model), _anchors(frozen.student))
    comparison = [{
        "source_alias": args.source_alias, "task": args.task,
        "source_anchor": int(source), "uda_anchor": int(uda),
        "anchor_cosine": float(cosine), "assignment_margin": float(margin),
        "matching_quality": assignment["quality"],
        "interpretation_allowed": assignment["interpretation_allowed"],
    } for source, uda, cosine, margin in zip(
        assignment["source_anchor"], assignment["uda_anchor"], assignment["cosine"], assignment["assignment_margin"],
    )]
    source_val_identity = {
        int(row["anchor"]): row for row in collected[1]
        if row["checkpoint_role"] == "source_best" and row["split"] == "source_val"
    }
    uda_val_identity = {
        int(row["anchor"]): row for row in collected[1]
        if row["checkpoint_role"] == "uda_final" and row["split"] == "source_val"
    }
    for row in comparison:
        source_metric = source_val_identity.get(row["source_anchor"], {})
        uda_metric = uda_val_identity.get(row["uda_anchor"], {})
        if row["interpretation_allowed"] and source_metric and uda_metric:
            for metric in ("exact_match", "window_iou_mean", "profile_correlation_mean"):
                row[f"source_{metric}"] = source_metric[metric]
                row[f"uda_{metric}"] = uda_metric[metric]
                row[f"uda_minus_source_{metric}"] = float(uda_metric[metric]) - float(source_metric[metric])
        else:
            row["comparison_status"] = "not_interpreted_low_or_ambiguous_anchor_match"
    for name, rows in zip(REQUIRED_OUTPUTS[:4], collected): _write_csv(output / name, rows)
    _write_csv(output / "source_uda_identity_comparison.csv", comparison)
    settings = settings_fingerprint(
        manifests["source_best"]["checkpoint_sha256"],
        manifests["uda_final"]["checkpoint_sha256"], args.data_root, args.seed,
        args.batch_size, args.pixel_budget,
    )
    if args.settings_fingerprint and args.settings_fingerprint != settings:
        raise ValueError("scheduler/analysis settings fingerprint mismatch")
    _write_json(output / "checkpoint_manifest.json", {
        "status": "complete", "protocol_version": STAGE1_PROTOCOL_VERSION,
        "source_alias": args.source_alias, "task": args.task,
        "source_checkpoint_sha256": manifests["source_best"]["checkpoint_sha256"],
        "uda_checkpoint_sha256": manifests["uda_final"]["checkpoint_sha256"],
        "settings_fingerprint": settings, "checkpoints": manifests,
        "source_protocol": source_protocol, "uda_protocol": uda_protocol,
        "anchor_matching_quality": assignment["quality"],
        "target_test_used": False, "target_labels_used": False,
        "training_performed": False,
    })


def run_summary(args):
    root = Path(args.output_root)
    combined = {name: [] for name in REQUIRED_OUTPUTS[:5]}
    for folder in sorted(path for path in root.iterdir() if path.is_dir() and path.name in ("AT1", "FR1", "FR2", "DK1")):
        for name in combined:
            path = folder / name
            if path.is_file():
                with path.open(newline="", encoding="utf-8") as stream:
                    combined[name].extend(csv.DictReader(stream))
    for name, rows in combined.items(): _write_csv(root / name, rows)
    summary = []
    for alias in ("AT1", "FR1", "FR2", "DK1"):
        manifest_path = root / alias / "checkpoint_manifest.json"
        if not manifest_path.is_file(): continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = {
            int(row["anchor"]): row
            for row in combined["anchor_identity_repeatability.csv"]
            if row.get("source_alias") == alias and row.get("split") == "source_val"
            and row.get("checkpoint_role") == "source_best"
        }
        strength = {
            int(row["anchor"]): row
            for row in combined["anchor_strength_reliability.csv"]
            if row.get("source_alias") == alias and row.get("split") == "source_val"
            and row.get("checkpoint_role") == "source_best"
            and row.get("scope") == "overall"
        }
        temporal = {
            int(row["anchor"]): row
            for row in combined["anchor_temporal_stability.csv"]
            if row.get("source_alias") == alias and row.get("split") == "source_val"
            and row.get("checkpoint_role") == "source_best"
        }
        for anchor in range(16):
            identity_row = identity.get(anchor, {}); strength_row = strength.get(anchor, {})
            temporal_row = temporal.get(anchor, {})
            summary.append({
                "source_alias": alias, "task": manifest["task"], "anchor": anchor,
                "anchor_matching_quality": manifest["anchor_matching_quality"],
                "valid_source_val_views": manifest["checkpoints"]["source_best"]["valid_val"],
                "invalid_source_val_views": manifest["checkpoints"]["source_best"]["invalid_val"],
                "identity_repeatability": identity_row.get("paired_repeatability", ""),
                "identity_increment_over_time": identity_row.get("increment_over_fixed_position", ""),
                "identity_increment_over_same_class": identity_row.get("increment_over_same_class", ""),
                "identity_status": "supported_beyond_controls" if str(identity_row.get("identity_beyond_time_supported", "")).lower() == "true" else "unsupported_or_time_explained",
                "matching_strength_status": "supported" if str(strength_row.get("absolute_matching_support", "")).lower() == "true" else "unsupported_or_uncertain",
                "p0r_joint_status": strength_row.get("p0r_joint_status", "missing"),
                "k1_direction_reliable_fraction": temporal_row.get("k1_direction_reliable_fraction", ""),
                "k2_direction_reliable_fraction": temporal_row.get("k2_direction_reliable_fraction", ""),
                "temporal_status": "reported_separately_for_k1_k2",
                "result_scope": "three_independent_evidence_axes_no_composite_score",
            })
    _write_csv(root / "ANCHOR_STAGE1_summary.csv", summary)
    report = [
        "# Anchor Stage-1 frozen audit", "",
        "This report is descriptive. It does not assign physical-event semantics or define a training loss.", "",
    ]
    for row in summary[::16]:
        source_rows = [value for value in summary if value["source_alias"] == row["source_alias"]]
        identity_supported = sum(value["identity_status"] == "supported_beyond_controls" for value in source_rows)
        matching_supported = sum(value["matching_strength_status"] == "supported" for value in source_rows)
        report.append(
            f"- {row['source_alias']} / {row['task']}: valid Source Val paired views "
            f"{row['valid_source_val_views']}, invalid {row['invalid_source_val_views']}; "
            f"Source/UDA Anchor matching: {row['anchor_matching_quality']}; "
            f"identity beyond controls supported for {identity_supported}/16 Anchors; "
            f"absolute matching support supported for {matching_supported}/16 Anchors."
        )
    report.extend([
        "", "Identity, absolute matching support, and K1/K2 temporal stability remain separate evidence axes.",
        "K1 and K2 phase directions use independent Source-Train magnitude references; low K1 magnitude does not invalidate K2.",
        "Analytic rotation errors test mathematical equivariance only and are not treated as observation repeatability.",
    ])
    (root / "ANCHOR_STAGE1_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source", "summary"), required=True)
    parser.add_argument("--source-alias", choices=("AT1", "FR1", "FR2", "DK1"))
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--uda-checkpoint")
    parser.add_argument("--data-root", default="/data/user/dataset/timematch_data")
    parser.add_argument("--p0r-root", default="experiments/anchor_matching_calibration_p0r_seed1")
    parser.add_argument("--output-root", default="experiments/structure_g_anchor_stage1_seed1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--settings-fingerprint", default="")
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "summary": run_summary(args)
    else: run_source(args)


if __name__ == "__main__":
    main()
