#!/usr/bin/env python3
"""Frozen numerical audit for the B2 phase response (no training)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.structure_representation_chain_audit import (  # noqa: E402
    DOMAINS,
    TASKS,
    build_audit_datasets,
    deterministic_loader,
    load_source_model,
    seed_all,
)
from methods.structure_da.phase_equivariance import rotate_phase_moments  # noqa: E402


BLOCKS = ("presence", "coverage", "k1_real", "k1_imag", "k2_real", "k2_imag")


def _effective_rank(values, eps=1e-12):
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.
    values = values - values.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(values)
    probability = singular.square()
    if float(probability.sum()) <= eps:
        return 0.
    probability = probability / probability.sum()
    return float(torch.exp(-(probability * probability.clamp_min(eps).log()).sum()))


def response_summary(response, block_size):
    response = torch.as_tensor(response, dtype=torch.float32)
    if response.ndim != 2 or response.shape[1] != 6 * int(block_size):
        raise ValueError("response must be [samples, 6 * block_size]")
    standardized = (
        (response - response.mean(0, keepdim=True))
        / response.std(0, unbiased=False, keepdim=True).clamp_min(1e-8)
    )
    rows = []
    for index, name in enumerate(BLOCKS):
        start, stop = index * block_size, (index + 1) * block_size
        block = response[:, start:stop]
        standardized_block = standardized[:, start:stop]
        rows.append({
            "block": name,
            "mean": float(block.mean()),
            "variance": float(block.var(unbiased=False)),
            "rms": float(block.square().mean().sqrt()),
            "effective_rank_raw": _effective_rank(block),
            "effective_rank_standardized": _effective_rank(standardized_block),
        })
    rows.append({
        "block": "full_response",
        "mean": float(response.mean()),
        "variance": float(response.var(unbiased=False)),
        "rms": float(response.square().mean().sqrt()),
        "effective_rank_raw": _effective_rank(response),
        "effective_rank_standardized": _effective_rank(standardized),
    })
    return rows


def evidence_profile(evidence, eps=1e-8):
    evidence = torch.as_tensor(evidence)
    if evidence.ndim != 3:
        raise ValueError("relative evidence must be [samples, windows, anchors]")
    active = (evidence > eps).sum(1)
    peak = evidence.amax(1)
    positive_peak = peak[peak > eps]
    weak_threshold = (
        torch.quantile(positive_peak.float(), .25)
        if positive_peak.numel() else peak.new_tensor(0.)
    )
    total = max(active.numel(), 1)
    return {
        "zero_evidence_fraction": float((active == 0).sum() / total),
        "single_peak_fraction": float((active == 1).sum() / total),
        "multi_peak_fraction": float((active > 1).sum() / total),
        "weak_evidence_threshold_q25": float(weak_threshold),
        "weak_evidence_fraction": float(
            ((peak > eps) & (peak <= weak_threshold)).sum() / total
        ),
    }


def frozen_check_blockers(response, rotation_error, tolerance=1e-5):
    blockers = []
    if not torch.isfinite(torch.as_tensor(response)).all():
        blockers.append("non_finite_response")
    if not np.isfinite(rotation_error) or float(rotation_error) > tolerance:
        blockers.append("phase_rotation_error")
    return blockers


def _distribution(prefix, values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    quantiles = torch.quantile(values, torch.tensor([.25, .5, .75]))
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(unbiased=False)),
        f"{prefix}_min": float(values.amin()),
        f"{prefix}_q25": float(quantiles[0]),
        f"{prefix}_median": float(quantiles[1]),
        f"{prefix}_q75": float(quantiles[2]),
        f"{prefix}_max": float(values.amax()),
    }


def _write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_checkpoints(args):
    entries = []
    for task in args.tasks:
        entries.extend((
            (f"G_{task}", task, Path(args.g_root) / f"G_{task}_seed1/fold_0/checkpoint_last.pt"),
            (f"B_{task}", task, Path(args.b_root) / f"B_{task}_seed1/fold_0/checkpoint_last.pt"),
        ))
    return entries


def _explicit_checkpoints(values):
    entries = []
    for value in values:
        if "=" not in value:
            raise ValueError("--checkpoint must be LABEL=PATH")
        label, path = value.split("=", 1)
        matches = [task for task in TASKS if task in label]
        if len(matches) != 1:
            raise ValueError(f"checkpoint label must contain exactly one task: {label}")
        entries.append((label, matches[0], Path(path)))
    return entries


@torch.no_grad()
def _extract_checkpoint(label, task, checkpoint, args):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    device = torch.device(args.device)
    model, config = load_source_model(checkpoint, device, "phase_moment")
    source_alias, target_alias = TASKS[task]
    expected_mode = "softmax" if label.startswith("G_") else "independent"
    actual_mode = getattr(config, "phase_response_mode", "softmax")
    if actual_mode != expected_mode:
        raise ValueError(
            f"checkpoint {label} response mode mismatch: "
            f"expected={expected_mode}, actual={actual_mode}"
        )
    if getattr(config, "source", None) != DOMAINS[source_alias]:
        raise ValueError(
            f"checkpoint {label} source mismatch: "
            f"expected={DOMAINS[source_alias]}, actual={getattr(config, 'source', None)}"
        )
    allowed_targets = {DOMAINS[source_alias], DOMAINS[target_alias]}
    if getattr(config, "target", None) not in allowed_targets:
        raise ValueError(
            f"checkpoint {label} target mismatch: "
            f"expected one of {sorted(allowed_targets)}, "
            f"actual={getattr(config, 'target', None)}"
        )
    datasets, _ = build_audit_datasets(
        config, DOMAINS[source_alias], DOMAINS[target_alias],
        args.data_root, args.seed,
    )
    original_values, b2_values, similarity_values, evidence_values = [], [], [], []
    peak_values, coverage_values, k1_values, k2_values = [], [], [], [], []
    maximum_rotation_error = 0.
    for batch in deterministic_loader(
        datasets["source_val"], args.batch_size, args.num_workers,
        pixel_budget=args.pixel_budget,
    ):
        output = model(
            batch["pixels"].to(device),
            batch["valid_pixels"].to(device),
            batch["positions"].to(device),
            batch["extra"].to(device),
            return_dict=True,
        )
        similarity = output["shapelet_similarity"]
        _, _, centers = model.structure_branch.window_extractor(
            output["exposed_curve"], return_centers=True,
        )
        details = {
            "similarity": similarity,
            "candidate_mask": torch.ones(
                similarity.shape[:2], dtype=torch.bool, device=device,
            ),
        }
        base = model.structure_branch.compose_phase_response(
            details, centers, phase_shift=0, mode="b2",
        )
        shifted = model.structure_branch.compose_phase_response(
            details, centers, phase_shift=31, mode="b2",
        )
        expected = rotate_phase_moments(
            base["phase_moments"],
            torch.full((similarity.shape[0],), 31., device=device),
            shapelet_count=similarity.shape[-1], harmonics=(1, 2),
            period_days=model.structure_branch.exposer.period_days,
        )
        maximum_rotation_error = max(
            maximum_rotation_error,
            float((shifted["phase_moments"] - expected).abs().amax()),
        )
        moments = base["phase_moments"].reshape(
            similarity.shape[0], 2, 2, similarity.shape[-1],
        )
        original_values.append(output["shapelet_response"].float().cpu())
        b2_values.append(base["response"].float().cpu())
        similarity_values.append(similarity.float().cpu())
        evidence_values.append(base["relative_evidence"].float().cpu())
        peak_values.append(base["peak_evidence"].float().cpu())
        coverage_values.append(base["concentration"].float().cpu())
        magnitude = moments.square().sum(2).sqrt()
        k1_values.append(magnitude[:, 0].float().cpu())
        k2_values.append(magnitude[:, 1].float().cpu())
    original = torch.cat(original_values)
    b2 = torch.cat(b2_values)
    similarity = torch.cat(similarity_values)
    evidence = torch.cat(evidence_values)
    peak = torch.cat(peak_values)
    coverage = torch.cat(coverage_values)
    k1, k2 = torch.cat(k1_values), torch.cat(k2_values)
    count = b2.shape[1] // 6
    block_rows = []
    for representation, values in (
        ("original", original), ("offline_b2", b2),
        ("offline_b2_minus_original", b2 - original),
    ):
        for row in response_summary(values, count):
            block_rows.append({
                "checkpoint": label, "task": task,
                "trained_response_mode": getattr(config, "phase_response_mode", "softmax"),
                "representation": representation, **row,
            })
    profile = evidence_profile(evidence)
    diagnostic = {
        "checkpoint": label, "task": task,
        "trained_response_mode": actual_mode,
        "sample_count": len(b2),
        "similarity_mean": float(similarity.mean()),
        "similarity_std": float(similarity.std(unbiased=False)),
        "similarity_min": float(similarity.amin()),
        "similarity_max": float(similarity.amax()),
        "phase_rotation_max_abs_error": maximum_rotation_error,
        **_distribution("relative_evidence", evidence),
        **_distribution("presence", b2[:, :count]),
        **_distribution("peak_evidence", peak),
        **_distribution("coverage", coverage),
        **_distribution("k1_magnitude", k1),
        **_distribution("k2_magnitude", k2),
        **profile,
    }
    blockers = frozen_check_blockers(b2, maximum_rotation_error)
    manifest = {
        "checkpoint": str(checkpoint), "task": task, "label": label,
        "checkpoint_sha256": _sha256(checkpoint),
        "epoch": int(torch.load(checkpoint, map_location="cpu", weights_only=False).get("epoch", -1)),
        "trained_response_mode": actual_mode,
        "source": getattr(config, "source", ""),
        "target": getattr(config, "target", ""),
        "seed": int(getattr(config, "seed", -1)),
        "sample_count": len(b2), "blockers": blockers,
        "training_allowed": not blockers,
        "target_test_accessed": False,
    }
    return block_rows, diagnostic, manifest


def run(args):
    seed_all(args.seed)
    output = Path(args.output_root)
    entries = (
        _explicit_checkpoints(args.checkpoint)
        if args.checkpoint else _default_checkpoints(args)
    )
    block_rows, diagnostics, manifests = [], [], []
    for label, task, checkpoint in entries:
        rows, diagnostic, manifest = _extract_checkpoint(
            label, task, checkpoint, args,
        )
        block_rows.extend(rows)
        diagnostics.append(diagnostic)
        manifests.append(manifest)
        print(f"B2_FROZEN_CHECK|label={label}|task={task}|blockers={manifest['blockers']}")
    _write_csv(output / "B2_response_blocks.csv", block_rows)
    _write_csv(output / "B2_evidence_diagnostics.csv", diagnostics)
    output.mkdir(parents=True, exist_ok=True)
    (output / "B2_frozen_manifest.json").write_text(
        json.dumps(manifests, indent=2, sort_keys=True), encoding="utf-8",
    )
    lines = [
        "# B2 Frozen Check", "",
        "This check evaluates B2 numerics on frozen G/B representations; it does not evaluate classification performance.", "",
        "| Checkpoint | Task | Samples | Rotation error | Blockers |",
        "|---|---|---:|---:|---|",
    ]
    for diagnostic, manifest in zip(diagnostics, manifests):
        lines.append(
            f"| {manifest['label']} | {manifest['task']} | {manifest['sample_count']} | "
            f"{diagnostic['phase_rotation_max_abs_error']:.3e} | "
            f"{','.join(manifest['blockers']) or 'none'} |"
        )
    (output / "B2_frozen_check.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    blockers = [item for manifest in manifests for item in manifest["blockers"]]
    if blockers:
        raise RuntimeError(f"B2 frozen check blocked training: {sorted(set(blockers))}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="experiments/structure_b2_seed1/frozen_check")
    parser.add_argument("--g-root", default="experiments/structure_e_cross_seed1/outputs/uda")
    parser.add_argument("--b-root", default="experiments/structure_g_response_ab_seed1/outputs/uda")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=tuple(TASKS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pixel-budget", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
