#!/usr/bin/env python3
"""09: Oracle true-class Phase population structure diagnostic.

This script only observes the complete population of numerically-valid cached
``gamma_{i,y_i}`` warps from experiment 06. It never runs registration,
production legality, Phase confirmation, class/group centers, clustering, group
count selection, Stable Label, Teacher/Student updates, or model training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from methods.structure_da.phase_population_diagnostic import (
    common_language_effect,
    deterministic_equal_class_subset,
    deterministic_random_subset,
    exact_phase_knn,
    gamma_to_unit_phase_vectors,
    knn_label_statistics,
    phase_distance_to_identity,
    phase_pair_distances,
    sample_within_cross_pairs,
)
from methods.structure_da.sample_phase_diagnostic import classical_mds, phase_distance_matrix


DEFAULT_K = (5, 10, 20)
DEFAULT_CLASS_NAMES = (
    "corn", "horsebeans", "meadow", "spring_barley", "spring_oat",
    "winter_barley", "winter_rapeseed", "winter_rye", "winter_triticale", "winter_wheat",
)


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value, default=float("nan")) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _i(value) -> int:
    return int(float(value))


def _load_population(path: Path) -> tuple[list[int], np.ndarray, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"09 requires experiment-06 Stage-A cache and never recomputes registration: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = list(payload.get("records", ()))
    sample_ids = [int(v) for v in payload.get("sample_ids", ())]
    true_classes = np.asarray(payload.get("true_classes", ()), dtype=np.int64)
    if len(records) != len(sample_ids) or len(records) != len(true_classes):
        raise ValueError("06 Stage-A cache has inconsistent record/sample/class lengths")
    gammas: list[Tensor] = []
    valid_ids: list[int] = []
    valid_classes: list[int] = []
    for sample_id, true_class, record in zip(sample_ids, true_classes.tolist(), records):
        valid = bool(record.get("numerically_valid", False))
        gamma = record.get("gamma")
        if not valid or not isinstance(gamma, Tensor):
            continue
        gamma = gamma.detach().cpu().double().flatten()
        if not torch.isfinite(gamma).all().item():
            continue
        valid_ids.append(int(sample_id)); valid_classes.append(int(true_class)); gammas.append(gamma)
    if not gammas:
        raise RuntimeError("06 cache contains no numerically-valid oracle true-class gamma")
    lengths = {int(g.numel()) for g in gammas}
    if len(lengths) != 1:
        raise ValueError("cached gamma values do not share one registration grid")
    return valid_ids, np.asarray(valid_classes, dtype=np.int64), torch.stack(gammas, dim=0)


def _join_audit_rows(sample_ids: Sequence[int], path07: Path, path08: Path) -> tuple[dict[int, dict], dict[int, dict]]:
    rows07 = {_i(r["sample_id"]): r for r in _read_csv(path07)}
    rows08 = {_i(r["sample_id"]): r for r in _read_csv(path08)}
    required = set(map(int, sample_ids))
    missing07 = sorted(required.difference(rows07))[:5]
    missing08 = sorted(required.difference(rows08))[:5]
    if missing07 or missing08:
        raise ValueError(f"07/08 sample population mismatch: missing07={missing07}, missing08={missing08}")
    return rows07, rows08


def _class_names(rows07: dict[int, dict], sample_ids: Sequence[int], labels: np.ndarray) -> dict[int, str]:
    result: dict[int, str] = {}
    for sid, label in zip(sample_ids, labels.tolist()):
        row = rows07[int(sid)]
        fallback = DEFAULT_CLASS_NAMES[int(label)] if 0 <= int(label) < len(DEFAULT_CLASS_NAMES) else str(label)
        name = row.get("class_name") or row.get("true_class_name") or fallback
        result[int(label)] = str(name)
    return result


def _integration_mean(values: Tensor) -> Tensor:
    if values.shape[1] < 2:
        return values.mean(dim=1)
    weights = torch.ones(values.shape[1], dtype=torch.float64)
    weights[[0, -1]] *= 0.5
    weights /= weights.sum()
    return (values.double() * weights[None, :]).sum(dim=1)


def _population_descriptors(gammas: Tensor, days: float) -> tuple[Tensor, np.ndarray, np.ndarray]:
    identity = torch.linspace(0.0, 1.0, gammas.shape[1], dtype=torch.float64)
    displacement = gammas.double() - identity[None, :]
    shift = _integration_mean(displacement)
    centered = displacement - shift[:, None]
    if displacement.shape[1] > 1:
        weights = torch.ones(displacement.shape[1], dtype=torch.float64)
        weights[[0, -1]] *= 0.5; weights /= weights.sum()
        nonlinear = torch.sqrt((centered.square() * weights[None, :]).sum(dim=1).clamp_min(0.0))
    else:
        nonlinear = torch.zeros_like(shift)
    return displacement, (shift * days).numpy(), (nonlinear * days).numpy()


def _plot_heatmap(path: Path, displacement_days: np.ndarray, mean_shift_days: np.ndarray) -> None:
    order = np.argsort(mean_shift_days, kind="stable")
    fig, ax = plt.subplots(figsize=(12, 9))
    image = ax.imshow(displacement_days[order], aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_xlabel("canonical registration grid")
    ax.set_ylabel("all numerically-valid samples (sorted by label-free mean displacement)")
    ax.set_title("Oracle true-class gamma displacement population")
    fig.colorbar(image, ax=ax, label="gamma(t)-t [days]")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_quantile_band(path: Path, displacement_days: np.ndarray, seed: int) -> None:
    x = np.linspace(0.0, 365.0, displacement_days.shape[1])
    q10, q25, q50, q75, q90 = np.quantile(displacement_days, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
    rng = np.random.default_rng(seed)
    subset = rng.choice(displacement_days.shape[0], size=min(80, displacement_days.shape[0]), replace=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx in subset:
        ax.plot(x, displacement_days[idx], linewidth=0.5, alpha=0.12)
    ax.fill_between(x, q10, q90, alpha=0.15, label="10-90%")
    ax.fill_between(x, q25, q75, alpha=0.25, label="25-75%")
    ax.plot(x, q50, linewidth=2.0, label="median")
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("canonical day"); ax.set_ylabel("displacement [days]")
    ax.set_title("Population displacement quantile bands"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _transition_code(values: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    order = ["wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong"]
    mapping = {name: i for i, name in enumerate(order)}
    return np.asarray([mapping.get(str(v), -1) for v in values], dtype=np.int64), order


def _scatter_numeric(path: Path, x: np.ndarray, y: np.ndarray, color: np.ndarray, title: str, color_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(x, y, c=color, s=8, alpha=0.55, cmap="viridis")
    ax.set_xlabel("mean displacement [days]"); ax.set_ylabel("nonlinear residual [days]"); ax.set_title(title)
    fig.colorbar(points, ax=ax, label=color_label); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _scatter_class(path: Path, x: np.ndarray, y: np.ndarray, labels: np.ndarray, names: dict[int, str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in sorted(np.unique(labels).tolist()):
        mask = labels == cid
        ax.scatter(x[mask], y[mask], s=8, alpha=0.5, label=names.get(int(cid), str(cid)))
    ax.set_xlabel("mean displacement [days]"); ax.set_ylabel("nonlinear residual [days]"); ax.set_title(title)
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _scatter_transition(path: Path, x: np.ndarray, y: np.ndarray, transitions: Sequence[str], title: str) -> None:
    codes, order = _transition_code(transitions)
    fig, ax = plt.subplots(figsize=(9, 7))
    for code, name in enumerate(order):
        mask = codes == code
        ax.scatter(x[mask], y[mask], s=8, alpha=0.5, label=name)
    ax.set_xlabel("mean displacement [days]"); ax.set_ylabel("nonlinear residual [days]"); ax.set_title(title)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _embedding(vectors: Tensor, indices: np.ndarray) -> np.ndarray:
    subset_vectors = vectors[torch.as_tensor(indices, dtype=torch.long)]
    inner = subset_vectors @ subset_vectors.T
    distance = torch.acos(inner.clamp(-1.0, 1.0)); distance.fill_diagonal_(0.0)
    return classical_mds(distance, dimensions=2).numpy()


def _phase_plot_class(path: Path, coords: np.ndarray, labels: np.ndarray, names: dict[int, str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in sorted(np.unique(labels).tolist()):
        mask = labels == cid
        ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.65, label=names.get(int(cid), str(cid)))
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2"); ax.set_title(title); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _phase_plot_numeric(path: Path, coords: np.ndarray, values: np.ndarray, title: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    p = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=12, alpha=0.7, cmap="viridis")
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2"); ax.set_title(title); fig.colorbar(p, ax=ax, label=label)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _phase_plot_transition(path: Path, coords: np.ndarray, transitions: Sequence[str], title: str) -> None:
    codes, order = _transition_code(transitions)
    fig, ax = plt.subplots(figsize=(9, 7))
    for code, name in enumerate(order):
        mask = codes == code
        ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.65, label=name)
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2"); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_embedding_set(prefix: Path, coords: np.ndarray, indices: np.ndarray, labels: np.ndarray, names: dict[int, str], delta_p: np.ndarray, transitions: Sequence[str], identity_distance: np.ndarray, best_alpha: np.ndarray, *, numbered: bool) -> None:
    prefix.mkdir(parents=True, exist_ok=True)
    base = prefix.parent if numbered else prefix
    def target(number: str, name: str) -> Path:
        return (base / f"{number}_{name}.png") if numbered else (prefix / f"{name}.png")
    _phase_plot_class(target("07", "phase_space_true_class"), coords, labels[indices], names, "Phase-space visualization colored by true class")
    _phase_plot_numeric(target("08", "phase_space_delta_p"), coords, delta_p[indices], "Same Phase coordinates colored by delta p_true", "delta p_true")
    _phase_plot_transition(target("09", "phase_space_transition"), coords, [transitions[i] for i in indices], "Same Phase coordinates colored by hard transition")
    _phase_plot_numeric(target("10", "phase_space_identity_distance"), coords, identity_distance[indices], "Same Phase coordinates colored by d_Gamma(gamma,id)", "Phase distance to identity")
    _phase_plot_numeric(target("11", "phase_space_best_alpha_diagnostic"), coords, best_alpha[indices], "Same Phase coordinates colored by experiment-08 best alpha", "diagnostic best alpha")


def _plot_pair_distribution(path: Path, within: np.ndarray, cross: np.ndarray) -> None:
    values = np.concatenate([within, cross])
    bins = np.linspace(float(np.min(values)), float(np.max(values)), 60)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(within, bins=bins, density=True, alpha=0.45, label="within true class")
    ax.hist(cross, bins=bins, density=True, alpha=0.45, label="cross true class")
    ax.set_xlabel("Fisher-Rao Phase distance"); ax.set_ylabel("density"); ax.set_title("Within-class vs cross-class Phase distance")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_knn_distributions(path: Path, stats: dict[int, tuple[np.ndarray, np.ndarray]], which: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for k, pair in sorted(stats.items()):
        values = pair[which]
        ax.hist(values, bins=30, density=True, histtype="step", linewidth=1.8, label=f"k={k}")
    ax.set_xlabel("same-class neighbor fraction" if which == 0 else "class mixing entropy")
    ax.set_ylabel("density"); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_knn_vs_delta(path: Path, stats: dict[int, tuple[np.ndarray, np.ndarray]], delta_p: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for k, (same, _entropy) in sorted(stats.items()):
        ax.scatter(same, delta_p, s=5, alpha=0.18, label=f"k={k}")
    ax.set_xlabel("same-class Phase-neighbor fraction"); ax.set_ylabel("delta p_true"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _per_class_outputs(output_dir: Path, displacement_days: np.ndarray, mean_shift: np.ndarray, nonlinear: np.ndarray, labels: np.ndarray, names: dict[int, str]) -> list[dict]:
    root = output_dir / "per_class"; root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    x = np.linspace(0.0, 365.0, displacement_days.shape[1])
    fig, ax = plt.subplots(figsize=(12, 7))
    for cid in sorted(np.unique(labels).tolist()):
        mask = labels == cid; curves = displacement_days[mask]
        q10, q25, q50, q75, q90 = np.quantile(curves, [0.10,0.25,0.50,0.75,0.90], axis=0)
        name = names.get(int(cid), str(cid))
        ax.plot(x, q50, linewidth=1.5, label=name)
        cfig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].fill_between(x, q10, q90, alpha=0.15); axes[0].fill_between(x, q25, q75, alpha=0.25); axes[0].plot(x, q50, linewidth=1.8)
        axes[0].axhline(0.0, linestyle="--", linewidth=0.8); axes[0].set_title(f"{name}: displacement bands"); axes[0].set_xlabel("canonical day"); axes[0].set_ylabel("days")
        axes[1].hist(mean_shift[mask], bins=35); axes[1].set_title("mean displacement"); axes[1].set_xlabel("days")
        axes[2].hist(nonlinear[mask], bins=35); axes[2].set_title("nonlinear residual"); axes[2].set_xlabel("days")
        cfig.tight_layout(); cfig.savefig(root / f"class_{cid:02d}_{name}.png", dpi=170); plt.close(cfig)
        rows.append({"class_id": int(cid), "class_name": name, "n_samples": int(mask.sum()), "mean_shift_median_days": float(np.median(mean_shift[mask])), "mean_shift_iqr_days": float(np.quantile(mean_shift[mask],.75)-np.quantile(mean_shift[mask],.25)), "nonlinear_residual_median_days": float(np.median(nonlinear[mask])), "nonlinear_residual_iqr_days": float(np.quantile(nonlinear[mask],.75)-np.quantile(nonlinear[mask],.25))})
    ax.axhline(0.0, linestyle="--", linewidth=0.8); ax.set_xlabel("canonical day"); ax.set_ylabel("median displacement [days]"); ax.set_title("Per-class median displacement functions (oracle-only coloring)"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(output_dir / "15_per_class_displacement_summary.png", dpi=180); plt.close(fig)
    _write_csv(root / "per_class_descriptor_summary.csv", rows)
    return rows


def _describe(values: np.ndarray) -> dict[str, float]:
    return {"median": float(np.median(values)), "q25": float(np.quantile(values,.25)), "q75": float(np.quantile(values,.75)), "iqr": float(np.quantile(values,.75)-np.quantile(values,.25))}


def run(args) -> None:
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids, labels, gammas = _load_population(args.stage_a_cache.resolve())
    if args.expected_valid_count and len(sample_ids) != args.expected_valid_count:
        raise RuntimeError(f"expected {args.expected_valid_count} numerically-valid gamma values, found {len(sample_ids)}")
    rows07, rows08 = _join_audit_rows(sample_ids, args.audit07_sample_csv.resolve(), args.audit08_sample_csv.resolve())
    names = _class_names(rows07, sample_ids, labels)
    delta_p = np.asarray([_f(rows07[sid].get("delta_prob_vs_no")) for sid in sample_ids], dtype=np.float64)
    transitions = [rows07[sid].get("transition_vs_no", "") for sid in sample_ids]
    best_alpha = np.asarray([_f(rows08[sid].get("best_alpha_by_true_prob")) for sid in sample_ids], dtype=np.float64)

    print(f"ORACLE_PHASE_POPULATION_09_START|n={len(sample_ids)}|grid={gammas.shape[1]}|registration_calls=0|clustering=false", flush=True)
    displacement, mean_shift, nonlinear = _population_descriptors(gammas, args.days_per_year)
    displacement_days = displacement.numpy() * args.days_per_year
    vectors = gamma_to_unit_phase_vectors(gammas)
    identity_distance = phase_distance_to_identity(vectors).numpy()

    random_idx = deterministic_random_subset(len(sample_ids), args.mds_random_size, args.random_seed)
    equal_idx = deterministic_equal_class_subset(labels, per_class=args.mds_equal_per_class, seed=args.random_seed + 1)
    random_coords = _embedding(vectors, random_idx)
    equal_coords = _embedding(vectors, equal_idx)
    random_coord_map = {int(idx):(float(x),float(y)) for idx,(x,y) in zip(random_idx.tolist(), random_coords.tolist())}

    print(f"ORACLE_PHASE_POPULATION_09_KNN_START|n={len(sample_ids)}|kmax={max(args.knn_k)}|device={args.knn_device}", flush=True)
    device = torch.device(args.knn_device if args.knn_device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    neighbor_indices, neighbor_distances = exact_phase_knn(vectors, k_max=max(args.knn_k), device=device, block_size=args.knn_block_size)
    knn_stats = knn_label_statistics(neighbor_indices, labels, k_values=args.knn_k)
    print("ORACLE_PHASE_POPULATION_09_KNN_READY", flush=True)

    within_l, within_r, cross_l, cross_r = sample_within_cross_pairs(labels, count_each=args.pair_samples, seed=args.random_seed + 2)
    within_dist = phase_pair_distances(vectors, within_l, within_r)
    cross_dist = phase_pair_distances(vectors, cross_l, cross_r)
    effect = common_language_effect(within_dist, cross_dist)

    rows: list[dict] = []
    for idx, (sid, cid) in enumerate(zip(sample_ids, labels.tolist())):
        coord = random_coord_map.get(idx, (float("nan"), float("nan")))
        row = {
            "sample_id": int(sid), "true_class": int(cid), "class_name": names.get(int(cid), str(cid)),
            "mean_displacement_days": float(mean_shift[idx]), "nonlinear_residual_days": float(nonlinear[idx]),
            "phase_distance_to_identity": float(identity_distance[idx]),
            "pred_no": rows07[sid].get("pred_no", rows07[sid].get("pred_no_phase", "")),
            "pred_gamma": rows07[sid].get("pred_oracle_gamma", ""),
            "transition_type": transitions[idx],
            "p_true_no": _f(rows07[sid].get("true_prob_no")),
            "p_true_gamma": _f(rows07[sid].get("true_prob_oracle_gamma")),
            "delta_p_true": float(delta_p[idx]), "best_alpha_diagnostic": float(best_alpha[idx]),
            "mds_x": coord[0], "mds_y": coord[1], "mds_subset_flag": bool(idx in random_coord_map),
        }
        for k in args.knn_k:
            same, entropy = knn_stats[int(k)]
            row[f"same_class_knn_k{k}"] = float(same[idx]); row[f"class_entropy_knn_k{k}"] = float(entropy[idx])
        rows.append(row)
    _write_csv(output_dir / "01_sample_phase_population.csv", rows)
    np.savez_compressed(output_dir / "02_phase_curves.npz", sample_id=np.asarray(sample_ids,dtype=np.int64), true_class=labels, gamma=gammas.numpy().astype(np.float32), displacement=displacement.numpy().astype(np.float32), grid=np.linspace(0.0,1.0,gammas.shape[1],dtype=np.float32))

    _plot_heatmap(output_dir / "03_global_displacement_heatmap.png", displacement_days, mean_shift)
    _plot_quantile_band(output_dir / "04_global_displacement_quantile_band.png", displacement_days, args.random_seed)
    _scatter_class(output_dir / "05_shift_vs_nonlinearity_true_class.png", mean_shift, nonlinear, labels, names, "Mean displacement vs nonlinear residual, colored by true class")
    _scatter_numeric(output_dir / "06_shift_vs_nonlinearity_delta_p.png", mean_shift, nonlinear, delta_p, "Mean displacement vs nonlinear residual, colored by delta p_true", "delta p_true")
    _scatter_transition(output_dir / "06b_shift_vs_nonlinearity_transition.png", mean_shift, nonlinear, transitions, "Same descriptor coordinates colored by hard transition")
    _scatter_numeric(output_dir / "06c_shift_vs_nonlinearity_identity_distance.png", mean_shift, nonlinear, identity_distance, "Same descriptor coordinates colored by d_Gamma(gamma,id)", "Phase distance to identity")
    _scatter_numeric(output_dir / "06d_shift_vs_nonlinearity_best_alpha.png", mean_shift, nonlinear, best_alpha, "Same descriptor coordinates colored by experiment-08 best alpha", "diagnostic best alpha")

    _plot_embedding_set(output_dir / "_random", random_coords, random_idx, labels, names, delta_p, transitions, identity_distance, best_alpha, numbered=True)
    equal_root = output_dir / "phase_space_equal_class"; equal_root.mkdir(exist_ok=True)
    _plot_embedding_set(equal_root, equal_coords, equal_idx, labels, names, delta_p, transitions, identity_distance, best_alpha, numbered=False)
    _write_csv(output_dir / "phase_space_random_subset.csv", [{"sample_id":sample_ids[int(idx)],"true_class":int(labels[idx]),"mds_x":float(x),"mds_y":float(y)} for idx,(x,y) in zip(random_idx,random_coords)])
    _write_csv(equal_root / "equal_class_subset.csv", [{"sample_id":sample_ids[int(idx)],"true_class":int(labels[idx]),"mds_x":float(x),"mds_y":float(y)} for idx,(x,y) in zip(equal_idx,equal_coords)])

    _plot_pair_distribution(output_dir / "12_within_vs_cross_class_distance.png", within_dist, cross_dist)
    _plot_knn_distributions(output_dir / "13_knn_same_class_fraction.png", knn_stats, 0)
    _plot_knn_distributions(output_dir / "14_knn_class_entropy.png", knn_stats, 1)
    _plot_knn_vs_delta(output_dir / "14b_knn_same_class_fraction_vs_delta_p.png", knn_stats, delta_p)
    per_class = _per_class_outputs(output_dir, displacement_days, mean_shift, nonlinear, labels, names)

    knn_summary: list[dict] = []
    for k, (same, entropy) in sorted(knn_stats.items()):
        scopes = [("ALL", np.ones(len(labels),dtype=bool))] + [(names.get(int(cid),str(cid)), labels==cid) for cid in sorted(np.unique(labels).tolist())]
        for scope, mask in scopes:
            knn_summary.append({"scope":scope,"k":int(k),"n":int(mask.sum()),"same_class_fraction_mean":float(np.mean(same[mask])),"same_class_fraction_median":float(np.median(same[mask])),"class_entropy_mean":float(np.mean(entropy[mask])),"class_entropy_median":float(np.median(entropy[mask])),"delta_p_mean":float(np.nanmean(delta_p[mask]))})
    _write_csv(output_dir / "knn_population_summary.csv", knn_summary)
    np.savez_compressed(output_dir / "phase_knn_neighbors.npz", sample_id=np.asarray(sample_ids,dtype=np.int64), neighbor_indices=neighbor_indices, neighbor_distances=neighbor_distances)

    population_summary = {
        "n_numerically_valid": len(sample_ids), "registration_grid_size": int(gammas.shape[1]),
        "mean_displacement_days": _describe(mean_shift), "nonlinear_residual_days": _describe(nonlinear), "phase_distance_to_identity": _describe(identity_distance),
        "within_class_phase_distance": _describe(within_dist), "cross_class_phase_distance": _describe(cross_dist), "within_vs_cross_effect": effect,
        "knn": {str(k): {"same_class_fraction_mean":float(np.mean(pair[0])),"same_class_fraction_median":float(np.median(pair[0])),"class_entropy_mean":float(np.mean(pair[1])),"class_entropy_median":float(np.median(pair[1]))} for k,pair in sorted(knn_stats.items())},
        "random_mixing_reference": {
            "same_class_probability": float(sum((labels == cid).mean() ** 2 for cid in np.unique(labels))),
            "global_class_entropy": float(-sum(p * math.log(p) for p in [(labels == cid).mean() for cid in np.unique(labels)] if p > 0)),
        },
        "allowed_interpretations": ["cross_class_shared_structure", "continuous_cross_class_shared_structure", "mostly_class_specific_structure"],
        "automatic_interpretation": None,
        "clustering_performed": False, "group_count_selected": False, "representative_phase_constructed": False,
    }
    _json_dump(output_dir / "16_population_structure_summary.json", population_summary)
    manifest = {
        "experiment":"09_oracle_true_class_phase_population_structure_diagnostic", "phase":"III", "oracle_only":True,
        "stage_a_cache":str(args.stage_a_cache.resolve()), "audit07_sample_csv":str(args.audit07_sample_csv.resolve()), "audit08_sample_csv":str(args.audit08_sample_csv.resolve()),
        "n_population":len(sample_ids), "all_numerically_valid_used":True, "production_legality_used":False, "gain_filter_used":False, "beneficial_harmful_filter_used":False,
        "class_center_used":False, "group_center_used":False, "phase_confirmation_used":False, "registration_solver_called":False,
        "clustering_performed":False, "group_count_selected":False, "representative_phase_constructed":False,
        "distance_metric":"Fisher-Rao phase distance d_Gamma via normalized sqrt(gamma_dot) sphere",
        "mds":{"method":"classical metric MDS for visualization only","random_subset_size":int(len(random_idx)),"random_seed":int(args.random_seed),"random_sampling_rule":"label-free uniform sample","equal_class_subset_size":int(len(equal_idx)),"equal_class_per_class":int(args.mds_equal_per_class),"equal_class_oracle_only":True},
        "pair_audit":{"within_pairs":int(len(within_dist)),"cross_pairs":int(len(cross_dist)),"seed":int(args.random_seed+2)},
        "knn":{"k":list(map(int,args.knn_k)),"population":"all numerically-valid samples","neighbor_search_uses_labels":False,"labels_used_only_posthoc":True,"device":str(device),"block_size":int(args.knn_block_size)},
    }
    _json_dump(output_dir / "00_manifest.json", manifest)
    readme = f"""# 09 Oracle True-Class Phase Population Structure Diagnostic\n\n本实验只观察全部数值有效的 oracle true-class 样本级配准变换在 Phase 空间中的群体结构。\n\n- population: {len(sample_ids)} / expected {args.expected_valid_count}\n- registration solver calls: 0\n- production legality / gain / class dispersion filtering: 不使用\n- beneficial / harmful: 只作为事后着色，不过滤 population\n- Phase metric: Fisher–Rao $d_\\Gamma$，不是 $\\|\\gamma_i-\\gamma_j\\|_2$\n- clustering: **未执行**\n- automatic group count: **未执行**\n- representative Phase: **未构造**\n\n二维 MDS 只用于可视化。主 MDS 使用 label-free random subset ({len(random_idx)} samples)；另有 oracle-only equal-class subset ({len(equal_idx)} samples)。所有总体 displacement、pair-distance 抽样统计和全体 kNN 统计均基于完整 numerically-valid population。\n\nPhase kNN 的邻居搜索只使用 $d_\\Gamma$；真实类别仅在邻居确定以后计算 same-class fraction 与 mixing entropy。\n\n本实验不自动选择阶段 III 的 A/B/C 解释，只输出证据。允许的最终理论解释仅为：跨类别共享结构、连续但跨类别共享的 Phase 结构、或 Phase 主要按类别分裂。\n"""
    (output_dir / "README_中文说明.md").write_text(readme, encoding="utf-8")
    print(f"ORACLE_PHASE_POPULATION_09_DONE|output={output_dir}|n={len(sample_ids)}|registration_calls=0|clustering=false", flush=True)


def _parse_k(text: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(v.strip()) for v in text.split(",") if v.strip())))
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("knn-k must contain positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a-cache", type=Path, required=True)
    parser.add_argument("--audit07-sample-csv", type=Path, required=True)
    parser.add_argument("--audit08-sample-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-valid-count", type=int, default=10634)
    parser.add_argument("--days-per-year", type=float, default=365.0)
    parser.add_argument("--random-seed", type=int, default=20260812)
    parser.add_argument("--mds-random-size", type=int, default=1000)
    parser.add_argument("--mds-equal-per-class", type=int, default=100)
    parser.add_argument("--pair-samples", type=int, default=100000)
    parser.add_argument("--knn-k", type=_parse_k, default=DEFAULT_K)
    parser.add_argument("--knn-device", default="auto")
    parser.add_argument("--knn-block-size", type=int, default=512)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
