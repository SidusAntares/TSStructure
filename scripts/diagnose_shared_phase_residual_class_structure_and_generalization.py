#!/usr/bin/env python3
"""11: Shared-Phase residual class structure and generalization diagnostic.

This oracle-only experiment consumes the fixed oracle sample-gamma population
from experiment 06/07 and the exact leave-one-class-out shared Phase centers
saved by experiment 10.  It factorizes each source->target sample gamma into

    gamma_sample = delta_minus_class o residual

under the repository's actual gamma direction, enforces a Fisher--Rao
reconstruction hard gate, estimates residual class centers, performs strictly
within-class 5-fold cross-fitting, and evaluates only frozen-classifier time
position changes.  It never reruns registration, trains parameters, clusters
residuals, tunes alpha, gates classes, or changes the Phase mechanism.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import visualize_stage2_phase_alignment as phasevis
from methods.structure_da.confirmed_phase_view import align_target_positions_to_source
from methods.structure_da.phase_geometry import phase_distance
from methods.structure_da.residual_phase_diagnostic import (
    build_residual_population,
    compose_phase,
    deterministic_class_folds,
    pairwise_center_distances,
    phase_distances_to_center,
    residual_center_summary,
)
from methods.structure_da.sample_phase_diagnostic import TOnlyPhaseRegistration
from methods.structure_da.shared_domain_phase_diagnostic import (
    phase_distance_to_identity_for_gamma,
    sample_equal_weights,
    weighted_frechet_mean_gamma,
)

EXPECTED_SAMPLES = 10634
EXPECTED_CLASSES = 10
DEFAULT_CV_FOLDS = 5
DEFAULT_CV_SEED = 20260812
DEFAULT_RECONSTRUCTION_TOLERANCE = 1e-7


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_div(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def _binary_f1(labels: np.ndarray, predictions: np.ndarray, class_id: int) -> float:
    tp = int(np.sum((labels == class_id) & (predictions == class_id)))
    fp = int(np.sum((labels != class_id) & (predictions == class_id)))
    fn = int(np.sum((labels == class_id) & (predictions != class_id)))
    p = _safe_div(tp, tp + fp); r = _safe_div(tp, tp + fn)
    if not math.isfinite(p) or not math.isfinite(r) or p + r == 0.0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def _macro_f1(labels: np.ndarray, predictions: np.ndarray, class_ids: Sequence[int]) -> float:
    return float(np.mean([_binary_f1(labels, predictions, int(c)) for c in class_ids]))


def _macro_recall(labels: np.ndarray, predictions: np.ndarray, class_ids: Sequence[int]) -> float:
    values = []
    for c in class_ids:
        mask = labels == int(c)
        values.append(float(np.mean(predictions[mask] == int(c))) if np.any(mask) else float("nan"))
    values = [v for v in values if math.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def _transition(before: bool, after: bool) -> str:
    if before and after:
        return "correct_to_correct"
    if before and not after:
        return "correct_to_wrong"
    if not before and after:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def _registration_from_payload(payload: dict) -> TOnlyPhaseRegistration:
    values = dict(payload)
    gamma = values.get("gamma")
    if isinstance(gamma, Tensor):
        values["gamma"] = gamma.detach().cpu().double()
    values["reject_reasons"] = tuple(values.get("reject_reasons", ()))
    return TOnlyPhaseRegistration(**values)


def _load_gamma_population(path: Path) -> tuple[list[TOnlyPhaseRegistration], list[int], np.ndarray, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"11 requires experiment-06 Stage-A cache and never recomputes registration: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = [_registration_from_payload(row) for row in payload.get("records", ())]
    sample_ids = [int(v) for v in payload.get("sample_ids", ())]
    labels = np.asarray(payload.get("true_classes", ()), dtype=np.int64)
    if len(records) != len(sample_ids) or len(records) != labels.size:
        raise ValueError("06 Stage-A cache arrays have inconsistent lengths")
    kept_records: list[TOnlyPhaseRegistration] = []
    kept_ids: list[int] = []
    kept_labels: list[int] = []
    gammas: list[Tensor] = []
    for record, sid, label in zip(records, sample_ids, labels.tolist()):
        if not bool(record.numerically_valid) or not isinstance(record.gamma, Tensor):
            continue
        gamma = record.gamma.detach().cpu().double().flatten()
        if not torch.isfinite(gamma).all().item() or not torch.all(gamma[1:] > gamma[:-1]).item():
            continue
        kept_records.append(record); kept_ids.append(int(sid)); kept_labels.append(int(label)); gammas.append(gamma)
    if not gammas:
        raise RuntimeError("06 cache contains no numerically-valid oracle true-class gamma")
    if len({int(g.numel()) for g in gammas}) != 1:
        raise ValueError("cached gamma values do not share one registration grid")
    return kept_records, kept_ids, np.asarray(kept_labels, dtype=np.int64), torch.stack(gammas)


def _normalize_population_identity(
    records: Sequence[TOnlyPhaseRegistration],
    sample_ids: Sequence[int],
    labels: np.ndarray,
    dataset_parcels: Sequence[int],
) -> tuple[list[TOnlyPhaseRegistration], list[int], np.ndarray]:
    parcels = [int(v) for v in dataset_parcels]
    ids = [int(v) for v in sample_ids]
    if set(ids) == set(parcels):
        return list(records), ids, labels.copy()
    if ids and all(0 <= value < len(parcels) for value in ids):
        mapped = [parcels[value] for value in ids]
        return [replace(record, sample_id=mapped[i]) for i, record in enumerate(records)], mapped, labels.copy()
    raise ValueError("06 sample IDs are neither parcel IDs nor valid target-test local indices")


def _load_07_rows(path: Path, required_ids: Sequence[int]) -> dict[int, dict[str, str]]:
    rows = {int(float(row["sample_id"])): row for row in _read_csv(path)}
    missing = sorted(set(map(int, required_ids)) - set(rows))
    if missing:
        raise ValueError(f"07 sample CSV is missing experiment-11 parcels: {missing[:5]}")
    return rows




def _load_experiment10_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "10_class_balanced_shared_domain_phase_leave_one_class_out":
        raise ValueError("experiment-10 manifest protocol mismatch")
    required_false = (
        "production_legality_filter", "beneficial_harmful_filter", "clustering",
        "M1_M2", "class_specific_alpha", "teacher_student", "stable_label",
        "training_updates", "domain_shape_transport",
        "classification_outcomes_used_to_fit_center",
    )
    for key in required_false:
        if bool(payload.get(key, False)):
            raise ValueError(f"experiment-10 manifest violates frozen protocol: {key}=true")
    integrity = str(payload.get("held_out_class_integrity", ""))
    if "zero" not in integrity.lower():
        raise ValueError("experiment-10 manifest does not certify zero held-out-class estimation weight")
    return payload


def _load_shared_centers(path: Path, classes: Sequence[str], grid_size: int) -> dict[int, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"11 requires experiment-10 shared Phase curves and never recomputes them: {path}"
        )
    payload = np.load(path, allow_pickle=False)
    if "class_names" in payload:
        saved = [str(v) for v in payload["class_names"].tolist()]
        if saved != list(classes):
            raise ValueError("experiment-10 class_names differ from runtime classes")
    result: dict[int, Tensor] = {}
    for class_id in range(len(classes)):
        key = f"delta_minus_class_{class_id}"
        if key not in payload:
            raise KeyError(f"experiment-10 NPZ is missing {key}")
        gamma = torch.as_tensor(payload[key], dtype=torch.float64).flatten()
        if gamma.numel() != int(grid_size):
            raise ValueError(f"{key} grid size differs from sample gamma grid")
        if not torch.isfinite(gamma).all().item() or not torch.all(gamma[1:] > gamma[:-1]).item():
            raise ValueError(f"{key} must be finite and strictly increasing")
        result[class_id] = gamma
    return result


def _full_centers_and_geometry(
    residuals: Tensor, labels: np.ndarray, classes: Sequence[str]
) -> tuple[dict[int, object], list[dict], Tensor]:
    centers: dict[int, object] = {}
    rows: list[dict] = []
    center_stack: list[Tensor] = []
    for class_id, class_name in enumerate(classes):
        subset = residuals[torch.from_numpy(labels == class_id)]
        summary = residual_center_summary(class_id, subset)
        centers[class_id] = summary
        center_stack.append(summary.center)
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "n_samples": int(subset.shape[0]),
            "residual_center_to_identity": summary.center_to_identity,
            "within_class_dispersion_mean_squared": summary.dispersion_mean_squared,
            "distance_to_center_median": summary.distance_median,
            "distance_to_center_q25": summary.distance_q25,
            "distance_to_center_q75": summary.distance_q75,
            "distance_to_center_iqr": summary.distance_q75 - summary.distance_q25,
            "distance_to_center_q90": summary.distance_q90,
            "frechet_objective": summary.estimator.objective,
            "frechet_iterations": summary.estimator.iterations,
            "frechet_converged": summary.estimator.converged,
            "frechet_final_tangent_norm": summary.estimator.tangent_norm,
        })
    matrix = pairwise_center_distances(torch.stack(center_stack))
    return centers, rows, matrix


def _crossfit_centers(
    residuals: Tensor,
    sample_ids: Sequence[int],
    labels: np.ndarray,
    classes: Sequence[str],
    *,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, dict[tuple[int, int], Tensor], list[dict], dict[int, Tensor]]:
    folds = deterministic_class_folds(sample_ids, labels, n_folds=n_folds, seed=seed)
    full_centers = {
        c: residual_center_summary(c, residuals[torch.from_numpy(labels == c)]).center
        for c in range(len(classes))
    }
    fold_centers: dict[tuple[int, int], Tensor] = {}
    rows: list[dict] = []
    provisional: dict[tuple[int, int], dict] = {}
    for class_id, class_name in enumerate(classes):
        for fold in range(n_folds):
            train_mask_np = (labels == class_id) & (folds != fold)
            test_mask_np = (labels == class_id) & (folds == fold)
            train = residuals[torch.from_numpy(train_mask_np)]
            test = residuals[torch.from_numpy(test_mask_np)]
            estimator = weighted_frechet_mean_gamma(train, sample_equal_weights(train.shape[0]))
            center = estimator.gamma
            fold_centers[(class_id, fold)] = center
            train_dist = phase_distances_to_center(train, center)
            test_dist = phase_distances_to_center(test, center)
            provisional[(class_id, fold)] = {
                "class_id": class_id,
                "class_name": class_name,
                "fold": fold,
                "n_train": int(train.shape[0]),
                "n_test": int(test.shape[0]),
                "center_to_full_center_distance": float(phase_distance(center, full_centers[class_id]).item()),
                "center_to_identity_distance": phase_distance_to_identity_for_gamma(center),
                "train_dispersion": float(train_dist.square().mean().item()),
                "test_distance_to_fold_center_median": float(torch.median(test_dist).item()),
                "test_distance_to_fold_center_q90": float(torch.quantile(test_dist, 0.9).item()),
                "frechet_objective": estimator.objective,
                "frechet_converged": estimator.converged,
            }
        stack = torch.stack([fold_centers[(class_id, f)] for f in range(n_folds)])
        pairwise = pairwise_center_distances(stack)
        off = pairwise[~torch.eye(n_folds, dtype=torch.bool)]
        pair_mean = float(off.mean().item())
        pair_max = float(off.max().item())
        for fold in range(n_folds):
            others = torch.cat([pairwise[fold, :fold], pairwise[fold, fold + 1:]])
            item = provisional[(class_id, fold)]
            item["fold_center_pairwise_class_mean"] = pair_mean
            item["fold_center_pairwise_class_max"] = pair_max
            item["this_fold_mean_distance_to_other_fold_centers"] = float(others.mean().item())
            rows.append(item)
    return folds, fold_centers, rows, full_centers


def _evaluate_five_way(
    *,
    model,
    loader,
    shared_centers: Mapping[int, Tensor],
    composite_centers: Mapping[tuple[int, int], Tensor],
    fold_by_parcel: Mapping[int, int],
    rows07: Mapping[int, Mapping[str, str]],
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = phasevis._move_batch(raw_batch, device)
            backbone = model.forward_backbone(
                batch["pixels"], batch["valid_pixels"], batch["positions"],
                batch.get("extra"), time_mask=batch.get("time_mask"),
            )
            native = backbone.normalized_positions.detach()
            mask = backbone.time_mask.detach()
            labels = batch["label"].long()
            parcels = batch["parcel_index"].detach().cpu().long()

            shared_positions = native.clone()
            residual_positions = native.clone()
            for class_id in sorted(shared_centers):
                selected = labels == int(class_id)
                if torch.any(selected):
                    shared_positions[selected] = align_target_positions_to_source(
                        native[selected], mask[selected], shared_centers[int(class_id)]
                    )
            for row_index, parcel in enumerate(parcels.tolist()):
                class_id = int(labels[row_index].item())
                fold = int(fold_by_parcel[int(parcel)])
                gamma = composite_centers[(class_id, fold)]
                residual_positions[row_index:row_index + 1] = align_target_positions_to_source(
                    native[row_index:row_index + 1], mask[row_index:row_index + 1], gamma
                )

            no_out = model.forward_from_backbone(
                backbone, batch["positions"], batch.get("extra"), return_geometry=False,
            )
            shared_out = model.forward_from_backbone(
                backbone, batch["positions"], batch.get("extra"),
                temporal_positions_override=shared_positions, return_geometry=False,
            )
            residual_out = model.forward_from_backbone(
                backbone, batch["positions"], batch.get("extra"),
                temporal_positions_override=residual_positions, return_geometry=False,
            )
            no_prob = torch.softmax(no_out.logits.float(), dim=-1)
            shared_prob = torch.softmax(shared_out.logits.float(), dim=-1)
            residual_prob = torch.softmax(residual_out.logits.float(), dim=-1)
            no_pred = no_out.logits.argmax(dim=-1)
            shared_pred = shared_out.logits.argmax(dim=-1)
            residual_pred = residual_out.logits.argmax(dim=-1)

            for row_index, parcel in enumerate(parcels.tolist()):
                parcel = int(parcel); true_class = int(labels[row_index].item())
                base = rows07.get(parcel)
                if base is None:
                    raise KeyError(f"parcel {parcel} is missing from experiment 07")
                if int(float(base["true_class"])) != true_class:
                    raise ValueError("experiment-07 true class differs from current target-test label")
                replay_prob = float(no_prob[row_index, true_class].item())
                if abs(replay_prob - float(base["true_prob_no"])) > 5e-5:
                    raise RuntimeError("frozen No-Phase probability replay differs from experiment 07")
                if int(no_pred[row_index].item()) != int(float(base["pred_no"])):
                    raise RuntimeError("frozen No-Phase prediction replay differs from experiment 07")
                pred_shared = int(shared_pred[row_index].item())
                pred_residual = int(residual_pred[row_index].item())
                pred_no = int(float(base["pred_no"])); pred_tm = int(float(base["pred_timematch"])); pred_sample = int(float(base["pred_oracle_gamma"]))
                p_no = float(base["true_prob_no"]); p_tm = float(base["true_prob_timematch"]); p_sample = float(base["true_prob_oracle_gamma"])
                p_shared = float(shared_prob[row_index, true_class].item())
                p_residual = float(residual_prob[row_index, true_class].item())
                rows.append({
                    "sample_id": parcel,
                    "true_class": true_class,
                    "fold": int(fold_by_parcel[parcel]),
                    "pred_no_phase": pred_no,
                    "pred_timematch": pred_tm,
                    "pred_shared_phase": pred_shared,
                    "pred_shared_plus_residual_cv": pred_residual,
                    "pred_sample_gamma": pred_sample,
                    "true_prob_no_phase": p_no,
                    "true_prob_timematch": p_tm,
                    "true_prob_shared_phase": p_shared,
                    "true_prob_shared_plus_residual_cv": p_residual,
                    "true_prob_sample_gamma": p_sample,
                    "transition_residual_vs_shared": _transition(pred_shared == true_class, pred_residual == true_class),
                    "transition_residual_vs_no": _transition(pred_no == true_class, pred_residual == true_class),
                })
    return rows


def _classification_tables(sample_rows: Sequence[dict], classes: Sequence[str]) -> tuple[list[dict], dict]:
    rows = list(sample_rows)
    labels = np.asarray([int(r["true_class"]) for r in rows], dtype=np.int64)
    methods = {
        "no_phase": ("pred_no_phase", "true_prob_no_phase"),
        "timematch": ("pred_timematch", "true_prob_timematch"),
        "shared_phase": ("pred_shared_phase", "true_prob_shared_phase"),
        "shared_plus_residual_cv": ("pred_shared_plus_residual_cv", "true_prob_shared_plus_residual_cv"),
        "sample_gamma": ("pred_sample_gamma", "true_prob_sample_gamma"),
    }
    predictions = {name: np.asarray([int(r[pkey]) for r in rows], dtype=np.int64) for name, (pkey, _) in methods.items()}
    class_ids = list(range(len(classes)))
    per_class: list[dict] = []
    for class_id, class_name in enumerate(classes):
        mask = labels == class_id
        item = {"class_id": class_id, "class_name": class_name, "n_samples": int(mask.sum())}
        for name, (_pkey, prob_key) in methods.items():
            item[f"recall_{name}"] = float(np.mean(predictions[name][mask] == class_id))
            item[f"mean_true_prob_{name}"] = float(np.mean([float(rows[i][prob_key]) for i in np.flatnonzero(mask)]))
        item["delta_recall_residual_vs_shared"] = item["recall_shared_plus_residual_cv"] - item["recall_shared_phase"]
        item["delta_recall_residual_vs_no"] = item["recall_shared_plus_residual_cv"] - item["recall_no_phase"]
        item["delta_recall_residual_vs_sample"] = item["recall_shared_plus_residual_cv"] - item["recall_sample_gamma"]
        item["delta_true_prob_residual_vs_shared"] = item["mean_true_prob_shared_plus_residual_cv"] - item["mean_true_prob_shared_phase"]
        item["delta_true_prob_residual_vs_no"] = item["mean_true_prob_shared_plus_residual_cv"] - item["mean_true_prob_no_phase"]
        item["delta_true_prob_residual_vs_sample"] = item["mean_true_prob_shared_plus_residual_cv"] - item["mean_true_prob_sample_gamma"]
        per_class.append(item)
    overall = {}
    for name, (_pkey, prob_key) in methods.items():
        pred = predictions[name]
        overall[name] = {
            "accuracy": float(np.mean(pred == labels)),
            "macro_f1": _macro_f1(labels, pred, class_ids),
            "macro_recall": _macro_recall(labels, pred, class_ids),
            "mean_true_class_probability": float(np.mean([float(r[prob_key]) for r in rows])),
        }
    return per_class, overall


def _validate_experiment10_shared_replay(path: Path, per_class: Sequence[dict], *, tolerance: float = 5e-5) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    previous = {row["class_name"]: row for row in _read_csv(path)}
    for row in per_class:
        old = previous.get(str(row["class_name"]))
        if old is None:
            raise ValueError(f"experiment-10 CSV missing class {row['class_name']}")
        if abs(float(old["recall_shared_phase_minus_class"]) - float(row["recall_shared_phase"])) > tolerance:
            raise RuntimeError(f"shared Phase recall replay differs from experiment 10 for {row['class_name']}")
        if abs(float(old["mean_true_prob_shared_phase_minus_class"]) - float(row["mean_true_prob_shared_phase"])) > tolerance:
            raise RuntimeError(f"shared Phase true-prob replay differs from experiment 10 for {row['class_name']}")


def _transition_rows(sample_rows: Sequence[dict], classes: Sequence[str], key: str) -> list[dict]:
    output: list[dict] = []
    scopes = [("ALL", None)] + [(name, i) for i, name in enumerate(classes)]
    for name, class_id in scopes:
        rows = [r for r in sample_rows if class_id is None or int(r["true_class"]) == class_id]
        counts = {value: 0 for value in ("wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong")}
        for row in rows:
            counts[str(row[key])] += 1
        output.append({"class_id": "ALL" if class_id is None else class_id, "class_name": name, "n_samples": len(rows), **counts})
    return output


def _confusion(sample_rows: Sequence[dict], pred_key: str, n_classes: int) -> np.ndarray:
    result = np.zeros((n_classes, n_classes), dtype=np.int64)
    for row in sample_rows:
        result[int(row["true_class"]), int(row[pred_key])] += 1
    return result


def _plot_confusion(path: Path, matrix: np.ndarray, classes: Sequence[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, aspect="auto")
    fig.colorbar(im, ax=ax, label="count")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def _plot_distance_to_identity(path: Path, sample_rows: Sequence[dict], classes: Sequence[str]) -> None:
    values = [[float(r["residual_distance_to_identity"]) for r in sample_rows if int(r["true_class"]) == c] for c in range(len(classes))]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(values, showfliers=False)
    ax.set_xticks(range(1, len(classes) + 1), classes, rotation=45, ha="right")
    ax.set_ylabel("d_Gamma(residual, identity)")
    ax.set_title("Residual Phase distance to identity by true class")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def _plot_center_displacement(path: Path, centers: Mapping[int, object], classes: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    first = next(iter(centers.values())).center
    grid = np.linspace(0.0, 1.0, first.numel())
    days = grid * 364.0
    for class_id, class_name in enumerate(classes):
        gamma = centers[class_id].center.numpy()
        ax.plot(days, (gamma - grid) * 364.0, label=class_name, linewidth=1.4)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("canonical day"); ax.set_ylabel("residual center displacement (days)")
    ax.set_title("Full-class residual Phase center displacement")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def _plot_matrix(path: Path, matrix: np.ndarray, classes: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, aspect="auto")
    fig.colorbar(im, ax=ax, label="d_Gamma")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_title("Pairwise residual-class-center Fisher--Rao distance")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def _plot_dispersion(path: Path, rows: Sequence[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(rows)), [float(r["within_class_dispersion_mean_squared"]) for r in rows])
    ax.set_xticks(range(len(rows)), [str(r["class_name"]) for r in rows], rotation=45, ha="right")
    ax.set_ylabel("mean squared d_Gamma to residual center")
    ax.set_title("Within-class residual Phase dispersion")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def _plot_crossfit_stability(path: Path, rows: Sequence[dict], classes: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for class_id in range(len(classes)):
        subset = [r for r in rows if int(r["class_id"]) == class_id]
        x = np.full(len(subset), class_id, dtype=float) + np.linspace(-0.18, 0.18, len(subset))
        ax.scatter(x, [float(r["center_to_full_center_distance"]) for r in subset], s=22)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_ylabel("d_Gamma(fold center, full-class center)")
    ax.set_title("5-fold residual-center stability (raw distances; no threshold)")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def _plot_per_class(path: Path, rows: Sequence[dict], metric_prefix: str, ylabel: str) -> None:
    methods = ["no_phase", "timematch", "shared_phase", "shared_plus_residual_cv", "sample_gamma"]
    x = np.arange(len(rows)); width = 0.16
    fig, ax = plt.subplots(figsize=(14, 6))
    for offset, method in enumerate(methods):
        ax.bar(x + (offset - 2) * width, [float(r[f"{metric_prefix}_{method}"]) for r in rows], width=width, label=method)
    ax.set_xticks(x, [str(r["class_name"]) for r in rows], rotation=45, ha="right")
    ax.set_ylabel(ylabel); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def _plot_overall(path: Path, overall: Mapping[str, Mapping[str, float]]) -> None:
    methods = list(overall)
    x = np.arange(len(methods)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [overall[m]["accuracy"] for m in methods], width, label="Accuracy")
    ax.bar(x + width / 2, [overall[m]["macro_f1"] for m in methods], width, label="Macro-F1")
    ax.set_xticks(x, methods, rotation=20, ha="right"); ax.set_ylim(0.0, 1.0); ax.legend()
    ax.set_title("Frozen classifier: residual-Phase cross-fit comparison")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def _write_readme(path: Path) -> None:
    text = """# 实验 11：共享 Phase 剩余结构与泛化诊断

本实验是阶段 IV 的分叉实验，只判断类别条件剩余 Phase 是否稳定、可跨样本泛化并具有分类价值。实验不训练模型、不运行 registration、不使用旧 legality、不聚类 residual、不调 alpha、不做 residual gating，也不修改 Stage 2 控制流。

## 复合方向

仓库定义 `gamma(u_source)=u_target`，实验 10 的 `delta_-c` 也是 source→target；冻结分类器对 target 时间位置使用 `gamma^{-1}`。因此本实验从代码方向推导得到：

`gamma_sample = delta_-c ∘ residual`，所以 `residual = delta_-c^{-1} ∘ gamma_sample`。

脚本会把每个 residual 再与 `delta_-c` 复合，并用正式 Fisher--Rao `d_Gamma` 与原 sample gamma 比较。只要任一重构误差超过 manifest 中的纯数值 tolerance，实验立即停止，后续 residual 结果不会生成。

## 文件逐项说明

- `00_manifest.json`：记录输入文件、source→target 方向、复合公式、5-fold seed、重构数值 tolerance 以及全部禁止机制。无坐标轴。它只能证明实验协议配置，不能证明 residual 有分类价值。
- `01_residual_phase_curves.npz`：保存全部样本的 `gamma_true_class`、`shared_phase_minus_class`、`residual_phase`、sample fold，以及 10 个 full residual centers、50 个 cross-fit fold centers和相应 composite Phase。使用全部 10,634 个数值有效样本。用于精确复算几何，不可单独推出正式 UDA 机制。
- `02_residual_sample_summary.csv`：每个 target-test 样本一行。包含 residual 到 identity/full-center/fold-center 的 Phase 距离、重构误差、fold，以及五路冻结分类结果。距离越小表示 Phase 几何越近；分类概率越大仅代表当前 frozen classifier 更支持真实类别。
- `03_residual_per_class_summary.csv`：每类 full residual Fréchet center 到 identity 的距离、类内 mean squared dispersion、median/IQR/q90 distance 和 Karcher 收敛信息。中心离 identity 大但 dispersion 也大，不能解释为稳定类别 residual。
- `04_residual_distance_to_identity_by_class.png`：横轴真实类别，纵轴 `d_Gamma(r_i,id)`；箱线图使用该类全部 residual。看剥离共享主效应后哪些类仍有较强剩余 Phase；不能因为值大就定义独立 Phase group。
- `05_residual_class_center_displacement.png`：横轴 canonical day，纵轴 full residual center 的 `r(t)-t`（days），10 类叠图。看中心轮廓是否不同；视觉差异不能代替类内稳定性和交叉留出分类证据。
- `06_residual_class_center_pairwise_distance.png`：横纵轴真实类别，像素值为 full residual centers 之间 Fisher--Rao `d_Gamma`。看 corn 等中心是否真正远离其他类别；不聚类、不输出组数。
- `07_residual_within_class_dispersion.png`：横轴真实类别，纵轴 mean squared `d_Gamma(r_i,rbar_c)`。越小表示类内 residual 越集中，但不能单独证明分类可泛化。
- `08_crossfit_center_stability.csv`：每类每 fold 一行。记录 4-fold train/1-fold test 数量、fold center 到 full center/identity 的距离、train dispersion、held-out test 到 fold center距离，以及 fold centers 两两距离摘要。无稳定阈值，直接报告原始值。
- `09_crossfit_center_stability.png`：横轴真实类别，纵轴 `d_Gamma(rbar_c^(-f),rbar_c)`，每类 5 个点。越小表示 fold center 对样本组成更稳定；不能自动决定是否启用 residual。
- `10_per_class_classification_comparison.csv`：每类五路 `No Phase / TimeMatch / shared / shared+residual_cv / sample gamma` 的 Recall、mean true-class probability 和 residual 相对三种基线的差值。`shared+residual_cv` 每个样本只能使用未见过该样本 fold 的 residual center。
- `11_per_class_recall_comparison.png`：横轴真实类别，纵轴 Recall，五路并列。重点看 corn 是否从 shared 恢复并超过 No Phase，以及 spring_oat/winter_rye/winter_wheat 和 beneficial controls 的响应。
- `12_per_class_true_probability_comparison.png`：横轴真实类别，纵轴 mean true-class probability，五路并列。用于补充 hard Recall 的连续证据。
- `13_overall_classification_comparison.json`：完整 target-test 的 Accuracy、Macro-F1、macro Recall 和 mean true-class probability。`shared+residual_cv` 只由每个样本自己的 held-out fold center组成，不能用 full-class residual center评价。
- `14_overall_classification_comparison.png`：横轴五种时间处理，纵轴 Accuracy/Macro-F1。只能作整体摘要，不能替代逐类和 cross-fit 稳定性分析。
- `15_transition_vs_shared_phase.csv`：`shared → shared+residual_cv` 的 wrong→correct、correct→wrong、correct→correct、wrong→wrong，含 ALL 和逐类。用于判断 residual 是净修复还是只是交换错误样本。
- `16_transition_vs_no_phase.csv`：`No Phase → shared+residual_cv` 的同类 hard transition。用于判断 residual 是否真正恢复超过原生时间输入。
- `17_confusion_shared_phase.png`：横轴 predicted、纵轴 true class；只使用 leave-one-class-out shared Phase。
- `18_confusion_shared_plus_residual_cv.png`：横轴 predicted、纵轴 true class；每个样本使用 cross-fitted residual center。用于定位类别条件 residual 是否修复或制造特定 confusion。
- `19_confusion_sample_gamma.png`：oracle sample-gamma confusion matrix，作为上限/个体化对照，不用于 residual center 拟合。
- `20_residual_diagnostic_summary.json`：重构误差、full/cross-fit residual 几何、focus/control classes 和整体分类的机器可读摘要。脚本不会按 target-test 结果自动设计 gating/alpha/新聚类；`automatic_route_decision=null`，由实验结果按冻结判据在理论窗口最终二选一。
- `README_中文说明.md`：本说明。

## 实验能支持什么

只有当 residual class center 在 Fisher--Rao 几何下有受控类内 dispersion、5-fold center 对组成稳定，并且 `shared+residual_cv` 在真正 held-out samples 上系统性优于 shared，才支持“共享主效应之外存在可跨样本泛化的类别条件剩余 Phase”。

## 实验不能支持什么

本实验不能直接建立“每类一个正式 Phase”，不能用 target-test 分类结果选择 residual center，不能据结果设置类别 alpha、residual gate 或聚类，也不能证明正式 UDA 中可以无标签恢复 residual。若 residual 不稳定或 cross-fit 分类不泛化，应停止继续拆 Phase，而不是继续增加自由度。
"""
    path.write_text(text, encoding="utf-8")


def run(args) -> dict:
    calibration = torch.load(args.calibration_checkpoint.resolve(), map_location="cpu", weights_only=False)
    runtime = calibration.get("runtime_config")
    if not isinstance(runtime, dict):
        raise ValueError("calibration checkpoint is missing runtime_config")
    classes = [str(v) for v in runtime["classes"]]
    if len(classes) != EXPECTED_CLASSES:
        raise ValueError(f"experiment 11 expects {EXPECTED_CLASSES} classes")
    source = str(runtime["source"]); target = str(runtime["target"]); seed = int(runtime["seed"])
    fold = int(args.fold); data_root = str(args.data_root or runtime["data_root"])
    closed_set = bool(runtime.get("closed_set", True)); combine = bool(runtime.get("combine_spring_and_winter", False))
    time_mode = str(runtime.get("time_coordinate_mode", "canonical_day_of_year"))
    val_ratio = float(runtime.get("val_ratio", 0.1)); test_ratio = float(runtime.get("test_ratio", 0.2))

    device = torch.device(args.device)
    model_checkpoint = torch.load(args.model_checkpoint.resolve(), map_location="cpu", weights_only=False)
    model = phasevis._build_model(runtime, calibration, device, model_checkpoint=model_checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    source_all = phasevis._eligible_parcels(data_root, source, classes, closed_set=closed_set, combine_spring_and_winter=combine, time_coordinate_mode=time_mode)
    target_all = phasevis._eligible_parcels(data_root, target, classes, closed_set=closed_set, combine_spring_and_winter=combine, time_coordinate_mode=time_mode)
    splits = phasevis._reconstruct_fold_splits(source_all, target_all, source=source, target=target, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio, fold=fold)
    target_test_parcels = np.asarray(sorted(splits[target]["test"]), dtype=np.int64)
    target_loader = phasevis._selected_loader(
        data_root, target, classes, target_test_parcels,
        closed_set=closed_set, combine_spring_and_winter=combine, time_coordinate_mode=time_mode,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    records, sample_ids, labels, gammas = _load_gamma_population(args.stage_a_cache.resolve())
    records, sample_ids, labels = _normalize_population_identity(records, sample_ids, labels, target_loader.dataset.get_parcel_indices().tolist())
    if len(sample_ids) != int(args.expected_valid_count):
        raise ValueError(f"experiment 11 requires exactly {args.expected_valid_count} numerically-valid gammas, got {len(sample_ids)}")
    if set(sample_ids) != set(map(int, target_test_parcels.tolist())):
        raise ValueError("06 gamma population does not match reconstructed target-test")
    if sorted(np.unique(labels).tolist()) != list(range(len(classes))):
        raise ValueError("experiment 11 requires all runtime classes")

    _load_experiment10_manifest(args.experiment10_manifest.resolve())
    shared_centers = _load_shared_centers(args.experiment10_phase_npz.resolve(), classes, gammas.shape[1])
    rows07 = _load_07_rows(args.audit07_sample_csv.resolve(), sample_ids)
    for sid, label in zip(sample_ids, labels.tolist()):
        if int(float(rows07[int(sid)]["true_class"])) != int(label):
            raise ValueError("07 true class does not match gamma population")

    print(
        "RESIDUAL_PHASE_11_START|"
        f"source={source}|target={target}|n={len(sample_ids)}|classes={len(classes)}|"
        "registration_calls=0|training_updates=false|clustering=false|crossfit=5fold",
        flush=True,
    )

    # HARD GATE: derive and reconstruct residuals before any structural/classifier interpretation.
    reconstruction = build_residual_population(
        gammas, labels, shared_centers,
        reconstruction_tolerance=float(args.reconstruction_tolerance),
    )
    residuals = reconstruction.residuals
    print(
        "RESIDUAL_PHASE_11_RECONSTRUCTION_READY|"
        f"max_dGamma={reconstruction.max_error:.3e}|mean_dGamma={reconstruction.mean_error:.3e}|"
        f"fail_count={reconstruction.fail_count}|tolerance={reconstruction.tolerance:.3e}",
        flush=True,
    )

    full_centers, per_class_geometry, center_pairwise = _full_centers_and_geometry(residuals, labels, classes)
    folds, fold_centers, crossfit_rows, _ = _crossfit_centers(
        residuals, sample_ids, labels, classes,
        n_folds=int(args.cv_folds), seed=int(args.cv_seed),
    )
    if int(args.cv_folds) != DEFAULT_CV_FOLDS:
        raise ValueError("experiment 11 frozen protocol requires 5 folds")

    composite_centers = {
        (class_id, fold_id): compose_phase(shared_centers[class_id], fold_centers[(class_id, fold_id)])
        for class_id in range(len(classes)) for fold_id in range(int(args.cv_folds))
    }
    fold_by_parcel = {int(sid): int(fold_value) for sid, fold_value in zip(sample_ids, folds.tolist())}

    eval_rows = _evaluate_five_way(
        model=model, loader=target_loader,
        shared_centers=shared_centers,
        composite_centers=composite_centers,
        fold_by_parcel=fold_by_parcel,
        rows07=rows07, device=device,
    )
    if len(eval_rows) != len(sample_ids):
        raise RuntimeError("cross-fit classifier evaluation did not cover complete target-test")
    per_class_classification, overall = _classification_tables(eval_rows, classes)
    _validate_experiment10_shared_replay(args.experiment10_per_class_csv.resolve(), per_class_classification)

    residual_index = {int(sid): i for i, sid in enumerate(sample_ids)}
    full_center_by_class = {c: full_centers[c].center for c in range(len(classes))}
    sample_rows: list[dict] = []
    for row in eval_rows:
        sid = int(row["sample_id"]); class_id = int(row["true_class"]); index = residual_index[sid]
        residual = residuals[index]
        fold_id = int(row["fold"])
        item = dict(row)
        item.update({
            "residual_distance_to_identity": phase_distance_to_identity_for_gamma(residual),
            "residual_distance_to_full_class_center": float(phase_distance(residual, full_center_by_class[class_id]).item()),
            "residual_distance_to_fold_center": float(phase_distance(residual, fold_centers[(class_id, fold_id)]).item()),
            "residual_reconstruction_phase_error": float(reconstruction.phase_errors[index].item()),
        })
        sample_rows.append(item)

    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, gammas.shape[1])
    shared_per_sample = torch.stack([shared_centers[int(c)] for c in labels.tolist()])
    full_center_stack = torch.stack([full_centers[c].center for c in range(len(classes))])
    fold_center_stack = torch.stack([fold_centers[(c, f)] for c in range(len(classes)) for f in range(int(args.cv_folds))]).reshape(len(classes), int(args.cv_folds), -1)
    composite_stack = torch.stack([composite_centers[(c, f)] for c in range(len(classes)) for f in range(int(args.cv_folds))]).reshape(len(classes), int(args.cv_folds), -1)
    np.savez_compressed(
        output / "01_residual_phase_curves.npz",
        grid=grid,
        sample_id=np.asarray(sample_ids, dtype=np.int64),
        true_class=labels,
        fold=folds,
        gamma_true_class=gammas.numpy(),
        shared_phase_minus_class=shared_per_sample.numpy(),
        residual_phase=residuals.numpy(),
        residual_full_class_centers=full_center_stack.numpy(),
        residual_crossfit_centers=fold_center_stack.numpy(),
        shared_plus_residual_crossfit_centers=composite_stack.numpy(),
        class_names=np.asarray(classes),
    )
    _write_csv(output / "02_residual_sample_summary.csv", sample_rows)
    _write_csv(output / "03_residual_per_class_summary.csv", per_class_geometry)
    _plot_distance_to_identity(output / "04_residual_distance_to_identity_by_class.png", sample_rows, classes)
    _plot_center_displacement(output / "05_residual_class_center_displacement.png", full_centers, classes)
    _plot_matrix(output / "06_residual_class_center_pairwise_distance.png", center_pairwise.numpy(), classes)
    _plot_dispersion(output / "07_residual_within_class_dispersion.png", per_class_geometry)
    _write_csv(output / "08_crossfit_center_stability.csv", crossfit_rows)
    _plot_crossfit_stability(output / "09_crossfit_center_stability.png", crossfit_rows, classes)
    _write_csv(output / "10_per_class_classification_comparison.csv", per_class_classification)
    _plot_per_class(output / "11_per_class_recall_comparison.png", per_class_classification, "recall", "Recall")
    _plot_per_class(output / "12_per_class_true_probability_comparison.png", per_class_classification, "mean_true_prob", "mean true-class probability")
    _json_dump(output / "13_overall_classification_comparison.json", overall)
    _plot_overall(output / "14_overall_classification_comparison.png", overall)
    _write_csv(output / "15_transition_vs_shared_phase.csv", _transition_rows(sample_rows, classes, "transition_residual_vs_shared"))
    _write_csv(output / "16_transition_vs_no_phase.csv", _transition_rows(sample_rows, classes, "transition_residual_vs_no"))
    _plot_confusion(output / "17_confusion_shared_phase.png", _confusion(sample_rows, "pred_shared_phase", len(classes)), classes, "Leave-one-class shared Phase")
    _plot_confusion(output / "18_confusion_shared_plus_residual_cv.png", _confusion(sample_rows, "pred_shared_plus_residual_cv", len(classes)), classes, "Shared + cross-fitted residual Phase")
    _plot_confusion(output / "19_confusion_sample_gamma.png", _confusion(sample_rows, "pred_sample_gamma", len(classes)), classes, "Oracle sample gamma")

    perclass_map = {str(row["class_name"]): row for row in per_class_classification}
    geometry_map = {str(row["class_name"]): row for row in per_class_geometry}
    summary = {
        "protocol": "11_shared_phase_residual_class_structure_and_generalization_diagnostic",
        "source": source, "target": target, "seed": seed, "fold": fold,
        "numerically_valid_gamma": len(sample_ids),
        "residual_factorization": "gamma_sample = delta_minus_class o residual; residual = inverse(delta_minus_class) o gamma_sample",
        "residual_reconstruction_max_error": reconstruction.max_error,
        "residual_reconstruction_mean_error": reconstruction.mean_error,
        "residual_reconstruction_fail_count": reconstruction.fail_count,
        "residual_reconstruction_tolerance": reconstruction.tolerance,
        "cv": {"folds": int(args.cv_folds), "seed": int(args.cv_seed), "split_scope": "independent within each true class"},
        "focus_classes": {
            name: {"geometry": geometry_map.get(name), "classification": perclass_map.get(name)}
            for name in ("corn", "spring_oat", "winter_rye", "winter_wheat") if name in perclass_map
        },
        "beneficial_controls": {
            name: {"geometry": geometry_map.get(name), "classification": perclass_map.get(name)}
            for name in ("horsebeans", "spring_barley", "winter_barley", "winter_triticale") if name in perclass_map
        },
        "overall": overall,
        "automatic_route_decision": None,
        "final_allowed_routes": {
            "route_1_continue_phase_decomposition": "only if cross-fitted residual centers are stable and shared+residual_cv has systematic held-out classification value",
            "route_2_stop_phase_splitting": "if residual is unstable or cross-fitted classification does not generalize; proceed to unlabeled shared-Phase recovery and stable Stage-2 design",
        },
        "no_additional_mechanism_allowed_from_script": True,
    }
    _json_dump(output / "20_residual_diagnostic_summary.json", summary)
    manifest = {
        "protocol": summary["protocol"],
        "stage_a_cache": str(args.stage_a_cache.resolve()),
        "audit07_sample_csv": str(args.audit07_sample_csv.resolve()),
        "experiment10_manifest": str(args.experiment10_manifest.resolve()),
        "experiment10_phase_npz": str(args.experiment10_phase_npz.resolve()),
        "experiment10_per_class_csv": str(args.experiment10_per_class_csv.resolve()),
        "calibration_checkpoint": str(args.calibration_checkpoint.resolve()),
        "model_checkpoint": str(args.model_checkpoint.resolve()),
        "gamma_direction": "gamma(u_source)=u_target; target classifier positions use gamma^{-1}",
        "derived_residual_formula": "residual = inverse(delta_minus_class) o gamma_sample",
        "reconstruction_formula": "gamma_reconstructed = delta_minus_class o residual",
        "reconstruction_metric": "Fisher-Rao d_Gamma",
        "reconstruction_tolerance": float(args.reconstruction_tolerance),
        "cv_folds": int(args.cv_folds),
        "cv_seed": int(args.cv_seed),
        "crossfit_leakage_rule": "held-out fold samples have zero contribution to their residual center",
        "target_true_label_use": "oracle-only: choose delta_-class, form residual class population/folds, and evaluate",
        "registration_calls": 0,
        "production_legality_filter": False,
        "beneficial_harmful_filter": False,
        "clustering": False,
        "M1_M2": False,
        "class_specific_alpha": False,
        "residual_gating": False,
        "teacher_student": False,
        "stable_label": False,
        "training_updates": False,
        "domain_shape_transport": False,
        "classification_outcomes_used_to_fit_residual_center": False,
        "automatic_route_decision": None,
    }
    _json_dump(output / "00_manifest.json", manifest)
    _write_readme(output / "README_中文说明.md")
    print(
        f"RESIDUAL_PHASE_11_DONE|output={output}|n={len(sample_ids)}|"
        "registration_calls=0|training_updates=false|clustering=false",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-cache", type=Path, required=True)
    parser.add_argument("--audit07-sample-csv", type=Path, required=True)
    parser.add_argument("--experiment10-manifest", type=Path, required=True)
    parser.add_argument("--experiment10-phase-npz", type=Path, required=True)
    parser.add_argument("--experiment10-per-class-csv", type=Path, required=True)
    parser.add_argument("--calibration-checkpoint", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-valid-count", type=int, default=EXPECTED_SAMPLES)
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--cv-seed", type=int, default=DEFAULT_CV_SEED)
    parser.add_argument("--reconstruction-tolerance", type=float, default=DEFAULT_RECONSTRUCTION_TOLERANCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
