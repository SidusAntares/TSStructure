#!/usr/bin/env python3
"""10: Class-balanced shared Domain Phase leave-one-class-out diagnostic.

The experiment consumes the complete numerically-valid oracle true-class gamma
population already computed by experiment 06. It estimates one class-balanced
Fisher--Rao Frechet center, ten leave-one-class-out centers, and one diagnostic
sample-equal center. No registration, clustering, training, legality filter,
Phase grouping, class-specific strength tuning, Stable Label, Teacher/Student,
or Domain Shape transport is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
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

import diagnose_sample_level_phase_validity as sample06
import visualize_stage2_phase_alignment as phasevis
from methods.structure_da.confirmed_phase_view import align_target_positions_to_source
from methods.structure_da.shared_domain_phase_diagnostic import (
    WeightedFrechetMeanResult,
    center_phase_distance_matrix,
    class_balanced_weights,
    phase_distance_to_identity_for_gamma,
    sample_equal_weights,
    weighted_frechet_mean_gamma,
    weighted_objective,
)
from methods.structure_da.sample_phase_diagnostic import TOnlyPhaseRegistration


EXPECTED_CLASSES = 10
EXPECTED_SAMPLES = 10634


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


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
    for class_id in class_ids:
        mask = labels == int(class_id)
        values.append(float(np.mean(predictions[mask] == int(class_id))) if np.any(mask) else float("nan"))
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


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
            f"10 requires experiment-06 Stage-A cache and never recomputes registration: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = [_registration_from_payload(row) for row in payload.get("records", ())]
    sample_ids = [int(v) for v in payload.get("sample_ids", ())]
    labels = np.asarray(payload.get("true_classes", ()), dtype=np.int64)
    if len(records) != len(sample_ids) or len(records) != labels.size:
        raise ValueError("06 Stage-A cache arrays have inconsistent lengths")
    valid_records: list[TOnlyPhaseRegistration] = []
    valid_ids: list[int] = []
    valid_labels: list[int] = []
    gammas: list[Tensor] = []
    for record, sid, label in zip(records, sample_ids, labels.tolist()):
        if not bool(record.numerically_valid) or not isinstance(record.gamma, Tensor):
            continue
        gamma = record.gamma.detach().cpu().double().flatten()
        if not torch.isfinite(gamma).all().item() or not torch.all(gamma[1:] > gamma[:-1]).item():
            continue
        valid_records.append(record); valid_ids.append(int(sid)); valid_labels.append(int(label)); gammas.append(gamma)
    if not gammas:
        raise RuntimeError("06 Stage-A cache contains no numerically-valid oracle true-class gamma")
    lengths = {int(g.numel()) for g in gammas}
    if len(lengths) != 1:
        raise ValueError("cached gamma values do not share one registration grid")
    return valid_records, valid_ids, np.asarray(valid_labels, dtype=np.int64), torch.stack(gammas)


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
        mapped_records = [replace(record, sample_id=mapped[i]) for i, record in enumerate(records)]
        return mapped_records, mapped, labels.copy()
    raise ValueError("06 sample IDs are neither stable parcel IDs nor valid target-test local indices")


def _load_07_rows(path: Path, required_ids: Sequence[int]) -> dict[int, dict[str, str]]:
    rows = {int(float(row["sample_id"])): row for row in _read_csv(path)}
    missing = sorted(set(map(int, required_ids)).difference(rows))[:5]
    if missing:
        raise ValueError(f"07 sample CSV is missing experiment-10 parcels: {missing}")
    return rows


def _estimate_centers(gammas: Tensor, labels: np.ndarray, classes: Sequence[str]):
    if len(classes) != EXPECTED_CLASSES:
        raise ValueError(f"experiment 10 expects {EXPECTED_CLASSES} classes")
    results: dict[str, WeightedFrechetMeanResult] = {}
    weights_by_name: dict[str, Tensor] = {}

    all_weights = class_balanced_weights(labels)
    print("SHARED_PHASE_10_CENTER|name=delta_all|status=start|weighting=class_balanced", flush=True)
    results["delta_all"] = weighted_frechet_mean_gamma(gammas, all_weights)
    weights_by_name["delta_all"] = all_weights
    print("SHARED_PHASE_10_CENTER|name=delta_all|status=ready", flush=True)

    for class_id, class_name in enumerate(classes):
        name = f"delta_minus_class_{class_id}"
        weights = class_balanced_weights(labels, held_out_class=class_id)
        print(f"SHARED_PHASE_10_CENTER|name={name}|held_out={class_name}|status=start", flush=True)
        results[name] = weighted_frechet_mean_gamma(gammas, weights)
        weights_by_name[name] = weights
        print(f"SHARED_PHASE_10_CENTER|name={name}|held_out={class_name}|status=ready", flush=True)

    sample_weights = sample_equal_weights(gammas.shape[0])
    print("SHARED_PHASE_10_CENTER|name=delta_sample_equal|status=start|weighting=sample_equal", flush=True)
    results["delta_sample_equal"] = weighted_frechet_mean_gamma(gammas, sample_weights)
    weights_by_name["delta_sample_equal"] = sample_weights
    print("SHARED_PHASE_10_CENTER|name=delta_sample_equal|status=ready", flush=True)
    return results, weights_by_name


def _center_summary(
    gammas: Tensor,
    labels: np.ndarray,
    classes: Sequence[str],
    results: dict[str, WeightedFrechetMeanResult],
    weights: dict[str, Tensor],
) -> tuple[list[dict], Tensor, list[str]]:
    required_names = ["delta_all", *[f"delta_minus_class_{c}" for c in range(len(classes))]]
    required_gammas = torch.stack([results[name].gamma for name in required_names])
    pairwise = center_phase_distance_matrix(required_gammas)
    all_gamma = results["delta_all"].gamma
    rows: list[dict] = []
    for index, name in enumerate(required_names):
        held_out = None if name == "delta_all" else int(name.rsplit("_", 1)[-1])
        result = results[name]
        rows.append({
            "center_name": name,
            "center_role": "class_balanced_all" if held_out is None else "class_balanced_leave_one_class_out",
            "held_out_class_id": "" if held_out is None else held_out,
            "held_out_class_name": "" if held_out is None else classes[held_out],
            "included_class_count": len(classes) if held_out is None else len(classes) - 1,
            "d_to_delta_all": 0.0 if held_out is None else float(pairwise[0, index].item()),
            "d_to_identity": phase_distance_to_identity_for_gamma(result.gamma),
            "weighted_frechet_objective": result.objective,
            "weighted_objective_recomputed": weighted_objective(gammas, result.gamma, weights[name]),
            "iterations": result.iterations,
            "converged": result.converged,
            "final_tangent_norm": result.tangent_norm,
        })
    sample = results["delta_sample_equal"]
    sample_pair = center_phase_distance_matrix(torch.stack([all_gamma, sample.gamma]))
    rows.append({
        "center_name": "delta_sample_equal",
        "center_role": "diagnostic_sample_equal_control",
        "held_out_class_id": "",
        "held_out_class_name": "",
        "included_class_count": len(classes),
        "d_to_delta_all": float(sample_pair[0, 1].item()),
        "d_to_identity": phase_distance_to_identity_for_gamma(sample.gamma),
        "weighted_frechet_objective": sample.objective,
        "weighted_objective_recomputed": weighted_objective(gammas, sample.gamma, weights["delta_sample_equal"]),
        "iterations": sample.iterations,
        "converged": sample.converged,
        "final_tangent_norm": sample.tangent_norm,
    })
    return rows, pairwise, required_names


def _evaluate_shared_leave_one_class(
    *,
    model,
    loader,
    centers: dict[int, Tensor],
    rows07: dict[int, dict[str, str]],
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
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
            for class_id in sorted(centers):
                selected = labels == int(class_id)
                if not torch.any(selected):
                    continue
                shared_positions[selected] = align_target_positions_to_source(
                    native[selected], mask[selected], centers[int(class_id)]
                )
            shared_out = model.forward_from_backbone(
                backbone, batch["positions"], batch.get("extra"),
                temporal_positions_override=shared_positions, return_geometry=False,
            )
            no_out = model.forward_from_backbone(
                backbone, batch["positions"], batch.get("extra"), return_geometry=False,
            )
            shared_probs = torch.softmax(shared_out.logits.float(), dim=-1)
            shared_pred = shared_out.logits.argmax(dim=-1)
            no_probs = torch.softmax(no_out.logits.float(), dim=-1)
            no_pred = no_out.logits.argmax(dim=-1)

            for row_index, parcel in enumerate(parcels.tolist()):
                parcel = int(parcel); true_class = int(labels[row_index].item())
                base = rows07.get(parcel)
                if base is None:
                    raise KeyError(f"target parcel {parcel} is missing from 07 sample CSV")
                if int(float(base["true_class"])) != true_class:
                    raise ValueError("07 true class differs from reconstructed target-test label")
                # Replaying native predictions here is a protocol consistency check.
                p_no = float(no_probs[row_index, true_class].item())
                p_no_07 = float(base["true_prob_no"])
                if abs(p_no - p_no_07) > 5e-5 or int(no_pred[row_index].item()) != int(float(base["pred_no"])):
                    raise RuntimeError("frozen No-Phase replay does not match experiment 07")
                pred_shared = int(shared_pred[row_index].item())
                p_shared = float(shared_probs[row_index, true_class].item())
                pred_no = int(float(base["pred_no"])); pred_tm = int(float(base["pred_timematch"])); pred_sample = int(float(base["pred_oracle_gamma"]))
                p_tm = float(base["true_prob_timematch"]); p_sample = float(base["true_prob_oracle_gamma"])
                rows.append({
                    "sample_id": parcel,
                    "true_class": true_class,
                    "pred_no_phase": pred_no,
                    "pred_timematch": pred_tm,
                    "pred_sample_gamma": pred_sample,
                    "pred_shared_phase_minus_class": pred_shared,
                    "true_prob_no_phase": p_no_07,
                    "true_prob_timematch": p_tm,
                    "true_prob_sample_gamma": p_sample,
                    "true_prob_shared_phase_minus_class": p_shared,
                    "transition_shared_vs_no": _transition(pred_no == true_class, pred_shared == true_class),
                    "transition_shared_vs_sample_gamma": _transition(pred_sample == true_class, pred_shared == true_class),
                })
    return rows


def _classification_tables(sample_rows: Sequence[dict], classes: Sequence[str]) -> tuple[list[dict], dict]:
    rows = list(sample_rows)
    labels = np.asarray([int(r["true_class"]) for r in rows], dtype=np.int64)
    keys = {
        "no_phase": "pred_no_phase",
        "timematch": "pred_timematch",
        "sample_gamma": "pred_sample_gamma",
        "shared_phase_minus_class": "pred_shared_phase_minus_class",
    }
    prob_keys = {
        "no_phase": "true_prob_no_phase",
        "timematch": "true_prob_timematch",
        "sample_gamma": "true_prob_sample_gamma",
        "shared_phase_minus_class": "true_prob_shared_phase_minus_class",
    }
    predictions = {name: np.asarray([int(r[key]) for r in rows], dtype=np.int64) for name, key in keys.items()}
    class_ids = list(range(len(classes)))
    per_class: list[dict] = []
    for class_id, class_name in enumerate(classes):
        mask = labels == class_id
        item = {"class_id": class_id, "class_name": class_name, "n_samples": int(mask.sum())}
        for name in keys:
            item[f"recall_{name}"] = float(np.mean(predictions[name][mask] == class_id))
            item[f"mean_true_prob_{name}"] = float(np.mean([float(rows[i][prob_keys[name]]) for i in np.flatnonzero(mask)]))
        item["delta_recall_shared_vs_no"] = item["recall_shared_phase_minus_class"] - item["recall_no_phase"]
        item["delta_recall_shared_vs_timematch"] = item["recall_shared_phase_minus_class"] - item["recall_timematch"]
        item["delta_recall_shared_vs_sample_gamma"] = item["recall_shared_phase_minus_class"] - item["recall_sample_gamma"]
        item["delta_true_prob_shared_vs_no"] = item["mean_true_prob_shared_phase_minus_class"] - item["mean_true_prob_no_phase"]
        item["delta_true_prob_shared_vs_timematch"] = item["mean_true_prob_shared_phase_minus_class"] - item["mean_true_prob_timematch"]
        item["delta_true_prob_shared_vs_sample_gamma"] = item["mean_true_prob_shared_phase_minus_class"] - item["mean_true_prob_sample_gamma"]
        per_class.append(item)

    overall = {"n_samples": len(rows), "methods": {}}
    for name in keys:
        pred = predictions[name]
        overall["methods"][name] = {
            "accuracy": float(np.mean(pred == labels)),
            "macro_f1": _macro_f1(labels, pred, class_ids),
            "macro_recall": _macro_recall(labels, pred, class_ids),
            "mean_true_class_probability": float(np.mean([float(r[prob_keys[name]]) for r in rows])),
        }
    return per_class, overall


def _transition_rows(sample_rows: Sequence[dict], classes: Sequence[str], key: str) -> list[dict]:
    order = ("wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong")
    result: list[dict] = []
    scopes = [("ALL", None), *[(classes[c], c) for c in range(len(classes))]]
    for name, class_id in scopes:
        rows = [r for r in sample_rows if class_id is None or int(r["true_class"]) == class_id]
        counts = {value: sum(str(r[key]) == value for r in rows) for value in order}
        result.append({"scope": name, "class_id": "" if class_id is None else class_id, "n_samples": len(rows), **counts})
    return result


def _confusion(sample_rows: Sequence[dict], pred_key: str, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for row in sample_rows:
        matrix[int(row["true_class"]), int(row[pred_key])] += 1
    return matrix


def _plot_drift(path: Path, center_rows: Sequence[dict], classes: Sequence[str]) -> None:
    loo = [row for row in center_rows if row["center_role"] == "class_balanced_leave_one_class_out"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([row["held_out_class_name"] for row in loo], [float(row["d_to_delta_all"]) for row in loo])
    ax.set_ylabel("d_Gamma(delta_-c, delta_all)"); ax.set_xlabel("held-out class")
    ax.set_title("Leave-one-class shared-Phase drift (no threshold)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_phase_overlay(path: Path, results: dict[str, WeightedFrechetMeanResult], classes: Sequence[str]) -> None:
    gamma_all = results["delta_all"].gamma.numpy(); k = gamma_all.size
    x = np.linspace(0.0, 365.0, k); identity = np.linspace(0.0, 1.0, k)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, (gamma_all - identity) * 365.0, linewidth=2.5, label="delta_all")
    for class_id, class_name in enumerate(classes):
        gamma = results[f"delta_minus_class_{class_id}"].gamma.numpy()
        ax.plot(x, (gamma - identity) * 365.0, linewidth=1.0, alpha=0.75, label=f"minus {class_name}")
    sample = results["delta_sample_equal"].gamma.numpy()
    ax.plot(x, (sample - identity) * 365.0, linewidth=1.8, linestyle="--", label="sample-equal control")
    ax.axhline(0.0, linewidth=1.0, linestyle=":")
    ax.set_xlabel("canonical day"); ax.set_ylabel("gamma(t)-t [days]")
    ax.set_title("Class-balanced shared Phase: all and leave-one-class-out candidates")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_center_distance(path: Path, matrix: Tensor, classes: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix.numpy(), aspect="equal", interpolation="nearest")
    labels = ["all", *[f"-{name}" for name in classes]]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title("11x11 Fisher-Rao distance among class-balanced shared Phase candidates")
    fig.colorbar(image, ax=ax, label="d_Gamma")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_per_class_metric(path: Path, per_class: Sequence[dict], metric_prefix: str, ylabel: str) -> None:
    methods = ("no_phase", "timematch", "sample_gamma", "shared_phase_minus_class")
    x = np.arange(len(per_class)); width = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    for offset, method in enumerate(methods):
        values = [float(row[f"{metric_prefix}_{method}"]) for row in per_class]
        ax.bar(x + (offset - 1.5) * width, values, width=width, label=method)
    ax.set_xticks(x, [str(r["class_name"]) for r in per_class], rotation=35, ha="right")
    ax.set_ylabel(ylabel); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_overall(path: Path, overall: dict) -> None:
    methods = list(overall["methods"])
    accuracy = [overall["methods"][m]["accuracy"] for m in methods]
    f1 = [overall["methods"][m]["macro_f1"] for m in methods]
    x = np.arange(len(methods)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, accuracy, width=width, label="Accuracy")
    ax.bar(x + width/2, f1, width=width, label="Macro-F1")
    ax.set_xticks(x, methods, rotation=20, ha="right"); ax.set_ylim(0.0, 1.0); ax.legend()
    ax.set_title("Frozen classifier comparison: LOO shared Phase is oracle-only")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_confusion(path: Path, matrix: np.ndarray, classes: Sequence[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, interpolation="nearest", aspect="equal")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class"); ax.set_title(title)
    fig.colorbar(image, ax=ax, label="count")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _write_readme(path: Path) -> None:
    text = """# 实验 10 中文说明

## 实验唯一问题

完全不使用真实类别 `c` 的任何样本，由其他 9 类的数值有效 oracle true-class gamma 估计出的类别平衡共享 Phase `delta_-c`，是否仍能稳定地帮助 held-out 类别 `c`？

所有共享中心都是 Fisher--Rao Phase 几何上的加权 Fréchet/Karcher mean，不是逐时间点 gamma 普通平均。类别平衡中心让每个真实类别的总权重相同；`delta_sample_equal` 仅作为“大类别是否拉动中心”的诊断对照。

本实验是 oracle-only 离线诊断。真实 target label 只用于构造类别平衡权重、定义 leave-one-class-out、以及事后分类评价。每个类别 `c` 的 target 样本只使用从未见过类别 `c` 的 `delta_-c`。模型参数、PSE latent `H_i` 均不改变，只通过 `gamma^{-1}` 改变送入冻结 LTAE 的 target 时间位置。

## 文件说明

- `00_manifest.json`：协议、输入、禁止机制、Fréchet estimator 语义。无坐标轴。应检查 `registration_calls=0`、`clustering=false`、`training_updates=false`。只能证明运行协议符合实验 10，不能证明 Domain Phase 有效。
- `01_shared_phase_curves.npz`：`delta_all`、10 个 `delta_-c`、以及 sample-equal 诊断中心的完整 gamma 数组和 canonical grid。用于精确复算 Phase 几何；不能从数组本身判断分类价值。
- `02_shared_phase_summary.csv`：每个共享中心的 role、held-out class、`d_to_delta_all`、`d_to_identity`、Fréchet objective、迭代收敛信息。用于检查统计稳定性和 estimator 数值行为；不设置 drift threshold。
- `03_leave_one_class_phase_drift.png`：横轴为被删除类别，纵轴为 `d_Gamma(delta_-c,delta_all)`。看单个类别是否强烈拉动共享中心。原始漂移只作诊断，不自动判“稳定/不稳定”。
- `04_shared_phase_displacement_overlay.png`：横轴 canonical day，纵轴 `gamma(t)-t`（days）。叠加 `delta_all`、10 个 LOO 中心，并以虚线显示 sample-equal control。看共享时间变化轮廓是否对类别组成稳定；不能据此聚类。
- `05_shared_phase_pairwise_distance.png`：横纵轴都是 `delta_all + 10 个 delta_-c`，像素值是正式 Fisher--Rao `d_Gamma`。看 11 个候选是否处在同一稳定 Phase 区域；不输出自动模式数。
- `06_per_class_classification_comparison.csv`：每类 `No Phase / TimeMatch / sample gamma / delta_-c` 的 Recall、mean true-class probability 和差值。`delta_-c` 完全没有用该类样本估计，是 held-out-class 泛化的主表。
- `07_per_class_recall_comparison.png`：横轴真实类别，纵轴 Recall，四种时间处理并列。重点看 spring_oat、winter_rye、winter_wheat 是否缓解 sample-gamma 负迁移，同时看 horsebeans、spring_barley、winter_barley、winter_triticale 是否保留收益。
- `08_per_class_true_probability_comparison.png`：横轴真实类别，纵轴 mean true-class probability。用于观察 hard Recall 之外的连续语义变化。
- `09_overall_classification_comparison.json`：将每类样本分别使用对应 `delta_-c` 后组合成完整 target-test 的 Accuracy、Macro-F1、macro Recall、mean true-class probability，并同时给出另外三种基线。这个 overall 仍是 oracle-only，因为真实类别决定使用哪个 LOO center。
- `10_overall_classification_comparison.png`：横轴为四种处理，纵轴为 Accuracy/Macro-F1。只能作为整体摘要，不能替代逐类 held-out 泛化分析。
- `11_transition_vs_no_phase.csv`：`delta_-c` 相对 No Phase 的 `wrong_to_correct / correct_to_wrong / correct_to_correct / wrong_to_wrong`，含 ALL 和逐类统计。重点看共享 Phase 是否产生净修复。
- `12_transition_vs_sample_gamma.csv`：`delta_-c` 相对 oracle sample gamma 的 hard transition。重点看是否减少实验 08 暴露的 sample-level `correct_to_wrong`，同时保留 `wrong_to_correct`。
- `13_confusion_no_phase.png`：No Phase confusion matrix；横轴 predicted、纵轴 true class。
- `14_confusion_sample_gamma.png`：oracle true-class sample gamma confusion matrix；用于复现实验 07/08 的 sample-level 时间校正行为。
- `15_confusion_leave_one_class_shared_phase.png`：LOO shared Phase confusion matrix；重点检查 winter_wheat→winter_triticale、winter_rye→winter_triticale/winter_barley 等是否减少。
- `16_shared_phase_diagnostic_summary.json`：共享中心漂移、class-balanced vs sample-equal 距离、held-out-class gain 数量、重点/控制类结果的机器可读摘要。脚本不会自动选择 A/B/C/D 理论结论。
- `README_中文说明.md`：本说明。

## 能证明什么

如果多个 `delta_-c` 对 `delta_all` 的原始 Phase 漂移普遍较小，且多数 held-out class 在完全未参与自身 Phase 估计时仍相对 No Phase 获益，同时受损类改善、beneficial control 不被明显破坏，则强支持“跨类别统计中存在可泛化的共享 Domain Phase 主效应”。

## 不能证明什么

本实验不能证明这个 Fréchet center 是最终 Domain Phase estimator；不能证明正式 UDA 中可无标签恢复它；不能据结果自动引入 Phase grouping、类别特异 Phase、不同 alpha、Teacher/Student、Stable Label 或 Domain Shape。若统计稳定但分类响应差异大，应先把“Domain Phase 是否存在”和“怎样应用到 classifier”分开解释。
"""
    path.write_text(text, encoding="utf-8")


def run(args) -> dict:
    calibration = torch.load(args.calibration_checkpoint.resolve(), map_location="cpu", weights_only=False)
    runtime = calibration.get("runtime_config")
    if not isinstance(runtime, dict):
        raise ValueError("calibration checkpoint is missing runtime_config")
    classes = [str(v) for v in runtime["classes"]]
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
        raise ValueError(f"experiment 10 requires exactly {args.expected_valid_count} numerically-valid gamma values, got {len(sample_ids)}")
    if set(sample_ids) != set(map(int, target_test_parcels.tolist())):
        raise ValueError("06 numerically-valid gamma population does not match reconstructed target-test")
    if sorted(np.unique(labels).tolist()) != list(range(len(classes))):
        raise ValueError("experiment 10 requires all runtime classes in the gamma population")

    rows07 = _load_07_rows(args.audit07_sample_csv.resolve(), sample_ids)
    for sid, label in zip(sample_ids, labels.tolist()):
        if int(float(rows07[int(sid)]["true_class"])) != int(label):
            raise ValueError("07 true class does not match 06 gamma population")

    print(
        "SHARED_PHASE_10_START|"
        f"source={source}|target={target}|n={len(sample_ids)}|classes={len(classes)}|"
        "registration_calls=0|clustering=false|training_updates=false|weighting=class_balanced",
        flush=True,
    )
    center_results, center_weights = _estimate_centers(gammas, labels, classes)
    center_rows, center_matrix, center_names = _center_summary(gammas, labels, classes, center_results, center_weights)

    loo_centers = {class_id: center_results[f"delta_minus_class_{class_id}"].gamma for class_id in range(len(classes))}
    sample_rows = _evaluate_shared_leave_one_class(
        model=model, loader=target_loader, centers=loo_centers, rows07=rows07, device=device,
    )
    if len(sample_rows) != len(sample_ids):
        raise RuntimeError("shared-Phase evaluation did not cover complete held-out target-test")
    per_class, overall = _classification_tables(sample_rows, classes)
    transitions_no = _transition_rows(sample_rows, classes, "transition_shared_vs_no")
    transitions_sample = _transition_rows(sample_rows, classes, "transition_shared_vs_sample_gamma")

    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, gammas.shape[1])
    np.savez_compressed(
        output / "01_shared_phase_curves.npz",
        grid=grid,
        delta_all=center_results["delta_all"].gamma.numpy(),
        **{f"delta_minus_class_{c}": center_results[f"delta_minus_class_{c}"].gamma.numpy() for c in range(len(classes))},
        delta_sample_equal=center_results["delta_sample_equal"].gamma.numpy(),
        class_names=np.asarray(classes),
    )
    _write_csv(output / "02_shared_phase_summary.csv", center_rows)
    _plot_drift(output / "03_leave_one_class_phase_drift.png", center_rows, classes)
    _plot_phase_overlay(output / "04_shared_phase_displacement_overlay.png", center_results, classes)
    _plot_center_distance(output / "05_shared_phase_pairwise_distance.png", center_matrix, classes)
    _write_csv(output / "06_per_class_classification_comparison.csv", per_class)
    _plot_per_class_metric(output / "07_per_class_recall_comparison.png", per_class, "recall", "Recall")
    _plot_per_class_metric(output / "08_per_class_true_probability_comparison.png", per_class, "mean_true_prob", "mean true-class probability")
    _json_dump(output / "09_overall_classification_comparison.json", overall)
    _plot_overall(output / "10_overall_classification_comparison.png", overall)
    _write_csv(output / "11_transition_vs_no_phase.csv", transitions_no)
    _write_csv(output / "12_transition_vs_sample_gamma.csv", transitions_sample)
    _plot_confusion(output / "13_confusion_no_phase.png", _confusion(sample_rows, "pred_no_phase", len(classes)), classes, "No Phase")
    _plot_confusion(output / "14_confusion_sample_gamma.png", _confusion(sample_rows, "pred_sample_gamma", len(classes)), classes, "Oracle sample gamma")
    _plot_confusion(output / "15_confusion_leave_one_class_shared_phase.png", _confusion(sample_rows, "pred_shared_phase_minus_class", len(classes)), classes, "Leave-one-class-out shared Phase")

    drift = [float(row["d_to_delta_all"]) for row in center_rows if row["center_role"] == "class_balanced_leave_one_class_out"]
    per_class_by_name = {row["class_name"]: row for row in per_class}
    summary = {
        "protocol": "10_class_balanced_shared_domain_phase_leave_one_class_out",
        "source": source, "target": target, "seed": seed, "fold": fold,
        "numerically_valid_gamma": len(sample_ids),
        "registration_calls": 0,
        "training_updates": False,
        "class_balanced_center": {
            "d_to_identity": phase_distance_to_identity_for_gamma(center_results["delta_all"].gamma),
            "objective": center_results["delta_all"].objective,
            "converged": center_results["delta_all"].converged,
        },
        "leave_one_class_drift_raw": {
            "values": {row["held_out_class_name"]: float(row["d_to_delta_all"]) for row in center_rows if row["center_role"] == "class_balanced_leave_one_class_out"},
            "mean": float(np.mean(drift)), "median": float(np.median(drift)), "max": float(np.max(drift)),
            "threshold_used": None,
        },
        "class_balanced_vs_sample_equal_distance": float(next(row["d_to_delta_all"] for row in center_rows if row["center_name"] == "delta_sample_equal")),
        "held_out_classes_improved_vs_no_count": int(sum(float(row["delta_recall_shared_vs_no"]) > 0.0 for row in per_class)),
        "held_out_classes_improved_true_prob_vs_no_count": int(sum(float(row["delta_true_prob_shared_vs_no"]) > 0.0 for row in per_class)),
        "focus_classes": {name: per_class_by_name[name] for name in ("spring_oat", "winter_rye", "winter_wheat") if name in per_class_by_name},
        "beneficial_control_classes": {name: per_class_by_name[name] for name in ("horsebeans", "spring_barley", "winter_barley", "winter_triticale") if name in per_class_by_name},
        "overall": overall,
        "automatic_stage_iv_conclusion": None,
        "allowed_interpretations": {
            "A": "shared Phase stable and generalizes to most held-out classes",
            "B": "shared Phase statistically stable but classifier response is class-dependent",
            "C": "single shared Phase is composition-sensitive because deleting classes causes substantial raw drift",
            "D": "only if later evidence forces multiple natural cross-class modes; experiment 10 does not cluster",
        },
    }
    _json_dump(output / "16_shared_phase_diagnostic_summary.json", summary)
    manifest = {
        "protocol": summary["protocol"],
        "stage_a_cache": str(args.stage_a_cache.resolve()),
        "audit07_sample_csv": str(args.audit07_sample_csv.resolve()),
        "calibration_checkpoint": str(args.calibration_checkpoint.resolve()),
        "model_checkpoint": str(args.model_checkpoint.resolve()),
        "target_true_label_use": "oracle-only: correct gamma population, class-balanced weights, leave-one-class-out definition, and evaluation",
        "center_estimator": "weighted intrinsic Fisher-Rao Frechet/Karcher mean on the Phase SRVF unit sphere",
        "class_balanced_weight": "each included true class has equal total weight; samples divide that class mass uniformly",
        "sample_equal_control": "diagnostic only; every sample has equal weight",
        "held_out_class_integrity": "delta_-c assigns exactly zero estimation weight to every sample with true class c",
        "gamma_direction": "center gamma is source->target; target classifier positions use gamma^{-1}",
        "registration_calls": 0,
        "production_legality_filter": False,
        "beneficial_harmful_filter": False,
        "clustering": False,
        "M1_M2": False,
        "class_specific_phase_selection": False,
        "class_specific_alpha": False,
        "teacher_student": False,
        "stable_label": False,
        "training_updates": False,
        "domain_shape_transport": False,
        "classification_outcomes_used_to_fit_center": False,
        "drift_threshold": None,
    }
    _json_dump(output / "00_manifest.json", manifest)
    _write_readme(output / "README_中文说明.md")
    print(f"SHARED_PHASE_10_DONE|output={output}|registration_calls=0|clustering=false", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-cache", type=Path, required=True)
    parser.add_argument("--audit07-sample-csv", type=Path, required=True)
    parser.add_argument("--calibration-checkpoint", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-valid-count", type=int, default=EXPECTED_SAMPLES)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
