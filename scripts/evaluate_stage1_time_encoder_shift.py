#!/usr/bin/env python3
"""Evaluate a Stage-1 time-encoder ablation with a TimeMatch-style scalar shift.

This script is intended for the second-layer causal comparison after retraining
Stage 1 with ``--time_encoder_type timematch_fixed_sinusoidal``.

It:
1. reconstructs the exact train/val/test split from the run config;
2. restores ``fold_0/stage1_best.pt``;
3. re-estimates the target->source scalar shift by TimeMatch's initial
   Inception-Score scan on target-train data;
4. evaluates the same target-test set with no shift and the selected scalar
   shift;
5. writes a Chinese README and CSV/JSON diagnostics.

Target labels are never used to choose the scalar shift.  They are used only for
post-hoc test metrics and oracle shift-scan columns.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, REPOSITORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import compare_stage2_phase_vs_timematch_shift as layer1
import visualize_stage2_phase_alignment as phasevis
import train as train_module
from dataset import create_evaluation_loaders
from methods.structure_da import TSStructureModel


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _macro_f1(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    return layer1._macro_f1(labels, predictions, tuple(range(num_classes)))


def _build_model(config: Namespace, checkpoint: dict, device: torch.device) -> TSStructureModel:
    model = TSStructureModel(
        num_classes=config.num_classes,
        input_dim=config.input_dim,
        with_extra=config.with_extra,
        time_reference=getattr(config, "time_reference", 0.0),
        time_scale=config.time_scale,
        tau_fast_init=config.tau_fast_init,
        tau_slow_init=config.tau_slow_init,
        tau_min=config.tau_min,
        delta_tau_min=config.delta_tau_min,
        trend_num_basis=config.trend_num_basis,
        structure_num_basis=config.structure_num_basis,
        canonical_grid_size=config.canonical_grid_size,
        roughness_grid_size=config.roughness_grid_size,
        trend_smoothing=config.trend_smoothing,
        structure_smoothing=config.structure_smoothing,
        n_head=config.n_head,
        d_k=config.d_k,
        d_model=config.d_model,
        ltae_mlp=train_module._int_list(config.ltae_mlp),
        dropout=config.dropout,
        classifier_hidden=train_module._int_list(config.classifier_hidden),
        max_initial_frequency=config.time2vec_max_frequency,
        time_encoder_type=config.time_encoder_type,
        timematch_pe_period=config.timematch_pe_period,
        timematch_pe_max_shift=config.timematch_pe_max_shift,
    )
    train_module.load_structure_da_state_dict(model, checkpoint["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def _evaluate_views(
    model: TSStructureModel,
    loader,
    *,
    device: torch.device,
    shift_days: float,
    time_scale_days: float,
    num_classes: int,
) -> dict:
    labels: List[Tensor] = []
    pred_no: List[Tensor] = []
    pred_shift: List[Tensor] = []
    prob_true_no: List[Tensor] = []
    prob_true_shift: List[Tensor] = []

    encoder = model.temporal_module.raw_encoder.shared_ltae.shared_time_encoder
    uses_time2vec = hasattr(encoder, "linear_weight") and hasattr(encoder, "phase")
    for raw_batch in loader:
        batch = phasevis._move_batch(raw_batch, device)
        backbone = model.forward_backbone(
            batch["pixels"],
            batch["valid_pixels"],
            batch["positions"],
            batch.get("extra"),
            time_mask=batch.get("time_mask"),
        )
        no = model.forward_from_backbone(
            backbone,
            batch["positions"],
            batch.get("extra"),
            return_geometry=False,
        )
        shifted_positions = layer1._scalar_positions(
            backbone, shift_days, time_scale_days
        )
        scalar_context = (
            layer1._timematch_time_extrapolation(model)
            if uses_time2vec
            else contextlib.nullcontext()
        )
        with scalar_context:
            shifted = model.forward_from_backbone(
                backbone,
                batch["positions"],
                batch.get("extra"),
                temporal_positions_override=shifted_positions,
                return_geometry=False,
            )

        y = batch["label"].detach().long()
        p0 = torch.softmax(no.logits.float(), dim=-1)
        ps = torch.softmax(shifted.logits.float(), dim=-1)
        labels.append(y.cpu())
        pred_no.append(p0.argmax(dim=-1).cpu())
        pred_shift.append(ps.argmax(dim=-1).cpu())
        rows = torch.arange(len(y), device=device)
        prob_true_no.append(p0[rows, y].cpu())
        prob_true_shift.append(ps[rows, y].cpu())

    y = torch.cat(labels).numpy().astype(np.int64)
    p0 = torch.cat(pred_no).numpy().astype(np.int64)
    ps = torch.cat(pred_shift).numpy().astype(np.int64)
    true0 = torch.cat(prob_true_no).numpy().astype(np.float64)
    trues = torch.cat(prob_true_shift).numpy().astype(np.float64)
    return {
        "samples": int(len(y)),
        "no_shift_accuracy": float((y == p0).mean()),
        "shift_accuracy": float((y == ps).mean()),
        "no_shift_macro_f1": _macro_f1(y, p0, num_classes),
        "shift_macro_f1": _macro_f1(y, ps, num_classes),
        "no_shift_true_probability_mean": float(true0.mean()),
        "shift_true_probability_mean": float(trues.mean()),
        "accuracy_gain": float((y == ps).mean() - (y == p0).mean()),
        "macro_f1_gain": float(
            _macro_f1(y, ps, num_classes) - _macro_f1(y, p0, num_classes)
        ),
        "wrong_to_right": int(np.logical_and(y != p0, y == ps).sum()),
        "right_to_wrong": int(np.logical_and(y == p0, y != ps).sum()),
    }


def _write_readme(
    path: Path,
    *,
    run_dir: Path,
    config: Namespace,
    checkpoint: dict,
    shift_result: dict,
    test_result: dict,
) -> None:
    source_val = checkpoint.get("source_val", {})
    text = f"""# 第二层 B：TimeMatch 固定时间位置编码 Stage-1 对照

## 1. 实验身份

训练目录：

`{run_dir}`

任务：

`{config.source} → {config.target}`

seed：`{config.seed}`

时间编码：

`{config.time_encoder_type}`

本实验只改变 Raw T/S LTAE 的时间位置编码；PSE、分解、T/S private LayerNorm、
shared attention、projection、classifier 和 Stage-1 losses 保持原设计。

TimeMatch fixed PE 参数：

- `T = {config.timematch_pe_period}`
- `max_temporal_shift / positional offset = {config.timematch_pe_max_shift} days`
- calendar scale = `{config.time_scale} days`

## 2. Stage-1 source 结果

selected epoch：`{checkpoint.get("epoch", "unknown")}`

source-val Macro-F1：`{source_val.get("macro_f1", "unknown")}`

## 3. TimeMatch-style scalar shift

目标域训练集上使用 **Inception Score** 无监督扫描 shift，
target true label 不参与 shift 选择。

最终选择：

`{int(shift_result["selected_shift_days"]):+d} days`

oracle best accuracy shift：

`{int(shift_result["oracle_best_accuracy_shift_days"]):+d} days`

oracle best Macro-F1 shift：

`{int(shift_result["oracle_best_macro_f1_shift_days"]):+d} days`

oracle 列只用于事后诊断。

## 4. 目标域 test

No shift：

- Acc = `{test_result["no_shift_accuracy"]:.6f}`
- Macro-F1 = `{test_result["no_shift_macro_f1"]:.6f}`
- mean true-class probability = `{test_result["no_shift_true_probability_mean"]:.6f}`

Selected scalar shift：

- Acc = `{test_result["shift_accuracy"]:.6f}`
- Macro-F1 = `{test_result["shift_macro_f1"]:.6f}`
- mean true-class probability = `{test_result["shift_true_probability_mean"]:.6f}`

Gain：

- Acc = `{test_result["accuracy_gain"]:+.6f}`
- Macro-F1 = `{test_result["macro_f1_gain"]:+.6f}`
- wrong→right = `{test_result["wrong_to_right"]}`
- right→wrong = `{test_result["right_to_wrong"]}`

## 5. 文件

`shift_scan.csv`
：每个整数 shift 的 Inception Score，以及仅用于分析的 oracle Acc/F1。

`shift_scan.png`
：shift 扫描曲线。

`target_test_summary.json`
：No shift 与 selected scalar shift 的正式对照数字。

`experiment_manifest.json`
：本次 run、checkpoint 和时间编码配置。

## 6. 与 Time2Vec 基线的比较方式

不要只比较 `shift gain`，同时比较：

1. source-val Macro-F1；
2. target no-shift Macro-F1；
3. IS-selected scalar-shift Macro-F1；
4. scalar shift 带来的 Macro-F1 gain。

如果 fixed PE 的 source/no-shift 不下降，同时 scalar shift gain 明显大于 Time2Vec，
才支持“当前 Time2Vec 没有充分利用 calendar shift”这一判断。
"""
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "train_config.json"
    checkpoint_path = run_dir / f"fold_{args.fold}" / "stage1_best.pt"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config = Namespace(**config_data)
    config.output_dir = str(run_dir)
    config.model = "structure_da"
    if not hasattr(config, "time_encoder_type"):
        config.time_encoder_type = "continuous_time2vec"
    if not hasattr(config, "timematch_pe_period"):
        config.timematch_pe_period = 1000.0
    if not hasattr(config, "timematch_pe_max_shift"):
        config.timematch_pe_max_shift = 100.0

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    eligible_indices, _ = train_module.prepare_data_protocol(config)
    random.seed(config.seed)
    splits = train_module.create_train_val_test_folds(
        [config.source, config.target],
        config.num_folds,
        eligible_indices,
        config.val_ratio,
        config.test_ratio,
    )[args.fold]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = _build_model(config, checkpoint, device)

    shift_loader = layer1._timematch_estimation_loader(
        config.data_root,
        config.target,
        config.classes,
        splits[config.target]["train"],
        closed_set=config.closed_set,
        combine_spring_and_winter=config.combine_spring_and_winter,
        time_coordinate_mode=config.time_coordinate_mode,
        batch_size=args.timematch_batch_size,
        num_workers=args.num_workers,
        num_pixels=args.timematch_num_pixels,
        seed=config.seed,
    )
    print(
        "SECOND_LAYER_FIXED_PE_SHIFT_SCAN_START|"
        f"run={run_dir}|encoder={config.time_encoder_type}",
        flush=True,
    )
    shift_result = layer1._estimate_timematch_scalar_shift(
        model,
        shift_loader,
        device=device,
        min_shift=-args.timematch_max_shift,
        max_shift=args.timematch_max_shift,
        max_batches=args.timematch_estimation_batches,
        num_classes=config.num_classes,
        time_scale_days=config.time_scale,
    )
    selected_shift = int(shift_result["selected_shift_days"])

    _, target_test_loader = create_evaluation_loaders(
        config.target, splits, config, sample_pixels_val=False
    )
    test_result = _evaluate_views(
        model,
        target_test_loader,
        device=device,
        shift_days=selected_shift,
        time_scale_days=config.time_scale,
        num_classes=config.num_classes,
    )

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "shift_scan.csv", shift_result["rows"])
    layer1._plot_shift_scan(out / "shift_scan.png", shift_result, args.dpi)
    (out / "target_test_summary.json").write_text(
        json.dumps(test_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "source": config.source,
        "target": config.target,
        "seed": int(config.seed),
        "time_encoder_type": config.time_encoder_type,
        "timematch_pe_period": float(config.timematch_pe_period),
        "timematch_pe_max_shift": float(config.timematch_pe_max_shift),
        "selected_shift_days": selected_shift,
        "source_val": checkpoint.get("source_val", {}),
    }
    (out / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(
        out / "README_中文说明.md",
        run_dir=run_dir,
        config=config,
        checkpoint=checkpoint,
        shift_result=shift_result,
        test_result=test_result,
    )
    print(
        "SECOND_LAYER_FIXED_PE_EVAL_COMPLETE|"
        f"shift={selected_shift:+d}|"
        f"no_f1={test_result['no_shift_macro_f1']:.6f}|"
        f"shift_f1={test_result['shift_macro_f1']:.6f}|"
        f"output={out}",
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--timematch-max-shift", default=60, type=int)
    parser.add_argument("--timematch-estimation-batches", default=100, type=int)
    parser.add_argument("--timematch-batch-size", default=128, type=int)
    parser.add_argument("--timematch-num-pixels", default=64, type=int)
    parser.add_argument("--dpi", default=180, type=int)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
