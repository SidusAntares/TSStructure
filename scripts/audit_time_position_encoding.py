#!/usr/bin/env python3
"""Second-layer A diagnostic: where temporal-position sensitivity is attenuated.

No training is performed. For the same frozen Stage-1 model and target samples,
trace a scalar calendar shift through the temporal stack. If a supplied calibration
checkpoint contains a confirmed Domain Phase, that transform is audited as an
additional optional view; a confirmed Phase is not required for this second-layer
diagnostic.

    time encoding -> branch token -> key -> attention -> pooled representation

Two encoding formulas are evaluated:
1. the checkpoint's trained ContinuousTime2Vec;
2. the official TimeMatch fixed sinusoidal formula, injected post-hoc only for
   mechanistic sensitivity analysis.

The TimeMatch post-hoc path is NOT a performance ablation because downstream
weights were trained with Time2Vec.  The causal performance comparison requires
a separately retrained Stage-1 model and is handled by the Stage-1 encoder
variant added in the same patch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, REPOSITORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import compare_stage2_phase_vs_timematch_shift as layer1
import evaluate_stage1_time_encoder_shift as stage1eval
import visualize_stage2_phase_alignment as phasevis
import train as train_module
from models.ltae import ContinuousTime2Vec, TimeMatchFixedSinusoidal


EPS = 1e-8


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _norm_ratio(delta: Tensor, reference: Tensor) -> float:
    return float(delta.double().norm().item() / (reference.double().norm().item() + EPS))


def _attention_tv(after: Tensor, before: Tensor) -> float:
    # [1, H, 1, L], averaged across heads.
    diff = (after.double() - before.double()).abs()
    return float((0.5 * diff.sum(dim=-1)).mean().item())


def _attention_cosine(after: Tensor, before: Tensor) -> float:
    a = after.double().reshape(-1)
    b = before.double().reshape(-1)
    denom = a.norm() * b.norm()
    if denom.item() <= EPS:
        return 1.0
    return float(torch.dot(a, b).item() / denom.item())


def _current_encoding(shared, positions: Tensor, mask: Tensor) -> Tensor:
    encoder = shared.shared_time_encoder
    if isinstance(encoder, ContinuousTime2Vec):
        # Use the exact learned Time2Vec formula without the production [0,1]
        # gate so a TimeMatch-style scalar translation remains a translation.
        return layer1._unbounded_time2vec_forward(
            encoder, positions, time_mask=mask
        )
    return encoder(positions, time_mask=mask)


def _trace_branch(
    shared,
    *,
    component: Tensor,
    positions: Tensor,
    mask: Tensor,
    branch: str,
    time_encoding: Tensor,
) -> dict:
    if branch == "trend":
        input_norm = shared.trend_input_norm
        output_norm = shared.trend_output_norm
    elif branch == "structure":
        input_norm = shared.structure_input_norm
        output_norm = shared.structure_output_norm
    else:
        raise ValueError("branch must be trend or structure")

    safe = torch.where(mask.unsqueeze(-1), component, torch.zeros_like(component))
    projected = torch.relu(input_norm(shared.shared_input_projection(safe)))
    branch_tokens = projected + time_encoding
    key = shared.attention_heads.key(branch_tokens)
    pooled, attention = shared.attention_heads(branch_tokens, time_mask=mask)
    representation = output_norm(shared.dropout(shared.shared_projection(pooled)))
    return {
        "projected": projected.detach(),
        "time_encoding": time_encoding.detach(),
        "tokens": branch_tokens.detach(),
        "key": key.detach(),
        "attention": attention.detach(),
        "representation": representation.detach(),
    }


def _view_metrics(before: dict, after: dict) -> dict:
    delta_encoding = after["time_encoding"] - before["time_encoding"]
    return {
        "time_encoding_relative_change": _norm_ratio(
            delta_encoding, before["time_encoding"]
        ),
        "time_change_to_content_norm": _norm_ratio(
            delta_encoding, before["projected"]
        ),
        "key_relative_change": _norm_ratio(
            after["key"] - before["key"], before["key"]
        ),
        "attention_total_variation": _attention_tv(
            after["attention"], before["attention"]
        ),
        "attention_cosine_similarity": _attention_cosine(
            after["attention"], before["attention"]
        ),
        "representation_relative_change": _norm_ratio(
            after["representation"] - before["representation"],
            before["representation"],
        ),
        "time_encoding_norm": float(before["time_encoding"].double().norm().item()),
        "content_token_norm": float(before["projected"].double().norm().item()),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def _summaries(sample_rows: Sequence[dict]) -> List[dict]:
    keys = (
        "time_encoding_relative_change",
        "time_change_to_content_norm",
        "key_relative_change",
        "attention_total_variation",
        "attention_cosine_similarity",
        "representation_relative_change",
        "time_encoding_norm",
        "content_token_norm",
    )
    summaries: List[dict] = []
    groups = sorted(
        {
            (row["encoder"], row["branch"], row["transform"])
            for row in sample_rows
        }
    )
    for encoder, branch, transform in groups:
        subset = [
            row
            for row in sample_rows
            if row["encoder"] == encoder
            and row["branch"] == branch
            and row["transform"] == transform
        ]
        item = {
            "encoder": encoder,
            "branch": branch,
            "transform": transform,
            "samples": len(subset),
        }
        for key in keys:
            item[f"{key}_mean"] = _mean(float(row[key]) for row in subset)
            item[f"{key}_median"] = float(
                np.median([float(row[key]) for row in subset])
            )
        summaries.append(item)
    return summaries


def _plot_metric(
    path: Path,
    summaries: Sequence[dict],
    metric: str,
    title: str,
    ylabel: str,
    dpi: int,
) -> None:
    labels: List[str] = []
    values: List[float] = []
    for row in summaries:
        labels.append(
            f"{row['encoder']}\n{row['branch']}\n{row['transform']}"
        )
        values.append(float(row[f"{metric}_mean"]))
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(labels)), 5.5))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _write_readme(
    path: Path,
    *,
    checkpoint: Path,
    source: str,
    target: str,
    shift_days: float,
    samples: int,
    phase_available: bool,
    input_mode: str,
) -> None:
    phase_line = (
        "3. confirmed Domain Phase 的 `gamma^-1(t_target)`（额外诊断）。"
        if phase_available
        else "3. 本次没有 confirmed non-identity Domain Phase，因此不计算 Domain Phase 分支。"
    )
    text = f"""# 第二层 A：时间位置编码逐层敏感度诊断

## 1. 本次实验目的

本目录不是新的训练结果，而是 **0-step / no-gradient** 诊断。

输入模式：`{input_mode}`

样本范围：完整 closed-set target population。true label 仅用于每类均匀抽样，
不参与参数更新、shift 选择或任何训练。

模型 checkpoint：

`{checkpoint}`

任务：`{source} → {target}`

对同一批目标域样本比较：

1. 原始时间位置；
2. TimeMatch-style 固定平移 `{shift_days:+.1f}` 天；
{phase_line}

第二层的核心问题是固定 calendar shift 从哪一层开始被削弱：

```text
time position
→ time encoding
→ T/S content + time
→ attention key
→ attention distribution
→ pooled LTAE representation
```

因此 **Domain Phase 不是运行本诊断的前置条件**。第一层已经单独比较过
scalar shift 与 Domain Phase；本目录主要审计时间编码和 LTAE 对 scalar shift 的响应。

本次共记录 {samples} 条 `encoder × branch × transform` 样本诊断行。

## 2. 两种时间编码

### `checkpoint_time_encoder`

使用 checkpoint 中真实训练过的时间编码。当前基线通常是 `ContinuousTime2Vec`。

### `timematch_fixed_posthoc`

使用 TimeMatch 官方固定 sinusoidal positional encoding 的解析连续形式：

- `T = 1000`；
- day coordinate 使用 `normalized_position × 365`；
- 与官方 `positions + max_temporal_shift` 一致，默认再加 `+100 day` 偏置；
- 对整数 day 与 TimeMatch 的固定 embedding table 数值一致；
- fractional day 直接按相同 sin/cos 公式计算，不做 round。

**重要：该分支只是 post-hoc 机制诊断。**
后面的 attention / classifier 权重并没有用这种 PE 训练，所以不能根据它的分类结果判断
TimeMatch PE 是否优于 Time2Vec。真正的性能比较必须重新训练 Stage 1。

## 3. CSV

### `sensitivity_samples.csv`

每行是一条 `样本 × encoder × T/S branch × transform`。

主要指标：

- `time_encoding_relative_change`：时间编码本身的相对变化；
- `time_change_to_content_norm`：时间编码变化相对于内容 token 的尺度；
- `key_relative_change`：进入 attention 后 key 的相对变化；
- `attention_total_variation`：attention 权重变化；
- `attention_cosine_similarity`：前后 attention 的相似程度，越接近 1 越不敏感；
- `representation_relative_change`：最终 T/S LTAE representation 的相对变化。

### `sensitivity_summary.csv`

对上述量按 encoder / T-S branch / transform 汇总 mean 与 median。

## 4. 图片

`plots/time_encoding_response.png`：时间编码自身响应。

`plots/key_response.png`：时间变化经过 key projection 后还剩多少。

`plots/attention_response.png`：真正 attention distribution 改变多少。

`plots/representation_response.png`：最终 pooled T/S representation 改变多少。

## 5. 判读

如果 `Time2Vec encoding change` 很小而 `TimeMatch fixed PE change` 明显大，
优先说明当前 Time2Vec 本身缺乏 calendar-shift 敏感度。

如果 Time2Vec encoding change 明显但 key change 很小，说明 Stage-1 学出的
key projection 在抑制时间编码方向。

如果 key change 明显但 attention TV 很小 / cosine 接近 1，说明 master query +
softmax 后时间变化被进一步削弱。

如果直到 representation 都有明显变化，但 source-target alignment / classifier 改善仍弱，
问题才更靠近后续表示融合与分类器。
"""
    path.write_text(text, encoding="utf-8")


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    groups: List[dict] = []
    class_to_group: Dict[int, dict] = {}
    phase_available = False

    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        config_path = run_dir / "train_config.json"
        checkpoint_path = run_dir / f"fold_{args.fold}" / "stage1_best.pt"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config = argparse.Namespace(**config_data)
        config.output_dir = str(run_dir)
        config.model = "structure_da"
        if args.data_root is not None:
            config.data_root = str(args.data_root)
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
        model = stage1eval._build_model(config, checkpoint, device).eval()

        source = str(config.source)
        target = str(config.target)
        classes = [str(v) for v in config.classes]
        data_root = str(config.data_root)
        seed = int(config.seed)
        closed_set = bool(config.closed_set)
        combine = bool(config.combine_spring_and_winter)
        time_mode = str(config.time_coordinate_mode)
        calendar_days = float(config.time_scale)
        train_indices = splits
        input_mode = "stage1_run"
    else:
        checkpoint_path = args.checkpoint.resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        runtime = checkpoint.get("runtime_config")
        if not isinstance(runtime, dict):
            raise ValueError("calibration checkpoint must contain runtime_config")

        source = str(runtime["source"])
        target = str(runtime["target"])
        classes = [str(v) for v in runtime["classes"]]
        data_root = str(args.data_root or runtime["data_root"])
        seed = int(runtime["seed"])
        closed_set = bool(phasevis._runtime_value(runtime, "closed_set", True))
        combine = bool(phasevis._runtime_value(runtime, "combine_spring_and_winter", False))
        time_mode = str(
            phasevis._runtime_value(runtime, "time_coordinate_mode", "canonical_day_of_year")
        )
        val_ratio = float(phasevis._runtime_value(runtime, "val_ratio", 0.1))
        test_ratio = float(phasevis._runtime_value(runtime, "test_ratio", 0.2))
        calendar_days = float(phasevis._runtime_value(runtime, "time_scale", 365.0))
        model = phasevis._build_model(runtime, checkpoint, device).eval()

        try:
            groups = phasevis._checkpoint_group_payloads(checkpoint)
        except ValueError as error:
            if "no confirmed non-identity Domain Phase group" not in str(error):
                raise
            groups = []
        class_to_group = phasevis._class_to_group(groups) if groups else {}
        phase_available = bool(class_to_group)

        source_all = phasevis._eligible_parcels(
            data_root,
            source,
            classes,
            closed_set=closed_set,
            combine_spring_and_winter=combine,
            time_coordinate_mode=time_mode,
        )
        target_all = phasevis._eligible_parcels(
            data_root,
            target,
            classes,
            closed_set=closed_set,
            combine_spring_and_winter=combine,
            time_coordinate_mode=time_mode,
        )
        train_indices = phasevis._reconstruct_fold_train_indices(
            source_all,
            target_all,
            source=source,
            target=target,
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            fold=args.fold,
        )
        input_mode = "calibration_checkpoint"

    shared = model.temporal_module.raw_encoder.shared_ltae
    all_classes = tuple(range(len(classes)))
    if args.classes is None:
        requested_classes = (
            tuple(sorted(class_to_group)) if phase_available else all_classes
        )
    else:
        requested_classes = tuple(args.classes)
        unknown = sorted(set(requested_classes) - set(all_classes))
        if unknown:
            raise ValueError(
                "requested class ids are outside the closed set: "
                + ",".join(str(v) for v in unknown)
            )

    # This is a mechanistic, no-training diagnostic.  Use the full eligible
    # closed-set target population for deterministic per-class sampling instead
    # of reconstructing the Stage-1 target split.  Target labels are used only
    # to balance the diagnostic sample; they never update parameters or choose
    # a temporal transform.  This also makes the audit independent of fragile
    # split-reconstruction details in a saved Stage-1 run.
    target_population = phasevis._eligible_parcels(
        data_root,
        target,
        classes,
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
    )
    if target_population.size == 0:
        raise ValueError("closed-set target population is empty")
    target_meta = phasevis._metadata_train_dataset(
        data_root,
        target,
        classes,
        set(int(value) for value in target_population.tolist()),
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
    )
    if len(target_meta) == 0:
        raise ValueError(
            "eligible target parcels were found but the diagnostic metadata dataset is empty"
        )
    target_parcels = phasevis._uniform_selected_parcels(
        target_meta, requested_classes, args.samples_per_class
    )
    target_loader = phasevis._selected_loader(
        data_root,
        target,
        classes,
        target_parcels,
        closed_set=closed_set,
        combine_spring_and_winter=combine,
        time_coordinate_mode=time_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    target_records = phasevis._collect_geometry(
        model,
        target_loader,
        device=device,
        target=phase_available,
        class_to_group=class_to_group,
    )

    fixed = TimeMatchFixedSinusoidal(
        shared.d_model,
        time_reference=0.0,
        time_scale=1.0,
        calendar_scale_days=calendar_days,
        position_offset_days=args.timematch_pe_max_shift,
        period=args.timematch_pe_period,
    ).to(device=device, dtype=next(model.parameters()).dtype).eval()

    sample_rows: List[dict] = []
    for class_id in requested_classes:
        for record in target_records.get(class_id, []):
            mask = record["mask"].to(device=device, dtype=torch.bool).unsqueeze(0)
            pos0 = record["positions"].to(device=device).unsqueeze(0)
            scalar = torch.where(
                mask,
                pos0 + float(args.shift_days) / calendar_days,
                torch.zeros_like(pos0),
            )
            views = {"scalar_shift": scalar}
            if phase_available and "positions_after" in record:
                views["domain_phase"] = record["positions_after"].to(device=device).unsqueeze(0)
            components = {
                "trend": record["trend_tokens"].to(device=device).unsqueeze(0),
                "structure": record["structure_tokens"].to(device=device).unsqueeze(0),
            }

            current0 = _current_encoding(shared, pos0, mask)
            fixed0 = fixed(pos0, time_mask=mask)
            for transform, pos1 in views.items():
                current1 = _current_encoding(shared, pos1, mask)
                fixed1 = fixed(pos1, time_mask=mask)
                for branch, component in components.items():
                    for encoder_name, enc0, enc1 in (
                        ("checkpoint_time_encoder", current0, current1),
                        ("timematch_fixed_posthoc", fixed0, fixed1),
                    ):
                        before = _trace_branch(
                            shared,
                            component=component,
                            positions=pos0,
                            mask=mask,
                            branch=branch,
                            time_encoding=enc0,
                        )
                        after = _trace_branch(
                            shared,
                            component=component,
                            positions=pos1,
                            mask=mask,
                            branch=branch,
                            time_encoding=enc1,
                        )
                        metrics = _view_metrics(before, after)
                        sample_rows.append(
                            {
                                "class_id": int(class_id),
                                "class_name": classes[class_id],
                                "parcel_index": int(record["parcel_index"]),
                                "phase_group_id": int(record.get("group_id", -1)),
                                "encoder": encoder_name,
                                "branch": branch,
                                "transform": transform,
                                **metrics,
                            }
                        )

    summaries = _summaries(sample_rows)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "sensitivity_samples.csv", sample_rows)
    _write_csv(out / "sensitivity_summary.csv", summaries)
    _plot_metric(
        out / "plots" / "time_encoding_response.png",
        summaries,
        "time_encoding_relative_change",
        "Time encoding response to temporal correction",
        "Relative encoding change",
        args.dpi,
    )
    _plot_metric(
        out / "plots" / "key_response.png",
        summaries,
        "key_relative_change",
        "Attention-key response to temporal correction",
        "Relative key change",
        args.dpi,
    )
    _plot_metric(
        out / "plots" / "attention_response.png",
        summaries,
        "attention_total_variation",
        "Attention-distribution response",
        "Mean total variation",
        args.dpi,
    )
    _plot_metric(
        out / "plots" / "representation_response.png",
        summaries,
        "representation_relative_change",
        "Pooled LTAE representation response",
        "Relative representation change",
        args.dpi,
    )
    manifest = {
        "input_mode": input_mode,
        "checkpoint": str(checkpoint_path),
        "source": source,
        "target": target,
        "classes": list(requested_classes),
        "shift_days": float(args.shift_days),
        "phase_available": phase_available,
        "phase_group_count": len(groups),
        "timematch_pe_period": float(args.timematch_pe_period),
        "timematch_pe_max_shift": float(args.timematch_pe_max_shift),
        "samples_per_class": int(args.samples_per_class),
        "sampling_scope": "full_closed_set_target_population",
        "target_population_size": int(target_population.size),
        "diagnostic_rows": len(sample_rows),
        "posthoc_fixed_pe_is_not_performance_ablation": True,
    }
    _json_dump(out / "diagnostic_manifest.json", manifest)
    _write_readme(
        out / "README_中文说明.md",
        checkpoint=checkpoint_path,
        source=source,
        target=target,
        shift_days=args.shift_days,
        samples=len(sample_rows),
        phase_available=phase_available,
        input_mode=input_mode,
    )
    print(
        "SECOND_LAYER_TIME_SENSITIVITY_COMPLETE|"
        f"rows={len(sample_rows)}|phase_available={str(phase_available).lower()}|output={out}",
        flush=True,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", default=None, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--classes", default=None, type=lambda v: tuple(int(x) for x in v.split(",")))
    parser.add_argument("--samples-per-class", default=32, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--shift-days", default=-21.0, type=float)
    parser.add_argument("--timematch-pe-period", default=1000.0, type=float)
    parser.add_argument("--timematch-pe-max-shift", default=100.0, type=float)
    parser.add_argument("--dpi", default=180, type=int)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
