#!/usr/bin/env python3
"""Experiment 13B: fixed high-purity seed driven short semantic adaptation.

This program is deliberately target-label-free.  It trains two arms from the
same Stage-1 checkpoint: source-only continued training and source + immutable
T0 seeds.  PSE/LTAE/time encoder/classifier are trainable; geometry is frozen.
EMA Teacher is observation-only.  Oracle evaluation is a separate script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

import visualize_stage2_phase_alignment as phasevis
from dataset import BalancedBatchSampler, PixelSetData, GroupByShapesBatchSampler, worker_init_fn
from methods.structure_da.ema_teacher import Stage2EMATeacher
from methods.structure_da.seed_warmup_diagnostic import (
    class_balanced_seed_cross_entropy,
    configure_13b_semantic_student,
    cosine_feature_displacement,
    finite_hyperparameters,
    normalize_seed_manifest_rows,
    posterior_entropy,
    posterior_margin,
    seed_manifest_fingerprint,
    validate_checkpoint_epochs,
)
from transforms import Identity, Normalize, RandomSamplePixels, ToTensor


PROTOCOL = "13B_fixed_seed_short_semantic_adaptation_diagnostic"
EXPECTED_RUNTIME = ("austria/33UVP/2017", "denmark/32VNH/2017", 1, 0)
FORBIDDEN_TARGET_LABEL_KEYS = {"label", "true_label", "candidate_correct", "seed_correct"}


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key); seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LabelFreeDataset(Dataset):
    """Remove target truth at the dataset boundary."""
    def __init__(self, base: Dataset) -> None:
        self.base = base
    def __len__(self) -> int:
        return len(self.base)
    def __getitem__(self, index: int):
        item = dict(self.base[index])
        item.pop("label", None)
        return item


class FixedSeedDataset(Dataset):
    """Expose only immutable pseudo-labels from T0; never target truth."""
    def __init__(self, base: PixelSetData, pseudo_by_parcel: dict[int, int]) -> None:
        self.base = base
        self.pseudo_by_parcel = dict(pseudo_by_parcel)
        parcels = base.get_parcel_indices().astype(np.int64)
        if set(map(int, parcels.tolist())) != set(self.pseudo_by_parcel):
            raise ValueError("seed dataset parcels do not exactly match T0")
        self.pseudo_labels = np.asarray([self.pseudo_by_parcel[int(v)] for v in parcels], dtype=np.int64)
    def __len__(self) -> int:
        return len(self.base)
    def __getitem__(self, index: int):
        item = dict(self.base[index])
        parcel = int(item["parcel_index"])
        item.pop("label", None)
        item["pseudo_label"] = int(self.pseudo_by_parcel[parcel])
        return item


def _train_source_loader(runtime: dict, splits: dict, *, seed: int, num_workers: int, batch_size: int) -> DataLoader:
    transform = transforms.Compose([RandomSamplePixels(int(runtime.get("num_pixels", 64))), Normalize(), ToTensor()])
    dataset = PixelSetData(
        data_root=str(runtime["data_root"]), dataset_name=str(runtime["source"]), classes=list(runtime["classes"]),
        transform=transform, indices=splits[str(runtime["source"])]["train"], with_extra=False,
        closed_set=bool(runtime.get("closed_set", True)),
        combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter", False)),
        time_coordinate_mode=str(runtime.get("time_coordinate_mode", "canonical_day_of_year")),
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset=dataset,
        batch_sampler=BalancedBatchSampler(dataset.get_labels(), batch_size, seed=seed),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def _seed_loader(runtime: dict, seed_records, *, seed: int, num_workers: int, batch_size: int) -> DataLoader:
    pseudo_by = {int(item.sample_id): int(item.pseudo_label) for item in seed_records}
    transform = transforms.Compose([RandomSamplePixels(int(runtime.get("num_pixels", 64))), Normalize(), ToTensor()])
    base = PixelSetData(
        data_root=str(runtime["data_root"]), dataset_name=str(runtime["target"]), classes=list(runtime["classes"]),
        transform=transform, indices=set(pseudo_by), with_extra=False,
        closed_set=bool(runtime.get("closed_set", True)),
        combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter", False)),
        time_coordinate_mode=str(runtime.get("time_coordinate_mode", "canonical_day_of_year")),
    )
    dataset = FixedSeedDataset(base, pseudo_by)
    if len(dataset) <= len(set(dataset.pseudo_labels.tolist())):
        raise ValueError("T0 is too small for class-balanced target batches")
    resolved_bs = min(int(batch_size), len(dataset))
    if resolved_bs <= len(set(dataset.pseudo_labels.tolist())):
        resolved_bs = len(set(dataset.pseudo_labels.tolist())) + 1
    generator = torch.Generator().manual_seed(int(seed) + 991)
    return DataLoader(
        dataset=dataset,
        batch_sampler=BalancedBatchSampler(dataset.pseudo_labels, resolved_bs, seed=seed + 991),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def _scan_loader(data_root: str, domain: str, classes: Sequence[str], parcels: Iterable[int], runtime: dict, *, batch_size: int, num_workers: int, strip_label: bool) -> DataLoader:
    base = PixelSetData(
        data_root=data_root, dataset_name=domain, classes=list(classes),
        transform=transforms.Compose([Identity(), Normalize(), ToTensor()]),
        indices=set(int(v) for v in parcels), with_extra=False,
        closed_set=bool(runtime.get("closed_set", True)),
        combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter", False)),
        time_coordinate_mode=str(runtime.get("time_coordinate_mode", "canonical_day_of_year")),
    )
    dataset = LabelFreeDataset(base) if strip_label else base
    # GroupByShapesBatchSampler understands PixelSetData metadata; build it on
    # the base dataset, then apply the same integer indices to the label-free
    # wrapper so target truth never reaches the collated batch.
    sampler = GroupByShapesBatchSampler(base, batch_size, by_time=True, by_pixel_dim=True)
    return DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn,
    )


def _move(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}


def _forward(model, batch: dict):
    return model(
        batch["pixels"], batch["valid_pixels"], batch["positions"],
        batch.get("extra"), return_geometry=False,
    )


@torch.no_grad()
def _scan_model(model, loader, device: torch.device, *, allow_source_labels: bool = False) -> dict:
    model.eval()
    ids=[]; post=[]; features=[]; labels=[]
    for raw in loader:
        if not allow_source_labels and FORBIDDEN_TARGET_LABEL_KEYS.intersection(raw.keys()):
            raise RuntimeError("target truth leaked into 13B unlabeled scan")
        batch = _move(raw, device)
        out = _forward(model, batch)
        ids.append(batch["parcel_index"].detach().cpu().long())
        post.append(torch.softmax(out.logits.detach().float(), dim=1).cpu())
        features.append(out.fused_repr.detach().float().cpu())
        if allow_source_labels:
            labels.append(batch["label"].detach().cpu().long())
    sample_ids = torch.cat(ids); posterior = torch.cat(post); feature = torch.cat(features)
    order = torch.argsort(sample_ids)
    result = {"sample_id": sample_ids[order], "posterior": posterior[order], "feature": feature[order]}
    if allow_source_labels:
        result["label"] = torch.cat(labels)[order]
    return result


def _save_unlabeled_scan(path: Path, *, scan: dict, initial: dict, seed_by: dict[int,int], epoch: int, arm: str, role: str) -> list[dict]:
    ids = scan["sample_id"].long(); posterior = scan["posterior"].float(); feature = scan["feature"].float()
    if not torch.equal(ids, initial["sample_id"].long()):
        raise ValueError("target scan sample ordering changed across checkpoints")
    initial_post = initial["posterior"].float(); initial_feat = initial["feature"].float()
    initial_candidate = initial_post.argmax(dim=1)
    top1 = posterior.argmax(dim=1); top1_prob = posterior.max(dim=1).values
    margin = posterior_margin(posterior); entropy = posterior_entropy(posterior)
    displacement = cosine_feature_displacement(feature, initial_feat)
    seed_member = torch.tensor([int(v.item()) in seed_by for v in ids], dtype=torch.bool)
    seed_pseudo = torch.tensor([seed_by.get(int(v.item()), -1) for v in ids], dtype=torch.long)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_id=ids.numpy().astype(np.int64),
        posterior=posterior.numpy().astype(np.float32),
        feature=feature.numpy().astype(np.float32),
        initial_candidate=initial_candidate.numpy().astype(np.int64),
        seed_member=seed_member.numpy(),
        seed_pseudo_label=seed_pseudo.numpy().astype(np.int64),
        epoch=np.asarray([epoch], dtype=np.int64),
    )
    rows=[]
    for i, sid in enumerate(ids.tolist()):
        rows.append({
            "sample_id": int(sid), "arm": arm, "checkpoint_epoch": int(epoch), "model_role": role,
            "seed_member": bool(seed_member[i].item()), "seed_pseudo_label": int(seed_pseudo[i].item()),
            "initial_candidate": int(initial_candidate[i].item()), "current_top1": int(top1[i].item()),
            "top1_probability": float(top1_prob[i].item()), "margin": float(margin[i].item()),
            "entropy": float(entropy[i].item()), "feature_cosine_displacement_from_stage1": float(displacement[i].item()),
        })
    return rows


def _save_source_scan(path: Path, scan: dict, *, epoch: int, arm: str, role: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_id=scan["sample_id"].numpy().astype(np.int64),
        posterior=scan["posterior"].numpy().astype(np.float32),
        feature=scan["feature"].numpy().astype(np.float32),
        true_label=scan["label"].numpy().astype(np.int64),
        epoch=np.asarray([epoch], dtype=np.int64), arm=np.asarray([arm]), model_role=np.asarray([role]),
    )


def _model_state_hash(model) -> str:
    h=hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _train_arm(
    *, arm: str, model, source_loader, seed_loader, initial_target: dict, target_train_loader,
    target_val_loader, source_val_loader, seed_by: dict[int,int], output: Path, device: torch.device,
    epochs: int, steps_per_epoch: int, checkpoints: tuple[int,...], lr: float, weight_decay: float,
    lambda_target: float, ema_decay: float, amp_enabled: bool, seed: int,
) -> tuple[list[dict], dict]:
    _set_seed(seed)
    policy = configure_13b_semantic_student(model)
    named = dict(model.named_parameters())
    params = [named[name] for name in policy.trainable_parameter_names]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*steps_per_epoch, eta_min=0.0)
    teacher = Stage2EMATeacher.from_student(model, policy, decay=ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and device.type=="cuda"))
    trajectory_rows=[]
    source_stream=hashlib.sha256()

    def checkpoint(epoch: int) -> None:
        student_scan=_scan_model(model, target_train_loader, device)
        teacher_scan=_scan_model(teacher.model(), target_train_loader, device)
        trajectory_rows.extend(_save_unlabeled_scan(output/"target_trajectory_unlabeled"/arm/f"epoch_{epoch:03d}_student.npz", scan=student_scan, initial=initial_target, seed_by=seed_by, epoch=epoch, arm=arm, role="student"))
        trajectory_rows.extend(_save_unlabeled_scan(output/"target_trajectory_unlabeled"/arm/f"epoch_{epoch:03d}_teacher.npz", scan=teacher_scan, initial=initial_target, seed_by=seed_by, epoch=epoch, arm=arm, role="teacher"))
        student_val=_scan_model(model, target_val_loader, device)
        teacher_val=_scan_model(teacher.model(), target_val_loader, device)
        np.savez_compressed(output/"target_val_unlabeled"/arm/f"epoch_{epoch:03d}_student.npz", sample_id=student_val["sample_id"].numpy(), posterior=student_val["posterior"].numpy(), feature=student_val["feature"].numpy())
        np.savez_compressed(output/"target_val_unlabeled"/arm/f"epoch_{epoch:03d}_teacher.npz", sample_id=teacher_val["sample_id"].numpy(), posterior=teacher_val["posterior"].numpy(), feature=teacher_val["feature"].numpy())
        source_student=_scan_model(model, source_val_loader, device, allow_source_labels=True)
        source_teacher=_scan_model(teacher.model(), source_val_loader, device, allow_source_labels=True)
        _save_source_scan(output/"source_val_trajectory"/arm/f"epoch_{epoch:03d}_student.npz", source_student, epoch=epoch, arm=arm, role="student")
        _save_source_scan(output/"source_val_trajectory"/arm/f"epoch_{epoch:03d}_teacher.npz", source_teacher, epoch=epoch, arm=arm, role="teacher")
        torch.save({
            "protocol": PROTOCOL, "arm": arm, "epoch": epoch,
            "student_state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
            "teacher_state_dict": {k:v.detach().cpu() for k,v in teacher.model().state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "trainable_parameter_names": policy.trainable_parameter_names,
        }, output/"checkpoints"/arm/f"epoch_{epoch:03d}.pt")
        print(f"SEED13B_CHECKPOINT|arm={arm}|epoch={epoch}|target_labels_used=false", flush=True)

    for p in (output/"target_val_unlabeled"/arm, output/"source_val_trajectory"/arm, output/"checkpoints"/arm): p.mkdir(parents=True, exist_ok=True)
    checkpoint(0)
    source_iter=iter(source_loader); seed_iter=(None if seed_loader is None else iter(seed_loader))
    for epoch in range(1, epochs+1):
        model.train(); meters={"source":0.0,"target":0.0,"total":0.0}; steps=0
        for _ in range(steps_per_epoch):
            try: source_raw=next(source_iter)
            except StopIteration:
                source_iter=iter(source_loader); source_raw=next(source_iter)
            source_stream.update(source_raw["parcel_index"].detach().cpu().numpy().astype(np.int64).tobytes())
            source=_move(source_raw, device)
            optimizer.zero_grad(set_to_none=True)
            amp_on=bool(amp_enabled and device.type=="cuda")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_on):
                source_out=_forward(model, source)
                source_loss=F.cross_entropy(source_out.logits, source["label"].long())
                target_loss=source_loss.sum()*0.0
                if arm=="main":
                    if seed_iter is None: raise RuntimeError("main arm requires fixed T0 loader")
                    try: target_raw=next(seed_iter)
                    except StopIteration:
                        seed_iter=iter(seed_loader); target_raw=next(seed_iter)
                    if FORBIDDEN_TARGET_LABEL_KEYS.intersection(target_raw.keys()):
                        raise RuntimeError("target truth leaked into 13B target-seed training batch")
                    target=_move(target_raw, device)
                    target_out=_forward(model, target)
                    target_loss=class_balanced_seed_cross_entropy(target_out.logits, target["pseudo_label"].long())
                total=source_loss + (float(lambda_target)*target_loss if arm=="main" else 0.0)
            scaler.scale(total).backward(); scaler.step(optimizer); scaler.update(); scheduler.step(); teacher.update_after_optimizer_step(model)
            meters["source"]+=float(source_loss.detach()); meters["target"]+=float(target_loss.detach()); meters["total"]+=float(total.detach()); steps+=1
        print(f"SEED13B_TRAIN_EPOCH|arm={arm}|epoch={epoch}/{epochs}|steps={steps}|source_loss={meters['source']/steps:.6f}|target_loss={meters['target']/steps:.6f}|total_loss={meters['total']/steps:.6f}|lambda_target={(lambda_target if arm=='main' else 0.0):.6g}", flush=True)
        if epoch in checkpoints: checkpoint(epoch)
    return trajectory_rows, {"source_stream_hash":source_stream.hexdigest(),"final_student_hash":_model_state_hash(model),"final_teacher_hash":_model_state_hash(teacher.model()),"trainable_parameter_names":list(policy.trainable_parameter_names)}


def _write_readme(path: Path, config: dict) -> None:
    class_names = ", ".join(f"{i}:{name}" for i, name in enumerate(config["classes"]))
    path.write_text(f"""# 实验 13B：高纯度种子驱动的短程语义适应诊断

## 实验目的

本实验只回答：**在 shared Domain Phase 尚未建立时，固定高纯度低覆盖 T0 是否足以驱动健康的 target semantic adaptation，并产生 Stage-1 静态状态中不存在的动态信息。**

固定对象：`AT1 → DK1`，seed=`{config['seed']}`，fold=`{config['fold']}`；target 动态分析 split=`target-train`，泛化诊断 split=`target-val`，source semantic retention 使用 `source-val`。Stage-1 checkpoint：`{config['stage1_checkpoint']}`。

类别映射：{class_names}。

## 训练边界

两组都从完全相同的 Stage-1 checkpoint 初始化：

- `source_only`：只优化 source true-label CE；
- `main`：优化 source true-label CE + 固定 T0 的 class-balanced target seed CE。

Student 可训练边界为 **PSE_S + raw LTAE + TimeEncoder + Classifier**。独立的 Stage-1 geometry anchor 代表冻结 `PSE_G + decomposition + SRVF/source geometry`；本实验不运行 registration、不更新 Phase、不构造 Stable Label。EMA Teacher 只跟随 Student 并输出轨迹，**禁止产生新 pseudo-label 或修改 T0**。

T0 在训练前从上游 manifest 读取并复制到 `01_initial_seed_manifest.csv`。其 SHA256 为 `{config['seed_manifest_sha256']}`，seed 数量 `{config['seed_count']}`，具有 seed 的类别 `{config['seed_classes']}`。训练启动后不 add/remove/relabel/refresh。

固定优化配置：Adam，LR=`{config['lr']}`，weight decay=`{config['weight_decay']}`，lambda_t=`{config['lambda_target']}`，EMA decay=`{config['ema_decay']}`。训练窗口 `{config['epochs']}` epochs × `{config['steps_per_epoch']}` steps；预先固定 checkpoint=`{config['checkpoint_epochs']}`。不存在 best epoch、oracle early stopping 或动态 lambda。

## 无标签输出

- `00_stage2_warmup_config.json`：checkpoint、T0 hash、优化器、LR、lambda_t、EMA、训练预算、参数更新边界与 source-stream 一致性。
- `01_initial_seed_manifest.csv`：只读 T0。字段 `sample_id/pseudo_label/seed_selected/selection_evidence`；禁止包含 target true label。
- `02_stage1_target_reference_unlabeled.npz`：Stage-1 target-train `posterior/feature/initial_candidate`，作为所有动态量的 r=0 reference。
- `02b_stage1_source_val_reference.npz`：冻结 source anchor 在 source-val 上的 reference posterior/feature；source label 在 source supervision/held-out audit 中合法。
- `02c_stage1_target_val_reference_unlabeled.npz`：Stage-1 target-val reference，不含 target truth。
- `03_target_trajectory_unlabeled.csv`：`sample × arm × checkpoint × model_role` 的 `current_top1/top1_probability/margin/entropy/feature_cosine_displacement_from_stage1`，并保留固定 `initial_candidate/seed_member/seed_pseudo_label`。
- `target_trajectory_unlabeled/<arm>/epoch_XXX_{student,teacher}.npz`：完整 target-train posterior 与 current-LTAE feature。动态 CSV 不压缩成任何 `dynamic_reliability_score`。
- `target_val_unlabeled/`：固定 checkpoint 的 target-val posterior/feature；训练程序不读取其 true label。
- `source_val_trajectory/`：固定 checkpoint 的 source held-out Student/Teacher 轨迹。
- `checkpoints/`：预先固定 epoch 的 Student、EMA Teacher、optimizer、scheduler 状态；**不从中选择 oracle 最优 epoch**。

## Oracle-only 后处理

只有训练程序完整写出 `03_target_trajectory_unlabeled.csv` 并确认两组 source stream 一致以后，独立 `evaluate_stage2_seed_warmup_13b_oracle.py` 才读取 target metadata。以下所有 target truth 均为 **oracle-only diagnostic，不参与训练或无监督决策**。

`oracle/model_trajectory_metrics.csv` 给出 Accuracy/Macro-F1/Weighted-F1 及相对 epoch0 的变化，分别覆盖 target-train、target-val、source-val；核心因果比较是同一固定 checkpoint 的 `main - source_only`，而不是挑最好 epoch。

`oracle/seed_nonseed_trajectory.csv` 对 Stage-1 initially-wrong 样本报告 true-class probability change、initial-wrong-class probability change、二者同时向正确方向移动的比例和 current-correct fraction。**non_seed** 是判断 T0 信息是否通过共享参数传播到未监督 target population 的关键组。

`oracle/class_trajectory.csv` 对 Student/Teacher 分别报告每类 precision、recall、F1、support、pred_count。重点检查 spring_barley / winter_rye / winter_wheat 是否恢复，以及 spring_oat / winter_triticale 是否继续异常吸收。

`oracle/critical_flow_trajectory.csv` 使用 **Stage-1 时刻固定 cohort**，绝不在后续 checkpoint 重定义：`barley→oat`、`rye→triticale`、`wheat→triticale`，以及 `barley→barley/oat→oat/triticale→triticale/rye→rye/wheat→wheat` 健康参考。对错误 cohort，`mean_delta_correct_prob > 0` 且 `mean_delta_initial_candidate_prob < 0` 表示总体沿“更可纠正”方向移动；它只是 oracle 数学诊断，不是未来 revoke/gate 规则。

`oracle/main_vs_source_only_contrast.csv` 直接给出固定 checkpoint 上 Main 相对 source-only 的 Accuracy/Macro-F1/Weighted-F1 差值，用于排除“只是多做了 source training steps”的解释。

## 可以与不能得出的结论

本实验可以判断：固定 T0 是否产生超出 source-only continuation 的 target 改善；non-seed 是否出现系统性正确方向 movement；关键错误流是否开始降低初始错误类别概率并提高 true-class 概率；健康类别和 source held-out semantic 是否被明显破坏。

本实验**不能**据此定义动态 revoke threshold、Teacher relabel、dynamic reliability score、shared/class-conditioned Domain Phase、MMD/DANN/CORAL/contrastive/prototype alignment、class-specific rule 或最终 `C(c)→T(c)`。代码不会自动给出“13B 成功/部分完成/失败”结论，最终判定必须对完整 trajectory 做理论解释。

本实验不生成依赖 oracle 选择的图，因此不存在通过图形挑 best checkpoint/threshold 的步骤。
""", encoding="utf-8")


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--calibration-checkpoint", type=Path, required=True)
    p.add_argument("--model-checkpoint", type=Path, required=True)
    p.add_argument("--seed-manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps-per-epoch", type=int, default=500)
    p.add_argument("--checkpoint-epochs", default="0,1,3,5")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-target", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main():
    args=parse_args(); finite_hyperparameters(lr=args.lr,weight_decay=args.weight_decay,lambda_target=args.lambda_target,ema_decay=args.ema_decay)
    checkpoints=validate_checkpoint_epochs(args.epochs, tuple(int(v) for v in args.checkpoint_epochs.split(",")))
    calibration=torch.load(args.calibration_checkpoint.resolve(), map_location="cpu", weights_only=False)
    runtime=dict(calibration.get("runtime_config") or {})
    if not runtime: raise ValueError("calibration checkpoint is missing runtime_config")
    runtime["data_root"]=str(args.data_root.resolve()) if args.data_root else str(runtime["data_root"])
    actual=(str(runtime["source"]),str(runtime["target"]),int(runtime["seed"]),int(args.fold))
    if actual!=EXPECTED_RUNTIME: raise ValueError(f"first 13B run is frozen to AT1->DK1 seed=1 fold=0; got {actual}")
    classes=[str(v) for v in runtime["classes"]]; device=torch.device(args.device); seed=int(runtime["seed"])
    model_checkpoint=torch.load(args.model_checkpoint.resolve(), map_location="cpu", weights_only=False)
    anchor=phasevis._build_model(runtime, calibration, device, model_checkpoint=model_checkpoint)
    anchor.eval(); [p.requires_grad_(False) for p in anchor.parameters()]
    geometry_anchor=phasevis._build_model(runtime, calibration, device, model_checkpoint=model_checkpoint)
    geometry_anchor.eval(); [p.requires_grad_(False) for p in geometry_anchor.parameters()]

    source_all=phasevis._eligible_parcels(runtime["data_root"],runtime["source"],classes,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    target_all=phasevis._eligible_parcels(runtime["data_root"],runtime["target"],classes,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    splits=phasevis._reconstruct_fold_splits(source_all,target_all,source=runtime["source"],target=runtime["target"],seed=seed,val_ratio=float(runtime.get("val_ratio",0.1)),test_ratio=float(runtime.get("test_ratio",0.2)),fold=args.fold)
    target_train=sorted(splits[runtime["target"]]["train"]); target_val=sorted(splits[runtime["target"]]["val"]); source_val=sorted(splits[runtime["source"]]["val"])
    target_train_loader=_scan_loader(runtime["data_root"],runtime["target"],classes,target_train,runtime,batch_size=args.batch_size,num_workers=args.num_workers,strip_label=True)
    target_val_loader=_scan_loader(runtime["data_root"],runtime["target"],classes,target_val,runtime,batch_size=args.batch_size,num_workers=args.num_workers,strip_label=True)
    source_val_loader=_scan_loader(runtime["data_root"],runtime["source"],classes,source_val,runtime,batch_size=args.batch_size,num_workers=args.num_workers,strip_label=False)
    initial=_scan_model(anchor,target_train_loader,device)
    initial_candidate={int(sid):int(c) for sid,c in zip(initial["sample_id"].tolist(), initial["posterior"].argmax(dim=1).tolist())}
    raw_seed_rows=_load_csv(args.seed_manifest.resolve())
    records=normalize_seed_manifest_rows(raw_seed_rows, valid_sample_ids=target_train, initial_candidates=initial_candidate, num_classes=len(classes))
    seed_by={r.sample_id:r.pseudo_label for r in records}; seed_hash=seed_manifest_fingerprint(records)

    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/"02_stage1_target_reference_unlabeled.npz",sample_id=initial["sample_id"].numpy(),posterior=initial["posterior"].numpy(),feature=initial["feature"].numpy(),initial_candidate=initial["posterior"].argmax(dim=1).numpy())
    anchor_source_val=_scan_model(anchor,source_val_loader,device,allow_source_labels=True)
    _save_source_scan(output/"02b_stage1_source_val_reference.npz",anchor_source_val,epoch=0,arm="baseline",role="source_anchor")
    anchor_target_val=_scan_model(anchor,target_val_loader,device)
    np.savez_compressed(output/"02c_stage1_target_val_reference_unlabeled.npz",sample_id=anchor_target_val["sample_id"].numpy(),posterior=anchor_target_val["posterior"].numpy(),feature=anchor_target_val["feature"].numpy())
    normalized_rows=[{"sample_id":r.sample_id,"pseudo_label":r.pseudo_label,"seed_selected":True,"selection_evidence":r.selection_evidence} for r in records]
    _write_csv(output/"01_initial_seed_manifest.csv",normalized_rows)
    config={"protocol":PROTOCOL,"source":runtime["source"],"target":runtime["target"],"seed":seed,"fold":args.fold,"classes":classes,"stage1_checkpoint":str(args.model_checkpoint.resolve()),"calibration_checkpoint":str(args.calibration_checkpoint.resolve()),"upstream_seed_manifest":str(args.seed_manifest.resolve()),"seed_manifest_sha256":seed_hash,"seed_count":len(records),"seed_classes":sorted(set(r.pseudo_label for r in records)),"epochs":args.epochs,"steps_per_epoch":args.steps_per_epoch,"checkpoint_epochs":list(checkpoints),"optimizer":"Adam","lr":args.lr,"weight_decay":args.weight_decay,"lambda_target":args.lambda_target,"ema_decay":args.ema_decay,"scheduler":"CosineAnnealingLR_stepwise","student_trainable":"PSE + raw LTAE/time encoder + classifier","geometry_state":"separate frozen Stage-1 copy; no registration run","teacher_role":"EMA observation only; never supplies pseudo-label","target_truth_in_training":False,"seed_refresh":False,"relabel":False,"phase_update":False,"alignment_loss":False,"best_epoch_selection":False,"source_anchor_state_hash":_model_state_hash(anchor),"geometry_anchor_state_hash":_model_state_hash(geometry_anchor)}
    _json_dump(output/"00_stage2_warmup_config.json",config); _write_readme(output/"README_中文说明.md",config)
    print(f"SEED13B_T0_FROZEN|n={len(records)}|classes={','.join(map(str,config['seed_classes']))}|sha256={seed_hash}|target_truth_used=false",flush=True)

    all_rows=[]; arm_meta={}
    for arm in ("source_only","main"):
        _set_seed(seed)
        student=phasevis._build_model(runtime,calibration,device,model_checkpoint=model_checkpoint)
        source_loader=_train_source_loader(runtime,splits,seed=seed,num_workers=args.num_workers,batch_size=args.batch_size)
        target_seed_loader=None if arm=="source_only" else _seed_loader(runtime,records,seed=seed,num_workers=args.num_workers,batch_size=args.batch_size)
        rows,meta=_train_arm(arm=arm,model=student,source_loader=source_loader,seed_loader=target_seed_loader,initial_target=initial,target_train_loader=target_train_loader,target_val_loader=target_val_loader,source_val_loader=source_val_loader,seed_by=seed_by,output=output,device=device,epochs=args.epochs,steps_per_epoch=args.steps_per_epoch,checkpoints=checkpoints,lr=args.lr,weight_decay=args.weight_decay,lambda_target=args.lambda_target,ema_decay=args.ema_decay,amp_enabled=args.amp,seed=seed)
        all_rows.extend(rows); arm_meta[arm]=meta
    if arm_meta["source_only"]["source_stream_hash"] != arm_meta["main"]["source_stream_hash"]:
        raise RuntimeError("source-only and Main did not receive the same source parcel stream")
    _write_csv(output/"03_target_trajectory_unlabeled.csv",all_rows)
    config["arm_metadata"]=arm_meta; config["source_stream_identical_between_arms"]=True
    _json_dump(output/"00_stage2_warmup_config.json",config)
    print("SEED13B_UNLABELED_COMPLETE|oracle_evaluation_ready=true|target_truth_used=false",flush=True)

if __name__=="__main__": main()
