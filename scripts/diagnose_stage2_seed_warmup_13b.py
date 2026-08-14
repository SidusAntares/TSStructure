#!/usr/bin/env python3
"""Experiment 13B: bootstrap pseudo-label strategy short adaptation comparison.

Five label-free bootstrap strategies plus source-only and an explicit oracle
upper-bound control are trained with the same short semantic trainer.  Initial
pseudo-labels are generated once at r=0 and remain immutable.  No Phase update,
registration refresh, pseudo-label refresh/relabel, alignment loss or Teacher
feedback is permitted.
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
    BOOTSTRAP_ARMS,
    LABEL_FREE_BOOTSTRAP_ARMS,
    FixedSeedRecord,
    class_balanced_seed_cross_entropy,
    confidence_geometry_veto_bootstrap,
    configure_13b_semantic_student,
    cosine_prototype_labels,
    dapl_bootstrap,
    diagonal_gaussian_conformity_percentile,
    finite_hyperparameters,
    ipl_bootstrap,
    posterior_entropy,
    posterior_margin,
    cosine_feature_displacement,
    seed_manifest_fingerprint,
    tfda_nn_bootstrap,
    timematch_confidence_bootstrap,
    validate_checkpoint_epochs,
)
from transforms import Identity, Normalize, RandomSamplePixels, ToTensor


PROTOCOL = "13B_bootstrap_strategy_short_adaptation_comparison_v2"
EXPECTED_RUNTIME = ("austria/33UVP/2017", "denmark/32VNH/2017", 1, 0)
FORBIDDEN_TARGET_LABEL_KEYS = {"label", "true_label", "candidate_correct", "seed_correct", "confusion_flow"}


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
                fields.append(key); seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class LabelFreeDataset(Dataset):
    def __init__(self, base: Dataset) -> None: self.base = base
    def __len__(self) -> int: return len(self.base)
    def __getitem__(self, index: int):
        item = dict(self.base[index]); item.pop("label", None); return item


class FixedSeedDataset(Dataset):
    """Expose immutable strategy pseudo-labels instead of target truth."""
    def __init__(self, base: PixelSetData, pseudo_by_parcel: dict[int, int]) -> None:
        self.base = base; self.pseudo_by_parcel = dict(pseudo_by_parcel)
        parcels = base.get_parcel_indices().astype(np.int64)
        if set(map(int, parcels.tolist())) != set(self.pseudo_by_parcel):
            raise ValueError("target supervision dataset does not exactly match its frozen manifest")
        self.pseudo_labels = np.asarray([self.pseudo_by_parcel[int(v)] for v in parcels], dtype=np.int64)
    def __len__(self) -> int: return len(self.base)
    def __getitem__(self, index: int):
        item = dict(self.base[index]); parcel = int(item["parcel_index"])
        item.pop("label", None); item["pseudo_label"] = int(self.pseudo_by_parcel[parcel]); return item


def _train_source_loader(runtime: dict, splits: dict, *, seed: int, num_workers: int, batch_size: int) -> DataLoader:
    transform = transforms.Compose([RandomSamplePixels(int(runtime.get("num_pixels", 64))), Normalize(), ToTensor()])
    ds = PixelSetData(
        data_root=str(runtime["data_root"]), dataset_name=str(runtime["source"]), classes=list(runtime["classes"]),
        transform=transform, indices=splits[str(runtime["source"])]["train"], with_extra=False,
        closed_set=bool(runtime.get("closed_set", True)), combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter", False)),
        time_coordinate_mode=str(runtime.get("time_coordinate_mode", "canonical_day_of_year")),
    )
    return DataLoader(
        dataset=ds, batch_sampler=BalancedBatchSampler(ds.get_labels(), batch_size, seed=seed), num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(seed),
    )


def _target_supervision_loader(runtime: dict, records: Sequence[FixedSeedRecord], *, seed: int, num_workers: int, batch_size: int) -> DataLoader | None:
    if not records:
        return None
    pseudo_by = {int(r.sample_id): int(r.pseudo_label) for r in records}
    transform = transforms.Compose([RandomSamplePixels(int(runtime.get("num_pixels", 64))), Normalize(), ToTensor()])
    base = PixelSetData(
        data_root=str(runtime["data_root"]), dataset_name=str(runtime["target"]), classes=list(runtime["classes"]),
        transform=transform, indices=set(pseudo_by), with_extra=False, closed_set=bool(runtime.get("closed_set", True)),
        combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter", False)),
        time_coordinate_mode=str(runtime.get("time_coordinate_mode", "canonical_day_of_year")),
    )
    ds = FixedSeedDataset(base, pseudo_by)
    n_classes = len(set(ds.pseudo_labels.tolist()))
    if len(ds) <= n_classes:
        # Explicitly preserve a near-empty bootstrap as a failed/degenerate arm
        # instead of silently changing its selection rule.
        return None
    resolved_bs = min(int(batch_size), len(ds))
    if resolved_bs <= n_classes: resolved_bs = n_classes + 1
    return DataLoader(
        dataset=ds, batch_sampler=BalancedBatchSampler(ds.pseudo_labels, resolved_bs, seed=seed + 991),
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed + 991),
    )


def _scan_loader(data_root: str, domain: str, classes: Sequence[str], parcels: Iterable[int], runtime: dict, *, batch_size: int, num_workers: int, strip_label: bool) -> DataLoader:
    base = PixelSetData(
        data_root=data_root, dataset_name=domain, classes=list(classes), transform=transforms.Compose([Identity(), Normalize(), ToTensor()]),
        indices=set(int(v) for v in parcels), with_extra=False, closed_set=bool(runtime.get("closed_set", True)),
        combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter", False)),
        time_coordinate_mode=str(runtime.get("time_coordinate_mode", "canonical_day_of_year")),
    )
    ds = LabelFreeDataset(base) if strip_label else base
    sampler = GroupByShapesBatchSampler(base, batch_size, by_time=True, by_pixel_dim=True)
    return DataLoader(dataset=ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn)


def _move(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}


def _forward(model, batch: dict):
    return model(batch["pixels"], batch["valid_pixels"], batch["positions"], batch.get("extra"), return_geometry=False)


@torch.no_grad()
def _scan_model(model, loader, device: torch.device, *, allow_labels: bool = False) -> dict:
    model.eval(); ids=[]; post=[]; features=[]; labels=[]
    for raw in loader:
        if not allow_labels and FORBIDDEN_TARGET_LABEL_KEYS.intersection(raw.keys()):
            raise RuntimeError("target truth leaked into a label-free 13B scan")
        batch=_move(raw,device); out=_forward(model,batch)
        ids.append(batch["parcel_index"].detach().cpu().long()); post.append(torch.softmax(out.logits.detach().float(),dim=1).cpu()); features.append(out.fused_repr.detach().float().cpu())
        if allow_labels: labels.append(batch["label"].detach().cpu().long())
    sample_id=torch.cat(ids); posterior=torch.cat(post); feature=torch.cat(features); order=torch.argsort(sample_id)
    result={"sample_id":sample_id[order],"posterior":posterior[order],"feature":feature[order]}
    if allow_labels: result["label"]=torch.cat(labels)[order]
    return result


def _model_state_hash(model) -> str:
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _records_to_rows(records: Sequence[FixedSeedRecord], strategy: str) -> list[dict]:
    return [{"sample_id":r.sample_id,"pseudo_label":r.pseudo_label,"seed_selected":True,"strategy":strategy,"selection_evidence":r.selection_evidence} for r in records]


def _load_structure_inputs(structure_dir: Path, initial: dict, *, source: str, target: str, seed: int, fold: int) -> dict:
    """Load only 13A-2 label-free target observables plus source-labeled cache."""
    root=structure_dir.resolve()
    obs_path=root/"01_unlabeled_sample_observables.csv"; dense_path=root/"02_unlabeled_dense_vectors.npz"; source_path=root/"cache/source_frozen_ltae_features.pt"
    for path in (obs_path,dense_path,source_path):
        if not path.is_file(): raise FileNotFoundError(f"13B requires 13A-2 structural input: {path}")
    rows=_load_csv(obs_path)
    forbidden={"true_label","candidate_correct","confusion_flow","seed_correct"}
    if rows and forbidden.intersection(rows[0]): raise ValueError("13A-2 structure observable unexpectedly contains target oracle fields")
    with np.load(dense_path,allow_pickle=False) as z: dense={k:z[k] for k in z.files}
    source_cache=torch.load(source_path,map_location="cpu",weights_only=False)
    if bool(source_cache.get("contains_target_true_labels",False)): raise ValueError("source feature cache claims target truth contamination")
    if (str(source_cache.get("source")),str(source_cache.get("target")),int(source_cache.get("seed")),int(source_cache.get("fold"))) != (source,target,seed,fold):
        raise ValueError("13A-2 source feature cache runtime does not match 13B")
    source_obs=dict(source_cache.get("observables") or {})
    ids=initial["sample_id"].numpy().astype(np.int64); post=initial["posterior"].numpy().astype(np.float32); feat=initial["feature"].numpy().astype(np.float32)
    if not np.array_equal(dense["sample_id"].astype(np.int64),ids): raise ValueError("13A-2 target dense ordering differs from current Stage-1 scan")
    if np.max(np.abs(dense["raw_posterior"].astype(np.float32)-post))>2e-5: raise ValueError("13A-2 posterior does not match current Stage-1 checkpoint")
    if np.max(np.abs(dense["frozen_ltae_feature"].astype(np.float32)-feat))>2e-4: raise ValueError("13A-2 LTAE feature does not match current Stage-1 checkpoint")
    by_id={int(r["sample_id"]):r for r in rows}
    if set(by_id)!=set(map(int,ids.tolist())): raise ValueError("13A-2 observable rows do not cover current target-train exactly")
    aligned=[by_id[int(v)] for v in ids.tolist()]
    return {"rows":aligned,"dense":dense,"source":source_obs,"root":root}


def _generate_label_free_bootstraps(initial: dict, structure: dict, *, args, num_classes: int) -> tuple[dict[str,tuple[FixedSeedRecord,...]], dict]:
    ids=initial["sample_id"].numpy().astype(np.int64); post=initial["posterior"].numpy().astype(np.float64); target_feat=initial["feature"].numpy().astype(np.float64); raw_pred=post.argmax(axis=1)
    source=structure["source"]; sx=np.asarray(source["features"],dtype=np.float64); sy=np.asarray(source["labels"],dtype=np.int64); sids=np.asarray(source["sample_ids"],dtype=np.int64)
    knn=np.asarray(structure["dense"]["target_knn_indices"],dtype=np.int64)
    if not np.array_equal(np.asarray(source["raw_pred"]).shape,np.asarray(source["labels"]).shape): raise ValueError("invalid source feature cache")

    _, conformity_pct=diagonal_gaussian_conformity_percentile(sids,sx,sy,target_feat,raw_pred,num_classes=num_classes,folds=args.conformity_crossfit_folds)
    proto=cosine_prototype_labels(sx,sy,target_feat,num_classes=num_classes)
    rows=structure["rows"]
    t_pct=np.asarray([float(r["T_registered_error_source_percentile"]) for r in rows],dtype=np.float64)
    s_pct=np.asarray([float(r["S_registered_error_source_percentile"]) for r in rows],dtype=np.float64)

    records={
        "PL_TIMEMATCH_CONF":timematch_confidence_bootstrap(ids,post,threshold=args.pseudo_threshold),
        "PL_DAPL_BOOT":dapl_bootstrap(ids,post,conformity_pct,confidence_threshold=args.pseudo_threshold,conformity_threshold=args.conformity_percentile_threshold),
        "PL_IPL_BOOT":ipl_bootstrap(ids,post,proto,knn,k=args.bootstrap_knn_k,support_threshold=args.ipl_support_threshold),
        "PL_TFDA_NN":tfda_nn_bootstrap(ids,post,knn,k=args.bootstrap_knn_k),
        "PL_CONF_GEOM":confidence_geometry_veto_bootstrap(ids,post,t_pct,s_pct,confidence_threshold=args.pseudo_threshold,conflict_percentile=args.geometry_conflict_percentile),
    }
    if tuple(records) != LABEL_FREE_BOOTSTRAP_ARMS: raise RuntimeError("13B label-free bootstrap matrix drifted")
    diagnostics={
        "raw_candidate":raw_pred.astype(np.int64), "dapl_conformity_percentile":conformity_pct.astype(np.float32),
        "ipl_prototype_label":proto.astype(np.int64), "T_registered_error_source_percentile":t_pct.astype(np.float32),
        "S_registered_error_source_percentile":s_pct.astype(np.float32),
    }
    return records, diagnostics


def _save_scan(path: Path, *, scan: dict, initial: dict, seed_by: dict[int,int], epoch: int, arm: str, role: str, oracle_arm: bool) -> list[dict]:
    ids=scan["sample_id"].long(); posterior=scan["posterior"].float(); feature=scan["feature"].float()
    if not torch.equal(ids,initial["sample_id"].long()): raise ValueError("target scan ordering changed across checkpoints")
    initial_post=initial["posterior"].float(); initial_feat=initial["feature"].float(); initial_candidate=initial_post.argmax(dim=1)
    top1=posterior.argmax(dim=1); top1_prob=posterior.max(dim=1).values; margin=posterior_margin(posterior); entropy=posterior_entropy(posterior); displacement=cosine_feature_displacement(feature,initial_feat)
    seed_member=torch.tensor([int(v.item()) in seed_by for v in ids],dtype=torch.bool); seed_pseudo=torch.tensor([seed_by.get(int(v.item()),-1) for v in ids],dtype=torch.long)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,sample_id=ids.numpy().astype(np.int64),posterior=posterior.numpy().astype(np.float32),feature=feature.numpy().astype(np.float32),initial_candidate=initial_candidate.numpy().astype(np.int64),seed_member=seed_member.numpy(),seed_pseudo_label=seed_pseudo.numpy().astype(np.int64),epoch=np.asarray([epoch],dtype=np.int64))
    rows=[]
    for i,sid in enumerate(ids.tolist()):
        rows.append({"sample_id":int(sid),"arm":arm,"checkpoint_epoch":int(epoch),"model_role":role,"pseudo_labeled":bool(seed_member[i].item()),"pseudo_label":int(seed_pseudo[i].item()),"initial_candidate":int(initial_candidate[i].item()),"current_top1":int(top1[i].item()),"top1_probability":float(top1_prob[i].item()),"margin":float(margin[i].item()),"entropy":float(entropy[i].item()),"feature_cosine_displacement_from_stage1":float(displacement[i].item()),"oracle_arm":bool(oracle_arm)})
    return rows


def _save_source_scan(path: Path, scan: dict, *, epoch: int, arm: str, role: str) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,sample_id=scan["sample_id"].numpy().astype(np.int64),posterior=scan["posterior"].numpy().astype(np.float32),feature=scan["feature"].numpy().astype(np.float32),true_label=scan["label"].numpy().astype(np.int64),epoch=np.asarray([epoch],dtype=np.int64),arm=np.asarray([arm]),model_role=np.asarray([role]))


def _train_arm(*, arm: str, model, source_loader, target_loader, initial_target: dict, target_train_loader, target_val_loader, source_val_loader, seed_by: dict[int,int], output: Path, device: torch.device, epochs: int, steps_per_epoch: int, checkpoints: tuple[int,...], lr: float, weight_decay: float, lambda_target: float, ema_decay: float, amp_enabled: bool, seed: int, oracle_arm: bool) -> tuple[list[dict],dict]:
    _set_seed(seed); policy=configure_13b_semantic_student(model); named=dict(model.named_parameters()); params=[named[n] for n in policy.trainable_parameter_names]
    optimizer=torch.optim.Adam(params,lr=lr,weight_decay=weight_decay); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs*steps_per_epoch,eta_min=0.0); teacher=Stage2EMATeacher.from_student(model,policy,decay=ema_decay); scaler=torch.amp.GradScaler("cuda",enabled=(amp_enabled and device.type=="cuda"))
    rows=[]; source_stream=hashlib.sha256(); target_steps=0
    trajectory_root=output/("oracle_control/trajectory" if oracle_arm else "target_trajectory_unlabeled")/arm
    target_val_root=output/("oracle_control/target_val" if oracle_arm else "target_val_unlabeled")/arm

    def checkpoint(epoch:int)->None:
        s=_scan_model(model,target_train_loader,device); t=_scan_model(teacher.model(),target_train_loader,device)
        rows.extend(_save_scan(trajectory_root/f"epoch_{epoch:03d}_student.npz",scan=s,initial=initial_target,seed_by=seed_by,epoch=epoch,arm=arm,role="student",oracle_arm=oracle_arm)); rows.extend(_save_scan(trajectory_root/f"epoch_{epoch:03d}_teacher.npz",scan=t,initial=initial_target,seed_by=seed_by,epoch=epoch,arm=arm,role="teacher",oracle_arm=oracle_arm))
        sv=_scan_model(model,target_val_loader,device); tv=_scan_model(teacher.model(),target_val_loader,device); target_val_root.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target_val_root/f"epoch_{epoch:03d}_student.npz",sample_id=sv["sample_id"].numpy(),posterior=sv["posterior"].numpy(),feature=sv["feature"].numpy()); np.savez_compressed(target_val_root/f"epoch_{epoch:03d}_teacher.npz",sample_id=tv["sample_id"].numpy(),posterior=tv["posterior"].numpy(),feature=tv["feature"].numpy())
        ss=_scan_model(model,source_val_loader,device,allow_labels=True); st=_scan_model(teacher.model(),source_val_loader,device,allow_labels=True); _save_source_scan(output/"source_val_trajectory"/arm/f"epoch_{epoch:03d}_student.npz",ss,epoch=epoch,arm=arm,role="student"); _save_source_scan(output/"source_val_trajectory"/arm/f"epoch_{epoch:03d}_teacher.npz",st,epoch=epoch,arm=arm,role="teacher")
        ck=output/"checkpoints"/arm/f"epoch_{epoch:03d}.pt"; ck.parent.mkdir(parents=True,exist_ok=True); torch.save({"protocol":PROTOCOL,"arm":arm,"epoch":epoch,"student_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"teacher_state_dict":{k:v.detach().cpu() for k,v in teacher.model().state_dict().items()},"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"trainable_parameter_names":policy.trainable_parameter_names,"oracle_arm":oracle_arm},ck)
        print(f"SEED13B_CHECKPOINT|arm={arm}|epoch={epoch}|oracle_arm={str(oracle_arm).lower()}",flush=True)

    checkpoint(0); source_iter=iter(source_loader); target_iter=None if target_loader is None else iter(target_loader)
    for epoch in range(1,epochs+1):
        model.train(); meters={"source":0.0,"target":0.0,"total":0.0}; steps=0
        for _ in range(steps_per_epoch):
            try: source_raw=next(source_iter)
            except StopIteration: source_iter=iter(source_loader); source_raw=next(source_iter)
            source_stream.update(source_raw["parcel_index"].detach().cpu().numpy().astype(np.int64).tobytes()); source=_move(source_raw,device); optimizer.zero_grad(set_to_none=True); amp_on=bool(amp_enabled and device.type=="cuda")
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp_on):
                source_out=_forward(model,source); source_loss=F.cross_entropy(source_out.logits,source["label"].long()); target_loss=source_loss.sum()*0.0
                if target_iter is not None:
                    try: target_raw=next(target_iter)
                    except StopIteration: target_iter=iter(target_loader); target_raw=next(target_iter)
                    if FORBIDDEN_TARGET_LABEL_KEYS.intersection(target_raw.keys()): raise RuntimeError("target truth leaked through target supervision Dataset")
                    target=_move(target_raw,device); target_out=_forward(model,target); target_loss=class_balanced_seed_cross_entropy(target_out.logits,target["pseudo_label"].long()); target_steps+=1
                total=source_loss+(float(lambda_target)*target_loss if target_iter is not None else 0.0)
            scaler.scale(total).backward(); scaler.step(optimizer); scaler.update(); scheduler.step(); teacher.update_after_optimizer_step(model); meters["source"]+=float(source_loss.detach()); meters["target"]+=float(target_loss.detach()); meters["total"]+=float(total.detach()); steps+=1
        print(f"SEED13B_TRAIN_EPOCH|arm={arm}|epoch={epoch}/{epochs}|steps={steps}|target_steps={target_steps}|source_loss={meters['source']/steps:.6f}|target_loss={meters['target']/steps:.6f}|total_loss={meters['total']/steps:.6f}",flush=True)
        if epoch in checkpoints: checkpoint(epoch)
    return rows,{"source_stream_hash":source_stream.hexdigest(),"target_optimization_steps":int(target_steps),"target_supervision_count":len(seed_by),"target_supervision_classes":sorted(set(seed_by.values())),"target_loader_active":target_loader is not None,"final_student_hash":_model_state_hash(model),"final_teacher_hash":_model_state_hash(teacher.model()),"trainable_parameter_names":list(policy.trainable_parameter_names),"oracle_arm":oracle_arm}


def _write_readme(path:Path,config:dict)->None:
    path.write_text(f"""# 实验 13B：多种初始伪标签策略的短程适应比较

## 定位

本实验不以 Stage-1 静态 pseudo-label precision 提前淘汰 bootstrap。五种预定义的 label-free 初始监督策略全部经过完全相同的短程参数更新，价值由训练后的 target model trajectory、关键错误流、健康类别保持与 source semantic retention 共同判断。

固定任务：AT1 → DK1，seed=1，fold=0。Stage-1 checkpoint：`{config['stage1_checkpoint']}`。shared Domain Phase、class-conditioned Phase、registration refresh、pseudo-label refresh/relabel、Teacher pseudo-label feedback、MMD/DANN/CORAL/contrastive/prototype alignment 全部关闭。

## 七个配置

- `CTRL_SOURCE`：仅 source true-label CE。
- `PL_TIMEMATCH_CONF`：raw classifier top-1，confidence > `{config['bootstrap_parameters']['pseudo_threshold']}`；这里只抽取 TimeMatch pseudo-label confidence baseline，不启用 temporal shift。
- `PL_DAPL_BOOT`：同一 confidence gate + source true-label frozen LTAE diagonal-Gaussian conformity；source 5-fold cross-fit 标定 percentile ≤ `{config['bootstrap_parameters']['conformity_percentile_threshold']}`。
- `PL_IPL_BOOT`：raw classifier label 与 source cosine prototype label 一致，并要求 target KNN-{config['bootstrap_parameters']['bootstrap_knn_k']} raw-candidate support ≥ `{config['bootstrap_parameters']['ipl_support_threshold']}`。
- `PL_TFDA_NN`：target KNN-{config['bootstrap_parameters']['bootstrap_knn_k']} 邻居 posterior 平均后 argmax 直接产生 bootstrap pseudo-label；所有 target-train 样本进入 T0。该标签允许不同于单样本 raw top-1。
- `PL_CONF_GEOM`：TimeMatch confidence pool 上，若 T 或 S registered-error 的 source-class percentile ≥ `{config['bootstrap_parameters']['geometry_conflict_percentile']}`，仅 veto、不改类别。
- `CTRL_ORACLE`：target-train true labels，**oracle supervised diagnostic upper bound**；只有在五个 label-free manifest 已全部写盘后才构造，不参与任何 bootstrap 策略选择。

TransPL 不进入第一轮，因为它要求额外 VQ representation 与 transition model，会同时改变表示和 pseudo-label mechanism，破坏本实验只比较 bootstrap acquisition 的归因边界。

## 固定训练器

全部 7 组从相同 Stage-1 checkpoint 初始化。Student 的 PSE + raw LTAE/TimeEncoder + Classifier 全部更新；geometry side 保持独立冻结 Stage-1 状态。优化器 Adam，LR={config['lr']}，weight_decay={config['weight_decay']}，lambda_t={config['lambda_target']}，EMA={config['ema_decay']}。统一 `{config['epochs']} × {config['steps_per_epoch']}` update budget，checkpoint={config['checkpoint_epochs']}。target pseudo-label dataset 大小不决定 update 数：只要 manifest 可形成 batch，每个训练 step 都接受一个 class-balanced target batch。

所有 T0 只在 r=0 生成一次，随后固定；不 refresh、不 add/remove、不 relabel。EMA Teacher 只观察。

## 标签边界与输出

`01_bootstrap_manifests/*.csv`、`01_bootstrap_strategy_manifest.json`、`02_stage1_target_reference_unlabeled.npz` 与 `03_target_trajectory_unlabeled.csv` 不含 target truth。五种 label-free bootstrap 均只读取 Stage-1 frozen posterior/LTAE feature、source true labels，以及 13A-2 已落盘的 label-free KNN/geometry observables。

`oracle_control/` 明确属于 supervised diagnostic upper bound；`oracle/` 是所有训练结束后的后验 evaluator。target true label 不参与五种 bootstrap 的 threshold、selection、loss、checkpoint selection 或 early stopping。

本实验没有 best epoch。所有结论必须基于预先固定的 trajectory，而不能事后挑 target F1 最好的 checkpoint。
""",encoding="utf-8")


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--calibration-checkpoint",type=Path,required=True); p.add_argument("--model-checkpoint",type=Path,required=True); p.add_argument("--structure-diagnostic-dir",type=Path,required=True); p.add_argument("--data-root",type=Path,default=None); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--fold",type=int,default=0); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--num-workers",type=int,default=4); p.add_argument("--epochs",type=int,default=5); p.add_argument("--steps-per-epoch",type=int,default=500); p.add_argument("--checkpoint-epochs",default="0,1,3,5"); p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--lambda-target",type=float,default=1.0); p.add_argument("--ema-decay",type=float,default=0.99); p.add_argument("--pseudo-threshold",type=float,default=0.9); p.add_argument("--conformity-percentile-threshold",type=float,default=0.95); p.add_argument("--conformity-crossfit-folds",type=int,default=5); p.add_argument("--bootstrap-knn-k",type=int,default=20); p.add_argument("--ipl-support-threshold",type=float,default=0.5); p.add_argument("--geometry-conflict-percentile",type=float,default=0.95); p.add_argument("--amp",action="store_true"); return p.parse_args()


def main():
    args=parse_args(); finite_hyperparameters(lr=args.lr,weight_decay=args.weight_decay,lambda_target=args.lambda_target,ema_decay=args.ema_decay); checkpoints=validate_checkpoint_epochs(args.epochs,tuple(int(v) for v in args.checkpoint_epochs.split(",")))
    calibration=torch.load(args.calibration_checkpoint.resolve(),map_location="cpu",weights_only=False); runtime=dict(calibration.get("runtime_config") or {})
    if not runtime: raise ValueError("calibration checkpoint is missing runtime_config")
    runtime["data_root"]=str(args.data_root.resolve()) if args.data_root else str(runtime["data_root"]); actual=(str(runtime["source"]),str(runtime["target"]),int(runtime["seed"]),int(args.fold))
    if actual!=EXPECTED_RUNTIME: raise ValueError(f"first 13B run is frozen to AT1->DK1 seed=1 fold=0; got {actual}")
    classes=[str(v) for v in runtime["classes"]]; device=torch.device(args.device); seed=int(runtime["seed"]); model_checkpoint=torch.load(args.model_checkpoint.resolve(),map_location="cpu",weights_only=False)
    anchor=phasevis._build_model(runtime,calibration,device,model_checkpoint=model_checkpoint); anchor.eval(); [p.requires_grad_(False) for p in anchor.parameters()]
    geometry_anchor=phasevis._build_model(runtime,calibration,device,model_checkpoint=model_checkpoint); geometry_anchor.eval(); [p.requires_grad_(False) for p in geometry_anchor.parameters()]

    source_all=phasevis._eligible_parcels(runtime["data_root"],runtime["source"],classes,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year"))); target_all=phasevis._eligible_parcels(runtime["data_root"],runtime["target"],classes,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    splits=phasevis._reconstruct_fold_splits(source_all,target_all,source=runtime["source"],target=runtime["target"],seed=seed,val_ratio=float(runtime.get("val_ratio",0.1)),test_ratio=float(runtime.get("test_ratio",0.2)),fold=args.fold)
    target_train=sorted(splits[runtime["target"]]["train"]); target_val=sorted(splits[runtime["target"]]["val"]); source_val=sorted(splits[runtime["source"]]["val"])
    target_train_loader=_scan_loader(runtime["data_root"],runtime["target"],classes,target_train,runtime,batch_size=args.batch_size,num_workers=args.num_workers,strip_label=True); target_val_loader=_scan_loader(runtime["data_root"],runtime["target"],classes,target_val,runtime,batch_size=args.batch_size,num_workers=args.num_workers,strip_label=True); source_val_loader=_scan_loader(runtime["data_root"],runtime["source"],classes,source_val,runtime,batch_size=args.batch_size,num_workers=args.num_workers,strip_label=False)
    initial=_scan_model(anchor,target_train_loader,device); structure=_load_structure_inputs(args.structure_diagnostic_dir,initial,source=str(runtime["source"]),target=str(runtime["target"]),seed=seed,fold=args.fold)

    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/"02_stage1_target_reference_unlabeled.npz",sample_id=initial["sample_id"].numpy(),posterior=initial["posterior"].numpy(),feature=initial["feature"].numpy(),initial_candidate=initial["posterior"].argmax(dim=1).numpy())
    anchor_source_val=_scan_model(anchor,source_val_loader,device,allow_labels=True); _save_source_scan(output/"02b_stage1_source_val_reference.npz",anchor_source_val,epoch=0,arm="STAGE1",role="source_anchor"); anchor_target_val=_scan_model(anchor,target_val_loader,device); np.savez_compressed(output/"02c_stage1_target_val_reference_unlabeled.npz",sample_id=anchor_target_val["sample_id"].numpy(),posterior=anchor_target_val["posterior"].numpy(),feature=anchor_target_val["feature"].numpy())

    records_by_arm,bootstrap_diag=_generate_label_free_bootstraps(initial,structure,args=args,num_classes=len(classes)); manifest_meta={}
    manifest_dir=output/"01_bootstrap_manifests"
    for arm in LABEL_FREE_BOOTSTRAP_ARMS:
        records=records_by_arm[arm]; path=manifest_dir/f"{arm}.csv"; _write_csv(path,_records_to_rows(records,arm)); manifest_meta[arm]={"path":str(path),"sha256":seed_manifest_fingerprint(records),"count":len(records),"classes":sorted(set(r.pseudo_label for r in records)),"target_truth_used":False}
        print(f"SEED13B_BOOTSTRAP_FROZEN|arm={arm}|n={len(records)}|classes={','.join(map(str,manifest_meta[arm]['classes']))}|target_truth_used=false",flush=True)
    np.savez_compressed(output/"01_bootstrap_diagnostics_unlabeled.npz",sample_id=initial["sample_id"].numpy().astype(np.int64),**bootstrap_diag)
    bootstrap_parameters={"pseudo_threshold":args.pseudo_threshold,"conformity_percentile_threshold":args.conformity_percentile_threshold,"conformity_crossfit_folds":args.conformity_crossfit_folds,"bootstrap_knn_k":args.bootstrap_knn_k,"ipl_support_threshold":args.ipl_support_threshold,"geometry_conflict_percentile":args.geometry_conflict_percentile}
    _json_dump(output/"01_bootstrap_strategy_manifest.json",{"protocol":PROTOCOL,"label_free_strategies":list(LABEL_FREE_BOOTSTRAP_ARMS),"parameters":bootstrap_parameters,"manifests":manifest_meta,"structure_diagnostic_dir":str(structure["root"]),"target_truth_used":False,"transpl_included":False})

    oracle_manifest=output/"oracle_control/CTRL_ORACLE_target_truth_manifest.csv"
    config={"protocol":PROTOCOL,"source":runtime["source"],"target":runtime["target"],"seed":seed,"fold":args.fold,"classes":classes,"stage1_checkpoint":str(args.model_checkpoint.resolve()),"calibration_checkpoint":str(args.calibration_checkpoint.resolve()),"structure_diagnostic_dir":str(structure["root"]),"arms":list(BOOTSTRAP_ARMS),"label_free_bootstrap_arms":list(LABEL_FREE_BOOTSTRAP_ARMS),"bootstrap_parameters":bootstrap_parameters,"bootstrap_manifests":manifest_meta,"ctrl_oracle_manifest":str(oracle_manifest),"epochs":args.epochs,"steps_per_epoch":args.steps_per_epoch,"checkpoint_epochs":list(checkpoints),"optimizer":"Adam","lr":args.lr,"weight_decay":args.weight_decay,"lambda_target":args.lambda_target,"ema_decay":args.ema_decay,"scheduler":"CosineAnnealingLR_stepwise","student_trainable":"PSE + raw LTAE/time encoder + classifier","geometry_state":"separate frozen Stage-1 copy; no registration/Phase update","teacher_role":"EMA observation only","pseudo_label_refresh":False,"relabel_after_r0":False,"alignment_loss":False,"best_epoch_selection":False,"target_test_used":False,"transpl_included":False,"oracle_control_constructed_after_label_free_training":True,"source_anchor_state_hash":_model_state_hash(anchor),"geometry_anchor_state_hash":_model_state_hash(geometry_anchor)}
    _json_dump(output/"00_stage2_bootstrap_config.json",config); _write_readme(output/"README_中文说明.md",config)

    # First train CTRL_SOURCE + all five label-free bootstrap arms.  Target
    # truth has not been loaded anywhere in this process at this point.
    records_by_arm["CTRL_SOURCE"]=tuple(); label_free_rows=[]; arm_meta={}
    for arm in ("CTRL_SOURCE",)+LABEL_FREE_BOOTSTRAP_ARMS:
        _set_seed(seed); student=phasevis._build_model(runtime,calibration,device,model_checkpoint=model_checkpoint); source_loader=_train_source_loader(runtime,splits,seed=seed,num_workers=args.num_workers,batch_size=args.batch_size); records=records_by_arm[arm]; target_loader=None if arm=="CTRL_SOURCE" else _target_supervision_loader(runtime,records,seed=seed,num_workers=args.num_workers,batch_size=args.batch_size); seed_by={r.sample_id:r.pseudo_label for r in records}
        rows,meta=_train_arm(arm=arm,model=student,source_loader=source_loader,target_loader=target_loader,initial_target=initial,target_train_loader=target_train_loader,target_val_loader=target_val_loader,source_val_loader=source_val_loader,seed_by=seed_by,output=output,device=device,epochs=args.epochs,steps_per_epoch=args.steps_per_epoch,checkpoints=checkpoints,lr=args.lr,weight_decay=args.weight_decay,lambda_target=args.lambda_target,ema_decay=args.ema_decay,amp_enabled=args.amp,seed=seed,oracle_arm=False); arm_meta[arm]=meta; label_free_rows.extend(rows)
    _write_csv(output/"03_target_trajectory_unlabeled.csv",label_free_rows)
    print("SEED13B_LABEL_FREE_ARMS_COMPLETE|arms=6|target_truth_loaded=false",flush=True)

    # Only now construct and train the supervised diagnostic upper bound.
    target_truth_loader=_scan_loader(runtime["data_root"],runtime["target"],classes,target_train,runtime,batch_size=args.batch_size,num_workers=args.num_workers,strip_label=False); target_truth=_scan_model(anchor,target_truth_loader,device,allow_labels=True)
    if not torch.equal(target_truth["sample_id"],initial["sample_id"]): raise ValueError("oracle target-train ordering differs from frozen reference")
    oracle_records=tuple(FixedSeedRecord(int(sid),int(label),"oracle_true_label_diagnostic_upper_bound") for sid,label in zip(target_truth["sample_id"].tolist(),target_truth["label"].tolist())); records_by_arm["CTRL_ORACLE"]=oracle_records
    _write_csv(oracle_manifest,[{"sample_id":r.sample_id,"pseudo_label":r.pseudo_label,"selection_evidence":r.selection_evidence,"oracle_use":"supervised diagnostic upper bound"} for r in oracle_records])
    _set_seed(seed); student=phasevis._build_model(runtime,calibration,device,model_checkpoint=model_checkpoint); source_loader=_train_source_loader(runtime,splits,seed=seed,num_workers=args.num_workers,batch_size=args.batch_size); target_loader=_target_supervision_loader(runtime,oracle_records,seed=seed,num_workers=args.num_workers,batch_size=args.batch_size); seed_by={r.sample_id:r.pseudo_label for r in oracle_records}
    oracle_rows,oracle_meta=_train_arm(arm="CTRL_ORACLE",model=student,source_loader=source_loader,target_loader=target_loader,initial_target=initial,target_train_loader=target_train_loader,target_val_loader=target_val_loader,source_val_loader=source_val_loader,seed_by=seed_by,output=output,device=device,epochs=args.epochs,steps_per_epoch=args.steps_per_epoch,checkpoints=checkpoints,lr=args.lr,weight_decay=args.weight_decay,lambda_target=args.lambda_target,ema_decay=args.ema_decay,amp_enabled=args.amp,seed=seed,oracle_arm=True); arm_meta["CTRL_ORACLE"]=oracle_meta; _write_csv(output/"oracle_control/CTRL_ORACLE_trajectory_index.csv",oracle_rows)

    source_hashes={meta["source_stream_hash"] for meta in arm_meta.values()}
    if len(source_hashes)!=1: raise RuntimeError("13B arms did not receive the same source parcel stream")
    config["arm_metadata"]=arm_meta; config["source_stream_identical_between_arms"]=True; _json_dump(output/"00_stage2_bootstrap_config.json",config)
    print("SEED13B_TRAINING_COMPLETE|groups=7|source_stream_identical=true|best_epoch_selection=false|oracle_evaluation_ready=true",flush=True)


if __name__=="__main__": main()
