#!/usr/bin/env python3
"""Experiment 14: shared Domain Phase iterative Stage-2 loop.

Main training is strictly label-free on target data.  One identity warm-up is
run once and forked into NO_PHASE, STATIC_DOMAIN_PHASE and
ITERATIVE_DOMAIN_PHASE.  Target truth is reserved for the separate evaluator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

import diagnose_sample_level_phase_validity as samplediag
import visualize_stage2_phase_alignment as phasevis
from dataset import BalancedBatchSampler, GroupByShapesBatchSampler, PixelSetData, worker_init_fn
from methods.structure_da.domain_phase_loop import (
    PHASE_ARMS,
    SharedDomainPhaseEstimate,
    actual_phase_for_arm,
    build_shared_domain_phase,
    class_center_distance_rows,
    identity_phase,
    map_source_batch_positions,
    map_target_batch_positions,
    phase_distance_value,
    validate_phase,
)
from methods.structure_da.ema_teacher import Stage2EMATeacher
from methods.structure_da.registration_geometry import evaluate_registration_geometry
from methods.structure_da.sample_phase_diagnostic import (
    TOnlyPhaseRegistration,
    TRegistrationGeometryCache,
    solve_t_only_registrations,
)
from methods.structure_da.seed_warmup_diagnostic import configure_13b_semantic_student
from methods.structure_da.stage2_objective import Stage2Objective, Stage2ObjectiveConfig
from methods.structure_da.stage2_trainer import DeviceBatchLoader, build_stage2_registration_extractor
from transforms import Identity, Normalize, RandomSamplePixels, ToTensor

PROTOCOL = "14_shared_domain_phase_iterative_loop_v1"
EXPECTED_RUNTIME = ("austria/33UVP/2017", "denmark/32VNH/2017", 1, 0)
PAIR_CACHE_SCHEMA = "14_phase_registration_pair_cache_v1"
RAW_GEOMETRY_SCHEMA = "13A_raw_candidate_geometry_v1"
FORBIDDEN_TARGET_LABEL_KEYS = {"label", "true_label", "candidate_correct", "confusion_flow"}


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


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp); tmp.replace(path)


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _state_hash(model) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode("utf-8")); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class LabelFreeDataset(Dataset):
    def __init__(self, base: Dataset) -> None: self.base = base
    def __len__(self) -> int: return len(self.base)
    def __getitem__(self, index: int):
        item = dict(self.base[index]); item.pop("label", None); return item


def _move(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in batch.items()}


def _source_train_loader(runtime: dict, splits: dict, *, seed: int, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose([RandomSamplePixels(int(runtime.get("num_pixels", 64))), Normalize(), ToTensor()])
    ds = PixelSetData(
        data_root=str(runtime["data_root"]), dataset_name=str(runtime["source"]), classes=list(runtime["classes"]),
        transform=transform, indices=splits[str(runtime["source"])]["train"], with_extra=False,
        closed_set=bool(runtime.get("closed_set", True)),
        combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter", False)),
        time_coordinate_mode=str(runtime.get("time_coordinate_mode", "canonical_day_of_year")),
    )
    return DataLoader(
        ds, batch_sampler=BalancedBatchSampler(ds.get_labels(), batch_size, seed=seed),
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )


def _target_train_loader(runtime: dict, parcels: Iterable[int], *, seed: int, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose([RandomSamplePixels(int(runtime.get("num_pixels", 64))), Normalize(), ToTensor()])
    base = PixelSetData(
        data_root=str(runtime["data_root"]), dataset_name=str(runtime["target"]), classes=list(runtime["classes"]),
        transform=transform, indices=set(map(int, parcels)), with_extra=False,
        closed_set=bool(runtime.get("closed_set", True)),
        combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter", False)),
        time_coordinate_mode=str(runtime.get("time_coordinate_mode", "canonical_day_of_year")),
    )
    ds = LabelFreeDataset(base)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed + 1777),
    )


def _scan_loader(runtime: dict, domain: str, parcels: Iterable[int], *, batch_size: int, num_workers: int, strip_label: bool) -> DataLoader:
    base = PixelSetData(
        data_root=str(runtime["data_root"]), dataset_name=domain, classes=list(runtime["classes"]),
        transform=transforms.Compose([Identity(), Normalize(), ToTensor()]), indices=set(map(int, parcels)),
        with_extra=False, closed_set=bool(runtime.get("closed_set", True)),
        combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter", False)),
        time_coordinate_mode=str(runtime.get("time_coordinate_mode", "canonical_day_of_year")),
    )
    ds = LabelFreeDataset(base) if strip_label else base
    return DataLoader(
        ds, batch_sampler=GroupByShapesBatchSampler(base, batch_size, by_time=True, by_pixel_dim=True),
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn,
    )


def _raw_forward(model, batch: dict):
    return model(batch["pixels"], batch["valid_pixels"], batch["positions"], batch.get("extra"), return_geometry=False)


def _forward_source_phase(model, batch: dict, gamma: Tensor):
    backbone = model.forward_backbone(
        batch["pixels"], batch["valid_pixels"], batch["positions"], batch.get("extra"),
        time_mask=batch.get("time_mask"), compute_decomposition=False,
    )
    mapped = map_source_batch_positions(backbone.normalized_positions, backbone.time_mask, gamma)
    return model.forward_from_backbone(
        backbone, batch["positions"], batch.get("extra"), temporal_positions_override=mapped, return_geometry=False,
    )


def _forward_target_teacher_phase(model, batch: dict, gamma: Tensor):
    backbone = model.forward_backbone(
        batch["pixels"], batch["valid_pixels"], batch["positions"], batch.get("extra"),
        time_mask=batch.get("time_mask"), compute_decomposition=False,
    )
    mapped = map_target_batch_positions(backbone.normalized_positions, backbone.time_mask, gamma)
    return model.forward_from_backbone(
        backbone, batch["positions"], batch.get("extra"), temporal_positions_override=mapped, return_geometry=False,
    )


@torch.no_grad()
def _scan_target_teacher(model, loader, device: torch.device, gamma: Tensor, *, threshold: float, previous_top1: Mapping[int, int] | None = None) -> dict:
    model.eval(); ids=[]; post=[]; feats=[]
    for raw in loader:
        if FORBIDDEN_TARGET_LABEL_KEYS.intersection(raw.keys()):
            raise RuntimeError("target truth leaked into experiment-14 main scan")
        batch=_move(raw,device); out=_forward_target_teacher_phase(model,batch,gamma)
        ids.append(batch["parcel_index"].detach().cpu().long())
        post.append(torch.softmax(out.logits.detach().float(),dim=1).cpu())
        feats.append(out.fused_repr.detach().float().cpu())
    sample_id=torch.cat(ids); posterior=torch.cat(post); feature=torch.cat(feats); order=torch.argsort(sample_id)
    sample_id=sample_id[order]; posterior=posterior[order]; feature=feature[order]
    confidence,pred=posterior.max(dim=1); mask=confidence>float(threshold)
    changed=torch.zeros_like(mask)
    if previous_top1 is not None:
        changed=torch.tensor([previous_top1.get(int(s),int(p))!=int(p) for s,p in zip(sample_id.tolist(),pred.tolist())],dtype=torch.bool)
    return {"sample_id":sample_id,"posterior":posterior,"feature":feature,"pred":pred,"confidence":confidence,"mask":mask,"changed":changed}


@torch.no_grad()
def _scan_raw(model, loader, device: torch.device, *, allow_labels: bool) -> dict:
    model.eval(); ids=[]; post=[]; labels=[]
    for raw in loader:
        if not allow_labels and FORBIDDEN_TARGET_LABEL_KEYS.intersection(raw.keys()):
            raise RuntimeError("target truth leaked into label-free scan")
        batch=_move(raw,device); out=_raw_forward(model,batch)
        ids.append(batch["parcel_index"].detach().cpu().long()); post.append(torch.softmax(out.logits.detach().float(),dim=1).cpu())
        if allow_labels: labels.append(batch["label"].detach().cpu().long())
    sample_id=torch.cat(ids); posterior=torch.cat(post); order=torch.argsort(sample_id)
    result={"sample_id":sample_id[order],"posterior":posterior[order]}
    if allow_labels: result["label"]=torch.cat(labels)[order]
    return result


def _classification_metrics(posterior: Tensor, labels: Tensor, num_classes: int) -> dict:
    pred=posterior.argmax(dim=1); labels=labels.long()
    accuracy=float((pred==labels).float().mean().item())
    f1=[]; weights=[]
    for c in range(num_classes):
        tp=int(((pred==c)&(labels==c)).sum()); fp=int(((pred==c)&(labels!=c)).sum()); fn=int(((pred!=c)&(labels==c)).sum()); n=int((labels==c).sum())
        precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
        score=2*precision*recall/(precision+recall) if precision+recall else 0.0
        f1.append(score); weights.append(n)
    macro=float(np.mean(f1)); weighted=float(np.average(f1,weights=weights)) if sum(weights) else 0.0
    return {"accuracy":accuracy,"macro_f1":macro,"weighted_f1":weighted}


def _save_refresh(root: Path, *, arm: str, refresh_id: int, phase_epoch: int, scan: dict, previous_top1: Mapping[int,int] | None) -> dict[int,int]:
    root.mkdir(parents=True,exist_ok=True)
    ids=scan["sample_id"].numpy().astype(np.int64); post=scan["posterior"].numpy().astype(np.float32)
    pred=scan["pred"].numpy().astype(np.int64); conf=scan["confidence"].numpy().astype(np.float32); mask=scan["mask"].numpy().astype(bool); changed=scan["changed"].numpy().astype(bool)
    np.savez_compressed(root/f"refresh_{refresh_id:03d}.npz",sample_id=ids,posterior=post,predicted_class=pred,confidence=conf,mask=mask,changed_top1=changed,phase_epoch=np.asarray([phase_epoch],dtype=np.int64))
    rows=[{"sample_id":int(ids[i]),"arm":arm,"refresh_id":int(refresh_id),"phase_epoch":int(phase_epoch),"predicted_class":int(pred[i]),"confidence":float(conf[i]),"mask":bool(mask[i]),"changed_top1":bool(changed[i])} for i in range(len(ids))]
    _write_csv(root/f"refresh_{refresh_id:03d}.csv",rows)
    counts=np.bincount(pred[mask],minlength=post.shape[1]) if np.any(mask) else np.zeros(post.shape[1],dtype=np.int64)
    print(f"PHASE14_PSEUDO_REFRESH|arm={arm}|refresh={refresh_id}|accepted={int(mask.sum())}/{len(mask)}|class_counts={','.join(map(str,counts.tolist()))}",flush=True)
    return {int(s):int(p) for s,p in zip(ids.tolist(),pred.tolist())}


def _minimal_record_payload(record: TOnlyPhaseRegistration) -> dict:
    return {
        "sample_id":int(record.sample_id),"class_id":int(record.class_id),
        "gamma":None if record.gamma is None else record.gamma.detach().cpu().double(),
        "numerically_valid":bool(record.numerically_valid),"solver_error":record.solver_error,
        "target_trend_valid":bool(record.target_trend_valid),"pre_common_support_t":float(record.pre_common_support_t),
        "t_only_legal":bool(record.t_only_legal),"reject_reasons":tuple(record.reject_reasons),
    }


def _load_pair_cache(path: Path, *, source: str, target: str, seed: int, fold: int, model_checkpoint: Path) -> dict[tuple[int,int],dict]:
    if not path.is_file(): return {}
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("schema")!=PAIR_CACHE_SCHEMA: raise ValueError("unsupported experiment-14 pair cache schema")
    for key,expected in (("source",source),("target",target),("seed",seed),("fold",fold)):
        if str(payload.get(key))!=str(expected): raise ValueError(f"experiment-14 pair cache {key} mismatch")
    if Path(str(payload.get("model_checkpoint"))).resolve()!=model_checkpoint.resolve(): raise ValueError("experiment-14 pair cache Stage-1 checkpoint mismatch")
    records={}
    for row in payload.get("records",()):
        item=dict(row); gamma=item.get("gamma")
        if isinstance(gamma,Tensor): item["gamma"]=gamma.detach().cpu().double()
        key=(int(item["sample_id"]),int(item["class_id"]))
        if key in records: raise ValueError("duplicate sample×class registration pair")
        records[key]=item
    print(f"PHASE14_PAIR_CACHE_RESUME|pairs={len(records)}|path={path}",flush=True)
    return records


def _save_pair_cache(path: Path, records: Mapping[tuple[int,int],dict], *, source: str, target: str, seed: int, fold: int, model_checkpoint: Path) -> None:
    ordered=[records[key] for key in sorted(records)]
    _atomic_torch_save({"schema":PAIR_CACHE_SCHEMA,"source":source,"target":target,"seed":int(seed),"fold":int(fold),"model_checkpoint":str(model_checkpoint.resolve()),"contains_target_true_labels":False,"records":ordered},path)


def _prefill_raw_candidate_cache(pair_records: dict[tuple[int,int],dict], raw_path: Path, *, initial_scan: dict, model_checkpoint: Path, source: str, target: str, seed: int, fold: int) -> int:
    if not raw_path.is_file(): return 0
    payload=torch.load(raw_path,map_location="cpu",weights_only=False)
    if payload.get("schema")!=RAW_GEOMETRY_SCHEMA: raise ValueError("unsupported 13A raw-candidate cache schema")
    if bool(payload.get("contains_target_true_labels",False)): raise ValueError("13A raw-candidate cache reports target truth contamination")
    for key,expected in (("source",source),("target",target),("seed",seed),("fold",fold)):
        if str(payload.get(key))!=str(expected): raise ValueError(f"13A raw-candidate cache {key} mismatch")
    if Path(str(payload.get("model_checkpoint"))).resolve()!=model_checkpoint.resolve(): raise ValueError("13A raw-candidate cache Stage-1 checkpoint mismatch")
    raw_by_id={int(s):int(c) for s,c in zip(initial_scan["sample_id"].tolist(),initial_scan["posterior"].argmax(dim=1).tolist())}
    added=0
    for row in payload.get("records",()):
        sid=int(row["sample_id"]); cid=int(row["raw_pred"])
        if sid not in raw_by_id or raw_by_id[sid]!=cid: raise ValueError("13A raw-candidate cache assignment differs from current Stage-1 scan")
        key=(sid,cid)
        if key in pair_records: continue
        gamma=row.get("gamma")
        pair_records[key]={"sample_id":sid,"class_id":cid,"gamma":None if gamma is None else gamma.detach().cpu().double(),"numerically_valid":bool(row.get("registration_numerically_valid",False)),"solver_error":row.get("solver_error"),"target_trend_valid":bool(row.get("target_trend_valid",False)),"pre_common_support_t":float(row.get("T_pre_common_support",float("nan"))),"t_only_legal":False,"reject_reasons":tuple(),"source":"13A_raw_candidate_geometry"}
        added+=1
    print(f"PHASE14_PAIR_CACHE_PREFILL|from_13A={added}",flush=True); return added


@torch.no_grad()
def _solve_missing_pairs(*, desired: Mapping[int,int], pair_records: dict[tuple[int,int],dict], runtime: dict, geometry_model, source_reg_bank, reg_extractor, scan_config, device: torch.device, batch_size: int, num_workers: int, workers: int, dp_chunk_size: int, cache_path: Path, source: str, target: str, seed: int, fold: int, model_checkpoint: Path) -> None:
    missing={int(sid):int(cid) for sid,cid in desired.items() if (int(sid),int(cid)) not in pair_records}
    if not missing: return
    loader=_scan_loader(runtime,target,missing.keys(),batch_size=batch_size,num_workers=num_workers,strip_label=True)
    geometry_model.eval(); solved_total=0
    for raw in loader:
        if FORBIDDEN_TARGET_LABEL_KEYS.intersection(raw.keys()): raise RuntimeError("target truth leaked into Phase registration refresh")
        batch=_move(raw,device); parcels=batch["parcel_index"].detach().cpu().long()
        output=geometry_model(batch["pixels"],batch["valid_pixels"],batch["positions"],batch.get("extra"),return_geometry=True)
        reg=evaluate_registration_geometry(output.trend,output.positions,output.mask,reg_extractor)
        t_cache=TRegistrationGeometryCache(
            sample_ids=parcels,
            trend_srvf_reg=reg.trend_srvf.detach().cpu(),
            trend_support_reg=reg.trend_support.detach().cpu(),
            trend_valid=reg.trend_valid.detach().cpu(),
            registration_grid=reg.registration_grid.detach().cpu(),
        )
        assignments=[(i,missing[int(sid)]) for i,sid in enumerate(parcels.tolist())]
        solved=solve_t_only_registrations(source_reg_bank,t_cache,assignments,scan_config,workers=workers,progress_label="PHASE14_LAZY_DP",max_target_samples_per_pool=max(1,min(int(dp_chunk_size),len(assignments))))
        for record in solved:
            item=_minimal_record_payload(record); item["source"]="experiment14_lazy_dp"; pair_records[(int(record.sample_id),int(record.class_id))]=item
        solved_total+=len(solved)
        _save_pair_cache(cache_path,pair_records,source=source,target=target,seed=seed,fold=fold,model_checkpoint=model_checkpoint)
    print(f"PHASE14_PAIR_CACHE_EXTEND|new_pairs={solved_total}|total_pairs={len(pair_records)}",flush=True)


def _estimate_phase(*, scan: dict, pair_records: dict[tuple[int,int],dict], runtime: dict, geometry_model, source_reg_bank, reg_extractor, scan_config, device: torch.device, batch_size: int, num_workers: int, workers: int, dp_chunk_size: int, cache_path: Path, source: str, target: str, seed: int, fold: int, model_checkpoint: Path, num_classes: int) -> SharedDomainPhaseEstimate:
    ids=scan["sample_id"].tolist(); pred=scan["pred"].tolist(); mask=scan["mask"].tolist()
    accepted={int(sid):int(cid) for sid,cid,keep in zip(ids,pred,mask) if bool(keep)}
    _solve_missing_pairs(desired=accepted,pair_records=pair_records,runtime=runtime,geometry_model=geometry_model,source_reg_bank=source_reg_bank,reg_extractor=reg_extractor,scan_config=scan_config,device=device,batch_size=batch_size,num_workers=num_workers,workers=workers,dp_chunk_size=dp_chunk_size,cache_path=cache_path,source=source,target=target,seed=seed,fold=fold,model_checkpoint=model_checkpoint)
    gammas_by_class={c:[] for c in range(num_classes)}; accepted_counts={c:0 for c in range(num_classes)}
    for sid,cid in accepted.items():
        accepted_counts[cid]+=1; row=pair_records.get((sid,cid))
        if row is None: continue
        gamma=row.get("gamma")
        if bool(row.get("numerically_valid",False)) and isinstance(gamma,Tensor): gammas_by_class[cid].append(gamma)
    return build_shared_domain_phase(gammas_by_class,accepted_count_by_class=accepted_counts,num_classes=num_classes,k_reg=int(scan_config.k_reg))


def _save_phase_state(root: Path, *, arm: str, phase_epoch: int, actual_gamma: Tensor, estimate: SharedDomainPhaseEstimate | None, previous_gamma: Tensor, applied: bool) -> None:
    root.mkdir(parents=True,exist_ok=True); gamma=validate_phase(actual_gamma); prev=validate_phase(previous_gamma)
    payload={"protocol":PROTOCOL,"arm":arm,"phase_epoch":int(phase_epoch),"actual_gamma":gamma,"applied":bool(applied),"distance_to_identity":phase_distance_value(gamma,identity_phase(gamma.numel())),"distance_to_previous":phase_distance_value(gamma,prev),"estimate_valid":None if estimate is None else bool(estimate.valid),"estimate_invalid_reason":None if estimate is None else estimate.invalid_reason,"participating_classes":() if estimate is None else estimate.participating_classes,"accepted_count_by_class":() if estimate is None else estimate.accepted_count_by_class,"valid_gamma_count_by_class":() if estimate is None else estimate.valid_gamma_count_by_class,"class_centers":{} if estimate is None else {int(c):g.detach().cpu().double() for c,g in estimate.class_centers}}
    torch.save(payload,root/f"phase_{phase_epoch:03d}.pt")
    rows=[] if estimate is None else class_center_distance_rows(estimate,gamma)
    for row in rows: row.update({"arm":arm,"phase_epoch":int(phase_epoch),"phase_valid":bool(estimate.valid),"phase_applied":bool(applied)})
    _write_csv(root/f"phase_{phase_epoch:03d}_class_summary.csv",rows)
    print(f"PHASE14_PHASE_STATE|arm={arm}|phase_epoch={phase_epoch}|applied={str(applied).lower()}|valid={str(estimate.valid if estimate is not None else False).lower()}|classes={','.join(map(str,estimate.participating_classes if estimate is not None else ())) }|d_id={payload['distance_to_identity']:.8g}|d_prev={payload['distance_to_previous']:.8g}",flush=True)


def _build_optimizer_state(model, *, lr: float, weight_decay: float, total_steps: int, ema_decay: float):
    policy=configure_13b_semantic_student(model); named=dict(model.named_parameters()); params=[named[n] for n in policy.trainable_parameter_names]
    optimizer=torch.optim.Adam(params,lr=lr,weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_steps,eta_min=0.0)
    teacher=Stage2EMATeacher.from_student(model,policy,decay=ema_decay)
    return policy,optimizer,scheduler,teacher


def _train_window(*, model, teacher: Stage2EMATeacher, optimizer, scheduler, scaler, objective: Stage2Objective, source_loader, target_loader, device: torch.device, gamma: Tensor, steps: int, amp: bool, pseudo_threshold: float, label: str) -> dict:
    source_iter=iter(source_loader); target_iter=iter(target_loader); model.train(); meters={"source":0.0,"target":0.0,"total":0.0,"accepted":0,"target_seen":0}
    for _ in range(steps):
        try: source_raw=next(source_iter)
        except StopIteration: source_iter=iter(source_loader); source_raw=next(source_iter)
        try: target_raw=next(target_iter)
        except StopIteration: target_iter=iter(target_loader); target_raw=next(target_iter)
        if FORBIDDEN_TARGET_LABEL_KEYS.intersection(target_raw.keys()): raise RuntimeError("target truth leaked into experiment-14 training")
        source=_move(source_raw,device); target=_move(target_raw,device); optimizer.zero_grad(set_to_none=True)
        amp_on=bool(amp and device.type=="cuda")
        with torch.no_grad():
            teacher_out=_forward_target_teacher_phase(teacher.model(),target,gamma)
            teacher_post=torch.softmax(teacher_out.logits.detach().float(),dim=1)
            conf,pseudo=teacher_post.max(dim=1)
        keep=conf>float(pseudo_threshold)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp_on):
            source_out=_forward_source_phase(model,source,gamma)
            target_out=_raw_forward(model,target)
            if bool(keep.any().item()):
                obj=objective(source_to_target_logits=source_out.logits,source_labels=source["label"].long(),native_target_logits=target_out.logits[keep],stable_target_labels=pseudo[keep].long())
            else:
                obj=objective(source_to_target_logits=source_out.logits,source_labels=source["label"].long(),native_target_logits=None,stable_target_labels=None)
        scaler.scale(obj.total).backward(); scaler.step(optimizer); scaler.update(); scheduler.step(); teacher.update_after_optimizer_step(model)
        meters["source"]+=float(obj.source_to_target.detach()); meters["target"]+=float(obj.native_target.detach()); meters["total"]+=float(obj.total.detach()); meters["accepted"]+=int(keep.sum()); meters["target_seen"]+=int(keep.numel())
    out={"source_loss":meters["source"]/steps,"target_loss":meters["target"]/steps,"total_loss":meters["total"]/steps,"accepted_fraction":meters["accepted"]/max(1,meters["target_seen"]),"steps":int(steps)}
    print(f"PHASE14_TRAIN_WINDOW|window={label}|steps={steps}|source_loss={out['source_loss']:.6f}|target_loss={out['target_loss']:.6f}|total_loss={out['total_loss']:.6f}|accepted_fraction={out['accepted_fraction']:.6f}",flush=True)
    return out


def _source_val_metrics(model, loader, device: torch.device, num_classes: int) -> dict:
    scan=_scan_raw(model,loader,device,allow_labels=True); return _classification_metrics(scan["posterior"],scan["label"],num_classes)


def _save_target_val(root: Path, *, arm: str, epoch: int, student, teacher, loader, device: torch.device) -> None:
    root.mkdir(parents=True,exist_ok=True)
    for role,model in (("student",student),("teacher",teacher.model())):
        scan=_scan_raw(model,loader,device,allow_labels=False)
        np.savez_compressed(root/f"epoch_{epoch:03d}_{role}.npz",sample_id=scan["sample_id"].numpy().astype(np.int64),posterior=scan["posterior"].numpy().astype(np.float32),arm=np.asarray([arm]),epoch=np.asarray([epoch],dtype=np.int64))


def _save_checkpoint(path: Path, *, arm: str, epoch: int, student, teacher, optimizer, scheduler, scaler, gamma: Tensor, policy) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"protocol":PROTOCOL,"arm":arm,"epoch":int(epoch),"student_state_dict":{k:v.detach().cpu() for k,v in student.state_dict().items()},"teacher_state_dict":{k:v.detach().cpu() for k,v in teacher.model().state_dict().items()},"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"scaler_state_dict":scaler.state_dict(),"actual_phase":validate_phase(gamma),"trainable_parameter_names":policy.trainable_parameter_names},path)


def _write_readme(path: Path, config: dict) -> None:
    path.write_text(f"""# 实验 14：共享 Domain Phase 迭代闭环构建与训练效应验证

任务固定为 AT1 → DK1，seed=1，fold=0。Stage-1 checkpoint：`{config['stage1_checkpoint']}`。

本实验先运行一次 `{config['warmup_epochs']}×{config['steps_per_epoch']}` identity warm-up，然后从完全相同的 Student、EMA Teacher、optimizer、scheduler 状态分叉出 `NO_PHASE`、`STATIC_DOMAIN_PHASE`、`ITERATIVE_DOMAIN_PHASE` 三个配置，各运行 `{config['phase_epochs']}×{config['steps_per_epoch']}`。因此三组的训练差异只来自 shared Domain Phase 的使用方式。

Domain Phase 方向固定为 `delta: source_time -> target_time`。source true-label branch 使用 `delta(t_s)`；Teacher 在 `delta^-1(t_t)` 上产生 target pseudo-label；Student target branch 始终使用原始 `t_t`。当前不实现 class-conditioned Phase、Domain Shape、额外 alignment loss、DAPL/IPL/TFDA 联合、agreement gate 或 supervision 状态机。

几何分支由独立 Stage-1 模型副本提供并保持冻结。Phase observation 只由当前 Teacher confidence mask 决定类别资格；registration 的 gain/Shape 等旧 legality 不作为新的 reliability gate。数值求解失败的 gamma 只能作为不可计算记录被排除。类别内与类别间中心都使用 Fisher–Rao/Karcher mean；shared Phase 先做类别内中心，再对非空类别中心等权聚合。只有一个非空类别时不更新 Phase。

`pseudo_refresh/`、`phase_refresh/`、`target_val_unlabeled/` 全部在 main training 中生成，不读取 target truth。target true labels 只由独立 `evaluate_stage2_shared_domain_phase_14_oracle.py` 后验加入。训练过程中不做 target oracle early stopping，也不选择 best epoch。

registration cache `cache/phase_registration_pairs.pt` 按 `sample_id × current pseudo-label class` 断点续算；若 13A raw-candidate cache 与同一 Stage-1 checkpoint 匹配，会先复用其中已有的 raw-candidate gamma。后续只有 pseudo-label 变到此前未注册类别时才新增 exact-DP pair，不重新训练几何分支。
""",encoding="utf-8")


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--calibration-checkpoint",type=Path,required=True); p.add_argument("--model-checkpoint",type=Path,required=True)
    p.add_argument("--source-registration-bank-cache",type=Path,required=True); p.add_argument("--raw-candidate-geometry-cache",type=Path,required=True)
    p.add_argument("--data-root",type=str,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",type=str,default="cuda:0"); p.add_argument("--fold",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=64); p.add_argument("--num-workers",type=int,default=4); p.add_argument("--registration-workers",type=int,default=4); p.add_argument("--dp-target-chunk-size",type=int,default=512)
    p.add_argument("--warmup-epochs",type=int,default=5); p.add_argument("--phase-epochs",type=int,default=20); p.add_argument("--steps-per-epoch",type=int,default=500)
    p.add_argument("--pseudo-threshold",type=float,default=0.9); p.add_argument("--lambda-target",type=float,default=1.0); p.add_argument("--focal-gamma",type=float,default=1.0); p.add_argument("--ema-decay",type=float,default=0.9999); p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--amp",action="store_true")
    return p.parse_args()


def main() -> None:
    args=parse_args(); device=torch.device(args.device); calibration=torch.load(args.calibration_checkpoint.resolve(),map_location="cpu",weights_only=False); runtime=dict(calibration["runtime_config"]); runtime["data_root"]=args.data_root
    source,target,seed,fold=str(runtime["source"]),str(runtime["target"]),int(runtime["seed"]),int(args.fold)
    if (source,target,seed,fold)!=EXPECTED_RUNTIME: raise ValueError("experiment 14 first run is frozen to AT1->DK1 seed=1 fold=0")
    classes=list(runtime["classes"]); num_classes=len(classes); model_checkpoint=torch.load(args.model_checkpoint.resolve(),map_location="cpu",weights_only=False)
    if not 0.0<=args.pseudo_threshold<=1.0: raise ValueError("pseudo threshold must lie in [0,1]")
    if args.warmup_epochs!=5 or args.phase_epochs!=20 or args.steps_per_epoch!=500: print("PHASE14_NONDEFAULT_BUDGET|explicit_cli_override=true",flush=True)
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True); cache_dir=output/"cache"; cache_dir.mkdir(parents=True,exist_ok=True)
    _set_seed(seed)

    source_all=phasevis._eligible_parcels(args.data_root,source,classes,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    target_all=phasevis._eligible_parcels(args.data_root,target,classes,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    splits=phasevis._reconstruct_fold_splits(source_all,target_all,source=source,target=target,seed=seed,val_ratio=float(runtime.get("val_ratio",0.1)),test_ratio=float(runtime.get("test_ratio",0.2)),fold=fold)
    source_train=sorted(splits[source]["train"]); source_val=sorted(splits[source]["val"]); target_train=sorted(splits[target]["train"]); target_val=sorted(splits[target]["val"])
    target_scan_loader=_scan_loader(runtime,target,target_train,batch_size=args.batch_size,num_workers=args.num_workers,strip_label=True)
    target_val_loader=_scan_loader(runtime,target,target_val,batch_size=args.batch_size,num_workers=args.num_workers,strip_label=True)
    source_val_loader=_scan_loader(runtime,source,source_val,batch_size=args.batch_size,num_workers=args.num_workers,strip_label=False)

    geometry_model=phasevis._build_model(runtime,calibration,device,model_checkpoint=model_checkpoint); geometry_model.eval(); [p.requires_grad_(False) for p in geometry_model.parameters()]; geometry_hash_before=_state_hash(geometry_model)
    scan_config=samplediag._scan_config(runtime,args.registration_workers); reg_extractor=build_stage2_registration_extractor(geometry_model,device=device,k_reg=scan_config.k_reg)
    source_geometry_loader=phasevis._selected_loader(args.data_root,source,classes,np.asarray(source_train,dtype=np.int64),closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")),batch_size=args.batch_size,num_workers=args.num_workers)
    source_reg_bank=samplediag._load_or_build_registration_bank(args.source_registration_bank_cache.resolve(),model=geometry_model,source_train_loader=DeviceBatchLoader(source_geometry_loader,device),num_classes=num_classes,device=device,reg_extractor=reg_extractor)
    if not bool(source_reg_bank.ready.all().item()): raise RuntimeError("experiment 14 requires all source registration classes ready")

    # Stage-1 label-free reference, used only to validate reusable 13A gamma cache.
    anchor=phasevis._build_model(runtime,calibration,device,model_checkpoint=model_checkpoint); anchor.eval(); [p.requires_grad_(False) for p in anchor.parameters()]
    initial_scan=_scan_raw(anchor,target_scan_loader,device,allow_labels=False)
    stage1_val=_scan_raw(anchor,target_val_loader,device,allow_labels=False)
    stage1_dir=output/"stage1_reference"; stage1_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(stage1_dir/"target_train_unlabeled.npz",sample_id=initial_scan["sample_id"].numpy().astype(np.int64),posterior=initial_scan["posterior"].numpy().astype(np.float32))
    np.savez_compressed(stage1_dir/"target_val_unlabeled.npz",sample_id=stage1_val["sample_id"].numpy().astype(np.int64),posterior=stage1_val["posterior"].numpy().astype(np.float32))
    _json_dump(stage1_dir/"source_val_metrics.json",_source_val_metrics(anchor,source_val_loader,device,num_classes))
    pair_cache_path=cache_dir/"phase_registration_pairs.pt"; pair_records=_load_pair_cache(pair_cache_path,source=source,target=target,seed=seed,fold=fold,model_checkpoint=args.model_checkpoint.resolve())
    _prefill_raw_candidate_cache(pair_records,args.raw_candidate_geometry_cache.resolve(),initial_scan=initial_scan,model_checkpoint=args.model_checkpoint.resolve(),source=source,target=target,seed=seed,fold=fold)
    _save_pair_cache(pair_cache_path,pair_records,source=source,target=target,seed=seed,fold=fold,model_checkpoint=args.model_checkpoint.resolve())

    # One common warm-up.  The exact resulting semantic/optimizer/Teacher state is forked into all arms.
    student=phasevis._build_model(runtime,calibration,device,model_checkpoint=model_checkpoint)
    total_steps=(args.warmup_epochs+args.phase_epochs)*args.steps_per_epoch
    policy,optimizer,scheduler,teacher=_build_optimizer_state(student,lr=args.lr,weight_decay=args.weight_decay,total_steps=total_steps,ema_decay=args.ema_decay)
    objective=Stage2Objective(num_classes=num_classes,config=Stage2ObjectiveConfig(lambda_target=args.lambda_target,focal_gamma=args.focal_gamma)).to(device)
    scaler=torch.amp.GradScaler("cuda",enabled=(args.amp and device.type=="cuda"))
    identity=identity_phase(scan_config.k_reg)
    warmup_rows=[]
    for epoch in range(1,args.warmup_epochs+1):
        source_loader=_source_train_loader(runtime,splits,seed=seed+epoch,batch_size=args.batch_size,num_workers=args.num_workers)
        target_loader=_target_train_loader(runtime,target_train,seed=seed+epoch,batch_size=args.batch_size,num_workers=args.num_workers)
        stats=_train_window(model=student,teacher=teacher,optimizer=optimizer,scheduler=scheduler,scaler=scaler,objective=objective,source_loader=source_loader,target_loader=target_loader,device=device,gamma=identity,steps=args.steps_per_epoch,amp=args.amp,pseudo_threshold=args.pseudo_threshold,label=f"warmup:{epoch}")
        stats.update({"stage":"warmup","epoch":epoch}); warmup_rows.append(stats)
        _save_target_val(output/"warmup/target_val_unlabeled",arm="COMMON_WARMUP",epoch=epoch,student=student,teacher=teacher,loader=target_val_loader,device=device)
        sm=_source_val_metrics(student,source_val_loader,device,num_classes); tm=_source_val_metrics(teacher.model(),source_val_loader,device,num_classes)
        warmup_rows.append({"stage":"warmup_source_val","epoch":epoch,"role":"student",**sm})
        warmup_rows.append({"stage":"warmup_source_val","epoch":epoch,"role":"teacher",**tm})
    _write_csv(output/"00_warmup_training.csv",warmup_rows)
    warmup_refresh=_scan_target_teacher(teacher.model(),target_scan_loader,device,identity,threshold=args.pseudo_threshold)
    _save_refresh(output/"warmup/pseudo_refresh",arm="COMMON_WARMUP",refresh_id=0,phase_epoch=0,scan=warmup_refresh,previous_top1=None)

    first_estimate=_estimate_phase(scan=warmup_refresh,pair_records=pair_records,runtime=runtime,geometry_model=geometry_model,source_reg_bank=source_reg_bank,reg_extractor=reg_extractor,scan_config=scan_config,device=device,batch_size=args.batch_size,num_workers=args.num_workers,workers=args.registration_workers,dp_chunk_size=args.dp_target_chunk_size,cache_path=pair_cache_path,source=source,target=target,seed=seed,fold=fold,model_checkpoint=args.model_checkpoint.resolve(),num_classes=num_classes)
    first_phase=first_estimate.gamma if first_estimate.valid else identity
    _save_phase_state(output/"warmup/phase_refresh",arm="COMMON_WARMUP",phase_epoch=1,actual_gamma=first_phase,estimate=first_estimate,previous_gamma=identity,applied=bool(first_estimate.valid))

    warmup_state={"student":deepcopy({k:v.detach().cpu() for k,v in student.state_dict().items()}),"teacher":deepcopy({k:v.detach().cpu() for k,v in teacher.model().state_dict().items()}),"optimizer":deepcopy(optimizer.state_dict()),"scheduler":deepcopy(scheduler.state_dict()),"scaler":deepcopy(scaler.state_dict())}
    torch.save({"protocol":PROTOCOL,"warmup_epochs":args.warmup_epochs,"student_state_dict":warmup_state["student"],"teacher_state_dict":warmup_state["teacher"],"optimizer_state_dict":warmup_state["optimizer"],"scheduler_state_dict":warmup_state["scheduler"],"scaler_state_dict":warmup_state["scaler"],"first_phase":first_phase,"first_phase_valid":first_estimate.valid},output/"warmup/common_warmup_checkpoint.pt")

    arm_summaries={}; training_rows=[]
    for arm in PHASE_ARMS:
        _set_seed(seed+4000); model=phasevis._build_model(runtime,calibration,device,model_checkpoint=model_checkpoint); model.load_state_dict(warmup_state["student"],strict=True)
        arm_policy,arm_optimizer,arm_scheduler,arm_teacher=_build_optimizer_state(model,lr=args.lr,weight_decay=args.weight_decay,total_steps=total_steps,ema_decay=args.ema_decay)
        arm_teacher.model().load_state_dict(warmup_state["teacher"],strict=True); arm_optimizer.load_state_dict(warmup_state["optimizer"]); arm_scheduler.load_state_dict(warmup_state["scheduler"]); arm_scaler=torch.amp.GradScaler("cuda",enabled=(args.amp and device.type=="cuda")); arm_scaler.load_state_dict(warmup_state["scaler"])
        current=identity if arm=="NO_PHASE" else first_phase.clone(); fixed_first=None if arm=="NO_PHASE" else first_phase.clone()
        current_phase_valid = bool(arm != "NO_PHASE" and first_estimate.valid)
        previous_top1={int(s):int(p) for s,p in zip(warmup_refresh["sample_id"].tolist(),warmup_refresh["pred"].tolist())}
        arm_root=output/"arms"/arm; refresh_root=arm_root/"pseudo_refresh"; phase_root=arm_root/"phase_refresh"; val_root=arm_root/"target_val_unlabeled"
        initial_estimate=None if arm=="NO_PHASE" else first_estimate
        _save_phase_state(phase_root,arm=arm,phase_epoch=1,actual_gamma=current,estimate=initial_estimate,previous_gamma=identity,applied=current_phase_valid)
        source_rows=[]
        for epoch in range(1,args.phase_epochs+1):
            phase_used=current.clone()
            source_loader=_source_train_loader(runtime,splits,seed=seed+10000+epoch,batch_size=args.batch_size,num_workers=args.num_workers)
            target_loader=_target_train_loader(runtime,target_train,seed=seed+10000+epoch,batch_size=args.batch_size,num_workers=args.num_workers)
            stats=_train_window(model=model,teacher=arm_teacher,optimizer=arm_optimizer,scheduler=arm_scheduler,scaler=arm_scaler,objective=objective,source_loader=source_loader,target_loader=target_loader,device=device,gamma=phase_used,steps=args.steps_per_epoch,amp=args.amp,pseudo_threshold=args.pseudo_threshold,label=f"{arm}:{epoch}")
            stats.update({"arm":arm,"epoch":epoch,"phase_distance_to_identity":phase_distance_value(phase_used,identity)}); training_rows.append(stats)
            refresh=_scan_target_teacher(arm_teacher.model(),target_scan_loader,device,phase_used,threshold=args.pseudo_threshold,previous_top1=previous_top1)
            previous_top1=_save_refresh(refresh_root,arm=arm,refresh_id=epoch,phase_epoch=epoch,scan=refresh,previous_top1=previous_top1)
            _save_target_val(val_root,arm=arm,epoch=epoch,student=model,teacher=arm_teacher,loader=target_val_loader,device=device)
            sm=_source_val_metrics(model,source_val_loader,device,num_classes); tm=_source_val_metrics(arm_teacher.model(),source_val_loader,device,num_classes)
            source_rows.append({"arm":arm,"epoch":epoch,"role":"student",**sm}); source_rows.append({"arm":arm,"epoch":epoch,"role":"teacher",**tm})
            _save_checkpoint(arm_root/"checkpoints"/f"epoch_{epoch:03d}.pt",arm=arm,epoch=epoch,student=model,teacher=arm_teacher,optimizer=arm_optimizer,scheduler=arm_scheduler,scaler=arm_scaler,gamma=phase_used,policy=arm_policy)

            # A new Phase may only be constructed at the epoch boundary and is
            # used by the *next* epoch.  No same-minibatch circular proof.
            if epoch < args.phase_epochs:
                estimate=None
                if arm=="ITERATIVE_DOMAIN_PHASE":
                    estimate=_estimate_phase(scan=refresh,pair_records=pair_records,runtime=runtime,geometry_model=geometry_model,source_reg_bank=source_reg_bank,reg_extractor=reg_extractor,scan_config=scan_config,device=device,batch_size=args.batch_size,num_workers=args.num_workers,workers=args.registration_workers,dp_chunk_size=args.dp_target_chunk_size,cache_path=pair_cache_path,source=source,target=target,seed=seed,fold=fold,model_checkpoint=args.model_checkpoint.resolve(),num_classes=num_classes)
                next_phase=actual_phase_for_arm(arm,identity=identity,first_phase=fixed_first,previous_phase=current,proposal=estimate,phase_epoch=epoch+1)
                if arm == "ITERATIVE_DOMAIN_PHASE" and estimate is not None and estimate.valid:
                    current_phase_valid = True
                _save_phase_state(phase_root,arm=arm,phase_epoch=epoch+1,actual_gamma=next_phase,estimate=estimate,previous_gamma=current,applied=current_phase_valid)
                current=next_phase
        _write_csv(arm_root/"source_val_metrics.csv",source_rows)
        arm_summaries[arm]={"final_student_hash":_state_hash(model),"final_teacher_hash":_state_hash(arm_teacher.model()),"final_phase_distance_to_identity":phase_distance_value(current,identity)}
    _write_csv(output/"01_training_windows.csv",training_rows)

    geometry_hash_after=_state_hash(geometry_model)
    if geometry_hash_after!=geometry_hash_before: raise RuntimeError("frozen geometry model changed during experiment 14")
    config={"protocol":PROTOCOL,"source":source,"target":target,"seed":seed,"fold":fold,"classes":classes,"stage1_checkpoint":str(args.model_checkpoint.resolve()),"calibration_checkpoint":str(args.calibration_checkpoint.resolve()),"source_registration_bank_cache":str(args.source_registration_bank_cache.resolve()),"raw_candidate_geometry_cache":str(args.raw_candidate_geometry_cache.resolve()),"pair_registration_cache":str(pair_cache_path),"warmup_epochs":args.warmup_epochs,"phase_epochs":args.phase_epochs,"steps_per_epoch":args.steps_per_epoch,"phase_refresh_interval_epochs":1,"phase_arms":list(PHASE_ARMS),"pseudo_label_mechanism":"TimeMatch-compatible EMA Teacher top1 + confidence threshold","pseudo_threshold":args.pseudo_threshold,"lambda_target":args.lambda_target,"focal_gamma":args.focal_gamma,"ema_decay":args.ema_decay,"optimizer":"Adam","lr":args.lr,"weight_decay":args.weight_decay,"scheduler":"CosineAnnealingLR over warmup+phase total steps","student_trainable":"PSE_S + LTAE + TimeEncoder + Classifier","geometry_branch":"separate frozen Stage-1 copy","phase_center":"two-stage Fisher-Rao/Karcher: sample->class center, equal class-center aggregation","phase_observation_gate":"Teacher confidence mask only; numerical solver validity is computational validity, not reliability gate","source_training_time":"delta(t_s)","teacher_target_time":"delta^{-1}(t_t)","student_target_time":"native t_t","raw_source_loss":False,"class_conditioned_phase":False,"domain_shape":False,"extra_alignment_losses":False,"target_truth_used_in_main_training":False,"target_oracle_early_stopping":False,"warmup_forked_once_for_all_arms":True,"geometry_state_hash":geometry_hash_before,"arm_summaries":arm_summaries}
    _json_dump(output/"00_experiment14_config.json",config); _write_readme(output/"README_中文说明.md",config)
    print(f"PHASE14_TRAINING_COMPLETE|arms=3|warmup_shared=true|geometry_unchanged=true|target_truth_used=false|output={output}",flush=True)


if __name__=="__main__": main()
