#!/usr/bin/env python3
"""Oracle-only evaluator for experiment 14.

This script is intentionally separate from the main training process.  It joins
target truth only after all three Phase arms have completed.
"""
from __future__ import annotations

import argparse
import csv
import json
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

import diagnose_stage2_bootstrap_temporal_state as boot12
import visualize_stage2_phase_alignment as phasevis
from methods.structure_da.domain_phase_loop import (
    PHASE_ARMS,
    build_shared_domain_phase,
    identity_phase,
    phase_distance_value,
)

PROTOCOL = "14_shared_domain_phase_iterative_loop_v1"


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]; seen=set()
    for row in rows:
        for key in row:
            if key not in seen: fields.append(key); seen.add(key)
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _json_dump(path: Path,payload) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=True)+"\n",encoding="utf-8")


def _metrics(pred: np.ndarray, truth: np.ndarray, num_classes: int) -> dict:
    pred=np.asarray(pred,dtype=np.int64); truth=np.asarray(truth,dtype=np.int64)
    if pred.shape!=truth.shape or pred.ndim!=1: raise ValueError("pred/truth mismatch")
    acc=float(np.mean(pred==truth)) if pred.size else float("nan"); f1=[]; weights=[]
    for c in range(num_classes):
        tp=int(np.sum((pred==c)&(truth==c))); fp=int(np.sum((pred==c)&(truth!=c))); fn=int(np.sum((pred!=c)&(truth==c))); n=int(np.sum(truth==c))
        p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0; score=2*p*r/(p+r) if p+r else 0.0
        f1.append(score); weights.append(n)
    return {"accuracy":acc,"macro_f1":float(np.mean(f1)),"weighted_f1":float(np.average(f1,weights=weights)) if sum(weights) else float("nan")}


def _per_class_rows(pred: np.ndarray,truth: np.ndarray,classes: Sequence[str],**prefix) -> list[dict]:
    rows=[]
    for c,name in enumerate(classes):
        tp=int(np.sum((pred==c)&(truth==c))); fp=int(np.sum((pred==c)&(truth!=c))); fn=int(np.sum((pred!=c)&(truth==c))); n=int(np.sum(truth==c)); pred_n=int(np.sum(pred==c))
        p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0; f1=2*p*r/(p+r) if p+r else 0.0
        rows.append({**prefix,"class_id":c,"class_name":name,"precision":p,"recall":r,"f1":f1,"true_count":n,"pred_count":pred_n})
    return rows


def _label_map(config: dict, calibration: dict, data_root: str, *, split: str) -> dict[int,int]:
    runtime=dict(calibration["runtime_config"]); runtime["data_root"]=data_root; classes=list(runtime["classes"]); source=str(runtime["source"]); target=str(runtime["target"]); seed=int(runtime["seed"]); fold=int(config["fold"])
    source_all=phasevis._eligible_parcels(data_root,source,classes,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    target_all=phasevis._eligible_parcels(data_root,target,classes,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    splits=phasevis._reconstruct_fold_splits(source_all,target_all,source=source,target=target,seed=seed,val_ratio=float(runtime.get("val_ratio",0.1)),test_ratio=float(runtime.get("test_ratio",0.2)),fold=fold)
    meta=phasevis._metadata_dataset(data_root,target,classes,splits[target][split],closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    return {int(s):int(y) for s,y in zip(meta.get_parcel_indices().tolist(),meta.get_labels().tolist())}


def _aligned_truth(ids: np.ndarray,label_by: Mapping[int,int]) -> np.ndarray:
    missing=[int(v) for v in ids.tolist() if int(v) not in label_by]
    if missing: raise ValueError(f"oracle label map misses {len(missing)} samples")
    return np.asarray([label_by[int(v)] for v in ids.tolist()],dtype=np.int64)


def _critical_flow_rows(pred: np.ndarray,truth: np.ndarray,classes: Sequence[str],**prefix) -> list[dict]:
    index={name:i for i,name in enumerate(classes)}; specs=[("spring_barley","spring_oat"),("winter_rye","winter_triticale"),("winter_wheat","winter_triticale")]
    rows=[]
    for src,dst in specs:
        if src not in index or dst not in index: continue
        s=index[src]; d=index[dst]; cohort=truth==s; n=int(cohort.sum()); wrong=int(np.sum(cohort&(pred==d))); correct=int(np.sum(cohort&(pred==s)))
        rows.append({**prefix,"true_class":src,"absorbed_into":dst,"cohort_n":n,"absorbed_n":wrong,"absorbed_rate":wrong/n if n else float("nan"),"correct_top1_n":correct,"correct_top1_rate":correct/n if n else float("nan")})
    return rows


def _load_oracle_phase(path: Path, *, num_classes: int, k_reg: int, config: dict, train_labels: Mapping[int, int]):
    if not path.is_file(): return None
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("schema") != boot12.ORACLE_CACHE_SCHEMA:
        raise ValueError("unsupported experiment-12 oracle registration cache schema")
    for key in ("source", "target", "seed", "fold"):
        if str(payload.get(key)) != str(config.get(key)):
            raise ValueError(f"oracle registration cache {key} mismatch")
    cached_model=payload.get("model_checkpoint")
    if cached_model is None or Path(str(cached_model)).resolve() != Path(str(config["stage1_checkpoint"])).resolve():
        raise ValueError("oracle registration cache Stage-1 checkpoint mismatch")
    records=[boot12._record_from_payload(row) for row in payload.get("records",())]
    expected_ids=set(map(int,train_labels.keys())); seen=set()
    gammas={c:[] for c in range(num_classes)}; accepted={c:0 for c in range(num_classes)}
    for record in records:
        sid=int(record.sample_id); cid=int(record.class_id)
        if sid not in expected_ids:
            raise ValueError("oracle registration cache contains sample outside target-train")
        if sid in seen:
            raise ValueError("duplicate target sample in oracle registration cache")
        if cid != int(train_labels[sid]):
            raise ValueError("oracle registration cache true-class identity mismatch")
        seen.add(sid); accepted[cid]+=1
        if record.gamma is not None and record.numerically_valid: gammas[cid].append(record.gamma)
    if seen != expected_ids:
        raise ValueError(f"oracle registration cache incomplete: {len(seen)}/{len(expected_ids)} target-train samples")
    estimate=build_shared_domain_phase(gammas,accepted_count_by_class=accepted,num_classes=num_classes,k_reg=k_reg)
    if not estimate.valid: raise RuntimeError("oracle true-class registrations cannot form a valid shared Phase")
    return estimate


def _fixed_initial_error_cohorts(exp: Path, train_labels: Mapping[int, int], classes: Sequence[str]) -> dict[str, dict]:
    path=exp/"stage1_reference"/"target_train_unlabeled.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing Stage-1 target-train reference: {path}")
    with np.load(path,allow_pickle=False) as z:
        ids=z["sample_id"].astype(np.int64); posterior=z["posterior"].astype(np.float64)
    truth=_aligned_truth(ids,train_labels); initial=posterior.argmax(axis=1); index={name:i for i,name in enumerate(classes)}
    specs=[("barley_to_oat","spring_barley","spring_oat"),("rye_to_triticale","winter_rye","winter_triticale"),("wheat_to_triticale","winter_wheat","winter_triticale")]
    cohorts={}
    for key,src,dst in specs:
        if src not in index or dst not in index: continue
        src_id=index[src]; dst_id=index[dst]; keep=(truth==src_id)&(initial==dst_id)
        cohorts[key]={"sample_ids":set(map(int,ids[keep].tolist())),"true_class_id":src_id,"wrong_class_id":dst_id,"true_class":src,"initial_wrong_class":dst}
    return cohorts


def _fixed_cohort_rows(ids: np.ndarray, posterior: np.ndarray, mask: np.ndarray, cohorts: Mapping[str, dict], *, arm: str, epoch: int) -> list[dict]:
    ids=np.asarray(ids,dtype=np.int64); posterior=np.asarray(posterior,dtype=np.float64); mask=np.asarray(mask,dtype=bool); pred=posterior.argmax(axis=1); row_by_id={int(s):i for i,s in enumerate(ids.tolist())}; rows=[]
    for name,spec in cohorts.items():
        positions=[row_by_id[sid] for sid in sorted(spec["sample_ids"]) if sid in row_by_id]
        if len(positions)!=len(spec["sample_ids"]):
            raise ValueError(f"refresh is missing samples from fixed cohort {name}")
        idx=np.asarray(positions,dtype=np.int64); n=int(idx.size); true_id=int(spec["true_class_id"]); wrong_id=int(spec["wrong_class_id"])
        if n:
            p_true=posterior[idx,true_id]; p_wrong=posterior[idx,wrong_id]; current=pred[idx]
            rows.append({"arm":arm,"epoch":int(epoch),"cohort":name,"true_class":spec["true_class"],"initial_wrong_class":spec["initial_wrong_class"],"cohort_n":n,"accepted_fraction":float(mask[idx].mean()),"mean_p_true":float(p_true.mean()),"median_p_true":float(np.median(p_true)),"mean_p_initial_wrong":float(p_wrong.mean()),"median_p_initial_wrong":float(np.median(p_wrong)),"still_initial_wrong_rate":float(np.mean(current==wrong_id)),"corrected_top1_rate":float(np.mean(current==true_id)),"other_wrong_rate":float(np.mean((current!=wrong_id)&(current!=true_id)))})
        else:
            rows.append({"arm":arm,"epoch":int(epoch),"cohort":name,"true_class":spec["true_class"],"initial_wrong_class":spec["initial_wrong_class"],"cohort_n":0,"accepted_fraction":float("nan"),"mean_p_true":float("nan"),"median_p_true":float("nan"),"mean_p_initial_wrong":float("nan"),"median_p_initial_wrong":float("nan"),"still_initial_wrong_rate":float("nan"),"corrected_top1_rate":float("nan"),"other_wrong_rate":float("nan")})
    return rows


def _phase_files(exp: Path):
    for arm in PHASE_ARMS:
        for path in sorted((exp/"arms"/arm/"phase_refresh").glob("phase_*.pt")):
            yield arm,path


def _plots(exp: Path, phase_rows: list[dict], pseudo_rows: list[dict], val_rows: list[dict]) -> None:
    figdir=exp/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    # Phase curves.
    plt.figure(figsize=(10,6)); grid=None
    for arm in PHASE_ARMS:
        files=sorted((exp/"arms"/arm/"phase_refresh").glob("phase_*.pt"))
        for path in files:
            payload=torch.load(path,map_location="cpu",weights_only=False); gamma=payload["actual_gamma"].detach().cpu().numpy(); x=np.linspace(0,1,len(gamma)); grid=x
            epoch=int(payload["phase_epoch"])
            if arm=="ITERATIVE_DOMAIN_PHASE" and epoch not in {1,5,10,15,20}: continue
            if arm!="ITERATIVE_DOMAIN_PHASE" and epoch!=1: continue
            plt.plot(x,gamma,label=f"{arm}:{epoch}")
    if grid is not None: plt.plot(grid,grid,linestyle="--",label="identity")
    plt.xlabel("source normalized time"); plt.ylabel("target normalized time"); plt.title("Shared Domain Phase curves"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(figdir/"phase_curves.png",dpi=180); plt.close()

    if phase_rows:
        plt.figure(figsize=(9,5))
        for arm in PHASE_ARMS:
            rows=[r for r in phase_rows if r["arm"]==arm]
            if rows: plt.plot([r["phase_epoch"] for r in rows],[r["distance_to_previous"] for r in rows],marker="o",label=arm)
        plt.xlabel("Phase epoch"); plt.ylabel("Fisher-Rao distance to previous Phase"); plt.title("Phase drift trajectory"); plt.legend(); plt.tight_layout(); plt.savefig(figdir/"phase_drift_trajectory.png",dpi=180); plt.close()

        final_path=exp/"arms"/"ITERATIVE_DOMAIN_PHASE"/"phase_refresh"/"phase_020.pt"
        if final_path.is_file():
            payload=torch.load(final_path,map_location="cpu",weights_only=False); shared=payload["actual_gamma"].detach().cpu().numpy(); x=np.linspace(0,1,len(shared))
            plt.figure(figsize=(10,6)); plt.plot(x,shared,linewidth=2,label="shared delta_20")
            for cid,center in sorted(payload.get("class_centers",{}).items()): plt.plot(x,center.detach().cpu().numpy(),alpha=0.65,label=f"class {cid}")
            plt.plot(x,x,linestyle="--",label="identity"); plt.xlabel("source normalized time"); plt.ylabel("target normalized time"); plt.title("Final iterative class Phase centers and shared Domain Phase"); plt.legend(fontsize=7,ncol=2); plt.tight_layout(); plt.savefig(figdir/"iterative_final_class_centers.png",dpi=180); plt.close()

    if pseudo_rows:
        plt.figure(figsize=(9,5))
        common=[r for r in pseudo_rows if r["arm"]=="COMMON_WARMUP"]
        for arm in PHASE_ARMS:
            rows=common+[r for r in pseudo_rows if r["arm"]==arm]
            if rows: plt.plot([r["epoch"] for r in rows],[r["accepted_macro_f1"] for r in rows],marker="o",label=f"{arm} F1")
        plt.xlabel("epoch"); plt.ylabel("accepted pseudo-label Macro-F1"); plt.title("Pseudo-label quality trajectory"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(figdir/"pseudo_label_macro_f1.png",dpi=180); plt.close()
        plt.figure(figsize=(9,5))
        for arm in PHASE_ARMS:
            rows=common+[r for r in pseudo_rows if r["arm"]==arm]
            if rows: plt.plot([r["epoch"] for r in rows],[r["coverage"] for r in rows],marker="o",label=arm)
        plt.xlabel("epoch"); plt.ylabel("coverage"); plt.title("Pseudo-label coverage trajectory"); plt.legend(); plt.tight_layout(); plt.savefig(figdir/"pseudo_label_coverage.png",dpi=180); plt.close()

    if val_rows:
        plt.figure(figsize=(9,5))
        for arm in PHASE_ARMS:
            rows=[r for r in val_rows if r["arm"]==arm and r["role"]=="student"]
            if rows: plt.plot([r["epoch"] for r in rows],[r["macro_f1"] for r in rows],marker="o",label=arm)
        plt.xlabel("epoch"); plt.ylabel("target-val Macro-F1"); plt.title("Target-val model trajectory"); plt.legend(); plt.tight_layout(); plt.savefig(figdir/"target_val_macro_f1.png",dpi=180); plt.close()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--experiment-dir",type=Path,required=True); p.add_argument("--calibration-checkpoint",type=Path,required=True); p.add_argument("--data-root",type=str,required=True); p.add_argument("--oracle-registration-cache",type=Path,default=None); args=p.parse_args()
    exp=args.experiment_dir.resolve(); config=json.loads((exp/"00_experiment14_config.json").read_text(encoding="utf-8"))
    if config.get("protocol")!=PROTOCOL: raise ValueError("not an experiment-14 output directory")
    calibration=torch.load(args.calibration_checkpoint.resolve(),map_location="cpu",weights_only=False); classes=list(calibration["runtime_config"]["classes"]); num_classes=len(classes)
    train_labels=_label_map(config,calibration,args.data_root,split="train"); val_labels=_label_map(config,calibration,args.data_root,split="val")
    oracle_dir=exp/"oracle"; oracle_dir.mkdir(parents=True,exist_ok=True)

    pseudo_summary=[]; per_class=[]; flow_rows=[]; confusion_rows=[]; fixed_flow_rows=[]
    fixed_cohorts=_fixed_initial_error_cohorts(exp,train_labels,classes)

    def audit_refresh(path: Path, *, arm: str, epoch: int) -> None:
        with np.load(path,allow_pickle=False) as z:
            ids=z["sample_id"].astype(np.int64); posterior=z["posterior"].astype(np.float64); pred=z["predicted_class"].astype(np.int64); mask=z["mask"].astype(bool)
        truth=_aligned_truth(ids,train_labels); all_m=_metrics(pred,truth,num_classes); accepted_m=_metrics(pred[mask],truth[mask],num_classes) if mask.any() else {"accuracy":float("nan"),"macro_f1":float("nan"),"weighted_f1":float("nan")}
        pseudo_summary.append({"arm":arm,"epoch":int(epoch),"coverage":float(mask.mean()),"accepted_n":int(mask.sum()),"all_accuracy":all_m["accuracy"],"all_macro_f1":all_m["macro_f1"],"all_weighted_f1":all_m["weighted_f1"],"accepted_precision":accepted_m["accuracy"],"accepted_macro_f1":accepted_m["macro_f1"],"accepted_weighted_f1":accepted_m["weighted_f1"]})
        per_class.extend(_per_class_rows(pred,truth,classes,arm=arm,epoch=int(epoch),scope="all_teacher_top1"))
        if mask.any(): per_class.extend(_per_class_rows(pred[mask],truth[mask],classes,arm=arm,epoch=int(epoch),scope="accepted_teacher_top1"))
        flow_rows.extend(_critical_flow_rows(pred,truth,classes,arm=arm,epoch=int(epoch),scope="all")); flow_rows.extend(_critical_flow_rows(pred[mask],truth[mask],classes,arm=arm,epoch=int(epoch),scope="accepted") if mask.any() else [])
        fixed_flow_rows.extend(_fixed_cohort_rows(ids,posterior,mask,fixed_cohorts,arm=arm,epoch=int(epoch)))
        cm=np.zeros((num_classes,num_classes),dtype=np.int64)
        for y,predv in zip(truth.tolist(),pred.tolist()): cm[int(y),int(predv)]+=1
        for y in range(num_classes):
            for q in range(num_classes): confusion_rows.append({"arm":arm,"epoch":int(epoch),"true_class":y,"pred_class":q,"count":int(cm[y,q])})

    warmup_refresh=exp/"warmup"/"pseudo_refresh"/"refresh_000.npz"
    if warmup_refresh.is_file(): audit_refresh(warmup_refresh,arm="COMMON_WARMUP",epoch=0)
    for arm in PHASE_ARMS:
        for path in sorted((exp/"arms"/arm/"pseudo_refresh").glob("refresh_*.npz")):
            audit_refresh(path,arm=arm,epoch=int(path.stem.split("_")[-1]))
    _write_csv(oracle_dir/"pseudo_label_trajectory.csv",pseudo_summary); _write_csv(oracle_dir/"pseudo_label_per_class.csv",per_class); _write_csv(oracle_dir/"critical_error_flow_trajectory.csv",flow_rows); _write_csv(oracle_dir/"critical_fixed_cohort_trajectory.csv",fixed_flow_rows); _write_csv(oracle_dir/"pseudo_label_confusion_long.csv",confusion_rows)

    val_rows=[]
    stage1_path=exp/"stage1_reference"/"target_val_unlabeled.npz"
    if stage1_path.is_file():
        with np.load(stage1_path,allow_pickle=False) as z: ids=z["sample_id"].astype(np.int64); posterior=z["posterior"].astype(np.float64)
        truth=_aligned_truth(ids,val_labels); val_rows.append({"arm":"STAGE1","epoch":0,"role":"student",**_metrics(posterior.argmax(axis=1),truth,num_classes)})
    for path in sorted((exp/"warmup"/"target_val_unlabeled").glob("epoch_*.npz")):
        role="teacher" if path.stem.endswith("teacher") else "student"; epoch=int(path.stem.split("_")[1])
        with np.load(path,allow_pickle=False) as z: ids=z["sample_id"].astype(np.int64); posterior=z["posterior"].astype(np.float64)
        truth=_aligned_truth(ids,val_labels); val_rows.append({"arm":"COMMON_WARMUP","epoch":epoch,"role":role,**_metrics(posterior.argmax(axis=1),truth,num_classes)})
    for arm in PHASE_ARMS:
        for path in sorted((exp/"arms"/arm/"target_val_unlabeled").glob("epoch_*.npz")):
            role="teacher" if path.stem.endswith("teacher") else "student"; epoch=int(path.stem.split("_")[1])
            with np.load(path,allow_pickle=False) as z: ids=z["sample_id"].astype(np.int64); posterior=z["posterior"].astype(np.float64)
            truth=_aligned_truth(ids,val_labels); m=_metrics(posterior.argmax(axis=1),truth,num_classes); val_rows.append({"arm":arm,"epoch":epoch,"role":role,**m})
    _write_csv(oracle_dir/"target_val_metrics.csv",val_rows)

    phase_rows=[]; phase_class_rows=[]; oracle_estimate=None
    oracle_cache=args.oracle_registration_cache.resolve() if args.oracle_registration_cache is not None else None
    first_phase_path=exp/"warmup/phase_refresh/phase_001.pt"; k_reg=int(torch.load(first_phase_path,map_location="cpu",weights_only=False)["actual_gamma"].numel())
    if oracle_cache is not None and oracle_cache.is_file():
        oracle_estimate=_load_oracle_phase(oracle_cache,num_classes=num_classes,k_reg=k_reg,config=config,train_labels=train_labels)
        torch.save({"protocol":PROTOCOL,"gamma":oracle_estimate.gamma,"class_centers":{int(c):g for c,g in oracle_estimate.class_centers},"source":"target-train true-class registration cache; oracle-only"},oracle_dir/"oracle_shared_phase.pt")
    for arm,path in _phase_files(exp):
        payload=torch.load(path,map_location="cpu",weights_only=False); gamma=payload["actual_gamma"]; epoch=int(payload["phase_epoch"])
        row={"arm":arm,"phase_epoch":epoch,"distance_to_identity":float(payload["distance_to_identity"]),"distance_to_previous":float(payload["distance_to_previous"]),"applied":bool(payload["applied"]),"estimate_valid":payload.get("estimate_valid"),"distance_to_oracle":float("nan") if oracle_estimate is None else phase_distance_value(gamma,oracle_estimate.gamma)}; phase_rows.append(row)
        centers=payload.get("class_centers",{})
        oracle_centers={} if oracle_estimate is None else dict(oracle_estimate.class_centers)
        for cid,center in centers.items():
            cid=int(cid); phase_class_rows.append({"arm":arm,"phase_epoch":epoch,"class_id":cid,"class_name":classes[cid],"distance_class_center_to_shared":phase_distance_value(center,gamma),"distance_class_center_to_oracle_class":float("nan") if cid not in oracle_centers else phase_distance_value(center,oracle_centers[cid])})
    _write_csv(oracle_dir/"phase_trajectory_oracle.csv",phase_rows); _write_csv(oracle_dir/"phase_class_center_oracle.csv",phase_class_rows)

    _plots(exp,phase_rows,pseudo_summary,val_rows)
    _json_dump(oracle_dir/"oracle_evaluation_manifest.json",{"protocol":PROTOCOL,"target_truth_use":"post-hoc target-train pseudo-label audit and target-val metrics only","oracle_phase_cache":None if oracle_cache is None else str(oracle_cache),"oracle_phase_computed":oracle_estimate is not None,"checkpoint_selection":False,"training_mutated":False})
    print(f"PHASE14_ORACLE_EVALUATION_COMPLETE|pseudo_rows={len(pseudo_summary)}|target_val_rows={len(val_rows)}|oracle_phase={str(oracle_estimate is not None).lower()}|output={oracle_dir}",flush=True)


if __name__=="__main__": main()
