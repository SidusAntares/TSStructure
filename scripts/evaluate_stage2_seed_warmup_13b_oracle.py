#!/usr/bin/env python3
"""Oracle-only post-hoc evaluator for experiment 13B.

Run only after diagnose_stage2_seed_warmup_13b.py prints
SEED13B_UNLABELED_COMPLETE.  Target truth is used here for diagnosis only and
never feeds training, T0, checkpoint selection, EMA or sample weighting.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import sklearn.metrics
import torch

import visualize_stage2_phase_alignment as phasevis

ORACLE_NOTE = "oracle-only diagnostic，不参与训练或无监督决策"


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields=[]; seen=set()
    for row in rows:
        for key in row:
            if key not in seen: fields.append(key); seen.add(key)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def _load_csv(path: Path) -> list[dict]:
    with path.open("r",encoding="utf-8",newline="") as f: return [dict(r) for r in csv.DictReader(f)]


def _metric(y: np.ndarray, pred: np.ndarray) -> dict:
    if y.size==0: return {"n":0,"accuracy":float("nan"),"macro_f1":float("nan"),"weighted_f1":float("nan")}
    return {"n":int(y.size),"accuracy":float(sklearn.metrics.accuracy_score(y,pred)),"macro_f1":float(sklearn.metrics.f1_score(y,pred,average="macro",zero_division=0)),"weighted_f1":float(sklearn.metrics.f1_score(y,pred,average="weighted",zero_division=0))}


def _labels_for(data_root: str, domain: str, classes: list[str], indices, runtime: dict) -> dict[int,int]:
    ds=phasevis._metadata_dataset(data_root,domain,classes,set(int(v) for v in indices),closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    return {int(p):int(y) for p,y in zip(ds.get_parcel_indices().tolist(),ds.get_labels().tolist())}


def _aligned_labels(ids: np.ndarray, mapping: dict[int,int]) -> np.ndarray:
    if set(map(int,ids.tolist())) != set(mapping): raise ValueError("saved scan and oracle metadata cover different samples")
    return np.asarray([mapping[int(v)] for v in ids],dtype=np.int64)


def _load_scan(path: Path) -> dict:
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def _class_rows(y: np.ndarray, pred: np.ndarray, classes: list[str], *, arm: str, epoch: int, role: str) -> list[dict]:
    precision,recall,f1,support=sklearn.metrics.precision_recall_fscore_support(y,pred,labels=np.arange(len(classes)),zero_division=0)
    return [{"arm":arm,"checkpoint_epoch":epoch,"model_role":role,"class_id":c,"class_name":classes[c],"precision":float(precision[c]),"recall":float(recall[c]),"f1":float(f1[c]),"support":int(support[c]),"pred_count":int(np.sum(pred==c)),"oracle_use":ORACLE_NOTE} for c in range(len(classes))]


def _cohort_specs(classes: list[str]):
    name_to_id={name:i for i,name in enumerate(classes)}
    requested=[
        ("barley_to_oat","spring_barley","spring_oat"),
        ("rye_to_triticale","winter_rye","winter_triticale"),
        ("wheat_to_triticale","winter_wheat","winter_triticale"),
        ("barley_to_barley","spring_barley","spring_barley"),
        ("oat_to_oat","spring_oat","spring_oat"),
        ("triticale_to_triticale","winter_triticale","winter_triticale"),
        ("rye_to_rye","winter_rye","winter_rye"),
        ("wheat_to_wheat","winter_wheat","winter_wheat"),
    ]
    return [(key,name_to_id[t],name_to_id[p]) for key,t,p in requested if t in name_to_id and p in name_to_id]


def _critical_rows(*, y, initial_pred, initial_post, current_post, arm, epoch, role, classes):
    pred=current_post.argmax(axis=1); rows=[]
    for key,true_id,initial_id in _cohort_specs(classes):
        mask=(y==true_id)&(initial_pred==initial_id); n=int(mask.sum())
        if n==0: continue
        p_true=current_post[mask,true_id]; p_initial=current_post[mask,initial_id]
        d_true=p_true-initial_post[mask,true_id]; d_initial=p_initial-initial_post[mask,initial_id]
        current=pred[mask]
        absorber_name = {"spring_barley":"spring_oat","winter_rye":"winter_triticale","winter_wheat":"winter_triticale"}.get(classes[true_id])
        absorber_id = classes.index(absorber_name) if absorber_name in classes else None
        rows.append({
            "cohort":key,"arm":arm,"checkpoint_epoch":epoch,"model_role":role,"n":n,
            "true_class":classes[true_id],"initial_candidate_class":classes[initial_id],
            "mean_true_prob":float(np.mean(p_true)),"median_true_prob":float(np.median(p_true)),
            "mean_initial_candidate_prob":float(np.mean(p_initial)),"median_initial_candidate_prob":float(np.median(p_initial)),
            "mean_delta_correct_prob":float(np.mean(d_true)),"median_delta_correct_prob":float(np.median(d_true)),
            "mean_delta_initial_candidate_prob":float(np.mean(d_initial)),"median_delta_initial_candidate_prob":float(np.median(d_initial)),
            "top1_true_fraction":float(np.mean(current==true_id)),"top1_initial_candidate_fraction":float(np.mean(current==initial_id)),
            "top1_other_wrong_fraction":float(np.mean((current!=true_id)&(current!=initial_id))),
            "named_absorber_class":absorber_name or "",
            "top1_named_absorber_fraction":float(np.mean(current==absorber_id)) if absorber_id is not None else float("nan"),
            "correct_direction_fraction":float(np.mean((d_true>0)&(d_initial<0))) if true_id!=initial_id else float("nan"),
            "oracle_use":ORACLE_NOTE,
        })
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument("--experiment-dir",type=Path,required=True); p.add_argument("--calibration-checkpoint",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); args=p.parse_args()
    root=args.experiment_dir.resolve(); config=json.loads((root/"00_stage2_warmup_config.json").read_text(encoding="utf-8"))
    if not bool(config.get("source_stream_identical_between_arms")): raise RuntimeError("13B unlabeled training did not finish or source stream mismatch")
    required=[root/"03_target_trajectory_unlabeled.csv",root/"02_stage1_target_reference_unlabeled.npz"]
    if any(not x.is_file() for x in required): raise FileNotFoundError("13B unlabeled trajectory is incomplete")
    calibration=torch.load(args.calibration_checkpoint.resolve(),map_location="cpu",weights_only=False); runtime=dict(calibration.get("runtime_config") or {})
    classes=[str(v) for v in runtime["classes"]]; source=str(runtime["source"]); target=str(runtime["target"]); seed=int(runtime["seed"]); fold=int(config["fold"])
    source_all=phasevis._eligible_parcels(str(args.data_root),source,classes,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    target_all=phasevis._eligible_parcels(str(args.data_root),target,classes,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    splits=phasevis._reconstruct_fold_splits(source_all,target_all,source=source,target=target,seed=seed,val_ratio=float(runtime.get("val_ratio",0.1)),test_ratio=float(runtime.get("test_ratio",0.2)),fold=fold)
    train_labels=_labels_for(str(args.data_root),target,classes,splits[target]["train"],runtime); val_labels=_labels_for(str(args.data_root),target,classes,splits[target]["val"],runtime)
    seed_rows=_load_csv(root/"01_initial_seed_manifest.csv"); seed_by={int(r["sample_id"]):int(r["pseudo_label"]) for r in seed_rows}
    initial=_load_scan(root/"02_stage1_target_reference_unlabeled.npz"); ids=initial["sample_id"].astype(np.int64); y=_aligned_labels(ids,train_labels); initial_post=initial["posterior"].astype(np.float64); initial_pred=initial_post.argmax(axis=1); seed_member=np.asarray([int(v) in seed_by for v in ids],dtype=bool); seed_correct=np.asarray([seed_by.get(int(v),-1)==int(t) if int(v) in seed_by else False for v,t in zip(ids,y)],dtype=bool)

    oracle_rows=[]; metric_rows=[]; class_rows=[]; critical_rows=[]; group_rows=[]
    checkpoints=[int(v) for v in config["checkpoint_epochs"]]
    for arm in ("source_only","main"):
        for epoch in checkpoints:
            scans={role:_load_scan(root/"target_trajectory_unlabeled"/arm/f"epoch_{epoch:03d}_{role}.npz") for role in ("student","teacher")}
            for role,scan in scans.items():
                if not np.array_equal(scan["sample_id"].astype(np.int64),ids): raise ValueError("target train ordering drifted")
                post=scan["posterior"].astype(np.float64); pred=post.argmax(axis=1)
                for group,mask in (("all",np.ones(len(y),dtype=bool)),("seed",seed_member),("non_seed",~seed_member)):
                    m=_metric(y[mask],pred[mask]); metric_rows.append({"scope":"target_train","group":group,"arm":arm,"checkpoint_epoch":epoch,"model_role":role,**m,"oracle_use":ORACLE_NOTE})
                    initially_wrong=(initial_pred!=y)&mask
                    if np.any(initially_wrong):
                        row=np.arange(len(y)); dcorrect=post[row,y]-initial_post[row,y]; dwrong=post[row,initial_pred]-initial_post[row,initial_pred]
                        group_rows.append({"group":group,"arm":arm,"checkpoint_epoch":epoch,"model_role":role,"n_initially_wrong":int(initially_wrong.sum()),"mean_delta_correct_prob":float(np.mean(dcorrect[initially_wrong])),"median_delta_correct_prob":float(np.median(dcorrect[initially_wrong])),"mean_delta_initial_wrong_prob":float(np.mean(dwrong[initially_wrong])),"median_delta_initial_wrong_prob":float(np.median(dwrong[initially_wrong])),"correct_direction_fraction":float(np.mean((dcorrect[initially_wrong]>0)&(dwrong[initially_wrong]<0))),"current_correct_fraction":float(np.mean(pred[initially_wrong]==y[initially_wrong])),"oracle_use":ORACLE_NOTE})
                class_rows.extend(_class_rows(y,pred,classes,arm=arm,epoch=epoch,role=role)); critical_rows.extend(_critical_rows(y=y,initial_pred=initial_pred,initial_post=initial_post,current_post=post,arm=arm,epoch=epoch,role=role,classes=classes))
            s=scans["student"]; t=scans["teacher"]; row_idx=np.arange(len(y)); sp=s["posterior"].astype(np.float64); tp=t["posterior"].astype(np.float64)
            for i,sid in enumerate(ids.tolist()):
                init_wrong=int(initial_pred[i]); true=int(y[i])
                oracle_rows.append({"sample_id":int(sid),"arm":arm,"checkpoint_epoch":epoch,"true_label":true,"true_class_name":classes[true],"seed_member":bool(seed_member[i]),"seed_pseudo_label":int(seed_by.get(int(sid),-1)),"seed_correct":bool(seed_correct[i]) if seed_member[i] else "","initial_candidate":init_wrong,"initial_candidate_correct":bool(init_wrong==true),"student_current_top1":int(sp[i].argmax()),"student_current_correct":bool(sp[i].argmax()==true),"teacher_current_top1":int(tp[i].argmax()),"teacher_current_correct":bool(tp[i].argmax()==true),"student_delta_correct_prob":float(sp[i,true]-initial_post[i,true]),"student_delta_initial_wrong_prob":float(sp[i,init_wrong]-initial_post[i,init_wrong]) if init_wrong!=true else float("nan"),"teacher_delta_correct_prob":float(tp[i,true]-initial_post[i,true]),"teacher_delta_initial_wrong_prob":float(tp[i,init_wrong]-initial_post[i,init_wrong]) if init_wrong!=true else float("nan"),"oracle_use":ORACLE_NOTE})

            # Target-val generalization, fixed checkpoints only.
            for role in ("student","teacher"):
                v=_load_scan(root/"target_val_unlabeled"/arm/f"epoch_{epoch:03d}_{role}.npz"); vy=_aligned_labels(v["sample_id"].astype(np.int64),val_labels); vp=v["posterior"].argmax(axis=1); metric_rows.append({"scope":"target_val","group":"all","arm":arm,"checkpoint_epoch":epoch,"model_role":role,**_metric(vy,vp),"oracle_use":ORACLE_NOTE})
                sv=_load_scan(root/"source_val_trajectory"/arm/f"epoch_{epoch:03d}_{role}.npz"); sy=sv["true_label"].astype(np.int64); spred=sv["posterior"].argmax(axis=1); metric_rows.append({"scope":"source_val","group":"all","arm":arm,"checkpoint_epoch":epoch,"model_role":role,**_metric(sy,spred),"oracle_use":"source labels only"})

    # Fixed r=0 deltas; never select the best checkpoint.
    baseline={(r["scope"],r["group"],r["arm"],r["model_role"]):r["macro_f1"] for r in metric_rows if r["checkpoint_epoch"]==0}
    for r in metric_rows: r["delta_macro_f1_from_epoch0"]=float(r["macro_f1"]-baseline[(r["scope"],r["group"],r["arm"],r["model_role"])])
    # Main - source-only causal contrast at each pre-fixed checkpoint.
    lookup={(r["scope"],r["group"],r["checkpoint_epoch"],r["model_role"],r["arm"]):r for r in metric_rows}
    contrasts=[]
    for scope in ("target_train","target_val","source_val"):
        groups=("all","seed","non_seed") if scope=="target_train" else ("all",)
        for group in groups:
            for epoch in checkpoints:
                for role in ("student","teacher"):
                    a=lookup.get((scope,group,epoch,role,"main")); b=lookup.get((scope,group,epoch,role,"source_only"))
                    if a and b: contrasts.append({"scope":scope,"group":group,"checkpoint_epoch":epoch,"model_role":role,"main_minus_source_only_accuracy":float(a["accuracy"]-b["accuracy"]),"main_minus_source_only_macro_f1":float(a["macro_f1"]-b["macro_f1"]),"main_minus_source_only_weighted_f1":float(a["weighted_f1"]-b["weighted_f1"]),"oracle_use":ORACLE_NOTE if scope.startswith("target") else "source labels only"})

    oracle=root/"oracle"; oracle.mkdir(parents=True,exist_ok=True)
    _write_csv(oracle/"target_trajectory_oracle.csv",oracle_rows); _write_csv(oracle/"model_trajectory_metrics.csv",metric_rows); _write_csv(oracle/"class_trajectory.csv",class_rows); _write_csv(oracle/"critical_flow_trajectory.csv",critical_rows); _write_csv(oracle/"seed_nonseed_trajectory.csv",group_rows); _write_csv(oracle/"main_vs_source_only_contrast.csv",contrasts)
    summary={"protocol":config["protocol"],"oracle_use":ORACLE_NOTE,"checkpoint_selection":"none; all epochs pre-fixed before oracle join","target_truth_used_in_training":False,"seed_manifest_hash":config["seed_manifest_sha256"],"outputs":["target_trajectory_oracle.csv","model_trajectory_metrics.csv","class_trajectory.csv","critical_flow_trajectory.csv","seed_nonseed_trajectory.csv","main_vs_source_only_contrast.csv"],"automatic_13B_verdict":None}
    (oracle/"13b_oracle_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"SEED13B_ORACLE_COMPLETE|target_truth_use={ORACLE_NOTE}|best_epoch_selection=false",flush=True)

if __name__=="__main__": main()
