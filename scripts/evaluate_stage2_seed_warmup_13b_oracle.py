#!/usr/bin/env python3
"""Oracle-only post-hoc evaluator for experiment 13B bootstrap comparison.

All five label-free bootstrap manifests and all fixed training trajectories must
already exist.  Target truth is used here to explain bootstrap quality and
model movement; it never changes a bootstrap rule, training loss, checkpoint,
or model state.  CTRL_ORACLE is explicitly treated as a supervised diagnostic
upper bound rather than a UDA arm.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Sequence

REPOSITORY_ROOT=Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path: sys.path.insert(0,str(REPOSITORY_ROOT))

import numpy as np
import sklearn.metrics
import torch

import visualize_stage2_phase_alignment as phasevis
from methods.structure_da.seed_warmup_diagnostic import BOOTSTRAP_ARMS, LABEL_FREE_BOOTSTRAP_ARMS

ORACLE_NOTE="oracle-only diagnostic，不参与训练或无监督决策"


def _write_csv(path:Path,rows:Sequence[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text("",encoding="utf-8"); return
    fields=[]; seen=set()
    for row in rows:
        for key in row:
            if key not in seen: fields.append(key); seen.add(key)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def _load_csv(path:Path)->list[dict]:
    if not path.is_file() or path.stat().st_size==0: return []
    with path.open("r",encoding="utf-8",newline="") as f: return [dict(r) for r in csv.DictReader(f)]


def _load_scan(path:Path)->dict:
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def _metric(y:np.ndarray,pred:np.ndarray)->dict:
    if y.size==0: return {"n":0,"accuracy":float("nan"),"macro_f1":float("nan"),"weighted_f1":float("nan")}
    return {"n":int(y.size),"accuracy":float(sklearn.metrics.accuracy_score(y,pred)),"macro_f1":float(sklearn.metrics.f1_score(y,pred,average="macro",zero_division=0)),"weighted_f1":float(sklearn.metrics.f1_score(y,pred,average="weighted",zero_division=0))}


def _labels_for(data_root:str,domain:str,classes:list[str],indices,runtime:dict)->dict[int,int]:
    ds=phasevis._metadata_dataset(data_root,domain,classes,set(int(v) for v in indices),closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    return {int(p):int(y) for p,y in zip(ds.get_parcel_indices().tolist(),ds.get_labels().tolist())}


def _aligned_labels(ids:np.ndarray,mapping:dict[int,int])->np.ndarray:
    if set(map(int,ids.tolist()))!=set(mapping): raise ValueError("saved scan and oracle metadata cover different samples")
    return np.asarray([mapping[int(v)] for v in ids],dtype=np.int64)


def _class_rows(y:np.ndarray,pred:np.ndarray,classes:list[str],*,arm:str,epoch:int,role:str)->list[dict]:
    precision,recall,f1,support=sklearn.metrics.precision_recall_fscore_support(y,pred,labels=np.arange(len(classes)),zero_division=0)
    return [{"arm":arm,"checkpoint_epoch":epoch,"model_role":role,"class_id":c,"class_name":classes[c],"precision":float(precision[c]),"recall":float(recall[c]),"f1":float(f1[c]),"support":int(support[c]),"pred_count":int(np.sum(pred==c)),"oracle_use":ORACLE_NOTE} for c in range(len(classes))]


def _cohort_specs(classes:list[str]):
    n={name:i for i,name in enumerate(classes)}; req=[("barley_to_oat","spring_barley","spring_oat"),("rye_to_triticale","winter_rye","winter_triticale"),("wheat_to_triticale","winter_wheat","winter_triticale"),("barley_to_barley","spring_barley","spring_barley"),("oat_to_oat","spring_oat","spring_oat"),("triticale_to_triticale","winter_triticale","winter_triticale"),("rye_to_rye","winter_rye","winter_rye"),("wheat_to_wheat","winter_wheat","winter_wheat")]
    return [(key,n[t],n[p]) for key,t,p in req if t in n and p in n]


def _critical_rows(*,y,initial_pred,initial_post,current_post,pseudo_member,arm,epoch,role,classes):
    pred=current_post.argmax(axis=1); rows=[]
    for key,true_id,initial_id in _cohort_specs(classes):
        base=(y==true_id)&(initial_pred==initial_id)
        for subset,extra in (("all",np.ones(len(y),dtype=bool)),("pseudo_labeled",pseudo_member),("non_pseudo_labeled",~pseudo_member)):
            mask=base&extra; n=int(mask.sum())
            if n==0: continue
            ptrue=current_post[mask,true_id]; pinit=current_post[mask,initial_id]; dtrue=ptrue-initial_post[mask,true_id]; dinit=pinit-initial_post[mask,initial_id]; cur=pred[mask]
            rows.append({"cohort":key,"cohort_subset":subset,"arm":arm,"checkpoint_epoch":epoch,"model_role":role,"n":n,"true_class":classes[true_id],"initial_candidate_class":classes[initial_id],"mean_true_prob":float(np.mean(ptrue)),"median_true_prob":float(np.median(ptrue)),"mean_initial_candidate_prob":float(np.mean(pinit)),"median_initial_candidate_prob":float(np.median(pinit)),"mean_delta_correct_prob":float(np.mean(dtrue)),"median_delta_correct_prob":float(np.median(dtrue)),"mean_delta_initial_candidate_prob":float(np.mean(dinit)),"median_delta_initial_candidate_prob":float(np.median(dinit)),"top1_true_fraction":float(np.mean(cur==true_id)),"top1_initial_candidate_fraction":float(np.mean(cur==initial_id)),"top1_other_wrong_fraction":float(np.mean((cur!=true_id)&(cur!=initial_id))),"correct_direction_fraction":float(np.mean((dtrue>0)&(dinit<0))) if true_id!=initial_id else float("nan"),"oracle_use":ORACLE_NOTE})
    return rows


def _manifest_for_arm(root:Path,config:dict,arm:str)->dict[int,int]:
    if arm=="CTRL_SOURCE": return {}
    if arm=="CTRL_ORACLE": path=Path(config["ctrl_oracle_manifest"])
    else: path=Path(config["bootstrap_manifests"][arm]["path"])
    if not path.is_absolute(): path=root/path
    rows=_load_csv(path)
    return {int(r["sample_id"]):int(r["pseudo_label"]) for r in rows}


def _trajectory_path(root:Path,arm:str,epoch:int,role:str)->Path:
    if arm=="CTRL_ORACLE": return root/"oracle_control/trajectory"/arm/f"epoch_{epoch:03d}_{role}.npz"
    return root/"target_trajectory_unlabeled"/arm/f"epoch_{epoch:03d}_{role}.npz"


def _target_val_path(root:Path,arm:str,epoch:int,role:str)->Path:
    if arm=="CTRL_ORACLE": return root/"oracle_control/target_val"/arm/f"epoch_{epoch:03d}_{role}.npz"
    return root/"target_val_unlabeled"/arm/f"epoch_{epoch:03d}_{role}.npz"


def main():
    p=argparse.ArgumentParser(); p.add_argument("--experiment-dir",type=Path,required=True); p.add_argument("--calibration-checkpoint",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); args=p.parse_args(); root=args.experiment_dir.resolve()
    config=json.loads((root/"00_stage2_bootstrap_config.json").read_text(encoding="utf-8"))
    if tuple(config.get("arms",[]))!=BOOTSTRAP_ARMS: raise RuntimeError("13B arm matrix is incomplete or reordered")
    if not bool(config.get("source_stream_identical_between_arms")): raise RuntimeError("13B training did not finish with identical source stream")
    if bool(config.get("transpl_included")): raise RuntimeError("TransPL must not be included in first-round 13B")
    required=[root/"03_target_trajectory_unlabeled.csv",root/"02_stage1_target_reference_unlabeled.npz",root/"01_bootstrap_strategy_manifest.json"]
    if any(not x.is_file() for x in required): raise FileNotFoundError("13B trajectory/bootstrap outputs are incomplete")

    calibration=torch.load(args.calibration_checkpoint.resolve(),map_location="cpu",weights_only=False); runtime=dict(calibration.get("runtime_config") or {}); classes=[str(v) for v in runtime["classes"]]; source=str(runtime["source"]); target=str(runtime["target"]); seed=int(runtime["seed"]); fold=int(config["fold"])
    source_all=phasevis._eligible_parcels(str(args.data_root),source,classes,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year"))); target_all=phasevis._eligible_parcels(str(args.data_root),target,classes,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    splits=phasevis._reconstruct_fold_splits(source_all,target_all,source=source,target=target,seed=seed,val_ratio=float(runtime.get("val_ratio",0.1)),test_ratio=float(runtime.get("test_ratio",0.2)),fold=fold); train_labels=_labels_for(str(args.data_root),target,classes,splits[target]["train"],runtime); val_labels=_labels_for(str(args.data_root),target,classes,splits[target]["val"],runtime)
    initial=_load_scan(root/"02_stage1_target_reference_unlabeled.npz"); ids=initial["sample_id"].astype(np.int64); y=_aligned_labels(ids,train_labels); initial_post=initial["posterior"].astype(np.float64); initial_pred=initial_post.argmax(axis=1); checkpoints=[int(v) for v in config["checkpoint_epochs"]]

    # First diagnose the frozen T0 produced by every strategy.  These values are
    # explanatory only: no strategy is skipped after this point.
    bootstrap_quality=[]; bootstrap_confusion=[]; manifest_by_arm={arm:_manifest_for_arm(root,config,arm) for arm in BOOTSTRAP_ARMS}
    id_to_pos={int(v):i for i,v in enumerate(ids.tolist())}
    for arm in BOOTSTRAP_ARMS:
        pseudo_by=manifest_by_arm[arm]; member=np.asarray([int(v) in pseudo_by for v in ids],dtype=bool); pseudo=np.asarray([pseudo_by.get(int(v),-1) for v in ids],dtype=np.int64); selected=pseudo[member]; truth=y[member]
        q=_metric(truth,selected) if selected.size else _metric(np.asarray([],dtype=np.int64),np.asarray([],dtype=np.int64))
        bootstrap_quality.append({"arm":arm,"n_selected":int(member.sum()),"coverage":float(member.mean()),"pseudo_label_precision":q["accuracy"],"pseudo_label_macro_f1":q["macro_f1"],"raw_candidate_relabel_fraction":float(np.mean(selected!=initial_pred[member])) if selected.size else float("nan"),"oracle_control":arm=="CTRL_ORACLE","oracle_use":ORACLE_NOTE})
        if selected.size:
            cm=sklearn.metrics.confusion_matrix(truth,selected,labels=np.arange(len(classes)))
            for true_id in range(len(classes)):
                for pseudo_id in range(len(classes)):
                    if cm[true_id,pseudo_id]: bootstrap_confusion.append({"arm":arm,"true_class_id":true_id,"true_class_name":classes[true_id],"pseudo_class_id":pseudo_id,"pseudo_class_name":classes[pseudo_id],"n":int(cm[true_id,pseudo_id]),"oracle_use":ORACLE_NOTE})

    oracle_rows=[]; metric_rows=[]; class_rows=[]; critical_rows=[]; movement_rows=[]
    for arm in BOOTSTRAP_ARMS:
        pseudo_by=manifest_by_arm[arm]; pseudo_member=np.asarray([int(v) in pseudo_by for v in ids],dtype=bool); pseudo_label=np.asarray([pseudo_by.get(int(v),-1) for v in ids],dtype=np.int64); pseudo_correct=np.asarray([(pseudo_by.get(int(v),-1)==int(t)) if int(v) in pseudo_by else False for v,t in zip(ids,y)],dtype=bool)
        for epoch in checkpoints:
            scans={role:_load_scan(_trajectory_path(root,arm,epoch,role)) for role in ("student","teacher")}
            for role,scan in scans.items():
                if not np.array_equal(scan["sample_id"].astype(np.int64),ids): raise ValueError("target train ordering drifted")
                post=scan["posterior"].astype(np.float64); pred=post.argmax(axis=1); row=np.arange(len(y)); dcorrect=post[row,y]-initial_post[row,y]; dwrong=post[row,initial_pred]-initial_post[row,initial_pred]
                for group,mask in (("all",np.ones(len(y),dtype=bool)),("pseudo_labeled",pseudo_member),("non_pseudo_labeled",~pseudo_member)):
                    metric_rows.append({"scope":"target_train","group":group,"arm":arm,"checkpoint_epoch":epoch,"model_role":role,**_metric(y[mask],pred[mask]),"oracle_use":ORACLE_NOTE})
                    iw=(initial_pred!=y)&mask
                    if np.any(iw): movement_rows.append({"group":group,"arm":arm,"checkpoint_epoch":epoch,"model_role":role,"n_initially_wrong":int(iw.sum()),"mean_delta_correct_prob":float(np.mean(dcorrect[iw])),"median_delta_correct_prob":float(np.median(dcorrect[iw])),"mean_delta_initial_wrong_prob":float(np.mean(dwrong[iw])),"median_delta_initial_wrong_prob":float(np.median(dwrong[iw])),"correct_direction_fraction":float(np.mean((dcorrect[iw]>0)&(dwrong[iw]<0))),"current_correct_fraction":float(np.mean(pred[iw]==y[iw])),"oracle_use":ORACLE_NOTE})
                class_rows.extend(_class_rows(y,pred,classes,arm=arm,epoch=epoch,role=role)); critical_rows.extend(_critical_rows(y=y,initial_pred=initial_pred,initial_post=initial_post,current_post=post,pseudo_member=pseudo_member,arm=arm,epoch=epoch,role=role,classes=classes))
            sp=scans["student"]["posterior"].astype(np.float64); tp=scans["teacher"]["posterior"].astype(np.float64)
            for i,sid in enumerate(ids.tolist()):
                true=int(y[i]); init=int(initial_pred[i]); oracle_rows.append({"sample_id":int(sid),"arm":arm,"checkpoint_epoch":epoch,"true_label":true,"true_class_name":classes[true],"pseudo_labeled":bool(pseudo_member[i]),"pseudo_label":int(pseudo_label[i]),"pseudo_label_correct":bool(pseudo_correct[i]) if pseudo_member[i] else "","initial_candidate":init,"initial_candidate_correct":bool(init==true),"student_current_top1":int(sp[i].argmax()),"student_current_correct":bool(sp[i].argmax()==true),"teacher_current_top1":int(tp[i].argmax()),"teacher_current_correct":bool(tp[i].argmax()==true),"student_delta_correct_prob":float(sp[i,true]-initial_post[i,true]),"student_delta_initial_wrong_prob":float(sp[i,init]-initial_post[i,init]) if init!=true else float("nan"),"teacher_delta_correct_prob":float(tp[i,true]-initial_post[i,true]),"teacher_delta_initial_wrong_prob":float(tp[i,init]-initial_post[i,init]) if init!=true else float("nan"),"oracle_use":ORACLE_NOTE})
            for role in ("student","teacher"):
                v=_load_scan(_target_val_path(root,arm,epoch,role)); vy=_aligned_labels(v["sample_id"].astype(np.int64),val_labels); metric_rows.append({"scope":"target_val","group":"all","arm":arm,"checkpoint_epoch":epoch,"model_role":role,**_metric(vy,v["posterior"].argmax(axis=1)),"oracle_use":ORACLE_NOTE}); sv=_load_scan(root/"source_val_trajectory"/arm/f"epoch_{epoch:03d}_{role}.npz"); metric_rows.append({"scope":"source_val","group":"all","arm":arm,"checkpoint_epoch":epoch,"model_role":role,**_metric(sv["true_label"].astype(np.int64),sv["posterior"].argmax(axis=1)),"oracle_use":"source labels only"})

    baseline={(r["scope"],r["group"],r["arm"],r["model_role"]):r["macro_f1"] for r in metric_rows if r["checkpoint_epoch"]==0}
    for r in metric_rows: r["delta_macro_f1_from_epoch0"]=float(r["macro_f1"]-baseline[(r["scope"],r["group"],r["arm"],r["model_role"])])
    lookup={(r["scope"],r["group"],r["checkpoint_epoch"],r["model_role"],r["arm"]):r for r in metric_rows}; contrasts=[]
    for arm in BOOTSTRAP_ARMS:
        if arm=="CTRL_SOURCE": continue
        for scope in ("target_train","target_val","source_val"):
            for epoch in checkpoints:
                for role in ("student","teacher"):
                    a=lookup.get((scope,"all",epoch,role,arm)); s=lookup.get((scope,"all",epoch,role,"CTRL_SOURCE")); o=lookup.get((scope,"all",epoch,role,"CTRL_ORACLE"))
                    if a and s and o: contrasts.append({"arm":arm,"scope":scope,"checkpoint_epoch":epoch,"model_role":role,"arm_minus_source_macro_f1":float(a["macro_f1"]-s["macro_f1"]),"arm_minus_source_accuracy":float(a["accuracy"]-s["accuracy"]),"arm_minus_oracle_macro_f1":float(a["macro_f1"]-o["macro_f1"]),"oracle_minus_source_macro_f1":float(o["macro_f1"]-s["macro_f1"]),"oracle_use":ORACLE_NOTE if scope.startswith("target") else "source labels only"})

    oracle=root/"oracle"; oracle.mkdir(parents=True,exist_ok=True); _write_csv(oracle/"bootstrap_strategy_quality.csv",bootstrap_quality); _write_csv(oracle/"bootstrap_strategy_confusion.csv",bootstrap_confusion); _write_csv(oracle/"target_trajectory_oracle.csv",oracle_rows); _write_csv(oracle/"model_trajectory_metrics.csv",metric_rows); _write_csv(oracle/"class_trajectory.csv",class_rows); _write_csv(oracle/"critical_flow_trajectory.csv",critical_rows); _write_csv(oracle/"pseudo_nonpseudo_movement.csv",movement_rows); _write_csv(oracle/"strategy_vs_controls.csv",contrasts)
    summary={"protocol":config["protocol"],"oracle_use":ORACLE_NOTE,"arms":list(BOOTSTRAP_ARMS),"checkpoint_selection":"none; all checkpoints fixed before oracle evaluation","target_test_used":False,"transpl_included":False,"static_pseudo_label_quality_is_explanatory_only":True,"automatic_13B_verdict":None}
    (oracle/"13b_oracle_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print("SEED13B_ORACLE_COMPLETE|groups=7|best_epoch_selection=false|transpl=false",flush=True)


if __name__=="__main__": main()
