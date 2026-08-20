#!/usr/bin/env python3
"""Reusable TimeMatch loaders, scalar search, and pseudo-label scans.

This ports the official TimeMatch adaptation control flow onto the semantic model interface:
IS initial scalar shift, AM epoch refresh, shifted-source focal supervision,
online EMA-Teacher confidence pseudo labels on shifted target, native/strong
Student target training, stepwise cosine LR and state_dict-wide EMA.

Target truth is never loaded in this process.  Domain Phase/Shape are absent.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Iterable, Sequence

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import transforms

from dataset import GroupByShapesBatchSampler, PixelSetData, worker_init_fn
from methods.structure_da.timematch_utils import select_shift_from_probabilities
from transforms import Identity, Normalize, RandomSamplePixels, RandomSampleTimeSteps, ToTensor

class LabelFreeDataset(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        item=dict(self.base[i]);item.pop("label",None);return item

class RandomTemporalShiftMetadata:
    """Equivalent TimeMatch per-sample temporal augmentation without mutating canonical raw time."""
    def __init__(self,max_shift=60,p=1.0):self.max_shift=int(max_shift);self.p=float(p)
    def __call__(self,sample):
        shift=random.randint(-self.max_shift,self.max_shift) if random.random()<self.p else 0
        sample["temporal_aug_shift_days"]=int(shift);return sample

class TupleDataset(Dataset):
    def __init__(self,weak,strong):
        if len(weak)!=len(strong):raise ValueError("weak/strong dataset length mismatch")
        self.weak=weak;self.strong=strong
    def __len__(self):return len(self.weak)
    def __getitem__(self,i):return self.weak[i],self.strong[i]

class DeviceBatchLoader:
    def __init__(self,loader,device):self.loader=loader;self.device=device
    def __len__(self):return len(self.loader)
    def __iter__(self):
        for batch in self.loader:yield move(batch,self.device)

def write_csv(path:Path,rows:Sequence[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def set_seed(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def move(batch,device):return {k:(v.to(device) if isinstance(v,Tensor) else v) for k,v in batch.items()}

def _shifted_positions(backbone,shift_days,time_scale:float)->Tensor:
    base=backbone.normalized_positions
    if isinstance(shift_days,Tensor):
        s=shift_days.to(device=base.device,dtype=base.dtype)
        if s.ndim==1:s=s[:,None]
        if s.shape not in ((base.shape[0],1),base.shape):raise ValueError("per-sample scalar shift shape mismatch")
        delta=s/float(time_scale)
    else:delta=float(shift_days)/float(time_scale)
    out=base+delta
    return torch.where(backbone.time_mask,out,torch.zeros_like(out))

def forward_scalar(model,batch,*,shift_days=0.0,augmentation_shift_days=None):
    bb=model.forward_backbone(batch["pixels"],batch["valid_pixels"],batch["positions"],batch.get("extra"),time_mask=batch.get("time_mask"),compute_decomposition=False)
    total=shift_days
    if augmentation_shift_days is not None:
        aug=augmentation_shift_days
        if not isinstance(aug,Tensor):aug=torch.as_tensor(aug,device=bb.tokens.device)
        total=aug.to(bb.tokens.device,dtype=bb.tokens.dtype)+float(shift_days)
    pos=_shifted_positions(bb,total,float(model.backbone.time_scale))
    return model.forward_from_backbone(bb,batch["positions"],batch.get("extra"),temporal_positions_override=pos,return_geometry=False)

def shift_probability_cube(model,loader,device,shifts:Sequence[int],*,max_batches:int,shift_chunk:int=8)->np.ndarray:
    model.eval();parts=[]
    with torch.no_grad():
        for bi,raw in enumerate(loader):
            if bi>=max_batches:break
            batch=move(raw,device);bb=model.forward_backbone(batch["pixels"],batch["valid_pixels"],batch["positions"],batch.get("extra"),time_mask=batch.get("time_mask"),compute_decomposition=False)
            b,l,d=bb.tokens.shape; batch_parts=[]
            for start in range(0,len(shifts),shift_chunk):
                chunk=list(shifts[start:start+shift_chunk]);sc=len(chunk)
                latent=bb.tokens[:,None].expand(b,sc,l,d).reshape(b*sc,l,d)
                mask=bb.time_mask[:,None].expand(b,sc,l).reshape(b*sc,l)
                base=bb.normalized_positions[:,None].expand(b,sc,l)
                delta=torch.as_tensor(chunk,device=base.device,dtype=base.dtype)[None,:,None]/float(model.backbone.time_scale)
                pos=torch.where(bb.time_mask[:,None],base+delta,torch.zeros_like(base)).reshape(b*sc,l)
                raw=model.temporal_module.raw_encoder(latent=latent,positions=pos,mask=mask);logits=model.classifier(raw.fused_repr)
                probs=torch.softmax(logits.float(),dim=1).reshape(b,sc,-1);batch_parts.append(probs.cpu())
            parts.append(torch.cat(batch_parts,dim=1))
    if not parts:raise RuntimeError("TimeMatch shift estimator received no target batches")
    return torch.cat(parts,dim=0).numpy().astype(np.float64)

def estimate_shift(model,loader,device,*,min_shift,max_shift,sample_size,estimator,class_distr,shift_chunk,output,epoch_tag):
    shifts=list(range(int(min_shift),int(max_shift)+1));cube=shift_probability_cube(model,loader,device,shifts,max_batches=sample_size,shift_chunk=shift_chunk);selected,rows=select_shift_from_probabilities(cube,shifts,estimator=estimator,class_distribution_target=class_distr)
    for r in rows:r["epoch_tag"]=epoch_tag
    write_csv(output/"shift_scans"/f"{epoch_tag}.csv",rows);print(f"TM10_SHIFT|epoch_tag={epoch_tag}|estimator={estimator}|selected={selected:+d}|range={min_shift}:{max_shift}|samples={cube.shape[0]}",flush=True);return selected


def pseudo_labels_from_weak_loader(model,loader,device,shift_days:int)->np.ndarray:
    labels=[];model.eval()
    with torch.no_grad():
        for raw in loader:
            batch=move(raw,device);out=forward_scalar(model,batch,shift_days=shift_days);labels.extend(out.logits.argmax(1).detach().cpu().tolist())
    if not labels:raise RuntimeError("TimeMatch initial pseudo-label scan was empty")
    return np.asarray(labels,dtype=np.int64)

def scan_teacher_pseudo(model,loader,device,shift_days:int):
    ids=[];posts=[];model.eval()
    with torch.no_grad():
        for raw in loader:
            batch=move(raw,device);out=forward_scalar(model,batch,shift_days=shift_days);ids.append(batch["parcel_index"].detach().cpu());posts.append(torch.softmax(out.logits.float(),dim=1).cpu())
    ids=torch.cat(ids).long();post=torch.cat(posts).float();order=torch.argsort(ids);return {"sample_id":ids[order],"posterior":post[order]}

def scan_loader(data_root:str,domain:str,classes:Sequence[str],parcels:Iterable[int],runtime:dict,*,batch_size:int,num_workers:int,strip_label:bool)->DataLoader:
    base=PixelSetData(data_root=data_root,dataset_name=domain,classes=list(classes),transform=transforms.Compose([Identity(),Normalize(),ToTensor()]),indices=set(int(v) for v in parcels),with_extra=False,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    dataset=LabelFreeDataset(base) if strip_label else base
    sampler=GroupByShapesBatchSampler(base,batch_size,by_time=True,by_pixel_dim=True)
    return DataLoader(dataset=dataset,batch_sampler=sampler,num_workers=num_workers,pin_memory=torch.cuda.is_available(),worker_init_fn=worker_init_fn)

@torch.no_grad()
def scan_model(model,loader,device,*,allow_labels:bool=False)->dict:
    model.eval();ids=[];posteriors=[];features=[];labels=[]
    for raw in loader:
        if not allow_labels and "label" in raw:
            raise RuntimeError("target truth leaked into a label-free TimeMatch scan")
        batch=move(raw,device);out=model(batch["pixels"],batch["valid_pixels"],batch["positions"],batch.get("extra"),return_geometry=False)
        ids.append(batch["parcel_index"].detach().cpu().long());posteriors.append(torch.softmax(out.logits.detach().float(),dim=1).cpu());features.append(out.fused_repr.detach().float().cpu())
        if allow_labels:labels.append(batch["label"].detach().cpu().long())
    sample_id=torch.cat(ids);posterior=torch.cat(posteriors);feature=torch.cat(features);order=torch.argsort(sample_id)
    result={"sample_id":sample_id[order],"posterior":posterior[order],"feature":feature[order]}
    if allow_labels:result["label"]=torch.cat(labels)[order]
    return result

def eligible_parcels(data_root:str,domain:str,classes:Sequence[str],runtime:dict)->np.ndarray:
    dataset=PixelSetData(data_root=data_root,dataset_name=domain,classes=list(classes),transform=None,indices=None,with_extra=False,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    return dataset.get_parcel_indices()

def reconstruct_fold_splits(source_parcels:np.ndarray,target_parcels:np.ndarray,*,source:str,target:str,seed:int,val_ratio:float,test_ratio:float,fold:int)->dict:
    rng=random.Random(seed);requested=None
    for _ in range(fold+1):
        requested={}
        for name,raw in ((source,source_parcels),(target,target_parcels)):
            indices=[int(value) for value in raw.tolist()];n=len(indices);n_test=int(test_ratio*n);n_val=int(val_ratio*n);n_train=n-n_test-n_val;rng.shuffle(indices)
            requested[name]={"train":set(indices[:n_train]),"val":set(indices[n_train:n_train+n_val]),"test":set(indices[n_train+n_val:])}
    return requested

def selected_loader(data_root:str,domain:str,classes:Sequence[str],parcel_indices:np.ndarray,runtime:dict,*,batch_size:int,num_workers:int)->DataLoader:
    dataset=PixelSetData(data_root=data_root,dataset_name=domain,classes=list(classes),transform=transforms.Compose([Identity(),Normalize(),ToTensor()]),indices=set(int(value) for value in parcel_indices.tolist()),with_extra=False,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    return DataLoader(dataset=dataset,batch_sampler=GroupByShapesBatchSampler(dataset,batch_size,by_time=True,by_pixel_dim=True),num_workers=num_workers,pin_memory=torch.cuda.is_available(),worker_init_fn=worker_init_fn)

def registration_config(runtime:dict,workers:int):
    def required(name):
        value=runtime.get(name)
        if value is None:raise ValueError(f"runtime_config is missing {name}")
        return float(value)
    return SimpleNamespace(registration_lambda=required("stage2_registration_lambda"),registration_gain_ratio_max=required("stage2_registration_gain_ratio_max"),registration_min_common_support=required("stage2_registration_min_common_support"),registration_max_roughness=required("stage2_registration_max_roughness"),registration_min_increment=required("stage2_registration_min_increment"),registration_max_local_speed=required("stage2_registration_max_local_speed"),registration_max_deviation=required("stage2_registration_max_deviation"),class_hypothesis_margin=required("stage2_class_hypothesis_margin"),k_reg=128,registration_workers=int(workers))

def make_loaders(runtime,splits,*,batch_size,num_workers,seed,seq_length,num_pixels,max_shift_aug,shift_aug_p,with_shift_aug):
    strong_parts=[RandomSamplePixels(num_pixels),RandomSampleTimeSteps(seq_length)]
    if with_shift_aug:strong_parts.append(RandomTemporalShiftMetadata(max_shift_aug,shift_aug_p))
    else:strong_parts.append(Identity())
    strong_parts += [Normalize(),ToTensor()]
    strong=transforms.Compose(strong_parts); weak=transforms.Compose([RandomSamplePixels(num_pixels),Normalize(),ToTensor()])
    source_ds=PixelSetData(str(runtime["data_root"]),str(runtime["source"]),list(runtime["classes"]),strong,indices=splits[str(runtime["source"])]["train"],with_extra=False,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    labels=source_ds.get_labels();freq=Counter(labels);weights=[1.0/freq[x] for x in labels];sampler=WeightedRandomSampler(weights,len(labels),replacement=True)
    source_loader=DataLoader(source_ds,batch_size=batch_size,sampler=sampler,drop_last=True,num_workers=num_workers,pin_memory=torch.cuda.is_available(),worker_init_fn=worker_init_fn)
    base=PixelSetData(str(runtime["data_root"]),str(runtime["target"]),list(runtime["classes"]),None,indices=splits[str(runtime["target"])]["train"],with_extra=False,closed_set=bool(runtime.get("closed_set",True)),combine_spring_and_winter=bool(runtime.get("combine_spring_and_winter",False)),time_coordinate_mode=str(runtime.get("time_coordinate_mode","canonical_day_of_year")))
    weak_ds=deepcopy(base);weak_ds.transform=weak;strong_ds=deepcopy(base);strong_ds.transform=strong
    # Strip target truth at Dataset boundary for both training and shift estimation.
    pair=TupleDataset(LabelFreeDataset(weak_ds),LabelFreeDataset(strong_ds));target_loader=DataLoader(pair,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=num_workers,pin_memory=torch.cuda.is_available(),worker_init_fn=worker_init_fn)
    noaug=deepcopy(base);noaug.transform=weak;noaug=LabelFreeDataset(noaug);shift_loader=DataLoader(noaug,batch_size=batch_size,shuffle=True,drop_last=False,num_workers=num_workers,pin_memory=torch.cuda.is_available(),worker_init_fn=worker_init_fn)
    return source_loader,shift_loader,target_loader
