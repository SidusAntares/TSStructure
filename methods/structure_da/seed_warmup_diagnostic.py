"""Pure helpers for experiment 13B bootstrap-strategy short adaptation.

13B compares *initial* target pseudo-label acquisition strategies under one
identical short semantic trainer.  No helper in this module accepts target true
labels.  The only exception in the full experiment is the explicitly separate
CTRL_ORACLE upper-bound arm, constructed after all label-free manifests have
already been written to disk.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .stage2_parameter_policy import Stage2ParameterPolicy


BOOTSTRAP_ARMS = (
    "CTRL_SOURCE",
    "PL_TIMEMATCH_CONF",
    "PL_DAPL_BOOT",
    "PL_IPL_BOOT",
    "PL_TFDA_NN",
    "PL_CONF_GEOM",
    "CTRL_ORACLE",
)
LABEL_FREE_BOOTSTRAP_ARMS = BOOTSTRAP_ARMS[1:6]

FORBIDDEN_SEED_FIELDS = {
    "true_label",
    "true_class_name",
    "candidate_correct",
    "seed_correct",
    "confusion_flow",
    "oracle_precision",
    "oracle_f1",
}


@dataclass(frozen=True)
class FixedSeedRecord:
    sample_id: int
    pseudo_label: int
    selection_evidence: str


def configure_13b_semantic_student(model: nn.Module) -> Stage2ParameterPolicy:
    """Train PSE + complete raw temporal encoder + classifier; freeze geometry."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    try:
        pse = model.backbone.pixel_set_encoder
        raw = model.temporal_module.raw_encoder
        classifier = model.classifier
        model.backbone.decomposition
        model.temporal_module.trend_geometry
        model.temporal_module.structure_geometry
    except AttributeError as error:
        raise ValueError("model does not expose the 13B semantic/geometry modules") from error

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (pse, raw, classifier):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        (trainable if parameter.requires_grad else frozen).append(name)

    all_names = {name for name, _ in model.named_parameters()}
    if set(trainable) & set(frozen) or set(trainable) | set(frozen) != all_names:
        raise RuntimeError("13B parameter policy does not partition model parameters")
    if not any(name.startswith("backbone.pixel_set_encoder.") for name in trainable):
        raise RuntimeError("13B requires trainable PSE parameters")
    forbidden_prefixes = (
        "backbone.decomposition.",
        "temporal_module.trend_geometry.",
        "temporal_module.structure_geometry.",
    )
    if any(name.startswith(forbidden_prefixes) for name in trainable):
        raise RuntimeError("13B geometry/decomposition parameters must remain frozen")
    return Stage2ParameterPolicy(tuple(trainable), tuple(frozen))


def class_balanced_seed_cross_entropy(logits: Tensor, pseudo_labels: Tensor) -> Tensor:
    """Mean CE per present pseudo-label class, then mean across present classes."""
    if logits.ndim != 2 or logits.shape[0] < 1:
        raise ValueError("logits must have shape [B,C] with B>0")
    if pseudo_labels.shape != (logits.shape[0],) or pseudo_labels.dtype != torch.long:
        raise ValueError("pseudo_labels must have shape [B] and dtype torch.long")
    if torch.any(pseudo_labels < 0).item() or torch.any(pseudo_labels >= logits.shape[1]).item():
        raise ValueError("pseudo_labels contain an invalid class id")
    per_sample = F.cross_entropy(logits, pseudo_labels, reduction="none")
    class_losses = []
    for class_id in torch.unique(pseudo_labels, sorted=True):
        class_losses.append(per_sample[pseudo_labels == class_id].mean())
    if not class_losses:
        raise ValueError("target seed batch is empty")
    return torch.stack(class_losses).mean()


def normalize_seed_manifest_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    valid_sample_ids: Iterable[int],
    initial_candidates: Mapping[int, int] | None,
    num_classes: int,
    require_initial_candidate_match: bool = True,
) -> tuple[FixedSeedRecord, ...]:
    """Validate a label-free bootstrap manifest without using target truth.

    ``require_initial_candidate_match`` remains True by default for strategies
    whose label source is the frozen classifier.  TFDA-NN is intentionally
    allowed to assign a neighborhood-aggregated label different from raw top-1.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer >=2")
    valid = {int(v) for v in valid_sample_ids}
    seen: set[int] = set()
    normalized: list[FixedSeedRecord] = []
    for row in rows:
        leaked = FORBIDDEN_SEED_FIELDS.intersection(str(key) for key in row.keys())
        if leaked:
            raise ValueError("seed manifest contains forbidden oracle fields: " + ",".join(sorted(leaked)))
        if "sample_id" not in row or "pseudo_label" not in row:
            raise ValueError("seed manifest requires sample_id and pseudo_label")
        sample_id = int(row["sample_id"])
        pseudo_label = int(row["pseudo_label"])
        if sample_id in seen:
            raise ValueError(f"duplicate seed sample_id {sample_id}")
        seen.add(sample_id)
        if sample_id not in valid:
            raise ValueError(f"seed sample_id {sample_id} is not in target-train")
        if not 0 <= pseudo_label < num_classes:
            raise ValueError(f"seed pseudo_label {pseudo_label} is outside class range")
        if require_initial_candidate_match:
            if initial_candidates is None or sample_id not in initial_candidates:
                raise ValueError(f"initial Stage-1 candidate missing for seed {sample_id}")
            if int(initial_candidates[sample_id]) != pseudo_label:
                raise ValueError(
                    f"seed {sample_id} pseudo_label={pseudo_label} relabels Stage-1 candidate "
                    f"{initial_candidates[sample_id]}"
                )
        selected = row.get("seed_selected", True)
        if isinstance(selected, str):
            selected = selected.strip().lower() not in {"0", "false", "no", ""}
        if not bool(selected):
            continue
        evidence = str(row.get("selection_evidence", row.get("selection_metadata", "fixed_bootstrap")))
        normalized.append(FixedSeedRecord(sample_id, pseudo_label, evidence))
    return tuple(sorted(normalized, key=lambda item: item.sample_id))


def seed_manifest_fingerprint(records: Sequence[FixedSeedRecord]) -> str:
    h = hashlib.sha256()
    for item in sorted(records, key=lambda row: row.sample_id):
        h.update(f"{item.sample_id}:{item.pseudo_label}:{item.selection_evidence}\n".encode("utf-8"))
    return h.hexdigest()


def _validate_bootstrap_arrays(sample_ids: Sequence[int], posterior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(sample_ids, dtype=np.int64)
    post = np.asarray(posterior, dtype=np.float64)
    if ids.ndim != 1 or post.ndim != 2 or post.shape[0] != ids.size or post.shape[1] < 2:
        raise ValueError("sample_ids/posterior must have shapes [N] and [N,C]")
    if len(np.unique(ids)) != ids.size or not np.isfinite(post).all():
        raise ValueError("bootstrap arrays must contain unique ids and finite posterior")
    return ids, post


def _records_from_mask(
    sample_ids: np.ndarray,
    pseudo_labels: np.ndarray,
    mask: np.ndarray,
    evidence: Sequence[str] | str,
) -> tuple[FixedSeedRecord, ...]:
    mask = np.asarray(mask, dtype=bool)
    labels = np.asarray(pseudo_labels, dtype=np.int64)
    if mask.shape != sample_ids.shape or labels.shape != sample_ids.shape:
        raise ValueError("bootstrap mask/labels must have shape [N]")
    if isinstance(evidence, str):
        evidence_values = [evidence] * sample_ids.size
    else:
        evidence_values = list(evidence)
        if len(evidence_values) != sample_ids.size:
            raise ValueError("selection evidence length mismatch")
    return tuple(
        FixedSeedRecord(int(sample_ids[i]), int(labels[i]), str(evidence_values[i]))
        for i in np.flatnonzero(mask)
    )


def timematch_confidence_bootstrap(
    sample_ids: Sequence[int], posterior: np.ndarray, *, threshold: float = 0.9
) -> tuple[FixedSeedRecord, ...]:
    """TimeMatch-style raw top-1 confidence bootstrap; no temporal shift."""
    ids, post = _validate_bootstrap_arrays(sample_ids, posterior)
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("confidence threshold must lie in [0,1]")
    pred = post.argmax(axis=1)
    conf = post[np.arange(ids.size), pred]
    evidence = [f"raw_top1_confidence={v:.8g}>threshold={float(threshold):.8g}" for v in conf]
    return _records_from_mask(ids, pred, conf > float(threshold), evidence)


def _diag_gaussian_score(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    floor = max(float(np.median(var[var > 0])) * 1e-6 if np.any(var > 0) else 0.0, 1e-8)
    safe_var = np.maximum(var, floor)
    return np.mean((x - mean) ** 2 / safe_var, axis=1)


def diagonal_gaussian_conformity_percentile(
    source_ids: Sequence[int],
    source_features: np.ndarray,
    source_labels: Sequence[int],
    target_features: np.ndarray,
    target_candidates: Sequence[int],
    *,
    num_classes: int,
    folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Source-only cross-fit diagonal-Gaussian conformity calibration.

    Source own-class scores are obtained by deterministic class-stratified
    cross-fit so a source sample is never scored by Gaussian statistics that
    contain itself.  Target candidates are then scored by full source-class
    Gaussian statistics and converted to source-class empirical percentiles.
    """
    sids = np.asarray(source_ids, dtype=np.int64)
    sx = np.asarray(source_features, dtype=np.float64)
    sy = np.asarray(source_labels, dtype=np.int64)
    tx = np.asarray(target_features, dtype=np.float64)
    tc = np.asarray(target_candidates, dtype=np.int64)
    if sx.ndim != 2 or tx.ndim != 2 or sx.shape[1] != tx.shape[1]:
        raise ValueError("source/target features must share [*,D]")
    if sy.shape != (sx.shape[0],) or sids.shape != sy.shape or tc.shape != (tx.shape[0],):
        raise ValueError("source/target metadata shape mismatch")
    if int(folds) < 2:
        raise ValueError("conformity cross-fit requires at least two folds")
    reference_score = np.full(sx.shape[0], np.nan, dtype=np.float64)
    target_score = np.full(tx.shape[0], np.nan, dtype=np.float64)

    for cid in range(int(num_classes)):
        idx = np.flatnonzero(sy == cid)
        if idx.size < int(folds) * 2:
            raise ValueError(f"source class {cid} is too small for {folds}-fold conformity calibration")
        # Stable fold assignment depends only on source sample id ordering.
        idx = idx[np.argsort(sids[idx], kind="stable")]
        fold_id = np.arange(idx.size, dtype=np.int64) % int(folds)
        for fold in range(int(folds)):
            held = idx[fold_id == fold]
            train = idx[fold_id != fold]
            mean = sx[train].mean(axis=0)
            var = sx[train].var(axis=0, ddof=1)
            reference_score[held] = _diag_gaussian_score(sx[held], mean, var)
        qmask = tc == cid
        if np.any(qmask):
            mean = sx[idx].mean(axis=0)
            var = sx[idx].var(axis=0, ddof=1)
            target_score[qmask] = _diag_gaussian_score(tx[qmask], mean, var)

    percentile = np.full(tx.shape[0], np.nan, dtype=np.float64)
    for cid in range(int(num_classes)):
        ref = np.sort(reference_score[(sy == cid) & np.isfinite(reference_score)])
        mask = (tc == cid) & np.isfinite(target_score)
        if ref.size and np.any(mask):
            percentile[mask] = np.searchsorted(ref, target_score[mask], side="right") / ref.size
    if not np.isfinite(target_score).all() or not np.isfinite(percentile).all():
        raise RuntimeError("source conformity calibration produced non-finite target values")
    return target_score, percentile


def dapl_bootstrap(
    sample_ids: Sequence[int],
    posterior: np.ndarray,
    conformity_percentile: Sequence[float],
    *,
    confidence_threshold: float = 0.9,
    conformity_threshold: float = 0.95,
) -> tuple[FixedSeedRecord, ...]:
    ids, post = _validate_bootstrap_arrays(sample_ids, posterior)
    percentile = np.asarray(conformity_percentile, dtype=np.float64)
    if percentile.shape != ids.shape or not np.isfinite(percentile).all():
        raise ValueError("conformity percentile must be finite [N]")
    pred = post.argmax(axis=1); conf = post[np.arange(ids.size), pred]
    mask = (conf > float(confidence_threshold)) & (percentile <= float(conformity_threshold))
    evidence = [
        f"confidence={conf[i]:.8g}>{float(confidence_threshold):.8g};diag_gaussian_source_percentile={percentile[i]:.8g}<={float(conformity_threshold):.8g}"
        for i in range(ids.size)
    ]
    return _records_from_mask(ids, pred, mask, evidence)


def cosine_prototype_labels(source_features: np.ndarray, source_labels: Sequence[int], target_features: np.ndarray, *, num_classes: int) -> np.ndarray:
    sx = np.asarray(source_features, dtype=np.float64); sy = np.asarray(source_labels, dtype=np.int64); tx = np.asarray(target_features, dtype=np.float64)
    if sx.ndim != 2 or tx.ndim != 2 or sx.shape[1] != tx.shape[1] or sy.shape != (sx.shape[0],):
        raise ValueError("prototype feature arrays are inconsistent")
    centers=[]
    for cid in range(int(num_classes)):
        cls=sx[sy==cid]
        if cls.size==0: raise ValueError(f"source class {cid} has no features")
        centers.append(cls.mean(axis=0))
    centers=np.asarray(centers,dtype=np.float64)
    tx=tx/np.maximum(np.linalg.norm(tx,axis=1,keepdims=True),1e-12)
    centers=centers/np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-12)
    return (tx@centers.T).argmax(axis=1).astype(np.int64)


def target_knn_candidate_support(raw_pred: Sequence[int], target_knn_indices: np.ndarray, *, k: int = 20) -> np.ndarray:
    pred=np.asarray(raw_pred,dtype=np.int64); nbr=np.asarray(target_knn_indices,dtype=np.int64)
    if nbr.ndim!=2 or nbr.shape[0]!=pred.size or int(k)<1 or int(k)>nbr.shape[1]:
        raise ValueError("invalid target KNN array/k")
    if np.any(nbr[:,:int(k)]<0) or np.any(nbr[:,:int(k)]>=pred.size):
        raise ValueError("target KNN indices outside target range")
    return np.mean(pred[nbr[:,:int(k)]]==pred[:,None],axis=1)


def ipl_bootstrap(
    sample_ids: Sequence[int], posterior: np.ndarray, prototype_labels: Sequence[int], target_knn_indices: np.ndarray,
    *, k: int = 20, support_threshold: float = 0.5,
) -> tuple[FixedSeedRecord, ...]:
    ids,post=_validate_bootstrap_arrays(sample_ids,posterior); pred=post.argmax(axis=1)
    proto=np.asarray(prototype_labels,dtype=np.int64)
    if proto.shape!=ids.shape: raise ValueError("prototype labels must have shape [N]")
    support=target_knn_candidate_support(pred,target_knn_indices,k=k)
    mask=(proto==pred)&(support>=float(support_threshold))
    evidence=[f"classifier={pred[i]};prototype={proto[i]};target_knn{k}_raw_candidate_fraction={support[i]:.8g}>={float(support_threshold):.8g}" for i in range(ids.size)]
    return _records_from_mask(ids,pred,mask,evidence)


def tfda_nn_bootstrap(sample_ids: Sequence[int], posterior: np.ndarray, target_knn_indices: np.ndarray, *, k: int = 20) -> tuple[FixedSeedRecord, ...]:
    """Assign every target sample the argmax of mean neighbor posterior."""
    ids,post=_validate_bootstrap_arrays(sample_ids,posterior); nbr=np.asarray(target_knn_indices,dtype=np.int64)
    if nbr.ndim!=2 or nbr.shape[0]!=ids.size or int(k)<1 or int(k)>nbr.shape[1]:
        raise ValueError("invalid target KNN array/k")
    if np.any(nbr[:,:int(k)]<0) or np.any(nbr[:,:int(k)]>=ids.size):
        raise ValueError("target KNN indices outside target range")
    mean_post=post[nbr[:,:int(k)]].mean(axis=1); pseudo=mean_post.argmax(axis=1)
    confidence=mean_post[np.arange(ids.size),pseudo]
    evidence=[f"target_knn{k}_mean_posterior_argmax={pseudo[i]};aggregated_probability={confidence[i]:.8g}" for i in range(ids.size)]
    return _records_from_mask(ids,pseudo,np.ones(ids.size,dtype=bool),evidence)


def confidence_geometry_veto_bootstrap(
    sample_ids: Sequence[int], posterior: np.ndarray,
    t_registered_error_percentile: Sequence[float], s_registered_error_percentile: Sequence[float],
    *, confidence_threshold: float = 0.9, conflict_percentile: float = 0.95,
) -> tuple[FixedSeedRecord, ...]:
    ids,post=_validate_bootstrap_arrays(sample_ids,posterior); pred=post.argmax(axis=1); conf=post[np.arange(ids.size),pred]
    tp=np.asarray(t_registered_error_percentile,dtype=np.float64); sp=np.asarray(s_registered_error_percentile,dtype=np.float64)
    if tp.shape!=ids.shape or sp.shape!=ids.shape: raise ValueError("geometry percentiles must have shape [N]")
    finite=np.isfinite(tp)&np.isfinite(sp)
    # Missing geometry is not interpreted as a conflict; the observable remains
    # recorded and only a measured extreme source-calibrated anomaly can veto.
    conflict=finite&((tp>=float(conflict_percentile))|(sp>=float(conflict_percentile)))
    mask=(conf>float(confidence_threshold))&(~conflict)
    evidence=[f"confidence={conf[i]:.8g}>{float(confidence_threshold):.8g};T_reg_pct={tp[i]:.8g};S_reg_pct={sp[i]:.8g};conflict_cut={float(conflict_percentile):.8g};geometry_available={bool(finite[i])}" for i in range(ids.size)]
    return _records_from_mask(ids,pred,mask,evidence)


def validate_checkpoint_epochs(total_epochs: int, checkpoints: Sequence[int]) -> tuple[int, ...]:
    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int) or total_epochs < 1:
        raise ValueError("total_epochs must be a positive integer")
    values = tuple(int(v) for v in checkpoints)
    if not values or values[0] != 0 or values[-1] != total_epochs:
        raise ValueError("checkpoint schedule must start at 0 and end at total_epochs")
    if tuple(sorted(set(values))) != values:
        raise ValueError("checkpoint epochs must be strictly increasing and unique")
    if any(v < 0 or v > total_epochs for v in values):
        raise ValueError("checkpoint epoch outside training window")
    if len(values) < 4:
        raise ValueError("13B requires initial, early, middle and end checkpoints")
    return values


def posterior_margin(probabilities: Tensor) -> Tensor:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [N,C], C>=2")
    top2 = probabilities.topk(k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def posterior_entropy(probabilities: Tensor, eps: float = 1e-12) -> Tensor:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N,C]")
    q = probabilities.clamp_min(float(eps))
    return -(q * q.log()).sum(dim=1)


def cosine_feature_displacement(current: Tensor, initial: Tensor, eps: float = 1e-12) -> Tensor:
    if current.shape != initial.shape or current.ndim != 2:
        raise ValueError("current and initial features must share shape [N,D]")
    a = current / current.norm(dim=1, keepdim=True).clamp_min(float(eps))
    b = initial / initial.norm(dim=1, keepdim=True).clamp_min(float(eps))
    return 1.0 - (a * b).sum(dim=1)


def finite_hyperparameters(*, lr: float, weight_decay: float, lambda_target: float, ema_decay: float) -> None:
    for name, value in (("lr", lr), ("weight_decay", weight_decay), ("lambda_target", lambda_target)):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(lr) <= 0.0:
        raise ValueError("lr must be greater than zero")
    if not math.isfinite(float(ema_decay)) or not 0.0 <= float(ema_decay) < 1.0:
        raise ValueError("ema_decay must satisfy 0 <= decay < 1")
