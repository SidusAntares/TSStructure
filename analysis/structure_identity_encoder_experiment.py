"""Side-effect-light experiment primitives for source-only 07B retrieval."""
from __future__ import annotations

import contextlib
import copy
import math
import random
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from analysis.multivariate_local_waveform_scan_diagnostic import relative_queries as _queries


def relative_queries(segment: Mapping, points: int = 32, period_days: float = 365.0):
    return _queries(segment, points=points, period_days=period_days)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_sample_split(rows: Sequence[Mapping], seed: int = 1,
                            train_ratio: float = .70, validation_ratio: float = .15):
    by_class = defaultdict(set)
    for row in rows:
        by_class[int(row["class_id"])].add(int(row["sample_id"]))
    result = {"train": [], "validation": [], "test": []}
    for class_id in sorted(by_class):
        ids = np.asarray(sorted(by_class[class_id]), dtype=np.int64)
        rng = np.random.default_rng(seed + class_id)
        rng.shuffle(ids)
        count = len(ids)
        train_stop = int(math.floor(count * train_ratio))
        val_stop = train_stop + int(math.floor(count * validation_ratio))
        result["train"].extend(ids[:train_stop].tolist())
        result["validation"].extend(ids[train_stop:val_stop].tolist())
        result["test"].extend(ids[val_stop:].tolist())
    return {key: sorted(value) for key, value in result.items()}


def assign_groups_to_split(groups: Sequence[Mapping], split: Mapping[str, Sequence[int]]):
    lookup = {int(sample): name for name, samples in split.items() for sample in samples}
    output = {name: [] for name in split}
    for group in groups:
        sample_id = int(group["sample_id"])
        if sample_id not in lookup:
            raise ValueError(f"sample absent from split: {sample_id}")
        output[lookup[sample_id]].append(group)
    return output


def _stack_rows(rows: Sequence[Mapping], key: str, width: int):
    values = [np.asarray(row[key], dtype=np.float64).reshape(-1, width) for row in rows]
    values = [value for value in values if len(value)]
    if not values:
        return np.zeros((1, width), dtype=np.float64)
    return np.concatenate(values, axis=0)


def _moments(values):
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return mean, np.maximum(std, 1e-6)


def fit_normalization_statistics(train_instances: Sequence[Mapping]):
    amplitude = _stack_rows(train_instances, "amplitude", np.asarray(train_instances[0]["amplitude"]).shape[-1])
    events = _stack_rows(train_instances, "events", 4)
    fine = _stack_rows(train_instances, "fine", 6)
    a_mean, a_std = _moments(amplitude)
    e_mean, e_std = _moments(events)
    f_mean, f_std = _moments(fine)
    return {
        "amplitude_mean": a_mean.tolist(), "amplitude_std": a_std.tolist(),
        "event_mean": e_mean.tolist(), "event_std": e_std.tolist(),
        "fine_mean": f_mean.tolist(), "fine_std": f_std.tolist(),
    }


def build_candidate_group(reference_id, positive_id, direction, candidates,
                          sample_id, class_id):
    pool = [dict(row) for row in candidates
            if bool(row.get("accepted")) and str(row.get("direction")) == str(direction)]
    pool.sort(key=lambda row: (row.get("segment_id", row.get("coarse_segment_id", ""))))
    id_key = "segment_id" if pool and "segment_id" in pool[0] else "coarse_segment_id"
    ids = [str(row[id_key]) for row in pool]
    if str(positive_id) not in ids:
        raise ValueError("G+ positive is absent from accepted same-direction candidates")
    positive = ids.index(str(positive_id))
    if positive:
        pool.insert(0, pool.pop(positive)); ids.insert(0, ids.pop(positive)); positive = 0
    return dict(reference_structure_id=str(reference_id), sample_id=int(sample_id),
                class_id=int(class_id), candidates=pool, candidate_ids=ids,
                positive_id=str(positive_id), positive_index=positive)


@dataclass
class PaddedCandidateEmbeddings:
    embeddings: torch.Tensor
    mask: torch.Tensor
    positive_index: torch.Tensor


def pad_candidate_embeddings(groups: Sequence[torch.Tensor], positive_indices: Sequence[int]):
    if not groups:
        raise ValueError("empty candidate batch")
    maximum, width = max(len(group) for group in groups), groups[0].shape[-1]
    output = groups[0].new_zeros((len(groups), maximum, width))
    mask = torch.zeros((len(groups), maximum), dtype=torch.bool, device=output.device)
    for index, group in enumerate(groups):
        output[index, :len(group)] = group
        mask[index, :len(group)] = True
    return PaddedCandidateEmbeddings(output, mask,
                                     torch.as_tensor(positive_indices, device=output.device))


def cosine_candidate_scores(reference: torch.Tensor, candidates: torch.Tensor):
    return torch.einsum("bd,bnd->bn", F.normalize(reference, dim=-1),
                        F.normalize(candidates, dim=-1))


def listwise_ranking_loss(scores, candidate_mask, positive_index, temperature=.07):
    eligible = candidate_mask.sum(dim=1) >= 2
    if not bool(eligible.any()):
        return scores.sum() * 0.0, 0
    logits = scores[eligible] / float(temperature)
    logits = logits.masked_fill(~candidate_mask[eligible], -torch.inf)
    return F.cross_entropy(logits, positive_index[eligible]), int(eligible.sum().item())


def positive_consistency_loss(reference, positive):
    return (1.0 - F.cosine_similarity(reference, positive, dim=-1)).mean()


def structure_identity_loss(scores, candidate_mask, positive_index, reference,
                            positive, temperature=.07, lambda_pos=.1):
    rank, rank_groups = listwise_ranking_loss(scores, candidate_mask, positive_index, temperature)
    consistency = positive_consistency_loss(reference, positive)
    return {"loss": rank + float(lambda_pos) * consistency,
            "rank_loss": rank, "positive_loss": consistency,
            "rank_groups": rank_groups}


def retrieval_rows(groups: Sequence[Mapping], score_arrays: Sequence[np.ndarray]):
    output = []
    for group, raw_scores in zip(groups, score_arrays):
        scores = np.asarray(raw_scores, dtype=np.float64)
        ids = list(group["candidate_ids"])
        positive_index = ids.index(str(group["positive_id"]))
        order = sorted(range(len(ids)), key=lambda index: (-scores[index], ids[index]))
        rank = order.index(positive_index) + 1
        top = order[0]
        second = order[1] if len(order) > 1 else None
        negatives = [scores[i] for i in range(len(scores)) if i != positive_index]
        output.append(dict(group_id=group.get("group_id", ""), sample_id=group.get("sample_id", ""),
            reference_structure_id=group.get("reference_structure_id", ""), positive_id=group["positive_id"],
            selected_id=ids[top], exact=top == positive_index, positive_rank=rank, top2=rank <= 2,
            mrr=1.0 / rank, pairwise_win_rate=float(np.mean(scores[positive_index] > negatives)) if negatives else np.nan,
            top1_similarity=float(scores[top]), margin=float(scores[top] - scores[second]) if second is not None else np.nan,
            positive_margin=float(scores[positive_index] - max(negatives)) if negatives else np.nan,
            unique_best=bool(sum(np.isclose(scores, scores[top])) == 1), num_candidates=len(ids)))
    return output


def calibrate_rejection_thresholds(rows: Sequence[Mapping], minimum_precision=.95):
    if not rows:
        raise ValueError("validation rows are empty")
    similarities = sorted({float(row["top1_similarity"]) for row in rows})
    margins = sorted({float(row["margin"]) for row in rows if np.isfinite(row["margin"])})
    candidates = []
    for similarity in similarities:
        for margin in margins:
            accepted = [row for row in rows if float(row["top1_similarity"]) >= similarity
                        and float(row["margin"]) >= margin]
            if not accepted:
                continue
            precision = float(np.mean([bool(row["exact"]) for row in accepted]))
            if precision >= float(minimum_precision):
                candidates.append((len(accepted), precision, -similarity, -margin, similarity, margin))
    if not candidates:
        return dict(similarity_threshold=float("inf"), margin_threshold=float("inf"),
                    accepted=0, precision=float("nan"), coverage=0.0,
                    minimum_precision=float(minimum_precision))
    count, precision, _, _, similarity, margin = max(candidates)
    return dict(similarity_threshold=float(similarity), margin_threshold=float(margin),
                accepted=int(count), precision=float(precision), coverage=count / len(rows),
                minimum_precision=float(minimum_precision))


def apply_rejection_thresholds(rows: Sequence[Mapping], thresholds: Mapping):
    output = []
    for row in rows:
        item = dict(row)
        item["accepted"] = bool(float(row["top1_similarity"]) >= float(thresholds["similarity_threshold"])
                                and float(row["margin"]) >= float(thresholds["margin_threshold"]))
        output.append(item)
    return output


def evaluate_with_frozen_thresholds(rows: Sequence[Mapping], thresholds: Mapping):
    evaluated = apply_rejection_thresholds(rows, thresholds)
    accepted = [row for row in evaluated if row["accepted"]]
    correct = sum(bool(row["exact"]) for row in accepted)
    return dict(precision=correct / len(accepted) if accepted else np.nan,
                coverage=len(accepted) / len(evaluated) if evaluated else np.nan,
                accepted_correct=correct, accepted_wrong=len(accepted) - correct,
                rejected=len(evaluated) - len(accepted), rows=evaluated)


def summarize_retrieval(rows: Sequence[Mapping]):
    multi = [row for row in rows if int(row.get("num_candidates", 0)) >= 2]
    if not multi:
        return {key: np.nan for key in ("top1", "top2", "mrr", "pairwise", "positive_margin", "unique_best")}
    return dict(top1=float(np.mean([row["exact"] for row in multi])),
                top2=float(np.mean([row["top2"] for row in multi])),
                mrr=float(np.mean([row["mrr"] for row in multi])),
                pairwise=float(np.nanmean([row["pairwise_win_rate"] for row in multi])),
                positive_margin=float(np.nanmean([row["positive_margin"] for row in multi])),
                unique_best=float(np.mean([row["unique_best"] for row in multi])))


def variant_packets(groups: Sequence[Mapping], variants: Sequence[str]):
    return {name: copy.deepcopy(list(groups)) for name in variants}


def validate_train_reference(reference: Mapping):
    valid_events = len(reference.get("events", ())) > 0
    valid_fine = len(reference.get("fine", ())) > 0
    return {"valid_reference": bool(valid_events and valid_fine),
            "invalid_reason": "" if valid_events and valid_fine else
            ("empty_events" if not valid_events else "empty_fine")}


@contextlib.contextmanager
def staged_source_output(final: Path, required: Sequence[str]):
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".tmp_{final.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        yield staging
        missing = [name for name in required
                   if not (staging / name).is_file() or not (staging / name).stat().st_size]
        if missing:
            raise RuntimeError("incomplete 07B output: " + ", ".join(missing))
        backup = final.parent / f".old_{final.name}_{uuid.uuid4().hex}"
        if final.exists():
            final.replace(backup)
        try:
            staging.replace(final)
        except Exception:
            if backup.exists() and not final.exists():
                backup.replace(final)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
