"""Reusable hashing, EMA, distribution, and shift-selection utilities."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_int_sequence(values: Iterable[int]) -> str:
    h = hashlib.sha256()
    arr = np.asarray(sorted(int(v) for v in values), dtype="<i8")
    h.update(arr.tobytes())
    return h.hexdigest()


def hash_split_pair(source_ids: Iterable[int], target_ids: Iterable[int]) -> str:
    h = hashlib.sha256()
    h.update(b"source\0")
    h.update(bytes.fromhex(hash_int_sequence(source_ids)))
    h.update(b"target\0")
    h.update(bytes.fromhex(hash_int_sequence(target_ids)))
    return h.hexdigest()


def json_fingerprint(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key in sorted(value, key=lambda x: str(x)):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value[key], child))
        return out
    if isinstance(value, (list, tuple)):
        out = {}
        for i, item in enumerate(value):
            child = f"{prefix}[{i}]"
            out.update(_flatten(item, child))
        if not value:
            out[prefix] = []
        return out
    return {prefix: value}


def config_diff(reference: Mapping[str, object], current: Mapping[str, object]) -> list[dict]:
    """Return an explicit leaf-wise diff.  Missing fields are never hidden."""
    a, b = _flatten(reference), _flatten(current)
    rows: list[dict] = []
    for key in sorted(set(a) | set(b)):
        av = a.get(key, "<MISSING>")
        bv = b.get(key, "<MISSING>")
        if av != bv:
            rows.append({"field": key, "reference": av, "current": bv})
    return rows


def cosine_lr_trace(*, lr: float, total_steps: int) -> list[dict]:
    """Generate PyTorch CosineAnnealingLR's stepwise trace without model state."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    p = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam([p], lr=float(lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(total_steps), eta_min=0.0)
    rows = []
    for step in range(1, total_steps + 1):
        before = float(opt.param_groups[0]["lr"])
        opt.step(); sched.step()
        after = float(opt.param_groups[0]["lr"])
        rows.append({"step": step, "lr_before_scheduler": before, "lr_after_scheduler": after})
    return rows


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8")); h.update(str(value.dtype).encode("ascii")); h.update(value.numpy().tobytes())
    return h.hexdigest()


def tensor_checksum(value: Tensor | None) -> str:
    if value is None:
        return "none"
    x = value.detach().cpu().contiguous()
    h = hashlib.sha256(); h.update(str(x.dtype).encode("ascii")); h.update(str(tuple(x.shape)).encode("ascii")); h.update(x.numpy().tobytes())
    return h.hexdigest()


def state_difference(student: nn.Module, teacher: nn.Module) -> dict:
    sp, tp = dict(student.named_parameters()), dict(teacher.named_parameters())
    if set(sp) != set(tp):
        raise ValueError("Student/Teacher parameter names differ")
    sq = 0.0; max_abs = 0.0; per_parameter = []
    for name in sorted(sp):
        d = (sp[name].detach().float().cpu() - tp[name].detach().float().cpu())
        l2 = float(torch.linalg.vector_norm(d).item())
        ma = float(d.abs().max().item()) if d.numel() else 0.0
        sq += l2 * l2; max_abs = max(max_abs, ma)
        per_parameter.append({"name": name, "l2": l2, "max_abs": ma})
    sb, tb = dict(student.named_buffers()), dict(teacher.named_buffers())
    buffer_rows = []
    for name in sorted(set(sb) | set(tb)):
        if name not in sb or name not in tb:
            buffer_rows.append({"name": name, "status": "missing", "student_present": name in sb, "teacher_present": name in tb})
            continue
        a, b = sb[name].detach().cpu(), tb[name].detach().cpu()
        if a.is_floating_point() or a.is_complex():
            diff = a.float() - b.float(); l2 = float(torch.linalg.vector_norm(diff).item()); ma = float(diff.abs().max().item()) if diff.numel() else 0.0
            exact = bool(torch.equal(a, b))
        else:
            exact = bool(torch.equal(a, b)); l2 = float(torch.linalg.vector_norm((a.to(torch.float64)-b.to(torch.float64))).item()); ma = float((a.to(torch.float64)-b.to(torch.float64)).abs().max().item()) if a.numel() else 0.0
        kind = "other"
        if name.endswith("running_mean"): kind = "running_mean"
        elif name.endswith("running_var"): kind = "running_var"
        elif name.endswith("num_batches_tracked"): kind = "num_batches_tracked"
        buffer_rows.append({"name": name, "kind": kind, "l2": l2, "max_abs": ma, "exact_equal": exact, "student_dtype": str(a.dtype), "teacher_dtype": str(b.dtype)})
    bn_modes = []
    tm = dict(teacher.named_modules())
    for name, module in student.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            other = tm[name]
            bn_modes.append({
                "name": name,
                "student_training": bool(module.training),
                "teacher_training": bool(other.training),
                "student_track_running_stats": bool(module.track_running_stats),
                "teacher_track_running_stats": bool(other.track_running_stats),
            })
    return {
        "student_training": bool(student.training),
        "teacher_training": bool(teacher.training),
        "parameter_l2": math.sqrt(sq),
        "parameter_max_abs": max_abs,
        "per_parameter": per_parameter,
        "buffers": buffer_rows,
        "batchnorm_modes": bn_modes,
    }


@torch.no_grad()
def official_timematch_ema_update(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    """Match jnyborg/timematch update_ema_variables state_dict semantics.

    The original implementation iterates over *all* state_dict values, not
    named trainable parameters only.  This therefore includes registered BN
    buffers.  Integer buffers are computed in floating arithmetic then copied
    back with PyTorch's destination dtype conversion, matching copy_ semantics.
    """
    d = float(decay)
    if not math.isfinite(d) or not 0.0 <= d < 1.0:
        raise ValueError("EMA decay must satisfy 0 <= decay < 1")
    ss, ts = student.state_dict(), teacher.state_dict()
    if list(ss) != list(ts):
        raise ValueError("Student/Teacher state_dict order differs")
    for name in ss:
        model_v, ema_v = ss[name], ts[name]
        if model_v.is_floating_point() or model_v.is_complex():
            ema_v.copy_(d * ema_v + (1.0 - d) * model_v)
        else:
            mixed = d * ema_v.to(torch.float64) + (1.0 - d) * model_v.to(torch.float64)
            ema_v.copy_(mixed)
    teacher.eval()


def class_distribution(labels: Sequence[int] | np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("labels must be non-empty 1-D")
    return np.bincount(y, minlength=int(num_classes)).astype(np.float64) / float(y.size)


def select_shift_from_probabilities(
    probabilities: np.ndarray,
    shifts: Sequence[int],
    *,
    estimator: str,
    class_distribution_target: np.ndarray | None = None,
) -> tuple[int, list[dict]]:
    """Label-free IS/ENT/AM shift score used by official TimeMatch.

    probabilities has shape [N,S,C] and contains no target truth.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 3 or p.shape[1] != len(shifts) or p.shape[2] < 2:
        raise ValueError("probabilities must have shape [N,S,C]")
    if not np.isfinite(p).all() or p.shape[0] == 0:
        raise ValueError("probabilities must be finite and non-empty")
    eps = 1e-5; py = p.mean(axis=0); pred = p.argmax(axis=2)
    estimator = str(estimator).upper()
    inception = np.mean(np.sum(p * (np.log(p + eps) - np.log(py[None] + eps)), axis=2), axis=0)
    entropy = np.mean(np.sum(-p * np.log(p + eps), axis=2), axis=0)
    am = np.full(len(shifts), np.nan, dtype=np.float64)
    if estimator == "IS":
        score = inception; best = int(np.argmax(score)); direction = "max"
    elif estimator == "ENT":
        score = entropy; best = int(np.argmin(score)); direction = "min"
    elif estimator == "AM":
        if class_distribution_target is None:
            raise ValueError("AM requires previous pseudo-label class distribution")
        c = np.asarray(class_distribution_target, dtype=np.float64)
        if c.shape != (p.shape[2],):
            raise ValueError("class distribution shape mismatch")
        one_hot_py = np.zeros_like(py)
        for j in range(len(shifts)):
            one_hot_py[j] = np.bincount(pred[:, j], minlength=p.shape[2]) / float(p.shape[0])
        kl = np.sum(c[None] * (np.log(c[None] + eps) - np.log(one_hot_py + eps)), axis=1)
        am = kl + entropy; score = am; best = int(np.argmin(score)); direction = "min"
    else:
        raise ValueError(f"unsupported TimeMatch shift estimator {estimator!r}")
    rows = []
    for j, shift in enumerate(shifts):
        rows.append({
            "shift_days": int(shift), "estimator": estimator, "selection_direction": direction,
            "inception_score": float(inception[j]), "entropy": float(entropy[j]),
            "am_score": float(am[j]) if np.isfinite(am[j]) else None,
            "selected": bool(j == best),
        })
    return int(shifts[best]), rows


def write_json(path: str | Path, payload: object) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
