"""Pure diagnostic helpers for sample-level Phase validity audits.

This module is intentionally outside the training/Domain-Phase decision path.
Stage-A registration legality is *T-only*: it receives only T registration
geometry and numerical warp constraints.  Shape/S-SRVF is evaluated later by
:func:`evaluate_shape_validation`, so it remains an independent diagnostic
signal rather than a hidden admission gate.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
import time
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from .phase_evidence import compute_gamma_diagnostics, empirical_cdf
from .phase_registration import (
    check_gamma_legality,
    resample_gamma,
    warp_q_gamma,
    warp_support_gamma,
)
from .prototype_bank import SourcePrototypeBank, support_aware_q_distance
from .registration_geometry import SourceRegistrationPrototypeBank, TargetGeometryCache
from .target_hypothesis_scan import PhaseHypothesisScanConfig


@dataclass(frozen=True)
class TRegistrationGeometryCache:
    """Trend-only target cache used to generate and gate sample Phase.

    The absence of any Structure/Shape field is deliberate: Stage-A Phase
    proposal and legality must not use S-SRVF, because S is reserved for
    independent post-hoc validation.
    """

    sample_ids: Tensor
    trend_srvf_reg: Tensor
    trend_support_reg: Tensor
    trend_valid: Tensor
    registration_grid: Tensor


@dataclass(frozen=True)
class TOnlyPhaseRegistration:
    sample_index: int
    sample_id: int
    class_id: int
    gamma: Tensor | None
    target_trend_valid: bool
    pre_common_support_t: float
    t_identity_error: float | None
    t_registered_error: float | None
    t_gain_ratio: float | None
    common_support_t: float | None
    gamma_finite: bool
    gamma_endpoint_error: float | None
    gamma_strictly_increasing: bool
    gamma_min_increment: float | None
    gamma_max_local_speed: float | None
    gamma_roughness: float | None
    phase_deviation: float | None
    numerically_valid: bool
    t_only_legal: bool
    reject_reasons: tuple[str, ...]
    solver_error: str | None = None


@dataclass(frozen=True)
class RawShapeValidation:
    sample_index: int
    sample_id: int
    class_id: int
    raw_shape_distance: float | None
    q_distance_percentile: float | None
    common_support_shape: float | None
    computable: bool


@dataclass(frozen=True)
class RawShapeSelection:
    sample_id: int
    selected_class_id: int | None
    selected_distance: float | None
    second_class_id: int | None
    second_distance: float | None
    margin: float | None
    selectable_class_ids: tuple[int, ...]


_WORKER_SOURCE_NP: np.ndarray | None = None
_WORKER_TARGET_NP: np.ndarray | None = None


def remap_local_sample_ids_to_parcels(
    local_sample_ids: Tensor,
    parcel_indices: Sequence[int],
) -> Tensor:
    """Map dataset-local sample indices back to stable parcel identities.

    ``TargetGeometryCache.sample_ids`` comes from
    ``target_hypothesis_scan._batch_sample_ids()``, which prefers the batch
    ``index`` field when both ``index`` and ``parcel_index`` are present.
    For ``PixelSetData`` selected subsets that ``index`` is the local dataset
    position, while downstream 06 diagnostics are keyed by stable
    ``parcel_index``. This helper makes that boundary explicit without
    changing the shared Stage-2 scanner contract.
    """
    local = local_sample_ids.detach().to(device="cpu", dtype=torch.long)
    if local.ndim != 1:
        raise ValueError("local_sample_ids must be one-dimensional")
    parcels = np.asarray(parcel_indices, dtype=np.int64)
    if parcels.ndim != 1:
        raise ValueError("parcel_indices must be one-dimensional")
    if local.numel() == 0:
        return local
    minimum = int(local.min().item())
    maximum = int(local.max().item())
    if minimum < 0 or maximum >= len(parcels):
        raise IndexError(
            "target geometry cache contains sample indices outside the selected dataset"
        )
    mapped = torch.from_numpy(parcels[local.numpy()].copy()).to(dtype=torch.long)
    if torch.unique(mapped).numel() != mapped.numel():
        raise ValueError("parcel_index identities must be unique within the selected dataset")
    return mapped


def trend_only_cache(cache: TargetGeometryCache) -> TRegistrationGeometryCache:
    return TRegistrationGeometryCache(
        sample_ids=cache.sample_ids.detach().cpu(),
        trend_srvf_reg=cache.trend_srvf_reg.detach().cpu(),
        trend_support_reg=cache.trend_support_reg.detach().cpu(),
        trend_valid=cache.trend_valid.detach().cpu(),
        registration_grid=cache.registration_grid.detach().cpu(),
    )


def _worker_init(source_np: np.ndarray, target_np: np.ndarray) -> None:
    global _WORKER_SOURCE_NP, _WORKER_TARGET_NP
    _WORKER_SOURCE_NP = source_np
    _WORKER_TARGET_NP = target_np
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _worker(payload: tuple[int, int, int, int, float]):
    sample_index, sample_id, class_id, target_local_index, lam = payload
    if _WORKER_SOURCE_NP is None or _WORKER_TARGET_NP is None:
        raise RuntimeError("sample Phase worker cache is unavailable")
    try:
        from fdasrsf import curve_functions as cf

        gamma_np = cf.optimum_reparam_curve(
            q1=_WORKER_SOURCE_NP[class_id],
            q2=_WORKER_TARGET_NP[target_local_index],
            lam=float(lam),
            method="DP",
        )
        gamma = np.asarray(gamma_np, dtype=np.float64)
        if gamma.ndim != 1:
            return sample_index, sample_id, class_id, None, "invalid_gamma_rank"
        return sample_index, sample_id, class_id, gamma, None
    except Exception as error:  # pragma: no cover - real solver faults
        return sample_index, sample_id, class_id, None, type(error).__name__


def _integration_weights(grid: Tensor) -> Tensor:
    weights = torch.ones_like(grid, dtype=torch.float64)
    if weights.numel() > 1:
        weights[[0, -1]] *= 0.5
    return weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)


def _common_support(a: Tensor, b: Tensor, weights: Tensor) -> float:
    return float((weights * torch.minimum(a.double(), b.double())).sum().item())


def _gamma_reasons(legality, config: PhaseHypothesisScanConfig) -> list[str]:
    reasons: list[str] = []
    if not legality.finite or legality.endpoint_error > 1e-6:
        reasons.append("gamma_endpoint")
    if (not legality.strictly_increasing) or legality.min_increment < config.registration_min_increment:
        reasons.append("gamma_increment")
    if legality.max_local_speed > config.registration_max_local_speed:
        reasons.append("gamma_speed")
    if legality.roughness > config.registration_max_roughness:
        reasons.append("gamma_roughness")
    if legality.phase_deviation > config.registration_max_deviation:
        reasons.append("gamma_deviation")
    return reasons


def solve_t_only_registrations(
    source_bank: SourceRegistrationPrototypeBank,
    target_cache: TRegistrationGeometryCache,
    assignments: Sequence[tuple[int, int]],
    config: PhaseHypothesisScanConfig,
    *,
    workers: int | None = None,
    progress_label: str = "SAMPLE_PHASE_DP",
    max_target_samples_per_pool: int = 512,
) -> tuple[TOnlyPhaseRegistration, ...]:
    """Solve selected sample×class registrations and apply *T-only* gates.

    ``assignments`` contains ``(sample_index, class_id)`` pairs.  This function
    has no Shape/S-SRVF input and therefore cannot use S as a proposal/admission
    signal.  It applies only target-T validity, T common support, warp numerical
    constraints, and the T registration-gain gate.
    """
    if not assignments:
        return ()
    source_ready = source_bank.ready.detach().cpu().bool()
    source_q = source_bank.trend_srvf.detach().cpu().double().contiguous()
    source_support = source_bank.trend_support.detach().cpu().double().contiguous()
    target_q = target_cache.trend_srvf_reg.detach().cpu().double().contiguous()
    target_support = target_cache.trend_support_reg.detach().cpu().double().contiguous()
    target_valid = target_cache.trend_valid.detach().cpu().bool()
    sample_ids = target_cache.sample_ids.detach().cpu().long()
    grid = target_cache.registration_grid.detach().cpu().double()
    weights = _integration_weights(grid)

    normalized: list[tuple[int, int]] = []
    for sample_index, class_id in assignments:
        sample_index = int(sample_index)
        class_id = int(class_id)
        if not 0 <= sample_index < target_q.shape[0]:
            raise IndexError("sample_index is outside target registration cache")
        if not 0 <= class_id < source_q.shape[0]:
            raise IndexError("class_id is outside source registration bank")
        if not bool(source_ready[class_id].item()):
            raise ValueError(f"source registration class {class_id} is not ready")
        normalized.append((sample_index, class_id))

    if max_target_samples_per_pool < 1:
        raise ValueError("max_target_samples_per_pool must be positive")
    source_np = np.ascontiguousarray(source_q.numpy().transpose(0, 2, 1))
    # Full target-test T geometry can exceed a gigabyte in float64.  Spawned
    # workers must therefore receive only a bounded sample chunk instead of a
    # copy of the complete target cache.
    unique_samples = list(dict.fromkeys(sample_index for sample_index, _class_id in normalized))
    assignment_by_sample: dict[int, list[int]] = {}
    for sample_index, class_id in normalized:
        assignment_by_sample.setdefault(sample_index, []).append(class_id)
    raw: list[tuple] = []
    started = time.monotonic()
    total_pairs = len(normalized)
    completed_total = 0
    for chunk_start in range(0, len(unique_samples), max_target_samples_per_pool):
        chunk_samples = unique_samples[chunk_start : chunk_start + max_target_samples_per_pool]
        target_local = {sample_index: local for local, sample_index in enumerate(chunk_samples)}
        stage_target_np = np.ascontiguousarray(
            target_q[chunk_samples].numpy().transpose(0, 2, 1)
        )
        payloads = []
        for sample_index in chunk_samples:
            sample_id = int(sample_ids[sample_index].item())
            for class_id in assignment_by_sample[sample_index]:
                payloads.append((
                    sample_index, sample_id, class_id, target_local[sample_index],
                    float(config.registration_lambda),
                ))
        worker_count = int(config.registration_workers if workers is None else workers)
        worker_count = max(1, min(worker_count, len(payloads), os.cpu_count() or 1))
        chunksize = max(1, len(payloads) // max(1, worker_count * 8))
        env_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        previous_env = {key: os.environ.get(key) for key in env_keys}
        for key in env_keys:
            os.environ[key] = "1"
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
                initializer=_worker_init,
                initargs=(source_np, stage_target_np),
            ) as executor:
                for item in executor.map(_worker, payloads, chunksize=chunksize):
                    raw.append(item)
                    completed_total += 1
            print(
                f"{progress_label}_PROGRESS|completed={completed_total}/{total_pairs}"
                f"|sample_chunk={len(chunk_samples)}|workers={worker_count}"
                f"|seconds={time.monotonic() - started:.2f}",
                flush=True,
            )
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    outputs: list[TOnlyPhaseRegistration] = []
    for sample_index, sample_id, class_id, gamma_np, solver_error in raw:
        target_is_valid = bool(target_valid[sample_index].item())
        pre_support = _common_support(source_support[class_id], target_support[sample_index], weights)
        if solver_error is not None or gamma_np is None:
            reasons = ["solver_failed"]
            if not target_is_valid:
                reasons.insert(0, "target_trend_invalid")
            if pre_support < config.registration_min_common_support:
                reasons.insert(0, "pre_support")
            outputs.append(TOnlyPhaseRegistration(
                sample_index=sample_index, sample_id=sample_id, class_id=class_id,
                gamma=None, target_trend_valid=target_is_valid,
                pre_common_support_t=pre_support,
                t_identity_error=None, t_registered_error=None, t_gain_ratio=None,
                common_support_t=None, gamma_finite=False, gamma_endpoint_error=None,
                gamma_strictly_increasing=False, gamma_min_increment=None,
                gamma_max_local_speed=None, gamma_roughness=None,
                phase_deviation=None, numerically_valid=False, t_only_legal=False,
                reject_reasons=tuple(reasons), solver_error=solver_error,
            ))
            continue

        gamma = torch.as_tensor(gamma_np, dtype=torch.float64, device="cpu").contiguous()
        legality = check_gamma_legality(
            gamma.float(), grid.float(),
            registration_min_increment=config.registration_min_increment,
            registration_max_local_speed=config.registration_max_local_speed,
            registration_max_roughness=config.registration_max_roughness,
            registration_max_deviation=config.registration_max_deviation,
        )
        numerically_valid = bool(
            legality.finite and legality.endpoint_error <= 1e-6 and legality.strictly_increasing
        )
        diagnostics = None
        if numerically_valid:
            diagnostics = compute_gamma_diagnostics(
                sample_id=sample_id,
                class_id=class_id,
                gamma=gamma.float(),
                source_trend_srvf=source_q[class_id].float(),
                target_trend_srvf=target_q[sample_index].float(),
                source_support=source_support[class_id].float(),
                target_support=target_support[sample_index].float(),
                integration_weights=weights.float(),
                registration_grid=grid.float(),
            )
        reasons: list[str] = []
        if not target_is_valid:
            reasons.append("target_trend_invalid")
        if pre_support < config.registration_min_common_support:
            reasons.append("pre_support")
        reasons.extend(_gamma_reasons(legality, config))
        if diagnostics is None or not math.isfinite(float(diagnostics.gain_ratio)):
            reasons.append("gain_unavailable")
        elif diagnostics.gain_ratio > config.registration_gain_ratio_max:
            reasons.append("gain")
        outputs.append(TOnlyPhaseRegistration(
            sample_index=sample_index, sample_id=sample_id, class_id=class_id,
            gamma=gamma, target_trend_valid=target_is_valid,
            pre_common_support_t=pre_support,
            t_identity_error=None if diagnostics is None else float(diagnostics.e_id),
            t_registered_error=None if diagnostics is None else float(diagnostics.e_reg),
            t_gain_ratio=None if diagnostics is None else float(diagnostics.gain_ratio),
            common_support_t=None if diagnostics is None else float(diagnostics.common_support),
            gamma_finite=bool(legality.finite),
            gamma_endpoint_error=float(legality.endpoint_error),
            gamma_strictly_increasing=bool(legality.strictly_increasing),
            gamma_min_increment=float(legality.min_increment),
            gamma_max_local_speed=float(legality.max_local_speed),
            gamma_roughness=float(legality.roughness),
            phase_deviation=float(legality.phase_deviation),
            numerically_valid=numerically_valid,
            t_only_legal=bool(numerically_valid and not reasons),
            reject_reasons=tuple(reasons), solver_error=None,
        ))
    return tuple(outputs)


def evaluate_shape_validation(
    registration: TOnlyPhaseRegistration,
    *,
    target_cache: TargetGeometryCache,
    source_bank: SourcePrototypeBank,
) -> RawShapeValidation:
    """Evaluate raw S-SRVF distance *after* T-only Phase generation/gating."""
    if registration.gamma is None or not registration.numerically_valid:
        return RawShapeValidation(
            sample_index=registration.sample_index, sample_id=registration.sample_id,
            class_id=registration.class_id, raw_shape_distance=None,
            q_distance_percentile=None, common_support_shape=None, computable=False,
        )
    sample_index = registration.sample_index
    class_id = registration.class_id
    shape_grid = target_cache.shape_grid.detach().cpu().double()
    reg_grid = target_cache.registration_grid.detach().cpu().double()
    gamma_shape = resample_gamma(registration.gamma, reg_grid, shape_grid)
    target_q = target_cache.structure_srvf_shape[sample_index].detach().cpu().float()
    target_support = target_cache.structure_support_shape[sample_index].detach().cpu().float()
    aligned_q = warp_q_gamma(target_q, gamma_shape.float()).squeeze(0)
    aligned_support = warp_support_gamma(target_support, gamma_shape.float(), shape_grid.float())
    proto = source_bank.shape_srvf[class_id].detach().cpu().float()
    proto_support = source_bank.shape_support[class_id].detach().cpu().float()
    weights = _integration_weights(shape_grid).float()
    distance = support_aware_q_distance(
        aligned_q.unsqueeze(0), proto.unsqueeze(0),
        aligned_support.unsqueeze(0), proto_support.unsqueeze(0), weights,
    )
    if not bool(distance.valid[0, 0].item()):
        return RawShapeValidation(
            sample_index=sample_index, sample_id=registration.sample_id,
            class_id=class_id, raw_shape_distance=None, q_distance_percentile=None,
            common_support_shape=float(distance.common_support[0, 0].item()), computable=False,
        )
    value = float(distance.distance[0, 0].item())
    samples = source_bank.q_distance_samples[class_id].detach().cpu().double()
    percentile = None
    if samples.numel() > 0:
        percentile = float(empirical_cdf(samples, torch.tensor([value], dtype=torch.float64))[0].item())
    return RawShapeValidation(
        sample_index=sample_index, sample_id=registration.sample_id,
        class_id=class_id, raw_shape_distance=value,
        q_distance_percentile=percentile,
        common_support_shape=float(distance.common_support[0, 0].item()), computable=True,
    )


def select_raw_shape_candidate(
    registrations: Iterable[TOnlyPhaseRegistration],
    shape_validations: Iterable[RawShapeValidation],
) -> RawShapeSelection:
    """Select the minimum *raw* S distance among T-only-legal candidates.

    No classifier, Teacher, pseudo-label history, class balance, percentile
    normalization, or distance calibration is used here.  This is intentionally
    the simplest Stage-B baseline diagnostic.
    """
    reg_by_key = {(r.sample_id, r.class_id): r for r in registrations}
    usable: list[tuple[float, int]] = []
    sample_ids: set[int] = set()
    for shape in shape_validations:
        sample_ids.add(int(shape.sample_id))
        reg = reg_by_key.get((shape.sample_id, shape.class_id))
        if reg is None or not reg.t_only_legal or not shape.computable:
            continue
        if shape.raw_shape_distance is None or not math.isfinite(float(shape.raw_shape_distance)):
            continue
        usable.append((float(shape.raw_shape_distance), int(shape.class_id)))
    if len(sample_ids) > 1:
        raise ValueError("raw-shape selection must receive one sample at a time")
    sample_id = next(iter(sample_ids)) if sample_ids else (
        int(next(iter(reg_by_key.values())).sample_id) if reg_by_key else -1
    )
    usable.sort(key=lambda item: (item[0], item[1]))
    if not usable:
        return RawShapeSelection(
            sample_id=sample_id, selected_class_id=None, selected_distance=None,
            second_class_id=None, second_distance=None, margin=None,
            selectable_class_ids=(),
        )
    best = usable[0]
    second = usable[1] if len(usable) > 1 else None
    return RawShapeSelection(
        sample_id=sample_id,
        selected_class_id=best[1], selected_distance=best[0],
        second_class_id=None if second is None else second[1],
        second_distance=None if second is None else second[0],
        margin=None if second is None else float(second[0] - best[0]),
        selectable_class_ids=tuple(class_id for _distance, class_id in usable),
    )


def phase_distance_matrix(gammas: Tensor) -> Tensor:
    """Vectorized Fisher--Rao distance matrix for diagnostic visualization."""
    if not isinstance(gammas, Tensor) or gammas.ndim != 2 or gammas.shape[1] < 2:
        raise ValueError("gammas must have shape [N,K] with K>=2")
    values = gammas.detach().cpu().double().contiguous()
    if values.shape[0] == 0 or not torch.isfinite(values).all().item():
        raise ValueError("gammas must be finite and non-empty")
    interval_count = values.shape[1] - 1
    derivative = torch.diff(values, dim=1) * interval_count
    if torch.any(derivative < -1e-12).item():
        raise ValueError("gammas must be monotonically nondecreasing")
    psi = torch.sqrt(derivative.clamp_min(0.0))
    step = 1.0 / interval_count
    norm = torch.sqrt((psi.square().sum(dim=1) * step).clamp_min(1e-15))
    psi = psi / norm[:, None]
    inner = (psi @ psi.T) * step
    distance = torch.acos(inner.clamp(-1.0, 1.0))
    distance.fill_diagonal_(0.0)
    return distance.detach()


def classical_mds(distance: Tensor, dimensions: int = 2) -> Tensor:
    """Classical metric MDS used only for visualization, never clustering."""
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    d = distance.detach().cpu().double()
    n = d.shape[0]
    if n == 0:
        return torch.empty((0, dimensions), dtype=torch.float64)
    if n == 1:
        return torch.zeros((1, dimensions), dtype=torch.float64)
    squared = d.square()
    eye_center = torch.eye(n, dtype=torch.float64) - torch.full((n, n), 1.0 / n, dtype=torch.float64)
    gram = -0.5 * eye_center @ squared @ eye_center
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    keep = min(dimensions, n)
    coords = eigenvectors[:, :keep] * torch.sqrt(eigenvalues[:keep].clamp_min(0.0))[None, :]
    if keep < dimensions:
        coords = torch.cat([coords, torch.zeros((n, dimensions - keep), dtype=coords.dtype)], dim=1)
    return coords.detach()
