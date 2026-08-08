"""Progressive all-class target evidence acquisition for Domain Phase.

For every representative target sample, this module performs the exact
vector-valued T-SRVF registration against every ready source class.  The same
class-conditioned gamma is then applied to target Shape geometry, and the raw
Shape distance defines the candidate pseudo-label.  Statistical gates decide
only whether the associated sample-class alignment is reliable enough to enter
Domain Phase estimation; they do not erase the candidate pseudo-label itself.

All successful pairwise gammas are cached so later Phase-state logic can reuse
them without rerunning fdasrsf.  No target ground-truth labels are read here.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
import math
import os
import platform
import time
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from .phase_evidence import compute_gamma_diagnostics, empirical_cdf
from .phase_registration import (
    FdasrsfCurveRegistrationAdapter,
    check_gamma_legality,
    resample_gamma,
    warp_q_gamma,
    warp_support_gamma,
)
from .prototype_bank import SourcePrototypeBank, support_aware_q_distance
from .registration_geometry import (
    SourceRegistrationPrototypeBank,
    TargetGeometryCache,
    evaluate_registration_geometry,
)
from .temporal_srvf import TemporalSRVFExtractor


@dataclass(frozen=True)
class PhaseHypothesisScanConfig:
    """Registration/Shape reliability gates for all-class evidence scanning.

    ``proposal_*`` fields are retained only for CLI/checkpoint compatibility
    with V6.  Round A deliberately ignores them: every ready source class is
    registered for every selected target sample.
    """

    registration_lambda: float
    registration_gain_ratio_max: float
    registration_min_common_support: float
    registration_max_roughness: float
    registration_min_increment: float
    registration_max_local_speed: float
    registration_max_deviation: float
    class_hypothesis_margin: float

    k_reg: int = 128
    class_hypothesis_max: int = 2
    proposal_classifier_topk: int = 2
    proposal_identity_topk: int = 2
    registration_workers: int = 1

    def __post_init__(self) -> None:
        if self.k_reg < 2:
            raise ValueError("k_reg must be at least 2")
        if self.class_hypothesis_max != 2:
            raise ValueError("class_hypothesis_max must equal 2")
        if self.class_hypothesis_margin < 0:
            raise ValueError("class_hypothesis_margin must be non-negative")
        # Deprecated V6 proposal settings remain type-checked for config
        # compatibility, but do not affect class selection in Round A.
        for name in ("proposal_classifier_topk", "proposal_identity_topk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.registration_workers, bool)
            or not isinstance(self.registration_workers, int)
            or self.registration_workers < 1
        ):
            raise ValueError("registration_workers must be a positive integer")


@dataclass(frozen=True)
class PairwiseClassAlignment:
    """Cached result of one target-sample × source-class exact registration.

    ``numerically_valid`` means the solver returned a finite, endpoint-valid,
    strictly increasing gamma that can be used to evaluate Shape.  The stricter
    ``phase_evidence_eligible`` flag additionally applies the configured
    support/deformation/gain and Shape-computability reliability gates.  Source
    q-range exceedance is retained as a diagnostic, not a Phase-discovery gate.
    """

    sample_id: int
    class_id: int
    gamma: Tensor | None

    t_identity_error: float | None
    t_registered_error: float | None
    t_gain_ratio: float | None
    pre_common_support_t: float
    common_support_t: float | None

    gamma_finite: bool
    gamma_endpoint_error: float | None
    gamma_strictly_increasing: bool
    gamma_min_increment: float | None
    gamma_max_local_speed: float | None
    gamma_roughness: float | None
    phase_deviation: float | None

    q_shape_distance: float | None
    q_distance_percentile: float | None
    common_support_shape: float | None

    numerically_valid: bool
    phase_evidence_eligible: bool
    reject_reasons: tuple[str, ...]
    solver_error: str | None = None


@dataclass(frozen=True)
class CandidatePseudoLabel:
    """Geometry-first pseudo-label defined by minimum raw aligned Shape distance."""

    sample_id: int
    class_id: int
    q_shape_distance: float
    q_distance_percentile: float | None
    phase_evidence_eligible: bool
    ambiguous: bool
    secondary_class_id: int | None = None
    secondary_q_shape_distance: float | None = None
    secondary_phase_evidence_eligible: bool | None = None


@dataclass(frozen=True)
class TargetClassPhaseHypothesis:
    sample_id: int
    class_id: int
    gamma: Tensor
    t_identity_error: float
    t_registered_error: float
    t_gain_ratio: float
    q_shape_distance: float
    q_distance_percentile: float
    common_support_t: float
    common_support_shape: float
    roughness: float
    phase_deviation: float
    preferred: bool
    ambiguous_class: bool
    evidence_weight: float


@dataclass(frozen=True)
class TargetHypothesisScanResult:
    hypotheses: tuple[TargetClassPhaseHypothesis, ...]
    num_samples: int
    num_pairwise_attempted: int
    num_pre_support_rejected: int
    num_solver_failed: int
    num_gamma_rejected: int
    num_gain_rejected: int
    num_shape_support_rejected: int
    num_outer_rejected: int
    samples_with_zero_hypothesis: int
    samples_with_one_hypothesis: int
    samples_with_two_hypotheses: int

    # Round-A geometry-first records.  Defaults preserve older hand-built test
    # fixtures while downstream rounds migrate to the richer evidence model.
    pairwise_alignments: tuple[PairwiseClassAlignment, ...] = ()
    candidate_pseudo_labels: tuple[CandidatePseudoLabel, ...] = ()

    # Extended runtime diagnostics.
    num_solver_calls: int = 0
    num_ready_classes: int = 0
    num_all_class_pairs: int = 0
    num_proposal_pairs: int = 0  # deprecated alias; equals all-class attempts
    num_gamma_endpoint_rejected: int = 0
    num_gamma_increment_rejected: int = 0
    num_gamma_speed_rejected: int = 0
    num_gamma_roughness_rejected: int = 0
    num_gamma_deviation_rejected: int = 0
    scanned_sample_ids: tuple[int, ...] = ()
    pairwise_class_counts: tuple[int, ...] = ()
    proposal_class_counts: tuple[int, ...] = ()  # deprecated alias
    diagnostic_quantiles: tuple[tuple[str, float, float, float], ...] = ()


@dataclass(frozen=True)
class _SolverTask:
    sample_index: int
    sample_id: int
    class_id: int


@dataclass(frozen=True)
class _SolverResult:
    sample_index: int
    sample_id: int
    class_id: int
    gamma: Tensor | None
    error_name: str | None


@dataclass(frozen=True)
class _CpuSourceScanCache:
    trend_srvf: Tensor
    trend_support: Tensor
    ready: Tensor
    trend_np: np.ndarray
    shape_srvf: Tensor
    shape_support: Tensor
    q_distance_samples: tuple[Tensor, ...]
    q_outer: Tensor


def _build_cpu_source_cache(
    stage1_bank: SourcePrototypeBank,
    source_reg_bank: SourceRegistrationPrototypeBank,
) -> _CpuSourceScanCache:
    trend_srvf = source_reg_bank.trend_srvf.detach().to(device="cpu", dtype=torch.float32).contiguous()
    trend_support = source_reg_bank.trend_support.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if source_reg_bank.ready.shape != stage1_bank.ready.shape:
        raise ValueError("source registration and Stage-1 ready masks must share shape")
    ready = (
        source_reg_bank.ready.detach().to(device="cpu", dtype=torch.bool)
        & stage1_bank.ready.detach().to(device="cpu", dtype=torch.bool)
    ).contiguous()
    trend_np = np.ascontiguousarray(
        trend_srvf.to(dtype=torch.float64).numpy().transpose(0, 2, 1)
    )
    shape_srvf = stage1_bank.shape_srvf.detach().to(device="cpu", dtype=torch.float32).contiguous()
    shape_support = stage1_bank.shape_support.detach().to(device="cpu", dtype=torch.float32).contiguous()
    q_distance_samples = tuple(
        item.detach().to(device="cpu", dtype=torch.float64).contiguous()
        for item in stage1_bank.q_distance_samples
    )
    q_outer = stage1_bank.q_quantiles[:, 2].detach().to(device="cpu", dtype=torch.float64).contiguous()
    return _CpuSourceScanCache(
        trend_srvf=trend_srvf,
        trend_support=trend_support,
        ready=ready,
        trend_np=trend_np,
        shape_srvf=shape_srvf,
        shape_support=shape_support,
        q_distance_samples=q_distance_samples,
        q_outer=q_outer,
    )


_WORKER_SOURCE_NP: np.ndarray | None = None
_WORKER_TARGET_NP: np.ndarray | None = None


def _registration_worker_init(source_np: np.ndarray, target_np: np.ndarray) -> None:
    # Children inherit OMP/MKL/OPENBLAS=1 before module import from the parent
    # spawn environment.  Keep explicit guards here too for non-BLAS code.
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


def _fdasrsf_worker(payload: tuple[int, int, int, int, float]):
    sample_index, sample_id, class_id, target_local_index, lam = payload
    if _WORKER_SOURCE_NP is None or _WORKER_TARGET_NP is None:
        raise RuntimeError("registration worker cache is unavailable")
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
    except Exception as error:  # pragma: no cover - exercised on real solver faults
        return sample_index, sample_id, class_id, None, type(error).__name__


def _integration_weights(grid: Tensor) -> Tensor:
    weights = torch.ones_like(grid, dtype=torch.float64)
    weights[[0, -1]] *= 0.5
    weights = weights / weights.sum()
    return weights.to(dtype=grid.dtype)


def _batch_sample_ids(batch: dict, batch_size: int, fallback_start: int) -> Tensor:
    for name in ("index", "parcel_index"):
        value = batch.get(name)
        if isinstance(value, Tensor) and value.shape == (batch_size,):
            return value.detach().to(device="cpu", dtype=torch.long)
    return torch.arange(fallback_start, fallback_start + batch_size, dtype=torch.long)


def _build_target_geometry_cache(
    model: nn.Module,
    target_scan_loader,
    *,
    device: torch.device,
    shape_grid: Tensor,
    shape_extractor: TemporalSRVFExtractor,
    reg_extractor: TemporalSRVFExtractor,
) -> TargetGeometryCache:
    """Cache the selected target evidence once; expensive DP never reruns the model."""
    del shape_extractor
    rows: list[tuple[int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = []
    registration_grid: Tensor | None = None
    fallback_start = 0
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            for batch in target_scan_loader:
                output = model(
                    batch["pixels"],
                    batch["valid_pixels"],
                    batch["positions"],
                    batch.get("extra"),
                    return_geometry=True,
                )
                if output.geometry is None:
                    raise RuntimeError("target phase scan requires functional geometry")
                reg = evaluate_registration_geometry(
                    output.trend, output.positions, output.mask, reg_extractor
                )
                if registration_grid is None:
                    registration_grid = reg.registration_grid.detach().cpu()
                batch_size = int(output.trend.shape[0])
                sample_ids = _batch_sample_ids(batch, batch_size, fallback_start)
                fallback_start += batch_size
                for index in range(batch_size):
                    rows.append(
                        (
                            int(sample_ids[index].item()),
                            reg.trend_srvf[index].detach().cpu(),
                            reg.trend_support[index].detach().cpu(),
                            reg.trend_valid[index].detach().cpu(),
                            output.geometry.structure_srvf[index].detach().cpu(),
                            output.geometry.structure_support[index].detach().cpu(),
                            output.geometry.structure_valid[index].detach().cpu(),
                        )
                    )
    finally:
        model.train(was_training)
    if registration_grid is None or not rows:
        raise RuntimeError("target phase scan loader produced no batches")

    # GroupByShapesBatchSampler may reorder samples.  Sorting by stable dataset
    # id makes evidence order independent of pixel-width batching.
    rows.sort(key=lambda row: row[0])
    return TargetGeometryCache(
        sample_ids=torch.tensor([row[0] for row in rows], dtype=torch.long),
        trend_srvf_reg=torch.stack([row[1] for row in rows]),
        trend_support_reg=torch.stack([row[2] for row in rows]),
        trend_valid=torch.stack([row[3] for row in rows]),
        structure_srvf_shape=torch.stack([row[4] for row in rows]),
        structure_support_shape=torch.stack([row[5] for row in rows]),
        structure_valid=torch.stack([row[6] for row in rows]),
        registration_grid=registration_grid,
        shape_grid=shape_grid,
    )


def _common_support(support_a: Tensor, support_b: Tensor, integration_weights: Tensor) -> float:
    common = integration_weights * torch.minimum(support_a, support_b)
    return float(common.sum().item())


def _gamma_rejection_flags(legality, config: PhaseHypothesisScanConfig) -> tuple[str, ...]:
    flags: list[str] = []
    if not legality.finite or legality.endpoint_error > 1e-6:
        flags.append("endpoint")
    if (not legality.strictly_increasing) or legality.min_increment < config.registration_min_increment:
        flags.append("increment")
    if legality.max_local_speed > config.registration_max_local_speed:
        flags.append("speed")
    if legality.roughness > config.registration_max_roughness:
        flags.append("roughness")
    if legality.phase_deviation > config.registration_max_deviation:
        flags.append("deviation")
    return tuple(flags)


def _shape_evaluate(
    gamma: Tensor,
    class_id: int,
    *,
    cache: TargetGeometryCache,
    sample_index: int,
    source_cache: _CpuSourceScanCache,
    integration_shape: Tensor,
) -> tuple[float, float | None, float] | None:
    """Evaluate aligned Shape without applying Phase-evidence reliability gates."""
    reg_grid = cache.registration_grid
    shape_grid = cache.shape_grid
    gamma_shape = resample_gamma(gamma, reg_grid, shape_grid)
    target_shape = cache.structure_srvf_shape[sample_index]
    target_support = cache.structure_support_shape[sample_index]
    aligned = warp_q_gamma(target_shape, gamma_shape).squeeze(0)
    aligned_support = warp_support_gamma(target_support, gamma_shape, shape_grid)
    proto = source_cache.shape_srvf[class_id].to(aligned.dtype)
    proto_support = source_cache.shape_support[class_id].to(aligned_support.dtype)
    distance = support_aware_q_distance(
        aligned.unsqueeze(0),
        proto.unsqueeze(0),
        aligned_support.unsqueeze(0),
        proto_support.unsqueeze(0),
        integration_shape.to(dtype=aligned.dtype),
    )
    if not bool(distance.valid[0, 0].item()):
        return None
    q_dist = float(distance.distance[0, 0].item())
    common_shape = float(distance.common_support[0, 0].item())
    samples = source_cache.q_distance_samples[class_id]
    pct: float | None = None
    if samples.numel() > 0:
        pct = float(
            empirical_cdf(
                samples,
                torch.tensor([q_dist], dtype=torch.float64),
            )[0].item()
        )
    return q_dist, pct, common_shape


def _alignment_to_hypothesis(
    alignment: PairwiseClassAlignment,
    *,
    preferred: bool,
    ambiguous_class: bool,
    evidence_weight: float,
) -> TargetClassPhaseHypothesis:
    if not alignment.phase_evidence_eligible or alignment.gamma is None:
        raise ValueError("only Phase-evidence-eligible alignments can become hypotheses")
    required = (
        alignment.t_identity_error,
        alignment.t_registered_error,
        alignment.t_gain_ratio,
        alignment.q_shape_distance,
        alignment.q_distance_percentile,
        alignment.common_support_t,
        alignment.common_support_shape,
        alignment.gamma_roughness,
        alignment.phase_deviation,
    )
    if any(value is None for value in required):
        raise ValueError("eligible alignment is missing required diagnostics")
    return TargetClassPhaseHypothesis(
        sample_id=alignment.sample_id,
        class_id=alignment.class_id,
        gamma=alignment.gamma.detach().cpu().double(),
        t_identity_error=float(alignment.t_identity_error),
        t_registered_error=float(alignment.t_registered_error),
        t_gain_ratio=float(alignment.t_gain_ratio),
        q_shape_distance=float(alignment.q_shape_distance),
        q_distance_percentile=float(alignment.q_distance_percentile),
        common_support_t=float(alignment.common_support_t),
        common_support_shape=float(alignment.common_support_shape),
        roughness=float(alignment.gamma_roughness),
        phase_deviation=float(alignment.phase_deviation),
        preferred=preferred,
        ambiguous_class=ambiguous_class,
        evidence_weight=float(evidence_weight),
    )


def _select_candidate_and_hypotheses(
    alignments: Iterable[PairwiseClassAlignment],
    *,
    ambiguity_margin: float,
) -> tuple[CandidatePseudoLabel | None, tuple[TargetClassPhaseHypothesis, ...]]:
    """Select raw-Shape argmin first, then apply reliability to Phase evidence."""
    usable = [
        item
        for item in alignments
        if item.q_shape_distance is not None and math.isfinite(float(item.q_shape_distance))
    ]
    if not usable:
        return None, ()
    usable.sort(key=lambda item: (float(item.q_shape_distance), int(item.class_id)))
    primary = usable[0]
    secondary = usable[1] if len(usable) > 1 else None
    gap = (
        float(secondary.q_shape_distance) - float(primary.q_shape_distance)
        if secondary is not None
        else float("inf")
    )
    ambiguous = secondary is not None and gap < float(ambiguity_margin)
    candidate = CandidatePseudoLabel(
        sample_id=primary.sample_id,
        class_id=primary.class_id,
        q_shape_distance=float(primary.q_shape_distance),
        q_distance_percentile=primary.q_distance_percentile,
        phase_evidence_eligible=primary.phase_evidence_eligible,
        ambiguous=ambiguous,
        secondary_class_id=secondary.class_id if ambiguous and secondary is not None else None,
        secondary_q_shape_distance=(
            float(secondary.q_shape_distance) if ambiguous and secondary is not None else None
        ),
        secondary_phase_evidence_eligible=(
            secondary.phase_evidence_eligible if ambiguous and secondary is not None else None
        ),
    )

    evidence_candidates = [primary]
    if ambiguous and secondary is not None:
        evidence_candidates.append(secondary)
    eligible = [item for item in evidence_candidates if item.phase_evidence_eligible]
    if not eligible:
        return candidate, ()
    weight = 1.0 / len(eligible)
    hypotheses = tuple(
        _alignment_to_hypothesis(
            item,
            preferred=(item.class_id == primary.class_id and len(eligible) == 1),
            ambiguous_class=ambiguous,
            evidence_weight=weight,
        )
        for item in eligible
    )
    return candidate, hypotheses


def _quantiles(values: Iterable[float]) -> tuple[float, float, float]:
    tensor = torch.tensor(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=torch.float64,
    )
    if tensor.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    q = torch.quantile(tensor, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    return tuple(float(value.item()) for value in q)  # type: ignore[return-value]


class TargetPhaseHypothesisScanner:
    """Cache target geometry once and acquire nested exact-DP evidence progressively."""

    def __init__(
        self,
        model: nn.Module,
        target_scan_loader,
        stage1_prototype_bank: SourcePrototypeBank,
        source_registration_bank: SourceRegistrationPrototypeBank,
        config: PhaseHypothesisScanConfig,
        *,
        device: torch.device,
        shape_extractor: TemporalSRVFExtractor,
        reg_extractor: TemporalSRVFExtractor,
        evidence_seed: int = 0,
        adapter: FdasrsfCurveRegistrationAdapter | None = None,
    ) -> None:
        self.model = model
        self.stage1_bank = stage1_prototype_bank
        self.source_reg_bank = source_registration_bank
        self.config = config
        self.device = device
        self.adapter = adapter
        shape_grid = shape_extractor.functional_lift.canonical_grid.detach().cpu()
        start = time.monotonic()
        print("TARGET_GEOMETRY_CACHE_START")
        self.cache = _build_target_geometry_cache(
            model,
            target_scan_loader,
            device=device,
            shape_grid=shape_grid,
            shape_extractor=shape_extractor,
            reg_extractor=reg_extractor,
        )
        print(
            "TARGET_GEOMETRY_CACHE_READY|"
            f"samples={self.cache.sample_ids.numel()}|seconds={time.monotonic() - start:.2f}"
        )
        self.source_cache = _build_cpu_source_cache(
            stage1_prototype_bank, source_registration_bank
        )
        self.target_trend_np = np.ascontiguousarray(
            self.cache.trend_srvf_reg.to(dtype=torch.float64).numpy().transpose(0, 2, 1)
        )
        generator = torch.Generator(device="cpu").manual_seed(int(evidence_seed))
        self.order = torch.randperm(self.cache.sample_ids.numel(), generator=generator)
        self.integration_reg = _integration_weights(self.cache.registration_grid)
        self.integration_shape = _integration_weights(self.cache.shape_grid)
        self.scanned = 0
        self.hypotheses: list[TargetClassPhaseHypothesis] = []
        self.pairwise_alignments: list[PairwiseClassAlignment] = []
        self.candidate_pseudo_labels: list[CandidatePseudoLabel] = []
        self.sample_cardinality: dict[int, int] = {}
        self.counts = {
            "attempted": 0,
            "solver_calls": 0,
            "pre_support": 0,
            "solver_failed": 0,
            "gamma_rejected": 0,
            "endpoint": 0,
            "increment": 0,
            "speed": 0,
            "roughness": 0,
            "deviation": 0,
            "gain_rejected": 0,
            "shape_support_rejected": 0,
            "outer_rejected": 0,
        }
        self.pairwise_class_counts = [0] * int(source_registration_bank.ready.numel())
        self.diagnostics: dict[str, list[float]] = {
            "common_support": [],
            "gain_ratio": [],
            "roughness": [],
            "max_local_speed": [],
            "phase_deviation": [],
            "q_distance": [],
            "q_percentile": [],
        }

    @property
    def total_cached_samples(self) -> int:
        return int(self.cache.sample_ids.numel())

    def sample_ids_for_budget(self, budget: int) -> tuple[int, ...]:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("evidence budget must be a positive integer")
        budget = min(budget, self.total_cached_samples)
        return tuple(
            int(self.cache.sample_ids[int(index)].item())
            for index in self.order[:budget].tolist()
        )

    def _solve(self, tasks: list[_SolverTask]) -> list[_SolverResult]:
        if not tasks:
            return []
        # Injected adapters are always executed serially for deterministic unit
        # tests.  The production fdasrsf path uses process-level parallelism on
        # Linux, where the server runs.
        can_parallel = (
            self.adapter is None
            and self.config.registration_workers > 1
            and platform.system().lower() == "linux"
        )
        if not can_parallel:
            adapter = self.adapter or FdasrsfCurveRegistrationAdapter(
                registration_lambda=self.config.registration_lambda
            )
            results = []
            for task in tasks:
                try:
                    source_np = self.source_cache.trend_np[task.class_id]
                    target_np = self.target_trend_np[task.sample_index]
                    gamma = adapter.register(
                        torch.from_numpy(source_np.T.copy()),
                        torch.from_numpy(target_np.T.copy()),
                    ).detach().cpu().double()
                    results.append(
                        _SolverResult(task.sample_index, task.sample_id, task.class_id, gamma, None)
                    )
                except Exception as error:
                    results.append(
                        _SolverResult(
                            task.sample_index,
                            task.sample_id,
                            task.class_id,
                            None,
                            type(error).__name__,
                        )
                    )
            return results

        # Spawn workers once per evidence stage with immutable NumPy caches.
        # Individual jobs then carry only integer indices, instead of repeatedly
        # pickling two [D,K] float64 arrays through the process pipe.
        stage_sample_indices = tuple(dict.fromkeys(task.sample_index for task in tasks))
        target_local = {sample_index: local for local, sample_index in enumerate(stage_sample_indices)}
        stage_target_np = np.ascontiguousarray(
            self.target_trend_np[list(stage_sample_indices)]
        )
        payloads = [
            (
                task.sample_index,
                task.sample_id,
                task.class_id,
                target_local[task.sample_index],
                self.config.registration_lambda,
            )
            for task in tasks
        ]
        workers = min(self.config.registration_workers, len(tasks), os.cpu_count() or 1)
        # Spawn rather than fork: the parent already owns a CUDA context.  Set
        # BLAS/OpenMP limits before child creation so imports in spawned workers
        # observe them from process start.
        chunksize = max(1, len(payloads) // max(1, workers * 8))
        raw = []
        solve_start = time.monotonic()
        progress_every = max(64, workers * chunksize)
        env_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        previous_env = {key: os.environ.get(key) for key in env_keys}
        for key in env_keys:
            os.environ[key] = "1"
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=_registration_worker_init,
                initargs=(self.source_cache.trend_np, stage_target_np),
            ) as executor:
                iterator = executor.map(_fdasrsf_worker, payloads, chunksize=chunksize)
                for completed, item in enumerate(iterator, start=1):
                    raw.append(item)
                    if completed % progress_every == 0 or completed == len(payloads):
                        print(
                            "TARGET_DP_PROGRESS|"
                            f"completed={completed}/{len(payloads)}"
                            f"|workers={workers}"
                            f"|seconds={time.monotonic() - solve_start:.2f}"
                        )
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        return [
            _SolverResult(
                sample_index=item[0],
                sample_id=item[1],
                class_id=item[2],
                gamma=None if item[3] is None else torch.as_tensor(item[3], dtype=torch.float64),
                error_name=item[4],
            )
            for item in raw
        ]

    def scan_to_budget(self, budget: int) -> TargetHypothesisScanResult:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("phase evidence budget must be a positive integer")
        budget = min(budget, self.total_cached_samples)
        if budget < self.scanned:
            raise ValueError("phase evidence budget cannot decrease")
        if budget == self.scanned:
            return self.result()

        stage_start = time.monotonic()
        new_cache_indices = [int(i) for i in self.order[self.scanned : budget].tolist()]
        ready_classes = [
            int(index)
            for index in torch.nonzero(self.source_cache.ready, as_tuple=False).flatten().tolist()
        ]
        ready_count = len(ready_classes)
        tasks: list[_SolverTask] = []
        pre_common_support: dict[tuple[int, int], float] = {}

        # Round A formal semantics: every representative target sample is
        # registered against every ready source class.  No classifier or
        # identity-T proposal is allowed to shrink this search space.
        for sample_index in new_cache_indices:
            sample_id = int(self.cache.sample_ids[sample_index].item())
            for class_id in ready_classes:
                self.counts["attempted"] += 1
                self.pairwise_class_counts[class_id] += 1
                source_support = self.source_cache.trend_support[class_id]
                target_support = self.cache.trend_support_reg[sample_index]
                common = _common_support(source_support, target_support, self.integration_reg)
                pre_common_support[(sample_index, class_id)] = common
                self.diagnostics["common_support"].append(common)
                if common < self.config.registration_min_common_support:
                    # Reliability mark only: do not erase this class before the
                    # geometry-first pseudo-label can be formed.
                    self.counts["pre_support"] += 1
                tasks.append(_SolverTask(sample_index, sample_id, class_id))

        self.counts["solver_calls"] += len(tasks)
        print(
            "TARGET_HYPOTHESIS_SCAN_STAGE_START|"
            f"budget={budget}|new_samples={len(new_cache_indices)}"
            f"|ready_classes={ready_count}|all_class_pairs={len(tasks)}"
            f"|solver_calls={len(tasks)}|workers={self.config.registration_workers}"
        )

        solved = self._solve(tasks)
        alignments_by_sample: dict[int, list[PairwiseClassAlignment]] = {
            index: [] for index in new_cache_indices
        }
        reg_grid = self.cache.registration_grid

        for solved_pair in solved:
            sample_index = solved_pair.sample_index
            class_id = solved_pair.class_id
            pre_common = pre_common_support[(sample_index, class_id)]
            if solved_pair.error_name is not None or solved_pair.gamma is None:
                self.counts["solver_failed"] += 1
                alignment = PairwiseClassAlignment(
                    sample_id=solved_pair.sample_id,
                    class_id=class_id,
                    gamma=None,
                    t_identity_error=None,
                    t_registered_error=None,
                    t_gain_ratio=None,
                    pre_common_support_t=pre_common,
                    common_support_t=None,
                    gamma_finite=False,
                    gamma_endpoint_error=None,
                    gamma_strictly_increasing=False,
                    gamma_min_increment=None,
                    gamma_max_local_speed=None,
                    gamma_roughness=None,
                    phase_deviation=None,
                    q_shape_distance=None,
                    q_distance_percentile=None,
                    common_support_shape=None,
                    numerically_valid=False,
                    phase_evidence_eligible=False,
                    reject_reasons=("solver_failed",),
                    solver_error=solved_pair.error_name,
                )
                self.pairwise_alignments.append(alignment)
                alignments_by_sample[sample_index].append(alignment)
                continue

            gamma = solved_pair.gamma.to(dtype=torch.float32)
            legality = check_gamma_legality(
                gamma,
                reg_grid,
                registration_min_increment=self.config.registration_min_increment,
                registration_max_local_speed=self.config.registration_max_local_speed,
                registration_max_roughness=self.config.registration_max_roughness,
                registration_max_deviation=self.config.registration_max_deviation,
            )
            self.diagnostics["roughness"].append(legality.roughness)
            self.diagnostics["max_local_speed"].append(legality.max_local_speed)
            self.diagnostics["phase_deviation"].append(legality.phase_deviation)
            gamma_flags = _gamma_rejection_flags(legality, self.config)
            if gamma_flags:
                self.counts["gamma_rejected"] += 1
                for flag in gamma_flags:
                    self.counts[flag] += 1

            # Basic numerical validity is intentionally weaker than the
            # configured plausibility gates: finite + endpoint-valid + strictly
            # increasing is enough to evaluate Shape for pseudo-label ranking.
            numerically_valid = (
                legality.finite
                and legality.endpoint_error <= 1e-6
                and legality.strictly_increasing
            )
            diagnostics = None
            evaluated = None
            if numerically_valid:
                source_q = self.source_cache.trend_srvf[class_id].to(gamma.dtype)
                target_q = self.cache.trend_srvf_reg[sample_index].to(gamma.dtype)
                source_support = self.source_cache.trend_support[class_id].to(gamma.dtype)
                target_support = self.cache.trend_support_reg[sample_index].to(gamma.dtype)
                diagnostics = compute_gamma_diagnostics(
                    sample_id=solved_pair.sample_id,
                    class_id=class_id,
                    gamma=gamma,
                    source_trend_srvf=source_q,
                    target_trend_srvf=target_q,
                    source_support=source_support,
                    target_support=target_support,
                    integration_weights=self.integration_reg.to(gamma.dtype),
                    registration_grid=reg_grid.to(gamma.dtype),
                )
                self.diagnostics["gain_ratio"].append(diagnostics.gain_ratio)
                if (
                    not math.isfinite(diagnostics.gain_ratio)
                    or diagnostics.gain_ratio > self.config.registration_gain_ratio_max
                ):
                    self.counts["gain_rejected"] += 1
                evaluated = _shape_evaluate(
                    gamma.detach().cpu().double(),
                    class_id,
                    cache=self.cache,
                    sample_index=sample_index,
                    source_cache=self.source_cache,
                    integration_shape=self.integration_shape,
                )

            q_dist: float | None = None
            pct: float | None = None
            common_shape: float | None = None
            if numerically_valid and evaluated is None:
                self.counts["shape_support_rejected"] += 1
            elif evaluated is not None:
                q_dist, pct, common_shape = evaluated
                self.diagnostics["q_distance"].append(q_dist)
                if pct is not None:
                    self.diagnostics["q_percentile"].append(pct)
                outer = float(self.source_cache.q_outer[class_id].item())
                if q_dist > outer:
                    self.counts["outer_rejected"] += 1

            reasons: list[str] = []
            if pre_common < self.config.registration_min_common_support:
                reasons.append("pre_support")
            reasons.extend(f"gamma_{flag}" for flag in gamma_flags)
            if diagnostics is None or not math.isfinite(diagnostics.gain_ratio):
                reasons.append("gain_unavailable")
            elif diagnostics.gain_ratio > self.config.registration_gain_ratio_max:
                reasons.append("gain")
            if evaluated is None:
                reasons.append("shape_support")
            elif pct is None:
                reasons.append("q_cdf_unavailable")

            # Source q95 is intentionally diagnostic-only here.  Domain Shape
            # shift can move a correctly matched target sample outside the
            # source within-class q range; using that range as a prerequisite
            # for discovering Domain Phase creates a circular dependency.
            # ``outer_rejected`` is still counted above for calibration and
            # downstream stable-label analysis.
            eligible = numerically_valid and not reasons
            alignment = PairwiseClassAlignment(
                sample_id=solved_pair.sample_id,
                class_id=class_id,
                gamma=gamma.detach().cpu().double(),
                t_identity_error=None if diagnostics is None else diagnostics.e_id,
                t_registered_error=None if diagnostics is None else diagnostics.e_reg,
                t_gain_ratio=None if diagnostics is None else diagnostics.gain_ratio,
                pre_common_support_t=pre_common,
                common_support_t=None if diagnostics is None else diagnostics.common_support,
                gamma_finite=legality.finite,
                gamma_endpoint_error=legality.endpoint_error,
                gamma_strictly_increasing=legality.strictly_increasing,
                gamma_min_increment=legality.min_increment,
                gamma_max_local_speed=legality.max_local_speed,
                gamma_roughness=legality.roughness,
                phase_deviation=legality.phase_deviation,
                q_shape_distance=q_dist,
                q_distance_percentile=pct,
                common_support_shape=common_shape,
                numerically_valid=numerically_valid,
                phase_evidence_eligible=eligible,
                reject_reasons=tuple(reasons),
                solver_error=None,
            )
            self.pairwise_alignments.append(alignment)
            alignments_by_sample[sample_index].append(alignment)

        for sample_index in new_cache_indices:
            candidate, kept = _select_candidate_and_hypotheses(
                alignments_by_sample[sample_index],
                ambiguity_margin=self.config.class_hypothesis_margin,
            )
            sample_id = int(self.cache.sample_ids[sample_index].item())
            if candidate is not None:
                self.candidate_pseudo_labels.append(candidate)
            self.sample_cardinality[sample_id] = len(kept)
            self.hypotheses.extend(kept)

        self.scanned = budget
        result = self.result()
        print(
            "TARGET_HYPOTHESIS_SCAN_STAGE_DONE|"
            f"budget={budget}|candidate_labels={len(result.candidate_pseudo_labels)}"
            f"|hypotheses={len(result.hypotheses)}"
            f"|zero={result.samples_with_zero_hypothesis}"
            f"|one={result.samples_with_one_hypothesis}"
            f"|two={result.samples_with_two_hypotheses}"
            f"|pre_support={result.num_pre_support_rejected}"
            f"|solver_failed={result.num_solver_failed}"
            f"|gamma_rejected={result.num_gamma_rejected}"
            f"|gamma_endpoint={result.num_gamma_endpoint_rejected}"
            f"|gamma_increment={result.num_gamma_increment_rejected}"
            f"|gamma_speed={result.num_gamma_speed_rejected}"
            f"|gamma_roughness={result.num_gamma_roughness_rejected}"
            f"|gamma_deviation={result.num_gamma_deviation_rejected}"
            f"|gain_rejected={result.num_gain_rejected}"
            f"|shape_support_rejected={result.num_shape_support_rejected}"
            f"|outer_rejected={result.num_outer_rejected}"
            f"|seconds={time.monotonic() - stage_start:.2f}"
        )
        quantile_text = "|".join(
            f"{name}_p10={p10:.4g}|{name}_p50={p50:.4g}|{name}_p90={p90:.4g}"
            for name, p10, p50, p90 in result.diagnostic_quantiles
        )
        if quantile_text:
            print(
                "TARGET_HYPOTHESIS_SCAN_DISTRIBUTIONS|"
                f"budget={budget}|" + quantile_text
            )
        return result

    def result(self) -> TargetHypothesisScanResult:
        sample_ids = tuple(
            int(self.cache.sample_ids[int(index)].item())
            for index in self.order[: self.scanned].tolist()
        )
        zero = sum(self.sample_cardinality.get(sample_id, 0) == 0 for sample_id in sample_ids)
        one = sum(self.sample_cardinality.get(sample_id, 0) == 1 for sample_id in sample_ids)
        two = sum(self.sample_cardinality.get(sample_id, 0) == 2 for sample_id in sample_ids)
        quantiles = tuple(
            (name, *_quantiles(values)) for name, values in self.diagnostics.items()
        )
        ready_count = int(self.source_cache.ready.sum().item())
        return TargetHypothesisScanResult(
            hypotheses=tuple(self.hypotheses),
            num_samples=self.scanned,
            num_pairwise_attempted=self.counts["attempted"],
            num_pre_support_rejected=self.counts["pre_support"],
            num_solver_failed=self.counts["solver_failed"],
            num_gamma_rejected=self.counts["gamma_rejected"],
            num_gain_rejected=self.counts["gain_rejected"],
            num_shape_support_rejected=self.counts["shape_support_rejected"],
            num_outer_rejected=self.counts["outer_rejected"],
            samples_with_zero_hypothesis=zero,
            samples_with_one_hypothesis=one,
            samples_with_two_hypotheses=two,
            pairwise_alignments=tuple(self.pairwise_alignments),
            candidate_pseudo_labels=tuple(self.candidate_pseudo_labels),
            num_solver_calls=self.counts["solver_calls"],
            num_ready_classes=ready_count,
            num_all_class_pairs=self.scanned * ready_count,
            num_proposal_pairs=self.counts["attempted"],
            num_gamma_endpoint_rejected=self.counts["endpoint"],
            num_gamma_increment_rejected=self.counts["increment"],
            num_gamma_speed_rejected=self.counts["speed"],
            num_gamma_roughness_rejected=self.counts["roughness"],
            num_gamma_deviation_rejected=self.counts["deviation"],
            scanned_sample_ids=sample_ids,
            pairwise_class_counts=tuple(self.pairwise_class_counts),
            proposal_class_counts=tuple(self.pairwise_class_counts),
            diagnostic_quantiles=quantiles,
        )


def scan_target_class_phase_hypotheses(
    model: nn.Module,
    target_scan_loader,
    stage1_prototype_bank: SourcePrototypeBank,
    source_registration_bank: SourceRegistrationPrototypeBank,
    config: PhaseHypothesisScanConfig,
    *,
    device: torch.device,
    shape_extractor: TemporalSRVFExtractor,
    reg_extractor: TemporalSRVFExtractor,
    adapter: FdasrsfCurveRegistrationAdapter | None = None,
) -> TargetHypothesisScanResult:
    """Compatibility wrapper: scan the supplied evidence loader to completion."""
    scanner = TargetPhaseHypothesisScanner(
        model,
        target_scan_loader,
        stage1_prototype_bank,
        source_registration_bank,
        config,
        device=device,
        shape_extractor=shape_extractor,
        reg_extractor=reg_extractor,
        evidence_seed=0,
        adapter=adapter,
    )
    return scanner.scan_to_budget(scanner.total_cached_samples)
