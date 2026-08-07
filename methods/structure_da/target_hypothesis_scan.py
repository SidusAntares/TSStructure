"""Full target class-conditioned phase hypothesis scan.

For each target sample the scan caches its K_reg trend geometry and K_shape
Shape geometry once (O(N_target)), then runs one DP2 registration against each
ready source class (O(N_target x C)). The returned gamma is resampled to the
Shape grid, applied to the target S-SRVF, and scored against the Stage-1
source Shape prototypes. The score is turned into a percentile through the
Stage-1 empirical source intra-class distance distribution. At most two
hypotheses per sample are retained, cold-starting from T registration and
aligned Shape distance only — no classifier, no fused features, no target
labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .phase_evidence import (
    PairwisePhaseCandidate,
    compute_gamma_diagnostics,
    empirical_cdf,
)
from .phase_registration import (
    FdasrsfDP2RegistrationAdapter,
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
    """Thresholds and constants for the hypothesis scan.

    Only the structural constants are frozen: ``k_reg=128``, ``method=DP2``,
    ``penalty=roughness`` and ``class_hypothesis_max=2``. The experimental
    thresholds are passed explicitly by tests and frozen in a later round.
    """

    registration_lambda: float
    registration_dp_grid_dim: int

    registration_gain_ratio_max: float
    registration_min_common_support: float
    registration_max_roughness: float
    registration_min_increment: float
    registration_max_local_speed: float
    registration_max_deviation: float

    class_hypothesis_margin: float

    k_reg: int = 128
    class_hypothesis_max: int = 2


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


def _integration_weights(grid: Tensor) -> Tensor:
    weights = torch.ones_like(grid, dtype=torch.float64)
    weights[[0, -1]] *= 0.5
    weights = weights / weights.sum()
    return weights.to(dtype=grid.dtype)


def _build_target_geometry_cache(
    model: nn.Module,
    target_scan_loader,
    *,
    device: torch.device,
    shape_grid: Tensor,
    shape_extractor: TemporalSRVFExtractor,
    reg_extractor: TemporalSRVFExtractor,
) -> TargetGeometryCache:
    """Cache each target sample's K_reg trend and K_shape Shape geometry once."""
    sample_ids: list[int] = []
    trend_srvf_reg: list[Tensor] = []
    trend_support_reg: list[Tensor] = []
    trend_valid: list[Tensor] = []
    structure_srvf_shape: list[Tensor] = []
    structure_support_shape: list[Tensor] = []
    structure_valid: list[Tensor] = []
    registration_grid: Tensor | None = None
    global_index = 0

    with torch.inference_mode():
        for batch in target_scan_loader:
            output = model(
                batch["pixels"],
                batch["valid_pixels"],
                batch["positions"],
                batch.get("extra"),
                return_geometry=True,
            )
            trend = output.trend
            mask = output.mask
            positions = output.positions
            reg = evaluate_registration_geometry(
                trend, positions, mask, reg_extractor
            )
            geometry = output.geometry
            if registration_grid is None:
                registration_grid = reg.registration_grid.detach().cpu()
            batch_size = trend.shape[0]
            for index in range(batch_size):
                sample_ids.append(global_index)
                global_index += 1
                trend_srvf_reg.append(reg.trend_srvf[index].detach().cpu())
                trend_support_reg.append(reg.trend_support[index].detach().cpu())
                trend_valid.append(reg.trend_valid[index].detach().cpu())
                structure_srvf_shape.append(
                    geometry.structure_srvf[index].detach().cpu()
                )
                structure_support_shape.append(
                    geometry.structure_support[index].detach().cpu()
                )
                structure_valid.append(
                    geometry.structure_valid[index].detach().cpu()
                )
    if registration_grid is None:
        raise RuntimeError("target scan loader produced no batches")
    return TargetGeometryCache(
        sample_ids=torch.tensor(sample_ids, dtype=torch.long),
        trend_srvf_reg=torch.stack(trend_srvf_reg),
        trend_support_reg=torch.stack(trend_support_reg),
        trend_valid=torch.stack(trend_valid),
        structure_srvf_shape=torch.stack(structure_srvf_shape),
        structure_support_shape=torch.stack(structure_support_shape),
        structure_valid=torch.stack(structure_valid),
        registration_grid=registration_grid,
        shape_grid=shape_grid,
    )


def _process_pairwise_candidates(
    *,
    sample_index: int,
    sample_id: int,
    cache: TargetGeometryCache,
    stage1_bank: SourcePrototypeBank,
    source_reg_bank: SourceRegistrationPrototypeBank,
    config: PhaseHypothesisScanConfig,
    adapter: FdasrsfDP2RegistrationAdapter,
    device: torch.device,
    integration_reg: Tensor,
    integration_shape: Tensor,
    solver_diagnostics: list,
) -> list[PairwisePhaseCandidate]:
    """Run DP2 against every ready source class for one target sample."""
    candidates: list[PairwisePhaseCandidate] = []
    reg_grid = cache.registration_grid.to(device=device)
    target_trend = cache.trend_srvf_reg[sample_index].to(device=device)
    target_trend_support = cache.trend_support_reg[sample_index].to(device=device)

    for class_id in range(source_reg_bank.trend_srvf.shape[0]):
        if not source_reg_bank.ready[class_id].item():
            continue
        source_trend = source_reg_bank.trend_srvf[class_id].to(device=device)
        source_trend_support = source_reg_bank.trend_support[class_id].to(device=device)

        # Pre-registration support gate: do not call the solver without enough
        # common support.
        pre_common = _common_support(
            source_trend_support, target_trend_support, integration_reg
        )
        if pre_common < config.registration_min_common_support:
            solver_diagnostics.append("pre_support")
            continue

        # DP2: single solver call per sample-class pair.
        try:
            gamma = adapter.register(source_trend, target_trend, reg_grid)
        except Exception as error:  # solver isolation per pair
            solver_diagnostics.append(f"solver_error:{type(error).__name__}")
            continue

        legality = check_gamma_legality(
            gamma,
            reg_grid,
            registration_min_increment=config.registration_min_increment,
            registration_max_local_speed=config.registration_max_local_speed,
            registration_max_roughness=config.registration_max_roughness,
            registration_max_deviation=config.registration_max_deviation,
        )
        if not legality.legal:
            solver_diagnostics.append("gamma_illegal")
            continue

        diagnostics = compute_gamma_diagnostics(
            sample_id=sample_id,
            class_id=class_id,
            gamma=gamma,
            source_trend_srvf=source_trend,
            target_trend_srvf=target_trend,
            source_support=source_trend_support,
            target_support=target_trend_support,
            integration_weights=integration_reg,
            registration_grid=reg_grid,
            adapter=adapter,
        )
        candidates.append(
            PairwisePhaseCandidate(
                sample_id=sample_id,
                class_id=class_id,
                gamma=gamma,
                t_identity_error=diagnostics.e_id,
                t_registered_error=diagnostics.e_reg,
                t_gain_ratio=diagnostics.gain_ratio,
                common_support=diagnostics.common_support,
                roughness=legality.roughness,
                min_increment=legality.min_increment,
                max_local_speed=legality.max_local_speed,
                phase_deviation=legality.phase_deviation,
                legal=True,
                reject_reason=None,
            )
        )
    return candidates


def _common_support(
    support_a: Tensor, support_b: Tensor, integration_weights: Tensor
) -> float:
    common = integration_weights * torch.minimum(support_a, support_b)
    return float(common.sum().item())


def _shape_evaluate(
    candidate: PairwisePhaseCandidate,
    *,
    cache: TargetGeometryCache,
    sample_index: int,
    stage1_bank: SourcePrototypeBank,
    config: PhaseHypothesisScanConfig,
    device: torch.device,
    integration_shape: Tensor,
) -> tuple[float, float, float] | None:
    """Resample gamma to the Shape grid, warp target S-SRVF, return (dist, pct, common)."""
    reg_grid = cache.registration_grid.to(device=device)
    shape_grid = cache.shape_grid.to(device=device)
    gamma_shape = resample_gamma(candidate.gamma.to(device=device), reg_grid, shape_grid)
    target_shape = cache.structure_srvf_shape[sample_index].to(device=device)
    target_support = cache.structure_support_shape[sample_index].to(device=device)
    aligned = warp_q_gamma(target_shape.unsqueeze(0), gamma_shape).squeeze(0)
    aligned_support = warp_support_gamma(target_support, gamma_shape, shape_grid)
    proto = stage1_bank.shape_srvf[candidate.class_id].to(device=device)
    proto_support = stage1_bank.shape_support[candidate.class_id].to(device=device)
    distance = support_aware_q_distance(
        aligned.unsqueeze(0),
        proto.unsqueeze(0),
        aligned_support.unsqueeze(0),
        proto_support.unsqueeze(0),
        integration_shape.to(device=device),
    )
    if not distance.valid[0, 0].item():
        return None
    q_dist = float(distance.distance[0, 0].item())
    common_shape = float(distance.common_support[0, 0].item())
    samples = stage1_bank.q_distance_samples[candidate.class_id]
    if samples is None or samples.numel() == 0:
        return None
    pct = float(
        empirical_cdf(
            samples.to(device=device), torch.tensor([q_dist], device=device)
        )[0].item()
    )
    return q_dist, pct, common_shape


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
    adapter: FdasrsfDP2RegistrationAdapter | None = None,
) -> TargetHypothesisScanResult:
    """Run the full target class-conditioned phase hypothesis scan.

    The scan is fully no-grad. Target labels are never read, so permuting the
    target batch labels does not change the result.
    """
    if adapter is None:
        adapter = FdasrsfDP2RegistrationAdapter(
            registration_lambda=config.registration_lambda,
            registration_dp_grid_dim=config.registration_dp_grid_dim,
        )
    shape_grid = shape_extractor.functional_lift.canonical_grid.detach().cpu()
    was_training = model.training
    try:
        model.eval()
        cache = _build_target_geometry_cache(
            model,
            target_scan_loader,
            device=device,
            shape_grid=shape_grid,
            shape_extractor=shape_extractor,
            reg_extractor=reg_extractor,
        )
        integration_reg = _integration_weights(cache.registration_grid)
        integration_shape = _integration_weights(cache.shape_grid)

        hypotheses: list[TargetClassPhaseHypothesis] = []
        counts = {
            "attempted": 0,
            "pre_support": 0,
            "solver_failed": 0,
            "gamma_rejected": 0,
            "gain_rejected": 0,
            "shape_support_rejected": 0,
            "outer_rejected": 0,
        }
        zero_count = 0
        one_count = 0
        two_count = 0

        num_samples = cache.sample_ids.shape[0]
        for sample_index in range(num_samples):
            sample_id = int(cache.sample_ids[sample_index].item())
            solver_diagnostics: list = []
            candidates = _process_pairwise_candidates(
                sample_index=sample_index,
                sample_id=sample_id,
                cache=cache,
                stage1_bank=stage1_prototype_bank,
                source_reg_bank=source_registration_bank,
                config=config,
                adapter=adapter,
                device=device,
                integration_reg=integration_reg,
                integration_shape=integration_shape,
                solver_diagnostics=solver_diagnostics,
            )
            for diag in solver_diagnostics:
                if diag == "pre_support":
                    counts["pre_support"] += 1
                elif diag.startswith("solver_error"):
                    counts["solver_failed"] += 1
                elif diag == "gamma_illegal":
                    counts["gamma_rejected"] += 1
            counts["attempted"] += len(candidates) + sum(
                1 for d in solver_diagnostics if d.startswith("solver_error") or d == "gamma_illegal"
            )

            sample_hypotheses: list[TargetClassPhaseHypothesis] = []
            for candidate in candidates:
                # T gain gate.
                if candidate.t_gain_ratio > config.registration_gain_ratio_max:
                    counts["gain_rejected"] += 1
                    continue
                evaluated = _shape_evaluate(
                    candidate,
                    cache=cache,
                    sample_index=sample_index,
                    stage1_bank=stage1_prototype_bank,
                    config=config,
                    device=device,
                    integration_shape=integration_shape,
                )
                if evaluated is None:
                    counts["shape_support_rejected"] += 1
                    continue
                q_dist, pct, common_shape = evaluated
                # Outer gate: distance must be within the source outer distance.
                outer = stage1_prototype_bank.q_quantiles[candidate.class_id, 2].item()
                if q_dist > outer:
                    counts["outer_rejected"] += 1
                    continue
                sample_hypotheses.append(
                    TargetClassPhaseHypothesis(
                        sample_id=sample_id,
                        class_id=candidate.class_id,
                        gamma=candidate.gamma,
                        t_identity_error=candidate.t_identity_error,
                        t_registered_error=candidate.t_registered_error,
                        t_gain_ratio=candidate.t_gain_ratio,
                        q_shape_distance=q_dist,
                        q_distance_percentile=pct,
                        common_support_t=candidate.common_support,
                        common_support_shape=common_shape,
                        roughness=candidate.roughness,
                        phase_deviation=candidate.phase_deviation,
                        preferred=False,
                        ambiguous_class=False,
                        evidence_weight=1.0,
                    )
                )

            # Sort by percentile ascending and keep at most two.
            sample_hypotheses.sort(key=lambda h: h.q_distance_percentile)
            kept = sample_hypotheses[: config.class_hypothesis_max]
            if len(kept) == 1:
                kept[0] = _replace(
                    kept[0], preferred=True, ambiguous_class=False, evidence_weight=1.0
                )
                one_count += 1
            elif len(kept) == 2:
                u1 = kept[0].q_distance_percentile
                u2 = kept[1].q_distance_percentile
                gap = u2 - u1
                if gap >= config.class_hypothesis_margin:
                    kept[0] = _replace(kept[0], preferred=True, ambiguous_class=False, evidence_weight=0.5)
                    kept[1] = _replace(kept[1], preferred=False, ambiguous_class=False, evidence_weight=0.5)
                else:
                    kept[0] = _replace(kept[0], preferred=False, ambiguous_class=True, evidence_weight=0.5)
                    kept[1] = _replace(kept[1], preferred=False, ambiguous_class=True, evidence_weight=0.5)
                two_count += 1
            else:
                zero_count += 1
            hypotheses.extend(kept)

        return TargetHypothesisScanResult(
            hypotheses=tuple(hypotheses),
            num_samples=num_samples,
            num_pairwise_attempted=counts["attempted"],
            num_pre_support_rejected=counts["pre_support"],
            num_solver_failed=counts["solver_failed"],
            num_gamma_rejected=counts["gamma_rejected"],
            num_gain_rejected=counts["gain_rejected"],
            num_shape_support_rejected=counts["shape_support_rejected"],
            num_outer_rejected=counts["outer_rejected"],
            samples_with_zero_hypothesis=zero_count,
            samples_with_one_hypothesis=one_count,
            samples_with_two_hypotheses=two_count,
        )
    finally:
        model.train(was_training)


def _replace(hypothesis: TargetClassPhaseHypothesis, **changes) -> TargetClassPhaseHypothesis:
    return TargetClassPhaseHypothesis(
        **{**hypothesis.__dict__, **changes}
    )
