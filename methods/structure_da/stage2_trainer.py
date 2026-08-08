"""Round-7 orchestration for the frozen TSStructure V3 Stage-2 adaptation.

This module wires the already-frozen Round 3--6 geometry/statistics modules to
one blockwise training loop.  It intentionally contains no new domain loss,
clustering rule, registration solver or pseudo-label CE.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .confirmed_phase_view import (
    build_confirmed_class_to_group_map,
    build_confirmed_phase_view,
)
from .domain_phase_state import (
    DomainPhaseConfig,
    DomainPhaseState,
    PhaseDecisionStatus,
    PhaseGroupStatus,
    update_domain_phase_state,
)
from .domain_shape_state import (
    DomainShapeConfig,
    DomainShapeState,
    DomainShapeStatus,
    update_domain_shape_state,
)
from .ema_teacher import Stage2EMATeacher
from .phase_registration import SourceRegistrationPrototypeBank
from .prototype_bank import SourcePrototypeBank, support_aware_q_distance
from .shape_transport import (
    SyntheticSourceExample,
    build_phase_only_synthetic_source_example,
    build_synthetic_source_example,
    evaluate_synthetic_source_diagnostics,
)
from .source_prototype_scanner import refresh_source_fused_statistics
from .stable_target_labels import (
    StableLabelConfig,
    StableTargetLabelScanResult,
    scan_stable_target_labels_from_candidates,
)
from .stage2_objective import Stage2Objective, Stage2ObjectiveConfig
from .stage2_parameter_policy import Stage2ParameterPolicy
from .stage2_calibration import export_stage2_calibration_statistics
from .target_hypothesis_scan import (
    PhaseHypothesisScanConfig,
    TargetHypothesisScanResult,
    TargetPhaseHypothesisScanner,
)
from .temporal_srvf import TemporalSRVFExtractor


@dataclass(frozen=True)
class Stage2TrainerConfig:
    phase_scan: PhaseHypothesisScanConfig
    phase: DomainPhaseConfig
    stable_labels: StableLabelConfig
    shape: DomainShapeConfig
    objective: Stage2ObjectiveConfig
    ema_decay: float
    lambda_delta: float
    total_epochs: int = 60
    adaptation_block_epochs: int = 20
    steps_per_epoch: int | None = None
    amp_enabled: bool = False
    amp_dtype: str = "float16"
    phase_evidence_initial_samples: int = 64
    phase_evidence_max_samples: int = 512
    evidence_seed: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.ema_decay)) or not 0.0 <= float(self.ema_decay) < 1.0:
            raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")
        if not math.isfinite(float(self.lambda_delta)) or not 0.0 <= float(self.lambda_delta) <= 1.0:
            raise ValueError("lambda_delta must lie in [0,1]")
        for name in ("total_epochs", "adaptation_block_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.steps_per_epoch is not None and (
            isinstance(self.steps_per_epoch, bool)
            or not isinstance(self.steps_per_epoch, int)
            or self.steps_per_epoch < 1
        ):
            raise ValueError("steps_per_epoch must be a positive integer or None")
        if self.amp_dtype not in ("float16", "bfloat16"):
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
        for name in ("phase_evidence_initial_samples", "phase_evidence_max_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.phase_evidence_initial_samples > self.phase_evidence_max_samples:
            raise ValueError(
                "phase_evidence_initial_samples cannot exceed phase_evidence_max_samples"
            )


@dataclass(frozen=True)
class Stage2StatisticsSnapshot:
    phase_state: DomainPhaseState
    stable_labels: StableTargetLabelScanResult
    shape_state: DomainShapeState


@dataclass(frozen=True)
class TargetHypothesisCache:
    source_geometry_version: int
    result: TargetHypothesisScanResult


class DeviceBatchLoader:
    """Iterate an existing deterministic loader with tensors moved to one device."""

    def __init__(self, loader, device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield {
                key: value.to(device=self.device) if isinstance(value, Tensor) else value
                for key, value in batch.items()
            }


def build_stage2_registration_extractor(
    model: nn.Module,
    *,
    device: torch.device,
    k_reg: int = 128,
) -> TemporalSRVFExtractor:
    """Build the already-frozen Round-3 K_reg extractor without registration."""
    structure = model.temporal_module.structure_geometry
    functional = structure.functional_lift
    extractor = type(structure)(
        feature_dim=model.backbone.feature_dim,
        num_basis=functional.num_basis,
        canonical_grid_size=k_reg,
        roughness_grid_size=functional.roughness_grid_size,
        smoothing_weight=functional.smoothing_weight,
        time_reference=0.0,
        time_scale=1.0,
        min_mean_support=0.0,
        min_dynamic_energy=0.0,
    )
    return extractor.to(device=device)


@dataclass(frozen=True)
class Stage2RunResult:
    best_target_val_f1: float
    best_target_val_epoch: int | None
    final_diagnostic_target_test: dict | None
    adaptation_performed: bool = True
    abstain_reason: str | None = None


def _integration_weights(model: nn.Module, device: torch.device) -> Tensor:
    grid = model.temporal_module.structure_geometry.functional_lift.canonical_grid
    weights = torch.ones_like(grid, device=device, dtype=torch.float32)
    weights[[0, -1]] *= 0.5
    return weights / weights.sum()


def _batch_tensor(batch: dict, name: str, device: torch.device):
    value = batch.get(name)
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise ValueError(f"batch[{name!r}] must be a tensor")
    return value.to(device=device)


def _source_sample_ids(batch: dict, batch_size: int) -> list[int]:
    value = batch.get("parcel_index")
    if isinstance(value, Tensor) and value.shape == (batch_size,):
        return [int(item) for item in value.tolist()]
    return list(range(batch_size))


def _stack_synthetic(
    examples: list[SyntheticSourceExample],
    valid_flags: list[Tensor],
    *,
    device: torch.device,
) -> dict[str, Tensor] | None:
    if not examples:
        return None
    return {
        "trend": torch.stack([item.trend_tokens for item in examples]).to(device=device),
        "structure": torch.stack([item.structure_tokens for item in examples]).to(device=device),
        "positions": torch.stack([item.target_style_positions for item in examples]).to(device=device),
        "mask": torch.stack([item.mask for item in examples]).to(device=device),
        "labels": torch.tensor([item.class_id for item in examples], device=device, dtype=torch.long),
        "q": torch.stack([item.q_shape for item in examples]).to(device=device),
        "q_support": torch.stack([item.q_support for item in examples]).to(device=device),
        "q_valid": torch.stack(valid_flags).to(device=device, dtype=torch.bool),
    }


def _confirmed_phase_exists(state: DomainPhaseState) -> bool:
    return state.decision_status in (
        PhaseDecisionStatus.IDENTITY_CONFIRMED,
        PhaseDecisionStatus.NONIDENTITY_CONFIRMED,
    )


def _adaptation_available(snapshot: Stage2StatisticsSnapshot) -> bool:
    decision = snapshot.phase_state.decision_status
    if decision is PhaseDecisionStatus.NONIDENTITY_CONFIRMED:
        return True
    if decision is PhaseDecisionStatus.IDENTITY_CONFIRMED:
        return snapshot.shape_state.status is DomainShapeStatus.CONFIRMED
    return False


def _phase_groups_log_value(state: DomainPhaseState) -> str:
    if not state.groups:
        return "-"
    return ";".join(
        f"{group.group_id}:{group.status.value}:{','.join(str(c) for c in group.member_classes)}"
        for group in state.groups
    )


def _optional_metric(value: float | None) -> str:
    return "none" if value is None else f"{float(value):.6g}"


def _phase_summary(state: DomainPhaseState) -> dict:
    confirmed_membership = {
        str(class_id): group.group_id
        for group in state.groups
        if group.status is PhaseGroupStatus.CONFIRMED
        for class_id in group.member_classes
    }
    return {
        "scan_index": state.scan_index,
        "m": state.m,
        "decision_status": state.decision_status.value,
        "decision_stability_age": state.decision_stability_age,
        "identity_evidence_classes": list(state.identity_evidence_classes),
        "identity_evidence_count": state.identity_evidence_count,
        "residual_evidence_count": state.residual_evidence_count,
        "residual_evidence_classes": list(state.residual_evidence_classes),
        "valid_phase_classes": list(state.valid_phase_classes),
        "rejected_classes": list(state.rejected_classes),
        "g0_classes": list(state.rejected_classes),
        "confirmed_group_membership": confirmed_membership,
        "groups": [
            {
                "group_id": group.group_id,
                "member_classes": list(group.member_classes),
                "status": group.status.value,
                "confirmation_age": group.confirmation_age,
                "center_drift": group.center_drift,
            }
            for group in state.groups
        ],
    }


def _shape_summary(state: DomainShapeState) -> dict:
    delta_norm = None
    if state.delta is not None:
        grid_size = state.delta.shape[0]
        weights = torch.full(
            (grid_size,),
            1.0 / (grid_size - 1),
            device=state.delta.device,
            dtype=state.delta.dtype,
        )
        weights[[0, -1]] *= 0.5
        delta_norm = float(torch.sqrt((weights * state.delta.square().sum(-1)).sum()).item())
    return {
        "scan_index": state.scan_index,
        "status": state.status.value,
        "valid_classes": list(state.valid_classes),
        "rho_shape": state.rho_shape,
        "delta_norm": delta_norm,
        "leave_one_out_drift": state.leave_one_out_drift,
        "center_drift": state.center_drift,
        "confirmation_age": state.confirmation_age,
    }


def _phase_state_payload(state: DomainPhaseState) -> dict:
    return {
        "scan_index": state.scan_index,
        "m": state.m,
        "decision_status": state.decision_status.value,
        "decision_stability_age": state.decision_stability_age,
        "identity_evidence_classes": tuple(state.identity_evidence_classes),
        "identity_evidence_count": state.identity_evidence_count,
        "residual_evidence_count": state.residual_evidence_count,
        "residual_evidence_classes": tuple(state.residual_evidence_classes),
        "valid_phase_classes": tuple(state.valid_phase_classes),
        "rejected_classes": tuple(state.rejected_classes),
        "class_centers": tuple(
            {
                "class_id": item.class_id,
                "center_gamma": item.center_gamma.detach().cpu(),
                "candidate_count": item.candidate_count,
                "effective_evidence_count": item.effective_evidence_count,
                "dispersion": item.dispersion,
                "diameter": item.diameter,
                "median_distance": item.median_distance,
                "center_drift": item.center_drift,
                "valid": item.valid,
                "reject_reason": item.reject_reason,
            }
            for item in state.class_centers
        ),
        "groups": tuple(
            {
                "group_id": group.group_id,
                "member_classes": tuple(group.member_classes),
                "center_gamma": group.center_gamma.detach().cpu(),
                "within_dispersion": group.within_dispersion,
                "diameter": group.diameter,
                "core_radius": group.core_radius,
                "sample_evidence_count": group.sample_evidence_count,
                "class_count": group.class_count,
                "center_drift": group.center_drift,
                "status": group.status.value,
                "confirmation_age": group.confirmation_age,
            }
            for group in state.groups
        ),
    }


def _shape_state_payload(state: DomainShapeState) -> dict:
    return {
        "scan_index": state.scan_index,
        "status": state.status.value,
        "valid_classes": tuple(state.valid_classes),
        "delta": None if state.delta is None else state.delta.detach().cpu(),
        "interactions": tuple(item.detach().cpu() for item in state.interactions),
        "rho_shape": state.rho_shape,
        "leave_one_out_drift": state.leave_one_out_drift,
        "center_drift": state.center_drift,
        "confirmation_age": state.confirmation_age,
        "class_centers": tuple(
            {
                "class_id": item.class_id,
                "center_q": item.center_q.detach().cpu(),
                "center_support": item.center_support.detach().cpu(),
                "sample_count": item.sample_count,
                "effective_weight": item.effective_weight,
                "source_distance": item.source_distance,
                "residual_q": item.residual_q.detach().cpu(),
                "valid": item.valid,
                "reject_reason": item.reject_reason,
            }
            for item in state.class_centers
        ),
    }


def _stable_label_payload(result: StableTargetLabelScanResult) -> dict:
    return {
        "num_samples": result.num_samples,
        "num_stable_labels": result.num_stable_labels,
        "stable_class_counts": tuple(result.stable_class_counts),
        "num_phase_compatible": result.num_phase_compatible,
        "num_phase_incompatible": result.num_phase_incompatible,
        "labels": tuple(
            {
                "sample_id": item.sample_id,
                "class_id": item.class_id,
                "group_id": item.group_id,
                "confidence_summary": item.confidence_summary,
            }
            for item in result.stable_labels
        ),
        "candidates": tuple(
            {
                "sample_id": item.sample_id,
                "class_id": item.class_id,
                "group_id": item.group_id,
                "phase_compatible": item.phase_compatible,
                "phase_distance_to_group": item.phase_distance_to_group,
                "candidate_q_shape_distance": item.candidate_q_shape_distance,
                "candidate_ambiguous": item.candidate_ambiguous,
                "passed_classifier": item.passed_classifier,
                "passed_fused": item.passed_fused,
                "passed_q": item.passed_q,
                "accepted": item.accepted,
                "reject_reason": item.reject_reason,
            }
            for item in result.candidates
        ),
    }


def _target_hypothesis_payload(result: TargetHypothesisScanResult) -> dict:
    """Checkpoint the frozen Round-A/B evidence without rerunning registration."""
    return {
        "num_samples": result.num_samples,
        "scanned_sample_ids": tuple(result.scanned_sample_ids),
        "candidate_pseudo_labels": tuple(
            {
                "sample_id": item.sample_id,
                "class_id": item.class_id,
                "q_shape_distance": item.q_shape_distance,
                "q_distance_percentile": item.q_distance_percentile,
                "phase_evidence_eligible": item.phase_evidence_eligible,
                "ambiguous": item.ambiguous,
                "secondary_class_id": item.secondary_class_id,
                "secondary_q_shape_distance": item.secondary_q_shape_distance,
                "secondary_phase_evidence_eligible": item.secondary_phase_evidence_eligible,
            }
            for item in result.candidate_pseudo_labels
        ),
        "pairwise_alignments": tuple(
            {
                "sample_id": item.sample_id,
                "class_id": item.class_id,
                "gamma": None if item.gamma is None else item.gamma.detach().cpu(),
                "t_identity_error": item.t_identity_error,
                "t_registered_error": item.t_registered_error,
                "t_gain_ratio": item.t_gain_ratio,
                "pre_common_support_t": item.pre_common_support_t,
                "common_support_t": item.common_support_t,
                "q_shape_distance": item.q_shape_distance,
                "q_distance_percentile": item.q_distance_percentile,
                "common_support_shape": item.common_support_shape,
                "numerically_valid": item.numerically_valid,
                "phase_evidence_eligible": item.phase_evidence_eligible,
                "reject_reasons": tuple(item.reject_reasons),
                "solver_error": item.solver_error,
            }
            for item in result.pairwise_alignments
        ),
    }


def _bank_to_cpu(bank: SourcePrototypeBank) -> dict:
    return {
        "trend_srvf": bank.trend_srvf.detach().cpu(),
        "shape_srvf": bank.shape_srvf.detach().cpu(),
        "trend_support": bank.trend_support.detach().cpu(),
        "shape_support": bank.shape_support.detach().cpu(),
        "fused": bank.fused.detach().cpu(),
        "class_counts": bank.class_counts.detach().cpu(),
        "ready": bank.ready.detach().cpu(),
        "q_distance_samples": tuple(item.detach().cpu() for item in bank.q_distance_samples),
        "f_distance_samples": tuple(item.detach().cpu() for item in bank.f_distance_samples),
        "q_quantiles": bank.q_quantiles.detach().cpu(),
        "f_quantiles": bank.f_quantiles.detach().cpu(),
        "version": bank.version,
    }


class Stage2Trainer:
    """Train the explicit Stage-2 student against block-frozen statistics."""

    def __init__(
        self,
        *,
        student: nn.Module,
        policy: Stage2ParameterPolicy,
        ema_teacher: Stage2EMATeacher,
        optimizer: Optimizer,
        source_loader,
        source_scan_loader,
        target_statistics_loader,
        source_prototype_bank: SourcePrototypeBank,
        source_registration_bank: SourceRegistrationPrototypeBank,
        reg_extractor: TemporalSRVFExtractor,
        config: Stage2TrainerConfig,
        device: torch.device,
        output_dir: str,
        runtime_config: dict | None = None,
        writer=None,
    ) -> None:
        self.student = student
        self.policy = policy
        self.ema_teacher = ema_teacher
        self.optimizer = optimizer
        self.source_loader = source_loader
        self.source_scan_loader = source_scan_loader
        self.target_statistics_loader = target_statistics_loader
        self.source_prototype_bank = source_prototype_bank
        self.source_geometry_version = int(source_prototype_bank.version)
        self.source_registration_bank = source_registration_bank
        self.reg_extractor = reg_extractor
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.runtime_config = {} if runtime_config is None else dict(runtime_config)
        self.writer = writer
        self.objective = Stage2Objective(
            num_classes=int(source_prototype_bank.ready.numel()),
            config=config.objective,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                config.amp_enabled
                and device.type == "cuda"
                and config.amp_dtype == "float16"
            ),
        )
        self.hypothesis_cache: TargetHypothesisCache | None = None
        self.phase_scanner: TargetPhaseHypothesisScanner | None = None
        self.statistics: Stage2StatisticsSnapshot | None = None
        self.hypothesis_scan_count = 0
        self.phase_evidence_stages = 0
        self.shape_evidence_stages = 0
        self.shape_evidence_sample_ids: tuple[int, ...] = ()
        self.successful_optimizer_steps = 0
        self._validate_optimizer_boundary()

    def _validate_optimizer_boundary(self) -> None:
        expected = {
            id(parameter)
            for name, parameter in self.student.named_parameters()
            if name in set(self.policy.trainable_parameter_names)
        }
        actual = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        if actual != expected:
            raise ValueError(
                "Stage-2 optimizer parameters must exactly match Stage2ParameterPolicy.trainable_parameter_names"
            )

    def _phase_evidence_budgets(self, available: int) -> tuple[int, ...]:
        maximum = min(self.config.phase_evidence_max_samples, available)
        current = min(self.config.phase_evidence_initial_samples, maximum)
        budgets: list[int] = []
        while current < maximum:
            budgets.append(current)
            current = min(maximum, current * 2)
        if not budgets or budgets[-1] != maximum:
            budgets.append(maximum)
        return tuple(budgets)

    def _get_phase_scanner(self) -> TargetPhaseHypothesisScanner:
        if self.phase_scanner is None:
            self.phase_scanner = TargetPhaseHypothesisScanner(
                self.ema_teacher.model(),
                self.target_statistics_loader,
                self.source_prototype_bank,
                self.source_registration_bank,
                self.config.phase_scan,
                device=self.device,
                shape_extractor=self.student.temporal_module.structure_geometry,
                reg_extractor=self.reg_extractor,
                evidence_seed=self.config.evidence_seed,
            )
            self.hypothesis_scan_count += 1
        return self.phase_scanner

    def _stable_and_shape_from_fixed_phase(
        self,
        phase_state: DomainPhaseState,
        previous_shape: DomainShapeState | None,
        *,
        sample_ids: tuple[int, ...],
    ) -> tuple[StableTargetLabelScanResult, DomainShapeState]:
        if self.hypothesis_cache is None:
            raise RuntimeError("target hypothesis cache is unavailable")
        stable_result = scan_stable_target_labels_from_candidates(
            ema_teacher=self.ema_teacher,
            target_loader=self.target_statistics_loader,
            hypothesis_result=self.hypothesis_cache.result,
            phase_state=phase_state,
            phase_config=self.config.phase,
            source_prototype_bank=self.source_prototype_bank,
            config=self.config.stable_labels,
            sample_ids=sample_ids,
        )
        shape_state = update_domain_shape_state(
            stable_result,
            self.source_prototype_bank,
            self.config.shape,
            previous_state=previous_shape,
        )
        return stable_result, shape_state

    @torch.no_grad()
    def initialize_statistics(self) -> Stage2StatisticsSnapshot:
        """Acquire progressive Domain Phase evidence, then estimate Shape.

        Round B deliberately scans every configured nested evidence budget.
        A confirmed group at an intermediate budget is not treated as proof
        that the complete domain-level model order is settled: later evidence
        may reveal a second multi-class Phase group.  Exact pairwise gammas are
        cached by the scanner and never recomputed at block boundaries.
        """
        scanner = self._get_phase_scanner()
        phase_state: DomainPhaseState | None = None
        final_result: TargetHypothesisScanResult | None = None

        for budget in self._phase_evidence_budgets(scanner.total_cached_samples):
            final_result = scanner.scan_to_budget(budget)
            self.phase_evidence_stages += 1
            phase_state = update_domain_phase_state(
                final_result,
                self.config.phase,
                previous_state=phase_state,
            )
            print(
                "STAGE2_PHASE_EVIDENCE_STAGE|"
                f"budget={budget}|phase_scan_index={phase_state.scan_index}"
                f"|phase_m={phase_state.m}"
                f"|phase_decision={phase_state.decision_status.value}"
                f"|decision_age={phase_state.decision_stability_age}"
                f"|confirmed_phase={str(_confirmed_phase_exists(phase_state)).lower()}"
                f"|valid_classes={','.join(str(c) for c in phase_state.valid_phase_classes) or '-'}"
                f"|groups={_phase_groups_log_value(phase_state)}"
                f"|g0={','.join(str(c) for c in phase_state.rejected_classes) or '-'}"
                f"|hypotheses={len(final_result.hypotheses)}"
                f"|residual_evidence={phase_state.residual_evidence_count}"
                f"|residual_classes={','.join(str(c) for c in phase_state.residual_evidence_classes) or '-'}"
                f"|identity_classes={','.join(str(c) for c in phase_state.identity_evidence_classes) or '-'}"
                f"|solver_calls={final_result.num_solver_calls}"
            )

        assert final_result is not None
        assert phase_state is not None
        self.hypothesis_cache = TargetHypothesisCache(
            source_geometry_version=self.source_geometry_version,
            result=final_result,
        )

        shape_state: DomainShapeState | None = None
        stable_result: StableTargetLabelScanResult | None = None
        if _confirmed_phase_exists(phase_state):
            for budget in self._phase_evidence_budgets(scanner.total_cached_samples):
                sample_ids = scanner.sample_ids_for_budget(budget)
                stable_result, shape_state = self._stable_and_shape_from_fixed_phase(
                    phase_state,
                    shape_state,
                    sample_ids=sample_ids,
                )
                self.shape_evidence_stages += 1
                self.shape_evidence_sample_ids = sample_ids
                stable_coverage = (
                    stable_result.num_stable_labels / stable_result.num_samples
                    if stable_result.num_samples
                    else 0.0
                )
                shape_metrics = _shape_summary(shape_state)
                print(
                    "STAGE2_SHAPE_EVIDENCE_STAGE|"
                    f"budget={budget}|stable_labels={stable_result.num_stable_labels}"
                    f"|stable_coverage={stable_coverage:.4f}"
                    f"|candidate_views={stable_result.num_candidate_views}"
                    f"|phase_compatible={stable_result.num_phase_compatible}"
                    f"|phase_incompatible={stable_result.num_phase_incompatible}"
                    f"|cls_pass={stable_result.num_classifier_pass}"
                    f"|fused_pass={stable_result.num_fused_pass}"
                    f"|q_pass={stable_result.num_q_pass}"
                    f"|ambiguous={stable_result.num_ambiguous_rejected}"
                    f"|shape_status={shape_state.status.value}"
                    f"|shape_valid_classes={','.join(str(c) for c in shape_state.valid_classes) or '-'}"
                    f"|rho_shape={_optional_metric(shape_state.rho_shape)}"
                    f"|delta_norm={_optional_metric(shape_metrics['delta_norm'])}"
                    f"|loo_drift={_optional_metric(shape_state.leave_one_out_drift)}"
                    f"|center_drift={_optional_metric(shape_state.center_drift)}"
                )
                if shape_state.status is DomainShapeStatus.CONFIRMED:
                    break
        else:
            self.shape_evidence_sample_ids = scanner.sample_ids_for_budget(
                final_result.num_samples
            )
            stable_result, shape_state = self._stable_and_shape_from_fixed_phase(
                phase_state,
                None,
                sample_ids=self.shape_evidence_sample_ids,
            )

        assert stable_result is not None
        assert shape_state is not None
        snapshot = Stage2StatisticsSnapshot(
            phase_state=phase_state,
            stable_labels=stable_result,
            shape_state=shape_state,
        )
        self.statistics = snapshot
        print(
            "STAGE2_STATISTICS|"
            f"scan_index={phase_state.scan_index}|phase_m={phase_state.m}"
            f"|phase_decision={phase_state.decision_status.value}"
            f"|confirmed_phase={str(_confirmed_phase_exists(phase_state)).lower()}"
            f"|stable_labels={stable_result.num_stable_labels}"
            f"|shape_status={shape_state.status.value}"
            f"|hypothesis_scans={self.hypothesis_scan_count}"
            f"|phase_evidence_stages={self.phase_evidence_stages}"
            f"|phase_evidence_samples={final_result.num_samples}"
            f"|shape_evidence_stages={self.shape_evidence_stages}"
            f"|shape_evidence_samples={len(self.shape_evidence_sample_ids)}"
        )
        return snapshot

    @torch.no_grad()
    def settle_statistics(
        self,
        *,
        previous: Stage2StatisticsSnapshot | None,
    ) -> Stage2StatisticsSnapshot:
        """Refresh stable-label/Shape statistics while freezing Domain Phase.

        Exact registration is never repeated while source geometry is frozen.
        A block boundary supplies a genuinely new EMA/fused representation, so
        it may advance Domain Shape confirmation, but it cannot manufacture a
        new Phase confirmation from replayed hypotheses.
        """
        if previous is None:
            return self.initialize_statistics()
        if self.hypothesis_cache is None:
            raise RuntimeError("target hypothesis cache is unavailable")
        stable_result, shape_state = self._stable_and_shape_from_fixed_phase(
            previous.phase_state,
            previous.shape_state,
            sample_ids=self.shape_evidence_sample_ids,
        )
        snapshot = Stage2StatisticsSnapshot(
            phase_state=previous.phase_state,
            stable_labels=stable_result,
            shape_state=shape_state,
        )
        self.statistics = snapshot
        shape_metrics = _shape_summary(shape_state)
        stable_coverage = (
            stable_result.num_stable_labels / stable_result.num_samples
            if stable_result.num_samples
            else 0.0
        )
        print(
            "STAGE2_STATISTICS_REFRESH|"
            f"phase_m={previous.phase_state.m}"
            f"|phase_decision={previous.phase_state.decision_status.value}"
            f"|confirmed_phase={str(_confirmed_phase_exists(previous.phase_state)).lower()}"
            f"|stable_labels={stable_result.num_stable_labels}"
            f"|stable_coverage={stable_coverage:.4f}"
            f"|phase_compatible={stable_result.num_phase_compatible}"
            f"|phase_incompatible={stable_result.num_phase_incompatible}"
            f"|cls_pass={stable_result.num_classifier_pass}"
            f"|fused_pass={stable_result.num_fused_pass}"
            f"|q_pass={stable_result.num_q_pass}"
            f"|shape_status={shape_state.status.value}"
            f"|rho_shape={_optional_metric(shape_state.rho_shape)}"
            f"|delta_norm={_optional_metric(shape_metrics['delta_norm'])}"
            f"|loo_drift={_optional_metric(shape_state.leave_one_out_drift)}"
            f"|center_drift={_optional_metric(shape_state.center_drift)}"
            f"|hypothesis_scans={self.hypothesis_scan_count}"
        )
        return snapshot

    @torch.no_grad()
    def refresh_source_features_and_statistics(self) -> Stage2StatisticsSnapshot:
        """Refresh EMA fused source state, then reuse cached target hypotheses."""
        self.source_prototype_bank = refresh_source_fused_statistics(
            self.ema_teacher.model(),
            self.source_scan_loader,
            self.source_prototype_bank,
            device=self.device,
        )
        return self.settle_statistics(previous=self.statistics)

    def _source_forward(self, batch: dict):
        pixels = _batch_tensor(batch, "pixels", self.device)
        valid_pixels = _batch_tensor(batch, "valid_pixels", self.device)
        positions = _batch_tensor(batch, "positions", self.device)
        extra = _batch_tensor(batch, "extra", self.device)
        time_mask = _batch_tensor(batch, "time_mask", self.device)
        labels = _batch_tensor(batch, "label", self.device)
        if pixels is None or valid_pixels is None or positions is None or labels is None:
            raise ValueError("source batch must contain pixels, valid_pixels, positions and label")
        labels = labels.to(dtype=torch.long)
        with torch.no_grad():
            backbone = self.student.forward_backbone(
                pixels,
                valid_pixels,
                positions,
                extra,
                time_mask=time_mask,
            )
            trend, structure = self.student._trend_and_structure(backbone)
            structure_geometry = self.student.temporal_module.structure_geometry(
                structure,
                backbone.normalized_positions,
                backbone.time_mask,
            )
            trend = trend.detach()
            structure = structure.detach()
            normalized_positions = backbone.normalized_positions.detach()
            mask = backbone.time_mask.detach()
        amp_dtype = getattr(torch, self.config.amp_dtype)
        amp_on = self.config.amp_enabled and (
            self.device.type == "cuda" or amp_dtype == torch.bfloat16
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=amp_dtype,
            enabled=amp_on,
        ):
            raw = self.student.temporal_module.raw_encoder(
                trend=trend,
                structure=structure,
                positions=normalized_positions,
                mask=mask,
            )
            logits = self.student.classifier(raw.fused_repr)
        return (
            logits,
            raw.fused_repr,
            labels,
            trend,
            structure,
            normalized_positions,
            mask,
            structure_geometry,
        )

    @torch.no_grad()
    def _build_synthetic_batch(
        self,
        batch: dict,
        *,
        trend: Tensor,
        structure: Tensor,
        positions: Tensor,
        mask: Tensor,
        labels: Tensor,
        structure_geometry,
    ) -> dict[str, Tensor] | None:
        if self.statistics is None:
            raise RuntimeError("Stage-2 statistics must be initialized before training")
        phase_state = self.statistics.phase_state
        if not _confirmed_phase_exists(phase_state):
            return None
        shape_state = self.statistics.shape_state
        shape_confirmed = shape_state.status is DomainShapeStatus.CONFIRMED
        sample_ids = _source_sample_ids(batch, labels.shape[0])
        examples: list[SyntheticSourceExample] = []
        valid_flags: list[Tensor] = []
        for row in range(labels.shape[0]):
            class_id = int(labels[row].item())
            if shape_confirmed:
                example = build_synthetic_source_example(
                    source_sample_id=sample_ids[row],
                    class_id=class_id,
                    source_structure_function=structure_geometry.functional.function[row],
                    source_q_shape=structure_geometry.srvf[row],
                    source_q_support=structure_geometry.support_confidence[row],
                    source_positions=positions[row],
                    mask=mask[row],
                    phase_state=phase_state,
                    domain_shape_state=shape_state,
                    decomposition=self.student.backbone.decomposition,
                    lambda_delta=self.config.lambda_delta,
                )
            else:
                example = build_phase_only_synthetic_source_example(
                    source_sample_id=sample_ids[row],
                    class_id=class_id,
                    source_trend_tokens=trend[row],
                    source_structure_tokens=structure[row],
                    source_q_shape=structure_geometry.srvf[row],
                    source_q_support=structure_geometry.support_confidence[row],
                    source_positions=positions[row],
                    mask=mask[row],
                    phase_state=phase_state,
                )
            if example is not None:
                examples.append(example)
                valid_flags.append(structure_geometry.structure_valid[row].detach())
        return _stack_synthetic(examples, valid_flags, device=self.device)

    def _set_student_training_modes(self) -> None:
        """Train only the Stage-2 task path while frozen geometry stays deterministic."""
        self.student.train()
        self.student.backbone.eval()
        self.student.temporal_module.trend_geometry.eval()
        self.student.temporal_module.structure_geometry.eval()

    def train_step(self, batch: dict) -> dict[str, float]:
        if self.statistics is None:
            raise RuntimeError("Stage-2 statistics must be initialized before training")
        if not _adaptation_available(self.statistics):
            raise RuntimeError(
                "Stage-2 optimizer step is forbidden without confirmed structural adaptation evidence"
            )
        self._set_student_training_modes()
        self.optimizer.zero_grad(set_to_none=True)
        (
            source_logits,
            source_fused,
            source_labels,
            trend,
            structure,
            positions,
            mask,
            structure_geometry,
        ) = self._source_forward(batch)
        synthetic = self._build_synthetic_batch(
            batch,
            trend=trend,
            structure=structure,
            positions=positions,
            mask=mask,
            labels=source_labels,
            structure_geometry=structure_geometry,
        )

        synthetic_logits = None
        if synthetic is not None:
            amp_dtype = getattr(torch, self.config.amp_dtype)
            amp_on = self.config.amp_enabled and (
                self.device.type == "cuda" or amp_dtype == torch.bfloat16
            )
            with torch.autocast(
                device_type=self.device.type,
                dtype=amp_dtype,
                enabled=amp_on,
            ):
                synthetic_raw = self.student.temporal_module.raw_encoder(
                    trend=synthetic["trend"],
                    structure=synthetic["structure"],
                    positions=synthetic["positions"],
                    mask=synthetic["mask"],
                )
                synthetic_logits = self.student.classifier(synthetic_raw.fused_repr)

        shape_state = self.statistics.shape_state
        objective_output = self.objective(
            source_logits=source_logits,
            source_fused_repr=source_fused,
            source_labels=source_labels,
            source_q=structure_geometry.srvf.detach(),
            source_q_support=structure_geometry.support_confidence.detach(),
            source_q_valid=structure_geometry.structure_valid.detach(),
            source_prototype_bank=self.source_prototype_bank,
            integration_weights=_integration_weights(self.student, self.device),
            synthetic_logits=synthetic_logits,
            synthetic_labels=None if synthetic is None else synthetic["labels"],
            synthetic_q=None if synthetic is None else synthetic["q"],
            synthetic_q_support=None if synthetic is None else synthetic["q_support"],
            synthetic_q_valid=None if synthetic is None else synthetic["q_valid"],
            domain_shape_state=(
                shape_state
                if synthetic is not None and shape_state.status is DomainShapeStatus.CONFIRMED
                else None
            ),
            lambda_delta=(
                self.config.lambda_delta
                if synthetic is not None and shape_state.status is DomainShapeStatus.CONFIRMED
                else None
            ),
        )
        previous_scale = float(self.scaler.get_scale())
        self.scaler.scale(objective_output.total).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        new_scale = float(self.scaler.get_scale())
        step_succeeded = not self.scaler.is_enabled() or new_scale >= previous_scale
        if step_succeeded:
            self.ema_teacher.update_after_optimizer_step(self.student)
            self.successful_optimizer_steps += 1

        return {
            "loss": float(objective_output.total.detach().item()),
            "source_cls": float(objective_output.source_cls.detach().item()),
            "source_proto": float(objective_output.source_proto.detach().item()),
            "source_consistency": float(objective_output.source_consistency.detach().item()),
            "synthetic_cls": float(objective_output.synthetic_cls.detach().item()),
            "synthetic_consistency": float(objective_output.synthetic_consistency.detach().item()),
            "source_count": float(objective_output.source_count),
            "synthetic_count": float(objective_output.synthetic_count),
            "optimizer_step_succeeded": float(step_succeeded),
        }

    def train_epoch(self, epoch: int) -> dict[str, float]:
        meters: dict[str, float] = {}
        steps = 0
        limit = self.config.steps_per_epoch or len(self.source_loader)
        for batch in self.source_loader:
            if steps >= limit:
                break
            metrics = self.train_step(batch)
            for key, value in metrics.items():
                meters[key] = meters.get(key, 0.0) + value
            steps += 1
        if steps == 0:
            raise RuntimeError("source training loader produced no Stage-2 batches")
        averages = {key: value / steps for key, value in meters.items()}
        print(
            "STAGE2_TRAIN|"
            f"epoch={epoch}/{self.config.total_epochs}|steps={steps}"
            f"|loss={averages['loss']:.4f}|source_cls={averages['source_cls']:.4f}"
            f"|source_proto={averages['source_proto']:.4f}"
            f"|source_cons={averages['source_consistency']:.4f}"
            f"|synthetic_cls={averages['synthetic_cls']:.4f}"
            f"|synthetic_cons={averages['synthetic_consistency']:.4f}"
            f"|synthetic_count={averages['synthetic_count']:.2f}"
            f"|optimizer_step_success={averages['optimizer_step_succeeded']:.2f}"
        )
        if self.writer is not None:
            for key, value in averages.items():
                self.writer.add_scalar(f"stage2/train/{key}", value, epoch)
        return averages

    def write_calibration_statistics(self) -> dict[str, str]:
        if self.statistics is None or self.hypothesis_cache is None:
            raise RuntimeError("Stage-2 calibration requires initialized statistics and hypothesis cache")
        paths = export_stage2_calibration_statistics(
            output_dir=self.output_dir,
            hypothesis_result=self.hypothesis_cache.result,
            phase_state=self.statistics.phase_state,
            stable_result=self.statistics.stable_labels,
            shape_state=self.statistics.shape_state,
            source_prototype_bank=self.source_prototype_bank,
        )
        print(
            "STAGE2_CALIBRATION_EXPORT|"
            f"summary={paths['summary']}"
            f"|pairwise={paths['pairwise']}"
            f"|candidates={paths['candidates']}"
            f"|stable_candidates={paths['stable_candidates']}"
            f"|geometry={paths['geometry']}"
        )
        return paths

    @torch.no_grad()
    def write_final_shape_synthesis_audit(
        self,
        *,
        samples_per_class: int = 3,
        max_batches: int = 64,
    ) -> dict[str, str]:
        """Export the final non-gating Shape/synthesis audit for diagnostic runs.

        This routine is deliberately read-only: it reuses the already-confirmed
        Phase/Shape state, samples true-labelled *source* examples only, performs
        no registration, and never calls ``optimizer.step``.  The exported JSON
        answers two questions that are intentionally stronger than the aggregate
        ``rho_shape`` statistic:

        1. do the valid class residuals ``R_c`` point in the same direction as
           the shared ``Delta``; and
        2. does the actual source->target synthesis remain finite, preserve the
           source label, apply exactly the confirmed Phase coordinate map, and
           move q geometry toward the observed target class center when one is
           available?
        """
        if self.statistics is None:
            raise RuntimeError("Stage-2 statistics are unavailable")
        if isinstance(samples_per_class, bool) or samples_per_class < 1:
            raise ValueError("samples_per_class must be a positive integer")
        if isinstance(max_batches, bool) or max_batches < 1:
            raise ValueError("max_batches must be a positive integer")

        os.makedirs(self.output_dir, exist_ok=True)
        json_path = os.path.join(self.output_dir, "stage2_shape_synthesis_audit.json")
        tensor_path = os.path.join(self.output_dir, "stage2_shape_synthesis_audit.pt")

        phase_state = self.statistics.phase_state
        shape_state = self.statistics.shape_state
        payload: dict = {
            "phase_decision": phase_state.decision_status.value,
            "phase_m": int(phase_state.m),
            "shape_status": shape_state.status.value,
            "lambda_delta": float(self.config.lambda_delta),
            "optimizer_steps": int(self.successful_optimizer_steps),
            "shape": {},
            "synthesis": {},
        }
        tensor_payload: dict = {}

        if shape_state.delta is None:
            payload["shape"] = {"available": False, "reason": "delta_unavailable"}
            payload["synthesis"] = {"available": False, "reason": "delta_unavailable"}
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            torch.save(tensor_payload, tensor_path)
            return {"json": json_path, "tensors": tensor_path}

        delta = shape_state.delta.detach().to(device=self.device, dtype=torch.float32)
        weights = _integration_weights(self.student, self.device).to(delta)

        def weighted_inner(left: Tensor, right: Tensor) -> Tensor:
            return (weights * (left * right).sum(dim=-1)).sum()

        def weighted_norm(value: Tensor) -> Tensor:
            return torch.sqrt(weighted_inner(value, value).clamp_min(0.0))

        delta_norm_t = weighted_norm(delta)
        delta_norm = float(delta_norm_t.item())
        valid_centers = [item for item in shape_state.class_centers if item.valid]
        interaction_by_class = {
            int(class_id): interaction.detach().to(delta)
            for class_id, interaction in zip(shape_state.valid_classes, shape_state.interactions)
        }
        class_rows: list[dict] = []
        tensor_payload["delta"] = delta.detach().cpu()
        tensor_payload["class_rows"] = {}

        for center in valid_centers:
            class_id = int(center.class_id)
            residual = center.residual_q.detach().to(delta)
            interaction = interaction_by_class[class_id]
            residual_norm_t = weighted_norm(residual)
            interaction_norm_t = weighted_norm(interaction)
            denominator = residual_norm_t * delta_norm_t
            cosine = (
                float((weighted_inner(residual, delta) / denominator).item())
                if float(denominator.item()) > 1e-12
                else None
            )

            source_q = self.source_prototype_bank.shape_srvf[class_id].detach().to(delta)
            source_support = self.source_prototype_bank.shape_support[class_id].detach().to(delta)
            target_q = center.center_q.detach().to(delta)
            target_support = center.center_support.detach().to(delta)

            def distance_to_target(candidate_q: Tensor) -> float | None:
                output = support_aware_q_distance(
                    candidate_q.unsqueeze(0),
                    target_q.unsqueeze(0),
                    source_support.unsqueeze(0),
                    target_support.unsqueeze(0),
                    weights,
                )
                if not bool(output.valid[0, 0].item()):
                    return None
                return float(output.distance[0, 0].item())

            before = distance_to_target(source_q)
            lambda_after = distance_to_target(source_q + float(self.config.lambda_delta) * delta)
            full_after = distance_to_target(source_q + delta)
            common_support = (source_support * target_support).clamp(0.0, 1.0)
            active_common_support = common_support > 1e-6
            row = {
                "class_id": class_id,
                "sample_count": int(center.sample_count),
                "effective_weight": float(center.effective_weight),
                "source_distance": float(center.source_distance),
                "common_support_mean": float(common_support.mean().item()),
                "common_support_active_rate": float(active_common_support.float().mean().item()),
                "residual_norm": float(residual_norm_t.item()),
                "interaction_norm": float(interaction_norm_t.item()),
                "interaction_over_delta": (
                    float(interaction_norm_t.item()) / delta_norm if delta_norm > 0.0 else None
                ),
                "cos_residual_delta": cosine,
                "distance_to_target_before": before,
                "distance_to_target_after_lambda_delta": lambda_after,
                "distance_to_target_after_full_delta": full_after,
                "lambda_delta_improved": (
                    None if before is None or lambda_after is None else lambda_after <= before
                ),
                "full_delta_improved": (
                    None if before is None or full_after is None else full_after <= before
                ),
            }
            class_rows.append(row)
            tensor_payload["class_rows"][class_id] = {
                "source_q": source_q.cpu(),
                "source_support": source_support.cpu(),
                "target_center_q": target_q.cpu(),
                "target_center_support": target_support.cpu(),
                "common_support": common_support.cpu(),
                "residual_q": residual.cpu(),
                "interaction_q": interaction.cpu(),
            }

        cosines = [row["cos_residual_delta"] for row in class_rows if row["cos_residual_delta"] is not None]
        if valid_centers:
            common_support_stack = torch.stack([
                (
                    self.source_prototype_bank.shape_support[int(center.class_id)].detach().to(delta)
                    * center.center_support.detach().to(delta)
                ).clamp(0.0, 1.0)
                for center in valid_centers
            ])
            supported_class_count = (common_support_stack > 1e-6).sum(dim=0)
            insufficient_support_mask = supported_class_count < 2
            delta_energy_total = weighted_inner(delta, delta)
            delta_energy_low_support = (
                weights * insufficient_support_mask.to(delta.dtype) * delta.square().sum(dim=-1)
            ).sum()
            low_support_delta_energy_ratio = (
                float((delta_energy_low_support / delta_energy_total).item())
                if float(delta_energy_total.item()) > 1e-12
                else 0.0
            )
            min_supported_classes = int(supported_class_count.min().item())
            mean_supported_classes = float(supported_class_count.float().mean().item())
        else:
            low_support_delta_energy_ratio = None
            min_supported_classes = 0
            mean_supported_classes = 0.0
        payload["shape"] = {
            "available": True,
            "rho_shape": shape_state.rho_shape,
            "delta_norm": delta_norm,
            "leave_one_out_drift": shape_state.leave_one_out_drift,
            "center_drift": shape_state.center_drift,
            "confirmation_age": int(shape_state.confirmation_age),
            "valid_classes": list(shape_state.valid_classes),
            "min_cos_residual_delta": min(cosines) if cosines else None,
            "mean_cos_residual_delta": sum(cosines) / len(cosines) if cosines else None,
            "low_support_delta_energy_ratio": low_support_delta_energy_ratio,
            "min_supported_classes_per_grid_point": min_supported_classes,
            "mean_supported_classes_per_grid_point": mean_supported_classes,
            "all_full_delta_improved": all(
                row["full_delta_improved"] is True
                for row in class_rows
                if row["full_delta_improved"] is not None
            ),
            "class_rows": class_rows,
        }

        # Synthesis is meaningful only after the actual training preconditions
        # are satisfied.  The audit never relaxes these preconditions.
        if not _adaptation_available(self.statistics):
            payload["synthesis"] = {"available": False, "reason": "adaptation_unavailable"}
        elif shape_state.status is not DomainShapeStatus.CONFIRMED:
            payload["synthesis"] = {"available": False, "reason": "shape_not_confirmed"}
        else:
            confirmed_source_classes: set[int]
            if phase_state.decision_status is PhaseDecisionStatus.IDENTITY_CONFIRMED:
                confirmed_source_classes = set(range(int(self.source_prototype_bank.ready.numel())))
            else:
                confirmed_source_classes = {
                    int(class_id)
                    for group in phase_state.groups
                    if group.status is PhaseGroupStatus.CONFIRMED
                    for class_id in group.member_classes
                }

            counts = {class_id: 0 for class_id in sorted(confirmed_source_classes)}
            rows: list[dict] = []
            examples_for_tensor: dict[int, dict] = {}
            previous_mode = self.student.training
            self.student.eval()
            try:
                for batch_index, batch in enumerate(self.source_loader):
                    if batch_index >= max_batches:
                        break
                    (
                        source_logits,
                        _source_fused,
                        source_labels,
                        _trend,
                        _structure,
                        positions,
                        mask,
                        structure_geometry,
                    ) = self._source_forward(batch)
                    sample_ids = _source_sample_ids(batch, source_labels.shape[0])

                    pending: list[tuple[int, SyntheticSourceExample]] = []
                    for row_index in range(source_labels.shape[0]):
                        class_id = int(source_labels[row_index].item())
                        if class_id not in counts or counts[class_id] >= samples_per_class:
                            continue
                        example = build_synthetic_source_example(
                            source_sample_id=sample_ids[row_index],
                            class_id=class_id,
                            source_structure_function=structure_geometry.functional.function[row_index],
                            source_q_shape=structure_geometry.srvf[row_index],
                            source_q_support=structure_geometry.support_confidence[row_index],
                            source_positions=positions[row_index],
                            mask=mask[row_index],
                            phase_state=phase_state,
                            domain_shape_state=shape_state,
                            decomposition=self.student.backbone.decomposition,
                            lambda_delta=self.config.lambda_delta,
                        )
                        if example is not None:
                            pending.append((row_index, example))

                    if not pending:
                        if counts and all(value >= samples_per_class for value in counts.values()):
                            break
                        continue

                    synthetic_batch = _stack_synthetic(
                        [item[1] for item in pending],
                        [structure_geometry.structure_valid[item[0]].detach() for item in pending],
                        device=self.device,
                    )
                    assert synthetic_batch is not None
                    synthetic_raw = self.student.temporal_module.raw_encoder(
                        trend=synthetic_batch["trend"],
                        structure=synthetic_batch["structure"],
                        positions=synthetic_batch["positions"],
                        mask=synthetic_batch["mask"],
                    )
                    synthetic_logits = self.student.classifier(synthetic_raw.fused_repr)

                    for local_index, (source_index, example) in enumerate(pending):
                        class_id = int(example.class_id)
                        diagnostics = evaluate_synthetic_source_diagnostics(
                            example=example,
                            source_q_shape=structure_geometry.srvf[source_index],
                            source_logits=source_logits[source_index],
                            synthetic_logits=synthetic_logits[local_index],
                            source_positions=positions[source_index],
                            phase_state=phase_state,
                            domain_shape_state=shape_state,
                            source_prototype_bank=self.source_prototype_bank,
                        )
                        source_prediction = int(source_logits[source_index].argmax().item())
                        synthetic_prediction = int(synthetic_logits[local_index].argmax().item())
                        source_q = structure_geometry.srvf[source_index].detach().to(example.q_shape)
                        q_shift_error = weighted_norm(
                            (example.q_shape - source_q) - float(self.config.lambda_delta) * delta
                        )
                        valid_position_mask = example.mask
                        position_delta = (
                            example.target_style_positions[valid_position_mask]
                            - positions[source_index].to(example.target_style_positions)[valid_position_mask]
                        )
                        row = {
                            "source_sample_id": int(example.source_sample_id),
                            "class_id": class_id,
                            "group_id": int(example.group_id),
                            "source_prediction": source_prediction,
                            "synthetic_prediction": synthetic_prediction,
                            "source_correct": source_prediction == class_id,
                            "synthetic_correct": synthetic_prediction == class_id,
                            "prediction_changed": source_prediction != synthetic_prediction,
                            "finite": bool(diagnostics.finite),
                            "valid_support": bool(diagnostics.valid_support),
                            "label_preserved": bool(diagnostics.label_preserved),
                            "target_shape_distance_before": diagnostics.target_shape_distance_before,
                            "target_shape_distance_after": diagnostics.target_shape_distance_after,
                            "target_shape_improved": diagnostics.target_shape_improved,
                            "phase_leakage": diagnostics.phase_leakage,
                            "shape_class_separation_margin": diagnostics.shape_class_separation_margin,
                            "classifier_margin": diagnostics.classifier_margin,
                            "q_shift_error": float(q_shift_error.item()),
                            "position_shift_mean_abs": (
                                float(position_delta.abs().mean().item()) if position_delta.numel() else 0.0
                            ),
                            "position_shift_max_abs": (
                                float(position_delta.abs().max().item()) if position_delta.numel() else 0.0
                            ),
                        }
                        rows.append(row)
                        counts[class_id] += 1
                        if class_id not in examples_for_tensor:
                            examples_for_tensor[class_id] = {
                                "source_sample_id": int(example.source_sample_id),
                                "source_q": source_q.cpu(),
                                "synthetic_q": example.q_shape.detach().cpu(),
                                "source_positions": positions[source_index].detach().cpu(),
                                "target_style_positions": example.target_style_positions.detach().cpu(),
                                "mask": example.mask.detach().cpu(),
                                "trend_tokens": example.trend_tokens.detach().cpu(),
                                "structure_tokens": example.structure_tokens.detach().cpu(),
                                "source_logits": source_logits[source_index].detach().cpu(),
                                "synthetic_logits": synthetic_logits[local_index].detach().cpu(),
                            }
                    if counts and all(value >= samples_per_class for value in counts.values()):
                        break
            finally:
                self.student.train(previous_mode)

            tensor_payload["synthetic_examples"] = examples_for_tensor
            generated = len(rows)
            center_rows = [row for row in rows if row["target_shape_improved"] is not None]
            payload["synthesis"] = {
                "available": True,
                "requested_samples_per_class": int(samples_per_class),
                "max_batches": int(max_batches),
                "confirmed_source_classes": sorted(confirmed_source_classes),
                "sample_counts": {str(key): value for key, value in counts.items()},
                "generated": generated,
                "all_finite": bool(rows) and all(row["finite"] for row in rows),
                "all_valid_support": bool(rows) and all(row["valid_support"] for row in rows),
                "source_accuracy": (
                    sum(row["source_correct"] for row in rows) / generated if generated else None
                ),
                "synthetic_accuracy": (
                    sum(row["synthetic_correct"] for row in rows) / generated if generated else None
                ),
                "label_preserved_rate": (
                    sum(row["label_preserved"] for row in rows) / generated if generated else None
                ),
                "prediction_changed_rate": (
                    sum(row["prediction_changed"] for row in rows) / generated if generated else None
                ),
                "target_shape_improved_rate": (
                    sum(row["target_shape_improved"] is True for row in center_rows) / len(center_rows)
                    if center_rows
                    else None
                ),
                "max_phase_leakage": max(
                    (row["phase_leakage"] for row in rows if row["phase_leakage"] is not None),
                    default=None,
                ),
                "min_shape_class_separation_margin": min(
                    (
                        row["shape_class_separation_margin"]
                        for row in rows
                        if row["shape_class_separation_margin"] is not None
                    ),
                    default=None,
                ),
                "min_classifier_margin": min(
                    (row["classifier_margin"] for row in rows if row["classifier_margin"] is not None),
                    default=None,
                ),
                "max_q_shift_error": max((row["q_shift_error"] for row in rows), default=None),
                "rows": rows,
            }

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        torch.save(tensor_payload, tensor_path)
        print(
            "STAGE2_SHAPE_SYNTHESIS_AUDIT|"
            f"json={json_path}|tensors={tensor_path}"
            f"|shape_status={shape_state.status.value}"
            f"|synthetic_examples={payload['synthesis'].get('generated', 0)}"
        )
        return {"json": json_path, "tensors": tensor_path}

    def write_shape_diagnostics(self, epoch: int, *, suffix: str = "") -> None:
        if self.statistics is None:
            raise RuntimeError("Stage-2 statistics are unavailable")
        name_suffix = f"_{suffix}" if suffix else ""
        json_path = os.path.join(
            self.output_dir, f"shape_diagnostics_{epoch:03d}{name_suffix}.json"
        )
        stable = self.statistics.stable_labels
        hypothesis = None if self.hypothesis_cache is None else self.hypothesis_cache.result
        payload = {
            "epoch": epoch,
            "phase": _phase_summary(self.statistics.phase_state),
            "stable_label_coverage": (
                stable.num_stable_labels / stable.num_samples if stable.num_samples else 0.0
            ),
            "stable_label_class_counts": list(stable.stable_class_counts),
            "shape": _shape_summary(self.statistics.shape_state),
            "source_geometry_version": self.source_geometry_version,
            "target_hypothesis_scan_count": self.hypothesis_scan_count,
            "phase_evidence_stages": self.phase_evidence_stages,
            "phase_evidence_samples": None if hypothesis is None else hypothesis.num_samples,
            "shape_evidence_stages": self.shape_evidence_stages,
            "shape_evidence_samples": len(self.shape_evidence_sample_ids),
            "phase_solver_calls": None if hypothesis is None else hypothesis.num_solver_calls,
            "phase_proposal_pairs": None if hypothesis is None else hypothesis.num_proposal_pairs,
            "phase_all_class_pairs": None if hypothesis is None else hypothesis.num_all_class_pairs,
            "phase_rejections": None if hypothesis is None else {
                "pre_support": hypothesis.num_pre_support_rejected,
                "solver_failed": hypothesis.num_solver_failed,
                "gamma": hypothesis.num_gamma_rejected,
                "gamma_endpoint": hypothesis.num_gamma_endpoint_rejected,
                "gamma_increment": hypothesis.num_gamma_increment_rejected,
                "gamma_speed": hypothesis.num_gamma_speed_rejected,
                "gamma_roughness": hypothesis.num_gamma_roughness_rejected,
                "gamma_deviation": hypothesis.num_gamma_deviation_rejected,
                "gain": hypothesis.num_gain_rejected,
                "shape_support": hypothesis.num_shape_support_rejected,
                "shape_outer": hypothesis.num_outer_rejected,
            },
            "phase_diagnostic_quantiles": None if hypothesis is None else {
                name: {"p10": p10, "p50": p50, "p90": p90}
                for name, p10, p50, p90 in hypothesis.diagnostic_quantiles
            },
            "phase_proposal_class_counts": None if hypothesis is None else list(
                hypothesis.proposal_class_counts
            ),
            "phase_scanned_sample_ids": None if hypothesis is None else list(
                hypothesis.scanned_sample_ids
            ),
            "phase_hypotheses": None if hypothesis is None else [
                {
                    "sample_id": item.sample_id,
                    "class_id": item.class_id,
                    "t_gain_ratio": item.t_gain_ratio,
                    "q_shape_distance": item.q_shape_distance,
                    "q_distance_percentile": item.q_distance_percentile,
                    "common_support_t": item.common_support_t,
                    "common_support_shape": item.common_support_shape,
                    "roughness": item.roughness,
                    "phase_deviation": item.phase_deviation,
                    "preferred": item.preferred,
                    "ambiguous_class": item.ambiguous_class,
                    "evidence_weight": item.evidence_weight,
                }
                for item in hypothesis.hypotheses
            ],
            "candidate_pseudo_labels": None if hypothesis is None else [
                {
                    "sample_id": item.sample_id,
                    "class_id": item.class_id,
                    "q_shape_distance": item.q_shape_distance,
                    "q_distance_percentile": item.q_distance_percentile,
                    "phase_evidence_eligible": item.phase_evidence_eligible,
                    "ambiguous": item.ambiguous,
                    "secondary_class_id": item.secondary_class_id,
                    "secondary_q_shape_distance": item.secondary_q_shape_distance,
                }
                for item in hypothesis.candidate_pseudo_labels
            ],
            "stable_phase_compatible": stable.num_phase_compatible,
            "stable_phase_incompatible": stable.num_phase_incompatible,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


    @torch.no_grad()
    def write_oracle_shape_snapshot(self, epoch: int) -> None:
        """Write-only target true-label Shape diagnostic; never returns state."""
        if self.statistics is None:
            return None
        phase_state = self.statistics.phase_state
        class_to_group = build_confirmed_class_to_group_map(phase_state)
        if not class_to_group:
            torch.save(
                {"epoch": epoch, "class_centers": {}},
                os.path.join(self.output_dir, f"oracle_target_shape_{epoch:03d}.pt"),
            )
            return None
        teacher = self.ema_teacher.model()
        sums: dict[int, Tensor] = {}
        support_sums: dict[int, Tensor] = {}
        counts: dict[int, int] = {}
        fallback_index = 0
        for batch in self.target_statistics_loader:
            labels = batch.get("label")
            pixels = batch.get("pixels")
            if not isinstance(labels, Tensor) or not isinstance(pixels, Tensor):
                raise ValueError("oracle target snapshot requires target labels and pixels")
            batch_size = int(pixels.shape[0])
            batch_ids = batch.get("index")
            if not isinstance(batch_ids, Tensor) or batch_ids.shape != (batch_size,):
                batch_ids = torch.arange(
                    fallback_index, fallback_index + batch_size, dtype=torch.long
                )
            batch_ids = batch_ids.detach().to(device="cpu", dtype=torch.long)
            fallback_index += batch_size
            rows_by_group: dict[int, list[int]] = {}
            groups = {}
            for row, class_id_value in enumerate(labels.tolist()):
                group = class_to_group.get(int(class_id_value))
                if group is None:
                    continue
                rows_by_group.setdefault(group.group_id, []).append(row)
                groups[group.group_id] = group
            for group_id, rows in rows_by_group.items():
                subset = {}
                for key, value in batch.items():
                    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
                        subset[key] = value[rows]
                    else:
                        subset[key] = value
                sample_ids = torch.tensor(
                    [int(batch_ids[row].item()) for row in rows], dtype=torch.long
                )
                view = build_confirmed_phase_view(
                    model=teacher,
                    batch=subset,
                    sample_ids=sample_ids,
                    group=groups[group_id],
                )
                subset_labels = labels[rows]
                for local_index, class_id_value in enumerate(subset_labels.tolist()):
                    class_id = int(class_id_value)
                    q = view.aligned_q_shape[local_index].detach().cpu()
                    support = view.aligned_q_support[local_index].detach().cpu()
                    weighted = q * support.unsqueeze(-1)
                    if class_id not in sums:
                        sums[class_id] = weighted
                        support_sums[class_id] = support
                        counts[class_id] = 1
                    else:
                        sums[class_id] += weighted
                        support_sums[class_id] += support
                        counts[class_id] += 1
        centers = {
            class_id: sums[class_id] / (support_sums[class_id].unsqueeze(-1) + 1e-8)
            for class_id in sums
        }
        torch.save(
            {
                "epoch": epoch,
                "class_centers": centers,
                "class_support": support_sums,
                "class_counts": counts,
            },
            os.path.join(self.output_dir, f"oracle_target_shape_{epoch:03d}.pt"),
        )
        return None

    def save_ema_checkpoint(
        self,
        filename: str,
        *,
        epoch: int,
        target_val: dict | None,
    ) -> str:
        if self.statistics is None:
            raise RuntimeError("Stage-2 statistics are unavailable")
        path = os.path.join(self.output_dir, filename)
        teacher = self.ema_teacher.model()
        state = {
            "stage": "stage2",
            "epoch": epoch,
            "state_dict": {
                key: value.detach().cpu() for key, value in teacher.state_dict().items()
            },
            "target_val": None if target_val is None else {
                "accuracy": target_val.get("accuracy"),
                "macro_f1": target_val.get("macro_f1"),
            },
            "phase_state_summary": _phase_summary(self.statistics.phase_state),
            "domain_shape_state_summary": _shape_summary(self.statistics.shape_state),
            "phase_state": _phase_state_payload(self.statistics.phase_state),
            "domain_shape_state": _shape_state_payload(self.statistics.shape_state),
            "source_geometry_version": self.source_geometry_version,
            "source_prototype_bank": _bank_to_cpu(self.source_prototype_bank),
            "phase_evidence_sample_ids": (
                ()
                if self.hypothesis_cache is None
                else tuple(self.hypothesis_cache.result.scanned_sample_ids)
            ),
            "shape_evidence_sample_ids": tuple(self.shape_evidence_sample_ids),
            "target_hypothesis_state": (
                None
                if self.hypothesis_cache is None
                else _target_hypothesis_payload(self.hypothesis_cache.result)
            ),
            "stable_label_state": _stable_label_payload(self.statistics.stable_labels),
            "runtime_config": self.runtime_config,
            "stage2_config": {
                key: value
                for key, value in self.runtime_config.items()
                if key.startswith("stage2_")
            },
            "successful_optimizer_steps": self.successful_optimizer_steps,
        }
        torch.save(state, path)
        print(f"STAGE2_CHECKPOINT|path={path}|epoch={epoch}")
        return path


def run_stage2_statistics_diagnostic(trainer) -> Stage2StatisticsSnapshot:
    """Run Stage-2 statistics only; never perform an optimizer step.

    This mode exists for scientific diagnostics of Domain Phase evidence
    acquisition and downstream Stable Label / Domain Shape gates.  It writes
    the same initial statistics payload used by training but exits before any
    Stage-2 parameter update, so an ``M=0`` diagnostic cannot silently become
    source-only continuation training.
    """
    snapshot = trainer.initialize_statistics()
    print("STAGE2_INIT_COMPLETE|statistics_ready=true")
    trainer.write_shape_diagnostics(0, suffix="initial")
    if hasattr(trainer, "write_calibration_statistics"):
        trainer.write_calibration_statistics()
    if hasattr(trainer, "write_final_shape_synthesis_audit"):
        trainer.write_final_shape_synthesis_audit()
    stable = snapshot.stable_labels
    print(
        "STAGE2_DIAGNOSTIC_COMPLETE|"
        f"phase_m={snapshot.phase_state.m}"
        f"|confirmed_phase={str(_confirmed_phase_exists(snapshot.phase_state)).lower()}"
        f"|stable_labels={stable.num_stable_labels}"
        f"|shape_status={snapshot.shape_state.status.value}"
        f"|optimizer_steps={trainer.successful_optimizer_steps}"
    )
    return snapshot


def run_stage2_training(
    trainer,
    *,
    evaluate_target_val: Callable[[nn.Module, int], dict],
    evaluate_target_test: Callable[[nn.Module, int], dict],
) -> Stage2RunResult:
    """Run the exact 60-epoch/20-epoch-block protocol (or smoke overrides).

    Target-test metrics are deliberately write-only with respect to training
    decisions: only target validation Macro-F1 selects the analysis checkpoint.
    """
    snapshot = trainer.initialize_statistics()
    print("STAGE2_INIT_COMPLETE|statistics_ready=true")
    if isinstance(snapshot, Stage2StatisticsSnapshot) and not _adaptation_available(snapshot):
        decision = snapshot.phase_state.decision_status
        if decision is PhaseDecisionStatus.UNCONFIRMED:
            reason = "phase_unconfirmed"
        else:
            reason = "identity_phase_without_confirmed_shape"
        print(
            "STAGE2_ABSTAIN|"
            f"reason={reason}|phase_decision={decision.value}"
            f"|shape_status={snapshot.shape_state.status.value}"
            "|optimizer_steps=0|retain_stage1_best=true"
        )
        trainer.write_shape_diagnostics(0, suffix="no_adaptation")
        trainer.save_ema_checkpoint(
            "stage2_last_ema.pt",
            epoch=0,
            target_val=None,
        )
        final_test = evaluate_target_test(trainer.ema_teacher.model(), 0)
        return Stage2RunResult(
            best_target_val_f1=float("nan"),
            best_target_val_epoch=None,
            final_diagnostic_target_test=final_test,
            adaptation_performed=False,
            abstain_reason=reason,
        )
    total_epochs = trainer.config.total_epochs
    block_epochs = trainer.config.adaptation_block_epochs
    best_f1 = float("-inf")
    best_epoch: int | None = None
    final_test: dict | None = None
    formal_diagnostic_epochs = {20, 40, 60}

    for epoch in range(1, total_epochs + 1):
        trainer.train_epoch(epoch)
        teacher = trainer.ema_teacher.model()
        val_metrics = evaluate_target_val(teacher, epoch)
        val_f1 = float(val_metrics["macro_f1"])
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            trainer.save_ema_checkpoint(
                "stage2_best_target_val_ema.pt",
                epoch=epoch,
                target_val=val_metrics,
            )

        is_block_boundary = epoch % block_epochs == 0 or epoch == total_epochs
        if not is_block_boundary:
            continue

        if epoch in formal_diagnostic_epochs:
            trainer.save_ema_checkpoint(
                f"stage2_ema_{epoch:03d}.pt",
                epoch=epoch,
                target_val=val_metrics,
            )
            final_test = evaluate_target_test(teacher, epoch)
        trainer.write_shape_diagnostics(epoch)
        trainer.refresh_source_features_and_statistics()
        if epoch == total_epochs:
            trainer.write_shape_diagnostics(epoch, suffix="final")

    trainer.save_ema_checkpoint(
        "stage2_last_ema.pt",
        epoch=total_epochs,
        target_val=None,
    )
    return Stage2RunResult(
        best_target_val_f1=best_f1,
        best_target_val_epoch=best_epoch,
        final_diagnostic_target_test=final_test,
        adaptation_performed=True,
        abstain_reason=None,
    )
