"""Phase-only TimeMatch-style Stage-2 adaptation.

Classification uses the complete frozen-PSE latent process through one LTAE.
Decomposition/SRVF geometry is no-gradient evidence for Domain Phase and Stable
Labels only. Domain Shape is intentionally absent from this control flow.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .confirmed_phase_view import map_source_positions_to_target
from .domain_phase_state import (
    DomainPhaseConfig,
    DomainPhaseState,
    PhaseDecisionStatus,
    PhaseGroupStatus,
    update_domain_phase_state,
)
from .ema_teacher import Stage2EMATeacher
from .phase_registration import SourceRegistrationPrototypeBank
from .prototype_bank import SourcePrototypeBank
from .stable_target_labels import (
    StableLabelConfig,
    StableTargetLabelScanResult,
    scan_stable_target_labels_from_confirmed_phase,
)
from .stage2_objective import Stage2Objective, Stage2ObjectiveConfig
from .stage2_parameter_policy import Stage2ParameterPolicy
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
    objective: Stage2ObjectiveConfig
    ema_decay: float
    total_epochs: int = 60
    steps_per_epoch: int | None = None
    amp_enabled: bool = False
    amp_dtype: str = "float16"
    phase_evidence_initial_samples: int = 64
    phase_evidence_max_samples: int = 512
    evidence_seed: int = 0
    target_time_keep_ratio: float = 0.8

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.ema_decay)) or not 0.0 <= float(self.ema_decay) < 1.0:
            raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")
        for name in ("total_epochs",):
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
        if self.phase_evidence_initial_samples < 1 or self.phase_evidence_max_samples < 1:
            raise ValueError("phase evidence budgets must be positive")
        if self.phase_evidence_initial_samples > self.phase_evidence_max_samples:
            raise ValueError("phase_evidence_initial_samples cannot exceed max")
        if not math.isfinite(float(self.target_time_keep_ratio)) or not 0.0 < float(self.target_time_keep_ratio) <= 1.0:
            raise ValueError("target_time_keep_ratio must lie in (0,1]")


@dataclass(frozen=True)
class Stage2StatisticsSnapshot:
    phase_state: DomainPhaseState
    stable_labels: StableTargetLabelScanResult
    phase_routes: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class TargetHypothesisCache:
    source_geometry_version: int
    result: TargetHypothesisScanResult


@dataclass(frozen=True)
class Stage2RunResult:
    best_target_val_f1: float
    best_target_val_epoch: int | None
    final_diagnostic_target_test: dict | None
    adaptation_performed: bool = True
    abstain_reason: str | None = None


class DeviceBatchLoader:
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


def _batch_tensor(batch: dict, name: str, device: torch.device):
    value = batch.get(name)
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise ValueError(f"batch[{name!r}] must be a tensor")
    return value.to(device=device)


def _confirmed_phase_exists(state: DomainPhaseState) -> bool:
    return state.decision_status in (
        PhaseDecisionStatus.IDENTITY_CONFIRMED,
        PhaseDecisionStatus.NONIDENTITY_CONFIRMED,
    )


def _confirmed_groups(state: DomainPhaseState) -> dict[int, object]:
    return {
        int(group.group_id): group
        for group in state.groups
        if group.status is PhaseGroupStatus.CONFIRMED
    }


def _derive_phase_routes(
    phase_state: DomainPhaseState,
    stable_result: StableTargetLabelScanResult,
    num_classes: int,
) -> tuple[int | None, ...]:
    if phase_state.decision_status is not PhaseDecisionStatus.NONIDENTITY_CONFIRMED:
        return (None,) * num_classes
    groups = _confirmed_groups(phase_state)
    if not groups:
        return (None,) * num_classes
    if len(groups) == 1:
        return (next(iter(groups)),) * num_classes

    routes: list[int | None] = [None] * num_classes
    for group_id, group in groups.items():
        for class_id in group.member_classes:
            if 0 <= int(class_id) < num_classes:
                existing = routes[int(class_id)]
                if existing is not None and existing != group_id:
                    raise ValueError(f"class {class_id} belongs to multiple Phase groups")
                routes[int(class_id)] = group_id

    evidence: list[dict[int, float]] = [dict() for _ in range(num_classes)]
    for item in stable_result.stable_labels:
        class_id = int(item.class_id)
        group_id = int(item.group_id)
        if 0 <= class_id < num_classes and group_id in groups:
            evidence[class_id][group_id] = evidence[class_id].get(group_id, 0.0) + max(
                0.0, float(item.confidence_summary)
            )
    for class_id, by_group in enumerate(evidence):
        if routes[class_id] is not None or not by_group:
            continue
        ranked = sorted(by_group.items(), key=lambda pair: (-pair[1], pair[0]))
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            routes[class_id] = ranked[0][0]
    return tuple(routes)


def _adaptation_available(snapshot: Stage2StatisticsSnapshot) -> bool:
    if snapshot.phase_state.decision_status is PhaseDecisionStatus.UNCONFIRMED:
        return False
    if snapshot.phase_state.decision_status is PhaseDecisionStatus.NONIDENTITY_CONFIRMED:
        return True
    return snapshot.stable_labels.num_stable_labels > 0


def _optional_metric(value: float | None) -> str:
    return "none" if value is None else f"{float(value):.6g}"


def _phase_groups_log_value(state: DomainPhaseState) -> str:
    if not state.groups:
        return "-"
    return ";".join(
        f"{g.group_id}:{g.status.value}:classes={','.join(str(c) for c in g.member_classes)}"
        f":disp={g.within_dispersion:.6g}:diam={g.diameter:.6g}"
        f":radius={g.core_radius:.6g}:drift={_optional_metric(g.center_drift)}"
        for g in state.groups
    )


def _phase_rejections_log_value(state: DomainPhaseState) -> str:
    rejected = [
        f"{center.class_id}:{center.reject_reason or 'group_model'}"
        for center in state.class_centers
        if center.class_id in state.rejected_classes
    ]
    return ",".join(rejected) if rejected else "-"


def _phase_state_payload(state: DomainPhaseState) -> dict:
    """Serialize the complete no-gradient Domain Phase state for diagnostics.

    Stage-2 checkpoints are also the durable input to post-hoc Phase audits.
    Group centers alone are sufficient to *apply* a confirmed Phase, but they
    are not sufficient to test the domain-level hypothesis itself: that audit
    additionally needs the reliable/rejected class centers and their geometry.
    Keep every tensor detached on CPU so checkpointing cannot create a hidden
    gradient path.
    """
    return {
        "scan_index": int(state.scan_index),
        "m": int(state.m),
        "decision_status": state.decision_status.value,
        "decision_stability_age": int(state.decision_stability_age),
        "valid_phase_classes": tuple(int(v) for v in state.valid_phase_classes),
        "rejected_classes": tuple(int(v) for v in state.rejected_classes),
        "identity_evidence_classes": tuple(int(v) for v in state.identity_evidence_classes),
        "identity_evidence_count": float(state.identity_evidence_count),
        "residual_evidence_classes": tuple(int(v) for v in state.residual_evidence_classes),
        "residual_evidence_count": int(state.residual_evidence_count),
        "class_centers": tuple(
            {
                "class_id": int(center.class_id),
                "center_gamma": center.center_gamma.detach().cpu(),
                "candidate_count": int(center.candidate_count),
                "effective_evidence_count": float(center.effective_evidence_count),
                "dispersion": float(center.dispersion),
                "diameter": float(center.diameter),
                "median_distance": float(center.median_distance),
                "center_drift": center.center_drift,
                "valid": bool(center.valid),
                "reject_reason": center.reject_reason,
            }
            for center in state.class_centers
        ),
        "groups": tuple(
            {
                "group_id": int(g.group_id),
                "member_classes": tuple(int(v) for v in g.member_classes),
                "status": g.status.value,
                "center_gamma": g.center_gamma.detach().cpu(),
                "confirmation_age": int(g.confirmation_age),
                "center_drift": g.center_drift,
                "within_dispersion": float(g.within_dispersion),
                "diameter": float(g.diameter),
                "core_radius": float(g.core_radius),
                "sample_evidence_count": float(g.sample_evidence_count),
                "class_count": int(g.class_count),
            }
            for g in state.groups
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
    def __init__(
        self,
        *,
        student: nn.Module,
        policy: Stage2ParameterPolicy,
        ema_teacher: Stage2EMATeacher,
        optimizer: Optimizer,
        scheduler=None,
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
        target_train_loader=None,
        target_stable_label_loader=None,
    ) -> None:
        self.student = student
        self.policy = policy
        self.ema_teacher = ema_teacher
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.source_loader = source_loader
        self.source_scan_loader = source_scan_loader
        self.target_statistics_loader = target_statistics_loader
        self.target_stable_label_loader = (
            target_statistics_loader if target_stable_label_loader is None else target_stable_label_loader
        )
        self.target_train_loader = (
            self.target_stable_label_loader if target_train_loader is None else target_train_loader
        )
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
            num_classes=int(source_prototype_bank.ready.numel()), config=config.objective
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(config.amp_enabled and device.type == "cuda" and config.amp_dtype == "float16"),
        )
        self.hypothesis_cache: TargetHypothesisCache | None = None
        self.phase_scanner: TargetPhaseHypothesisScanner | None = None
        self.statistics: Stage2StatisticsSnapshot | None = None
        self.hypothesis_scan_count = 0
        self.phase_evidence_stages = 0
        self.stable_label_refresh_count = 0
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
            raise ValueError("Stage-2 optimizer parameters must exactly match parameter policy")

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

    def _stable_from_fixed_phase(self, phase_state: DomainPhaseState) -> StableTargetLabelScanResult:
        result = scan_stable_target_labels_from_confirmed_phase(
            ema_teacher=self.ema_teacher,
            target_loader=self.target_stable_label_loader,
            phase_state=phase_state,
            source_prototype_bank=self.source_prototype_bank,
            config=self.config.stable_labels,
        )
        self.stable_label_refresh_count += 1
        return result

    def _log_stable_label_diagnostics(self, result: StableTargetLabelScanResult) -> None:
        refresh = self.stable_label_refresh_count
        print(
            "STAGE2_STABLE_LABEL_ORACLE|"
            f"refresh={refresh}|oracle_only=true|evaluated={result.oracle_num_evaluated}"
            f"|accuracy={_optional_metric(result.oracle_accuracy)}"
            f"|macro_f1={_optional_metric(result.oracle_macro_f1)}"
            f"|precision={','.join(f'{v:.4f}' for v in result.oracle_precision) or '-'}"
            f"|recall={','.join(f'{v:.4f}' for v in result.oracle_recall) or '-'}"
            f"|support={','.join(str(v) for v in result.oracle_support) or '-'}"
            f"|gate_rejections={','.join(f'{k}:{v}' for k,v in result.gate_rejection_counts) or '-'}"
        )
        if self.writer is not None:
            if result.oracle_accuracy is not None:
                self.writer.add_scalar("stage2/stable_label_oracle_accuracy", result.oracle_accuracy, refresh)
            if result.oracle_macro_f1 is not None:
                self.writer.add_scalar("stage2/stable_label_oracle_macro_f1", result.oracle_macro_f1, refresh)

    @torch.no_grad()
    def initialize_statistics(self) -> Stage2StatisticsSnapshot:
        scanner = self._get_phase_scanner()
        phase_state: DomainPhaseState | None = None
        final_result: TargetHypothesisScanResult | None = None
        for budget in self._phase_evidence_budgets(scanner.total_cached_samples):
            final_result = scanner.scan_to_budget(budget)
            self.phase_evidence_stages += 1
            phase_state = update_domain_phase_state(
                final_result, self.config.phase, previous_state=phase_state
            )
            print(
                "STAGE2_PHASE_EVIDENCE_STAGE|"
                f"budget={budget}|phase_scan_index={phase_state.scan_index}"
                f"|phase_m={phase_state.m}|phase_decision={phase_state.decision_status.value}"
                f"|decision_age={phase_state.decision_stability_age}"
                f"|confirmed_phase={str(_confirmed_phase_exists(phase_state)).lower()}"
                f"|valid_classes={','.join(str(c) for c in phase_state.valid_phase_classes) or '-'}"
                f"|rejected={_phase_rejections_log_value(phase_state)}"
                f"|groups={_phase_groups_log_value(phase_state)}"
                f"|hypotheses={len(final_result.hypotheses)}|solver_calls={final_result.num_solver_calls}"
            )
        assert phase_state is not None and final_result is not None
        self.hypothesis_cache = TargetHypothesisCache(self.source_geometry_version, final_result)
        stable_result = self._stable_from_fixed_phase(phase_state)
        routes = _derive_phase_routes(
            phase_state, stable_result, int(self.source_prototype_bank.ready.numel())
        )
        snapshot = Stage2StatisticsSnapshot(phase_state, stable_result, routes)
        self.statistics = snapshot
        coverage = stable_result.num_stable_labels / stable_result.num_samples if stable_result.num_samples else 0.0
        print(
            "STAGE2_STATISTICS|"
            f"scan_index={phase_state.scan_index}|phase_m={phase_state.m}"
            f"|phase_decision={phase_state.decision_status.value}"
            f"|stable_labels={stable_result.num_stable_labels}|stable_coverage={coverage:.4f}"
            "|phase_only=true"
            f"|hypothesis_scans={self.hypothesis_scan_count}"
            f"|phase_evidence_stages={self.phase_evidence_stages}"
            f"|phase_evidence_samples={final_result.num_samples}"
            f"|stable_refreshes={self.stable_label_refresh_count}"
        )
        self._log_stable_label_diagnostics(stable_result)
        return snapshot

    @torch.no_grad()
    def refresh_stable_labels(self) -> Stage2StatisticsSnapshot:
        if self.statistics is None:
            return self.initialize_statistics()
        stable_result = self._stable_from_fixed_phase(self.statistics.phase_state)
        routes = _derive_phase_routes(
            self.statistics.phase_state,
            stable_result,
            int(self.source_prototype_bank.ready.numel()),
        )
        self.statistics = Stage2StatisticsSnapshot(
            self.statistics.phase_state, stable_result, routes
        )
        coverage = stable_result.num_stable_labels / stable_result.num_samples if stable_result.num_samples else 0.0
        print(
            "STAGE2_STABLE_LABEL_REFRESH|"
            f"refresh={self.stable_label_refresh_count}|stable_labels={stable_result.num_stable_labels}"
            f"|stable_coverage={coverage:.4f}"
            f"|class_counts={','.join(str(v) for v in stable_result.stable_class_counts)}"
            f"|routes={','.join('none' if v is None else str(v) for v in routes)}"
        )
        self._log_stable_label_diagnostics(stable_result)
        return self.statistics

    def _source_phase_positions(
        self,
        positions: Tensor,
        mask: Tensor,
        labels: Tensor,
    ) -> Tensor:
        assert self.statistics is not None
        phase_state = self.statistics.phase_state
        if phase_state.decision_status is PhaseDecisionStatus.IDENTITY_CONFIRMED:
            return positions.detach()
        groups = _confirmed_groups(phase_state)
        routes = self.statistics.phase_routes
        mapped = positions.detach().clone()
        for row in range(labels.shape[0]):
            class_id = int(labels[row].item())
            route = routes[class_id] if class_id < len(routes) else None
            if route is None or int(route) not in groups:
                continue
            mapped[row] = map_source_positions_to_target(
                positions[row].detach(), mask[row].detach(), groups[int(route)].center_gamma
            )
        return mapped.detach()

    def _source_forward_phase(self, batch: dict) -> tuple[Tensor, Tensor]:
        pixels = _batch_tensor(batch, "pixels", self.device)
        valid_pixels = _batch_tensor(batch, "valid_pixels", self.device)
        positions = _batch_tensor(batch, "positions", self.device)
        extra = _batch_tensor(batch, "extra", self.device)
        time_mask = _batch_tensor(batch, "time_mask", self.device)
        labels = _batch_tensor(batch, "label", self.device)
        if pixels is None or valid_pixels is None or positions is None or labels is None:
            raise ValueError("source batch must contain pixels, valid_pixels, positions and label")
        labels = labels.long()
        with torch.no_grad():
            backbone = self.student.forward_backbone(
                pixels, valid_pixels, positions, extra, time_mask=time_mask,
                compute_decomposition=False,
            )
            latent = backbone.tokens.detach()
            native_positions = backbone.normalized_positions.detach()
            mask = backbone.time_mask.detach()
            target_style_positions = self._source_phase_positions(native_positions, mask, labels)
        amp_dtype = getattr(torch, self.config.amp_dtype)
        amp_on = self.config.amp_enabled and (
            self.device.type == "cuda" or amp_dtype == torch.bfloat16
        )
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=amp_on):
            raw = self.student.temporal_module.raw_encoder(
                latent=latent, positions=target_style_positions, mask=mask
            )
            logits = self.student.classifier(raw.fused_repr)
        return logits, labels

    def _stable_target_lookup(self) -> dict[int, int]:
        if self.statistics is None:
            return {}
        return {
            int(item.sample_id): int(item.class_id)
            for item in self.statistics.stable_labels.stable_labels
        }

    def _strong_native_target_time_mask(self, base_mask: Tensor) -> Tensor:
        if base_mask.ndim != 2:
            raise ValueError("base target time mask must have shape [B,L]")
        base_mask = base_mask.to(device=self.device, dtype=torch.bool)
        ratio = float(self.config.target_time_keep_ratio)
        if ratio >= 1.0:
            return base_mask
        strong = torch.zeros_like(base_mask)
        for row in range(base_mask.shape[0]):
            valid = torch.nonzero(base_mask[row], as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            keep = max(min(2, int(valid.numel())), int(math.ceil(valid.numel() * ratio)))
            keep = min(int(valid.numel()), keep)
            chosen = valid if keep == valid.numel() else valid[torch.randperm(valid.numel(), device=self.device)[:keep]]
            strong[row, chosen] = True
        return strong

    def _target_forward_native(self, batch: dict) -> tuple[Tensor | None, Tensor | None]:
        stable_lookup = self._stable_target_lookup()
        if not stable_lookup:
            return None, None
        sample_ids = batch.get("index")
        if not isinstance(sample_ids, Tensor) or sample_ids.ndim != 1:
            raise ValueError("target batch must contain one-dimensional tensor index")
        rows: list[int] = []
        labels: list[int] = []
        for row, sample_id in enumerate(sample_ids.detach().cpu().tolist()):
            label = stable_lookup.get(int(sample_id))
            if label is not None:
                rows.append(row)
                labels.append(label)
        if not rows:
            return None, None

        index = torch.tensor(rows, device=self.device, dtype=torch.long)
        pixels = _batch_tensor(batch, "pixels", self.device)
        valid_pixels = _batch_tensor(batch, "valid_pixels", self.device)
        positions = _batch_tensor(batch, "positions", self.device)
        extra = _batch_tensor(batch, "extra", self.device)
        time_mask = _batch_tensor(batch, "time_mask", self.device)
        if pixels is None or valid_pixels is None or positions is None:
            raise ValueError("target batch must contain pixels, valid_pixels and positions")
        batch_size = pixels.shape[0]

        def select(value):
            if value is None:
                return None
            return value.index_select(0, index) if value.ndim > 0 and value.shape[0] == batch_size else value

        pixels, valid_pixels, positions, extra, time_mask = map(
            select, (pixels, valid_pixels, positions, extra, time_mask)
        )
        base_mask = (
            torch.ones(pixels.shape[:2], device=self.device, dtype=torch.bool)
            if time_mask is None
            else time_mask.bool()
        )
        strong_mask = self._strong_native_target_time_mask(base_mask)
        with torch.no_grad():
            backbone = self.student.forward_backbone(
                pixels, valid_pixels, positions, extra, time_mask=strong_mask,
                compute_decomposition=False,
            )
            latent = backbone.tokens.detach()
            native_positions = backbone.normalized_positions.detach()
            mask = backbone.time_mask.detach()
        amp_dtype = getattr(torch, self.config.amp_dtype)
        amp_on = self.config.amp_enabled and (
            self.device.type == "cuda" or amp_dtype == torch.bfloat16
        )
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=amp_on):
            raw = self.student.temporal_module.raw_encoder(
                latent=latent, positions=native_positions, mask=mask
            )
            logits = self.student.classifier(raw.fused_repr)
        return logits, torch.tensor(labels, device=self.device, dtype=torch.long)

    def _set_student_training_modes(self) -> None:
        self.student.train()
        self.student.backbone.eval()
        self.student.temporal_module.trend_geometry.eval()
        self.student.temporal_module.structure_geometry.eval()

    def train_step(self, source_batch: dict, target_batch: dict | None = None) -> dict[str, float]:
        if self.statistics is None:
            raise RuntimeError("Stage-2 statistics must be initialized before training")
        if not _adaptation_available(self.statistics):
            raise RuntimeError("Stage-2 optimizer step requires confirmed Phase evidence")
        self._set_student_training_modes()
        self.optimizer.zero_grad(set_to_none=True)
        source_logits, source_labels = self._source_forward_phase(source_batch)
        target_logits = target_labels = None
        if target_batch is not None:
            target_logits, target_labels = self._target_forward_native(target_batch)
        objective = self.objective(
            source_to_target_logits=source_logits,
            source_labels=source_labels,
            native_target_logits=target_logits,
            stable_target_labels=target_labels,
        )
        previous_scale = float(self.scaler.get_scale())
        self.scaler.scale(objective.total).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        step_succeeded = (not self.scaler.is_enabled()) or float(self.scaler.get_scale()) >= previous_scale
        if step_succeeded:
            if self.scheduler is not None:
                self.scheduler.step()
            self.ema_teacher.update_after_optimizer_step(self.student)
            self.successful_optimizer_steps += 1
        unresolved = sum(route is None for route in self.statistics.phase_routes)
        return {
            "loss": float(objective.total.detach().item()),
            "source_to_target": float(objective.source_to_target.detach().item()),
            "native_target": float(objective.native_target.detach().item()),
            "source_count": float(objective.source_count),
            "target_count": float(objective.target_count),
            "unresolved_phase_routes": float(unresolved),
            "optimizer_step_succeeded": float(step_succeeded),
        }

    def train_epoch(self, epoch: int) -> dict[str, float]:
        steps_limit = self.config.steps_per_epoch or len(self.source_loader)
        target_iterator = iter(self.target_train_loader)
        totals: dict[str, float] = {}
        steps = 0
        for source_batch in self.source_loader:
            if steps >= steps_limit:
                break
            try:
                target_batch = next(target_iterator)
            except StopIteration:
                target_iterator = iter(self.target_train_loader)
                target_batch = next(target_iterator)
            metrics = self.train_step(source_batch, target_batch)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        if steps == 0:
            raise RuntimeError("Stage-2 source loader produced no batches")
        avg = {key: value / steps for key, value in totals.items()}
        lr = float(self.optimizer.param_groups[0]["lr"])
        print(
            "STAGE2_TRAIN|"
            f"epoch={epoch}/{self.config.total_epochs}|steps={steps}|lr={lr:.8g}"
            f"|loss={avg['loss']:.4f}|source_to_target={avg['source_to_target']:.4f}"
            f"|native_target={avg['native_target']:.4f}|source_count={avg['source_count']:.2f}"
            f"|target_count={avg['target_count']:.2f}"
            f"|unresolved_routes={avg['unresolved_phase_routes']:.2f}"
            f"|optimizer_step_success={avg['optimizer_step_succeeded']:.2f}"
        )
        return avg

    def save_ema_checkpoint(self, filename: str, *, epoch: int, target_val: dict | None) -> str:
        if self.statistics is None:
            raise RuntimeError("cannot checkpoint before Stage-2 statistics initialization")
        path = os.path.join(self.output_dir, filename)
        state = {
            "stage": "stage2_phase_only",
            "epoch": int(epoch),
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in self.ema_teacher.model().state_dict().items()
            },
            "target_val": None if target_val is None else {
                "accuracy": target_val.get("accuracy"),
                "macro_f1": target_val.get("macro_f1"),
            },
            "phase_state": _phase_state_payload(self.statistics.phase_state),
            "phase_routes": tuple(self.statistics.phase_routes),
            "source_geometry_version": self.source_geometry_version,
            "source_prototype_bank": _bank_to_cpu(self.source_prototype_bank),
            "runtime_config": self.runtime_config,
            "successful_optimizer_steps": self.successful_optimizer_steps,
            "domain_shape": "disabled",
        }
        torch.save(state, path)
        print(f"STAGE2_CHECKPOINT|path={path}|epoch={epoch}")
        return path


def run_stage2_statistics_diagnostic(trainer: Stage2Trainer) -> Stage2StatisticsSnapshot:
    snapshot = trainer.initialize_statistics()
    print("STAGE2_INIT_COMPLETE|statistics_ready=true|phase_only=true")
    calibration_path = trainer.save_ema_checkpoint(
        "stage2_calibration_state.pt", epoch=0, target_val=None
    )
    print(
        "STAGE2_DIAGNOSTIC_COMPLETE|"
        f"phase_m={snapshot.phase_state.m}"
        f"|confirmed_phase={str(_confirmed_phase_exists(snapshot.phase_state)).lower()}"
        f"|stable_labels={snapshot.stable_labels.num_stable_labels}"
        "|phase_only=true"
        f"|optimizer_steps={trainer.successful_optimizer_steps}"
        f"|calibration_state={calibration_path}"
    )
    return snapshot


def run_stage2_training(
    trainer: Stage2Trainer,
    *,
    evaluate_target_val: Callable[[nn.Module, int], dict],
    evaluate_target_test: Callable[[nn.Module, int], dict],
) -> Stage2RunResult:
    snapshot = trainer.initialize_statistics()
    print("STAGE2_INIT_COMPLETE|statistics_ready=true|phase_only=true")
    if not _adaptation_available(snapshot):
        reason = (
            "phase_unconfirmed"
            if snapshot.phase_state.decision_status is PhaseDecisionStatus.UNCONFIRMED
            else "confirmed_phase_without_stable_target_evidence"
        )
        print(
            "STAGE2_ABSTAIN|"
            f"reason={reason}|phase_decision={snapshot.phase_state.decision_status.value}"
            "|phase_only=true|optimizer_steps=0|retain_stage1_best=true"
        )
        trainer.save_ema_checkpoint("stage2_last_ema.pt", epoch=0, target_val=None)
        final_test = evaluate_target_test(trainer.ema_teacher.model(), 0)
        return Stage2RunResult(float("nan"), None, final_test, False, reason)

    total_epochs = trainer.config.total_epochs
    best_f1 = float("-inf")
    best_epoch: int | None = None
    final_test: dict | None = None
    formal_diagnostic_epochs = {20, 40, 60}

    for epoch in range(1, total_epochs + 1):
        if epoch > 1:
            trainer.refresh_stable_labels()
        trainer.train_epoch(epoch)
        teacher = trainer.ema_teacher.model()
        val_metrics = evaluate_target_val(teacher, epoch)
        val_f1 = float(val_metrics["macro_f1"])
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            trainer.save_ema_checkpoint(
                "stage2_best_target_val_ema.pt", epoch=epoch, target_val=val_metrics
            )
        if epoch in formal_diagnostic_epochs:
            trainer.save_ema_checkpoint(
                f"stage2_ema_{epoch:03d}.pt", epoch=epoch, target_val=val_metrics
            )
            final_test = evaluate_target_test(teacher, epoch)

    trainer.save_ema_checkpoint("stage2_last_ema.pt", epoch=total_epochs, target_val=None)
    return Stage2RunResult(best_f1, best_epoch, final_test, True, None)
