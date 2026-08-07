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
from .prototype_bank import SourcePrototypeBank
from .shape_transport import (
    SyntheticSourceExample,
    build_phase_only_synthetic_source_example,
    build_synthetic_source_example,
)
from .source_prototype_scanner import refresh_source_fused_statistics
from .stable_target_labels import (
    StableLabelConfig,
    StableTargetLabelScanResult,
    scan_stable_target_labels,
)
from .stage2_objective import Stage2Objective, Stage2ObjectiveConfig
from .stage2_parameter_policy import Stage2ParameterPolicy
from .target_hypothesis_scan import (
    PhaseHypothesisScanConfig,
    TargetHypothesisScanResult,
    scan_target_class_phase_hypotheses,
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
    return any(group.status is PhaseGroupStatus.CONFIRMED for group in state.groups)


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
        self.statistics: Stage2StatisticsSnapshot | None = None
        self.hypothesis_scan_count = 0
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

    def _scan_target_hypotheses(self) -> TargetHypothesisScanResult:
        cache = self.hypothesis_cache
        if cache is not None and cache.source_geometry_version == self.source_geometry_version:
            return cache.result
        result = scan_target_class_phase_hypotheses(
            self.ema_teacher.model(),
            self.target_statistics_loader,
            self.source_prototype_bank,
            self.source_registration_bank,
            self.config.phase_scan,
            device=self.device,
            shape_extractor=self.student.temporal_module.structure_geometry,
            reg_extractor=self.reg_extractor,
        )
        self.hypothesis_scan_count += 1
        self.hypothesis_cache = TargetHypothesisCache(
            source_geometry_version=self.source_geometry_version,
            result=result,
        )
        return result

    @torch.no_grad()
    def settle_statistics(
        self,
        *,
        previous: Stage2StatisticsSnapshot | None,
    ) -> Stage2StatisticsSnapshot:
        """Advance frozen phase/label/Shape states without repeating registration."""
        hypothesis_result = self._scan_target_hypotheses()
        phase_state = None if previous is None else previous.phase_state
        shape_state = None if previous is None else previous.shape_state
        stable_result: StableTargetLabelScanResult | None = None
        max_passes = (
            self.config.phase.phase_confirmation_patience
            + self.config.shape.shape_confirmation_patience
        )
        for _ in range(max_passes):
            phase_state = update_domain_phase_state(
                hypothesis_result,
                self.config.phase,
                previous_state=phase_state,
            )
            stable_result = scan_stable_target_labels(
                ema_teacher=self.ema_teacher,
                target_loader=self.target_statistics_loader,
                hypothesis_result=hypothesis_result,
                phase_state=phase_state,
                source_prototype_bank=self.source_prototype_bank,
                config=self.config.stable_labels,
            )
            shape_state = update_domain_shape_state(
                stable_result,
                self.source_prototype_bank,
                self.config.shape,
                previous_state=shape_state,
            )
            if phase_state.m == 0:
                break
            if not _confirmed_phase_exists(phase_state):
                continue
            if shape_state.status is DomainShapeStatus.CONFIRMED:
                break
            if shape_state.status is DomainShapeStatus.UNAVAILABLE:
                break
        assert phase_state is not None
        assert shape_state is not None
        assert stable_result is not None
        snapshot = Stage2StatisticsSnapshot(
            phase_state=phase_state,
            stable_labels=stable_result,
            shape_state=shape_state,
        )
        self.statistics = snapshot
        print(
            "STAGE2_STATISTICS|"
            f"scan_index={phase_state.scan_index}|phase_m={phase_state.m}"
            f"|confirmed_phase={str(_confirmed_phase_exists(phase_state)).lower()}"
            f"|stable_labels={stable_result.num_stable_labels}"
            f"|shape_status={shape_state.status.value}"
            f"|hypothesis_scans={self.hypothesis_scan_count}"
        )
        return snapshot

    def initialize_statistics(self) -> Stage2StatisticsSnapshot:
        return self.settle_statistics(previous=None)

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

    def write_shape_diagnostics(self, epoch: int, *, suffix: str = "") -> None:
        if self.statistics is None:
            raise RuntimeError("Stage-2 statistics are unavailable")
        name_suffix = f"_{suffix}" if suffix else ""
        json_path = os.path.join(
            self.output_dir, f"shape_diagnostics_{epoch:03d}{name_suffix}.json"
        )
        tensor_path = os.path.join(
            self.output_dir, f"shape_diagnostics_{epoch:03d}{name_suffix}.pt"
        )
        stable = self.statistics.stable_labels
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
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        torch.save(
            {
                "delta": (
                    None
                    if self.statistics.shape_state.delta is None
                    else self.statistics.shape_state.delta.detach().cpu()
                ),
                "interactions": tuple(
                    item.detach().cpu() for item in self.statistics.shape_state.interactions
                ),
            },
            tensor_path,
        )

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
        global_index = 0
        for batch in self.target_statistics_loader:
            labels = batch.get("label")
            pixels = batch.get("pixels")
            if not isinstance(labels, Tensor) or not isinstance(pixels, Tensor):
                raise ValueError("oracle target snapshot requires target labels and pixels")
            batch_size = int(pixels.shape[0])
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
                    [global_index + row for row in rows], dtype=torch.long
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
            global_index += batch_size
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
            "source_geometry_version": self.source_geometry_version,
            "source_prototype_bank": _bank_to_cpu(self.source_prototype_bank),
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
    trainer.initialize_statistics()
    print("STAGE2_INIT_COMPLETE|statistics_ready=true")
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
        trainer.write_oracle_shape_snapshot(epoch)
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
    )
