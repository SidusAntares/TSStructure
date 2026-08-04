"""Joint phase-aware source/target training for Structure DA."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import sklearn.metrics
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset import (
    BalancedBatchSampler,
    PixelSetData,
    create_train_loader,
    worker_init_fn,
)
from transforms import Normalize, RandomSamplePixels, ToTensor
from utils.train_utils import AverageMeter, cycle, progress_bar_disabled, to_cuda

from .diagnostics import (
    DecompositionDiagnostics,
    compute_decomposition_diagnostics,
    merge_decomposition_diagnostics,
    summarize_decomposition_diagnostics,
)
from .eden_alignment import EDENDomainAlignmentOutput
from .full_model import StructureAwareDomainAdaptationModel
from .phase_aware_objective import (
    PhaseAwareTaskLossOutput,
    PhaseAwareTaskLossWeights,
    PhaseAwareTaskObjective,
    TrendLedGeometryLossOutput,
)
from .quality_fusion import (
    TwoScaleQualityLossOutput,
    TwoScaleQualityObjective,
    concatenate_two_scale_quality_outputs,
)


def _positive_int(name: str, value: int | None, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_nonnegative(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not math.isfinite(converted) or converted < 0 or (positive and converted == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return converted


def resolve_domain_score_weight(epoch_index: int, warmup_epochs: int) -> float:
    """Resolve the normalized domain-score contribution for a zero-based epoch."""

    if isinstance(epoch_index, bool) or not isinstance(epoch_index, int) or epoch_index < 0:
        raise ValueError("epoch_index must be a nonnegative integer")
    if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int) or warmup_epochs < 0:
        raise ValueError("warmup_epochs must be a nonnegative integer")
    if warmup_epochs <= 1:
        return 1.0
    return min(1.0, max(0.0, epoch_index / float(warmup_epochs - 1)))


def resolve_grl_warmup_max_iters(
    epochs: int,
    steps_per_epoch: int,
    *,
    fraction: float | None = 0.2,
    override: int | None = None,
) -> int:
    """Resolve an absolute GRL warm-up while preserving an explicit override."""

    _positive_int("epochs", epochs)
    _positive_int("steps_per_epoch", steps_per_epoch)
    if fraction is not None and override is not None:
        raise ValueError(
            "grl_warmup_fraction and grl_warmup_max_iters cannot both be specified"
        )
    if override is not None:
        _positive_int("grl_warmup_max_iters", override)
        return override
    resolved_fraction = 0.2 if fraction is None else _finite_nonnegative(
        "grl_warmup_fraction", fraction
    )
    if resolved_fraction > 1:
        raise ValueError("grl_warmup_fraction must lie in [0, 1]")
    return max(1, round(epochs * steps_per_epoch * resolved_fraction))


@dataclass(frozen=True)
class JointStructureDALossOutput:
    reported_total_loss: Tensor
    task: PhaseAwareTaskLossOutput
    geometry: TrendLedGeometryLossOutput


@dataclass(frozen=True)
class JointStructureDADiagnostics:
    """Detached finite scalar diagnostics from one joint training step."""

    scalars: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        invalid = [
            name
            for name, value in self.scalars.items()
            if not isinstance(value, Tensor)
            or value.ndim != 0
            or not torch.isfinite(value).item()
        ]
        if invalid:
            raise FloatingPointError(
                "training diagnostics must be finite scalar tensors: "
                + ", ".join(invalid)
            )


@dataclass(frozen=True)
class JointStructureDATrainStepOutput:
    losses: JointStructureDALossOutput
    alignment: EDENDomainAlignmentOutput
    quality: TwoScaleQualityLossOutput
    source_batch_size: int
    target_batch_size: int
    mean_alpha_trend: Tensor
    mean_alpha_structure: Tensor
    diagnostics: JointStructureDADiagnostics
    source_decomposition_diagnostics: DecompositionDiagnostics
    target_decomposition_diagnostics: DecompositionDiagnostics
    phase_class_diagnostics: Mapping[str, "PerClassPhaseDiagnosticsAccumulator"]


@dataclass(frozen=True)
class JointStructureDATrainingConfig:
    epochs: int
    steps_per_epoch: int | None
    lr: float
    weight_decay: float
    geometry_weight: float = 1.0
    classification_weight: float = 1.0
    quality_weight: float = 1.0
    source_shape_weight: float = 1.0
    source_raw_weight: float = 1.0
    global_domain_weight: float = 1.0
    target_semantic_weight: float = 1.0
    quality_classification_weight: float = 1.0
    quality_domain_weight: float = 1.0
    quality_domain_score_warmup_epochs: int = 5
    candidate_init_warp_amplitude: float = 0.015
    phase_identity_tolerance: float = 1e-4
    phase_candidate_unique_tolerance: float = 1e-4
    time_reference: float = 0.0
    time_scale: float = 365.0
    time_coordinate_mode: str = "canonical_day_of_year"
    amp: bool = False
    amp_dtype: str = "float16"
    log_step: int = 10
    progress_bar: str = "auto"
    classes: Sequence[str] = ()
    balance_source: bool = True

    def __post_init__(self) -> None:
        _positive_int("epochs", self.epochs)
        _positive_int("steps_per_epoch", self.steps_per_epoch, optional=True)
        _positive_int("log_step", self.log_step)
        if (
            isinstance(self.quality_domain_score_warmup_epochs, bool)
            or not isinstance(self.quality_domain_score_warmup_epochs, int)
            or self.quality_domain_score_warmup_epochs < 0
        ):
            raise ValueError(
                "quality_domain_score_warmup_epochs must be a nonnegative integer"
            )
        if not isinstance(self.amp, bool):
            raise ValueError("amp must be boolean")
        if not isinstance(self.balance_source, bool):
            raise ValueError("balance_source must be boolean")
        if self.amp_dtype not in ("float16", "bfloat16"):
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
        object.__setattr__(self, "lr", _finite_nonnegative("lr", self.lr, positive=True))
        object.__setattr__(
            self, "weight_decay", _finite_nonnegative("weight_decay", self.weight_decay)
        )
        for name in (
            "geometry_weight",
            "classification_weight",
            "quality_weight",
            "source_shape_weight",
            "source_raw_weight",
            "global_domain_weight",
            "target_semantic_weight",
            "quality_classification_weight",
            "quality_domain_weight",
            "candidate_init_warp_amplitude",
            "phase_identity_tolerance",
        ):
            object.__setattr__(self, name, _finite_nonnegative(name, getattr(self, name)))
        object.__setattr__(
            self,
            "phase_candidate_unique_tolerance",
            _finite_nonnegative(
                "phase_candidate_unique_tolerance",
                self.phase_candidate_unique_tolerance,
                positive=True,
            ),
        )
        try:
            time_reference = float(self.time_reference)
            time_scale = float(self.time_scale)
        except (TypeError, ValueError) as error:
            raise ValueError("time_reference and time_scale must be finite") from error
        if not math.isfinite(time_reference):
            raise ValueError("time_reference must be finite")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and greater than zero")
        if self.time_coordinate_mode != "canonical_day_of_year":
            raise ValueError(
                "time_coordinate_mode must be 'canonical_day_of_year'"
            )
        object.__setattr__(self, "time_reference", time_reference)
        object.__setattr__(self, "time_scale", time_scale)
        progress_bar_disabled(self.progress_bar)


def create_joint_structure_da_train_loaders(config, splits):
    """Create V3 train loaders, balancing only the labelled source domain."""

    source_transform = transforms.Compose(
        [RandomSamplePixels(config.num_pixels), Normalize(), ToTensor()]
    )
    target_transform = transforms.Compose(
        [RandomSamplePixels(config.num_pixels), Normalize(), ToTensor()]
    )
    common = dict(
        data_root=config.data_root,
        classes=config.classes,
        with_extra=False,
        closed_set=config.closed_set,
        combine_spring_and_winter=config.combine_spring_and_winter,
        time_coordinate_mode=getattr(
            config, "time_coordinate_mode", "canonical_day_of_year"
        ),
    )
    source_dataset = PixelSetData(
        dataset_name=config.source,
        transform=source_transform,
        indices=splits[config.source]["train"],
        **common,
    )
    target_dataset = PixelSetData(
        dataset_name=config.target,
        transform=target_transform,
        indices=splits[config.target]["train"],
        **common,
    )
    if getattr(config, "balance_source", True):
        source_loader = DataLoader(
            dataset=source_dataset,
            batch_sampler=BalancedBatchSampler(
                source_dataset.get_labels(), config.batch_size, seed=config.seed
            ),
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )
    else:
        source_loader = create_train_loader(
            source_dataset, config.batch_size, config.num_workers
        )
    target_loader = create_train_loader(
        target_dataset, config.batch_size, config.num_workers
    )
    return source_loader, target_loader


class PerClassPhaseDiagnosticsAccumulator:
    """Bounded per-class Phase counters; target labels are supplied by the caller."""

    def __init__(
        self, num_classes: int, num_candidates: int, max_phase_samples: int = 4096
    ):
        self.num_classes = int(num_classes)
        self.num_candidates = int(num_candidates)
        self.max_phase_samples = int(max_phase_samples)
        self._rows = [
            dict(count=0, phase_magnitudes=torch.empty(0, dtype=torch.float32))
            for _ in range(self.num_classes)
        ]

    def update(self, *, labels, label_valid, phase_base_valid, candidate_trainable,
               candidate_acceptable, candidate_unique_count, candidate_collapse,
               trend_ambiguous, structure_enabled, structure_changed, structure_veto,
               phase_status, selected_candidate, phase_magnitude, accepted_shift,
               shape_valid) -> None:
        tensors = {
            name: value.detach()
            for name, value in locals().items()
            if isinstance(value, Tensor)
        }
        labels = tensors["labels"].long()
        label_valid = tensors["label_valid"].bool()
        label_valid = (
            label_valid & (labels >= 0) & (labels < self.num_classes)
        )
        metric_names = (
            "phase_base_valid",
            "candidate_trainable",
            "candidate_acceptable",
            "candidate_unique_count",
            "candidate_collapse",
            "trend_ambiguous",
            "structure_enabled",
            "structure_changed",
            "structure_veto",
            "valid_identity",
            "valid_nonidentity",
            "failure",
            "head_denominator",
            "structure_changed_denominator",
            "structure_veto_denominator",
            "accepted_shift_sum",
            "accepted_shift_count",
            "shape_valid",
        )
        status = tensors["phase_status"]
        selected = tensors["selected_candidate"]
        successful = status != 0
        enabled = tensors["structure_enabled"].bool()
        per_sample = {
            "phase_base_valid": tensors["phase_base_valid"].float(),
            "candidate_trainable": tensors["candidate_trainable"].float().sum(dim=-1),
            "candidate_acceptable": tensors["candidate_acceptable"].float().sum(dim=-1),
            "candidate_unique_count": tensors["candidate_unique_count"].float(),
            "candidate_collapse": tensors["candidate_collapse"].float(),
            "trend_ambiguous": tensors["trend_ambiguous"].float(),
            "structure_enabled": tensors["structure_enabled"].float(),
            "structure_changed": tensors["structure_changed"].float(),
            "structure_veto": tensors["structure_veto"].float(),
            "valid_identity": (status == 1).float(),
            "valid_nonidentity": (status == 2).float(),
            "failure": (status == 0).float(),
            "head_denominator": (selected >= 0).float(),
            "structure_changed_denominator": enabled.float(),
            "structure_veto_denominator": enabled.float(),
            "accepted_shift_sum": torch.where(
                successful,
                tensors["accepted_shift"].float(),
                torch.zeros_like(tensors["accepted_shift"], dtype=torch.float32),
            ),
            "accepted_shift_count": successful.float(),
            "shape_valid": tensors["shape_valid"].float(),
        }
        valid_labels = labels[label_valid]
        counts = torch.bincount(
            valid_labels, minlength=self.num_classes
        )
        per_sample_values = torch.stack(
            [per_sample[name] for name in metric_names], dim=-1
        )
        aggregates = torch.zeros(
            self.num_classes,
            len(metric_names),
            dtype=per_sample_values.dtype,
            device=per_sample_values.device,
        )
        aggregates.index_add_(
            0, valid_labels, per_sample_values[label_valid]
        )
        learned = label_valid & (selected >= 0) & (
            selected < self.num_candidates
        )
        joint_index = (
            labels[learned] * self.num_candidates + selected[learned]
        )
        head_counts = torch.bincount(
            joint_index,
            minlength=self.num_classes * self.num_candidates,
        ).reshape(self.num_classes, self.num_candidates)
        aggregated_cpu = torch.cat(
            [
                counts.to(dtype=aggregates.dtype).unsqueeze(-1),
                aggregates,
                head_counts.to(dtype=aggregates.dtype),
            ],
            dim=-1,
        ).cpu()
        counts_cpu = aggregated_cpu[:, 0]
        aggregates_cpu = aggregated_cpu[
            :, 1 : 1 + len(metric_names)
        ]
        head_counts_cpu = aggregated_cpu[:, 1 + len(metric_names) :]
        for class_id in range(self.num_classes):
            n = int(counts_cpu[class_id])
            if n == 0:
                continue
            row = self._rows[class_id]
            row["count"] += n
            for index, name in enumerate(metric_names):
                row[name] = row.get(name, 0.0) + float(
                    aggregates_cpu[class_id, index]
                )
            for head in range(self.num_candidates):
                name = f"head_{head}"
                row[name] = row.get(name, 0.0) + float(
                    head_counts_cpu[class_id, head]
                )

        magnitude_valid = label_valid & successful
        magnitude_labels = labels[magnitude_valid]
        magnitude_values = tensors["phase_magnitude"][magnitude_valid].flatten()
        if magnitude_values.numel():
            magnitude_batch = torch.stack(
                [magnitude_labels.float(), magnitude_values.float()], dim=-1
            ).cpu()
            magnitude_labels = magnitude_batch[:, 0].long()
            magnitude_values = magnitude_batch[:, 1]
        for class_id, row in enumerate(self._rows):
            remaining = self.max_phase_samples - len(row["phase_magnitudes"])
            if remaining <= 0 or magnitude_values.numel() == 0:
                continue
            values = magnitude_values[magnitude_labels == class_id][:remaining]
            row["phase_magnitudes"] = torch.cat(
                [row["phase_magnitudes"], values]
            )

    def merge(self, other: "PerClassPhaseDiagnosticsAccumulator") -> None:
        for target, source in zip(self._rows, other._rows):
            target["count"] += source["count"]
            for name, value in source.items():
                if name == "count":
                    continue
                if name == "phase_magnitudes":
                    remaining = self.max_phase_samples - len(target[name])
                    target[name] = torch.cat(
                        [target[name], value[:max(remaining, 0)]]
                    )
                else:
                    target[name] = target.get(name, 0.0) + value

    def add_sample_metric(
        self, labels: Tensor, label_valid: Tensor, name: str, values: Tensor
    ) -> None:
        """Add a detached per-sample diagnostic with its own valid denominator."""
        labels = labels.detach().long()
        values = values.detach().float()
        valid = (
            label_valid.detach().bool()
            & torch.isfinite(values)
            & (labels >= 0)
            & (labels < self.num_classes)
        )
        valid_labels = labels[valid]
        sums = torch.bincount(
            valid_labels,
            weights=values[valid],
            minlength=self.num_classes,
        ).cpu()
        counts = torch.bincount(
            valid_labels, minlength=self.num_classes
        ).cpu()
        for class_id, row in enumerate(self._rows):
            if counts[class_id]:
                row[f"extra_sum_{name}"] = (
                    row.get(f"extra_sum_{name}", 0.0)
                    + float(sums[class_id])
                )
                row[f"extra_count_{name}"] = (
                    row.get(f"extra_count_{name}", 0.0)
                    + int(counts[class_id])
                )

    def summaries(self) -> dict[int, dict[str, float]]:
        output = {}
        for class_id, row in enumerate(self._rows):
            count = row["count"]
            if count == 0:
                continue
            safe = lambda numerator, denominator=count: float(numerator) / denominator if denominator else math.nan
            magnitudes = row["phase_magnitudes"]
            result = {
                "sample_count": count,
                "phase_base_valid_rate": safe(row["phase_base_valid"]),
                "candidate_trainable_rate": safe(row["candidate_trainable"], count * self.num_candidates),
                "candidate_acceptable_rate": safe(row["candidate_acceptable"], count * self.num_candidates),
                "candidate_unique_count_mean": safe(row["candidate_unique_count"]),
                "candidate_collapse_rate": safe(row["candidate_collapse"]),
                "trend_ambiguity_rate": safe(row["trend_ambiguous"]),
                "structure_disambiguation_enabled_rate": safe(row["structure_enabled"]),
                "structure_changed_preferred_rate": safe(row["structure_changed"], row["structure_changed_denominator"]),
                "structure_changed_preferred_overall_rate": safe(row["structure_changed"]),
                "structure_veto_all_rate": safe(row["structure_veto"], row["structure_veto_denominator"]),
                "valid_identity_rate": safe(row["valid_identity"]),
                "valid_nonidentity_rate": safe(row["valid_nonidentity"]),
                "failure_rate": safe(row["failure"]),
                "phase_magnitude_mean": float(magnitudes.mean()) if magnitudes.numel() else math.nan,
                "phase_magnitude_p95": float(torch.quantile(magnitudes, .95)) if magnitudes.numel() else math.nan,
                "accepted_warp_shift_mean": safe(row["accepted_shift_sum"], row["accepted_shift_count"]),
                "structure_shape_valid_rate": safe(row["shape_valid"]),
            }
            for head in range(self.num_candidates):
                result[f"candidate_head_{head}_selected_rate"] = safe(row[f"head_{head}"], row["head_denominator"])
            for name, value in row.items():
                if name.startswith("extra_sum_"):
                    metric = name.removeprefix("extra_sum_")
                    result[metric] = safe(value, row[f"extra_count_{metric}"])
            output[class_id] = result
        return output


def _sample_to_device(sample, device):
    if torch.device(device).type != "cpu":
        return to_cuda(sample, device)
    return tuple(
        sample[name].to(device) if name in sample else None
        for name in ("pixels", "valid_pixels", "positions", "extra")
    )


def _check_loss_scalars(**losses: Tensor) -> None:
    invalid = [
        name
        for name, value in losses.items()
        if not isinstance(value, Tensor)
        or value.ndim != 0
        or not torch.isfinite(value.detach()).item()
    ]
    if invalid:
        raise FloatingPointError("invalid scalar loss before backward: " + ", ".join(invalid))


def _check_bounded_training_values(model, quality, alignment) -> None:
    for name, value in (
        ("alpha_trend", quality.alpha_trend),
        ("alpha_structure", quality.alpha_structure),
    ):
        detached = value.detach()
        if (
            not torch.isfinite(detached).all().item()
            or not ((detached >= 0) & (detached <= 1)).all().item()
        ):
            raise FloatingPointError(f"{name} must contain finite values in [0, 1]")
    coefficient = alignment.coefficient.detach()
    if (
        coefficient.ndim != 0
        or not torch.isfinite(coefficient).item()
        or not model.alignment.grl.low <= coefficient.item() <= model.alignment.grl.high
    ):
        raise FloatingPointError("GRL coefficient lies outside its configured bounds")


@contextmanager
def _frozen_parameters(parameters):
    parameters = tuple(parameters)
    states = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, state in zip(parameters, states):
            parameter.requires_grad_(state)


def _mean_norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(value, dim=-1).mean()


def _rate(value: Tensor) -> Tensor:
    return value.float().mean()


def _phase_diagnostics(prefix: str, output, original_positions: Tensor) -> dict[str, Tensor]:
    selection = output.temporal.core.selection
    mask = output.backbone.time_mask
    positions = original_positions
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand_as(mask)
    positions = positions.to(
        device=output.temporal.aligned_positions.device,
        dtype=output.temporal.aligned_positions.dtype,
    )
    shift = (output.temporal.aligned_positions - positions).abs()
    mean_shift = shift.masked_select(mask).mean()
    candidate_count = selection.candidate_trainable_mask.shape[1]
    if candidate_count > 1:
        off_diagonal = ~torch.eye(
            candidate_count,
            dtype=torch.bool,
            device=selection.candidate_pairwise_distance.device,
        )
        pairwise = selection.candidate_pairwise_distance[:, off_diagonal]
        pairwise_mean = pairwise.mean()
        pairwise_min = pairwise.amin()
    else:
        pairwise_mean = selection.candidate_pairwise_distance.sum() * 0.0
        pairwise_min = pairwise_mean
    selected = selection.selected_candidate_index
    values = {
        f"{prefix}_phase_valid_rate": _rate(selection.phase_valid),
        f"{prefix}_valid_nonidentity_rate": _rate(selection.phase_status == 2),
        f"{prefix}_valid_identity_rate": _rate(selection.phase_status == 1),
        f"{prefix}_failure_rate": _rate(selection.phase_status == 0),
        f"{prefix}_candidate_trainable_rate": _rate(selection.candidate_trainable_mask),
        f"{prefix}_candidate_acceptable_rate": _rate(selection.candidate_acceptable_mask),
        f"{prefix}_candidate_pairwise_distance_mean": pairwise_mean,
        f"{prefix}_candidate_pairwise_distance_min": pairwise_min,
        f"{prefix}_candidate_unique_count_mean": selection.candidate_unique_count.float().mean(),
        f"{prefix}_candidate_collapse_rate": _rate(selection.candidate_collapse_mask),
        f"{prefix}_T_ambiguity_rate": _rate(selection.trend_candidate_ambiguous),
        f"{prefix}_S_changed_preferred_rate": _rate(
            selection.structure_disambiguation_used
            & (selected >= 0)
            & (selected != selection.trend_preferred_candidate_index)
        ),
        f"{prefix}_S_veto_rate": _rate(selection.structure_candidate_vetoed),
        f"{prefix}_candidate_selected_is_identity_rate": _rate(
            selection.candidate_selected_is_identity
        ),
        f"{prefix}_shape_valid_rate": _rate(output.representation.shape_valid),
        f"{prefix}_aligned_position_mean_absolute_shift": mean_shift,
    }
    for candidate in range(candidate_count):
        values[f"{prefix}_candidate_head_{candidate}_selected_rate"] = _rate(
            selected == candidate
        )
    return values


def _phase_class_accumulator(output, original_positions, labels, label_valid, num_classes):
    selection = output.temporal.core.selection
    mask = output.backbone.time_mask
    positions = original_positions
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand_as(mask)
    positions = positions.to(output.temporal.aligned_positions)
    token_shift = (output.temporal.aligned_positions - positions).abs()
    accepted_shift = (token_shift * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    selected = selection.selected_candidate_index
    selected_magnitude = selection.candidate_phase_magnitude.gather(
        1, selected.clamp_min(0).unsqueeze(1)
    ).squeeze(1)
    selected_magnitude = torch.where(selected >= 0, selected_magnitude, torch.zeros_like(selected_magnitude))
    accumulator = PerClassPhaseDiagnosticsAccumulator(
        num_classes, selection.candidate_trainable_mask.shape[1]
    )
    accumulator.update(
        labels=labels,
        label_valid=label_valid,
        phase_base_valid=selection.phase_base_valid,
        candidate_trainable=selection.candidate_trainable_mask,
        candidate_acceptable=selection.candidate_acceptable_mask,
        candidate_unique_count=selection.candidate_unique_count,
        candidate_collapse=selection.candidate_collapse_mask,
        trend_ambiguous=selection.trend_candidate_ambiguous,
        structure_enabled=selection.structure_disambiguation_used,
        structure_changed=(selection.structure_disambiguation_used & (selected >= 0)
                           & (selected != selection.trend_preferred_candidate_index)),
        structure_veto=selection.structure_candidate_vetoed,
        phase_status=selection.phase_status,
        selected_candidate=selected,
        phase_magnitude=selected_magnitude,
        accepted_shift=accepted_shift,
        shape_valid=selection.structure_shape_valid,
    )
    return accumulator


def _fusion_diagnostics(prefix: str, output) -> dict[str, Tensor]:
    quality = output.representation.quality
    return {
        f"{prefix}_alpha_trend_mean": quality.alpha_trend.mean(),
        f"{prefix}_alpha_trend_std": quality.alpha_trend.std(unbiased=False),
        f"{prefix}_alpha_structure_mean": quality.alpha_structure.mean(),
        f"{prefix}_alpha_structure_std": quality.alpha_structure.std(unbiased=False),
        f"{prefix}_raw_trend_norm": _mean_norm(output.representation.trend_embedding),
        f"{prefix}_raw_structure_norm": _mean_norm(output.representation.structure_embedding),
        f"{prefix}_shape_norm": _mean_norm(output.representation.shape_feature),
        f"{prefix}_weighted_trend_norm": _mean_norm(quality.weighted_trend),
        f"{prefix}_weighted_structure_norm": _mean_norm(quality.weighted_structure),
        f"{prefix}_fused_norm": _mean_norm(output.representation.fused_feature),
    }


def _prototype_diagnostics(model, prototype, target_batch_size: int) -> dict[str, Tensor]:
    state = model.prototype_alignment
    denominator = max(target_batch_size, 1)
    values = {}
    for name in ("q", "z", "trend", "structure"):
        values[f"{name}_prototype_ready_rate"] = getattr(
            state, f"{name}_prototype_ready"
        ).float().mean()
        values[f"{name}_radius_ready_rate"] = getattr(
            state, f"{name}_radius_ready"
        ).float().mean()
    for name in (
        "teacher",
        "inner",
        "middle",
        "outer",
        "z_pull",
        "trend_relation",
        "structure_relation",
        "trend_pull",
        "structure_pull",
    ):
        values[f"target_{name}_rate"] = getattr(
            prototype, f"target_{name}_count"
        ).float() / denominator
    return values


def _build_diagnostics(
    model,
    source_output,
    target_output,
    source_positions,
    target_positions,
    source_labels,
    losses,
    alignment,
    quality,
    domain_score_weight: float,
) -> JointStructureDADiagnostics:
    task = losses.task
    prototype = task.prototype
    geometry = losses.geometry
    decomposition = model.backbone.decomposition
    values = {
        "loss_reported_total": losses.reported_total_loss,
        "loss_task_total": task.total_loss,
        "loss_geometry_total": geometry.total_loss,
        "loss_classification": task.classification_loss,
        "loss_quality_total": task.quality_loss,
        "loss_quality_classification": quality.classification_loss,
        "loss_quality_domain": quality.domain_loss,
        "loss_source_shape": task.source_shape_loss,
        "loss_q_compact": prototype.q_compact_loss,
        "loss_q_separate": prototype.q_separate_loss,
        "loss_z_proto": prototype.z_proto_loss,
        "loss_q_to_z_source": prototype.q_to_z_source_loss,
        "loss_source_raw": task.source_raw_loss,
        "loss_trend_proto": prototype.trend_proto_loss,
        "loss_structure_proto": prototype.structure_proto_loss,
        "loss_global_domain": task.global_domain_loss,
        "loss_target_semantic": task.target_semantic_loss,
        "loss_q_to_z_target": prototype.q_to_z_target_loss,
        "loss_z_pull": prototype.z_pull_loss,
        "loss_q_to_trend_target": prototype.q_to_trend_target_loss,
        "loss_q_to_structure_target": prototype.q_to_structure_target_loss,
        "loss_trend_pull": prototype.trend_pull_loss,
        "loss_structure_pull": prototype.structure_pull_loss,
        "loss_geometry_candidate": geometry.candidate_loss,
        "loss_geometry_center": geometry.center_loss,
        "source_train_accuracy": (
            source_output.representation.logits.argmax(-1) == source_labels
        ).float().mean(),
        "domain_accuracy": alignment.accuracy,
        "grl_coefficient": alignment.coefficient,
        "domain_score_weight": source_output.representation.logits.new_tensor(
            domain_score_weight
        ),
        "tau_fast": decomposition.tau_fast,
        "tau_slow": decomposition.tau_slow,
        "tau_gap": decomposition.tau_slow - decomposition.tau_fast,
    }
    values.update(_fusion_diagnostics("source", source_output))
    values.update(_fusion_diagnostics("target", target_output))
    values.update(_phase_diagnostics("source", source_output, source_positions))
    values.update(_phase_diagnostics("target", target_output, target_positions))
    values.update(
        _prototype_diagnostics(
            model, prototype, target_output.representation.logits.shape[0]
        )
    )
    return JointStructureDADiagnostics(
        {name: value.detach() for name, value in values.items()}
    )


def _decomposition_diagnostics(
    output, positions: Tensor, eps: float
) -> DecompositionDiagnostics:
    decomposition = output.backbone.decomposition
    return compute_decomposition_diagnostics(
        output.backbone.tokens,
        decomposition.trend,
        decomposition.dynamics,
        decomposition.residual,
        positions,
        output.backbone.time_mask,
        eps=eps,
    )


def joint_structure_da_train_step(
    model,
    source_sample,
    target_sample,
    task_optimizer,
    geometry_optimizer,
    quality_objective,
    task_objective,
    training_config,
    device,
    *,
    domain_score_weight: float = 1.0,
    task_scheduler=None,
    geometry_scheduler=None,
    task_scaler=None,
) -> JointStructureDATrainStepOutput:
    if not isinstance(model, StructureAwareDomainAdaptationModel):
        raise ValueError("model must be StructureAwareDomainAdaptationModel")
    if not isinstance(quality_objective, TwoScaleQualityObjective):
        raise ValueError("quality_objective must be TwoScaleQualityObjective")
    if not isinstance(task_objective, PhaseAwareTaskObjective):
        raise ValueError("task_objective must be PhaseAwareTaskObjective")
    source_labels = source_sample["label"].to(device=device, dtype=torch.long)
    source_tensors = _sample_to_device(source_sample, device)
    target_tensors = _sample_to_device(target_sample, device)
    task_optimizer.zero_grad(set_to_none=True)
    geometry_optimizer.zero_grad(set_to_none=True)
    amp_dtype = getattr(torch, training_config.amp_dtype)
    amp_enabled = training_config.amp and (
        torch.device(device).type == "cuda" or amp_dtype == torch.bfloat16
    )

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        source_backbone = model.forward_backbone(*source_tensors)
        target_backbone = model.forward_backbone(*target_tensors)

    with torch.autocast(device_type=device.type, enabled=False):
        geometry_output = model.forward_geometry_from_backbones(
            source_backbone,
            source_tensors[2],
            target_backbone,
            target_tensors[2],
        )
        geometry_weighted = (
            training_config.geometry_weight * geometry_output.total_loss.float()
        )
    _check_loss_scalars(geometry_weighted=geometry_weighted)
    geometry_weighted.backward()
    geometry_optimizer.step()
    if geometry_scheduler is not None:
        geometry_scheduler.step()
    geometry_optimizer.zero_grad(set_to_none=True)

    geometry_parameters = tuple(model.geometry_parameters())
    with _frozen_parameters(geometry_parameters):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            source_output = model.forward_from_backbone(
                source_backbone,
                source_tensors[2],
                domain_score_weight=domain_score_weight,
            )
            target_output = model.forward_from_backbone(
                target_backbone,
                target_tensors[2],
                domain_score_weight=domain_score_weight,
            )
            target_shape_feature_da = model.forward_target_shape_feature_da(
                target_output
            )
            target_shape_teacher_feature = (
                model.forward_target_shape_teacher_feature(target_output)
            )
            merged_quality = concatenate_two_scale_quality_outputs(
                source_output.representation.quality,
                target_output.representation.quality,
            )
            source_batch_size = source_labels.shape[0]
            target_batch_size = target_tensors[0].shape[0]
            domain_labels = torch.cat(
                [
                    torch.ones(source_batch_size, dtype=torch.long, device=device),
                    torch.zeros(target_batch_size, dtype=torch.long, device=device),
                ]
            )
            source_mask = domain_labels == 1
            class_labels = torch.cat(
                [
                    source_labels,
                    torch.zeros(target_batch_size, dtype=torch.long, device=device),
                ]
            )
            quality_loss = quality_objective(
                merged_quality, class_labels, domain_labels, source_mask
            )
            alignment = model.align(source_output, target_output)
            prototype = model.prototype_losses(
                source_output,
                source_labels,
                target_output,
                target_shape_feature_da,
                target_shape_teacher_feature,
            )
            task_loss = task_objective(
                source_output.representation.logits,
                source_labels,
                quality_loss,
                alignment,
                prototype,
            )
        _check_loss_scalars(
            task_total=task_loss.total_loss,
            classification=task_loss.classification_loss,
            quality=task_loss.quality_loss,
            source_shape=task_loss.source_shape_loss,
            source_raw=task_loss.source_raw_loss,
            global_domain=task_loss.global_domain_loss,
            target_semantic=task_loss.target_semantic_loss,
        )
        if task_scaler is not None and task_scaler.is_enabled():
            scale_before = task_scaler.get_scale()
            task_scaler.scale(task_loss.total_loss).backward()
            task_scaler.step(task_optimizer)
            task_scaler.update()
            task_step_succeeded = task_scaler.get_scale() >= scale_before
        else:
            task_loss.total_loss.backward()
            task_optimizer.step()
            task_step_succeeded = True
    if any(parameter.grad is not None for parameter in geometry_parameters):
        raise RuntimeError("task backward must not populate warp-estimator gradients")
    if task_scheduler is not None and task_step_succeeded:
        task_scheduler.step()
    task_optimizer.zero_grad(set_to_none=True)

    source_state_update_executed = False
    if task_step_succeeded:
        model.update_source_state_from_output(
            source_output, source_tensors[2], source_labels
        )
        source_state_update_executed = True
    reported_total = (
        task_loss.total_loss.detach()
        + training_config.geometry_weight * geometry_output.total_loss.detach()
    )
    losses = JointStructureDALossOutput(
        reported_total,
        task_loss,
        geometry_output.loss,
    )
    _check_bounded_training_values(model, merged_quality, alignment)
    diagnostics = _build_diagnostics(
        model,
        source_output,
        target_output,
        source_tensors[2],
        target_tensors[2],
        source_labels,
        losses,
        alignment,
        quality_loss,
        domain_score_weight,
    )
    step_succeeded = torch.as_tensor(
        float(task_step_succeeded), device=device
    )
    step_scalars = dict(diagnostics.scalars)
    step_scalars.update(
        task_optimizer_step_succeeded=step_succeeded,
        task_optimizer_step_skipped=1.0 - step_succeeded,
        task_optimizer_skip_rate=1.0 - step_succeeded,
        task_step_success_rate=step_succeeded,
        task_step_skip_rate=1.0 - step_succeeded,
        source_state_update_executed=torch.as_tensor(
            float(source_state_update_executed), device=device
        ),
        source_state_update_rate=torch.as_tensor(
            float(source_state_update_executed), device=device
        ),
        trend_reference_support=(
            model.temporal_features.core.trend_template.running_support
            .mean()
            .detach()
        ),
        structure_diagnostic_reference_support=(
            model.temporal_features.core.structure_diagnostic_template
            .running_support.mean()
            .detach()
        ),
    )
    diagnostics = JointStructureDADiagnostics(step_scalars)
    eps = model.backbone.decomposition.eps
    source_class_diagnostics = _phase_class_accumulator(
        source_output,
        source_tensors[2],
        source_labels,
        torch.ones_like(source_labels, dtype=torch.bool),
        len(training_config.classes),
    )
    prototype_module = model.prototype_alignment
    source_valid = torch.ones_like(source_labels, dtype=torch.bool)
    for metric, ready in (
        ("shape_geometry_prototype_ready_rate", prototype_module.q_prototype_ready),
        ("shape_feature_prototype_ready_rate", prototype_module.z_prototype_ready),
        ("trend_raw_prototype_ready_rate", prototype_module.trend_prototype_ready),
        ("structure_raw_prototype_ready_rate", prototype_module.structure_prototype_ready),
    ):
        source_class_diagnostics.add_sample_metric(
            source_labels, source_valid, metric, ready[source_labels].float()
        )
    source_class_diagnostics.add_sample_metric(
        source_labels, source_valid, "alpha_trend_mean",
        source_output.representation.quality.alpha_trend.squeeze(-1),
    )
    source_class_diagnostics.add_sample_metric(
        source_labels, source_valid, "alpha_structure_mean",
        source_output.representation.quality.alpha_structure.squeeze(-1),
    )
    target_class_diagnostics = _phase_class_accumulator(
        target_output,
        target_tensors[2],
        prototype.target_pseudo_label,
        prototype.target_teacher_mask,
        len(training_config.classes),
    )
    for metric, values in (
        ("q_z_consistency_rate", prototype.target_teacher_mask.float()),
        ("geometry_teacher_inner_rate", prototype.target_inner_mask.float()),
        ("geometry_teacher_middle_rate", prototype.target_middle_mask.float()),
        ("geometry_teacher_outer_rate", prototype.target_outer_mask.float()),
    ):
        target_class_diagnostics.add_sample_metric(
            prototype.target_pseudo_label,
            prototype.target_teacher_mask,
            metric,
            values,
        )
    return JointStructureDATrainStepOutput(
        losses=losses,
        alignment=alignment,
        quality=quality_loss,
        source_batch_size=source_batch_size,
        target_batch_size=target_batch_size,
        mean_alpha_trend=merged_quality.alpha_trend.mean().detach(),
        mean_alpha_structure=merged_quality.alpha_structure.mean().detach(),
        diagnostics=diagnostics,
        source_decomposition_diagnostics=_decomposition_diagnostics(
            source_output, source_tensors[2], eps
        ),
        target_decomposition_diagnostics=_decomposition_diagnostics(
            target_output, target_tensors[2], eps
        ),
        phase_class_diagnostics={
            "source": source_class_diagnostics,
            "target": target_class_diagnostics,
        },
    )


def _resolve_steps(training_config, source_loader, target_loader) -> int:
    source_steps, target_steps = len(source_loader), len(target_loader)
    if source_steps == 0 or target_steps == 0:
        raise ValueError("source and target training loaders must be nonempty")
    return training_config.steps_per_epoch or source_steps


def resolve_steps_per_epoch(training_config, source_loader, target_loader) -> int:
    return _resolve_steps(training_config, source_loader, target_loader)


@torch.inference_mode()
def _collect_validation_structure_contributions(
    model,
    val_loader,
    device,
    classes: Sequence[str],
) -> tuple[dict[str, float], Tensor, Tensor]:
    if not classes:
        raise ValueError("classes must be nonempty")
    mode_names = (
        "occlusion_full",
        "occlusion_no_shape",
        "occlusion_trend_only",
        "occlusion_structure_only",
        "occlusion_shape_only",
    )
    mode_logits: dict[str, list[Tensor]] = {name: [] for name in mode_names}
    labels: list[Tensor] = []
    phase_sums: dict[str, float] = {}
    phase_samples = 0
    model.eval()
    for sample in val_loader:
        target = sample["label"].to(device=device, dtype=torch.long)
        pixels, valid_pixels, positions, extra = _sample_to_device(sample, device)
        output = model.forward_details(pixels, valid_pixels, positions, extra)
        batch_size = int(target.shape[0])
        for name, value in _phase_diagnostics(
            "validation", output, positions
        ).items():
            short_name = name.removeprefix("validation_")
            phase_sums[short_name] = (
                phase_sums.get(short_name, 0.0)
                + float(value.item()) * batch_size
            )
        phase_samples += batch_size
        quality = output.representation.quality
        trend = quality.weighted_trend
        structure = quality.weighted_structure
        shape = quality.shape_feature
        zero_trend = torch.zeros_like(trend)
        zero_structure = torch.zeros_like(structure)
        zero_shape = torch.zeros_like(shape)
        features = {
            "occlusion_full": torch.cat([trend, structure, shape], -1),
            "occlusion_no_shape": torch.cat([trend, structure, zero_shape], -1),
            "occlusion_trend_only": torch.cat([trend, zero_structure, zero_shape], -1),
            "occlusion_structure_only": torch.cat([zero_trend, structure, zero_shape], -1),
            "occlusion_shape_only": torch.cat([zero_trend, zero_structure, shape], -1),
        }
        for name, feature in features.items():
            logits = (
                output.representation.logits
                if name == "occlusion_full"
                else model.representation.classifier(feature)
            )
            mode_logits[name].append(logits.detach().cpu())
        labels.append(target.detach().cpu())
    if not labels:
        raise ValueError("validation loader must be nonempty")
    all_labels = torch.cat(labels)
    metrics: dict[str, float] = {}
    for name, chunks in mode_logits.items():
        logits = torch.cat(chunks)
        predictions = logits.argmax(-1)
        metrics[f"{name}_loss"] = float(F.cross_entropy(logits, all_labels).item())
        metrics[f"{name}_f1"] = float(
            sklearn.metrics.f1_score(
                all_labels.numpy(),
                predictions.numpy(),
                average="macro",
                zero_division=0,
            )
        )
    metrics["delta_shape"] = metrics["occlusion_full_f1"] - metrics["occlusion_no_shape_f1"]
    metrics["delta_trend"] = metrics["occlusion_full_f1"] - metrics["occlusion_structure_only_f1"]
    metrics["delta_structure"] = metrics["occlusion_full_f1"] - metrics["occlusion_trend_only_f1"]
    for name, total in phase_sums.items():
        metrics[f"phase_{name}"] = total / max(phase_samples, 1)
    predictions = torch.cat(mode_logits["occlusion_full"]).argmax(-1)
    return metrics, all_labels, predictions


@torch.inference_mode()
def validation_structure_contributions(
    model,
    val_loader,
    device,
    classes: Sequence[str],
) -> dict[str, float]:
    return _collect_validation_structure_contributions(
        model, val_loader, device, classes
    )[0]


def _validation_with_structure_contributions(
    best_f1,
    best_model_path,
    training_config,
    device,
    epoch,
    model,
    val_loader,
    writer,
) -> tuple[float, dict[str, float]]:
    metrics, labels, predictions = _collect_validation_structure_contributions(
        model, val_loader, device, training_config.classes
    )
    val_loss, val_f1 = metrics["occlusion_full_loss"], metrics["occlusion_full_f1"]
    val_accuracy = float(sklearn.metrics.accuracy_score(labels.numpy(), predictions.numpy()))
    val_kappa = float(
        sklearn.metrics.cohen_kappa_score(
            labels.numpy(),
            predictions.numpy(),
            labels=list(range(len(training_config.classes))),
        )
    )
    if writer is not None:
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_accuracy, epoch)
        writer.add_scalar("val/f1", val_f1, epoch)
        writer.add_scalar("val/kappa", val_kappa, epoch)
    print(
        f"Validation result: loss={val_loss:.4f}, "
        f"acc={val_accuracy:.2f}, f1={val_f1:.4f}"
    )
    if val_f1 > best_f1:
        best_f1 = val_f1
        if best_model_path is not None:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "training_config": asdict(training_config),
                },
                best_model_path,
            )
    return best_f1, metrics


def _format_epoch_diagnostics(
    prefix: str, epoch: int, domain: str, values: Mapping[str, Tensor]
) -> str:
    fields = [f"{prefix}|epoch={epoch}|domain={domain}"]
    fields.extend(f"{name}={value.item():.6g}" for name, value in values.items())
    return "|".join(fields)


def _write_diagnostics(writer, step: int, diagnostics: Mapping[str, Tensor]) -> None:
    if writer is None:
        return
    for name, value in diagnostics.items():
        if name.startswith("loss_quality"):
            group = "quality"
        elif name.startswith("loss_geometry"):
            group = "geometry"
        elif name.startswith("loss_"):
            group = "loss"
        elif name.startswith("target_") and not name.startswith("target_alpha"):
            group = "target"
        elif "prototype" in name or "radius" in name:
            group = "prototype"
        elif name.startswith("source_") and any(
            token in name
            for token in (
                "phase_",
                "candidate_",
                "T_ambiguity",
                "S_changed",
                "S_veto",
                "valid_identity",
                "valid_nonidentity",
                "failure_rate",
                "shape_valid",
                "aligned_",
            )
        ):
            group = "phase/source"
        elif name.startswith("target_") and any(
            token in name
            for token in (
                "phase_",
                "candidate_",
                "T_ambiguity",
                "S_changed",
                "S_veto",
                "valid_identity",
                "valid_nonidentity",
                "failure_rate",
                "shape_valid",
                "aligned_",
            )
        ):
            group = "phase/target"
        elif name.startswith("source_"):
            group = "fusion/source"
        elif name.startswith("target_"):
            group = "fusion/target"
        elif name in ("domain_accuracy", "grl_coefficient", "domain_score_weight"):
            group = "domain"
        else:
            group = "loss"
        writer.add_scalar(f"train/{group}/{name}", value.item(), step)


def _log_validation_contributions(writer, epoch: int, metrics) -> None:
    fields = [f"VAL_CONTRIBUTION|epoch={epoch}"]
    for name, value in metrics.items():
        fields.append(f"{name}={value:.6g}")
        if writer is not None:
            writer.add_scalar(f"val/feature_occlusion/{name}", value, epoch - 1)
    print("|".join(fields))


def train_joint_structure_da(
    model,
    source_loader,
    target_loader,
    val_loader,
    training_config,
    writer,
    device,
    best_model_path,
):
    steps_per_epoch = _resolve_steps(training_config, source_loader, target_loader)
    model.to(device)
    task_optimizer = torch.optim.Adam(
        model.task_parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
    )
    geometry_optimizer = torch.optim.Adam(
        model.geometry_parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
    )
    task_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        task_optimizer, T_max=training_config.epochs * steps_per_epoch, eta_min=0
    )
    geometry_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        geometry_optimizer, T_max=training_config.epochs * steps_per_epoch, eta_min=0
    )
    task_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            training_config.amp
            and device.type == "cuda"
            and training_config.amp_dtype == "float16"
        ),
    )
    quality_objective = TwoScaleQualityObjective(
        classification_weight=training_config.quality_classification_weight,
        domain_weight=training_config.quality_domain_weight,
    )
    task_objective = PhaseAwareTaskObjective(
        PhaseAwareTaskLossWeights(
            classification=training_config.classification_weight,
            quality=training_config.quality_weight,
            source_shape=training_config.source_shape_weight,
            source_raw=training_config.source_raw_weight,
            global_domain=training_config.global_domain_weight,
            target_semantic=training_config.target_semantic_weight,
        )
    )
    best_f1 = float("-inf")
    for epoch in range(training_config.epochs):
        domain_score_weight = resolve_domain_score_weight(
            epoch, training_config.quality_domain_score_warmup_epochs
        )
        progress_disabled = progress_bar_disabled(training_config.progress_bar)
        model.train()
        meters: dict[str, AverageMeter] = {}
        decomposition_epoch = {"source": None, "target": None}
        phase_class_epoch = {"source": None, "target": None}
        source_class_batch_presence = [0] * len(training_config.classes)
        source_iterator, target_iterator = cycle(source_loader), cycle(target_loader)
        bar = None if progress_disabled else tqdm(range(steps_per_epoch))
        step_iterator = range(steps_per_epoch) if bar is None else bar
        for local_step in step_iterator:
            global_step = epoch * steps_per_epoch + local_step
            result = joint_structure_da_train_step(
                model,
                next(source_iterator),
                next(target_iterator),
                task_optimizer,
                geometry_optimizer,
                quality_objective,
                task_objective,
                training_config,
                device,
                domain_score_weight=domain_score_weight,
                task_scheduler=task_scheduler,
                geometry_scheduler=geometry_scheduler,
                task_scaler=task_scaler,
            )
            for name, value in result.diagnostics.scalars.items():
                meters.setdefault(name, AverageMeter()).update(value.item())
            for domain, accumulator in result.phase_class_diagnostics.items():
                if phase_class_epoch[domain] is None:
                    phase_class_epoch[domain] = accumulator
                else:
                    phase_class_epoch[domain].merge(accumulator)
            for class_id in result.phase_class_diagnostics["source"].summaries():
                source_class_batch_presence[class_id] += 1
            for domain, diagnostics in (
                ("source", result.source_decomposition_diagnostics),
                ("target", result.target_decomposition_diagnostics),
            ):
                current = decomposition_epoch[domain]
                decomposition_epoch[domain] = (
                    diagnostics
                    if current is None
                    else merge_decomposition_diagnostics(current, diagnostics)
                )
            diagnostics = result.diagnostics.scalars
            lr = task_optimizer.param_groups[0]["lr"]
            if progress_disabled and global_step % training_config.log_step == 0:
                print(
                    f"TRAIN_STEP|epoch={epoch + 1}/{training_config.epochs}"
                    f"|step={local_step + 1}/{steps_per_epoch}"
                    f"|total={diagnostics['loss_reported_total'].item():.4f}"
                    f"|task={diagnostics['loss_task_total'].item():.4f}"
                    f"|geometry={diagnostics['loss_geometry_total'].item():.4f}"
                    f"|cls={diagnostics['loss_classification'].item():.4f}"
                    f"|quality={diagnostics['loss_quality_total'].item():.4f}"
                    f"|source_shape={diagnostics['loss_source_shape'].item():.4f}"
                    f"|source_raw={diagnostics['loss_source_raw'].item():.4f}"
                    f"|global_da={diagnostics['loss_global_domain'].item():.4f}"
                    f"|target_semantic={diagnostics['loss_target_semantic'].item():.4f}"
                    f"|teacher_rate={diagnostics['target_teacher_rate'].item():.3f}"
                    f"|phase_valid_s={diagnostics['source_phase_valid_rate'].item():.3f}"
                    f"|phase_valid_t={diagnostics['target_phase_valid_rate'].item():.3f}"
                    f"|domain_acc={diagnostics['domain_accuracy'].item():.3f}"
                    f"|grl={diagnostics['grl_coefficient'].item():.3f}"
                    f"|q_dom_w={domain_score_weight:.3f}|lr={lr:.2e}"
                )
            if global_step % training_config.log_step == 0:
                _write_diagnostics(writer, global_step, diagnostics)
                if writer is not None:
                    writer.add_scalar("train/loss/lr", lr, global_step)
            if bar is not None:
                bar.set_postfix(
                    total=f"{diagnostics['loss_reported_total'].item():.3f}",
                    task=f"{diagnostics['loss_task_total'].item():.3f}",
                    geometry=f"{diagnostics['loss_geometry_total'].item():.3f}",
                    grl=f"{diagnostics['grl_coefficient'].item():.3f}",
                    lr=f"{lr:.2e}",
                )
        if progress_disabled:
            averages = {name: meter.avg for name, meter in meters.items()}
            print(
                f"TRAIN_EPOCH|epoch={epoch + 1}/{training_config.epochs}"
                f"|steps={steps_per_epoch}|total={averages['loss_reported_total']:.4f}"
                f"|task={averages['loss_task_total']:.4f}"
                f"|geometry={averages['loss_geometry_total']:.4f}"
                f"|cls={averages['loss_classification']:.4f}"
                f"|quality={averages['loss_quality_total']:.4f}"
                f"|source_shape={averages['loss_source_shape']:.4f}"
                f"|source_raw={averages['loss_source_raw']:.4f}"
                f"|global_da={averages['loss_global_domain']:.4f}"
                f"|target_semantic={averages['loss_target_semantic']:.4f}"
                f"|task_step_success_rate={averages['task_step_success_rate']:.3f}"
                f"|task_step_skip_rate={averages['task_step_skip_rate']:.3f}"
                f"|source_state_update_rate={averages['source_state_update_rate']:.3f}"
            )
        for domain in ("source", "target"):
            prefix = f"{domain}_"
            phase_fields = [
                f"PHASE_EPOCH|epoch={epoch + 1}|domain={domain}"
            ]
            for name, meter in meters.items():
                short_name = name.removeprefix(prefix)
                if name.startswith(prefix) and (
                    short_name.startswith("candidate_")
                    or short_name.startswith("T_")
                    or short_name.startswith("S_")
                    or short_name
                    in (
                        "valid_identity_rate",
                        "valid_nonidentity_rate",
                        "failure_rate",
                    )
                ):
                    phase_fields.append(f"{short_name}={meter.avg:.6g}")
            print("|".join(phase_fields))
        for domain, label_source in (
            ("source", "source_true"),
            ("target", "target_geometry_pseudo"),
        ):
            accumulator = phase_class_epoch[domain]
            if accumulator is None:
                continue
            for class_id, summary in accumulator.summaries().items():
                fields = [
                    f"PHASE_CLASS_EPOCH|epoch={epoch + 1}|split={domain}",
                    f"label_source={label_source}",
                    f"class_id={class_id}",
                ]
                fields.extend(f"{name}={value:.6g}" for name, value in summary.items())
                print("|".join(fields))
                if writer is not None:
                    path = f"phase/{label_source}_class_{class_id}"
                    for name, value in summary.items():
                        if math.isfinite(float(value)):
                            writer.add_scalar(f"{path}/{name}", value, epoch)
        source_summaries = phase_class_epoch["source"].summaries()
        for class_id, summary in source_summaries.items():
            sampled = summary["sample_count"]
            presence = source_class_batch_presence[class_id] / steps_per_epoch
            print(
                f"SOURCE_BALANCE_EPOCH|epoch={epoch + 1}|class_id={class_id}"
                f"|source_sample_count={sampled}|source_batch_presence_rate={presence:.6g}"
            )
            if writer is not None:
                writer.add_scalar(f"protocol/source_sample_count_class_{class_id}", sampled, epoch)
                writer.add_scalar(f"protocol/source_batch_presence_rate_class_{class_id}", presence, epoch)
        for domain in ("source", "target"):
            summary = summarize_decomposition_diagnostics(decomposition_epoch[domain])
            print(_format_epoch_diagnostics("DECOMP_EPOCH", epoch + 1, domain, summary))
            if writer is not None:
                for name, value in summary.items():
                    writer.add_scalar(
                        f"train/decomposition/{domain}/{name}", value.item(), epoch
                    )
        model.eval()
        best_f1, metrics = _validation_with_structure_contributions(
            best_f1,
            best_model_path,
            training_config,
            device,
            epoch,
            model,
            val_loader,
            writer,
        )
        _log_validation_contributions(writer, epoch + 1, metrics)
    return best_f1
