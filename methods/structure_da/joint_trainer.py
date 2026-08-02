"""Joint source/target training for the structure-aware model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import sklearn.metrics
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset import PixelSetData, create_train_loader
from transforms import Normalize, RandomSamplePixels, ToTensor
from utils.train_utils import AverageMeter, cycle, progress_bar_disabled, to_cuda

from .eden_alignment import EDENDomainAlignmentOutput
from .diagnostics import (
    ContributionDiagnostics,
    DecompositionDiagnostics,
    compute_decomposition_diagnostics,
    compute_structure_contribution_diagnostics,
    merge_contribution_diagnostics,
    merge_decomposition_diagnostics,
    summarize_contribution_diagnostics,
    summarize_decomposition_diagnostics,
)
from .full_model import StructureAwareDomainAdaptationModel
from .quality_fusion import (
    HierarchicalQualityObjective,
    QualityLossOutput,
    concatenate_hierarchical_quality_outputs,
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
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
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
    if resolved_fraction > 1.0:
        raise ValueError("grl_warmup_fraction must lie in [0, 1]")
    return max(1, round(epochs * steps_per_epoch * resolved_fraction))


@dataclass(frozen=True)
class JointStructureDALossOutput:
    total_loss: Tensor
    task_loss: Tensor
    quality_loss: QualityLossOutput
    geometry_loss: Tensor
    alignment_loss: Tensor


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
    source_batch_size: int
    target_batch_size: int
    mean_alpha_trend: Tensor
    mean_alpha_dynamics: Tensor
    mean_alpha_residual: Tensor
    mean_beta_trend_temporal: Tensor
    mean_beta_dynamics_temporal: Tensor
    diagnostics: JointStructureDADiagnostics
    source_decomposition_diagnostics: DecompositionDiagnostics
    target_decomposition_diagnostics: DecompositionDiagnostics
    source_contribution_diagnostics: ContributionDiagnostics
    target_contribution_diagnostics: ContributionDiagnostics


@dataclass(frozen=True)
class JointStructureDATrainingConfig:
    epochs: int
    steps_per_epoch: int | None
    lr: float
    weight_decay: float
    task_weight: float = 1.0
    geometry_weight: float = 1.0
    alignment_weight: float = 1.0
    structural_classification_weight: float = 1.0
    structural_domain_weight: float = 1.0
    component_classification_weight: float = 1.0
    component_domain_weight: float = 1.0
    quality_domain_score_warmup_epochs: int = 5
    amp: bool = False
    amp_dtype: str = "float16"
    log_step: int = 10
    progress_bar: str = "auto"
    classes: Sequence[str] = ()

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
        if self.amp_dtype not in ("float16", "bfloat16"):
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
        object.__setattr__(self, "lr", _finite_nonnegative("lr", self.lr, positive=True))
        object.__setattr__(self, "weight_decay", _finite_nonnegative("weight_decay", self.weight_decay))
        for name in (
            "task_weight", "geometry_weight", "alignment_weight",
            "structural_classification_weight", "structural_domain_weight",
            "component_classification_weight", "component_domain_weight",
        ):
            object.__setattr__(self, name, _finite_nonnegative(name, getattr(self, name)))
        progress_bar_disabled(self.progress_bar)


def create_joint_structure_da_train_loaders(config, splits):
    """Create independently shuffled source and target loaders without extras."""
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
    return (
        create_train_loader(source_dataset, config.batch_size, config.num_workers),
        create_train_loader(target_dataset, config.batch_size, config.num_workers),
    )


def _sample_to_device(sample, device):
    if torch.device(device).type != "cpu":
        return to_cuda(sample, device)
    return tuple(
        sample[name].to(device) if name in sample else None
        for name in ("pixels", "valid_pixels", "positions", "extra")
    )


def _scalar_description(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        return f"non-tensor({type(value).__name__})"
    if value.ndim != 0:
        return f"shape={tuple(value.shape)}"
    return repr(value.detach().item())


def _check_loss_scalars(**losses: Tensor) -> None:
    invalid = [
        name
        for name, value in losses.items()
        if not isinstance(value, Tensor)
        or value.ndim != 0
        or not torch.isfinite(value.detach()).item()
    ]
    if invalid:
        values = ", ".join(
            f"{name}={_scalar_description(value)}"
            for name, value in losses.items()
        )
        raise FloatingPointError(
            f"invalid scalar loss before backward ({', '.join(invalid)}): {values}"
        )


def _check_bounded_training_values(model, quality, alignment) -> None:
    coefficients = {
        "alpha_trend": quality.alpha_trend,
        "alpha_dynamics": quality.alpha_dynamics,
        "alpha_residual": quality.alpha_residual,
        "beta_trend_temporal": quality.beta_trend_temporal,
        "beta_dynamics_temporal": quality.beta_dynamics_temporal,
    }
    for name, value in coefficients.items():
        detached = value.detach()
        if (
            not torch.isfinite(detached).all().item()
            or not ((detached >= 0) & (detached <= 1)).all().item()
        ):
            raise FloatingPointError(
                f"{name} must contain only finite values in [0, 1]"
            )

    grl = alignment.coefficient.detach()
    grl_low = model.alignment.grl.low
    grl_high = model.alignment.grl.high
    if (
        grl.ndim != 0
        or not torch.isfinite(grl).item()
        or not grl_low <= grl.item() <= grl_high
    ):
        raise FloatingPointError(
            "GRL coefficient must be a finite scalar in "
            f"[{grl_low}, {grl_high}]"
        )


def _masked_mean_square(value: Tensor, time_mask: Tensor) -> Tensor:
    mask = time_mask
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return value.square().masked_select(mask.expand_as(value)).mean()


def _mean_l2_norm(value: Tensor) -> Tensor:
    return torch.linalg.vector_norm(value, dim=-1).mean()


def _domain_diagnostics(prefix: str, output, eps: float) -> dict[str, Tensor]:
    decomposition = output.backbone.decomposition
    time_mask = output.backbone.time_mask
    energies = {
        "trend": _masked_mean_square(decomposition.trend, time_mask),
        "dynamics": _masked_mean_square(decomposition.dynamics, time_mask),
        "residual": _masked_mean_square(decomposition.residual, time_mask),
    }
    energy_total = sum(energies.values())
    reconstruction = (
        decomposition.trend + decomposition.dynamics + decomposition.residual
    )
    reconstruction_error = _masked_mean_square(
        reconstruction - output.backbone.tokens,
        time_mask,
    ) / (
        _masked_mean_square(output.backbone.tokens, time_mask) + eps
    )
    values = {
        f"{prefix}_{name}_energy_fraction": energy / (energy_total + eps)
        for name, energy in energies.items()
    }
    values[f"{prefix}_reconstruction_relative_error"] = reconstruction_error
    values.update(
        {
            f"{prefix}_temporal_T_valid_rate": output.temporal.trend.encoded.valid.float().mean(),
            f"{prefix}_temporal_D_valid_rate": output.temporal.dynamics.encoded.valid.float().mean(),
            f"{prefix}_raw_T_norm": _mean_l2_norm(output.representation.trend_embedding),
            f"{prefix}_raw_D_norm": _mean_l2_norm(output.representation.dynamics_embedding),
            f"{prefix}_raw_R_norm": _mean_l2_norm(output.representation.residual_embedding),
            f"{prefix}_temporal_T_norm": _mean_l2_norm(output.representation.temporal_features.trend),
            f"{prefix}_temporal_D_norm": _mean_l2_norm(output.representation.temporal_features.dynamics),
            f"{prefix}_raw_fusion_norm": _mean_l2_norm(output.representation.quality.raw_fusion),
            f"{prefix}_temporal_fusion_norm": _mean_l2_norm(output.representation.quality.temporal_fusion),
            f"{prefix}_fused_feature_norm": _mean_l2_norm(output.representation.quality.fused_feature),
        }
    )
    quality = output.representation.quality
    for name, coefficient in (
        ("alpha_T", quality.alpha_trend),
        ("alpha_D", quality.alpha_dynamics),
        ("alpha_R", quality.alpha_residual),
        ("beta_T_temporal", quality.beta_trend_temporal),
        ("beta_D_temporal", quality.beta_dynamics_temporal),
    ):
        values[f"{prefix}_{name}_mean"] = coefficient.mean()
        values[f"{prefix}_{name}_std"] = coefficient.std(unbiased=False)
    return values


def _build_diagnostics(
    model,
    source_output,
    target_output,
    source_labels,
    losses,
    geometry,
    alignment,
) -> JointStructureDADiagnostics:
    trend_geometry = geometry.temporal.trend.geometry
    dynamics_geometry = geometry.temporal.dynamics.geometry
    decomposition = model.backbone.decomposition
    values = {
        "source_train_accuracy": (
            source_output.representation.logits.argmax(dim=-1) == source_labels
        ).float().mean(),
        "domain_accuracy": alignment.accuracy,
        "grl_coefficient": alignment.coefficient,
        "loss_total": losses.total_loss,
        "loss_task": losses.task_loss,
        "loss_quality_total": losses.quality_loss.total_loss,
        "loss_quality_structural_cls": losses.quality_loss.structural_classification_loss,
        "loss_quality_structural_domain": losses.quality_loss.structural_domain_loss,
        "loss_quality_component_cls": losses.quality_loss.component_classification_loss,
        "loss_quality_component_domain": losses.quality_loss.component_domain_loss,
        "loss_geometry_total": losses.geometry_loss,
        "loss_alignment": losses.alignment_loss,
        "geometry_T_alignment": trend_geometry.alignment_loss,
        "geometry_T_roughness": trend_geometry.roughness_loss,
        "geometry_T_unsupported": trend_geometry.unsupported_loss,
        "geometry_T_phase_center": trend_geometry.center_loss,
        "geometry_D_alignment": dynamics_geometry.alignment_loss,
        "geometry_D_roughness": dynamics_geometry.roughness_loss,
        "geometry_D_unsupported": dynamics_geometry.unsupported_loss,
        "geometry_D_phase_center": dynamics_geometry.center_loss,
        "tau_fast": decomposition.tau_fast,
        "tau_slow": decomposition.tau_slow,
        "tau_gap": decomposition.tau_slow - decomposition.tau_fast,
    }
    values.update(_domain_diagnostics("source", source_output, decomposition.eps))
    values.update(_domain_diagnostics("target", target_output, decomposition.eps))
    return JointStructureDADiagnostics(
        scalars={name: value.detach() for name, value in values.items()}
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


def _contribution_diagnostics(output) -> ContributionDiagnostics:
    representation = output.representation
    quality = representation.quality
    return compute_structure_contribution_diagnostics(
        alpha_T=quality.alpha_trend,
        alpha_D=quality.alpha_dynamics,
        alpha_R=quality.alpha_residual,
        beta_T_temporal=quality.beta_trend_temporal,
        beta_D_temporal=quality.beta_dynamics_temporal,
        temporal_T=representation.temporal_features.trend,
        temporal_D=representation.temporal_features.dynamics,
        raw_fusion=quality.raw_fusion,
        temporal_fusion=quality.temporal_fusion,
        temporal_T_valid=representation.temporal_features.trend_valid,
        temporal_D_valid=representation.temporal_features.dynamics_valid,
    )


def joint_structure_da_train_step(
    model,
    source_sample,
    target_sample,
    task_optimizer,
    geometry_optimizer,
    quality_objective,
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
    if not isinstance(quality_objective, HierarchicalQualityObjective):
        raise ValueError("quality_objective must be HierarchicalQualityObjective")
    source_labels = source_sample["label"].to(device=device, dtype=torch.long)
    source_tensors = _sample_to_device(source_sample, device)
    target_tensors = _sample_to_device(target_sample, device)
    task_optimizer.zero_grad(set_to_none=True)
    geometry_optimizer.zero_grad(set_to_none=True)
    amp_enabled = training_config.amp and device.type == "cuda"
    amp_dtype = getattr(torch, training_config.amp_dtype)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        source_backbone = model.forward_backbone(*source_tensors)
        target_backbone = model.forward_backbone(*target_tensors)
        model.update_source_state_from_backbone(
            model.detach_backbone_for_state(source_backbone), source_tensors[2]
        )
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

        task_loss = F.cross_entropy(
            source_output.representation.logits, source_labels
        )
        merged_quality = concatenate_hierarchical_quality_outputs(
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
        alignment_loss = alignment.loss
    with torch.autocast(device_type=device.type, enabled=False):
        geometry = model.forward_source_geometry(source_output, source_tensors[2])
        geometry_loss = geometry.total_loss.float()
    total_loss = (
        training_config.task_weight * task_loss
        + quality_loss.total_loss
        + training_config.geometry_weight * geometry_loss
        + training_config.alignment_weight * alignment_loss
    )
    _check_loss_scalars(
        task_loss=task_loss,
        quality_loss=quality_loss.total_loss,
        geometry_loss=geometry_loss,
        alignment_loss=alignment_loss,
        total_loss=total_loss,
    )
    _check_bounded_training_values(model, merged_quality, alignment)
    loss_output = JointStructureDALossOutput(
        total_loss=total_loss,
        task_loss=task_loss,
        quality_loss=quality_loss,
        geometry_loss=geometry_loss,
        alignment_loss=alignment_loss,
    )
    diagnostics = _build_diagnostics(
        model,
        source_output,
        target_output,
        source_labels,
        loss_output,
        geometry,
        alignment,
    )
    diagnostic_eps = model.backbone.decomposition.eps
    source_decomposition_diagnostics = _decomposition_diagnostics(
        source_output, source_tensors[2], diagnostic_eps
    )
    target_decomposition_diagnostics = _decomposition_diagnostics(
        target_output, target_tensors[2], diagnostic_eps
    )
    source_contribution_diagnostics = _contribution_diagnostics(source_output)
    target_contribution_diagnostics = _contribution_diagnostics(target_output)
    weighted_geometry_loss = training_config.geometry_weight * geometry_loss
    task_total_loss = (
        training_config.task_weight * task_loss
        + quality_loss.total_loss
        + training_config.alignment_weight * alignment_loss
    )
    weighted_geometry_loss.backward()
    geometry_optimizer.step()
    if geometry_scheduler is not None:
        geometry_scheduler.step()
    if task_scaler is not None and task_scaler.is_enabled():
        scale_before = task_scaler.get_scale()
        task_scaler.scale(task_total_loss).backward()
        task_scaler.step(task_optimizer)
        task_scaler.update()
        task_step_succeeded = task_scaler.get_scale() >= scale_before
    else:
        task_total_loss.backward()
        task_optimizer.step()
        task_step_succeeded = True
    if task_scheduler is not None and task_step_succeeded:
        task_scheduler.step()
    quality = merged_quality
    return JointStructureDATrainStepOutput(
        losses=loss_output,
        alignment=alignment,
        source_batch_size=source_batch_size,
        target_batch_size=target_batch_size,
        mean_alpha_trend=quality.alpha_trend.mean(),
        mean_alpha_dynamics=quality.alpha_dynamics.mean(),
        mean_alpha_residual=quality.alpha_residual.mean(),
        mean_beta_trend_temporal=quality.beta_trend_temporal.mean(),
        mean_beta_dynamics_temporal=quality.beta_dynamics_temporal.mean(),
        diagnostics=diagnostics,
        source_decomposition_diagnostics=source_decomposition_diagnostics,
        target_decomposition_diagnostics=target_decomposition_diagnostics,
        source_contribution_diagnostics=source_contribution_diagnostics,
        target_contribution_diagnostics=target_contribution_diagnostics,
    )


def _resolve_steps(training_config, source_loader, target_loader) -> int:
    source_steps, target_steps = len(source_loader), len(target_loader)
    if source_steps == 0 or target_steps == 0:
        raise ValueError("source and target training loaders must be nonempty")
    return training_config.steps_per_epoch or source_steps


def resolve_steps_per_epoch(training_config, source_loader, target_loader) -> int:
    return _resolve_steps(training_config, source_loader, target_loader)


def _diagnostic_tensorboard_tag(name: str) -> str | None:
    direct = {
        "source_train_accuracy": "train/source_train_accuracy",
        "tau_fast": "train/decomposition/tau_fast",
        "tau_slow": "train/decomposition/tau_slow",
        "tau_gap": "train/decomposition/tau_gap",
        "geometry_T_alignment": "train/geometry/trend_alignment",
        "geometry_T_roughness": "train/geometry/trend_roughness",
        "geometry_T_unsupported": "train/geometry/trend_unsupported",
        "geometry_T_phase_center": "train/geometry/trend_phase_center",
        "geometry_D_alignment": "train/geometry/dynamics_alignment",
        "geometry_D_roughness": "train/geometry/dynamics_roughness",
        "geometry_D_unsupported": "train/geometry/dynamics_unsupported",
        "geometry_D_phase_center": "train/geometry/dynamics_phase_center",
    }
    if name in direct:
        return direct[name]
    for domain in ("source", "target"):
        prefix = f"{domain}_"
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        suffix_tags = {
            "trend_energy_fraction": "energy_fraction_trend",
            "dynamics_energy_fraction": "energy_fraction_dynamics",
            "residual_energy_fraction": "energy_fraction_residual",
            "reconstruction_relative_error": "reconstruction_relative_error",
            "temporal_T_valid_rate": "temporal_valid_trend",
            "temporal_D_valid_rate": "temporal_valid_dynamics",
            "raw_T_norm": "raw_norm_trend",
            "raw_D_norm": "raw_norm_dynamics",
            "raw_R_norm": "raw_norm_residual",
            "temporal_T_norm": "temporal_norm_trend",
            "temporal_D_norm": "temporal_norm_dynamics",
            "raw_fusion_norm": "fusion_norm_raw",
            "temporal_fusion_norm": "fusion_norm_temporal",
            "fused_feature_norm": "fusion_norm_final",
            "alpha_T_mean": "alpha_trend_mean",
            "alpha_T_std": "alpha_trend_std",
            "alpha_D_mean": "alpha_dynamics_mean",
            "alpha_D_std": "alpha_dynamics_std",
            "alpha_R_mean": "alpha_residual_mean",
            "alpha_R_std": "alpha_residual_std",
            "beta_T_temporal_mean": "beta_trend_temporal_mean",
            "beta_T_temporal_std": "beta_trend_temporal_std",
            "beta_D_temporal_mean": "beta_dynamics_temporal_mean",
            "beta_D_temporal_std": "beta_dynamics_temporal_std",
        }
        if suffix in suffix_tags:
            return f"train/{domain}/{suffix_tags[suffix]}"
    return None


@torch.inference_mode()
def _collect_validation_structure_contributions(
    model,
    val_loader,
    device,
    classes: Sequence[str],
) -> tuple[dict[str, float], Tensor, Tensor]:
    """Evaluate fusion-block counterfactuals from one detailed pass per batch."""

    if not classes:
        raise ValueError("classes must be nonempty")
    mode_logits: dict[str, list[Tensor]] = {
        "full": [],
        "raw_only": [],
    }
    labels: list[Tensor] = []
    model.eval()
    for sample in val_loader:
        target = sample["label"].to(device=device, dtype=torch.long)
        pixels, valid_pixels, positions, extra = _sample_to_device(sample, device)
        output = model.forward_details(pixels, valid_pixels, positions, extra)
        quality = output.representation.quality
        raw = quality.raw_fusion
        temporal = quality.temporal_fusion
        zero_temporal = torch.zeros_like(temporal)
        mode_logits["full"].append(output.representation.logits.detach().cpu())
        features = {
            "raw_only": torch.cat([raw, zero_temporal], dim=-1),
        }
        for name, feature in features.items():
            mode_logits[name].append(
                model.representation.classifier(feature).detach().cpu()
            )
        labels.append(target.detach().cpu())
    if not labels:
        raise ValueError("validation loader must be nonempty")
    all_labels = torch.cat(labels)
    per_mode: dict[str, dict[str, float]] = {}
    for name, chunks in mode_logits.items():
        logits = torch.cat(chunks)
        predictions = logits.argmax(dim=-1)
        per_mode[name] = {
            "macro_f1": float(
                sklearn.metrics.f1_score(
                    all_labels.numpy(),
                    predictions.numpy(),
                    average="macro",
                    zero_division=0,
                )
            ),
            "cross_entropy": float(
                F.cross_entropy(logits, all_labels, reduction="mean").item()
            ),
        }
    metrics = {
        **{
            f"{name}_loss": values["cross_entropy"]
            for name, values in per_mode.items()
        },
        **{
            f"{name}_f1": values["macro_f1"]
            for name, values in per_mode.items()
        },
        "delta_structure": (
            per_mode["full"]["macro_f1"]
            - per_mode["raw_only"]["macro_f1"]
        ),
    }
    return metrics, all_labels, torch.cat(mode_logits["full"]).argmax(dim=-1)


@torch.inference_mode()
def validation_structure_contributions(
    model,
    val_loader,
    device,
    classes: Sequence[str],
) -> dict[str, float]:
    """Return full-dataset counterfactual contribution metrics."""

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
    val_loss = metrics["full_loss"]
    val_f1 = metrics["full_f1"]
    val_accuracy = float(
        sklearn.metrics.accuracy_score(labels.numpy(), predictions.numpy())
    )
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
        print(f"Validation F1 improved from {best_f1:.4f} to {val_f1:.4f}!")
        best_f1 = val_f1
        if best_model_path is not None:
            print(f"Saving best model to {best_model_path}")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_f1": best_f1,
                },
                best_model_path,
            )
    else:
        print(f"Validation F1 did not improve from {best_f1:.4f}.")
    return best_f1, metrics


def _merge_optional(current, update, merge):
    return update if current is None else merge(current, update)


def _format_epoch_diagnostics(
    prefix: str, epoch: int, domain: str, values: Mapping[str, Tensor]
) -> str:
    fields = [f"{prefix}|epoch={epoch}|domain={domain}"]
    for name, value in values.items():
        scalar = value.item()
        fields.append(f"{name}={scalar:.6g}")
    return "|".join(fields)


def _write_epoch_diagnostics(
    writer,
    epoch: int,
    domain: str,
    category: str,
    values: Mapping[str, Tensor],
) -> None:
    if writer is None:
        return
    for name, value in values.items():
        writer.add_scalar(
            f"train/diagnostics/{domain}/{category}/{name}", value.item(), epoch
        )


def _log_validation_contributions(writer, epoch: int, metrics) -> None:
    fields = [f"VAL_CONTRIBUTION|epoch={epoch}"]
    for name, value in metrics.items():
        fields.append(f"{name}={value:.6g}")
        if writer is not None:
            writer.add_scalar(f"val/counterfactual/{name}", value, epoch - 1)
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
        task_optimizer,
        T_max=training_config.epochs * steps_per_epoch,
        eta_min=0,
    )
    geometry_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        geometry_optimizer,
        T_max=training_config.epochs * steps_per_epoch,
        eta_min=0,
    )
    amp_enabled = training_config.amp and device.type == "cuda"
    task_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and training_config.amp_dtype == "float16",
    )
    quality_objective = HierarchicalQualityObjective(
        structural_classification_weight=training_config.structural_classification_weight,
        structural_domain_weight=training_config.structural_domain_weight,
        component_classification_weight=training_config.component_classification_weight,
        component_domain_weight=training_config.component_domain_weight,
    )
    best_f1 = float("-inf")
    meter_names = (
        "total", "task", "quality_total", "quality_structural_cls",
        "quality_structural_domain", "quality_component_cls",
        "quality_component_domain", "geometry", "alignment",
    )
    for epoch in range(training_config.epochs):
        domain_score_weight = resolve_domain_score_weight(
            epoch, training_config.quality_domain_score_warmup_epochs
        )
        progress_disabled = progress_bar_disabled(training_config.progress_bar)
        model.train()
        meters = {name: AverageMeter() for name in meter_names}
        diagnostic_meters: dict[str, AverageMeter] = {}
        decomposition_epoch = {"source": None, "target": None}
        contribution_epoch = {"source": None, "target": None}
        lr_meter = AverageMeter()
        source_iterator = cycle(source_loader)
        target_iterator = cycle(target_loader)
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
                training_config,
                device,
                domain_score_weight=domain_score_weight,
                task_scheduler=task_scheduler,
                geometry_scheduler=geometry_scheduler,
                task_scaler=task_scaler,
            )
            losses = result.losses
            values = {
                "total": losses.total_loss,
                "task": losses.task_loss,
                "quality_total": losses.quality_loss.total_loss,
                "quality_structural_cls": losses.quality_loss.structural_classification_loss,
                "quality_structural_domain": losses.quality_loss.structural_domain_loss,
                "quality_component_cls": losses.quality_loss.component_classification_loss,
                "quality_component_domain": losses.quality_loss.component_domain_loss,
                "geometry": losses.geometry_loss,
                "alignment": losses.alignment_loss,
            }
            for name, value in values.items():
                meters[name].update(value.detach().item(), n=result.source_batch_size)
            for name, value in result.diagnostics.scalars.items():
                diagnostic_meters.setdefault(name, AverageMeter()).update(
                    value.item()
                )
            decomposition_epoch["source"] = _merge_optional(
                decomposition_epoch["source"],
                result.source_decomposition_diagnostics,
                merge_decomposition_diagnostics,
            )
            decomposition_epoch["target"] = _merge_optional(
                decomposition_epoch["target"],
                result.target_decomposition_diagnostics,
                merge_decomposition_diagnostics,
            )
            contribution_epoch["source"] = _merge_optional(
                contribution_epoch["source"],
                result.source_contribution_diagnostics,
                merge_contribution_diagnostics,
            )
            contribution_epoch["target"] = _merge_optional(
                contribution_epoch["target"],
                result.target_contribution_diagnostics,
                merge_contribution_diagnostics,
            )
            lr = task_optimizer.param_groups[0]["lr"]
            lr_meter.update(lr)
            diagnostics = result.diagnostics.scalars
            if (
                progress_disabled
                and global_step % training_config.log_step == 0
            ):
                print(
                    f"TRAIN_STEP|epoch={epoch + 1}/{training_config.epochs}"
                    f"|step={local_step + 1}/{steps_per_epoch}"
                    f"|total={diagnostics['loss_total'].item():.4f}"
                    f"|task={diagnostics['loss_task'].item():.4f}"
                    f"|q_total={diagnostics['loss_quality_total'].item():.4f}"
                    f"|geometry={diagnostics['loss_geometry_total'].item():.4f}"
                    f"|alignment={diagnostics['loss_alignment'].item():.4f}"
                    f"|train_acc={diagnostics['source_train_accuracy'].item():.4f}"
                    f"|domain_acc={diagnostics['domain_accuracy'].item():.4f}"
                    f"|grl={diagnostics['grl_coefficient'].item():.3f}"
                    f"|q_dom_w={domain_score_weight:.3f}"
                    f"|lr={lr:.2e}"
                )
            if global_step % training_config.log_step == 0 and writer is not None:
                tags = {
                    "total": "train/loss_total",
                    "task": "train/loss_task",
                    "quality_total": "train/loss_quality_total",
                    "quality_structural_cls": "train/loss_quality_structural_cls",
                    "quality_structural_domain": "train/loss_quality_structural_domain",
                    "quality_component_cls": "train/loss_quality_component_cls",
                    "quality_component_domain": "train/loss_quality_component_domain",
                    "geometry": "train/loss_geometry",
                    "alignment": "train/loss_alignment",
                }
                for name, tag in tags.items():
                    writer.add_scalar(tag, meters[name].val, global_step)
                writer.add_scalar("train/domain_accuracy", result.alignment.accuracy.detach().item(), global_step)
                writer.add_scalar("train/grl_coefficient", result.alignment.coefficient.detach().item(), global_step)
                writer.add_scalar(
                    "train/quality/domain_score_weight",
                    domain_score_weight,
                    global_step,
                )
                for tag, value in (
                    ("train/alpha_trend", result.mean_alpha_trend),
                    ("train/alpha_dynamics", result.mean_alpha_dynamics),
                    ("train/alpha_residual", result.mean_alpha_residual),
                    ("train/beta_trend_temporal", result.mean_beta_trend_temporal),
                    ("train/beta_dynamics_temporal", result.mean_beta_dynamics_temporal),
                ):
                    writer.add_scalar(tag, value.detach().item(), global_step)
                for name, value in diagnostics.items():
                    tag = _diagnostic_tensorboard_tag(name)
                    if tag is not None:
                        writer.add_scalar(tag, value.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
            if bar is not None:
                bar.set_postfix(
                    total=f"{meters['total'].avg:.3f}",
                    task=f"{meters['task'].avg:.3f}",
                    geometry=f"{meters['geometry'].avg:.3f}",
                    align=f"{meters['alignment'].avg:.3f}",
                    grl=f"{result.alignment.coefficient.item():.3f}",
                    lr=f"{lr:.2e}",
                )
        if progress_disabled:
            diagnostic_average = {
                name: meter.avg for name, meter in diagnostic_meters.items()
            }
            print(
                f"TRAIN_EPOCH|epoch={epoch + 1}/{training_config.epochs}"
                f"|steps={steps_per_epoch}|total={meters['total'].avg:.4f}"
                f"|task={meters['task'].avg:.4f}"
                f"|q_total={meters['quality_total'].avg:.4f}"
                f"|q_struct_cls={meters['quality_structural_cls'].avg:.4f}"
                f"|q_struct_dom={meters['quality_structural_domain'].avg:.4f}"
                f"|q_comp_cls={meters['quality_component_cls'].avg:.4f}"
                f"|q_comp_dom={meters['quality_component_domain'].avg:.4f}"
                f"|geometry={meters['geometry'].avg:.4f}"
                f"|alignment={meters['alignment'].avg:.4f}"
                f"|train_acc={diagnostic_average['source_train_accuracy']:.4f}"
                f"|domain_acc={diagnostic_average['domain_accuracy']:.4f}"
                f"|grl={diagnostic_average['grl_coefficient']:.3f}"
                f"|q_dom_w={domain_score_weight:.3f}"
                f"|lr={lr_meter.avg:.2e}"
            )
            print(
                f"STRUCTURE_EPOCH|epoch={epoch + 1}"
                f"|tau_fast={diagnostic_average['tau_fast']:.6f}"
                f"|tau_slow={diagnostic_average['tau_slow']:.6f}"
                f"|tau_gap={diagnostic_average['tau_gap']:.6f}"
                f"|energy_T_s={diagnostic_average['source_trend_energy_fraction']:.4f}"
                f"|energy_D_s={diagnostic_average['source_dynamics_energy_fraction']:.4f}"
                f"|energy_R_s={diagnostic_average['source_residual_energy_fraction']:.4f}"
                f"|energy_T_t={diagnostic_average['target_trend_energy_fraction']:.4f}"
                f"|energy_D_t={diagnostic_average['target_dynamics_energy_fraction']:.4f}"
                f"|energy_R_t={diagnostic_average['target_residual_energy_fraction']:.4f}"
                f"|reconstruction_s={diagnostic_average['source_reconstruction_relative_error']:.3e}"
                f"|reconstruction_t={diagnostic_average['target_reconstruction_relative_error']:.3e}"
                f"|temporal_T_valid_s={diagnostic_average['source_temporal_T_valid_rate']:.4f}"
                f"|temporal_D_valid_s={diagnostic_average['source_temporal_D_valid_rate']:.4f}"
                f"|temporal_T_valid_t={diagnostic_average['target_temporal_T_valid_rate']:.4f}"
                f"|temporal_D_valid_t={diagnostic_average['target_temporal_D_valid_rate']:.4f}"
                f"|raw_fusion_norm_s={diagnostic_average['source_raw_fusion_norm']:.4f}"
                f"|raw_fusion_norm_t={diagnostic_average['target_raw_fusion_norm']:.4f}"
                f"|temporal_fusion_norm_s={diagnostic_average['source_temporal_fusion_norm']:.4f}"
                f"|temporal_fusion_norm_t={diagnostic_average['target_temporal_fusion_norm']:.4f}"
            )
            print(
                f"QUALITY_EPOCH|epoch={epoch + 1}"
                f"|domain_score_weight={domain_score_weight:.3f}"
                f"|alpha_T_s={diagnostic_average['source_alpha_T_mean']:.4f}"
                f"|alpha_T_t={diagnostic_average['target_alpha_T_mean']:.4f}"
                f"|alpha_D_s={diagnostic_average['source_alpha_D_mean']:.4f}"
                f"|alpha_D_t={diagnostic_average['target_alpha_D_mean']:.4f}"
                f"|alpha_R_s={diagnostic_average['source_alpha_R_mean']:.4f}"
                f"|alpha_R_t={diagnostic_average['target_alpha_R_mean']:.4f}"
                f"|beta_T_temporal_s={diagnostic_average['source_beta_T_temporal_mean']:.4f}"
                f"|beta_T_temporal_t={diagnostic_average['target_beta_T_temporal_mean']:.4f}"
                f"|beta_D_temporal_s={diagnostic_average['source_beta_D_temporal_mean']:.4f}"
                f"|beta_D_temporal_t={diagnostic_average['target_beta_D_temporal_mean']:.4f}"
            )
            print(
                f"GEOMETRY_EPOCH|epoch={epoch + 1}"
                f"|T_align={diagnostic_average['geometry_T_alignment']:.4f}"
                f"|T_rough={diagnostic_average['geometry_T_roughness']:.4f}"
                f"|T_unsupported={diagnostic_average['geometry_T_unsupported']:.4f}"
                f"|T_center={diagnostic_average['geometry_T_phase_center']:.4f}"
                f"|D_align={diagnostic_average['geometry_D_alignment']:.4f}"
                f"|D_rough={diagnostic_average['geometry_D_roughness']:.4f}"
                f"|D_unsupported={diagnostic_average['geometry_D_unsupported']:.4f}"
                f"|D_center={diagnostic_average['geometry_D_phase_center']:.4f}"
            )
        for domain in ("source", "target"):
            decomposition_summary = summarize_decomposition_diagnostics(
                decomposition_epoch[domain]
            )
            contribution_summary = summarize_contribution_diagnostics(
                contribution_epoch[domain]
            )
            print(
                _format_epoch_diagnostics(
                    "DECOMP_EPOCH", epoch + 1, domain, decomposition_summary
                )
            )
            print(
                _format_epoch_diagnostics(
                    "CONTRIBUTION_EPOCH",
                    epoch + 1,
                    domain,
                    contribution_summary,
                )
            )
            _write_epoch_diagnostics(
                writer, epoch, domain, "decomposition", decomposition_summary
            )
            _write_epoch_diagnostics(
                writer, epoch, domain, "contribution", contribution_summary
            )
        model.eval()
        best_f1, validation_contributions = _validation_with_structure_contributions(
            best_f1,
            best_model_path,
            training_config,
            device,
            epoch,
            model,
            val_loader,
            writer,
        )
        _log_validation_contributions(
            writer, epoch + 1, validation_contributions
        )
    return best_f1
