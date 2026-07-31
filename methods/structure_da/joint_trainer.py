"""Joint source/target training for the structure-aware model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset import PixelSetData, create_train_loader
from evaluation import validation
from transforms import Normalize, RandomSamplePixels, ToTensor
from utils.train_utils import AverageMeter, cycle, progress_bar_disabled, to_cuda

from .eden_alignment import EDENDomainAlignmentOutput
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


@dataclass(frozen=True)
class JointStructureDALossOutput:
    total_loss: Tensor
    task_loss: Tensor
    quality_loss: QualityLossOutput
    geometry_loss: Tensor
    alignment_loss: Tensor


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
    mean_beta_trend_channel: Tensor
    mean_beta_dynamics_channel: Tensor


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
    log_step: int = 10
    progress_bar: str = "auto"
    classes: Sequence[str] = ()

    def __post_init__(self) -> None:
        _positive_int("epochs", self.epochs)
        _positive_int("steps_per_epoch", self.steps_per_epoch, optional=True)
        _positive_int("log_step", self.log_step)
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
        "beta_trend_channel": quality.beta_trend_channel,
        "beta_dynamics_channel": quality.beta_dynamics_channel,
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


def joint_structure_da_train_step(
    model,
    source_sample,
    target_sample,
    optimizer,
    quality_objective,
    training_config,
    device,
) -> JointStructureDATrainStepOutput:
    if not isinstance(model, StructureAwareDomainAdaptationModel):
        raise ValueError("model must be StructureAwareDomainAdaptationModel")
    if not isinstance(quality_objective, HierarchicalQualityObjective):
        raise ValueError("quality_objective must be HierarchicalQualityObjective")
    source_labels = source_sample["label"].to(device=device, dtype=torch.long)
    source_tensors = _sample_to_device(source_sample, device)
    target_tensors = _sample_to_device(target_sample, device)
    optimizer.zero_grad(set_to_none=True)
    model.update_source_state(*source_tensors)
    source_output = model.forward_details(*source_tensors)
    target_output = model.forward_details(*target_tensors)

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
    geometry = model.forward_source_geometry(source_output, source_tensors[2])
    geometry_loss = geometry.total_loss
    alignment = model.align(source_output, target_output)
    alignment_loss = alignment.loss
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
    total_loss.backward()
    optimizer.step()
    quality = merged_quality
    return JointStructureDATrainStepOutput(
        losses=JointStructureDALossOutput(
            total_loss=total_loss,
            task_loss=task_loss,
            quality_loss=quality_loss,
            geometry_loss=geometry_loss,
            alignment_loss=alignment_loss,
        ),
        alignment=alignment,
        source_batch_size=source_batch_size,
        target_batch_size=target_batch_size,
        mean_alpha_trend=quality.alpha_trend.mean(),
        mean_alpha_dynamics=quality.alpha_dynamics.mean(),
        mean_alpha_residual=quality.alpha_residual.mean(),
        mean_beta_trend_temporal=quality.beta_trend_temporal.mean(),
        mean_beta_dynamics_temporal=quality.beta_dynamics_temporal.mean(),
        mean_beta_trend_channel=quality.beta_trend_channel.mean(),
        mean_beta_dynamics_channel=quality.beta_dynamics_channel.mean(),
    )


def _resolve_steps(training_config, source_loader, target_loader) -> int:
    source_steps, target_steps = len(source_loader), len(target_loader)
    if source_steps == 0 or target_steps == 0:
        raise ValueError("source and target training loaders must be nonempty")
    return training_config.steps_per_epoch or max(source_steps, target_steps)


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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.epochs * steps_per_epoch,
        eta_min=0,
    )
    quality_objective = HierarchicalQualityObjective(
        structural_classification_weight=training_config.structural_classification_weight,
        structural_domain_weight=training_config.structural_domain_weight,
        component_classification_weight=training_config.component_classification_weight,
        component_domain_weight=training_config.component_domain_weight,
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_f1 = float("-inf")
    meter_names = (
        "total", "task", "quality_total", "quality_structural_cls",
        "quality_structural_domain", "quality_component_cls",
        "quality_component_domain", "geometry", "alignment",
        "domain_accuracy", "alpha_T", "alpha_D", "alpha_R",
        "beta_T_temp", "beta_D_temp", "beta_T_channel", "beta_D_channel",
    )
    for epoch in range(training_config.epochs):
        progress_disabled = progress_bar_disabled(training_config.progress_bar)
        model.train()
        meters = {name: AverageMeter() for name in meter_names}
        source_iterator = cycle(source_loader)
        target_iterator = cycle(target_loader)
        bar = tqdm(range(steps_per_epoch), disable=progress_disabled)
        for local_step in bar:
            global_step = epoch * steps_per_epoch + local_step
            result = joint_structure_da_train_step(
                model,
                next(source_iterator),
                next(target_iterator),
                optimizer,
                quality_objective,
                training_config,
                device,
            )
            scheduler.step()
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
            diagnostic_values = {
                "domain_accuracy": result.alignment.accuracy,
                "alpha_T": result.mean_alpha_trend,
                "alpha_D": result.mean_alpha_dynamics,
                "alpha_R": result.mean_alpha_residual,
                "beta_T_temp": result.mean_beta_trend_temporal,
                "beta_D_temp": result.mean_beta_dynamics_temporal,
                "beta_T_channel": result.mean_beta_trend_channel,
                "beta_D_channel": result.mean_beta_dynamics_channel,
            }
            diagnostic_batch_size = (
                result.source_batch_size + result.target_batch_size
            )
            for name, value in diagnostic_values.items():
                meters[name].update(
                    value.detach().item(), n=diagnostic_batch_size
                )
            lr = optimizer.param_groups[0]["lr"]
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
                for tag, value in (
                    ("train/alpha_trend", result.mean_alpha_trend),
                    ("train/alpha_dynamics", result.mean_alpha_dynamics),
                    ("train/alpha_residual", result.mean_alpha_residual),
                    ("train/beta_trend_temporal", result.mean_beta_trend_temporal),
                    ("train/beta_dynamics_temporal", result.mean_beta_dynamics_temporal),
                    ("train/beta_trend_channel", result.mean_beta_trend_channel),
                    ("train/beta_dynamics_channel", result.mean_beta_dynamics_channel),
                ):
                    writer.add_scalar(tag, value.detach().item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
            if not progress_disabled:
                bar.set_postfix(
                    total=f"{meters['total'].avg:.3f}",
                    task=f"{meters['task'].avg:.3f}",
                    geometry=f"{meters['geometry'].avg:.3f}",
                    align=f"{meters['alignment'].avg:.3f}",
                    grl=f"{result.alignment.coefficient.item():.3f}",
                    lr=f"{lr:.2e}",
                )
        if progress_disabled:
            print(
                f"TRAIN_EPOCH|epoch={epoch + 1}/{training_config.epochs}"
                f"|steps={steps_per_epoch}|total={meters['total'].avg:.4f}"
                f"|task={meters['task'].avg:.4f}"
                f"|quality={meters['quality_total'].avg:.4f}"
                f"|geometry={meters['geometry'].avg:.4f}"
                f"|alignment={meters['alignment'].avg:.4f}"
                f"|domain_accuracy={meters['domain_accuracy'].avg:.4f}"
                f"|alpha_T={meters['alpha_T'].avg:.4f}"
                f"|alpha_D={meters['alpha_D'].avg:.4f}"
                f"|alpha_R={meters['alpha_R'].avg:.4f}"
                f"|beta_T_temp={meters['beta_T_temp'].avg:.4f}"
                f"|beta_D_temp={meters['beta_D_temp'].avg:.4f}"
                f"|beta_T_channel={meters['beta_T_channel'].avg:.4f}"
                f"|beta_D_channel={meters['beta_D_channel'].avg:.4f}"
                f"|grl={result.alignment.coefficient.item():.3f}|lr={lr:.2e}"
            )
        model.eval()
        best_f1 = validation(
            best_f1,
            best_model_path,
            training_config,
            criterion,
            device,
            epoch,
            model,
            val_loader,
            writer,
        )
    return best_f1
