"""Formal joint source/target training loop for Structure DA."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset import PixelSetData, create_train_loader
from evaluation import validation
from transforms import Normalize, RandomSamplePixels, ToTensor
from utils.train_utils import AverageMeter, cycle, progress_bar_disabled, to_cuda

from .losses import (
    LossWeights,
    StructureDALosses,
    classification_loss,
    component_diversity_loss,
    compose_total_loss,
    quality_classification_loss,
    quality_domain_loss,
    structural_adversarial_loss,
)
from .schedules import grl_coefficient, quality_gate_progress


def _positive_int(name, value, *, optional=False):
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_real(name, value, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{name} must be a finite {qualifier}real number")
    return value


@dataclass(frozen=True)
class StructureDATrainingConfig:
    epochs: int
    steps_per_epoch: Optional[int]
    lr: float
    weight_decay: float
    quality_warmup_steps: Optional[int]
    grl_warmup_steps: Optional[int]
    grl_gamma: float
    loss_weights: LossWeights
    log_step: int
    progress_bar: str = "auto"
    classes: Sequence[str] = ()

    def __post_init__(self):
        _positive_int("epochs", self.epochs)
        _positive_int("steps_per_epoch", self.steps_per_epoch, optional=True)
        _positive_int("quality_warmup_steps", self.quality_warmup_steps, optional=True)
        _positive_int("grl_warmup_steps", self.grl_warmup_steps, optional=True)
        _positive_int("log_step", self.log_step)
        object.__setattr__(self, "lr", _finite_real("lr", self.lr, positive=True))
        object.__setattr__(self, "weight_decay", _finite_real("weight_decay", self.weight_decay))
        object.__setattr__(self, "grl_gamma", _finite_real("grl_gamma", self.grl_gamma, positive=True))
        if not isinstance(self.loss_weights, LossWeights):
            raise ValueError("loss_weights must be a LossWeights instance")
        progress_bar_disabled(self.progress_bar)


@dataclass(frozen=True)
class ResolvedStructureDATraining:
    steps_per_epoch: int
    quality_warmup_steps: int
    grl_warmup_steps: int


@dataclass(frozen=True)
class StructureDATrainStepOutput:
    losses: StructureDALosses
    quality_progress: float
    grl_coefficient: float
    source_batch_size: int
    target_batch_size: int


def create_structure_da_train_loaders(config, splits):
    """Create independently shuffled source and target training loaders."""
    source_transform = transforms.Compose(
        [RandomSamplePixels(config.num_pixels), Normalize(), ToTensor()]
    )
    target_transform = transforms.Compose(
        [RandomSamplePixels(config.num_pixels), Normalize(), ToTensor()]
    )
    common = dict(
        data_root=config.data_root,
        classes=config.classes,
        with_extra=config.with_extra,
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


def resolve_structure_da_training(training_config, source_loader, target_loader):
    source_steps, target_steps = len(source_loader), len(target_loader)
    if source_steps == 0 or target_steps == 0:
        domains = []
        if source_steps == 0:
            domains.append("source")
        if target_steps == 0:
            domains.append("target")
        raise ValueError(
            f"Empty {' and '.join(domains)} training loader; batch_size may be "
            "larger than the available training split."
        )
    steps = training_config.steps_per_epoch or max(source_steps, target_steps)
    return ResolvedStructureDATraining(
        steps_per_epoch=steps,
        quality_warmup_steps=training_config.quality_warmup_steps or steps,
        grl_warmup_steps=training_config.grl_warmup_steps or steps,
    )


def _sample_to_device(sample, device):
    # ``to_cuda`` is the established data path; a CPU equivalent keeps unit and
    # smoke tests possible on hosts without CUDA.
    if torch.device(device).type != "cpu":
        return to_cuda(sample, device)
    return tuple(
        sample[name].to(device) if name in sample else None
        for name in ("pixels", "valid_pixels", "positions", "extra")
    )


def structure_da_train_step(
    model, source_sample, target_sample, optimizer, device, global_step,
    quality_warmup_steps, grl_warmup_steps, grl_gamma, loss_weights,
):
    """Run exactly one joint source/target optimization step."""
    source_labels = source_sample["label"].to(device=device, dtype=torch.long)
    source_tensors = _sample_to_device(source_sample, device)
    target_tensors = _sample_to_device(target_sample, device)
    rho = quality_gate_progress(global_step, quality_warmup_steps)
    lambda_grl = grl_coefficient(global_step, grl_warmup_steps, gamma=grl_gamma)

    optimizer.zero_grad(set_to_none=True)
    source_output = model.forward_details(*source_tensors, quality_progress=rho)
    target_output = model.forward_details(*target_tensors, quality_progress=rho)
    adaptation_output = model.adapt(
        source_output, target_output, grl_coefficient=lambda_grl
    )
    losses = compose_total_loss(
        classification_loss(source_output.component, source_labels),
        quality_domain_loss(source_output.component, target_output.component),
        quality_classification_loss(source_output.component, source_labels),
        component_diversity_loss(source_output.component, source_labels),
        structural_adversarial_loss(adaptation_output),
        weights=loss_weights,
    )
    losses.total.backward()
    optimizer.step()
    return StructureDATrainStepOutput(
        losses=losses,
        quality_progress=rho,
        grl_coefficient=lambda_grl,
        source_batch_size=source_labels.shape[0],
        target_batch_size=target_tensors[0].shape[0],
    )


def train_structure_da(
    model, source_loader, target_loader, val_loader, training_config,
    writer, device, best_model_path,
):
    """Train jointly, validate each epoch, and retain the best checkpoint."""
    resolved = resolve_structure_da_training(
        training_config, source_loader, target_loader
    )
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_config.lr,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.epochs * resolved.steps_per_epoch,
        eta_min=0,
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_f1 = float("-inf")
    names = ("total", "classification", "quality_domain", "quality_classification", "diversity", "structural_adversarial")

    for epoch in range(training_config.epochs):
        model.train()
        meters = {name: AverageMeter() for name in names}
        source_iter, target_iter = cycle(source_loader), cycle(target_loader)
        bar = tqdm(
            range(resolved.steps_per_epoch),
            disable=progress_bar_disabled(training_config.progress_bar),
        )
        for local_step in bar:
            global_step = epoch * resolved.steps_per_epoch + local_step
            result = structure_da_train_step(
                model, next(source_iter), next(target_iter), optimizer, device,
                global_step, resolved.quality_warmup_steps,
                resolved.grl_warmup_steps, training_config.grl_gamma,
                training_config.loss_weights,
            )
            scheduler.step()
            for name in names:
                meters[name].update(
                    getattr(result.losses, name).detach().item(),
                    n=result.source_batch_size,
                )
            lr = optimizer.param_groups[0]["lr"]
            if writer is not None and global_step % training_config.log_step == 0:
                for name in names:
                    tag = {"classification": "cls", "quality_domain": "qdom", "quality_classification": "qcls", "diversity": "div", "structural_adversarial": "sda"}.get(name, name)
                    writer.add_scalar(f"train/loss_{tag}", meters[name].val, global_step)
                writer.add_scalar("train/quality_progress", result.quality_progress, global_step)
                writer.add_scalar("train/grl_coefficient", result.grl_coefficient, global_step)
                writer.add_scalar("train/lr", lr, global_step)
            bar.set_postfix(
                lr=f"{lr:.2e}", total=f"{meters['total'].avg:.3f}",
                cls=f"{meters['classification'].avg:.3f}",
                sda=f"{meters['structural_adversarial'].avg:.3f}",
                rho=f"{result.quality_progress:.2f}",
                grl=f"{result.grl_coefficient:.2f}",
            )
        model.eval()
        best_f1 = validation(
            best_f1, best_model_path, training_config, criterion, device,
            epoch, model, val_loader, writer,
        )
    return best_f1
