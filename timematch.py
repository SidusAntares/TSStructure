from torch.utils.data.sampler import WeightedRandomSampler
import sklearn.metrics
from collections import Counter
from copy import deepcopy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset import PixelSetData
from evaluation import validation
from models.reimts_classifier import (
    PatchOccupancyMeter,
    ReIMTSClassificationOutput,
    reimts_classification_loss,
)
from transforms import (
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
    RandomTemporalShift,
    Identity,
)
from utils.focal_loss import FocalLoss
from utils.train_utils import (
    AverageMeter,
    cycle,
    format_duration,
    format_log_block,
    progress_bar_disabled,
    to_cuda,
)


def format_pseudo_histogram(labels, class_names):
    """Render accepted pseudo-label counts with one class per line."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(class_names))
    return [
        "pseudo class histogram:",
        *[f"  {name}: {int(counts[index])}" for index, name in enumerate(class_names)],
    ]


class PseudoLabelMeter:
    """Aggregate teacher confidence and offline pseudo-label diagnostics."""

    def __init__(self):
        self.confidences = []
        self.pseudo_targets = []
        self.accepted = []
        self.true_targets = []

    def update(self, confidences, pseudo_targets, accepted, true_targets):
        self.confidences.append(confidences.detach().cpu())
        self.pseudo_targets.append(pseudo_targets.detach().cpu())
        self.accepted.append(accepted.detach().bool().cpu())
        self.true_targets.append(torch.as_tensor(true_targets).detach().cpu())

    def summary(self, num_classes):
        confidences = torch.cat(self.confidences).numpy()
        pseudo_targets = torch.cat(self.pseudo_targets).numpy()
        accepted = torch.cat(self.accepted).numpy().astype(bool)
        true_targets = torch.cat(self.true_targets).numpy()
        accepted_count = int(accepted.sum())
        return {
            "seen": int(confidences.size),
            "accepted": accepted_count,
            "confidence_mean_all": (
                float(confidences.mean()) if confidences.size else 0.0
            ),
            "confidence_mean_accepted": (
                float(confidences[accepted].mean()) if accepted_count else 0.0
            ),
            "macro_f1_debug": (
                sklearn.metrics.f1_score(
                    true_targets[accepted],
                    pseudo_targets[accepted],
                    average="macro",
                    zero_division=0,
                )
                if accepted_count
                else 0.0
            ),
            "accepted_labels": pseudo_targets[accepted],
        }


def format_shift_diagnostics(
    estimator,
    shifts,
    scores,
    maximize,
    accuracy_scores,
    sample_batches,
    runtime_seconds,
    selected_index=None,
    spatial_encoder_time=0.0,
    reimts_mtan_time=0.0,
    total_feature_preparation_time=0.0,
    ltae_classifier_total_time=0.0,
):
    """Format score ranking without participating in shift selection."""
    scores = np.asarray(scores)
    accuracy_scores = np.asarray(accuracy_scores)
    ranked = np.argsort(scores)
    if maximize:
        ranked = ranked[::-1]
    if selected_index is not None:
        selected_index = int(selected_index)
        ranked = np.asarray(
            [selected_index, *[index for index in ranked if index != selected_index]]
        )
    best = int(ranked[0])
    second = int(ranked[1]) if len(ranked) > 1 else None
    gap = (
        abs(float(scores[second]) - float(scores[best]))
        if second is not None
        else None
    )
    oracle = int(np.argmax(accuracy_scores))
    lines = [
        f"estimator: {estimator}",
        f"range: [{shifts[0]}, {shifts[-1]}]",
        f"candidate_count: {len(shifts)}",
        f"sample_batches: {sample_batches}",
        "",
        f"selected_shift: {shifts[best]}",
        f"selected_score: {float(scores[best]):.6f}",
        "",
        "second_best_shift: "
        f"{shifts[second] if second is not None else 'unavailable'}",
        "second_best_score: "
        f"{float(scores[second]):.6f}" if second is not None else
        "second_best_score: unavailable",
        f"score_gap: {gap:.6f}" if gap is not None else "score_gap: unavailable",
        "",
        "top5:",
    ]
    lines.extend(
        f"  {rank}. shift={shifts[index]} score={float(scores[index]):.6f}"
        for rank, index in enumerate(ranked[:5], 1)
    )
    lines.extend([
        "",
        "debug oracle:",
        f"  best_accuracy_shift: {shifts[oracle]}",
        f"  best_accuracy: {float(accuracy_scores[oracle]):.6f}",
        "",
        "feature preparation:",
        f"  spatial_encoder_time: {spatial_encoder_time:.6f} s",
        f"  reimts_mtan_time: {reimts_mtan_time:.6f} s",
        "  total_feature_preparation_time: "
        f"{total_feature_preparation_time:.6f} s",
        "",
        "candidate evaluation:",
        f"  candidate_count: {len(shifts)}",
        "  ltae_classifier_total_time: "
        f"{ltae_classifier_total_time:.6f} s",
        "  mean_time_per_candidate: "
        f"{ltae_classifier_total_time / max(1, len(shifts) * sample_batches):.6f} s",
        "",
        "runtime:",
        f"  total_shift_estimation_time: {runtime_seconds:.6f} s",
    ])
    return format_log_block("[SHIFT ESTIMATION]", lines)


def format_timematch_epoch_summary(
    epoch,
    epochs,
    shift,
    pseudo,
    histogram_lines,
    losses,
    timing,
    target_updates,
    patch_lines,
):
    """Build the single structured stdout block for one TimeMatch epoch."""
    candidate_count = shift["max"] - shift["min"] + 1
    acceptance = pseudo["accepted"] / pseudo["seen"] if pseudo["seen"] else 0.0
    lines = [
        "shift:",
        f"  estimator: {shift['estimator']}",
        f"  search_range: [{shift['min']}, {shift['max']}]",
        f"  target_to_source: {shift['target_to_source']}",
        f"  source_to_target: {shift['source_to_target']}",
        f"  candidate_count: {candidate_count}",
        f"  shift_estimation_time: {shift['seconds']:.2f} s",
        "",
        "pseudo labels:",
        f"  seen: {pseudo['seen']}",
        f"  accepted: {pseudo['accepted']}",
        f"  acceptance_rate: {acceptance:.2%}",
        f"  confidence_mean_all: {pseudo['confidence_mean_all']:.4f}",
        "  confidence_mean_accepted: "
        f"{pseudo['confidence_mean_accepted']:.4f}",
        f"  pseudo_macro_f1_debug: {pseudo['macro_f1_debug']:.4f}",
        f"  target_updates: {target_updates}",
        "",
        *histogram_lines,
        "",
        "loss:",
        f"  source: {losses['source']:.6f}",
        f"  target: {losses['target']:.6f}",
        f"  total: {losses['total']:.6f}",
        f"  lr: {losses['lr']:.2e}",
        "",
        "patch occupancy:",
        *[f"  {line}" for line in patch_lines],
        "",
        "timing:",
        f"  training_epoch: {timing['training']:.2f} s",
        f"  validation: {timing['validation']:.2f} s",
        f"  total_epoch: {timing['total']:.2f} s",
        f"  elapsed: {format_duration(timing['elapsed'])}",
    ]
    return format_log_block(f"[TIMEMATCH] Epoch {epoch}/{epochs}", lines)


def _check_temporal_index_range(model, positions, applied_shift, tag):
    if positions.numel() == 0:
        return
    if not hasattr(model, "temporal_encoder"):
        return

    temporal_encoder = model.temporal_encoder
    min_pos = int(positions.min().item())
    max_pos = int(positions.max().item())
    min_idx = min_pos + applied_shift + temporal_encoder.max_temporal_shift
    max_idx = max_pos + applied_shift + temporal_encoder.max_temporal_shift
    table_size = temporal_encoder.positional_enc.num_embeddings

    if min_idx < 0 or max_idx >= table_size:
        raise ValueError(
            f"{tag} temporal indices out of range: "
            f"positions=[{min_pos}, {max_pos}], shift={applied_shift}, "
            f"embedding_indices=[{min_idx}, {max_idx}], table_size={table_size}. "
            "This usually means an extra temporal shift was applied on top of TimeMatch "
            "alignment or the positional encoding range is inconsistent with the dataset dates."
        )


def forward_model_with_shift(
    model, pixels, mask, positions, extra, shift
):
    """Apply the model's global shift capability or baseline date shift."""
    capability = getattr(model, "forward_with_shift", None)
    if callable(capability):
        return capability(pixels, mask, positions, extra, shift)
    return model.forward(pixels, mask, positions + shift, extra)


def forward_model_for_loss_with_shift(
    model, pixels, mask, positions, extra, shift
):
    """Return sample logits plus occupancy diagnostics when supported."""
    capability = getattr(model, "forward_for_loss", None)
    if callable(capability):
        return capability(pixels, mask, positions, extra, shift=shift)
    return forward_model_with_shift(
        model, pixels, mask, positions, extra, shift
    )


def sample_logits_from_output(output):
    if isinstance(output, ReIMTSClassificationOutput):
        return output.sample_logits
    return output


def slice_student_output(output, start, stop):
    if isinstance(output, ReIMTSClassificationOutput):
        return ReIMTSClassificationOutput(
            sample_logits=output.sample_logits[start:stop],
            patch_valid=output.patch_valid[start:stop],
        )
    return output[start:stop]


def student_classification_loss(
    output, targets, criterion, loss_mode="sample"
):
    if isinstance(output, ReIMTSClassificationOutput):
        return reimts_classification_loss(
            output, targets, criterion, mode=loss_mode
        )
    return criterion(output, targets)


def train_timematch(student, config, writer, val_loader, device, best_model_path, fold_num, splits):
    training_run_started = time.perf_counter()
    source_loader, target_loader_no_aug, target_loader = get_data_loaders(splits, config, config.balance_source)

    # Setup model
    pretrained_path = f"{config.weights}/fold_{fold_num}"
    pretrained_weights = torch.load(f"{pretrained_path}/model.pt", weights_only=False)["state_dict"]
    student.load_state_dict(pretrained_weights)
    teacher = deepcopy(student)
    student.to(device)
    teacher.to(device)

    # Training setup
    global_step, best_f1 = 0, 0
    if config.use_focal_loss:
        criterion = FocalLoss(gamma=config.focal_loss_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    reimts_loss_mode = getattr(config, "reimts_loss_mode", "sample")
    patch_diagnostics = getattr(config, "reimts_patch_diagnostics", False)

    steps_per_epoch = config.steps_per_epoch

    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0)

    source_iter = iter(cycle(source_loader))
    target_iter = iter(cycle(target_loader))
    min_shift, max_shift = -config.max_temporal_shift, config.max_temporal_shift
    target_to_source_shift = 0

    # To evaluate how well we estimate class distribution
    target_labels = target_loader_no_aug.dataset.get_labels()
    actual_class_distr = estimate_class_distribution(target_labels, config.num_classes)

    # estimate an initial guess for shift using Inception Score
    if config.estimate_shift:
        shift_estimator = 'IS' if config.shift_estimator == 'AM' else config.shift_estimator
        target_to_source_shift = estimate_temporal_shift(
            teacher,
            target_loader_no_aug,
            device,
            min_shift=min_shift,
            max_shift=max_shift,
            sample_size=config.sample_size,
            shift_estimator=shift_estimator,
            progress_bar=getattr(config, "progress_bar", "auto"),
            diagnostics=patch_diagnostics,
        )
        if target_to_source_shift >= 0:
            min_shift = 0
        else:
            max_shift = 0

        # Use estimated shift to get initial pseudo labels
        pseudo_softmaxes = get_pseudo_labels(
            teacher,
            target_loader_no_aug,
            device,
            target_to_source_shift,
            n=None,
            progress_bar=getattr(config, "progress_bar", "auto"),
        )
        all_pseudo_labels = torch.max(pseudo_softmaxes, dim=1)[1]

    source_to_target_shift = 0
    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()
        shift_seconds = 0.0
        shift_search_min, shift_search_max = min_shift, max_shift
        progress_bar = tqdm(
            range(steps_per_epoch),
            desc=f"TimeMatch Epoch {epoch + 1}/{config.epochs}",
            disable=progress_bar_disabled(
                getattr(config, "progress_bar", "auto")
            ),
        )
        source_loss_meter = AverageMeter()
        target_loss_meter = AverageMeter()
        total_loss_meter = AverageMeter()
        pseudo_meter = PseudoLabelMeter()
        source_patch_meter = PatchOccupancyMeter()
        target_weak_patch_meter = PatchOccupancyMeter()
        target_updates = 0

        if config.estimate_shift:
            estimated_class_distr = estimate_class_distribution(all_pseudo_labels, config.num_classes)
            writer.add_scalar("train/kl_d", kl_divergence(actual_class_distr, estimated_class_distr), epoch)
            shift_started = time.perf_counter()
            target_to_source_shift = estimate_temporal_shift(teacher,
                    target_loader_no_aug, device, estimated_class_distr,
                    min_shift=min_shift, max_shift=max_shift, sample_size=config.sample_size,
                    shift_estimator=config.shift_estimator,
                    progress_bar=getattr(config, "progress_bar", "auto"),
                    diagnostics=patch_diagnostics)
            shift_seconds = time.perf_counter() - shift_started
            if epoch == 0:
                if config.shift_source:
                    source_to_target_shift = -target_to_source_shift
                else:
                    source_to_target_shift = 0
                min_shift, max_shift = min(target_to_source_shift, 0), max(0, target_to_source_shift)
            writer.add_scalar("train/temporal_shift", target_to_source_shift, epoch)

        student.train()
        teacher.eval()  # don't update BN or use dropout for teacher

        training_epoch_started = time.perf_counter()
        for step in progress_bar:
            sample_source, (sample_target_weak, sample_target_strong) = next(source_iter), next(target_iter)

            # Get pseudo labels from teacher
            pixels_t_weak, mask_t_weak, position_t_weak, extra_t_weak = to_cuda(sample_target_weak, device)
            with torch.no_grad():
                if step == 0:
                    teacher_output = forward_model_for_loss_with_shift(
                        teacher,
                        pixels_t_weak,
                        mask_t_weak,
                        position_t_weak,
                        extra_t_weak,
                        target_to_source_shift,
                    )
                    if isinstance(teacher_output, ReIMTSClassificationOutput):
                        target_weak_patch_meter.update(teacher_output.patch_valid)
                    teacher_logits = sample_logits_from_output(teacher_output)
                else:
                    teacher_logits = forward_model_with_shift(
                        teacher, pixels_t_weak, mask_t_weak,
                        position_t_weak, extra_t_weak,
                        target_to_source_shift,
                    )
                teacher_preds = F.softmax(teacher_logits, dim=1)
            pseudo_conf, pseudo_targets = torch.max(teacher_preds, dim=1)
            pseudo_mask = pseudo_conf > config.pseudo_threshold
            num_pseudo = int(pseudo_mask.sum().item())
            target_updates += num_pseudo
            pseudo_meter.update(
                pseudo_conf,
                pseudo_targets,
                pseudo_mask,
                sample_target_weak['label'],
            )

            # Update student on shifted source data and pseudo-labeled target data
            pixels_s, mask_s, position_s, extra_s = to_cuda(sample_source, device)
            source_labels = sample_source['label'].cuda(device, non_blocking=True)
            pixels_t, mask_t, position_t, extra_t = to_cuda(sample_target_strong, device)
            output_target = None
            loss_target = 0.0
            if config.domain_specific_bn:
                _check_temporal_index_range(student, position_s, source_to_target_shift, "source")
                output_source = forward_model_for_loss_with_shift(
                    student,
                    pixels_s,
                    mask_s,
                    position_s,
                    extra_s,
                    source_to_target_shift,
                )
                if num_pseudo >= 2:  # at least 2 examples required for BN
                    _check_temporal_index_range(student, position_t[pseudo_mask], 0, "target")
                    output_target = forward_model_for_loss_with_shift(
                        student,
                        pixels_t[pseudo_mask],
                        mask_t[pseudo_mask],
                        position_t[pseudo_mask],
                        extra_t[pseudo_mask],
                        0,
                    )
            else:
                _check_temporal_index_range(student, position_s, source_to_target_shift, "source")
                if num_pseudo > 0:
                    selected_pixels_t = pixels_t[pseudo_mask]
                    selected_mask_t = mask_t[pseudo_mask]
                    selected_position_t = position_t[pseudo_mask]
                    selected_extra_t = extra_t[pseudo_mask]

                    _check_temporal_index_range(student, selected_position_t, 0, "target")

                    pixels = torch.cat([pixels_s, selected_pixels_t], dim=0)
                    mask = torch.cat([mask_s, selected_mask_t], dim=0)
                    position = torch.cat([position_s, selected_position_t], dim=0)
                    shift = torch.cat(
                        [
                            torch.full_like(position_s, source_to_target_shift),
                            torch.zeros_like(selected_position_t),
                        ],
                        dim=0,
                    )
                    extra = torch.cat([extra_s, selected_extra_t], dim=0)

                    output = forward_model_for_loss_with_shift(
                        student, pixels, mask, position, extra, shift
                    )
                    source_batch_size = pixels_s.shape[0]
                    output_source = slice_student_output(
                        output, 0, source_batch_size
                    )
                    output_target = slice_student_output(
                        output, source_batch_size, None
                    )
                else:
                    output_source = forward_model_for_loss_with_shift(
                        student,
                        pixels_s,
                        mask_s,
                        position_s,
                        extra_s,
                        source_to_target_shift,
                    )
                    output_target = None

            loss_source = student_classification_loss(
                output_source,
                source_labels,
                criterion,
                loss_mode=reimts_loss_mode,
            )
            if isinstance(output_source, ReIMTSClassificationOutput):
                source_patch_meter.update(output_source.patch_valid)
            if output_target is not None:
                loss_target = student_classification_loss(
                    output_target,
                    pseudo_targets[pseudo_mask],
                    criterion,
                    loss_mode=reimts_loss_mode,
                )
            loss = loss_source + config.trade_off * loss_target

            # compute loss and backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            update_ema_variables(student, teacher, config.ema_decay)

            # Metrics
            source_loss_meter.update(loss_source.item())
            target_loss_meter.update(
                loss_target.item() if torch.is_tensor(loss_target) else float(loss_target)
            )
            total_loss_meter.update(loss.item())
            progress_bar.set_postfix(loss=f"{total_loss_meter.avg:.3f}")

            if step % config.log_step == 0:
                writer.add_scalar("train/loss", total_loss_meter.val, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/target_updates", len(torch.nonzero(pseudo_mask)), global_step)

            global_step += 1

        progress_bar.close()
        training_seconds = time.perf_counter() - training_epoch_started

        # Evaluate pseudo labels
        pseudo_summary = pseudo_meter.summary(config.num_classes)
        pseudo_count = pseudo_summary["accepted"]
        conf_pseudo_f1 = pseudo_summary["macro_f1_debug"]
        all_pseudo_labels = torch.cat(pseudo_meter.pseudo_targets).numpy()
        writer.add_scalar("train/pseudo_f1", conf_pseudo_f1, epoch)
        writer.add_scalar("train/pseudo_count", pseudo_count, epoch)

        validation_seconds = 0.0
        if config.run_validation:
            validation_started = time.perf_counter()
            if config.output_student:
                student.eval()
                best_f1 = validation(best_f1, None, config, criterion, device, epoch, student, val_loader, writer)
            else:
                teacher.eval()
                best_f1 = validation(best_f1, None, config, criterion, device, epoch, teacher, val_loader, writer)
            validation_seconds = time.perf_counter() - validation_started

        patch_lines = [
            *source_patch_meter.format_lines("source strong:"),
            *target_weak_patch_meter.format_lines("target weak:"),
        ]
        print(format_timematch_epoch_summary(
            epoch=epoch + 1,
            epochs=config.epochs,
            shift={
                "estimator": getattr(config, "shift_estimator", "AM"),
                "min": shift_search_min,
                "max": shift_search_max,
                "target_to_source": target_to_source_shift,
                "source_to_target": source_to_target_shift,
                "seconds": shift_seconds,
            },
            pseudo=pseudo_summary,
            histogram_lines=format_pseudo_histogram(
                pseudo_summary["accepted_labels"], config.classes
            ),
            losses={
                "source": source_loss_meter.avg,
                "target": target_loss_meter.avg,
                "total": total_loss_meter.avg,
                "lr": optimizer.param_groups[0]["lr"],
            },
            timing={
                "training": training_seconds,
                "validation": validation_seconds,
                "total": time.perf_counter() - epoch_started,
                "elapsed": time.perf_counter() - training_run_started,
            },
            target_updates=target_updates,
            patch_lines=patch_lines,
        ))

    # Save model final model 
    if config.output_student:
        torch.save({'state_dict': student.state_dict()}, best_model_path)
    else:
        torch.save({'state_dict': teacher.state_dict()}, best_model_path)
    return best_f1

def estimate_class_distribution(labels, num_classes):
    return np.bincount(labels, minlength=num_classes) / len(labels)

def kl_divergence(actual, estimated):
    return np.sum(actual * (np.log(actual + 1e-5) - np.log(estimated + 1e-5)))

@torch.no_grad()
def update_ema_variables(model, ema, decay=0.99):
    for ema_v, model_v in zip(ema.state_dict().values(), model.state_dict().values()):
        ema_v.copy_(decay * ema_v + (1. - decay) * model_v)


def get_data_loaders(splits, config, balance_source=True):
    weak_aug = transforms.Compose([
        RandomSamplePixels(config.num_pixels),
        Normalize(),
        ToTensor(),
    ])

    strong_aug = transforms.Compose([
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            RandomTemporalShift(
                max_shift=config.max_shift_aug,
                p=config.shift_aug_p,
            ) if config.with_shift_aug else Identity(),
            Normalize(),
            ToTensor(),
    ])

    source_dataset = PixelSetData(config.data_root, config.source,
            config.classes, strong_aug,
            indices=splits[config.source]['train'],
            closed_set=config.closed_set,
            combine_spring_and_winter=config.combine_spring_and_winter,)

    if balance_source:
        source_labels = source_dataset.get_labels()
        freq = Counter(source_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_labels]
        sampler = WeightedRandomSampler(source_weights, len(source_labels))
        print("using balanced loader for source")
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=config.batch_size,
            drop_last=True,
        )
    else:
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )

    target_dataset = PixelSetData(config.data_root, config.target,
            config.classes, None,
            indices=splits[config.target]['train'],
            closed_set=config.closed_set,
            combine_spring_and_winter=config.combine_spring_and_winter)

    strong_dataset = deepcopy(target_dataset)
    strong_dataset.transform = strong_aug
    weak_dataset = deepcopy(target_dataset)
    weak_dataset.transform = weak_aug
    target_dataset_weak_strong = TupleDataset(weak_dataset, strong_dataset)

    no_aug_dataset = deepcopy(target_dataset)
    no_aug_dataset.transform = weak_aug
    # For shift estimation
    target_loader_no_aug = data.DataLoader(
        no_aug_dataset,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # For mean teacher training
    target_loader_weak_strong = data.DataLoader(
        target_dataset_weak_strong,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    print(f'size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)')
    print(f'size of target dataset: {len(target_dataset)} ({len(target_loader_weak_strong)} batches)')

    return source_loader, target_loader_no_aug, target_loader_weak_strong


class TupleDataset(data.Dataset):
    def __init__(self, dataset1, dataset2):
        super().__init__()
        self.weak = dataset1
        self.strong = dataset2
        assert len(dataset1) == len(dataset2)
        self.len = len(dataset1)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.weak[index], self.strong[index])


@torch.no_grad()
def estimate_temporal_shift(
    model,
    target_loader,
    device,
    class_distribution=None,
    min_shift=-60,
    max_shift=60,
    sample_size=100,
    shift_estimator='IS',
    progress_bar='auto',
    diagnostics=False,
):
    estimation_started = time.perf_counter()
    shifts = list(range(min_shift, max_shift + 1))
    model.eval()
    if sample_size is None:
        sample_size = len(target_loader)

    target_iter = iter(target_loader)
    shift_softmaxes, labels = [], []
    spatial_encoder_time = 0.0
    reimts_mtan_time = 0.0
    total_feature_preparation_time = 0.0
    ltae_classifier_total_time = 0.0
    for _ in tqdm(
        range(sample_size),
        desc=f'Estimating shift between [{min_shift}, {max_shift}]',
        disable=progress_bar_disabled(progress_bar),
    ):
        sample = next(target_iter)
        labels.extend(sample['label'].tolist())
        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        prepare_capability = getattr(model, "prepare_shift_features", None)
        evaluate_capability = getattr(
            model, "forward_from_shift_features", None
        )
        if callable(prepare_capability) and callable(evaluate_capability):
            features = prepare_capability(
                pixels, valid_pixels, positions, extra
            )
            spatial_encoder_time += float(
                getattr(features, "spatial_encoder_time", 0.0)
            )
            reimts_mtan_time += float(
                getattr(features, "reimts_mtan_time", 0.0)
            )
            total_feature_preparation_time += float(
                getattr(
                    features,
                    "total_feature_preparation_time",
                    getattr(features, "spatial_encoder_time", 0.0)
                    + getattr(features, "reimts_mtan_time", 0.0),
                )
            )
            if features.tokens.is_cuda:
                torch.cuda.synchronize(features.tokens.device)
            candidate_started = time.perf_counter()
            shift_logits = torch.stack(
                [
                    evaluate_capability(features, shift)
                    for shift in shifts
                ],
                dim=1,
            )
            if shift_logits.is_cuda:
                torch.cuda.synchronize(shift_logits.device)
            ltae_classifier_total_time += (
                time.perf_counter() - candidate_started
            )
        else:
            if pixels.is_cuda:
                torch.cuda.synchronize(pixels.device)
            spatial_started = time.perf_counter()
            spatial_feats = model.spatial_encoder.forward(
                pixels, valid_pixels, extra
            )
            if spatial_feats.is_cuda:
                torch.cuda.synchronize(spatial_feats.device)
            spatial_seconds = time.perf_counter() - spatial_started
            spatial_encoder_time += spatial_seconds
            total_feature_preparation_time += spatial_seconds
            candidate_started = time.perf_counter()
            shift_logits = torch.stack(
                [
                    model.decoder(
                        model.temporal_encoder(spatial_feats, positions + shift)
                    )
                    for shift in shifts
                ],
                dim=1,
            )
            if shift_logits.is_cuda:
                torch.cuda.synchronize(shift_logits.device)
            ltae_classifier_total_time += (
                time.perf_counter() - candidate_started
            )
        shift_probs = F.softmax(shift_logits, dim=2)
        shift_softmaxes.append(shift_probs)
    shift_softmaxes = torch.cat(shift_softmaxes).cpu().numpy()  # (N, n_shifts, n_classes)
    labels = np.array(labels)
    shift_predictions = np.argmax(shift_softmaxes, axis=2)  # (N, n_shifts)

    # shift_f1_scores = [f1_score(labels, shift_predictions, num_classes) for shift_predictions in all_shift_predictions]
    shift_acc_scores = [(labels == predictions).mean() for predictions in np.moveaxis(shift_predictions, 0, 1)]
    p_yx = shift_softmaxes # (N, n_shifts, n_classes)
    p_y = shift_softmaxes.mean(axis=0)  # (n_shifts, n_classes)


    if shift_estimator == 'IS':
        scores = np.mean(np.sum(p_yx * (np.log(p_yx + 1e-5) - np.log(p_y[np.newaxis] + 1e-5)), axis=2), axis=0)  # (n_shifts)
        maximize = True
        best_shift_idx = int(np.argsort(scores)[::-1][0])

    elif shift_estimator == 'ENT':
        scores = -np.mean(np.sum(p_yx * np.log(p_yx + 1e-5), axis=2), axis=0)  # (n_shifts)
        maximize = False
        best_shift_idx = int(np.argsort(scores)[0])

    elif shift_estimator == 'AM':
        assert class_distribution is not None, 'Target class distribution required to compute AM score'

        # estimate class distribution
        one_hot_p_y = np.zeros_like(p_y)
        for i in range(len(shifts)):
            one_hot = np.zeros((shift_softmaxes.shape[0], shift_softmaxes.shape[-1]))  # (n, classes)
            one_hot[np.arange(one_hot.shape[0]), shift_predictions[:, i]] = 1
            one_hot_p_y[i] = one_hot.mean(axis=0)

        c_train = class_distribution
        # kl_d = np.sum(c_train * (np.log(c_train + 1e-5) - np.log(p_y + 1e-5)), axis=1) # soft class distr
        kl_d = np.sum(c_train * (np.log(c_train + 1e-5) - np.log(one_hot_p_y + 1e-5)), axis=1)
        entropy = np.mean(np.sum(-p_yx * np.log(p_yx + 1e-5), axis=2), axis=0)
        scores = kl_d + entropy
        maximize = False
        best_shift_idx = int(np.argsort(scores)[0])
    elif shift_estimator == 'ACC':  # for upperbound comparison
        scores = np.asarray(shift_acc_scores)
        maximize = True
        best_shift_idx = int(np.argmax(scores))
    else:
        raise NotImplementedError

    best_shift = shifts[best_shift_idx]
    print(format_shift_diagnostics(
        estimator=shift_estimator,
        shifts=shifts,
        scores=scores,
        maximize=maximize,
        accuracy_scores=shift_acc_scores,
        sample_batches=sample_size,
        runtime_seconds=time.perf_counter() - estimation_started,
        selected_index=best_shift_idx,
        spatial_encoder_time=spatial_encoder_time,
        reimts_mtan_time=reimts_mtan_time,
        total_feature_preparation_time=total_feature_preparation_time,
        ltae_classifier_total_time=ltae_classifier_total_time,
    ))
    return best_shift




@torch.no_grad()
def get_pseudo_labels(
    model,
    data_loader,
    device,
    best_shift,
    n=500,
    progress_bar='auto',
):
    model.eval()
    pseudo_softmaxes = []
    indices = []
    for i, sample in enumerate(
        tqdm(
            data_loader,
            desc="Computing pseudo labels",
            disable=progress_bar_disabled(progress_bar),
        )
    ):
        if n is not None and i == n:
            break
        indices.extend(sample["index"].tolist())

        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        logits = forward_model_with_shift(
            model,
            pixels,
            valid_pixels,
            positions,
            extra,
            best_shift,
        )
        probs = F.softmax(logits, dim=1).cpu()
        pseudo_softmaxes.extend(probs.tolist())

    indices = torch.as_tensor(indices)
    pseudo_softmaxes = torch.as_tensor(pseudo_softmaxes)
    pseudo_softmaxes = pseudo_softmaxes[torch.argsort(indices)]

    return pseudo_softmaxes
