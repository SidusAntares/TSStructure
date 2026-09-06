from torch.utils.data.sampler import WeightedRandomSampler
import sklearn.metrics
from collections import Counter
from copy import deepcopy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset import PixelSetData
from evaluation import validation
from models.fredn.diagnostics import log_fredn_diagnostics
from models.shape_alignment import (
    ShapeAlignment,
    SourceShapeReferenceBank,
    preserve_rng_state,
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
    progress_bar_disabled,
    to_cuda,
)


def _check_temporal_index_range(model, positions, applied_shift, tag):
    if positions.numel() == 0:
        return

    min_pos = int(positions.min().item())
    max_pos = int(positions.max().item())
    if hasattr(model, "get_temporal_encoders"):
        temporal_encoders = model.get_temporal_encoders()
    else:
        temporal_encoders = (model.temporal_encoder,)
    for encoder_index, temporal_encoder in enumerate(temporal_encoders):
        min_idx = min_pos + applied_shift + temporal_encoder.max_temporal_shift
        max_idx = max_pos + applied_shift + temporal_encoder.max_temporal_shift
        table_size = temporal_encoder.positional_enc.num_embeddings

        if min_idx < 0 or max_idx >= table_size:
            raise ValueError(
                f"{tag} temporal indices out of range: encoder={encoder_index}, "
                f"positions=[{min_pos}, {max_pos}], shift={applied_shift}, "
                f"embedding_indices=[{min_idx}, {max_idx}], table_size={table_size}. "
                "This usually means an extra temporal shift was applied on top of TimeMatch "
                "alignment or the positional encoding range is inconsistent with the dataset dates."
            )


def _forward_with_temporal_shift(
    model,
    pixels,
    mask,
    positions,
    extra,
    temporal_shift=0,
    collect_diagnostics=False,
):
    if hasattr(model, "forward_with_temporal_shift"):
        if getattr(model, "supports_fredn_diagnostics", False):
            return model.forward_with_temporal_shift(
                pixels,
                mask,
                positions,
                extra,
                temporal_shift=temporal_shift,
                collect_diagnostics=collect_diagnostics,
            )
        return model.forward_with_temporal_shift(
            pixels,
            mask,
            positions,
            extra,
            temporal_shift=temporal_shift,
        )
    return model.forward(pixels, mask, positions + temporal_shift, extra)


def _forward_with_spatial_capture(
    model,
    pixels,
    mask,
    positions,
    extra,
    temporal_shift=0,
    collect_diagnostics=False,
):
    """Capture the spatial tensor produced by this exact semantic forward."""
    captured = []

    def capture_spatial(_module, _inputs, output):
        captured.append(output)

    handle = model.spatial_encoder.register_forward_hook(capture_spatial)
    try:
        logits = _forward_with_temporal_shift(
            model,
            pixels,
            mask,
            positions,
            extra,
            temporal_shift=temporal_shift,
            collect_diagnostics=collect_diagnostics,
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            "shape alignment expected exactly one spatial encoder call in the "
            f"semantic forward, captured {len(captured)}"
        )
    return logits, captured[0]


def _shape_loss_from_capture(
    shape_alignment,
    captured_selected,
    target_positions,
    pseudo_targets,
    pseudo_mask,
    target_to_source_shift,
):
    """Apply the sole pseudo gate and express accepted targets in source time."""
    selected_positions = target_positions[pseudo_mask]
    selected_classes = pseudo_targets[pseudo_mask]
    if captured_selected.shape[0] != selected_positions.shape[0]:
        raise ValueError("captured target features do not match the pseudo mask")
    return shape_alignment(
        captured_selected,
        selected_positions + target_to_source_shift,
        selected_classes,
    )


def _add_shape_loss(timematch_loss, shape_result, shape_lambda):
    if shape_result is None:
        return timematch_loss
    return timematch_loss + shape_lambda * shape_result.loss


def _prepare_temporal_features(
    model,
    spatial_feats,
    positions,
    collect_diagnostics=False,
):
    if hasattr(model, "prepare_temporal_features"):
        if getattr(model, "supports_fredn_diagnostics", False):
            return model.prepare_temporal_features(
                spatial_feats,
                positions,
                collect_diagnostics=collect_diagnostics,
            )
        return model.prepare_temporal_features(spatial_feats, positions)
    return spatial_feats


def _classify_prepared(model, prepared, positions, temporal_shift=0):
    if hasattr(model, "classify_prepared"):
        return model.classify_prepared(
            prepared,
            positions,
            temporal_shift=temporal_shift,
        )
    return model.decoder(
        model.temporal_encoder(prepared, positions + temporal_shift)
    )


def _build_source_shape_alignment(student, config, splits, device, checkpoint_path):
    """Build and persist a frozen source-train-only shape reference."""
    if getattr(config, "model", "pseltae") != "pseltae":
        raise ValueError("shape alignment is training-only support for model=pseltae")
    weak_transform = transforms.Compose(
        [RandomSamplePixels(config.num_pixels), Normalize(), ToTensor()]
    )
    with preserve_rng_state(seed=config.shape_reference_seed):
        candidates = PixelSetData(
            config.data_root,
            config.source,
            config.classes,
            transform=None,
            indices=splits[config.source]["train"],
            with_extra=config.with_extra,
            closed_set=config.closed_set,
            combine_spring_and_winter=config.combine_spring_and_winter,
        )
        labels = candidates.get_labels()
        parcel_indices = candidates.get_parcel_indices()
        rng = np.random.default_rng(config.shape_reference_seed)
        selected_parcels = []
        for class_id in range(config.num_classes):
            class_parcels = parcel_indices[labels == class_id].copy()
            if len(class_parcels) == 0:
                raise ValueError(
                    f"source train split has no shape reference for class {class_id}"
                )
            rng.shuffle(class_parcels)
            selected_parcels.extend(
                class_parcels[: config.shape_reference_per_class].tolist()
            )
        reference_dataset = PixelSetData(
            config.data_root,
            config.source,
            config.classes,
            transform=weak_transform,
            indices=selected_parcels,
            with_extra=config.with_extra,
            closed_set=config.closed_set,
            combine_spring_and_winter=config.combine_spring_and_winter,
        )
        reference_loader = data.DataLoader(
            reference_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        was_training = student.training
        student.eval()
        features, positions, reference_labels = [], [], []
        with torch.no_grad():
            for sample in reference_loader:
                pixels, valid, timestamps, extra = to_cuda(sample, device)
                features.append(student.spatial_encoder(pixels, valid, extra))
                positions.append(timestamps)
                reference_labels.append(
                    sample["label"].to(device, non_blocking=True)
                )
        student.train(was_training)
        bank = SourceShapeReferenceBank.from_source_features(
            torch.cat(features),
            torch.cat(positions),
            torch.cat(reference_labels),
            modes=config.shape_modes,
            grid_points=config.shape_grid_points,
            period_days=config.shape_fourier_period_days,
            reg=config.shape_fourier_reg,
            prominence_rel=config.shape_prominence_rel,
            min_distance_days=config.shape_min_distance_days,
        ).to(device)

    os.makedirs(config.fold_dir, exist_ok=True)
    torch.save(
        bank.export_payload(), os.path.join(config.fold_dir, "shape_reference.pt")
    )
    manifest = bank.manifest(
        config.source,
        checkpoint_path,
        config.classes,
        config.shape_reference_per_class,
    )
    with open(
        os.path.join(config.fold_dir, "shape_reference_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)
    return ShapeAlignment(
        bank,
        morph_weight=config.shape_morph_weight,
        event_weight=config.shape_event_weight,
    ).to(device)


def _log_shape_result(writer, result, timematch_loss, config, pseudo_mask, step):
    weighted = config.shape_lambda * result.loss.detach()
    denominator = timematch_loss.detach().abs().clamp_min(1e-12)
    values = {
        "shape/raw_loss": result.loss.detach(),
        "shape/weighted_loss": weighted,
        "shape/morph_loss": result.morph_loss.detach(),
        "shape/event_loss": result.event_loss.detach(),
        "shape/selected_target_count": pseudo_mask.sum().detach(),
        "shape/selected_target_rate": pseudo_mask.float().mean().detach(),
        "shape/morph_corr_mean": result.morph_corr_mean,
        "shape/event_amp_abs_gap": result.event_amp_abs_gap,
        "shape/loss_ratio": weighted / denominator,
        "shape/no_event_reference_count": result.no_event_reference_count,
    }
    for mode, loss in result.mode_losses.items():
        values[f"shape/mode{mode}_loss"] = loss.detach()
    for name, value in values.items():
        writer.add_scalar(name, value, step)


@torch.no_grad()
def _collect_shape_epoch_diagnostics(
    teacher,
    shape_alignment,
    target_loader,
    device,
    target_to_source_shift,
    pseudo_threshold,
    max_batches,
):
    totals = {}
    selected_total = 0
    if max_batches <= 0:
        return totals
    with preserve_rng_state():
        for batch_index, sample in enumerate(target_loader):
            if batch_index >= max_batches:
                break
            pixels, valid, positions, extra = to_cuda(sample, device)
            logits, spatial = _forward_with_spatial_capture(
                teacher,
                pixels,
                valid,
                positions,
                extra,
                temporal_shift=target_to_source_shift,
            )
            probabilities = F.softmax(logits, dim=1)
            confidence, pseudo_classes = probabilities.max(dim=1)
            selected = confidence > pseudo_threshold
            count = int(selected.sum().item())
            if count == 0:
                continue
            batch_metrics = shape_alignment.diagnostics(
                spatial[selected],
                positions[selected] + target_to_source_shift,
                pseudo_classes[selected],
            )
            selected_total += count
            for name, value in batch_metrics.items():
                totals[name] = totals.get(name, value.new_zeros(())) + value * count
    if selected_total:
        totals = {name: value / selected_total for name, value in totals.items()}
    return totals


def train_timematch(student, config, writer, val_loader, device, best_model_path, fold_num, splits):
    source_loader, target_loader_no_aug, target_loader = get_data_loaders(splits, config, config.balance_source)

    # Setup model
    pretrained_path = f"{config.weights}/fold_{fold_num}"
    pretrained_weights = torch.load(f"{pretrained_path}/model.pt", weights_only=False)["state_dict"]
    student.load_state_dict(pretrained_weights)
    teacher = deepcopy(student)
    student.to(device)
    teacher.to(device)
    shape_alignment = None
    if getattr(config, "shape_align", False):
        shape_alignment = _build_source_shape_alignment(
            student,
            config,
            splits,
            device,
            os.path.join(pretrained_path, "model.pt"),
        )

    # Training setup
    global_step, best_f1 = 0, 0
    if config.use_focal_loss:
        criterion = FocalLoss(gamma=config.focal_loss_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()

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
        progress_bar = tqdm(
            range(steps_per_epoch),
            desc=f"TimeMatch Epoch {epoch + 1}/{config.epochs}",
            disable=progress_bar_disabled(
                getattr(config, "progress_bar", "auto")
            ),
        )
        loss_meter = AverageMeter()

        if config.estimate_shift:
            estimated_class_distr = estimate_class_distribution(all_pseudo_labels, config.num_classes)
            writer.add_scalar("train/kl_d", kl_divergence(actual_class_distr, estimated_class_distr), epoch)
            target_to_source_shift = estimate_temporal_shift(teacher,
                    target_loader_no_aug, device, estimated_class_distr,
                    min_shift=min_shift, max_shift=max_shift, sample_size=config.sample_size,
                    shift_estimator=config.shift_estimator,
                    progress_bar=getattr(config, "progress_bar", "auto"))
            if epoch == 0:
                if config.shift_source:
                    source_to_target_shift = -target_to_source_shift
                else:
                    source_to_target_shift = 0
                min_shift, max_shift = min(target_to_source_shift, 0), max(0, target_to_source_shift)
            writer.add_scalar("train/temporal_shift", target_to_source_shift, epoch)

        student.train()
        teacher.eval()  # don't update BN or use dropout for teacher

        all_labels, all_pseudo_labels, all_pseudo_mask = [], [], []
        for step in progress_bar:
            collect_diagnostics = global_step % config.log_step == 0
            sample_source, (sample_target_weak, sample_target_strong) = next(source_iter), next(target_iter)

            # Get pseudo labels from teacher
            pixels_t_weak, mask_t_weak, position_t_weak, extra_t_weak = to_cuda(sample_target_weak, device)
            with torch.no_grad():
                teacher_preds = F.softmax(
                    _forward_with_temporal_shift(
                        teacher,
                        pixels_t_weak,
                        mask_t_weak,
                        position_t_weak,
                        extra_t_weak,
                        temporal_shift=target_to_source_shift,
                    ),
                    dim=1,
                )
            pseudo_conf, pseudo_targets = torch.max(teacher_preds, dim=1)
            pseudo_mask = pseudo_conf > config.pseudo_threshold
            num_pseudo = int(pseudo_mask.sum().item())

            # Update student on shifted source data and pseudo-labeled target data
            pixels_s, mask_s, position_s, extra_s = to_cuda(sample_source, device)
            source_labels = sample_source['label'].cuda(device, non_blocking=True)
            pixels_t, mask_t, position_t, extra_t = to_cuda(sample_target_strong, device)
            logits_target = None
            captured_target = None
            loss_target = 0.0
            if config.domain_specific_bn:
                _check_temporal_index_range(student, position_s, source_to_target_shift, "source")
                logits_source = _forward_with_temporal_shift(
                    student,
                    pixels_s,
                    mask_s,
                    position_s,
                    extra_s,
                    temporal_shift=source_to_target_shift,
                    collect_diagnostics=(
                        collect_diagnostics and num_pseudo < 2
                    ),
                )
                if num_pseudo >= 2:  # at least 2 examples required for BN
                    _check_temporal_index_range(student, position_t[pseudo_mask], 0, "target")
                    target_args = (
                        student,
                        pixels_t[pseudo_mask],
                        mask_t[pseudo_mask],
                        position_t[pseudo_mask],
                        extra_t[pseudo_mask],
                    )
                    if shape_alignment is None:
                        logits_target = _forward_with_temporal_shift(
                            *target_args,
                            collect_diagnostics=collect_diagnostics,
                        )
                    else:
                        logits_target, captured_target = _forward_with_spatial_capture(
                            *target_args,
                            collect_diagnostics=collect_diagnostics,
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
                    position = torch.cat(
                        [position_s, selected_position_t],
                        dim=0,
                    )
                    temporal_shift = torch.cat(
                        [
                            torch.full(
                                (position_s.shape[0], 1),
                                source_to_target_shift,
                                device=position_s.device,
                                dtype=position_s.dtype,
                            ),
                            torch.zeros(
                                (selected_position_t.shape[0], 1),
                                device=selected_position_t.device,
                                dtype=selected_position_t.dtype,
                            ),
                        ],
                        dim=0,
                    )
                    extra = torch.cat([extra_s, selected_extra_t], dim=0)

                    concat_args = (student, pixels, mask, position, extra)
                    if shape_alignment is None:
                        logits = _forward_with_temporal_shift(
                            *concat_args,
                            temporal_shift=temporal_shift,
                            collect_diagnostics=collect_diagnostics,
                        )
                        captured = None
                    else:
                        logits, captured = _forward_with_spatial_capture(
                            *concat_args,
                            temporal_shift=temporal_shift,
                            collect_diagnostics=collect_diagnostics,
                        )
                    source_batch_size = pixels_s.shape[0]
                    logits_source = logits[:source_batch_size]
                    logits_target = logits[source_batch_size:]
                    if captured is not None:
                        captured_target = captured[source_batch_size:]
                else:
                    logits_source = _forward_with_temporal_shift(
                        student,
                        pixels_s,
                        mask_s,
                        position_s,
                        extra_s,
                        temporal_shift=source_to_target_shift,
                        collect_diagnostics=collect_diagnostics,
                    )
                    logits_target = None

            loss_source = criterion(logits_source, source_labels)
            if logits_target is not None:
                loss_target = criterion(logits_target, pseudo_targets[pseudo_mask])
            timematch_loss = loss_source + config.trade_off * loss_target
            shape_result = None
            if shape_alignment is not None and captured_target is not None:
                shape_result = _shape_loss_from_capture(
                    shape_alignment,
                    captured_target,
                    position_t,
                    pseudo_targets,
                    pseudo_mask,
                    target_to_source_shift,
                )
            loss = _add_shape_loss(
                timematch_loss,
                shape_result,
                getattr(config, "shape_lambda", 0.1),
            )

            # compute loss and backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            update_ema_variables(student, teacher, config.ema_decay)

            # Metrics
            loss_meter.update(loss.item())
            progress_bar.set_postfix(loss=f"{loss_meter.avg:.3f}")
            all_labels.extend(sample_target_weak['label'].tolist())
            all_pseudo_labels.extend(pseudo_targets.tolist())
            all_pseudo_mask.extend(pseudo_mask.tolist())

            if global_step % config.log_step == 0:
                writer.add_scalar("train/loss", loss_meter.val, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/target_updates", len(torch.nonzero(pseudo_mask)), global_step)
                log_fredn_diagnostics(student, writer, global_step)
                if shape_result is not None:
                    _log_shape_result(
                        writer,
                        shape_result,
                        timematch_loss,
                        config,
                        pseudo_mask,
                        global_step,
                    )

            global_step += 1

        progress_bar.close()

        # Evaluate pseudo labels
        all_labels, all_pseudo_labels, all_pseudo_mask = np.array(all_labels), np.array(all_pseudo_labels), np.array(all_pseudo_mask)
        pseudo_count = all_pseudo_mask.sum()
        conf_pseudo_f1 = sklearn.metrics.f1_score(all_labels[all_pseudo_mask], all_pseudo_labels[all_pseudo_mask], average='macro', zero_division=0)
        print(f"Teacher pseudo label F1 {conf_pseudo_f1:.3f} (n={pseudo_count})")
        writer.add_scalar("train/pseudo_f1", conf_pseudo_f1, epoch)
        writer.add_scalar("train/pseudo_count", pseudo_count, epoch)

        if shape_alignment is not None:
            shape_diagnostics = _collect_shape_epoch_diagnostics(
                teacher,
                shape_alignment,
                target_loader_no_aug,
                device,
                target_to_source_shift,
                config.pseudo_threshold,
                config.shape_diag_batches,
            )
            for name, value in shape_diagnostics.items():
                writer.add_scalar(f"shape_diag/{name}", value, epoch)

        writer.add_scalar("train/pseudo_f1", conf_pseudo_f1, epoch)
        writer.add_scalar("train/pseudo_count", pseudo_count, epoch)

        if config.run_validation:
            if config.output_student:
                student.eval()
                best_f1 = validation(best_f1, None, config, criterion, device, epoch, student, val_loader, writer)
            else:
                teacher.eval()
                best_f1 = validation(best_f1, None, config, criterion, device, epoch, teacher, val_loader, writer)

    # Save model final model 
    if config.output_student:
        torch.save({'state_dict': student.state_dict()}, best_model_path)
    else:
        torch.save({'state_dict': teacher.state_dict()}, best_model_path)

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
):
    shifts = list(range(min_shift, max_shift + 1))
    model.eval()
    if sample_size is None:
        sample_size = len(target_loader)

    target_iter = iter(target_loader)
    shift_softmaxes, labels = [], []
    for _ in tqdm(
        range(sample_size),
        desc=f'Estimating shift between [{min_shift}, {max_shift}]',
        disable=progress_bar_disabled(progress_bar),
    ):
        sample = next(target_iter)
        labels.extend(sample['label'].tolist())
        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        spatial_feats = model.spatial_encoder.forward(pixels, valid_pixels, extra)
        prepared = _prepare_temporal_features(model, spatial_feats, positions)
        shift_logits = torch.stack(
            [
                _classify_prepared(
                    model,
                    prepared,
                    positions,
                    temporal_shift=shift,
                )
                for shift in shifts
            ],
            dim=1,
        )
        shift_probs = F.softmax(shift_logits, dim=2)
        shift_softmaxes.append(shift_probs)
    shift_softmaxes = torch.cat(shift_softmaxes).cpu().numpy()  # (N, n_shifts, n_classes)
    labels = np.array(labels)
    shift_predictions = np.argmax(shift_softmaxes, axis=2)  # (N, n_shifts)

    # shift_f1_scores = [f1_score(labels, shift_predictions, num_classes) for shift_predictions in all_shift_predictions]
    shift_acc_scores = [(labels == predictions).mean() for predictions in np.moveaxis(shift_predictions, 0, 1)]
    print(f"Most accurate shift {shifts[np.argmax(shift_acc_scores)]} with {np.max(shift_acc_scores):.3f}")

    p_yx = shift_softmaxes # (N, n_shifts, n_classes)
    p_y = shift_softmaxes.mean(axis=0)  # (n_shifts, n_classes)


    if shift_estimator == 'IS':
        inception_score = np.mean(np.sum(p_yx * (np.log(p_yx + 1e-5) - np.log(p_y[np.newaxis] + 1e-5)), axis=2), axis=0)  # (n_shifts)

        shift_indices_ranked = np.argsort(inception_score)[::-1]  # max is best
        best_shift_idx = shift_indices_ranked[0]
        best_shift = shifts[best_shift_idx]
        print(f"Best Inception Score shift {best_shift} with accuracy {shift_acc_scores[best_shift_idx]:.3f}")
        return best_shift

    elif shift_estimator == 'ENT':
        entropy_score = -np.mean(np.sum(p_yx * np.log(p_yx + 1e-5), axis=2), axis=0)  # (n_shifts)
        shift_indices_ranked = np.argsort(entropy_score)  # min is best
        best_shift_idx = shift_indices_ranked[0]
        best_shift = shifts[best_shift_idx]
        print(f"Best Entropy Score shift {best_shift} with accuracy {shift_acc_scores[best_shift_idx]:.3f}")
        return best_shift

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
        am = kl_d + entropy
        shift_indices_ranked = np.argsort(am)  # min is best
        best_shift_idx = shift_indices_ranked[0]
        best_shift = shifts[best_shift_idx]
        print(f"Best AM Score shift {best_shift} with accuracy {shift_acc_scores[best_shift_idx]:.3f}")

        return best_shift
    elif shift_estimator == 'ACC':  # for upperbound comparison
        shift_indices_ranked = np.argsort(shift_acc_scores)[::-1]  # max is best
        return shifts[np.argmax(shift_acc_scores)]
    else:
        raise NotImplementedError




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
        logits = _forward_with_temporal_shift(
            model,
            pixels,
            valid_pixels,
            positions,
            extra,
            temporal_shift=best_shift,
        )
        probs = F.softmax(logits, dim=1).cpu()
        pseudo_softmaxes.extend(probs.tolist())

    indices = torch.as_tensor(indices)
    pseudo_softmaxes = torch.as_tensor(pseudo_softmaxes)
    pseudo_softmaxes = pseudo_softmaxes[torch.argsort(indices)]

    return pseudo_softmaxes
