from collections import Counter
from copy import deepcopy
from collections import defaultdict
import math
import os

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from dataset import PixelSetData
from evaluation import validation
from models.fourier_reconstruction import BatchedDirectFourierAnalyzer, BatchedDirectFourierSynthesizer
from models.stclassifier import PseStructureProtoLTae
from transforms import Normalize, RandomSamplePixels, RandomSampleTimeSteps, ToTensor, RandomTemporalShift, Identity
from utils.focal_loss import FocalLoss
from utils.train_utils import AverageMeter, bool_flag, cycle, progress_bar_disabled, to_cuda
from methods.structure_da.prototype_losses import (
    accumulate_class_feature_sums,
    centroid_alignment_summary,
    class_balanced_shape_pseudo_loss,
    class_relative_domain_alignment,
    memory_class_balanced_shape_pseudo_loss,
    memory_class_relative_domain_alignment,
    compose_da_loss,
    compose_structure_v5_da_loss,
    ensure_finite_structure_loss,
    instance_batch_statistics,
    instance_prototype_loss,
    log_shape_health,
    prototype_ramp,
    shapelet_data_support_loss,
    shapelet_diversity_loss,
    selected_shape_pseudo_loss,
    masked_pseudo_classification_loss,
    shared_private_separation_loss,
    structure_domain_adversarial_loss,
    structure_private_domain_loss,
    update_instance_bank,
)
from models.structure_da.prototype_bank import ClassFeatureMemory


def _reset_trainable_module(module):
    for child in module.modules():
        if child is not module and hasattr(child, "reset_parameters"):
            child.reset_parameters()


def load_v4_source_for_v5(model, checkpoint):
    """Load a V4 source checkpoint and initialize only V5 domain modules."""
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    allowed = (
        "structure_branch.domain_projector.",
        "private_domain_classifier.",
    )
    invalid_missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "incompatible V4 source checkpoint: "
            f"invalid_missing={invalid_missing}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    _reset_trainable_module(model.domain_classifier)
    _reset_trainable_module(model.structure_branch.domain_projector)
    _reset_trainable_module(model.private_domain_classifier)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def load_v6_source_for_v6(model, checkpoint):
    """Strictly load the independently trained V6 source before UDA."""
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    _reset_trainable_module(model.domain_classifier)
    _reset_trainable_module(model.structure_branch.domain_projector)
    _reset_trainable_module(model.private_domain_classifier)
    return {"missing_keys": [], "unexpected_keys": []}


def _new_epoch_structure_stats():
    return {"instance_cos": []}


def _collect_epoch_structure_stats(accumulator, batch):
    accumulator["instance_cos"].append(batch["instance_cos"].detach())


def _summarize_epoch_structure_stats(accumulator, prefix, expected_scales=()):
    if not accumulator["instance_cos"]:
        return {f"{prefix}_instance_cos_to_correct_proto": float("nan")}
    return {f"{prefix}_instance_cos_to_correct_proto": float(torch.cat(accumulator["instance_cos"]).mean())}


def _prototype_bank_stats(model):
    row = {}
    bank = model.instance_prototype_bank
    active = bank.prototypes[bank.initialized]
    similarity = active @ active.T
    off_diagonal = similarity[~torch.eye(active.shape[0], dtype=torch.bool, device=active.device)]
    row["prototype_instance_initialized_classes"] = int(bank.initialized.sum())
    row["prototype_instance_mean_pairwise_cos"] = float(off_diagonal.mean()) if off_diagonal.numel() else 0.
    row["prototype_instance_min_pairwise_cos"] = float(off_diagonal.min()) if off_diagonal.numel() else 0.
    return row


def add_shift_estimation_arguments(parser):
    parser.add_argument("--shift-estimation-view", "--shift_estimation_view", dest="shift_estimation_view", default="raw", choices=["raw", "fourier_recon"])
    parser.add_argument("--shift-fourier-num-modes", "--shift_fourier_num_modes", dest="shift_fourier_num_modes", default=13, type=int)
    parser.add_argument("--shift-fourier-reg", "--shift_fourier_reg", dest="shift_fourier_reg", default=1e-3, type=float)
    parser.add_argument("--shift-fourier-period-days", "--shift_fourier_period_days", dest="shift_fourier_period_days", default=365.0, type=float)
    parser.add_argument("--shift-fourier-solver", "--shift_fourier_solver", dest="shift_fourier_solver", default="dense_direct", choices=["dense_direct"])
    return parser


def _shift_estimation_kwargs(config):
    view = getattr(config, "shift_estimation_view", "raw")
    if view == "raw":
        return {"shift_estimation_view": "raw"}
    return {
        "shift_estimation_view": view,
        "shift_fourier_num_modes": config.shift_fourier_num_modes,
        "shift_fourier_reg": config.shift_fourier_reg,
        "shift_fourier_period_days": config.shift_fourier_period_days,
        "shift_fourier_solver": config.shift_fourier_solver,
    }

def _estimate_temporal_shift_for_config(
    model,
    target_loader,
    device,
    config,
    class_distribution=None,
    **kwargs,
):
    if class_distribution is not None:
        kwargs["class_distribution"] = class_distribution
    return estimate_temporal_shift(
        model,
        target_loader,
        device,
        **kwargs,
        **_shift_estimation_kwargs(config),
    )

def _log_shift_view_config(config):
    view = getattr(config, "shift_estimation_view", "raw")
    print(f"SHIFT_ESTIMATION_VIEW|{view}")
    if view == "fourier_recon":
        print(
            "SHIFT_FOURIER_CONFIG|"
            f"modes={config.shift_fourier_num_modes}|"
            f"reg={config.shift_fourier_reg}|"
            f"period_days={config.shift_fourier_period_days}|"
            f"solver={config.shift_fourier_solver}"
        )

def _log_shift_view_compare(epoch, initial_diagnostics, epoch_diagnostics):
    raw_is = initial_diagnostics["raw_selected_shift"]
    recon_is = initial_diagnostics["selected_shift"]
    raw_am = epoch_diagnostics["raw_selected_shift"]
    recon_am = epoch_diagnostics["selected_shift"]
    print(
        "SHIFT_VIEW_COMPARE|"
        f"epoch={epoch}|raw_shift={raw_am}|recon_shift={recon_am}|"
        f"delta={recon_am - raw_am}|"
        f"raw_is_shift={raw_is}|recon_is_shift={recon_is}|"
        f"raw_am_shift={raw_am}|recon_am_shift={recon_am}"
    )
    print(
        "RECON_SHIFT_VIEW_DIAG|"
        f"mean_confidence={epoch_diagnostics['mean_confidence']:.6f}|"
        f"prediction_entropy={epoch_diagnostics['prediction_entropy']:.6f}|"
        f"num_predicted_classes={epoch_diagnostics['num_predicted_classes']}|"
        f"am_score_range={epoch_diagnostics['score_range']:.6f}|"
        f"is_score_range={initial_diagnostics['score_range']:.6f}"
    )


def _initialize_timematch_shift(model, target_loader, device, config):
    """Replay the established IS -> pseudo distribution initialization."""
    if not config.estimate_shift:
        return 0, None, None

    estimator = "IS" if config.shift_estimator == "AM" else config.shift_estimator
    initial_shift, diagnostics = _estimate_temporal_shift_for_config(
        model, target_loader, device, config,
        min_shift=-config.max_temporal_shift,
        max_shift=config.max_temporal_shift,
        sample_size=config.sample_size,
        shift_estimator=estimator,
        progress_bar=getattr(config, "progress_bar", "auto"),
        compare_raw=getattr(config, "shift_estimation_view", "raw") == "fourier_recon",
        return_diagnostics=True,
        include_label_diagnostics=False,
    )
    print(
        "INITIAL_SHIFT|"
        f"source={config.source}|target={config.target}|"
        f"view={getattr(config, 'shift_estimation_view', 'raw')}|"
        f"shift_days={initial_shift}"
    )
    if config.shift_estimator != "AM":
        return initial_shift, None, diagnostics

    pseudo = get_pseudo_labels(
        model, target_loader, device, initial_shift, n=None,
        progress_bar=getattr(config, "progress_bar", "auto"),
    )
    distribution = estimate_class_distribution(
        torch.argmax(pseudo, dim=1).cpu().numpy(), config.num_classes
    )
    return initial_shift, distribution, diagnostics


def _reestimate_timematch_shift(
    model, target_loader, device, config, initial_shift,
    class_distribution, initial_diagnostics, epoch,
):
    """Perform the established epoch-wise AM update without target labels."""
    if not config.estimate_shift or config.shift_estimator != "AM":
        return initial_shift
    min_shift, max_shift = (
        (0, config.max_temporal_shift)
        if initial_shift >= 0
        else (-config.max_temporal_shift, 0)
    )
    shift, diagnostics = _estimate_temporal_shift_for_config(
        model, target_loader, device, config,
        class_distribution=class_distribution,
        min_shift=min_shift,
        max_shift=max_shift,
        sample_size=config.sample_size,
        shift_estimator="AM",
        progress_bar=getattr(config, "progress_bar", "auto"),
        compare_raw=getattr(config, "shift_estimation_view", "raw") == "fourier_recon",
        return_diagnostics=True,
        include_label_diagnostics=False,
    )
    if getattr(config, "shift_estimation_view", "raw") == "fourier_recon":
        _log_shift_view_compare(epoch, initial_diagnostics, diagnostics)
    print(f"EPOCH_SHIFT|epoch={epoch}|target_to_source_days={shift}")
    return shift

def _check_temporal_index_range(model, positions, applied_shift, tag):
    if positions.numel() == 0:
        return

    min_pos = int(positions.min().item())
    max_pos = int(positions.max().item())
    shift_tensor = torch.as_tensor(applied_shift)
    min_shift = float(shift_tensor.min().item())
    max_shift = float(shift_tensor.max().item())
    if hasattr(model, "get_temporal_encoders"):
        temporal_encoders = model.get_temporal_encoders()
    else:
        temporal_encoders = (model.temporal_encoder,)
    for encoder_index, temporal_encoder in enumerate(temporal_encoders):
        min_idx = min_pos + min_shift + temporal_encoder.max_temporal_shift
        max_idx = max_pos + max_shift + temporal_encoder.max_temporal_shift
        table_size = temporal_encoder.positional_enc.num_embeddings

        if min_idx < 0 or max_idx >= table_size:
            raise ValueError(
                f"{tag} temporal indices out of range: encoder={encoder_index}, "
                f"positions=[{min_pos}, {max_pos}], shift=[{min_shift}, {max_shift}], "
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
        return model.forward_with_temporal_shift(
            pixels,
            mask,
            positions,
            extra,
            temporal_shift=temporal_shift,
        )
    return model.forward(pixels, mask, positions + temporal_shift, extra)

def _prepare_temporal_features(
    model,
    spatial_feats,
    positions,
    collect_diagnostics=False,
):
    if hasattr(model, "prepare_temporal_features"):
        return model.prepare_temporal_features(spatial_feats, positions)
    return spatial_feats

def _classify_prepared(
    model, prepared, positions, temporal_shift=0, prepared_structure=None,
):
    if hasattr(model, "classify_prepared"):
        kwargs = {"temporal_shift": temporal_shift}
        if prepared_structure is not None:
            kwargs["prepared_structure"] = prepared_structure
        return model.classify_prepared(prepared, positions, **kwargs)
    return model.decoder(
        model.temporal_encoder(prepared, positions + temporal_shift)
    )


def _classify_shift_grid(model, prepared, positions, shifts):
    structure = (
        model.prepare_structure(prepared, positions)
        if hasattr(model, "prepare_structure") else None
    )
    return torch.stack([
        _classify_prepared(
            model, prepared, positions, temporal_shift=shift,
            prepared_structure=structure,
        )
        for shift in shifts
    ], dim=1)


@torch.no_grad()
def _update_structure_feature_memories(
    memories, source_output, source_labels, teacher_output,
    pseudo, confidence, pseudo_mask, pseudo_threshold,
):
    memories["shape_source"].update_source(
        source_output["shapelet_response"], source_labels,
    )
    memories["stats_source"].update_source(
        source_output["shape_stats_feature"], source_labels,
    )
    if pseudo_mask.any():
        memories["shape_target"].update_target(
            teacher_output["shapelet_response"][pseudo_mask], pseudo[pseudo_mask],
            confidence[pseudo_mask], pseudo_threshold,
        )
        memories["stats_target"].update_target(
            teacher_output["shape_stats_feature"][pseudo_mask], pseudo[pseudo_mask],
            confidence[pseudo_mask], pseudo_threshold,
        )


def _structure_memory_checkpoint(memories):
    return {
        name: {
            key: value.detach().cpu().clone()
            for key, value in memory.state_dict().items()
        }
        for name, memory in memories.items()
    }


@torch.no_grad()
def _memory_center_gap(source_memory, target_memory, support_saturation=4):
    reliability = target_memory.reliability(support_saturation)
    valid = (
        source_memory.initialized & target_memory.initialized
        & (reliability > 0)
    )
    if not valid.any():
        return reliability.new_tensor(0.)
    weights = reliability[valid]
    weights = weights / weights.sum().clamp_min(1e-12)
    source_center = (weights[:, None] * source_memory.prototypes[valid]).sum(0)
    target_center = (weights[:, None] * target_memory.prototypes[valid]).sum(0)
    return (target_center - source_center).norm()

def _train_structure_proto_timematch_v3_reference(
    student, config, writer, val_loader, device, best_model_path, fold_num, splits,
):
    if config.with_shift_aug:
        raise ValueError(
            "structure prototype TimeMatch requires identical canonical window indexing; "
            "set --with_shift_aug false"
        )
    source_loader, target_loader_no_aug, target_loader = get_data_loaders(
        splits, config, config.balance_source
    )
    checkpoint_path = os.path.join(config.weights, f"fold_{fold_num}", "model.pt")
    packet = torch.load(checkpoint_path, weights_only=False)
    student.load_state_dict(packet["state_dict"])
    if not student.instance_prototype_bank.initialized.all():
        raise RuntimeError("source instance prototype bank is incomplete")
    student.to(device)
    teacher = deepcopy(student).to(device)
    teacher.eval()
    memories = {
        "shape_source": ClassFeatureMemory(
            config.num_classes, student.shape_classifier.in_features, momentum=.9,
        ).to(device),
        "shape_target": ClassFeatureMemory(
            config.num_classes, student.shape_classifier.in_features, momentum=.9,
        ).to(device),
        "stats_source": ClassFeatureMemory(
            config.num_classes, 64, momentum=.9,
        ).to(device),
        "stats_target": ClassFeatureMemory(
            config.num_classes, 64, momentum=.9,
        ).to(device),
    }
    criterion = (
        FocalLoss(gamma=config.focal_loss_gamma)
        if config.use_focal_loss else torch.nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * config.steps_per_epoch, eta_min=0,
    )
    source_iter, target_iter = iter(cycle(source_loader)), iter(cycle(target_loader))
    initial_shift, class_distribution, initial_diagnostics = _initialize_timematch_shift(
        teacher, target_loader_no_aug, device, config
    )
    target_to_source_shift = initial_shift
    best_f1 = 0
    global_step = 0
    for epoch in range(config.epochs):
        target_ramp = prototype_ramp(
            epoch, config.proto_ramp_epochs, config.proto_ramp_start,
        )
        target_to_source_shift = _reestimate_timematch_shift(
            teacher, target_loader_no_aug, device, config,
            initial_shift, class_distribution, initial_diagnostics, epoch,
        )
        source_to_target_shift = (
            -target_to_source_shift if getattr(config, "shift_source", True) else 0
        )
        student.train()
        teacher.eval()
        source_epoch_stats = _new_epoch_structure_stats()
        target_epoch_stats = _new_epoch_structure_stats()
        centroid_sums = {
            "source_instance": torch.zeros(
                config.num_classes, student.instance_dim, device=device,
            ),
            "target_instance": torch.zeros(
                config.num_classes, student.instance_dim, device=device,
            ),
        }
        centroid_counts = {
            "source": torch.zeros(config.num_classes, device=device),
            "target": torch.zeros(config.num_classes, device=device),
        }
        progress = tqdm(
            range(config.steps_per_epoch),
            desc=f"StructureProto TimeMatch {epoch + 1}/{config.epochs}",
            disable=progress_bar_disabled(getattr(config, "progress_bar", "auto")),
        )
        shape_source_loss_sum = 0.
        shape_source_correct = 0
        shape_source_count = 0
        shape_target_loss_sum = 0.
        shape_target_correct = 0
        shape_target_count = 0
        shape_target_batches = 0
        v2_epoch_sums = defaultdict(float)
        v2_epoch_batches = 0
        pseudo_total_count = 0
        pseudo_accepted_count = 0
        for epoch_step in progress:
            source_sample = next(source_iter)
            target_weak, target_strong = next(target_iter)
            pw, mw, tw, ew = to_cuda(target_weak, device)
            with torch.no_grad():
                teacher_output = teacher.forward_with_temporal_shift(
                    pw, mw, tw, ew, temporal_shift=target_to_source_shift,
                    return_dict=True,
                )
                probabilities = F.softmax(teacher_output["logits"], dim=1)
                confidence, pseudo = probabilities.max(1)
                pseudo_mask = confidence > config.pseudo_threshold
            pseudo_total_count += int(confidence.numel())
            pseudo_accepted_count += int(pseudo_mask.sum())

            ps, ms, ts, es = to_cuda(source_sample, device)
            source_labels = source_sample["label"].cuda(device=device, non_blocking=True)
            source_output = student.forward_with_temporal_shift(
                ps, ms, ts, es, temporal_shift=source_to_target_shift,
                return_dict=True,
            )
            loss_cls_source = criterion(source_output["logits"], source_labels)
            loss_shape_source = criterion(source_output["shape_logits"], source_labels)
            source_ins = instance_prototype_loss(
                source_output["instance_feature"], source_labels,
                student.instance_prototype_bank, config.proto_temperature,
            )
            source_proto = config.proto_instance_weight * source_ins
            with torch.no_grad():
                accumulate_class_feature_sums(
                    centroid_sums["source_instance"], centroid_counts["source"],
                    source_output["instance_feature"], source_labels,
                )

            target_count = int(pseudo_mask.sum())
            if target_count >= 2:
                pt, mt, tt, et = to_cuda(target_strong, device)
                target_output = student(
                    pt[pseudo_mask], mt[pseudo_mask], tt[pseudo_mask], et[pseudo_mask],
                    return_dict=True,
                )
                target_labels = pseudo[pseudo_mask]
                loss_pseudo_target = criterion(target_output["logits"], target_labels)
                target_confidence = confidence[pseudo_mask]
                _update_structure_feature_memories(
                    memories, source_output, source_labels, teacher_output,
                    pseudo, confidence, pseudo_mask, config.pseudo_threshold,
                )
                target_shape = memory_class_balanced_shape_pseudo_loss(
                    target_output["shape_logits"], target_labels, target_confidence,
                    memories["shape_target"], config.pseudo_threshold,
                    config.focal_loss_gamma,
                )
                loss_shape_target = target_shape["loss"]
                target_shape_accuracy = float(
                    (target_output["shape_logits"].detach().argmax(1) == target_labels)
                    .float().mean()
                )
                shape_alignment = memory_class_relative_domain_alignment(
                    memories["shape_source"], memories["shape_target"],
                    target_output["shapelet_response"], target_labels,
                    target_confidence, config.pseudo_threshold, distance="mse",
                )
                stats_alignment = memory_class_relative_domain_alignment(
                    memories["stats_source"], memories["stats_target"],
                    target_output["shape_stats_feature"], target_labels,
                    target_confidence, config.pseudo_threshold, distance="smooth_l1",
                )
                target_ins = instance_prototype_loss(
                    target_output["instance_feature"], target_labels,
                    student.instance_prototype_bank, config.proto_temperature,
                )
                target_proto = config.proto_instance_weight * target_ins
                with torch.no_grad():
                    accumulate_class_feature_sums(
                        centroid_sums["target_instance"], centroid_counts["target"],
                        target_output["instance_feature"], target_labels,
                    )
                _collect_epoch_structure_stats(
                    target_epoch_stats,
                    instance_batch_statistics(target_output, target_labels, student.instance_prototype_bank),
                )
            else:
                _update_structure_feature_memories(
                    memories, source_output, source_labels, teacher_output,
                    pseudo, confidence, pseudo_mask, config.pseudo_threshold,
                )
                zero = source_output["logits"].sum() * 0
                loss_pseudo_target = target_ins = target_proto = zero
                loss_shape_target = zero
                target_shape_accuracy = 0.
                target_shape = {
                    "valid_classes": 0,
                    "mean_class_reliability": zero.detach(),
                }
                shape_alignment = stats_alignment = {
                    "total_loss": zero, "global_loss": zero,
                    "relative_loss": zero, "center_gap": zero.detach(),
                    "valid_classes": 0, "relative_active": False,
                }

            loss_diversity = shapelet_diversity_loss(
                student.structure_branch.shapelet_dictionary.anchors,
                config.shapelet_diversity_margin,
            )
            loss_shaping = shapelet_data_support_loss(
                source_output["shape_tokens"],
                student.structure_branch.shapelet_dictionary.anchors,
                config.shapelet_shaping_temperature,
            )

            loss = compose_da_loss(
                loss_cls_source, loss_pseudo_target, config.trade_off,
                source_ins, target_ins, loss_diversity, loss_shaping,
                target_ramp, config.proto_instance_weight,
                config.shapelet_diversity_weight, config.shapelet_shaping_weight,
            )
            loss = (
                loss
                + config.shape_class_weight * loss_shape_source
                + target_ramp * getattr(config, "shape_target_weight", .05)
                * loss_shape_target
                + target_ramp * getattr(config, "shape_align_weight", .05)
                * shape_alignment["total_loss"]
                + target_ramp * getattr(config, "stats_align_weight", .02)
                * stats_alignment["total_loss"]
            )
            optimizer.zero_grad()
            ensure_finite_structure_loss(loss)
            loss.backward()
            if epoch_step == 0:
                log_shape_health(writer, epoch, student, source_output)
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=5., error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            update_instance_bank(
                source_output["instance_feature"], source_labels,
                student.instance_prototype_bank,
            )
            _collect_epoch_structure_stats(
                source_epoch_stats,
                instance_batch_statistics(source_output, source_labels, student.instance_prototype_bank),
            )
            source_batch_count = int(source_labels.shape[0])
            shape_source_loss_sum += float(loss_shape_source.detach()) * source_batch_count
            shape_source_correct += int(
                (source_output["shape_logits"].detach().argmax(1) == source_labels).sum()
            )
            shape_source_count += source_batch_count
            if target_count >= 2:
                shape_target_loss_sum += float(loss_shape_target.detach())
                shape_target_correct += int(round(target_shape_accuracy * target_count))
                shape_target_count += target_count
                shape_target_batches += 1
            concentration = source_output["shapelet_concentration"].detach().float()
            singular = torch.linalg.svdvals(
                source_output["shapelet_response"].detach().float()
            )
            probability = singular / singular.sum().clamp_min(1e-12)
            response_rank = torch.exp(
                -(probability * probability.clamp_min(1e-12).log()).sum()
            )
            v2_values = {
                "shape_strength_std": strength.std(dim=0, unbiased=False).mean(),
                "shape_concentration_mean": concentration.mean(),
                "shape_concentration_std": concentration.std(unbiased=False),
                "shape_response_effective_rank": response_rank,
                "target_shape_valid_classes": target_shape["valid_classes"],
                "target_shape_mean_reliability": target_shape["mean_class_reliability"],
                "loss_shape_target_balanced": loss_shape_target.detach(),
                "loss_shape_align_global": shape_alignment["global_loss"].detach(),
                "loss_shape_align_relative": shape_alignment["relative_loss"].detach(),
                "loss_stats_align_global": stats_alignment["global_loss"].detach(),
                "loss_stats_align_relative": stats_alignment["relative_loss"].detach(),
                "shape_domain_center_gap": shape_alignment["center_gap"],
                "stats_domain_center_gap": stats_alignment["center_gap"],
                "shape_valid_align_classes": shape_alignment["valid_classes"],
                "stats_valid_align_classes": stats_alignment["valid_classes"],
                "shape_relative_active_batch_ratio": float(
                    shape_alignment["relative_active"]
                ),
                "stats_relative_active_batch_ratio": float(
                    stats_alignment["relative_active"]
                ),
            }
            for name, value in v2_values.items():
                value = torch.as_tensor(value, device=device).detach()
                v2_epoch_sums[name] = v2_epoch_sums.get(name, torch.zeros_like(value)) + value
            v2_epoch_batches += 1
            update_ema_variables(student, teacher, config.ema_decay)
            if global_step % config.log_step == 0:
                metrics = {
                    "loss_cls_source": loss_cls_source,
                    "loss_pseudo_target": loss_pseudo_target,
                    "loss_shape_source": loss_shape_source,
                    "loss_shape_target": loss_shape_target,
                    "loss_proto_instance_source": source_ins,
                    "loss_proto_instance_target": target_ins,
                    "loss_shapelet_diversity": loss_diversity,
                    "shapelet_shaping_loss": loss_shaping,
                    "loss_proto_source_total": source_proto,
                    "loss_proto_target_total": target_proto,
                    "proto_ramp_source": source_ins.new_tensor(1.),
                    "proto_ramp_target": source_ins.new_tensor(target_ramp),
                    "loss_total": loss,
                }
                for name, value in metrics.items():
                    writer.add_scalar(f"train/{name}", value.detach(), global_step)
                print("STRUCTURE_PROTO_DA|" + "|".join(
                    f"{name}={float(value.detach()):.6f}" for name, value in metrics.items()
                ))
            global_step += 1
        progress.close()
        source_shape_loss_epoch = shape_source_loss_sum / max(shape_source_count, 1)
        source_shape_accuracy_epoch = shape_source_correct / max(shape_source_count, 1)
        target_shape_loss_epoch = shape_target_loss_sum / max(shape_target_batches, 1)
        target_shape_accuracy_epoch = shape_target_correct / max(shape_target_count, 1)
        writer.add_scalar("epoch/shape_source_loss", source_shape_loss_epoch, epoch)
        writer.add_scalar("epoch/shape_source_accuracy", source_shape_accuracy_epoch, epoch)
        writer.add_scalar("epoch/shape_target_loss", target_shape_loss_epoch, epoch)
        writer.add_scalar("epoch/shape_target_pseudo_accuracy", target_shape_accuracy_epoch, epoch)
        print(
            f"SHAPE_AUX_EPOCH|epoch={epoch}|source_loss={source_shape_loss_epoch:.6f}|"
            f"source_accuracy={source_shape_accuracy_epoch:.6f}|"
            f"target_loss={target_shape_loss_epoch:.6f}|"
            f"target_pseudo_accuracy={target_shape_accuracy_epoch:.6f}|"
            f"source_samples={shape_source_count}|target_samples={shape_target_count}"
        )
        v2_epoch = {
            name: float((total / max(v2_epoch_batches, 1)).cpu())
            for name, total in v2_epoch_sums.items()
        }
        for name, value in v2_epoch.items():
            writer.add_scalar(f"epoch/{name}", value, epoch)
        print("SHAPE_V2_EPOCH|epoch=" + str(epoch) + "|" + "|".join(
            f"{name}={value:.6f}" for name, value in v2_epoch.items()
        ))
        target_reliability = memories["shape_target"].reliability()
        initialized_target = memories["shape_target"].initialized
        active_reliability = target_reliability[initialized_target]
        if active_reliability.numel():
            memory_mean_reliability = float(active_reliability.mean())
            memory_min_reliability = float(active_reliability.min())
            memory_mean_support = float(
                memories["shape_target"].sample_count[initialized_target]
                .float().mean()
            )
        else:
            memory_mean_reliability = 0.
            memory_min_reliability = 0.
            memory_mean_support = 0.
        v3_epoch = {
            "pseudo_coverage": pseudo_accepted_count / max(pseudo_total_count, 1),
            "shape_memory_source_classes": int(memories["shape_source"].initialized.sum()),
            "shape_memory_target_classes": int(memories["shape_target"].initialized.sum()),
            "stats_memory_source_classes": int(memories["stats_source"].initialized.sum()),
            "stats_memory_target_classes": int(memories["stats_target"].initialized.sum()),
            "target_memory_mean_reliability": memory_mean_reliability,
            "target_memory_min_reliability": memory_min_reliability,
            "target_memory_mean_support": memory_mean_support,
            "shape_relative_active_batch_ratio": v2_epoch["shape_relative_active_batch_ratio"],
            "stats_relative_active_batch_ratio": v2_epoch["stats_relative_active_batch_ratio"],
            "shape_memory_center_gap": float(_memory_center_gap(
                memories["shape_source"], memories["shape_target"],
            )),
            "stats_memory_center_gap": float(_memory_center_gap(
                memories["stats_source"], memories["stats_target"],
            )),
            "loss_shape_align_global": v2_epoch["loss_shape_align_global"],
            "loss_shape_align_relative": v2_epoch["loss_shape_align_relative"],
            "loss_stats_align_global": v2_epoch["loss_stats_align_global"],
            "loss_stats_align_relative": v2_epoch["loss_stats_align_relative"],
        }
        for name, value in v3_epoch.items():
            writer.add_scalar(f"epoch/{name}", value, epoch)
        print("SHAPE_V3_EPOCH|epoch=" + str(epoch) + "|" + "|".join(
            f"{name}={value:.6f}" for name, value in v3_epoch.items()
        ))
        prototype_row = {"epoch": epoch, **_prototype_bank_stats(student)}
        prototype_row.update(_summarize_epoch_structure_stats(
            source_epoch_stats, "source", config.shape_window_scales,
        ))
        prototype_row.update(_summarize_epoch_structure_stats(
            target_epoch_stats, "target", config.shape_window_scales,
        ))
        print("STRUCTURE_DIAG|" + "|".join(
            f"{key}={value}" for key, value in prototype_row.items()
        ))
        for level in ("instance",):
            alignment = centroid_alignment_summary(
                centroid_sums["source_instance"], centroid_counts["source"],
                centroid_sums["target_instance"], centroid_counts["target"],
            )
            valid_classes = alignment["valid_classes"].detach().cpu().tolist()
            per_class_values = alignment["per_class"].detach().cpu().tolist()
            per_class = ",".join(
                f"{class_id}:{value:.6f}"
                for class_id, value in zip(valid_classes, per_class_values)
            )
            macro = float(alignment["macro_cos"].detach().cpu())
            print(
                f"DOMAIN_CENTROID_ALIGN|epoch={epoch}|level={level}|"
                f"macro_cos={macro:.6f}|valid_classes={len(valid_classes)}|"
                f"per_class={per_class}"
            )
        previous_best = best_f1
        if config.run_validation:
            student.eval()
            best_f1 = validation(
                best_f1, None, config, criterion, device, epoch,
                student if config.output_student else teacher, val_loader, writer,
            )
        state_model = student if config.output_student else teacher
        checkpoint = {
            "epoch": epoch,
            "state_dict": state_model.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_temporal_shift": target_to_source_shift,
            "config": vars(config),
            "best_f1": best_f1,
            "structure_memory": _structure_memory_checkpoint(memories),
        }
        torch.save(checkpoint, os.path.join(config.fold_dir, "checkpoint_last.pt"))
        if best_f1 > previous_best or not os.path.isfile(best_model_path):
            torch.save(checkpoint, best_model_path)
            torch.save(checkpoint, os.path.join(config.fold_dir, "checkpoint_best.pt"))


def _train_structure_proto_timematch(
    student, config, writer, val_loader, device, best_model_path, fold_num, splits,
):
    if config.with_shift_aug:
        raise ValueError(
            "structure prototype TimeMatch requires identical canonical window indexing; "
            "set --with_shift_aug false"
        )
    source_loader, target_loader_no_aug, target_loader = get_data_loaders(
        splits, config, config.balance_source,
    )
    checkpoint_path = os.path.join(config.weights, f"fold_{fold_num}", "model.pt")
    compatibility = load_v6_source_for_v6(
        student, torch.load(checkpoint_path, weights_only=False),
    )
    print(
        "STRUCTURE_V6_SOURCE_LOAD|"
        f"checkpoint={checkpoint_path}|"
        f"missing={','.join(compatibility['missing_keys'])}|unexpected=none"
    )
    student.to(device)
    teacher = deepcopy(student).to(device)
    teacher.eval()
    criterion = (
        FocalLoss(gamma=config.focal_loss_gamma)
        if config.use_focal_loss else torch.nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.Adam(
        student.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    total_steps = config.epochs * config.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0,
    )
    source_iter, target_iter = iter(cycle(source_loader)), iter(cycle(target_loader))
    initial_shift, class_distribution, initial_diagnostics = _initialize_timematch_shift(
        teacher, target_loader_no_aug, device, config,
    )
    target_to_source_shift = initial_shift
    best_f1 = 0
    global_step = 0
    for epoch in range(config.epochs):
        target_to_source_shift = _reestimate_timematch_shift(
            teacher, target_loader_no_aug, device, config,
            initial_shift, class_distribution, initial_diagnostics, epoch,
        )
        source_to_target_shift = (
            -target_to_source_shift if getattr(config, "shift_source", True) else 0
        )
        student.train()
        teacher.eval()
        epoch_sums = defaultdict(float)
        pseudo_total = pseudo_accepted = 0
        source_shape_correct = source_samples = 0
        shared_domain_correct = private_domain_correct = 0
        domain_sample_count = 0
        progress = tqdm(
            range(config.steps_per_epoch),
            desc=f"StructureV6 TimeMatch {epoch + 1}/{config.epochs}",
            disable=progress_bar_disabled(getattr(config, "progress_bar", "auto")),
        )
        for epoch_step in progress:
            source_sample = next(source_iter)
            target_weak, target_strong = next(target_iter)

            pw, mw, tw, ew = to_cuda(target_weak, device)
            with torch.no_grad():
                teacher_logits = teacher.forward_with_temporal_shift(
                    pw, mw, tw, ew, temporal_shift=target_to_source_shift,
                )
                probabilities = F.softmax(teacher_logits, dim=1)
                confidence, pseudo = probabilities.max(1)
                pseudo_mask = confidence > config.pseudo_threshold
            pseudo_total += int(pseudo_mask.numel())
            pseudo_accepted += int(pseudo_mask.sum())

            ps, ms, ts, es = to_cuda(source_sample, device)
            source_labels = source_sample["label"].cuda(device=device, non_blocking=True)
            source_output = student.forward_with_temporal_shift(
                ps, ms, ts, es, temporal_shift=source_to_target_shift,
                return_dict=True,
            )
            pt, mt, tt, et = to_cuda(target_strong, device)
            target_output = student(pt, mt, tt, et, return_dict=True)

            loss_cls_source = criterion(source_output["logits"], source_labels)
            loss_pseudo_target = masked_pseudo_classification_loss(
                target_output["logits"], pseudo, pseudo_mask, criterion,
            )
            loss_shape_source = criterion(source_output["shape_logits"], source_labels)
            loss_diversity = shapelet_diversity_loss(
                student.structure_branch.shapelet_dictionary.anchors,
                config.shapelet_diversity_margin,
            )
            progress_value = global_step / max(total_steps - 1, 1)
            grl_alpha = 2. / (1. + math.exp(-10. * progress_value)) - 1.
            shared_domain = structure_domain_adversarial_loss(
                student.domain_classifier,
                source_output["shape_shared_feature"],
                target_output["shape_shared_feature"],
                grl_alpha,
            )
            private_domain = structure_private_domain_loss(
                student.private_domain_classifier,
                source_output["shape_domain_feature"],
                target_output["shape_domain_feature"],
            )
            loss_separation = shared_private_separation_loss(
                torch.cat((
                    source_output["shape_shared_feature"],
                    target_output["shape_shared_feature"],
                )),
                torch.cat((
                    source_output["shape_domain_feature"],
                    target_output["shape_domain_feature"],
                )),
            )
            loss = compose_structure_v5_da_loss(
                loss_cls_source, loss_pseudo_target, loss_shape_source,
                loss_diversity, shared_domain["loss"], private_domain["loss"],
                loss_separation, config.trade_off,
                config.shape_class_weight, config.shapelet_diversity_weight,
                config.shared_adv_weight, config.private_domain_weight,
                config.separation_weight,
            )

            optimizer.zero_grad()
            ensure_finite_structure_loss(loss)
            loss.backward()
            if epoch_step == 0:
                log_shape_health(writer, epoch, student, source_output)
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=5., error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            update_ema_variables(student, teacher, config.ema_decay)

            source_count = int(source_labels.shape[0])
            target_count = int(pseudo.shape[0])
            source_samples += source_count
            source_shape_correct += int(
                (source_output["shape_logits"].detach().argmax(1) == source_labels).sum()
            )
            domain_sample_count += source_count + target_count
            shared_domain_correct += int(round(
                float(shared_domain["source_accuracy"]) * source_count
                + float(shared_domain["target_accuracy"]) * target_count
            ))
            private_domain_correct += int(round(
                float(private_domain["source_accuracy"]) * source_count
                + float(private_domain["target_accuracy"]) * target_count
            ))
            values = {
                "loss_cls_source": loss_cls_source.detach(),
                "loss_pseudo_target": loss_pseudo_target.detach(),
                "loss_shape_source": loss_shape_source.detach(),
                "loss_shapelet_diversity": loss_diversity.detach(),
                "loss_shared_adv": shared_domain["loss"].detach(),
                "loss_private_domain": private_domain["loss"].detach(),
                "loss_separation": loss_separation.detach(),
                "grl_alpha": source_output["logits"].new_tensor(grl_alpha),
            }
            for name, value in values.items():
                epoch_sums[name] += float(value)
            if global_step % config.log_step == 0:
                metrics = {**values, "loss_total": loss.detach()}
                for name, value in metrics.items():
                    writer.add_scalar(f"train/{name}", value, global_step)
                print("STRUCTURE_V6_DA|" + "|".join(
                    f"{name}={float(value):.6f}" for name, value in metrics.items()
                ))
            global_step += 1
        progress.close()

        batches = max(config.steps_per_epoch, 1)
        epoch_values = {name: value / batches for name, value in epoch_sums.items()}
        epoch_values.update({
            "pseudo_coverage": pseudo_accepted / max(pseudo_total, 1),
            "shared_domain_accuracy": shared_domain_correct / max(domain_sample_count, 1),
            "private_domain_accuracy": private_domain_correct / max(domain_sample_count, 1),
            "source_shape_accuracy": source_shape_correct / max(source_samples, 1),
        })
        for name, value in epoch_values.items():
            writer.add_scalar(f"epoch/{name}", value, epoch)
        print("SHAPE_V6_EPOCH|epoch=" + str(epoch) + "|" + "|".join(
            f"{name}={value:.6f}" for name, value in epoch_values.items()
        ))

        previous_best = best_f1
        if config.run_validation:
            student.eval()
            best_f1 = validation(
                best_f1, None, config, criterion, device, epoch,
                student if config.output_student else teacher, val_loader, writer,
            )
        state_model = student if config.output_student else teacher
        checkpoint = {
            "epoch": epoch,
            "state_dict": state_model.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_temporal_shift": target_to_source_shift,
            "config": vars(config),
            "best_f1": best_f1,
        }
        torch.save(checkpoint, os.path.join(config.fold_dir, "checkpoint_last.pt"))
        if best_f1 > previous_best or not os.path.isfile(best_model_path):
            torch.save(checkpoint, best_model_path)
            torch.save(checkpoint, os.path.join(config.fold_dir, "checkpoint_best.pt"))


def train_timematch(student, config, writer, val_loader, device, best_model_path, fold_num, splits):
    if isinstance(student, PseStructureProtoLTae):
        return _train_structure_proto_timematch(
            student, config, writer, val_loader, device,
            best_model_path, fold_num, splits,
        )

    source_loader, target_loader_no_aug, target_loader = get_data_loaders(
        splits, config, config.balance_source
    )
    checkpoint_path = os.path.join(config.weights, f"fold_{fold_num}", "model.pt")
    student.load_state_dict(torch.load(checkpoint_path, weights_only=False)["state_dict"])
    student.to(device)
    teacher = deepcopy(student).to(device)
    teacher.eval()
    criterion = FocalLoss(gamma=config.focal_loss_gamma) if config.use_focal_loss else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * config.steps_per_epoch, eta_min=0
    )
    source_iter, target_iter = iter(cycle(source_loader)), iter(cycle(target_loader))
    initial_shift, class_distribution, initial_diagnostics = _initialize_timematch_shift(
        teacher, target_loader_no_aug, device, config
    )
    target_to_source_shift = initial_shift
    best_f1, global_step = 0, 0

    for epoch in range(config.epochs):
        target_to_source_shift = _reestimate_timematch_shift(
            teacher, target_loader_no_aug, device, config,
            initial_shift, class_distribution, initial_diagnostics, epoch,
        )
        source_to_target_shift = (
            -target_to_source_shift if getattr(config, "shift_source", True) else 0
        )
        student.train()
        teacher.eval()
        progress = tqdm(
            range(config.steps_per_epoch),
            desc=f"TimeMatch Epoch {epoch + 1}/{config.epochs}",
            disable=progress_bar_disabled(getattr(config, "progress_bar", "auto")),
        )
        labels_epoch, pseudo_epoch, mask_epoch = [], [], []
        for _ in progress:
            source_sample = next(source_iter)
            target_weak, target_strong = next(target_iter)
            pw, mw, tw, ew = to_cuda(target_weak, device)
            with torch.no_grad():
                teacher_logits = _forward_with_temporal_shift(
                    teacher, pw, mw, tw, ew,
                    temporal_shift=target_to_source_shift,
                )
                confidence, pseudo = F.softmax(teacher_logits, dim=1).max(1)
                pseudo_mask = confidence > config.pseudo_threshold

            ps, ms, ts, es = to_cuda(source_sample, device)
            source_labels = source_sample["label"].cuda(device=device, non_blocking=True)
            pt, mt, tt, et = to_cuda(target_strong, device)
            pseudo_count = int(pseudo_mask.sum())
            if getattr(config, "domain_specific_bn", True):
                logits_source = _forward_with_temporal_shift(
                    student, ps, ms, ts, es,
                    temporal_shift=source_to_target_shift,
                )
                logits_target = None
                if pseudo_count >= 2:
                    logits_target = _forward_with_temporal_shift(
                        student, pt[pseudo_mask], mt[pseudo_mask],
                        tt[pseudo_mask], et[pseudo_mask],
                    )
            elif pseudo_count:
                pixels = torch.cat((ps, pt[pseudo_mask]))
                masks = torch.cat((ms, mt[pseudo_mask]))
                positions = torch.cat((ts, tt[pseudo_mask]))
                extras = torch.cat((es, et[pseudo_mask]))
                source_shift = (
                    source_to_target_shift
                    if torch.is_tensor(source_to_target_shift)
                    else torch.full((ts.shape[0], 1), source_to_target_shift, device=ts.device, dtype=ts.dtype)
                )
                shifts = torch.cat((
                    source_shift,
                    torch.zeros((pseudo_count, 1), device=tt.device, dtype=tt.dtype),
                ))
                logits = _forward_with_temporal_shift(
                    student, pixels, masks, positions, extras, temporal_shift=shifts,
                )
                logits_source, logits_target = logits[:ps.shape[0]], logits[ps.shape[0]:]
            else:
                logits_source = _forward_with_temporal_shift(
                    student, ps, ms, ts, es,
                    temporal_shift=source_to_target_shift,
                )
                logits_target = None
            loss_source = criterion(logits_source, source_labels)
            loss_target = (
                criterion(logits_target, pseudo[pseudo_mask])
                if logits_target is not None else logits_source.sum() * 0
            )
            loss = loss_source + config.trade_off * loss_target
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            update_ema_variables(student, teacher, config.ema_decay)
            if global_step % config.log_step == 0:
                writer.add_scalar("train/loss_cls_source", loss_source.detach(), global_step)
                writer.add_scalar("train/loss_pseudo_target", loss_target.detach(), global_step)
                writer.add_scalar("train/loss_total", loss.detach(), global_step)
            labels_epoch.extend(target_weak["label"].tolist())
            pseudo_epoch.extend(pseudo.tolist())
            mask_epoch.extend(pseudo_mask.tolist())
            global_step += 1
        progress.close()

        selected = np.asarray(mask_epoch, dtype=bool)
        if selected.any():
            pseudo_f1 = sklearn.metrics.f1_score(
                np.asarray(labels_epoch)[selected],
                np.asarray(pseudo_epoch)[selected],
                average="macro", zero_division=0,
            )
            writer.add_scalar("train/pseudo_f1", pseudo_f1, epoch)
        previous_best = best_f1
        if config.run_validation:
            selected_model = student if config.output_student else teacher
            selected_model.eval()
            best_f1 = validation(
                best_f1, None, config, criterion, device, epoch,
                selected_model, val_loader, writer,
            )
        selected_model = student if config.output_student else teacher
        checkpoint = {
            "epoch": epoch,
            "state_dict": selected_model.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_temporal_shift": target_to_source_shift,
            "config": vars(config),
            "best_f1": best_f1,
        }
        fold_dir = getattr(config, "fold_dir", os.path.dirname(best_model_path) or ".")
        torch.save(checkpoint, os.path.join(fold_dir, "checkpoint_last.pt"))
        if best_f1 > previous_best or not os.path.isfile(best_model_path):
            torch.save(checkpoint, best_model_path)
            torch.save(checkpoint, os.path.join(fold_dir, "checkpoint_best.pt"))


def estimate_class_distribution(labels, num_classes):
    return np.bincount(labels, minlength=num_classes) / len(labels)

def kl_divergence(actual, estimated):
    return np.sum(actual * (np.log(actual + 1e-5) - np.log(estimated + 1e-5)))

@torch.no_grad()
def update_ema_variables(model, ema, decay=0.99):
    model_parameters = dict(model.named_parameters())
    for name, ema_value in ema.named_parameters():
        ema_value.copy_(decay * ema_value + (1. - decay) * model_parameters[name])
    model_buffers = dict(model.named_buffers())
    for name, ema_value in ema.named_buffers():
        if "prototype_bank" not in name:
            ema_value.copy_(model_buffers[name])

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

def _select_temporal_shift(
    shift_softmaxes,
    labels,
    shifts,
    shift_estimator,
    class_distribution,
    print_summary,
):
    shift_predictions = np.argmax(shift_softmaxes, axis=2)
    shift_acc_scores = (None if labels is None else np.asarray([
        (labels == predictions).mean()
        for predictions in np.moveaxis(shift_predictions, 0, 1)
    ]))
    if print_summary and shift_acc_scores is not None:
        print(
            f"Most accurate shift {shifts[np.argmax(shift_acc_scores)]} "
            f"with {np.max(shift_acc_scores):.3f}"
        )

    p_yx = shift_softmaxes
    p_y = shift_softmaxes.mean(axis=0)
    if shift_estimator == 'IS':
        scores = np.mean(
            np.sum(
                p_yx
                * (
                    np.log(p_yx + 1e-5)
                    - np.log(p_y[np.newaxis] + 1e-5)
                ),
                axis=2,
            ),
            axis=0,
        )
        best_shift_idx = np.argsort(scores)[::-1][0]
        summary_name = "Inception Score"
    elif shift_estimator == 'ENT':
        scores = -np.mean(
            np.sum(p_yx * np.log(p_yx + 1e-5), axis=2), axis=0
        )
        best_shift_idx = np.argsort(scores)[0]
        summary_name = "Entropy Score"
    elif shift_estimator == 'AM':
        assert class_distribution is not None, (
            'Target class distribution required to compute AM score'
        )
        one_hot_p_y = np.zeros_like(p_y)
        for i in range(len(shifts)):
            one_hot = np.zeros(
                (shift_softmaxes.shape[0], shift_softmaxes.shape[-1])
            )
            one_hot[np.arange(one_hot.shape[0]), shift_predictions[:, i]] = 1
            one_hot_p_y[i] = one_hot.mean(axis=0)
        kl_d = np.sum(
            class_distribution
            * (
                np.log(class_distribution + 1e-5)
                - np.log(one_hot_p_y + 1e-5)
            ),
            axis=1,
        )
        entropy = np.mean(
            np.sum(-p_yx * np.log(p_yx + 1e-5), axis=2), axis=0
        )
        scores = kl_d + entropy
        best_shift_idx = np.argsort(scores)[0]
        summary_name = "AM Score"
    elif shift_estimator == 'ACC':
        if shift_acc_scores is None:
            raise ValueError("accuracy shift selection requires labels")
        scores = shift_acc_scores
        best_shift_idx = np.argmax(scores)
        summary_name = "Accuracy"
    else:
        raise NotImplementedError(shift_estimator)

    best_shift = shifts[best_shift_idx]
    if print_summary and shift_estimator != 'ACC':
        message = f"Best {summary_name} shift {best_shift}"
        if shift_acc_scores is not None:
            message += f" with accuracy {shift_acc_scores[best_shift_idx]:.3f}"
        print(message)
    selected_probs = p_yx[:, best_shift_idx]
    selected_predictions = shift_predictions[:, best_shift_idx]
    diagnostics = {
        "selected_shift": best_shift,
        "mean_confidence": float(selected_probs.max(axis=1).mean()),
        "prediction_entropy": float(
            np.mean(
                np.sum(
                    -selected_probs * np.log(selected_probs + 1e-5), axis=1
                )
            )
        ),
        "num_predicted_classes": int(np.unique(selected_predictions).size),
        "score_range": float(np.max(scores) - np.min(scores)),
    }
    return best_shift, diagnostics

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
    shift_estimation_view='raw',
    shift_fourier_num_modes=13,
    shift_fourier_reg=1e-3,
    shift_fourier_period_days=365.0,
    shift_fourier_solver='dense_direct',
    compare_raw=False,
    return_diagnostics=False,
    include_label_diagnostics=True,
):
    shifts = list(range(min_shift, max_shift + 1))
    model.eval()
    if shift_estimation_view not in ('raw', 'fourier_recon'):
        raise ValueError(f"unsupported shift estimation view: {shift_estimation_view}")
    analyzer = synthesizer = None
    if shift_estimation_view == 'fourier_recon':
        if shift_fourier_solver != 'dense_direct':
            raise ValueError(
                f"unsupported shift Fourier solver: {shift_fourier_solver}"
            )
        analyzer = BatchedDirectFourierAnalyzer(
            num_modes=shift_fourier_num_modes,
            period_days=shift_fourier_period_days,
            reg=shift_fourier_reg,
        )
        synthesizer = BatchedDirectFourierSynthesizer(
            num_modes=shift_fourier_num_modes,
            period_days=shift_fourier_period_days,
        )
    if sample_size is None:
        sample_size = len(target_loader)

    target_iter = iter(target_loader)
    shift_softmaxes, raw_shift_softmaxes, labels = [], [], []
    for _ in tqdm(
        range(sample_size),
        desc=f'Estimating shift between [{min_shift}, {max_shift}]',
        disable=progress_bar_disabled(progress_bar),
    ):
        sample = next(target_iter)
        if include_label_diagnostics:
            labels.extend(sample['label'].tolist())
        pixels, valid_pixels, positions, extra = to_cuda(sample, device)
        spatial_feats = model.spatial_encoder.forward(pixels, valid_pixels, extra)
        raw_prepared = _prepare_temporal_features(model, spatial_feats, positions)
        prepared = raw_prepared
        if shift_estimation_view == 'fourier_recon':
            coeffs, _ = analyzer(prepared, positions)
            prepared = synthesizer(coeffs, positions)
        shift_logits = _classify_shift_grid(model, prepared, positions, shifts)
        shift_probs = F.softmax(shift_logits, dim=2)
        shift_softmaxes.append(shift_probs)
        if compare_raw:
            if shift_estimation_view == 'raw':
                raw_shift_softmaxes.append(shift_probs)
            else:
                raw_logits = _classify_shift_grid(
                    model, raw_prepared, positions, shifts,
                )
                raw_shift_softmaxes.append(F.softmax(raw_logits, dim=2))
    shift_softmaxes = torch.cat(shift_softmaxes).cpu().numpy()  # (N, n_shifts, n_classes)
    labels = np.array(labels) if include_label_diagnostics else None
    best_shift, diagnostics = _select_temporal_shift(
        shift_softmaxes,
        labels,
        shifts,
        shift_estimator,
        class_distribution,
        print_summary=True,
    )
    if compare_raw:
        raw_softmaxes = torch.cat(raw_shift_softmaxes).cpu().numpy()
        raw_shift, raw_diagnostics = _select_temporal_shift(
            raw_softmaxes,
            labels,
            shifts,
            shift_estimator,
            class_distribution,
            print_summary=False,
        )
        diagnostics["raw_selected_shift"] = raw_shift
        diagnostics["raw_score_range"] = raw_diagnostics["score_range"]
    if return_diagnostics:
        return best_shift, diagnostics
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
