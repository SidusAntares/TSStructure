import argparse
import json
import os
import pickle as pkl
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import (
    BalancedBatchSampler,
    PixelSetData,
    create_evaluation_loaders,
    create_train_loader,
    worker_init_fn,
)
from evaluation import evaluation
from methods.structure_da import (
    DeviceBatchLoader,
    DomainPhaseConfig,
    DomainShapeConfig,
    PhaseHypothesisScanConfig,
    SourceClassificationTrainer,
    SourcePrototypeBank,
    StableLabelConfig,
    Stage1Objective,
    Stage2EMATeacher,
    Stage2ObjectiveConfig,
    Stage2Trainer,
    Stage2TrainerConfig,
    TSStructureModel,
    build_source_prototype_bank,
    build_source_registration_prototypes,
    build_stage2_registration_extractor,
    configure_stage2_parameter_policy,
    create_feature_snapshot_manager,
    finalize_distance_statistics,
    run_stage2_training,
)
from transforms import Identity, Normalize, RandomSamplePixels, ToTensor
from utils import label_utils
from utils.metrics import overall_classification_report
from utils.train_utils import bool_flag


def load_structure_da_state_dict(model, state_dict):
    """Load current checkpoints and reject removed global-alignment weights clearly."""

    legacy_keys = sorted(key for key in state_dict if key.startswith("alignment."))
    if legacy_keys:
        raise RuntimeError(
            "checkpoint is incompatible: it contains the removed global fused-feature "
            "domain alignment branch (alignment.*); retrain with the current model"
        )
    model.load_state_dict(state_dict)


def create_source_train_loader(config, splits):
    """Create the labelled source training loader, balanced when requested."""

    source_transform = transforms.Compose(
        [RandomSamplePixels(config.num_pixels), Normalize(), ToTensor()]
    )
    source_dataset = PixelSetData(
        data_root=config.data_root,
        dataset_name=config.source,
        classes=config.classes,
        transform=source_transform,
        indices=splits[config.source]["train"],
        with_extra=False,
        closed_set=config.closed_set,
        combine_spring_and_winter=config.combine_spring_and_winter,
        time_coordinate_mode=getattr(
            config, "time_coordinate_mode", "canonical_day_of_year"
        ),
    )
    if getattr(config, "balance_source", True):
        return DataLoader(
            dataset=source_dataset,
            batch_sampler=BalancedBatchSampler(
                source_dataset.get_labels(), config.batch_size, seed=config.seed
            ),
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )
    return create_train_loader(source_dataset, config.batch_size, config.num_workers)


def create_source_scan_loader(config, splits):
    """Deterministic full-source loader for prototype scans.

    Unlike the training loader this uses no random pixel sampling, no shuffle,
    no balancing and no drop-last, so every source training sample is seen
    exactly once in a fixed order. Batches are grouped by parcel pixel
    dimension so variable-width pixel sets still collate.
    """

    from dataset import GroupByShapesBatchSampler

    scan_transform = transforms.Compose([Identity(), Normalize(), ToTensor()])
    scan_dataset = PixelSetData(
        data_root=config.data_root,
        dataset_name=config.source,
        classes=config.classes,
        transform=scan_transform,
        indices=splits[config.source]["train"],
        with_extra=False,
        closed_set=config.closed_set,
        combine_spring_and_winter=config.combine_spring_and_winter,
        time_coordinate_mode=getattr(
            config, "time_coordinate_mode", "canonical_day_of_year"
        ),
    )
    scan_batch_size = getattr(config, "eval_batch_size", None) or config.batch_size
    return DataLoader(
        dataset=scan_dataset,
        batch_sampler=GroupByShapesBatchSampler(
            scan_dataset, scan_batch_size, by_pixel_dim=True
        ),
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )


def create_target_statistics_loader(config, splits):
    """Deterministic full target-train loader for Stage-2 statistics only."""
    from dataset import GroupByShapesBatchSampler

    scan_transform = transforms.Compose([Identity(), Normalize(), ToTensor()])
    scan_dataset = PixelSetData(
        data_root=config.data_root,
        dataset_name=config.target,
        classes=config.classes,
        transform=scan_transform,
        indices=splits[config.target]["train"],
        with_extra=False,
        closed_set=config.closed_set,
        combine_spring_and_winter=config.combine_spring_and_winter,
        time_coordinate_mode=getattr(
            config, "time_coordinate_mode", "canonical_day_of_year"
        ),
    )
    scan_batch_size = getattr(config, "eval_batch_size", None) or config.batch_size
    return DataLoader(
        dataset=scan_dataset,
        batch_sampler=GroupByShapesBatchSampler(
            scan_dataset, scan_batch_size, by_pixel_dim=True
        ),
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )


def _apply_stage2_config_file(config) -> None:
    path = getattr(config, "stage2_config", None)
    if not path:
        return
    with open(path, "r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("--stage2_config must contain a JSON object")
    allowed = {name for name in vars(config) if name.startswith("stage2_")}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            "unknown Stage-2 configuration keys: " + ", ".join(unknown)
        )
    for name, value in values.items():
        if getattr(config, name) is None:
            setattr(config, name, value)


def _missing_stage2_configuration(config) -> list[str]:
    required = (
        "stage2_registration_lambda",
        "stage2_registration_gain_ratio_max",
        "stage2_registration_min_common_support",
        "stage2_registration_max_roughness",
        "stage2_registration_min_increment",
        "stage2_registration_max_local_speed",
        "stage2_registration_max_deviation",
        "stage2_class_hypothesis_margin",
        "stage2_phase_min_samples_per_class",
        "stage2_phase_class_dispersion_max",
        "stage2_phase_class_diameter_max",
        "stage2_phase_group_dispersion_max",
        "stage2_phase_group_diameter_max",
        "stage2_phase_group_core_separation",
        "stage2_phase_global_radius",
        "stage2_phase_confirmation_patience",
        "stage2_phase_center_drift_max",
        "stage2_stable_tau_f",
        "stage2_stable_tau_q",
        "stage2_shape_min_valid_classes",
        "stage2_shape_min_samples_per_class",
        "stage2_shape_shared_ratio_min",
        "stage2_shape_leave_one_out_drift_max",
        "stage2_shape_center_drift_max",
        "stage2_shape_effect_norm_max",
        "stage2_shape_confirmation_patience",
        "stage2_lambda_src_proto",
        "stage2_lambda_src_cons",
        "stage2_lambda_syn",
        "stage2_lambda_syn_cons",
        "stage2_objective_tau_q",
        "stage2_fused_margin",
        "stage2_ema_decay",
        "stage2_lambda_delta",
    )
    missing = [name for name in required if getattr(config, name, None) is None]
    gate_pairs = (
        ("classifier gate", "stage2_cls_confidence_min", "stage2_cls_margin_min"),
        ("fused gate", "stage2_fused_confidence_min", "stage2_fused_margin_min"),
        ("q gate", "stage2_q_confidence_min", "stage2_q_margin_min"),
    )
    for label, confidence, margin in gate_pairs:
        if getattr(config, confidence, None) is None and getattr(config, margin, None) is None:
            missing.append(label + f" ({confidence} or {margin})")
    return missing


def build_stage2_config(config) -> Stage2TrainerConfig:
    missing = _missing_stage2_configuration(config)
    if missing:
        raise ValueError(
            "missing Stage-2 statistical configuration: " + ", ".join(missing)
        )
    return Stage2TrainerConfig(
        phase_scan=PhaseHypothesisScanConfig(
            registration_lambda=config.stage2_registration_lambda,
            registration_gain_ratio_max=config.stage2_registration_gain_ratio_max,
            registration_min_common_support=config.stage2_registration_min_common_support,
            registration_max_roughness=config.stage2_registration_max_roughness,
            registration_min_increment=config.stage2_registration_min_increment,
            registration_max_local_speed=config.stage2_registration_max_local_speed,
            registration_max_deviation=config.stage2_registration_max_deviation,
            class_hypothesis_margin=config.stage2_class_hypothesis_margin,
        ),
        phase=DomainPhaseConfig(
            phase_min_samples_per_class=config.stage2_phase_min_samples_per_class,
            phase_class_dispersion_max=config.stage2_phase_class_dispersion_max,
            phase_class_diameter_max=config.stage2_phase_class_diameter_max,
            phase_group_dispersion_max=config.stage2_phase_group_dispersion_max,
            phase_group_diameter_max=config.stage2_phase_group_diameter_max,
            phase_group_core_separation=config.stage2_phase_group_core_separation,
            phase_global_radius=config.stage2_phase_global_radius,
            phase_confirmation_patience=config.stage2_phase_confirmation_patience,
            phase_center_drift_max=config.stage2_phase_center_drift_max,
        ),
        stable_labels=StableLabelConfig(
            tau_f=config.stage2_stable_tau_f,
            tau_q=config.stage2_stable_tau_q,
            cls_confidence_min=config.stage2_cls_confidence_min,
            cls_margin_min=config.stage2_cls_margin_min,
            fused_confidence_min=config.stage2_fused_confidence_min,
            fused_margin_min=config.stage2_fused_margin_min,
            q_confidence_min=config.stage2_q_confidence_min,
            q_margin_min=config.stage2_q_margin_min,
        ),
        shape=DomainShapeConfig(
            shape_min_valid_classes=config.stage2_shape_min_valid_classes,
            shape_min_samples_per_class=config.stage2_shape_min_samples_per_class,
            shape_shared_ratio_min=config.stage2_shape_shared_ratio_min,
            shape_leave_one_out_drift_max=config.stage2_shape_leave_one_out_drift_max,
            shape_center_drift_max=config.stage2_shape_center_drift_max,
            shape_effect_norm_max=config.stage2_shape_effect_norm_max,
            shape_confirmation_patience=config.stage2_shape_confirmation_patience,
        ),
        objective=Stage2ObjectiveConfig(
            lambda_src_proto=config.stage2_lambda_src_proto,
            lambda_src_cons=config.stage2_lambda_src_cons,
            lambda_syn=config.stage2_lambda_syn,
            lambda_syn_cons=config.stage2_lambda_syn_cons,
            tau_q=config.stage2_objective_tau_q,
            fused_margin=config.stage2_fused_margin,
        ),
        ema_decay=config.stage2_ema_decay,
        lambda_delta=config.stage2_lambda_delta,
        total_epochs=config.stage2_epochs,
        adaptation_block_epochs=config.stage2_block_epochs,
        steps_per_epoch=config.stage2_steps_per_epoch,
        amp_enabled=bool(getattr(config, "amp", False)),
        amp_dtype=getattr(config, "amp_dtype", "float16"),
    )


def _comma_separated_ints(value: str) -> list[int]:
    if isinstance(value, str):
        items = [int(part) for part in value.split(",") if part]
        if not items:
            raise argparse.ArgumentTypeError("expected a comma-separated integer list")
        return items
    raise argparse.ArgumentTypeError("expected a comma-separated integer list")


def _int_list(value):
    if isinstance(value, list):
        return value
    return _comma_separated_ints(value)


def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    indices, _ = prepare_data_protocol(config)
    folds = create_train_val_test_folds(
        [config.source, config.target],
        config.num_folds,
        indices,
        config.val_ratio,
        config.test_ratio,
    )
    print("SPLIT_PROTOCOL|name=random_parcel_split")

    if config.overall:
        overall_performance(config)
        return

    stage2_runtime_config = None if config.eval else build_stage2_config(config)
    all_folds_have_formal_test = True

    for fold_num, splits in enumerate(folds):
        print(f'Starting fold {fold_num}...')

        if config.closed_set:
            print_closed_set_counts(config, indices, splits)

        config.fold_dir = os.path.join(config.output_dir, f'fold_{fold_num}')
        config.fold_num = fold_num

        sample_pixels_val = config.sample_pixels_val
        if config.eval:
            _, target_test_loader = create_evaluation_loaders(
                config.target, splits, config, sample_pixels_val
            )
            source_val_loader = None
            target_val_loader = None
            target_statistics_loader = None
        else:
            source_val_loader, _ = create_evaluation_loaders(
                config.source, splits, config, sample_pixels_val
            )
            target_val_loader, target_test_loader = create_evaluation_loaders(
                config.target, splits, config, sample_pixels_val
            )
            target_statistics_loader = DeviceBatchLoader(
                create_target_statistics_loader(config, splits), device
            )

        source_loader = None
        source_scan_loader = None
        if not config.eval:
            source_loader = create_source_train_loader(config, splits)
            source_scan_loader = DeviceBatchLoader(
                create_source_scan_loader(config, splits), device
            )
            eval_batch_size = config.eval_batch_size or config.batch_size
            amp_enabled = bool(
                getattr(config, "amp", False)
                and (
                    device.type == "cuda"
                    or getattr(config, "amp_dtype", "float16") == "bfloat16"
                )
            )
            print(
                "TRAIN_PROTOCOL|"
                f"batch_size={config.batch_size}"
                f"|eval_batch_size={eval_batch_size}"
                f"|source_loader_steps={len(source_loader)}"
                f"|stage1_epochs={config.stage1_epochs}"
                f"|stage2_epochs={config.stage2_epochs}"
                f"|stage2_block_epochs={config.stage2_block_epochs}"
                f"|warmup_epochs={config.source_warmup_epochs}"
                f"|amp={str(amp_enabled).lower()}"
                f"|amp_dtype={getattr(config, 'amp_dtype', 'float16')}"
            )

        model = TSStructureModel(
            num_classes=config.num_classes,
            input_dim=config.input_dim,
            with_extra=config.with_extra,
            time_reference=getattr(config, "time_reference", 0.0),
            time_scale=config.time_scale,
            tau_fast_init=config.tau_fast_init,
            tau_slow_init=config.tau_slow_init,
            tau_min=config.tau_min,
            delta_tau_min=config.delta_tau_min,
            trend_num_basis=config.trend_num_basis,
            structure_num_basis=config.structure_num_basis,
            canonical_grid_size=config.canonical_grid_size,
            roughness_grid_size=config.roughness_grid_size,
            trend_smoothing=config.trend_smoothing,
            structure_smoothing=config.structure_smoothing,
            n_head=config.n_head,
            d_k=config.d_k,
            d_model=config.d_model,
            ltae_mlp=_int_list(config.ltae_mlp),
            dropout=config.dropout,
            classifier_hidden=_int_list(config.classifier_hidden),
            max_initial_frequency=config.time2vec_max_frequency,
        )
        model.to(device)

        if config.eval:
            stage2_last = os.path.join(config.fold_dir, "stage2_last_ema.pt")
            fallback = os.path.join(config.fold_dir, "model.pt")
            checkpoint_path = stage2_last if os.path.isfile(stage2_last) else fallback
            print(f"Restoring evaluation model from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, weights_only=False)["state_dict"]
            load_structure_da_state_dict(model, state_dict)
            test_metrics = evaluation(
                model,
                target_test_loader,
                device,
                config.classes,
                criterion=torch.nn.CrossEntropyLoss(),
                mode='test',
                progress_bar=getattr(config, "progress_bar", "auto"),
            )
            save_results(test_metrics, config)
            continue

        from torch.utils.tensorboard import SummaryWriter

        assert source_loader is not None
        assert source_scan_loader is not None
        assert source_val_loader is not None
        assert target_statistics_loader is not None
        assert target_val_loader is not None
        assert stage2_runtime_config is not None

        print(model)
        print('Number of Stage-1 trainable parameters:', get_num_trainable_params(model))
        writer = SummaryWriter(
            log_dir=f'{config.tensorboard_log_dir}_fold{fold_num}', purge_step=0
        )
        feature_snapshot_manager = create_feature_snapshot_manager(
            model, config, splits, device=device
        )
        best_model_path = os.path.join(config.fold_dir, 'model.pt')
        train_source_classification(
            model, source_loader, source_val_loader,
            config, writer, device, best_model_path,
            feature_snapshot_manager,
            source_scan_loader=source_scan_loader,
        )

        # Formal Stage-1 -> Stage-2 boundary: select source-val best, then
        # recompute the final frozen full-source geometry/statistics once.
        stage1_best_path = os.path.join(config.fold_dir, "stage1_best.pt")
        stage1_checkpoint = torch.load(stage1_best_path, weights_only=False)
        load_structure_da_state_dict(model, stage1_checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        source_bank = build_source_prototype_bank(
            model, source_scan_loader, config.num_classes, device=device
        )
        source_bank, _ = finalize_distance_statistics(
            model, source_scan_loader, source_bank, device=device
        )
        reg_extractor = build_stage2_registration_extractor(
            model, device=device, k_reg=stage2_runtime_config.phase_scan.k_reg
        )
        source_registration_bank = build_source_registration_prototypes(
            model,
            source_scan_loader,
            config.num_classes,
            device=device,
            reg_extractor=reg_extractor,
        )

        policy = configure_stage2_parameter_policy(model)
        named_parameters = dict(model.named_parameters())
        stage2_parameters = [
            named_parameters[name] for name in policy.trainable_parameter_names
        ]
        stage2_optimizer = torch.optim.Adam(
            stage2_parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        ema_teacher = Stage2EMATeacher.from_student(
            model, policy, decay=stage2_runtime_config.ema_decay
        )
        stage2_trainer = Stage2Trainer(
            student=model,
            policy=policy,
            ema_teacher=ema_teacher,
            optimizer=stage2_optimizer,
            source_loader=source_loader,
            source_scan_loader=source_scan_loader,
            target_statistics_loader=target_statistics_loader,
            source_prototype_bank=source_bank,
            source_registration_bank=source_registration_bank,
            reg_extractor=reg_extractor,
            config=stage2_runtime_config,
            device=device,
            output_dir=config.fold_dir,
            runtime_config=dict(vars(config)),
            writer=writer,
        )

        def evaluate_target_val(teacher_model, epoch):
            metrics = evaluation(
                teacher_model,
                target_val_loader,
                device,
                config.classes,
                criterion=torch.nn.CrossEntropyLoss(),
                mode='val',
                progress_bar=getattr(config, "progress_bar", "auto"),
            )
            writer.add_scalar("stage2/target_val_accuracy", metrics["accuracy"], epoch)
            writer.add_scalar("stage2/target_val_macro_f1", metrics["macro_f1"], epoch)
            print(
                f"STAGE2_TARGET_VAL|epoch={epoch}|accuracy={metrics['accuracy']:.4f}"
                f"|macro_f1={metrics['macro_f1']:.4f}"
            )
            return metrics

        def evaluate_target_test(teacher_model, epoch):
            metrics = evaluation(
                teacher_model,
                target_test_loader,
                device,
                config.classes,
                criterion=torch.nn.CrossEntropyLoss(),
                mode='test',
                progress_bar=getattr(config, "progress_bar", "auto"),
            )
            print(
                f"DIAGNOSTIC_TARGET_TEST|epoch={epoch}|accuracy={metrics['accuracy']:.4f}"
                f"|macro_f1={metrics['macro_f1']:.4f}"
            )
            return metrics

        stage2_result = run_stage2_training(
            stage2_trainer,
            evaluate_target_val=evaluate_target_val,
            evaluate_target_test=evaluate_target_test,
        )
        writer.close()

        if stage2_result.final_diagnostic_target_test is not None:
            test_metrics = stage2_result.final_diagnostic_target_test
            print(
                f"Final diagnostic test for {config.experiment_name}: "
                f"accuracy={test_metrics['accuracy']:.4f}, "
                f"f1={test_metrics['macro_f1']:.4f}"
            )
            print(test_metrics['classification_report'])
            save_results(test_metrics, config)
        else:
            all_folds_have_formal_test = False
            print(
                "STAGE2_SMOKE_COMPLETE|no formal epoch-20/40/60 target test was run"
            )

    if all_folds_have_formal_test:
        overall_performance(config)


def train_source_classification(
    model,
    source_loader,
    source_val_loader,
    config,
    writer,
    device,
    best_model_path,
    feature_snapshot_manager=None,
    source_scan_loader=None,
):
    epochs = config.stage1_epochs
    warmup_epochs = config.source_warmup_epochs
    steps_per_epoch = getattr(config, "steps_per_epoch", None)
    if steps_per_epoch is None or steps_per_epoch <= 0:
        steps_per_epoch = len(source_loader)
    if isinstance(steps_per_epoch, bool) or steps_per_epoch < 1:
        raise ValueError("--steps_per_epoch must be a positive integer or None")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=0
    )
    amp_enabled = bool(
        getattr(config, "amp", False)
        and (
            device.type == "cuda"
            or getattr(config, "amp_dtype", "float16") == "bfloat16"
        )
    )
    objective = Stage1Objective(
        num_classes=config.num_classes,
        lambda_q=config.lambda_q,
        lambda_f=config.lambda_f,
        lambda_q_to_cls=config.lambda_q_to_cls,
        margin_q=config.margin_q,
        margin_f=config.margin_f,
        tau_q=config.tau_q,
    )
    trainer = SourceClassificationTrainer(
        model,
        optimizer,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=getattr(config, "amp_dtype", "float16"),
        objective=objective,
    )

    best_f1 = float("-inf")
    bank: SourcePrototypeBank | None = None
    bank_version = 0
    stage1_dir = config.fold_dir
    tmp_best_path = os.path.join(stage1_dir, "stage1_best_model_tmp.pt")

    # A non-warmup epoch must always start with a bank produced by a
    # deterministic full-source scan.  The normal protocol builds the first
    # bank at the end of the final warmup epoch; if warmup is disabled, build
    # it once before epoch 1 instead.
    if epochs > 0 and warmup_epochs <= 0:
        if source_scan_loader is None:
            raise RuntimeError("source_scan_loader is required for prototype refresh")
        print("PROTOTYPE_REFRESH|epoch=0")
        bank = build_source_prototype_bank(
            model,
            source_scan_loader,
            config.num_classes,
            device=device,
        )
        bank_version += 1
        print(
            f"PROTOTYPE_READY|epoch=0|version={bank_version}"
            f"|ready_classes={len(bank.ready_classes())}"
        )

    for epoch in range(epochs):
        warmup = epoch < warmup_epochs
        model.train()
        meters = defaultdict(float)
        steps = 0
        for batch in source_loader:
            if steps >= steps_per_epoch:
                break
            metrics = trainer.train_step(batch, warmup=warmup, bank=bank)
            for name, value in metrics.items():
                meters[name] += value
            steps += 1
            if steps % config.log_step == 0:
                avg = {name: value / steps for name, value in meters.items()}
                print(
                    f"TRAIN_STEP|epoch={epoch + 1}/{epochs}|step={steps}/{steps_per_epoch}"
                    f"|warmup={str(warmup).lower()}"
                    f"|loss={avg['loss']:.4f}|cls={avg['classification_loss']:.4f}"
                    f"|q_proto={avg['q_proto_loss']:.4f}|f_proto={avg['f_proto_loss']:.4f}"
                    f"|q_to_cls={avg['q_to_cls_loss']:.4f}|accuracy={avg['accuracy']:.4f}"
                )
        if steps == 0:
            raise RuntimeError("source training loader produced no batches")
        averages = {name: value / steps for name, value in meters.items()}
        print(
            f"TRAIN_EPOCH|epoch={epoch + 1}/{epochs}|steps={steps}"
            f"|warmup={str(warmup).lower()}"
            f"|loss={averages['loss']:.4f}|cls={averages['classification_loss']:.4f}"
            f"|q_proto={averages['q_proto_loss']:.4f}|f_proto={averages['f_proto_loss']:.4f}"
            f"|q_to_cls={averages['q_to_cls_loss']:.4f}"
            f"|q_valid={int(averages['q_valid_count'])}"
            f"|f_valid={int(averages['f_valid_count'])}"
            f"|consistency_valid={int(averages['consistency_valid_count'])}"
            f"|accuracy={averages['accuracy']:.4f}"
        )
        for name, value in averages.items():
            writer.add_scalar(f"train/{name}", value, epoch)
        ready_classes = 0 if bank is None else len(bank.ready_classes())
        print(
            f"STAGE1_EPOCH|epoch={epoch + 1}/{epochs}|warmup={str(warmup).lower()}"
            f"|prototype_ready_classes={ready_classes}|prototype_version={bank_version}"
        )
        scheduler.step()
        if feature_snapshot_manager is not None:
            snapshot_result = feature_snapshot_manager.capture(epoch + 1)
            if (
                writer is not None
                and snapshot_result is not None
                and snapshot_result.status == "FAILED"
            ):
                writer.add_scalar("diagnostics/feature_snapshot_failed", 1, epoch)

        model.eval()
        best_f1, metrics = _source_validation(
            best_f1, best_model_path, config, device, epoch, model, source_val_loader, writer
        )
        if metrics["macro_f1"] >= best_f1:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "source_val": {
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                    },
                },
                tmp_best_path,
            )

        # Refresh at epoch boundaries only.  In particular, the final warmup
        # epoch must produce the bank consumed by the first non-warmup epoch.
        # Every subsequent non-warmup epoch refreshes the bank for the next
        # epoch; no mini-batch ever mutates or replaces it.
        next_epoch_is_non_warmup = (epoch + 1) >= warmup_epochs
        if epoch < epochs - 1 and next_epoch_is_non_warmup:
            if source_scan_loader is None:
                raise RuntimeError("source_scan_loader is required for prototype refresh")
            print(f"PROTOTYPE_REFRESH|epoch={epoch + 1}")
            bank = build_source_prototype_bank(
                model,
                source_scan_loader,
                config.num_classes,
                device=device,
            )
            bank_version += 1
            print(
                f"PROTOTYPE_READY|epoch={epoch + 1}|version={bank_version}"
                f"|ready_classes={len(bank.ready_classes())}"
            )

    _finalize_stage1_checkpoints(
        model,
        optimizer,
        scheduler,
        trainer.scaler,
        config,
        device,
        tmp_best_path,
        source_scan_loader,
    )
    return best_f1


def _finalize_stage1_checkpoints(
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    device,
    tmp_best_path,
    source_scan_loader,
):
    """Produce stage1_best.pt and stage1_last.pt with full prototype state."""

    def emit_checkpoint(path: str, state: dict, epoch: int) -> None:
        torch.save(state, path)
        print(f"STAGE1_CHECKPOINT|path={path}|epoch={epoch}")

    def build_prototype_state(model_state: dict, epoch: int, source_val: dict) -> dict:
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        bank = build_source_prototype_bank(
            model, source_scan_loader, config.num_classes, device=device
        )
        bank, examples = finalize_distance_statistics(
            model, source_scan_loader, bank, device=device
        )
        return {
            "stage": "stage1",
            "epoch": epoch,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "source_val": {
                "accuracy": source_val.get("accuracy", 0.0),
                "macro_f1": source_val.get("macro_f1", 0.0),
            },
            "prototype_bank": {
                "trend_srvf": bank.trend_srvf.detach().cpu(),
                "shape_srvf": bank.shape_srvf.detach().cpu(),
                "trend_support": bank.trend_support.detach().cpu(),
                "shape_support": bank.shape_support.detach().cpu(),
                "fused": bank.fused.detach().cpu(),
                "class_counts": bank.class_counts.detach().cpu(),
                "ready": bank.ready.detach().cpu(),
                "q_distance_samples": tuple(v.detach().cpu() for v in bank.q_distance_samples),
                "f_distance_samples": tuple(v.detach().cpu() for v in bank.f_distance_samples),
                "q_quantiles": bank.q_quantiles.detach().cpu(),
                "f_quantiles": bank.f_quantiles.detach().cpu(),
                "version": bank.version,
            },
            "shape_examples": _to_cpu_examples(examples),
            "stage1_config": _stage1_config_dict(config),
        }

    stage1_dir = config.fold_dir
    last_path = os.path.join(stage1_dir, "stage1_last.pt")
    best_path = os.path.join(stage1_dir, "stage1_best.pt")
    last_model_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }

    if os.path.isfile(tmp_best_path):
        best_state = torch.load(tmp_best_path, weights_only=False)
        best_checkpoint = build_prototype_state(
            best_state["state_dict"],
            best_state["epoch"],
            best_state["source_val"],
        )
        emit_checkpoint(best_path, best_checkpoint, best_state["epoch"])

    last_state = {
        "epoch": config.stage1_epochs - 1,
        "state_dict": last_model_state,
        "best_f1": 0.0,
        "source_val": {"accuracy": 0.0, "macro_f1": 0.0},
    }
    last_checkpoint = build_prototype_state(
        last_state["state_dict"],
        last_state["epoch"],
        last_state["source_val"],
    )
    emit_checkpoint(last_path, last_checkpoint, last_state["epoch"])

    if os.path.isfile(tmp_best_path):
        os.remove(tmp_best_path)


def _to_cpu_examples(examples: list[dict]) -> list[dict]:
    out = []
    for example in examples:
        item = dict(example)
        for key in ("q_shape", "support", "canonical_grid", "original_positions"):
            value = item.get(key)
            if isinstance(value, torch.Tensor):
                item[key] = value.detach().cpu()
        out.append(item)
    return out


def _stage1_config_dict(config) -> dict:
    names = (
        "stage1_epochs",
        "source_warmup_epochs",
        "lambda_q",
        "lambda_f",
        "lambda_q_to_cls",
        "margin_q",
        "margin_f",
        "tau_q",
        "num_classes",
        "canonical_grid_size",
    )
    return {name: getattr(config, name, None) for name in names}


def _source_validation(
    best_f1, best_model_path, config, device, epoch, model, val_loader, writer
):
    val_metrics = evaluation(
        model,
        val_loader,
        device,
        config.classes,
        torch.nn.CrossEntropyLoss(),
        mode='val',
        progress_bar=getattr(config, "progress_bar", "auto"),
    )
    val_loss, val_acc, val_f1, val_kappa = (
        val_metrics['loss'],
        val_metrics['accuracy'],
        val_metrics['macro_f1'],
        val_metrics['kappa'],
    )
    writer.add_scalar('val/loss', val_loss, global_step=epoch)
    writer.add_scalar('val/accuracy', val_acc, global_step=epoch)
    writer.add_scalar('val/f1', val_f1, global_step=epoch)
    writer.add_scalar('val/kappa', val_kappa, global_step=epoch)
    print(f"Validation result: loss={val_loss:.4f}, acc={val_acc:.2f}, f1={val_f1:.4f}")
    if val_f1 > best_f1:
        print(f'Validation F1 improved from {best_f1:.4f} to {val_f1:.4f}!')
        best_f1 = val_f1
        if best_model_path is not None:
            print(f'Saving best model to {best_model_path}')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'best_f1': best_f1}, best_model_path)
    else:
        print(f'Validation F1 did not improve from {best_f1:.4f}.')
    return best_f1, val_metrics


def prepare_data_protocol(config):
    dataset_time_options = (
        {"time_coordinate_mode": config.time_coordinate_mode}
        if hasattr(config, "time_coordinate_mode")
        else {}
    )
    candidate_classes = label_utils.get_classes(
        config.source.split('/')[0],
        combine_spring_and_winter=config.combine_spring_and_winter,
    )

    if not config.closed_set:
        source_data = PixelSetData(
            config.data_root,
            config.source,
            candidate_classes,
            combine_spring_and_winter=config.combine_spring_and_winter,
            **dataset_time_options,
        )
        labels, counts = np.unique(source_data.get_labels(), return_counts=True)
        source_classes = [
            candidate_classes[int(label)]
            for label, count in zip(labels, counts)
            if count >= 200
        ]
        print('Using classes:', source_classes)
        config.classes = source_classes
        config.num_classes = len(source_classes)
        target_data = PixelSetData(
            config.data_root,
            config.target,
            source_classes,
            combine_spring_and_winter=config.combine_spring_and_winter,
            **dataset_time_options,
        )
        return {
            config.source: len(source_data),
            config.target: len(target_data),
        }, None

    candidate_classes = [
        class_name
        for class_name in candidate_classes
        if class_name != "unknown"
    ]
    candidate_source_dataset = PixelSetData(
        config.data_root,
        config.source,
        candidate_classes,
        closed_set=True,
        combine_spring_and_winter=config.combine_spring_and_winter,
        **dataset_time_options,
    )
    labels, counts = np.unique(
        candidate_source_dataset.get_labels(), return_counts=True
    )
    source_classes = [
        candidate_classes[int(label)]
        for label, count in zip(labels, counts)
        if count >= 200
    ]
    if not source_classes:
        raise ValueError(
            f"No source classes in {config.source} have at least 200 samples"
        )

    source_count_by_class = {
        candidate_classes[int(label)]: int(count)
        for label, count in zip(labels, counts)
        if count >= 200
    }
    config.classes = source_classes
    config.num_classes = len(source_classes)

    source_protocol_dataset = PixelSetData(
        config.data_root,
        config.source,
        config.classes,
        closed_set=True,
        combine_spring_and_winter=config.combine_spring_and_winter,
        **dataset_time_options,
    )
    target_protocol_dataset = PixelSetData(
        config.data_root,
        config.target,
        config.classes,
        closed_set=True,
        combine_spring_and_winter=config.combine_spring_and_winter,
        **dataset_time_options,
    )
    eligible_indices = {
        config.source: source_protocol_dataset.get_parcel_indices().tolist(),
        config.target: target_protocol_dataset.get_parcel_indices().tolist(),
    }
    protocol = {
        "closed_set": True,
        "source": config.source,
        "target": config.target,
        "min_source_samples_per_class": 200,
        "combine_spring_and_winter": config.combine_spring_and_winter,
        "time_reference": getattr(config, "time_reference", 0.0),
        "time_scale": getattr(config, "time_scale", 365.0),
        "time_coordinate_mode": getattr(
            config, "time_coordinate_mode", "canonical_day_of_year"
        ),
        "classes": config.classes,
        "class_to_idx": {
            class_name: index for index, class_name in enumerate(config.classes)
        },
        "source_class_counts": source_count_by_class,
        "eligible_source_samples": len(eligible_indices[config.source]),
        "eligible_target_samples": len(eligible_indices[config.target]),
        "seed": config.seed,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
    }

    print(
        "CLOSED_SET_PROTOCOL|"
        f"source={config.source}|target={config.target}|"
        f"num_classes={config.num_classes}|classes={','.join(config.classes)}"
    )
    with open(os.path.join(config.output_dir, "closed_set_protocol.json"), "w") as f:
        json.dump(protocol, f, indent=4)

    return eligible_indices, protocol


def print_closed_set_counts(config, eligible_indices, splits):
    print(
        "CLOSED_SET_COUNTS|"
        f"source_total={len(eligible_indices[config.source])}|"
        f"target_total={len(eligible_indices[config.target])}|"
        f"source_train={len(splits[config.source]['train'])}|"
        f"source_val={len(splits[config.source]['val'])}|"
        f"source_test={len(splits[config.source]['test'])}|"
        f"target_train={len(splits[config.target]['train'])}|"
        f"target_val={len(splits[config.target]['val'])}|"
        f"target_test={len(splits[config.target]['test'])}"
    )


def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataset_size(data_root, dataset):
    dir = os.path.join(data_root, dataset)
    return len([name for name in os.listdir(os.path.join(dir, 'data')) if name.endswith('.zarr')])


def create_train_val_test_folds(datasets, num_folds, num_indices, val_ratio=0.1, test_ratio=0.2):
    folds = []
    for _ in range(num_folds):
        splits = {}
        for dataset in datasets:
            if isinstance(num_indices, dict):
                index_spec = num_indices[dataset]
            else:
                index_spec = num_indices
            if isinstance(index_spec, (int, np.integer)):
                indices = list(range(int(index_spec)))
            else:
                indices = list(index_spec)
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val

            random.shuffle(indices)

            train_indices = set(indices[:n_train])
            val_indices = set(indices[n_train:n_train + n_val])
            test_indices = set(indices[n_train + n_val:])
            assert train_indices.isdisjoint(val_indices)
            assert train_indices.isdisjoint(test_indices)
            assert val_indices.isdisjoint(test_indices)
            assert train_indices | val_indices | test_indices == set(indices)

            splits[dataset] = {'train': train_indices, 'val': val_indices, 'test': test_indices}
        folds.append(splits)
    return folds


def save_results(metrics, config):
    out_dir = config.fold_dir
    metrics = deepcopy(metrics)
    conf_mat = metrics.pop('confusion_matrix')
    class_report = metrics.pop('classification_report')
    target_name = str(config.target).replace('/', '_')

    with open(os.path.join(out_dir, f'test_metrics_{target_name}.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
    with open(os.path.join(out_dir, f'class_report_{target_name}.txt'), 'w') as outfile:
        outfile.write(str(class_report))
    pkl.dump(conf_mat, open(os.path.join(out_dir, f'conf_mat_{target_name}.pkl'), 'wb'))


def overall_performance(config):
    overall_metrics = defaultdict(list)
    target_name = str(config.target).replace("/", "_")

    cms = []
    for fold in range(config.num_folds):
        fold_dir = os.path.join(config.output_dir, f'fold_{fold}')
        test_metrics = json.load(open(os.path.join(fold_dir, f'test_metrics_{target_name}.json')))
        for metric, value in test_metrics.items():
            overall_metrics[metric].append(value)
        cm = pkl.load(open(os.path.join(fold_dir, f'conf_mat_{target_name}.pkl'), 'rb'))
        cms.append(cm)

    for i, row in enumerate(np.mean(cms, axis=0)):
        print(config.classes[i], row.astype(int))

    print(f'Overall result across {config.num_folds} folds:')
    print(overall_classification_report(cms, config.classes))
    for metric, values in overall_metrics.items():
        values = np.array(values)
        if metric == 'loss':
            print(f"{metric}: {np.mean(values):.4}±{np.std(values):.4}")
        else:
            values *= 100
            print(f"{metric}: {np.mean(values):.1f}±{np.std(values):.1f}")

    with open(os.path.join(config.output_dir, f'overall_{target_name}.json'), 'w') as file:
        file.write(json.dumps(overall_metrics, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Setup parameters
    parser.add_argument('--data_root', default='/data/user/dataset/timematch_data', type=str,
                        help='Path to datasets root directory')
    source_balance = parser.add_mutually_exclusive_group()
    source_balance.add_argument(
        '--balance-source', dest='balance_source', action='store_true',
        help='balance source-domain classes in training batches (default)',
    )
    source_balance.add_argument(
        '--no-balance-source', dest='balance_source', action='store_false',
        help='disable source-domain class-balanced batches',
    )
    parser.set_defaults(balance_source=True)

    available_tiles = ['denmark/32VNH/2017', 'france/30TXT/2017', 'france/31TCJ/2017', 'austria/33UVP/2017']

    parser.add_argument('--source', default='denmark/32VNH/2017', help='source dataset', choices=available_tiles)
    parser.add_argument('--target', default='france/30TXT/2017', help='target dataset', choices=available_tiles)
    parser.add_argument('--num_folds', default=1, type=int, help='Number of train/test folds for cross validation')
    parser.add_argument("--val_ratio", default=0.1, type=float,
                        help='Ratio of training data to use for validation. Default 0.1.')
    parser.add_argument("--test_ratio", default=0.2, type=float,
                        help='Ratio of training data to use for testing. Default 0.2.')
    parser.add_argument('--sample_pixels_val', type=bool_flag, default=True, help='speed up validation at the cost of randomness')
    parser.add_argument('--output_dir', default='outputs', help='Path to the folder where the results should be stored')
    parser.add_argument('-e', '--experiment_name', default=None, help='Name of the experiment')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations')
    parser.add_argument('--log_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--eval', action='store_true', help='run only evaluation')
    parser.add_argument('--overall', action='store_true', help='print overall results, if exists')
    parser.add_argument('--combine_spring_and_winter', default=False, type=bool_flag)
    parser.add_argument(
        '--closed_set',
        default=True,
        type=bool_flag,
        help='use source-defined closed-set protocol',
    )
    parser.add_argument(
        '--progress_bar',
        default='auto',
        choices=['auto', 'on', 'off'],
        help=(
            'tqdm mode: auto enables progress bars only on an interactive '
            'stderr; on always enables; off always disables'
        ),
    )

    # Training configuration
    parser.add_argument('--stage1_epochs', default=100, type=int, help='Number of source-only epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument(
        '--eval_batch_size',
        default=None,
        type=int,
        help='Validation/test batch size; defaults to --batch_size',
    )
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay rate')
    parser.add_argument('--num_pixels', default=64, type=int, help='Number of pixels to sample from the input sample')
    parser.set_defaults(model='structure_da')
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input sample')
    parser.add_argument('--with_extra', default=False, type=bool_flag, help='whether to input extra geometric features to the PSE')
    parser.add_argument('--tensorboard_log_dir', default='runs')
    parser.add_argument(
        '--feature_snapshot_interval', default=0, type=int,
        help='epoch interval for compact PSE/geometry snapshots; 0 disables',
    )
    parser.add_argument(
        '--feature_snapshot_samples_per_class', default=32, type=int,
    )
    parser.add_argument(
        '--feature_snapshot_batch_size', default=8, type=int,
        help='snapshot inference batch size; CUDA OOM retries halve it down to 1',
    )
    parser.add_argument(
        '--feature_snapshot_dtype', default='float16', choices=['float16', 'float32'],
    )
    parser.add_argument('--feature_snapshot_dir', default=None)
    parser.add_argument('--steps_per_epoch', default=None, type=int,
                        help='limit the number of training steps per epoch; default uses the full source loader')

    # Stage-1 prototype supervision
    parser.add_argument('--source_warmup_epochs', default=5, type=int,
                        help='epochs of CE-only warmup before prototype supervision starts')
    parser.add_argument('--lambda_q', default=0.1, type=float,
                        help='weight of the Shape prototype relative-margin loss')
    parser.add_argument('--lambda_f', default=0.1, type=float,
                        help='weight of the fused-feature prototype relative-margin loss')
    parser.add_argument('--lambda_q_to_cls', default=0.1, type=float,
                        help='weight of the q-to-classifier consistency KL')
    parser.add_argument('--margin_q', default=0.1, type=float,
                        help='relative margin for the Shape prototype loss')
    parser.add_argument('--margin_f', default=0.1, type=float,
                        help='relative margin for the fused-feature prototype loss')
    parser.add_argument('--tau_q', default=0.1, type=float,
                        help='temperature for the Shape geometry class distribution')

    # Stage-2 orchestration. Scientific thresholds/weights intentionally have
    # no defaults; provide them explicitly or through --stage2_config JSON.
    parser.add_argument('--stage2_config', default=None, type=str)
    parser.add_argument('--stage2_epochs', default=60, type=int)
    parser.add_argument('--stage2_block_epochs', default=20, type=int)
    parser.add_argument('--stage2_steps_per_epoch', default=None, type=int)

    parser.add_argument('--stage2_registration_lambda', default=None, type=float)
    parser.add_argument('--stage2_registration_gain_ratio_max', default=None, type=float)
    parser.add_argument('--stage2_registration_min_common_support', default=None, type=float)
    parser.add_argument('--stage2_registration_max_roughness', default=None, type=float)
    parser.add_argument('--stage2_registration_min_increment', default=None, type=float)
    parser.add_argument('--stage2_registration_max_local_speed', default=None, type=float)
    parser.add_argument('--stage2_registration_max_deviation', default=None, type=float)
    parser.add_argument('--stage2_class_hypothesis_margin', default=None, type=float)

    parser.add_argument('--stage2_phase_min_samples_per_class', default=None, type=float)
    parser.add_argument('--stage2_phase_class_dispersion_max', default=None, type=float)
    parser.add_argument('--stage2_phase_class_diameter_max', default=None, type=float)
    parser.add_argument('--stage2_phase_group_dispersion_max', default=None, type=float)
    parser.add_argument('--stage2_phase_group_diameter_max', default=None, type=float)
    parser.add_argument('--stage2_phase_group_core_separation', default=None, type=float)
    parser.add_argument('--stage2_phase_global_radius', default=None, type=float)
    parser.add_argument('--stage2_phase_confirmation_patience', default=None, type=int)
    parser.add_argument('--stage2_phase_center_drift_max', default=None, type=float)

    parser.add_argument('--stage2_stable_tau_f', default=None, type=float)
    parser.add_argument('--stage2_stable_tau_q', default=None, type=float)
    parser.add_argument('--stage2_cls_confidence_min', default=None, type=float)
    parser.add_argument('--stage2_cls_margin_min', default=None, type=float)
    parser.add_argument('--stage2_fused_confidence_min', default=None, type=float)
    parser.add_argument('--stage2_fused_margin_min', default=None, type=float)
    parser.add_argument('--stage2_q_confidence_min', default=None, type=float)
    parser.add_argument('--stage2_q_margin_min', default=None, type=float)

    parser.add_argument('--stage2_shape_min_valid_classes', default=None, type=int)
    parser.add_argument('--stage2_shape_min_samples_per_class', default=None, type=int)
    parser.add_argument('--stage2_shape_shared_ratio_min', default=None, type=float)
    parser.add_argument('--stage2_shape_leave_one_out_drift_max', default=None, type=float)
    parser.add_argument('--stage2_shape_center_drift_max', default=None, type=float)
    parser.add_argument('--stage2_shape_effect_norm_max', default=None, type=float)
    parser.add_argument('--stage2_shape_confirmation_patience', default=None, type=int)

    parser.add_argument('--stage2_lambda_src_proto', default=None, type=float)
    parser.add_argument('--stage2_lambda_src_cons', default=None, type=float)
    parser.add_argument('--stage2_lambda_syn', default=None, type=float)
    parser.add_argument('--stage2_lambda_syn_cons', default=None, type=float)
    parser.add_argument('--stage2_objective_tau_q', default=None, type=float)
    parser.add_argument('--stage2_fused_margin', default=None, type=float)
    parser.add_argument('--stage2_ema_decay', default=None, type=float)
    parser.add_argument('--stage2_lambda_delta', default=None, type=float)

    # Model hyperparameters
    parser.add_argument('--canonical_grid_size', default=64, type=int)
    parser.add_argument('--roughness_grid_size', default=256, type=int)
    parser.add_argument('--trend_num_basis', default=12, type=int)
    parser.add_argument('--structure_num_basis', default=12, type=int)
    parser.add_argument('--trend_smoothing', default=1e-2, type=float)
    parser.add_argument('--structure_smoothing', default=1e-3, type=float)
    parser.add_argument('--n_head', default=16, type=int)
    parser.add_argument('--d_k', default=8, type=int)
    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--ltae_mlp', default='256,128', type=_comma_separated_ints)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--classifier_hidden', default='64,32', type=_comma_separated_ints)
    parser.add_argument('--time2vec_max_frequency', default=16.0, type=float)

    # Optimization / precision
    parser.add_argument('--amp', default=False, type=bool_flag)
    parser.add_argument(
        '--amp_dtype',
        default='float16',
        choices=['float16', 'bfloat16'],
    )
    parser.add_argument('--time_reference', default=0.0, type=float)
    parser.add_argument('--time_scale', default=365.0, type=float)
    parser.add_argument(
        '--time_coordinate_mode',
        default='canonical_day_of_year',
        choices=['canonical_day_of_year'],
    )
    parser.add_argument('--tau_fast_init', default=0.05, type=float)
    parser.add_argument('--tau_slow_init', default=0.20, type=float)
    parser.add_argument('--tau_min', default=1e-4, type=float)
    parser.add_argument('--delta_tau_min', default=1e-4, type=float)

    cfg = parser.parse_args()
    _apply_stage2_config_file(cfg)

    # Setup folders based on name
    if cfg.experiment_name is not None:
        cfg.tensorboard_log_dir = os.path.join(cfg.tensorboard_log_dir, cfg.experiment_name)
        cfg.output_dir = os.path.join(cfg.output_dir, cfg.experiment_name)

    os.makedirs(cfg.output_dir, exist_ok=True)
    for fold in range(cfg.num_folds):
        os.makedirs(os.path.join(cfg.output_dir, f'fold_{fold}'), exist_ok=True)

    # write training config to file
    if not cfg.eval:
        with open(os.path.join(cfg.output_dir, 'train_config.json'), 'w') as f:
            f.write(json.dumps(vars(cfg), indent=4))
    print(cfg)
    main(cfg)
