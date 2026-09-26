import argparse
from collections import defaultdict
from copy import deepcopy
from distutils.util import strtobool
import json
import os
import pickle as pkl
import random

import numpy as np
import torch
import torch.backends.cudnn
from torchvision.transforms import transforms
from tqdm import tqdm

from competitors.dann.dann import train_dann
from competitors.jumbot.jumbot import train_jumbot
from competitors.mmd.train_mmd import train_mmd
from competitors.alda.train_alda import train_alda
from dataset import PixelSetData, create_evaluation_loaders, create_train_loader
from evaluation import evaluation, validation
from models.stclassifier import (
    PseFourierReconLTae,
    PseGru,
    PseLTae,
    PseTae,
    PseTempCNN,
    PseStructureProtoLTae,
)
from models.structure_da.discriminative_structure import (
    initialize_shapelet_dictionary_from_tokens,
)
from methods.structure_da.prototype_losses import (
    compose_structure_v4_source_loss,
    ensure_finite_structure_loss,
    log_shape_health,
    shapelet_diversity_loss,
)
from timematch import add_shift_estimation_arguments, train_timematch
from transforms import Normalize, RandomSamplePixels, RandomSampleTimeSteps, ToTensor, RandomTemporalShift, Identity
from utils import label_utils
from utils.focal_loss import FocalLoss
from utils.metrics import overall_classification_report
from utils.train_utils import (
    AverageMeter,
    bool_flag,
    progress_bar_disabled,
    to_cuda,
)



def add_model_arguments(parser):
    parser.add_argument(
        '--model',
        default='pseltae',
        choices=['psetae', 'pseltae', 'psetcnn', 'psegru', 'psefourierreconltae', 'psestructureprotoltae'],
    )
    parser.add_argument('--fourier_num_modes', default=13, type=int)
    parser.add_argument('--fourier_reg', default=1e-3, type=float)
    parser.add_argument('--fourier_period_days', default=365.0, type=float)
    parser.add_argument(
        '--fourier_solver',
        default='dense_direct',
        choices=['dense_direct'],
    )
    parser.add_argument('--structure-branch', dest='structure_branch', default=False, type=bool_flag)
    parser.add_argument('--structure-exposer', dest='structure_exposer', default='fourier', choices=['fourier'])
    parser.add_argument('--shape-dim', dest='shape_dim', default=128, type=int)
    parser.add_argument('--shape-window-scales', dest='shape_window_scales', nargs='+', default=[24], type=int)
    parser.add_argument('--shape-window-stride', dest='shape_window_stride', default=8, type=int)
    parser.add_argument('--shapelet-count', dest='shapelet_count', default=32, type=int)
    parser.add_argument(
        '--shapelet-init', dest='shapelet_init', default='random',
        choices=['kmeans', 'random'],
    )
    parser.add_argument('--shapelet-beta', dest='shapelet_beta', default=5., type=float)
    parser.add_argument('--shape-resample-length', dest='shape_resample_length', default=16, type=int)
    parser.add_argument('--shapelet-diversity-margin', dest='shapelet_diversity_margin', default=.5, type=float)
    parser.add_argument('--shapelet-diversity-weight', dest='shapelet_diversity_weight', default=.01, type=float)
    parser.add_argument('--shapelet-shaping-weight', dest='shapelet_shaping_weight', default=.01, type=float)
    parser.add_argument('--shapelet-shaping-temperature', dest='shapelet_shaping_temperature', default=.1, type=float)
    parser.add_argument('--shape-class-weight', dest='shape_class_weight', default=.1, type=float)
    parser.add_argument('--shape-target-weight', dest='shape_target_weight', default=.05, type=float)
    parser.add_argument('--shape-align-weight', dest='shape_align_weight', default=.05, type=float)
    parser.add_argument('--stats-align-weight', dest='stats_align_weight', default=.02, type=float)
    parser.add_argument('--proto-momentum', dest='proto_momentum', default=.9, type=float)
    parser.add_argument('--proto-temperature', dest='proto_temperature', default=.1, type=float)
    parser.add_argument('--proto-instance-weight', dest='proto_instance_weight', default=.1, type=float)
    parser.add_argument('--proto-init-epoch', dest='proto_init_epoch', default=1, type=int)
    parser.add_argument('--proto-ramp-start', dest='proto_ramp_start', default=.1, type=float)
    parser.add_argument('--proto-ramp-epochs', dest='proto_ramp_epochs', default=5, type=int)
    return parser


def create_model(config):
    common = dict(
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        with_extra=config.with_extra,
    )
    if config.model == 'pseltae':
        return PseLTae(**common)
    if config.model == 'psetae':
        return PseTae(**common)
    if config.model == 'psetcnn':
        return PseTempCNN(**common)
    if config.model == 'psegru':
        return PseGru(**common)
    if config.model == 'psefourierreconltae':
        return PseFourierReconLTae(
            **common,
            fourier_num_modes=config.fourier_num_modes,
            fourier_reg=config.fourier_reg,
            fourier_period_days=config.fourier_period_days,
            fourier_solver=config.fourier_solver,
        )
    if config.model == 'psestructureprotoltae':
        model = PseStructureProtoLTae(
            **common,
            shape_dim=config.shape_dim,
            shape_window_scales=config.shape_window_scales,
            shape_window_stride=config.shape_window_stride,
            shapelet_count=config.shapelet_count,
            shapelet_beta=config.shapelet_beta,
            shape_resample_length=config.shape_resample_length,
            fourier_num_modes=config.fourier_num_modes,
            fourier_reg=config.fourier_reg,
            fourier_period_days=config.fourier_period_days,
        )
        model.instance_prototype_bank.momentum = config.proto_momentum
        return model
    raise NotImplementedError(config.model)


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

    if config.overall:
        overall_performance(config)
        return

    for fold_num, splits in enumerate(folds):
        print(f'Starting fold {fold_num}...')

        if config.closed_set:
            print_closed_set_counts(config, indices, splits)

        config.fold_dir = os.path.join(config.output_dir, f'fold_{fold_num}')
        config.fold_num = fold_num

        sample_pixels_val = config.sample_pixels_val or (config.eval and config.temporal_shift)
        val_loader, test_loader = create_evaluation_loaders(config.target, splits, config, sample_pixels_val)

        model = create_model(config)
        if isinstance(model, PseStructureProtoLTae):
            with open(os.path.join(config.fold_dir, 'manifest.json'), 'w') as stream:
                json.dump(
                    {
                        'method': 'discriminative_structure_occurrence_phase_v6',
                        'structure_exposer': config.structure_exposer,
                        'fourier_num_modes': config.fourier_num_modes,
                        'shape_dim': config.shape_dim,
                        'shape_window_scales': config.shape_window_scales,
                        'shape_window_stride': config.shape_window_stride,
                        'shapelet_count': config.shapelet_count,
                        'shapelet_beta': config.shapelet_beta,
                        'shape_resample_length': config.shape_resample_length,
                        'shapelet_diversity_margin': config.shapelet_diversity_margin,
                        'shapelet_diversity_weight': config.shapelet_diversity_weight,
                        'shape_class_weight': config.shape_class_weight,
                        'shapelet_response': 'strength+concentration',
                        'shape_response_dim': 2 * config.shapelet_count,
                        'structure_decomposition': 'shared_private',
                        'shared_dim': 64,
                        'domain_dim': 32,
                        'shared_usage': 'morphology_domain_invariant',
                        'domain_usage': 'domain_only',
                        'occurrence_phase': 'weighted_circular_sin_cos',
                        'phase_response_dim': 2 * config.shapelet_count,
                        'phase_projector': 'linear_gelu_linear_layernorm',
                        'semantic_fusion': 'layernorm(shared+phase)',
                        'semantic_usage': 'shape_classifier+Qshape',
                        'phase_loss': False,
                        'shared_domain_objective': 'GRL',
                        'private_domain_objective': 'domain_classification',
                        'separation': 'cross_covariance',
                        'shared_adv_weight': getattr(config, 'shared_adv_weight', .1),
                        'private_domain_weight': getattr(config, 'private_domain_weight', .1),
                        'separation_weight': getattr(config, 'separation_weight', .01),
                        'target_shape_loss': False,
                        'instance_prototype': False,
                        'shape_support_loss': False,
                        'shape_stats_alignment': False,
                        'ema_memory': False,
                        'shapelet_init': config.shapelet_init,
                        'shape_aux_classifier': 'linear',
                    },
                    stream,
                    indent=2,
                )
        
        model.to(config.device)

        best_model_path = os.path.join(config.fold_dir, 'model.pt')

        if not config.eval:
            print(model)
            print('Number of trainable parameters:', get_num_trainable_params(model))

            # if os.path.isfile(best_model_path):
            #     answer = input(f'Model already exists at {best_model_path}! Override y/[n]? ')
            #     override = strtobool(answer) if len(answer) > 0 else False
            #     if not override:
            #         print('Skipping fold', fold_num)
            #         continue

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=f'{config.tensorboard_log_dir}_fold{fold_num}', purge_step=0)
            if config.method == 'timematch':
                train_timematch(model, config, writer, val_loader, device, best_model_path, fold_num, splits)
            elif config.method == 'dann':
                train_dann(model, config, writer, val_loader, device, best_model_path, fold_num, splits)
            elif config.method == 'mmd':
                train_mmd(model, config, writer, val_loader, device, best_model_path, fold_num, splits)
            elif config.method == 'jumbot':
                train_jumbot(model, config, writer, val_loader, device, best_model_path, fold_num, splits)
            elif config.method == 'alda':
                train_alda(model, config, writer, val_loader, device, best_model_path, fold_num, splits)
            else:
                train_supervised(model, config, writer, splits, val_loader, device, best_model_path)

        print('Restoring best model weights for testing...')

        state_dict = torch.load(best_model_path, weights_only=False)['state_dict']
        model.load_state_dict(state_dict)
        test_metrics = evaluation(
            model,
            test_loader,
            device,
            config.classes,
            mode='test',
            progress_bar=getattr(config, "progress_bar", "auto"),
        )

        print(f"Test result for {config.experiment_name}: accuracy={test_metrics['accuracy']:.4f}, f1={test_metrics['macro_f1']:.4f}")
        print(test_metrics['classification_report'])

        save_results(test_metrics, config)

    overall_performance(config)


def prepare_data_protocol(config):
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
    )
    target_protocol_dataset = PixelSetData(
        config.data_root,
        config.target,
        config.classes,
        closed_set=True,
        combine_spring_and_winter=config.combine_spring_and_winter,
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


def _prototype_diagnostics(model, epoch):
    row = {'epoch': epoch}
    bank = model.instance_prototype_bank
    active = bank.prototypes[bank.initialized]
    similarities = active @ active.T
    off_diagonal = similarities[~torch.eye(active.shape[0], dtype=torch.bool, device=active.device)]
    row['prototype_instance_initialized_classes'] = int(bank.initialized.sum())
    row['prototype_instance_mean_pairwise_cos'] = float(off_diagonal.mean()) if off_diagonal.numel() else 0.
    row['prototype_instance_min_pairwise_cos'] = float(off_diagonal.min()) if off_diagonal.numel() else 0.
    return row


def _new_structure_epoch_stats():
    return {'instance_cos': []}


def _collect_structure_epoch_stats(accumulator, batch):
    accumulator['instance_cos'].append(batch['instance_cos'].detach())


def _summarize_structure_epoch_stats(accumulator, epoch, domain):
    values = torch.cat(accumulator['instance_cos'])
    return {'epoch': epoch, f'{domain}_instance_cos_to_correct_proto': float(values.mean())}

def get_dataset_size(data_root, dataset):
    dir = os.path.join(data_root, dataset)
    return len([name for name in os.listdir(os.path.join(dir, 'data')) if name.endswith('.zarr')])


def initialize_shapelet_dictionary_from_source(
    model, source_loader, device, seed, max_tokens=50_000,
):
    """Initialize shared shapelets once from unlabeled source-train shape tokens."""
    if not isinstance(model, PseStructureProtoLTae):
        raise TypeError("source shapelet initialization requires PseStructureProtoLTae")
    if max_tokens < model.structure_branch.shapelet_dictionary.anchors.shape[0]:
        raise ValueError("max_tokens must cover every shapelet anchor")
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_rng is not None:
        torch.cuda.manual_seed_all(seed)
    was_training = model.training
    collected = []
    model.eval()
    try:
        with torch.no_grad():
            for sample in source_loader:
                pixels = sample['pixels'].to(device)
                mask = sample['valid_pixels'].to(device)
                positions = sample['positions'].to(device)
                extra = sample.get('extra')
                if extra is not None:
                    extra = extra.to(device)
                spatial = model.spatial_encoder(pixels, mask, extra)
                branch = model.structure_branch
                exposed, _ = branch.exposer(spatial, positions)
                window_groups, _ = branch.window_extractor(exposed)
                tokens = torch.cat([
                    branch.token_generator(windows) for windows in window_groups
                ], dim=1).reshape(-1, model.shape_dim)
                remaining = max_tokens - sum(value.shape[0] for value in collected)
                collected.append(tokens[:remaining].cpu())
                if sum(value.shape[0] for value in collected) >= max_tokens:
                    break
    finally:
        model.train(was_training)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    if not collected:
        raise RuntimeError("source loader produced no shape tokens")
    tokens = torch.cat(collected, dim=0)
    diagnostics = initialize_shapelet_dictionary_from_tokens(
        model.structure_branch.shapelet_dictionary, tokens, seed,
    )
    print(
        "SHAPELET_KMEANS_INIT|"
        f"tokens={diagnostics['tokens']}|anchors={diagnostics['anchors']}|"
        f"pairwise_cos_mean={diagnostics['pairwise_cos_mean']:.6f}|"
        f"pairwise_cos_max={diagnostics['pairwise_cos_max']:.6f}"
    )
    return diagnostics

def train_supervised(model, config, writer, splits, val_loader, device, best_model_path):
    model.to(device)

    best_f1 = 0
    structure_proto = isinstance(model, PseStructureProtoLTae)

    train_transform = transforms.Compose([
        RandomSamplePixels(config.num_pixels),
        RandomSampleTimeSteps(config.seq_length),
        RandomTemporalShift(max_shift=config.max_shift_aug, p=config.shift_aug_p) if config.with_shift_aug else Identity(),
        Normalize(),
        ToTensor(),
    ])
    dataset_name = config.source
    if config.train_on_target:
        dataset_name = config.target

    dataset = PixelSetData(
        config.data_root,
        dataset_name,
        config.classes,
        train_transform,
        indices=splits[dataset_name]['train'],
        closed_set=config.closed_set,
        combine_spring_and_winter=config.combine_spring_and_winter,
    )
    data_loader = create_train_loader(dataset, config.batch_size, config.num_workers)
    print(f'training dataset: {dataset_name}, n={len(dataset)}, batches={len(data_loader)}')

    if (
        structure_proto
        and not config.train_on_target
        and getattr(config, 'shapelet_init', 'random') == 'kmeans'
    ):
        initialize_shapelet_dictionary_from_source(
            model, data_loader, device, config.seed,
        )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    criterion = FocalLoss(gamma=config.focal_loss_gamma)
    steps_per_epoch = len(data_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0)

    best_f1 = 0
    for epoch in range(config.epochs):
        model.train()
        loss_meter = AverageMeter()
        shape_source_loss_sum = 0.
        shape_source_correct = 0
        shape_source_count = 0

        progress_bar = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=f'Epoch {epoch + 1}/{config.epochs}',
            disable=progress_bar_disabled(
                getattr(config, "progress_bar", "auto")
            ),
        )
        global_step = epoch * len(data_loader)
        for step, sample in progress_bar:
            targets = sample['label'].cuda(device=device, non_blocking=True)

            pixels, mask, positions, extra = to_cuda(sample, device)
            if structure_proto:
                structured = model(pixels, mask, positions, extra, return_dict=True)
                outputs = structured["logits"]
                loss_cls = criterion(outputs, targets)
                loss_shape_source = criterion(structured["shape_logits"], targets)
                loss_diversity = shapelet_diversity_loss(
                    model.structure_branch.shapelet_dictionary.anchors,
                    config.shapelet_diversity_margin,
                )
                loss = compose_structure_v4_source_loss(
                    loss_cls, loss_shape_source, loss_diversity,
                    config.shape_class_weight, config.shapelet_diversity_weight,
                )
            else:
                outputs = model.forward(pixels, mask, positions, extra)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            if structure_proto:
                ensure_finite_structure_loss(loss)
            loss.backward()
            if structure_proto:
                if step == 0:
                    log_shape_health(writer, epoch, model, structured)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5., error_if_nonfinite=True,
                )
            optimizer.step()
            if structure_proto:
                batch_count = int(targets.shape[0])
                shape_source_loss_sum += float(loss_shape_source.detach()) * batch_count
                shape_source_correct += int(
                    (structured["shape_logits"].detach().argmax(1) == targets).sum()
                )
                shape_source_count += batch_count
            scheduler.step()

            loss_meter.update(loss.item(), n=config.batch_size)

            if step % config.log_step == 0:
                lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(lr=f'{lr:.1E}', loss=f"{loss_meter.avg:.3f}")
                writer.add_scalar("train/loss", loss_meter.val, global_step + step)
                writer.add_scalar("train/lr", lr, global_step + step)
                if structure_proto:
                    writer.add_scalar("train/loss_cls_source", loss_cls.detach(), global_step + step)
                    writer.add_scalar("train/loss_shape_source", loss_shape_source.detach(), global_step + step)
                    writer.add_scalar("train/loss_shapelet_diversity", loss_diversity.detach(), global_step + step)

        progress_bar.close()

        if structure_proto:
            shape_source_loss = shape_source_loss_sum / max(shape_source_count, 1)
            shape_source_accuracy = shape_source_correct / max(shape_source_count, 1)
            writer.add_scalar("epoch/shape_source_loss", shape_source_loss, epoch)
            writer.add_scalar("epoch/shape_source_accuracy", shape_source_accuracy, epoch)
            print(
                f"SHAPE_AUX_EPOCH|epoch={epoch}|domain=source|"
                f"loss={shape_source_loss:.6f}|accuracy={shape_source_accuracy:.6f}|"
                f"samples={shape_source_count}"
            )

        model.eval()
        previous_best = best_f1
        best_f1 = validation(best_f1, best_model_path, config, criterion, device, epoch, model, val_loader, writer)
        if structure_proto:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': vars(config),
                'best_f1': best_f1,
            }
            torch.save(checkpoint, os.path.join(config.fold_dir, 'checkpoint_last.pt'))
            if best_f1 > previous_best or not os.path.isfile(best_model_path):
                torch.save(checkpoint, best_model_path)
                torch.save(checkpoint, os.path.join(config.fold_dir, 'checkpoint_best.pt'))


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

    for i,row in enumerate(np.mean(cms, axis=0)):
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
    parser.add_argument('--num_blocks', default=100, type=int, help='Number of geographical blocks in dataset for splitting. Default 100.')

    available_tiles = ['denmark/32VNH/2017', 'france/30TXT/2017', 'france/31TCJ/2017', 'austria/33UVP/2017']

    parser.add_argument('--source', default='denmark/32VNH/2017', help='source dataset', choices=available_tiles)
    parser.add_argument('--target', default='france/30TXT/2017', help='target dataset', choices=available_tiles)
    parser.add_argument('--num_folds', default=1, type=int, help='Number of train/test folds for cross validation')
    parser.add_argument("--val_ratio", default=0.1, type=float,
                        help='Ratio of training data to use for validation. Default 10%%.')
    parser.add_argument("--test_ratio", default=0.2, type=float,
                        help='Ratio of training data to use for testing. Default 20%%.')
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
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay rate')
    parser.add_argument('--focal_loss_gamma', default=1.0, type=float, help='gamma value for focal loss')
    parser.add_argument('--num_pixels', default=64, type=int, help='Number of pixels to sample from the input sample')
    parser.add_argument('--seq_length', default=30, type=int, help='Number of time steps to sample from the input sample')
    add_model_arguments(parser)
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input sample')
    parser.add_argument('--with_extra', default=False, type=bool_flag, help='whether to input extra geometric features to the PSE')
    parser.add_argument('--tensorboard_log_dir', default='runs')
    parser.add_argument('--train_on_target', default=False, action='store_true', help='supervised training on target for upper bound comparison')

    parser.add_argument('--with_shift_aug', default=False, type=bool_flag, help='whether to apply random temporal shift augmentation')
    parser.add_argument('--shift_aug_p', default=1.0, type=float, help='probability to apply temporal shift augmentation')
    parser.add_argument('--max_shift_aug', default=60, type=int, help='highest shift to apply for temporal shift augmentation')

    # Specific parameters for each training method
    subparsers = parser.add_subparsers(dest='method')

    # DANN + CDAN
    dann = subparsers.add_parser('dann')
    dann.add_argument('--adv_loss', type=str, default='DANN', choices=['DANN', 'CDAN', 'CDAN+E'])
    dann.add_argument('--use_default_optim', type=bool_flag, default=True, help="whether to use default optimizer")
    dann.add_argument('--weights', type=str, help='path to source trained model weights')
    dann.add_argument("--steps_per_epoch", type=int, default=500, help='n steps per epoch')
    dann.add_argument('--epochs', default=20, type=int, help='Number of epochs per fold')
    dann.add_argument("--trade_off", default=1.0, type=float, help='weight of adversarial loss')
    dann.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    # MMD loss (DAN)
    mmd = subparsers.add_parser('mmd')
    mmd.add_argument('--use_default_optim', type=bool_flag, default=True, help="whether to use default optimizer")
    mmd.add_argument('--weights', type=str, help='path to source trained model weights')
    mmd.add_argument("--steps_per_epoch", type=int, default=500, help='n steps per epoch')
    mmd.add_argument('--epochs', default=20, type=int, help='Number of epochs per fold')
    mmd.add_argument("--trade_off", default=1.0, type=float, help='weight of adversarial loss')
    mmd.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    # JUMBOT
    jumbot = subparsers.add_parser('jumbot')
    jumbot.add_argument('--weights', type=str, help='path to source trained model weights')
    jumbot.add_argument("--steps_per_epoch", type=int, default=500, help='n steps per epoch')
    jumbot.add_argument('--epochs', default=20, type=int, help='Number of epochs per fold')
    jumbot.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    jumbot.add_argument('--eta1', default=0.01, type=float, help='feature comparison coefficient')
    jumbot.add_argument('--eta2', default=1.0, type=float, help='label comparison coefficient')
    jumbot.add_argument('--epsilon', default=0.01, type=float, help='marginal coefficient')
    jumbot.add_argument('--tau', default=0.5, type=float, help='entropic regularization')

    # ALDA
    alda = subparsers.add_parser('alda')
    alda.add_argument('--use_default_optim', type=bool_flag, default=True, help="whether to use default optimizer")
    alda.add_argument('--weights', type=str, help='path to source trained model weights')
    alda.add_argument("--steps_per_epoch", type=int, default=500, help='n steps per epoch')
    alda.add_argument('--epochs', default=20, type=int, help='Number of epochs per fold')
    alda.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    alda.add_argument("--trade_off", default=1.0, type=float, help='weight of adversarial loss')
    alda.add_argument("--pseudo_threshold", default=0.7, type=float, help='confidence threshold for assigning pseudo labels')


    # TimeMatch
    timematch = subparsers.add_parser('timematch')
    timematch.add_argument('--weights', type=str, help='path to source trained model weights')
    timematch.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    timematch.add_argument("--pseudo_threshold", default=0.9, type=float, help='confidence threshold for assigning pseudo labels')
    timematch.add_argument("--ema_decay", default=0.9999, type=float, help='decay rate for mean teacher')
    timematch.add_argument("--trade_off", type=float, default=2.0, help='weight for unsupervised loss')
    timematch.add_argument('--shared-adv-weight', dest='shared_adv_weight', default=.1, type=float)
    timematch.add_argument('--private-domain-weight', dest='private_domain_weight', default=.1, type=float)
    timematch.add_argument('--separation-weight', dest='separation_weight', default=.01, type=float)
    timematch.add_argument("--estimate_shift", type=bool_flag, default=True, help='whether to account for temporal shift')
    timematch.add_argument('--epochs', default=20, type=int, help='Number of epochs per fold')
    timematch.add_argument("--steps_per_epoch", type=int, default=500, help='n steps per epoch')
    timematch.add_argument("--balance_source", type=bool_flag, default=True, help='class balanced batches for source')
    timematch.add_argument("--use_focal_loss", type=bool_flag, default=True, help='use focal loss or cross entropy')
    timematch.add_argument("--shift_source", type=bool_flag, default=True, help='whether to apply temporal shift to source data')
    timematch.add_argument("--sample_size", type=int, default=100, help='number of batches to sample for estimating shift')
    timematch.add_argument("--max_temporal_shift", type=int, default=60, help='maximum temporal shift to consider')
    timematch.add_argument("--domain_specific_bn", type=bool_flag, default=True, help='whether to use domain specific batch normalization')
    timematch.add_argument("--shift_estimator", type=str, default='AM', choices=['AM', 'IS', 'ACC', 'ENT'])
    add_shift_estimation_arguments(timematch)
    timematch.add_argument('--run_validation', default=True, action='store_true', help='whether to run validation each epoch')
    timematch.add_argument("--output_student", type=bool_flag, default=True, help='output student or teacher')

    cfg = parser.parse_args()


    # Setup folders based on name
    if cfg.experiment_name is not None:
        cfg.tensorboard_log_dir = os.path.join(cfg.tensorboard_log_dir, cfg.experiment_name)
        cfg.output_dir = os.path.join(cfg.output_dir, cfg.experiment_name)

    os.makedirs(cfg.output_dir, exist_ok=True)
    for fold in range(cfg.num_folds):
        os.makedirs(os.path.join(cfg.output_dir, 'fold_{}'.format(fold)), exist_ok=True)


    # write training config to file
    if not cfg.eval:
        with open(os.path.join(cfg.output_dir, 'train_config.json'), 'w') as f:
            f.write(json.dumps(vars(cfg), indent=4))
    print(cfg)
    main(cfg)
