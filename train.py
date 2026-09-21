import argparse
import csv
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
from torch.utils.tensorboard import SummaryWriter
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
from methods.structure_da.prototype_losses import (
    compose_source_loss,
    initialize_source_banks,
    select_top_shape_tokens,
    two_level_prototype_losses,
    update_source_banks,
    structure_batch_statistics,
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
    parser.add_argument('--shape-ratio', dest='shape_ratio', default=.6, type=float)
    parser.add_argument('--shape-window-scales', dest='shape_window_scales', nargs='+', default=[16, 32], type=int)
    parser.add_argument('--shape-window-stride', dest='shape_window_stride', default=8, type=int)
    parser.add_argument('--proto-momentum', dest='proto_momentum', default=.9, type=float)
    parser.add_argument('--proto-temperature', dest='proto_temperature', default=.1, type=float)
    parser.add_argument('--proto-shape-mix', dest='proto_shape_mix', default=.01, type=float)
    parser.add_argument('--source-proto-weight', dest='source_proto_weight', default=1., type=float)
    parser.add_argument('--target-proto-weight', dest='target_proto_weight', default=1., type=float)
    parser.add_argument('--proto-warmup-epochs', dest='proto_warmup_epochs', default=1, type=int)
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
            fourier_num_modes=config.fourier_num_modes,
            fourier_reg=config.fourier_reg,
            fourier_period_days=config.fourier_period_days,
        )
        model.shape_prototype_bank.momentum = config.proto_momentum
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
                        'method': 'discriminative_structure_dual_prototype_v1',
                        'structure_exposer': config.structure_exposer,
                        'fourier_num_modes': config.fourier_num_modes,
                        'shape_dim': config.shape_dim,
                        'shape_ratio': config.shape_ratio,
                        'shape_window_scales': config.shape_window_scales,
                        'shape_window_stride': config.shape_window_stride,
                        'proto_momentum': config.proto_momentum,
                        'proto_temperature': config.proto_temperature,
                        'proto_shape_mix': config.proto_shape_mix,
                        'source_proto_weight': config.source_proto_weight,
                        'target_proto_weight': config.target_proto_weight,
                        'prototype_update_domain': 'source_ground_truth_only',
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


def _append_csv(path, row):
    exists = os.path.isfile(path)
    with open(path, 'a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _prototype_diagnostics(model, epoch):
    row = {'epoch': epoch}
    for name, bank in (
        ('prototype_shape', model.shape_prototype_bank),
        ('prototype_instance', model.instance_prototype_bank),
    ):
        active = bank.prototypes[bank.initialized]
        similarities = active @ active.T
        off_diagonal = similarities[~torch.eye(active.shape[0], dtype=torch.bool, device=active.device)]
        row[f'{name}_initialized_classes'] = int(bank.initialized.sum())
        row[f'{name}_mean_pairwise_cos'] = float(off_diagonal.mean()) if off_diagonal.numel() else 0.
        row[f'{name}_min_pairwise_cos'] = float(off_diagonal.min()) if off_diagonal.numel() else 0.
    return row


def _new_structure_epoch_stats():
    return {
        'instance_cos': [], 'shape_cos': [], 'attention_entropy': [],
        'top_counts': [], 'selected_by_scale': defaultdict(lambda: [0., 0.]),
    }


def _collect_structure_epoch_stats(accumulator, batch):
    for key in ('instance_cos', 'shape_cos', 'attention_entropy', 'top_counts'):
        accumulator[key].append(batch[key].detach())
    for scale, (selected, valid) in batch['selected_by_scale'].items():
        accumulator['selected_by_scale'][scale][0] += float(selected)
        accumulator['selected_by_scale'][scale][1] += float(valid)


def _summarize_structure_epoch_stats(accumulator, epoch, domain):
    def merged(key):
        return torch.cat(accumulator[key])
    entropy = merged('attention_entropy')
    counts = merged('top_counts')
    row = {
        'epoch': epoch,
        f'{domain}_instance_cos_to_correct_proto': float(merged('instance_cos').mean()),
        f'{domain}_shape_cos_to_correct_proto': float(merged('shape_cos').mean()),
        'attention_entropy_mean': float(entropy.mean()),
        'attention_entropy_p10': float(torch.quantile(entropy, .1)),
        'attention_entropy_p90': float(torch.quantile(entropy, .9)),
        'top_shape_count_mean': float(counts.mean()),
        'top_shape_count_min': int(counts.min()),
        'top_shape_count_max': int(counts.max()),
    }
    for scale, (selected, valid) in sorted(accumulator['selected_by_scale'].items()):
        row[f'selected_fraction_scale_{scale}'] = selected / max(valid, 1.)
    return row

def get_dataset_size(data_root, dataset):
    dir = os.path.join(data_root, dataset)
    return len([name for name in os.listdir(os.path.join(dir, 'data')) if name.endswith('.zarr')])

def train_supervised(model, config, writer, splits, val_loader, device, best_model_path):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_f1 = 0
    structure_proto = isinstance(model, PseStructureProtoLTae)
    if structure_proto and config.epochs <= config.proto_warmup_epochs:
        raise ValueError("structure prototype source training requires at least one epoch after warm-up")

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

    criterion = FocalLoss(gamma=config.focal_loss_gamma)
    steps_per_epoch = len(data_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch, eta_min=0)

    best_f1 = 0
    for epoch in range(config.epochs):
        if structure_proto and epoch == config.proto_warmup_epochs:
            model.eval()
            def initialization_batches():
                for initialization_sample in data_loader:
                    initialization_labels = initialization_sample['label'].cuda(device=device, non_blocking=True)
                    init_pixels, init_mask, init_positions, init_extra = to_cuda(initialization_sample, device)
                    yield (
                        model(init_pixels, init_mask, init_positions, init_extra, return_dict=True),
                        initialization_labels,
                    )
            initialize_source_banks(
                initialization_batches(),
                model.shape_prototype_bank,
                model.instance_prototype_bank,
                config.shape_ratio,
            )
            print(
                "STRUCTURE_PROTO_INITIALIZED|"
                f"shape_classes={int(model.shape_prototype_bank.initialized.sum())}|"
                f"instance_classes={int(model.instance_prototype_bank.initialized.sum())}"
            )
            best_f1 = 0
        model.train()
        loss_meter = AverageMeter()
        epoch_structure_stats = _new_structure_epoch_stats() if structure_proto else None

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
                if epoch < config.proto_warmup_epochs:
                    loss_instance = outputs.sum() * 0
                    loss_shape = outputs.sum() * 0
                    loss_proto = outputs.sum() * 0
                    loss = loss_cls
                    selected_for_stats = select_top_shape_tokens(
                        structured['shape_attention'], structured['shape_mask'], config.shape_ratio,
                    )
                else:
                    loss_instance, loss_shape, _, selected_for_stats = two_level_prototype_losses(
                        structured, targets,
                        model.shape_prototype_bank, model.instance_prototype_bank,
                        config.shape_ratio, config.proto_temperature,
                        config.proto_shape_mix,
                    )
                    composed = compose_source_loss(
                        loss_cls, loss_instance, loss_shape,
                        config.proto_shape_mix, config.source_proto_weight,
                    )
                    loss, loss_proto = composed.total, composed.prototype_total
            else:
                outputs = model.forward(pixels, mask, positions, extra)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if structure_proto and epoch >= config.proto_warmup_epochs:
                update_source_banks(
                    structured, targets,
                    model.shape_prototype_bank, model.instance_prototype_bank,
                    config.shape_ratio,
                )
            if structure_proto:
                _collect_structure_epoch_stats(
                    epoch_structure_stats,
                    structure_batch_statistics(
                        structured, targets,
                        model.shape_prototype_bank, model.instance_prototype_bank,
                        selected_for_stats,
                    ),
                )
            scheduler.step()

            loss_meter.update(loss.item(), n=config.batch_size)

            if step % config.log_step == 0:
                lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(lr=f'{lr:.1E}', loss=f"{loss_meter.avg:.3f}")
                writer.add_scalar("train/loss", loss_meter.val, global_step + step)
                writer.add_scalar("train/lr", lr, global_step + step)
                if structure_proto:
                    writer.add_scalar("train/loss_cls_source", loss_cls.detach(), global_step + step)
                    writer.add_scalar("train/loss_proto_instance_source", loss_instance.detach(), global_step + step)
                    writer.add_scalar("train/loss_proto_shape_source", loss_shape.detach(), global_step + step)
                    writer.add_scalar("train/loss_proto_source_total", loss_proto.detach(), global_step + step)
                    entropy = -(structured["shape_attention"].clamp_min(1e-12).log()
                                * structured["shape_attention"]).sum(-1).mean()
                    writer.add_scalar("structure/attention_entropy_mean", entropy.detach(), global_step + step)

        progress_bar.close()

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
            epoch_stats = _summarize_structure_epoch_stats(
                epoch_structure_stats, epoch, 'source',
            )
            prototype_stats = _prototype_diagnostics(model, epoch)
            prototype_stats.update({
                key: value for key, value in epoch_stats.items()
                if key.startswith('source_')
            })
            _append_csv(
                os.path.join(config.fold_dir, 'prototype_stats.csv'),
                prototype_stats,
            )
            _append_csv(
                os.path.join(config.fold_dir, 'attention_stats.csv'),
                {key: value for key, value in epoch_stats.items()
                 if not key.startswith('source_')},
            )


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
    if config.model == 'psestructureprotoltae':
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as outfile:
            json.dump(metrics, outfile, indent=4)
        np.savetxt(os.path.join(out_dir, 'confusion_matrix.csv'), conf_mat, delimiter=',', fmt='%d')
        with open(os.path.join(out_dir, 'class_metrics.csv'), 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['class_id', 'class_name', 'precision', 'recall', 'f1', 'support'])
            writer.writeheader()
            for class_id, class_name in enumerate(config.classes):
                tp = float(conf_mat[class_id, class_id])
                support = float(conf_mat[class_id].sum())
                predicted = float(conf_mat[:, class_id].sum())
                precision = tp / predicted if predicted else 0.
                recall = tp / support if support else 0.
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
                writer.writerow(dict(class_id=class_id, class_name=class_name, precision=precision, recall=recall, f1=f1, support=int(support)))


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
