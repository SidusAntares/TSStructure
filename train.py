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
from dataset import PixelSetData, create_evaluation_loaders
from evaluation import evaluation
from methods.structure_da import (
    HierarchicalQualityObjective,
    JointStructureDATrainingConfig,
    StructureAwareDomainAdaptationModel,
    create_joint_structure_da_train_loaders,
    resolve_grl_warmup_max_iters,
    resolve_steps_per_epoch,
    train_joint_structure_da,
)
from utils import label_utils
from utils.metrics import overall_classification_report
from utils.train_utils import bool_flag



def main(config):
    if config.with_extra:
        raise ValueError(
            "the channel-preserving structure model does not use geometric extra features"
        )
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

        sample_pixels_val = config.sample_pixels_val
        if config.eval:
            _, target_test_loader = create_evaluation_loaders(
                config.target, splits, config, sample_pixels_val
            )
            source_val_loader = None
        else:
            source_val_loader, _ = create_evaluation_loaders(
                config.source, splits, config, sample_pixels_val
            )
            _, target_test_loader = create_evaluation_loaders(
                config.target, splits, config, sample_pixels_val
            )

        training_config = None
        source_loader = None
        target_loader = None
        if config.eval:
            actual_grl_warmup_max_iters = (
                getattr(config, "grl_warmup_max_iters", None) or 1
            )
        else:
            source_loader, target_loader = create_joint_structure_da_train_loaders(
                config, splits
            )
            training_config = JointStructureDATrainingConfig(
                epochs=config.epochs,
                steps_per_epoch=config.steps_per_epoch,
                lr=config.lr,
                weight_decay=config.weight_decay,
                task_weight=config.lambda_task,
                geometry_weight=config.lambda_geometry,
                alignment_weight=config.lambda_alignment,
                structural_classification_weight=config.lambda_structural_cls,
                structural_domain_weight=config.lambda_structural_domain,
                component_classification_weight=config.lambda_component_cls,
                component_domain_weight=config.lambda_component_domain,
                quality_domain_score_warmup_epochs=(
                    config.quality_domain_score_warmup_epochs
                ),
                amp=getattr(config, "amp", False),
                amp_dtype=getattr(config, "amp_dtype", "float16"),
                log_step=config.log_step,
                progress_bar=config.progress_bar,
                classes=tuple(config.classes),
            )
            resolved_steps_per_epoch = resolve_steps_per_epoch(
                training_config, source_loader, target_loader
            )
            grl_fraction = getattr(config, "grl_warmup_fraction", None)
            grl_override = getattr(config, "grl_warmup_max_iters", None)
            actual_grl_warmup_max_iters = resolve_grl_warmup_max_iters(
                config.epochs,
                resolved_steps_per_epoch,
                fraction=grl_fraction,
                override=grl_override,
            )
            displayed_fraction = (
                0.2 if grl_fraction is None and grl_override is None
                else grl_fraction
            )
            eval_batch_size = config.eval_batch_size or config.batch_size
            amp_enabled = bool(training_config.amp and device.type == "cuda")
            print(
                "TRAIN_PROTOCOL|"
                f"batch_size={config.batch_size}"
                f"|eval_batch_size={eval_batch_size}"
                f"|source_loader_steps={len(source_loader)}"
                f"|target_loader_steps={len(target_loader)}"
                f"|resolved_steps_per_epoch={resolved_steps_per_epoch}"
                f"|epochs={config.epochs}"
                f"|total_steps={config.epochs * resolved_steps_per_epoch}"
                f"|grl_warmup_fraction={displayed_fraction}"
                f"|grl_warmup_max_iters={actual_grl_warmup_max_iters}"
                "|quality_domain_score_warmup_epochs="
                f"{config.quality_domain_score_warmup_epochs}"
                f"|amp={str(amp_enabled).lower()}"
                f"|amp_dtype={training_config.amp_dtype}"
            )

        model = StructureAwareDomainAdaptationModel(
            num_classes=config.num_classes,
            num_channels=config.input_dim,
            channel_feature_dim=config.channel_feature_dim,
            pixel_hidden_dim=config.pixel_hidden_dim,
            structure_dim=config.structure_dim,
            time_scale=config.time_scale,
            tau_fast_init=config.tau_fast_init,
            tau_slow_init=config.tau_slow_init,
            tau_min=config.tau_min,
            delta_tau_min=config.delta_tau_min,
            alignment_hidden_dim=config.domain_hidden_dim,
            grl_max_iters=actual_grl_warmup_max_iters,
        )
        
        model.to(config.device)

        best_model_path = os.path.join(config.fold_dir, 'model.pt')

        if not config.eval:
            from torch.utils.tensorboard import SummaryWriter

            print(model)
            print('Number of trainable parameters:', get_num_trainable_params(model))

            # if os.path.isfile(best_model_path):
            #     answer = input(f'Model already exists at {best_model_path}! Override y/[n]? ')
            #     override = strtobool(answer) if len(answer) > 0 else False
            #     if not override:
            #         print('Skipping fold', fold_num)
            #         continue

            writer = SummaryWriter(log_dir=f'{config.tensorboard_log_dir}_fold{fold_num}', purge_step=0)
            train_joint_structure_da(
                model, source_loader, target_loader, source_val_loader,
                training_config, writer, device, best_model_path,
            )

        print('Restoring best model weights for testing...')

        state_dict = torch.load(best_model_path, weights_only=False)['state_dict']
        model.load_state_dict(state_dict)

        test_metrics = evaluation(
            model,
            target_test_loader,
            device,
            config.classes,
            criterion=torch.nn.CrossEntropyLoss(),
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
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs per fold')
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
    parser.add_argument('--steps_per_epoch', default=None, type=int)
    parser.add_argument('--channel_feature_dim', default=16, type=int)
    parser.add_argument('--pixel_hidden_dim', default=16, type=int)
    parser.add_argument('--structure_dim', default=128, type=int)
    parser.add_argument('--domain_hidden_dim', default=128, type=int)
    grl_warmup_group = parser.add_mutually_exclusive_group()
    grl_warmup_group.add_argument(
        '--grl_warmup_max_iters',
        default=None,
        type=int,
        help='explicit GRL warm-up step override',
    )
    grl_warmup_group.add_argument(
        '--grl_warmup_fraction',
        default=None,
        type=float,
        help='GRL warm-up fraction; defaults to 0.20 unless an override is used',
    )
    parser.add_argument('--amp', default=False, type=bool_flag)
    parser.add_argument(
        '--amp_dtype',
        default='float16',
        choices=['float16', 'bfloat16'],
    )
    parser.add_argument('--lambda_task', default=1.0, type=float)
    parser.add_argument('--lambda_geometry', default=1.0, type=float)
    parser.add_argument('--lambda_alignment', default=1.0, type=float)
    parser.add_argument('--lambda_structural_cls', default=1.0, type=float)
    parser.add_argument('--lambda_structural_domain', default=1.0, type=float)
    parser.add_argument('--lambda_component_cls', default=1.0, type=float)
    parser.add_argument('--lambda_component_domain', default=1.0, type=float)
    parser.add_argument(
        '--quality_domain_score_warmup_epochs',
        default=5,
        type=int,
    )
    parser.add_argument('--time_scale', default=366.0, type=float)
    parser.add_argument('--tau_fast_init', default=0.05, type=float)
    parser.add_argument('--tau_slow_init', default=0.20, type=float)
    parser.add_argument('--tau_min', default=1e-4, type=float)
    parser.add_argument('--delta_tau_min', default=1e-4, type=float)

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
