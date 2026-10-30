"""Independent source-only imputer pretraining and imputed TimeMatch UDA."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from train_m1_source import TILES, boolean


def build_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('pretrain', 'uda'))
    parser.add_argument('--variant', choices=('A', 'B'))
    parser.add_argument('--source', required=True, choices=TILES)
    parser.add_argument('--target', required=True, choices=TILES)
    parser.add_argument('--weights', required=True, help='Original historical Source leaf containing fold_0/model.pt')
    parser.add_argument('--imputer_checkpoint')
    parser.add_argument('--output_dir', required=True, help='Exact per-task output leaf (must not exist)')
    parser.add_argument('--experiment_name')
    parser.add_argument('--tensorboard_log_dir')
    parser.add_argument('--data_root', default='/data/user/dataset/timematch_data')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', choices=('pseltae',), default='pseltae')
    parser.add_argument('--progress_bar', choices=('auto', 'on', 'off'), default='off')
    for name, value in dict(seed=1, num_workers=8, epochs=20, steps_per_epoch=500,
                            batch_size=128, seq_length=30, num_pixels=64, input_dim=10,
                            num_folds=1, num_blocks=100, log_step=10, max_shift_aug=60,
                            sample_size=100, max_temporal_shift=60).items():
        parser.add_argument('--' + name, type=int, default=value)
    for name, value in dict(lr=1e-4, weight_decay=1e-4, imputer_lr=1e-4,
                            imputer_weight_decay=1e-4, nll_weight=.01,
                            source_aug_weight=1., focal_loss_gamma=1., val_ratio=.1,
                            test_ratio=.2, shift_aug_p=1., pseudo_threshold=.9,
                            ema_decay=.9999, trade_off=2.).items():
        parser.add_argument('--' + name, type=float, default=value)
    parser.add_argument('--reconstruction_weight', type=float)
    for name, value in dict(closed_set=True, sample_pixels_val=True,
                            combine_spring_and_winter=False, with_extra=False,
                            with_shift_aug=False, balance_source=True, use_focal_loss=True,
                            estimate_shift=True, shift_source=True, domain_specific_bn=True,
                            run_validation=False, log_target_labels=False,
                            strict_final_student=True, output_student=True).items():
        parser.add_argument('--' + name, type=boolean, default=value)
    parser.add_argument('--shift_estimator', choices=('AM',), default='AM')
    parser.add_argument('--evidence_mode', choices=('off',), default='off')
    parser.add_argument('--dry_run', action='store_true')
    cfg = parser.parse_args(argv)
    if cfg.stage == 'uda' and (not cfg.variant or not cfg.imputer_checkpoint):
        parser.error('UDA requires --variant and --imputer_checkpoint')
    if cfg.stage == 'pretrain' and (cfg.variant or cfg.imputer_checkpoint):
        parser.error('pretrain does not accept a variant or an imputer checkpoint')
    if cfg.source == cfg.target:
        parser.error('specify the downstream source/target pair; pretraining uses source only internally')
    if (cfg.seed, cfg.num_workers, cfg.epochs, cfg.num_folds) != (1, 8, 20, 1):
        parser.error('matched experiment requires seed=1, num_workers=8, epochs=20 and num_folds=1')
    if (not cfg.closed_set or cfg.with_extra or cfg.with_shift_aug or cfg.combine_spring_and_winter
            or not cfg.domain_specific_bn or cfg.run_validation or cfg.log_target_labels
            or not cfg.output_student or not cfg.strict_final_student):
        parser.error('use the original closed-set inputs and final Student protocol without target-label diagnostics')
    for name, value in vars(cfg).items():
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f'{name} must be finite')
    for name in ('steps_per_epoch', 'batch_size', 'seq_length', 'num_pixels', 'input_dim',
                 'num_blocks', 'log_step', 'sample_size'):
        if getattr(cfg, name) <= 0:
            parser.error(f'{name} must be positive')
    if min(cfg.lr, cfg.imputer_lr) <= 0 or min(cfg.weight_decay, cfg.imputer_weight_decay,
                                              cfg.nll_weight, cfg.source_aug_weight) < 0:
        parser.error('learning rates must be positive and loss/decay weights nonnegative')
    if not (0 <= cfg.val_ratio < 1 and 0 <= cfg.test_ratio < 1 and 0 < cfg.val_ratio + cfg.test_ratio < 1):
        parser.error('split ratios must leave training samples')
    cfg.reconstruction_weight = (1. if cfg.stage == 'pretrain' else .1) if cfg.reconstruction_weight is None else cfg.reconstruction_weight
    if cfg.reconstruction_weight < 0:
        parser.error('reconstruction_weight must be nonnegative')
    cfg.experiment_name = cfg.experiment_name or Path(cfg.output_dir).name
    cfg.tensorboard_log_dir = cfg.tensorboard_log_dir or str(Path(cfg.output_dir) / 'tensorboard')
    cfg.mode = None
    cfg.m1 = cfg.eval = cfg.overall = cfg.train_on_target = False
    cfg.method = 'timematch'
    cfg.imputer_epochs = 20
    cfg.pretrain_recovery_epochs = 10
    cfg.imputer_options = dict(feature_dim=128, d_model=256, num_latents=32,
                               rounds=2, self_blocks=3, heads=8, dropout=.1)
    cfg.imputation_protocol = dict(top_fraction=.25, max_anchors=2, short_gap_days=20,
        long_gap_days=40, short_queries='midpoint', long_queries='thirds', max_queries=8,
        recovery_heads=[1, 2], recovery_sides='both', recovery_total_points=[2, 3],
        date_coordinate='original TimeMatch days_after(start_date)', extrapolation=False,
        reconstruction='mean Huber + nll_weight * mean per-feature Gaussian NLL',
        log_variance_bounds=[-6, 4], classification_position='interpolate frozen LTAE position table')
    return cfg


def validate_source(config):
    """Reject M0/M1 and any Source protocol other than the archived original."""
    leaf = Path(config.weights)
    checkpoint = leaf / 'fold_0/model.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Original Source checkpoint missing: {checkpoint}')
    metadata = leaf / 'train_config.json'
    source = json.loads(metadata.read_text(encoding='utf-8'))
    expected = dict(source=config.source, target=config.source, seed=1, num_workers=8,
                    epochs=100, model='pseltae', closed_set=True, num_folds=1,
                    num_blocks=100, val_ratio=.1, test_ratio=.2, sample_pixels_val=True,
                    batch_size=128, seq_length=30, num_pixels=64, input_dim=10,
                    combine_spring_and_winter=False, with_extra=False, with_shift_aug=False,
                    lr=.001, weight_decay=.0001, focal_loss_gamma=1., shift_aug_p=1.,
                    max_shift_aug=60)
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f'Original Source config mismatch: {key}={source.get(key)!r}, expected {value!r}')
    if source.get('mode') in ('M0', 'M1') or source.get('m1', False) or source.get('method') is not None:
        raise ValueError('Original Source required; M0/M1/UDA checkpoints are forbidden')
    config.source_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def source_provenance():
    root = Path(__file__).resolve().parent
    paths = [root / 'train_imputed_timematch.py', root / 'train.py', root / 'timematch.py',
             root / 'dataset.py', root / 'transforms.py', root / 'evaluation.py']
    paths.extend(sorted((root / 'models').rglob('*.py')))
    paths.extend(sorted((root / 'utils').glob('*.py')))
    return {'sha256': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in paths if p.is_file()}}


def main(argv=None):
    cfg = build_config(argv)
    if cfg.dry_run:
        print(json.dumps(vars(cfg), indent=2))
        return
    validate_source(cfg)
    if cfg.stage == 'uda' and not Path(cfg.imputer_checkpoint).is_file():
        raise FileNotFoundError(f'Imputer epoch-20 checkpoint missing: {cfg.imputer_checkpoint}')
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'fold_0').mkdir()
    (output / 'config.json').write_text(json.dumps(vars(cfg), indent=2), encoding='utf-8')
    (output / 'code_provenance.json').write_text(json.dumps(source_provenance(), indent=2), encoding='utf-8')
    from utils.imputation_training import pretrain_imputer, run_imputation_uda
    if cfg.stage == 'pretrain':
        pretrain_imputer(cfg)
    else:
        run_imputation_uda(cfg)


if __name__ == '__main__':
    main()
