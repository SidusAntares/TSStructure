"""Independent, fresh Source training for M1 and its matched M0 control."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


BASE_REVISION = '8e4594de40581755db1d2aa6e4633273b17a9928'
TILES = ('austria/33UVP/2017', 'france/30TXT/2017',
         'france/31TCJ/2017', 'denmark/32VNH/2017')


def boolean(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('expected true or false')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('M0', 'M1'), default='M1')
    parser.add_argument('--source', required=True, choices=TILES)
    parser.add_argument('--target', choices=TILES, help='Must equal source; defaults to source')
    parser.add_argument('--data_root', default='/data/user/dataset/timematch_data')
    parser.add_argument('--output_dir', default=None, help='Source experiment parent directory')
    parser.add_argument('--experiment_name', default=None)
    parser.add_argument('--tensorboard_log_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--progress_bar', choices=('auto', 'on', 'off'), default='auto')
    parser.add_argument('--model', choices=('pseltae',), default='pseltae')
    for name, default in (('seed', 1), ('num_workers', 8), ('epochs', 100),
                          ('batch_size', 128), ('num_pixels', 64), ('seq_length', 30),
                          ('input_dim', 10), ('num_folds', 1), ('num_blocks', 100),
                          ('log_step', 10), ('max_shift_aug', 60),
                          ('m1_warmup_epochs', 5), ('m1_degenerate_patience', 5)):
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in (('lr', .001), ('weight_decay', .0001),
                          ('focal_loss_gamma', 1.), ('val_ratio', .1), ('test_ratio', .2),
                          ('shift_aug_p', 1.), ('m1_rho', .5), ('m1_lambda_keep', 1.),
                          ('m1_lambda_budget', .1), ('m1_sharpness', 1.),
                          ('m1_tau_start', .5), ('m1_tau_end', .05),
                          ('m1_degenerate_tolerance', .001)):
        parser.add_argument('--' + name, type=float, default=default)
    for name, default in (('closed_set', True), ('sample_pixels_val', True),
                          ('combine_spring_and_winter', False), ('with_extra', False),
                          ('with_shift_aug', False)):
        parser.add_argument('--' + name, type=boolean, default=default)
    parser.add_argument('--dry_run', action='store_true', help='Validate and print config without training')
    return parser


def build_config(argv=None):
    parser = build_parser()
    cfg = parser.parse_args(argv)
    cfg.target = cfg.target or cfg.source
    if cfg.target != cfg.source:
        parser.error('Source training requires --target equal to --source')
    if cfg.seed != 1 or cfg.num_workers != 8:
        parser.error('matched Source experiments require seed=1 and num_workers=8')
    if not cfg.closed_set or cfg.with_shift_aug or cfg.with_extra or cfg.combine_spring_and_winter:
        parser.error('use the original closed-set protocol without shift augmentation or extra features')
    for name in ('epochs', 'batch_size', 'num_pixels', 'seq_length', 'num_folds',
                 'input_dim', 'num_blocks', 'log_step', 'm1_degenerate_patience'):
        if getattr(cfg, name) <= 0:
            parser.error(f'{name} must be positive')
    for name, value in vars(cfg).items():
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f'{name} must be finite')
    if not 0 < cfg.m1_rho < 1 or cfg.m1_tau_start <= 0 or cfg.m1_tau_end <= 0:
        parser.error('require 0 < m1_rho < 1 and positive temperatures')
    if cfg.m1_tau_end > cfg.m1_tau_start:
        parser.error('m1_tau_end must not exceed m1_tau_start')
    if min(cfg.m1_lambda_keep, cfg.m1_lambda_budget, cfg.m1_sharpness,
           cfg.m1_degenerate_tolerance, cfg.m1_warmup_epochs, cfg.weight_decay) < 0 or cfg.lr <= 0:
        parser.error('loss weights, tolerance, warmup and weight decay must be nonnegative; lr must be positive')
    if not (0 <= cfg.val_ratio < 1 and 0 <= cfg.test_ratio < 1 and
            0 < cfg.val_ratio + cfg.test_ratio < 1):
        parser.error('validation and test ratios must leave training samples')
    prefix = cfg.mode.lower()
    root = Path('outputs') / f'{prefix}_soft_top50_seed1'
    cfg.output_dir = str(Path(cfg.output_dir or root / 'source') /
                         (cfg.experiment_name or f'{prefix}_source_{cfg.source.split("/")[1]}_seed1'))
    cfg.experiment_name = cfg.experiment_name or f'{prefix}_source_{cfg.source.split("/")[1]}_seed1'
    cfg.tensorboard_log_dir = str(Path(cfg.tensorboard_log_dir or
                                      Path(cfg.output_dir).parent.parent / 'tensorboard') /
                                  cfg.experiment_name)
    cfg.m1 = cfg.mode == 'M1'
    cfg.method = None
    cfg.train_on_target = cfg.eval = cfg.overall = False
    cfg.base_revision = BASE_REVISION
    cfg.code_revision = os.environ.get('CODE_REVISION') or None
    return cfg


def source_provenance():
    root = Path(__file__).resolve().parent
    paths = [root / 'train_m1_source.py', root / 'train.py', root / 'dataset.py',
             root / 'transforms.py', root / 'evaluation.py']
    paths.extend(sorted((root / 'models').rglob('*.py')))
    paths.extend(sorted((root / 'utils').glob('*.py')))
    return {'base_revision': BASE_REVISION,
            'code_revision': os.environ.get('CODE_REVISION') or None,
            'sha256': {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in paths if path.is_file()}}


def main(argv=None):
    cfg = build_config(argv)
    if cfg.dry_run:
        print(json.dumps(vars(cfg), indent=2))
        return
    output = Path(cfg.output_dir)
    # Atomic leaf creation prevents accidental checkpoint reuse or overwrite.
    output.mkdir(parents=True, exist_ok=False)
    for fold in range(cfg.num_folds):
        (output / f'fold_{fold}').mkdir()
    (output / 'train_config.json').write_text(json.dumps(vars(cfg), indent=2), encoding='utf-8')
    (output / 'code_provenance.json').write_text(json.dumps(source_provenance(), indent=2), encoding='utf-8')
    from train import main as train_main
    train_main(cfg)


if __name__ == '__main__':
    main()
