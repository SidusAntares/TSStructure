"""Isolated source-only structure finetuning and segment-selected TimeMatch UDA."""
import argparse
import hashlib
import json
import math
from pathlib import Path

TILES = ('austria/33UVP/2017', 'france/30TXT/2017',
         'france/31TCJ/2017', 'denmark/32VNH/2017')


def boolean(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('expected true or false')


def build_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('finetune', 'uda'))
    parser.add_argument('--variant', required=True, choices=('inshape', 'attention'))
    parser.add_argument('--source', required=True, choices=TILES)
    parser.add_argument('--target', required=True, choices=TILES)
    parser.add_argument('--weights', required=True, help='Historical Original Source leaf containing fold_0/model.pt')
    parser.add_argument('--structure_checkpoint', help='Own variant/task finetune/fold_0/model.pt, required for UDA')
    parser.add_argument('--output_dir', required=True, help='Exact new stage leaf; existing directories are refused')
    parser.add_argument('--experiment_name')
    parser.add_argument('--tensorboard_log_dir')
    parser.add_argument('--data_root', default='/data/user/dataset/timematch_data')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', choices=('pseltae',), default='pseltae')
    parser.add_argument('--progress_bar', choices=('auto', 'on', 'off'), default='off')
    integers = dict(seed=1, num_workers=8, epochs=20, steps_per_epoch=500, batch_size=128,
        seq_length=30, num_pixels=64, input_dim=10, num_folds=1, num_blocks=100,
        log_step=10, max_shift_aug=60, sample_size=100, max_temporal_shift=60,
        pelt_min_size=3, selector_width=64, selector_heads=4, selector_layers=1)
    floats = dict(lr=1e-4, weight_decay=1e-4, selector_lr=1e-4, keep_weight=.5,
        sparse_weight=.01, pelt_penalty=.1, reinmax_tau=1., selector_dropout=0., focal_loss_gamma=1.,
        val_ratio=.1, test_ratio=.2, shift_aug_p=1., pseudo_threshold=.9,
        ema_decay=.9999, trade_off=2.)
    booleans = dict(closed_set=True, sample_pixels_val=True, combine_spring_and_winter=False,
        with_extra=False, with_shift_aug=False, balance_source=True, use_focal_loss=True,
        estimate_shift=True, shift_source=True, domain_specific_bn=True, run_validation=False,
        log_target_labels=False, strict_final_student=True, output_student=True)
    for name, value in integers.items():
        parser.add_argument('--' + name, type=int, default=value)
    for name, value in floats.items():
        parser.add_argument('--' + name, type=float, default=value)
    for name, value in booleans.items():
        parser.add_argument('--' + name, type=boolean, default=value)
    parser.add_argument('--shift_estimator', choices=('AM',), default='AM')
    parser.add_argument('--evidence_mode', choices=('off',), default='off')
    parser.add_argument('--dry_run', action='store_true')
    cfg = parser.parse_args(argv)
    if cfg.source == cfg.target:
        parser.error('specify the downstream source/target pair; finetuning uses source only internally')
    if cfg.stage == 'uda' and not cfg.structure_checkpoint:
        parser.error('UDA requires --structure_checkpoint from its own final source finetuning')
    if cfg.stage == 'finetune' and cfg.structure_checkpoint:
        parser.error('finetune starts from --weights and does not accept --structure_checkpoint')
    # The approved run is a fixed matched comparison; refuse silent protocol drift.
    fixed = dict(integers, **floats, **booleans)
    for name in ('log_step',):
        fixed.pop(name)
    for name, expected in fixed.items():
        value = getattr(cfg, name)
        if isinstance(value, float) and not math.isfinite(value):
            parser.error(f'{name} must be finite')
        if value != expected:
            parser.error(f'matched segment experiment requires {name}={expected}')
    if cfg.log_step <= 0:
        parser.error('log_step must be positive')
    cfg.mode = None
    cfg.m1 = cfg.eval = cfg.overall = cfg.train_on_target = False
    cfg.method = 'timematch'
    cfg.experiment_name = cfg.experiment_name or Path(cfg.output_dir).name
    cfg.tensorboard_log_dir = cfg.tensorboard_log_dir or str(Path(cfg.output_dir) / 'tensorboard')
    cfg.structure_protocol = dict(source_full_weight=.5, source_keep_weight=.5,
        sparsity='mean per-head expected segment count', target='one Keep pseudo Focal loss',
        teacher='weak original observations, EMA PSE-LTAE classifier; no Teacher selector',
        segmentation='PELT RBF on normalized content, penalty=0.1*T, minimum 3',
        selection='ReinMax tau=1', optimizer='Adam', scheduler='CosineAnnealingLR',
        selector='shared CLS Transformer; length fraction in both variants; attention adds detached mass and enrichment',
        finetune='20 full source epochs, fixed final checkpoint',
        uda='20 epochs x 500 steps, fixed final joint Student and selector',
        imputation=False)
    return cfg


def validate_source(config):
    """Verify the archived 100-epoch Original Source before touching output."""
    leaf = Path(config.weights)
    checkpoint = leaf / 'fold_0/model.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Original Source checkpoint missing: {checkpoint}')
    source = json.loads((leaf / 'train_config.json').read_text(encoding='utf-8'))
    expected = dict(source=config.source, target=config.source, seed=1, num_workers=8,
        epochs=100, model='pseltae', closed_set=True, num_folds=1, num_blocks=100,
        val_ratio=.1, test_ratio=.2, sample_pixels_val=True, batch_size=128,
        seq_length=30, num_pixels=64, input_dim=10, combine_spring_and_winter=False,
        with_extra=False, with_shift_aug=False, lr=.001, weight_decay=.0001,
        focal_loss_gamma=1., shift_aug_p=1., max_shift_aug=60)
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f'Original Source config mismatch: {key}={source.get(key)!r}, expected {value!r}')
    if source.get('mode') is not None or source.get('m1', False) or source.get('method') is not None:
        raise ValueError('Original Source requires mode=None, m1=False and method=None')
    config.source_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def source_provenance():
    root = Path(__file__).resolve().parent
    paths = [root / 'train_segment_timematch.py', root / 'train.py', root / 'timematch.py',
        root / 'dataset.py', root / 'transforms.py', root / 'evaluation.py',
        root / 'closedset_scripts/run_segment_timematch_8jobs_3gpu.sh']
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
    if cfg.stage == 'uda':
        checkpoint = Path(cfg.structure_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Final structure finetuning checkpoint missing: {checkpoint}')
        cfg.structure_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'fold_0').mkdir()
    (output / 'config.json').write_text(json.dumps(vars(cfg), indent=2), encoding='utf-8')
    (output / 'code_provenance.json').write_text(json.dumps(source_provenance(), indent=2), encoding='utf-8')
    from utils.segment_training import finetune_structure, run_structure_uda
    if cfg.stage == 'finetune':
        finetune_structure(cfg)
    else:
        run_structure_uda(cfg)


if __name__ == '__main__':
    main()
