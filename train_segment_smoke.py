"""Bounded Source structure finetuning probe; never launches a full run or UDA."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from train_segment_timematch import build_config as build_training_config, validate_source


def build_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    for name, default in dict(steps=200, probe_every=25, probe_samples=128,
                              gradient_every=25, health_window=20, warmup_steps=20).items():
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--pelt_penalty', type=float, default=.8)
    parser.add_argument('--pelt_penalty_mode', choices=('absolute',), default='absolute')
    parser.add_argument('--sparse_weight', type=float, choices=(.01, .03, .1), default=.01)
    diagnostic, remaining = parser.parse_known_args(argv)
    if not 1 <= diagnostic.steps <= 1000:
        parser.error('steps must be between 1 and 1000')
    for name in ('probe_every', 'probe_samples', 'gradient_every', 'health_window'):
        if getattr(diagnostic, name) <= 0:
            parser.error(f'{name} must be positive')
    if diagnostic.warmup_steps < 0:
        parser.error('warmup_steps must be nonnegative')
    if not math.isfinite(diagnostic.pelt_penalty) or diagnostic.pelt_penalty != .8:
        parser.error('smoke experiment requires pelt_penalty=0.8 (absolute)')
    if any(token == '--stage' or token.startswith('--stage=') for token in remaining):
        parser.error('smoke has no selectable training stage')
    cfg = build_training_config(['--stage', 'finetune', *remaining])
    for name, value in vars(diagnostic).items():
        setattr(cfg, name, value)
    cfg.mode = cfg.stage = 'smoke'
    cfg.scheduler_epochs = 20
    cfg.fold_dir = str(Path(cfg.output_dir) / 'fold_0')
    cfg.structure_protocol = dict(source_full_weight=.5, source_keep_weight=.5,
        sparsity='mean per-head expected segment count',
        sparse_weight=cfg.sparse_weight,
        segmentation='PELT RBF on feature-zscored content, median-positive bandwidth with gamma multiplier 1, absolute penalty=0.8, minimum 3',
        selection='ReinMax tau=1', optimizer='Adam', scheduler='CosineAnnealingLR',
        scheduler_horizon='20 full Source epochs; probe stops at the configured steps',
        selector='shared CLS Transformer; length fraction in both variants; attention adds detached mass and enrichment',
        finetune='bounded Source-only smoke probe', uda=False, imputation=False)
    return cfg


def source_provenance():
    root = Path(__file__).resolve().parent
    paths = [root / 'train_segment_smoke.py', root / 'train_segment_timematch.py',
             root / 'train.py', root / 'dataset.py', root / 'transforms.py',
             root / 'evaluation.py', root / 'closedset_scripts/run_segment_smoke_8jobs_3gpu.sh']
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
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    Path(cfg.fold_dir).mkdir()
    (output / 'config.json').write_text(json.dumps(vars(cfg), indent=2), encoding='utf-8')
    (output / 'code_provenance.json').write_text(json.dumps(source_provenance(), indent=2), encoding='utf-8')
    from utils.segment_smoke import run_smoke
    run_smoke(cfg)


if __name__ == '__main__':
    main()
