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
    parser.add_argument('--stage', required=True, choices=('finetune', 'joint_source', 'uda'))
    parser.add_argument('--source_training_epochs', type=int, choices=(20, 100), default=20)
    parser.add_argument('--variant', required=True, choices=('inshape', 'attention', 'attention_shared'))
    parser.add_argument('--source', required=True, choices=TILES)
    parser.add_argument('--target', required=True, choices=TILES)
    parser.add_argument('--weights', required=True, help='Historical Original Source leaf containing fold_0/model.pt')
    parser.add_argument('--structure_checkpoint', help='Own variant/task finetune/fold_0/model.pt, required for UDA')
    parser.add_argument('--finetune_ltae_only', action='store_true',
                        help='Train LTAE and Selector while freezing PSE and classifier')
    parser.add_argument('--segment_context', action='store_true',
                        help='Apply one self-attention layer across candidate segment embeddings')
    parser.add_argument('--segment_refine', action='store_true',
                        help='Use class-guided calendar boundaries during source finetuning')
    parser.add_argument('--refinement_rounds', type=int, choices=(1, 2), default=0,
                        help='Run one or two alternating Split and Merge rounds in Source finetuning')
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
    floats = dict(lr=1e-4, weight_decay=1e-4, selector_lr=1e-4, keep_weight=1.,
        sparse_weight=.01, pelt_penalty=.8, reinmax_tau=1., selector_dropout=0., focal_loss_gamma=1.,
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
    parser.add_argument('--pelt_penalty_mode', choices=('absolute',), default='absolute')
    parser.add_argument('--dry_run', action='store_true')
    cfg = parser.parse_args(argv)
    if cfg.refinement_rounds:
        if cfg.stage != 'finetune':
            parser.error('--refinement_rounds requires --stage finetune')
        cfg.segment_refine = True
    if cfg.source == cfg.target:
        parser.error('specify the downstream source/target pair; finetuning uses source only internally')
    if cfg.stage == 'uda' and not cfg.structure_checkpoint:
        parser.error('UDA requires --structure_checkpoint from its own final source finetuning')
    if cfg.stage != 'uda' and cfg.structure_checkpoint:
        parser.error('finetune starts from --weights and does not accept --structure_checkpoint')
    if cfg.finetune_ltae_only and cfg.stage != 'finetune':
        parser.error('--finetune_ltae_only requires --stage finetune')
    if cfg.segment_context and (cfg.variant != 'inshape' or cfg.stage == 'joint_source'
                                or cfg.source_training_epochs != 20 or cfg.finetune_ltae_only):
        parser.error('--segment_context requires the matched inshape finetune/UDA route')
    if cfg.segment_refine and (cfg.variant != 'inshape' or cfg.stage == 'joint_source'
                              or cfg.source_training_epochs != 20 or cfg.finetune_ltae_only
                              or cfg.segment_context):
        parser.error('--segment_refine requires the pure INSHAPE finetune/UDA route')
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
    if cfg.refinement_rounds:
        cfg.refinement_adapt_epochs = 5
        cfg.epochs = cfg.source_training_epochs = 10 + 2 * cfg.refinement_adapt_epochs * cfg.refinement_rounds
        cfg.source_schedule_epochs = 30
    cfg.mode = None
    cfg.selector_protocol = 'inshape_drop_proxy_detached_v1' if cfg.variant == 'inshape' else 'legacy'
    if cfg.segment_context:
        cfg.selector_protocol = 'inshape_drop_proxy_context_v1'
    if cfg.variant == 'attention_shared':
        cfg.selector_protocol = 'inshape_attention_coverage_v1'
    if cfg.variant in ('inshape', 'attention_shared'):
        cfg.selector_width, cfg.selector_dropout, cfg.reinmax_tau, cfg.sparse_weight = 32, .1, 1.3, 3.
    cfg.source_init = 'scratch' if cfg.stage == 'joint_source' else 'checkpoint'
    if cfg.stage == 'uda' and cfg.source_training_epochs == 100:
        cfg.source_init = 'scratch'
    if cfg.stage == 'uda' and cfg.source_training_epochs == 100:
        cfg.source_init = 'scratch'
    if cfg.stage == 'joint_source':
        if cfg.variant != 'inshape':
            parser.error('Joint Source comparison requires restored INSHAPE')
        cfg.epochs, cfg.lr, cfg.selector_lr, cfg.source_training_epochs = 100, .001, .0005, 100
    cfg.m1 = cfg.eval = cfg.overall = cfg.train_on_target = False
    cfg.method = 'timematch'
    cfg.experiment_name = cfg.experiment_name or Path(cfg.output_dir).name
    cfg.tensorboard_log_dir = cfg.tensorboard_log_dir or str(Path(cfg.output_dir) / 'tensorboard')
    cfg.structure_protocol = dict(source_full_weight=0., source_keep_weight=1.,
        classification_path='keep_only_unified', sparse_updates='Selector only; no PSE/LTAE/Classifier',
        sparsity='mean per-head expected segment count', target='one Keep pseudo Focal loss',
        teacher='weak original observations, EMA PSE-LTAE classifier; no Teacher selector',
        segmentation='PELT RBF on feature-zscored content, median-positive bandwidth, absolute penalty=0.8, minimum 3',
        selection='ReinMax tau=1', optimizer='Adam', scheduler='CosineAnnealingLR',
        selector='shared CLS Transformer; length fraction in both variants; attention adds detached mass and enrichment',
        finetune='20 full source epochs, fixed final checkpoint',
        uda='20 epochs x 500 steps, fixed final joint Student and selector',
        imputation=False)
    if cfg.variant in ('inshape', 'attention_shared'):
        cfg.structure_protocol.update(selector='pure content CLS Transformer; one shared segment gate; initial bias p=0.8',
            sparsity='sampled pre-repair observation coverage times detached (2*coverage-1)',
            sparse_weight=cfg.sparse_weight, sparse_updates='Selector only', selection='ReinMax tau=1.3')
    if cfg.variant == 'attention_shared':
        cfg.structure_protocol.update(selector='content CLS plus detached 16-head mass and enrichment; one shared gate; zero-initialized attention weights',
                                      attention_conditions='32 features: 16 masses and 16 observation-count normalized enrichments')
    if cfg.variant == 'inshape':
        cfg.structure_protocol.update(keep_input='only retained PSE features and original dates enter LTAE',
                                      selector_input_gradient='detached PSE features; Keep classifier trains PSE',
                                      keep_gate_gradient='frozen LTAE pooling proxy, gates 0.05 + 0.95 * hard gate, forward canceled')
    if cfg.finetune_ltae_only:
        cfg.structure_protocol.update(finetune_trainable='LTAE and Selector',
                                      finetune_frozen='PSE and classifier including normalization buffers',
                                      selector_input_gradient='frozen PSE features')
    if cfg.segment_context:
        cfg.structure_protocol.update(segment_context='one 1-head residual self-attention over valid segment CLS vectors; zero output initialization')
    if cfg.segment_refine:
        cfg.structure_protocol.update(
            source_reference='full source training dates, frozen PSE, one class-feedback update after epoch 10',
            finetune_trainable='LTAE and Selector',
            finetune_frozen='PSE and classifier including normalization buffers',
            uda_reference='native per-view PELT; uses the refined Source checkpoint only')
    if cfg.refinement_rounds:
        cfg.structure_protocol.update(
            source_reference='full dates with frozen PSE; Split and Merge separated by Selector/LTAE training',
            finetune=f'{cfg.epochs} source epochs; 10 warmup, then 5 after each Split and each Merge',
            refinement_rounds=cfg.refinement_rounds,
            source_schedule_epochs=cfg.source_schedule_epochs)
    cfg.structure_protocol.update(source_init=cfg.source_init, source_training_epochs=cfg.source_training_epochs,
        actual_stage_epochs=cfg.epochs)
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
        root / 'closedset_scripts/run_segment_timematch_8jobs_3gpu.sh',
        root / 'closedset_scripts/run_segment_compare_8jobs_4gpu.sh',
        root / 'closedset_scripts/run_segment_attention_finetune_4gpu.sh']
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
    if cfg.stage != 'uda':
        finetune_structure(cfg)
    else:
        run_structure_uda(cfg)


if __name__ == '__main__':
    main()
