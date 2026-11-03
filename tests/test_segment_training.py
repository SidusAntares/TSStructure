from types import SimpleNamespace

import torch
import pytest

from models.stclassifier import PseLTae
from models.segment_selector import SegmentSelector
from utils.segment_training import train_source_structure_epochs, StructureRuntime


def models():
    torch.manual_seed(7)
    return PseLTae(num_classes=3, with_extra=False), SegmentSelector(width=16, heads=2)


def sample():
    return dict(pixels=torch.rand(4, 30, 10, 8), valid_pixels=torch.ones(4, 30, 8),
                positions=torch.arange(30)[None].expand(4, -1)*11,
                extra=torch.zeros(4, 4), label=torch.tensor([0, 1, 2, 0]))


def config(tmp_path):
    return SimpleNamespace(device='cpu', epochs=20, lr=1e-4, selector_lr=1e-4,
        weight_decay=1e-4, focal_loss_gamma=1., keep_weight=1., sparse_weight=.01,
        pelt_penalty=.8, pelt_penalty_mode='absolute', pelt_min_size=3, variant='attention', fold_dir=str(tmp_path),
        original_source_checkpoint_sha256='original_source', source='source', seed=1)


def test_source_twenty_epoch_joint_finetune_exports_final_not_best(tmp_path):
    model, selector = models()
    old_model = {k: v.clone() for k, v in model.state_dict().items()}
    old_selector = {k: v.clone() for k, v in selector.state_dict().items()}
    rows = train_source_structure_epochs(model, selector, [sample()], config(tmp_path))
    assert len(rows) == 20 and rows[-1]['epoch'] == 20
    packet = torch.load(tmp_path/'model.pt', weights_only=False)
    assert packet['role'] == 'source_structure' and packet['completed_epochs'] == 20
    assert packet['original_source_checkpoint_sha256'] == 'original_source'
    assert packet['classification_path'] == 'keep_only_unified'
    assert 'full' not in rows[0]
    assert any(not torch.equal(v, old_model[k]) for k, v in packet['state_dict'].items())
    assert any(not torch.equal(v, old_selector[k]) for k, v in packet['selector_state_dict'].items())
    assert model.temporal_encoder.positional_enc.weight.requires_grad is False
    torch.testing.assert_close(packet['state_dict']['temporal_encoder.positional_enc.weight'],
                               old_model['temporal_encoder.positional_enc.weight'], rtol=0, atol=0)


def test_structure_runtime_source_uses_single_keep_classification(tmp_path):
    from utils.focal_loss import FocalLoss
    model, selector = models()
    runtime = StructureRuntime(config(tmp_path), selector)
    runtime.attach(model, model)
    data = sample()
    logits = runtime.classification(data['pixels'], data['valid_pixels'], data['positions'], data['extra'])
    criterion = FocalLoss(gamma=1)
    torch.testing.assert_close(runtime.source_loss(criterion, data['label']), criterion(logits, data['label']))
    assert runtime.full_source is None
    assert model.temporal_encoder.mlp[0].norm.num_batches_tracked == 1


def test_uda_structure_trains_both_domains_without_target_labels(tmp_path, monkeypatch):
    import timematch
    model, selector = models()
    source = tmp_path/'fine/fold_0'
    source.mkdir(parents=True)
    torch.save({'state_dict': model.state_dict()}, source/'model.pt')
    cfg = config(tmp_path)
    cfg.__dict__.update(weights=str(tmp_path/'fine'), num_classes=3, balance_source=True,
        use_focal_loss=True, steps_per_epoch=1, max_temporal_shift=60,
        estimate_shift=False, pseudo_threshold=0., trade_off=2., domain_specific_bn=True,
        ema_decay=.9999, log_step=1, run_validation=False, log_target_labels=False,
        output_student=True, progress_bar='off', structure_checkpoint_sha256='fine20')
    cfg._structure_runtime = StructureRuntime(cfg, selector)
    batch = sample()
    unlabeled = {k: v for k, v in batch.items() if k != 'label'}
    monkeypatch.setattr(timematch, 'get_data_loaders', lambda *a: ([batch], None, [(unlabeled, unlabeled)]))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    calls = []
    original = cfg._structure_runtime.classification
    def classification(*args, **kwargs):
        domain = kwargs.get('domain', args[4] if len(args) > 4 else 'source')
        calls.append(domain)
        return original(*args, **kwargs)
    monkeypatch.setattr(cfg._structure_runtime, 'classification', classification)
    writer = SimpleNamespace(add_scalar=lambda *a, **kw: None)
    out = tmp_path/'final.pt'
    timematch.train_timematch(model, cfg, writer, None, torch.device('cpu'), str(out), 0, {})
    packet = torch.load(out, weights_only=False)
    assert packet['role'] == 'student' and packet['completed_epochs'] == 20
    assert 'selector_state_dict' in packet and packet['structure_initial_sha256'] == 'fine20'
    assert calls.count('source') == calls.count('target') == 20


@pytest.mark.parametrize('source_stage', ['finetune', 'joint_source'])
def test_independent_entry_finetune_then_uda_and_endpoint_reports(tmp_path, monkeypatch, capsys, source_stage):
    import json
    import sys
    import types
    import dataset
    import timematch
    import train_segment_timematch as entry
    from test_m1_training import training_module
    train = training_module(monkeypatch)
    source, target = 'austria/33UVP/2017', 'denmark/32VNH/2017'
    weights = tmp_path/'closedset_source_33UVP_seed1'
    (weights/'fold_0').mkdir(parents=True)
    model, _ = models()
    torch.save({'state_dict': model.state_dict()}, weights/'fold_0/model.pt')
    common = ['--source', source, '--target', target, '--weights', str(weights), '--device', 'cpu',
              '--variant', 'inshape']
    old_builder = entry.build_config
    source_config = vars(old_builder(common+['--stage', 'finetune', '--output_dir', str(tmp_path/'fine')])).copy()
    source_config.update(target=source, epochs=100, lr=.001, method=None)
    (weights/'train_config.json').write_text(json.dumps(source_config))
    (weights/'closed_set_protocol.json').write_text(json.dumps({'classes': ['a', 'b', 'c']}))
    def protocol(cfg):
        cfg.classes, cfg.num_classes = ['a', 'b', 'c'], 3
        return {cfg.source: list(range(200)), cfg.target: list(range(200))}, None
    monkeypatch.setattr(train, 'prepare_data_protocol', protocol)
    batch = sample()
    unlabeled = {k: v for k, v in batch.items() if k != 'label'}
    seen = []
    def pixels(root, domain, *args, **kwargs):
        seen.append(domain)
        return [batch]
    monkeypatch.setattr(dataset, 'PixelSetData', pixels)
    monkeypatch.setattr(dataset, 'create_train_loader', lambda *a: [batch])
    evaluation_ids = []
    def evaluation_loader(domain, splits, cfg, *args):
        evaluation_ids.append((domain, set(splits[domain]['test'])))
        return [batch], [batch]
    monkeypatch.setattr(dataset, 'create_evaluation_loaders', evaluation_loader)
    class Writer:
        def __init__(self, **kwargs): pass
        def add_scalar(self, *a, **kw): pass
        def close(self): pass
    tb = types.ModuleType('torch.utils.tensorboard')
    tb.SummaryWriter = Writer
    monkeypatch.setitem(sys.modules, 'torch.utils.tensorboard', tb)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    entry.main(common+['--stage', source_stage, '--output_dir', str(tmp_path/'fine')])
    assert seen == [source]
    initial = tmp_path/'fine/fold_0/model.pt'
    assert initial.is_file()
    source_packet = torch.load(initial, weights_only=False)
    assert source_packet['completed_epochs'] == (100 if source_stage == 'joint_source' else 20)
    assert source_packet['source_init'] == ('scratch' if source_stage == 'joint_source' else 'checkpoint')
    def synthetic_budget(argv=None):
        cfg = old_builder(argv)
        if cfg.stage == 'uda': cfg.steps_per_epoch = 1
        return cfg
    monkeypatch.setattr(entry, 'build_config', synthetic_budget)
    monkeypatch.setattr(timematch, 'get_data_loaders', lambda *a: ([batch], None, [(unlabeled, unlabeled)]))
    monkeypatch.setattr(timematch, 'estimate_temporal_shift', lambda *a, **kw: 8)
    monkeypatch.setattr(timematch, 'get_pseudo_labels', lambda *a, **kw: torch.eye(3))
    leaf = tmp_path/'uda'
    entry.main(common+['--stage', 'uda', '--structure_checkpoint', str(initial), '--output_dir', str(leaf),
                       '--source_training_epochs', '100' if source_stage == 'joint_source' else '20'])
    packet = torch.load(leaf/'fold_0/model.pt', weights_only=False)
    assert packet['role'] == 'student' and packet['completed_epochs'] == 20
    tested = json.loads((leaf/'fold_0/tested_checkpoint.json').read_text())
    assert tested['variant'] == 'inshape' and tested['structure_initial_sha256']
    for domain in ('austria_33UVP_2017', 'denmark_32VNH_2017'):
        report = json.loads((leaf/f'fold_0/test_{domain}.json').read_text())
        assert 'full' in report and 'keep' in report
        assert (leaf/f'fold_0/logits_{domain}.npz').is_file()
    raw = json.loads((leaf/'split_indices.json').read_text())
    fine = json.loads((tmp_path/'fine/split_indices.json').read_text())
    assert evaluation_ids[-2][1] == set(raw[source]['test']) & set(fine[source]['test'])
    assert evaluation_ids[-1][1] == set(raw[target]['test'])
    text = capsys.readouterr().out
    assert 'STRUCTURE_TEST_START' in text and 'input=full|accuracy=' in text and 'input=keep|accuracy=' in text
