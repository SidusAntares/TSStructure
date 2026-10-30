from types import SimpleNamespace

import torch

from models.stclassifier import PseLTae
from models.timeperceiver_imputer import TimePerceiverImputer
from utils.imputation_training import augmented_classification, feature_reconstruction, train_imputer_epochs


def setup():
    torch.manual_seed(3)
    model = PseLTae(num_classes=3, with_extra=False).eval()
    imputer = TimePerceiverImputer(d_model=32, num_latents=4, rounds=1,
                                  self_blocks=1, heads=8, dropout=0.)
    dates = torch.arange(12)[None].expand(4, -1)*30
    features = torch.rand(4, 12, 128, requires_grad=True)
    attention = torch.zeros(4, 16, 12)
    attention[:, :, 5] = 1
    return model, imputer, features, dates, attention


def test_production_imputer_configuration_forward_and_backward():
    torch.manual_seed(11)
    imputer = TimePerceiverImputer()
    assert imputer.latents.shape == (32, 256) and len(imputer.rounds) == 2
    assert all(len(block.blocks) == 3 for block in imputer.rounds)
    features = torch.rand(2, 30, 128, requires_grad=True)
    dates = torch.arange(30)[None].expand(2, -1)*12
    queries = torch.tensor([[42., 126.], [54., 138.]])
    result = imputer(features, dates, queries)
    assert result['prediction'].shape == (2, 2, 128)
    loss = result['prediction'].square().mean() + result['log_variance'].square().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(features.grad).all()
    assert imputer.latents.grad.abs().sum() > 0


def test_source_and_target_classification_gradient_contract():
    for detached in (False, True):
        model, imputer, features, dates, attention = setup()
        logits, info = augmented_classification(model, imputer, features, dates, attention,
                                                detach_prediction=detached)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0])).backward()
        assert features.grad is not None and features.grad.abs().sum() > 0
        assert model.decoder[0].linear.weight.grad.abs().sum() > 0
        grads = [p.grad for p in imputer.parameters() if p.grad is not None]
        assert bool(grads) != detached
        assert all(torch.isfinite(g).all() for g in grads)
        assert info['query_count'] > 0


def test_recovery_updates_only_imputer_and_uses_visible_brackets():
    model, imputer, features, dates, _ = setup()
    info = feature_reconstruction(model, imputer, features, dates)
    assert info['count'] > 0
    info['loss'].backward()
    assert features.grad is None
    assert all(p.grad is None for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in imputer.parameters())


def test_no_gaps_recovers_original_classification_and_fractional_days_work():
    model, imputer, features, dates, attention = setup()
    dense = dates // 10
    expected = model.decoder(model.temporal_encoder(features, dense))
    got, info = augmented_classification(model, imputer, features, dense, attention)
    assert info['query_count'] == 0
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    # Thirds are preserved as fractional dates rather than rounded onto a new grid.
    dates = torch.arange(12)[None].expand(4, -1)*41
    got, info = augmented_classification(model, imputer, features, dates, attention, shift=-10)
    assert torch.isfinite(got).all() and info['query_count'] > 0


def test_twenty_epoch_frozen_source_smoke(tmp_path):
    model, imputer, _, dates, _ = setup()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    old = {k: v.clone() for k, v in imputer.state_dict().items()}
    batch = dict(pixels=torch.rand(4, 12, 10, 8), valid_pixels=torch.ones(4, 12, 8),
                 positions=dates, extra=torch.zeros(4, 4), label=torch.tensor([0, 1, 2, 0]))
    config = SimpleNamespace(device='cpu', imputer_lr=1e-4, imputer_weight_decay=1e-4,
        imputer_epochs=20, pretrain_recovery_epochs=10, reconstruction_weight=1.,
        source_aug_weight=1., nll_weight=.01, focal_loss_gamma=1., progress_bar='off',
        fold_dir=str(tmp_path), source='source', source_checkpoint_sha256='synthetic')
    history = train_imputer_epochs(model, imputer, [batch], config)
    assert len(history) == 20
    assert all(row['classification'] == 0 for row in history[:10])
    assert all(row['classification'] > 0 for row in history[10:])
    assert all(torch.equal(value, initial[k]) for k, value in model.state_dict().items())
    assert any(not torch.equal(value, old[k]) for k, value in imputer.state_dict().items())
    packet = torch.load(tmp_path/'imputer_epoch20.pt', weights_only=False)
    assert packet['completed_epochs'] == 20


def test_imputed_uda_exports_joint_final_student_without_target_labels(tmp_path, monkeypatch):
    import timematch
    from utils.imputation_training import ImputationRuntime
    model, imputer, _, dates, _ = setup()
    source_dir = tmp_path/'source/fold_0'
    source_dir.mkdir(parents=True)
    torch.save({'state_dict': model.state_dict()}, source_dir/'model.pt')
    old_student = {k: v.clone() for k, v in model.state_dict().items()}
    old_imputer = {k: v.clone() for k, v in imputer.state_dict().items()}
    sample = dict(pixels=torch.rand(4, 12, 10, 8), valid_pixels=torch.ones(4, 12, 8),
                  positions=dates, extra=torch.zeros(4, 4), label=torch.tensor([0, 1, 2, 0]))
    unlabeled = {key: value for key, value in sample.items() if key != 'label'}
    monkeypatch.setattr(timematch, 'get_data_loaders', lambda *a: ([sample], None, [(unlabeled, unlabeled)]))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    cfg = SimpleNamespace(weights=str(tmp_path/'source'), num_classes=3, balance_source=True,
        use_focal_loss=True, focal_loss_gamma=1., lr=1e-4, imputer_lr=1e-4, weight_decay=1e-4,
        imputer_weight_decay=1e-4, epochs=20, steps_per_epoch=1, max_temporal_shift=60,
        estimate_shift=False, pseudo_threshold=0., trade_off=2., domain_specific_bn=True,
        ema_decay=.9999, log_step=1, run_validation=False, log_target_labels=False,
        output_student=True, progress_bar='off', variant='A', reconstruction_weight=.1,
        nll_weight=.01, imputer_checkpoint_sha256='same_initial_imputer',
        source_checkpoint_sha256='original_source', fold_dir=str(tmp_path))
    cfg._imputation_runtime = ImputationRuntime(cfg, imputer)
    writer = SimpleNamespace(add_scalar=lambda *a, **kw: None)
    output = tmp_path/'final.pt'
    timematch.train_timematch(model, cfg, writer, None, torch.device('cpu'), str(output), 0, {})
    packet = torch.load(output, weights_only=False)
    assert packet['role'] == 'student' and packet['completed_epochs'] == 20
    assert packet['imputer_initial_sha256'] == 'same_initial_imputer'
    assert any(not torch.equal(value, old_student[k]) for k, value in packet['state_dict'].items())
    assert any(not torch.equal(value, old_imputer[k]) for k, value in packet['imputer_state_dict'].items())


def test_independent_entry_pretrain_then_both_variants_end_to_end(tmp_path, monkeypatch):
    import json
    import sys
    import types
    import dataset
    import timematch
    import train_imputed_timematch as entry
    import utils.imputation_training as runtime
    from test_m1_training import training_module
    train = training_module(monkeypatch)
    source, target = 'austria/33UVP/2017', 'denmark/32VNH/2017'
    weights = tmp_path/'closedset_source_33UVP_seed1'
    (weights/'fold_0').mkdir(parents=True)
    model, _, _, dates, _ = setup()
    torch.save({'state_dict': model.state_dict()}, weights/'fold_0/model.pt')
    common = ['--source', source, '--target', target, '--weights', str(weights), '--device', 'cpu']
    source_config = vars(entry.build_config(common+['--stage', 'pretrain', '--output_dir', str(tmp_path/'pretrain')])).copy()
    source_config.update(target=source, epochs=100, lr=.001, method=None)
    (weights/'train_config.json').write_text(json.dumps(source_config))
    (weights/'closed_set_protocol.json').write_text(json.dumps({'classes': ['a', 'b', 'c']}))
    def protocol(cfg):
        cfg.classes, cfg.num_classes = ['a', 'b', 'c'], 3
        return {cfg.source: list(range(20)), cfg.target: list(range(20))}, None
    monkeypatch.setattr(train, 'prepare_data_protocol', protocol)
    batch = dict(pixels=torch.rand(4, 12, 10, 8), valid_pixels=torch.ones(4, 12, 8),
                 positions=dates, extra=torch.zeros(4, 4), label=torch.tensor([0, 1, 2, 0]))
    unlabeled = {k: v for k, v in batch.items() if k != 'label'}
    seen = []
    def pixels(root, domain, *args, **kwargs):
        seen.append(domain)
        return [batch]
    monkeypatch.setattr(dataset, 'PixelSetData', pixels)
    monkeypatch.setattr(dataset, 'create_train_loader', lambda *a: [batch])
    monkeypatch.setattr(dataset, 'create_evaluation_loaders', lambda *a: ([batch], [batch]))
    monkeypatch.setattr(runtime, '_new_imputer', lambda cfg: TimePerceiverImputer(
        d_model=32, num_latents=4, rounds=1, self_blocks=1, heads=8, dropout=0.))
    class Writer:
        def __init__(self, **kwargs): pass
        def add_scalar(self, *a, **kw): pass
        def close(self): pass
    tb = types.ModuleType('torch.utils.tensorboard')
    tb.SummaryWriter = Writer
    monkeypatch.setitem(sys.modules, 'torch.utils.tensorboard', tb)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    entry.main(common+['--stage', 'pretrain', '--output_dir', str(tmp_path/'pretrain')])
    assert seen == [source]
    initial = tmp_path/'pretrain/fold_0/imputer_epoch20.pt'
    assert initial.is_file()
    assert (tmp_path/'pretrain/fold_0/test_austria_33UVP_2017.json').is_file()
    monkeypatch.setattr(timematch, 'get_data_loaders', lambda *a: ([batch], None, [(unlabeled, unlabeled)]))
    monkeypatch.setattr(timematch, 'estimate_temporal_shift', lambda *a, **kw: 8)
    monkeypatch.setattr(timematch, 'get_pseudo_labels', lambda *a, **kw: torch.eye(3))
    hashes = []
    for variant in ('A', 'B'):
        leaf = tmp_path/variant
        entry.main(common+['--stage', 'uda', '--variant', variant,
            '--imputer_checkpoint', str(initial), '--output_dir', str(leaf), '--steps_per_epoch', '1',
            '--pseudo_threshold', '0.0', '--sample_size', '1'])
        tested = json.loads((leaf/'fold_0/tested_checkpoint.json').read_text())
        hashes.append(tested['imputer_initial_sha256'])
        for domain in ('austria_33UVP_2017', 'denmark_32VNH_2017'):
            report = json.loads((leaf/f'fold_0/test_{domain}.json').read_text())
            assert 'original' in report and 'augmented' in report
            assert (leaf/f'fold_0/logits_{domain}.npz').is_file()
    assert hashes[0] == hashes[1]
