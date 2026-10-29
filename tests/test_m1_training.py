import csv
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def training_module(monkeypatch):
    if 'train' not in sys.modules:
        tensorboard = types.ModuleType('torch.utils.tensorboard')
        tensorboard.SummaryWriter = object
        monkeypatch.setitem(sys.modules, 'torch.utils.tensorboard', tensorboard)
        for name, function in [('competitors.dann.dann', 'train_dann'),
                               ('competitors.jumbot.jumbot', 'train_jumbot'),
                               ('competitors.mmd.train_mmd', 'train_mmd'),
                               ('competitors.alda.train_alda', 'train_alda')]:
            module = types.ModuleType(name)
            setattr(module, function, lambda *a: None)
            monkeypatch.setitem(sys.modules, name, module)
    import train
    return train


def test_m1_rejects_target_validation_before_reading_any_data(monkeypatch):
    train = training_module(monkeypatch)
    with pytest.raises(ValueError, match='source==target'):
        train.main(SimpleNamespace(m1=True, source='source', target='target', method=None,
                                   train_on_target=False, model='pseltae'))


def test_m1_source_training_smoke_validates_saves_and_restores(tmp_path, monkeypatch):
    train = training_module(monkeypatch)
    from models.stclassifier import PseLTae
    from utils.m1_training import paired_source_evaluation
    torch.manual_seed(13)
    model = PseLTae(num_classes=4, with_extra=False)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    sample = dict(pixels=torch.rand(4, 30, 10, 8), valid_pixels=torch.ones(4, 30, 8),
                  positions=torch.arange(30)[None].repeat(4, 1)*10,
                  extra=torch.zeros(4, 4), label=torch.arange(4))
    used_domains = []
    def source_dataset(root, name, *args, **kwargs):
        used_domains.append(name)
        return [sample, sample]
    monkeypatch.setattr(train, 'PixelSetData', source_dataset)
    monkeypatch.setattr(train, 'create_train_loader', lambda *a: [sample, sample])
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    config = SimpleNamespace(
        m1=True, mode='M1', source='source', target='source', data_root='synthetic',
        train_on_target=False, closed_set=True, combine_spring_and_winter=False,
        classes=['a', 'b', 'c', 'd'], batch_size=4, num_workers=8, epochs=3,
        num_pixels=8, seq_length=30, with_shift_aug=False, max_shift_aug=60, shift_aug_p=1.,
        lr=.001, weight_decay=.0001, focal_loss_gamma=1., log_step=1, progress_bar='off',
        fold_dir=str(tmp_path), m1_rho=.5, m1_tau_start=.5, m1_tau_end=.1,
        m1_warmup_epochs=1, m1_lambda_keep=1., m1_lambda_budget=.1, m1_sharpness=1.,
        m1_degenerate_tolerance=.001, m1_degenerate_patience=5,
    )
    logged = []
    writer = SimpleNamespace(add_scalar=lambda *a, **kw: logged.append(a), add_histogram=lambda *a: None)
    path = tmp_path / 'model.pt'
    train.train_supervised(model, config, writer, {'source': {'train': set(range(8))}},
                           [sample], torch.device('cpu'), str(path))
    assert used_domains == ['source']
    with (tmp_path / 'm1_training.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert all(np.isfinite(float(row[key])) for row in rows for key in ('full', 'keep', 'budget', 'total', 'per_head_std_min'))
    assert any(not torch.equal(value, initial[key]) for key, value in model.state_dict().items())
    assert model.temporal_encoder.mlp[0].norm.num_batches_tracked == 6
    packet = torch.load(path, map_location='cpu', weights_only=False)
    assert packet['selection_domain'] == 'source' and packet['role'] == 'source_supervised'
    assert torch.load(tmp_path / 'model_last.pt', weights_only=False)['completed_epochs'] == 3
    restored = PseLTae(num_classes=4, with_extra=False)
    restored.load_state_dict(packet['state_dict'])
    paired = paired_source_evaluation(restored, [sample], config, torch.device('cpu'), .1)
    assert all(np.isfinite(paired[name]['loss']) for name in ('full', 'keep'))
    assert any(row[0] == 'm1/loss_keep' for row in logged)


def test_original_uda_can_disable_all_true_target_monitoring(tmp_path, monkeypatch):
    import timematch
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(2, 3)
            self.temporal_encoder = SimpleNamespace(max_temporal_shift=100,
                positional_enc=SimpleNamespace(num_embeddings=565))
        def forward(self, pixels, valid, positions, extra): return self.classifier(pixels)
    student = Student()
    source = tmp_path / 'source/fold_0'
    source.mkdir(parents=True)
    torch.save({'state_dict': student.state_dict()}, source / 'model.pt')
    sample = dict(pixels=torch.rand(4, 2), valid_pixels=torch.ones(4, 1),
                  positions=torch.arange(4)[:, None], extra=torch.zeros(4, 4), label=torch.tensor([0, 1, 2, 0]))
    unlabeled = {key: value for key, value in sample.items() if key != 'label'}
    def forbidden_labels(): raise AssertionError('True target labels were accessed')
    target = SimpleNamespace(dataset=SimpleNamespace(get_labels=forbidden_labels))
    monkeypatch.setattr(timematch, 'get_data_loaders', lambda *a: ([sample], target, [(unlabeled, unlabeled)]))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(timematch, 'estimate_temporal_shift', lambda *a, **kw: 8)
    monkeypatch.setattr(timematch, 'get_pseudo_labels', lambda *a, **kw: torch.eye(3))
    config = SimpleNamespace(weights=str(tmp_path / 'source'), balance_source=True, num_classes=3,
        use_focal_loss=False, steps_per_epoch=1, lr=.0001, weight_decay=.0001, epochs=20,
        max_temporal_shift=60, estimate_shift=True, shift_estimator='AM', shift_source=True, sample_size=1,
        pseudo_threshold=0., domain_specific_bn=True, trade_off=2., ema_decay=.9999,
        log_step=1, run_validation=False, log_target_labels=False, output_student=True, progress_bar='off')
    logged = []
    writer = SimpleNamespace(add_scalar=lambda *a: logged.append(a))
    path = tmp_path / 'final.pt'
    timematch.train_timematch(student, config, writer, None, torch.device('cpu'), str(path), 0, {})
    assert not any(row[0] in ('train/pseudo_f1', 'train/kl_d') for row in logged)
    assert torch.load(path, weights_only=False)['completed_epochs'] == 20


@pytest.mark.parametrize('mode', ['M0', 'M1'])
def test_independent_source_entry_full_pipeline_smoke(tmp_path, monkeypatch, mode):
    import json
    import train_m1_source as entry
    train = training_module(monkeypatch)
    source = 'austria/33UVP/2017'
    sample = dict(pixels=torch.rand(4, 30, 10, 8), valid_pixels=torch.ones(4, 30, 8),
                  positions=torch.arange(30)[None].repeat(4, 1)*10,
                  extra=torch.zeros(4, 4), label=torch.arange(4))
    observed = []
    def protocol(config):
        assert config.source == config.target == source
        config.classes, config.num_classes = ['a', 'b', 'c', 'd'], 4
        return {source: list(range(20))}, None
    def evaluation_loaders(name, *args):
        observed.append(name)
        return [sample], [sample]
    monkeypatch.setattr(train, 'prepare_data_protocol', protocol)
    monkeypatch.setattr(train, 'create_evaluation_loaders', evaluation_loaders)
    monkeypatch.setattr(train, 'PixelSetData', lambda *a, **kw: [sample, sample])
    monkeypatch.setattr(train, 'create_train_loader', lambda *a: [sample, sample])
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(train, 'SummaryWriter', lambda **kw: SimpleNamespace(
        add_scalar=lambda *a, **kw: None, add_histogram=lambda *a: None))
    entry.main(['--source', source, '--mode', mode, '--device', 'cpu', '--epochs', '2',
                '--batch_size', '4', '--num_pixels', '8', '--output_dir', str(tmp_path),
                '--experiment_name', 'smoke_'+mode, '--tensorboard_log_dir', str(tmp_path / 'tensorboard'),
                '--progress_bar', 'off'])
    root = tmp_path / ('smoke_'+mode)
    assert observed == [source]
    assert (root / 'fold_0/test_metrics_austria_33UVP_2017.json').is_file()
    kept = json.loads((root / 'fold_0/soft_top50_keep_source_test_metrics.json').read_text())
    assert kept['domain'] == source and np.isfinite(kept['macro_f1'])
    packet = torch.load(root / 'fold_0/model.pt', map_location='cpu', weights_only=False)
    assert packet['selection_domain'] == source
    assert (root / 'fold_0/m1_training.csv').exists() == (mode == 'M1')


@pytest.mark.parametrize('estimator', ['IS', 'ENT', 'AM'])
def test_real_shift_estimator_needs_no_target_labels(monkeypatch, capsys, estimator):
    import timematch
    from models.stclassifier import PseLTae
    torch.manual_seed(19)
    model = PseLTae(num_classes=3, with_extra=False)
    sample = dict(pixels=torch.rand(4, 30, 10, 8), valid_pixels=torch.ones(4, 30, 8),
                  positions=torch.arange(30)[None].repeat(4, 1)*10,
                  extra=torch.zeros(4, 4))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    kwargs = dict(min_shift=-1, max_shift=1, sample_size=1, shift_estimator=estimator,
                  class_distribution=np.ones(3)/3, progress_bar='off')
    shift = timematch.estimate_temporal_shift(model, [sample], torch.device('cpu'),
                                            log_target_labels=False, **kwargs)
    output = capsys.readouterr().out
    assert shift in (-1, 0, 1) and 'accuracy' not in output and 'Most accurate' not in output
    labeled = dict(sample, label=torch.arange(4) % 3)
    assert timematch.estimate_temporal_shift(model, [labeled], torch.device('cpu'), **kwargs) == shift
    with pytest.raises(ValueError, match='requires true target labels'):
        timematch.estimate_temporal_shift(model, [], torch.device('cpu'),
                                         shift_estimator='ACC', log_target_labels=False)


@pytest.mark.parametrize('augmentation,expected_dates', [('strong', 30), ('weak', 60)])
def test_target_student_augmentation_preserves_source_and_teacher(monkeypatch, augmentation, expected_dates):
    import timematch
    # Match the server's legacy NumPy alias while exercising the real transforms.
    monkeypatch.setitem(np.__dict__, 'long', np.int64)
    class SyntheticDataset:
        def __init__(self, root, name, classes, transform, **kwargs):
            self.transform = transform
        def __len__(self):
            return 4
        def __getitem__(self, index):
            sample = dict(pixels=np.ones((60, 10, 64), dtype=np.float32),
                          positions=np.arange(60), extra=np.zeros(4), label=0)
            return self.transform(sample)
    monkeypatch.setattr(timematch, 'PixelSetData', SyntheticDataset)
    class Loader:
        def __init__(self, dataset, **kwargs): self.dataset = dataset
        def __len__(self): return 1
    monkeypatch.setattr(timematch.data, 'DataLoader', Loader)
    config = SimpleNamespace(num_pixels=64, seq_length=30, with_shift_aug=False,
                             data_root='synthetic', source='source', target='target', classes=['a'],
                             closed_set=True, combine_spring_and_winter=False, num_workers=8,
                             batch_size=4, student_target_aug=augmentation)
    source, no_aug, paired = timematch.get_data_loaders(
        {'source': {'train': [0]}, 'target': {'train': [0]}}, config, balance_source=False)
    teacher, student = paired.dataset[0]
    assert source.dataset[0]['pixels'].shape[0] == 30
    assert teacher['pixels'].shape[0] == no_aug.dataset[0]['pixels'].shape[0] == 60
    assert student['pixels'].shape[0] == expected_dates
