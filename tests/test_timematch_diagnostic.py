import importlib.util

import numpy as np
import pytest
import torch


def test_diagnostic_module_exists():
    assert importlib.util.find_spec('utils.timematch_diagnostic') is not None


def test_intervals_select_only_top_points_and_distinguish_mass_from_count():
    from utils.timematch_diagnostic import select_structure
    # Top points are 0,2,5,9. 75% of mass fits [0,2]; 75% of count needs [0,5].
    attention = np.array([.45, .005, .30, .005, .005, .13, .005, .005, .005, .085])
    dates = np.array([0, 1, 2, 3, 4, 5, 20, 30, 40, 100])
    mass = select_structure(attention, dates, 'mass75', top_fraction=.4)
    count = select_structure(attention, dates, 'count75', top_fraction=.4)
    top = select_structure(attention, dates, 'top30', top_fraction=.4)
    assert np.flatnonzero(mass).tolist() == [0, 2]
    assert np.flatnonzero(count).tolist() == [0, 2, 5]
    assert np.flatnonzero(top).tolist() == [0, 2, 5, 9]
    assert not mass[1] and not count[1]


def test_gate_and_mean_change_only_one_head_without_renormalization():
    from utils.timematch_diagnostic import counterfactual_embeddings
    attention = torch.tensor([[[.5, .3, .2], [.2, .3, .5]]])
    values = torch.arange(1., 7.).reshape(1, 2, 3, 1)
    mask = torch.tensor([[[True, False, False], [False, False, True]]])
    full = (attention[..., None] * values).sum(-2).flatten(1)
    for mode, g in [('mean', None), ('gate01', .1), ('gate05', .5)]:
        results = counterfactual_embeddings(attention, values, mask, mode)
        assert torch.allclose(results[:, 0], full)
        assert results[0, 1, 1] == full[0, 1]
        assert results[0, 2, 0] == full[0, 0]
        if g is not None:
            assert torch.allclose(results[0, 1, 0], full[0, 0] - (1-g)*.5*1.)
        else:
            assert torch.allclose(results[0, 1, 0], full[0, 0] - .5*(1.-2.5))


def test_fourier_display_preserves_smooth_signal_on_irregular_days():
    from utils.timematch_diagnostic import fourier_display
    dates = np.arange(0., 365., 11.) + np.sin(np.arange(34))*.5
    signal = np.cos(2*np.pi*dates/365)
    grid, curve, observed = fourier_display(dates, signal)
    assert len(grid) == len(curve) == 365
    assert np.max(np.abs(observed-signal)) < .02
    assert np.max(np.abs(curve-np.cos(2*np.pi*grid/365))) < .03


def test_final_student_loader_rejects_validation_best_or_wrong_epoch(tmp_path):
    from utils.timematch_diagnostic import load_final_student
    model = torch.nn.Linear(2, 2)
    path = tmp_path / 'checkpoint_epoch20.pt'
    torch.save({'state_dict': model.state_dict(), 'completed_epochs': 10,
                'role': 'student'}, path)
    with pytest.raises(ValueError, match='20'):
        load_final_student(path)
    torch.save({'state_dict': model.state_dict(), 'epoch': 19, 'best_f1': 1.}, path)
    with pytest.raises(ValueError):
        load_final_student(path)
    torch.save({'state_dict': model.state_dict(), 'completed_epochs': 20,
                'role': 'student'}, path)
    assert load_final_student(path)['completed_epochs'] == 20


def test_snapshot_preserves_training_mode_rng_buffers_and_parameters(tmp_path):
    from types import SimpleNamespace
    from utils.timematch_diagnostic import save_student_snapshot
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout()).train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    config = SimpleNamespace(source='src', target='dst', seed=1)
    path = save_student_snapshot(model, config, {}, tmp_path, 10, -8, 8)
    packet = torch.load(path, map_location='cpu', weights_only=False)
    assert packet['student_shifts'] == {'source': -8, 'target': 0}
    assert packet['completed_epochs'] == 10
    assert model.training
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_render_stage_has_six_structure_and_eighteen_grouped_bar_figures(tmp_path):
    from analysis.timematch_diagnostic import plot_stage
    from utils.timematch_diagnostic import CLASSES, STRUCTURES, select_structure
    dates = np.arange(30)*12
    attention = np.tile(np.linspace(1., 3., 30)/60., (16, 1))
    masks = np.array([[select_structure(head, dates, method) for head in attention]
                      for method in STRUCTURES])
    representatives = [dict(name=name, parcel=i, dates=dates, masks=masks,
                            grid=dates, curve=np.sin(dates/100+i),
                            observed=np.sin(dates/100+i), shift=0)
                       for i, name in enumerate(CLASSES)]
    results = dict(representatives=representatives, means=np.ones((3, 3, 3, 16)),
                   errors=np.zeros((3, 3, 3, 16)), counts=[100, 100, 100])
    structures, bars = plot_stage('AT1_DK1', 0, {'source': results, 'target': results}, tmp_path)
    assert len(structures) == 6 and len(bars) == 18
    assert all((tmp_path / path).is_file() for path in structures + bars)
    assert len((tmp_path / 'contributions.csv').read_text().splitlines()) == 2*3*3*3*16+1


def test_analyze_domain_matches_real_classifier_and_saves_per_sample_evidence(tmp_path):
    from analysis.timematch_diagnostic import analyze_domain
    from models.stclassifier import PseLTae
    from utils.timematch_diagnostic import CLASSES
    torch.manual_seed(4)
    model = PseLTae(num_classes=3, with_extra=False).eval()

    class SyntheticDataset:
        dataset_name = 'synthetic'
        samples = [(None, i, i//2, None) for i in range(6)]

        def __getitem__(self, index):
            rng = np.random.default_rng(index)
            return dict(pixels=rng.random((20, 10, 4)).astype(np.float32),
                        valid_pixels=np.ones((20, 4), dtype=np.float32),
                        positions=np.arange(20)*15, extra=np.zeros(4, dtype=np.float32),
                        label=index//2, parcel_index=index)

    selected = {name: [i*2, i*2+1] for i, name in enumerate(CLASSES)}
    with torch.no_grad():
        result = analyze_domain(model, SyntheticDataset(), selected,
                                dict(seed=1, classes=list(CLASSES)), -5,
                                np.zeros(128), np.eye(128)[0], torch.device('cpu'), tmp_path)
    assert result['counts'] == [2, 2, 2]
    assert np.isfinite(result['means']).all()
    for name in CLASSES:
        packet = np.load(tmp_path / f'{name}.npz')
        assert packet['contributions'].shape == (3, 3, 2, 16)
        assert packet['structures'].shape == (3, 2, 16, 20)
        assert np.all(packet['structures'][:2] <= packet['structures'][2:3])
        assert packet['applied_shift'] == -5


def test_fixed_selection_uses_100_or_all_and_is_reproducible():
    from analysis.timematch_diagnostic import _selection
    from utils.timematch_diagnostic import CLASSES

    class Dataset:
        dataset_name = 'synthetic'

        def get_labels(self):
            return np.repeat(np.arange(3), [120, 3, 101])

    selection = _selection(Dataset(), list(CLASSES), 1)
    assert [len(selection[name]) for name in CLASSES] == [100, 3, 100]
    assert selection == _selection(Dataset(), list(CLASSES), 1)


def test_missing_classifier_class_and_empty_test_pool_select_no_samples():
    from analysis.timematch_diagnostic import _selection

    class Dataset:
        dataset_name = 'synthetic'

        def get_labels(self):
            return np.array([0, 0])

    selected = _selection(Dataset(), ['spring_barley', 'spring_oat', 'other'], 1)
    assert len(selected['spring_barley']) == 2
    assert selected['spring_wheat'] == []
    assert selected['spring_oat'] == []


def test_na_diagnostics_do_not_invoke_model_or_fabricate_zero_contributions(tmp_path):
    from analysis.timematch_diagnostic import analyze_domain, plot_stage
    from utils.timematch_diagnostic import CLASSES

    class Dataset:
        dataset_name = 'synthetic'

    # All requested classes are unsupported: training can still be entirely valid.
    results = analyze_domain(None, Dataset(), {name: [] for name in CLASSES},
                             dict(seed=1, classes=['other1', 'other2']), 8,
                             None, None, torch.device('cpu'), tmp_path / 'data')
    assert results['counts'] == [0, 0, 0]
    assert np.isnan(results['means']).all()
    assert all(info['status'] == 'N/A' for info in results['availability'].values())
    assert all(info['reason'] == 'not_in_source_classifier' for info in results['availability'].values())
    packet = np.load(tmp_path / 'data/spring_wheat.npz')
    assert packet['status'] == 'N/A'
    assert packet['contributions'].shape == (3, 3, 0, 16)
    structures, bars = plot_stage('AT1_DK1', 0, {'source': results, 'target': results}, tmp_path / 'figures')
    assert len(structures) == 6 and len(bars) == 18
    import csv
    with (tmp_path / 'figures/contributions.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(row['mean'] == 'N/A' and row['count'] == '0' for row in rows)
    assert all(row['status'] == 'N/A' for row in rows)


def test_source_validation_allows_missing_requested_class_but_checks_weights(tmp_path, capsys):
    import json
    from analysis.timematch_diagnostic import validate_source
    from models.stclassifier import PseLTae
    root = tmp_path / 'source'
    (root / 'fold_0').mkdir(parents=True)
    config = dict(source='src', target='src', model='pseltae', seed=1, epochs=100,
                  closed_set=True, combine_spring_and_winter=False, input_dim=10,
                  with_extra=False, method=None)
    (root / 'train_config.json').write_text(json.dumps(config))
    (root / 'closed_set_protocol.json').write_text(json.dumps(dict(
        source='src', closed_set=True, classes=['spring_barley', 'spring_oat'],
    )))
    torch.save({'state_dict': PseLTae(num_classes=2, with_extra=False).state_dict()}, root / 'fold_0/model.pt')
    validate_source(root, 'src', 1)
    output = capsys.readouterr().out
    assert 'SOURCE_VALID|' in output and 'spring_wheat' in output and 'N/A' in output
    torch.save({'state_dict': {}}, root / 'fold_0/model.pt')
    with pytest.raises(RuntimeError):
        validate_source(root, 'src', 1)


def test_real_training_hooks_preserve_updates_and_export_epoch20_student(tmp_path, monkeypatch):
    """20 tiny real optimizer steps; fake only server data/GPU and child rendering."""
    from types import SimpleNamespace
    import timematch
    from utils.timematch_diagnostic import load_final_student

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(2, 3)
            self.temporal_encoder = SimpleNamespace(
                max_temporal_shift=100, positional_enc=SimpleNamespace(num_embeddings=565))

        def forward(self, pixels, valid, positions, extra):
            return self.classifier(pixels)

    torch.manual_seed(123)
    initial = Student().state_dict()
    pretrained = tmp_path / 'source/fold_0'
    pretrained.mkdir(parents=True)
    torch.save({'state_dict': initial}, pretrained / 'model.pt')
    sample = dict(pixels=torch.randn(4, 2), valid_pixels=torch.ones(4, 1),
                  positions=torch.tensor([[1], [2], [3], [4]]), extra=torch.zeros(4, 4),
                  label=torch.tensor([0, 1, 2, 0]))
    target = SimpleNamespace(dataset=SimpleNamespace(get_labels=lambda: np.array([0, 1, 2, 0])))
    monkeypatch.setattr(timematch, 'get_data_loaders', lambda *a: ([sample], target, [(sample, sample)]))
    monkeypatch.setattr(timematch, 'estimate_temporal_shift', lambda *a, **kw: 8)
    monkeypatch.setattr(timematch, 'get_pseudo_labels', lambda *a, **kw: torch.eye(3))
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    child_stages = []

    def child_run(command, **kwargs):
        path = command[command.index('--snapshot')+1]
        packet = torch.load(path, map_location='cpu', weights_only=False)
        child_stages.append(packet['completed_epochs'])
        assert packet['student_shifts'] == {'source': -8, 'target': 0}
        print('CHILD_DIAGNOSTIC_ONLY', file=kwargs['stdout'])

    monkeypatch.setattr('utils.timematch_diagnostic.subprocess.run', child_run)
    config = SimpleNamespace(
        weights=str(tmp_path / 'source'), balance_source=True, num_classes=3,
        use_focal_loss=False, steps_per_epoch=1, lr=.001, weight_decay=0.,
        epochs=20, max_temporal_shift=60, estimate_shift=True, shift_estimator='AM',
        shift_source=True, sample_size=1, progress_bar='off', pseudo_threshold=0.,
        domain_specific_bn=True, trade_off=2., ema_decay=.99, log_step=1,
        run_validation=False, output_student=True, model='pseltae', num_folds=1,
        diagnostic_task='AT1_DK1', diagnostic_log=str(tmp_path / 'diagnostic.log'),
        diagnostic_dir=None,
    )
    writer = SimpleNamespace(add_scalar=lambda *a: None)
    baseline, diagnostic = Student(), Student()
    timematch.train_timematch(baseline, config, writer, None, torch.device('cpu'),
                             tmp_path / 'baseline.pt', 0, {})
    config.diagnostic_dir = str(tmp_path / 'diagnostic')
    timematch.train_timematch(diagnostic, config, writer, None, torch.device('cpu'),
                             tmp_path / 'final.pt', 0, {})
    assert child_stages == [0, 10, 20]
    assert all(torch.equal(value, diagnostic.state_dict()[key])
               for key, value in baseline.state_dict().items())
    snapshot = load_final_student(tmp_path / 'diagnostic/checkpoints/checkpoint_epoch20.pt')
    exported = load_final_student(tmp_path / 'final.pt')
    assert all(torch.equal(value, exported['state_dict'][key])
               for key, value in snapshot['state_dict'].items())
    assert 'CHILD_DIAGNOSTIC_ONLY' in (tmp_path / 'diagnostic.log').read_text()


@pytest.mark.parametrize('missing_wheat', [False, True])
def test_three_stage_render_and_verification_rejects_wrong_final_export(tmp_path, monkeypatch, missing_wheat):
    from types import SimpleNamespace
    import analysis.timematch_diagnostic as diagnostic
    from models.stclassifier import PseLTae
    from utils.timematch_diagnostic import CLASSES, save_student_snapshot
    class_names = [name for name in CLASSES if not (missing_wheat and name == 'spring_wheat')]
    class_count = len(class_names)

    class Dataset:
        dataset_name = 'synthetic'
        samples = [(None, i, i//2, None) for i in range(class_count*2)]

        def get_labels(self):
            return np.repeat(np.arange(class_count), 2)

        def get_parcel_indices(self):
            return np.arange(class_count*2)

        def __len__(self):
            return class_count*2

        def __getitem__(self, index):
            rng = np.random.default_rng(index)
            return dict(pixels=rng.random((20, 10, 4)).astype(np.float32),
                        valid_pixels=np.ones((20, 4), dtype=np.float32),
                        positions=np.arange(20)*15, extra=np.zeros(4, dtype=np.float32),
                        label=index//2, parcel_index=index)

    monkeypatch.setattr(diagnostic, '_dataset', lambda *a: Dataset())
    config = SimpleNamespace(source='src', target='dst', seed=1, classes=class_names,
                             num_classes=class_count, input_dim=10, with_extra=False, device='cpu')
    model = PseLTae(num_classes=class_count, with_extra=False).eval()
    for stage in (0, 10, 20):
        snapshot = save_student_snapshot(model, config, {}, tmp_path / 'checkpoints', stage, -8, 8)
        diagnostic.render(snapshot, tmp_path, 'AT1_DK1')
    import json
    manifest = json.loads((tmp_path / 'epoch_20/manifest.json').read_text())
    wheat = manifest['class_availability']['target']['spring_wheat']
    assert wheat['status'] == ('N/A' if missing_wheat else 'available')
    assert wheat['count'] == (0 if missing_wheat else 2)
    if missing_wheat:
        packet = np.load(tmp_path / 'epoch_20/data/target/spring_wheat.npz')
        assert packet['status'] == 'N/A'
        assert packet['contributions'].shape == (3, 3, 0, 16)
    assert len(list(tmp_path.glob('epoch_*/source/*/structures.png'))) == 9
    assert len(list(tmp_path.glob('epoch_*/*/*/contributions_*.png'))) == 54
    training = tmp_path / 'training'
    training.mkdir()
    torch.save(dict(state_dict=model.state_dict(), completed_epochs=20, role='student'), training / 'model.pt')
    diagnostic.write_json(training / 'tested_checkpoint.json', {
        'completed_epochs': 20, 'sha256': diagnostic.sha256(snapshot),
    })
    diagnostic.verify_task(tmp_path, 'AT1_DK1', training)
    torch.save(dict(state_dict={}, epoch=9, best_f1=1.), training / 'model.pt')
    with pytest.raises(ValueError, match='20-epoch'):
        diagnostic.verify_task(tmp_path, 'AT1_DK1', training)


def test_training_entry_tests_epoch20_even_if_model_pt_contains_validation_best(tmp_path, monkeypatch):
    import json
    import sys
    import types
    from types import SimpleNamespace
    # Optional competitors/TensorBoard are unrelated to the model-selection test.
    if 'train' not in sys.modules:
        stub = types.ModuleType('torch.utils.tensorboard')
        stub.SummaryWriter = object
        monkeypatch.setitem(sys.modules, 'torch.utils.tensorboard', stub)
        for module_name, function_name in (
            ('competitors.dann.dann', 'train_dann'),
            ('competitors.jumbot.jumbot', 'train_jumbot'),
            ('competitors.mmd.train_mmd', 'train_mmd'),
            ('competitors.alda.train_alda', 'train_alda'),
        ):
            stub = types.ModuleType(module_name)
            setattr(stub, function_name, lambda *a: None)
            monkeypatch.setitem(sys.modules, module_name, stub)
    import train
    from utils.timematch_diagnostic import save_student_snapshot
    config = SimpleNamespace(
        seed=1, device='cpu', source='src', target='dst', num_folds=1,
        val_ratio=.1, test_ratio=.2, overall=False, closed_set=False,
        sample_pixels_val=True, model='pseltae', input_dim=2, num_classes=2,
        with_extra=False, output_dir=str(tmp_path / 'uda'), eval=False,
        tensorboard_log_dir=str(tmp_path / 'tensorboard'), method='timematch',
        diagnostic_dir=str(tmp_path / 'diagnostic'), classes=['a', 'b'], experiment_name='test',
    )
    (tmp_path / 'uda/fold_0').mkdir(parents=True)
    monkeypatch.setattr(train, 'prepare_data_protocol', lambda cfg: ({}, None))
    monkeypatch.setattr(train, 'create_train_val_test_folds', lambda *a: [{}])
    monkeypatch.setattr(train, 'create_evaluation_loaders', lambda *a: (None, None))
    monkeypatch.setattr(train, 'PseLTae', lambda **kw: torch.nn.Linear(2, 2))
    monkeypatch.setattr(train, 'SummaryWriter', lambda **kw: None)
    monkeypatch.setattr(train, 'save_results', lambda *a: None)
    monkeypatch.setattr(train, 'overall_performance', lambda *a: None)

    def finish_training(model, cfg, writer, loader, device, model_path, fold, splits):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(20.)
        save_student_snapshot(model, cfg, {}, tmp_path / 'diagnostic/checkpoints', 20, -8, 8)
        # An adversarial validation-best file must never control final testing.
        torch.save({'state_dict': {key: value*0-999 for key, value in model.state_dict().items()},
                    'epoch': 9, 'best_f1': 1.}, model_path)

    monkeypatch.setattr(train, 'train_timematch', finish_training)
    tested = []

    def evaluate(model, *args, **kwargs):
        assert all(torch.all(parameter == 20.) for parameter in model.parameters())
        tested.append(True)
        return dict(accuracy=1., macro_f1=1., classification_report='synthetic test')

    monkeypatch.setattr(train, 'evaluation', evaluate)
    train.main(config)
    assert tested == [True]
    packet = json.loads((tmp_path / 'uda/fold_0/tested_checkpoint.json').read_text())
    assert packet['completed_epochs'] == 20
    assert packet['path'].endswith('checkpoint_epoch20.pt')


def test_child_failure_goes_to_diagnostic_log_and_stops_training(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from utils.timematch_diagnostic import StudentDiagnostic
    config = SimpleNamespace(epochs=20, output_student=True, model='pseltae', num_folds=1,
                             diagnostic_dir=str(tmp_path), diagnostic_log=str(tmp_path / 'diagnostic.log'),
                             diagnostic_task='AT1_DK1')

    def fail(*args, **kwargs):
        print('specific diagnostic error', file=kwargs['stderr'])
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr('utils.timematch_diagnostic.subprocess.run', fail)
    with pytest.raises(RuntimeError, match='Diagnostic failed'):
        StudentDiagnostic(config, {}).capture(0, torch.nn.Linear(2, 2), 0, 0)
    assert 'specific diagnostic error' in (tmp_path / 'diagnostic.log').read_text()
