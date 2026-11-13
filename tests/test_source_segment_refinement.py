import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.segment_selector import SegmentSelector
from models.stclassifier import PseLTae
from test_segment_launcher import bash_bin, setup_run
from test_segment_training import config
from utils import source_segment_refinement as refinement
from utils.segment_training import train_source_structure_epochs


def test_calendar_labels_skip_empty_segments_and_preserve_padding():
    dates = torch.tensor([[1, 8, 16, 99], [1, 3, 17, 19]])
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    cuts = [np.array([4, 6, 12]), np.array([5, 10])]
    assert refinement.calendar_batch(dates, valid, cuts).tolist() == [[0, 1, 2, -1], [0, 0, 1, 1]]


def test_split_requires_local_content_and_does_not_duplicate_existing_cuts():
    dates = np.arange(0, 120, 10)
    shape = np.array([0, 0, 0, 0, 3, 4, 3, 0, 0, 0, 0, 0])[:, None]
    proto = dict(id=0, frequency=.8, region=0, bounds=[[40]*3, [60]*3], shape=shape[4:7])
    added = refinement.split_cuts(dates, shape, np.array([]), [proto], min_size=3)
    assert added == [35, 65]
    assert not refinement.split_cuts(dates, shape, np.array(added), [proto], min_size=3)
    assert not refinement.split_cuts(dates, -shape, np.array([]), [proto], min_size=3)


def test_merge_preserves_independent_choices_when_their_loss_is_lower(monkeypatch):
    dates = torch.arange(9)*7
    features = torch.zeros(9, 1)
    bank = dict(mean=np.zeros(1), basis=np.eye(1), scale=np.ones(1), classes={})
    cuts = np.array([17.5, 38.5])
    gates, probs = np.array([True, True, False]), np.array([.8, .9, .2])
    for joint_cost, expected in ((.30, []), (.105, [17.5])):
        def evaluate(model, x, t, y, masks, gamma):
            assert masks.sum(1).tolist() == [3, 3, 6]
            return np.array([[.10, .20, joint_cost]])
        monkeypatch.setattr(refinement, 'merge_costs', evaluate)
        _, added, removed = refinement.refine_parcel(
            None, features, dates, 0, cuts, gates, probs, bank, min_size=3, gamma=1)
        assert added == [] and removed == expected


class FixedParcels:
    def __init__(self):
        generator = torch.Generator().manual_seed(2)
        self.pixels = torch.rand(6, 12, 10, 8, generator=generator)

    def __len__(self):
        return 6

    def get_parcel_indices(self):
        return np.arange(100, 106)

    def get_labels(self):
        return np.array([0, 1, 0, 1, 0, 1])

    def __getitem__(self, index):
        return dict(pixels=self.pixels[index], valid_pixels=torch.ones(12, 8),
                    positions=torch.arange(12)*7, extra=torch.zeros(4),
                    parcel_index=100+index, label=index % 2)


def test_refinement_uses_one_reference_pelt_and_freezes_source_encoders(tmp_path, monkeypatch):
    from torch.utils.data import default_collate
    import utils.segment_structure as structure

    dataset = FixedParcels()
    batch = default_collate([dataset[i] for i in range(len(dataset))])
    for key in ('pixels', 'valid_pixels', 'positions'):
        batch[key] = batch[key][:, ::2]
    cfg = config(tmp_path)
    cfg.__dict__.update(epochs=11, variant='inshape', output_dir=tmp_path, num_workers=0,
                        selector_protocol='inshape_drop_proxy_detached_v1')
    model = PseLTae(num_classes=2, with_extra=False)
    selector = SegmentSelector(width=16, heads=2, protocol=cfg.selector_protocol)
    before = {k: v.clone() for k, v in model.state_dict().items()
              if k.startswith(('spatial_encoder.', 'decoder.'))}
    monkeypatch.setattr(structure, 'segment_pelt', lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError('PELT rerun during minibatch training')))
    ref = refinement.SourceRefinement(dataset, cfg)
    train_source_structure_epochs(model, selector, [batch], cfg, refinement=ref)
    for k, value in before.items():
        torch.testing.assert_close(model.state_dict()[k], value, atol=0, rtol=0)
    assert (ref.folder / 'update_epoch10.pt').is_file()
    rows = [json.loads(line) for line in (ref.folder / 'parcels.jsonl').read_text().splitlines()]
    assert [row['parcel'] for row in rows] == list(dataset.get_parcel_indices())
    assert all(len(row['segments']) == len(row['final'])+1 for row in rows)
    assert torch.load(tmp_path/'model.pt', weights_only=False)['segment_refine']


@pytest.mark.parametrize('rounds,adapt', [(1, 5), (2, 5), (3, 3)])
def test_merge_recomputes_selector_on_split_cuts_after_training(tmp_path, monkeypatch, rounds, adapt):
    from torch.utils.data import default_collate

    dataset = FixedParcels()
    batch = default_collate([dataset[i] for i in range(len(dataset))])
    cfg = config(tmp_path)
    cfg.__dict__.update(epochs=10+2*adapt*rounds, variant='inshape', output_dir=tmp_path, num_workers=0,
                        refinement_rounds=rounds, refinement_adapt_epochs=adapt,
                        source_schedule_epochs=28 if adapt == 3 else 30,
                        selector_protocol='inshape_drop_proxy_detached_v1')
    model = PseLTae(num_classes=2, with_extra=False)
    selector = SegmentSelector(width=16, heads=2, protocol=cfg.selector_protocol)
    ref = refinement.SourceRefinement(dataset, cfg)
    monkeypatch.setattr(refinement, 'segment_pelt', lambda features, **_: torch.zeros(
        features.shape[:2], dtype=torch.long))
    phase_weights = []
    known_cuts = [17.5, 38.5, 59.5]

    def fit_bank(model, selector):
        phase_weights.append(selector.scorer[0].weight.detach().clone())
        prototype = dict(id=0, region=0, bounds=[[21]*3, [35]*3], shape=np.zeros((3, 8)))
        ref.bank = dict(mean=np.zeros(128), basis=np.eye(128)[:8], scale=np.full(8, 100.),
                        classes={0: [prototype], 1: [prototype]})

    def split(dates, shape, cuts, prototypes, min_size):
        return [known_cuts[len(cuts)]]

    def merge(model, features, dates, y, cuts, gates, probabilities, protected, gamma):
        assert cuts.tolist() == known_cuts[:len(phase_weights)]
        assert protected == [cuts[-1]]
        assert len(gates) == len(probabilities) == len(cuts)+1
        assert not torch.equal(selector.scorer[0].weight, phase_weights[-1])
        return []

    if adapt == 3:
        from utils import refinement_probe
        ref.probe = refinement_probe.RefinementProbe(dataset, cfg, ref.folder)
        monkeypatch.setattr(refinement_probe, 'segment_pelt', refinement.segment_pelt)
        monkeypatch.setattr(refinement_probe, 'split_cuts', split)
        monkeypatch.setattr(refinement_probe, 'merge_cuts', merge)
    monkeypatch.setattr(ref, 'fit_bank', fit_bank)
    monkeypatch.setattr(refinement, 'split_cuts', split)
    monkeypatch.setattr(refinement, 'merge_cuts', merge)
    history = train_source_structure_epochs(model, selector, [batch], cfg, refinement=ref)
    assert len(history) == cfg.epochs and len(phase_weights) == rounds+1
    assert [(row['epoch'], row['phase']) for row in ref.round_history] == [
        (10+adapt*i, 'merge' if i % 2 else 'split') for i in range(2*rounds)]
    packet = torch.load(tmp_path/'model.pt', weights_only=False)
    assert packet['refinement_rounds'] == rounds
    assert packet['refinement_round_history'] == ref.round_history
    if adapt == 3:
        rows = json.loads((ref.folder/'validation_probe.json').read_text())
        assert len(rows) == 25 and all(row['targets'] == 6 and row['target_pairs'] == 6 for row in rows)
        assert rows[0]['complete_fraction'] == 0
        assert rows[-1]['epoch'] == 28 and rows[-1]['epochs_since_edit'] == 3
        assert len({row['reference_sha256'] for row in rows}) == 1


@torch.no_grad()
def test_refined_source_trace_renders_png_and_reports_pair_metrics(tmp_path, capsys):
    from analysis.segment_cut_transfer import plot_class, refined_source_trace, report_class_edits

    model = PseLTae(num_classes=2, with_extra=False).eval()
    selector = SegmentSelector(width=16, heads=2, protocol='inshape_drop_proxy_detached_v1').eval()
    job = {'config': dict(pelt_penalty=.8, pelt_min_size=3, focal_loss_gamma=1.)}
    bank = dict(mean=np.zeros(128), basis=np.eye(128)[:8], scale=np.ones(8), classes={})
    args = SimpleNamespace(seed=1, device='cpu')
    dataset = FixedParcels()
    traces = [refined_source_trace(model, selector, job, dataset, i, args, (model, selector, bank))
              for i in (0, 2)]
    for trace in traces:
        trace['curve'] = trace['features'][:, 0]
    image = tmp_path/'00_corn.png'
    plot_class(traces, traces[0], image, 'Synthetic final Student', .2)
    assert image.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    report_class_edits('AT1_DK1', 'corn', traces)
    record = json.loads(capsys.readouterr().out.split('REFINED_PLOT_CLASS|')[1])
    assert record['initial']['prototype_match_coverage'] is None
    assert record['initial'] == record['refined']


def test_guided_round_refreshes_matches_and_keeps_split_protection_until_merge(tmp_path, monkeypatch):
    from utils import guided_segment_refinement as guided

    dataset = FixedParcels()
    cfg = config(tmp_path)
    cfg.__dict__.update(epochs=13, output_dir=tmp_path, shapelet_guided_merge=True,
                        refinement_adapt_epochs=3, refinement_epoch_plan=[1, 3, 3])
    ref = refinement.SourceRefinement(dataset, cfg)
    ref.cuts[100] = np.array([17.5, 59.5])
    dates = torch.arange(12)*7
    features = torch.zeros(12, 128)
    match = guided.ShapeletMatch(2, 3, 9, .1)
    next_match = guided.ShapeletMatch(3, 3, 6, .2)
    batches = [([100], [0], dates[None], features[None])]
    matching = []
    received = []
    module = torch.nn.Identity()

    def fit_bank(*args):
        ref.bank = dict(classes={}, mean=np.zeros(128), basis=np.eye(128)[:8], scale=np.ones(8))

    def match_shapes(*args):
        matching.append(len(matching)+1)
        return [match if matching[-1] == 1 else next_match]

    def merge(model, features, positions, y, cuts, gates, protected, matches, gamma, settings):
        received.append((protected, matches))
        return [], None

    monkeypatch.setattr(ref, 'fit_bank', fit_bank)
    monkeypatch.setattr(ref, 'views', lambda *args: iter(batches))
    monkeypatch.setattr(ref, 'selections', lambda *args: iter([
        (100, 0, dates, features, ref.cuts[100], np.ones(3, dtype=bool), np.ones(3))]))
    monkeypatch.setattr(guided, 'match_shapelets', match_shapes)
    monkeypatch.setattr(guided, 'guided_merge', merge)
    for epoch in (1, 4, 7, 10):
        ref.refine_round_step(module, module, epoch)
    assert matching == [1, 2]
    assert received == [((), [match]), ([38.5], [next_match])]
    assert [r['shapelet_matches'] for r in ref.round_history if r['phase'] == 'split'] == [1, 1]


def test_four_task_launcher_waits_for_final_students_before_plots(tmp_path):
    setup_run(tmp_path)
    script = 'run_segment_compare_8jobs_4gpu.sh'
    shutil.copyfile(Path(__file__).resolve().parents[1]/'closedset_scripts'/script,
                    tmp_path/'closedset_scripts'/script)
    subprocess.run([bash_bin(), 'closedset_scripts/'+script], cwd=tmp_path, check=True, timeout=40,
                   env=dict(os.environ, DATA_ROOT='data', PYTHON_BIN='/bin/echo', EXPERIMENT='refine',
                            SEGMENT_CONTEXT='0', SEGMENT_REFINE='1', RESUME='0'))
    logs = tmp_path/'logs/refine'
    for task in ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1'):
        assert '--stage finetune' in (logs/f'{task}_finetune_source.log').read_text()
        uda = (logs/f'{task}_finetune_uda.log').read_text()
        assert '--segment_refine' in uda and '--source_training_epochs 20' in uda
    events = (logs/'launcher.log').read_text()
    assert events.count('TASK_COMPLETE|') == 4
    assert events.rfind('TASK_COMPLETE|') < events.index('ANALYSIS_START|')
    analysis_command = (tmp_path/'logs/refine_visualization/analysis.log').read_text()
    assert shlex.split(analysis_command) == [
        '-u', '-m', 'analysis.segment_cut_transfer', '--mode', 'refined_plots',
        '--experiment_root', 'outputs/refine', '--data_root', 'data',
        '--output_dir', 'outputs/refine_visualization',
    ]


@pytest.mark.parametrize('adapt,budgets,guided', [(5, (20, 30), False), (3, (16, 22, 28), False),
                                                (3, (22, 28, 34), True)])
def test_refinement_comparison_launcher_runs_each_source_budget(tmp_path, adapt, budgets, guided):
    from train_segment_timematch import build_config

    setup_run(tmp_path)
    script = 'run_segment_compare_8jobs_4gpu.sh'
    shutil.copyfile(Path(__file__).resolve().parents[1]/'closedset_scripts'/script,
                    tmp_path/'closedset_scripts'/script)
    subprocess.run([bash_bin(), 'closedset_scripts/'+script], cwd=tmp_path, check=True, timeout=40,
                   env=dict(os.environ, DATA_ROOT='data', PYTHON_BIN='/bin/echo', EXPERIMENT='compare',
                            SEGMENT_CONTEXT='0', SEGMENT_REFINE='0', SEGMENT_REFINE_COMPARE='1', RESUME='0',
                            REFINEMENT_ADAPT_EPOCHS=str(adapt), SHAPELET_GUIDED_MERGE=str(int(guided))))
    logs = tmp_path/'logs/compare'
    for task in ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1'):
        routes = ('two_rounds', 'three_rounds', 'four_rounds') if guided else (
            'one_round', 'two_rounds', 'three_rounds')
        for route, epochs in zip(routes, budgets):
            command = shlex.split((logs/f'{task}_{route}_source.log').read_text())
            cfg = build_config(command[2:])
            assert cfg.stage == 'finetune' and cfg.epochs == epochs
            assert cfg.source_schedule_epochs == budgets[-1]
            assert cfg.shapelet_guided_merge is guided
            assert cfg.output_dir == f'outputs/compare/{route}/{task}/source'
    assert not list(logs.glob('*_uda.log'))
    events = (logs/'launcher.log').read_text()
    assert events.count('TASK_COMPLETE|') == 4*len(budgets)
    assert events.rfind('TASK_COMPLETE|') < events.index('ANALYSIS_START|')
    analysis = [
        '-u', '-m', 'analysis.segment_refinement_compare', '--experiment_root', 'outputs/compare',
        '--data_root', 'data', '--output_dir', 'outputs/compare_visualization', '--plot_task', 'AT1_DK1']
    if adapt == 3:
        analysis += ['--refinement_adapt_epochs', '3']
    if guided:
        analysis += ['--shapelet_guided_merge']
    assert shlex.split((tmp_path/'logs/compare_visualization/analysis.log').read_text()) == analysis


def test_refinement_comparison_keeps_the_same_plot_parcels_and_boundary_pairs(tmp_path):
    from analysis.segment_refinement_compare import plot_class, summarize_task

    def run(epoch, cuts):
        rows = {}
        for parcel, final in enumerate(cuts, 10):
            segments = [dict(selected=i == 0, cluster=1) for i in range(len(final)+1)]
            rows[parcel] = dict(label=0, initial=[20 if parcel == 10 else 40],
                                final=final, segments=segments)
        test = dict(completed_epochs=epoch, full=dict(accuracy=.8, macro_f1=.75),
                    keep=dict(accuracy=.7, macro_f1=.65))
        return dict(config=dict(classes=['corn'], source='AT1', original_source_checkpoint_sha256='same'),
                    rows=rows, bank={'0': [{}]}, history=[], test=test)

    one, two = run(20, [[22], [36]]), run(30, [[25], [31]])
    stats, plot_ids = summarize_task(one, two, seed=1, samples_per_class=2)
    assert set(plot_ids[0]) == {10, 11}
    assert stats['initial']['paired_cut_f1'] == 0
    assert stats['one_round']['mean_matched_error_days'] == 14
    assert stats['two_rounds']['mean_matched_error_days'] == 6
    assert stats['two_rounds']['heldout_native_pelt']['keep']['macro_f1'] == .65
    dates = np.arange(0, 81, 10)
    path = tmp_path / 'paired.png'
    plot_class('AT1_DK1', 'corn', plot_ids[0], one, two,
               {parcel: (dates, np.sin(dates/10)) for parcel in plot_ids[0]}, path)
    assert path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    third = run(28, [[26], [30]])
    stats, plot_ids = summarize_task(one, two, seed=1, samples_per_class=2, three=third)
    assert stats['three_rounds']['mean_matched_error_days'] == 4
    plot_class('AT1_DK1', 'corn', plot_ids[0], one, two,
               {parcel: (dates, np.sin(dates/10)) for parcel in plot_ids[0]}, path, three=third)
    assert path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'


def test_complete_structure_requires_one_well_bounded_selected_segment():
    from utils.refinement_probe import complete_window

    assert complete_window((5, 15), [(6, 15)])
    assert not complete_window((5, 15), [(5, 10), (10, 15)])
    assert not complete_window((5, 15), [(0, 30)])


def test_missing_history_uses_source_log_or_completed_checkpoint(tmp_path):
    from analysis.segment_refinement_compare import load_round_history

    fold = tmp_path/'fold_0'
    fold.mkdir()
    log = tmp_path/'source.log'
    history = [dict(epoch=10, phase='split', cuts_changed=4), dict(epoch=15, phase='merge', cuts_changed=1)]
    packet = dict(role='source_structure', completed_epochs=20, refinement_rounds=1)
    torch.save(packet, fold/'model.pt')
    assert load_round_history(tmp_path, 1, log) == (None, 'unavailable')
    log.write_text(''.join('SEGMENT_REFINE_PHASE|'+json.dumps(row)+'\n' for row in history), encoding='utf-8')
    assert load_round_history(tmp_path, 1, log) == (history, 'source_log')
    torch.save(dict(packet, refinement_round_history=history), fold/'model.pt')
    assert load_round_history(tmp_path, 1, None) == (history, 'checkpoint')


def test_missing_history_does_not_relabel_legacy_checkpoint(tmp_path):
    from analysis.segment_refinement_compare import load_round_history

    fold = tmp_path/'fold_0'
    fold.mkdir()
    torch.save(dict(role='source_structure', completed_epochs=20, segment_refine=True,
                    refinement_snapshot_sha256='legacy'), fold/'model.pt')
    with pytest.raises(ValueError, match='checkpoint does not confirm the requested 1 refinement'):
        load_round_history(tmp_path, 1, None)


def test_resume_reuses_valid_sources_and_backs_up_only_the_mismatched_run(tmp_path):
    setup_run(tmp_path)
    script = 'run_segment_compare_8jobs_4gpu.sh'
    shutil.copyfile(Path(__file__).resolve().parents[1]/'closedset_scripts'/script,
                    tmp_path/'closedset_scripts'/script)
    results = tmp_path/'outputs/compare'
    logs = tmp_path/'logs/compare'
    logs.mkdir(parents=True)
    bad = results/'one_round/AT1_DK1'
    domains = dict(AT1_DK1='austria_33UVP_2017', FR1_FR2='france_30TXT_2017',
                   FR2_DK1='france_31TCJ_2017', DK1_AT1='denmark_32VNH_2017')
    for task, domain in domains.items():
        for route in ('one_round', 'two_rounds'):
            source = results/route/task/'source'
            for path in (f'fold_0/test_{domain}.json', 'fold_0/model.pt', 'refinement/parcels.jsonl',
                         'refinement/prototypes.json'):
                file = source/path
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('original')
            (logs/f'{task}_{route}_source.log').write_text('original log')
    mock = tmp_path/'mock-python'
    mock.write_bytes(b'''#!/usr/bin/env bash
if [[ "$1" == -c ]]; then
    [[ "${3:-}" != outputs/compare/one_round/AT1_DK1/source ]] || exit 2
    exit 0
fi
echo "$*"
''')
    mock.chmod(0o755)
    subprocess.run([bash_bin(), 'closedset_scripts/'+script], cwd=tmp_path, check=True, timeout=40,
                   env=dict(os.environ, DATA_ROOT='data', PYTHON_BIN='./mock-python', EXPERIMENT='compare',
                            SEGMENT_CONTEXT='0', SEGMENT_REFINE='0', SEGMENT_REFINE_COMPARE='1',
                            RESUME='1', REPAIR_SOURCE='1'))
    events = (logs/'launcher.log').read_text()
    assert events.count('SOURCE_REUSE|') == 7
    assert events.count('SOURCE_BACKUP|') == 1
    backup, = bad.glob('source_before_repair.*')
    assert (backup/'source/fold_0/model.pt').read_text() == 'original'
    assert (backup/'source.log').read_text() == 'original log'
    assert '--refinement_rounds 1 --output_dir outputs/compare/one_round/AT1_DK1/source' in (
        logs/'AT1_DK1_one_round_source.log').read_text()
    assert [path.name for path in (tmp_path/'outputs').glob('compare*')] == ['compare']
