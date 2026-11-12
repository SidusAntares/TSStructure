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


@pytest.mark.parametrize('rounds', [1, 2])
def test_merge_recomputes_selector_on_split_cuts_after_training(tmp_path, monkeypatch, rounds):
    from torch.utils.data import default_collate

    dataset = FixedParcels()
    batch = default_collate([dataset[i] for i in range(len(dataset))])
    cfg = config(tmp_path)
    cfg.__dict__.update(epochs=10+10*rounds, variant='inshape', output_dir=tmp_path, num_workers=0,
                        refinement_rounds=rounds, refinement_adapt_epochs=5, source_schedule_epochs=30,
                        selector_protocol='inshape_drop_proxy_detached_v1')
    model = PseLTae(num_classes=2, with_extra=False)
    selector = SegmentSelector(width=16, heads=2, protocol=cfg.selector_protocol)
    ref = refinement.SourceRefinement(dataset, cfg)
    monkeypatch.setattr(refinement, 'segment_pelt', lambda features, **_: torch.zeros(
        features.shape[:2], dtype=torch.long))
    phase_weights = []

    def fit_bank(model, selector):
        phase_weights.append(selector.scorer[0].weight.detach().clone())
        ref.bank = dict(mean=np.zeros(128), basis=np.eye(128)[:8], scale=np.ones(8), classes={})

    def split(dates, shape, cuts, prototypes, min_size):
        return [17.5] if not len(cuts) else [59.5]

    def merge(model, features, dates, y, cuts, gates, probabilities, protected, gamma):
        assert cuts.tolist() == ([17.5] if len(phase_weights) == 1 else [17.5, 59.5])
        assert protected == [cuts[-1]]
        assert len(gates) == len(probabilities) == len(cuts)+1
        assert not torch.equal(selector.scorer[0].weight, phase_weights[-1])
        return []

    monkeypatch.setattr(ref, 'fit_bank', fit_bank)
    monkeypatch.setattr(refinement, 'split_cuts', split)
    monkeypatch.setattr(refinement, 'merge_cuts', merge)
    history = train_source_structure_epochs(model, selector, [batch], cfg, refinement=ref)
    assert len(history) == cfg.epochs and len(phase_weights) == rounds+1
    assert [(row['epoch'], row['phase']) for row in ref.round_history] == [
        phase for i in range(rounds) for phase in ((10+10*i, 'split'), (15+10*i, 'merge'))]
    assert torch.load(tmp_path/'model.pt', weights_only=False)['refinement_rounds'] == rounds


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


def test_refinement_comparison_launcher_runs_two_source_budgets_for_each_task(tmp_path):
    from train_segment_timematch import build_config

    setup_run(tmp_path)
    script = 'run_segment_compare_8jobs_4gpu.sh'
    shutil.copyfile(Path(__file__).resolve().parents[1]/'closedset_scripts'/script,
                    tmp_path/'closedset_scripts'/script)
    subprocess.run([bash_bin(), 'closedset_scripts/'+script], cwd=tmp_path, check=True, timeout=40,
                   env=dict(os.environ, DATA_ROOT='data', PYTHON_BIN='/bin/echo', EXPERIMENT='compare',
                            SEGMENT_CONTEXT='0', SEGMENT_REFINE='0', SEGMENT_REFINE_COMPARE='1', RESUME='0'))
    logs = tmp_path/'logs/compare'
    for task in ('AT1_DK1', 'FR1_FR2', 'FR2_DK1', 'DK1_AT1'):
        for route, epochs in (('one_round', 20), ('two_rounds', 30)):
            command = shlex.split((logs/f'{task}_{route}_source.log').read_text())
            cfg = build_config(command[2:])
            assert cfg.stage == 'finetune' and cfg.epochs == epochs and cfg.source_schedule_epochs == 30
            assert cfg.output_dir == f'outputs/compare/{route}/{task}/source'
    assert not list(logs.glob('*_uda.log'))
    events = (logs/'launcher.log').read_text()
    assert events.count('TASK_COMPLETE|') == 8
    assert events.rfind('TASK_COMPLETE|') < events.index('ANALYSIS_START|')
    assert shlex.split((tmp_path/'logs/compare_visualization/analysis.log').read_text()) == [
        '-u', '-m', 'analysis.segment_refinement_compare', '--experiment_root', 'outputs/compare',
        '--data_root', 'data', '--output_dir', 'outputs/compare_visualization', '--plot_task', 'AT1_DK1']


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
