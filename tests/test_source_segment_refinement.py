import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    assert '--mode refined_plots' in (tmp_path/'logs/refine_visualization/analysis.log').read_text()
