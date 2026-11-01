import json
from pathlib import Path

import numpy as np
import pytest

from analysis import segment_structure_visualization as vis


def test_ifft_is_display_only_preserves_input():
    dates = np.array([3., 8., 16., 18., 29.])
    values = np.array([1., -2., 3., 2., 1.])
    original_dates, original_values = dates.copy(), values.copy()
    grid, curve = vis.ifft_display(dates, values)
    assert grid.shape == curve.shape == (27,)
    assert np.isfinite(curve).all()
    np.testing.assert_array_equal(dates, original_dates)
    np.testing.assert_array_equal(values, original_values)
    assert grid[0] == dates[0] and grid[-1] == dates[-1]
    assert 'imputation_training' not in Path(vis.__file__).read_text()


def _jobs(root):
    torch = pytest.importorskip('torch')
    paths = []
    for variant in vis.VARIANTS:
        for task in vis.TASKS:
            leaf = root / variant / task / 'uda'
            (leaf / 'fold_0').mkdir(parents=True)
            path = leaf / 'fold_0' / 'model.pt'
            torch.save(dict(role='student', completed_epochs=20, variant=variant,
                            source_to_target_shift=7, state_dict={}, selector_state_dict={}), path)
            (leaf / 'resolved_config.json').write_text(json.dumps(dict(source='source', target='target', classes=['wheat'])))
            (leaf / 'split_indices.json').write_text(json.dumps({
                d: dict(train=[1], val=[2], test=[3]) for d in ('source', 'target')}))
            paths.append(path)
    return paths


def test_all_eight_required_before_any_output(tmp_path):
    root, output = tmp_path / 'experiment', tmp_path / 'experiment_visualization'
    paths = _jobs(root)
    before = {str(p): p.read_bytes() for p in paths}
    paths[-1].unlink()
    with pytest.raises(ValueError, match='All jobs must finish'):
        vis.main(['--experiment_root', str(root), '--data_root', 'unused', '--output_dir', str(output),
                  '--device', 'cpu', '--tasks', vis.TASKS[0], '--variants', 'inshape'])
    assert not output.exists()
    for path in paths[:-1]:
        assert path.read_bytes() == before[str(path)]


def test_preflight_rejects_teacher_and_incomplete(tmp_path):
    torch = pytest.importorskip('torch')
    paths = _jobs(tmp_path)
    assert len(vis.validate_jobs(tmp_path)) == 8
    packet = torch.load(paths[0], weights_only=False)
    for changes in (dict(role='teacher'), dict(completed_epochs=19)):
        torch.save(dict(packet, **changes), paths[0])
        with pytest.raises(ValueError, match='final Student'):
            vis.validate_jobs(tmp_path)


def test_all_stage_evaluation_split_preferred_over_raw_uda_holdout(tmp_path):
    paths = _jobs(tmp_path)
    leaf = paths[0].parent.parent
    raw = {d: dict(train=[1], val=[2], test=[3, 4]) for d in ('source', 'target')}
    evaluation = {d: dict(train=[1], val=[2], test=[3] if d == 'source' else [3, 4])
                  for d in ('source', 'target')}
    (leaf / 'split_indices.json').write_text(json.dumps(raw))
    strict_path = leaf / 'evaluation_split_indices.json'
    strict_path.write_text(json.dumps(evaluation))
    jobs = vis.validate_jobs(tmp_path)
    assert len(jobs) == 8
    job = jobs[0]
    assert vis.dataset_indices(job, 'source', 'test') == {3}
    assert 4 not in vis.dataset_indices(job, 'source', 'test')
    assert vis.dataset_indices(job, 'source', 'train') == {1}
    assert vis.dataset_indices(job, 'target', 'test') == {3, 4}
    assert job['evaluation_split_provenance'] == str(strict_path)
    # Older checkpoints retain their saved UDA split as the explicit fallback.
    assert vis.dataset_indices(jobs[1], 'source', 'test') == {3}
    assert jobs[1]['evaluation_split_provenance'].endswith('split_indices.json')


def test_shared_projection_and_sixteen_panel_pair(tmp_path):
    pytest.importorskip('matplotlib')
    features = [np.array([[0., 0.], [1., 2.], [2., 4.]])]
    mean, component = vis.fit_projection(features)
    curve = ((features[0] - mean) @ component).tolist()
    source = dict(dates=[0, 4, 9], curve=curve, segments=[0, 0, 1],
                  gate=[[1, 0]] * 16, probabilities=[[.9, .1]]*16, parcel_id=1)
    target = dict(dates=[1, 5, 10], curve=(np.asarray(curve)+2).tolist(),
                  segments=[0, 1, 1], gate=[[0, 1]]*16, probabilities=[[.1, .9]]*16, parcel_id=2)
    assert vis.plot_pair(source, target, tmp_path / 'pair', 'Wheat') == 16
    assert (tmp_path / 'pair.png').stat().st_size > 1000
    packet = json.loads((tmp_path / 'pair.json').read_text())
    assert packet['heads'] == 16 and packet['display_only']
    assert packet['source'] == source and packet['target'] == target
    assert packet['y_limits'][0] < min(curve)
    assert packet['y_limits'][1] > max(target['curve'])


def test_output_cannot_overwrite_training(tmp_path):
    with pytest.raises(ValueError, match='separate'):
        vis.main(['--experiment_root', str(tmp_path), '--data_root', 'unused',
                  '--output_dir', str(tmp_path / 'plots')])


def test_analysis_heldout_pairs_missing_class_and_readonly_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('matplotlib')
    from types import SimpleNamespace
    from utils import segment_structure
    class Dataset:
        metadata = {'start_date': 20170101}
        def __init__(self, parcel):
            self.parcel = parcel
        def __len__(self):
            return 1
        def get_parcel_indices(self):
            return np.array([self.parcel])
        def get_labels(self):
            return np.array([0])
        def __getitem__(self, index):
            assert index == 0
            return dict(pixels=np.ones((3, 2, 2), np.float32), valid_pixels=np.ones((3, 2), np.float32),
                        positions=np.array([0, 4, 9]), extra=np.zeros(4, np.float32),
                        parcel_index=self.parcel, label=0)
    spatial_calls = []
    class Model:
        def spatial_encoder(self, pixels, mask, extra):
            spatial_calls.append(1)
            return torch.tensor([[[0., 0.], [1., 2.], [2., 4.]]])
    shifts = []
    def forward(model, selector, pixels, mask, positions, extra, **kwargs):
        assert not torch.is_grad_enabled()
        shifts.append(positions[0].tolist())
        return None, torch.tensor([[2., 1.]]), dict(
            features=torch.tensor([[[0., 0.], [1., 2.], [2., 4.]]]),
            segments=torch.tensor([[0, 0, 1]]), gate=torch.ones(1, 16, 2),
            probabilities=torch.ones(1, 16, 2)*.7, point_mask=torch.ones(1, 16, 3),
            attention=torch.ones(1, 16, 3)/3)
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'original trained checkpoint')
    job = dict(variant='attention', task='AT1_DK1', checkpoint=checkpoint,
               config=dict(classes=['wheat', 'missing'], source='source', target='target'))
    monkeypatch.setattr(vis, 'load_job', lambda job, args: (Model(), None, dict(source_to_target_shift=7),
                                                          Dataset(1), Dataset(3), Dataset(13)))
    monkeypatch.setattr(segment_structure, 'structure_forward', forward)
    args = SimpleNamespace(output_dir=str(tmp_path / 'visualization'), seed=1, device='cpu',
                           pca_parcels=256, pairs_per_class=3, keep_fraction=.2, num_workers=8)
    manifest = vis.analyze_job(job, args)
    assert len(spatial_calls) == 1
    assert shifts == [[7, 11, 16], [0, 4, 9]]
    assert manifest['source_train_pca_parcel_ids'] == [1]
    assert manifest['class_coverage']['wheat']['pairs'][0]['source_parcel_id'] == 3
    assert manifest['class_coverage']['wheat']['pairs'][0]['target_parcel_id'] == 13
    assert manifest['class_coverage']['missing']['status'] == 'N/A'
    assert checkpoint.read_bytes() == b'original trained checkpoint'
