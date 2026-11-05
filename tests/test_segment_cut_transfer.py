from types import SimpleNamespace

import numpy as np
import pytest

from analysis import segment_cut_transfer as analysis


def test_source_cut_transfer_uses_dates_and_collapses_empty_intervals():
    cuts = analysis.candidate_edges([1, 3, 9, 15, 17], [0, 0, 1, 2, 2])[1:-1] + 2
    np.testing.assert_array_equal(cuts, [8, 14])
    np.testing.assert_array_equal(analysis.transfer_segments([2, 8, 13, 16], cuts), [0, 1, 1, 2])
    np.testing.assert_array_equal(analysis.transfer_segments([2, 16, 21], cuts), [0, 1, 1])


def test_class_figure_contains_cross_domain_and_source_rows_without_sidecars(tmp_path):
    pytest.importorskip('matplotlib')
    source = dict(dates=np.array([0, 5, 15, 20]), curve=np.array([1., 4., 0., 2.]),
                  segments=np.array([0, 0, 1, 1]), gate=np.array([True, False]), parcel=101)
    other = dict(source, gate=np.array([False, True]), parcel=102)
    target = dict(source, dates=source['dates'] + 1, parcel=201)
    assert analysis.retained_overlap(source, other) == 0.
    assert analysis.retained_overlap(source, source) == 1.
    path = tmp_path/'wheat.png'
    analysis.plot_class([source, other], target, path, 'Same-class wheat', .2)
    assert list(tmp_path.iterdir()) == [path]
    assert path.read_bytes().startswith(b'\x89PNG')


def test_cut_diagnostic_reuses_target_features_and_pairs_only_supported_classes(monkeypatch, capsys):
    torch = pytest.importorskip('torch')
    from sklearn.metrics import f1_score
    import dataset
    from utils import segment_structure

    source = SimpleNamespace(get_labels=lambda: np.array([0]))
    target = SimpleNamespace(get_labels=lambda: np.array([0, 1]))
    model, selector = object(), object()
    monkeypatch.setattr(analysis, 'load_job', lambda *a, **kw:
                        (model, selector, {'source_to_target_shift': 2}, None, source, target))
    monkeypatch.setattr(analysis, 'collect_observations', lambda *a:
                        dict(dates=np.array([0, 4, 8, 12]), segments=np.array([0, 0, 1, 1]), parcel=101))
    sample = dict(pixels=torch.ones(2, 4, 3, 2), valid_pixels=torch.ones(2, 4, 2),
                  positions=torch.tensor([[2, 6, 9, 14], [2, 6, 9, 14]]), label=torch.tensor([0, 1]))
    monkeypatch.setattr(dataset, 'create_evaluation_loaders', lambda *a: (None, [sample]))
    features = torch.arange(24).reshape(2, 4, 3).float()
    native = dict(segments=torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]]),
                  features=features, candidate_count=torch.tensor([2, 2]))
    monkeypatch.setattr(segment_structure, 'structure_forward', lambda *a, **kw:
                        (None, torch.tensor([[0., 1.], [0., 1.]]), native))
    def classify(m, s, x, dates, segments, deterministic):
        assert m is model and s is selector and deterministic
        torch.testing.assert_close(x, features[:1])
        torch.testing.assert_close(dates, sample['positions'][:1])
        torch.testing.assert_close(segments, torch.tensor([[0, 0, 1, 1]]))
        return torch.tensor([[1., 0.]]), {'candidate_count': torch.tensor([2])}
    monkeypatch.setattr(segment_structure, '_classify_retained_observations', classify)
    config = dict(classes=['wheat', 'oat'], target='target', seed=1, sample_pixels_val=True,
                  pelt_penalty=.8, pelt_min_size=3, pelt_penalty_mode='absolute')
    saved = dict(accuracy=.5, macro_f1=f1_score([0, 1], [1, 1], average='macro'))
    job = dict(config=config, variant='inshape', evaluation_splits={}, test_report={'keep': saved})
    args = SimpleNamespace(data_root='unused', device='cpu', num_workers=0, seed=1)
    analysis.target_cut_diagnostic(job, args)
    log = capsys.readouterr().out
    assert 'target_axis_cuts=[8.0]' in log
    assert 'PAIRED_RESULT n=1/2 changed_cuts=1 native_acc=0.000000 source_cut_acc=1.000000' in log
