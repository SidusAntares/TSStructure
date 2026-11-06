from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analysis import segment_sampling_stability as stability
from models.segment_selector import SegmentSelector
from test_segment_structure import _model
from utils import segment_structure


def test_boundary_matching_cannot_match_a_cut_twice():
    result = stability.boundary_agreement(np.array([4, 5, 20, 25]), np.array([5, 18]), 0, 21, 2)
    assert result['matched_cuts'] == 2
    assert result['full_cuts_in_span'] == 3
    assert result['boundary_f1'] == pytest.approx(.8)


def test_kept_interval_overlap_separates_endpoints_from_selection():
    full = np.array([[0., 4.], [6., 10.]])
    thinned = np.array([[2., 4.], [6., 8.]])
    assert stability.interval_iou(full, thinned, (2., 8.)) == 1.
    assert stability.interval_iou(full, thinned) == .5


def test_paired_sampling_keeps_features_and_dates_identical_across_cut_conditions(monkeypatch):
    torch.manual_seed(8)
    model = _model().eval()
    selector = SegmentSelector(feature_dim=8, width=16, heads=4, protocol='inshape_drop_proxy_detached_v1').eval()
    dates = np.arange(12)*5
    sample = dict(pixels=np.random.default_rng(1).normal(size=(12, 3, 5)),
                  valid_pixels=np.ones((12, 5)), positions=dates, extra=np.zeros(4),
                  parcel_index=42, label=1)
    class Dataset:
        def get_parcel_indices(self):
            return np.array([42])

        def __getitem__(self, index):
            return sample

    calls = []
    original = segment_structure._classify_retained_observations
    def classify(model, selector, features, positions, segments, deterministic):
        calls.append((features, positions, segments))
        return original(model, selector, features, positions, segments, deterministic)
    monkeypatch.setattr(segment_structure, '_classify_retained_observations', classify)
    job = dict(variant='inshape', config=dict(pelt_penalty=.0, pelt_min_size=2, pelt_penalty_mode='absolute'))
    args = SimpleNamespace(seed=1, device='cpu', dates=6, repeats=4, cut_tolerance=7.)
    with torch.no_grad():
        rows = stability.parcel_trials(model, selector, job, Dataset(), 0, 3, args)
    assert len(rows) == 4 and len(calls) == 2
    features, positions, labels = calls[1]
    torch.testing.assert_close(features[:4], features[4:])
    torch.testing.assert_close(positions[:4], positions[4:])
    for i, row in enumerate(rows):
        assert len(set(row['sampled_positions'])) == 6
        np.testing.assert_array_equal(positions[i].numpy(), np.asarray(row['sampled_positions'])+3)
        expected = stability.transfer_segments(row['sampled_positions'], row['full_boundaries'])
        np.testing.assert_array_equal(labels[4+i].numpy(), expected)
    assert stability.summarize(rows, 'ALL')['parcels'] == 1
