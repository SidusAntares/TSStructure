import numpy as np
import pytest
import torch

from utils import guided_segment_refinement as guided
from utils.source_segment_refinement import SETTINGS


def test_match_spans_old_segments_and_only_missing_outer_cut_is_added():
    dates = np.arange(15)*5
    shape = np.zeros((15, 1))
    shape[3:12, 0] = np.arange(9)+1
    prototype = dict(id=2, frequency=.4, bounds=[[15]*3, [55]*3], shape=shape[3:12])
    competitor = dict(prototype, id=4, frequency=.3)
    matches = guided.match_shapelets(dates, shape, [competitor, prototype], 3, SETTINGS)
    assert matches == [guided.ShapeletMatch(2, 3, 12, 0.)]
    cuts = np.array([12.5, 27.5, 42.5])
    assert guided.split_matches(dates, cuts, matches, 3, SETTINGS) == [57.5]


@pytest.mark.parametrize('protected,triple_cost,expected,states', [
    ([12.5, 57.5], .10, ([27.5, 42.5], 'guided'), [4]),
    ([12.5, 27.5], .10, ([42.5], 'ordinary'), [3]),
    ([12.5, 57.5], .20, ([27.5], 'ordinary'), [4, 3]),
])
def test_guided_merge_priority_cost_and_exact_protection(monkeypatch, protected, triple_cost, expected, states):
    dates = torch.arange(15)*5
    cuts = np.array([12.5, 27.5, 42.5, 57.5])
    gates = np.array([False, True, True, True, False])
    called = []

    def costs(model, features, positions, y, masks, gamma, states, coverage_cost):
        called.append(states)
        assert masks.sum(1).tolist() == [9] + [6]*(states-1)
        assert len({tuple(mask.tolist()) for mask in masks}) == states
        return np.array([[triple_cost if states == 4 else .10] + [.10]*(states-1)])

    monkeypatch.setattr(guided, 'merge_costs', costs)
    monkeypatch.setattr(guided, 'dtw', lambda *args: pytest.fail('Merge must reuse cached matches'))
    result = guided.guided_merge(None, torch.zeros(15, 1), dates, 0, cuts, gates, protected,
                                 [guided.ShapeletMatch(2, 3, 12, .2)], 1, SETTINGS)
    assert result == expected
    assert called == states


def test_guided_merge_rejects_internal_observation_gaps(monkeypatch):
    dates = torch.tensor([0, 5, 10, 40, 45, 50, 55, 60, 65])
    monkeypatch.setattr(guided, 'merge_costs', lambda *args, **kwargs: pytest.fail('Gap should reject candidate'))
    result = guided.guided_merge(None, torch.zeros(9, 1), dates, 0, np.array([47.5]),
                                 np.array([True, True]), (), (), 1, SETTINGS)
    assert result == ([], None)
