import importlib.util
import itertools

import numpy as np
import pytest

from utils.pelt_diagnostic import make_cost, pelt_boundaries, summarize_records, sweep_sequence


def exhaustive(cost, n, penalty, min_size):
    candidates = []
    for mask in itertools.product((False, True), repeat=n - 1):
        bounds = [0] + [i + 1 for i, flag in enumerate(mask) if flag] + [n]
        if min(np.diff(bounds)) >= min_size:
            objective = sum(cost(a, b) for a, b in zip(bounds, bounds[1:])) + penalty * (len(bounds) - 2)
            candidates.append((objective, bounds))
    return min(candidates)[0]


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('penalty', [0., .1, 1.])
def test_pelt_matches_exhaustive(seed, penalty):
    x = np.random.default_rng(seed).normal(size=(10, 3))
    x[5:] += 3
    cost, _ = make_cost(x)
    bounds = pelt_boundaries(cost, len(x), penalty, min_size=2)
    objective = sum(cost(a, b) for a, b in zip(bounds, bounds[1:])) + penalty * (len(bounds) - 2)
    assert objective == pytest.approx(exhaustive(cost, len(x), penalty, 2))


def test_default_cost_matches_production_and_does_not_mutate():
    pytest.importorskip('torch')
    from utils.segment_structure import _pelt_boundaries, _rbf_cost
    x = np.random.default_rng(42).normal(size=(30, 5))
    x[:, 0] = 9
    original = x.copy()
    cost, stats = make_cost(x)
    expected = _rbf_cost(x)
    for a in range(30):
        for b in range(a + 1, 31):
            assert cost(a, b) == pytest.approx(expected(a, b), abs=1e-11)
    for penalty in [0., .1, 1., 3.]:
        assert pelt_boundaries(cost, 30, penalty) == _pelt_boundaries(x, penalty, 3)
    np.testing.assert_array_equal(x, original)
    assert stats['constant_feature_fraction'] == .2


def test_factorial_sweep_and_penalty_modes():
    x = np.r_[np.zeros((4, 2)), np.ones((8, 2))]
    rows = sweep_sequence(x, np.arange(12) * 5)
    assert len(rows) == 90
    selected = [r for r in rows if r['normalization'] == 'feature' and r['gamma_multiplier'] == 1 and r['penalty_coefficient'] == .3]
    assert {r['penalty_mode']: r['actual_penalty'] for r in selected} == pytest.approx({'absolute': .3, 'per_observation': 3.6})
    assert {r['penalty_mode']: r['segment_count'] for r in selected} == {'absolute': 2, 'per_observation': 1}
    assert all(r['boundaries'][0] == 0 and r['boundaries'][-1] == 12 for r in rows)
    split = next(r for r in selected if r['segment_count'] == 2)
    assert split['segment_lengths_observations'] == [4, 8]
    assert split['segment_lengths_days'] == [15., 35.]
    assert split['best_single_split_gain'] > 0


def test_gamma_changes_boundaries():
    x = np.r_[np.zeros((6, 1)), np.ones((6, 1))]
    low, _ = make_cost(x, gamma_multiplier=.01)
    high, _ = make_cost(x, gamma_multiplier=4)
    assert pelt_boundaries(low, 12, .5) == [0, 12]
    assert pelt_boundaries(high, 12, .5) == [0, 6, 12]


def test_normalizations_preserve_relative_channels_and_constants():
    x = np.array([[0., 0.], [10., 1.], [20., 0.], [30., 1.]])
    feature, _ = make_cost(x)
    global_cost, _ = make_cost(x, normalization='global')
    raw, _ = make_cost(x, normalization='none')
    assert feature(0, 4) != pytest.approx(global_cost(0, 4))
    assert global_cost(0, 4) == pytest.approx(raw(0, 4))
    constant, stats = make_cost(np.ones((4, 2)))
    assert constant(0, 4) == 0
    assert stats['constant_feature_fraction'] == 1
    assert np.isfinite(list(stats.values())).all()


def test_short_sequence_dates_and_validation():
    rows = sweep_sequence(np.ones((2, 3)), np.array(['2026-01-01', '2026-01-04'], dtype='datetime64[D]'))
    assert all(r['boundaries'] == [0, 2] and r['segment_lengths_days'] == [3.] for r in rows)
    for x in [np.ones((0, 2)), np.ones(3), np.array([[np.nan]])]:
        with pytest.raises(ValueError):
            make_cost(x)
    for dates in [[1], [2, 1], [1, np.nan]]:
        with pytest.raises(ValueError):
            sweep_sequence(np.ones((2, 1)), dates)
    with pytest.raises(ValueError):
        make_cost(np.ones((2, 1)), gamma_multiplier=0)


def test_summary_groups_conditions_without_labels():
    rows = sweep_sequence(np.r_[np.zeros((6, 1)), np.ones((6, 1))], np.arange(12), penalties=(.3,), normalizations=('feature',), gamma_multipliers=(1,), penalty_modes=('absolute',))
    a = dict(rows[0], task='source', view='full')
    b = dict(rows[0], task='source', view='full', segment_count=4, single_segment=False)
    summary = summarize_records([a, b])
    assert len(summary) == 1
    assert summary[0]['sequence_count'] == 2
    assert summary[0]['mean_K'] == 3
    assert summary[0]['median_K'] == 3
    assert summary[0]['single_segment_rate'] == 0
    assert summary[0]['p10_K'] == pytest.approx(2.2)
    assert summary[0]['p90_K'] == pytest.approx(3.8)


@pytest.mark.parametrize('min_size', [1, 3, 7])
def test_pelt_matches_unpruned_dp_on_longer_random_sequences(min_size):
    for seed in range(5):
        x = np.random.default_rng(seed).normal(size=(45, 4))
        x[15:30] += 2
        cost, _ = make_cost(x)
        for penalty in [0., .03, .5, 3.]:
            objective = np.full(46, np.inf)
            objective[0] = -penalty
            for end in range(min_size, 46):
                objective[end] = min(objective[start] + cost(start, end) + penalty for start in range(end - min_size + 1))
            bounds = pelt_boundaries(cost, 45, penalty, min_size)
            actual = sum(cost(a, b) for a, b in zip(bounds, bounds[1:])) + penalty * (len(bounds) - 2)
            assert actual == pytest.approx(objective[-1], abs=1e-10)


def test_module_has_no_training_imports():
    spec = importlib.util.find_spec('utils.pelt_diagnostic')
    source = open(spec.origin, encoding='utf-8').read()
    import ast
    imports = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all(not (getattr(n, 'module', '') or '').startswith(('torch', 'utils.segment_structure')) for n in imports)
