from __future__ import annotations

import datetime as dt
import pickle

import numpy as np
import pytest

import dataset
from dataset import BalancedBatchSampler


def _write_dataset(root, name: str, *, dates, start_date) -> None:
    folder = root / name
    (folder / "data").mkdir(parents=True)
    (folder / "meta").mkdir()
    metadata = {
        "dates": list(dates),
        "start_date": start_date,
        "parcels": [
            {
                "label": 10,
                "n_pixels": 1,
                "geometric_features": [0.0],
            }
        ],
    }
    with open(folder / "meta" / "metadata.pkl", "wb") as handle:
        pickle.dump(metadata, handle)


@pytest.mark.parametrize(
    "date,expected",
    [
        (dt.date(2017, 1, 1), 0.0),
        (20170228, 58.0),
        ("20200229", 58.5),
        (dt.datetime(2019, 3, 1), 59.0),
        (20171231, 364.0),
    ],
)
def test_canonical_day_position_uses_fixed_non_leap_calendar(date, expected) -> None:
    assert dataset.canonical_day_position(date) == expected


def test_canonical_day_position_matches_month_day_across_years() -> None:
    assert dataset.canonical_day_position(20160301) == dataset.canonical_day_position(
        20170301
    )
    assert dataset.canonical_day_position(20200228) == dataset.canonical_day_position(
        20190228
    )


def test_dataset_positions_ignore_domain_local_start_date(tmp_path, monkeypatch) -> None:
    dates = [20170101, 20170228, 20170301, 20171231]
    _write_dataset(
        tmp_path,
        "france/source/2017",
        dates=dates,
        start_date=20161215,
    )
    _write_dataset(
        tmp_path,
        "france/target/2017",
        dates=dates,
        start_date=20170101,
    )
    monkeypatch.setattr(
        dataset.label_utils,
        "get_code_to_class",
        lambda *_args, **_kwargs: {10: "crop"},
    )

    source = dataset.PixelSetData(
        str(tmp_path), "france/source/2017", ["crop"], closed_set=True
    )
    target = dataset.PixelSetData(
        str(tmp_path), "france/target/2017", ["crop"], closed_set=True
    )

    assert source.date_positions == target.date_positions == [0.0, 58.0, 59.0, 364.0]


@pytest.mark.parametrize(
    "dates",
    [
        [20170101, 20170101],
        [20170301, 20170228],
    ],
)
def test_dataset_rejects_nonincreasing_dates_with_dataset_identity(
    tmp_path, monkeypatch, dates
) -> None:
    name = "france/tile/2017"
    _write_dataset(tmp_path, name, dates=dates, start_date=20170101)
    monkeypatch.setattr(
        dataset.label_utils,
        "get_code_to_class",
        lambda *_args, **_kwargs: {10: "crop"},
    )

    with pytest.raises(ValueError) as error:
        dataset.PixelSetData(str(tmp_path), name, ["crop"], closed_set=True)

    message = str(error.value)
    assert name in message
    assert str(dates[0]) in message
    assert str(dates[1]) in message


def test_dataset_rejects_invalid_date_with_dataset_identity(
    tmp_path, monkeypatch
) -> None:
    name = "france/tile/2017"
    _write_dataset(
        tmp_path,
        name,
        dates=[20170228, 20170229],
        start_date=20170101,
    )
    monkeypatch.setattr(
        dataset.label_utils,
        "get_code_to_class",
        lambda *_args, **_kwargs: {10: "crop"},
    )

    with pytest.raises(ValueError, match=r"france/tile/2017.*20170229"):
        dataset.PixelSetData(str(tmp_path), name, ["crop"], closed_set=True)


def test_dataset_rejects_unsupported_time_coordinate_mode(
    tmp_path, monkeypatch
) -> None:
    name = "france/tile/2017"
    _write_dataset(tmp_path, name, dates=[20170101], start_date=20170101)
    monkeypatch.setattr(
        dataset.label_utils,
        "get_code_to_class",
        lambda *_args, **_kwargs: {10: "crop"},
    )

    with pytest.raises(ValueError, match="canonical_day_of_year"):
        dataset.PixelSetData(
            str(tmp_path),
            name,
            ["crop"],
            closed_set=True,
            time_coordinate_mode="sample_min_max",
        )


def test_balanced_batch_sampler_is_seed_reproducible() -> None:
    labels = np.repeat(np.arange(3), 6)
    first = list(BalancedBatchSampler(labels, 6, seed=11))
    second = list(BalancedBatchSampler(labels, 6, seed=11))
    different = list(BalancedBatchSampler(labels, 6, seed=12))
    assert first == second
    assert first != different
    assert all(set(labels[batch]) == {0, 1, 2} for batch in first)


def test_balanced_batch_sampler_preserves_requested_batch_size() -> None:
    labels = np.repeat(np.arange(3), 9)
    batches = list(BalancedBatchSampler(labels, 8, seed=2))
    assert all(len(batch) == 8 for batch in batches)
    assert all(np.bincount(labels[batch], minlength=3).max() <= 3 for batch in batches)
