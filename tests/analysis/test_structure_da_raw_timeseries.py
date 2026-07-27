import datetime as dt

import numpy as np

from analysis.structure_da.raw_timeseries import (
    ParcelCurve,
    aggregate_parcel_curves,
    normalize_parcel_pixels,
    parse_acquisition_dates,
)


def test_pixel_mean_then_equal_parcel_mean_ignores_parcel_pixel_count():
    small = np.zeros((2, 1, 1), dtype=np.float32)
    large = np.full((2, 1, 9), 65535, dtype=np.float32)

    small_curve = normalize_parcel_pixels(small)
    large_curve = normalize_parcel_pixels(large)
    aggregate = aggregate_parcel_curves([
        ParcelCurve("DK1", "corn", (dt.date(2017, 1, 1), dt.date(2017, 1, 2)), small_curve),
        ParcelCurve("DK1", "corn", (dt.date(2017, 1, 1), dt.date(2017, 1, 2)), large_curve),
    ])["DK1", "corn"]

    assert np.allclose(aggregate.mean, 0.5)
    assert aggregate.n_parcels == 2


def test_dates_are_real_calendar_dates_and_missing_groups_are_absent():
    dates = parse_acquisition_dates([20170105, "20170317"])
    groups = aggregate_parcel_curves([
        ParcelCurve("AT1", "wheat", dates, np.ones((2, 1))),
    ])

    assert dates == (dt.date(2017, 1, 5), dt.date(2017, 3, 17))
    assert ("AT1", "wheat") in groups
    assert ("DK1", "wheat") not in groups
