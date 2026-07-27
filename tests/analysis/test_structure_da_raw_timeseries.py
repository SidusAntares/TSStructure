import datetime as dt

import numpy as np

from analysis.structure_da import raw_timeseries as raw
from analysis.structure_da.raw_timeseries import (
    ParcelCurve,
    RawAggregate,
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


def test_parcel_ndvi_uses_normalized_red_and_nir_parcel_means():
    pixels = np.zeros((2, 10, 3), dtype=np.float32)
    pixels[:, 2, :] = 10000
    pixels[:, 3, :] = 30000

    ndvi = raw.compute_parcel_ndvi(pixels)
    red = 10000 / 65535
    nir = 30000 / 65535

    assert np.allclose(ndvi, (nir - red) / (nir + red + 1e-8))


def test_ndvi_aggregation_keeps_parcels_equally_weighted():
    dates = (dt.date(2017, 1, 1), dt.date(2017, 1, 11))
    small = np.zeros((2, 10, 1), dtype=np.float32)
    small[:, 2, :] = 10000
    small[:, 3, :] = 10000
    large = np.zeros((2, 10, 9), dtype=np.float32)
    large[:, 2, :] = 10000
    large[:, 3, :] = 20000

    aggregate = aggregate_parcel_curves([
        ParcelCurve("DK1", "corn", dates, raw.compute_parcel_ndvi(small)),
        ParcelCurve("DK1", "corn", dates, raw.compute_parcel_ndvi(large)),
    ])["DK1", "corn"]

    assert np.allclose(aggregate.mean, 1 / 6)
    assert aggregate.n_parcels == 2


def test_day_of_year_is_derived_from_real_dates():
    dates = parse_acquisition_dates([20170101, 20170301, 20171231])

    assert raw.day_of_years(dates) == (1, 60, 365)


def test_ndvi_tables_do_not_invent_missing_class_domain_rows():
    aggregate = RawAggregate(
        "AT1", "wheat", (dt.date(2017, 1, 5),),
        np.array([0.2]), np.array([0.0]), np.array([0.0]), 3,
    )

    long_table, support, _ = raw.build_ndvi_tables({("AT1", "wheat"): aggregate})

    assert {
        "class_name", "domain", "date", "day_of_year", "ndvi_mean",
        "ndvi_std", "ndvi_sem", "n_parcels", "red_band", "red_index",
        "nir_band", "nir_index",
    } <= set(long_table.columns)
    assert list(long_table["domain"]) == ["AT1"]
    assert list(support["domain"]) == ["AT1"]
    assert not (long_table["domain"] == "DK1").any()


def test_shape_normalized_ndvi_plot_is_created(tmp_path):
    aggregate = RawAggregate(
        "FR1", "corn",
        (dt.date(2017, 3, 1), dt.date(2017, 4, 1), dt.date(2017, 5, 1)),
        np.array([0.1, 0.5, 0.3]), np.array([0.01, 0.02, 0.01]),
        np.array([0.01, 0.02, 0.01]), 4,
    )
    output = tmp_path / "corn.png"

    raw.plot_ndvi_class_curves(
        "corn", {("FR1", "corn"): aggregate}, output, shape_normalized=True
    )

    assert output.is_file()


def test_ndvi_descriptors_report_peak_and_minimum_doy():
    aggregate = RawAggregate(
        "DK1", "barley",
        (dt.date(2017, 2, 1), dt.date(2017, 6, 10), dt.date(2017, 9, 1)),
        np.array([0.2, 0.8, 0.1]), np.zeros(3), np.zeros(3), 5,
    )

    _, _, descriptors = raw.build_ndvi_tables({("DK1", "barley"): aggregate})
    row = descriptors.iloc[0]

    assert row["ndvi_peak_doy"] == 161
    assert row["ndvi_min_doy"] == 244


def test_repository_band_mapping_identifies_red_and_nir():
    assert raw.NDVI_RED_BAND == "B04"
    assert raw.NDVI_RED_INDEX == 2
    assert raw.NDVI_NIR_BAND == "B08"
    assert raw.NDVI_NIR_INDEX == 3
