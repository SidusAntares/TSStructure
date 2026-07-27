import datetime as dt
import importlib

import numpy as np
import pandas as pd

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


def test_parcel_ndvi_averages_pixel_level_ndvi():
    pixels = np.zeros((1, 10, 2), dtype=np.float32)
    pixels[0, 2, :] = [10000, 30000]
    pixels[0, 3, :] = [30000, 40000]

    ndvi = raw.compute_parcel_ndvi(pixels)
    scaled_red = np.array([10000, 30000], dtype=np.float64) / 65535
    scaled_nir = np.array([30000, 40000], dtype=np.float64) / 65535
    pixel_ndvi = (scaled_nir - scaled_red) / (scaled_nir + scaled_red + 1e-8)
    expected = pixel_ndvi.mean()
    ratio_of_means = (
        (scaled_nir.mean() - scaled_red.mean())
        / (scaled_nir.mean() + scaled_red.mean() + 1e-8)
    )

    assert np.allclose(ndvi, [expected])
    assert not np.isclose(ndvi[0], ratio_of_means)


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


def _decomposition_diagnostics():
    return importlib.import_module("analysis.structure_da.decomposition_diagnostics")


def test_ndvi_decomposition_reuses_real_operator_and_reconstructs_irregular_series():
    diagnostics = _decomposition_diagnostics()
    from methods.structure_da.decomposition import SymmetricTimeKernelDecomposition

    rng = np.random.default_rng(11)
    doys = [10, 23, 41, 78, 121]
    frame = pd.DataFrame({
        "class_name": "corn", "domain": "FR1",
        "date": [f"2017-01-{index + 1:02d}" for index in range(len(doys))],
        "day_of_year": doys, "ndvi_mean": rng.normal(size=len(doys)),
        "n_parcels": 7,
    })

    components, reconstruction = diagnostics.decompose_ndvi_frame(frame)

    assert diagnostics.SymmetricTimeKernelDecomposition is SymmetricTimeKernelDecomposition
    assert reconstruction.iloc[0]["max_abs_reconstruction_error"] < 1e-5
    reconstructed = components.pivot(
        index="day_of_year", columns="component", values="value"
    ).sum(axis=1)
    assert np.allclose(reconstructed.to_numpy(), frame["ndvi_mean"], atol=1e-5)


def test_ndvi_decomposition_sorts_rows_and_does_not_invent_missing_domains():
    diagnostics = _decomposition_diagnostics()
    frame = pd.DataFrame([
        {"class_name": "wheat", "domain": "FR2", "date": "2017-03-01", "day_of_year": 60, "ndvi_mean": 0.3, "n_parcels": 4},
        {"class_name": "corn", "domain": "FR1", "date": "2017-02-01", "day_of_year": 32, "ndvi_mean": 0.2, "n_parcels": 3},
        {"class_name": "wheat", "domain": "FR1", "date": "2017-01-10", "day_of_year": 10, "ndvi_mean": 0.1, "n_parcels": 5},
        {"class_name": "wheat", "domain": "FR2", "date": "2017-01-20", "day_of_year": 20, "ndvi_mean": 0.2, "n_parcels": 4},
        {"class_name": "corn", "domain": "FR1", "date": "2017-01-05", "day_of_year": 5, "ndvi_mean": 0.1, "n_parcels": 3},
    ])

    components, reconstruction = diagnostics.decompose_ndvi_frame(frame)
    keys = list(components[["class_name", "domain", "day_of_year", "component"]].itertuples(index=False, name=None))
    component_order = {"trend": 0, "dynamics": 1, "residual": 2}

    assert keys == sorted(keys, key=lambda row: (row[0], row[1], row[2], component_order[row[3]]))
    assert set(zip(reconstruction["class_name"], reconstruction["domain"])) == {
        ("corn", "FR1"), ("wheat", "FR1"), ("wheat", "FR2")
    }


def test_ndvi_decomposition_writes_tables_and_missing_domain_figure(tmp_path):
    diagnostics = _decomposition_diagnostics()
    csv_path = tmp_path / "ndvi.csv"
    output_dir = tmp_path / "output"
    pd.DataFrame({
        "class_name": ["corn"] * 4,
        "domain": ["FR1", "FR1", "FR2", "FR2"],
        "date": ["2017-01-10", "2017-03-01", "2017-01-20", "2017-04-01"],
        "day_of_year": [10, 60, 20, 91],
        "ndvi_mean": [0.1, 0.5, 0.2, 0.6],
        "n_parcels": [3, 3, 4, 4],
    }).to_csv(csv_path, index=False)

    result = diagnostics.run_ndvi_decomposition(csv_path, output_dir)

    assert (output_dir / "tables/ndvi_decomposition/ndvi_decomposition_long.csv").is_file()
    assert (output_dir / "tables/ndvi_decomposition/reconstruction_check.csv").is_file()
    assert (output_dir / "figures/raw_timeseries/ndvi_decomposition/corn.png").is_file()
    assert set(result["components"]["domain"]) == {"FR1", "FR2"}


def test_cli_accepts_ndvi_decomposition_subcommand():
    from scripts.analyze_structure_da import build_parser

    args = build_parser().parse_args([
        "ndvi-decomposition", "--ndvi-csv", "ndvi.csv", "--output-dir", "out",
    ])

    assert args.command == "ndvi-decomposition"
