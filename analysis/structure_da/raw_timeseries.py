"""Read-only raw parcel time-series diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DOMAIN_DATASETS = {
    "DK1": "denmark/32VNH/2017",
    "FR2": "france/30TXT/2017",
    "FR1": "france/31TCJ/2017",
    "AT1": "austria/33UVP/2017",
}
DOMAIN_COLORS = {"DK1": "C0", "FR2": "C1", "FR1": "C2", "AT1": "C3"}

# Stored channel order follows preprocessing.read_s2_image(): 10 m bands first,
# followed by the resampled 20 m bands.
SENTINEL2_CHANNELS = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12")
NDVI_RED_BAND = "B04"
NDVI_RED_INDEX = SENTINEL2_CHANNELS.index(NDVI_RED_BAND)
NDVI_NIR_BAND = "B08"
NDVI_NIR_INDEX = SENTINEL2_CHANNELS.index(NDVI_NIR_BAND)


@dataclass(frozen=True)
class ParcelCurve:
    domain: str
    class_name: str
    dates: tuple[dt.date, ...]
    curve: np.ndarray


@dataclass(frozen=True)
class RawAggregate:
    domain: str
    class_name: str
    dates: tuple[dt.date, ...]
    mean: np.ndarray
    std: np.ndarray
    sem: np.ndarray
    n_parcels: int


def parse_acquisition_dates(values: Iterable[object]) -> tuple[dt.date, ...]:
    result = []
    for value in values:
        text = str(value)
        result.append(dt.datetime.strptime(text, "%Y%m%d").date())
    return tuple(result)


def normalize_parcel_pixels(pixels: np.ndarray) -> np.ndarray:
    """Return one parcel curve [T,C], using training-compatible scaling."""

    pixels = np.asarray(pixels)
    if pixels.ndim != 3:
        raise ValueError("pixels must have shape [T,C,S]")
    return np.clip(pixels, 0, 65535).astype(np.float64).mean(axis=-1) / 65535.0


def compute_parcel_ndvi(pixels: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute pixel-level NDVI, then average equally over a parcel's pixels."""

    pixels = np.asarray(pixels)
    if pixels.ndim != 3:
        raise ValueError("pixels must have shape [T,C,S]")
    if pixels.shape[1] <= max(NDVI_RED_INDEX, NDVI_NIR_INDEX):
        raise ValueError("pixels do not contain the repository-defined RED/NIR channels")
    scaled = np.clip(pixels, 0, 65535).astype(np.float64) / 65535.0
    red = scaled[:, NDVI_RED_INDEX, :]
    nir = scaled[:, NDVI_NIR_INDEX, :]
    pixel_ndvi = (nir - red) / (nir + red + eps)
    return pixel_ndvi.mean(axis=-1)


def day_of_years(dates: Iterable[dt.date]) -> tuple[int, ...]:
    """Derive DOY from parsed real acquisition dates."""

    return tuple(date.timetuple().tm_yday for date in dates)


def sample_grouped_parcels(
    items: Iterable[tuple[str, str, object]],
    samples_per_group: int,
    sample_seed: int,
) -> dict[tuple[str, str], list[object]]:
    """Deterministically reservoir-sample a bounded number of items per group."""

    if isinstance(samples_per_group, bool) or samples_per_group < 1:
        raise ValueError("samples_per_group must be a positive integer")
    rng = np.random.default_rng(sample_seed)
    reservoirs: dict[tuple[str, str], list[object]] = {}
    seen: dict[tuple[str, str], int] = {}
    for domain, class_name, payload in items:
        key = (domain, class_name)
        seen[key] = seen.get(key, 0) + 1
        reservoir = reservoirs.setdefault(key, [])
        if len(reservoir) < samples_per_group:
            reservoir.append(payload)
            continue
        replacement = int(rng.integers(0, seen[key]))
        if replacement < samples_per_group:
            reservoir[replacement] = payload
    return reservoirs


def collect_ndvi_diagnostic_parcels(
    data_root: Path | str,
    samples_per_group: int = 5,
    sample_seed: int = 1,
    classes: Iterable[str] | None = None,
    dataset_factory: Callable[[Path, str, tuple[str, ...]], object] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[str]]:
    """Stream all parcels while retaining only bounded NDVI examples per group."""

    if isinstance(samples_per_group, bool) or samples_per_group < 1:
        raise ValueError("samples_per_group must be a positive integer")
    data_root = Path(data_root)
    if dataset_factory is None:
        if not data_root.is_dir():
            raise FileNotFoundError(f"TimeMatch data root does not exist: {data_root}")
        from dataset import PixelSetData
        from utils import label_utils

        all_classes = tuple(label_utils.get_classes("denmark", "france", "austria"))

        def dataset_factory(root: Path, name: str, labels: tuple[str, ...]):
            return PixelSetData(
                str(root), name, labels, transform=None, closed_set=True,
                combine_spring_and_winter=False,
            )
    else:
        all_classes = tuple(classes or ())
        if not all_classes:
            raise ValueError("classes must be provided with a custom dataset_factory")

    selected = None if classes is None else tuple(dict.fromkeys(classes))
    unknown = set(selected or ()).difference(all_classes)
    if unknown:
        raise ValueError(f"unknown classes requested: {sorted(unknown)}")
    rng = np.random.default_rng(sample_seed)
    reservoirs: dict[tuple[str, str], list[dict[str, object]]] = {}
    seen: dict[tuple[str, str], int] = {}
    accumulators: dict[tuple[str, str], dict[str, object]] = {}
    class_sets: dict[str, set[str]] = {}

    for domain, dataset_name in DOMAIN_DATASETS.items():
        dataset = dataset_factory(data_root, dataset_name, all_classes)
        dates = parse_acquisition_dates(dataset.metadata["dates"])
        doys = np.asarray(day_of_years(dates), dtype=np.float64)
        present: set[str] = set()
        for parcel_index in range(len(dataset)):
            sample = dataset[parcel_index]
            class_name = all_classes[int(sample["label"])]
            if selected is not None and class_name not in selected:
                continue
            ndvi = compute_parcel_ndvi(sample["pixels"])
            if ndvi.shape != doys.shape:
                raise ValueError("parcel NDVI length must match acquisition dates")
            valid = np.isfinite(ndvi) & np.isfinite(doys)
            if "valid_pixels" in sample:
                pixel_valid = np.asarray(sample["valid_pixels"], dtype=bool)
                if pixel_valid.ndim == 2 and pixel_valid.shape[0] == len(ndvi):
                    valid &= pixel_valid.any(axis=-1)
            if not valid.any():
                continue
            key = (domain, class_name)
            present.add(class_name)
            accumulator = accumulators.setdefault(
                key,
                {
                    "dates": dates,
                    "sum": np.zeros_like(ndvi, dtype=np.float64),
                    "valid_count": np.zeros_like(ndvi, dtype=np.int64),
                    "n_parcels": 0,
                },
            )
            accumulator["sum"][valid] += ndvi[valid]
            accumulator["valid_count"][valid] += 1
            accumulator["n_parcels"] += 1

            seen[key] = seen.get(key, 0) + 1
            payload = {
                "domain": domain,
                "class_name": class_name,
                "parcel_index": parcel_index,
                "dates": dates,
                "doys": doys.copy(),
                "ndvi": ndvi.copy(),
                "valid": valid.copy(),
            }
            reservoir = reservoirs.setdefault(key, [])
            if len(reservoir) < samples_per_group:
                reservoir.append(payload)
            else:
                replacement = int(rng.integers(0, seen[key]))
                if replacement < samples_per_group:
                    reservoir[replacement] = payload
        class_sets[domain] = present

    common = sorted(set.intersection(*class_sets.values())) if class_sets else []
    if selected is not None:
        missing = set(selected).difference(common)
        if missing:
            raise ValueError(
                "requested classes are not shared by all domains: "
                + ", ".join(sorted(missing))
            )
        common = list(selected)
    rows: list[dict[str, object]] = []
    for (domain, class_name), accumulator in sorted(accumulators.items()):
        if class_name not in common:
            continue
        valid_count = accumulator["valid_count"]
        mean = np.divide(
            accumulator["sum"], valid_count,
            out=np.full_like(accumulator["sum"], np.nan),
            where=valid_count > 0,
        )
        for date, doy, value, count in zip(
            accumulator["dates"], day_of_years(accumulator["dates"]), mean, valid_count
        ):
            if not np.isfinite(value):
                continue
            rows.append({
                "class_name": class_name,
                "domain": domain,
                "date": date.isoformat(),
                "day_of_year": doy,
                "ndvi_mean": float(value),
                "n_parcels": int(accumulator["n_parcels"]),
                "n_valid_parcels": int(count),
            })
    sampled = [
        parcel
        for key in sorted(reservoirs)
        if key[1] in common
        for parcel in reservoirs[key]
    ]
    return pd.DataFrame(rows), sampled, common


def aggregate_parcel_curves(parcels: Iterable[ParcelCurve]) -> dict[tuple[str, str], RawAggregate]:
    """Aggregate parcel curves with equal weight per parcel."""

    grouped: dict[tuple[str, str], list[ParcelCurve]] = {}
    for parcel in parcels:
        grouped.setdefault((parcel.domain, parcel.class_name), []).append(parcel)
    result = {}
    for key, values in grouped.items():
        dates = values[0].dates
        if any(value.dates != dates for value in values):
            raise ValueError(f"parcels in {key} must share acquisition dates")
        curves = np.stack([value.curve for value in values])
        std = curves.std(axis=0, ddof=0)
        result[key] = RawAggregate(
            domain=key[0], class_name=key[1], dates=dates,
            mean=curves.mean(axis=0), std=std,
            sem=std / np.sqrt(len(values)), n_parcels=len(values),
        )
    return result


def build_ndvi_tables(
    aggregates: dict[tuple[str, str], RawAggregate],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build NDVI observations, parcel support, and temporal descriptors."""

    mapping = {
        "red_band": NDVI_RED_BAND, "red_index": NDVI_RED_INDEX,
        "nir_band": NDVI_NIR_BAND, "nir_index": NDVI_NIR_INDEX,
    }
    long_rows, support_rows, descriptor_rows = [], [], []
    for aggregate in aggregates.values():
        doys = day_of_years(aggregate.dates)
        for date, doy, mean, std, sem in zip(
            aggregate.dates, doys, aggregate.mean, aggregate.std, aggregate.sem
        ):
            long_rows.append({
                "class_name": aggregate.class_name, "domain": aggregate.domain,
                "date": date.isoformat(), "day_of_year": doy,
                "ndvi_mean": mean, "ndvi_std": std, "ndvi_sem": sem,
                "n_parcels": aggregate.n_parcels, **mapping,
            })
        support_rows.append({
            "class_name": aggregate.class_name, "domain": aggregate.domain,
            "n_parcels": aggregate.n_parcels, **mapping,
        })
        peak = int(np.argmax(aggregate.mean))
        minimum = int(np.argmin(aggregate.mean))
        descriptor_rows.append({
            "class_name": aggregate.class_name, "domain": aggregate.domain,
            "n_parcels": aggregate.n_parcels,
            "ndvi_mean_over_time": aggregate.mean.mean(),
            "ndvi_std_over_time": aggregate.mean.std(ddof=0),
            "ndvi_range": np.ptp(aggregate.mean),
            "ndvi_peak_value": aggregate.mean[peak], "ndvi_peak_doy": doys[peak],
            "ndvi_min_value": aggregate.mean[minimum], "ndvi_min_doy": doys[minimum],
            **mapping,
        })
    return pd.DataFrame(long_rows), pd.DataFrame(support_rows), pd.DataFrame(descriptor_rows)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_schedule(schedule: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 3.8))
    for row, domain in enumerate(DOMAIN_DATASETS):
        subset = schedule[schedule["domain"] == domain]
        axis.scatter(pd.to_datetime(subset["date"]), np.full(len(subset), row), marker="|", s=150, label=domain)
    axis.set_yticks(range(4), DOMAIN_DATASETS.keys())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.set_title("Domain observation schedule (real acquisition dates)")
    axis.grid(axis="x", alpha=0.25)
    _save(fig, path)


def _plot_support(support: pd.DataFrame, path: Path) -> None:
    matrix = support.set_index("class_name").reindex(columns=DOMAIN_DATASETS.keys()).fillna(0)
    fig, axis = plt.subplots(figsize=(7, max(4.5, 0.4 * len(matrix))))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues")
    axis.set_xticks(range(4), matrix.columns)
    axis.set_yticks(range(len(matrix)), matrix.index)
    axis.set_title("Raw parcel support by class and domain")
    fig.colorbar(image, ax=axis, label="parcel count")
    _save(fig, path)


def _plot_class_curves(class_name: str, aggregates: dict[tuple[str, str], RawAggregate], path: Path, shape_normalized: bool) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(13, 15), sharex=False)
    for channel, axis in enumerate(axes.flat):
        absent = []
        for domain in DOMAIN_DATASETS:
            aggregate = aggregates.get((domain, class_name))
            if aggregate is None:
                absent.append(domain)
                continue
            mean = aggregate.mean[:, channel]
            sem = aggregate.sem[:, channel]
            if shape_normalized:
                denominator = mean.std(ddof=0) + 1e-8
                curve = (mean - mean.mean()) / denominator
                band = sem / denominator
            else:
                curve, band = mean, sem
            dates = pd.to_datetime(list(aggregate.dates))
            axis.plot(dates, curve, label=domain, linewidth=1.4)
            axis.fill_between(dates, curve - band, curve + band, alpha=0.10)
        axis.set_title(f"channel_{channel}")
        axis.grid(alpha=0.2)
        if absent:
            axis.text(0.01, 0.02, "absent: " + ", ".join(absent), transform=axis.transAxes, fontsize=7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    qualifier = "shape-normalized diagnostic" if shape_normalized else "absolute normalized reflectance"
    fig.suptitle(f"{class_name}: {qualifier}", y=1.01)
    _save(fig, path)


def _draw_ndvi_class(
    axis: plt.Axes, class_name: str,
    aggregates: dict[tuple[str, str], RawAggregate], shape_normalized: bool,
) -> None:
    for domain in DOMAIN_DATASETS:
        aggregate = aggregates.get((domain, class_name))
        if aggregate is None:
            continue
        mean = aggregate.mean
        sem = aggregate.sem
        if shape_normalized:
            # Diagnostic visualization only; this transform is never used for training.
            denominator = mean.std(ddof=0) + 1e-8
            mean = (mean - mean.mean()) / denominator
            sem = sem / denominator
        doys = np.asarray(day_of_years(aggregate.dates))
        color = DOMAIN_COLORS[domain]
        axis.plot(
            doys, mean, marker="o", markersize=3.5, linewidth=1.4,
            color=color, label=domain,
        )
        axis.fill_between(doys, mean - sem, mean + sem, color=color, alpha=0.10)
    axis.set_xlim(1, 366)
    axis.grid(alpha=0.2)


def plot_ndvi_class_curves(
    class_name: str, aggregates: dict[tuple[str, str], RawAggregate],
    path: Path, shape_normalized: bool = False,
) -> None:
    """Plot one class on each domain's unmodified real-observation DOY points."""

    fig, axis = plt.subplots(figsize=(8, 5))
    _draw_ndvi_class(axis, class_name, aggregates, shape_normalized)
    axis.set_xlabel("Day of year (DOY)")
    axis.set_ylabel("Shape-normalized NDVI" if shape_normalized else "NDVI")
    qualifier = (
        "Shape-normalized mean NDVI (diagnostic only)"
        if shape_normalized else "Parcel-mean NDVI by domain"
    )
    axis.set_title(
        f"{class_name}: {qualifier}\n"
        f"RED={NDVI_RED_BAND}[{NDVI_RED_INDEX}], NIR={NDVI_NIR_BAND}[{NDVI_NIR_INDEX}]"
    )
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend()
    _save(fig, path)


def _plot_ndvi_overview(
    classes: list[str], aggregates: dict[tuple[str, str], RawAggregate], path: Path,
) -> None:
    ncols = min(4, max(1, len(classes)))
    nrows = int(np.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.2 * nrows), squeeze=False)
    for axis, class_name in zip(axes.flat, classes):
        _draw_ndvi_class(axis, class_name, aggregates, False)
        axis.set_title(class_name, fontsize=9)
        axis.set_xlabel("DOY")
        axis.set_ylabel("NDVI")
    for axis in axes.flat[len(classes):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle(
        "Parcel-mean NDVI by domain | "
        f"RED={NDVI_RED_BAND}[{NDVI_RED_INDEX}], NIR={NDVI_NIR_BAND}[{NDVI_NIR_INDEX}]",
        y=1.01,
    )
    _save(fig, path)


def run_raw_analysis(data_root: Path | str, output_dir: Path | str) -> dict[str, object]:
    """Analyze all four raw domains without interpolation or data mutation."""

    from dataset import PixelSetData
    from utils import label_utils

    data_root, output_dir = Path(data_root), Path(output_dir)
    if not data_root.is_dir():
        raise FileNotFoundError(f"TimeMatch data root does not exist: {data_root}")
    all_classes = tuple(label_utils.get_classes("denmark", "france", "austria"))
    accumulators: dict[tuple[str, str], dict[str, object]] = {}
    schedule_rows, support_rows = [], []
    class_sets: dict[str, set[str]] = {}
    for domain, dataset_name in DOMAIN_DATASETS.items():
        dataset = PixelSetData(
            str(data_root), dataset_name, all_classes, transform=None,
            closed_set=True, combine_spring_and_winter=False,
        )
        dates = parse_acquisition_dates(dataset.metadata["dates"])
        positions = dataset.date_positions
        for date, position in zip(dates, positions):
            schedule_rows.append({
                "domain": domain, "date": date.isoformat(),
                "day_of_year": date.timetuple().tm_yday,
                "relative_position": position, "n_observations": 1,
            })
        counts: dict[str, int] = {}
        for index in range(len(dataset)):
            sample = dataset[index]
            class_name = all_classes[int(sample["label"])]
            counts[class_name] = counts.get(class_name, 0) + 1
            curve = normalize_parcel_pixels(sample["pixels"])
            ndvi = compute_parcel_ndvi(sample["pixels"])
            accumulator = accumulators.setdefault(
                (domain, class_name),
                {"dates": dates, "sum": np.zeros_like(curve),
                 "sum_sq": np.zeros_like(curve),
                 "ndvi_sum": np.zeros_like(ndvi),
                 "ndvi_sum_sq": np.zeros_like(ndvi), "count": 0},
            )
            accumulator["sum"] += curve
            accumulator["sum_sq"] += np.square(curve)
            accumulator["ndvi_sum"] += ndvi
            accumulator["ndvi_sum_sq"] += np.square(ndvi)
            accumulator["count"] += 1
        class_sets[domain] = set(counts)
        support_rows.extend({"class_name": name, "domain": domain, "parcel_count": count} for name, count in counts.items())

    aggregates, ndvi_aggregates = {}, {}
    for (domain, class_name), accumulator in accumulators.items():
        count = int(accumulator["count"])
        mean = accumulator["sum"] / count
        variance = np.maximum(accumulator["sum_sq"] / count - np.square(mean), 0)
        std = np.sqrt(variance)
        aggregates[domain, class_name] = RawAggregate(
            domain, class_name, accumulator["dates"], mean, std,
            std / np.sqrt(count), count,
        )
        ndvi_mean = accumulator["ndvi_sum"] / count
        ndvi_variance = np.maximum(
            accumulator["ndvi_sum_sq"] / count - np.square(ndvi_mean), 0
        )
        ndvi_std = np.sqrt(ndvi_variance)
        ndvi_aggregates[domain, class_name] = RawAggregate(
            domain, class_name, accumulator["dates"], ndvi_mean, ndvi_std,
            ndvi_std / np.sqrt(count), count,
        )
    tables = output_dir / "tables"
    figures = output_dir / "figures" / "raw_timeseries"
    tables.mkdir(parents=True, exist_ok=True)
    schedule = pd.DataFrame(schedule_rows)
    schedule.to_csv(tables / "domain_observation_schedule.csv", index=False)
    support_long = pd.DataFrame(support_rows)
    support = support_long.pivot(index="class_name", columns="domain", values="parcel_count").fillna(0).astype(int).reset_index()
    support.to_csv(tables / "raw_class_support.csv", index=False)
    union = sorted(set.union(*class_sets.values()))
    intersection = sorted(set.intersection(*class_sets.values()))
    pd.DataFrame(
        [{"set": "classes_union", "class_name": name} for name in union]
        + [{"set": "classes_intersection_all_four", "class_name": name} for name in intersection]
    ).to_csv(tables / "raw_class_sets.csv", index=False)
    descriptors = []
    for aggregate in aggregates.values():
        for channel in range(aggregate.mean.shape[1]):
            curve = aggregate.mean[:, channel]
            descriptors.append({
                "class_name": aggregate.class_name, "domain": aggregate.domain,
                "channel": f"channel_{channel}", "mean_over_time": curve.mean(),
                "std_over_time": curve.std(ddof=0), "range": np.ptp(curve),
                "argmax_date": aggregate.dates[int(np.argmax(curve))].isoformat(),
                "argmin_date": aggregate.dates[int(np.argmin(curve))].isoformat(),
                "n_parcels": aggregate.n_parcels,
            })
    pd.DataFrame(descriptors).to_csv(tables / "raw_temporal_descriptors.csv", index=False)
    ndvi_tables = tables / "ndvi_doy"
    ndvi_tables.mkdir(parents=True, exist_ok=True)
    ndvi_long, ndvi_support, ndvi_descriptors = build_ndvi_tables(ndvi_aggregates)
    ndvi_long.to_csv(ndvi_tables / "ndvi_doy_long.csv", index=False)
    ndvi_support.to_csv(ndvi_tables / "ndvi_domain_class_support.csv", index=False)
    ndvi_descriptors.to_csv(ndvi_tables / "ndvi_temporal_descriptors.csv", index=False)
    _plot_schedule(schedule, figures / "domain_observation_schedule.png")
    _plot_support(support, figures / "class_support.png")
    for class_name in union:
        safe_name = class_name.replace("/", "_")
        _plot_class_curves(class_name, aggregates, figures / "absolute" / f"{safe_name}.png", False)
        _plot_class_curves(class_name, aggregates, figures / "shape_normalized" / f"{safe_name}.png", True)
        plot_ndvi_class_curves(
            class_name, ndvi_aggregates, figures / "ndvi_doy" / f"{safe_name}.png"
        )
        plot_ndvi_class_curves(
            class_name, ndvi_aggregates,
            figures / "ndvi_doy_shape_normalized" / f"{safe_name}.png",
            shape_normalized=True,
        )
    _plot_ndvi_overview(union, ndvi_aggregates, figures / "ndvi_doy_overview.png")

    manifest = {
        "command": "raw", "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_experiment_dir": None, "data_root": str(data_root),
        "task_logs_used": [], "completed": None, "incomplete": None,
        "failed": None, "diagnostic_sampling_seed": None,
        "samples_per_class": None, "checkpoint_path": None,
        "ndvi_band_mapping": {
            "red_band": NDVI_RED_BAND, "red_index": NDVI_RED_INDEX,
            "nir_band": NDVI_NIR_BAND, "nir_index": NDVI_NIR_INDEX,
            "stored_channel_order": SENTINEL2_CHANNELS,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "aggregates": aggregates, "ndvi_aggregates": ndvi_aggregates,
        "manifest": manifest, "classes_union": union,
        "classes_intersection": intersection,
    }
