"""Read-only raw parcel time-series diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
from typing import Iterable

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
            accumulator = accumulators.setdefault(
                (domain, class_name),
                {"dates": dates, "sum": np.zeros_like(curve),
                 "sum_sq": np.zeros_like(curve), "count": 0},
            )
            accumulator["sum"] += curve
            accumulator["sum_sq"] += np.square(curve)
            accumulator["count"] += 1
        class_sets[domain] = set(counts)
        support_rows.extend({"class_name": name, "domain": domain, "parcel_count": count} for name, count in counts.items())

    aggregates = {}
    for (domain, class_name), accumulator in accumulators.items():
        count = int(accumulator["count"])
        mean = accumulator["sum"] / count
        variance = np.maximum(accumulator["sum_sq"] / count - np.square(mean), 0)
        std = np.sqrt(variance)
        aggregates[domain, class_name] = RawAggregate(
            domain, class_name, accumulator["dates"], mean, std,
            std / np.sqrt(count), count,
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
    _plot_schedule(schedule, figures / "domain_observation_schedule.png")
    _plot_support(support, figures / "class_support.png")
    for class_name in union:
        safe_name = class_name.replace("/", "_")
        _plot_class_curves(class_name, aggregates, figures / "absolute" / f"{safe_name}.png", False)
        _plot_class_curves(class_name, aggregates, figures / "shape_normalized" / f"{safe_name}.png", True)

    manifest = {
        "command": "raw", "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_experiment_dir": None, "data_root": str(data_root),
        "task_logs_used": [], "completed": None, "incomplete": None,
        "failed": None, "diagnostic_sampling_seed": None,
        "samples_per_class": None, "checkpoint_path": None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"aggregates": aggregates, "manifest": manifest, "classes_union": union, "classes_intersection": intersection}
