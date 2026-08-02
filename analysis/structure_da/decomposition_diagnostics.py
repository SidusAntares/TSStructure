"""Offline hierarchy diagnostics for the real symmetric time decomposition."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from methods.structure_da.decomposition import SymmetricTimeKernelDecomposition

from .raw_timeseries import (
    DOMAIN_COLORS,
    DOMAIN_DATASETS,
    collect_ndvi_diagnostic_parcels,
)


TAU_FAST = 0.05
TAU_SLOW = 0.20
TIME_SCALE = 365.0
FAST_SCALE_DAYS = TAU_FAST * TIME_SCALE
SLOW_SCALE_DAYS = TAU_SLOW * TIME_SCALE
RECONSTRUCTION_TOLERANCE = 1e-5
COMPONENT_ORDER = ("original", "trend", "structure", "dynamics", "residual")
COMPONENT_LABELS = {
    "original": "Original H",
    "trend": "Trend T",
    "structure": "Structure S = T + D",
    "dynamics": "Detail D = S - T",
    "residual": "Residual R = H - S",
}
REQUIRED_COLUMNS = {
    "class_name", "domain", "date", "day_of_year", "ndvi_mean", "n_parcels",
}
PAIR_CROSSING_COLUMNS = (
    "class_name", "domain", "component", "parcel_index_a", "parcel_index_b",
    "n_common_observations", "time_span_days", "crossing_count",
    "crossings_per_100_days", "first_crossing_doy", "last_crossing_doy",
    "crossing_doys", "has_crossing", "multiple_crossings",
)


def _decomposer() -> SymmetricTimeKernelDecomposition:
    model = SymmetricTimeKernelDecomposition(
        tau_fast_init=TAU_FAST,
        tau_slow_init=TAU_SLOW,
        time_scale=TIME_SCALE,
    )
    model.eval()
    return model


def decompose_ndvi_series(
    values: np.ndarray,
    doys: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Decompose one observed NDVI curve without interpolation or date insertion."""

    values = np.asarray(values, dtype=np.float64)
    doys = np.asarray(doys, dtype=np.float64)
    if values.ndim != 1 or doys.ndim != 1 or values.shape != doys.shape:
        raise ValueError("values and doys must be equal-length one-dimensional arrays")
    if values.size == 0:
        raise ValueError("NDVI series must contain at least one observation")
    valid = np.isfinite(values) & np.isfinite(doys)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask)
        if valid_mask.shape != values.shape:
            raise ValueError("valid_mask must have the same shape as values")
        valid &= valid_mask.astype(bool)
    if not valid.any():
        raise ValueError("NDVI series has no valid observations")

    safe_values = np.where(valid, values, 0.0).astype(np.float32)
    safe_doys = np.where(valid, doys, 0.0).astype(np.float64)
    with torch.no_grad():
        output = _decomposer()(
            torch.from_numpy(safe_values).view(1, -1, 1),
            torch.from_numpy(safe_doys),
            torch.from_numpy(valid).view(1, -1),
        )
    trend = output.trend[0, :, 0].cpu().numpy().astype(np.float64)
    dynamics = output.dynamics[0, :, 0].cpu().numpy().astype(np.float64)
    residual = output.residual[0, :, 0].cpu().numpy().astype(np.float64)
    arrays = {
        "original": values.copy(),
        "trend": trend,
        "structure": trend + dynamics,
        "dynamics": dynamics,
        "residual": residual,
    }
    for array in arrays.values():
        array[~valid] = np.nan
    structure_error = float(np.max(np.abs(
        arrays["structure"][valid]
        - (arrays["trend"][valid] + arrays["dynamics"][valid])
    )))
    input_error = float(np.max(np.abs(
        arrays["original"][valid]
        - (arrays["structure"][valid] + arrays["residual"][valid])
    )))
    if structure_error >= RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(f"S = T + D check failed: max error {structure_error:.3e}")
    if input_error >= RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(f"H = S + R check failed: max error {input_error:.3e}")
    return {
        "doys": doys.copy(),
        "valid": valid,
        **arrays,
        "max_abs_structure_error": structure_error,
        "max_abs_input_error": input_error,
    }


def decompose_ndvi_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose class-domain mean curves at their real acquisition dates."""

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"NDVI CSV is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("NDVI CSV contains no observations")
    component_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    for (class_name, domain), group in frame.groupby(
        ["class_name", "domain"], sort=True
    ):
        group = group.sort_values(["day_of_year", "date"], kind="stable")
        parcel_counts = group["n_parcels"].unique()
        if len(parcel_counts) != 1:
            raise ValueError(f"{class_name}/{domain} has inconsistent n_parcels values")
        result = decompose_ndvi_series(
            group["ndvi_mean"].to_numpy(dtype=np.float64),
            group["day_of_year"].to_numpy(dtype=np.float64),
        )
        n_parcels = int(parcel_counts[0])
        for row_index, row in enumerate(group.itertuples(index=False)):
            if not result["valid"][row_index]:
                continue
            for component in COMPONENT_ORDER:
                component_rows.append({
                    "class_name": class_name,
                    "domain": domain,
                    "date": str(row.date),
                    "day_of_year": int(row.day_of_year),
                    "component": component,
                    "value": float(result[component][row_index]),
                    "n_parcels": n_parcels,
                    "tau_fast": TAU_FAST,
                    "tau_slow": TAU_SLOW,
                    "fast_scale_days": FAST_SCALE_DAYS,
                    "slow_scale_days": SLOW_SCALE_DAYS,
                })
        reconstruction_rows.append({
            "class_name": class_name,
            "domain": domain,
            "n_observations": int(result["valid"].sum()),
            "n_parcels": n_parcels,
            "max_abs_structure_error": result["max_abs_structure_error"],
            "max_abs_input_error": result["max_abs_input_error"],
        })
    components = pd.DataFrame(component_rows)
    order = {name: index for index, name in enumerate(COMPONENT_ORDER)}
    components["_component_order"] = components["component"].map(order)
    components = components.sort_values(
        ["class_name", "domain", "day_of_year", "_component_order"],
        kind="stable",
    ).drop(columns="_component_order").reset_index(drop=True)
    return components, pd.DataFrame(reconstruction_rows)


def component_variation(
    values: np.ndarray, doys: np.ndarray, eps: float = 1e-8,
) -> dict[str, float | int]:
    """Compute exploratory variation on adjacent real, irregular observations."""

    values = np.asarray(values, dtype=np.float64)
    doys = np.asarray(doys, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim < 2 or doys.ndim != 1 or values.shape[0] != doys.size:
        raise ValueError("values must have time first and match one-dimensional doys")
    flattened = values.reshape(values.shape[0], -1)
    point_valid = np.isfinite(doys) & np.isfinite(flattened).all(axis=1)
    observed = flattened[point_valid]
    observed_doys = doys[point_valid]
    if len(observed_doys) < 2:
        return {"total_variation": 0.0, "roughness": 0.0, "n_intervals": 0}
    delta = observed[1:] - observed[:-1]
    norms = np.linalg.norm(delta, axis=1)
    delta_t = np.maximum(observed_doys[1:] - observed_doys[:-1], eps)
    return {
        "total_variation": float(norms.sum()),
        "roughness": float(np.sum(np.square(norms) / delta_t)),
        "n_intervals": int(len(delta_t)),
    }


def summarize_parcel_components(
    domain: str,
    class_name: str,
    records: Iterable[dict[str, object]],
) -> pd.DataFrame:
    """Summarize quantiles after decomposing each parcel independently."""

    records = list(records)
    if not records:
        return pd.DataFrame()
    reference_doys = np.asarray(records[0]["doys"], dtype=np.float64)
    for record in records[1:]:
        if not np.array_equal(np.asarray(record["doys"]), reference_doys):
            raise ValueError("parcel records in one class/domain must share acquisition dates")
    rows: list[dict[str, object]] = []
    for component in COMPONENT_ORDER:
        stacked = np.stack([np.asarray(record[component]) for record in records])
        for index, doy in enumerate(reference_doys):
            column = stacked[:, index]
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                continue
            rows.append({
                "class_name": class_name,
                "domain": domain,
                "day_of_year": float(doy),
                "component": component,
                "median": float(np.quantile(finite, 0.50)),
                "q25": float(np.quantile(finite, 0.25)),
                "q75": float(np.quantile(finite, 0.75)),
                "n_samples": int(finite.size),
            })
    return pd.DataFrame(rows)


def _validate_crossing_eps(crossing_eps: float) -> float:
    try:
        crossing_eps = float(crossing_eps)
    except (TypeError, ValueError) as error:
        raise ValueError("crossing_eps must be a finite number greater than or equal to zero") from error
    if not math.isfinite(crossing_eps) or crossing_eps < 0:
        raise ValueError("crossing_eps must be a finite number greater than or equal to zero")
    return crossing_eps


def count_pairwise_crossings(
    first_values: np.ndarray,
    second_values: np.ndarray,
    doys: np.ndarray,
    first_valid: np.ndarray,
    second_valid: np.ndarray,
    crossing_eps: float = 1e-4,
) -> dict[str, object]:
    """Count epsilon-aware sign changes on common real observations."""

    crossing_eps = _validate_crossing_eps(crossing_eps)
    first_values = np.asarray(first_values, dtype=np.float64)
    second_values = np.asarray(second_values, dtype=np.float64)
    doys = np.asarray(doys, dtype=np.float64)
    first_valid = np.asarray(first_valid, dtype=bool)
    second_valid = np.asarray(second_valid, dtype=bool)
    arrays = (first_values, second_values, doys, first_valid, second_valid)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("crossing inputs must be one-dimensional")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("crossing inputs must have identical shapes")

    common = (
        first_valid & second_valid & np.isfinite(first_values)
        & np.isfinite(second_values) & np.isfinite(doys)
    )
    common_doys = doys[common]
    first = first_values[common]
    second = second_values[common]
    order = np.argsort(common_doys, kind="stable")
    common_doys = common_doys[order]
    delta = (first - second)[order]
    states = np.where(
        delta > crossing_eps, 1, np.where(delta < -crossing_eps, -1, 0)
    )
    nonzero = states != 0
    signed_states = states[nonzero]
    signed_doys = common_doys[nonzero]
    crossing_doys: list[float] = []
    if len(signed_states) >= 2:
        changes = signed_states[1:] != signed_states[:-1]
        crossing_doys = [
            float((left + right) / 2.0)
            for left, right in zip(signed_doys[:-1][changes], signed_doys[1:][changes])
        ]
    crossing_count = len(crossing_doys)
    time_span_days = (
        float(common_doys[-1] - common_doys[0]) if len(common_doys) >= 2 else 0.0
    )
    return {
        "n_common_observations": int(len(common_doys)),
        "time_span_days": time_span_days,
        "crossing_count": crossing_count,
        "crossings_per_100_days": float(
            100.0 * crossing_count / max(time_span_days, 1.0)
        ),
        "first_crossing_doy": crossing_doys[0] if crossing_doys else np.nan,
        "last_crossing_doy": crossing_doys[-1] if crossing_doys else np.nan,
        "crossing_doys": crossing_doys,
        "has_crossing": crossing_count >= 1,
        "multiple_crossings": crossing_count >= 2,
    }


def _matching_doys(first: dict[str, object], second: dict[str, object]) -> np.ndarray:
    first_doys = np.asarray(first["doys"], dtype=np.float64)
    second_doys = np.asarray(second["doys"], dtype=np.float64)
    if (
        first_doys.shape != second_doys.shape
        or not np.allclose(first_doys, second_doys, equal_nan=True)
    ):
        raise ValueError("parcel records in one domain must share acquisition DOYs")
    return first_doys


def build_pairwise_crossing_table(
    grouped_records: dict[tuple[str, str], list[dict[str, object]]],
    crossing_eps: float,
) -> pd.DataFrame:
    """Build deterministic unordered-pair crossing diagnostics for T and S."""

    crossing_eps = _validate_crossing_eps(crossing_eps)
    rows: list[dict[str, object]] = []
    for (domain, class_name), records in sorted(grouped_records.items()):
        ordered = sorted(records, key=lambda record: record["parcel_index"])
        indices = [record["parcel_index"] for record in ordered]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate parcel_index in {class_name}/{domain}")
        for component in ("trend", "structure"):
            for first, second in combinations(ordered, 2):
                result = count_pairwise_crossings(
                    first[component], second[component], _matching_doys(first, second),
                    first["valid"], second["valid"], crossing_eps,
                )
                crossing_doys = ";".join(str(float(value)) for value in result.pop("crossing_doys"))
                rows.append({
                    "class_name": class_name,
                    "domain": domain,
                    "component": component,
                    "parcel_index_a": first["parcel_index"],
                    "parcel_index_b": second["parcel_index"],
                    **result,
                    "crossing_doys": crossing_doys,
                })
    table = pd.DataFrame(rows, columns=PAIR_CROSSING_COLUMNS)
    if not table.empty:
        table = table.sort_values(
            ["class_name", "domain", "component", "parcel_index_a", "parcel_index_b"],
            kind="stable",
        ).reset_index(drop=True)
    return table


def summarize_pairwise_crossings(pair_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize pair crossings and count parcels from their explicit identities."""

    columns = (
        "class_name", "domain", "component", "n_parcels", "n_pairs",
        "crossing_count_mean", "crossing_count_median", "crossing_count_q25",
        "crossing_count_q75", "crossing_count_p90",
        "crossings_per_100_days_median", "fraction_with_any_crossing",
        "fraction_with_multiple_crossings", "fraction_with_at_least_3_crossings",
    )
    if pair_table.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for keys, group in pair_table.groupby(
        ["class_name", "domain", "component"], sort=True
    ):
        counts = group["crossing_count"].to_numpy(dtype=np.float64)
        parcel_ids = set(group["parcel_index_a"]).union(group["parcel_index_b"])
        rows.append({
            "class_name": keys[0], "domain": keys[1], "component": keys[2],
            "n_parcels": len(parcel_ids), "n_pairs": len(group),
            "crossing_count_mean": float(np.mean(counts)),
            "crossing_count_median": float(np.quantile(counts, 0.50)),
            "crossing_count_q25": float(np.quantile(counts, 0.25)),
            "crossing_count_q75": float(np.quantile(counts, 0.75)),
            "crossing_count_p90": float(np.quantile(counts, 0.90)),
            "crossings_per_100_days_median": float(
                np.quantile(group["crossings_per_100_days"], 0.50)
            ),
            "fraction_with_any_crossing": float(np.mean(counts >= 1)),
            "fraction_with_multiple_crossings": float(np.mean(counts >= 2)),
            "fraction_with_at_least_3_crossings": float(np.mean(counts >= 3)),
        })
    return pd.DataFrame(rows, columns=columns)


def _pair_component_mse(
    first: dict[str, object], second: dict[str, object], component: str,
) -> float:
    doys = _matching_doys(first, second)
    first_values = np.asarray(first[component], dtype=np.float64)
    second_values = np.asarray(second[component], dtype=np.float64)
    first_valid = np.asarray(first["valid"], dtype=bool)
    second_valid = np.asarray(second["valid"], dtype=bool)
    if len({array.shape for array in (
        first_values, second_values, first_valid, second_valid,
    )}) != 1:
        raise ValueError("component values and validity masks must have identical shapes")
    common = (
        first_valid & second_valid & np.isfinite(first_values)
        & np.isfinite(second_values) & np.isfinite(doys)
    )
    if not common.any():
        return np.nan
    return float(np.mean(np.square(first_values[common] - second_values[common])))


def select_component_medoid(
    records: list[dict[str, object]],
    component: str,
    eps: float = 1e-8,
) -> dict[str, object]:
    """Select an actual parcel minimizing mean common-observation pair MSE."""

    if component not in ("trend", "structure"):
        raise ValueError("component must be 'trend' or 'structure'")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and greater than zero")
    if not records:
        raise ValueError("records must contain at least one parcel")
    ordered = sorted(records, key=lambda record: record["parcel_index"])
    candidates: list[tuple[float, object, dict[str, object], int]] = []
    for candidate in ordered:
        distances = [
            _pair_component_mse(candidate, other, component)
            for other in ordered if other is not candidate
        ]
        valid_distances = np.asarray(
            [value for value in distances if np.isfinite(value)], dtype=np.float64
        )
        if len(ordered) == 1:
            score = 0.0
        elif valid_distances.size:
            score = float(np.mean(valid_distances))
        else:
            score = np.inf
        candidates.append((score, candidate["parcel_index"], candidate, len(valid_distances)))
    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate[0] < best[0] - eps:
            best = candidate
    score, _, record, n_valid_pairs = best
    return {
        "record": record,
        "medoid_parcel_index": record["parcel_index"],
        "medoid_mean_pair_mse": float(score) if np.isfinite(score) else np.nan,
        "n_parcels": len(ordered),
        "n_valid_pairs": n_valid_pairs,
    }


def build_component_medoid_table(
    grouped_records: dict[tuple[str, str], list[dict[str, object]]],
) -> pd.DataFrame:
    """Select real trend and structure medoid parcels for every group."""

    rows: list[dict[str, object]] = []
    for (domain, class_name), records in sorted(grouped_records.items()):
        for component in ("trend", "structure"):
            result = select_component_medoid(records, component)
            rows.append({
                "class_name": class_name, "domain": domain, "component": component,
                "medoid_parcel_index": result["medoid_parcel_index"],
                "medoid_mean_pair_mse": result["medoid_mean_pair_mse"],
                "n_parcels": result["n_parcels"],
                "n_valid_pairs": result["n_valid_pairs"],
            })
    return pd.DataFrame(rows).sort_values(
        ["class_name", "domain", "component"], kind="stable"
    ).reset_index(drop=True)


def _distribution(values: Iterable[float], prefix: str) -> dict[str, float]:
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan, f"{prefix}_median": np.nan,
            f"{prefix}_q25": np.nan, f"{prefix}_q75": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.quantile(values, 0.50)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
    }


def build_identity_dispersion_table(
    grouped_records: dict[tuple[str, str], list[dict[str, object]]],
    medoid_table: pd.DataFrame,
) -> pd.DataFrame:
    """Measure unregistered within-group dispersion around actual medoids."""

    # Post-registration dispersion is intentionally deferred until the
    # final T-only phase registration objective and reference construction
    # are frozen. The current learned warp estimator cannot be used as an
    # untrained offline optimizer.
    rows: list[dict[str, object]] = []
    for (domain, class_name), records in sorted(grouped_records.items()):
        by_index = {record["parcel_index"]: record for record in records}
        for component in ("trend", "structure"):
            selected = medoid_table[
                (medoid_table["class_name"] == class_name)
                & (medoid_table["domain"] == domain)
                & (medoid_table["component"] == component)
            ]
            if len(selected) != 1:
                raise ValueError(f"missing unique medoid for {class_name}/{domain}/{component}")
            medoid_index = selected.iloc[0]["medoid_parcel_index"]
            if medoid_index not in by_index:
                raise ValueError("medoid_parcel_index does not identify an input record")
            medoid = by_index[medoid_index]
            medoid_distances = [
                _pair_component_mse(record, medoid, component) for record in records
            ]
            pair_distances = [
                _pair_component_mse(first, second, component)
                for first, second in combinations(records, 2)
            ]
            rows.append({
                "class_name": class_name, "domain": domain, "component": component,
                "medoid_parcel_index": medoid_index,
                **_distribution(medoid_distances, "identity_medoid_dispersion"),
                **_distribution(pair_distances, "identity_pairwise_dispersion"),
                "n_parcels": len(records),
                "n_valid_pairs": int(np.isfinite(pair_distances).sum()),
            })
    return pd.DataFrame(rows).sort_values(
        ["class_name", "domain", "component"], kind="stable"
    ).reset_index(drop=True)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_curves(axis: plt.Axes, frame: pd.DataFrame, component: str) -> None:
    component_frame = frame[frame["component"] == component]
    for domain in DOMAIN_DATASETS:
        curve = component_frame[component_frame["domain"] == domain].sort_values(
            "day_of_year", kind="stable"
        )
        if not curve.empty:
            axis.plot(
                curve["day_of_year"], curve["value"], marker="o", markersize=3.5,
                linewidth=1.4, color=DOMAIN_COLORS[domain], label=domain,
            )
    axis.set_xlim(1, 366)
    axis.set_ylabel(COMPONENT_LABELS[component])
    axis.grid(alpha=0.2)


def _share_trend_structure_limits(axes: Iterable[plt.Axes], frame: pd.DataFrame) -> None:
    values = frame[frame["component"].isin(("trend", "structure"))]["value"]
    if values.empty:
        return
    low, high = float(values.min()), float(values.max())
    margin = max((high - low) * 0.05, 1e-4)
    for axis in axes:
        axis.set_ylim(low - margin, high + margin)


def _add_legend(fig: plt.Figure, axis: plt.Axes) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles))


def _plot_class_decomposition(
    class_name: str, components: pd.DataFrame, path: Path,
) -> None:
    class_frame = components[components["class_name"] == class_name]
    fig, axes = plt.subplots(5, 1, figsize=(9, 15), sharex=True)
    for axis, component in zip(axes, COMPONENT_ORDER):
        _plot_curves(axis, class_frame, component)
    _share_trend_structure_limits((axes[1], axes[2]), class_frame)
    axes[-1].set_xlabel("Day of year (DOY)")
    _add_legend(fig, axes[0])
    fig.suptitle(
        f"{class_name}: H = S + R, S = T + D\n"
        f"tau_fast={TAU_FAST:.2f} (~{FAST_SCALE_DAYS:.2f} d), "
        f"tau_slow={TAU_SLOW:.2f} (~{SLOW_SCALE_DAYS:.1f} d)"
    )
    _save(fig, path)


def _plot_trend_structure_comparison(
    class_name: str, components: pd.DataFrame, path: Path,
) -> None:
    class_frame = components[components["class_name"] == class_name]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    _plot_curves(axes[0], class_frame, "trend")
    _plot_curves(axes[1], class_frame, "structure")
    _share_trend_structure_limits(axes, class_frame)
    axes[-1].set_xlabel("Day of year (DOY)")
    _add_legend(fig, axes[0])
    fig.suptitle(f"{class_name}: candidate phase anchor T vs shape carrier S\nH = S + R, S = T + D")
    _save(fig, path)


def _plot_parcel_example(record: dict[str, object], path: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(8, 13), sharex=True)
    valid = np.asarray(record["valid"], dtype=bool)
    doys = np.asarray(record["doys"])[valid]
    for axis, component in zip(axes, COMPONENT_ORDER):
        axis.plot(doys, np.asarray(record[component])[valid], marker="o", linewidth=1.3)
        axis.set_ylabel(COMPONENT_LABELS[component])
        axis.grid(alpha=0.2)
    _share_trend_structure_limits(
        (axes[1], axes[2]),
        pd.DataFrame({
            "component": np.repeat(("trend", "structure"), len(doys)),
            "value": np.concatenate((record["trend"][valid], record["structure"][valid])),
        }),
    )
    axes[-1].set_xlabel("Day of year (DOY)")
    fig.suptitle(
        f"{record['class_name']} / {record['domain']} / parcel {record['parcel_index']}\n"
        "H = S + R, S = T + D"
    )
    _save(fig, path)


def _plot_component_quantiles(
    class_name: str, summary: pd.DataFrame, component: str, path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    selected = summary[
        (summary["class_name"] == class_name) & (summary["component"] == component)
    ]
    for domain in DOMAIN_DATASETS:
        curve = selected[selected["domain"] == domain].sort_values("day_of_year")
        if curve.empty:
            continue
        x = curve["day_of_year"].to_numpy(dtype=float)
        axis.plot(x, curve["median"], color=DOMAIN_COLORS[domain], label=domain)
        axis.fill_between(
            x, curve["q25"].to_numpy(dtype=float), curve["q75"].to_numpy(dtype=float),
            color=DOMAIN_COLORS[domain], alpha=0.18,
        )
    axis.set_xlim(1, 366)
    axis.set_xlabel("Day of year (DOY)")
    axis.set_ylabel(COMPONENT_LABELS[component])
    axis.grid(alpha=0.2)
    axis.legend()
    fig.suptitle(f"{class_name}: parcel median and IQR for {COMPONENT_LABELS[component]}")
    _save(fig, path)


def _plot_class_spaghetti(
    class_name: str,
    grouped_records: dict[tuple[str, str], list[dict[str, object]]],
    crossing_summary: pd.DataFrame,
    path: Path,
) -> None:
    """Plot sampled parcel identities, pointwise medians, and actual medoids."""

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), squeeze=False)
    for column, domain in enumerate(DOMAIN_DATASETS):
        records = grouped_records.get((domain, class_name), [])
        for row, component in enumerate(("trend", "structure")):
            axis = axes[row, column]
            for record in records:
                values = np.asarray(record[component], dtype=np.float64)
                doys = np.asarray(record["doys"], dtype=np.float64)
                valid = (
                    np.asarray(record["valid"], dtype=bool)
                    & np.isfinite(values) & np.isfinite(doys)
                )
                order = np.argsort(doys[valid], kind="stable")
                axis.plot(
                    doys[valid][order], values[valid][order],
                    color=DOMAIN_COLORS[domain], linewidth=0.8, alpha=0.12,
                    marker=None,
                )
            if records:
                reference_doys = np.asarray(records[0]["doys"], dtype=np.float64)
                for record in records[1:]:
                    _matching_doys(records[0], record)
                pointwise_rows: list[tuple[float, float]] = []
                for index, doy in enumerate(reference_doys):
                    observed = [
                        float(record[component][index])
                        for record in records
                        if record["valid"][index]
                        and np.isfinite(record[component][index])
                        and np.isfinite(record["doys"][index])
                    ]
                    if observed:
                        pointwise_rows.append((float(doy), float(np.median(observed))))
                median_doys, median_values = zip(*pointwise_rows)
                axis.plot(
                    median_doys, median_values, color="black",
                    linewidth=2.2, linestyle="-", marker=None,
                    label="pointwise median",
                )
                medoid = select_component_medoid(records, component)
                medoid_record = medoid["record"]
                medoid_values = np.asarray(medoid_record[component], dtype=np.float64)
                medoid_doys = np.asarray(medoid_record["doys"], dtype=np.float64)
                medoid_valid = (
                    np.asarray(medoid_record["valid"], dtype=bool)
                    & np.isfinite(medoid_values) & np.isfinite(medoid_doys)
                )
                medoid_order = np.argsort(medoid_doys[medoid_valid], kind="stable")
                axis.plot(
                    medoid_doys[medoid_valid][medoid_order],
                    medoid_values[medoid_valid][medoid_order],
                    color="black", linewidth=1.6, linestyle="--", marker=None,
                    label=f"medoid parcel {medoid['medoid_parcel_index']}",
                )
            selected = crossing_summary[
                (crossing_summary["class_name"] == class_name)
                & (crossing_summary["domain"] == domain)
                & (crossing_summary["component"] == component)
            ]
            median_crossings = (
                float(selected.iloc[0]["crossing_count_median"])
                if len(selected) == 1 else 0.0
            )
            symbol = "T" if component == "trend" else "S"
            axis.set_title(
                f"{domain} | n_parcels={len(records)} | "
                f"{symbol} median pair crossings={median_crossings:g}"
            )
            axis.set_xlim(1, 366)
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel(COMPONENT_LABELS[component])
            if row == 1:
                axis.set_xlabel("Day of year (DOY)")
            if records:
                axis.legend(fontsize=8)
    fig.suptitle(
        f"{class_name}: individual parcel curves, pointwise median, and actual medoid"
    )
    _save(fig, path)


def _write_mean_outputs(
    components: pd.DataFrame,
    table_dir: Path,
    figure_dir: Path,
) -> list[str]:
    table_dir.mkdir(parents=True, exist_ok=True)
    components.to_csv(table_dir / "mean_components_long.csv", index=False)
    classes = sorted(components["class_name"].unique())
    for class_name in classes:
        safe_name = str(class_name).replace("/", "_")
        _plot_class_decomposition(
            class_name, components, figure_dir / "class_domain_mean" / f"{safe_name}.png"
        )
        _plot_trend_structure_comparison(
            class_name, components,
            figure_dir / "trend_structure_comparison" / f"{safe_name}.png",
        )
    return classes


def run_ndvi_decomposition(
    ndvi_csv: Path | str, output_dir: Path | str,
) -> dict[str, object]:
    """Decompose the existing compact class-domain mean NDVI table."""

    ndvi_csv, output_dir = Path(ndvi_csv), Path(output_dir)
    if not ndvi_csv.is_file():
        raise FileNotFoundError(f"NDVI CSV does not exist: {ndvi_csv}")
    components, reconstruction = decompose_ndvi_frame(pd.read_csv(ndvi_csv))
    table_dir = output_dir / "tables" / "ndvi_ts_decomposition"
    figure_dir = output_dir / "figures" / "raw_timeseries" / "ndvi_ts_decomposition"
    classes = _write_mean_outputs(components, table_dir, figure_dir)
    reconstruction.to_csv(table_dir / "reconstruction_check.csv", index=False)
    maximum = float(reconstruction[
        ["max_abs_structure_error", "max_abs_input_error"]
    ].to_numpy().max())
    return {
        "components": components,
        "reconstruction": reconstruction,
        "classes": classes,
        "max_reconstruction_error": maximum,
    }


def run_ndvi_ts_diagnostic(
    data_root: Path | str,
    output_dir: Path | str,
    samples_per_group: int = 5,
    sample_seed: int = 1,
    classes: Iterable[str] | None = None,
    dataset_factory: Callable[[Path, str, tuple[str, ...]], object] | None = None,
    crossing_eps: float = 1e-4,
) -> dict[str, object]:
    """Run mean and bounded per-parcel T/S hierarchy diagnostics."""

    crossing_eps = _validate_crossing_eps(crossing_eps)
    output_dir = Path(output_dir)
    mean_frame, sampled, common_classes = collect_ndvi_diagnostic_parcels(
        data_root, samples_per_group, sample_seed, classes, dataset_factory,
    )
    mean_components, mean_reconstruction = decompose_ndvi_frame(mean_frame)
    table_dir = output_dir / "tables" / "ndvi_ts_decomposition"
    figure_dir = output_dir / "figures" / "raw_timeseries" / "ndvi_ts_decomposition"
    output_classes = _write_mean_outputs(
        mean_components, table_dir, figure_dir,
    )

    grouped_records: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    variation_rows: list[dict[str, object]] = []
    parcel_reconstruction: list[dict[str, object]] = []
    for parcel in sampled:
        result = decompose_ndvi_series(parcel["ndvi"], parcel["doys"], parcel["valid"])
        record = {**parcel, **result}
        key = (str(parcel["domain"]), str(parcel["class_name"]))
        grouped_records[key].append(record)
        safe_class = str(parcel["class_name"]).replace("/", "_")
        _plot_parcel_example(
            record,
            figure_dir / "parcel_examples" / safe_class
            / f"{parcel['domain']}_parcel_{parcel['parcel_index']}.png",
        )
        parcel_reconstruction.append({
            "level": "parcel",
            "class_name": parcel["class_name"],
            "domain": parcel["domain"],
            "parcel_index": parcel["parcel_index"],
            "n_observations": int(np.asarray(result["valid"]).sum()),
            "n_parcels": 1,
            "max_abs_structure_error": result["max_abs_structure_error"],
            "max_abs_input_error": result["max_abs_input_error"],
        })
        for component in COMPONENT_ORDER:
            variation_rows.append({
                "class_name": parcel["class_name"],
                "domain": parcel["domain"],
                "parcel_index": parcel["parcel_index"],
                "component": component,
                **component_variation(result[component], result["doys"]),
            })

    summaries = [
        summarize_parcel_components(domain, class_name, records)
        for (domain, class_name), records in sorted(grouped_records.items())
    ]
    parcel_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    pair_crossings = build_pairwise_crossing_table(grouped_records, crossing_eps)
    crossing_summary = summarize_pairwise_crossings(pair_crossings)
    medoids = build_component_medoid_table(grouped_records)
    identity_dispersion = build_identity_dispersion_table(grouped_records, medoids)
    pair_crossings.to_csv(table_dir / "parcel_pair_crossings.csv", index=False)
    crossing_summary.to_csv(
        table_dir / "parcel_crossing_group_summary.csv", index=False
    )
    medoids.to_csv(table_dir / "parcel_component_medoids.csv", index=False)
    identity_dispersion.to_csv(
        table_dir / "parcel_identity_dispersion.csv", index=False
    )

    parcel_summary.to_csv(table_dir / "parcel_components_summary.csv", index=False)
    variation = pd.DataFrame(variation_rows)
    variation.to_csv(table_dir / "component_variation_per_parcel.csv", index=False)
    summary_rows: list[dict[str, object]] = []
    for keys, group in variation.groupby(["class_name", "domain", "component"], sort=True):
        for metric in ("total_variation", "roughness"):
            values = group[metric].to_numpy(dtype=float)
            summary_rows.append({
                "class_name": keys[0], "domain": keys[1], "component": keys[2],
                "metric": metric, "median": float(np.quantile(values, 0.50)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "n_parcels": int(len(values)),
            })
    variation_summary = pd.DataFrame(summary_rows)
    variation_summary.to_csv(
        table_dir / "component_variation_group_summary.csv", index=False
    )
    mean_reconstruction = mean_reconstruction.assign(
        level="class_domain_mean", parcel_index=pd.NA
    )
    reconstruction = pd.concat(
        [mean_reconstruction, pd.DataFrame(parcel_reconstruction)], ignore_index=True
    )
    reconstruction.to_csv(table_dir / "reconstruction_check.csv", index=False)

    for class_name in common_classes:
        safe_name = str(class_name).replace("/", "_")
        for component in ("trend", "structure"):
            _plot_component_quantiles(
                class_name, parcel_summary, component,
                figure_dir / "class_domain_quantiles" / component / f"{safe_name}.png",
            )
        _plot_class_spaghetti(
            class_name, grouped_records, crossing_summary,
            figure_dir / "class_domain_spaghetti" / f"{safe_name}.png",
        )
    return {
        "components": mean_components,
        "parcel_summary": parcel_summary,
        "variation": variation,
        "variation_summary": variation_summary,
        "reconstruction": reconstruction,
        "sampled_parcels": sampled,
        "classes": output_classes,
        "pair_crossings": pair_crossings,
        "crossing_summary": crossing_summary,
        "medoids": medoids,
        "identity_dispersion": identity_dispersion,
    }
