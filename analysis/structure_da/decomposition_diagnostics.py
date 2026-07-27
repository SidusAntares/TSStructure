"""Diagnose the real Structure DA time-kernel decomposition on mean NDVI curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from methods.structure_da.decomposition import SymmetricTimeKernelDecomposition

from .raw_timeseries import DOMAIN_COLORS, DOMAIN_DATASETS


TAU_FAST = 0.05
TAU_SLOW = 0.20
TIME_SCALE = 365.0
FAST_SCALE_DAYS = TAU_FAST * TIME_SCALE
SLOW_SCALE_DAYS = TAU_SLOW * TIME_SCALE
RECONSTRUCTION_TOLERANCE = 1e-5
COMPONENT_ORDER = ("trend", "dynamics", "residual")
REQUIRED_COLUMNS = {
    "class_name", "domain", "date", "day_of_year", "ndvi_mean", "n_parcels",
}


def decompose_ndvi_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose sorted class-domain mean NDVI curves without resampling."""

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"NDVI CSV is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("NDVI CSV contains no observations")

    decomposer = SymmetricTimeKernelDecomposition(
        tau_fast_init=TAU_FAST,
        tau_slow_init=TAU_SLOW,
        time_scale=TIME_SCALE,
    )
    decomposer.eval()
    component_rows, reconstruction_rows = [], []
    grouped = frame.groupby(["class_name", "domain"], sort=True)
    with torch.no_grad():
        for (class_name, domain), group in grouped:
            group = group.sort_values(["day_of_year", "date"], kind="stable")
            parcel_counts = group["n_parcels"].unique()
            if len(parcel_counts) != 1:
                raise ValueError(
                    f"{class_name}/{domain} has inconsistent n_parcels values"
                )
            values = group["ndvi_mean"].to_numpy(dtype=np.float32)
            doys = group["day_of_year"].to_numpy(dtype=np.float64)
            features = torch.from_numpy(values).view(1, -1, 1)
            positions = torch.from_numpy(doys)
            output = decomposer(features, positions)
            arrays = {
                "trend": output.trend[0, :, 0].cpu().numpy(),
                "dynamics": output.dynamics[0, :, 0].cpu().numpy(),
                "residual": output.residual[0, :, 0].cpu().numpy(),
            }
            reconstructed = sum(arrays.values())
            error = float(np.max(np.abs(values - reconstructed)))
            if error >= RECONSTRUCTION_TOLERANCE:
                raise RuntimeError(
                    f"NDVI decomposition reconstruction failed for {class_name}/{domain}: "
                    f"max error {error:.3e}"
                )
            n_parcels = int(parcel_counts[0])
            for row_index, row in enumerate(group.itertuples(index=False)):
                for component in COMPONENT_ORDER:
                    component_rows.append({
                        "class_name": class_name, "domain": domain,
                        "date": str(row.date), "day_of_year": int(row.day_of_year),
                        "component": component,
                        "value": float(arrays[component][row_index]),
                        "n_parcels": n_parcels,
                        "tau_fast": TAU_FAST, "tau_slow": TAU_SLOW,
                        "fast_scale_days": FAST_SCALE_DAYS,
                        "slow_scale_days": SLOW_SCALE_DAYS,
                    })
            reconstruction_rows.append({
                "class_name": class_name, "domain": domain,
                "n_observations": len(group), "n_parcels": n_parcels,
                "max_abs_reconstruction_error": error,
            })
    return pd.DataFrame(component_rows), pd.DataFrame(reconstruction_rows)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_class_decomposition(
    class_name: str, components: pd.DataFrame, path: Path,
) -> None:
    labels = {
        "trend": "Trend",
        "dynamics": "Structured dynamics",
        "residual": "Residual",
    }
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    class_frame = components[components["class_name"] == class_name]
    for axis, component in zip(axes, COMPONENT_ORDER):
        component_frame = class_frame[class_frame["component"] == component]
        for domain in DOMAIN_DATASETS:
            curve = component_frame[component_frame["domain"] == domain]
            if curve.empty:
                continue
            curve = curve.sort_values("day_of_year", kind="stable")
            axis.plot(
                curve["day_of_year"], curve["value"], marker="o",
                markersize=3.5, linewidth=1.4, color=DOMAIN_COLORS[domain],
                label=domain,
            )
        axis.set_xlim(1, 366)
        axis.set_ylabel(labels[component])
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Day of year (DOY)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, legend_labels, loc="upper center",
            bbox_to_anchor=(0.5, 0.88), ncol=len(handles),
        )
    fig.suptitle(
        f"{class_name}: NDVI decomposition diagnostic\n"
        "T = slow smoothing | D = fast smoothing - slow smoothing | "
        "R = original NDVI - fast smoothing\n"
        f"tau_fast={TAU_FAST:.2f} (~{FAST_SCALE_DAYS:.2f} d), "
        f"tau_slow={TAU_SLOW:.2f} (~{SLOW_SCALE_DAYS:.1f} d)"
    )
    _save(fig, path)


def run_ndvi_decomposition(
    ndvi_csv: Path | str, output_dir: Path | str,
) -> dict[str, object]:
    """Read the compact NDVI table and emit component tables and class figures."""

    ndvi_csv, output_dir = Path(ndvi_csv), Path(output_dir)
    if not ndvi_csv.is_file():
        raise FileNotFoundError(f"NDVI CSV does not exist: {ndvi_csv}")
    components, reconstruction = decompose_ndvi_frame(pd.read_csv(ndvi_csv))
    table_dir = output_dir / "tables" / "ndvi_decomposition"
    figure_dir = output_dir / "figures" / "raw_timeseries" / "ndvi_decomposition"
    table_dir.mkdir(parents=True, exist_ok=True)
    components.to_csv(table_dir / "ndvi_decomposition_long.csv", index=False)
    reconstruction.to_csv(table_dir / "reconstruction_check.csv", index=False)
    classes = sorted(components["class_name"].unique())
    for class_name in classes:
        safe_name = str(class_name).replace("/", "_")
        _plot_class_decomposition(class_name, components, figure_dir / f"{safe_name}.png")
    return {
        "components": components,
        "reconstruction": reconstruction,
        "classes": classes,
        "max_reconstruction_error": float(
            reconstruction["max_abs_reconstruction_error"].max()
        ),
    }
