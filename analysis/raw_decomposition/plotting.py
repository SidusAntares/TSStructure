"""Plot/save helpers for raw decomposition diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .methods import DecompositionResult


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def save_result_arrays(
    output_path: Path,
    *,
    raw_values: np.ndarray,
    raw_positions: np.ndarray,
    result: DecompositionResult,
) -> None:
    payload = {
        "raw_values": np.asarray(raw_values, dtype=np.float64),
        "raw_positions": np.asarray(raw_positions, dtype=np.float64),
    }
    for index, component in enumerate(result.components):
        safe = component.name.replace("/", "_").replace(" ", "_")
        payload[f"component_{index:02d}_{safe}_values"] = component.values
        payload[f"component_{index:02d}_{safe}_positions"] = component.positions
    np.savez_compressed(output_path, **payload)


def save_result_metadata(output_path: Path, result: DecompositionResult) -> None:
    payload = {
        "method": result.method,
        "components": [component.name for component in result.components],
        "metadata": _json_ready(dict(result.metadata)),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_result(
    output_path: Path,
    *,
    raw_values: np.ndarray,
    raw_positions: np.ndarray,
    result: DecompositionResult,
    title: str,
    dpi: int = 160,
) -> None:
    rows = 1 + len(result.components)
    figure, axes = plt.subplots(
        rows,
        1,
        figsize=(11.5, max(3.0, 2.15 * rows)),
        sharex=False,
        constrained_layout=True,
    )
    if rows == 1:
        axes = [axes]
    axes = np.asarray(axes, dtype=object).reshape(-1)

    axes[0].plot(raw_positions, raw_values, marker="o", linewidth=1.3, markersize=3.0)
    axes[0].set_title("raw observed series")
    axes[0].set_ylabel("value")
    axes[0].grid(alpha=0.18)

    for axis, component in zip(axes[1:], result.components):
        axis.plot(component.positions, component.values, marker="o", linewidth=1.2, markersize=2.6)
        axis.set_title(component.name)
        axis.set_ylabel("value")
        axis.grid(alpha=0.18)

    axes[-1].set_xlabel("TimeMatch acquisition position (actual observations; no interpolation)")
    figure.suptitle(title)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
