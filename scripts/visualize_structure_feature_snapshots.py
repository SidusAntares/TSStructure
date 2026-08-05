#!/usr/bin/env python3
"""Render compact PSE and aligned-structure snapshot curves without datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from methods.structure_da.feature_snapshots import load_selected_samples


def _load_epochs(snapshot_dir: Path) -> list[dict[str, np.ndarray]]:
    snapshots: list[dict[str, np.ndarray]] = []
    for path in sorted(snapshot_dir.glob("epoch_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            required = {
                "epoch", "pse_values", "pse_times", "pse_offsets",
                "aligned_shape", "shape_grid", "shape_valid",
            }
            if set(data.files) != required:
                raise ValueError(f"snapshot fields are invalid: {path}")
            snapshots.append({name: data[name].copy() for name in data.files})
    if not snapshots:
        raise ValueError("no epoch snapshots found")
    return snapshots


def _fit_joint_pca(arrays: list[np.ndarray]) -> PCA:
    values = np.concatenate(arrays, axis=0).astype(np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA requires at least two feature rows")
    return PCA(n_components=min(2, values.shape[0], values.shape[1])).fit(values)


def _plot_curves(
    path: Path,
    x_values: list[np.ndarray],
    feature_values: list[np.ndarray],
    labels: np.ndarray,
    parcels: np.ndarray,
    pca: PCA,
) -> None:
    figure, axes = plt.subplots(pca.n_components_, 1, squeeze=False, figsize=(8, 5))
    for x, features, label, parcel in zip(x_values, feature_values, labels, parcels):
        coordinates = pca.transform(features.astype(np.float32))
        for component in range(pca.n_components_):
            axes[component, 0].plot(
                x,
                coordinates[:, component],
                linewidth=1,
                alpha=0.75,
                label=f"parcel={int(parcel)}, class={int(label)}",
            )
            axes[component, 0].set_ylabel(f"PC{component + 1}")
    axes[-1, 0].set_xlabel("physical time / canonical grid")
    if feature_values:
        axes[0, 0].legend(fontsize=6, ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def visualize_snapshots(snapshot_dir: Path, output_dir: Path) -> dict[str, object]:
    """Fit one cross-epoch PCA per family and render every retained sample."""

    snapshot_dir, output_dir = Path(snapshot_dir), Path(output_dir)
    selected = load_selected_samples(snapshot_dir / "selected_samples.npz")
    snapshots = _load_epochs(snapshot_dir)
    count = len(selected.sample_index)
    for snapshot in snapshots:
        if snapshot["pse_offsets"].shape != (count + 1,) or snapshot["aligned_shape"].shape[0] != count:
            raise ValueError("snapshot sample count does not match selected samples")
    pse_pca = _fit_joint_pca([snapshot["pse_values"] for snapshot in snapshots])
    shape_rows = [
        snapshot["aligned_shape"][snapshot["shape_valid"].astype(bool)].reshape(-1, snapshot["aligned_shape"].shape[-1])
        for snapshot in snapshots
    ]
    shape_pca = _fit_joint_pca([rows for rows in shape_rows if len(rows)])

    for snapshot in snapshots:
        epoch = int(snapshot["epoch"].item())
        offsets = snapshot["pse_offsets"]
        valid = snapshot["shape_valid"].astype(bool)
        for domain_id, domain_name in ((0, "source"), (1, "target")):
            indices = np.flatnonzero(selected.domain == domain_id)
            pse_features = [snapshot["pse_values"][offsets[index]:offsets[index + 1]] for index in indices]
            pse_times = [snapshot["pse_times"][offsets[index]:offsets[index + 1]] for index in indices]
            _plot_curves(
                output_dir / f"epoch_{epoch:04d}_{domain_name}_pse.png",
                pse_times,
                pse_features,
                selected.label[indices],
                selected.sample_index[indices],
                pse_pca,
            )
            shape_indices = indices[valid[indices]]
            shape_features = [snapshot["aligned_shape"][index] for index in shape_indices]
            shape_times = [snapshot["shape_grid"] for _ in shape_indices]
            _plot_curves(
                output_dir / f"epoch_{epoch:04d}_{domain_name}_aligned_shape.png",
                shape_times,
                shape_features,
                selected.label[shape_indices],
                selected.sample_index[shape_indices],
                shape_pca,
            )
    return {
        "epochs": [int(snapshot["epoch"].item()) for snapshot in snapshots],
        "pse_pca_fit_count": 1,
        "shape_pca_fit_count": 1,
        "shape_curves_per_epoch": int(snapshots[0]["shape_valid"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize compact PSE and aligned_structure_srvf snapshots"
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = visualize_snapshots(args.snapshot_dir, args.output_dir)
    print(
        f"SNAPSHOT_VISUALIZATION|epochs={len(summary['epochs'])}"
        f"|output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
