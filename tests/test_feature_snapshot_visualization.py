from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.visualize_structure_feature_snapshots import visualize_snapshots


def test_visualizer_cli_help_runs_from_repository_root() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "visualize_structure_feature_snapshots.py"),
            "--help",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "snapshot-dir" in result.stdout


def _write_snapshot(root: Path, epoch: int, shift: float) -> None:
    values = np.arange(24, dtype=np.float32).reshape(8, 3) + shift
    aligned = np.arange(36, dtype=np.float32).reshape(4, 3, 3) + shift
    np.savez_compressed(
        root / f"epoch_{epoch:04d}.npz",
        epoch=np.asarray(epoch),
        pse_values=values,
        pse_times=np.asarray([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.float32),
        pse_offsets=np.asarray([0, 2, 4, 6, 8], dtype=np.int64),
        aligned_shape=aligned,
        shape_grid=np.linspace(0, 1, 3, dtype=np.float32),
        shape_valid=np.asarray([True, False, True, True]),
    )


def test_visualizer_uses_snapshots_only_and_writes_per_epoch_domain_plots(
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "snapshots"
    output_dir = tmp_path / "plots"
    snapshot_dir.mkdir()
    np.savez_compressed(
        snapshot_dir / "selected_samples.npz",
        sample_index=np.asarray([10, 11, 20, 21], dtype=np.int64),
        domain=np.asarray([0, 0, 1, 1], dtype=np.uint8),
        label=np.asarray([0, 1, 0, 1], dtype=np.int16),
        source_domain=np.asarray("source"),
        target_domain=np.asarray("target"),
        selection_seed=np.asarray(1, dtype=np.int64),
        samples_per_class=np.asarray(1, dtype=np.int64),
    )
    _write_snapshot(snapshot_dir, 25, 0.0)
    _write_snapshot(snapshot_dir, 50, 100.0)

    summary = visualize_snapshots(snapshot_dir, output_dir)

    assert summary["epochs"] == [25, 50]
    assert summary["pse_pca_fit_count"] == 1
    assert summary["shape_pca_fit_count"] == 1
    assert summary["shape_curves_per_epoch"] == 3
    expected = {
        output_dir / f"epoch_{epoch:04d}_{domain}_{family}.png"
        for epoch in (25, 50)
        for domain in ("source", "target")
        for family in ("pse", "aligned_shape")
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)
