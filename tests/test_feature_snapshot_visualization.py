from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

import scripts.visualize_structure_feature_snapshots as visualization
from scripts.visualize_structure_feature_snapshots import visualize_snapshots


EPOCHS = (25, 50, 75, 100)


def _write_selected(root: Path, *, compact: bool) -> None:
    np.savez_compressed(
        root / "selected_samples.npz",
        sample_index=np.asarray([10, 11, 12, 13, 20, 21, 22, 23], dtype=np.int64),
        domain=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8),
        label=np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int16),
        source_domain=np.asarray("source"),
        target_domain=np.asarray("target"),
        selection_seed=np.asarray(1, dtype=np.int64),
        samples_per_class=np.asarray(2, dtype=np.int64),
    )
    manifest = {
        "schema_version": 2 if compact else 1,
        "class_names": ["corn", "wheat"] if compact else None,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _legacy_snapshot(root: Path, epoch: int) -> None:
    rng = np.random.default_rng(epoch)
    np.savez_compressed(
        root / f"epoch_{epoch:04d}.npz",
        epoch=np.asarray(epoch),
        pse_values=rng.normal(size=(32, 6)).astype(np.float16),
        pse_times=np.tile(np.arange(4, dtype=np.float32), 8),
        pse_offsets=np.arange(0, 33, 4, dtype=np.int64),
        aligned_shape=rng.normal(size=(8, 5, 6)).astype(np.float16),
        shape_grid=np.linspace(0, 1, 5, dtype=np.float32),
        shape_valid=np.asarray([True, True, True, True, True, False, True, True]),
    )


def _compact_snapshot(root: Path, epoch: int) -> None:
    rng = np.random.default_rng(epoch)
    np.savez_compressed(
        root / f"epoch_{epoch:04d}.npz",
        epoch=np.asarray(epoch, dtype=np.int64),
        projection_fitted_epoch=np.asarray(25, dtype=np.int64),
        schema_version=np.asarray(2, dtype=np.int64),
        pse_pc_values=rng.normal(size=(32, 2)).astype(np.float16),
        pse_times=np.tile(np.arange(4, dtype=np.float32), 8),
        pse_offsets=np.arange(0, 33, 4, dtype=np.int64),
        aligned_shape_pc=rng.normal(size=(8, 5, 2)).astype(np.float16),
        shape_grid=np.linspace(0, 1, 5, dtype=np.float32),
        shape_valid=np.asarray([True, True, True, True, True, False, True, True]),
    )


def _write_basis(root: Path) -> None:
    np.savez_compressed(
        root / "projection_basis.npz",
        schema_version=np.asarray(2),
        fitted_epoch=np.asarray(25),
        num_components=np.asarray(2),
        pse_mean=np.zeros(6, dtype=np.float32),
        pse_components=np.eye(2, 6, dtype=np.float32),
        pse_explained_variance=np.ones(2, dtype=np.float32),
        pse_explained_variance_ratio=np.full(2, 0.5, dtype=np.float32),
        shape_mean=np.zeros(6, dtype=np.float32),
        shape_components=np.eye(2, 6, dtype=np.float32),
        shape_explained_variance=np.ones(2, dtype=np.float32),
        shape_explained_variance_ratio=np.full(2, 0.5, dtype=np.float32),
    )


def test_visualizer_cli_help_exposes_readability_controls() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repository_root / "scripts" / "visualize_structure_feature_snapshots.py"), "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for option in (
        "--snapshot-dir", "--output-dir", "--display-samples-per-class",
        "--separation-samples-per-class", "--components",
        "--robust-lower-percentile", "--robust-upper-percentile", "--dpi",
    ):
        assert option in result.stdout


def test_legacy_schema_fits_one_joint_basis_and_never_modifies_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_dir = tmp_path / "legacy"
    output_dir = tmp_path / "plots"
    snapshot_dir.mkdir()
    _write_selected(snapshot_dir, compact=False)
    for epoch in EPOCHS:
        _legacy_snapshot(snapshot_dir, epoch)
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in snapshot_dir.iterdir()
    }
    calls = 0
    original = visualization.fit_deterministic_pca

    def counted(values: np.ndarray, num_components: int = 2):
        nonlocal calls
        calls += 1
        return original(values, num_components)

    monkeypatch.setattr(visualization, "fit_deterministic_pca", counted)
    summary = visualize_snapshots(
        snapshot_dir, output_dir,
        display_samples_per_class=1,
        separation_samples_per_class=1,
        components=(1,),
        dpi=40,
    )
    assert summary["schema"] == "legacy_full_features"
    assert summary["pca_mode"] == "joint_all_epochs_in_memory"
    assert calls == 2
    assert summary["epochs"] == list(EPOCHS)
    assert summary["max_evolution_curves_per_axis"] <= 1
    assert summary["max_separation_curves_per_class"] <= 1
    assert summary["evolution_grid"] == (2, 4)
    assert summary["figures"] == 40
    assert (output_dir / "class_evolution" / "pse_class_00_pc1_full.png").is_file()
    assert (output_dir / "class_separation" / "shape_target_epoch_0100_pc1_robust.png").is_file()
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in snapshot_dir.iterdir()
    }
    assert after == before
    assert not plt.get_fignums()


def test_compact_schema_uses_stored_projection_and_writes_class_only_legends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_dir = tmp_path / "compact"
    output_dir = tmp_path / "plots"
    snapshot_dir.mkdir()
    _write_selected(snapshot_dir, compact=True)
    _write_basis(snapshot_dir)
    for epoch in EPOCHS:
        _compact_snapshot(snapshot_dir, epoch)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("compact schema must not refit PCA")

    monkeypatch.setattr(visualization, "fit_deterministic_pca", forbidden_fit)
    summary = visualize_snapshots(
        snapshot_dir, output_dir,
        display_samples_per_class=1,
        separation_samples_per_class=1,
        components=(1, 2),
        dpi=40,
    )
    assert summary["schema"] == "compact_fixed_pca"
    assert summary["pca_mode"] == "stored_fixed_epoch25_basis"
    assert summary["legend_labels"] == ["corn", "wheat"]
    assert all("parcel" not in label for label in summary["legend_labels"])
    assert summary["invalid_shape_skipped"] == len(EPOCHS)
    assert (output_dir / "pse_pc_outliers.tsv").is_file()
    assert (output_dir / "shape_pc_outliers.tsv").is_file()
    assert not plt.get_fignums()


def test_compact_schema_rejects_missing_projection_component(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "compact"
    snapshot_dir.mkdir()
    _write_selected(snapshot_dir, compact=True)
    _write_basis(snapshot_dir)
    for epoch in EPOCHS:
        _compact_snapshot(snapshot_dir, epoch)
    with pytest.raises(ValueError, match="components 1 and 2"):
        visualize_snapshots(snapshot_dir, tmp_path / "out", components=(3,))


def test_class_colors_are_stable_when_other_classes_are_absent() -> None:
    assert visualization._class_colors([0, 2])[2] == visualization._class_colors([1, 2])[2]


def test_compact_schema_requires_epoch25_projection_basis(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "compact"
    snapshot_dir.mkdir()
    _write_selected(snapshot_dir, compact=True)
    _write_basis(snapshot_dir)
    with np.load(snapshot_dir / "projection_basis.npz", allow_pickle=False) as data:
        fields = {name: data[name].copy() for name in data.files}
    fields["fitted_epoch"] = np.asarray(50)
    np.savez_compressed(snapshot_dir / "projection_basis.npz", **fields)
    for epoch in EPOCHS:
        _compact_snapshot(snapshot_dir, epoch)

    with pytest.raises(ValueError, match="epoch 25"):
        visualization._load_projected_epochs(snapshot_dir, count=8)
