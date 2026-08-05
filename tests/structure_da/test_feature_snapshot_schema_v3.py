from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import methods.structure_da.feature_snapshots as snapshots


def test_deterministic_pixel_indices_match_training_sampling_semantics() -> None:
    choose = snapshots.deterministic_pixel_indices
    enough = choose(10, 4, pixel_seed=7, domain="source", parcel_index=11)
    repeated = choose(2, 4, pixel_seed=7, domain="source", parcel_index=11)

    assert enough.shape == (4,)
    assert len(np.unique(enough)) == 4
    np.testing.assert_array_equal(repeated, np.asarray([0, 1, 0, 0]))
    np.testing.assert_array_equal(
        enough,
        choose(10, 4, pixel_seed=7, domain="source", parcel_index=11),
    )
    assert not np.array_equal(
        enough,
        choose(10, 4, pixel_seed=7, domain="target", parcel_index=11),
    )
    assert not np.array_equal(
        enough,
        choose(10, 4, pixel_seed=7, domain="source", parcel_index=12),
    )


def test_selected_pixel_application_fixes_width_and_masks_padding() -> None:
    sample = {
        "pixels": np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2),
        "valid_pixels": np.ones((2, 2), dtype=np.float32),
        "positions": np.asarray([1, 2], dtype=np.int64),
        "label": 0,
    }
    transformed = snapshots.prepare_snapshot_sample(
        sample, np.asarray([0, 1, 0, 0]), num_pixels=4
    )

    assert transformed["pixels"].shape == (2, 3, 4)
    assert transformed["valid_pixels"].shape == (2, 4)
    torch.testing.assert_close(
        transformed["valid_pixels"],
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]).repeat(2, 1),
    )


def test_weighted_pca_gives_each_parcel_equal_total_weight() -> None:
    short = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    long = np.repeat(np.asarray([[0.0, 10.0]]), 8, axis=0)
    values = np.concatenate((short, long), axis=0)
    weights = np.concatenate((np.full(2, 0.5), np.full(8, 0.125)))

    fit = snapshots.fit_deterministic_pca(values, num_components=2, weights=weights)

    np.testing.assert_allclose(fit.mean, np.asarray([0.5, 5.0]), atol=1e-6)
    assert fit.components.shape == (2, 2)
    for component in fit.components:
        assert component[np.argmax(np.abs(component))] >= 0


def test_snapshot_status_recomputes_failures_and_allows_successful_retry(
    tmp_path: Path,
) -> None:
    snapshots.update_snapshot_status(
        tmp_path,
        epoch=25,
        epoch_status={"status": "FAILED", "error_type": "RuntimeError", "error": "x"},
    )
    failed = json.loads((tmp_path / "snapshot_status.json").read_text(encoding="utf-8"))
    assert failed["has_failures"] is True

    snapshots.update_snapshot_status(
        tmp_path,
        epoch=25,
        epoch_status={"status": "SUCCESS", "batch_size": 8, "path": "epoch_0025.npz"},
    )
    retried = json.loads((tmp_path / "snapshot_status.json").read_text(encoding="utf-8"))
    assert retried["epochs"]["25"]["status"] == "SUCCESS"
    assert retried["has_failures"] is False
    assert not (tmp_path / "SNAPSHOT_FAILED").exists()


def test_phase_status_and_warp_are_normalized_for_storage() -> None:
    grid = np.linspace(0, 1, 4, dtype=np.float32)
    statuses = np.asarray([0, 1, 2], dtype=np.uint8)
    warp = np.asarray(
        [grid + 0.1, grid + 0.1, [0.0, 0.2, 0.7, 1.0]], dtype=np.float32
    )

    normalized = snapshots.normalize_accepted_warp(statuses, warp, grid)

    np.testing.assert_array_equal(normalized[0], grid)
    np.testing.assert_array_equal(normalized[1], grid)
    np.testing.assert_allclose(normalized[2], warp[2])


def test_factory_does_not_construct_dataset_when_interval_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(feature_snapshot_interval=0)
    assert snapshots.create_feature_snapshot_manager(
        object(), config, {}, device=torch.device("cpu")
    ) is None
