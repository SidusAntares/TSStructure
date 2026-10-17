import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from analysis.structure_g_anchor_stage1_audit import (
    STAGE1_PROTOCOL_VERSION,
    anchor_basis_assignment,
    build_cache_signature,
    evaluate_view_integrity,
    identity_control_summary,
    pair_compact_views,
    phase_stability_summary,
    strength_support_summary,
    validate_checkpoint_pair_metadata,
    validate_split_isolation,
)
from analysis.anchor_matching_calibration_audit import DeterministicViewDataset, circular_window_iou
from scripts.run_structure_g_anchor_stage1_seed1 import (
    artifact_complete,
    planned_jobs,
    run_scheduler,
)


def _view(parcels=(10, 11), shift=0.0):
    pixels = np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape(2, 3, 2, 4)
    pixels = pixels + shift
    return {
        "domain": np.asarray(["FR1", "FR1"]),
        "parcel_index": np.asarray(parcels),
        "label": np.asarray([0, 1]),
        "input_hash": np.asarray([f"h{shift}-0", f"h{shift}-1"]),
        "pixels_summary": pixels.mean((1, 2)),
        "positions": np.asarray([[1, 3, 5], [2, 4, 6]]) + int(shift),
        "valid_mask": np.asarray([[1, 1, 1], [1, 1, 0]], dtype=np.uint8),
        "similarity": np.zeros((2, 8, 16), dtype=np.float32) + shift,
        "response": np.zeros((2, 96), dtype=np.float32) + shift,
        "token_signature": np.zeros((2, 8, 4), dtype=np.float32) + shift,
    }


def test_actual_sampled_input_controls_view_validity():
    left = _view(shift=0)
    same = copy.deepcopy(left)
    changed = _view(shift=1)
    invalid = evaluate_view_integrity(left, same)
    valid = evaluate_view_integrity(left, changed)
    assert not invalid[0]["valid_view"]
    assert invalid[0]["invalid_reason"] == "identical_sampled_input"
    assert valid[0]["valid_view"]
    assert valid[0]["position_change_fraction"] > 0
    assert valid[0]["pixel_summary_relative_l2"] > 0


def test_deterministic_views_are_reproducible_but_independent():
    class RandomDataset:
        def __len__(self): return 2
        def __getitem__(self, index): return {"value": np.random.uniform(size=4)}
        def get_parcel_indices(self): return np.asarray([20, 21])
        def get_shapes(self): return [(30, 2, 100), (30, 2, 100)]
        def get_labels(self): return np.asarray([0, 1])

    dataset = RandomDataset()
    first = DeterministicViewDataset(dataset, "FR1", 1, 0)
    repeat = DeterministicViewDataset(dataset, "FR1", 1, 0)
    second = DeterministicViewDataset(dataset, "FR1", 1, 1)
    np.testing.assert_allclose(first[0]["value"], repeat[0]["value"])
    assert not np.allclose(first[0]["value"], second[0]["value"])


def test_pairing_is_by_domain_and_parcel_not_loader_order():
    left = _view((10, 11), 0)
    right = _view((11, 10), 1)
    right = {key: value[::-1] if np.asarray(value).ndim and len(value) == 2 else value
             for key, value in right.items()}
    paired_left, paired_right = pair_compact_views(left, right)
    np.testing.assert_array_equal(paired_left["parcel_index"], paired_right["parcel_index"])
    with pytest.raises(ValueError, match="identical parcels"):
        pair_compact_views(left, _view((10, 12), 1))


def test_circular_window_overlap_uses_real_wrapped_indices():
    assert circular_window_iou(56, 0, scale=24, grid_points=64) == pytest.approx(0.5)
    assert circular_window_iou(0, 32, scale=24, grid_points=64) == 0.0


def test_source_train_val_and_checkpoint_identity_are_strict():
    assert validate_split_isolation([1, 2], [3, 4])
    with pytest.raises(ValueError, match="leakage"):
        validate_split_isolation([1, 2], [2, 3])
    source = SimpleNamespace(
        source="france/30TXT/2017", classes=["a", "b"],
        shape_representation="phase_moment", phase_response_mode="softmax",
    )
    uda = SimpleNamespace(
        source="france/30TXT/2017", classes=["a", "b"],
        shape_representation="phase_moment", phase_response_mode="softmax",
    )
    assert validate_checkpoint_pair_metadata(source, uda, 19)
    with pytest.raises(ValueError, match="epoch 19"):
        validate_checkpoint_pair_metadata(source, uda, 18)


def test_identity_repeatability_is_compared_with_time_and_same_class_controls():
    paired = np.asarray([1, 1, 0, 1], dtype=float)
    fixed_position = np.asarray([1, 1, 1, 1], dtype=float)
    same_class = np.asarray([0, 1, 0, 0], dtype=float)
    row = identity_control_summary(paired, fixed_position, same_class)
    assert row["paired_repeatability"] == pytest.approx(.75)
    assert row["increment_over_fixed_position"] == pytest.approx(-.25)
    assert row["increment_over_same_class"] == pytest.approx(.5)
    assert not row["identity_beyond_time_supported"]


def test_constant_low_similarity_is_not_called_absolute_matching_support():
    q = np.full(40, -.8)
    repeated = np.ones(40)
    row = strength_support_summary(q, repeated, baseline_probability=.5)
    assert row["repeatability"] == 1.0
    assert row["absolute_matching_support"] is False
    assert row["failure_reason"] == "uniformly_low_absolute_similarity"


def test_k1_and_k2_low_magnitude_are_gated_independently():
    k1_a = np.asarray([[1e-5, 0.0], [2e-5, 0.0]])
    k1_b = np.asarray([[0.0, 1e-5], [0.0, 2e-5]])
    k2_a = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    k2_b = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    row = phase_stability_summary(k1_a, k1_b, k2_a, k2_b, .01, .5)
    assert row["k1_direction_sample_count"] == 0
    assert np.isnan(row["k1_direction_error_mean"])
    assert row["k2_direction_sample_count"] == 2
    assert np.isfinite(row["k2_direction_error_mean"])


def test_anchor_bases_use_hungarian_one_to_one_assignment():
    source = np.eye(3, dtype=np.float64)
    uda = source[[2, 0, 1]]
    result = anchor_basis_assignment(source, uda)
    assert result["source_anchor"].tolist() == [0, 1, 2]
    assert result["uda_anchor"].tolist() == [1, 2, 0]
    np.testing.assert_allclose(result["cosine"], 1.0)
    assert result["interpretation_allowed"]


def test_cache_signature_changes_with_protocol_sampling_and_checkpoint():
    base = dict(checkpoint_sha="abc", split_fingerprint="split", seed=1,
                sampling={"seq_length": 30, "num_pixels": 64})
    first = build_cache_signature(**base)
    assert first != build_cache_signature(**{**base, "checkpoint_sha": "def"})
    assert first != build_cache_signature(**{**base, "sampling": {"seq_length": 20}})
    assert first != build_cache_signature(**{**base, "protocol_version": "old"})


def test_frozen_audit_helpers_do_not_mutate_model_parameters():
    model = torch.nn.Linear(3, 2)
    before = [value.detach().clone() for value in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        model(torch.ones(2, 3))
    assert all(parameter.grad is None for parameter in model.parameters())
    for old, new in zip(before, model.parameters()):
        torch.testing.assert_close(old, new)


def _args(tmp_path, mode="dry-run"):
    return argparse.Namespace(
        mode=mode, data_root=tmp_path / "data",
        source_root=tmp_path / "source", uda_root=tmp_path / "uda",
        p0r_root=tmp_path / "p0r", output_root=tmp_path / "out",
        gpus=["0", "1", "2", "3"], batch_size=8, pixel_budget=64,
        seed=1, python="python", checkpoint_at1="", checkpoint_fr1="",
        checkpoint_fr2="", checkpoint_dk1="", uda_at1_dk1="",
        uda_fr1_fr2="", uda_fr2_dk1="", uda_dk1_at1="",
    )


def test_scheduler_plans_four_source_and_four_uda_checkpoints(capsys):
    args = _args(Path("unused_stage1_dry_run"))
    jobs = planned_jobs(args)
    assert [(job.source_alias, job.task) for job in jobs] == [
        ("AT1", "AT1_DK1"), ("FR1", "FR1_FR2"),
        ("FR2", "FR2_DK1"), ("DK1", "DK1_AT1"),
    ]
    run_scheduler(args)
    output = capsys.readouterr().out
    assert output.count("ANCHOR_STAGE1_PLAN|") == 4
    assert "git " not in output.lower()


def test_resume_requires_complete_manifest_and_all_outputs():
    fixture = Path("tests/stage1_artifact_fixture")
    shutil.rmtree(fixture, ignore_errors=True)
    args = _args(fixture, "resume")
    job = planned_jobs(args)[0]
    folder = Path(args.output_root) / job.source_alias
    folder.mkdir(parents=True)
    source_sha, uda_sha, signature = "source", "uda", "signature"
    (folder / "checkpoint_manifest.json").write_text(json.dumps({
        "status": "complete", "protocol_version": STAGE1_PROTOCOL_VERSION,
        "source_checkpoint_sha256": source_sha, "uda_checkpoint_sha256": uda_sha,
        "settings_fingerprint": signature, "target_test_used": False,
        "training_performed": False,
    }), encoding="utf-8")
    assert not artifact_complete(folder, source_sha, uda_sha, signature)
    shutil.rmtree(fixture, ignore_errors=True)
