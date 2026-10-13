import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest


def test_response96_block_layout_is_decoded_without_reordering():
    from analysis.structure_g_response_organization_audit import decode_response96

    response = np.arange(96, dtype=np.float64)[None, :]
    decoded = decode_response96(response)

    assert decoded.shape == (1, 16, 6)
    np.testing.assert_array_equal(decoded[0, :, 0], np.arange(0, 16))
    np.testing.assert_array_equal(decoded[0, :, 1], np.arange(16, 32))
    np.testing.assert_array_equal(decoded[0, :, 2], np.arange(32, 48))
    np.testing.assert_array_equal(decoded[0, :, 5], np.arange(80, 96))


def test_circular_statistics_respect_wraparound_and_reliability():
    from analysis.structure_g_response_organization_audit import circular_statistics

    angles = np.deg2rad(np.array([179.0, -179.0]))
    result = circular_statistics(np.cos(angles), np.sin(angles))

    assert abs(abs(result["direction_radians"]) - np.pi) < 0.03
    assert result["resultant_length"] > 0.99
    assert result["support"] == 2


def test_class_anchor_rows_mark_minority_support_as_not_estimable():
    from analysis.structure_g_response_organization_audit import class_anchor_distribution

    events = np.zeros((4, 16, 6), dtype=np.float64)
    labels = np.array([0, 0, 0, 1])
    rows = class_anchor_distribution(
        events, labels, ["major", "minor"], scope="SRC_AT1", split="source_train",
        min_support=2,
    )

    minor = [row for row in rows if row["class_id"] == 1]
    assert minor
    assert {row["estimable"] for row in minor} == {False}
    assert {row["support"] for row in minor} == {1}


def test_relation_selection_uses_source_train_only():
    from analysis.structure_g_response_organization_audit import select_cross_anchor_relations

    rng = np.random.default_rng(4)
    train = rng.normal(size=(40, 16, 6))
    labels = np.repeat([0, 1], 20)
    train[labels == 1, 0, 0] += 3
    train[labels == 1, 1, 0] += 3

    first = select_cross_anchor_relations(train, labels, max_relations=4)
    unrelated_val = rng.normal(size=(200, 16, 6))
    unrelated_labels = rng.integers(0, 2, size=200)
    second = select_cross_anchor_relations(train, labels, max_relations=4)

    assert first == second
    assert unrelated_val.shape[0] != train.shape[0]
    assert unrelated_labels.shape[0] == unrelated_val.shape[0]


def test_shrinkage_relation_is_finite_when_samples_are_fewer_than_dimensions():
    from analysis.structure_g_response_organization_audit import shrinkage_dependence

    rng = np.random.default_rng(9)
    values = rng.normal(size=(3, 16, 2))
    matrix = shrinkage_dependence(values)

    assert matrix.shape == (16, 16)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(np.diag(matrix), 1.0)


def _writable_test_dir():
    path = Path.cwd() / f".codex_response_org_test_{uuid.uuid4().hex}"
    path.mkdir()
    return path


def test_cache_protocol_rejects_target_test_and_target_train_labels():
    from analysis.structure_g_response_organization_audit import validate_cache_protocol

    tmp_path = _writable_test_dir()
    manifest = {
        "checkpoint_sha256": "abc",
        "target_test_used": True,
        "target_train_labels_cached": False,
        "anchor_scope": "G:AT1_DK1:abc",
    }
    path = tmp_path / "checkpoint_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Target Test"):
        validate_cache_protocol(tmp_path, expected_sha="abc")

    manifest["target_test_used"] = False
    manifest["target_train_labels_cached"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="Target Train"):
            validate_cache_protocol(tmp_path, expected_sha="abc")
    finally:
        shutil.rmtree(tmp_path)


def test_cache_sha_and_anchor_scope_are_strict():
    from analysis.structure_g_response_organization_audit import (
        assert_anchor_scope_isolated,
        validate_cache_protocol,
    )

    tmp_path = _writable_test_dir()
    manifest = {
        "checkpoint_sha256": "sha-one",
        "target_test_used": False,
        "target_train_labels_cached": False,
        "anchor_scope": "G:AT1_DK1:sha-one",
    }
    (tmp_path / "checkpoint_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    try:
        with pytest.raises(ValueError, match="SHA"):
            validate_cache_protocol(tmp_path, expected_sha="sha-two")
        with pytest.raises(ValueError, match="Anchor IDs"):
            assert_anchor_scope_isolated("SRC:AT1:sha-one", "G:AT1_DK1:sha-two")
    finally:
        shutil.rmtree(tmp_path)


def test_dry_run_declares_eight_unique_scopes_without_training():
    from scripts.run_structure_g_response_organization_seed1 import planned_scopes

    scopes = planned_scopes()
    assert len(scopes) == 8
    assert len({scope.identity for scope in scopes}) == 8
    assert {scope.stage for scope in scopes} == {"source", "uda_final"}
    assert all("train.py" not in " ".join(scope.command_hint) for scope in scopes)
