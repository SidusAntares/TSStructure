from pathlib import Path

import pytest
import torch


ANCHOR_KEY = "structure_branch.shapelet_dictionary.anchors"


def _checkpoint(path, anchors):
    torch.save({"state_dict": {ANCHOR_KEY: anchors.clone()}}, path)


def test_identical_anchor_dictionary_has_zero_drift():
    from analysis.anchor_drift_audit import compare_anchor_tensors

    anchors = torch.randn(16, 8)
    summary, rows = compare_anchor_tensors(anchors, anchors.clone())
    assert len(rows) == 16
    assert summary["mean_rowwise_cosine"] == pytest.approx(1.)
    assert summary["min_rowwise_cosine"] == pytest.approx(1.)
    assert summary["max_rowwise_cosine"] == pytest.approx(1.)
    assert summary["mean_relative_l2_drift"] == pytest.approx(0.)
    assert summary["gram_frobenius_drift"] == pytest.approx(0.)
    assert all(row["l2_distance"] == pytest.approx(0.) for row in rows)


def test_anchor_checkpoint_loading_is_read_only():
    from analysis.anchor_drift_audit import load_anchor_tensor

    checkpoint = Path("tests/_anchor_drift_read_only_test.pt")
    try:
        anchors = torch.randn(16, 8)
        _checkpoint(checkpoint, anchors)
        before = checkpoint.read_bytes()
        loaded = load_anchor_tensor(checkpoint)
        after = checkpoint.read_bytes()
        torch.testing.assert_close(loaded, anchors)
        assert before == after
    finally:
        checkpoint.unlink(missing_ok=True)


def test_anchor_audit_launcher_uses_frozen_v2clean_root():
    source = Path("scripts/run_anchor_drift_audit_v2clean_seed1.sh").read_text()
    assert "outputs/structure_proto_v2clean_4tasks_seed1" in source
    assert "outputs/anchor_drift_v2clean_seed1" in source
    assert "analysis/anchor_drift_audit.py" in source
    assert "train.py" not in source

