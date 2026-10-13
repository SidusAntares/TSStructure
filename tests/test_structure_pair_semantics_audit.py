import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.modules.setdefault("ot", types.ModuleType("ot"))

import analysis.structure_pair_semantics_audit as audit
import scripts.run_structure_pair_semantics_seed1 as scheduler


def test_all_unordered_pairs_and_insufficient_support_are_explicit():
    rows = audit.enumerate_class_pairs(np.array([0, 0, 0, 1, 1, 2]), ["a", "b", "c"], min_support=2)
    assert [(r["y_id"], r["c_id"]) for r in rows] == [(0, 1), (0, 2), (1, 2)]
    assert [r["status"] for r in rows] == ["eligible", "insufficient_support", "insufficient_support"]


def test_signed_pair_score_is_antisymmetric_and_group_indices_are_exact():
    candidate = {
        "pair_y": 0, "pair_c": 1, "anchor_id": 0,
        "effect": [2., 0., 0., 0., 0., 0.],
        "mean_y": [1., 0., 0., 0., 0., 0.],
        "mean_c": [-1., 0., 0., 0., 0., 0.],
        "pooled_std": [1.] * 6,
    }
    events = np.array([[[.5, 0., 0., 0., 0., 0.]]])
    forward = audit.signed_pair_evidence(candidate, events, 0, 1, "SCP")
    reverse = audit.signed_pair_evidence(candidate, events, 1, 0, "SCP")
    assert forward == pytest.approx(-reverse)
    assert audit.FEATURE_GROUPS == {
        "SC": (0, 1), "SC_K1": (0, 1, 2, 3),
        "SC_K2": (0, 1, 4, 5), "SCP": (0, 1, 2, 3, 4, 5),
    }


def test_source_only_group_selection_ignores_target_metric():
    rows = [
        {"group": "SC", "source_val_quality": .70, "target_val_quality": .99},
        {"group": "SC_K1", "source_val_quality": .75, "target_val_quality": .10},
        {"group": "SC_K2", "source_val_quality": .72, "target_val_quality": .90},
        {"group": "SCP", "source_val_quality": .74, "target_val_quality": .95},
    ]
    assert audit.select_group_source_only(rows, uncertainty=.01) == ("SC_K1", False)
    assert audit.select_group_source_only(rows, uncertainty=.06) == ("SC_K1", True)


def test_stratified_bootstrap_binary_metric_is_finite_and_deterministic():
    labels = np.array([0, 0, 1, 1])
    score = np.array([.1, .2, .8, .9])
    first = audit._bootstrap_binary_metric(labels, score, audit.roc_auc_score, repeats=20, seed=7)
    second = audit._bootstrap_binary_metric(labels, score, audit.roc_auc_score, repeats=20, seed=7)
    assert first == second == pytest.approx((1., 1.))


def test_target_train_rows_never_include_truth():
    rows = audit.teacher_rank_rows(
        np.array([[3., 2., 1.]]), confidence_bounds=(.5, .8),
        margin_bounds=(.2, 1.5), entropy_bounds=(.3, 1.0),
        parcel_index=np.array([7]), domain=np.array(["target"]), labels=None,
    )
    assert "true_label" not in rows[0]
    with pytest.raises(ValueError, match="Target Train"):
        audit.assert_unlabeled_target_train({"label": np.array([1])})


def test_teacher_topk_observable_group_is_separate_from_offline_truth_rank():
    logits = np.array([[4., 3.9, 0.], [4., 1., 0.]])
    observable = audit.teacher_rank_rows(
        logits, (.5, .8), (.2, 1.5), (.3, 1.0), np.array([1, 2]), np.array(["d", "d"]), labels=None,
    )
    offline = audit.teacher_rank_rows(
        logits, (.5, .8), (.2, 1.5), (.3, 1.0), np.array([1, 2]), np.array(["d", "d"]), labels=np.array([1, 2]),
    )
    assert all("true_rank_group" not in row for row in observable)
    assert [row["competition_group"] for row in observable] == [row["competition_group"] for row in offline]
    assert [row["true_rank_group"] for row in offline] == ["true_top2", "true_top3"]


def test_incomplete_or_cyclic_pair_graph_abstains_instead_of_argmax():
    assert audit.resolve_pair_graph([0, 1, 2], {(0, 1): 1., (0, 2): 1.})["status"] == "abstain_missing"
    cyclic = {(0, 1): 1., (1, 2): 1., (0, 2): -1.}
    assert audit.resolve_pair_graph([0, 1, 2], cyclic)["status"] == "abstain_cycle"
    complete = {(0, 1): 1., (0, 2): 2., (1, 2): 1.}
    assert audit.resolve_pair_graph([0, 1, 2], complete) == {"status": "winner", "class_id": 0}


def test_checkpoint_scopes_never_compare_anchor_ids():
    left = audit.CheckpointScope("E", "FR1_FR2", "sha-a")
    right = audit.CheckpointScope("G", "FR1_FR2", "sha-b")
    assert audit.anchor_ids_comparable(left, left)
    assert not audit.anchor_ids_comparable(left, right)


def _test_root(name):
    root = Path("tests") / ".pair_semantics_tmp" / name
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def test_cache_validation_checks_sha_classes_and_no_target_test():
    cache = _test_root("cache")
    (cache / "feature_cache_manifest.json").write_text(json.dumps({
        "target_test_used": False, "target_train_labels_cached": False,
        "files": {"teacher": {}},
        "applied_shifts": {"student": {s: 0 for s in audit.ALLOWED_SPLITS},
                           "teacher": {s: -15 for s in audit.ALLOWED_SPLITS}},
    }))
    (cache / "checkpoint_manifest.json").write_text(json.dumps({
        "checkpoint_sha256": "abc", "classes": ["a", "b"], "target_test_used": False,
    }))
    result = audit.validate_cache_identity(cache, "abc", ["a", "b"], require_feature_files=False)
    assert result["valid"] is True
    with pytest.raises(ValueError, match="SHA"):
        audit.validate_cache_identity(cache, "wrong", ["a", "b"], require_feature_files=False)
    packet = json.loads((cache / "feature_cache_manifest.json").read_text())
    packet["applied_shifts"]["teacher"]["target_val"] = 0
    (cache / "feature_cache_manifest.json").write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="shift"):
        audit.validate_cache_identity(cache, "abc", ["a", "b"], require_feature_files=False)


def test_scheduler_is_frozen_resumable_and_records_missing_paths(monkeypatch):
    tmp_path = _test_root("scheduler")
    args = scheduler.build_parser().parse_args(["dry-run", "--output-root", str(tmp_path / "out")])
    assert all("train.py" not in " ".join(scheduler.job_command(job, args)) for job in scheduler.build_job_graph([]))
    missing = scheduler.resolve_inventory([], [], scheduler.TASKS)
    assert len(missing["missing"]) == 8
    job = scheduler.Job("audit:E:FR1_FR2", "audit", "E", "FR1_FR2")
    output = tmp_path / "out" / "E_FR1_FR2"
    output.mkdir(parents=True)
    for name in audit.REQUIRED_OUTPUTS:
        (output / name).write_text("ok")
    assert scheduler.artifact_complete(job, args)
