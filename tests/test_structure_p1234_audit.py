import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# POT is a training-only dependency; this frozen audit never calls it.
sys.modules.setdefault("ot", types.ModuleType("ot"))

import analysis.structure_p1234_audit as audit
import scripts.run_structure_p1234_seed1 as scheduler


def test_p1_groups_use_same_samples_and_source_val_selection_only():
    rows = [
        {"group": "SC", "source_val_auroc": .7, "target_val_auroc": .99, "sample_key_digest": "same", "candidate_digest": "c"},
        {"group": "SC_K1", "source_val_auroc": .8, "target_val_auroc": .1, "sample_key_digest": "same", "candidate_digest": "c"},
        {"group": "SC_K2", "source_val_auroc": .75, "target_val_auroc": .8, "sample_key_digest": "same", "candidate_digest": "c"},
        {"group": "SCP", "source_val_auroc": .79, "target_val_auroc": .9, "sample_key_digest": "same", "candidate_digest": "c"},
    ]
    assert audit.select_p1_group(rows) == "SC_K1"
    assert audit.validate_p1_comparability(rows)
    assert audit.feature_indices(2, "SCP") == [2, 18, 34, 50, 66, 82]


def test_snapshot_manifest_never_substitutes_final_for_missing_best(monkeypatch):
    source, best, final = Path("source.pt"), Path("missing-best.pt"), Path("last.pt")
    monkeypatch.setattr(Path, "is_file", lambda self: self.name != "missing-best.pt")
    monkeypatch.setattr(audit, "sha256_file", lambda path: "digest-" + Path(path).name)
    rows = audit.available_snapshots(source, best, final)
    by_role = {row["role"]: row for row in rows}
    assert by_role["source"]["available"] is True
    assert by_role["best"]["available"] is False
    assert by_role["final"]["available"] is True
    assert by_role["best"]["path"] != by_role["final"]["path"]
    assert by_role["best"]["selection_role"] == "descriptive_only"


def test_p3_keep_reject_relabel_share_conflicts_and_target_train_is_unlabeled():
    teacher = np.array([0, 0, 1, 1])
    proposal = np.array([0, 1, 0, 0])
    conflict = np.array([False, True, True, True])
    strict = np.array([False, True, False, True])
    rows = audit.supervision_decisions(
        parcel_index=np.arange(4), teacher_prediction=teacher,
        confidence=np.full(4, .95), teacher_margin=np.ones(4),
        structure_proposal=proposal, structure_margin=np.array([0., 2., 1., 3.]),
        conflict_mask=conflict, strict_positive_mask=strict, include_truth=False,
    )
    assert [row["action"] for row in rows] == ["KEEP", "RELABEL", "REJECT", "RELABEL"]
    assert all("true_label" not in row for row in rows)
    reject_conflicts = audit.conflict_ids(rows, correction=False)
    correction_conflicts = audit.conflict_ids(rows, correction=True)
    assert reject_conflicts == correction_conflicts == {1, 2, 3}


def test_p3_impact_counts_correct_keep_wrong_reject_and_wrong_relabel():
    rows = [
        {"action": "KEEP", "teacher_prediction": 0, "proposed_label": 0, "true_label": 0},
        {"action": "REJECT", "teacher_prediction": 0, "proposed_label": -1, "true_label": 1},
        {"action": "RELABEL", "teacher_prediction": 0, "proposed_label": 1, "true_label": 1},
        {"action": "RELABEL", "teacher_prediction": 0, "proposed_label": 2, "true_label": 1},
    ]
    summary = audit.summarize_decision_impact(rows)
    assert summary["correct_supervision_kept"] == 1
    assert summary["wrong_supervision_rejected"] == 1
    assert summary["wrong_supervision_corrected"] == 1
    assert summary["wrong_relabels"] == 1
    original = [
        {"conflict": True, "action": "RELABEL", "teacher_prediction": 0, "proposed_label": 1, "true_label": 1},
        {"conflict": False, "action": "KEEP", "teacher_prediction": 0, "proposed_label": 0, "true_label": 0},
    ]
    reject = audit.apply_conflict_policy(original, correction=False)
    correction = audit.apply_conflict_policy(original, correction=True)
    assert reject[0]["action"] == "REJECT"
    assert correction[0]["action"] == "RELABEL"


def test_phase_rotation_preserves_norm_and_second_order_rotates_twice():
    phase = np.array([[1., 0., 0., 1.]])
    rotated = audit.rotate_phase_moments(phase, np.pi / 2)
    assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(phase))
    assert rotated[0, :2] == pytest.approx([0., 1.], abs=1e-7)
    assert rotated[0, 2:] == pytest.approx([0., -1.], abs=1e-7)


def test_p4_perturbations_are_deterministic_and_declared_in_range():
    windows = np.linspace(-1, 1, 2 * 24 * 3).reshape(2, 24, 3)
    first = audit.controlled_window_perturbations(windows, seed=7)
    second = audit.controlled_window_perturbations(windows, seed=7)
    assert set(first) == {"noise", "mean", "scale", "shape", "dynamics"}
    for name in first:
        assert np.allclose(first[name]["windows"], second[name]["windows"])
        assert first[name]["distribution_in_range"] is True
        assert first[name]["magnitude"] > 0


def test_scheduler_is_four_way_frozen_only_and_resume_safe(monkeypatch):
    jobs = scheduler.build_job_graph()
    assert scheduler.INITIAL_GPU_ASSIGNMENT == {
        "P1:E:FR1_FR2": "0", "P1:E:FR2_DK1": "1",
        "P1:G:FR1_FR2": "2", "P1:G:FR2_DK1": "3",
    }
    assert all("train.py" not in " ".join(scheduler.job_command(job, scheduler.build_parser().parse_args(["dry-run", "--audit-root", "audit"]))) for job in jobs)
    monkeypatch.setattr(Path, "is_file", lambda self: self.name in audit.STAGE_ARTIFACTS["P1"])
    p1 = next(job for job in jobs if job.key == "P1:E:FR1_FR2")
    assert scheduler.artifact_complete(p1, scheduler.build_parser().parse_args(["resume", "--audit-root", "audit"]))
    args = scheduler.build_parser().parse_args(["dry-run", "--audit-root", "audit"])
    assert scheduler.job_log_path(args, p1) == Path("logs/structure_p1234_seed1/P1_E_FR1_FR2.log")


def test_no_target_test_or_training_entrypoint():
    assert "target_test" not in audit.ALLOWED_AUDIT_SPLITS
    parser = audit.build_parser()
    assert "train" not in parser._option_string_actions["--mode"].choices
