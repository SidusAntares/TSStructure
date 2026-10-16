import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _separable(seed=1, samples=36):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), samples // 3)
    centers = np.zeros((3, 16), dtype=np.float32)
    centers[0, :2] = (2., -1.)
    centers[1, :2] = (-1., 2.)
    centers[2, :2] = (-1., -1.)
    values = centers[labels] + rng.normal(0., .15, (samples, 16))
    return values.astype(np.float32), labels.astype(np.int64)


def test_strength_blocks_and_source_target_label_boundary():
    from analysis.structure_g_joint_strength_audit import (
        assert_cache_boundary, strength16,
    )

    response = np.arange(4 * 96, dtype=np.float32).reshape(4, 96)
    np.testing.assert_array_equal(strength16(response), response[:, :16])
    source_train = {"label": np.arange(4), "parcel_index": np.arange(4)}
    source_val = {"label": np.arange(4), "parcel_index": np.arange(8, 12)}
    target_train = {"parcel_index": np.arange(4)}
    target_val = {"label": np.arange(4), "parcel_index": np.arange(4, 8)}
    assert_cache_boundary(source_train, source_val, target_train, target_val)
    with pytest.raises(ValueError, match="Target Train labels"):
        assert_cache_boundary(source_train, source_val, {**target_train, "label": np.arange(4)}, target_val)


def test_best_single_anchor_is_selected_only_from_source_train():
    from analysis.structure_g_joint_strength_audit import select_best_single_anchor

    train, labels = _separable()
    altered_val = np.zeros_like(train)
    first = select_best_single_anchor(train, labels, random_state=3)
    second = select_best_single_anchor(
        train, labels, random_state=3,
        forbidden_validation_features=altered_val,
    )
    assert first["anchor_id"] == second["anchor_id"]
    assert first["selection_split"] == "source_train_cv"


def test_joint_reference_uses_source_only_and_handles_missing_class():
    from analysis.structure_g_joint_strength_audit import fit_joint_reference

    values, labels = _separable()
    reference = fit_joint_reference(values[labels != 2], labels[labels != 2], 3)
    assert reference["centers"].shape == (3, 16)
    assert reference["supported"].tolist() == [True, True, False]
    assert np.isnan(reference["radius"][2])
    np.testing.assert_allclose(
        reference["global_center"], np.median(values[labels != 2], axis=0),
    )


def test_multimodal_selection_uses_train_cv_not_target():
    from analysis.structure_g_joint_strength_audit import select_prototype_count

    rng = np.random.default_rng(4)
    labels = np.repeat([0, 1], 40)
    values = rng.normal(0, .1, (80, 16))
    values[:20, 0] -= 2.; values[20:40, 0] += 2.; values[40:, 0] += 5.
    selected = select_prototype_count(values, labels, candidates=(1, 2), seed=2)
    assert selected["selection_split"] == "source_train_cv"
    assert selected["prototype_count"] in (1, 2)


def test_scaled_nearest_neighbor_matches_source_classes():
    from analysis.structure_g_joint_strength_audit import predict_scaled_nearest_neighbor

    train, labels = _separable(samples=36)
    prediction = predict_scaled_nearest_neighbor(
        train, labels, train + .01, np.ones(train.shape[1]),
    )
    np.testing.assert_array_equal(prediction, labels)


def test_observation_views_pair_strictly_by_parcel():
    from analysis.structure_g_joint_strength_audit import pair_observation_views

    left = {"parcel_index": np.array([7, 9]), "strength": np.array([[1.], [2.]])}
    right = {"parcel_index": np.array([9, 7]), "strength": np.array([[20.], [10.]])}
    paired = pair_observation_views(left, right)
    assert paired["parcel_index"].tolist() == [7, 9]
    assert paired["right"].reshape(-1).tolist() == [10., 20.]
    with pytest.raises(ValueError, match="parcel sets"):
        pair_observation_views(left, {"parcel_index": np.array([7]), "strength": np.array([[1.]])})


def test_candidate_constraint_has_gradient_without_updating_parameters():
    from analysis.structure_g_joint_strength_audit import joint_constraint_gradient

    values, labels = _separable(samples=12)
    strength = torch.tensor(values, requires_grad=True)
    response_tail = torch.zeros(len(values), 80)
    weight = torch.nn.Parameter(torch.randn(3, 96))
    bias = torch.nn.Parameter(torch.zeros(3))
    before = weight.detach().clone()
    result = joint_constraint_gradient(
        strength, response_tail, torch.tensor(labels),
        centers=torch.zeros(3, 16), scale=torch.ones(16),
        radius=torch.full((3,), .25), classifier_weight=weight,
        classifier_bias=bias, focal_gamma=1.,
    )
    assert result["joint_grad_norm"] > 0
    assert 0 <= result["joint_grad_coverage"] <= 1
    assert result["joint_grad_norm_per_sample"].shape == (len(labels),)
    assert torch.equal(weight.detach(), before)
    assert weight.grad is None and bias.grad is None


def test_readiness_is_source_and_unlabeled_target_only():
    from analysis.structure_g_joint_strength_audit import build_readiness

    source = {"joint_gain_ci_low": .03, "joint_gain": .08}
    target_train = {"ood_rate": .1, "source_val_ood_ci_high": .2}
    constraint = {
        "val_active_rate": .2, "view_penalty_rate": .05,
        "gradient_cosine_ci_low": -.2,
    }
    packet = build_readiness(source, target_train, constraint)
    assert packet["source_joint_supported"] is True
    assert packet["transfer_support"] is True
    assert packet["compactness_justified"] is True
    assert packet["shape_ce_redundancy_risk"] is False
    assert packet["target_val_used_for_readiness"] is False
    assert "target_val" not in json.dumps(packet["decision_inputs"]).lower()


def test_readiness_can_recommend_relative_relations_from_source_only_evidence():
    from analysis.structure_g_joint_strength_audit import build_readiness

    packet = build_readiness(
        {
            "joint_gain_ci_low": -.01,
            "stability_supported": True,
            "relative_relation_gain_ci_low": .04,
        },
        {"ood_rate": .1, "source_val_ood_ci_high": .2},
        {
            "val_active_rate": .2,
            "view_penalty_rate": .05,
            "gradient_cosine_ci_low": -.2,
        },
    )
    assert packet["source_joint_supported"] is False
    assert packet["recommended_next_step"] == "study_relative_anchor_relations"


def test_unlabeled_target_distribution_has_no_true_label_fields():
    from analysis.structure_g_joint_strength_audit import unlabeled_target_distribution_rows

    rows = unlabeled_target_distribution_rows(
        "AT1_DK1", np.array([0, 0, 1]), np.array([0, 1, 1]),
        np.array([False, True, False]), ["a", "b"],
    )
    assert sum(row["model_prediction_count"] for row in rows) == 3
    assert sum(row["nearest_structure_count"] for row in rows) == 3
    assert all("label" not in key.lower() for row in rows for key in row)


def test_pre_readiness_cache_never_loads_target_val():
    from analysis.structure_g_joint_strength_audit import load_pre_readiness_cache

    root = Path(".joint_strength_pre_readiness_test")
    shutil.rmtree(root, ignore_errors=True); root.mkdir()
    (root / "checkpoint_manifest.json").write_text(json.dumps({
        "checkpoint_sha256": "abc", "target_test_used": False,
        "target_train_labels_cached": False,
    }))
    requested = []

    def loader(cache_dir, role, split):
        requested.append(split)
        offset = 2 if split == "source_val" else 0
        packet = {"parcel_index": np.arange(2) + offset, "response": np.zeros((2, 96))}
        if split != "target_train": packet["label"] = np.arange(2)
        return packet

    arrays, _ = load_pre_readiness_cache(root, "abc", loader=loader)
    assert set(arrays) == {"source_train", "source_val", "target_train"}
    assert requested == ["source_train", "source_val", "target_train"]
    shutil.rmtree(root, ignore_errors=True)


def test_view_cache_is_bound_to_checkpoint_sha():
    from analysis.structure_g_joint_strength_audit import load_paired_views

    root = Path(".joint_strength_view_cache_test")
    shutil.rmtree(root, ignore_errors=True); root.mkdir()
    path = root / "views.npz"
    np.savez_compressed(
        path, parcel_index_1=[1], strength_1=np.zeros((1, 16)), label_1=[0],
        parcel_index_2=[1], strength_2=np.zeros((1, 16)), label_2=[0],
        checkpoint_sha256=np.asarray("good"),
    )
    assert load_paired_views(path, "good")["parcel_index"].tolist() == [1]
    with pytest.raises(ValueError, match="SHA"):
        load_paired_views(path, "stale")
    shutil.rmtree(root, ignore_errors=True)


def test_checkpoint_validation_and_anchor_scope_are_checkpoint_local():
    from analysis.structure_g_joint_strength_audit import (
        assert_anchor_scope, validate_g_checkpoint_config, validate_response96_state,
    )

    config = {
        "shape_representation": "phase_moment", "phase_response_mode": "softmax",
        "shapelet_count": 16, "target_structure_detach_epochs": 5,
        "shape_phase_k2_enabled": True, "seed": 1, "epochs": 20,
    }
    validate_g_checkpoint_config(config, epoch=19)
    # These switches were introduced after some original-G checkpoints. Their
    # missing values must retain the historical softmax/K2-enabled defaults.
    legacy = dict(config)
    legacy.pop("phase_response_mode")
    legacy.pop("shape_phase_k2_enabled")
    validate_g_checkpoint_config(legacy, epoch=19)
    validate_response96_state({"shape_classifier.weight": torch.zeros(3, 96)})
    with pytest.raises(ValueError, match="Response96"):
        validate_response96_state({"shape_classifier.weight": torch.zeros(3, 32)})
    assert_anchor_scope("sha-a", "sha-a")
    with pytest.raises(ValueError, match="Anchor IDs"):
        assert_anchor_scope("sha-a", "sha-b")


def test_scheduler_dry_run_has_four_independent_tasks(capsys):
    from scripts.run_structure_g_joint_strength_audit_seed1 import (
        build_parser, planned_tasks, run_scheduler,
    )

    args = build_parser().parse_args(["dry-run", "--gpus", "0", "1", "2", "3"])
    assert len(planned_tasks(args.g_root)) == 4
    run_scheduler(args)
    output = capsys.readouterr().out
    assert output.count("PLAN|") == 4
    assert "G_FR2_DK1_seed1" in output


def test_resume_completion_is_bound_to_checkpoint_sha():
    import hashlib
    from scripts.run_structure_g_joint_strength_audit_seed1 import _complete

    root = Path(".joint_strength_complete_test")
    shutil.rmtree(root, ignore_errors=True); root.mkdir()
    checkpoint = root / "checkpoint_last.pt"; checkpoint.write_bytes(b"current")
    required = (
        "joint_discrimination.csv", "class_geometry.csv", "observation_stability.csv",
        "target_transfer.csv", "constraint_feasibility.csv", "JOINT_READINESS.json",
        "TASK_report.md",
    )
    for name in required: (root / name).write_text("x")
    (root / "checkpoint_manifest.json").write_text(json.dumps({
        "checkpoint_sha256": hashlib.sha256(b"current").hexdigest(),
    }))
    assert _complete(root, checkpoint)
    checkpoint.write_bytes(b"changed")
    assert not _complete(root, checkpoint)
    shutil.rmtree(root, ignore_errors=True)


def test_four_task_report_answers_the_five_audit_questions():
    from analysis.structure_g_joint_strength_audit import merge_outputs

    root = Path(".joint_strength_merge_test")
    shutil.rmtree(root, ignore_errors=True)
    for index, task in enumerate(("AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1")):
        folder = root / "tasks" / task
        folder.mkdir(parents=True)
        packet = {
            "task": task,
            "source_joint_supported": True,
            "transfer_support": True,
            "compactness_justified": False,
            "shape_ce_redundancy_risk": True,
            "recommended_next_step": "insufficient_evidence",
            "target_val_used_for_readiness": False,
            "decision_inputs": {
                "source": {
                    "joint_gain": .1, "joint_gain_ci_low": .02,
                    "joint_gain_ci_high": .15, "stability_supported": True,
                },
                "target_train_unlabeled": {"ood_rate": .1},
                "constraint": {
                    "val_active_rate": .2, "view_penalty_rate": .3,
                    "gradient_cosine_mean": .5,
                },
            },
        }
        (folder / "JOINT_READINESS.json").write_text(json.dumps(packet))
    merge_outputs(root)
    report = (root / "JOINT_FINAL_report.md").read_text()
    for heading in (
        "1. Full Strength16 vs single Anchor",
        "2. Source joint stability",
        "3. Observation-view stability",
        "4. Source-to-Target support",
        "5. Candidate constraint feasibility",
    ):
        assert heading in report
    merged = json.loads((root / "JOINT_READINESS.json").read_text())
    assert merged["completed_tasks"] == 4
    assert merged["target_val_used_for_readiness"] is False
    shutil.rmtree(root, ignore_errors=True)
