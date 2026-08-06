from __future__ import annotations

import subprocess
import sys

import train


def test_train_does_not_import_joint_trainer() -> None:
    source = open("train.py", encoding="utf-8").read()
    assert "JointTrainer" not in source
    assert "joint_trainer" not in source
    assert "geometry_optimizer" not in source
    assert "ShapeFeatureEncoder" not in source
    assert "create_joint_structure_da_train_loaders" not in source


def test_train_constructs_source_only_model() -> None:
    assert hasattr(train, "create_source_train_loader")
    assert hasattr(train, "train_source_classification")
    assert not hasattr(train, "train_joint_structure_da")
    assert not hasattr(train, "create_joint_structure_da_train_loaders")


def test_train_help_exposes_stage1_and_removes_legacy_arguments() -> None:
    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--stage1_epochs" in result.stdout
    for option in (
        "--lambda_geometry",
        "--lambda_quality",
        "--warp_num_candidates",
        "--candidate_init_warp_amplitude",
        "--num_shape_basis",
        "--num_phase_basis",
        "--shape_dim",
        "--quality_domain_score_warmup_epochs",
        "--prototype_momentum",
        "--phase_gain_weight",
    ):
        assert option not in result.stdout
