import sys
import subprocess
import types
import os
from types import SimpleNamespace

import torch
import torch.nn as nn


for module_name, function_name in (
    ("competitors.dann.dann", "train_dann"),
    ("competitors.jumbot.jumbot", "train_jumbot"),
    ("competitors.mmd.train_mmd", "train_mmd"),
    ("competitors.alda.train_alda", "train_alda"),
):
    module_stub = types.ModuleType(module_name)
    setattr(module_stub, function_name, lambda *args, **kwargs: None)
    sys.modules.setdefault(module_name, module_stub)

import train
from models.reimts_classifier import PseReIMTSMTANLTAE
from models.reimts_classifier import ReIMTSClassificationOutput
from models.reimts_classifier import format_patch_diagnostics
from models.stclassifier import PseLTae


def test_create_model_registers_reimts_without_changing_pseltae():
    common = dict(input_dim=10, num_classes=3, with_extra=False)
    baseline = train.create_model(SimpleNamespace(model="pseltae", **common))
    reimts = train.create_model(
        SimpleNamespace(
            model="psereimtsmtanltae",
            reimts_levels=3,
            reimts_scale_factor=2,
            reimts_period=365,
            mtan_num_ref_points=8,
            mtan_latent_dim=128,
            mtan_heads=1,
            **common,
        )
    )

    assert isinstance(baseline, PseLTae)
    assert isinstance(reimts, PseReIMTSMTANLTAE)
    assert reimts.reimts_encoder.reference_points == (8, 8, 8)


def test_train_help_exposes_reimts_model_and_arguments():
    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "psereimtsmtanltae" in result.stdout
    assert "--reimts_levels" in result.stdout
    assert "--mtan_num_ref_points" in result.stdout
    assert "--reimts_loss_mode" in result.stdout
    assert "--reimts_patch_diagnostics" in result.stdout


class _SupervisedPatchModel:
    def forward_for_loss(self, pixels, mask, positions, extra):
        patch_logits = pixels
        return ReIMTSClassificationOutput(
            sample_logits=patch_logits.mean(dim=1),
            patch_logits=patch_logits,
            patch_valid=mask.bool(),
        )

    def forward(self, *args, **kwargs):
        raise AssertionError("patch-capable source training must use forward_for_loss")


def test_source_supervised_helper_uses_valid_patch_loss():
    model = _SupervisedPatchModel()
    patch_logits = torch.tensor(
        [[[3.0, 0.0], [0.0, 3.0], [2.0, 1.0], [9.0, -9.0]]]
    )
    patch_valid = torch.tensor([[True, True, True, False]])
    targets = torch.tensor([0])
    criterion = nn.CrossEntropyLoss()

    sample_logits, loss = train.forward_supervised_for_loss(
        model,
        patch_logits,
        patch_valid,
        torch.zeros(1, 1, dtype=torch.long),
        None,
        targets,
        criterion,
        loss_mode="patch",
    )
    expected = criterion(
        patch_logits[patch_valid],
        targets.unsqueeze(1).expand(-1, 4)[patch_valid],
    )

    assert sample_logits.shape == (1, 2)
    assert torch.allclose(loss, expected)


def test_tensorboard_fold_directory_stays_under_experiment_task():
    assert train.tensorboard_fold_dir("runs/round3/source_DK1", 0) == os.path.join(
        "runs/round3/source_DK1", "fold_0"
    )


def test_patch_diagnostics_report_counts_empty_rates_and_quarters():
    patch_valid = torch.tensor(
        [
            [True, True, True, True],
            [True, False, True, False],
            [False, False, False, False],
        ]
    )

    report = format_patch_diagnostics(patch_valid, "target strong")

    assert "[ReIMTS patch diagnostics] target strong" in report
    assert "lowest scale = 4 patches" in report
    assert "0: 1 (33.33%)" in report
    assert "2: 1 (33.33%)" in report
    assert "4: 1 (33.33%)" in report
    assert "empty patch rate: 50.00%" in report
    assert "Q1 empty rate: 33.33%" in report
    assert "Q2 empty rate: 66.67%" in report
    assert "Q3 empty rate: 33.33%" in report
    assert "Q4 empty rate: 66.67%" in report
