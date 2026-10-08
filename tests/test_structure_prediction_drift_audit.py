from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


def test_transition_groups_cover_all_best_final_cases():
    from analysis.structure_prediction_drift_audit import prediction_transition

    assert prediction_transition(True, False) == "correct_to_wrong"
    assert prediction_transition(False, True) == "wrong_to_correct"
    assert prediction_transition(True, True) == "always_correct"
    assert prediction_transition(False, False) == "always_wrong"


def test_stage_records_align_by_unique_sample_id_not_loader_order():
    from analysis.structure_prediction_drift_audit import align_stage_records

    stages = {
        "source": [{"sample_id": 8, "value": "s8"}, {"sample_id": 3, "value": "s3"}],
        "best": [{"sample_id": 3, "value": "b3"}, {"sample_id": 8, "value": "b8"}],
        "final": [{"sample_id": 8, "value": "f8"}, {"sample_id": 3, "value": "f3"}],
    }
    aligned = align_stage_records(stages)
    assert [row["sample_id"] for row in aligned] == [3, 8]
    assert aligned[0]["source"]["value"] == "s3"
    assert aligned[0]["best"]["value"] == "b3"


def test_stage_alignment_rejects_duplicate_or_missing_ids():
    from analysis.structure_prediction_drift_audit import align_stage_records

    with pytest.raises(ValueError, match="duplicate sample_id"):
        align_stage_records({
            "source": [{"sample_id": 1}, {"sample_id": 1}],
            "best": [{"sample_id": 1}], "final": [{"sample_id": 1}],
        })
    with pytest.raises(ValueError, match="sample ID sets differ"):
        align_stage_records({
            "source": [{"sample_id": 1}], "best": [{"sample_id": 1}],
            "final": [{"sample_id": 2}],
        })


def test_structure_pair_metrics_and_true_class_margin_are_exact():
    from analysis.structure_prediction_drift_audit import (
        structure_pair_metrics, true_class_margin,
    )

    left = {
        "tokens": torch.tensor([[[1., 0.], [0., 1.]]]),
        "similarity": torch.tensor([[[.9, .1], [.2, .8]]]),
        "response": torch.tensor([[1., 0.]]),
        "presence": torch.tensor([[.8, .2]]),
    }
    right = {
        "tokens": torch.tensor([[[1., 0.], [1., 0.]]]),
        "similarity": torch.tensor([[[.7, .3], [.9, .1]]]),
        "response": torch.tensor([[0., 1.]]),
        "presence": torch.tensor([[.8, .2]]),
    }
    values = structure_pair_metrics(left, right)
    assert values["mean_window_token_cosine"].item() == pytest.approx(.5)
    assert values["mean_absolute_similarity_change"].item() == pytest.approx(.45)
    assert values["shape_response_cosine"].item() == pytest.approx(0.)
    assert values["window_top1_anchor_agreement"].item() == pytest.approx(.5)
    assert values["presence_cosine"].item() == pytest.approx(1.)
    logits = torch.tensor([[3., 1., 2.], [0., 4., 1.]])
    margin = true_class_margin(logits, torch.tensor([0, 2]))
    torch.testing.assert_close(margin, torch.tensor([1., -3.]))


class _Spatial(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(float(delta)))

    def forward(self, pixels, mask, extra):
        return pixels + self.delta


class _FixedBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(2.))
        self.last_positions = None

    def forward(self, spatial, positions, include_legacy_query=False):
        self.last_positions = positions.detach().clone()
        response = spatial.mean((1, 2)).unsqueeze(1) * self.anchor
        tokens = spatial.mean(2)
        similarity = torch.stack((tokens, -tokens), -1)
        return {
            "shape_tokens": tokens.unsqueeze(-1),
            "shapelet_similarity": similarity,
            "shapelet_response": response,
            "shapelet_strength": response,
        }


class _SourceMeasurement(nn.Module):
    shape_injection = "direct_response_query"

    def __init__(self):
        super().__init__()
        self.structure_branch = _FixedBranch()
        self.shape_classifier = nn.Linear(1, 2, bias=False)
        self.shape_response_norm = nn.Identity()

    def _shape_evidence(self, structure):
        return structure["shapelet_response"]


class _CurrentModel(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.spatial_encoder = _Spatial(delta)


def test_fixed_source_measurement_uses_current_pse_and_never_mutates_source():
    from analysis.structure_prediction_drift_audit import fixed_source_structure

    source = _SourceMeasurement().eval()
    before = deepcopy(source.state_dict())
    pixels = torch.ones(2, 3, 4)
    positions = torch.tensor([[2, 7, 12], [3, 8, 13]])
    mask = torch.ones_like(pixels)
    extra = torch.zeros(2, 1)
    first = fixed_source_structure(
        source, _CurrentModel(1.).eval(), pixels, mask, positions, extra,
    )
    second = fixed_source_structure(
        source, _CurrentModel(3.).eval(), pixels, mask, positions, extra,
    )
    assert not torch.allclose(first["response"], second["response"])
    torch.testing.assert_close(source.structure_branch.last_positions, positions)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_structure_measurement_is_independent_of_prediction_shift():
    from analysis.structure_prediction_drift_audit import fixed_source_structure

    source = _SourceMeasurement().eval()
    current = _CurrentModel(2.).eval()
    pixels = torch.ones(1, 3, 4)
    positions = torch.tensor([[2, 7, 12]])
    measured_at_minus_30 = fixed_source_structure(
        source, current, pixels, torch.ones_like(pixels), positions,
        torch.zeros(1, 1),
    )
    # Prediction shifts are deliberately not an argument to the structure measurer.
    measured_at_plus_45 = fixed_source_structure(
        source, current, pixels, torch.ones_like(pixels), positions,
        torch.zeros(1, 1),
    )
    for key in ("tokens", "similarity", "response", "presence"):
        torch.testing.assert_close(measured_at_minus_30[key], measured_at_plus_45[key])


def test_structure_and_split_compatibility_checks_reject_mismatches():
    from analysis.structure_prediction_drift_audit import (
        _assert_split_compatible, _assert_structure_compatible,
    )

    source = {
        "shape_representation": "current", "shapelet_count": 16,
        "seed": 1, "val_ratio": .1, "test_ratio": .2,
        "classes": [0, 1], "combine_spring_and_winter": False,
    }
    _assert_structure_compatible(source, dict(source), "best")
    _assert_split_compatible(source, dict(source), "best")
    with pytest.raises(ValueError, match="structure configuration mismatch"):
        _assert_structure_compatible(source, {**source, "shapelet_count": 32}, "best")
    with pytest.raises(ValueError, match="data split configuration mismatch"):
        _assert_split_compatible(source, {**source, "seed": 2}, "final")


def test_checkpoint_role_validation_and_missing_final():
    from analysis.structure_prediction_drift_audit import inspect_checkpoint

    tmp_path = Path("tests/.structure_prediction_drift_checkpoint_test")
    tmp_path.mkdir(exist_ok=True)
    missing = inspect_checkpoint(
        tmp_path / "checkpoint_last.pt", "final", "austria/33UVP/2017",
        "denmark/32VNH/2017",
    )
    assert missing["status"] == "MISSING_FINAL"

    packet = {
        "epoch": 19, "global_temporal_shift": -5,
        "config": {
            "method": "timematch", "epochs": 20,
            "source": "austria/33UVP/2017", "target": "denmark/32VNH/2017",
        },
        "state_dict": {},
    }
    final = tmp_path / "checkpoint_last.pt"
    torch.save(packet, final)
    result = inspect_checkpoint(
        final, "final", packet["config"]["source"], packet["config"]["target"],
    )
    assert result["status"] == "OK"
    assert result["epoch"] == 19
    assert result["global_shift"] == -5

    wrong = tmp_path / "checkpoint_best.pt"
    torch.save(packet, wrong)
    with pytest.raises(ValueError, match="final checkpoint must be checkpoint_last"):
        inspect_checkpoint(
            wrong, "final", packet["config"]["source"], packet["config"]["target"],
        )
    final.unlink()
    wrong.unlink()
    tmp_path.rmdir()


def test_checkpoint_specs_cover_requested_runs_and_never_substitute_best_for_final():
    from analysis.structure_prediction_drift_audit import experiment_specs

    specs = experiment_specs(Path("outputs"))
    assert {(item["method"], item["task"]) for item in specs} == {
        ("v2", "DK1_AT1"), ("v2clean", "DK1_AT1"),
        ("v2", "FR1_FR2"), ("v2clean", "FR1_FR2"),
        ("state_org_foundation_presence", "AT1_DK1"),
    }
    for item in specs:
        assert item["final"].name == "checkpoint_last.pt"
        assert item["best"].name in ("checkpoint_best.pt", "model.pt")
        assert item["final"] != item["best"]


def test_launcher_is_read_only_and_runs_five_audits_then_merge():
    text = Path("scripts/run_structure_prediction_drift_audit_seed1.sh").read_text()
    assert "structure_prediction_drift_audit.py audit" in text
    assert "structure_prediction_drift_audit.py merge" in text
    for token in ("v2", "v2clean", "state_org_foundation_presence"):
        assert token in text
    assert "train.py" not in text
    assert "optimizer" not in text
    assert "outputs/structure_prediction_drift_seed1" in text
