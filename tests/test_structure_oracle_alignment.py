import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import timematch


def _config(**overrides):
    values = dict(
        shape_da_mode="batch_align",
        shape_alignment_label_source="pseudo",
        oracle_pseudo_labels=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cli_defaults_to_pseudo_and_accepts_oracle_without_reusing_main_oracle():
    source = Path("train.py").read_text(encoding="utf-8")
    assert '"--shape-alignment-label-source"' in source
    assert 'choices=["pseudo", "oracle"]' in source
    assert 'default="pseudo"' in source


def test_alignment_label_selection_only_substitutes_the_class_ids():
    pseudo = torch.tensor([2, 0, 1])
    target_gt = torch.tensor([1, 0, 2])
    assert torch.equal(
        timematch.select_shape_alignment_labels(pseudo, target_gt, "pseudo"), pseudo,
    )
    assert torch.equal(
        timematch.select_shape_alignment_labels(pseudo, target_gt, "oracle"), target_gt,
    )


def test_alignment_label_selection_does_not_modify_teacher_outputs_or_mask():
    pseudo = torch.tensor([1, 0, 1])
    confidence = torch.tensor([.95, .7, .99])
    trusted = confidence > .9
    original = (pseudo.clone(), confidence.clone(), trusted.clone())
    timematch.select_shape_alignment_labels(
        pseudo, torch.tensor([0, 0, 1]), "oracle",
    )
    for value, expected in zip((pseudo, confidence, trusted), original):
        torch.testing.assert_close(value, expected)


@pytest.mark.parametrize("mode", ["source_prototype", "boundary_support"])
def test_oracle_alignment_rejects_unsupported_shape_da_modes(mode):
    with pytest.raises(ValueError, match="shape_alignment_label_source=oracle"):
        timematch.validate_shape_alignment_label_source(
            _config(shape_da_mode=mode, shape_alignment_label_source="oracle")
        )


def test_oracle_alignment_rejects_oracle_main_pseudo_labels():
    with pytest.raises(ValueError, match="oracle_pseudo_labels=false"):
        timematch.validate_shape_alignment_label_source(_config(
            shape_alignment_label_source="oracle", oracle_pseudo_labels=True,
        ))


@pytest.mark.parametrize("mode", ["batch_align", "local_support"])
def test_oracle_alignment_accepts_only_the_two_diagnostic_modes(mode):
    timematch.validate_shape_alignment_label_source(_config(
        shape_da_mode=mode, shape_alignment_label_source="oracle",
    ))


def test_trainer_keeps_main_pseudo_and_alignment_labels_on_separate_paths():
    trainer = inspect.getsource(timematch._train_structure_proto_timematch)
    assert "alignment_target_labels = select_shape_alignment_labels(" in trainer
    assert "target_output[\"logits\"], training_target_labels" in trainer
    assert "target_output[\"shapelet_response\"], alignment_target_labels" in trainer
    assert "target_output,\n                    alignment_target_labels," in trainer


def test_oracle_manifest_is_explicit_about_the_only_gt_use():
    source = Path("train.py").read_text(encoding="utf-8")
    for field in (
        "oracle_alignment_diagnostic", "shape_alignment_label_source",
        "shape_alignment_class_only", "teacher_controls_trusted_mask",
        "oracle_pseudo_labels",
    ):
        assert field in source


def test_launcher_has_exact_four_oracle_jobs_and_frozen_controls():
    text = Path("scripts/run_structure_oracle_alignment_2tasks_4gpu_seed1.sh").read_text()
    assert 'experiment="${task}_${variant}_oracle_seed1"' in text
    for setting in (
        "--shape-alignment-label-source oracle",
        "--oracle-pseudo-labels false",
        "--adaptive-pseudo-selection false",
        "--shape-representation current",
        "--shape-injection current_query",
        "--shape-align-weight 0.05",
    ):
        assert setting in text
    assert "--local-support-k 5" in text
    assert "--local-support-temperature 0.1" in text
    assert 'shape_mode="batch_align"' in text
    assert 'shape_mode="local_support"' in text
    assert '--shape-da-mode "$shape_mode"' in text
    for call in (
        'run_job "$GPU0" AT1 "$AT1" DK1 "$DK1" center',
        'run_job "$GPU1" AT1 "$AT1" DK1 "$DK1" local',
        'run_job "$GPU2" FR2 "$FR2" DK1 "$DK1" center',
        'run_job "$GPU3" FR2 "$FR2" DK1 "$DK1" local',
    ):
        assert call in text


def test_no_model_parameter_files_are_modified_by_oracle_diagnostic():
    status = Path("models/stclassifier.py").read_text(encoding="utf-8")
    assert "shape_alignment_label_source" not in status
