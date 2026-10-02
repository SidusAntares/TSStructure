import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import timematch


def _config(**overrides):
    values = dict(
        model="psestructureprotoltae", shape_representation="current",
        shape_injection="current_query", shape_da_mode="boundary_support",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parser_accepts_boundary_support_and_existing_modes_remain():
    source = Path("train.py").read_text(encoding="utf-8")
    assert (
        'choices=["batch_align", "source_prototype", "boundary_support", "local_support"]'
        in source
    )


@pytest.mark.parametrize("field,value", [
    ("model", "pseltae"),
    ("shape_representation", "residual_response"),
    ("shape_injection", "direct_response_query"),
])
def test_boundary_support_rejects_every_non_current_configuration(field, value):
    with pytest.raises(ValueError, match="boundary_support"):
        timematch.validate_boundary_support_config(_config(**{field: value}))


def test_auxiliary_classifiers_are_32d_small_asymmetric_and_seed_reproducible():
    source = nn.Linear(32, 5)
    first = timematch.initialize_boundary_classifiers(source, seed=7)
    second = timematch.initialize_boundary_classifiers(source, seed=7)
    c1, c2 = first
    assert c1.in_features == c2.in_features == 32
    assert c1.out_features == c2.out_features == 5
    torch.testing.assert_close(c1.weight, source.weight)
    assert 0. < float((c2.weight - c1.weight).abs().max()) < .01
    torch.testing.assert_close(c2.weight, second[1].weight)


def test_classifier_step_detaches_student_features_and_has_valid_source_ce():
    c1, c2 = timematch.initialize_boundary_classifiers(nn.Linear(32, 3), seed=2)
    source = torch.randn(7, 32, requires_grad=True)
    target = torch.randn(8, 32, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    result = timematch.boundary_classifier_objective(
        c1, c2, source, labels, target, nn.CrossEntropyLoss(),
    )
    result["loss"].backward()
    assert source.grad is None
    assert target.grad is None
    assert c1.weight.grad is not None and c2.weight.grad is not None
    assert float(result["source_loss"]) > 0.
    assert float(result["target_discrepancy"]) >= 0.


def test_generator_discrepancy_only_updates_response_and_zero_for_identical_heads():
    c1 = nn.Linear(32, 4)
    c2 = nn.Linear(32, 4)
    c2.load_state_dict(c1.state_dict())
    response = torch.randn(6, 32, requires_grad=True)
    loss = timematch.boundary_generator_discrepancy(response, c1, c2)
    assert float(loss) == pytest.approx(0.)
    c1.zero_grad(); c2.zero_grad()
    with torch.no_grad():
        c2.weight.add_(.01 * torch.randn_like(c2.weight))
    loss = timematch.boundary_generator_discrepancy(response, c1, c2)
    loss.backward()
    assert response.grad is not None and torch.isfinite(response.grad).all()
    assert c1.weight.grad is None and c2.weight.grad is None


def test_boundary_discrepancy_interface_has_no_target_supervision_inputs():
    signature = inspect.signature(timematch.boundary_generator_discrepancy)
    assert tuple(signature.parameters) == ("target_response", "classifier_1", "classifier_2")


def test_student_state_dict_is_unchanged_by_auxiliary_heads():
    student = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
    keys = tuple(student.state_dict())
    timematch.initialize_boundary_classifiers(nn.Linear(32, 4), seed=1)
    assert tuple(student.state_dict()) == keys


def test_boundary_mode_bypasses_both_existing_alignment_helpers():
    trainer = inspect.getsource(timematch._train_structure_proto_timematch)
    assert 'if shape_da_mode == "boundary_support"' in trainer
    assert "boundary_generator_discrepancy(" in trainer
    assert "compute_shape_da_alignment(" in trainer


def test_checkpoint_keeps_auxiliary_heads_outside_student_state_dict():
    trainer = inspect.getsource(timematch._train_structure_proto_timematch)
    assert 'strict=True' in trainer
    assert 'checkpoint["boundary_classifier_1_state_dict"]' in trainer
    assert 'checkpoint["boundary_classifier_2_state_dict"]' in trainer


def test_launcher_is_uda_only_current_query_and_preserves_v2clean_settings():
    text = Path("scripts/run_structure_boundary_support_4tasks_4gpu_seed1.sh").read_text()
    assert "source_retrained=false" in text
    assert "target_labels_used_by_boundary=false" in text
    assert "--shape-representation current" in text
    assert "--shape-injection current_query" in text
    assert "--shape-da-mode boundary_support" in text
    assert "--source-minority-mode base" in text
    assert "--adaptive-pseudo-selection false" in text
    assert "--oracle-pseudo-labels false" in text
    assert "--epochs 100" not in text
    assert text.count("timematch --weights") == 1
    for call in (
        'run_task "$GPU0" AT1 "$AT1" DK1 "$DK1"',
        'run_task "$GPU1" FR1 "$FR1" FR2 "$FR2"',
        'run_task "$GPU2" FR2 "$FR2" DK1 "$DK1"',
        'run_task "$GPU3" DK1 "$DK1" AT1 "$AT1"',
    ):
        assert call in text
