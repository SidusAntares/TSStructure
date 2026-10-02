import inspect
from pathlib import Path

import pytest
import torch

import timematch
from methods.structure_da.prototype_losses import class_relative_domain_alignment


def _outputs(source, target):
    return {"shapelet_response": source}, {"shapelet_response": target}


def _alignment(view, source, target, shapelet_count=2):
    source_output, target_output = _outputs(source, target)
    labels = torch.tensor([0, 0, 1, 1])
    confidence = torch.tensor([.95, .99, .96, 1.])
    trusted = torch.ones(4, dtype=torch.bool)
    return timematch.compute_shape_da_alignment(
        "batch_align", source_output, labels, target_output, labels,
        confidence, trusted, .9, alignment_view=view,
        shapelet_count=shapelet_count,
    )


def test_cli_defaults_to_full_and_exposes_only_the_four_frozen_views():
    source = Path("train.py").read_text(encoding="utf-8")
    assert '"--shape-alignment-view"' in source
    assert 'choices=["full", "strength", "concentration", "none"]' in source
    assert 'default="full"' in source


def test_feature_selector_slices_by_configured_shapelet_count_not_literal_16():
    response = torch.arange(24.).reshape(3, 8)
    assert timematch.select_shape_alignment_feature(response, "full", 4) is response
    torch.testing.assert_close(
        timematch.select_shape_alignment_feature(response, "strength", 4),
        response[:, :4],
    )
    torch.testing.assert_close(
        timematch.select_shape_alignment_feature(response, "concentration", 4),
        response[:, 4:8],
    )
    assert timematch.select_shape_alignment_feature(response, "none", 4) is None


def test_default_full_is_numerically_identical_to_original_batch_alignment():
    source = torch.randn(4, 4)
    target = torch.randn(4, 4, requires_grad=True)
    actual = _alignment("full", source, target)
    labels = torch.tensor([0, 0, 1, 1])
    confidence = torch.tensor([.95, .99, .96, 1.])
    expected = class_relative_domain_alignment(
        source, labels, target, labels, confidence, .9,
        distance="mse", min_target_support=2, support_saturation=4,
    )
    for key in ("total_loss", "global_loss", "relative_loss", "center_gap"):
        torch.testing.assert_close(actual[key], expected[key])


@pytest.mark.parametrize("view,active,inactive", [
    ("strength", slice(0, 2), slice(2, 4)),
    ("concentration", slice(2, 4), slice(0, 2)),
])
def test_component_alignment_gradient_only_enters_the_selected_response_half(
    view, active, inactive,
):
    source = torch.tensor([
        [1., 0., 0., 1.], [1., .2, .1, 1.],
        [0., 1., 1., 0.], [.2, 1., 1., .1],
    ])
    target = (source + .25).clone().requires_grad_(True)
    result = _alignment(view, source, target)
    assert torch.isfinite(result["total_loss"])
    result["total_loss"].backward()
    assert float(target.grad[:, active].abs().sum()) > 0.
    assert float(target.grad[:, inactive].abs().sum()) == 0.


def test_none_returns_graph_compatible_exact_zero_without_calling_alignment():
    source = torch.randn(4, 4)
    target = torch.randn(4, 4, requires_grad=True)
    result = _alignment("none", source, target)
    assert float(result["total_loss"]) == 0.
    assert result["valid_classes"] == 0
    result["total_loss"].backward()
    torch.testing.assert_close(target.grad, torch.zeros_like(target))


def test_training_only_passes_view_to_batch_alignment_not_model_outputs():
    trainer = inspect.getsource(timematch._train_structure_proto_timematch)
    assert "alignment_view=config.shape_alignment_view" in trainer
    assert "shapelet_count=config.shapelet_count" in trainer
    model_source = Path("models/stclassifier.py").read_text(encoding="utf-8")
    assert "shape_alignment_view" not in model_source


def test_manifest_logs_dimensions_and_forbids_target_gt_in_launcher():
    train_source = Path("train.py").read_text(encoding="utf-8")
    assert "alignment_feature_dim" in train_source
    assert "shape_alignment_view" in train_source
    launcher = Path("scripts/run_structure_alignment_view_4tasks_seed1.sh").read_text()
    for view in ("strength", "concentration", "none"):
        assert view in launcher
    for setting in (
        "--shape-da-mode batch_align",
        "--shape-alignment-label-source pseudo",
        "--oracle-pseudo-labels false",
        "--adaptive-pseudo-selection false",
        "--shape-representation current",
        "--shape-injection current_query",
    ):
        assert setting in launcher
    assert "full" not in launcher.split("for view in", 1)[-1].split("done", 1)[0]
    assert "source_retrained=false" in launcher
    assert "target_gt_used=false" in launcher


def test_launcher_runs_each_round_serially_and_each_round_four_tasks_in_parallel():
    launcher = Path("scripts/run_structure_alignment_view_4tasks_seed1.sh").read_text()
    assert 'for view in strength concentration none; do' in launcher
    assert launcher.count('run_task "$GPU') == 4
    assert launcher.count("wait \"$PID") == 4
    for call in (
        'run_task "$GPU0" AT1 "$AT1" DK1 "$DK1" "$view"',
        'run_task "$GPU1" FR1 "$FR1" FR2 "$FR2" "$view"',
        'run_task "$GPU2" FR2 "$FR2" DK1 "$DK1" "$view"',
        'run_task "$GPU3" DK1 "$DK1" AT1 "$AT1" "$view"',
    ):
        assert call in launcher
