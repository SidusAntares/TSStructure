import numpy as np
import pytest
import torch

from evaluation import format_validation_summary
from models.reimts_classifier import PatchOccupancyMeter
from timematch import (
    PseudoLabelMeter,
    format_pseudo_histogram,
    format_shift_diagnostics,
    format_timematch_epoch_summary,
)
from utils.train_utils import format_duration, format_log_block, log_block

# train imports optional competitor modules in production; the repository test
# environment already provides their dependencies.
from train import format_run_complete, format_source_epoch_summary


def test_format_duration_uses_fixed_hour_minute_second_fields():
    assert format_duration(0) == "00:00:00"
    assert format_duration(3661.9) == "01:01:01"
    assert format_duration(25 * 3600 + 2) == "25:00:02"


def test_log_block_renders_readable_multiline_boundaries(capsys):
    rendered = format_log_block(
        "[SOURCE] Epoch 1/100",
        ["train:", "  loss: 0.842731", "  lr: 3.72e-04"],
    )

    assert rendered.splitlines()[0] == "=" * 72
    assert "[SOURCE] Epoch 1/100\n" in rendered
    assert "train:\n  loss: 0.842731\n  lr: 3.72e-04" in rendered
    assert rendered.splitlines()[-1] == "=" * 72

    log_block("[VALIDATION]", ["loss: 0.1"], border="-")
    captured = capsys.readouterr().out
    assert captured.startswith("\n" + "-" * 72)
    assert "\n[VALIDATION]\n" in captured
    assert captured.endswith("-" * 72 + "\n\n")


def test_patch_occupancy_meter_aggregates_batches_without_changing_predictions():
    meter = PatchOccupancyMeter()
    meter.update(torch.tensor([[True, True, True, True], [True, False, True, False]]))
    meter.update(torch.tensor([[False, False, False, False]]))

    summary = meter.summary()
    lines = meter.format_lines("ReIMTS patches:")

    assert summary["samples"] == 3
    assert summary["valid_per_sample_mean"] == 2.0
    assert summary["empty_patch_rate"] == 0.5
    assert summary["quarter_empty_rates"] == pytest.approx(
        [1 / 3, 2 / 3, 1 / 3, 2 / 3]
    )
    assert lines[0] == "ReIMTS patches:"
    assert "  valid patches/sample mean: 2.000" in lines
    assert "  Q4 empty rate: 66.67%" in lines


def test_pseudo_histogram_is_one_class_per_line_and_includes_zero_counts():
    rendered = format_pseudo_histogram(
        np.array([0, 0, 2]), ["corn", "horsebeans", "winter_wheat"]
    )

    assert rendered == [
        "pseudo class histogram:",
        "  corn: 2",
        "  horsebeans: 0",
        "  winter_wheat: 1",
    ]


def test_pseudo_meter_reports_zero_safe_confidence_and_target_updates():
    meter = PseudoLabelMeter()
    meter.update(
        confidences=torch.tensor([0.4, 0.8]),
        pseudo_targets=torch.tensor([0, 1]),
        accepted=torch.tensor([False, False]),
        true_targets=torch.tensor([0, 1]),
    )

    summary = meter.summary(num_classes=2)

    assert summary["seen"] == 2
    assert summary["accepted"] == 0
    assert summary["confidence_mean_all"] == pytest.approx(0.6)
    assert summary["confidence_mean_accepted"] == 0.0
    assert summary["macro_f1_debug"] == 0.0
    assert summary["accepted_labels"].size == 0


def test_pseudo_meter_preserves_existing_observed_class_macro_f1_debug():
    meter = PseudoLabelMeter()
    meter.update(
        confidences=torch.tensor([0.99, 0.98]),
        pseudo_targets=torch.tensor([0, 0]),
        accepted=torch.tensor([True, True]),
        true_targets=torch.tensor([0, 0]),
    )

    summary = meter.summary(num_classes=2)

    assert summary["macro_f1_debug"] == 1.0


def test_shift_diagnostics_ranks_minimum_scores_and_prints_top_entries_multiline():
    rendered = format_shift_diagnostics(
        estimator="AM",
        shifts=[-2, -1, 0, 1, 2, 3],
        scores=np.array([0.5, 0.1, 0.2, 0.4, 0.3, 0.6]),
        maximize=False,
        accuracy_scores=np.array([0.2, 0.4, 0.3, 0.9, 0.1, 0.0]),
        sample_batches=100,
        runtime_seconds=87.4,
        spatial_encoder_time=1.2,
        reimts_mtan_time=3.4,
        total_feature_preparation_time=4.6,
        ltae_classifier_total_time=8.0,
    )

    assert "[SHIFT ESTIMATION]" in rendered
    assert "selected_shift: -1" in rendered
    assert "second_best_shift: 0" in rendered
    assert "score_gap: 0.100000" in rendered
    assert "debug oracle:\n  best_accuracy_shift: 1" in rendered
    assert "feature preparation:" in rendered
    assert "spatial_encoder_time: 1.200000 s" in rendered
    assert "reimts_mtan_time: 3.400000 s" in rendered
    assert "total_feature_preparation_time: 4.600000 s" in rendered
    assert "candidate evaluation:" in rendered
    assert "ltae_classifier_total_time: 8.000000 s" in rendered
    assert "runtime:\n  total_shift_estimation_time: 87.400000 s" in rendered
    top_lines = [line for line in rendered.splitlines() if "shift=" in line]
    assert len(top_lines) == 5
    assert top_lines[0].strip().startswith("1. shift=-1")


def test_shift_diagnostics_handles_single_candidate_without_crashing():
    rendered = format_shift_diagnostics(
        estimator="AM",
        shifts=[0],
        scores=np.array([0.25]),
        maximize=False,
        accuracy_scores=np.array([0.5]),
        sample_batches=1,
        runtime_seconds=1.0,
    )

    assert "selected_shift: 0" in rendered
    assert "second_best_shift: unavailable" in rendered
    assert "score_gap: unavailable" in rendered


def test_timematch_epoch_summary_has_separate_shift_pseudo_loss_and_timing_sections():
    rendered = format_timematch_epoch_summary(
        epoch=6,
        epochs=20,
        shift={
            "estimator": "AM", "min": -42, "max": 0,
            "target_to_source": -31, "source_to_target": 31,
            "seconds": 87.4,
        },
        pseudo={
            "seen": 100, "accepted": 25, "confidence_mean_all": 0.78,
            "confidence_mean_accepted": 0.95, "macro_f1_debug": 0.61,
        },
        histogram_lines=["pseudo class histogram:", "  corn: 25"],
        losses={"source": 0.8, "target": 0.2, "total": 1.2, "lr": 1e-4},
        timing={"training": 10.0, "validation": 2.0, "total": 100.0, "elapsed": 600.0},
        target_updates=25,
        patch_lines=["ReIMTS patches (source strong):", "  empty patch rate: 0.07%"],
    )

    for field in (
        "[TIMEMATCH] Epoch 6/20", "shift:", "candidate_count: 43",
        "pseudo labels:", "acceptance_rate: 25.00%", "target_updates: 25",
        "pseudo class histogram:", "loss:", "patch occupancy:", "timing:",
        "elapsed: 00:10:00",
    ):
        assert field in rendered


def test_source_epoch_summary_contains_loss_patch_and_runtime_sections():
    meter = PatchOccupancyMeter()
    meter.update(torch.tensor([[True, True, True, True], [True, True, False, True]]))

    rendered = format_source_epoch_summary(
        epoch=17,
        epochs=100,
        loss=0.842731,
        lr=3.72e-4,
        patch_meter=meter,
        epoch_seconds=185.32,
        elapsed_seconds=3161,
    )

    for field in (
        "[SOURCE] Epoch 17/100", "train:", "loss: 0.842731",
        "lr: 3.72e-04", "ReIMTS patches:",
        "valid patches/sample mean: 3.500", "timing:",
        "epoch: 185.32 s", "elapsed: 00:52:41",
    ):
        assert field in rendered


def test_validation_summary_reports_checkpoint_decision_and_timing():
    rendered = format_validation_summary(
        metrics={"loss": 0.2, "accuracy": 0.8, "macro_f1": 0.7, "kappa": 0.6},
        best_before=0.65,
        best_after=0.7,
        checkpoint_saved=True,
        validation_seconds=12.5,
    )

    assert rendered.splitlines()[0] == "-" * 72
    for field in (
        "[VALIDATION]", "loss: 0.200000", "accuracy: 0.800000",
        "macro_f1: 0.700000", "kappa: 0.600000",
        "best_macro_f1_before: 0.650000",
        "best_macro_f1_after: 0.700000", "checkpoint_saved: true",
        "validation_time: 12.50 s",
    ):
        assert field in rendered


def test_run_complete_summary_has_total_and_final_metrics():
    rendered = format_run_complete(
        experiment="DK1_to_FR1",
        total_seconds=29862,
        best_val_macro_f1=0.71,
        test_macro_f1=0.69,
    )

    assert "[RUN COMPLETE]" in rendered
    assert "experiment: DK1_to_FR1" in rendered
    assert "total_runtime: 08:17:42" in rendered
    assert "best_val_macro_f1: 0.710000" in rendered
    assert "test_macro_f1: 0.690000" in rendered
