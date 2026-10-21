import math
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.visualize_timematch_ltae_attention import (
    _plot_pair_page,
    build_sample_pairs,
    fit_source_pc1,
    marker_rows,
    project_pc1,
    resolve_student_domain_shifts,
    selection_summary_row,
    select_attention_indices,
    validate_attention_tensor,
    validate_original_final_checkpoint,
)


def test_top_fraction_uses_ceil_and_deterministic_index_ties():
    weights = np.array([.4, .4, .1, .1, 0.], dtype=np.float64)
    selected = select_attention_indices(
        weights, np.arange(5), "top_fraction", .3,
    )
    assert selected.tolist() == [0, 1]


def test_absolute_threshold_allows_empty_selection():
    selected = select_attention_indices(
        np.array([.01, .02, .03]), np.arange(3), "absolute", .05,
    )
    assert selected.size == 0
    row = selection_summary_row(
        "FR1:42", np.array([.01, .02, .03]), np.array([10, 20, 30]),
        4, "absolute", .05, "alpha_gt_0p05",
    )
    assert row["selected_count"] == 0
    assert row["selected_indices"] == ""
    assert row["selected_dates"] == ""


def test_attention_validation_keeps_all_heads_and_normalizes_per_head():
    attention = np.full((2, 16, 1, 7), 1 / 7, dtype=np.float64)
    normalized = validate_attention_tensor(attention)
    assert normalized.shape == (2, 16, 7)
    np.testing.assert_allclose(normalized.sum(-1), 1., atol=1e-6)
    broken = attention.copy()
    broken[0, 3, 0, 0] += .2
    with pytest.raises(ValueError, match="sum to one"):
        validate_attention_tensor(broken)


def test_source_and_target_student_shifts_follow_timematch_training_roles():
    packet = {"global_temporal_shift": -15}
    config = SimpleNamespace(shift_source=True)
    assert resolve_student_domain_shifts(packet, config) == {
        "source": 15, "target": 0, "target_teacher": -15,
    }
    config.shift_source = False
    assert resolve_student_domain_shifts(packet, config)["source"] == 0


def test_pc1_is_one_source_fitted_basis_with_deterministic_sign():
    source = np.array([
        [-2., 0.], [-1., 0.], [1., 0.], [2., 0.],
    ])
    mean, component, explained = fit_source_pc1(source)
    assert explained == pytest.approx(1.)
    assert component[np.argmax(np.abs(component))] > 0
    projected = project_pc1(np.array([[3., 0.], [-3., 0.]]), mean, component)
    np.testing.assert_allclose(projected, [3., -3.], atol=1e-7)


def test_pairs_are_deterministic_distinct_and_label_correct():
    source = {
        0: ["s0", "s1", "s2", "s3"],
        1: ["s4", "s5", "s6", "s7"],
        2: ["s8", "s9", "s10", "s11"],
    }
    target = {
        0: ["t0", "t1"], 1: ["t2", "t3"], 2: ["t4", "t5"],
    }
    first = build_sample_pairs(source, target, [0, 1, 2], pairs_per_class=1, seed=7)
    second = build_sample_pairs(source, target, [0, 1, 2], pairs_per_class=1, seed=7)
    assert first == second
    same = [row for row in first if row["group"] == "same_domain_same_class"]
    cross = [row for row in first if row["group"] == "cross_domain_same_class"]
    different = [row for row in first if row["group"] == "same_domain_different_class"]
    assert all(row["first_id"] != row["second_id"] for row in same)
    assert all(row["first_label"] == row["second_label"] for row in same + cross)
    assert all(row["first_label"] != row["second_label"] for row in different)


def test_curve_markers_share_exact_observation_indices_between_views():
    sample = {
        "sample_id": "FR1:42",
        "positions": np.array([10, 20, 40]),
        "attention": np.full((16, 3), 1 / 3),
        "pse_pc1": np.array([1., 2., 3.]),
        "fourier_observed_pc1": np.array([1.5, 2.5, 3.5]),
    }
    rows = marker_rows(sample, head=0, mode="top_fraction", value=2 / 3)
    pse = [row for row in rows if row["curve_type"] == "pse_pc1"]
    fourier = [row for row in rows if row["curve_type"] == "fourier"]
    assert [row["observation_index"] for row in pse] == [0, 1]
    assert [row["observation_index"] for row in fourier] == [0, 1]
    assert [row["date"] for row in pse] == [10, 20]
    assert [row["date"] for row in fourier] == [10, 20]


def test_only_original_final_student_checkpoint_is_accepted():
    config = {
        "model": "pseltae", "source": "france/30TXT/2017",
        "target": "france/31TCJ/2017", "seed": 1, "epochs": 20,
        "output_student": True,
    }
    packet = {
        "config": config, "epoch": 19, "state_dict": {},
        "teacher_state_dict": {}, "global_temporal_shift": 15,
    }
    validate_original_final_checkpoint(
        "checkpoint_last.pt", packet, "france/30TXT/2017",
        "france/31TCJ/2017", 1,
    )
    with pytest.raises(ValueError, match="checkpoint_last.pt"):
        validate_original_final_checkpoint(
            "checkpoint_best.pt", packet, config["source"], config["target"], 1,
        )
    packet["epoch"] = 18
    with pytest.raises(ValueError, match="final epoch"):
        validate_original_final_checkpoint(
            "checkpoint_last.pt", packet, config["source"], config["target"], 1,
        )


def test_synthetic_attention_page_renders_as_openable_png():
    sample = {
        "sample_id": "FR1:42", "domain": "france/30TXT/2017",
        "parcel_id": 42, "label": 0, "applied_shift": 15,
        "positions": np.array([10., 20., 40.]),
        "valid_indices": np.arange(3),
        "attention": np.full((16, 3), 1 / 3),
        "pse_pc1": np.array([1., 2., 3.]),
        "fourier_grid": np.array([0., 30., 60.]),
        "fourier_grid_pc1": np.array([.5, 2.5, 1.5]),
        "fourier_observed_pc1": np.array([1.5, 2.5, 3.5]),
    }
    figure = _plot_pair_page(
        sample, sample, ["crop"], "FR1_FR2", "cross_domain_same_class",
        "top_fraction", .2, "pse_pc1",
    )
    stream = BytesIO()
    figure.savefig(stream, format="png")
    assert stream.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(figure.axes) == 16
