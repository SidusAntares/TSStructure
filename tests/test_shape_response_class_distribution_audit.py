import numpy as np
import subprocess
import sys
import torch
from pathlib import Path

from models.stclassifier import PseStructureProtoLTae


def _model(representation):
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=2, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation=representation, shape_injection="current_query",
        dropout=0.,
    ).eval()


def _features(points_by_class):
    values, labels = [], []
    for class_id, points in enumerate(points_by_class):
        values.extend(points)
        labels.extend([class_id] * len(points))
    return np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def test_real_forward_shape_response_and_strict_load_for_both_variants():
    from analysis.shape_response_class_distribution_audit import extract_shape_response

    for representation in ("current", "residual_response"):
        model = _model(representation)
        pixels = torch.randn(3, 8, 3, 4)
        valid = torch.ones(3, 8, 4)
        positions = torch.arange(8).repeat(3, 1) * 40
        spatial = model.spatial_encoder(pixels, valid, torch.zeros(3, 4))
        structure = model.prepare_structure(spatial, positions)
        torch.testing.assert_close(
            extract_shape_response(structure), structure["shapelet_response"],
        )
        restored = _model(representation)
        restored.load_state_dict(model.state_dict(), strict=True)


def test_identical_distributions_have_zero_shift_unit_ratio_and_perfect_local_support():
    from analysis.shape_response_class_distribution_audit import class_distribution_rows

    source, labels = _features([
        [[1., .1], [1., -.1], [.9, .2], [.9, -.2], [1., 0.]],
        [[.1, 1.], [-.1, 1.], [.2, .9], [-.2, .9], [0., 1.]],
    ])
    rows, confusion = class_distribution_rows(
        "current", "A_B", ["a", "b"], source, labels, source.copy(), labels.copy(),
    )
    assert confusion == []
    for row in rows:
        assert row["centroid_shift"] < 1e-10
        assert abs(row["radius_ratio"] - 1.) < 1e-10
        assert row["positive_margin_rate"] == 1.
        assert row["source_knn5_correct_rate"] == 1.
        assert np.isfinite(list(_numeric_values(row))).all()


def _numeric_values(row):
    return [value for value in row.values() if isinstance(value, (int, float, np.number))]


def test_class_shift_expansion_and_wrong_source_region_are_detected():
    from analysis.shape_response_class_distribution_audit import class_distribution_rows

    source, labels = _features([
        [[1., .02], [1., -.02], [.99, .03], [.99, -.03], [1., 0.]],
        [[.02, 1.], [-.02, 1.], [.03, .99], [-.03, .99], [0., 1.]],
    ])
    target, target_labels = _features([
        [[.1, 1.], [.2, .98], [-.1, .99], [.3, .95], [0., 1.]],
        [[.8, .6], [-.6, .8], [.7, .7], [-.7, .7], [0., 1.]],
    ])
    rows, confusion = class_distribution_rows(
        "residual_response", "A_B", ["a", "b"],
        source, labels, target, target_labels,
    )
    by_class = {row["class"]: row for row in rows}
    assert by_class["a"]["centroid_shift"] > .5
    assert by_class["a"]["positive_margin_rate"] < .5
    assert by_class["a"]["dominant_wrong_source_class"] == "b"
    assert by_class["a"]["dominant_wrong_count"] > 0
    assert by_class["b"]["radius_ratio"] > 1.
    assert any(row["target_class"] == "a" and row["wrong_source_class"] == "b"
               for row in confusion)


def test_multimodal_local_mismatch_is_visible_when_centroid_is_unchanged():
    from analysis.shape_response_class_distribution_audit import class_distribution_rows

    # Both target class centroids keep their source direction, while every target
    # mode lands in a local neighbourhood dominated by the other source class.
    source, labels = _features([
        [[1., .35], [1., .34], [1., .33], [1., -.35], [1., -.34], [1., -.33]],
        [[.8, .6], [.81, .59], [.79, .61], [.8, -.6], [.81, -.59], [.79, -.61]],
    ])
    target, target_labels = _features([
        [[.8, .6], [.81, .59], [.79, .61], [.8, -.6], [.81, -.59], [.79, -.61]],
        [[1., .35], [1., .34], [1., .33], [1., -.35], [1., -.34], [1., -.33]],
    ])
    rows, _ = class_distribution_rows(
        "current", "A_B", ["a", "b"], source, labels, target, target_labels,
    )
    assert max(row["centroid_shift"] for row in rows) < 1e-8
    assert min(row["source_knn5_correct_rate"] for row in rows) == 0.


def test_scope_is_four_tasks_two_variants_and_validation_only():
    import analysis.shape_response_class_distribution_audit as audit

    assert audit.VARIANTS == ("current", "residual_response")
    assert set(audit.TASKS) == {"AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1"}
    assert audit.AUDIT_SPLITS == ("source_train", "source_val", "target_val")
    manifest = audit.manifest_payload([], seed=1)
    assert manifest["test_split_accessed"] is False
    assert manifest["uda_checkpoint_used"] is False


def test_classes_without_actual_validation_support_are_excluded_without_test_fallback():
    from analysis.shape_response_class_distribution_audit import active_audit_classes

    original = ["a", "b", "c"]
    extracted = {
        "source_train": {"labels": np.array([0, 1, 2])},
        "source_val": {"labels": np.array([0, 1, 2])},
        "target_val": {"labels": np.array([0, 1, 1])},
    }
    active, excluded = active_audit_classes(extracted, original)
    assert active == ["a", "b"]
    assert excluded == ["c"]


def test_summary_requires_complete_four_by_two_grid_and_finite_outputs():
    from analysis.shape_response_class_distribution_audit import validate_outputs

    rows = []
    for variant in ("current", "residual_response"):
        for task in ("AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1"):
            rows.append({
                "variant": variant, "task": task, "mean_centroid_shift": .1,
                "mean_radius_ratio": 1., "mean_positive_margin_rate": .8,
                "mean_source_knn5_correct_rate": .7,
                "target_oracle_macro_f1": .6,
                "source_to_target_probe_macro_f1": .5,
            })
    validate_outputs(rows, [])
    broken = list(rows)
    broken[0] = dict(broken[0], mean_centroid_shift=float("nan"))
    try:
        validate_outputs(broken, [])
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite output must be rejected")


def test_direct_script_cli_is_importable():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "analysis/shape_response_class_distribution_audit.py", "--help"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
