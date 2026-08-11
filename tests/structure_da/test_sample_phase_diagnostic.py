from __future__ import annotations

import ast
import inspect
from pathlib import Path

import torch

from methods.structure_da.sample_phase_diagnostic import (
    RawShapeValidation,
    TOnlyPhaseRegistration,
    TRegistrationGeometryCache,
    classical_mds,
    phase_distance_matrix,
    remap_local_sample_ids_to_parcels,
    select_raw_shape_candidate,
    solve_t_only_registrations,
)


def _registration(class_id: int, *, legal: bool) -> TOnlyPhaseRegistration:
    return TOnlyPhaseRegistration(
        sample_index=0,
        sample_id=123,
        class_id=class_id,
        gamma=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        target_trend_valid=True,
        pre_common_support_t=1.0,
        t_identity_error=1.0,
        t_registered_error=0.5,
        t_gain_ratio=0.5,
        common_support_t=1.0,
        gamma_finite=True,
        gamma_endpoint_error=0.0,
        gamma_strictly_increasing=True,
        gamma_min_increment=0.5,
        gamma_max_local_speed=1.0,
        gamma_roughness=0.0,
        phase_deviation=0.0,
        numerically_valid=True,
        t_only_legal=legal,
        reject_reasons=() if legal else ("gain",),
        solver_error=None,
    )


def _shape(class_id: int, distance: float) -> RawShapeValidation:
    return RawShapeValidation(
        sample_index=0,
        sample_id=123,
        class_id=class_id,
        raw_shape_distance=distance,
        q_distance_percentile=None,
        common_support_shape=1.0,
        computable=True,
    )


def test_stage_a_registration_cache_is_trend_only():
    fields = set(TRegistrationGeometryCache.__dataclass_fields__)
    assert fields == {
        "sample_ids",
        "trend_srvf_reg",
        "trend_support_reg",
        "trend_valid",
        "registration_grid",
    }
    assert "shape" not in " ".join(fields).lower()
    assert "structure" not in " ".join(fields).lower()


def test_stage_a_solver_does_not_accept_shape_geometry_or_source_shape_bank():
    parameters = set(inspect.signature(solve_t_only_registrations).parameters)
    assert "source_bank" in parameters
    assert "target_cache" in parameters
    assert all("shape" not in name.lower() for name in parameters)
    assert all("structure" not in name.lower() for name in parameters)


def test_raw_s_selection_ignores_lower_distance_t_illegal_candidate():
    registrations = (
        _registration(0, legal=False),
        _registration(1, legal=True),
        _registration(2, legal=True),
    )
    shapes = (
        _shape(0, 0.01),
        _shape(1, 0.20),
        _shape(2, 0.35),
    )
    selected = select_raw_shape_candidate(registrations, shapes)
    assert selected.selected_class_id == 1
    assert selected.selected_distance == 0.20
    assert selected.second_class_id == 2
    assert abs(selected.margin - 0.15) < 1e-12
    assert selected.selectable_class_ids == (1, 2)


def test_raw_s_selection_has_no_classifier_teacher_or_stable_label_logic():
    tree = ast.parse(inspect.getsource(select_raw_shape_candidate))
    identifiers = {
        node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "classifier" not in identifiers
    assert "teacher" not in identifiers
    assert "stable" not in identifiers
    assert "q_distance_percentile" not in identifiers
    assert "calibration" not in identifiers


def test_phase_distance_matrix_and_mds_are_descriptive_geometry_only():
    gammas = torch.tensor(
        [
            [0.0, 0.50, 1.0],
            [0.0, 0.40, 1.0],
            [0.0, 0.65, 1.0],
        ],
        dtype=torch.float64,
    )
    distance = phase_distance_matrix(gammas)
    assert distance.shape == (3, 3)
    assert torch.allclose(distance, distance.T, atol=1e-12)
    assert torch.allclose(torch.diag(distance), torch.zeros(3, dtype=torch.float64))
    coords = classical_mds(distance, dimensions=2)
    assert coords.shape == (3, 2)
    assert torch.isfinite(coords).all()


def test_local_geometry_ids_are_mapped_back_to_stable_parcel_ids():
    local_ids = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    parcel_ids = [101, 205, 309, 412]
    mapped = remap_local_sample_ids_to_parcels(local_ids, parcel_ids)
    assert mapped.tolist() == [309, 101, 412, 205]


def test_06_launcher_keeps_full_stage_a_and_bounded_stage_b():
    text = (Path(__file__).resolve().parents[2] / "scripts" / "run_sample_level_phase_validity_at1_dk1_seed1.sh").read_text(encoding="utf-8")
    assert "stage_a=full_target_test" in text
    assert 'STAGE_B_SAMPLES_PER_CLASS="${STAGE_B_SAMPLES_PER_CLASS:-128}"' in text
    assert "--stage-b-samples-per-class" in text
    assert "--mds-samples-per-class" in text


def test_06_script_explicitly_forbids_clustering_decisions():
    text = (Path(__file__).resolve().parents[2] / "scripts" / "diagnose_sample_level_phase_validity.py").read_text(encoding="utf-8")
    assert '"clustering_performed": False' in text
    assert '"group_count_selected": False' in text
    assert '"s_used_for_gamma_generation_or_legality": False' in text
    assert '"distance_calibration_used": False' in text


def test_06_wraps_stage2_scanners_with_device_batch_loader():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "diagnose_sample_level_phase_validity.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(script)
    wrapped_names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "DeviceBatchLoader":
            if node.args and isinstance(node.args[0], ast.Name):
                wrapped_names.append(node.args[0].id)
    assert "source_train_loader" in wrapped_names
    assert "target_test_loader" in wrapped_names
