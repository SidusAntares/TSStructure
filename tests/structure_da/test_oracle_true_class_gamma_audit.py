from __future__ import annotations

from pathlib import Path

import torch

from methods.structure_da.oracle_gamma_audit import (
    PRODUCTION_RULE_ORDER,
    audit_current_phase_legality,
    conditional_probability,
)
from methods.structure_da.sample_phase_diagnostic import RawShapeValidation, TOnlyPhaseRegistration
from methods.structure_da.target_hypothesis_scan import PhaseHypothesisScanConfig


def _config() -> PhaseHypothesisScanConfig:
    return PhaseHypothesisScanConfig(
        registration_lambda=0.0,
        registration_gain_ratio_max=0.95,
        registration_min_common_support=0.6,
        registration_max_roughness=1e9,
        registration_min_increment=0.001,
        registration_max_local_speed=4.0,
        registration_max_deviation=0.25,
        class_hypothesis_margin=0.05,
        k_reg=128,
        registration_workers=1,
    )


def _record(**overrides) -> TOnlyPhaseRegistration:
    values = dict(
        sample_index=0,
        sample_id=101,
        class_id=3,
        gamma=torch.linspace(0.0, 1.0, 128, dtype=torch.float64),
        target_trend_valid=True,
        pre_common_support_t=0.8,
        t_identity_error=1.0,
        t_registered_error=0.5,
        t_gain_ratio=0.5,
        common_support_t=0.8,
        gamma_finite=True,
        gamma_endpoint_error=0.0,
        gamma_strictly_increasing=True,
        gamma_min_increment=0.01,
        gamma_max_local_speed=1.0,
        gamma_roughness=0.0,
        phase_deviation=0.1,
        numerically_valid=True,
        t_only_legal=True,
        reject_reasons=(),
        solver_error=None,
    )
    values.update(overrides)
    return TOnlyPhaseRegistration(**values)


def _shape(**overrides) -> RawShapeValidation:
    values = dict(
        sample_index=0,
        sample_id=101,
        class_id=3,
        raw_shape_distance=0.2,
        q_distance_percentile=0.4,
        common_support_shape=0.8,
        computable=True,
    )
    values.update(overrides)
    return RawShapeValidation(**values)


def test_current_legality_includes_production_s_availability_gates():
    decision = audit_current_phase_legality(_record(), _shape(computable=False, q_distance_percentile=None), _config())
    assert not decision.accepted
    assert decision.reject_reasons == ("shape_support",)

    decision = audit_current_phase_legality(_record(), _shape(q_distance_percentile=None), _config())
    assert not decision.accepted
    assert decision.reject_reasons == ("q_cdf_unavailable",)


def test_current_legality_does_not_reuse_06_target_trend_valid_extra_gate():
    decision = audit_current_phase_legality(_record(target_trend_valid=False), _shape(), _config())
    assert decision.accepted
    assert decision.reject_reasons == ()


def test_current_legality_separates_numerical_validity_from_configured_increment_gate():
    record = _record(gamma_min_increment=0.0005, numerically_valid=True)
    decision = audit_current_phase_legality(record, _shape(), _config())
    assert not decision.accepted
    assert "gamma_increment" in decision.reject_reasons


def test_current_legality_reports_all_actual_configured_reasons():
    record = _record(
        pre_common_support_t=0.5,
        gamma_min_increment=0.0005,
        gamma_max_local_speed=5.0,
        gamma_roughness=2e9,
        phase_deviation=0.3,
        t_gain_ratio=0.99,
    )
    decision = audit_current_phase_legality(record, _shape(), _config())
    assert decision.reject_reasons == (
        "pre_support",
        "gamma_increment",
        "gamma_speed",
        "gamma_roughness",
        "gamma_deviation",
        "gain",
    )


def test_conditional_probability_uses_condition_as_denominator():
    assert conditional_probability([True, False, True, False], [True, True, False, False]) == 0.5


def test_07_script_is_posthoc_and_contains_no_registration_solver_fallback():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "audit_oracle_true_class_gamma_effectiveness.py").read_text(encoding="utf-8")
    launcher = (root / "scripts" / "run_oracle_true_class_gamma_audit_at1_dk1_seed1.sh").read_text(encoding="utf-8")
    assert "solve_t_only_registrations" not in text
    assert "optimum_reparam_curve" not in text
    assert "exact_dp_calls=0" in launcher
    assert "finish_06_stage_a_first" in launcher


def test_07_scope_excludes_grouping_stable_label_teacher_and_class_centers():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "audit_oracle_true_class_gamma_effectiveness.py").read_text(encoding="utf-8")
    assert '"domain_phase_grouping_modified": False' in text
    assert '"stable_label_used": False' in text
    assert '"teacher_used": False' in text
    assert '"class_center_used": False' in text
    assert '"group_center_used": False' in text
    assert tuple(PRODUCTION_RULE_ORDER) == (
        "pre_support",
        "gamma_increment",
        "gamma_speed",
        "gamma_roughness",
        "gamma_deviation",
        "gain_unavailable",
        "gain",
        "shape_support",
        "q_cdf_unavailable",
    )
