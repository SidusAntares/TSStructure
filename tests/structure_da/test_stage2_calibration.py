import csv
import json

import torch

from methods.structure_da.domain_phase_state import (
    DomainPhaseState,
    PhaseDecisionStatus,
)
from methods.structure_da.domain_shape_state import DomainShapeState, DomainShapeStatus
from methods.structure_da.stable_target_labels import StableTargetLabelScanResult
from methods.structure_da.stage2_calibration import export_stage2_calibration_statistics
from methods.structure_da.prototype_bank import SourcePrototypeBank
from methods.structure_da.target_hypothesis_scan import (
    CandidatePseudoLabel,
    PairwiseClassAlignment,
    TargetHypothesisScanResult,
)


def _alignment(sample_id: int, class_id: int, q_distance: float, gain: float) -> PairwiseClassAlignment:
    gamma = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    return PairwiseClassAlignment(
        sample_id=sample_id,
        class_id=class_id,
        gamma=gamma,
        t_identity_error=1.0,
        t_registered_error=gain,
        t_gain_ratio=gain,
        pre_common_support_t=0.9,
        common_support_t=0.9,
        gamma_finite=True,
        gamma_endpoint_error=0.0,
        gamma_strictly_increasing=True,
        gamma_min_increment=1.0 / 7.0,
        gamma_max_local_speed=1.0,
        gamma_roughness=0.0,
        phase_deviation=0.0,
        q_shape_distance=q_distance,
        q_distance_percentile=0.5,
        common_support_shape=0.9,
        numerically_valid=True,
        phase_evidence_eligible=(class_id == 0),
        reject_reasons=() if class_id == 0 else ("gain", "shape_outer"),
    )


def test_calibration_export_contains_raw_pairwise_and_unthresholded_top2_gap(tmp_path) -> None:
    alignments = (
        _alignment(10, 0, 0.2, 0.9),
        _alignment(10, 1, 0.5, 0.98),
        _alignment(11, 0, 0.4, 0.92),
        _alignment(11, 1, 0.3, 0.93),
    )
    result = TargetHypothesisScanResult(
        hypotheses=(),
        num_samples=2,
        num_pairwise_attempted=4,
        num_pre_support_rejected=0,
        num_solver_failed=0,
        num_gamma_rejected=0,
        num_gain_rejected=2,
        num_shape_support_rejected=0,
        num_outer_rejected=2,
        samples_with_zero_hypothesis=0,
        samples_with_one_hypothesis=2,
        samples_with_two_hypotheses=0,
        pairwise_alignments=alignments,
        candidate_pseudo_labels=(
            CandidatePseudoLabel(10, 0, 0.2, 0.5, True, False),
            CandidatePseudoLabel(11, 1, 0.3, 0.5, False, False),
        ),
        num_solver_calls=4,
        num_ready_classes=2,
        num_all_class_pairs=4,
    )
    phase = DomainPhaseState(
        scan_index=0,
        m=0,
        class_centers=(),
        valid_phase_classes=(),
        groups=(),
        rejected_classes=(),
        decision_status=PhaseDecisionStatus.UNCONFIRMED,
    )
    stable = StableTargetLabelScanResult(
        candidates=(),
        stable_labels=(),
        num_samples=2,
        num_without_confirmed_phase=2,
        num_candidate_views=0,
        num_classifier_pass=0,
        num_fused_pass=0,
        num_q_pass=0,
        num_stable_labels=0,
        num_ambiguous_rejected=0,
        stable_class_counts=(0, 0),
    )
    shape = DomainShapeState(
        scan_index=0,
        status=DomainShapeStatus.UNAVAILABLE,
        class_centers=(),
        valid_classes=(),
        delta=None,
        interactions=(),
        rho_shape=None,
        leave_one_out_drift=None,
        center_drift=None,
        confirmation_age=0,
    )

    bank = SourcePrototypeBank(
        trend_srvf=torch.zeros(2, 8, 1),
        shape_srvf=torch.zeros(2, 8, 1),
        trend_support=torch.ones(2, 8),
        shape_support=torch.ones(2, 8),
        fused=torch.zeros(2, 2),
        class_counts=torch.ones(2),
        ready=torch.ones(2, dtype=torch.bool),
        q_distance_samples=(torch.tensor([0.1]), torch.tensor([0.1])),
        f_distance_samples=(torch.tensor([0.1]), torch.tensor([0.1])),
        q_quantiles=torch.tensor([[0.1, 0.2, 0.4], [0.1, 0.2, 0.4]]),
        f_quantiles=torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
        version=1,
    )
    paths = export_stage2_calibration_statistics(
        output_dir=str(tmp_path),
        hypothesis_result=result,
        phase_state=phase,
        stable_result=stable,
        shape_state=shape,
        source_prototype_bank=bank,
    )
    geometry = torch.load(paths["geometry"], weights_only=False)
    assert geometry["gammas"].shape == (4, 8)
    assert geometry["sample_ids"].tolist() == [10, 10, 11, 11]

    with open(paths["summary"], encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["num_pairwise_attempted"] == 4
    assert summary["num_phase_evidence_eligible"] == 2
    assert summary["candidate_metrics"]["raw_q_distance_gap"]["count"] == 2
    assert summary["rejection_reason_counts"] == {"gain": 2, "shape_outer": 2}
    assert summary["phase_state"]["decision_status"] == "unconfirmed"

    with open(paths["candidates"], encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert float(rows[0]["raw_q_distance_gap"]) == 0.3
    assert abs(float(rows[1]["raw_q_distance_gap"]) - 0.1) < 1e-12
