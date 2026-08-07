"""Two-stage structure model: Stage-1 source prototype training backbone."""

from models.ltae import ContinuousTime2Vec, TrendStructureSharedLTAE

from .backbone import StructureBackbone, StructureBackboneOutput
from .decomposition import DecompositionOutput, SymmetricTimeKernelDecomposition
from .diagnostics import (
    ContributionDiagnostics,
    DecompositionDiagnostics,
    DiagnosticMoments,
    DiagnosticStat,
    compute_decomposition_diagnostics,
    compute_structure_contribution_diagnostics,
    merge_contribution_diagnostics,
    merge_decomposition_diagnostics,
    summarize_contribution_diagnostics,
    summarize_decomposition_diagnostics,
)
from .feature_snapshots import (
    FeatureSnapshotConfig,
    FeatureSnapshotManager,
    SnapshotCaptureResult,
    create_feature_snapshot_manager,
    deterministic_class_selection,
    load_selected_samples,
)
from .full_model import TSStructureModel
from .phase_evidence import (
    GammaDiagnostics,
    PairwisePhaseCandidate,
    compute_gamma_diagnostics,
    empirical_cdf,
    shape_distance_to_prototype,
)
from .phase_geometry import (
    gamma_to_psi,
    pairwise_phase_distances,
    phase_distance,
    sqrt_mean_gamma,
    sqrt_median_gamma,
)
from .domain_phase_state import (
    DomainPhaseConfig,
    DomainPhaseState,
    PhaseClassCenter,
    PhaseGroup,
    PhaseGroupStatus,
    update_domain_phase_state,
)
from .confirmed_phase_view import (
    ConfirmedPhaseView,
    align_target_positions_to_source,
    build_confirmed_class_to_group_map,
    build_confirmed_phase_view,
)
from .ema_teacher import Stage2EMATeacher
from .phase_registration import (
    FdasrsfCurveRegistrationAdapter,
    GammaLegalityOutput,
    build_source_registration_prototypes,
    check_gamma_legality,
    resample_gamma,
    warp_q_gamma,
    warp_support_gamma,
)
from .prototype_bank import (
    QUANTILE_LEVELS,
    SourcePrototypeBank,
    SupportAwareDistanceOutput,
    support_aware_q_distance,
)
from .registration_geometry import (
    RegistrationGeometryOutput,
    SourceRegistrationPrototypeBank,
    TargetGeometryCache,
    evaluate_registration_geometry,
)
from .representation import (
    FunctionalGeometryOutput,
    RawTemporalRepresentation,
    TSStructureForwardOutput,
)
from .source_prototype_scanner import (
    build_source_prototype_bank,
    finalize_distance_statistics,
)
from .source_trainer import SourceClassificationTrainer, SourceTrainStepOutput
from .stage1_objective import Stage1LossOutput, Stage1Objective
from .stage2_parameter_policy import (
    Stage2ParameterPolicy,
    configure_stage2_parameter_policy,
)
from .stable_target_labels import (
    StableLabelConfig,
    StableTargetCandidate,
    StableTargetLabel,
    StableTargetLabelScanResult,
    evaluate_stable_target_candidate,
    scan_stable_target_labels,
)
from .target_hypothesis_scan import (
    PhaseHypothesisScanConfig,
    TargetClassPhaseHypothesis,
    TargetHypothesisScanResult,
    scan_target_class_phase_hypotheses,
)
from .temporal_functional import (
    SourceRunningStandardizer,
    TemporalFunctionalLift,
    TemporalFunctionalOutput,
)
from .temporal_geometry import PhaseTangentOutput, warp_to_identity_tangent
from .temporal_head import SharedTrendStructureLTAE
from .temporal_module import TrendStructureTemporalModule
from .temporal_srvf import (
    SourceRunningSupportScale,
    TemporalSRVFExtractor,
    TemporalSRVFOutput,
)

__all__ = [
    "QUANTILE_LEVELS",
    "ContinuousTime2Vec",
    "ContributionDiagnostics",
    "ConfirmedPhaseView",
    "DecompositionDiagnostics",
    "DecompositionOutput",
    "DomainPhaseConfig",
    "DomainPhaseState",
    "DiagnosticMoments",
    "DiagnosticStat",
    "FdasrsfCurveRegistrationAdapter",
    "FeatureSnapshotConfig",
    "FeatureSnapshotManager",
    "FunctionalGeometryOutput",
    "GammaDiagnostics",
    "GammaLegalityOutput",
    "PairwisePhaseCandidate",
    "PhaseHypothesisScanConfig",
    "PhaseClassCenter",
    "PhaseGroup",
    "PhaseGroupStatus",
    "PhaseTangentOutput",
    "RawTemporalRepresentation",
    "RegistrationGeometryOutput",
    "SharedTrendStructureLTAE",
    "SnapshotCaptureResult",
    "SourceClassificationTrainer",
    "SourcePrototypeBank",
    "SourceRegistrationPrototypeBank",
    "SourceRunningStandardizer",
    "SourceRunningSupportScale",
    "SourceTrainStepOutput",
    "StableLabelConfig",
    "StableTargetCandidate",
    "StableTargetLabel",
    "StableTargetLabelScanResult",
    "Stage1LossOutput",
    "Stage1Objective",
    "Stage2EMATeacher",
    "Stage2ParameterPolicy",
    "StructureBackbone",
    "StructureBackboneOutput",
    "SupportAwareDistanceOutput",
    "SymmetricTimeKernelDecomposition",
    "TSStructureForwardOutput",
    "TSStructureModel",
    "TargetClassPhaseHypothesis",
    "TargetGeometryCache",
    "TargetHypothesisScanResult",
    "TemporalFunctionalLift",
    "TemporalFunctionalOutput",
    "TemporalSRVFExtractor",
    "TemporalSRVFOutput",
    "TrendStructureSharedLTAE",
    "TrendStructureTemporalModule",
    "build_source_prototype_bank",
    "build_source_registration_prototypes",
    "build_confirmed_class_to_group_map",
    "build_confirmed_phase_view",
    "check_gamma_legality",
    "compute_decomposition_diagnostics",
    "compute_gamma_diagnostics",
    "compute_structure_contribution_diagnostics",
    "configure_stage2_parameter_policy",
    "create_feature_snapshot_manager",
    "deterministic_class_selection",
    "empirical_cdf",
    "evaluate_registration_geometry",
    "evaluate_stable_target_candidate",
    "finalize_distance_statistics",
    "gamma_to_psi",
    "load_selected_samples",
    "merge_contribution_diagnostics",
    "merge_decomposition_diagnostics",
    "pairwise_phase_distances",
    "phase_distance",
    "resample_gamma",
    "scan_target_class_phase_hypotheses",
    "scan_stable_target_labels",
    "shape_distance_to_prototype",
    "sqrt_mean_gamma",
    "sqrt_median_gamma",
    "summarize_contribution_diagnostics",
    "summarize_decomposition_diagnostics",
    "support_aware_q_distance",
    "warp_q_gamma",
    "warp_support_gamma",
    "warp_to_identity_tangent",
    "update_domain_phase_state",
    "align_target_positions_to_source",
]
