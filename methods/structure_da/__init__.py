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
from .prototype_bank import (
    QUANTILE_LEVELS,
    SourcePrototypeBank,
    SupportAwareDistanceOutput,
    support_aware_q_distance,
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
from .registration_geometry import (
    RegistrationGeometryOutput,
    SourceRegistrationPrototypeBank,
    TargetGeometryCache,
    evaluate_registration_geometry,
)
from .phase_registration import (
    FdasrsfDP2RegistrationAdapter,
    GammaLegalityOutput,
    build_source_registration_prototypes,
    check_gamma_legality,
    resample_gamma,
    warp_q_gamma,
    warp_support_gamma,
)
from .phase_evidence import (
    GammaDiagnostics,
    PairwisePhaseCandidate,
    compute_gamma_diagnostics,
    empirical_cdf,
    shape_distance_to_prototype,
)
from .target_hypothesis_scan import (
    PhaseHypothesisScanConfig,
    TargetClassPhaseHypothesis,
    TargetHypothesisScanResult,
    scan_target_class_phase_hypotheses,
)

__all__ = [
    "ContinuousTime2Vec",
    "ContributionDiagnostics",
    "DecompositionDiagnostics",
    "DecompositionOutput",
    "DiagnosticMoments",
    "DiagnosticStat",
    "FdasrsfDP2RegistrationAdapter",
    "FeatureSnapshotConfig",
    "FeatureSnapshotManager",
    "FunctionalGeometryOutput",
    "GammaDiagnostics",
    "GammaLegalityOutput",
    "PairwisePhaseCandidate",
    "PhaseHypothesisScanConfig",
    "PhaseTangentOutput",
    "QUANTILE_LEVELS",
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
    "Stage1LossOutput",
    "Stage1Objective",
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
    "check_gamma_legality",
    "compute_decomposition_diagnostics",
    "compute_gamma_diagnostics",
    "compute_structure_contribution_diagnostics",
    "create_feature_snapshot_manager",
    "deterministic_class_selection",
    "empirical_cdf",
    "evaluate_registration_geometry",
    "finalize_distance_statistics",
    "load_selected_samples",
    "merge_contribution_diagnostics",
    "merge_decomposition_diagnostics",
    "resample_gamma",
    "scan_target_class_phase_hypotheses",
    "shape_distance_to_prototype",
    "summarize_contribution_diagnostics",
    "summarize_decomposition_diagnostics",
    "support_aware_q_distance",
    "warp_q_gamma",
    "warp_support_gamma",
    "warp_to_identity_tangent",
]
