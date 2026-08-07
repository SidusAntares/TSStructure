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

__all__ = [
    "ContinuousTime2Vec",
    "ContributionDiagnostics",
    "DecompositionDiagnostics",
    "DecompositionOutput",
    "DiagnosticMoments",
    "DiagnosticStat",
    "FeatureSnapshotConfig",
    "FeatureSnapshotManager",
    "FunctionalGeometryOutput",
    "PhaseTangentOutput",
    "QUANTILE_LEVELS",
    "RawTemporalRepresentation",
    "SharedTrendStructureLTAE",
    "SnapshotCaptureResult",
    "SourceClassificationTrainer",
    "SourcePrototypeBank",
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
    "TemporalFunctionalLift",
    "TemporalFunctionalOutput",
    "TemporalSRVFExtractor",
    "TemporalSRVFOutput",
    "TrendStructureSharedLTAE",
    "TrendStructureTemporalModule",
    "build_source_prototype_bank",
    "compute_decomposition_diagnostics",
    "compute_structure_contribution_diagnostics",
    "create_feature_snapshot_manager",
    "deterministic_class_selection",
    "finalize_distance_statistics",
    "load_selected_samples",
    "merge_contribution_diagnostics",
    "merge_decomposition_diagnostics",
    "summarize_contribution_diagnostics",
    "summarize_decomposition_diagnostics",
    "support_aware_q_distance",
    "warp_to_identity_tangent",
]
