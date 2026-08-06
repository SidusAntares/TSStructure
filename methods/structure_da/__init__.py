"""Two-stage structure model: source-only CE backbone for Round 1."""

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
from .representation import (
    FunctionalGeometryOutput,
    RawTemporalRepresentation,
    TSStructureForwardOutput,
)
from .source_trainer import SourceClassificationTrainer, SourceTrainStepOutput
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
    "RawTemporalRepresentation",
    "SharedTrendStructureLTAE",
    "SnapshotCaptureResult",
    "SourceClassificationTrainer",
    "SourceRunningStandardizer",
    "SourceRunningSupportScale",
    "SourceTrainStepOutput",
    "StructureBackbone",
    "StructureBackboneOutput",
    "SymmetricTimeKernelDecomposition",
    "TSStructureForwardOutput",
    "TSStructureModel",
    "TemporalFunctionalLift",
    "TemporalFunctionalOutput",
    "TemporalSRVFExtractor",
    "TemporalSRVFOutput",
    "TrendStructureSharedLTAE",
    "TrendStructureTemporalModule",
    "compute_decomposition_diagnostics",
    "compute_structure_contribution_diagnostics",
    "create_feature_snapshot_manager",
    "deterministic_class_selection",
    "load_selected_samples",
    "merge_contribution_diagnostics",
    "merge_decomposition_diagnostics",
    "summarize_contribution_diagnostics",
    "summarize_decomposition_diagnostics",
    "warp_to_identity_tangent",
]
