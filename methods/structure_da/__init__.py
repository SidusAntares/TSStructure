"""Current structure-aware domain-adaptation building blocks."""

from .backbone import StructureBackbone, StructureBackboneOutput
from .channel_module import (
    ChannelStructureOutput,
    ChannelStructurePairOutput,
    MultiScaleChannelRelationStructure,
    SharedChannelStructureOperator,
    SourceRunningAttributeStandardizer,
    SourceRunningRelationEnergyScale,
)
from .decomposition import DecompositionOutput, SymmetricTimeKernelDecomposition
from .eden_alignment import (
    EDENDomainAlignmentOutput,
    EDENDomainDiscriminator,
    EDENFusedFeatureAlignment,
    WarmStartGradientReverseLayer,
)
from .full_model import (
    StructureAwareDomainAdaptationModel,
    StructureAwareForwardOutput,
    StructureAwareGeometryOutput,
)
from .joint_trainer import (
    JointStructureDADiagnostics,
    JointStructureDALossOutput,
    JointStructureDATrainStepOutput,
    JointStructureDATrainingConfig,
    create_joint_structure_da_train_loaders,
    joint_structure_da_train_step,
    resolve_domain_score_weight,
    train_joint_structure_da,
)
from .quality_fusion import (
    ComponentQualityBundle,
    HierarchicalQualityFusion,
    HierarchicalQualityObjective,
    HierarchicalQualityOutput,
    QualityLossOutput,
    QualityScoreOutput,
    QualityScorer,
    StructuralQualityBundle,
    concatenate_hierarchical_quality_outputs,
)
from .representation import (
    PairedStructureFeatures,
    QualityAwareClassifierOutput,
    QualityAwareComponentClassifier,
)
from .temporal_coordinates import (
    TemporalCoordinateOutput,
    TemporalShapePhaseCoordinates,
)
from .temporal_functional import (
    SourceRunningStandardizer,
    TemporalFunctionalLift,
    TemporalFunctionalOutput,
)
from .temporal_geometry import (
    PhaseTangentOutput,
    TemporalGeometryLossOutput,
    TemporalGeometryObjective,
    warp_to_identity_tangent,
)
from .temporal_head import (
    PhaseCoordinateEncoder,
    ShapeCoordinateEncoder,
    TemporalStructureEncoder,
    TemporalStructureFeatureOutput,
    TemporalStructureOutputHead,
)
from .temporal_module import (
    SharedTemporalStructureOperator,
    TemporalGeometryForwardOutput,
    TemporalGeometryPairOutput,
    TemporalStructureExtractor,
    TemporalStructureOutput,
    TemporalStructurePairOutput,
)
from .temporal_registration import (
    MonotoneWarpEstimator,
    MonotoneWarpOutput,
    SourceRunningSRVFTemplate,
    SourceSRVFTemplateOutput,
    TemporalRegistrationOutput,
    TemporalSRVFRegistration,
)
from .temporal_srvf import (
    SourceRunningSupportScale,
    TemporalSRVFExtractor,
    TemporalSRVFOutput,
)

__all__ = [
    "ChannelStructureOutput",
    "ChannelStructurePairOutput",
    "ComponentQualityBundle",
    "DecompositionOutput",
    "EDENDomainAlignmentOutput",
    "EDENDomainDiscriminator",
    "EDENFusedFeatureAlignment",
    "HierarchicalQualityFusion",
    "HierarchicalQualityObjective",
    "HierarchicalQualityOutput",
    "JointStructureDALossOutput",
    "JointStructureDADiagnostics",
    "JointStructureDATrainStepOutput",
    "JointStructureDATrainingConfig",
    "MonotoneWarpEstimator",
    "MonotoneWarpOutput",
    "MultiScaleChannelRelationStructure",
    "PairedStructureFeatures",
    "PhaseCoordinateEncoder",
    "PhaseTangentOutput",
    "QualityAwareClassifierOutput",
    "QualityAwareComponentClassifier",
    "QualityLossOutput",
    "QualityScoreOutput",
    "QualityScorer",
    "ShapeCoordinateEncoder",
    "SharedChannelStructureOperator",
    "SharedTemporalStructureOperator",
    "SourceRunningAttributeStandardizer",
    "SourceRunningRelationEnergyScale",
    "SourceRunningSRVFTemplate",
    "SourceRunningStandardizer",
    "SourceRunningSupportScale",
    "SourceSRVFTemplateOutput",
    "StructuralQualityBundle",
    "StructureAwareDomainAdaptationModel",
    "StructureAwareForwardOutput",
    "StructureAwareGeometryOutput",
    "StructureBackbone",
    "StructureBackboneOutput",
    "SymmetricTimeKernelDecomposition",
    "TemporalCoordinateOutput",
    "TemporalFunctionalLift",
    "TemporalFunctionalOutput",
    "TemporalGeometryForwardOutput",
    "TemporalGeometryLossOutput",
    "TemporalGeometryObjective",
    "TemporalGeometryPairOutput",
    "TemporalRegistrationOutput",
    "TemporalSRVFExtractor",
    "TemporalSRVFOutput",
    "TemporalSRVFRegistration",
    "TemporalShapePhaseCoordinates",
    "TemporalStructureEncoder",
    "TemporalStructureExtractor",
    "TemporalStructureFeatureOutput",
    "TemporalStructureOutput",
    "TemporalStructureOutputHead",
    "TemporalStructurePairOutput",
    "WarmStartGradientReverseLayer",
    "concatenate_hierarchical_quality_outputs",
    "create_joint_structure_da_train_loaders",
    "joint_structure_da_train_step",
    "resolve_domain_score_weight",
    "train_joint_structure_da",
    "warp_to_identity_tangent",
]
