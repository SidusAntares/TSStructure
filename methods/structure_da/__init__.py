"""Building blocks for structure-aware domain adaptation."""

from .adaptation import (
    JointStructuralOutput,
    JointStructuralSpaceBuilder,
    SDADiscriminator,
    StructuralAdversarialAdapter,
    StructuralAdversarialOutput,
    gradient_reverse,
)
from .backbone import (
    StructureBackbone,
    StructureBackboneOutput,
)
from .decomposition import DecompositionOutput, SymmetricTimeKernelDecomposition
from .losses import (
    LossWeights,
    StructureDALosses,
    classification_loss,
    component_diversity_loss,
    compose_total_loss,
    quality_classification_loss,
    quality_domain_loss,
    structural_adversarial_loss,
)
from .method import StructureDAForwardOutput, StructureDAModel
from .model import (
    ComponentLTAEInputs,
    ComponentQualityBundle,
    ComponentStructureClassifier,
    ComponentStructureOutput,
    EffectiveQualityGates,
    StructuralQualityBundle,
)
from .quality import (
    ComponentQualityOutput,
    ComponentQualityPerception,
    DiscriminabilityScorer,
    DiversityScorer,
    QualityScores,
    StructuralQualityOutput,
    StructuralQualityPerception,
    TransferabilityScorer,
)
from .schedules import (
    apply_quality_warmup,
    grl_coefficient,
    grl_progress,
    quality_gate_progress,
)
from .structure_ops import (
    ChannelRelationOperator,
    StructureOutput,
    TemporalRelationOperator,
    vectorize_channel_statistic,
)
from .temporal_functional import (
    SourceRunningStandardizer,
    TemporalFunctionalLift,
    TemporalFunctionalOutput,
)
from .temporal_srvf import (
    SourceRunningSupportScale,
    TemporalSRVFExtractor,
    TemporalSRVFOutput,
)
from .trainer import (
    ResolvedStructureDATraining,
    StructureDATrainingConfig,
    StructureDATrainStepOutput,
    create_structure_da_train_loaders,
    resolve_structure_da_training,
    structure_da_train_step,
    train_structure_da,
)

__all__ = [
    "ChannelRelationOperator",
    "ComponentLTAEInputs",
    "ComponentQualityBundle",
    "ComponentQualityOutput",
    "ComponentQualityPerception",
    "ComponentStructureClassifier",
    "ComponentStructureOutput",
    "DecompositionOutput",
    "DiscriminabilityScorer",
    "DiversityScorer",
    "EffectiveQualityGates",
    "JointStructuralOutput",
    "JointStructuralSpaceBuilder",
    "LossWeights",
    "QualityScores",
    "StructureOutput",
    "StructureDALosses",
    "StructureBackbone",
    "StructureBackboneOutput",
    "StructureDAForwardOutput",
    "StructureDAModel",
    "ResolvedStructureDATraining",
    "StructureDATrainingConfig",
    "StructureDATrainStepOutput",
    "SDADiscriminator",
    "SourceRunningStandardizer",
    "SourceRunningSupportScale",
    "StructuralAdversarialAdapter",
    "StructuralAdversarialOutput",
    "StructuralQualityOutput",
    "StructuralQualityPerception",
    "StructuralQualityBundle",
    "SymmetricTimeKernelDecomposition",
    "TemporalFunctionalLift",
    "TemporalFunctionalOutput",
    "TemporalSRVFExtractor",
    "TemporalSRVFOutput",
    "TemporalRelationOperator",
    "TransferabilityScorer",
    "apply_quality_warmup",
    "classification_loss",
    "component_diversity_loss",
    "compose_total_loss",
    "grl_coefficient",
    "grl_progress",
    "gradient_reverse",
    "quality_gate_progress",
    "quality_classification_loss",
    "quality_domain_loss",
    "structural_adversarial_loss",
    "vectorize_channel_statistic",
    "create_structure_da_train_loaders",
    "resolve_structure_da_training",
    "structure_da_train_step",
    "train_structure_da",
]
