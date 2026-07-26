"""Building blocks for structure-aware domain adaptation."""

from .decomposition import DecompositionOutput, SymmetricTimeKernelDecomposition
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
from .schedules import apply_quality_warmup, quality_gate_progress
from .structure_ops import (
    ChannelRelationOperator,
    StructureOutput,
    TemporalRelationOperator,
    vectorize_channel_statistic,
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
    "QualityScores",
    "StructureOutput",
    "StructuralQualityOutput",
    "StructuralQualityPerception",
    "StructuralQualityBundle",
    "SymmetricTimeKernelDecomposition",
    "TemporalRelationOperator",
    "TransferabilityScorer",
    "apply_quality_warmup",
    "quality_gate_progress",
    "vectorize_channel_statistic",
]
