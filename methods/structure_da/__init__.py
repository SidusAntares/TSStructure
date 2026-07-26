"""Building blocks for structure-aware domain adaptation."""

from .decomposition import DecompositionOutput, SymmetricTimeKernelDecomposition
from .model import (
    ComponentLTAEInputs,
    ComponentStructureClassifier,
    ComponentStructureOutput,
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
    "ComponentQualityOutput",
    "ComponentQualityPerception",
    "ComponentStructureClassifier",
    "ComponentStructureOutput",
    "DecompositionOutput",
    "DiscriminabilityScorer",
    "DiversityScorer",
    "QualityScores",
    "StructureOutput",
    "StructuralQualityOutput",
    "StructuralQualityPerception",
    "SymmetricTimeKernelDecomposition",
    "TemporalRelationOperator",
    "TransferabilityScorer",
    "apply_quality_warmup",
    "quality_gate_progress",
    "vectorize_channel_statistic",
]
