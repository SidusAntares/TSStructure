"""Building blocks for structure-aware domain adaptation."""

from .decomposition import DecompositionOutput, SymmetricTimeKernelDecomposition
from .model import (
    ComponentLTAEInputs,
    ComponentStructureClassifier,
    ComponentStructureOutput,
)
from .structure_ops import (
    ChannelRelationOperator,
    StructureOutput,
    TemporalRelationOperator,
    vectorize_channel_statistic,
)

__all__ = [
    "ChannelRelationOperator",
    "ComponentLTAEInputs",
    "ComponentStructureClassifier",
    "ComponentStructureOutput",
    "DecompositionOutput",
    "StructureOutput",
    "SymmetricTimeKernelDecomposition",
    "TemporalRelationOperator",
    "vectorize_channel_statistic",
]
