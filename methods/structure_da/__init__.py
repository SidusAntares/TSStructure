"""Building blocks for structure-aware domain adaptation."""

from .decomposition import DecompositionOutput, SymmetricTimeKernelDecomposition
from .structure_ops import (
    ChannelRelationOperator,
    StructureOutput,
    TemporalRelationOperator,
    vectorize_channel_statistic,
)

__all__ = [
    "ChannelRelationOperator",
    "DecompositionOutput",
    "StructureOutput",
    "SymmetricTimeKernelDecomposition",
    "TemporalRelationOperator",
    "vectorize_channel_statistic",
]
