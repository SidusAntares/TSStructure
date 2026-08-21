"""Training-free raw time-series decomposition utilities for visual diagnostics."""

from .methods import (
    ALL_METHODS,
    Component,
    DecompositionConfig,
    DecompositionResult,
    OptionalDependencyError,
    decompose,
    method_capabilities,
)

__all__ = [
    "ALL_METHODS",
    "Component",
    "DecompositionConfig",
    "DecompositionResult",
    "OptionalDependencyError",
    "decompose",
    "method_capabilities",
]
