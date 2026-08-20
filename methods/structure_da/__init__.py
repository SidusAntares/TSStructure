"""TimeMatch semantic path with an optional frozen temporal-geometry branch."""

from .decomposition import DecompositionOutput, SymmetricTimeKernelDecomposition
from .original_timematch import FrozenGeometryCopy, OriginalTimeMatchModel
from .phase_evidence import PairwisePhaseCandidate, empirical_cdf, shape_distance_to_prototype
from .phase_registration import (
    FdasrsfCurveRegistrationAdapter,
    check_gamma_legality,
    resample_gamma,
    warp_q_gamma,
    warp_support_gamma,
)
from .prototype_bank import SourcePrototypeBank, SupportAwareDistanceOutput, support_aware_q_distance
from .registration_geometry import (
    RegistrationGeometryOutput,
    SourceRegistrationPrototypeBank,
    TargetGeometryCache,
    evaluate_registration_geometry,
)
from .temporal_functional import SourceRunningStandardizer, TemporalFunctionalLift, TemporalFunctionalOutput
from .temporal_srvf import SourceRunningSupportScale, TemporalSRVFExtractor, TemporalSRVFOutput
from .timematch_nonlinear_phase import (
    ALPHA_BANK,
    candidate_phase_grid,
    requires_nonlinear_phase,
)

__all__ = [
    "ALPHA_BANK",
    "DecompositionOutput",
    "FrozenGeometryCopy",
    "OriginalTimeMatchModel",
    "FdasrsfCurveRegistrationAdapter",
    "PairwisePhaseCandidate",
    "RegistrationGeometryOutput",
    "SourcePrototypeBank",
    "SourceRegistrationPrototypeBank",
    "SourceRunningStandardizer",
    "SourceRunningSupportScale",
    "SupportAwareDistanceOutput",
    "SymmetricTimeKernelDecomposition",
    "TargetGeometryCache",
    "TemporalFunctionalLift",
    "TemporalFunctionalOutput",
    "TemporalSRVFExtractor",
    "TemporalSRVFOutput",
    "candidate_phase_grid",
    "check_gamma_legality",
    "empirical_cdf",
    "evaluate_registration_geometry",
    "resample_gamma",
    "requires_nonlinear_phase",
    "shape_distance_to_prototype",
    "support_aware_q_distance",
    "warp_q_gamma",
    "warp_support_gamma",
]
