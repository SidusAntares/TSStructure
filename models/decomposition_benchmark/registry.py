from typing import Tuple

from models.stclassifier import PseLTae

from .ceemdan import CeemdanSeDecompositionClassifier
from .dlinear import DLinearDecompositionClassifier
from .dwt import DwtDecompositionClassifier
from .micn import MicnDecompositionClassifier
from .stl import StlDecompositionClassifier
from .vmd import VmdDecompositionClassifier
from .xpatch import XPatchEmaDecompositionClassifier


IMPLEMENTED_BENCHMARK_MODELS: Tuple[str, ...] = (
    "raw_timematch",
    "dlinear_decomp",
    "micn_decomp",
    "xpatch_ema",
    "stl",
    "vmd",
    "dwt",
    "ceemdan_se",
)

# Frozen first-round benchmark plan. Entries are added to IMPLEMENTED_BENCHMARK_MODELS
# only after their method-specific temporal interface and fusion have been implemented.
PLANNED_BENCHMARK_MODELS: Tuple[str, ...] = (
    "raw_timematch",
    "dlinear_decomp",
    "micn_decomp",
    "xpatch_ema",
    "stl",
    "vmd",
    "dwt",
    "ceemdan_se",
    "timekan",
    "sscnn",
    "timemixer",
    "amd",
    "etsformer",
)


def is_benchmark_model(name: str) -> bool:
    return name in IMPLEMENTED_BENCHMARK_MODELS


def build_benchmark_model(
    name: str,
    input_dim: int,
    num_classes: int,
    with_extra: bool,
    dlinear_kernel: int = 25,
    micn_conv_kernels=(17, 49),
    xpatch_ema_alpha: float = 0.3,
    stl_period: int = 7,
    dwt_wavelet: str = "haar",
    dwt_level: int = 2,
    vmd_num_modes: int = 5,
    vmd_alpha: float = 2000.0,
    vmd_tau: float = 0.0,
    vmd_dc: int = 0,
    vmd_init: int = 1,
    vmd_tol: float = 1e-7,
    ceemdan_trials: int = 100,
    ceemdan_epsilon: float = 0.005,
    ceemdan_seed: int = 1,
    ceemdan_sampen_m: int = 2,
    ceemdan_sampen_r_ratio: float = 0.2,
    ceemdan_se_threshold_factor: float = 0.5,
):
    model_kwargs = {
        "input_dim": input_dim,
        "mlp1": [input_dim, 32, 64],
        "num_classes": num_classes,
        "with_extra": with_extra,
    }
    if name == "raw_timematch":
        return PseLTae(**model_kwargs)
    if name == "dlinear_decomp":
        return DLinearDecompositionClassifier(
            kernel_size=dlinear_kernel,
            **model_kwargs,
        )
    if name == "micn_decomp":
        return MicnDecompositionClassifier(
            conv_kernels=micn_conv_kernels,
            **model_kwargs,
        )
    if name == "xpatch_ema":
        return XPatchEmaDecompositionClassifier(
            alpha=xpatch_ema_alpha,
            **model_kwargs,
        )
    if name == "stl":
        return StlDecompositionClassifier(
            period=stl_period,
            **model_kwargs,
        )
    if name == "vmd":
        return VmdDecompositionClassifier(
            num_modes=vmd_num_modes,
            alpha=vmd_alpha,
            tau=vmd_tau,
            dc=vmd_dc,
            init=vmd_init,
            tol=vmd_tol,
            **model_kwargs,
        )
    if name == "dwt":
        return DwtDecompositionClassifier(
            wavelet=dwt_wavelet,
            level=dwt_level,
            **model_kwargs,
        )
    if name == "ceemdan_se":
        return CeemdanSeDecompositionClassifier(
            trials=ceemdan_trials,
            epsilon=ceemdan_epsilon,
            noise_seed=ceemdan_seed,
            sampen_m=ceemdan_sampen_m,
            sampen_r_ratio=ceemdan_sampen_r_ratio,
            threshold_factor=ceemdan_se_threshold_factor,
            **model_kwargs,
        )
    raise ValueError(
        f"Unknown or not-yet-implemented decomposition benchmark model: {name}. "
        f"Implemented: {IMPLEMENTED_BENCHMARK_MODELS}"
    )
