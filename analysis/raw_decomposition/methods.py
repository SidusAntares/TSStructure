"""Training-free decomposition methods used by the raw TimeMatch visual audit.

The module deliberately does not interpolate or impute.  Methods that were
originally designed for regularly sampled sequences are applied to the ordered
observation sequence exactly as supplied; their metadata marks that their
frequency/scale semantics are observation-index based.  Lomb--Scargle is the
one method in this registry that explicitly uses the physical acquisition
positions when estimating frequencies.

Each function accepts one scalar raw time series.  Multi-band TimeMatch parcels
are reduced spatially and split into scalar band/vegetation-index series by the
CLI in ``scripts/visualize_raw_decompositions.py`` before arriving here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ALL_METHODS: Tuple[str, ...] = (
    "autoformer",
    "fedformer",
    "dlinear",
    "micn",
    "timemixer_ma",
    "timemixer_dft",
    "xpatch_ema",
    "stl",
    "emd",
    "ceemdan",
    "vmd",
    "wavelet",
    "ssa",
    "fourier",
    "lomb_scargle",
)


class OptionalDependencyError(RuntimeError):
    """Raised when a selected decomposition backend is not installed."""


@dataclass(frozen=True)
class Component:
    """One decomposition component and the x coordinates used for plotting."""

    name: str
    values: np.ndarray
    positions: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        positions = np.asarray(self.positions, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"component {self.name!r} values must be one-dimensional")
        if positions.ndim != 1:
            raise ValueError(f"component {self.name!r} positions must be one-dimensional")
        if values.shape != positions.shape:
            raise ValueError(
                f"component {self.name!r} values/positions shape mismatch: "
                f"{values.shape} vs {positions.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"component {self.name!r} contains non-finite values")
        if not np.all(np.isfinite(positions)):
            raise ValueError(f"component {self.name!r} contains non-finite positions")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True)
class DecompositionResult:
    method: str
    components: Tuple[Component, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("a decomposition result must contain at least one component")


@dataclass(frozen=True)
class DecompositionConfig:
    """Defaults chosen to mirror the cited decomposition blocks where possible.

    They are visualization defaults, not tuned TimeMatch hyperparameters.  Every
    value is serialized into the output manifest so later sweeps are auditable.
    """

    autoformer_kernel: int = 25
    fedformer_kernel: int = 25
    dlinear_kernel: int = 25
    micn_conv_kernels: Tuple[int, ...] = (12, 16)
    timemixer_kernel: int = 25
    timemixer_downsample_window: int = 2
    timemixer_downsample_layers: int = 2
    timemixer_dft_top_k: int = 5
    xpatch_alpha: float = 0.30
    stl_period: int = 12
    stl_robust: bool = True
    emd_max_imf: int = 5
    ceemdan_max_imf: int = 5
    ceemdan_trials: int = 50
    random_seed: int = 1
    vmd_alpha: float = 2000.0
    vmd_tau: float = 0.0
    vmd_k: int = 5
    vmd_dc: int = 0
    vmd_init: int = 1
    vmd_tol: float = 1e-7
    wavelet_name: str = "db4"
    wavelet_level: int = 3
    ssa_window: int = 0  # 0 -> floor(T / 3), clipped to [2, T-1]
    ssa_components: int = 5
    fourier_low_fraction: float = 0.15
    fourier_mid_fraction: float = 0.40
    lomb_top_k: int = 3
    lomb_grid_size: int = 1024
    lomb_min_separation_bins: int = 8


_METHOD_CAPABILITIES: Dict[str, Dict[str, object]] = {
    "autoformer": {
        "paper": "Autoformer (NeurIPS 2021)",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": None,
    },
    "fedformer": {
        "paper": "FEDformer (ICML 2022)",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": None,
    },
    "dlinear": {
        "paper": "DLinear (AAAI 2023)",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": None,
    },
    "micn": {
        "paper": "MICN (ICLR 2023)",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": None,
    },
    "timemixer_ma": {
        "paper": "TimeMixer (ICLR 2024), moving-average decomposition",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": None,
    },
    "timemixer_dft": {
        "paper": "TimeMixer (ICLR 2024), DFT decomposition",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": None,
    },
    "xpatch_ema": {
        "paper": "xPatch (AAAI 2025)",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": None,
    },
    "stl": {
        "paper": "STL",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": "statsmodels",
    },
    "emd": {
        "paper": "Empirical Mode Decomposition",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": "EMD-signal (import PyEMD)",
    },
    "ceemdan": {
        "paper": "CEEMDAN",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": "EMD-signal (import PyEMD)",
    },
    "vmd": {
        "paper": "Variational Mode Decomposition",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": "vmdpy",
    },
    "wavelet": {
        "paper": "Discrete Wavelet Transform",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": "PyWavelets (import pywt)",
    },
    "ssa": {
        "paper": "Singular Spectrum Analysis",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": None,
    },
    "fourier": {
        "paper": "Fourier spectral decomposition",
        "requires_training": False,
        "uses_physical_time": False,
        "optional_dependency": None,
    },
    "lomb_scargle": {
        "paper": "Lomb-Scargle irregular spectral decomposition",
        "requires_training": False,
        "uses_physical_time": True,
        "optional_dependency": "scipy",
    },
}


def method_capabilities(method: Optional[str] = None):
    if method is None:
        return {name: dict(payload) for name, payload in _METHOD_CAPABILITIES.items()}
    if method not in _METHOD_CAPABILITIES:
        raise KeyError(method)
    return dict(_METHOD_CAPABILITIES[method])


def _validate_input(values: np.ndarray, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(positions, dtype=np.float64)
    if x.ndim != 1 or t.ndim != 1 or x.shape != t.shape:
        raise ValueError("values and positions must share one-dimensional shape [T]")
    if x.size < 4:
        raise ValueError("decomposition requires at least four observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("input series contains NaN/Inf; no imputation is performed")
    if not np.all(np.isfinite(t)):
        raise ValueError("input positions contain NaN/Inf")
    if not np.all(np.diff(t) > 0):
        raise ValueError("input positions must be strictly increasing")
    return x, t


def _odd_kernel(kernel: int) -> int:
    kernel = int(kernel)
    if kernel < 1:
        raise ValueError("moving-average kernel must be positive")
    if kernel % 2 == 0:
        raise ValueError("moving-average kernel must be odd for this series_decomp implementation")
    return kernel


def _series_decomp(values: np.ndarray, kernel: int) -> Tuple[np.ndarray, np.ndarray]:
    """Autoformer/DLinear edge-replicated moving average on observation index."""
    kernel = _odd_kernel(kernel)
    half = (kernel - 1) // 2
    padded = np.pad(values, (half, half), mode="edge")
    weights = np.full(kernel, 1.0 / kernel, dtype=np.float64)
    trend = np.convolve(padded, weights, mode="valid")
    seasonal = values - trend
    if trend.shape != values.shape:
        raise RuntimeError("internal moving-average output length mismatch")
    return seasonal, trend


def _multi_series_decomp(values: np.ndarray, kernels: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    kernels = tuple(_odd_kernel(k) for k in kernels)
    if not kernels:
        raise ValueError("multi-window decomposition requires at least one kernel")
    pairs = [_series_decomp(values, k) for k in kernels]
    seasonal = np.mean(np.stack([pair[0] for pair in pairs], axis=0), axis=0)
    trend = np.mean(np.stack([pair[1] for pair in pairs], axis=0), axis=0)
    return seasonal, trend


def _simple_result(method: str, positions: np.ndarray, named_values, metadata) -> DecompositionResult:
    components = tuple(
        Component(name=name, values=np.asarray(component), positions=positions)
        for name, component in named_values
    )
    return DecompositionResult(method=method, components=components, metadata=metadata)


def _autoformer(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    seasonal, trend = _series_decomp(values, cfg.autoformer_kernel)
    return _simple_result(
        "autoformer",
        positions,
        (("seasonal", seasonal), ("trend", trend)),
        {
            "kernel": cfg.autoformer_kernel,
            "operator": "edge-replicated moving average series_decomp",
            "time_semantics": "observation_index",
        },
    )


def _fedformer(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    seasonal, trend = _series_decomp(values, cfg.fedformer_kernel)
    return _simple_result(
        "fedformer",
        positions,
        (("seasonal", seasonal), ("trend", trend)),
        {
            "kernel": cfg.fedformer_kernel,
            "operator": "FEDformer training-free single-window series_decomp",
            "time_semantics": "observation_index",
            "note": (
                "FEDformer's original multi-window series_decomp_multi contains a "
                "learnable Linear+Softmax mixer and therefore is deliberately not "
                "used in this no-training visualization. With the same kernel, this "
                "single-window operator is mathematically the same moving-average "
                "decomposition used by Autoformer."
            ),
        },
    )


def _dlinear(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    seasonal, trend = _series_decomp(values, cfg.dlinear_kernel)
    return _simple_result(
        "dlinear",
        positions,
        (("remainder", seasonal), ("trend", trend)),
        {
            "kernel": cfg.dlinear_kernel,
            "operator": "edge-replicated moving average",
            "time_semantics": "observation_index",
        },
    )


def _micn(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    decomp_kernels = tuple(k + 1 if k % 2 == 0 else k for k in cfg.micn_conv_kernels)
    seasonal, trend = _multi_series_decomp(values, decomp_kernels)
    return _simple_result(
        "micn",
        positions,
        (("seasonal_init", seasonal), ("trend", trend)),
        {
            "conv_kernels": list(cfg.micn_conv_kernels),
            "decomp_kernels": list(decomp_kernels),
            "operator": "MICN multi-scale hybrid decomposition front-end only",
            "time_semantics": "observation_index",
        },
    )


def _downsample_average(values: np.ndarray, positions: np.ndarray, window: int):
    window = int(window)
    if window <= 1:
        raise ValueError("TimeMixer downsample window must be > 1")
    n = (values.size // window) * window
    if n < window:
        return None
    reduced_values = values[:n].reshape(-1, window).mean(axis=1)
    # Positions are only for plotting.  TimeMixer itself uses ordered index scale.
    reduced_positions = positions[:n].reshape(-1, window).mean(axis=1)
    return reduced_values, reduced_positions


def _timemixer_scales(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig):
    scales = [(values, positions)]
    current_values = values
    current_positions = positions
    for _ in range(int(cfg.timemixer_downsample_layers)):
        reduced = _downsample_average(
            current_values, current_positions, cfg.timemixer_downsample_window
        )
        if reduced is None or reduced[0].size < 4:
            break
        current_values, current_positions = reduced
        scales.append((current_values, current_positions))
    return scales


def _timemixer_ma(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    components = []
    scale_lengths = []
    for scale_id, (scale_values, scale_positions) in enumerate(
        _timemixer_scales(values, positions, cfg)
    ):
        seasonal, trend = _series_decomp(scale_values, cfg.timemixer_kernel)
        components.extend(
            (
                Component(f"scale{scale_id}_seasonal", seasonal, scale_positions),
                Component(f"scale{scale_id}_trend", trend, scale_positions),
            )
        )
        scale_lengths.append(int(scale_values.size))
    return DecompositionResult(
        method="timemixer_ma",
        components=tuple(components),
        metadata={
            "moving_average_kernel": cfg.timemixer_kernel,
            "downsample_window": cfg.timemixer_downsample_window,
            "downsample_layers_requested": cfg.timemixer_downsample_layers,
            "scale_lengths": scale_lengths,
            "operator": "TimeMixer multi-scale input + moving-average decomposition before learned mixing",
            "time_semantics": "observation_index",
            "plot_position_note": "pooled acquisition positions are shown only to locate downsampled points",
        },
    )


def _dft_topk(values: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    top_k = max(1, int(top_k))
    spectrum = np.fft.rfft(values)
    amplitude = np.abs(spectrum)
    if amplitude.size:
        amplitude[0] = 0.0
    candidate = np.arange(1, amplitude.size)
    if candidate.size == 0:
        seasonal = np.zeros_like(values)
        return seasonal, values.copy(), np.array([], dtype=np.int64)
    keep_count = min(top_k, candidate.size)
    selected = candidate[np.argsort(amplitude[candidate])[-keep_count:]]
    mask = np.zeros_like(spectrum, dtype=bool)
    mask[selected] = True
    retained = np.where(mask, spectrum, 0.0)
    seasonal = np.fft.irfft(retained, n=values.size)
    trend = values - seasonal
    return seasonal, trend, np.sort(selected)


def _timemixer_dft(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    components = []
    selected_bins = {}
    scale_lengths = []
    for scale_id, (scale_values, scale_positions) in enumerate(
        _timemixer_scales(values, positions, cfg)
    ):
        seasonal, trend, bins = _dft_topk(scale_values, cfg.timemixer_dft_top_k)
        components.extend(
            (
                Component(f"scale{scale_id}_seasonal_dft", seasonal, scale_positions),
                Component(f"scale{scale_id}_trend", trend, scale_positions),
            )
        )
        selected_bins[f"scale{scale_id}"] = bins.tolist()
        scale_lengths.append(int(scale_values.size))
    return DecompositionResult(
        method="timemixer_dft",
        components=tuple(components),
        metadata={
            "top_k": cfg.timemixer_dft_top_k,
            "selected_frequency_bins": selected_bins,
            "downsample_window": cfg.timemixer_downsample_window,
            "scale_lengths": scale_lengths,
            "operator": "TimeMixer-style top-k temporal DFT decomposition before learned mixing",
            "time_semantics": "observation_index",
        },
    )


def _xpatch_ema(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    alpha = float(cfg.xpatch_alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("xPatch EMA alpha must be in (0, 1]")
    trend = np.empty_like(values)
    trend[0] = values[0]
    for index in range(1, values.size):
        trend[index] = alpha * values[index] + (1.0 - alpha) * trend[index - 1]
    seasonal = values - trend
    return _simple_result(
        "xpatch_ema",
        positions,
        (("seasonal", seasonal), ("ema_trend", trend)),
        {
            "alpha": alpha,
            "operator": "xPatch exponential moving average decomposition",
            "time_semantics": "observation_index",
        },
    )


def _stl(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError as error:
        raise OptionalDependencyError(
            "STL requires statsmodels; install requirements_decomposition_visualization.txt"
        ) from error
    period = int(cfg.stl_period)
    if period < 2:
        raise ValueError("STL period must be >= 2")
    if values.size < 2 * period:
        period = max(2, values.size // 2)
    fitted = STL(values, period=period, robust=bool(cfg.stl_robust)).fit()
    return _simple_result(
        "stl",
        positions,
        (("trend", fitted.trend), ("seasonal", fitted.seasonal), ("residual", fitted.resid)),
        {
            "period": period,
            "robust": bool(cfg.stl_robust),
            "time_semantics": "observation_index",
            "note": "No interpolation is performed; STL sees the ordered observations as equally spaced indices.",
        },
    )


def _emd(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    try:
        from PyEMD import EMD
    except ImportError as error:
        raise OptionalDependencyError(
            "EMD requires EMD-signal (import name PyEMD); install requirements_decomposition_visualization.txt"
        ) from error
    model = EMD()
    model.emd(values, max_imf=int(cfg.emd_max_imf))
    imfs, residual = model.get_imfs_and_residue()
    imfs = np.asarray(imfs, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if imfs.ndim == 1 and imfs.size:
        imfs = imfs[None, :]
    if imfs.size == 0:
        imfs = np.empty((0, values.size), dtype=np.float64)
    components = [Component(f"imf_{i + 1}", imf, positions) for i, imf in enumerate(imfs)]
    components.append(Component("residual", residual, positions))
    return DecompositionResult(
        method="emd",
        components=tuple(components),
        metadata={
            "max_imf": int(cfg.emd_max_imf),
            "num_imfs_returned": int(len(imfs)),
            "time_semantics": "observation_index",
        },
    )


def _ceemdan(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    try:
        from PyEMD import CEEMDAN
    except ImportError as error:
        raise OptionalDependencyError(
            "CEEMDAN requires EMD-signal (import name PyEMD); install requirements_decomposition_visualization.txt"
        ) from error
    model = CEEMDAN(trials=int(cfg.ceemdan_trials))
    # PyEMD exposes noise_seed in current releases; keep compatibility with older ones.
    if hasattr(model, "noise_seed"):
        model.noise_seed(int(cfg.random_seed))
    model.ceemdan(values, max_imf=int(cfg.ceemdan_max_imf))
    imfs, residual = model.get_imfs_and_residue()
    imfs = np.asarray(imfs, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if imfs.ndim == 1 and imfs.size:
        imfs = imfs[None, :]
    if imfs.size == 0:
        imfs = np.empty((0, values.size), dtype=np.float64)
    components = [Component(f"ceemdan_imf_{i + 1}", imf, positions) for i, imf in enumerate(imfs)]
    components.append(Component("residual", residual, positions))
    return DecompositionResult(
        method="ceemdan",
        components=tuple(components),
        metadata={
            "max_imf": int(cfg.ceemdan_max_imf),
            "trials": int(cfg.ceemdan_trials),
            "seed": int(cfg.random_seed),
            "num_imfs_returned": int(len(imfs)),
            "time_semantics": "observation_index",
        },
    )


def _vmd(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    try:
        from vmdpy import VMD
    except ImportError as error:
        raise OptionalDependencyError(
            "VMD requires vmdpy; install requirements_decomposition_visualization.txt"
        ) from error
    modes, _, omega = VMD(
        values,
        float(cfg.vmd_alpha),
        float(cfg.vmd_tau),
        int(cfg.vmd_k),
        int(cfg.vmd_dc),
        int(cfg.vmd_init),
        float(cfg.vmd_tol),
    )
    modes = np.asarray(modes, dtype=np.float64)
    # vmdpy intentionally drops the final observation for odd-length inputs.
    # Preserve that native behavior instead of padding/interpolating.
    output_length = int(modes.shape[-1])
    aligned_values = values[:output_length]
    aligned_positions = positions[:output_length]
    residual = aligned_values - modes.sum(axis=0)
    components = [
        Component(f"mode_{i + 1}", mode, aligned_positions)
        for i, mode in enumerate(modes)
    ]
    components.append(Component("residual", residual, aligned_positions))
    final_omega = np.asarray(omega[-1] if np.ndim(omega) > 1 else omega, dtype=np.float64)
    return DecompositionResult(
        method="vmd",
        components=tuple(components),
        metadata={
            "alpha": float(cfg.vmd_alpha),
            "tau": float(cfg.vmd_tau),
            "K": int(cfg.vmd_k),
            "DC": int(cfg.vmd_dc),
            "init": int(cfg.vmd_init),
            "tol": float(cfg.vmd_tol),
            "final_center_frequencies": final_omega.tolist(),
            "input_length": int(values.size),
            "output_length": output_length,
            "odd_length_native_drop": bool(output_length != values.size),
            "time_semantics": "observation_index",
        },
    )


def _wavelet(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    try:
        import pywt
    except ImportError as error:
        raise OptionalDependencyError(
            "Wavelet decomposition requires PyWavelets; install requirements_decomposition_visualization.txt"
        ) from error
    wavelet = pywt.Wavelet(cfg.wavelet_name)
    max_level = pywt.dwt_max_level(values.size, wavelet.dec_len)
    level = min(int(cfg.wavelet_level), int(max_level))
    if level < 1:
        raise ValueError(
            f"series length {values.size} is too short for wavelet {cfg.wavelet_name!r}"
        )
    coeffs = pywt.wavedec(values, wavelet, level=level, mode="symmetric")
    components = []
    for coeff_index, coeff in enumerate(coeffs):
        isolated = [np.zeros_like(item) for item in coeffs]
        isolated[coeff_index] = coeff
        reconstructed = pywt.waverec(isolated, wavelet, mode="symmetric")[: values.size]
        if coeff_index == 0:
            name = f"approx_L{level}"
        else:
            detail_level = level - coeff_index + 1
            name = f"detail_L{detail_level}"
        components.append(Component(name, reconstructed, positions))
    residual = values - np.sum(np.stack([c.values for c in components], axis=0), axis=0)
    components.append(Component("reconstruction_residual", residual, positions))
    return DecompositionResult(
        method="wavelet",
        components=tuple(components),
        metadata={
            "wavelet": cfg.wavelet_name,
            "level_requested": int(cfg.wavelet_level),
            "level_used": int(level),
            "time_semantics": "observation_index",
        },
    )


def _diagonal_average(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    out = np.zeros(rows + cols - 1, dtype=np.float64)
    counts = np.zeros_like(out)
    for row in range(rows):
        out[row : row + cols] += matrix[row]
        counts[row : row + cols] += 1.0
    return out / counts


def _ssa(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    n = values.size
    window = int(cfg.ssa_window) if int(cfg.ssa_window) > 0 else n // 3
    window = min(max(2, window), n - 1)
    columns = n - window + 1
    trajectory = np.column_stack([values[i : i + window] for i in range(columns)])
    u, singular_values, vt = np.linalg.svd(trajectory, full_matrices=False)
    count = min(int(cfg.ssa_components), len(singular_values))
    components = []
    reconstructed = []
    for index in range(count):
        elementary = singular_values[index] * np.outer(u[:, index], vt[index])
        series = _diagonal_average(elementary)
        reconstructed.append(series)
        components.append(Component(f"rc_{index + 1}", series, positions))
    residual = values - np.sum(np.stack(reconstructed, axis=0), axis=0)
    components.append(Component("residual", residual, positions))
    return DecompositionResult(
        method="ssa",
        components=tuple(components),
        metadata={
            "window": int(window),
            "components_requested": int(cfg.ssa_components),
            "components_returned": int(count),
            "singular_values": singular_values[:count].tolist(),
            "time_semantics": "observation_index",
        },
    )


def _fourier(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    low = float(cfg.fourier_low_fraction)
    mid = float(cfg.fourier_mid_fraction)
    if not 0.0 < low < mid < 1.0:
        raise ValueError("Fourier fractions must satisfy 0 < low < mid < 1")
    spectrum = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(values.size, d=1.0)
    nyquist = 0.5
    normalized = frequencies / nyquist
    masks = {
        "low_frequency": normalized <= low,
        "mid_frequency": (normalized > low) & (normalized <= mid),
        "high_frequency": normalized > mid,
    }
    components = []
    reconstructed = []
    for name, mask in masks.items():
        part = np.fft.irfft(np.where(mask, spectrum, 0.0), n=values.size)
        reconstructed.append(part)
        components.append(Component(name, part, positions))
    residual = values - np.sum(np.stack(reconstructed, axis=0), axis=0)
    components.append(Component("reconstruction_residual", residual, positions))
    return DecompositionResult(
        method="fourier",
        components=tuple(components),
        metadata={
            "low_fraction_of_nyquist": low,
            "mid_fraction_of_nyquist": mid,
            "time_semantics": "observation_index",
            "note": "Frequency bins are cycles per observation, not cycles per physical day.",
        },
    )


def _select_lomb_peaks(power: np.ndarray, count: int, min_separation_bins: int) -> np.ndarray:
    order = np.argsort(power)[::-1]
    selected = []
    for index in order:
        if all(abs(int(index) - previous) >= min_separation_bins for previous in selected):
            selected.append(int(index))
            if len(selected) >= count:
                break
    return np.asarray(selected, dtype=np.int64)


def _lomb_scargle(values: np.ndarray, positions: np.ndarray, cfg: DecompositionConfig) -> DecompositionResult:
    try:
        from scipy.signal import lombscargle
    except ImportError as error:
        raise OptionalDependencyError(
            "Lomb-Scargle decomposition requires scipy"
        ) from error
    relative_t = positions - positions[0]
    span = float(relative_t[-1] - relative_t[0])
    deltas = np.diff(relative_t)
    median_delta = float(np.median(deltas))
    if span <= 0.0 or median_delta <= 0.0:
        raise ValueError("Lomb-Scargle requires positive physical time span")
    min_frequency = 1.0 / span
    max_frequency = 0.5 / median_delta
    if not min_frequency < max_frequency:
        max_frequency = 2.0 * min_frequency
    grid_size = max(64, int(cfg.lomb_grid_size))
    frequencies = np.linspace(min_frequency, max_frequency, grid_size)
    angular = 2.0 * np.pi * frequencies
    centered = values - np.mean(values)
    power = lombscargle(relative_t, centered, angular, normalize=True)
    peak_indices = _select_lomb_peaks(
        np.asarray(power),
        max(1, int(cfg.lomb_top_k)),
        max(1, int(cfg.lomb_min_separation_bins)),
    )
    selected_frequencies = frequencies[peak_indices]
    columns = [np.ones_like(relative_t)]
    for frequency in selected_frequencies:
        columns.append(np.cos(2.0 * np.pi * frequency * relative_t))
        columns.append(np.sin(2.0 * np.pi * frequency * relative_t))
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    baseline = np.full_like(values, coefficients[0])
    components = [Component("baseline", baseline, positions)]
    reconstructed = baseline.copy()
    cursor = 1
    for rank, frequency in enumerate(selected_frequencies, start=1):
        contribution = (
            coefficients[cursor] * np.cos(2.0 * np.pi * frequency * relative_t)
            + coefficients[cursor + 1] * np.sin(2.0 * np.pi * frequency * relative_t)
        )
        cursor += 2
        reconstructed += contribution
        components.append(
            Component(f"frequency_{rank}_{frequency:.6f}_cycles_per_day", contribution, positions)
        )
    residual = values - reconstructed
    components.append(Component("residual", residual, positions))
    return DecompositionResult(
        method="lomb_scargle",
        components=tuple(components),
        metadata={
            "frequency_unit": "cycles_per_day",
            "frequency_grid_min": float(min_frequency),
            "frequency_grid_max": float(max_frequency),
            "frequency_grid_size": int(grid_size),
            "selected_frequencies": selected_frequencies.tolist(),
            "selected_period_days": (1.0 / selected_frequencies).tolist(),
            "time_semantics": "physical_time",
        },
    )


_DISPATCH = {
    "autoformer": _autoformer,
    "fedformer": _fedformer,
    "dlinear": _dlinear,
    "micn": _micn,
    "timemixer_ma": _timemixer_ma,
    "timemixer_dft": _timemixer_dft,
    "xpatch_ema": _xpatch_ema,
    "stl": _stl,
    "emd": _emd,
    "ceemdan": _ceemdan,
    "vmd": _vmd,
    "wavelet": _wavelet,
    "ssa": _ssa,
    "fourier": _fourier,
    "lomb_scargle": _lomb_scargle,
}


def decompose(
    method: str,
    values: np.ndarray,
    positions: np.ndarray,
    config: Optional[DecompositionConfig] = None,
) -> DecompositionResult:
    """Run one registered training-free decomposition without resampling."""
    if method not in _DISPATCH:
        raise KeyError(f"unknown decomposition method {method!r}; choose from {ALL_METHODS}")
    x, t = _validate_input(values, positions)
    cfg = config or DecompositionConfig()
    return _DISPATCH[method](x, t, cfg)
