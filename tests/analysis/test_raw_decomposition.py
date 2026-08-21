import numpy as np
import pytest

from analysis.raw_decomposition import ALL_METHODS, DecompositionConfig, decompose
from analysis.raw_decomposition.timematch_raw import extract_signals, parse_signal_names


def _series(n=64):
    t = np.cumsum(np.linspace(3.0, 8.0, n))
    x = 0.02 * t + np.sin(2.0 * np.pi * np.arange(n) / 12.0) + 0.2 * np.cos(
        2.0 * np.pi * np.arange(n) / 5.0
    )
    return x, t


def _sum_same_grid_components(result, n):
    components = [component.values for component in result.components if component.values.size == n]
    return np.sum(np.stack(components, axis=0), axis=0)


def test_registry_contains_exactly_fifteen_methods():
    assert len(ALL_METHODS) == 15
    assert len(set(ALL_METHODS)) == 15


@pytest.mark.parametrize(
    "method",
    ["autoformer", "fedformer", "dlinear", "micn", "xpatch_ema", "ssa", "fourier", "lomb_scargle"],
)
def test_core_methods_return_finite_components(method):
    x, t = _series()
    result = decompose(method, x, t, DecompositionConfig())
    assert result.method == method
    assert result.components
    for component in result.components:
        assert component.values.shape == component.positions.shape
        assert np.all(np.isfinite(component.values))
        assert np.all(np.isfinite(component.positions))


@pytest.mark.parametrize("method", ["autoformer", "fedformer", "dlinear", "micn", "xpatch_ema", "ssa", "fourier", "lomb_scargle"])
def test_additive_same_grid_methods_reconstruct_input(method):
    x, t = _series()
    result = decompose(method, x, t, DecompositionConfig())
    reconstructed = _sum_same_grid_components(result, x.size)
    np.testing.assert_allclose(reconstructed, x, atol=1e-7, rtol=1e-7)


def test_timemixer_methods_keep_raw_scale_and_only_downsample_internally():
    x, t = _series()
    cfg = DecompositionConfig(timemixer_downsample_layers=2)
    for method in ("timemixer_ma", "timemixer_dft"):
        result = decompose(method, x, t, cfg)
        scale0 = [component for component in result.components if component.name.startswith("scale0_")]
        assert len(scale0) == 2
        np.testing.assert_allclose(scale0[0].values + scale0[1].values, x, atol=1e-7)
        assert any(component.values.size < x.size for component in result.components)


def test_raw_timematch_signal_extraction_does_not_change_time_length():
    rng = np.random.default_rng(3)
    pixels = rng.uniform(1000.0, 5000.0, size=(23, 10, 7))
    signals = extract_signals(
        pixels,
        signal_names=parse_signal_names("B4,B8,NDVI"),
        spatial_reduction="mean",
        pixel_index=0,
    )
    assert set(signals) == {"B4", "B8", "NDVI"}
    assert all(values.shape == (23,) for values in signals.values())
    assert np.all((-1.0 <= signals["NDVI"]) & (signals["NDVI"] <= 1.0))


def test_nonfinite_raw_series_is_rejected_instead_of_imputed():
    x, t = _series()
    x[7] = np.nan
    with pytest.raises(ValueError, match="no imputation"):
        decompose("autoformer", x, t)
