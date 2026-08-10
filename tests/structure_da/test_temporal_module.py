from __future__ import annotations

import torch

from methods.structure_da import (
    FunctionalGeometryOutput,
    LatentTemporalLTAE,
    PhaseOnlyTemporalModule,
    RawTemporalRepresentation,
)
from methods.structure_da.temporal_srvf import TemporalSRVFExtractor


def _module() -> PhaseOnlyTemporalModule:
    raw = LatentTemporalLTAE(
        in_channels=2, n_head=1, d_k=2, n_neurons=(8, 4), d_model=8,
        dropout=0.0, max_initial_frequency=4.0,
    )
    def geo():
        return TemporalSRVFExtractor(
            feature_dim=2, num_basis=4, canonical_grid_size=5,
            roughness_grid_size=64, min_mean_support=0.0, min_dynamic_energy=0.0,
        )
    return PhaseOnlyTemporalModule(raw, geo(), geo())


def _inputs():
    torch.manual_seed(72)
    latent = torch.randn(2, 6, 2)
    trend = torch.randn(2, 6, 2)
    structure = trend + 0.15 * torch.sin(trend)
    positions = torch.linspace(0, 1, 6).expand(2, -1)
    mask = torch.tensor([[True]*6, [True, False, True, True, False, True]])
    return latent, trend, structure, positions, mask


def test_module_separates_task_and_geometry_inputs() -> None:
    module = _module().eval()
    latent, trend, structure, positions, mask = _inputs()
    raw, geometry = module(latent, trend, structure, positions, mask, return_geometry=True)
    assert isinstance(raw, RawTemporalRepresentation)
    assert raw.fused_repr.shape == (2, 4)
    assert isinstance(geometry, FunctionalGeometryOutput)
    assert geometry.structure_srvf.shape == (2, 5, 2)


def test_task_embedding_depends_on_latent_not_decomposed_values() -> None:
    module = _module().eval()
    latent, trend, structure, positions, mask = _inputs()
    raw_a, _ = module(latent, trend, structure, positions, mask, return_geometry=False)
    raw_b, _ = module(latent, trend + 100.0, structure - 100.0, positions, mask, return_geometry=False)
    torch.testing.assert_close(raw_a.fused_repr, raw_b.fused_repr, rtol=0, atol=0)


def test_module_skips_geometry_when_disabled(monkeypatch) -> None:
    module = _module()
    latent, trend, structure, positions, mask = _inputs()
    calls = 0
    original = module.trend_geometry.forward
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(module.trend_geometry, "forward", counted)
    _, geometry = module(latent, trend, structure, positions, mask, return_geometry=False)
    assert geometry is None and calls == 0
