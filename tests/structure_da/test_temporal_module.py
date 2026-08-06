from __future__ import annotations

import torch

from methods.structure_da import (
    FunctionalGeometryOutput,
    RawTemporalRepresentation,
    SharedTrendStructureLTAE,
    TrendStructureTemporalModule,
)
from methods.structure_da.temporal_srvf import TemporalSRVFExtractor


def _inputs(
    *,
    batch_size: int = 2,
    length: int = 6,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(72)
    tokens = torch.randn(batch_size, length, 2, dtype=dtype)
    positions = torch.linspace(0, 345, length, dtype=dtype)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, False, True, True, False, True],
        ][:batch_size]
    )
    return tokens, positions, mask


def _module(**overrides) -> TrendStructureTemporalModule:
    values = dict(
        in_channels=2,
        n_head=1,
        d_k=2,
        n_neurons=(8, 4),
        d_model=8,
        dropout=0.0,
        max_initial_frequency=4.0,
    )
    values.update(overrides)
    raw_encoder = SharedTrendStructureLTAE(**values)
    trend_geometry = TemporalSRVFExtractor(
        feature_dim=2,
        num_basis=4,
        canonical_grid_size=5,
        roughness_grid_size=64,
        min_mean_support=0.0,
        min_dynamic_energy=0.0,
    )
    structure_geometry = TemporalSRVFExtractor(
        feature_dim=2,
        num_basis=4,
        canonical_grid_size=5,
        roughness_grid_size=64,
        min_mean_support=0.0,
        min_dynamic_energy=0.0,
    )
    return TrendStructureTemporalModule(
        raw_encoder=raw_encoder,
        trend_geometry=trend_geometry,
        structure_geometry=structure_geometry,
    )


def test_module_forward_returns_raw_and_geometry() -> None:
    module = _module()
    trend, positions, mask = _inputs()
    structure = trend + 0.15 * torch.sin(trend)

    raw, geometry = module(trend, structure, positions, mask, return_geometry=True)

    assert isinstance(raw, RawTemporalRepresentation)
    assert raw.trend_repr.shape == (2, 4)
    assert raw.structure_repr.shape == (2, 4)
    assert raw.fused_repr.shape == (2, 8)
    assert raw.positions_used.shape == (2, 6)
    assert isinstance(geometry, FunctionalGeometryOutput)
    assert geometry.trend_srvf.shape == (2, 5, 2)
    assert geometry.structure_srvf.shape == (2, 5, 2)
    assert geometry.trend_support.shape == (2, 5)
    assert geometry.structure_support.shape == (2, 5)
    assert geometry.canonical_grid.shape == (5,)
    assert geometry.trend_valid.shape == (2,)
    assert geometry.structure_valid.shape == (2,)


def test_module_skips_geometry_when_disabled(monkeypatch) -> None:
    module = _module()
    trend, positions, mask = _inputs()
    structure = trend + 0.15 * torch.sin(trend)
    calls = 0
    original = module.trend_geometry.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module.trend_geometry, "forward", counted)
    raw, geometry = module(trend, structure, positions, mask, return_geometry=False)
    assert geometry is None
    assert calls == 0


def test_module_is_pure_without_source_state_buffers() -> None:
    module = _module()
    state = module.state_dict()
    assert not any("running_srvf" in key or "running_support" in key for key in state)
    assert not hasattr(module, "update_source_state")
    assert not hasattr(module, "update_reference")
    assert not hasattr(module, "warp_estimator")
