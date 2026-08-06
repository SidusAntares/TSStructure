from __future__ import annotations

import pytest
import torch

from methods.structure_da.representation import (
    FunctionalGeometryOutput,
    RawTemporalRepresentation,
    TSStructureForwardOutput,
)


def test_raw_temporal_representation_validates_shapes() -> None:
    trend = torch.randn(3, 4)
    structure = torch.randn(3, 4)
    fused = torch.randn(3, 8)
    positions = torch.randn(3, 5)

    raw = RawTemporalRepresentation(trend, structure, fused, positions)
    assert raw.trend_repr.shape == (3, 4)
    assert raw.fused_repr.shape == (3, 8)

    with pytest.raises(ValueError, match="fused_repr"):
        RawTemporalRepresentation(trend, structure, torch.randn(3, 9), positions)


def test_raw_temporal_representation_rejects_wrong_positions_used() -> None:
    trend = torch.randn(3, 4)
    structure = torch.randn(3, 4)
    fused = torch.randn(3, 8)
    with pytest.raises(ValueError, match="positions_used"):
        RawTemporalRepresentation(trend, structure, fused, torch.randn(4, 5))


def test_functional_geometry_output_validates_shapes() -> None:
    srvf = torch.randn(2, 6, 3)
    support = torch.rand(2, 6)
    grid = torch.linspace(0, 1, 6)
    valid = torch.tensor([True, False])

    geometry = FunctionalGeometryOutput(
        trend_srvf=srvf,
        structure_srvf=srvf,
        trend_support=support,
        structure_support=support,
        canonical_grid=grid,
        trend_valid=valid,
        structure_valid=valid,
    )
    assert geometry.trend_srvf.shape == (2, 6, 3)
    assert geometry.canonical_grid.shape == (6,)

    with pytest.raises(ValueError, match="trend_srvf and structure_srvf"):
        FunctionalGeometryOutput(
            trend_srvf=srvf,
            structure_srvf=torch.randn(2, 6, 4),
            trend_support=support,
            structure_support=support,
            canonical_grid=grid,
            trend_valid=valid,
            structure_valid=valid,
        )
    with pytest.raises(ValueError, match="canonical_grid"):
        FunctionalGeometryOutput(
            trend_srvf=srvf,
            structure_srvf=srvf,
            trend_support=support,
            structure_support=support,
            canonical_grid=torch.linspace(0, 1, 7),
            trend_valid=valid,
            structure_valid=valid,
        )


def test_forward_output_requires_no_old_fields() -> None:
    output = TSStructureForwardOutput(
        logits=torch.randn(2, 3),
        fused_repr=torch.randn(2, 8),
        trend_repr=torch.randn(2, 4),
        structure_repr=torch.randn(2, 4),
        latent=torch.randn(2, 5, 4),
        trend=torch.randn(2, 5, 4),
        structure=torch.randn(2, 5, 4),
        dynamics=torch.randn(2, 5, 4),
        residual=None,
        positions=torch.randn(2, 5),
        mask=torch.ones(2, 5, dtype=torch.bool),
        geometry=None,
    )
    fields = set(output.__dataclass_fields__)
    assert not (fields & {"channel", "quality", "alpha", "z_shape", "phase"})
