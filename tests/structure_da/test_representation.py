from __future__ import annotations

import pytest
import torch

from methods.structure_da.representation import (
    FunctionalGeometryOutput,
    RawTemporalRepresentation,
    TSStructureForwardOutput,
)


def test_raw_temporal_representation_is_single_stream() -> None:
    fused = torch.randn(3, 4)
    positions = torch.randn(3, 5)
    raw = RawTemporalRepresentation(fused_repr=fused, positions_used=positions)
    assert raw.fused_repr.shape == (3, 4)
    assert set(raw.__dataclass_fields__) == {"fused_repr", "positions_used"}


def test_raw_temporal_representation_validates_positions() -> None:
    with pytest.raises(ValueError, match="positions_used"):
        RawTemporalRepresentation(torch.randn(3, 4), torch.randn(4, 5))


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


def test_forward_output_contains_no_ts_task_embeddings() -> None:
    output = TSStructureForwardOutput(
        logits=torch.randn(2, 3),
        fused_repr=torch.randn(2, 4),
        latent=torch.randn(2, 5, 4),
        trend=None,
        structure=None,
        dynamics=None,
        residual=None,
        positions=torch.randn(2, 5),
        mask=torch.ones(2, 5, dtype=torch.bool),
        geometry=None,
    )
    fields = set(output.__dataclass_fields__)
    assert not (fields & {"trend_repr", "structure_repr", "z_shape", "quality", "phase"})
