from __future__ import annotations

import torch

from methods.structure_da.temporal_coordinates import (
    TrendStructureCoordinateOutput,
    TrendStructureCoordinates,
)

def _trend_structure_coordinates(**kwargs) -> TrendStructureCoordinates:
    values = dict(
        feature_dim=3,
        canonical_grid_size=7,
        num_shape_basis=4,
        num_phase_basis=3,
        attribute_projection_dim=2,
    )
    values.update(kwargs)
    return TrendStructureCoordinates(**values)


def test_trend_structure_shape_coordinates_use_complete_aligned_structure() -> None:
    module = _trend_structure_coordinates().double()
    aligned = torch.randn(2, 7, 3, dtype=torch.float64)
    support = torch.rand(2, 7, dtype=torch.float64)
    widths = torch.full((2, 6), 1.0 / 6, dtype=torch.float64)
    output = module(
        aligned_structure_srvf=aligned,
        aligned_structure_support=support,
        interval_widths=widths,
        shape_valid=torch.tensor([True, True]),
        phase_valid=torch.tensor([True, True]),
    )

    basis_support = torch.einsum(
        "k,bk,km->bm",
        module.grid_integration_weights,
        support,
        module.shape_time_basis.square(),
    )
    numerator = torch.einsum(
        "k,bk,km,bkd->bmd",
        module.grid_integration_weights,
        support,
        module.shape_time_basis,
        aligned,
    )
    expected = numerator / (basis_support + module.eps).unsqueeze(-1)

    assert isinstance(output, TrendStructureCoordinateOutput)
    torch.testing.assert_close(output.shape_coordinates_fixed, expected)
    assert not torch.allclose(
        output.shape_coordinates_fixed,
        numerator / torch.sqrt(basis_support + module.eps).unsqueeze(-1),
    )
    torch.testing.assert_close(output.shape_support, support)
    assert output.shape_coordinates_fixed.shape == (2, 4, 3)
    assert output.shape_coordinates.shape == (2, 4, 2)


def test_trend_structure_shape_and_phase_validity_are_independent() -> None:
    module = _trend_structure_coordinates()
    output = module(
        aligned_structure_srvf=torch.randn(2, 7, 3),
        aligned_structure_support=torch.ones(2, 7),
        interval_widths=torch.tensor(
            [
                [0.10, 0.12, 0.18, 0.20, 0.22, 0.18],
                [0.18, 0.22, 0.20, 0.18, 0.12, 0.10],
            ]
        ),
        shape_valid=torch.tensor([True, False]),
        phase_valid=torch.tensor([False, True]),
    )

    assert output.shape_valid.tolist() == [True, False]
    assert output.phase_valid.tolist() == [False, True]
    for value in (
        output.shape_coordinates_fixed,
        output.shape_coordinates,
        output.shape_basis_support,
        output.shape_support,
    ):
        torch.testing.assert_close(value[1], torch.zeros_like(value[1]))
    for value in (
        output.phase_coordinates,
        output.phase_basis_coefficients,
        output.phase_magnitude,
        output.phase_tangent,
    ):
        torch.testing.assert_close(value[0], torch.zeros_like(value[0]))


def test_trend_structure_attribute_projection_has_gradient() -> None:
    module = _trend_structure_coordinates()
    aligned = torch.randn(2, 7, 3, requires_grad=True)
    output = module(
        aligned_structure_srvf=aligned,
        aligned_structure_support=torch.ones(2, 7),
        interval_widths=torch.full((2, 6), 1.0 / 6),
        shape_valid=torch.tensor([True, True]),
        phase_valid=torch.tensor([True, True]),
    )

    output.shape_coordinates.square().sum().backward()

    assert aligned.grad is not None and torch.isfinite(aligned.grad).all()
    gradient = module.attribute_projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
