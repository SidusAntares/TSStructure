"""Coordinates for the V3 trend-led phase and structure shape path."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .temporal_geometry import (
    _finite_real,
    _validate_interval_widths,
    warp_to_identity_tangent,
)

def _positive_integer(name: str, value: int, minimum: int = 1) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _weighted_orthogonal_cosine_basis(
    grid: Tensor,
    integration_weights: Tensor,
    num_basis: int,
    *,
    exclude_constant: bool,
    eps: float,
) -> Tensor:
    """Build fixed cosine functions orthonormal under weighted integration."""

    if not isinstance(grid, Tensor) or grid.ndim != 1:
        raise ValueError("grid must have shape [N]")
    if grid.shape[0] < 1 or not grid.is_floating_point():
        raise ValueError("grid must be a nonempty floating-point tensor")
    if not torch.isfinite(grid).all().item():
        raise ValueError("grid must contain only finite values")
    if (
        not isinstance(integration_weights, Tensor)
        or integration_weights.shape != grid.shape
    ):
        raise ValueError("integration_weights must have shape [N]")
    if not integration_weights.is_floating_point():
        raise ValueError("integration_weights must use a floating-point dtype")
    if not torch.isfinite(integration_weights).all().item():
        raise ValueError("integration_weights must contain only finite values")
    if torch.any(integration_weights <= 0).item():
        raise ValueError("integration_weights must be strictly positive")
    if (
        integration_weights.device != grid.device
        or integration_weights.dtype != grid.dtype
    ):
        raise ValueError("integration_weights must match grid device and dtype")
    if not isinstance(exclude_constant, bool):
        raise ValueError("exclude_constant must be boolean")
    num_basis = _positive_integer("num_basis", num_basis)
    maximum = grid.shape[0] - (1 if exclude_constant else 0)
    if num_basis > maximum:
        raise ValueError(
            f"num_basis must not exceed {maximum} for this grid"
        )
    eps = _finite_real("eps", eps)
    if eps <= 0:
        raise ValueError("eps must be greater than zero")

    first_frequency = 1 if exclude_constant else 0
    frequencies = torch.arange(
        first_frequency,
        first_frequency + num_basis,
        device=grid.device,
        dtype=grid.dtype,
    )
    candidates = torch.cos(
        torch.pi * grid.unsqueeze(-1) * frequencies.unsqueeze(0)
    )
    sqrt_weights = torch.sqrt(integration_weights).unsqueeze(-1)
    weighted_candidates = sqrt_weights * candidates
    orthogonal, _ = torch.linalg.qr(weighted_candidates, mode="reduced")
    return orthogonal / sqrt_weights


@dataclass(frozen=True)
class TrendStructureCoordinateOutput:
    shape_coordinates_fixed: Tensor
    shape_coordinates: Tensor
    phase_coordinates: Tensor

    shape_basis_support: Tensor
    shape_support: Tensor

    phase_basis_coefficients: Tensor
    phase_magnitude: Tensor
    phase_tangent: Tensor

    shape_valid: Tensor
    phase_valid: Tensor


class TrendStructureCoordinates(nn.Module):
    """Project the complete aligned structure SRVF onto fixed Shape bases."""

    def __init__(
        self,
        feature_dim: int,
        canonical_grid_size: int,
        num_shape_basis: int = 8,
        num_phase_basis: int = 8,
        attribute_projection_dim: int = 8,
        min_basis_support: float = 1e-4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        feature_dim = _positive_integer("feature_dim", feature_dim)
        canonical_grid_size = _positive_integer(
            "canonical_grid_size", canonical_grid_size, minimum=3
        )
        num_shape_basis = _positive_integer("num_shape_basis", num_shape_basis)
        if num_shape_basis > canonical_grid_size:
            raise ValueError(
                "num_shape_basis must not exceed canonical_grid_size"
            )
        num_phase_basis = _positive_integer("num_phase_basis", num_phase_basis)
        if num_phase_basis > canonical_grid_size - 2:
            raise ValueError(
                "num_phase_basis must not exceed canonical_grid_size - 2"
            )
        attribute_projection_dim = _positive_integer(
            "attribute_projection_dim", attribute_projection_dim
        )
        min_basis_support = _finite_real("min_basis_support", min_basis_support)
        eps = _finite_real("eps", eps)
        if min_basis_support < 0:
            raise ValueError("min_basis_support must be nonnegative")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")

        self.feature_dim = feature_dim
        self.canonical_grid_size = canonical_grid_size
        self.num_shape_basis = num_shape_basis
        self.num_phase_basis = num_phase_basis
        self.attribute_projection_dim = attribute_projection_dim
        self.min_basis_support = min_basis_support
        self.eps = eps

        canonical_grid = torch.linspace(0.0, 1.0, canonical_grid_size)
        grid_integration_weights = torch.full(
            (canonical_grid_size,), 1.0 / (canonical_grid_size - 1)
        )
        grid_integration_weights[[0, -1]] *= 0.5
        interval_midpoints = 0.5 * (
            canonical_grid[:-1] + canonical_grid[1:]
        )
        interval_integration_weights = torch.full(
            (canonical_grid_size - 1,), 1.0 / (canonical_grid_size - 1)
        )
        shape_time_basis = _weighted_orthogonal_cosine_basis(
            canonical_grid,
            grid_integration_weights,
            num_shape_basis,
            exclude_constant=False,
            eps=eps,
        )
        phase_time_basis = _weighted_orthogonal_cosine_basis(
            interval_midpoints,
            interval_integration_weights,
            num_phase_basis,
            exclude_constant=True,
            eps=eps,
        )
        self.register_buffer("canonical_grid", canonical_grid)
        self.register_buffer("grid_integration_weights", grid_integration_weights)
        self.register_buffer("interval_midpoints", interval_midpoints)
        self.register_buffer(
            "interval_integration_weights", interval_integration_weights
        )
        self.register_buffer("shape_time_basis", shape_time_basis)
        self.register_buffer("phase_time_basis", phase_time_basis)
        self.attribute_projection = nn.Linear(
            feature_dim, attribute_projection_dim, bias=False
        )
        nn.init.orthogonal_(self.attribute_projection.weight)

    def _validate_inputs(
        self,
        aligned_structure_srvf: Tensor,
        aligned_structure_support: Tensor,
        interval_widths: Tensor,
        shape_valid: Tensor,
        phase_valid: Tensor,
    ) -> None:
        expected = None
        if isinstance(aligned_structure_srvf, Tensor) and aligned_structure_srvf.ndim == 3:
            expected = (
                aligned_structure_srvf.shape[0],
                self.canonical_grid_size,
                self.feature_dim,
            )
        if (
            not isinstance(aligned_structure_srvf, Tensor)
            or aligned_structure_srvf.shape != expected
            or aligned_structure_srvf.shape[1:] != (
                self.canonical_grid_size,
                self.feature_dim,
            )
            or not aligned_structure_srvf.is_floating_point()
        ):
            raise ValueError(
                "aligned_structure_srvf must have floating shape [B, K, D]"
            )
        if not torch.isfinite(aligned_structure_srvf).all().item():
            raise ValueError("aligned_structure_srvf must be finite")
        batch_size = aligned_structure_srvf.shape[0]
        if (
            not isinstance(aligned_structure_support, Tensor)
            or aligned_structure_support.shape != (batch_size, self.canonical_grid_size)
            or not aligned_structure_support.is_floating_point()
        ):
            raise ValueError("aligned_structure_support must have shape [B, K]")
        if not torch.isfinite(aligned_structure_support).all().item():
            raise ValueError("aligned_structure_support must be finite")
        if torch.any((aligned_structure_support < 0) | (aligned_structure_support > 1)).item():
            raise ValueError("aligned_structure_support must lie in [0, 1]")
        _validate_interval_widths(interval_widths)
        if interval_widths.shape != (batch_size, self.canonical_grid_size - 1):
            raise ValueError("interval_widths must have shape [B, K-1]")
        for name, valid in (("shape_valid", shape_valid), ("phase_valid", phase_valid)):
            if (
                not isinstance(valid, Tensor)
                or valid.dtype != torch.bool
                or valid.shape != (batch_size,)
            ):
                raise ValueError(f"{name} must be a boolean tensor with shape [B]")
            if valid.device != aligned_structure_srvf.device:
                raise ValueError(f"{name} device must match aligned_structure_srvf")
        for name, tensor in (
            ("aligned_structure_support", aligned_structure_support),
            ("interval_widths", interval_widths),
        ):
            if tensor.device != aligned_structure_srvf.device or tensor.dtype != aligned_structure_srvf.dtype:
                raise ValueError(
                    f"{name} dtype and device must match aligned_structure_srvf"
                )
        for name, tensor in self.named_buffers():
            if tensor.device != aligned_structure_srvf.device or tensor.dtype != aligned_structure_srvf.dtype:
                raise ValueError(f"{name} dtype and device must match input")
        weight = self.attribute_projection.weight
        if weight.device != aligned_structure_srvf.device or weight.dtype != aligned_structure_srvf.dtype:
            raise ValueError("attribute_projection dtype and device must match input")

    def forward(
        self,
        *,
        aligned_structure_srvf: Tensor,
        aligned_structure_support: Tensor,
        interval_widths: Tensor,
        shape_valid: Tensor,
        phase_valid: Tensor,
    ) -> TrendStructureCoordinateOutput:
        self._validate_inputs(
            aligned_structure_srvf,
            aligned_structure_support,
            interval_widths,
            shape_valid,
            phase_valid,
        )
        shape_basis_support = torch.einsum(
            "k,bk,km->bm",
            self.grid_integration_weights,
            aligned_structure_support,
            self.shape_time_basis.square(),
        )
        numerator = torch.einsum(
            "k,bk,km,bkd->bmd",
            self.grid_integration_weights,
            aligned_structure_support,
            self.shape_time_basis,
            aligned_structure_srvf,
        )
        shape_coordinates_fixed = numerator / (
            shape_basis_support + self.eps
        ).unsqueeze(-1)
        shape_coordinates_fixed = torch.where(
            (shape_basis_support >= self.min_basis_support).unsqueeze(-1),
            shape_coordinates_fixed,
            torch.zeros_like(shape_coordinates_fixed),
        )
        shape_coordinates = self.attribute_projection(shape_coordinates_fixed)

        phase = warp_to_identity_tangent(interval_widths, eps=self.eps)
        phase_basis_coefficients = torch.einsum(
            "j,bj,jm->bm",
            self.interval_integration_weights,
            phase.tangent,
            self.phase_time_basis,
        )
        phase_coordinates = torch.cat(
            [phase_basis_coefficients, phase.magnitude.unsqueeze(-1)], dim=-1
        )

        shape_coordinates_fixed = torch.where(
            shape_valid[:, None, None],
            shape_coordinates_fixed,
            torch.zeros_like(shape_coordinates_fixed),
        )
        shape_coordinates = torch.where(
            shape_valid[:, None, None],
            shape_coordinates,
            torch.zeros_like(shape_coordinates),
        )
        shape_support = torch.where(
            shape_valid[:, None],
            aligned_structure_support,
            torch.zeros_like(aligned_structure_support),
        )
        shape_basis_support = torch.where(
            shape_valid[:, None],
            shape_basis_support,
            torch.zeros_like(shape_basis_support),
        )
        phase_basis_coefficients = torch.where(
            phase_valid[:, None],
            phase_basis_coefficients,
            torch.zeros_like(phase_basis_coefficients),
        )
        phase_coordinates = torch.where(
            phase_valid[:, None],
            phase_coordinates,
            torch.zeros_like(phase_coordinates),
        )
        phase_magnitude = torch.where(
            phase_valid, phase.magnitude, torch.zeros_like(phase.magnitude)
        )
        phase_tangent = torch.where(
            phase_valid[:, None], phase.tangent, torch.zeros_like(phase.tangent)
        )
        return TrendStructureCoordinateOutput(
            shape_coordinates_fixed=shape_coordinates_fixed,
            shape_coordinates=shape_coordinates,
            phase_coordinates=phase_coordinates,
            shape_basis_support=shape_basis_support,
            shape_support=shape_support,
            phase_basis_coefficients=phase_basis_coefficients,
            phase_magnitude=phase_magnitude,
            phase_tangent=phase_tangent,
            shape_valid=shape_valid,
            phase_valid=phase_valid,
        )
