"""Explicit fixed-basis shape and phase coordinates for temporal SRVFs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .temporal_geometry import (
    _finite_real,
    _validate_interval_widths,
    warp_to_identity_tangent,
)
from .temporal_registration import TemporalRegistrationOutput


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
class TemporalCoordinateOutput:
    shape_coordinates: Tensor
    phase_coordinates: Tensor

    shape_time_coefficients: Tensor
    phase_basis_coefficients: Tensor
    phase_magnitude: Tensor

    shape_support: Tensor
    shape_basis_support: Tensor

    shape_residual: Tensor
    phase_tangent: Tensor

    valid: Tensor


class TemporalShapePhaseCoordinates(nn.Module):
    """Extract explicit fixed-time-basis shape and phase coordinates."""

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
        num_shape_basis = _positive_integer(
            "num_shape_basis", num_shape_basis
        )
        if num_shape_basis > canonical_grid_size:
            raise ValueError(
                "num_shape_basis must not exceed canonical_grid_size"
            )
        num_phase_basis = _positive_integer(
            "num_phase_basis", num_phase_basis
        )
        max_phase_basis = canonical_grid_size - 2
        if num_phase_basis > max_phase_basis:
            raise ValueError(
                "num_phase_basis must not exceed canonical_grid_size - 2"
            )
        attribute_projection_dim = _positive_integer(
            "attribute_projection_dim", attribute_projection_dim
        )
        min_basis_support = _finite_real(
            "min_basis_support", min_basis_support
        )
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
        grid_integration_weights[0] *= 0.5
        grid_integration_weights[-1] *= 0.5
        interval_midpoints = 0.5 * (
            canonical_grid[:-1] + canonical_grid[1:]
        )
        interval_integration_weights = torch.full(
            (canonical_grid_size - 1,),
            1.0 / (canonical_grid_size - 1),
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
        self.register_buffer(
            "grid_integration_weights", grid_integration_weights
        )
        self.register_buffer("interval_midpoints", interval_midpoints)
        self.register_buffer(
            "interval_integration_weights", interval_integration_weights
        )
        self.register_buffer("shape_time_basis", shape_time_basis)
        self.register_buffer("phase_time_basis", phase_time_basis)

        self.attribute_projection = nn.Linear(
            feature_dim,
            attribute_projection_dim,
            bias=False,
        )
        nn.init.orthogonal_(self.attribute_projection.weight)

    def _validate_inputs(
        self,
        registration_output: TemporalRegistrationOutput,
    ) -> None:
        if not isinstance(registration_output, TemporalRegistrationOutput):
            raise ValueError(
                "registration_output must be a TemporalRegistrationOutput"
            )
        registered_srvf = registration_output.registered_srvf
        expected_srvf_shape = None
        if isinstance(registered_srvf, Tensor) and registered_srvf.ndim == 3:
            expected_srvf_shape = (
                registered_srvf.shape[0],
                self.canonical_grid_size,
                self.feature_dim,
            )
        if (
            not isinstance(registered_srvf, Tensor)
            or registered_srvf.ndim != 3
            or registered_srvf.shape[1] != self.canonical_grid_size
        ):
            raise ValueError(
                "registered_srvf must have shape [B, canonical_grid_size, feature_dim]"
            )
        if registered_srvf.shape[2] != self.feature_dim:
            raise ValueError(
                "registered_srvf feature dimension must match feature_dim"
            )
        if not registered_srvf.is_floating_point():
            raise ValueError("registered_srvf must use a floating-point dtype")
        if not torch.isfinite(registered_srvf).all().item():
            raise ValueError("registered_srvf must contain only finite values")
        batch_size = registered_srvf.shape[0]

        template_srvf = registration_output.template_srvf
        if (
            not isinstance(template_srvf, Tensor)
            or template_srvf.shape != expected_srvf_shape
        ):
            raise ValueError("template_srvf must have shape [B, K, D]")
        if not template_srvf.is_floating_point():
            raise ValueError("template_srvf must use a floating-point dtype")
        if not torch.isfinite(template_srvf).all().item():
            raise ValueError("template_srvf must contain only finite values")

        expected_support_shape = (batch_size, self.canonical_grid_size)
        for name, support in (
            ("registered_support", registration_output.registered_support),
            ("template_support", registration_output.template_support),
        ):
            if not isinstance(support, Tensor) or support.shape != expected_support_shape:
                raise ValueError(f"{name} must have shape [B, K]")
            if not support.is_floating_point():
                raise ValueError(f"{name} must use a floating-point dtype")
            if not torch.isfinite(support).all().item():
                raise ValueError(f"{name} must contain only finite values")
            if torch.any((support < 0) | (support > 1)).item():
                raise ValueError(f"{name} must lie in [0, 1]")

        interval_widths = registration_output.interval_widths
        _validate_interval_widths(interval_widths)
        if interval_widths.shape != (
            batch_size,
            self.canonical_grid_size - 1,
        ):
            raise ValueError("interval_widths must have shape [B, K-1]")
        registration_valid = registration_output.registration_valid
        if (
            not isinstance(registration_valid, Tensor)
            or registration_valid.dtype != torch.bool
            or registration_valid.shape != (batch_size,)
        ):
            raise ValueError(
                "registration_valid must be a boolean tensor with shape [B]"
            )

        for name, tensor in (
            ("template_srvf", template_srvf),
            ("registered_support", registration_output.registered_support),
            ("template_support", registration_output.template_support),
            ("interval_widths", interval_widths),
        ):
            if tensor.device != registered_srvf.device:
                raise ValueError(
                    f"{name} device must match registered_srvf device"
                )
            if tensor.dtype != registered_srvf.dtype:
                raise ValueError(
                    f"{name} dtype must match registered_srvf dtype"
                )
        if registration_valid.device != registered_srvf.device:
            raise ValueError(
                "registration_valid device must match registered_srvf device"
            )
        for name, tensor in self.named_buffers():
            if tensor.device != registered_srvf.device:
                raise ValueError(
                    f"{name} device must match registered_srvf device"
                )
            if tensor.dtype != registered_srvf.dtype:
                raise ValueError(
                    f"{name} dtype must match registered_srvf dtype"
                )
        if self.attribute_projection.weight.device != registered_srvf.device:
            raise ValueError(
                "attribute_projection device must match registered_srvf device"
            )
        if self.attribute_projection.weight.dtype != registered_srvf.dtype:
            raise ValueError(
                "attribute_projection dtype must match registered_srvf dtype"
            )

    def forward(
        self,
        registration_output: TemporalRegistrationOutput,
    ) -> TemporalCoordinateOutput:
        self._validate_inputs(registration_output)
        registered_srvf = registration_output.registered_srvf
        valid = registration_output.registration_valid
        shape_residual = (
            registered_srvf - registration_output.template_srvf
        )
        shape_support = (
            registration_output.registered_support
            * registration_output.template_support
        )
        shape_basis_support = torch.einsum(
            "k,bk,km->bm",
            self.grid_integration_weights,
            shape_support,
            self.shape_time_basis.square(),
        )
        shape_numerator = torch.einsum(
            "k,bk,km,bkd->bmd",
            self.grid_integration_weights,
            shape_support,
            self.shape_time_basis,
            shape_residual,
        )
        shape_time_coefficients = shape_numerator / torch.sqrt(
            shape_basis_support + self.eps
        ).unsqueeze(-1)
        shape_time_coefficients = torch.where(
            (shape_basis_support >= self.min_basis_support).unsqueeze(-1),
            shape_time_coefficients,
            torch.zeros_like(shape_time_coefficients),
        )
        shape_coordinates = self.attribute_projection(
            shape_time_coefficients
        )

        phase = warp_to_identity_tangent(
            registration_output.interval_widths,
            eps=self.eps,
        )
        phase_basis_coefficients = torch.einsum(
            "j,bj,jm->bm",
            self.interval_integration_weights,
            phase.tangent,
            self.phase_time_basis,
        )
        phase_coordinates = torch.cat(
            [
                phase_basis_coefficients,
                phase.magnitude.unsqueeze(-1),
            ],
            dim=-1,
        )

        shape_time_coefficients = torch.where(
            valid[:, None, None],
            shape_time_coefficients,
            torch.zeros_like(shape_time_coefficients),
        )
        shape_coordinates = torch.where(
            valid[:, None, None],
            shape_coordinates,
            torch.zeros_like(shape_coordinates),
        )
        phase_basis_coefficients = torch.where(
            valid[:, None],
            phase_basis_coefficients,
            torch.zeros_like(phase_basis_coefficients),
        )
        phase_coordinates = torch.where(
            valid[:, None],
            phase_coordinates,
            torch.zeros_like(phase_coordinates),
        )
        phase_magnitude = torch.where(
            valid,
            phase.magnitude,
            torch.zeros_like(phase.magnitude),
        )
        shape_residual = torch.where(
            valid[:, None, None],
            shape_residual,
            torch.zeros_like(shape_residual),
        )
        phase_tangent = torch.where(
            valid[:, None],
            phase.tangent,
            torch.zeros_like(phase.tangent),
        )
        return TemporalCoordinateOutput(
            shape_coordinates=shape_coordinates,
            phase_coordinates=phase_coordinates,
            shape_time_coefficients=shape_time_coefficients,
            phase_basis_coefficients=phase_basis_coefficients,
            phase_magnitude=phase_magnitude,
            shape_support=shape_support,
            shape_basis_support=shape_basis_support,
            shape_residual=shape_residual,
            phase_tangent=phase_tangent,
            valid=valid,
        )
