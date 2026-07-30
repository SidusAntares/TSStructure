"""Differentiable functional lifting on a shared physical time axis."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


_CUBIC_DEGREE = 3


@dataclass(frozen=True)
class TemporalFunctionalOutput:
    standardized_tokens: Tensor
    normalized_positions: Tensor
    coefficients: Tensor
    function: Tensor
    derivative: Tensor

    time_mask: Tensor
    solve_valid: Tensor

    num_valid_observations: Tensor
    num_distinct_observations: Tensor
    time_span: Tensor
    max_internal_gap: Tensor


def _finite_float(name: str, value: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


class SourceRunningStandardizer(nn.Module):
    """Standardize features with source-only exponential running moments."""

    def __init__(
        self,
        feature_dim: int,
        momentum: float = 0.99,
        min_scale: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        momentum = _finite_float("momentum", momentum)
        min_scale = _finite_float("min_scale", min_scale)
        eps = _finite_float("eps", eps)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if min_scale <= 0:
            raise ValueError("min_scale must be greater than zero")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")

        self.feature_dim = feature_dim
        self.momentum = momentum
        self.min_scale = min_scale
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(feature_dim))
        self.register_buffer("running_second_moment", torch.zeros(feature_dim))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def _validate_tokens(self, tokens: Tensor, name: str) -> None:
        if not isinstance(tokens, Tensor) or tokens.ndim != 3:
            raise ValueError(f"{name} must have shape [B, L, D]")
        if tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f"{name} feature dimension must equal feature_dim={self.feature_dim}"
            )
        if not tokens.is_floating_point():
            raise ValueError(f"{name} must use a floating-point dtype")

    @torch.no_grad()
    def update(self, source_tokens: Tensor, time_mask: Tensor) -> None:
        self._validate_tokens(source_tokens, "source_tokens")
        if not isinstance(time_mask, Tensor) or time_mask.dtype != torch.bool:
            raise ValueError("time_mask must be a boolean tensor with shape [B, L]")
        if time_mask.shape != source_tokens.shape[:2]:
            raise ValueError("time_mask must be a boolean tensor with shape [B, L]")

        mask = time_mask.to(device=source_tokens.device)
        valid_tokens = source_tokens[mask]
        if valid_tokens.shape[0] == 0:
            return
        if not torch.isfinite(valid_tokens).all().item():
            raise ValueError("valid source tokens must be finite")
        batch_mean = valid_tokens.mean(dim=0).to(
            device=self.running_mean.device, dtype=self.running_mean.dtype
        )
        batch_second_moment = valid_tokens.square().mean(dim=0).to(
            device=self.running_second_moment.device,
            dtype=self.running_second_moment.dtype,
        )
        if self.num_updates.item() == 0:
            self.running_mean.copy_(batch_mean)
            self.running_second_moment.copy_(batch_second_moment)
        else:
            self.running_mean.mul_(self.momentum).add_(
                batch_mean, alpha=1.0 - self.momentum
            )
            self.running_second_moment.mul_(self.momentum).add_(
                batch_second_moment, alpha=1.0 - self.momentum
            )
        self.num_updates.add_(1)

    def forward(self, tokens: Tensor) -> Tensor:
        self._validate_tokens(tokens, "tokens")
        if self.num_updates.item() == 0:
            return tokens
        running_mean = self.running_mean.to(device=tokens.device, dtype=tokens.dtype)
        running_second_moment = self.running_second_moment.to(
            device=tokens.device, dtype=tokens.dtype
        )
        variance = running_second_moment - running_mean.square()
        scale = torch.sqrt(variance.clamp_min(self.min_scale**2))
        return (tokens - running_mean) / scale.clamp_min(self.eps)


def _resolve_time_mask(
    time_mask: Tensor,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> Tensor:
    if not isinstance(time_mask, Tensor):
        raise ValueError("time_mask must be a torch.Tensor")
    if time_mask.ndim == 1:
        if time_mask.shape[0] != sequence_length:
            raise ValueError("time_mask must have shape [B, L] or [L]")
        time_mask = time_mask.unsqueeze(0).expand(batch_size, -1)
    elif time_mask.ndim == 2:
        if time_mask.shape != (batch_size, sequence_length):
            raise ValueError("time_mask must have shape [B, L] or [L]")
    else:
        raise ValueError("time_mask must have shape [B, L] or [L]")
    if (
        time_mask.is_complex()
        or not torch.isfinite(time_mask).all().item()
        or not torch.all((time_mask == 0) | (time_mask == 1)).item()
    ):
        raise ValueError("time_mask must contain only finite 0/1 values")
    return time_mask.to(device=device, dtype=torch.bool)


def _resolve_physical_positions(
    positions: Tensor,
    time_mask: Tensor,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    time_reference: float,
    time_scale: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not isinstance(positions, Tensor):
        raise ValueError("positions must be a torch.Tensor")
    if positions.ndim == 1:
        if positions.shape[0] != sequence_length:
            raise ValueError("positions must have shape [B, L] or [L]")
        positions = positions.unsqueeze(0).expand(batch_size, -1)
    elif positions.ndim == 2:
        if positions.shape != (batch_size, sequence_length):
            raise ValueError("positions must have shape [B, L] or [L]")
    else:
        raise ValueError("positions must have shape [B, L] or [L]")
    if positions.is_complex() or positions.dtype == torch.bool:
        raise ValueError("positions must contain finite real values")
    position_dtype = torch.float64 if positions.dtype == torch.float64 else torch.float32
    if not positions.is_floating_point():
        position_dtype = torch.float64
    positions = positions.to(device=device, dtype=position_dtype)
    if not torch.isfinite(positions).all().item():
        raise ValueError("positions must contain only finite values")

    normalized_positions = (positions - time_reference) / time_scale
    below = time_mask & (normalized_positions < -eps)
    above = time_mask & (normalized_positions > 1.0 + eps)
    if torch.any(below | above).item():
        raise ValueError("valid normalized positions must lie in [0, 1]")
    normalized_positions = torch.where(
        time_mask & (normalized_positions < 0),
        torch.zeros_like(normalized_positions),
        normalized_positions,
    )
    normalized_positions = torch.where(
        time_mask & (normalized_positions > 1),
        torch.ones_like(normalized_positions),
        normalized_positions,
    )

    num_valid_observations = time_mask.sum(dim=1, dtype=torch.long)
    num_distinct_observations = num_valid_observations.clone()
    time_span = torch.zeros(
        batch_size, device=device, dtype=normalized_positions.dtype
    )
    max_internal_gap = torch.zeros_like(time_span)
    for batch_index in range(batch_size):
        valid_positions = normalized_positions[batch_index, time_mask[batch_index]]
        if valid_positions.numel() < 2:
            continue
        differences = valid_positions[1:] - valid_positions[:-1]
        if torch.any(differences <= 0).item():
            raise ValueError("valid positions must be strictly increasing")
        time_span[batch_index] = valid_positions[-1] - valid_positions[0]
        max_internal_gap[batch_index] = differences.max()
    return (
        normalized_positions,
        num_valid_observations,
        num_distinct_observations,
        time_span,
        max_internal_gap,
    )


def _basis_of_degree(positions: Tensor, knots: Tensor, degree: int) -> Tensor:
    num_basis = knots.numel() - _CUBIC_DEGREE - 1
    expanded = positions.unsqueeze(-1)
    basis = (
        (expanded >= knots[:-1]) & (expanded < knots[1:])
    ).to(dtype=positions.dtype)
    endpoint = positions == 1
    endpoint_basis = torch.zeros_like(basis)
    endpoint_basis[..., num_basis - 1] = 1
    basis = torch.where(endpoint.unsqueeze(-1), endpoint_basis, basis)
    for order in range(1, degree + 1):
        count = knots.numel() - order - 1
        left_denominator = knots[order : order + count] - knots[:count]
        right_denominator = (
            knots[order + 1 : order + count + 1] - knots[1 : count + 1]
        )
        left_safe = torch.where(
            left_denominator > 0,
            left_denominator,
            torch.ones_like(left_denominator),
        )
        right_safe = torch.where(
            right_denominator > 0,
            right_denominator,
            torch.ones_like(right_denominator),
        )
        left = (expanded - knots[:count]) / left_safe
        right = (knots[order + 1 : order + count + 1] - expanded) / right_safe
        left = left * (left_denominator > 0)
        right = right * (right_denominator > 0)
        basis = left * basis[..., :count] + right * basis[..., 1 : count + 1]
    return basis


def _basis_derivative(positions: Tensor, knots: Tensor, degree: int) -> Tensor:
    count = knots.numel() - degree - 1
    if degree == 0:
        return torch.zeros(
            *positions.shape, count, device=positions.device, dtype=positions.dtype
        )
    lower = _basis_of_degree(positions, knots, degree - 1)
    first_denominator = knots[degree : degree + count] - knots[:count]
    second_denominator = (
        knots[degree + 1 : degree + count + 1] - knots[1 : count + 1]
    )
    first_scale = torch.where(
        first_denominator > 0,
        degree / first_denominator,
        torch.zeros_like(first_denominator),
    )
    second_scale = torch.where(
        second_denominator > 0,
        degree / second_denominator,
        torch.zeros_like(second_denominator),
    )
    return (
        first_scale * lower[..., :count]
        - second_scale * lower[..., 1 : count + 1]
    )


def _basis_second_derivative(
    positions: Tensor, knots: Tensor, degree: int
) -> Tensor:
    count = knots.numel() - degree - 1
    if degree < 2:
        return torch.zeros(
            *positions.shape, count, device=positions.device, dtype=positions.dtype
        )
    lower_derivative = _basis_derivative(positions, knots, degree - 1)
    first_denominator = knots[degree : degree + count] - knots[:count]
    second_denominator = (
        knots[degree + 1 : degree + count + 1] - knots[1 : count + 1]
    )
    first_scale = torch.where(
        first_denominator > 0,
        degree / first_denominator,
        torch.zeros_like(first_denominator),
    )
    second_scale = torch.where(
        second_denominator > 0,
        degree / second_denominator,
        torch.zeros_like(second_denominator),
    )
    return (
        first_scale * lower_derivative[..., :count]
        - second_scale * lower_derivative[..., 1 : count + 1]
    )


def _evaluate_cubic_bspline(
    positions: Tensor, knots: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    knots = knots.to(device=positions.device, dtype=positions.dtype)
    basis = _basis_of_degree(positions, knots, _CUBIC_DEGREE)
    derivative = _basis_derivative(positions, knots, _CUBIC_DEGREE)
    second_derivative = _basis_second_derivative(
        positions, knots, _CUBIC_DEGREE
    )
    return basis, derivative, second_derivative


def _open_uniform_cubic_knots(num_basis: int) -> Tensor:
    internal_count = num_basis - _CUBIC_DEGREE - 1
    internal = torch.linspace(
        0.0, 1.0, internal_count + 2, dtype=torch.float64
    )[1:-1]
    return torch.cat(
        [
            torch.zeros(_CUBIC_DEGREE + 1, dtype=torch.float64),
            internal,
            torch.ones(_CUBIC_DEGREE + 1, dtype=torch.float64),
        ]
    )


class TemporalFunctionalLift(nn.Module):
    """Fit masked channel-component tokens as smooth functions of physical time."""

    def __init__(
        self,
        num_channels: int,
        channel_feature_dim: int,
        num_basis: int = 12,
        canonical_grid_size: int = 64,
        roughness_grid_size: int = 256,
        smoothing_weight: float = 1e-3,
        time_reference: float = 0.0,
        time_scale: float = 366.0,
        statistics_momentum: float = 0.99,
        min_feature_scale: float = 1e-3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        integer_values = {
            "num_channels": num_channels,
            "channel_feature_dim": channel_feature_dim,
            "num_basis": num_basis,
            "canonical_grid_size": canonical_grid_size,
            "roughness_grid_size": roughness_grid_size,
        }
        for name, value in integer_values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if channel_feature_dim <= 0:
            raise ValueError("channel_feature_dim must be positive")
        if num_basis < _CUBIC_DEGREE + 1:
            raise ValueError("num_basis must be at least degree + 1")
        if canonical_grid_size < 2:
            raise ValueError("canonical_grid_size must be at least 2")
        if roughness_grid_size < max(64, canonical_grid_size):
            raise ValueError(
                "roughness_grid_size must be at least max(64, canonical_grid_size)"
            )
        smoothing_weight = _finite_float("smoothing_weight", smoothing_weight)
        time_reference = _finite_float("time_reference", time_reference)
        time_scale = _finite_float("time_scale", time_scale)
        eps = _finite_float("eps", eps)
        if smoothing_weight < 0:
            raise ValueError("smoothing_weight must be non-negative")
        if time_scale <= 0:
            raise ValueError("time_scale must be greater than zero")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")

        self.num_channels = num_channels
        self.channel_feature_dim = channel_feature_dim
        self.feature_dim = num_channels * channel_feature_dim
        self.num_basis = num_basis
        self.canonical_grid_size = canonical_grid_size
        self.roughness_grid_size = roughness_grid_size
        self.smoothing_weight = smoothing_weight
        self.time_reference = time_reference
        self.time_scale = time_scale
        self.eps = eps
        self.standardizer = SourceRunningStandardizer(
            feature_dim=self.feature_dim,
            momentum=statistics_momentum,
            min_scale=min_feature_scale,
            eps=eps,
        )

        knots = _open_uniform_cubic_knots(num_basis)
        canonical_grid = torch.linspace(
            0.0, 1.0, canonical_grid_size, dtype=torch.float64
        )
        canonical_basis, canonical_derivative, _ = _evaluate_cubic_bspline(
            canonical_grid, knots
        )
        roughness_grid = torch.linspace(
            0.0, 1.0, roughness_grid_size, dtype=torch.float64
        )
        _, _, roughness_second_derivative = _evaluate_cubic_bspline(
            roughness_grid, knots
        )
        integration_weights = torch.ones(
            roughness_grid_size, dtype=torch.float64
        )
        integration_weights[[0, -1]] = 0.5
        integration_weights = integration_weights / (roughness_grid_size - 1)
        roughness_matrix = roughness_second_derivative.T @ (
            integration_weights.unsqueeze(-1) * roughness_second_derivative
        )
        roughness_matrix = 0.5 * (roughness_matrix + roughness_matrix.T)

        self.register_buffer("knots", knots)
        self.register_buffer("canonical_grid", canonical_grid)
        self.register_buffer("canonical_basis", canonical_basis)
        self.register_buffer(
            "canonical_basis_derivative", canonical_derivative
        )
        self.register_buffer("roughness_matrix", roughness_matrix)

    def _validate_component_tokens(self, component_tokens: Tensor) -> None:
        if not isinstance(component_tokens, Tensor) or component_tokens.ndim != 4:
            raise ValueError(
                "component_tokens must be a four-dimensional [B, L, C, P] tensor"
            )
        if component_tokens.shape[2] != self.num_channels:
            raise ValueError(
                "component_tokens channel dimension must equal "
                f"num_channels={self.num_channels}"
            )
        if component_tokens.shape[3] != self.channel_feature_dim:
            raise ValueError(
                "component_tokens feature dimension must equal "
                f"channel_feature_dim={self.channel_feature_dim}"
            )
        if not component_tokens.is_floating_point():
            raise ValueError("component_tokens must use a floating-point dtype")

    @torch.no_grad()
    def update_source_statistics(
        self, component_tokens: Tensor, time_mask: Tensor
    ) -> None:
        self._validate_component_tokens(component_tokens)
        batch_size, sequence_length = component_tokens.shape[:2]
        resolved_time_mask = _resolve_time_mask(
            time_mask,
            batch_size,
            sequence_length,
            component_tokens.device,
        )
        flattened = component_tokens.reshape(
            batch_size, sequence_length, self.feature_dim
        )
        self.standardizer.update(flattened, resolved_time_mask)

    def forward(
        self,
        component_tokens: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> TemporalFunctionalOutput:
        self._validate_component_tokens(component_tokens)
        batch_size, sequence_length = component_tokens.shape[:2]
        resolved_time_mask = _resolve_time_mask(
            time_mask,
            batch_size,
            sequence_length,
            component_tokens.device,
        )
        flattened = component_tokens.reshape(
            batch_size, sequence_length, self.feature_dim
        )
        if not torch.isfinite(flattened[resolved_time_mask]).all().item():
            raise ValueError("valid component tokens must be finite")
        standardized_tokens = self.standardizer(flattened)
        (
            normalized_positions,
            num_valid_observations,
            num_distinct_observations,
            time_span,
            max_internal_gap,
        ) = _resolve_physical_positions(
            positions,
            resolved_time_mask,
            batch_size,
            sequence_length,
            component_tokens.device,
            self.time_reference,
            self.time_scale,
            self.eps,
        )
        dtype = component_tokens.dtype
        normalized_positions = normalized_positions.to(dtype=dtype)
        time_span = time_span.to(dtype=dtype)
        max_internal_gap = max_internal_gap.to(dtype=dtype)

        fitting_positions = torch.where(
            resolved_time_mask,
            normalized_positions,
            torch.zeros_like(normalized_positions),
        )
        design_basis, _, _ = _evaluate_cubic_bspline(
            fitting_positions, self.knots
        )
        mask_values = resolved_time_mask.unsqueeze(-1).to(dtype=dtype)
        weighted_basis = design_basis * mask_values
        masked_tokens = torch.where(
            resolved_time_mask.unsqueeze(-1),
            standardized_tokens,
            torch.zeros_like(standardized_tokens),
        )
        gram = design_basis.transpose(1, 2) @ weighted_basis
        rhs = weighted_basis.transpose(1, 2) @ masked_tokens

        roughness = self.roughness_matrix.to(
            device=component_tokens.device, dtype=dtype
        )
        identity = torch.eye(
            self.num_basis, device=component_tokens.device, dtype=dtype
        )
        gram = gram + self.smoothing_weight * roughness + self.eps * identity
        cholesky, info = torch.linalg.cholesky_ex(gram)
        cholesky_valid = info == 0
        safe_cholesky = torch.where(
            cholesky_valid.reshape(batch_size, 1, 1),
            cholesky,
            identity.expand(batch_size, -1, -1),
        )
        safe_rhs = torch.where(
            cholesky_valid.reshape(batch_size, 1, 1),
            rhs,
            torch.zeros_like(rhs),
        )
        coefficients = torch.cholesky_solve(safe_rhs, safe_cholesky)

        canonical_basis = self.canonical_basis.to(
            device=component_tokens.device, dtype=dtype
        )
        canonical_derivative = self.canonical_basis_derivative.to(
            device=component_tokens.device, dtype=dtype
        )
        function = canonical_basis.unsqueeze(0) @ coefficients
        derivative = canonical_derivative.unsqueeze(0) @ coefficients
        finite_valid = (
            torch.isfinite(coefficients).reshape(batch_size, -1).all(dim=1)
            & torch.isfinite(function).reshape(batch_size, -1).all(dim=1)
            & torch.isfinite(derivative).reshape(batch_size, -1).all(dim=1)
        )
        observation_valid = num_distinct_observations >= 2
        solve_valid = observation_valid & cholesky_valid & finite_valid
        solve_mask = solve_valid.reshape(batch_size, 1, 1)
        coefficients = torch.where(
            solve_mask, coefficients, torch.zeros_like(coefficients)
        )
        function = torch.where(solve_mask, function, torch.zeros_like(function))
        derivative = torch.where(
            solve_mask, derivative, torch.zeros_like(derivative)
        )
        return TemporalFunctionalOutput(
            standardized_tokens=standardized_tokens,
            normalized_positions=normalized_positions,
            coefficients=coefficients,
            function=function,
            derivative=derivative,
            time_mask=resolved_time_mask,
            solve_valid=solve_valid,
            num_valid_observations=num_valid_observations,
            num_distinct_observations=num_distinct_observations,
            time_span=time_span,
            max_internal_gap=max_internal_gap,
        )
