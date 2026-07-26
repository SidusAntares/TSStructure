"""Explicit temporal and latent-channel structural relation operators."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn


Scalar = Union[float, torch.Tensor]


@dataclass(frozen=True)
class StructureOutput:
    """Local and sample-level readouts derived from one base relation.

    ``valid`` is a boolean tensor with shape ``[B]`` indicating whether each
    sample has enough observations (and channels, where applicable) to define
    the requested structure.
    """

    local: torch.Tensor
    statistic: torch.Tensor
    valid: torch.Tensor


def _positive_finite_float(name: str, value: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be a finite real number greater than zero")
    return converted


def _prepare_inputs(
    H: torch.Tensor,
    positions: torch.Tensor,
    time_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(H, torch.Tensor):
        raise ValueError("H must be a torch.Tensor")
    if H.ndim != 3:
        raise ValueError("H must have shape [B, L, D]")
    if H.shape[1] < 1:
        raise ValueError("H must contain at least one time point")
    if not H.is_floating_point():
        raise ValueError("H must use a floating-point dtype")

    batch_size, sequence_length, _ = H.shape
    if not isinstance(positions, torch.Tensor):
        raise ValueError("positions must be a torch.Tensor")
    if positions.ndim == 1:
        if positions.shape[0] != sequence_length:
            raise ValueError("positions length must match H")
        positions = positions.unsqueeze(0).expand(batch_size, -1)
    elif positions.ndim == 2:
        if positions.shape != (batch_size, sequence_length):
            raise ValueError("positions must have shape [B, L] or [L]")
    else:
        raise ValueError("positions must have shape [B, L] or [L]")
    if positions.is_complex() or positions.dtype == torch.bool:
        raise ValueError("positions must contain real numeric timestamps")

    if positions.is_floating_point():
        position_dtype = (
            torch.float64 if positions.dtype == torch.float64 else torch.float32
        )
    else:
        position_dtype = torch.float64
    positions = positions.to(device=H.device, dtype=position_dtype)
    if not torch.isfinite(positions).all().item():
        raise ValueError("positions must contain only finite values")

    if time_mask is None:
        mask = torch.ones(
            batch_size, sequence_length, dtype=torch.bool, device=H.device
        )
    else:
        if not isinstance(time_mask, torch.Tensor):
            raise ValueError("time_mask must be a torch.Tensor")
        if time_mask.ndim == 1:
            if time_mask.shape[0] != sequence_length:
                raise ValueError("time_mask length must match H")
            time_mask = time_mask.unsqueeze(0).expand(batch_size, -1)
        elif time_mask.ndim == 2:
            if time_mask.shape != (batch_size, sequence_length):
                raise ValueError("time_mask must have shape [B, L] or [L]")
        else:
            raise ValueError("time_mask must have shape [B, L] or [L]")
        if time_mask.is_complex() or not torch.all(
            (time_mask == 0) | (time_mask == 1)
        ).item():
            raise ValueError("time_mask must contain only boolean or 0/1 values")
        mask = time_mask.to(device=H.device, dtype=torch.bool)

    H_valid = torch.where(mask.unsqueeze(-1), H, torch.zeros_like(H))
    if not torch.isfinite(H_valid[mask]).all().item():
        raise ValueError("valid H values must be finite")
    return H_valid, positions, mask


def _computation_dtype(H: torch.Tensor) -> torch.dtype:
    if H.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return H.dtype


def _cast_finite_like(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    limit = torch.finfo(reference.dtype).max
    value = torch.nan_to_num(value, nan=0.0, posinf=limit, neginf=-limit)
    return value.clamp(min=-limit, max=limit).to(dtype=reference.dtype)


def _time_coverage_weights(
    positions: torch.Tensor,
    mask: torch.Tensor,
    time_scale: float,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute vectorized trapezoidal time-coverage weights in input order."""

    sort_keys = torch.where(mask, positions, torch.full_like(positions, math.inf))
    sorted_positions, sorted_indices = torch.sort(sort_keys, dim=1)
    sorted_mask = torch.gather(mask, 1, sorted_indices)
    sorted_positions = torch.where(
        sorted_mask, sorted_positions, torch.zeros_like(sorted_positions)
    )

    group_start = sorted_mask.clone()
    group_start[:, 1:] = sorted_mask[:, 1:] & (
        ~sorted_mask[:, :-1]
        | (sorted_positions[:, 1:] != sorted_positions[:, :-1])
    )
    group_ids = group_start.to(torch.long).cumsum(dim=1) - 1
    safe_group_ids = group_ids.clamp_min(0)
    group_positions = torch.zeros_like(sorted_positions).scatter_add(
        1,
        safe_group_ids,
        torch.where(group_start, sorted_positions, torch.zeros_like(sorted_positions)),
    )
    group_sizes = torch.zeros_like(sorted_positions).scatter_add(
        1, safe_group_ids, sorted_mask.to(sorted_positions.dtype)
    )
    number_of_groups = group_start.sum(dim=1, keepdim=True)
    group_mask = (
        torch.arange(positions.shape[1], device=positions.device).unsqueeze(0)
        < number_of_groups
    )

    adjacent_valid = group_mask[:, 1:] & group_mask[:, :-1]
    gaps = (group_positions[:, 1:] - group_positions[:, :-1]) / time_scale
    gaps = torch.where(adjacent_valid, gaps, torch.zeros_like(gaps))
    zero_column = torch.zeros(
        positions.shape[0], 1, dtype=positions.dtype, device=positions.device
    )
    left_gaps = torch.cat([zero_column, gaps], dim=1)
    right_gaps = torch.cat([gaps, zero_column], dim=1)
    coverage = 0.5 * (left_gaps + right_gaps)

    counts = sorted_mask.sum(dim=1, keepdim=True)
    valid = counts >= 2
    total = coverage.sum(dim=1, keepdim=True)
    group_weights = coverage / (total + eps) / group_sizes.clamp_min(1)
    normalized = torch.gather(group_weights, 1, safe_group_ids)
    normalized = normalized * sorted_mask.to(normalized.dtype)
    uniform = sorted_mask.to(positions.dtype) / counts.clamp_min(1).to(
        positions.dtype
    )
    sorted_weights = torch.where(total > 0, normalized, uniform)
    sorted_weights = torch.where(
        valid, sorted_weights, torch.zeros_like(sorted_weights)
    )

    weights = torch.zeros_like(sorted_weights).scatter(
        1, sorted_indices, sorted_weights
    )
    return weights.to(dtype=dtype)


def _prepare_tau(
    name: str,
    value: Scalar,
    reference: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.is_complex():
            raise ValueError(f"{name} must be a real scalar")
        prepared = value.to(device=reference.device, dtype=dtype).reshape(())
    else:
        try:
            prepared = torch.tensor(
                float(value), device=reference.device, dtype=dtype
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a real scalar") from error
    if not torch.isfinite(prepared).item():
        raise ValueError(f"{name} must be finite")
    return prepared


class TemporalRelationOperator(nn.Module):
    """Read local and statistical structure from pairwise temporal changes."""

    def __init__(
        self,
        time_scale: float = 365.0,
        eps_time: float = 1e-8,
        eps_norm: float = 1e-8,
    ) -> None:
        super().__init__()
        self.time_scale = _positive_finite_float("time_scale", time_scale)
        self.eps_time = _positive_finite_float("eps_time", eps_time)
        self.eps_norm = _positive_finite_float("eps_norm", eps_norm)

    def forward(
        self,
        H: torch.Tensor,
        positions: torch.Tensor,
        tau_fast: Scalar,
        tau_slow: Scalar,
        time_mask: Optional[torch.Tensor] = None,
    ) -> StructureOutput:
        """Derive temporal local rates and weighted sample statistics.

        ``tau_fast`` and ``tau_slow`` are caller-owned scalars. They remain in
        the autograd graph so the decomposition and relation operator can share
        exactly the same learned scales.
        """

        H_valid, positions, mask = _prepare_inputs(H, positions, time_mask)
        compute_dtype = _computation_dtype(H)
        tau_fast = _prepare_tau("tau_fast", tau_fast, H, compute_dtype)
        tau_slow = _prepare_tau("tau_slow", tau_slow, H, compute_dtype)
        if tau_fast.item() <= 0:
            raise ValueError("tau_fast must be greater than zero")
        if tau_slow.item() <= tau_fast.item():
            raise ValueError("tau_slow must be greater than tau_fast")

        H_compute = H_valid.to(dtype=compute_dtype)
        signed_delta = (
            positions.unsqueeze(1) - positions.unsqueeze(2)
        ) / self.time_scale
        signed_delta = signed_delta.to(dtype=compute_dtype)
        distance = torch.abs(signed_delta)

        H_difference = H_compute.unsqueeze(1) - H_compute.unsqueeze(2)
        H_difference = torch.where(
            (distance > 0).unsqueeze(-1),
            H_difference,
            torch.zeros_like(H_difference),
        )
        rate_denominator = distance + self.eps_time
        rate_denominator = torch.where(
            distance > 0,
            rate_denominator,
            torch.ones_like(rate_denominator),
        )
        rho = (
            torch.sign(signed_delta).unsqueeze(-1)
            * H_difference
            / rate_denominator.unsqueeze(-1)
        )

        sequence_length = H.shape[1]
        not_self = ~torch.eye(
            sequence_length, dtype=torch.bool, device=H.device
        ).unsqueeze(0)
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2) & not_self
        rho = torch.where(
            pair_mask.unsqueeze(-1), rho, torch.zeros_like(rho)
        )

        fast_local = self._local_readout(
            rho, distance, pair_mask, tau_fast
        )
        slow_local = self._local_readout(
            rho, distance, pair_mask, tau_slow
        )
        valid = mask.sum(dim=1) >= 2
        valid_local = valid.view(-1, 1, 1)
        fast_local = torch.where(
            valid_local, fast_local, torch.zeros_like(fast_local)
        )
        slow_local = torch.where(
            valid_local, slow_local, torch.zeros_like(slow_local)
        )

        coverage_weights = _time_coverage_weights(
            positions,
            mask,
            self.time_scale,
            self.eps_norm,
            compute_dtype,
        )
        fast_mean, fast_variance = self._weighted_moments(
            fast_local, coverage_weights
        )
        slow_mean, slow_variance = self._weighted_moments(
            slow_local, coverage_weights
        )
        local = torch.cat([fast_local, slow_local], dim=-1)
        statistic = torch.cat(
            [fast_mean, fast_variance, slow_mean, slow_variance], dim=-1
        )
        statistic = torch.where(
            valid.unsqueeze(-1), statistic, torch.zeros_like(statistic)
        )
        return StructureOutput(
            _cast_finite_like(local, H),
            _cast_finite_like(statistic, H),
            valid,
        )

    def _local_readout(
        self,
        rho: torch.Tensor,
        distance: torch.Tensor,
        pair_mask: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        kernel = torch.exp(-distance / tau) * pair_mask.to(distance.dtype)
        kernel_sum = kernel.sum(dim=-1, keepdim=True)
        normalizer = kernel_sum + self.eps_norm
        normalizer = torch.where(
            kernel_sum > 0, normalizer, torch.ones_like(normalizer)
        )
        weights = kernel / normalizer
        return torch.einsum("brs,brsd->brd", weights, rho)

    @staticmethod
    def _weighted_moments(
        local: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.einsum("bl,bld->bd", weights, local)
        centered = local - mean.unsqueeze(1)
        variance = torch.einsum("bl,bld->bd", weights, centered.square())
        return mean, variance


class ChannelRelationOperator(nn.Module):
    """Read latent-channel structure from a coverage-weighted correlation."""

    def __init__(self, time_scale: float = 365.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.time_scale = _positive_finite_float("time_scale", time_scale)
        self.eps = _positive_finite_float("eps", eps)

    def forward(
        self,
        H: torch.Tensor,
        positions: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
    ) -> StructureOutput:
        """Return channel-local responses and the off-diagonal correlation."""

        H_valid, positions, mask = _prepare_inputs(H, positions, time_mask)
        compute_dtype = _computation_dtype(H)
        H_compute = H_valid.to(dtype=compute_dtype)
        weights = _time_coverage_weights(
            positions, mask, self.time_scale, self.eps, compute_dtype
        )

        mean = torch.einsum("bl,bld->bd", weights, H_compute)
        centered = torch.where(
            mask.unsqueeze(-1),
            H_compute - mean.unsqueeze(1),
            torch.zeros_like(H_compute),
        )
        covariance = torch.einsum(
            "bl,bli,blj->bij", weights, centered, centered
        )
        inverse_std = torch.rsqrt(
            torch.diagonal(covariance, dim1=-2, dim2=-1) + self.eps
        )
        correlation = (
            covariance
            * inverse_std.unsqueeze(-1)
            * inverse_std.unsqueeze(-2)
        )

        feature_dim = H.shape[-1]
        diagonal_mask = torch.eye(
            feature_dim, dtype=torch.bool, device=H.device
        ).unsqueeze(0)
        channel_relation = correlation.masked_fill(diagonal_mask, 0)
        valid = (mask.sum(dim=1) >= 2) & (feature_dim >= 2)
        channel_relation = torch.where(
            valid.view(-1, 1, 1),
            channel_relation,
            torch.zeros_like(channel_relation),
        )

        standardized = centered * inverse_std.unsqueeze(1)
        local = torch.einsum(
            "bij,blj->bli", channel_relation, standardized
        ) / max(feature_dim - 1, 1)
        local = torch.where(
            mask.unsqueeze(-1) & valid.view(-1, 1, 1),
            local,
            torch.zeros_like(local),
        )
        return StructureOutput(
            _cast_finite_like(local, H),
            _cast_finite_like(channel_relation, H),
            valid,
        )


def vectorize_channel_statistic(statistic: torch.Tensor) -> torch.Tensor:
    """Vectorize ``[B,D,D]`` using row-major strict-upper-triangle ordering.

    The ordering is exactly ``torch.triu_indices(D, D, offset=1)``: entries
    from earlier rows precede entries from later rows, and the diagonal is
    excluded.
    """

    if not isinstance(statistic, torch.Tensor):
        raise ValueError("statistic must be a torch.Tensor")
    if statistic.ndim != 3 or statistic.shape[-2] != statistic.shape[-1]:
        raise ValueError("statistic must have shape [B, D, D]")
    feature_dim = statistic.shape[-1]
    indices = torch.triu_indices(
        feature_dim,
        feature_dim,
        offset=1,
        device=statistic.device,
    )
    return statistic[:, indices[0], indices[1]]
