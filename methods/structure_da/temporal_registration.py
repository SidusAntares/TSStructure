"""Generic pure monotone-warp utilities retained for later rounds.

Only stateless helper functions and frozen dataclasses live here. The learned
warp estimator and the source running SRVF template were removed from the main
chain in Round 1 and are not re-exported.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .temporal_functional import _finite_float


def _positive_integer(name: str, value: int, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


@dataclass(frozen=True)
class MonotoneWarpOutput:
    interval_logits: Tensor
    interval_widths: Tensor
    warp: Tensor
    warp_derivative: Tensor


@dataclass(frozen=True)
class MonotoneWarpCandidatesOutput:
    interval_logits: Tensor
    interval_widths: Tensor
    warp: Tensor
    warp_derivative: Tensor
    inverse_warp: Tensor


def invert_monotone_warp(
    warp: Tensor,
    query: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Invert endpoint-preserving piecewise-linear monotone warps."""

    eps = _finite_float("eps", eps)
    if eps <= 0:
        raise ValueError("eps must be greater than zero")
    if not isinstance(warp, Tensor) or warp.ndim < 1 or warp.shape[-1] < 2:
        raise ValueError("warp must have shape [..., K] with K >= 2")
    if not warp.is_floating_point():
        raise ValueError("warp must use a floating-point dtype")
    if not torch.isfinite(warp).all().item():
        raise ValueError("warp must contain only finite values")
    if torch.any((warp < 0) | (warp > 1)).item():
        raise ValueError("warp must lie in [0, 1]")
    if torch.any(warp[..., 1:] <= warp[..., :-1]).item():
        raise ValueError("warp must be strictly increasing")
    if not torch.allclose(
        warp[..., 0], torch.zeros_like(warp[..., 0]), atol=eps, rtol=0.0
    ):
        raise ValueError("warp must start at 0")
    if not torch.allclose(
        warp[..., -1], torch.ones_like(warp[..., -1]), atol=eps, rtol=0.0
    ):
        raise ValueError("warp must end at 1")

    grid_size = warp.shape[-1]
    leading_shape = warp.shape[:-1]
    if query is None:
        query_tensor = torch.linspace(
            0.0, 1.0, grid_size, device=warp.device, dtype=warp.dtype
        ).expand(*leading_shape, grid_size)
    else:
        try:
            query_tensor = torch.as_tensor(query, device=warp.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                "query must be a real tensor convertible to warp dtype"
            ) from error
        if query_tensor.is_complex() or query_tensor.dtype == torch.bool:
            raise ValueError("query must contain real numeric values")
        query_tensor = query_tensor.to(dtype=warp.dtype)
        if query_tensor.ndim < 1:
            raise ValueError("query must have shape [L] or [..., L]")
        if query_tensor.ndim == 1:
            query_tensor = query_tensor.expand(*leading_shape, query_tensor.shape[0])
        elif query_tensor.shape[:-1] != leading_shape:
            raise ValueError("query leading dimensions must match warp")
        if not torch.isfinite(query_tensor).all().item():
            raise ValueError("query must contain only finite values")
        if torch.any((query_tensor < 0) | (query_tensor > 1)).item():
            raise ValueError("query must lie in [0, 1]")

    output_length = query_tensor.shape[-1]
    flat_warp = warp.reshape(-1, grid_size).contiguous()
    flat_query = query_tensor.reshape(-1, output_length).contiguous()
    upper = torch.searchsorted(flat_warp, flat_query, right=True).clamp(
        min=1, max=grid_size - 1
    )
    lower = upper - 1
    lower_value = torch.gather(flat_warp, 1, lower)
    upper_value = torch.gather(flat_warp, 1, upper)
    fraction = (flat_query - lower_value) / (
        upper_value - lower_value
    ).clamp_min(eps)
    inverse = (lower.to(dtype=warp.dtype) + fraction) / (grid_size - 1)
    inverse = torch.where(flat_query == 0, torch.zeros_like(inverse), inverse)
    inverse = torch.where(flat_query == 1, torch.ones_like(inverse), inverse)
    return inverse.clamp(0.0, 1.0).reshape(*leading_shape, output_length)


def select_warp_candidate(
    candidates: MonotoneWarpCandidatesOutput,
    candidate_index: Tensor,
) -> MonotoneWarpOutput:
    """Select one warp candidate independently for every batch row."""

    if not isinstance(candidates, MonotoneWarpCandidatesOutput):
        raise ValueError("candidates must be a MonotoneWarpCandidatesOutput")
    if candidates.warp.ndim != 3:
        raise ValueError("candidate warp must have shape [B, G, K]")
    batch_size, num_candidates, grid_size = candidates.warp.shape
    expected_shapes = {
        "interval_logits": (batch_size, num_candidates, grid_size - 1),
        "interval_widths": (batch_size, num_candidates, grid_size - 1),
        "warp_derivative": (batch_size, num_candidates, grid_size),
        "inverse_warp": (batch_size, num_candidates, grid_size),
    }
    for name, expected_shape in expected_shapes.items():
        value = getattr(candidates, name)
        if not isinstance(value, Tensor) or value.shape != expected_shape:
            raise ValueError(f"candidate {name} has an invalid shape")
    if not isinstance(candidate_index, Tensor) or candidate_index.dtype != torch.long:
        raise ValueError("candidate_index must use torch.long dtype")
    if candidate_index.shape != (batch_size,):
        raise ValueError("candidate_index must have shape [B]")
    if candidate_index.device != candidates.warp.device:
        raise ValueError("candidate_index must be on the candidates device")
    if torch.any((candidate_index < 0) | (candidate_index >= num_candidates)).item():
        raise ValueError("candidate_index values must lie in the candidate range")

    def gather(value: Tensor) -> Tensor:
        index = candidate_index[:, None, None].expand(-1, 1, value.shape[-1])
        return torch.gather(value, 1, index).squeeze(1)

    return MonotoneWarpOutput(
        interval_logits=gather(candidates.interval_logits),
        interval_widths=gather(candidates.interval_widths),
        warp=gather(candidates.warp),
        warp_derivative=gather(candidates.warp_derivative),
    )


def _warp_sequence(sequence: Tensor, warp: Tensor) -> Tensor:
    device_type = sequence.device.type if isinstance(sequence, Tensor) else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        if isinstance(sequence, Tensor) and sequence.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            sequence = sequence.float()
        if isinstance(warp, Tensor) and warp.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            warp = warp.float()
        return _warp_sequence_float32(sequence, warp)


def _warp_sequence_float32(sequence: Tensor, warp: Tensor) -> Tensor:
    if not isinstance(sequence, Tensor) or sequence.ndim != 3:
        raise ValueError("sequence must have shape [B, K, D]")
    if not sequence.is_floating_point():
        raise ValueError("sequence must use a floating-point dtype")
    if sequence.shape[1] < 2:
        raise ValueError("sequence grid size must be at least 2")
    if not torch.isfinite(sequence).all().item():
        raise ValueError("sequence must contain only finite values")
    if not isinstance(warp, Tensor) or warp.shape != sequence.shape[:2]:
        raise ValueError("warp must have shape [B, K]")
    if not warp.is_floating_point():
        raise ValueError("warp must use a floating-point dtype")
    if warp.device != sequence.device or warp.dtype != sequence.dtype:
        raise ValueError("warp must match sequence device and dtype")
    if not torch.isfinite(warp).all().item():
        raise ValueError("warp must contain only finite values")
    if torch.any((warp < 0) | (warp > 1)).item():
        raise ValueError("warp must lie in [0, 1]")
    if torch.any(warp[:, 1:] <= warp[:, :-1]).item():
        raise ValueError("warp must be strictly increasing")

    input_tensor = sequence.transpose(1, 2).unsqueeze(2)
    grid_x = 2.0 * warp - 1.0
    grid_y = torch.zeros_like(grid_x)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)
    warped = F.grid_sample(
        input_tensor,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.squeeze(2).transpose(1, 2)


def _apply_srvf_group_action(
    srvf: Tensor,
    warp: Tensor,
    warp_derivative: Tensor,
    eps: float,
) -> Tensor:
    device_type = srvf.device.type if isinstance(srvf, Tensor) else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        if isinstance(srvf, Tensor) and srvf.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            srvf = srvf.float()
        if isinstance(warp, Tensor) and warp.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            warp = warp.float()
        if isinstance(warp_derivative, Tensor) and warp_derivative.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            warp_derivative = warp_derivative.float()
        return _apply_srvf_group_action_float32(
            srvf, warp, warp_derivative, eps
        )


def _apply_srvf_group_action_float32(
    srvf: Tensor,
    warp: Tensor,
    warp_derivative: Tensor,
    eps: float,
) -> Tensor:
    eps = _finite_float("eps", eps)
    if eps <= 0:
        raise ValueError("eps must be greater than zero")
    if not isinstance(warp_derivative, Tensor) or warp_derivative.shape != warp.shape:
        raise ValueError("warp_derivative must have shape [B, K]")
    if not warp_derivative.is_floating_point():
        raise ValueError("warp_derivative must use a floating-point dtype")
    if (
        warp_derivative.device != srvf.device
        or warp_derivative.dtype != srvf.dtype
    ):
        raise ValueError("warp_derivative must match srvf device and dtype")
    if not torch.isfinite(warp_derivative).all().item():
        raise ValueError("warp_derivative must contain only finite values")
    if torch.any(warp_derivative <= 0).item():
        raise ValueError("warp_derivative must be strictly positive")
    warped_srvf = _warp_sequence_float32(srvf, warp)
    return warped_srvf * torch.sqrt(
        warp_derivative.clamp_min(eps)
    ).unsqueeze(-1)
