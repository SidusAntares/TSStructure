"""Three-component decomposition of temporal feature sequences."""

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DecompositionOutput:
    """Trend, structured dynamics, and residual components of a sequence."""

    trend: torch.Tensor
    dynamics: torch.Tensor
    residual: torch.Tensor


def _inverse_softplus(value: float) -> float:
    """Return an unconstrained scalar whose softplus equals ``value``."""

    if value > 20.0:
        return value
    return math.log(math.expm1(value))


class SymmetricTimeKernelDecomposition(nn.Module):
    """Decompose temporal features with fast and slow symmetric time kernels.

    The two kernel scales are learned, while the decomposition itself contains
    no projections or task-specific logic. Timestamps are divided by a fixed
    global ``time_scale`` so spacing remains comparable between samples.
    Features have shape ``[B,L,D]`` and decomposition acts only along ``L``.
    """

    def __init__(
        self,
        tau_fast_init: float = 0.05,
        tau_slow_init: float = 0.20,
        tau_min: float = 1e-4,
        delta_tau_min: float = 1e-4,
        time_scale: float = 365.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        values = {
            "tau_fast_init": tau_fast_init,
            "tau_slow_init": tau_slow_init,
            "tau_min": tau_min,
            "delta_tau_min": delta_tau_min,
            "time_scale": time_scale,
            "eps": eps,
        }
        converted = {}
        for name, value in values.items():
            try:
                converted[name] = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a finite real number") from error
            if not math.isfinite(converted[name]):
                raise ValueError(f"{name} must be a finite real number")

        tau_fast_init = converted["tau_fast_init"]
        tau_slow_init = converted["tau_slow_init"]
        tau_min = converted["tau_min"]
        delta_tau_min = converted["delta_tau_min"]
        time_scale = converted["time_scale"]
        eps = converted["eps"]

        if time_scale <= 0:
            raise ValueError("time_scale must be greater than zero")
        if tau_min <= 0:
            raise ValueError("tau_min must be greater than zero")
        if delta_tau_min <= 0:
            raise ValueError("delta_tau_min must be greater than zero")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")
        if tau_fast_init <= tau_min:
            raise ValueError("tau_fast_init must be greater than tau_min")
        if tau_slow_init <= tau_fast_init + delta_tau_min:
            raise ValueError(
                "tau_slow_init must exceed tau_fast_init + delta_tau_min"
            )

        self.tau_min = tau_min
        self.delta_tau_min = delta_tau_min
        self.time_scale = time_scale
        self.eps = eps

        fast_offset = tau_fast_init - tau_min
        slow_gap = tau_slow_init - tau_fast_init - delta_tau_min
        self._tau_fast_unconstrained = nn.Parameter(
            torch.tensor(_inverse_softplus(fast_offset), dtype=torch.float32)
        )
        self._tau_gap_unconstrained = nn.Parameter(
            torch.tensor(_inverse_softplus(slow_gap), dtype=torch.float32)
        )

    @property
    def tau_fast(self) -> torch.Tensor:
        """Positive fast kernel scale."""

        return self.tau_min + F.softplus(self._tau_fast_unconstrained)

    @property
    def tau_slow(self) -> torch.Tensor:
        """Slow kernel scale, strictly greater than :attr:`tau_fast`."""

        return (
            self.tau_fast
            + self.delta_tau_min
            + F.softplus(self._tau_gap_unconstrained)
        )

    def forward(
        self,
        H: torch.Tensor,
        positions: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
    ) -> DecompositionOutput:
        device_type = H.device.type if isinstance(H, torch.Tensor) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            if isinstance(H, torch.Tensor) and H.dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                H = H.float()
            if isinstance(positions, torch.Tensor) and positions.dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                positions = positions.float()
            return self._forward_float32(H, positions, time_mask)

    def _forward_float32(
        self,
        H: torch.Tensor,
        positions: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
    ) -> DecompositionOutput:
        """Return the trend, dynamics, and residual of ``H``.

        Args:
            H: Floating-point temporal features with shape ``[B,L,D]``.
            positions: Real timestamps with shape ``[B, L]`` or ``[L]``.
            time_mask: Optional boolean or 0/1 validity mask with shape
                ``[B, L]`` or ``[L]``. Masked outputs are zero and masked
                values cannot contribute to valid queries.
        """

        if not isinstance(H, torch.Tensor):
            raise ValueError("H must be a torch.Tensor")
        if H.ndim != 3:
            raise ValueError("H must have shape [B, L, D]")
        if H.shape[1] < 1 or any(size < 1 for size in H.shape[2:]):
            raise ValueError("H feature dimensions must be non-empty")
        if not H.is_floating_point():
            raise ValueError("H must use a floating-point dtype")

        batch_size, sequence_length = H.shape[:2]
        positions = self._prepare_positions(
            positions, batch_size, sequence_length, H
        )
        mask = self._prepare_mask(time_mask, batch_size, sequence_length, H.device)
        mask_values = mask.to(dtype=H.dtype)

        mask_expanded = mask.reshape(
            batch_size,
            sequence_length,
            *([1] * (H.ndim - 2)),
        )
        H_valid = torch.where(mask_expanded, H, torch.zeros_like(H))
        if not torch.isfinite(H_valid[mask]).all().item():
            raise ValueError("valid H values must be finite")

        # Differences are formed before conversion to H.dtype so a large
        # common timestamp offset cannot erase smaller real-time intervals.
        pairwise_distance = torch.abs(
            positions.unsqueeze(-1) - positions.unsqueeze(-2)
        )
        pairwise_distance = pairwise_distance / self.time_scale
        pairwise_distance = pairwise_distance.to(dtype=H.dtype)

        tau_fast = self.tau_fast.to(device=H.device, dtype=H.dtype)
        tau_slow = self.tau_slow.to(device=H.device, dtype=H.dtype)
        M_fast = self._smooth(
            H_valid, pairwise_distance, mask_values, tau_fast
        )
        M_slow = self._smooth(
            H_valid, pairwise_distance, mask_values, tau_slow
        )

        trend = M_slow
        dynamics = M_fast - M_slow
        residual = H_valid - M_fast
        return DecompositionOutput(trend, dynamics, residual)

    def _prepare_positions(
        self,
        positions: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
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
            position_dtype = torch.int64
        positions = positions.to(device=reference.device, dtype=position_dtype)
        if not torch.isfinite(positions).all().item():
            raise ValueError("positions must contain only finite values")
        return positions

    @staticmethod
    def _prepare_mask(
        time_mask: Optional[torch.Tensor],
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if time_mask is None:
            return torch.ones(
                batch_size, sequence_length, dtype=torch.bool, device=device
            )
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
        return time_mask.to(device=device, dtype=torch.bool)

    def _smooth(
        self,
        H_valid: torch.Tensor,
        pairwise_distance: torch.Tensor,
        mask: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        kernel = torch.exp(-pairwise_distance / tau) * pair_mask
        normalizer = kernel.sum(dim=-1, keepdim=True) + self.eps
        # Invalid queries have an all-zero kernel. Giving only those rows a
        # unit denominator avoids 0/0 when eps underflows in low precision.
        normalizer = torch.where(
            mask.unsqueeze(-1).bool(), normalizer, torch.ones_like(normalizer)
        )
        weights = kernel / normalizer
        original_shape = H_valid.shape
        batch_size, sequence_length = original_shape[:2]
        flat_features = H_valid.reshape(batch_size, sequence_length, -1)
        smoothed = torch.bmm(weights, flat_features)
        smoothed = smoothed.reshape(original_shape)
        query_mask = mask.reshape(
            batch_size,
            sequence_length,
            *([1] * (H_valid.ndim - 2)),
        )
        return smoothed * query_mask
