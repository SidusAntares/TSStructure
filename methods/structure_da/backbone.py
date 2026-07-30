"""Channel-preserving input backbone for structure-aware models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.pse import ChannelPreservingPixelSetEncoder
from .decomposition import (
    DecompositionOutput,
    SymmetricTimeKernelDecomposition,
)


@dataclass(frozen=True)
class StructureBackboneOutput:
    """Channel tokens, resolved date validity, and temporal components."""

    channel_tokens: Tensor
    time_mask: Tensor
    decomposition: DecompositionOutput


def _resolve_time_mask(
    time_mask: Tensor | None,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> Tensor:
    if time_mask is None:
        return torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=device,
        )
    if not isinstance(time_mask, Tensor):
        raise ValueError("time_mask must be a torch.Tensor or None")
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


class StructureBackbone(nn.Module):
    """Encode parcel pixels per channel, then decompose only along time."""

    def __init__(
        self,
        num_channels: int = 10,
        channel_feature_dim: int = 16,
        pixel_hidden_dim: int = 16,
        tau_fast_init: float = 0.05,
        tau_slow_init: float = 0.20,
        tau_min: float = 1e-4,
        delta_tau_min: float = 1e-4,
        time_scale: float = 365.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.pixel_set_encoder = ChannelPreservingPixelSetEncoder(
            num_channels=num_channels,
            channel_feature_dim=channel_feature_dim,
            pixel_hidden_dim=pixel_hidden_dim,
            eps=eps,
        )
        self.decomposition = SymmetricTimeKernelDecomposition(
            tau_fast_init=tau_fast_init,
            tau_slow_init=tau_slow_init,
            tau_min=tau_min,
            delta_tau_min=delta_tau_min,
            time_scale=time_scale,
            eps=eps,
        )

    def forward(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        time_mask: Tensor | None = None,
    ) -> StructureBackboneOutput:
        channel_tokens = self.pixel_set_encoder(
            pixels,
            valid_pixels,
        )
        batch_size, sequence_length = channel_tokens.shape[:2]
        resolved_time_mask = _resolve_time_mask(
            time_mask,
            batch_size,
            sequence_length,
            pixels.device,
        )
        decomposition = self.decomposition(
            channel_tokens,
            positions,
            resolved_time_mask,
        )
        return StructureBackboneOutput(
            channel_tokens=channel_tokens,
            time_mask=resolved_time_mask,
            decomposition=decomposition,
        )
