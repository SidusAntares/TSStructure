"""Pixel-set input backbone for structure-aware models."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from models.pse import PixelSetEncoder
from .decomposition import (
    DecompositionOutput,
    SymmetricTimeKernelDecomposition,
)


@dataclass(frozen=True)
class StructureBackboneOutput:
    """PSE tokens, resolved date validity, and temporal components."""

    tokens: Tensor
    time_mask: Tensor
    normalized_positions: Tensor
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
    """Encode parcel pixel sets, then decompose their temporal features."""

    def __init__(
        self,
        input_dim: int = 10,
        mlp1: list[int] | None = None,
        pooling: str = "mean_std",
        mlp2: list[int] | None = None,
        with_extra: bool = False,
        extra_size: int = 4,
        tau_fast_init: float = 0.05,
        tau_slow_init: float = 0.20,
        tau_min: float = 1e-4,
        delta_tau_min: float = 1e-4,
        time_reference: float = 0.0,
        time_scale: float = 365.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if mlp1 is None:
            mlp1 = [input_dim, 32, 64]
        if mlp2 is None:
            mlp2 = [128, 128]
        else:
            mlp2 = list(mlp2)
        if with_extra:
            mlp2[0] += extra_size
        self.pixel_set_encoder = PixelSetEncoder(
            input_dim=input_dim,
            mlp1=mlp1,
            pooling=pooling,
            mlp2=mlp2,
            with_extra=with_extra,
            extra_size=extra_size,
        )
        self.feature_dim = self.pixel_set_encoder.output_dim
        try:
            self.time_reference = float(time_reference)
            self.time_scale = float(time_scale)
        except (TypeError, ValueError) as error:
            raise ValueError("time_reference and time_scale must be finite") from error
        if not math.isfinite(self.time_reference):
            raise ValueError("time_reference must be finite")
        if not math.isfinite(self.time_scale) or self.time_scale <= 0:
            raise ValueError("time_scale must be finite and greater than zero")
        self.eps = float(eps)
        self.decomposition = SymmetricTimeKernelDecomposition(
            tau_fast_init=tau_fast_init,
            tau_slow_init=tau_slow_init,
            tau_min=tau_min,
            delta_tau_min=delta_tau_min,
            time_scale=1.0,
            eps=eps,
        )

    def _normalize_positions(
        self,
        positions: Tensor,
        time_mask: Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Map shared physical positions to the model's normalized coordinates."""

        if not isinstance(positions, Tensor) or positions.is_complex() or positions.dtype == torch.bool:
            raise ValueError("physical positions must be a real torch.Tensor")
        batch_size, sequence_length = time_mask.shape
        if positions.ndim == 1 and positions.shape == (sequence_length,):
            resolved = positions.unsqueeze(0).expand(batch_size, -1)
        elif positions.ndim == 2 and positions.shape == (batch_size, sequence_length):
            resolved = positions
        else:
            raise ValueError("physical positions must have shape [L] or [B, L]")
        resolved = resolved.to(device=device, dtype=dtype)
        for sample_index in range(batch_size):
            valid = resolved[sample_index, time_mask[sample_index]]
            if not torch.isfinite(valid).all().item():
                raise ValueError(
                    f"sample {sample_index} physical positions must be finite"
                )
            if valid.numel() > 1 and not torch.all(valid[1:] > valid[:-1]).item():
                raise ValueError(
                    f"sample {sample_index} physical positions must be strictly increasing"
                )
        safe = torch.where(
            time_mask,
            resolved,
            torch.full_like(resolved, self.time_reference),
        )
        normalized = (safe - self.time_reference) / self.time_scale
        valid_normalized = normalized[time_mask]
        tolerance = 1e-6
        if valid_normalized.numel() and (
            torch.any(valid_normalized < -tolerance).item()
            or torch.any(valid_normalized > 1.0 + tolerance).item()
        ):
            raise ValueError("valid normalized positions must lie in [0, 1]")
        if not torch.isfinite(normalized).all().item():
            raise ValueError("normalized positions must be finite")
        return torch.where(time_mask, normalized, torch.zeros_like(normalized))

    def forward(
        self,
        pixels: Tensor,
        valid_pixels: Tensor,
        positions: Tensor,
        extra: Tensor | None,
        time_mask: Tensor | None = None,
    ) -> StructureBackboneOutput:
        tokens = self.pixel_set_encoder(
            pixels,
            valid_pixels,
            extra,
        )
        batch_size, sequence_length = tokens.shape[:2]
        resolved_time_mask = _resolve_time_mask(
            time_mask,
            batch_size,
            sequence_length,
            pixels.device,
        )
        # Keep physical time coordinates out of low-precision autocast.
        # The decomposition path intentionally promotes fp16/bf16 activations
        # to float32 for stable temporal geometry, and the raw LTAE consumes
        # those float32 T/S components with a float32 ContinuousTime2Vec.
        # Letting positions inherit an autocast token dtype would therefore
        # create an artificial dtype mismatch at the shared time encoder.
        position_dtype = (
            torch.float32
            if tokens.dtype in (torch.float16, torch.bfloat16)
            else tokens.dtype
        )
        normalized_positions = self._normalize_positions(
            positions,
            resolved_time_mask,
            dtype=position_dtype,
            device=tokens.device,
        )
        decomposition = self.decomposition(
            tokens,
            normalized_positions,
            resolved_time_mask,
        )
        return StructureBackboneOutput(
            tokens=tokens,
            time_mask=resolved_time_mask,
            normalized_positions=normalized_positions,
            decomposition=decomposition,
        )
