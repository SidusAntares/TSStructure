"""Single-stream LTAE classification head for Phase-only TSStructure."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from models.ltae import ContinuousTime2Vec, MultiHeadAttention, TimeMatchFixedSinusoidal

from .representation import RawTemporalRepresentation


class LatentTemporalLTAE(nn.Module):
    """Encode the complete PSE latent temporal process ``H`` with one LTAE."""

    def __init__(
        self,
        in_channels: int,
        n_head: int = 16,
        d_k: int = 8,
        n_neurons: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        d_model: int = 256,
        *,
        time_reference: float = 0.0,
        time_scale: float = 1.0,
        max_initial_frequency: float = 16.0,
        time_encoder_type: str = "continuous_time2vec",
        timematch_pe_period: float = 1000.0,
        timematch_pe_max_shift: float = 100.0,
        calendar_scale_days: float = 365.0,
    ) -> None:
        super().__init__()
        neurons = tuple(int(v) for v in n_neurons)
        if not neurons or neurons[0] != d_model:
            raise ValueError("n_neurons must be nonempty and start with d_model")
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        self.output_dim = neurons[-1]
        self.input_projection = nn.Linear(in_channels, d_model, bias=False)
        self.input_norm = nn.LayerNorm(d_model)
        if time_encoder_type == "continuous_time2vec":
            self.time_encoder = ContinuousTime2Vec(
                d_model,
                time_reference=time_reference,
                time_scale=time_scale,
                max_initial_frequency=max_initial_frequency,
            )
        elif time_encoder_type == "timematch_fixed_sinusoidal":
            self.time_encoder = TimeMatchFixedSinusoidal(
                d_model,
                time_reference=time_reference,
                time_scale=time_scale,
                calendar_scale_days=calendar_scale_days,
                position_offset_days=timematch_pe_max_shift,
                period=timematch_pe_period,
            )
        else:
            raise ValueError(
                "time_encoder_type must be 'continuous_time2vec' or "
                "'timematch_fixed_sinusoidal'"
            )
        self.time_encoder_type = time_encoder_type
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=d_model)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(neurons[:-1], neurons[1:]):
            layers.extend([nn.Linear(in_dim, out_dim, bias=False), nn.ReLU()])
        self.projection = nn.Sequential(*layers)
        self.dropout = nn.Dropout(float(dropout))
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        latent: Tensor,
        positions: Tensor,
        mask: Tensor,
    ) -> RawTemporalRepresentation:
        if not isinstance(latent, Tensor) or latent.ndim != 3:
            raise ValueError("latent must have shape [B,L,D]")
        if latent.shape[-1] != self.in_channels or not latent.is_floating_point():
            raise ValueError("latent must be floating point with in_channels features")
        if not isinstance(mask, Tensor) or mask.shape != latent.shape[:2]:
            raise ValueError("mask must have shape [B,L]")
        mask = mask.to(device=latent.device, dtype=torch.bool)
        if positions.ndim == 1:
            resolved_positions = positions.unsqueeze(0).expand(latent.shape[0], -1)
        elif positions.ndim == 2 and positions.shape == latent.shape[:2]:
            resolved_positions = positions
        else:
            raise ValueError("positions must have shape [L] or [B,L]")
        safe = torch.where(mask.unsqueeze(-1), latent, torch.zeros_like(latent))
        projected = torch.relu(self.input_norm(self.input_projection(safe)))
        time_encoding = self.time_encoder(resolved_positions, time_mask=mask)
        encoded, _ = self.attention_heads(projected + time_encoding, time_mask=mask)
        embedding = self.output_norm(self.dropout(self.projection(encoded)))
        sample_valid = mask.any(dim=-1)
        embedding = torch.where(
            sample_valid.unsqueeze(-1), embedding, torch.zeros_like(embedding)
        )
        return RawTemporalRepresentation(
            fused_repr=embedding,
            positions_used=resolved_positions,
        )
