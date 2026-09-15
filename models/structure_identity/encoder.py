"""Neural encoding of relative local structure evidence for 07B."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn
from torch.nn import functional as F


LOCAL_POINTS = 32


def amplitude_waveform(values: torch.Tensor) -> torch.Tensor:
    """Remove the first-point level while preserving window energy."""
    if values.ndim != 3:
        raise ValueError("waveform must be [B,32,D]")
    return values - values[:, :1]


def shape_waveform(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Endpoint-relative waveform with per-window Frobenius normalization."""
    relative = amplitude_waveform(values)
    scale = relative.square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(eps)
    return relative / scale


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=1),
            nn.Dropout(dropout),
        )
        self.norm = nn.GroupNorm(1, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values + self.net(values))


class MultiscaleWaveformEncoder(nn.Module):
    """Small independent Conv1d encoder for one 32-point waveform view."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, 1)
        self.multiscale = nn.ModuleList(
            nn.Conv1d(hidden_dim, hidden_dim, kernel, padding=kernel // 2)
            for kernel in (3, 5, 7)
        )
        self.projection = nn.Conv1d(hidden_dim * 3, hidden_dim, 1)
        self.blocks = nn.Sequential(
            ResidualTemporalBlock(hidden_dim, dropout),
            ResidualTemporalBlock(hidden_dim, dropout),
        )
        self.attention = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != LOCAL_POINTS:
            raise ValueError("waveform must contain exactly 32 relative-time points")
        hidden = self.input_projection(values.transpose(1, 2))
        hidden = self.projection(torch.cat([F.gelu(layer(hidden)) for layer in self.multiscale], dim=1))
        hidden = self.blocks(hidden)
        weights = torch.softmax(self.attention(hidden), dim=-1)
        return (hidden * weights).sum(dim=-1)


class SequenceTokenEncoder(nn.Module):
    """Variable-length typed sequence encoder with a safe EMPTY token."""
    def __init__(self, numeric_dim: int, num_types: int = 3, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, 8, padding_idx=0)
        self.projection = nn.Linear(numeric_dim + 8, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, 4, hidden_dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.sequence_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.empty_token = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.normal_(self.sequence_token, std=0.02)
        nn.init.normal_(self.empty_token, std=0.02)

    def forward(self, types: torch.Tensor, numeric: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if types.ndim != 2 or numeric.ndim != 3 or mask.shape != types.shape:
            raise ValueError("typed sequences require [B,N], [B,N,D], [B,N]")
        batch = types.shape[0]
        hidden = self.projection(torch.cat([self.type_embedding(types), numeric], dim=-1))
        token = self.sequence_token.expand(batch, -1, -1)
        sequence = torch.cat([token, hidden], dim=1)
        padding = torch.cat([
            torch.zeros(batch, 1, dtype=torch.bool, device=mask.device), ~mask.bool()
        ], dim=1)
        output = self.encoder(sequence, src_key_padding_mask=padding)[:, 0]
        empty = ~mask.any(dim=1)
        return torch.where(empty[:, None], self.empty_token.expand(batch, -1), output)


EventSequenceEncoder = SequenceTokenEncoder
FineSequenceEncoder = SequenceTokenEncoder


class ShapeFusionEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, output_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.shape_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            hidden_dim, 4, hidden_dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.output = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
        nn.init.normal_(self.shape_token, std=0.02)

    def forward(self, branch_tokens: Sequence[torch.Tensor]) -> torch.Tensor:
        if not branch_tokens:
            raise ValueError("at least one structure branch is required")
        batch = branch_tokens[0].shape[0]
        tokens = torch.stack(tuple(branch_tokens), dim=1)
        shape = self.shape_token.expand(batch, -1, -1)
        hidden = self.encoder(torch.cat([shape, tokens], dim=1))[:, 0]
        return F.normalize(self.output(hidden), dim=-1)


class StructureIdentityEncoder(nn.Module):
    """Encode only relative waveform/event/fine evidence; metadata is external."""
    VARIANTS = {
        "Waveform": ("shape", "amplitude"),
        "Event": ("event", "fine"),
        "Fusion": ("shape", "amplitude", "event", "fine"),
    }

    def __init__(self, waveform_dim: int, variant: str = "Fusion", dropout: float = 0.1):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown 07B variant: {variant}")
        self.variant = variant
        self.active_branches = self.VARIANTS[variant]
        self.shape_encoder = (MultiscaleWaveformEncoder(waveform_dim, dropout=dropout)
                              if "shape" in self.active_branches else None)
        self.amplitude_encoder = (MultiscaleWaveformEncoder(waveform_dim, dropout=dropout)
                                  if "amplitude" in self.active_branches else None)
        self.event_encoder = EventSequenceEncoder(4, dropout=dropout) if "event" in self.active_branches else None
        self.fine_encoder = FineSequenceEncoder(6, dropout=dropout) if "fine" in self.active_branches else None
        self.fusion = ShapeFusionEncoder(dropout=dropout)

    def forward(
        self,
        shape_waveform: torch.Tensor,
        amplitude_waveform: torch.Tensor,
        event_types: torch.Tensor,
        event_numeric: torch.Tensor,
        event_mask: torch.Tensor,
        fine_types: torch.Tensor,
        fine_numeric: torch.Tensor,
        fine_mask: torch.Tensor,
    ) -> torch.Tensor:
        if shape_waveform.shape[1] != LOCAL_POINTS or amplitude_waveform.shape[1] != LOCAL_POINTS:
            raise ValueError("waveform must contain exactly 32 relative-time points")
        tokens: Dict[str, torch.Tensor] = {}
        if "shape" in self.active_branches:
            tokens["shape"] = self.shape_encoder(shape_waveform)
        if "amplitude" in self.active_branches:
            tokens["amplitude"] = self.amplitude_encoder(amplitude_waveform)
        if "event" in self.active_branches:
            tokens["event"] = self.event_encoder(event_types, event_numeric, event_mask)
        if "fine" in self.active_branches:
            tokens["fine"] = self.fine_encoder(fine_types, fine_numeric, fine_mask)
        return self.fusion([tokens[name] for name in self.active_branches])
