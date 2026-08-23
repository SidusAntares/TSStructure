"""Minimal mTAN encoder used as the ReIMTS temporal backbone."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiTimeAttention(nn.Module):
    """Multi-time attention adapted from Ladbaby/PyOmniTS."""

    def __init__(self, input_dim, hidden_dim, embed_time=128, num_heads=1):
        super().__init__()
        if embed_time % num_heads != 0:
            raise ValueError("embed_time must be divisible by num_heads")
        self.embed_time = embed_time
        self.head_dim = embed_time // num_heads
        self.num_heads = num_heads
        self.value_dim = input_dim
        self.query_projection = nn.Linear(embed_time, embed_time)
        self.key_projection = nn.Linear(embed_time, embed_time)
        self.output_projection = nn.Linear(input_dim * num_heads, hidden_dim)

    def forward(self, query, key, value, mask):
        batch_size, _, value_dim = value.shape
        query = self.query_projection(query).view(
            query.shape[0], -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key_projection(key).view(
            key.shape[0], -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        scores = scores.unsqueeze(-1).expand(-1, -1, -1, -1, value_dim)
        expanded_mask = mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~expanded_mask, -1e9)
        attention = F.softmax(scores, dim=-2)
        attended = torch.sum(
            attention * value.unsqueeze(1).unsqueeze(2), dim=-2
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * value_dim
        )
        return self.output_projection(attended)


class MTANEncoder(nn.Module):
    """Encode irregular observations into the ReIMTS ``E_time`` tensor."""

    def __init__(
        self,
        input_dim=128,
        latent_dim=128,
        num_ref_points=8,
        num_heads=1,
        embed_time=128,
        period=365,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_ref_points = num_ref_points
        self.period = period
        self.register_buffer(
            "query", torch.linspace(0.0, 1.0, num_ref_points), persistent=True
        )
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.attention = MultiTimeAttention(
            input_dim=2 * input_dim,
            hidden_dim=latent_dim,
            embed_time=embed_time,
            num_heads=num_heads,
        )
        self.gru = nn.GRU(
            latent_dim,
            latent_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.to_distribution = nn.Sequential(
            nn.Linear(2 * latent_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 2 * latent_dim),
        )

    def _time_embedding(self, positions):
        positions = positions.unsqueeze(-1)
        return torch.cat(
            [self.linear(positions), torch.sin(self.periodic(positions))], dim=-1
        )

    def forward(self, values: Tensor, encoder_positions: Tensor, valid: Tensor):
        if values.ndim != 3:
            raise ValueError("values must have shape [B,L,D]")
        if encoder_positions.shape != values.shape[:2]:
            raise ValueError("encoder_positions must have shape [B,L]")
        if valid.shape != encoder_positions.shape:
            raise ValueError("valid must match encoder_positions")

        valid_features = valid.unsqueeze(-1).expand_as(values)
        masked_values = values * valid_features.to(values.dtype)
        encoder_input = torch.cat(
            [masked_values, valid_features.to(values.dtype)], dim=-1
        )
        normalized_time = encoder_positions.to(values.dtype) / float(self.period)
        key = self._time_embedding(normalized_time)
        query = self._time_embedding(self.query.to(values.dtype).unsqueeze(0))
        attention_mask = torch.cat([valid_features, valid_features], dim=-1)
        hidden = self.attention(query, key, encoder_input, attention_mask)
        hidden, _ = self.gru(hidden)
        distribution = self.to_distribution(hidden)
        qz0_mean = distribution[..., : self.latent_dim]
        qz0_logvar = distribution[..., self.latent_dim :]

        # Adapted from Ladbaby/PyOmniTS ReIMTS+mTAN implementation.
        # z0 corresponds to ReIMTS temporal representation E_time. Training
        # uses reparameterization; evaluation uses its deterministic mean.
        if self.training:
            epsilon = torch.randn_like(qz0_mean)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        else:
            z0 = qz0_mean
        return z0


class ReIMTSMTANEncoders(nn.Module):
    """Independent scale-specific mTAN encoders shared within each scale."""

    def __init__(
        self,
        levels=3,
        input_dim=128,
        latent_dim=128,
        num_ref_points=8,
        num_heads=1,
        period=365,
    ):
        super().__init__()
        self.scale_encoders = nn.ModuleList(
            [
                MTANEncoder(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    num_ref_points=num_ref_points,
                    num_heads=num_heads,
                    period=period,
                )
                for _ in range(levels)
            ]
        )
        self.reference_points = tuple(
            encoder.num_ref_points for encoder in self.scale_encoders
        )
