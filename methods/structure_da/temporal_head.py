"""Raw T/S temporal encoding with a shared LTAE body and private norms.

The trend and structure branches share one ContinuousTime2Vec, one input
projection Linear and one LTAE attention/projection body. Only the input and
output LayerNorms are branch-private. The final representation is the plain
concatenation ``[r_T | r_S]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from models.ltae import TrendStructureSharedLTAE

from .representation import RawTemporalRepresentation


class SharedTrendStructureLTAE(nn.Module):
    """Encode trend and structure into one shared LTAE representation."""

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
        time_scale: float = 365.0,
        max_initial_frequency: float = 16.0,
    ) -> None:
        super().__init__()
        self.shared_ltae = TrendStructureSharedLTAE(
            in_channels=in_channels,
            n_head=n_head,
            d_k=d_k,
            n_neurons=n_neurons,
            dropout=dropout,
            d_model=d_model,
            time_reference=time_reference,
            time_scale=time_scale,
            max_initial_frequency=max_initial_frequency,
        )
        self.component_dim = self.shared_ltae.component_dim

    def forward(
        self,
        trend: Tensor,
        structure: Tensor,
        positions: Tensor,
        mask: Tensor,
    ) -> RawTemporalRepresentation:
        """Return raw T/S embeddings for normalized physical positions.

        Args:
            trend: Temporal trend tokens with shape ``[B, L, D]``.
            structure: Temporal structure tokens with shape ``[B, L, D]``.
            positions: Backbone-normalized shared physical positions ``[B, L]``.
            mask: Boolean validity mask with shape ``[B, L]`` (True = valid).
        """
        trend_repr, structure_repr = self.shared_ltae(
            trend,
            structure,
            positions,
            time_mask=mask,
        )
        if positions.ndim == 1:
            resolved_positions = positions.unsqueeze(0).expand(trend.shape[0], -1)
        else:
            resolved_positions = positions
        fused_repr = torch.cat([trend_repr, structure_repr], dim=-1)
        return RawTemporalRepresentation(
            trend_repr=trend_repr,
            structure_repr=structure_repr,
            fused_repr=fused_repr,
            positions_used=resolved_positions,
        )
