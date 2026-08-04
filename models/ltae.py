"""
Lightweight Temporal Attention Encoder module
We modify the original LTAE to support variable time series lengths and domain-specific batch normalization

Credits:
The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
the Transformer served as code base for the present script.

paper: https://arxiv.org/abs/1706.03762
code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import copy
import math
from collections.abc import Sequence

from models.layers import LinearLayer, get_positional_encoding


class LTAE(nn.Module):
    def __init__(self,
                 in_channels=128,
                 n_head=16,
                 d_k=8,
                 n_neurons=[256, 128],
                 dropout=0.2,
                 d_model=256,
                 T=1000,
                 max_temporal_shift=100,
                 max_position=365,
                 ):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)

        """

        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.n_neurons = copy.deepcopy(n_neurons)
        self.max_temporal_shift = max_temporal_shift

        if d_model is not None:
            self.d_model = d_model
            # self.inconv = nn.Conv1d(in_channels, d_model, 1)
            self.inconv = LinearLayer(in_channels, d_model)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.positional_enc = nn.Embedding.from_pretrained(get_positional_encoding(max_position + 2*max_temporal_shift, self.d_model, T=T), freeze=True)
        # not splitting positional encoding seems to adapt better
        # sin_tab = get_positional_encoding(max_position + 2*max_temporal_shift, self.d_model // n_head, T=T)
        # self.positional_enc = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1), freeze=True)

        # self.inlayernorm = nn.LayerNorm(self.in_channels)
        # self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)

        assert (self.n_neurons[0] == self.d_model)

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.append(LinearLayer(self.n_neurons[i], self.n_neurons[i + 1]))

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions, return_att=False, time_mask=None):
        resolved_mask = None
        if time_mask is not None:
            resolved_mask = _resolve_time_mask(
                time_mask,
                batch_size=x.shape[0],
                sequence_length=x.shape[1],
                device=x.device,
            )
            x = torch.where(
                resolved_mask.unsqueeze(-1), x, torch.zeros_like(x)
            )
        if self.inconv is not None:
            x = self.inconv(x)
        enc_output = x + self.positional_enc(positions + self.max_temporal_shift)

        enc_output, attn = self.attention_heads(
            enc_output, time_mask=resolved_mask
        )

        enc_output = self.dropout(self.mlp(enc_output))
        if resolved_mask is not None:
            sample_valid = resolved_mask.any(dim=-1)
            enc_output = torch.where(
                sample_valid.unsqueeze(-1),
                enc_output,
                torch.zeros_like(enc_output),
            )

        if return_att:
            return enc_output, attn
        else:
            return enc_output


class ContinuousTime2Vec(nn.Module):
    """Learn a continuous linear-plus-periodic encoding of normalized time."""

    def __init__(
        self,
        output_dim: int,
        *,
        time_reference: float = 0.0,
        time_scale: float = 365.0,
        max_initial_frequency: float = 16.0,
    ) -> None:
        super().__init__()
        if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim < 2:
            raise ValueError("output_dim must be an integer at least 2")
        for name, value in (
            ("time_reference", time_reference),
            ("time_scale", time_scale),
            ("max_initial_frequency", max_initial_frequency),
        ):
            try:
                converted = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be finite") from error
            if not math.isfinite(converted):
                raise ValueError(f"{name} must be finite")
            if name == "time_scale" and converted <= 0:
                raise ValueError("time_scale must be finite and greater than zero")
            if name == "max_initial_frequency" and converted < 1:
                raise ValueError(
                    "max_initial_frequency must be finite and at least 1"
                )
        self.output_dim = output_dim
        self.time_reference = float(time_reference)
        self.time_scale = float(time_scale)
        self.max_initial_frequency = float(max_initial_frequency)
        self.linear_weight = nn.Parameter(torch.tensor(1.0))
        self.linear_bias = nn.Parameter(torch.tensor(0.0))
        self.frequencies = nn.Parameter(
            torch.linspace(1.0, self.max_initial_frequency, output_dim - 1)
        )
        self.phase = nn.Parameter(torch.zeros(output_dim - 1))

    def _resolve_inputs(
        self, positions: torch.Tensor, time_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(positions, torch.Tensor):
            raise ValueError("positions must be a torch.Tensor")
        if not positions.is_floating_point() or positions.is_complex():
            raise ValueError("positions must be a real floating-point tensor")
        if positions.ndim == 1:
            sequence_length = positions.shape[0]
            if time_mask is not None and isinstance(time_mask, torch.Tensor) and time_mask.ndim == 2:
                batch_size = time_mask.shape[0]
            else:
                batch_size = 1
            resolved_positions = positions.unsqueeze(0).expand(batch_size, -1)
        elif positions.ndim == 2:
            batch_size, sequence_length = positions.shape
            resolved_positions = positions
        else:
            raise ValueError("positions must have shape [L] or [B, L]")
        resolved_mask = (
            torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=positions.device,
            )
            if time_mask is None
            else _resolve_time_mask(
                time_mask,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=positions.device,
            )
        )
        if resolved_positions.device != self.linear_weight.device:
            raise ValueError("positions device must match module parameters")
        if resolved_positions.dtype != self.linear_weight.dtype:
            raise ValueError("positions dtype must match module parameters")
        valid_positions = resolved_positions[resolved_mask]
        if not torch.isfinite(valid_positions).all().item():
            raise ValueError("valid positions must be finite")
        normalized = (
            valid_positions - self.time_reference
        ) / self.time_scale
        tolerance = 1e-6
        if normalized.numel() and (
            torch.any(normalized < -tolerance).item()
            or torch.any(normalized > 1.0 + tolerance).item()
        ):
            raise ValueError("valid normalized positions must lie in [0, 1]")
        return resolved_positions, resolved_mask

    def forward(
        self,
        positions: torch.Tensor,
        *,
        time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        resolved_positions, resolved_mask = self._resolve_inputs(
            positions, time_mask
        )
        safe_positions = torch.where(
            resolved_mask, resolved_positions, torch.zeros_like(resolved_positions)
        )
        normalized = (
            (safe_positions - self.time_reference) / self.time_scale
        ).clamp(0.0, 1.0)
        linear = self.linear_weight * normalized + self.linear_bias
        periodic = torch.sin(
            normalized.unsqueeze(-1) * self.frequencies + self.phase
        )
        encoding = torch.cat([linear.unsqueeze(-1), periodic], dim=-1)
        encoding = torch.where(
            resolved_mask.unsqueeze(-1), encoding, torch.zeros_like(encoding)
        )
        if not torch.isfinite(encoding).all().item():
            raise ValueError("time encoding must contain only finite values")
        return encoding


class TrendStructureSharedLTAE(nn.Module):
    """Encode T/S with a shared input projection and branch-specific norms."""

    component_names = ("trend", "structure")

    def __init__(
        self,
        in_channels: int = 128,
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
        integer_values = {
            "in_channels": in_channels,
            "n_head": n_head,
            "d_k": d_k,
            "d_model": d_model,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        try:
            neurons = tuple(n_neurons)
        except TypeError as error:
            raise ValueError("n_neurons must be a sequence") from error
        if (
            not neurons
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in neurons)
            or neurons[0] != d_model
        ):
            raise ValueError(
                "n_neurons must be nonempty, positive, and start with d_model"
            )
        try:
            dropout = float(dropout)
        except (TypeError, ValueError) as error:
            raise ValueError("dropout must lie in [0, 1)") from error
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.in_channels = in_channels
        self.d_model = d_model
        self.n_neurons = list(neurons)
        self.component_dim = neurons[-1]
        self.shared_input_projection = nn.Linear(
            in_channels, d_model, bias=False
        )
        self.trend_input_norm = nn.LayerNorm(d_model)
        self.structure_input_norm = nn.LayerNorm(d_model)
        self.shared_time_encoder = ContinuousTime2Vec(
            d_model,
            time_reference=time_reference,
            time_scale=time_scale,
            max_initial_frequency=max_initial_frequency,
        )
        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=d_model
        )
        projection_layers = []
        for input_dim, output_dim in zip(neurons[:-1], neurons[1:]):
            projection_layers.extend(
                [nn.Linear(input_dim, output_dim, bias=False), nn.ReLU()]
            )
        self.shared_projection = nn.Sequential(*projection_layers)
        self.dropout = nn.Dropout(dropout)
        self.trend_output_norm = nn.LayerNorm(self.component_dim)
        self.structure_output_norm = nn.LayerNorm(self.component_dim)

    @property
    def trend_input_projection(self) -> nn.Linear:
        """The trend branch uses the single shared input projection."""

        return self.shared_input_projection

    @property
    def structure_input_projection(self) -> nn.Linear:
        """The structure branch uses the single shared input projection."""

        return self.shared_input_projection

    def forward(
        self,
        trend: torch.Tensor,
        structure: torch.Tensor,
        positions: torch.Tensor,
        *,
        time_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(trend, torch.Tensor) or trend.ndim != 3:
            raise ValueError("trend and structure must have shape [B, L, in_channels]")
        if (
            not isinstance(structure, torch.Tensor)
            or structure.shape != trend.shape
            or structure.dtype != trend.dtype
            or structure.device != trend.device
        ):
            raise ValueError("trend and structure must have identical shape, dtype, and device")
        if trend.shape[-1] != self.in_channels or not trend.is_floating_point():
            raise ValueError("trend and structure must be floating point with in_channels features")
        reference = next(self.parameters())
        if trend.dtype != reference.dtype or trend.device != reference.device:
            raise ValueError("components must match module dtype and device")
        batch_size, sequence_length = trend.shape[:2]
        resolved_mask = (
            torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=trend.device,
            )
            if time_mask is None
            else _resolve_time_mask(
                time_mask,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=trend.device,
            )
        )
        time_encoding = self.shared_time_encoder(
            positions, time_mask=resolved_mask
        )
        if time_encoding.shape[:2] != (batch_size, sequence_length):
            raise ValueError("positions must have shape [L] or [B, L]")
        branch_tokens = []
        input_norms = (self.trend_input_norm, self.structure_input_norm)
        for component, input_norm in zip((trend, structure), input_norms):
            safe = torch.where(
                resolved_mask.unsqueeze(-1), component, torch.zeros_like(component)
            )
            if not torch.isfinite(safe).all().item():
                raise ValueError("valid component values must be finite")
            projected = self.shared_input_projection(safe)
            branch_tokens.append(torch.relu(input_norm(projected)) + time_encoding)

        shared_outputs = []
        for tokens in branch_tokens:
            encoded, _ = self.attention_heads(tokens, time_mask=resolved_mask)
            shared_outputs.append(
                self.dropout(self.shared_projection(encoded))
            )
        sample_valid = resolved_mask.any(dim=-1)
        outputs = []
        output_norms = (self.trend_output_norm, self.structure_output_norm)
        for shared_output, output_norm in zip(shared_outputs, output_norms):
            normalized = output_norm(shared_output)
            outputs.append(
                torch.where(
                    sample_valid.unsqueeze(-1),
                    normalized,
                    torch.zeros_like(normalized),
                )
            )
        return tuple(outputs)


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.key = nn.Linear(d_in, n_head * d_k)
        self.query = nn.Parameter(torch.zeros(n_head, d_k)).requires_grad_(True)
        nn.init.normal_(self.query, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.temperature = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, time_mask=None):
        # Slightly more efficient re-implementation of LTAE
        B, T, C = x.size()
        q = self.query.repeat(B, 1, 1, 1).transpose(1, 2)  # (nh, hs) -> (B, nh, 1, d_k)
        k = self.key(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)  # (B, nh, T, d_k)
        v = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # self-attend; (B, nh, 1, d_k) x (B, nh, d_k, T) -> (B, nh, 1, T)
        att = (q @ k.transpose(-2, -1)) / self.temperature
        resolved_mask = None
        sample_valid = None
        if time_mask is not None:
            resolved_mask = _resolve_time_mask(
                time_mask,
                batch_size=B,
                sequence_length=T,
                device=x.device,
            )
            sample_valid = resolved_mask.any(dim=-1)
            safe_mask = resolved_mask.clone()
            safe_mask[~sample_valid, 0] = True
            att = att.masked_fill(
                ~safe_mask[:, None, None, :],
                torch.finfo(att.dtype).min,
            )
        att = self.softmax(att)
        att = self.dropout(att)
        if resolved_mask is not None:
            att = att * resolved_mask[:, None, None, :].to(dtype=att.dtype)
            att = att * sample_valid[:, None, None, None].to(dtype=att.dtype)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, C)
        if sample_valid is not None:
            y = torch.where(
                sample_valid.unsqueeze(-1), y, torch.zeros_like(y)
            )
        return y, att


def _resolve_time_mask(time_mask, batch_size, sequence_length, device):
    if not isinstance(time_mask, torch.Tensor):
        raise ValueError("time_mask must be a torch.Tensor or None")
    if time_mask.ndim == 1:
        if time_mask.shape != (sequence_length,):
            raise ValueError("time_mask must have shape [L] or [B, L]")
        time_mask = time_mask.unsqueeze(0).expand(batch_size, -1)
    elif time_mask.ndim == 2:
        if time_mask.shape != (batch_size, sequence_length):
            raise ValueError("time_mask must have shape [L] or [B, L]")
    else:
        raise ValueError("time_mask must have shape [L] or [B, L]")
    if time_mask.is_complex() or (
        time_mask.dtype != torch.bool
        and (
            not torch.isfinite(time_mask).all().item()
            or not torch.all((time_mask == 0) | (time_mask == 1)).item()
        )
    ):
        raise ValueError("time_mask must contain only finite 0/1 values")
    return time_mask.to(device=device, dtype=torch.bool)
