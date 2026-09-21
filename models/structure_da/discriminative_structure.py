"""Differentiable discriminative structure representation for irregular SITS."""

import torch
from torch import nn
from torch.nn import functional as F

from models.fourier_reconstruction import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
)


class FourierStructureExposer(nn.Module):
    """Expose a smooth structure view on a fixed periodic grid."""

    def __init__(self, num_modes=13, grid_points=64, period_days=365.0, reg=1e-3):
        super().__init__()
        self.analyzer = BatchedDirectFourierAnalyzer(num_modes, period_days, reg)
        self.synthesizer = BatchedDirectFourierSynthesizer(num_modes, period_days)
        self.grid_points = int(grid_points)
        self.period_days = float(period_days)
        self.register_buffer(
            "canonical_grid",
            torch.arange(self.grid_points, dtype=torch.float32)
            * (self.period_days / self.grid_points),
        )

    def forward(self, features, positions):
        coefficients, _ = self.analyzer(features, positions)
        grid = self.canonical_grid.to(device=features.device, dtype=features.dtype)
        grid = grid.unsqueeze(0).expand(features.shape[0], -1)
        return self.synthesizer(coefficients, grid), grid


class MultiScaleWindowExtractor(nn.Module):
    """Extract fixed, dense, non-extrema local windows in deterministic order."""

    def __init__(self, scales=(16, 32), stride=8):
        super().__init__()
        scales = tuple(int(value) for value in scales)
        if not scales or min(scales) < 2 or stride < 1:
            raise ValueError("window scales must be >=2 and stride must be positive")
        self.scales = scales
        self.stride = int(stride)

    def forward(self, curve):
        if curve.ndim != 3:
            raise ValueError("curve must be [B,G,D]")
        _, points, _ = curve.shape
        windows, scale_ids = [], []
        for scale in self.scales:
            if scale > points:
                raise ValueError("window scales cannot exceed the exposed curve length")
            starts = range(0, points, self.stride)
            indices = torch.stack([
                (torch.arange(scale, device=curve.device) + start) % points
                for start in starts
            ])
            windows.append(curve[:, indices])
            scale_ids.extend([scale] * indices.shape[0])
        if not windows:
            raise ValueError("no window scale fits the exposed curve")
        return windows, torch.tensor(scale_ids, device=curve.device)


class _ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden, dilation, dropout=.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, values):
        return self.norm((values + self.network(values)).transpose(1, 2)).transpose(1, 2)


class _VariableLengthTemporalEncoder(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.input_projection = nn.Conv1d(channels, hidden, 1)
        self.multiscale = nn.ModuleList(
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
            for kernel in (3, 5, 7)
        )
        self.multiscale_projection = nn.Conv1d(3 * hidden, hidden, 1)
        self.blocks = nn.ModuleList(
            _ResidualTemporalBlock(hidden, dilation) for dilation in (1, 2, 4)
        )
        self.temporal_score = nn.Conv1d(hidden, 1, 1)

    def forward(self, values, mask=None):
        batch, tokens, points, channels = values.shape
        flat = values.reshape(batch * tokens, points, channels).transpose(1, 2)
        encoded = self.input_projection(flat)
        encoded = self.multiscale_projection(torch.cat([
            F.gelu(layer(encoded)) for layer in self.multiscale
        ], dim=1))
        for block in self.blocks:
            encoded = block(encoded)
        scores = self.temporal_score(encoded).squeeze(1)
        if mask is not None:
            flat_mask = mask.reshape(batch * tokens, points)
            scores = scores.masked_fill(~flat_mask, -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        pooled = (encoded * attention.unsqueeze(1)).sum(-1)
        return pooled.reshape(batch, tokens, -1)


class ShapeTokenGenerator(nn.Module):
    """Encode normalized morphology/dynamics and absolute level/variation."""

    def __init__(self, channels, shape_dim=128, branch_dim=32, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.raw_encoder = _VariableLengthTemporalEncoder(channels, branch_dim)
        self.diff_encoder = _VariableLengthTemporalEncoder(channels, branch_dim)
        self.mean_encoder = nn.Sequential(nn.Linear(channels, branch_dim), nn.GELU(), nn.Linear(branch_dim, branch_dim))
        self.std_encoder = nn.Sequential(nn.Linear(channels, branch_dim), nn.GELU(), nn.Linear(branch_dim, branch_dim))
        self.fusion = nn.Sequential(
            nn.Linear(4 * branch_dim, shape_dim),
            nn.LayerNorm(shape_dim),
            nn.GELU(),
        )

    def components(self, windows, mask=None):
        if mask is None:
            mask = torch.ones(
                windows.shape[:3], dtype=torch.bool, device=windows.device,
            )
        weights = mask.to(windows.dtype).unsqueeze(-1)
        count = weights.sum(2).clamp_min(1)
        mean = (windows * weights).sum(2) / count
        centered = windows - mean.unsqueeze(2)
        variance = (centered.square() * weights).sum(2) / count
        std = variance.sqrt()
        normalized = centered / (std.unsqueeze(2) + self.eps)
        normalized = normalized * weights
        difference = torch.zeros_like(normalized)
        difference[:, :, 1:] = normalized[:, :, 1:] - normalized[:, :, :-1]
        difference = difference * weights
        return {"normalized": normalized, "difference": difference, "mean": mean, "std": std}

    def forward(self, windows, mask=None):
        parts = self.components(windows, mask)
        raw = self.raw_encoder(parts["normalized"], mask)
        difference = self.diff_encoder(parts["difference"], mask)
        mean = self.mean_encoder(parts["mean"])
        std = self.std_encoder(parts["std"])
        return self.fusion(torch.cat((raw, difference, mean, std), dim=-1))


class ShapeAttentionPool(nn.Module):
    def __init__(self, shape_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(shape_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, tokens, valid_token_mask):
        scores = self.score(tokens).squeeze(-1)
        scores = scores.masked_fill(~valid_token_mask, -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.where(valid_token_mask, attention, torch.zeros_like(attention))
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).eps)
        return torch.sum(tokens * attention.unsqueeze(-1), dim=1), attention, valid_token_mask


class DiscriminativeStructureBranch(nn.Module):
    def __init__(
        self, channels, shape_dim=128, num_modes=13, grid_points=64,
        period_days=365.0, reg=1e-3, window_scales=(16, 32), window_stride=8,
    ):
        super().__init__()
        self.exposer = FourierStructureExposer(num_modes, grid_points, period_days, reg)
        self.window_extractor = MultiScaleWindowExtractor(window_scales, window_stride)
        self.token_generator = ShapeTokenGenerator(channels, shape_dim)
        self.attention_pool = ShapeAttentionPool(shape_dim)

    def forward(self, features, positions):
        exposed, grid = self.exposer(features, positions)
        window_groups, scales = self.window_extractor(exposed)
        tokens = torch.cat([
            self.token_generator(windows) for windows in window_groups
        ], dim=1)
        token_mask = torch.ones(
            tokens.shape[:2], dtype=torch.bool, device=tokens.device,
        )
        class_token, attention, token_mask = self.attention_pool(tokens, token_mask)
        return {
            "shape_tokens": tokens,
            "shape_attention": attention,
            "shape_mask": token_mask,
            "shape_class_token": class_token,
            "shape_scales": scales,
            "exposed_curve": exposed,
            "exposed_grid": grid,
        }
