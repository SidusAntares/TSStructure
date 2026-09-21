"""Differentiable discriminative structure representation for irregular SITS."""

import math

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
        batch, points, channels = curve.shape
        max_scale = max(self.scales)
        windows, masks, scale_ids = [], [], []
        for scale in self.scales:
            if scale > points:
                continue
            for start in range(0, points - scale + 1, self.stride):
                window = curve[:, start:start + scale]
                padded = curve.new_zeros(batch, max_scale, channels)
                padded[:, :scale] = window
                valid = torch.zeros(batch, max_scale, dtype=torch.bool, device=curve.device)
                valid[:, :scale] = True
                windows.append(padded)
                masks.append(valid)
                scale_ids.append(scale)
        if not windows:
            raise ValueError("no window scale fits the exposed curve")
        return (
            torch.stack(windows, dim=1),
            torch.stack(masks, dim=1),
            torch.tensor(scale_ids, device=curve.device),
        )


class _TemporalBranch(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, values, mask):
        batch, tokens, points, channels = values.shape
        flat = values.reshape(batch * tokens, points, channels).transpose(1, 2)
        encoded = self.network(flat).transpose(1, 2)
        flat_mask = mask.reshape(batch * tokens, points).unsqueeze(-1)
        pooled = (encoded * flat_mask).sum(1) / flat_mask.sum(1).clamp_min(1)
        return pooled.reshape(batch, tokens, -1)


class ShapeTokenGenerator(nn.Module):
    """Encode normalized morphology/dynamics and absolute level/variation."""

    def __init__(self, channels, shape_dim=128, branch_dim=32, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.raw_encoder = _TemporalBranch(channels, branch_dim)
        self.diff_encoder = _TemporalBranch(channels, branch_dim)
        self.mean_encoder = nn.Sequential(nn.Linear(channels, branch_dim), nn.GELU(), nn.Linear(branch_dim, branch_dim))
        self.std_encoder = nn.Sequential(nn.Linear(channels, branch_dim), nn.GELU(), nn.Linear(branch_dim, branch_dim))
        self.fusion = nn.Sequential(
            nn.Linear(4 * branch_dim, shape_dim),
            nn.LayerNorm(shape_dim),
            nn.GELU(),
        )

    def components(self, windows, mask):
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

    def forward(self, windows, mask):
        parts = self.components(windows, mask)
        raw = self.raw_encoder(parts["normalized"], mask)
        difference = self.diff_encoder(parts["difference"], mask)
        mean = self.mean_encoder(parts["mean"])
        std = self.std_encoder(parts["std"])
        return self.fusion(torch.cat((raw, difference, mean, std), dim=-1))


class ShapeAttentionPool(nn.Module):
    def __init__(self, shape_dim):
        super().__init__()
        self.score = nn.Linear(shape_dim, 1)

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
        windows, window_mask, scales = self.window_extractor(exposed)
        tokens = self.token_generator(windows, window_mask)
        token_mask = window_mask.any(-1)
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
