"""Learnable complementary attribution of irregular Fourier coefficients."""

from typing import Dict

import torch
from torch import nn

from models.fredn.nufft import centered_modes


class _ResidualFrequencyLearner(nn.Module):
    """Small frequency MLP with a learned residual projection."""

    def __init__(self, num_modes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp_projection = nn.Sequential(
            nn.Linear(num_modes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modes),
        )
        self.residual_projection = nn.Linear(num_modes, num_modes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.mlp_projection(values) + self.residual_projection(values)


class ReImSpectralEncoder(nn.Module):
    """Encode a complex spectrum with one learner shared by real and imaginary parts."""

    def __init__(
        self,
        num_modes: int,
        channels: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        centered_modes(num_modes)
        if channels <= 0 or output_dim <= 0:
            raise ValueError("channels and output_dim must be positive")
        self.num_modes = num_modes
        self.channels = channels
        self.shared_reim_mlp = _ResidualFrequencyLearner(
            num_modes=num_modes,
            hidden_dim=2 * num_modes,
            dropout=dropout,
        )
        self.spectral_readout = nn.Linear(2 * num_modes, 1)
        self.channel_norm = nn.LayerNorm(channels)
        self.output_projection = nn.Sequential(
            nn.Linear(channels, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.ndim != 3:
            raise ValueError("coeffs must be [B,F,D]")
        if coeffs.shape[1:] != (self.num_modes, self.channels):
            raise ValueError("coeffs shape does not match configured modes/channels")
        if not coeffs.is_complex():
            raise ValueError("coeffs must be complex-valued")

        channel_spectra = coeffs.transpose(1, 2)
        real_features = self.shared_reim_mlp(channel_spectra.real)
        imaginary_features = self.shared_reim_mlp(channel_spectra.imag)
        reim_features = torch.cat((real_features, imaginary_features), dim=-1)
        channel_features = self.spectral_readout(reim_features).squeeze(-1)
        return self.output_projection(self.channel_norm(channel_features))


class FrequencyDisentangler(nn.Module):
    """Split coefficients with a soft mask shared by paired frequencies."""

    def __init__(self, num_modes: int, channels: int):
        super().__init__()
        centered_modes(num_modes)
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.num_modes = num_modes
        self.channels = channels
        max_magnitude = num_modes // 2
        magnitudes = torch.arange(max_magnitude + 1, dtype=torch.float32)
        initialization = -torch.log1p(magnitudes).unsqueeze(1)
        self.nonnegative_logits = nn.Parameter(
            initialization.expand(-1, channels).clone()
        )
        mode_magnitudes = centered_modes(num_modes).abs().to(torch.long)
        self.register_buffer("mode_magnitudes", mode_magnitudes, persistent=False)

    def expanded_mask(self) -> torch.Tensor:
        nonnegative_mask = torch.sigmoid(self.nonnegative_logits)
        return nonnegative_mask.index_select(0, self.mode_magnitudes)

    def forward(self, coeffs: torch.Tensor):
        if coeffs.ndim != 3:
            raise ValueError("coeffs must be [B,F,D]")
        if coeffs.shape[1:] != (self.num_modes, self.channels):
            raise ValueError("coeffs shape does not match configured modes/channels")
        if not coeffs.is_complex():
            raise ValueError("coeffs must be complex-valued")
        mask = self.expanded_mask().to(coeffs.real.dtype)
        trend = coeffs * mask.unsqueeze(0)
        seasonal = coeffs * (1.0 - mask).unsqueeze(0)
        return trend, seasonal, mask

    @staticmethod
    def diagnostics(
        coeffs: torch.Tensor,
        trend_coeffs: torch.Tensor,
        seasonal_coeffs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, object]:
        total_energy = coeffs.abs().square().sum().clamp_min(1e-12)
        trend_ratio = trend_coeffs.abs().square().sum() / total_energy
        seasonal_ratio = seasonal_coeffs.abs().square().sum() / total_energy
        detached_mask = mask.detach()
        return {
            "mask_mean": detached_mask.mean(),
            "mask_std": detached_mask.std(unbiased=False),
            "mask_lt_0.1": (detached_mask < 0.1).float().mean(),
            "mask_gt_0.9": (detached_mask > 0.9).float().mean(),
            "trend_energy_ratio": trend_ratio.detach(),
            "seasonal_energy_ratio": seasonal_ratio.detach(),
            "frequency_mask_mean": detached_mask.mean(dim=1),
        }
