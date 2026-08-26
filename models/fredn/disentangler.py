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
    def mask_diagnostics(mask: torch.Tensor) -> Dict[str, object]:
        """Summarize the expanded mask that is applied to Fourier coefficients."""
        if mask.ndim != 2:
            raise ValueError("mask must be [F,D]")
        if mask.is_complex():
            raise ValueError("mask must be real-valued")

        with torch.no_grad():
            detached_mask = mask.detach()
            quantile_levels = torch.tensor(
                [0.05, 0.25, 0.50, 0.75, 0.95],
                device=detached_mask.device,
                dtype=detached_mask.dtype,
            )
            frequency_quantiles = torch.quantile(
                detached_mask,
                quantile_levels,
                dim=1,
            )
            global_quantiles = torch.quantile(
                detached_mask.reshape(-1),
                quantile_levels,
            )
            feature_frequency_std = detached_mask.std(dim=0, unbiased=False)
            feature_frequency_range = (
                detached_mask.max(dim=0).values
                - detached_mask.min(dim=0).values
            )
            frequency_mean = detached_mask.mean(dim=1)
            magnitudes = centered_modes(
                detached_mask.shape[0],
                device=detached_mask.device,
                dtype=detached_mask.dtype,
            ).abs()
            centered_magnitudes = magnitudes - magnitudes.mean()
            centered_frequency_mean = frequency_mean - frequency_mean.mean()
            correlation_denominator = torch.sqrt(
                centered_magnitudes.square().sum()
                * centered_frequency_mean.square().sum()
            ).clamp_min(torch.finfo(detached_mask.dtype).eps)
            abs_freq_corr = (
                centered_magnitudes * centered_frequency_mean
            ).sum() / correlation_denominator

            feature_std_quantiles = torch.quantile(
                feature_frequency_std,
                torch.tensor(
                    [0.50, 0.90],
                    device=detached_mask.device,
                    dtype=detached_mask.dtype,
                ),
            )
            feature_range_quantiles = torch.quantile(
                feature_frequency_range,
                torch.tensor(
                    [0.50, 0.90],
                    device=detached_mask.device,
                    dtype=detached_mask.dtype,
                ),
            )
            near_half = (detached_mask - 0.5).abs() < 0.05
            low025 = detached_mask < 0.25
            high075 = detached_mask > 0.75

            return {
                "mask_mean": detached_mask.mean(),
                "mask_std": detached_mask.std(unbiased=False),
                "mask_min": detached_mask.min(),
                "mask_max": detached_mask.max(),
                "mask_p05": global_quantiles[0],
                "mask_p25": global_quantiles[1],
                "mask_p50": global_quantiles[2],
                "mask_p75": global_quantiles[3],
                "mask_p95": global_quantiles[4],
                "mask_near_half": near_half.float().mean(),
                "mask_low025": low025.float().mean(),
                "mask_high075": high075.float().mean(),
                "mask_lt_0.1": (detached_mask < 0.1).float().mean(),
                "mask_gt_0.9": (detached_mask > 0.9).float().mean(),
                "frequency_mask_mean": frequency_mean,
                "frequency_mask_std": detached_mask.std(dim=1, unbiased=False),
                "frequency_mask_p05": frequency_quantiles[0],
                "frequency_mask_p25": frequency_quantiles[1],
                "frequency_mask_p50": frequency_quantiles[2],
                "frequency_mask_p75": frequency_quantiles[3],
                "frequency_mask_p95": frequency_quantiles[4],
                "frequency_mask_near_half": near_half.float().mean(dim=1),
                "frequency_mask_low025": low025.float().mean(dim=1),
                "frequency_mask_high075": high075.float().mean(dim=1),
                "feature_freq_std_mean": feature_frequency_std.mean(),
                "feature_freq_std_median": feature_std_quantiles[0],
                "feature_freq_std_p90": feature_std_quantiles[1],
                "feature_freq_std_max": feature_frequency_std.max(),
                "feature_freq_range_mean": feature_frequency_range.mean(),
                "feature_freq_range_median": feature_range_quantiles[0],
                "feature_freq_range_p90": feature_range_quantiles[1],
                "feature_freq_range_max": feature_frequency_range.max(),
                "abs_freq_corr": abs_freq_corr,
            }

    @staticmethod
    def diagnostics(
        coeffs: torch.Tensor,
        trend_coeffs: torch.Tensor,
        seasonal_coeffs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, object]:
        with torch.no_grad():
            detached_coeffs = coeffs.detach()
            detached_trend = trend_coeffs.detach()
            detached_seasonal = seasonal_coeffs.detach()
            total_energy = detached_coeffs.abs().square().sum().clamp_min(1e-12)
            trend_total = detached_trend.abs().square().sum()
            seasonal_total = detached_seasonal.abs().square().sum()
            branch_total = (trend_total + seasonal_total).clamp_min(1e-12)

            input_by_frequency = detached_coeffs.abs().square().mean(dim=(0, 2))
            trend_by_frequency = detached_trend.abs().square().mean(dim=(0, 2))
            seasonal_by_frequency = detached_seasonal.abs().square().mean(
                dim=(0, 2)
            )

            diagnostics = FrequencyDisentangler.mask_diagnostics(mask)
            diagnostics.update(
                {
                    "trend_energy_ratio": trend_total / total_energy,
                    "seasonal_energy_ratio": seasonal_total / total_energy,
                    "branch_trend_energy_ratio": trend_total / branch_total,
                    "branch_seasonal_energy_ratio": seasonal_total / branch_total,
                    "input_energy_by_frequency": input_by_frequency
                    / input_by_frequency.sum().clamp_min(1e-12),
                    "trend_energy_by_frequency": trend_by_frequency
                    / trend_by_frequency.sum().clamp_min(1e-12),
                    "seasonal_energy_by_frequency": seasonal_by_frequency
                    / seasonal_by_frequency.sum().clamp_min(1e-12),
                }
            )
            return diagnostics
