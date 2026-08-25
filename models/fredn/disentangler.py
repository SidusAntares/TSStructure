"""Learnable complementary attribution of irregular Fourier coefficients."""

from typing import Dict

import torch
from torch import nn

from models.fredn.nufft import centered_modes


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
