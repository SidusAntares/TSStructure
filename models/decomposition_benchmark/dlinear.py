from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import RawComponentPseLtaeClassifier, TemporalComponentBatch


class DLinearSeriesDecomposition(nn.Module):
    """
    DLinear/Autoformer-style moving-average series decomposition.

    The operator is applied along observation order only. It does not interpolate,
    impute or resample TimeMatch acquisition times. Trend and seasonal components
    retain the original positions and pixel-set structure.
    """

    component_names = ("trend", "seasonal")

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("DLinear moving-average kernel_size must be a positive odd integer")
        self.kernel_size = kernel_size

    def moving_average(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4:
            raise ValueError("pixels must have shape [B, T, C, S]")
        batch, length, channels, num_pixels = pixels.shape
        series = pixels.permute(0, 2, 3, 1).reshape(batch * channels * num_pixels, 1, length)
        pad = (self.kernel_size - 1) // 2
        series = F.pad(series, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(series, kernel_size=self.kernel_size, stride=1)
        return trend.reshape(batch, channels, num_pixels, length).permute(0, 3, 1, 2).contiguous()

    def forward(self, pixels, mask, positions):
        trend = self.moving_average(pixels)
        seasonal = pixels - trend
        return [
            TemporalComponentBatch("trend", trend, mask, positions),
            TemporalComponentBatch("seasonal", seasonal, mask, positions),
        ]


class DLinearDecompositionClassifier(RawComponentPseLtaeClassifier):
    """Shared PSE + independent LTAEs + additive logits for DLinear decomposition."""

    def __init__(self, kernel_size: int = 25, **kwargs):
        super().__init__(
            decomposer=DLinearSeriesDecomposition(kernel_size=kernel_size),
            component_names=DLinearSeriesDecomposition.component_names,
            **kwargs,
        )
