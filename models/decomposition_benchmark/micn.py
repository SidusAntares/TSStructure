from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import RawComponentPseLtaeClassifier, TemporalComponentBatch


class MicnMultiScaleHybridDecomposition(nn.Module):
    """MICN's official multi-scale hybrid decomposition frontend only."""

    component_names = ("trend", "seasonal")

    def __init__(self, conv_kernels: Sequence[int] = (17, 49)):
        super().__init__()
        kernels = tuple(int(kernel) for kernel in conv_kernels)
        if not kernels or any(kernel <= 0 for kernel in kernels):
            raise ValueError("MICN conv_kernels must contain positive integers")
        self.conv_kernels = kernels
        self.decomposition_kernels = tuple(
            kernel + 1 if kernel % 2 == 0 else kernel for kernel in kernels
        )

    @staticmethod
    def _moving_average(pixels: torch.Tensor, kernel_size: int) -> torch.Tensor:
        batch, length, channels, num_pixels = pixels.shape
        series = pixels.permute(0, 2, 3, 1).reshape(
            batch * channels * num_pixels, 1, length
        )
        padding = (kernel_size - 1) // 2
        series = F.pad(series, (padding, padding), mode="replicate")
        moving_mean = F.avg_pool1d(series, kernel_size=kernel_size, stride=1)
        return moving_mean.reshape(batch, channels, num_pixels, length).permute(
            0, 3, 1, 2
        ).contiguous()

    def forward(self, pixels, mask, positions):
        if pixels.ndim != 4:
            raise ValueError("pixels must have shape [B, T, C, S]")
        trends = [
            self._moving_average(pixels, kernel)
            for kernel in self.decomposition_kernels
        ]
        trend = torch.stack(trends, dim=0).mean(dim=0)
        seasonal = pixels - trend
        return [
            TemporalComponentBatch("trend", trend, mask, positions),
            TemporalComponentBatch("seasonal", seasonal, mask, positions),
        ]


class MicnDecompositionClassifier(RawComponentPseLtaeClassifier):
    """Shared PSE, independent branches, and MICN benchmark additive logits."""

    def __init__(self, conv_kernels: Sequence[int] = (17, 49), **kwargs):
        super().__init__(
            decomposer=MicnMultiScaleHybridDecomposition(conv_kernels=conv_kernels),
            component_names=MicnMultiScaleHybridDecomposition.component_names,
            fusion_mode="additive_logits",
            **kwargs,
        )
