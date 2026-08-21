import numpy as np
import torch
import torch.nn as nn

from .components import RawComponentPseLtaeClassifier, TemporalComponentBatch


class StlDecomposition(nn.Module):
    """Observation-order STL applied independently to valid pixel trajectories."""

    component_names = ("trend", "seasonal", "residual")

    def __init__(self, period: int, robust: bool = False):
        super().__init__()
        period = int(period)
        if period < 2:
            raise ValueError("STL period must be at least 2")
        self.period = period
        self.robust = bool(robust)

    @staticmethod
    def _stl_class():
        try:
            from statsmodels.tsa.seasonal import STL
        except ImportError as error:
            raise ImportError(
                "stl requires statsmodels; install it with 'pip install statsmodels'"
            ) from error
        return STL

    @torch.no_grad()
    def forward(self, pixels, mask, positions):
        if pixels.ndim != 4 or mask.ndim != 3:
            raise ValueError("expected pixels [B, T, C, S] and mask [B, T, S]")
        if pixels.shape[0] != mask.shape[0] or pixels.shape[1] != mask.shape[1] or pixels.shape[3] != mask.shape[2]:
            raise ValueError("mask shape is incompatible with pixels")

        STL = self._stl_class()
        source = pixels.detach().cpu().numpy()
        valid_mask = mask.detach().cpu().numpy() > 0
        trend = np.zeros_like(source)
        seasonal = np.zeros_like(source)
        residual = source.copy()

        batch, _, channels, num_pixels = source.shape
        for batch_index in range(batch):
            for pixel_index in range(num_pixels):
                valid = valid_mask[batch_index, :, pixel_index]
                valid_count = int(valid.sum())
                if valid_count == 0:
                    continue
                if valid_count < 2 * self.period:
                    raise ValueError(
                        "STL requires at least two periods of valid observations per trajectory"
                    )
                for channel_index in range(channels):
                    trajectory = source[batch_index, valid, channel_index, pixel_index]
                    result = STL(
                        trajectory, period=self.period, robust=self.robust
                    ).fit()
                    trend[batch_index, valid, channel_index, pixel_index] = result.trend
                    seasonal[batch_index, valid, channel_index, pixel_index] = result.seasonal
                    residual[batch_index, valid, channel_index, pixel_index] = result.resid

        def as_input_tensor(values):
            return torch.as_tensor(values, device=pixels.device, dtype=pixels.dtype)

        return [
            TemporalComponentBatch("trend", as_input_tensor(trend), mask, positions),
            TemporalComponentBatch("seasonal", as_input_tensor(seasonal), mask, positions),
            TemporalComponentBatch("residual", as_input_tensor(residual), mask, positions),
        ]


class StlDecompositionClassifier(RawComponentPseLtaeClassifier):
    """Three STL branches fused by concatenating their LTAE embeddings."""

    def __init__(self, period: int = 7, **kwargs):
        super().__init__(
            decomposer=StlDecomposition(period=period),
            component_names=StlDecomposition.component_names,
            fusion_mode="concat_embeddings",
            **kwargs,
        )
