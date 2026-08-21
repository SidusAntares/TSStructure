import torch
import torch.nn as nn

from .components import RawComponentPseLtaeClassifier, TemporalComponentBatch


class XPatchEmaDecomposition(nn.Module):
    """xPatch exponential seasonal-trend decomposition frontend only."""

    component_names = ("trend", "seasonal")

    def __init__(self, alpha: float = 0.3):
        super().__init__()
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("xPatch EMA alpha must be in [0, 1]")
        self.alpha = alpha

    def forward(self, pixels, mask, positions):
        if pixels.ndim != 4:
            raise ValueError("pixels must have shape [B, T, C, S]")
        if pixels.shape[1] == 0:
            raise ValueError("xPatch EMA requires at least one observation")
        state = pixels[:, 0]
        trend_steps = [state]
        for time_index in range(1, pixels.shape[1]):
            state = self.alpha * pixels[:, time_index] + (1.0 - self.alpha) * state
            trend_steps.append(state)
        trend = torch.stack(trend_steps, dim=1)
        seasonal = pixels - trend
        return [
            TemporalComponentBatch("trend", trend, mask, positions),
            TemporalComponentBatch("seasonal", seasonal, mask, positions),
        ]


class XPatchEmaDecompositionClassifier(RawComponentPseLtaeClassifier):
    """Shared PSE and independent LTAEs with xPatch embedding concatenation."""

    def __init__(self, alpha: float = 0.3, **kwargs):
        super().__init__(
            decomposer=XPatchEmaDecomposition(alpha=alpha),
            component_names=XPatchEmaDecomposition.component_names,
            fusion_mode="concat_embeddings",
            **kwargs,
        )
