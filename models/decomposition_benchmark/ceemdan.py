from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .components import RawComponentPseLtaeClassifier, TemporalComponentBatch


def sample_entropy(sequence: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """Finite-series SampEn using Chebyshev distance and r = r_ratio * std.

    The m=2, r=0.2*std convention is widely used in CEEMDAN+entropy
    reconstruction work. Self matches are excluded. If no (m+1)-template pair
    matches the tolerance, the mathematically correct finite estimate is +inf;
    the CEEMDAN grouping code treats such a component as high-complexity.
    """

    values = np.asarray(sequence, dtype=np.float64).reshape(-1)
    m = int(m)
    r_ratio = float(r_ratio)
    if m < 1:
        raise ValueError("sample entropy embedding dimension m must be positive")
    if r_ratio <= 0:
        raise ValueError("sample entropy r_ratio must be positive")
    if values.size <= m + 1:
        return float("inf")
    if not np.all(np.isfinite(values)):
        raise ValueError("sample entropy input contains non-finite values")

    std = float(np.std(values))
    if std <= np.finfo(np.float64).eps:
        return 0.0
    tolerance = r_ratio * std

    # Use the same N-m starting positions for m and m+1 templates, so the
    # pair counts have the standard conditional-probability interpretation and
    # every (m+1) match is necessarily also an m match.
    template_count = values.size - m
    if template_count < 2:
        return float("inf")

    def matching_pairs(width: int) -> int:
        templates = np.stack(
            [values[index : index + width] for index in range(template_count)],
            axis=0,
        )
        matches = 0
        for index in range(template_count - 1):
            distance = np.max(
                np.abs(templates[index + 1 :] - templates[index]), axis=1
            )
            matches += int(np.count_nonzero(distance <= tolerance))
        return matches

    matches_m = matching_pairs(m)
    if matches_m == 0:
        return float("inf")
    matches_m1 = matching_pairs(m + 1)
    if matches_m1 == 0:
        return float("inf")
    return float(-np.log(matches_m1 / matches_m))


class CeemdanSeDecomposition(nn.Module):
    """CEEMDAN followed by literature-style adaptive SampEn reconstruction.

    CEEMDAN is applied independently to each valid raw trajectory in observation
    order. Its variable number of IMFs is converted into two fixed temporal
    components using adaptive Sample-Entropy discrimination: IMFs whose SampEn is
    greater than ``threshold_factor * mean(SampEn)`` are summed into the
    high-frequency component, and the remaining IMFs plus the CEEMDAN residual are
    summed into the low-frequency component. This follows the adaptive CEEMDAN-SE
    high/low-frequency discrimination rule used in published decomposition-
    reconstruction forecasting work and avoids a fixed max-IMF/zero-mask adapter.

    CEEMDAN itself does not consume TimeMatch acquisition-day gaps; positions are
    retained unchanged for downstream LTAE processing. No interpolation, imputation,
    or regular-grid conversion is performed.
    """

    component_names = ("high_frequency", "low_frequency")

    def __init__(
        self,
        trials: int = 100,
        epsilon: float = 0.005,
        noise_seed: int = 1,
        sampen_m: int = 2,
        sampen_r_ratio: float = 0.2,
        threshold_factor: float = 0.5,
    ):
        super().__init__()
        self.trials = int(trials)
        self.epsilon = float(epsilon)
        self.noise_seed = int(noise_seed)
        self.sampen_m = int(sampen_m)
        self.sampen_r_ratio = float(sampen_r_ratio)
        self.threshold_factor = float(threshold_factor)
        if self.trials < 1:
            raise ValueError("CEEMDAN trials must be at least 1")
        if self.epsilon <= 0:
            raise ValueError("CEEMDAN epsilon must be positive")
        if self.sampen_m < 1:
            raise ValueError("CEEMDAN SampEn m must be positive")
        if self.sampen_r_ratio <= 0:
            raise ValueError("CEEMDAN SampEn r ratio must be positive")
        if self.threshold_factor <= 0:
            raise ValueError("CEEMDAN SampEn threshold_factor must be positive")

    @staticmethod
    def _ceemdan_class():
        try:
            from PyEMD import CEEMDAN
        except ImportError as error:
            raise ImportError(
                "ceemdan_se requires EMD-signal (PyEMD); install it with "
                "'pip install EMD-signal'"
            ) from error
        return CEEMDAN

    @staticmethod
    def _validate_time_invariant_mask(mask: torch.Tensor) -> torch.Tensor:
        temporal_validity = mask > 0
        if not torch.equal(
            temporal_validity,
            temporal_validity[:, :1, :].expand_as(temporal_validity),
        ):
            raise ValueError(
                "CEEMDAN requires TimeMatch's time-invariant spatial pixel mask; "
                "temporal imputation is not allowed"
            )
        return temporal_validity[:, 0, :]

    def _make_ceemdan(self):
        CEEMDAN = self._ceemdan_class()
        # PyEMD documents parallel=False + noise_seed as the reproducible path.
        ceemdan = CEEMDAN(
            trials=self.trials,
            epsilon=self.epsilon,
            parallel=False,
        )
        return ceemdan

    def _group_components(self, components: np.ndarray, original: np.ndarray):
        components = np.asarray(components, dtype=np.float64)
        original = np.asarray(original, dtype=np.float64)
        if components.ndim != 2 or components.shape[1] != original.size:
            raise RuntimeError("CEEMDAN returned components with an unexpected shape")
        if components.shape[0] == 0:
            raise RuntimeError("CEEMDAN returned no components")

        # PyEMD CEEMDAN appends the final residue as its last returned component.
        # Keep that residue in the low-frequency branch, matching CEEMDAN-SE
        # reconstruction practice. A one-component result is therefore entirely low.
        if components.shape[0] == 1:
            return np.zeros_like(original), components[0].copy()

        imfs = components[:-1]
        residue = components[-1]
        entropies = np.asarray(
            [
                sample_entropy(
                    imf,
                    m=self.sampen_m,
                    r_ratio=self.sampen_r_ratio,
                )
                for imf in imfs
            ],
            dtype=np.float64,
        )
        finite = np.isfinite(entropies)
        if np.any(finite):
            threshold = self.threshold_factor * float(entropies[finite].mean())
            high_mask = (~finite) | (entropies > threshold)
        else:
            # With very short/noisy sequences SampEn may be +inf for every IMF.
            # Such components are maximally unresolved/complex, so keep them in the
            # high-frequency group while the CEEMDAN residue anchors the low branch.
            high_mask = np.ones(entropies.shape, dtype=bool)

        high = imfs[high_mask].sum(axis=0) if np.any(high_mask) else np.zeros_like(original)
        low = residue.copy()
        if np.any(~high_mask):
            low = low + imfs[~high_mask].sum(axis=0)
        return high, low

    def _decompose_trajectory(self, signal: np.ndarray, ceemdan) -> Sequence[np.ndarray]:
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim != 1:
            raise ValueError("CEEMDAN trajectory must be one-dimensional")
        if not np.all(np.isfinite(signal)):
            raise ValueError("CEEMDAN trajectory contains non-finite values")
        if np.ptp(signal) <= np.finfo(np.float64).eps:
            return np.zeros_like(signal), signal.copy()

        # Reset the documented CEEMDAN RNG before each trajectory. This makes the
        # deterministic benchmark independent of DataLoader/batch traversal order.
        ceemdan.noise_seed(self.noise_seed)
        components = ceemdan.ceemdan(signal, max_imf=-1, progress=False)
        high, low = self._group_components(components, signal)
        return high, low

    @torch.no_grad()
    def forward(self, pixels, mask, positions):
        if pixels.ndim != 4 or mask.ndim != 3 or positions.ndim != 2:
            raise ValueError(
                "expected pixels [B, T, C, S], mask [B, T, S], positions [B, T]"
            )
        batch, length, channels, num_pixels = pixels.shape
        if mask.shape != (batch, length, num_pixels):
            raise ValueError("mask shape is incompatible with pixels")
        if positions.shape != (batch, length):
            raise ValueError("positions shape is incompatible with pixels")

        valid_pixels = self._validate_time_invariant_mask(mask)
        source = (
            pixels.detach()
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .double()
            .numpy()
        )
        high = np.zeros((batch, channels, num_pixels, length), dtype=np.float64)
        low = np.zeros_like(high)
        ceemdan = self._make_ceemdan()
        valid_np = valid_pixels.cpu().numpy()
        for batch_index in range(batch):
            for pixel_index in range(num_pixels):
                if not valid_np[batch_index, pixel_index]:
                    continue
                for channel_index in range(channels):
                    high_value, low_value = self._decompose_trajectory(
                        source[batch_index, channel_index, pixel_index], ceemdan
                    )
                    high[batch_index, channel_index, pixel_index] = high_value
                    low[batch_index, channel_index, pixel_index] = low_value

        def as_component(name: str, values: np.ndarray) -> TemporalComponentBatch:
            component_pixels = torch.as_tensor(
                values, device=pixels.device, dtype=pixels.dtype
            ).permute(0, 3, 1, 2).contiguous()
            return TemporalComponentBatch(name, component_pixels, mask, positions)

        return [
            as_component("high_frequency", high),
            as_component("low_frequency", low),
        ]

    @staticmethod
    def reconstruct(components: Sequence[TemporalComponentBatch]) -> torch.Tensor:
        components = list(components)
        if len(components) != 2:
            raise ValueError("CEEMDAN-SE reconstruction expects two components")
        return components[0].pixels + components[1].pixels


class CeemdanSeDecompositionClassifier(RawComponentPseLtaeClassifier):
    """CEEMDAN-SE reconstruction with independent LTAEs and additive outputs."""

    def __init__(
        self,
        trials: int = 100,
        epsilon: float = 0.005,
        noise_seed: int = 1,
        sampen_m: int = 2,
        sampen_r_ratio: float = 0.2,
        threshold_factor: float = 0.5,
        **kwargs
    ):
        decomposer = CeemdanSeDecomposition(
            trials=trials,
            epsilon=epsilon,
            noise_seed=noise_seed,
            sampen_m=sampen_m,
            sampen_r_ratio=sampen_r_ratio,
            threshold_factor=threshold_factor,
        )
        super().__init__(
            decomposer=decomposer,
            component_names=decomposer.component_names,
            fusion_mode="additive_logits",
            **kwargs
        )
