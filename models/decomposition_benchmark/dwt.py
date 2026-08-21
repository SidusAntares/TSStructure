from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .components import RawComponentPseLtaeClassifier, TemporalComponentBatch


class DwtDecomposition(nn.Module):
    """PyWavelets multilevel DWT with coefficient-support acquisition positions."""

    # Zero extension makes every coefficient's real-observation support explicit
    # and monotonic without inventing acquisition times outside the input series.
    boundary_mode = "zero"

    def __init__(self, wavelet: str = "haar", level: int = 2):
        super().__init__()
        self.wavelet = str(wavelet)
        self.level = int(level)
        if self.level < 1:
            raise ValueError("DWT level must be at least 1")
        self.component_names = tuple(
            ["approximation_{}".format(self.level), "detail_{}".format(self.level)]
            + ["detail_{}".format(index) for index in range(self.level - 1, 0, -1)]
        )
        self._support_cache = {}
        self._last_input_length = None

    @staticmethod
    def _pywt():
        try:
            import pywt
        except ImportError as error:
            raise ImportError(
                "dwt requires PyWavelets; install it with 'pip install PyWavelets'"
            ) from error
        return pywt

    def _supports(self, length: int):
        cache_key = (length, self.wavelet, self.level, self.boundary_mode)
        if cache_key in self._support_cache:
            return self._support_cache[cache_key]
        pywt = self._pywt()
        basis = np.eye(length, dtype=np.float64)
        transformed = pywt.wavedec(
            basis,
            wavelet=self.wavelet,
            mode=self.boundary_mode,
            level=self.level,
            axis=-1,
        )
        supports = []
        for coefficients in transformed:
            component_supports = []
            for coefficient_index in range(coefficients.shape[-1]):
                indices = np.flatnonzero(
                    np.abs(coefficients[:, coefficient_index]) > 1e-12
                )
                if indices.size == 0:
                    raise RuntimeError("DWT coefficient has no source observation support")
                component_supports.append(indices)
            supports.append(component_supports)
        self._support_cache[cache_key] = supports
        return supports

    @torch.no_grad()
    def forward(self, pixels, mask, positions):
        if pixels.ndim != 4 or mask.ndim != 3 or positions.ndim != 2:
            raise ValueError(
                "expected pixels [B, T, C, S], mask [B, T, S], positions [B, T]"
            )
        batch, length, channels, num_pixels = pixels.shape
        if mask.shape != (batch, length, num_pixels) or positions.shape != (batch, length):
            raise ValueError("mask or positions shape is incompatible with pixels")

        temporal_validity = mask > 0
        if not torch.equal(
            temporal_validity,
            temporal_validity[:, :1, :].expand_as(temporal_validity),
        ):
            raise ValueError(
                "DWT requires TimeMatch's time-invariant spatial pixel mask; temporal imputation is not allowed"
            )

        pywt = self._pywt()
        max_level = pywt.dwt_max_level(length, pywt.Wavelet(self.wavelet).dec_len)
        if self.level > max_level:
            raise ValueError(
                "DWT level {} exceeds maximum {} for length {} and wavelet '{}'".format(
                    self.level, max_level, length, self.wavelet
                )
            )

        valid_pixels = temporal_validity[:, 0, :]
        series = pixels.detach().permute(0, 2, 3, 1).reshape(
            batch * channels * num_pixels, length
        ).cpu().numpy()
        valid_trajectories = valid_pixels.unsqueeze(1).expand(
            batch, channels, num_pixels
        ).reshape(-1).cpu().numpy()
        series[~valid_trajectories] = 0
        coefficient_arrays = pywt.wavedec(
            series,
            wavelet=self.wavelet,
            mode=self.boundary_mode,
            level=self.level,
            axis=-1,
        )

        supports = self._supports(length)
        components = []
        for name, coefficients, component_supports in zip(
            self.component_names, coefficient_arrays, supports
        ):
            coefficient_length = coefficients.shape[-1]
            component_pixels = torch.as_tensor(
                coefficients.reshape(batch, channels, num_pixels, coefficient_length),
                device=pixels.device,
                dtype=pixels.dtype,
            ).permute(0, 3, 1, 2).contiguous()
            component_mask = valid_pixels.unsqueeze(1).expand(
                batch, coefficient_length, num_pixels
            ).to(dtype=mask.dtype)
            support_positions = []
            for indices in component_supports:
                index_tensor = torch.as_tensor(indices, device=positions.device)
                summed = positions.index_select(1, index_tensor).sum(dim=1)
                support_positions.append(
                    torch.div(summed, len(indices), rounding_mode="floor")
                )
            component_positions = torch.stack(support_positions, dim=1)
            if torch.any(component_positions[:, 1:] < component_positions[:, :-1]):
                raise RuntimeError(
                    "DWT support centers are not monotonic for wavelet '{}' and mode '{}'".format(
                        self.wavelet, self.boundary_mode
                    )
                )
            components.append(
                TemporalComponentBatch(
                    name, component_pixels, component_mask, component_positions
                )
            )
        self._last_input_length = length
        return components

    @torch.no_grad()
    def reconstruct(self, components: Sequence[TemporalComponentBatch]):
        components = list(components)
        if tuple(component.name for component in components) != self.component_names:
            raise ValueError("DWT components are missing or out of order")
        if self._last_input_length is None:
            raise RuntimeError("reconstruct must follow a decomposition call")
        pywt = self._pywt()
        first = components[0].pixels
        batch, _, channels, num_pixels = first.shape
        coefficient_arrays = [
            component.pixels.detach().permute(0, 2, 3, 1).reshape(
                batch * channels * num_pixels, component.pixels.shape[1]
            ).cpu().numpy()
            for component in components
        ]
        reconstructed = pywt.waverec(
            coefficient_arrays,
            wavelet=self.wavelet,
            mode=self.boundary_mode,
            axis=-1,
        )[..., : self._last_input_length]
        return torch.as_tensor(
            reconstructed.reshape(
                batch, channels, num_pixels, self._last_input_length
            ),
            device=first.device,
            dtype=first.dtype,
        ).permute(0, 3, 1, 2).contiguous()


class DwtDecompositionClassifier(RawComponentPseLtaeClassifier):
    """Independent DWT-scale LTAEs fused by concatenating their embeddings."""

    def __init__(self, wavelet: str = "haar", level: int = 2, **kwargs):
        decomposer = DwtDecomposition(wavelet=wavelet, level=level)
        super().__init__(
            decomposer=decomposer,
            component_names=decomposer.component_names,
            fusion_mode="concat_embeddings",
            **kwargs,
        )
