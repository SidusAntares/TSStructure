from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .components import RawComponentPseLtaeClassifier, TemporalComponentBatch


class VmdDecomposition(nn.Module):
    """VMD modes kept as full observation-order temporal sequences.

    VMD is applied independently to each valid (channel, parcel-pixel) trajectory.
    It does not interpolate, impute, resample, or use acquisition-time spacing inside
    the VMD solver. The original TimeMatch positions are retained for the downstream
    LTAE. Modes are sorted by their final center frequency so branch identities are
    stable from low to high frequency.
    """

    def __init__(
        self,
        num_modes: int = 5,
        alpha: float = 2000.0,
        tau: float = 0.0,
        dc: int = 0,
        init: int = 1,
        tol: float = 1e-7,
    ):
        super().__init__()
        self.num_modes = int(num_modes)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.dc = int(dc)
        self.init = int(init)
        self.tol = float(tol)
        if self.num_modes < 1:
            raise ValueError("VMD num_modes must be at least 1")
        if self.alpha <= 0:
            raise ValueError("VMD alpha must be positive")
        if self.tol <= 0:
            raise ValueError("VMD tol must be positive")
        if self.dc not in (0, 1):
            raise ValueError("VMD dc must be 0 or 1")
        if self.init not in (0, 1, 2):
            raise ValueError("VMD init must be one of 0, 1, 2")
        self.component_names = tuple(
            "mode_{}".format(index + 1) for index in range(self.num_modes)
        )

    @staticmethod
    def _vmd_function():
        try:
            from vmdpy import VMD
        except ImportError as error:
            raise ImportError(
                "vmd requires vmdpy; install it with 'pip install vmdpy'"
            ) from error
        return VMD

    @staticmethod
    def _validate_time_invariant_mask(mask: torch.Tensor) -> torch.Tensor:
        temporal_validity = mask > 0
        if not torch.equal(
            temporal_validity,
            temporal_validity[:, :1, :].expand_as(temporal_validity),
        ):
            raise ValueError(
                "VMD requires TimeMatch's time-invariant spatial pixel mask; "
                "temporal imputation is not allowed"
            )
        return temporal_validity[:, 0, :]

    def _decompose_trajectory(self, signal: np.ndarray, vmd_function) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim != 1:
            raise ValueError("VMD trajectory must be one-dimensional")
        if not np.all(np.isfinite(signal)):
            raise ValueError("VMD trajectory contains non-finite values")

        # VMD's normalization/division steps are undefined for an exactly constant
        # zero-bandwidth signal. Preserve that degenerate signal in the lowest mode;
        # this is only a numerical guard and leaves ordinary trajectories untouched.
        if np.ptp(signal) <= np.finfo(np.float64).eps:
            modes = np.zeros((self.num_modes, signal.size), dtype=np.float64)
            modes[0] = signal
            return modes

        modes, _, omega = vmd_function(
            signal,
            self.alpha,
            self.tau,
            self.num_modes,
            self.dc,
            self.init,
            self.tol,
        )
        modes = np.asarray(modes, dtype=np.float64)
        if modes.shape != (self.num_modes, signal.size):
            raise RuntimeError(
                "vmdpy returned unexpected mode shape {}; expected {}".format(
                    modes.shape, (self.num_modes, signal.size)
                )
            )
        omega = np.asarray(omega)
        final_omega = omega[-1] if omega.ndim > 1 else omega
        if final_omega.shape[0] != self.num_modes:
            raise RuntimeError("vmdpy returned an unexpected center-frequency shape")
        order = np.argsort(final_omega)
        return modes[order]

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
        output = np.zeros(
            (self.num_modes, batch, channels, num_pixels, length), dtype=np.float64
        )
        vmd_function = self._vmd_function()
        valid_np = valid_pixels.cpu().numpy()
        for batch_index in range(batch):
            for pixel_index in range(num_pixels):
                if not valid_np[batch_index, pixel_index]:
                    continue
                for channel_index in range(channels):
                    output[:, batch_index, channel_index, pixel_index, :] = (
                        self._decompose_trajectory(
                            source[batch_index, channel_index, pixel_index],
                            vmd_function,
                        )
                    )

        components = []
        for mode_index, name in enumerate(self.component_names):
            component_pixels = torch.as_tensor(
                output[mode_index], device=pixels.device, dtype=pixels.dtype
            ).permute(0, 3, 1, 2).contiguous()
            components.append(
                TemporalComponentBatch(name, component_pixels, mask, positions)
            )
        return components

    @staticmethod
    def reconstruct(components: Sequence[TemporalComponentBatch]) -> torch.Tensor:
        components = list(components)
        if not components:
            raise ValueError("VMD reconstruction requires at least one component")
        return torch.stack([component.pixels for component in components], dim=0).sum(dim=0)


class VmdDecompositionClassifier(RawComponentPseLtaeClassifier):
    """VMD-LSTM-style independent temporal branches with output-level addition."""

    def __init__(
        self,
        num_modes: int = 5,
        alpha: float = 2000.0,
        tau: float = 0.0,
        dc: int = 0,
        init: int = 1,
        tol: float = 1e-7,
        **kwargs
    ):
        decomposer = VmdDecomposition(
            num_modes=num_modes,
            alpha=alpha,
            tau=tau,
            dc=dc,
            init=init,
            tol=tol,
        )
        super().__init__(
            decomposer=decomposer,
            component_names=decomposer.component_names,
            fusion_mode="additive_logits",
            **kwargs
        )
