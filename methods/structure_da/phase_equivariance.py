"""Target-internal temporal equivariance for phase-aware shape responses."""

import torch
from torch.nn import functional as F


def _phase_blocks(moments, shapelet_count, harmonics):
    shapelet_count = int(shapelet_count)
    harmonics = tuple(int(value) for value in harmonics)
    expected = 2 * shapelet_count * len(harmonics)
    if moments.ndim != 2 or moments.shape[1] != expected:
        raise ValueError(f"phase moments must be [B,{expected}]")
    return moments.reshape(moments.shape[0], len(harmonics), 2, shapelet_count)


def rotate_phase_moments(
    moments, delta, shapelet_count=16, harmonics=(1, 2), period_days=365.,
):
    """Rotate cos/sin occurrence moments using the real Fourier-shift sign."""
    blocks = _phase_blocks(moments, shapelet_count, harmonics)
    delta = torch.as_tensor(delta, device=moments.device, dtype=moments.dtype)
    if delta.ndim == 0:
        delta = delta.expand(moments.shape[0])
    elif delta.ndim == 2 and delta.shape == (moments.shape[0], 1):
        delta = delta[:, 0]
    if delta.ndim != 1 or delta.shape[0] != moments.shape[0]:
        raise ValueError("delta must be scalar, [B], or [B,1]")
    harmonic = torch.as_tensor(
        harmonics, device=moments.device, dtype=moments.dtype,
    )
    angle = 2. * torch.pi * delta[:, None] * harmonic[None] / float(period_days)
    cosine, sine = torch.cos(angle)[..., None], torch.sin(angle)[..., None]
    x, y = blocks[:, :, 0], blocks[:, :, 1]
    rotated = torch.stack((cosine * x - sine * y, sine * x + cosine * y), dim=2)
    return rotated.reshape_as(moments)


def phase_equivariance_loss(
    base, shifted, delta, shapelet_count=16, harmonics=(1, 2),
    period_days=365., eps=1e-8,
):
    """One-way occurrence invariance and magnitude-weighted phase equivariance."""
    base_strength = base["shapelet_strength"].detach()
    base_concentration = base["shapelet_concentration"].detach()
    base_moments = base["shapelet_phase_moments"].detach()
    occurrence = (
        F.smooth_l1_loss(shifted["shapelet_strength"], base_strength)
        + F.smooth_l1_loss(
            shifted["shapelet_concentration"], base_concentration,
        )
    )
    expected = rotate_phase_moments(
        base_moments, delta, shapelet_count, harmonics, period_days,
    )
    expected_blocks = _phase_blocks(expected, shapelet_count, harmonics)
    shifted_blocks = _phase_blocks(
        shifted["shapelet_phase_moments"], shapelet_count, harmonics,
    )
    magnitude = expected_blocks.square().sum(dim=2).sqrt().detach()
    squared_error = (shifted_blocks - expected_blocks).square().mean(dim=2)
    phase = (magnitude * squared_error).sum() / magnitude.sum().clamp_min(eps)
    return {
        "total_loss": occurrence + phase,
        "occurrence_loss": occurrence,
        "phase_loss": phase,
    }


def sample_structure_aug_shifts(batch_size, max_shift, device, generator=None):
    """Sample reproducible nonzero integer shifts from [-max,+max]."""
    batch_size, max_shift = int(batch_size), int(max_shift)
    if batch_size < 1 or max_shift < 1:
        raise ValueError("batch_size and max_shift must be positive")
    values = torch.randint(
        0, 2 * max_shift, (batch_size,), device=device, generator=generator,
    )
    return torch.where(
        values < max_shift, values - max_shift, values - max_shift + 1,
    )
