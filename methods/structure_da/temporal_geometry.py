"""Low-level phase geometry used by the V3 temporal path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def _finite_real(name: str, value: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


def _validate_interval_widths(interval_widths: Tensor) -> None:
    if not isinstance(interval_widths, Tensor) or interval_widths.ndim != 2:
        raise ValueError("interval_widths must have shape [B, J]")
    if interval_widths.shape[1] < 1:
        raise ValueError("interval_widths must contain at least one interval")
    if not interval_widths.is_floating_point():
        raise ValueError("interval_widths must use a floating-point dtype")
    if not torch.isfinite(interval_widths).all().item():
        raise ValueError("interval_widths must contain only finite values")
    if torch.any(interval_widths <= 0).item():
        raise ValueError("interval_widths must be strictly positive")
    width_sums = interval_widths.sum(dim=-1)
    if not torch.allclose(
        width_sums,
        torch.ones_like(width_sums),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("interval_widths must sum to one along intervals")


@dataclass(frozen=True)
class PhaseTangentOutput:
    interval_speed: Tensor
    warp_srvf: Tensor
    tangent: Tensor
    magnitude: Tensor


def warp_to_identity_tangent(
    interval_widths: Tensor,
    eps: float = 1e-8,
) -> PhaseTangentOutput:
    """Map monotone warp widths to the identity warp's SRVF tangent space."""

    eps = _finite_real("eps", eps)
    if eps <= 0:
        raise ValueError("eps must be greater than zero")
    _validate_interval_widths(interval_widths)

    interval_speed = interval_widths * interval_widths.shape[-1]
    warp_srvf = torch.sqrt(interval_speed.clamp_min(eps))
    identity = torch.ones_like(warp_srvf)
    cos_theta = (warp_srvf * identity).mean(dim=-1).clamp(
        min=-1.0 + eps,
        max=1.0,
    )
    identity_region = (1.0 - cos_theta) <= (0.5 * eps)
    safe_cos_theta = torch.where(
        identity_region,
        torch.zeros_like(cos_theta),
        cos_theta,
    )
    theta = torch.acos(safe_cos_theta)
    tangent_direction = (
        warp_srvf - safe_cos_theta.unsqueeze(-1) * identity
    )
    scale = theta / torch.sin(theta).clamp_min(eps)
    tangent = scale.unsqueeze(-1) * tangent_direction
    tangent = torch.where(
        (identity_region | (theta <= math.sqrt(eps))).unsqueeze(-1),
        torch.zeros_like(tangent),
        tangent,
    )
    magnitude = torch.sqrt(tangent.square().mean(dim=-1))
    return PhaseTangentOutput(
        interval_speed=interval_speed,
        warp_srvf=warp_srvf,
        tangent=tangent,
        magnitude=magnitude,
    )
