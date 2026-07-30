"""Support-aware geometric objectives for registered temporal SRVFs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .temporal_registration import TemporalRegistrationOutput


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


@dataclass(frozen=True)
class TemporalGeometryLossOutput:
    total_loss: Tensor

    alignment_loss: Tensor
    roughness_loss: Tensor
    unsupported_loss: Tensor
    center_loss: Tensor

    per_sample_alignment_error: Tensor
    per_sample_warp_roughness: Tensor
    per_sample_unsupported_error: Tensor

    interval_support: Tensor
    phase_tangent: Tensor
    phase_magnitude: Tensor

    active_mask: Tensor
    active_count: Tensor


class TemporalGeometryObjective(nn.Module):
    """Compute source-only support-aware registration geometry objectives."""

    def __init__(
        self,
        canonical_grid_size: int,
        alignment_weight: float = 1.0,
        roughness_weight: float = 1.0,
        unsupported_weight: float = 1.0,
        center_weight: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if (
            not isinstance(canonical_grid_size, int)
            or isinstance(canonical_grid_size, bool)
            or canonical_grid_size < 2
        ):
            raise ValueError("canonical_grid_size must be an integer at least 2")
        weights = {}
        for name, value in (
            ("alignment_weight", alignment_weight),
            ("roughness_weight", roughness_weight),
            ("unsupported_weight", unsupported_weight),
            ("center_weight", center_weight),
        ):
            converted = _finite_real(name, value)
            if converted < 0:
                raise ValueError(f"{name} must be nonnegative")
            weights[name] = converted
        eps = _finite_real("eps", eps)
        if eps <= 0:
            raise ValueError("eps must be greater than zero")

        self.canonical_grid_size = canonical_grid_size
        self.alignment_weight = weights["alignment_weight"]
        self.roughness_weight = weights["roughness_weight"]
        self.unsupported_weight = weights["unsupported_weight"]
        self.center_weight = weights["center_weight"]
        self.eps = eps

        grid_weights = torch.full(
            (canonical_grid_size,), 1.0 / (canonical_grid_size - 1)
        )
        grid_weights[0] *= 0.5
        grid_weights[-1] *= 0.5
        interval_weights = torch.full(
            (canonical_grid_size - 1,), 1.0 / (canonical_grid_size - 1)
        )
        self.register_buffer("grid_integration_weights", grid_weights)
        self.register_buffer(
            "interval_integration_weights", interval_weights
        )

    def _validate_inputs(
        self,
        registration_output: TemporalRegistrationOutput,
        source_mask: Tensor,
    ) -> None:
        if not isinstance(registration_output, TemporalRegistrationOutput):
            raise ValueError(
                "registration_output must be a TemporalRegistrationOutput"
            )
        registered_srvf = registration_output.registered_srvf
        if (
            not isinstance(registered_srvf, Tensor)
            or registered_srvf.ndim != 3
            or registered_srvf.shape[1] != self.canonical_grid_size
        ):
            raise ValueError(
                "registered_srvf must have shape [B, canonical grid, D]"
            )
        if not registered_srvf.is_floating_point():
            raise ValueError("registered_srvf must use a floating-point dtype")
        if not torch.isfinite(registered_srvf).all().item():
            raise ValueError("registered_srvf must contain only finite values")
        batch_size, grid_size, feature_dim = registered_srvf.shape
        template_srvf = registration_output.template_srvf
        if (
            not isinstance(template_srvf, Tensor)
            or template_srvf.shape != (batch_size, grid_size, feature_dim)
        ):
            raise ValueError("template_srvf must have shape [B, K, D]")
        if not template_srvf.is_floating_point():
            raise ValueError("template_srvf must use a floating-point dtype")
        if not torch.isfinite(template_srvf).all().item():
            raise ValueError("template_srvf must contain only finite values")

        expected_support_shape = (batch_size, grid_size)
        for name, support in (
            ("registered_support", registration_output.registered_support),
            ("template_support", registration_output.template_support),
        ):
            if not isinstance(support, Tensor) or support.shape != expected_support_shape:
                raise ValueError(f"{name} must have shape [B, K]")
            if not support.is_floating_point():
                raise ValueError(f"{name} must use a floating-point dtype")
            if not torch.isfinite(support).all().item():
                raise ValueError(f"{name} must contain only finite values")
            if torch.any((support < 0) | (support > 1)).item():
                raise ValueError(f"{name} must lie in [0, 1]")

        interval_widths = registration_output.interval_widths
        _validate_interval_widths(interval_widths)
        if interval_widths.shape != (batch_size, grid_size - 1):
            raise ValueError("interval_widths must have shape [B, K-1]")
        registration_valid = registration_output.registration_valid
        if (
            not isinstance(registration_valid, Tensor)
            or registration_valid.dtype != torch.bool
            or registration_valid.shape != (batch_size,)
        ):
            raise ValueError(
                "registration_valid must be a boolean tensor with shape [B]"
            )
        if not isinstance(source_mask, Tensor) or source_mask.dtype != torch.bool:
            raise ValueError("source_mask must be a boolean tensor with shape [B]")
        if source_mask.shape != (batch_size,):
            raise ValueError("source_mask must be a boolean tensor with shape [B]")

        for name, tensor in (
            ("template_srvf", template_srvf),
            ("registered_support", registration_output.registered_support),
            ("template_support", registration_output.template_support),
            ("interval_widths", interval_widths),
        ):
            if tensor.device != registered_srvf.device or tensor.dtype != registered_srvf.dtype:
                raise ValueError(
                    f"{name} must match registered_srvf device and dtype"
                )
        for name, tensor in (
            ("registration_valid", registration_valid),
            ("source_mask", source_mask),
        ):
            if tensor.device != registered_srvf.device:
                raise ValueError(f"{name} must match registered_srvf device")

    def forward(
        self,
        registration_output: TemporalRegistrationOutput,
        source_mask: Tensor,
    ) -> TemporalGeometryLossOutput:
        self._validate_inputs(registration_output, source_mask)
        registered_srvf = registration_output.registered_srvf
        sample_support = registration_output.registered_support
        template_support = registration_output.template_support
        interval_widths = registration_output.interval_widths
        dtype = registered_srvf.dtype
        device = registered_srvf.device
        grid_weights = self.grid_integration_weights.to(
            device=device, dtype=dtype
        )
        interval_weights = self.interval_integration_weights.to(
            device=device, dtype=dtype
        )
        active_mask = registration_output.registration_valid & source_mask
        active = active_mask.to(dtype=dtype)
        zero = (registered_srvf.sum() + interval_widths.sum()) * 0.0
        zero_per_sample = registered_srvf.sum(dim=(1, 2)) * 0.0

        alignment_support = sample_support * template_support
        squared_error = (
            registered_srvf - registration_output.template_srvf
        ).square().sum(dim=-1)
        alignment_numerator = (
            grid_weights * alignment_support * squared_error
        ).sum(dim=-1)
        alignment_denominator = (
            grid_weights * alignment_support
        ).sum(dim=-1)
        alignment_diagnostic_valid = active_mask & (
            alignment_denominator > self.eps
        )
        per_sample_alignment_error = torch.where(
            alignment_diagnostic_valid,
            alignment_numerator / alignment_denominator.clamp_min(self.eps),
            zero_per_sample,
        )
        alignment_loss = (active * alignment_numerator).sum() / (
            (active * alignment_denominator).sum() + self.eps
        )

        interval_support = 0.5 * (
            sample_support[:, :-1] + sample_support[:, 1:]
        )
        phase = warp_to_identity_tangent(interval_widths, eps=self.eps)
        log_speed = torch.log(phase.interval_speed.clamp_min(self.eps))

        if interval_widths.shape[1] == 1:
            roughness_numerator = phase.interval_speed.sum(dim=-1) * 0.0
            roughness_denominator = phase.interval_speed.sum(dim=-1) * 0.0
            per_sample_warp_roughness = zero_per_sample
            roughness_loss = zero
        else:
            log_speed_difference = log_speed[:, 1:] - log_speed[:, :-1]
            roughness_support = 0.5 * (
                interval_support[:, 1:] + interval_support[:, :-1]
            )
            roughness_numerator = (
                roughness_support * log_speed_difference.square()
            ).sum(dim=-1)
            roughness_denominator = roughness_support.sum(dim=-1)
            roughness_diagnostic_valid = active_mask & (
                roughness_denominator > self.eps
            )
            per_sample_warp_roughness = torch.where(
                roughness_diagnostic_valid,
                roughness_numerator
                / roughness_denominator.clamp_min(self.eps),
                zero_per_sample,
            )
            roughness_loss = (active * roughness_numerator).sum() / (
                (active * roughness_denominator).sum() + self.eps
            )

        unsupported_weight_grid = 1.0 - interval_support
        unsupported_numerator = (
            interval_weights
            * unsupported_weight_grid
            * log_speed.square()
        ).sum(dim=-1)
        unsupported_denominator = (
            interval_weights * unsupported_weight_grid
        ).sum(dim=-1)
        unsupported_diagnostic_valid = active_mask & (
            unsupported_denominator > self.eps
        )
        per_sample_unsupported_error = torch.where(
            unsupported_diagnostic_valid,
            unsupported_numerator
            / unsupported_denominator.clamp_min(self.eps),
            zero_per_sample,
        )
        unsupported_loss = (active * unsupported_numerator).sum() / (
            (active * unsupported_denominator).sum() + self.eps
        )

        sample_mean_support = (sample_support * grid_weights).sum(dim=-1)
        center_sample_weight = active * sample_mean_support
        center_weight_sum = center_sample_weight.sum()
        mean_tangent = (
            center_sample_weight.unsqueeze(-1) * phase.tangent
        ).sum(dim=0) / center_weight_sum.clamp_min(self.eps)
        raw_center_loss = (
            mean_tangent.square() * interval_weights
        ).sum()
        center_loss = torch.where(
            center_weight_sum > self.eps,
            raw_center_loss,
            zero,
        )

        total_loss = (
            self.alignment_weight * alignment_loss
            + self.roughness_weight * roughness_loss
            + self.unsupported_weight * unsupported_loss
            + self.center_weight * center_loss
        )
        return TemporalGeometryLossOutput(
            total_loss=total_loss,
            alignment_loss=alignment_loss,
            roughness_loss=roughness_loss,
            unsupported_loss=unsupported_loss,
            center_loss=center_loss,
            per_sample_alignment_error=per_sample_alignment_error,
            per_sample_warp_roughness=per_sample_warp_roughness,
            per_sample_unsupported_error=per_sample_unsupported_error,
            interval_support=interval_support,
            phase_tangent=phase.tangent,
            phase_magnitude=phase.magnitude,
            active_mask=active_mask,
            active_count=active_mask.sum(),
        )
