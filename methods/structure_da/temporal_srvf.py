"""Support-aware vector-valued SRVF extraction on canonical time grids."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .temporal_functional import (
    TemporalFunctionalLift,
    TemporalFunctionalOutput,
    _finite_float,
)


class SourceRunningSupportScale(nn.Module):
    """Track a source-only EMA scale for information-variance proxies."""

    def __init__(
        self,
        momentum: float = 0.99,
        initial_scale: float = 1.0,
        min_scale: float = 1e-6,
    ) -> None:
        super().__init__()
        momentum = _finite_float("momentum", momentum)
        initial_scale = _finite_float("initial_scale", initial_scale)
        min_scale = _finite_float("min_scale", min_scale)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if initial_scale <= 0:
            raise ValueError("initial_scale must be greater than zero")
        if min_scale <= 0:
            raise ValueError("min_scale must be greater than zero")

        self.momentum = momentum
        self.initial_scale = initial_scale
        self.min_scale = min_scale
        self.register_buffer("running_scale", torch.tensor(initial_scale))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(
        self,
        information_variance: Tensor,
        sample_valid: Tensor,
    ) -> None:
        if (
            not isinstance(information_variance, Tensor)
            or information_variance.ndim != 2
        ):
            raise ValueError("information_variance must have shape [B, K]")
        if not information_variance.is_floating_point():
            raise ValueError("information_variance must use a floating-point dtype")
        if not torch.isfinite(information_variance).all().item():
            raise ValueError("information_variance must contain only finite values")
        if torch.any(information_variance < 0).item():
            raise ValueError("information_variance must contain non-negative values")
        if not isinstance(sample_valid, Tensor) or sample_valid.dtype != torch.bool:
            raise ValueError("sample_valid must be a boolean tensor with shape [B]")
        if sample_valid.shape != (information_variance.shape[0],):
            raise ValueError("sample_valid must be a boolean tensor with shape [B]")

        valid = sample_valid.to(device=information_variance.device)
        if not torch.any(valid).item():
            return
        batch_scale = information_variance[valid].mean().to(
            device=self.running_scale.device,
            dtype=self.running_scale.dtype,
        ).clamp_min(self.min_scale)
        if self.num_updates.item() == 0:
            self.running_scale.copy_(batch_scale)
        else:
            self.running_scale.mul_(self.momentum).add_(
                batch_scale, alpha=1.0 - self.momentum
            ).clamp_min_(self.min_scale)
        self.num_updates.add_(1)

    def forward(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point dtype")
        if self.num_updates.item() == 0:
            return torch.tensor(self.initial_scale, device=device, dtype=dtype)
        return self.running_scale.to(device=device, dtype=dtype)


@dataclass(frozen=True)
class TemporalSRVFOutput:
    functional: TemporalFunctionalOutput

    support_confidence: Tensor
    mean_support: Tensor

    derivative_norm: Tensor
    dynamic_energy: Tensor

    srvf: Tensor
    structure_valid: Tensor


class TemporalSRVFExtractor(nn.Module):
    """Lift temporal tokens and extract a reliability-gated vector SRVF."""

    def __init__(
        self,
        num_channels: int,
        channel_feature_dim: int,
        num_basis: int = 12,
        canonical_grid_size: int = 64,
        roughness_grid_size: int = 256,
        smoothing_weight: float = 1e-3,
        time_reference: float = 0.0,
        time_scale: float = 366.0,
        statistics_momentum: float = 0.99,
        min_feature_scale: float = 1e-3,
        support_scale_momentum: float = 0.99,
        initial_support_scale: float = 1.0,
        min_support_scale: float = 1e-6,
        min_mean_support: float = 0.05,
        min_dynamic_energy: float = 1e-4,
        srvf_eps: float = 1e-8,
        derivative_norm_threshold: float = 1e-8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        min_mean_support = _finite_float("min_mean_support", min_mean_support)
        min_dynamic_energy = _finite_float(
            "min_dynamic_energy", min_dynamic_energy
        )
        srvf_eps = _finite_float("srvf_eps", srvf_eps)
        derivative_norm_threshold = _finite_float(
            "derivative_norm_threshold", derivative_norm_threshold
        )
        if not 0.0 <= min_mean_support <= 1.0:
            raise ValueError("min_mean_support must be in [0, 1]")
        if min_dynamic_energy < 0:
            raise ValueError("min_dynamic_energy must be non-negative")
        if srvf_eps <= 0:
            raise ValueError("srvf_eps must be greater than zero")
        if derivative_norm_threshold < 0:
            raise ValueError("derivative_norm_threshold must be non-negative")

        self.functional_lift = TemporalFunctionalLift(
            num_channels=num_channels,
            channel_feature_dim=channel_feature_dim,
            num_basis=num_basis,
            canonical_grid_size=canonical_grid_size,
            roughness_grid_size=roughness_grid_size,
            smoothing_weight=smoothing_weight,
            time_reference=time_reference,
            time_scale=time_scale,
            statistics_momentum=statistics_momentum,
            min_feature_scale=min_feature_scale,
            eps=eps,
        )
        self.support_scale = SourceRunningSupportScale(
            momentum=support_scale_momentum,
            initial_scale=initial_support_scale,
            min_scale=min_support_scale,
        )
        self.num_channels = num_channels
        self.channel_feature_dim = channel_feature_dim
        self.canonical_grid_size = canonical_grid_size
        self.min_support_scale = min_support_scale
        self.min_mean_support = min_mean_support
        self.min_dynamic_energy = min_dynamic_energy
        self.srvf_eps = srvf_eps
        self.derivative_norm_threshold = derivative_norm_threshold

        integration_weights = torch.full(
            (canonical_grid_size,),
            1.0 / (canonical_grid_size - 1),
            dtype=torch.float64,
        )
        integration_weights[[0, -1]] *= 0.5
        integration_weights.div_(integration_weights.sum())
        self.register_buffer("integration_weights", integration_weights)

    @torch.no_grad()
    def update_source_statistics(
        self,
        component_tokens: Tensor,
        time_mask: Tensor,
    ) -> None:
        self.functional_lift.update_source_statistics(component_tokens, time_mask)

    def _validate_functional_output(
        self, functional_output: TemporalFunctionalOutput
    ) -> None:
        if not isinstance(functional_output, TemporalFunctionalOutput):
            raise ValueError("functional_output must be a TemporalFunctionalOutput")
        information_variance = functional_output.information_variance
        if (
            information_variance.ndim != 2
            or information_variance.shape[1] != self.canonical_grid_size
        ):
            raise ValueError(
                "functional output canonical grid size does not match extractor"
            )
        batch_size = information_variance.shape[0]
        if functional_output.solve_valid.shape != (batch_size,):
            raise ValueError("functional solve_valid must have shape [B]")
        if functional_output.solve_valid.dtype != torch.bool:
            raise ValueError("functional solve_valid must be boolean")
        if (
            functional_output.derivative.ndim != 3
            or functional_output.derivative.shape[:2]
            != (batch_size, self.canonical_grid_size)
        ):
            raise ValueError(
                "functional derivative must have shape [B, K, D]"
            )

    @torch.no_grad()
    def update_source_support_scale(
        self,
        functional_output: TemporalFunctionalOutput,
    ) -> None:
        self._validate_functional_output(functional_output)
        self.support_scale.update(
            functional_output.information_variance,
            functional_output.solve_valid,
        )

    def forward(
        self,
        component_tokens: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> TemporalSRVFOutput:
        functional = self.functional_lift(
            component_tokens,
            positions,
            time_mask,
        )
        self._validate_functional_output(functional)
        dtype = component_tokens.dtype
        device = component_tokens.device
        information_variance = functional.information_variance
        tau_v = self.support_scale(device=device, dtype=dtype)
        support_confidence = 1.0 / (
            1.0
            + information_variance
            / tau_v.clamp_min(self.min_support_scale)
        )
        support_confidence = support_confidence.clamp(0.0, 1.0)
        support_confidence = torch.where(
            functional.solve_valid.unsqueeze(-1),
            support_confidence,
            torch.zeros_like(support_confidence),
        )

        integration_weights = self.integration_weights.to(
            device=device, dtype=dtype
        )
        mean_support = (
            support_confidence * integration_weights
        ).sum(dim=-1)
        derivative_norm = torch.linalg.vector_norm(
            functional.derivative,
            ord=2,
            dim=-1,
        )
        dynamic_energy = (
            integration_weights
            * support_confidence
            * derivative_norm
        ).sum(dim=-1)
        structure_valid = (
            functional.solve_valid
            & (mean_support >= self.min_mean_support)
            & (dynamic_energy >= self.min_dynamic_energy)
        )

        denominator = torch.sqrt(derivative_norm + self.srvf_eps)
        srvf = functional.derivative / denominator.unsqueeze(-1)
        srvf = torch.where(
            (derivative_norm > self.derivative_norm_threshold).unsqueeze(-1),
            srvf,
            torch.zeros_like(srvf),
        )
        srvf = torch.where(
            structure_valid.reshape(-1, 1, 1),
            srvf,
            torch.zeros_like(srvf),
        )
        return TemporalSRVFOutput(
            functional=functional,
            support_confidence=support_confidence,
            mean_support=mean_support,
            derivative_norm=derivative_norm,
            dynamic_energy=dynamic_energy,
            srvf=srvf,
            structure_valid=structure_valid,
        )
