"""Complete and globally shared temporal structure operators."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Iterator

import torch
from torch import Tensor, nn

from .temporal_coordinates import (
    TemporalCoordinateOutput,
    TemporalShapePhaseCoordinates,
)
from .temporal_functional import _resolve_time_mask
from .temporal_geometry import (
    TemporalGeometryLossOutput,
    TemporalGeometryObjective,
)
from .temporal_head import (
    TemporalStructureEncoder,
    TemporalStructureFeatureOutput,
)
from .temporal_registration import (
    TemporalRegistrationOutput,
    TemporalSRVFRegistration,
    _apply_srvf_group_action,
    _warp_sequence,
)


@dataclass(frozen=True)
class TemporalStructureOutput:
    registration: TemporalRegistrationOutput
    coordinates: TemporalCoordinateOutput
    encoded: TemporalStructureFeatureOutput
    geometry_registration: TemporalRegistrationOutput | None = None


@dataclass(frozen=True)
class TemporalGeometryForwardOutput:
    structure: TemporalStructureOutput
    geometry: TemporalGeometryLossOutput


@dataclass(frozen=True)
class TemporalStructurePairOutput:
    trend: TemporalStructureOutput
    dynamics: TemporalStructureOutput


@dataclass(frozen=True)
class TemporalGeometryPairOutput:
    trend: TemporalGeometryForwardOutput
    dynamics: TemporalGeometryForwardOutput
    total_loss: Tensor


def _validate_source_mask(
    source_mask: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    if (
        not isinstance(source_mask, Tensor)
        or source_mask.dtype != torch.bool
        or source_mask.shape != (batch_size,)
    ):
        raise ValueError("source_mask must be a boolean tensor with shape [B]")
    if source_mask.device != device:
        raise ValueError("source_mask device must match component tokens")


def _resolve_pair_positions(
    positions: Tensor,
    *,
    batch_size: int,
    sequence_length: int,
) -> Tensor:
    if not isinstance(positions, Tensor):
        raise ValueError("positions must be a torch.Tensor")
    if positions.ndim == 1:
        if positions.shape != (sequence_length,):
            raise ValueError("positions must have shape [L] or [B, L]")
        resolved = positions.unsqueeze(0).expand(batch_size, -1)
    elif positions.ndim == 2:
        if positions.shape != (batch_size, sequence_length):
            raise ValueError("positions must have shape [L] or [B, L]")
        resolved = positions
    else:
        raise ValueError("positions must have shape [L] or [B, L]")
    if resolved.is_complex() or resolved.dtype == torch.bool:
        raise ValueError("positions must contain finite real values")
    if not torch.isfinite(resolved).all().item():
        raise ValueError("positions must contain only finite values")
    return resolved


def _split_batched_dataclass(value, batch_size: int):
    """Split tensors with a leading 2B axis while preserving scalar metadata."""

    if isinstance(value, Tensor):
        if value.ndim > 0 and value.shape[0] == 2 * batch_size:
            return value[:batch_size], value[batch_size:]
        return value, value
    if is_dataclass(value):
        left = {}
        right = {}
        for field in fields(value):
            first, second = _split_batched_dataclass(
                getattr(value, field.name), batch_size
            )
            left[field.name] = first
            right[field.name] = second
        return type(value)(**left), type(value)(**right)
    return value, value


def _detach_dataclass(value):
    if isinstance(value, Tensor):
        return value.detach()
    if is_dataclass(value):
        return type(value)(
            **{
                field.name: _detach_dataclass(getattr(value, field.name))
                for field in fields(value)
            }
        )
    return value


def _floating_dataclass_to_float32(value):
    """Copy a dataclass tree while promoting low-precision tensors."""

    if isinstance(value, Tensor):
        if value.dtype in (torch.float16, torch.bfloat16):
            return value.float()
        return value
    if is_dataclass(value):
        return type(value)(
            **{
                field.name: _floating_dataclass_to_float32(
                    getattr(value, field.name)
                )
                for field in fields(value)
            }
        )
    return value


class TemporalStructureExtractor(nn.Module):
    """Compose registration, coordinates, encoding, and geometry for one component."""

    def __init__(
        self,
        feature_dim: int,
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
        template_momentum: float = 0.99,
        min_template_grid_weight: float = 1e-6,
        min_template_mean_support: float = 0.05,
        warp_hidden_dim: int = 64,
        warp_kernel_size: int = 5,
        warp_min_increment: float = 1e-4,
        num_shape_basis: int = 8,
        num_phase_basis: int = 8,
        attribute_projection_dim: int = 8,
        min_basis_support: float = 1e-4,
        coordinate_hidden_dim: int = 64,
        structure_dim: int = 128,
        dropout: float = 0.1,
        geometry_alignment_weight: float = 1.0,
        geometry_roughness_weight: float = 1.0,
        geometry_unsupported_weight: float = 1.0,
        geometry_center_weight: float = 1.0,
        srvf_eps: float = 1e-8,
        derivative_norm_threshold: float = 1e-8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.registration = TemporalSRVFRegistration(
            feature_dim=feature_dim,
            num_basis=num_basis,
            canonical_grid_size=canonical_grid_size,
            roughness_grid_size=roughness_grid_size,
            smoothing_weight=smoothing_weight,
            time_reference=time_reference,
            time_scale=time_scale,
            statistics_momentum=statistics_momentum,
            min_feature_scale=min_feature_scale,
            support_scale_momentum=support_scale_momentum,
            initial_support_scale=initial_support_scale,
            min_support_scale=min_support_scale,
            min_mean_support=min_mean_support,
            min_dynamic_energy=min_dynamic_energy,
            srvf_eps=srvf_eps,
            derivative_norm_threshold=derivative_norm_threshold,
            template_momentum=template_momentum,
            min_template_grid_weight=min_template_grid_weight,
            min_template_mean_support=min_template_mean_support,
            warp_hidden_dim=warp_hidden_dim,
            warp_kernel_size=warp_kernel_size,
            warp_min_increment=warp_min_increment,
            eps=eps,
        )
        self.coordinates = TemporalShapePhaseCoordinates(
            feature_dim=feature_dim,
            canonical_grid_size=canonical_grid_size,
            num_shape_basis=num_shape_basis,
            num_phase_basis=num_phase_basis,
            attribute_projection_dim=attribute_projection_dim,
            min_basis_support=min_basis_support,
            eps=eps,
        )
        self.encoder = TemporalStructureEncoder(
            num_shape_basis=num_shape_basis,
            attribute_projection_dim=attribute_projection_dim,
            num_phase_basis=num_phase_basis,
            coordinate_hidden_dim=coordinate_hidden_dim,
            structure_dim=structure_dim,
            dropout=dropout,
        )
        self.geometry_objective = TemporalGeometryObjective(
            canonical_grid_size=canonical_grid_size,
            alignment_weight=geometry_alignment_weight,
            roughness_weight=geometry_roughness_weight,
            unsupported_weight=geometry_unsupported_weight,
            center_weight=geometry_center_weight,
            eps=eps,
        )

    def _validate_component_tokens(self, component_tokens: Tensor) -> None:
        functional_lift = self.registration.srvf_extractor.functional_lift
        functional_lift._validate_component_tokens(component_tokens)
        reference_parameter = next(self.parameters())
        if component_tokens.device != reference_parameter.device:
            raise ValueError(
                "component_tokens device must match module parameter device"
            )
        if component_tokens.dtype != reference_parameter.dtype:
            raise ValueError(
                "component_tokens dtype must match module parameter dtype"
            )

    def _forward_registration(
        self,
        component_tokens: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> TemporalRegistrationOutput:
        self._validate_component_tokens(component_tokens)
        return self.registration(component_tokens, positions, time_mask)

    def _encode_registration(
        self,
        registration: TemporalRegistrationOutput,
        geometry_registration: TemporalRegistrationOutput | None = None,
    ) -> TemporalStructureOutput:
        device_type = registration.registered_srvf.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            float_registration = _floating_dataclass_to_float32(registration)
            coordinates = self.coordinates(float_registration)
        encoded = self.encoder(coordinates)
        return TemporalStructureOutput(
            registration=registration,
            coordinates=coordinates,
            encoded=encoded,
            geometry_registration=geometry_registration,
        )

    def _make_task_registration(
        self,
        registration: TemporalRegistrationOutput,
    ) -> TemporalRegistrationOutput:
        if not isinstance(registration, TemporalRegistrationOutput):
            raise ValueError(
                "registration must be a TemporalRegistrationOutput"
            )
        task_warp = registration.warp.detach()
        task_warp_derivative = registration.warp_derivative.detach()
        task_interval_logits = registration.interval_logits.detach()
        task_interval_widths = registration.interval_widths.detach()
        task_registered_srvf = _apply_srvf_group_action(
            registration.srvf_output.srvf,
            task_warp,
            task_warp_derivative,
            self.registration.eps,
        )
        task_registered_support = _warp_sequence(
            registration.srvf_output.support_confidence.unsqueeze(-1),
            task_warp,
        ).squeeze(-1)
        return TemporalRegistrationOutput(
            srvf_output=registration.srvf_output,
            template_srvf=registration.template_srvf,
            template_support=registration.template_support,
            template_initialized=registration.template_initialized,
            template_mean_support=registration.template_mean_support,
            interval_logits=task_interval_logits,
            interval_widths=task_interval_widths,
            warp=task_warp,
            warp_derivative=task_warp_derivative,
            registered_srvf=task_registered_srvf,
            registered_support=task_registered_support,
            registration_valid=registration.registration_valid,
        )

    def _make_geometry_registration(
        self,
        registration: TemporalRegistrationOutput,
    ) -> TemporalRegistrationOutput:
        if not isinstance(registration, TemporalRegistrationOutput):
            raise ValueError(
                "registration must be a TemporalRegistrationOutput"
            )
        geometry_srvf = registration.srvf_output.srvf.detach()
        geometry_support = registration.srvf_output.support_confidence.detach()
        registered_srvf = _apply_srvf_group_action(
            geometry_srvf,
            registration.warp,
            registration.warp_derivative,
            self.registration.eps,
        )
        registered_support = _warp_sequence(
            geometry_support.unsqueeze(-1),
            registration.warp,
        ).squeeze(-1)
        return TemporalRegistrationOutput(
            srvf_output=_detach_dataclass(registration.srvf_output),
            template_srvf=registration.template_srvf.detach(),
            template_support=registration.template_support.detach(),
            template_initialized=registration.template_initialized,
            template_mean_support=registration.template_mean_support.detach(),
            interval_logits=registration.interval_logits,
            interval_widths=registration.interval_widths,
            warp=registration.warp,
            warp_derivative=registration.warp_derivative,
            registered_srvf=registered_srvf,
            registered_support=registered_support,
            registration_valid=registration.registration_valid,
        )

    def forward_task(
        self,
        component_tokens: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> TemporalStructureOutput:
        registration = self._forward_registration(
            component_tokens, positions, time_mask
        )
        task_registration = self._make_task_registration(registration)
        geometry_registration = self._make_geometry_registration(registration)
        return self._encode_registration(
            task_registration, geometry_registration
        )

    def forward_task_pair(
        self,
        first: Tensor,
        second: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> tuple[TemporalStructureOutput, TemporalStructureOutput]:
        batch_size = first.shape[0]
        combined_tokens = torch.cat([first, second], dim=0)
        combined_positions = torch.cat([positions, positions], dim=0)
        combined_mask = torch.cat([time_mask, time_mask], dim=0)
        registration = self._forward_registration(
            combined_tokens, combined_positions, combined_mask
        )
        task_registration = self._make_task_registration(registration)
        geometry_registration = self._make_geometry_registration(registration)
        first_geometry, second_geometry = _split_batched_dataclass(
            geometry_registration, batch_size
        )
        first_task, second_task = _split_batched_dataclass(
            task_registration, batch_size
        )
        return (
            self._encode_registration(first_task, first_geometry),
            self._encode_registration(second_task, second_geometry),
        )

    def forward(
        self,
        component_tokens: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> TemporalStructureOutput:
        return self.forward_task(component_tokens, positions, time_mask)

    def forward_geometry(
        self,
        component_tokens: Tensor,
        positions: Tensor,
        time_mask: Tensor,
        source_mask: Tensor,
    ) -> TemporalGeometryForwardOutput:
        task = self.forward_task(component_tokens, positions, time_mask)
        return self.forward_geometry_from_task(task, source_mask)

    def forward_geometry_from_task(
        self,
        task: TemporalStructureOutput,
        source_mask: Tensor,
    ) -> TemporalGeometryForwardOutput:
        if not isinstance(task, TemporalStructureOutput):
            raise ValueError("task must be a TemporalStructureOutput")
        registration = task.geometry_registration
        if registration is None:
            raise ValueError("task output does not contain cached geometry registration")
        _validate_source_mask(
            source_mask,
            batch_size=registration.registered_srvf.shape[0],
            device=registration.registered_srvf.device,
        )
        structure = TemporalStructureOutput(
            registration=registration,
            coordinates=task.coordinates,
            encoded=task.encoded,
            geometry_registration=registration,
        )
        geometry = self.geometry_objective(registration, source_mask)
        return TemporalGeometryForwardOutput(
            structure=structure,
            geometry=geometry,
        )

    @torch.no_grad()
    def update_source_state(
        self,
        component_tokens: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> None:
        self._validate_component_tokens(component_tokens)
        self.registration.update_source_statistics(
            component_tokens, time_mask
        )
        first_pass = self.registration(
            component_tokens, positions, time_mask
        )
        self.registration.update_source_support_scale(
            first_pass.srvf_output.functional
        )
        second_pass = self.registration(
            component_tokens, positions, time_mask
        )
        self.registration.update_source_template(second_pass)

    def warp_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.registration.warp_estimator.parameters()

    def non_warp_parameters(self) -> Iterator[nn.Parameter]:
        warp_ids = {
            id(parameter)
            for parameter in self.registration.warp_estimator.parameters()
        }
        for parameter in self.parameters():
            if id(parameter) not in warp_ids:
                yield parameter


class SharedTemporalStructureOperator(nn.Module):
    """Apply one global temporal extractor to both trend and dynamics."""

    def __init__(self, extractor: TemporalStructureExtractor) -> None:
        super().__init__()
        if not isinstance(extractor, TemporalStructureExtractor):
            raise ValueError(
                "extractor must be a TemporalStructureExtractor"
            )
        self.extractor = extractor

    def _validate_pair_inputs(
        self,
        trend: Tensor,
        dynamics: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        for name, component in (("trend", trend), ("dynamics", dynamics)):
            if not isinstance(component, Tensor) or component.ndim != 3:
                raise ValueError(
                    f"{name} must be a three-dimensional [B, L, D] tensor"
                )
            if not component.is_floating_point():
                raise ValueError(f"{name} must use a floating-point dtype")
        if trend.shape != dynamics.shape:
            raise ValueError("trend and dynamics shape must match")
        if trend.dtype != dynamics.dtype:
            raise ValueError("trend and dynamics dtype must match")
        if trend.device != dynamics.device:
            raise ValueError("trend and dynamics device must match")
        self.extractor._validate_component_tokens(trend)
        self.extractor._validate_component_tokens(dynamics)
        batch_size, sequence_length = trend.shape[:2]
        resolved_positions = _resolve_pair_positions(
            positions,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        resolved_time_mask = _resolve_time_mask(
            time_mask,
            batch_size,
            sequence_length,
            trend.device,
        )
        if not torch.isfinite(trend[resolved_time_mask]).all().item():
            raise ValueError("valid trend values must be finite")
        if not torch.isfinite(dynamics[resolved_time_mask]).all().item():
            raise ValueError("valid dynamics values must be finite")
        return resolved_positions, resolved_time_mask

    def forward_task(
        self,
        trend: Tensor,
        dynamics: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> TemporalStructurePairOutput:
        resolved_positions, resolved_time_mask = self._validate_pair_inputs(
            trend, dynamics, positions, time_mask
        )
        trend_output, dynamics_output = self.extractor.forward_task_pair(
            trend,
            dynamics,
            resolved_positions,
            resolved_time_mask,
        )
        return TemporalStructurePairOutput(
            trend=trend_output,
            dynamics=dynamics_output,
        )

    def forward(
        self,
        trend: Tensor,
        dynamics: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> TemporalStructurePairOutput:
        return self.forward_task(trend, dynamics, positions, time_mask)

    def forward_geometry(
        self,
        trend: Tensor,
        dynamics: Tensor,
        positions: Tensor,
        time_mask: Tensor,
        source_mask: Tensor,
    ) -> TemporalGeometryPairOutput:
        task = self.forward_task(trend, dynamics, positions, time_mask)
        return self.forward_geometry_from_task(task, source_mask)

    def forward_geometry_from_task(
        self,
        task: TemporalStructurePairOutput,
        source_mask: Tensor,
    ) -> TemporalGeometryPairOutput:
        if not isinstance(task, TemporalStructurePairOutput):
            raise ValueError("task must be a TemporalStructurePairOutput")
        trend_output = self.extractor.forward_geometry_from_task(
            task.trend, source_mask
        )
        dynamics_output = self.extractor.forward_geometry_from_task(
            task.dynamics, source_mask
        )
        total_loss = 0.5 * (
            trend_output.geometry.total_loss
            + dynamics_output.geometry.total_loss
        )
        return TemporalGeometryPairOutput(
            trend=trend_output,
            dynamics=dynamics_output,
            total_loss=total_loss,
        )

    @torch.no_grad()
    def update_source_state(
        self,
        trend: Tensor,
        dynamics: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> None:
        resolved_positions, resolved_time_mask = self._validate_pair_inputs(
            trend, dynamics, positions, time_mask
        )
        combined_tokens = torch.cat([trend, dynamics], dim=0)
        combined_positions = torch.cat(
            [resolved_positions, resolved_positions], dim=0
        )
        combined_time_mask = torch.cat(
            [resolved_time_mask, resolved_time_mask], dim=0
        )
        self.extractor.update_source_state(
            combined_tokens,
            combined_positions,
            combined_time_mask,
        )
