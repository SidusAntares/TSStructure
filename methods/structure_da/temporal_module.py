"""V3 trend/structure temporal core and task-feature assembly."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .temporal_coordinates import (
    TrendStructureCoordinateOutput,
    TrendStructureCoordinates,
)
from .temporal_functional import _resolve_time_mask
from .temporal_head import ShapeFeatureEncoder, ShapeFeatureOutput
from .temporal_registration import (
    MonotoneWarpEstimator,
    SourceRunningSRVFTemplate,
    SourceSRVFTemplateOutput,
    _apply_srvf_group_action,
    _warp_sequence,
    invert_monotone_warp,
)
from .temporal_selection import (
    TrendStructurePhaseSelectionOutput,
    TrendStructureSelectionConfig,
    select_trend_structure_phase,
)
from .temporal_srvf import TemporalSRVFExtractor, TemporalSRVFOutput

@dataclass(frozen=True)
class TrendStructureTemporalCoreOutput:
    trend_srvf: TemporalSRVFOutput
    structure_srvf: TemporalSRVFOutput
    trend_template: SourceSRVFTemplateOutput
    structure_diagnostic_template: SourceSRVFTemplateOutput
    selection: TrendStructurePhaseSelectionOutput


class TrendStructureTemporalCore(nn.Module):
    """Extract independent T/S geometry and select one T-generated phase warp."""

    def __init__(
        self,
        feature_dim: int,
        trend_num_basis: int = 12,
        structure_num_basis: int = 12,
        canonical_grid_size: int = 64,
        roughness_grid_size: int = 256,
        trend_smoothing_weight: float = 1e-2,
        structure_smoothing_weight: float = 1e-3,
        time_reference: float = 0.0,
        time_scale: float = 365.0,
        statistics_momentum: float = 0.99,
        support_scale_momentum: float = 0.99,
        template_momentum: float = 0.99,
        min_feature_scale: float = 1e-3,
        initial_support_scale: float = 1.0,
        min_support_scale: float = 1e-6,
        min_mean_support: float = 0.05,
        min_dynamic_energy: float = 1e-4,
        min_template_grid_weight: float = 1e-6,
        min_template_mean_support: float = 0.05,
        warp_hidden_dim: int = 64,
        warp_kernel_size: int = 5,
        warp_min_increment: float = 1e-4,
        warp_num_candidates: int = 3,
        candidate_init_warp_amplitude: float = 0.015,
        selection_config: TrendStructureSelectionConfig | None = None,
        srvf_eps: float = 1e-8,
        derivative_norm_threshold: float = 1e-8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        common = dict(
            feature_dim=feature_dim,
            canonical_grid_size=canonical_grid_size,
            roughness_grid_size=roughness_grid_size,
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
            eps=eps,
        )
        self.trend_srvf_extractor = TemporalSRVFExtractor(
            num_basis=trend_num_basis,
            smoothing_weight=trend_smoothing_weight,
            **common,
        )
        self.structure_srvf_extractor = TemporalSRVFExtractor(
            num_basis=structure_num_basis,
            smoothing_weight=structure_smoothing_weight,
            **common,
        )
        self.trend_template = SourceRunningSRVFTemplate(
            canonical_grid_size=canonical_grid_size,
            feature_dim=feature_dim,
            momentum=template_momentum,
            min_grid_weight=min_template_grid_weight,
            eps=srvf_eps,
        )
        self.structure_diagnostic_template = SourceRunningSRVFTemplate(
            canonical_grid_size=canonical_grid_size,
            feature_dim=feature_dim,
            momentum=template_momentum,
            min_grid_weight=min_template_grid_weight,
            eps=srvf_eps,
        )
        self.warp_estimator = MonotoneWarpEstimator(
            feature_dim=feature_dim,
            canonical_grid_size=canonical_grid_size,
            hidden_dim=warp_hidden_dim,
            kernel_size=warp_kernel_size,
            min_increment=warp_min_increment,
            num_candidates=warp_num_candidates,
            candidate_init_warp_amplitude=candidate_init_warp_amplitude,
        )
        if selection_config is not None and not isinstance(
            selection_config, TrendStructureSelectionConfig
        ):
            raise ValueError(
                "selection_config must be a TrendStructureSelectionConfig or None"
            )
        self.selection_config = selection_config or TrendStructureSelectionConfig(
            min_common_support=min_template_mean_support,
            eps=eps,
        )
        self.feature_dim = feature_dim
        self.canonical_grid_size = canonical_grid_size

    def _validate_inputs(
        self,
        trend: Tensor,
        structure: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        for name, value in (("trend", trend), ("structure", structure)):
            if not isinstance(value, Tensor) or value.ndim != 3:
                raise ValueError(f"{name} must have shape [B, L, D]")
            if value.shape[-1] != self.feature_dim:
                raise ValueError(
                    f"{name} feature dimension must equal feature_dim={self.feature_dim}"
                )
            if not value.is_floating_point():
                raise ValueError(f"{name} must use a floating-point dtype")
        if trend.shape != structure.shape:
            raise ValueError("trend and structure shape must match")
        if trend.dtype != structure.dtype:
            raise ValueError("trend and structure dtype must match")
        if trend.device != structure.device:
            raise ValueError("trend and structure device must match")
        reference = next(self.warp_estimator.parameters())
        if trend.device != reference.device:
            raise ValueError("trend and structure device must match module parameters")
        if trend.dtype != reference.dtype:
            raise ValueError("trend and structure dtype must match module parameters")
        batch_size, sequence_length = trend.shape[:2]
        resolved_positions = _resolve_pair_positions(
            positions,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        resolved_mask = _resolve_time_mask(
            time_mask, batch_size, sequence_length, trend.device
        )
        if not torch.isfinite(trend[resolved_mask]).all().item():
            raise ValueError("valid trend values must be finite")
        if not torch.isfinite(structure[resolved_mask]).all().item():
            raise ValueError("valid structure values must be finite")
        return resolved_positions, resolved_mask

    def _read_templates(
        self, srvf: Tensor
    ) -> tuple[SourceSRVFTemplateOutput, SourceSRVFTemplateOutput]:
        arguments = dict(device=srvf.device, dtype=srvf.dtype)
        return self.trend_template(**arguments), self.structure_diagnostic_template(
            **arguments
        )

    def _select(
        self,
        trend_srvf: TemporalSRVFOutput,
        structure_srvf: TemporalSRVFOutput,
        trend_template: SourceSRVFTemplateOutput,
        structure_template: SourceSRVFTemplateOutput,
    ) -> TrendStructurePhaseSelectionOutput:
        batch_size = trend_srvf.srvf.shape[0]
        trend_template_srvf = trend_template.srvf.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        trend_template_support = trend_template.support.unsqueeze(0).expand(
            batch_size, -1
        )
        structure_template_srvf = structure_template.srvf.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        structure_template_support = structure_template.support.unsqueeze(0).expand(
            batch_size, -1
        )
        integration_weights = self.trend_srvf_extractor.integration_weights.to(
            device=trend_srvf.srvf.device, dtype=trend_srvf.srvf.dtype
        )
        template_mean_support = (
            trend_template_support * integration_weights
        ).sum(dim=-1) / integration_weights.sum()
        trend_registration_eligible = (
            trend_srvf.structure_valid
            & trend_template.initialized.expand(batch_size)
            & (
                template_mean_support
                >= self.selection_config.min_common_support
            )
        )
        candidates = self.warp_estimator.forward_candidates(
            trend_srvf.srvf.detach(),
            trend_template_srvf.detach(),
            trend_srvf.support_confidence.detach(),
            trend_template_support.detach(),
            trend_registration_eligible,
        )
        return select_trend_structure_phase(
            trend_srvf=trend_srvf.srvf,
            trend_support=trend_srvf.support_confidence,
            trend_valid=trend_srvf.structure_valid,
            trend_template_srvf=trend_template_srvf,
            trend_template_support=trend_template_support,
            trend_template_initialized=trend_template.initialized,
            structure_srvf=structure_srvf.srvf,
            structure_support=structure_srvf.support_confidence,
            structure_valid=structure_srvf.structure_valid,
            structure_template_srvf=structure_template_srvf,
            structure_template_support=structure_template_support,
            structure_template_initialized=structure_template.initialized,
            candidates=candidates,
            integration_weights=integration_weights,
            config=self.selection_config,
        )

    def forward(
        self,
        trend: Tensor,
        structure: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> TrendStructureTemporalCoreOutput:
        resolved_positions, resolved_mask = self._validate_inputs(
            trend, structure, positions, time_mask
        )
        trend_srvf = self.trend_srvf_extractor(
            trend, resolved_positions, resolved_mask
        )
        structure_srvf = self.structure_srvf_extractor(
            structure, resolved_positions, resolved_mask
        )
        trend_template, structure_template = self._read_templates(
            trend_srvf.srvf
        )
        selection = self._select(
            trend_srvf, structure_srvf, trend_template, structure_template
        )
        return TrendStructureTemporalCoreOutput(
            trend_srvf=trend_srvf,
            structure_srvf=structure_srvf,
            trend_template=trend_template,
            structure_diagnostic_template=structure_template,
            selection=selection,
        )

    def warp_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.warp_estimator.parameters()

    def non_warp_parameters(self) -> Iterator[nn.Parameter]:
        warp_ids = {id(parameter) for parameter in self.warp_estimator.parameters()}
        for parameter in self.parameters():
            if id(parameter) not in warp_ids:
                yield parameter

    @torch.no_grad()
    def update_source_state(
        self,
        trend: Tensor,
        structure: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> None:
        resolved_positions, resolved_mask = self._validate_inputs(
            trend, structure, positions, time_mask
        )
        self.trend_srvf_extractor.update_source_statistics(trend, resolved_mask)
        self.structure_srvf_extractor.update_source_statistics(
            structure, resolved_mask
        )
        first_trend = self.trend_srvf_extractor(
            trend, resolved_positions, resolved_mask
        )
        first_structure = self.structure_srvf_extractor(
            structure, resolved_positions, resolved_mask
        )
        self.trend_srvf_extractor.update_source_support_scale(
            first_trend.functional
        )
        self.structure_srvf_extractor.update_source_support_scale(
            first_structure.functional
        )
        second_trend = self.trend_srvf_extractor(
            trend, resolved_positions, resolved_mask
        )
        second_structure = self.structure_srvf_extractor(
            structure, resolved_positions, resolved_mask
        )
        before_trend, before_structure = self._read_templates(second_trend.srvf)
        trend_was_initialized = bool(before_trend.initialized.item())
        structure_was_initialized = bool(before_structure.initialized.item())

        if not trend_was_initialized:
            self.trend_template.update(
                second_trend.srvf,
                second_trend.support_confidence,
                second_trend.structure_valid,
            )
        if not structure_was_initialized:
            self.structure_diagnostic_template.update(
                second_structure.srvf,
                second_structure.support_confidence,
                second_structure.structure_valid,
            )

        refreshed_trend, refreshed_structure = self._read_templates(
            second_trend.srvf
        )
        if trend_was_initialized or structure_was_initialized:
            selection = self._select(
                second_trend,
                second_structure,
                refreshed_trend,
                refreshed_structure,
            )
            if trend_was_initialized:
                self.trend_template.update(
                    selection.accepted_trend_registered_srvf,
                    selection.accepted_trend_registered_support,
                    selection.phase_valid,
                )
            if structure_was_initialized:
                self.structure_diagnostic_template.update(
                    selection.accepted_structure_registered_srvf,
                    selection.accepted_structure_registered_support,
                    selection.structure_shape_valid,
                )


@dataclass(frozen=True)
class TrendStructureTaskFeatureOutput:
    core: TrendStructureTemporalCoreOutput
    coordinates: TrendStructureCoordinateOutput
    shape: ShapeFeatureOutput

    aligned_positions: Tensor
    aligned_positions_valid: Tensor

    aligned_structure_srvf: Tensor
    aligned_structure_support: Tensor


class TrendStructureTaskFeatureModule(nn.Module):
    """Build differentiable Shape and inverse-warped time task features."""

    def __init__(
        self,
        feature_dim: int,
        shape_output_dim: int,
        trend_num_basis: int = 12,
        structure_num_basis: int = 12,
        canonical_grid_size: int = 64,
        roughness_grid_size: int = 256,
        trend_smoothing_weight: float = 1e-2,
        structure_smoothing_weight: float = 1e-3,
        time_reference: float = 0.0,
        time_scale: float = 365.0,
        statistics_momentum: float = 0.99,
        support_scale_momentum: float = 0.99,
        template_momentum: float = 0.99,
        min_feature_scale: float = 1e-3,
        initial_support_scale: float = 1.0,
        min_support_scale: float = 1e-6,
        min_mean_support: float = 0.05,
        min_dynamic_energy: float = 1e-4,
        min_template_grid_weight: float = 1e-6,
        min_template_mean_support: float = 0.05,
        warp_hidden_dim: int = 64,
        warp_kernel_size: int = 5,
        warp_min_increment: float = 1e-4,
        warp_num_candidates: int = 3,
        candidate_init_warp_amplitude: float = 0.015,
        selection_config: TrendStructureSelectionConfig | None = None,
        srvf_eps: float = 1e-8,
        derivative_norm_threshold: float = 1e-8,
        eps: float = 1e-6,
        num_shape_basis: int = 8,
        num_phase_basis: int = 8,
        attribute_projection_dim: int = 8,
        min_basis_support: float = 1e-4,
        shape_hidden_dim: int = 128,
        shape_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.core = TrendStructureTemporalCore(
            feature_dim=feature_dim,
            trend_num_basis=trend_num_basis,
            structure_num_basis=structure_num_basis,
            canonical_grid_size=canonical_grid_size,
            roughness_grid_size=roughness_grid_size,
            trend_smoothing_weight=trend_smoothing_weight,
            structure_smoothing_weight=structure_smoothing_weight,
            time_reference=time_reference,
            time_scale=time_scale,
            statistics_momentum=statistics_momentum,
            support_scale_momentum=support_scale_momentum,
            template_momentum=template_momentum,
            min_feature_scale=min_feature_scale,
            initial_support_scale=initial_support_scale,
            min_support_scale=min_support_scale,
            min_mean_support=min_mean_support,
            min_dynamic_energy=min_dynamic_energy,
            min_template_grid_weight=min_template_grid_weight,
            min_template_mean_support=min_template_mean_support,
            warp_hidden_dim=warp_hidden_dim,
            warp_kernel_size=warp_kernel_size,
            warp_min_increment=warp_min_increment,
            warp_num_candidates=warp_num_candidates,
            candidate_init_warp_amplitude=candidate_init_warp_amplitude,
            selection_config=selection_config,
            srvf_eps=srvf_eps,
            derivative_norm_threshold=derivative_norm_threshold,
            eps=eps,
        )
        self.coordinates = TrendStructureCoordinates(
            feature_dim=feature_dim,
            canonical_grid_size=canonical_grid_size,
            num_shape_basis=num_shape_basis,
            num_phase_basis=num_phase_basis,
            attribute_projection_dim=attribute_projection_dim,
            min_basis_support=min_basis_support,
            eps=eps,
        )
        self.shape_encoder = ShapeFeatureEncoder(
            num_shape_basis=num_shape_basis,
            attribute_projection_dim=attribute_projection_dim,
            output_dim=shape_output_dim,
            hidden_dim=shape_hidden_dim,
            dropout=shape_dropout,
        )
        self.time_reference = float(time_reference)
        self.time_scale = float(time_scale)
        self.eps = float(eps)

    def _rebuild_task_aligned_structure(
        self,
        core: TrendStructureTemporalCoreOutput,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(core, TrendStructureTemporalCoreOutput):
            raise ValueError("core must be a TrendStructureTemporalCoreOutput")
        warp = core.selection.accepted_warp.warp.detach()
        warp_derivative = (
            core.selection.accepted_warp.warp_derivative.detach()
        )
        aligned_srvf = _apply_srvf_group_action(
            core.structure_srvf.srvf,
            warp,
            warp_derivative,
            self.eps,
        )
        aligned_support = _warp_sequence(
            core.structure_srvf.support_confidence.unsqueeze(-1),
            warp,
        ).squeeze(-1)
        shape_valid = core.selection.structure_shape_valid
        aligned_srvf = torch.where(
            shape_valid[:, None, None],
            aligned_srvf,
            torch.zeros_like(aligned_srvf),
        )
        aligned_support = torch.where(
            shape_valid[:, None],
            aligned_support,
            torch.zeros_like(aligned_support),
        )
        if not torch.isfinite(aligned_srvf).all().item() or not torch.isfinite(
            aligned_support
        ).all().item():
            raise ValueError("aligned structure outputs must be finite")
        return aligned_srvf, aligned_support

    def _align_positions(
        self,
        positions: Tensor,
        time_mask: Tensor,
        core: TrendStructureTemporalCoreOutput,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(core, TrendStructureTemporalCoreOutput):
            raise ValueError("core must be a TrendStructureTemporalCoreOutput")
        batch_size = core.selection.phase_valid.shape[0]
        if not isinstance(positions, Tensor) or not positions.is_floating_point():
            raise ValueError("positions must be a floating-point tensor")
        sequence_length = positions.shape[-1] if positions.ndim in (1, 2) else -1
        resolved_positions = _resolve_pair_positions(
            positions,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        resolved_mask = _resolve_time_mask(
            time_mask,
            batch_size,
            sequence_length,
            resolved_positions.device,
        )
        valid_positions = resolved_positions[resolved_mask]
        normalized_valid = (
            valid_positions - self.time_reference
        ) / self.time_scale
        if normalized_valid.numel() and (
            torch.any(normalized_valid < -self.eps).item()
            or torch.any(normalized_valid > 1.0 + self.eps).item()
        ):
            raise ValueError("valid normalized positions must lie in [0, 1]")
        normalized = (
            (resolved_positions - self.time_reference) / self.time_scale
        ).clamp(0.0, 1.0)
        safe_u = torch.where(
            resolved_mask, normalized, torch.zeros_like(normalized)
        )
        mapped_u = invert_monotone_warp(
            core.selection.accepted_warp.warp.detach(),
            query=safe_u,
            eps=self.eps,
        )
        aligned_positions = torch.where(
            core.selection.phase_valid[:, None],
            mapped_u,
            normalized,
        )
        aligned_positions = torch.where(
            resolved_mask,
            aligned_positions,
            torch.zeros_like(aligned_positions),
        ).detach()
        aligned_positions_valid = (
            core.selection.phase_valid & resolved_mask.any(dim=-1)
        )
        return aligned_positions, aligned_positions_valid

    def forward(
        self,
        trend: Tensor,
        structure: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> TrendStructureTaskFeatureOutput:
        core = self.core(trend, structure, positions, time_mask)
        aligned_structure_srvf, aligned_structure_support = (
            self._rebuild_task_aligned_structure(core)
        )
        coordinates = self.coordinates(
            aligned_structure_srvf=aligned_structure_srvf,
            aligned_structure_support=aligned_structure_support,
            interval_widths=(
                core.selection.accepted_warp.interval_widths.detach()
            ),
            shape_valid=core.selection.structure_shape_valid,
            phase_valid=core.selection.phase_valid,
        )
        shape = self.shape_encoder(
            coordinates.shape_coordinates,
            coordinates.shape_valid,
        )
        aligned_positions, aligned_positions_valid = self._align_positions(
            positions, time_mask, core
        )
        return TrendStructureTaskFeatureOutput(
            core=core,
            coordinates=coordinates,
            shape=shape,
            aligned_positions=aligned_positions,
            aligned_positions_valid=aligned_positions_valid,
            aligned_structure_srvf=aligned_structure_srvf,
            aligned_structure_support=aligned_structure_support,
        )

    def warp_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.core.warp_parameters()

    def non_warp_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.core.non_warp_parameters()
        yield from self.coordinates.parameters()
        yield from self.shape_encoder.parameters()

    @torch.no_grad()
    def update_source_state(
        self,
        trend: Tensor,
        structure: Tensor,
        positions: Tensor,
        time_mask: Tensor,
    ) -> None:
        self.core.update_source_state(trend, structure, positions, time_mask)


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
