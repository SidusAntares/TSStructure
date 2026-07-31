"""Support-aware multi-scale directed channel relation structure."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ChannelStructureOutput:
    """Task feature and auditable channel-relation intermediates."""

    feature: Tensor
    valid: Tensor
    state_relation: Tensor
    evolution_relation: Tensor
    relative_strength: Tensor
    state_reliability: Tensor
    evolution_reliability: Tensor
    reliable_edge_raw: Tensor
    edge_embedding: Tensor
    local_velocity: Tensor
    velocity_support: Tensor
    state_effective_pair_count: Tensor
    evolution_effective_pair_count: Tensor
    relation_mass: Tensor


@dataclass(frozen=True)
class ChannelStructurePairOutput:
    """Shared-extractor channel structure for trend and dynamics."""

    trend: ChannelStructureOutput
    dynamics: ChannelStructureOutput


@dataclass(frozen=True)
class _RelationComputation:
    state_relation: Tensor
    evolution_relation: Tensor
    relative_strength: Tensor
    state_reliability: Tensor
    evolution_reliability: Tensor
    local_velocity: Tensor
    velocity_support: Tensor
    state_effective_pair_count: Tensor
    evolution_effective_pair_count: Tensor
    state_energy_a: Tensor
    state_energy_b: Tensor
    evolution_energy_a: Tensor
    evolution_energy_b: Tensor


def _finite_float(name: str, value: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


def _positive_int(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "at least 2" if minimum == 2 else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _strict_binary_mask(mask: Tensor, name: str) -> Tensor:
    if not isinstance(mask, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor or None")
    if mask.is_complex() or (
        mask.dtype != torch.bool
        and (
            not mask.is_floating_point()
            and mask.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
        )
    ):
        raise ValueError(f"{name} must contain boolean or finite 0/1 values")
    if mask.dtype != torch.bool:
        if not torch.isfinite(mask).all().item() or not torch.all(
            (mask == 0) | (mask == 1)
        ).item():
            raise ValueError(f"{name} must contain only finite 0/1 values")
    return mask.to(dtype=torch.bool)


class SourceRunningAttributeStandardizer(nn.Module):
    """EMA source statistics shared over samples, times, and channels."""

    def __init__(
        self,
        attribute_dim: int,
        momentum: float = 0.99,
        min_scale: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.attribute_dim = _positive_int("attribute_dim", attribute_dim)
        self.momentum = _finite_float("momentum", momentum)
        self.min_scale = _finite_float("min_scale", min_scale)
        self.eps = _finite_float("eps", eps)
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be greater than zero")
        if self.eps <= 0:
            raise ValueError("eps must be greater than zero")

        self.register_buffer("running_mean", torch.zeros(attribute_dim))
        self.register_buffer("running_second_moment", torch.zeros(attribute_dim))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def _validate(self, tokens: Tensor, channel_mask: Tensor) -> Tensor:
        if not isinstance(tokens, Tensor) or tokens.ndim != 4:
            raise ValueError("tokens must have shape [B, L, C, P]")
        if not tokens.is_floating_point():
            raise ValueError("tokens must use a floating-point dtype")
        if tokens.shape[-1] != self.attribute_dim:
            raise ValueError("tokens attribute dimension must match attribute_dim")
        mask = _strict_binary_mask(channel_mask, "channel_mask")
        if mask.shape != tokens.shape[:3]:
            raise ValueError("channel_mask must have shape [B, L, C]")
        if mask.device != tokens.device:
            mask = mask.to(device=tokens.device)
        if self.running_mean.device != tokens.device:
            raise ValueError("standardizer and tokens must use the same device")
        if self.running_mean.dtype != tokens.dtype:
            raise ValueError("standardizer and tokens must use the same dtype")
        if not torch.isfinite(tokens[mask]).all().item():
            raise ValueError("valid tokens must be finite")
        return mask

    @torch.no_grad()
    def update(self, tokens: Tensor, channel_mask: Tensor) -> None:
        mask = self._validate(tokens, channel_mask)
        values = tokens[mask]
        if values.numel() == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_second = values.square().mean(dim=0)
        if self.num_updates.item() == 0:
            self.running_mean.copy_(batch_mean)
            self.running_second_moment.copy_(batch_second)
        else:
            self.running_mean.mul_(self.momentum).add_(
                batch_mean, alpha=1.0 - self.momentum
            )
            self.running_second_moment.mul_(self.momentum).add_(
                batch_second, alpha=1.0 - self.momentum
            )
        self.num_updates.add_(1)

    def forward(self, tokens: Tensor) -> Tensor:
        if not isinstance(tokens, Tensor) or tokens.ndim != 4:
            raise ValueError("tokens must have shape [B, L, C, P]")
        if not tokens.is_floating_point() or tokens.shape[-1] != self.attribute_dim:
            raise ValueError("tokens must be floating point with attribute_dim features")
        if self.running_mean.device != tokens.device or self.running_mean.dtype != tokens.dtype:
            raise ValueError("standardizer and tokens must use the same dtype and device")
        if self.num_updates.item() == 0:
            return tokens
        variance = self.running_second_moment - self.running_mean.square()
        scale = torch.sqrt(variance.clamp_min(self.min_scale**2))
        return (tokens - self.running_mean) / (scale + self.eps)


class SourceRunningRelationEnergyScale(nn.Module):
    """EMA source energy scales used only to calibrate reliability."""

    def __init__(
        self,
        momentum: float = 0.99,
        initial_state_scale: float = 1.0,
        initial_evolution_scale: float = 1.0,
        min_scale: float = 1e-6,
    ) -> None:
        super().__init__()
        self.momentum = _finite_float("momentum", momentum)
        state = _finite_float("initial_state_scale", initial_state_scale)
        evolution = _finite_float(
            "initial_evolution_scale", initial_evolution_scale
        )
        self.min_scale = _finite_float("min_scale", min_scale)
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if state <= 0 or evolution <= 0:
            raise ValueError("initial energy scales must be greater than zero")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be greater than zero")
        self.register_buffer("running_state_scale", torch.tensor(state))
        self.register_buffer("running_evolution_scale", torch.tensor(evolution))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    @staticmethod
    def _batch_scale(energy: Tensor, valid: Tensor) -> Tensor | None:
        if not isinstance(energy, Tensor) or not energy.is_floating_point():
            raise ValueError("energy must be a floating-point tensor")
        valid = _strict_binary_mask(valid, "energy valid mask")
        if valid.shape != energy.shape:
            raise ValueError("energy and valid mask must have identical shapes")
        valid = valid.to(device=energy.device)
        keep = valid & torch.isfinite(energy) & (energy > 0)
        if not keep.any().item():
            return None
        return energy[keep].mean()

    @torch.no_grad()
    def update(
        self,
        state_energy: Tensor,
        evolution_energy: Tensor,
        state_valid: Tensor,
        evolution_valid: Tensor,
    ) -> None:
        state = self._batch_scale(state_energy, state_valid)
        evolution = self._batch_scale(evolution_energy, evolution_valid)
        if state is None and evolution is None:
            return
        first = self.num_updates.item() == 0
        if state is not None:
            state = state.to(self.running_state_scale)
            if first:
                self.running_state_scale.copy_(state.clamp_min(self.min_scale))
            else:
                self.running_state_scale.mul_(self.momentum).add_(
                    state.clamp_min(self.min_scale), alpha=1.0 - self.momentum
                )
        if evolution is not None:
            evolution = evolution.to(self.running_evolution_scale)
            if first:
                self.running_evolution_scale.copy_(
                    evolution.clamp_min(self.min_scale)
                )
            else:
                self.running_evolution_scale.mul_(self.momentum).add_(
                    evolution.clamp_min(self.min_scale),
                    alpha=1.0 - self.momentum,
                )
        self.num_updates.add_(1)

    def forward(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")
        return (
            self.running_state_scale.to(device=device, dtype=dtype).clamp_min(
                self.min_scale
            ),
            self.running_evolution_scale.to(
                device=device, dtype=dtype
            ).clamp_min(self.min_scale),
        )


class MultiScaleChannelRelationStructure(nn.Module):
    """Extract support-aware directed channel relations on true timestamps."""

    def __init__(
        self,
        num_channels: int,
        token_dim: int,
        lag_centers: Sequence[float] = (
            -0.20,
            -0.10,
            -0.05,
            0.0,
            0.05,
            0.10,
            0.20,
        ),
        lag_widths: Sequence[float] = (
            0.05,
            0.04,
            0.03,
            0.03,
            0.03,
            0.04,
            0.05,
        ),
        velocity_bandwidth: float = 0.05,
        edge_hidden_dim: int = 32,
        structure_dim: int = 128,
        statistics_momentum: float = 0.99,
        energy_scale_momentum: float = 0.99,
        min_attribute_scale: float = 1e-3,
        min_energy_scale: float = 1e-6,
        min_velocity_effective_count: float = 2.0,
        min_velocity_time_spread: float = 1e-4,
        min_effective_pairs: float = 3.0,
        min_relation_mass: float = 1.0,
        time_reference: float = 0.0,
        time_scale: float = 366.0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_channels = _positive_int("num_channels", num_channels, minimum=2)
        self.token_dim = _positive_int("token_dim", token_dim)
        self.edge_hidden_dim = _positive_int("edge_hidden_dim", edge_hidden_dim)
        self.structure_dim = _positive_int("structure_dim", structure_dim)

        try:
            centers = tuple(_finite_float("lag_centers", value) for value in lag_centers)
            widths = tuple(_finite_float("lag_widths", value) for value in lag_widths)
        except TypeError as error:
            raise ValueError("lag centers and widths must be finite sequences") from error
        if not centers or len(centers) != len(widths):
            raise ValueError("lag_centers and lag_widths must have equal nonzero length")
        if any(width <= 0 for width in widths):
            raise ValueError("lag_widths must be greater than zero")
        if any(right <= left for left, right in zip(centers, centers[1:])):
            raise ValueError("lag_centers must be strictly increasing")
        zero_count = sum(abs(value) <= 1e-12 for value in centers)
        if zero_count != 1:
            raise ValueError("lag_centers must contain exactly one zero")
        for index in range(len(centers)):
            if not math.isclose(
                centers[index], -centers[-1 - index], abs_tol=1e-12
            ):
                raise ValueError("lag_centers must be symmetric around zero")
            if not math.isclose(
                widths[index], widths[-1 - index], rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("lag_widths must be symmetric for paired lags")

        self.velocity_bandwidth = _finite_float(
            "velocity_bandwidth", velocity_bandwidth
        )
        self.min_velocity_effective_count = _finite_float(
            "min_velocity_effective_count", min_velocity_effective_count
        )
        self.min_velocity_time_spread = _finite_float(
            "min_velocity_time_spread", min_velocity_time_spread
        )
        self.min_effective_pairs = _finite_float(
            "min_effective_pairs", min_effective_pairs
        )
        self.min_relation_mass = _finite_float(
            "min_relation_mass", min_relation_mass
        )
        self.time_reference = _finite_float("time_reference", time_reference)
        self.time_scale = _finite_float("time_scale", time_scale)
        dropout = _finite_float("dropout", dropout)
        layer_norm_eps = _finite_float("layer_norm_eps", layer_norm_eps)
        self.eps = _finite_float("eps", eps)
        for name, value in (
            ("velocity_bandwidth", self.velocity_bandwidth),
            ("min_velocity_effective_count", self.min_velocity_effective_count),
            ("min_velocity_time_spread", self.min_velocity_time_spread),
            ("min_effective_pairs", self.min_effective_pairs),
            ("time_scale", self.time_scale),
            ("layer_norm_eps", layer_norm_eps),
            ("eps", self.eps),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.min_relation_mass < 0:
            raise ValueError("min_relation_mass must be nonnegative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.num_lags = len(centers)
        source_indices = []
        target_indices = []
        for source in range(num_channels):
            for target in range(num_channels):
                if source != target:
                    source_indices.append(source)
                    target_indices.append(target)
        self.num_edges = len(source_indices)
        self._directed_pairs = tuple(zip(source_indices, target_indices))
        self.register_buffer("lag_centers", torch.tensor(centers))
        self.register_buffer("lag_widths", torch.tensor(widths))
        self.register_buffer(
            "source_channel_indices", torch.tensor(source_indices, dtype=torch.long)
        )
        self.register_buffer(
            "target_channel_indices", torch.tensor(target_indices, dtype=torch.long)
        )

        self.attribute_standardizer = SourceRunningAttributeStandardizer(
            token_dim,
            momentum=statistics_momentum,
            min_scale=min_attribute_scale,
            eps=self.eps,
        )
        self.energy_scale = SourceRunningRelationEnergyScale(
            momentum=energy_scale_momentum,
            min_scale=min_energy_scale,
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(3 * self.num_lags, edge_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim, bias=False),
        )
        self.output_projection = nn.Linear(
            self.num_edges * edge_hidden_dim, structure_dim, bias=False
        )
        self.output_head = nn.Sequential(
            self.output_projection,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(structure_dim, eps=layer_norm_eps),
        )

    def _validate_component(self, component_tokens: Tensor) -> tuple[int, int]:
        if not isinstance(component_tokens, Tensor) or component_tokens.ndim != 4:
            raise ValueError(
                "component_tokens must be four-dimensional with shape [B, L, C, P]"
            )
        if not component_tokens.is_floating_point():
            raise ValueError("component_tokens must use a floating-point dtype")
        batch_size, sequence_length, channels, attributes = component_tokens.shape
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("component_tokens batch and sequence dimensions must be non-empty")
        if channels != self.num_channels:
            raise ValueError("component_tokens num_channels dimension is invalid")
        if attributes != self.token_dim:
            raise ValueError("component_tokens token_dim dimension is invalid")
        parameter = next(self.parameters())
        if parameter.device != component_tokens.device:
            raise ValueError("module and component_tokens must use the same device")
        if parameter.dtype != component_tokens.dtype:
            raise ValueError("module and component_tokens must use the same dtype")
        return batch_size, sequence_length

    def _resolve_channel_mask(
        self,
        time_mask: Tensor | None,
        channel_mask: Tensor | None,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Tensor:
        resolved_time = None
        if time_mask is not None:
            resolved_time = _strict_binary_mask(time_mask, "time_mask")
            if resolved_time.ndim == 1:
                if resolved_time.shape != (sequence_length,):
                    raise ValueError("time_mask must have shape [L] or [B, L]")
                resolved_time = resolved_time.unsqueeze(0).expand(batch_size, -1)
            elif resolved_time.ndim == 2:
                if resolved_time.shape != (batch_size, sequence_length):
                    raise ValueError("time_mask must have shape [L] or [B, L]")
            else:
                raise ValueError("time_mask must have shape [L] or [B, L]")
            resolved_time = resolved_time.to(device=device)

        resolved_channel = None
        if channel_mask is not None:
            resolved_channel = _strict_binary_mask(channel_mask, "channel_mask")
            if resolved_channel.ndim == 2:
                if resolved_channel.shape != (sequence_length, self.num_channels):
                    raise ValueError(
                        "channel_mask must have shape [L, C] or [B, L, C]"
                    )
                resolved_channel = resolved_channel.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
            elif resolved_channel.ndim == 3:
                if resolved_channel.shape != (
                    batch_size,
                    sequence_length,
                    self.num_channels,
                ):
                    raise ValueError(
                        "channel_mask must have shape [L, C] or [B, L, C]"
                    )
            else:
                raise ValueError(
                    "channel_mask must have shape [L, C] or [B, L, C]"
                )
            resolved_channel = resolved_channel.to(device=device)

        if resolved_channel is None:
            resolved_channel = torch.ones(
                batch_size,
                sequence_length,
                self.num_channels,
                dtype=torch.bool,
                device=device,
            )
        if resolved_time is not None:
            resolved_channel = resolved_channel & resolved_time.unsqueeze(-1)
        return resolved_channel

    def _resolve_positions(
        self,
        positions: Tensor,
        channel_mask: Tensor,
        reference: Tensor,
    ) -> Tensor:
        batch_size, sequence_length = channel_mask.shape[:2]
        if not isinstance(positions, Tensor):
            raise ValueError("positions must be a torch.Tensor")
        if positions.is_complex() or positions.dtype == torch.bool:
            raise ValueError("positions must contain real numeric values")
        if positions.ndim == 1:
            if positions.shape != (sequence_length,):
                raise ValueError("positions must have shape [L] or [B, L]")
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        elif positions.ndim == 2:
            if positions.shape != (batch_size, sequence_length):
                raise ValueError("positions must have shape [L] or [B, L]")
        else:
            raise ValueError("positions must have shape [L] or [B, L]")
        positions = positions.to(device=reference.device, dtype=reference.dtype)
        active_time = channel_mask.any(dim=-1)
        if not torch.isfinite(positions[active_time]).all().item():
            raise ValueError("valid positions must be finite")
        normalized = (positions - self.time_reference) / self.time_scale
        if active_time.any().item():
            active_values = normalized[active_time]
            if (
                (active_values < -self.eps).any().item()
                or (active_values > 1.0 + self.eps).any().item()
            ):
                raise ValueError("valid normalized positions must lie in [0, 1]")
        for batch in range(batch_size):
            values = normalized[batch, active_time[batch]]
            if values.numel() > 1 and not torch.all(values[1:] > values[:-1]).item():
                raise ValueError("valid positions must be strictly increasing")
        normalized = torch.where(
            active_time, normalized.clamp(0.0, 1.0), torch.zeros_like(normalized)
        )
        return normalized

    def _prepare_inputs(
        self,
        component_tokens: Tensor,
        positions: Tensor,
        time_mask: Tensor | None,
        channel_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, sequence_length = self._validate_component(component_tokens)
        mask = self._resolve_channel_mask(
            time_mask,
            channel_mask,
            batch_size,
            sequence_length,
            component_tokens.device,
        )
        normalized_positions = self._resolve_positions(
            positions, mask, component_tokens
        )
        if not torch.isfinite(component_tokens[mask]).all().item():
            raise ValueError("valid component token values must be finite")
        safe_tokens = torch.where(
            mask.unsqueeze(-1), component_tokens, torch.zeros_like(component_tokens)
        )
        return safe_tokens, normalized_positions, mask

    def compute_channel_coverage_weights(
        self, positions: Tensor, channel_mask: Tensor
    ) -> Tensor:
        if positions.ndim != 2 or channel_mask.ndim != 3:
            raise ValueError("positions and channel_mask must have shapes [B,L] and [B,L,C]")
        if positions.shape != channel_mask.shape[:2] or channel_mask.shape[2] != self.num_channels:
            raise ValueError("positions and channel_mask shapes are incompatible")
        weights = positions.new_zeros(channel_mask.shape)
        for batch in range(channel_mask.shape[0]):
            for channel in range(self.num_channels):
                indices = torch.nonzero(
                    channel_mask[batch, :, channel], as_tuple=False
                ).squeeze(-1)
                count = indices.numel()
                if count == 0:
                    continue
                if count == 1:
                    weights[batch, indices[0], channel] = 1.0
                    continue
                times = positions[batch, indices]
                raw = torch.empty_like(times)
                raw[0] = (times[1] - times[0]) / 2.0
                raw[-1] = (times[-1] - times[-2]) / 2.0
                if count > 2:
                    raw[1:-1] = (times[2:] - times[:-2]) / 2.0
                total = raw.sum()
                if total > self.eps:
                    weights[batch, indices, channel] = raw / total
                else:
                    weights[batch, indices, channel] = 1.0 / count
        return weights

    def compute_local_velocity(
        self,
        standardized_tokens: Tensor,
        normalized_positions: Tensor,
        channel_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if standardized_tokens.ndim != 4:
            raise ValueError("standardized_tokens must have shape [B,L,C,P]")
        if normalized_positions.shape != standardized_tokens.shape[:2]:
            raise ValueError("normalized_positions must have shape [B,L]")
        if channel_mask.shape != standardized_tokens.shape[:3]:
            raise ValueError("channel_mask must have shape [B,L,C]")
        tokens = torch.where(
            channel_mask.unsqueeze(-1),
            standardized_tokens,
            torch.zeros_like(standardized_tokens),
        )
        delta = normalized_positions.unsqueeze(1) - normalized_positions.unsqueeze(2)
        base_kernel = torch.exp(
            -0.5 * (delta / self.velocity_bandwidth).square()
        )
        velocities = []
        supports = []
        effective_counts = []
        spreads = []
        for channel in range(self.num_channels):
            valid = channel_mask[:, :, channel]
            weights = (
                base_kernel
                * valid.unsqueeze(-1).to(tokens.dtype)
                * valid.unsqueeze(1).to(tokens.dtype)
            )
            weight_sum = weights.sum(dim=-1)
            safe_sum = weight_sum.clamp_min(self.eps)
            mean_time = torch.einsum(
                "bij,bj->bi", weights, normalized_positions
            ) / safe_sum
            values = tokens[:, :, channel]
            mean_value = torch.einsum("bij,bjp->bip", weights, values) / safe_sum.unsqueeze(-1)
            centered_time = normalized_positions.unsqueeze(1) - mean_time.unsqueeze(-1)
            centered_value = values.unsqueeze(1) - mean_value.unsqueeze(2)
            time_numerator = (weights * centered_time.square()).sum(dim=-1)
            slope_numerator = torch.einsum(
                "bij,bij,bijp->bip", weights, centered_time, centered_value
            )
            sufficient_spread = time_numerator > self.eps
            velocity = slope_numerator / time_numerator.clamp_min(self.eps).unsqueeze(-1)
            velocity = torch.where(
                (valid & sufficient_spread).unsqueeze(-1),
                velocity,
                torch.zeros_like(velocity),
            )
            effective = weight_sum.square() / (
                weights.square().sum(dim=-1) + self.eps
            )
            spread = time_numerator / safe_sum
            count_quality = (
                effective / self.min_velocity_effective_count
            ).clamp(0.0, 1.0)
            spread_quality = spread / (
                spread + self.min_velocity_time_spread
            )
            support = count_quality * spread_quality
            support = torch.where(
                valid & sufficient_spread & torch.isfinite(support),
                support,
                torch.zeros_like(support),
            )
            velocity = torch.where(
                torch.isfinite(velocity), velocity, torch.zeros_like(velocity)
            )
            velocities.append(velocity)
            supports.append(support)
            effective_counts.append(
                torch.where(valid, effective, torch.zeros_like(effective))
            )
            spreads.append(torch.where(valid, spread, torch.zeros_like(spread)))
        return (
            torch.stack(velocities, dim=2),
            torch.stack(supports, dim=2),
            torch.stack(effective_counts, dim=2),
            torch.stack(spreads, dim=2),
        )

    @staticmethod
    def _scatter_edges(edge_values: Tensor, source: Tensor, target: Tensor, channels: int) -> Tensor:
        output = edge_values.new_zeros(
            edge_values.shape[0], channels, channels, edge_values.shape[-1]
        )
        output[:, source, target, :] = edge_values
        return output

    def _compute_relations(
        self,
        standardized_tokens: Tensor,
        normalized_positions: Tensor,
        channel_mask: Tensor,
    ) -> _RelationComputation:
        coverage = self.compute_channel_coverage_weights(
            normalized_positions, channel_mask
        )
        channel_mean = torch.einsum(
            "blc,blcp->bcp", coverage, standardized_tokens
        )
        centered = standardized_tokens - channel_mean.unsqueeze(1)
        centered = torch.where(
            channel_mask.unsqueeze(-1), centered, torch.zeros_like(centered)
        )
        velocity, velocity_support, _, _ = self.compute_local_velocity(
            standardized_tokens, normalized_positions, channel_mask
        )

        delta = normalized_positions.unsqueeze(1) - normalized_positions.unsqueeze(2)
        lag_kernel = torch.exp(
            -0.5
            * (
                (delta.unsqueeze(1) - self.lag_centers.view(1, -1, 1, 1))
                / self.lag_widths.view(1, -1, 1, 1)
            ).square()
        )
        state_relations = []
        evolution_relations = []
        strengths = []
        state_counts = []
        evolution_counts = []
        state_energy_as = []
        state_energy_bs = []
        evolution_energy_as = []
        evolution_energy_bs = []
        for source, target in self._directed_pairs:
            source_mask = channel_mask[:, :, source].to(centered.dtype)
            target_mask = channel_mask[:, :, target].to(centered.dtype)
            pair_weight = (
                lag_kernel
                * source_mask[:, None, :, None]
                * target_mask[:, None, None, :]
            )
            source_state = centered[:, :, source]
            target_state = centered[:, :, target]
            state_numerator = torch.einsum(
                "bgij,bip,bjp->bg",
                pair_weight,
                source_state,
                target_state,
            )
            state_energy_a = torch.einsum(
                "bgij,bi->bg", pair_weight, source_state.square().sum(dim=-1)
            )
            state_energy_b = torch.einsum(
                "bgij,bj->bg", pair_weight, target_state.square().sum(dim=-1)
            )
            state_count = pair_weight.sum(dim=(-1, -2)).square() / (
                pair_weight.square().sum(dim=(-1, -2)) + self.eps
            )
            state_relation = state_numerator / torch.sqrt(
                state_energy_a * state_energy_b + self.eps
            )
            state_relation = torch.where(
                torch.isfinite(state_relation),
                state_relation.clamp(-1.0, 1.0),
                torch.zeros_like(state_relation),
            )

            source_support = velocity_support[:, :, source]
            target_support = velocity_support[:, :, target]
            evolution_weight = (
                pair_weight
                * source_support[:, None, :, None]
                * target_support[:, None, None, :]
            )
            source_velocity = velocity[:, :, source]
            target_velocity = velocity[:, :, target]
            evolution_numerator = torch.einsum(
                "bgij,bip,bjp->bg",
                evolution_weight,
                source_velocity,
                target_velocity,
            )
            evolution_energy_a = torch.einsum(
                "bgij,bi->bg",
                evolution_weight,
                source_velocity.square().sum(dim=-1),
            )
            evolution_energy_b = torch.einsum(
                "bgij,bj->bg",
                evolution_weight,
                target_velocity.square().sum(dim=-1),
            )
            evolution_count = evolution_weight.sum(dim=(-1, -2)).square() / (
                evolution_weight.square().sum(dim=(-1, -2)) + self.eps
            )
            evolution_relation = evolution_numerator / torch.sqrt(
                evolution_energy_a * evolution_energy_b + self.eps
            )
            evolution_relation = torch.where(
                torch.isfinite(evolution_relation),
                evolution_relation.clamp(-1.0, 1.0),
                torch.zeros_like(evolution_relation),
            )
            reliable_energy = (evolution_energy_a > self.eps) & (
                evolution_energy_b > self.eps
            )
            strength = torch.tanh(
                0.5
                * torch.log(
                    (evolution_energy_a + self.eps)
                    / (evolution_energy_b + self.eps)
                )
            )
            strength = torch.where(
                reliable_energy & torch.isfinite(strength),
                strength,
                torch.zeros_like(strength),
            )
            state_relations.append(state_relation)
            evolution_relations.append(evolution_relation)
            strengths.append(strength)
            state_counts.append(state_count)
            evolution_counts.append(evolution_count)
            state_energy_as.append(state_energy_a)
            state_energy_bs.append(state_energy_b)
            evolution_energy_as.append(evolution_energy_a)
            evolution_energy_bs.append(evolution_energy_b)

        state_relation_edges = torch.stack(state_relations, dim=1)
        evolution_relation_edges = torch.stack(evolution_relations, dim=1)
        strength_edges = torch.stack(strengths, dim=1)
        state_count_edges = torch.stack(state_counts, dim=1)
        evolution_count_edges = torch.stack(evolution_counts, dim=1)
        state_energy_a_edges = torch.stack(state_energy_as, dim=1)
        state_energy_b_edges = torch.stack(state_energy_bs, dim=1)
        evolution_energy_a_edges = torch.stack(evolution_energy_as, dim=1)
        evolution_energy_b_edges = torch.stack(evolution_energy_bs, dim=1)

        tau_state, tau_evolution = self.energy_scale(
            device=standardized_tokens.device, dtype=standardized_tokens.dtype
        )
        state_reliability_edges = (
            (state_count_edges / self.min_effective_pairs).clamp(0.0, 1.0)
            * state_energy_a_edges / (state_energy_a_edges + tau_state)
            * state_energy_b_edges / (state_energy_b_edges + tau_state)
        ).clamp(0.0, 1.0)
        evolution_reliability_edges = (
            (evolution_count_edges / self.min_effective_pairs).clamp(0.0, 1.0)
            * evolution_energy_a_edges
            / (evolution_energy_a_edges + tau_evolution)
            * evolution_energy_b_edges
            / (evolution_energy_b_edges + tau_evolution)
        ).clamp(0.0, 1.0)

        source = self.source_channel_indices
        target = self.target_channel_indices
        return _RelationComputation(
            state_relation=self._scatter_edges(
                state_relation_edges, source, target, self.num_channels
            ),
            evolution_relation=self._scatter_edges(
                evolution_relation_edges, source, target, self.num_channels
            ),
            relative_strength=self._scatter_edges(
                strength_edges, source, target, self.num_channels
            ),
            state_reliability=self._scatter_edges(
                state_reliability_edges, source, target, self.num_channels
            ),
            evolution_reliability=self._scatter_edges(
                evolution_reliability_edges, source, target, self.num_channels
            ),
            local_velocity=velocity,
            velocity_support=velocity_support,
            state_effective_pair_count=self._scatter_edges(
                state_count_edges, source, target, self.num_channels
            ),
            evolution_effective_pair_count=self._scatter_edges(
                evolution_count_edges, source, target, self.num_channels
            ),
            state_energy_a=state_energy_a_edges,
            state_energy_b=state_energy_b_edges,
            evolution_energy_a=evolution_energy_a_edges,
            evolution_energy_b=evolution_energy_b_edges,
        )

    @torch.no_grad()
    def update_source_state(
        self,
        component_tokens: Tensor,
        positions: Tensor,
        time_mask: Tensor | None = None,
        channel_mask: Tensor | None = None,
    ) -> None:
        safe_tokens, normalized_positions, mask = self._prepare_inputs(
            component_tokens, positions, time_mask, channel_mask
        )
        self.attribute_standardizer.update(safe_tokens, mask)
        standardized = self.attribute_standardizer(safe_tokens)
        standardized = torch.where(
            mask.unsqueeze(-1), standardized, torch.zeros_like(standardized)
        )
        relations = self._compute_relations(
            standardized, normalized_positions, mask
        )
        state_energy = torch.cat(
            [relations.state_energy_a, relations.state_energy_b], dim=-1
        )
        evolution_energy = torch.cat(
            [relations.evolution_energy_a, relations.evolution_energy_b], dim=-1
        )
        state_valid = torch.cat(
            [
                relations.state_effective_pair_count[
                    :, self.source_channel_indices, self.target_channel_indices
                ],
                relations.state_effective_pair_count[
                    :, self.source_channel_indices, self.target_channel_indices
                ],
            ],
            dim=-1,
        ) >= self.min_effective_pairs
        evolution_valid = torch.cat(
            [
                relations.evolution_effective_pair_count[
                    :, self.source_channel_indices, self.target_channel_indices
                ],
                relations.evolution_effective_pair_count[
                    :, self.source_channel_indices, self.target_channel_indices
                ],
            ],
            dim=-1,
        ) >= self.min_effective_pairs
        self.energy_scale.update(
            state_energy,
            evolution_energy,
            state_valid,
            evolution_valid,
        )

    def forward(
        self,
        component_tokens: Tensor,
        positions: Tensor,
        time_mask: Tensor | None = None,
        channel_mask: Tensor | None = None,
    ) -> ChannelStructureOutput:
        safe_tokens, normalized_positions, mask = self._prepare_inputs(
            component_tokens, positions, time_mask, channel_mask
        )
        standardized = self.attribute_standardizer(safe_tokens)
        standardized = torch.where(
            mask.unsqueeze(-1), standardized, torch.zeros_like(standardized)
        )
        relations = self._compute_relations(
            standardized, normalized_positions, mask
        )
        reliable_state = relations.state_reliability * relations.state_relation
        reliable_evolution = (
            relations.evolution_reliability * relations.evolution_relation
        )
        reliable_strength = (
            relations.evolution_reliability * relations.relative_strength
        )
        edge_raw_matrix = torch.cat(
            [reliable_state, reliable_evolution, reliable_strength], dim=-1
        )
        reliable_edge_raw = edge_raw_matrix[
            :, self.source_channel_indices, self.target_channel_indices
        ]
        edge_embedding = self.edge_encoder(reliable_edge_raw)
        feature = self.output_head(edge_embedding.flatten(start_dim=1))
        relation_mass = relations.state_reliability.sum(dim=(1, 2, 3)) + (
            relations.evolution_reliability.sum(dim=(1, 2, 3))
        )
        valid = relation_mass >= self.min_relation_mass
        feature = torch.where(
            valid.unsqueeze(-1), feature, torch.zeros_like(feature)
        )
        return ChannelStructureOutput(
            feature=feature,
            valid=valid,
            state_relation=relations.state_relation,
            evolution_relation=relations.evolution_relation,
            relative_strength=relations.relative_strength,
            state_reliability=relations.state_reliability,
            evolution_reliability=relations.evolution_reliability,
            reliable_edge_raw=reliable_edge_raw,
            edge_embedding=edge_embedding,
            local_velocity=relations.local_velocity,
            velocity_support=relations.velocity_support,
            state_effective_pair_count=relations.state_effective_pair_count,
            evolution_effective_pair_count=relations.evolution_effective_pair_count,
            relation_mass=relation_mass,
        )


class SharedChannelStructureOperator(nn.Module):
    """Apply one global channel extractor to trend and dynamics only."""

    def __init__(self, extractor: MultiScaleChannelRelationStructure) -> None:
        super().__init__()
        if not isinstance(extractor, MultiScaleChannelRelationStructure):
            raise ValueError(
                "extractor must be a MultiScaleChannelRelationStructure"
            )
        self.extractor = extractor

    @staticmethod
    def _validate_pair(trend: Tensor, dynamics: Tensor) -> None:
        if not isinstance(trend, Tensor) or not isinstance(dynamics, Tensor):
            raise ValueError("trend and dynamics must be tensors")
        if trend.shape != dynamics.shape:
            raise ValueError("trend and dynamics must have identical shape")
        if trend.dtype != dynamics.dtype:
            raise ValueError("trend and dynamics must have identical dtype")
        if trend.device != dynamics.device:
            raise ValueError("trend and dynamics must have identical device")

    @staticmethod
    def _duplicate_batch(value: Tensor | None, batch_ndim: int) -> Tensor | None:
        if value is None:
            return None
        if not isinstance(value, Tensor):
            raise ValueError("positions and masks must be tensors")
        if value.ndim == batch_ndim:
            return torch.cat([value, value], dim=0)
        return value

    def forward(
        self,
        trend: Tensor,
        dynamics: Tensor,
        positions: Tensor,
        time_mask: Tensor | None = None,
        channel_mask: Tensor | None = None,
    ) -> ChannelStructurePairOutput:
        self._validate_pair(trend, dynamics)
        return ChannelStructurePairOutput(
            trend=self.extractor(
                trend, positions, time_mask=time_mask, channel_mask=channel_mask
            ),
            dynamics=self.extractor(
                dynamics, positions, time_mask=time_mask, channel_mask=channel_mask
            ),
        )

    @torch.no_grad()
    def update_source_state(
        self,
        trend: Tensor,
        dynamics: Tensor,
        positions: Tensor,
        time_mask: Tensor | None = None,
        channel_mask: Tensor | None = None,
    ) -> None:
        self._validate_pair(trend, dynamics)
        combined = torch.cat([trend, dynamics], dim=0)
        combined_positions = self._duplicate_batch(positions, 2)
        combined_time_mask = self._duplicate_batch(time_mask, 2)
        combined_channel_mask = self._duplicate_batch(channel_mask, 3)
        self.extractor.update_source_state(
            combined,
            combined_positions,
            time_mask=combined_time_mask,
            channel_mask=combined_channel_mask,
        )
