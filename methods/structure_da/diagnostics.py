"""Pure, detached diagnostics for Structure DA training outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class DiagnosticStat:
    """Composable sum and valid-count sufficient statistic."""

    total: Tensor
    count: Tensor


@dataclass(frozen=True)
class DiagnosticMoments:
    """Composable population moments without averaging batch std values."""

    sum: Tensor
    sum_sq: Tensor
    count: Tensor


@dataclass(frozen=True)
class DecompositionDiagnostics:
    energies: Mapping[str, DiagnosticStat]
    centered_energies: Mapping[str, DiagnosticStat]
    cross_terms: Mapping[str, DiagnosticStat]
    reconstruction_energy: DiagnosticStat
    cosines: Mapping[str, DiagnosticStat]
    roughness: Mapping[str, DiagnosticStat]
    normalized_roughness: Mapping[str, DiagnosticStat]
    sample_count: Tensor
    eps: float


@dataclass(frozen=True)
class ContributionDiagnostics:
    gates: Mapping[str, DiagnosticMoments]
    effective_norms: Mapping[str, DiagnosticStat]
    fusion_shares: Mapping[str, DiagnosticStat]
    sample_count: Tensor
    eps: float


def _zero(device: torch.device, *, dtype: torch.dtype = torch.float64) -> Tensor:
    return torch.zeros((), device=device, dtype=dtype)


def _safe_mean(stat: DiagnosticStat) -> Tensor:
    return torch.where(
        stat.count > 0,
        stat.total / stat.count.clamp_min(1),
        torch.zeros_like(stat.total),
    )


def _safe_rate(count: Tensor, total: Tensor) -> Tensor:
    return torch.where(
        total > 0,
        count.to(torch.float64) / total.to(torch.float64),
        torch.zeros((), dtype=torch.float64, device=count.device),
    )


def _merge_stat(left: DiagnosticStat, right: DiagnosticStat) -> DiagnosticStat:
    return DiagnosticStat(left.total + right.total, left.count + right.count)


def _merge_moments(
    left: DiagnosticMoments, right: DiagnosticMoments
) -> DiagnosticMoments:
    return DiagnosticMoments(
        left.sum + right.sum,
        left.sum_sq + right.sum_sq,
        left.count + right.count,
    )


def _merge_mapping(left, right, merge):
    if left.keys() != right.keys():
        raise ValueError("diagnostic mappings must have identical keys")
    return {name: merge(left[name], right[name]) for name in left}


def _validate_decomposition_inputs(
    tensors: Mapping[str, Tensor], timestamps: Tensor, mask: Tensor
) -> tuple[int, int]:
    first = tensors["H"]
    if not isinstance(first, Tensor) or first.ndim < 3:
        raise ValueError("H/T/D/R must have shape [B, L, ...]")
    batch_size, sequence_length = first.shape[:2]
    for name, value in tensors.items():
        if not isinstance(value, Tensor) or value.shape != first.shape:
            raise ValueError(f"{name} must match H shape")
        if not value.is_floating_point() or value.is_complex():
            raise ValueError(f"{name} must use a real floating-point dtype")
        if value.device != first.device:
            raise ValueError(f"{name} must match H device")
    if (
        not isinstance(timestamps, Tensor)
        or timestamps.shape != (batch_size, sequence_length)
        or timestamps.dtype == torch.bool
        or timestamps.is_complex()
        or timestamps.device != first.device
    ):
        raise ValueError("timestamps must be real numeric with shape [B, L]")
    if (
        not isinstance(mask, Tensor)
        or mask.shape != (batch_size, sequence_length)
        or mask.dtype != torch.bool
        or mask.device != first.device
    ):
        raise ValueError("mask must be boolean with shape [B, L]")
    expanded_mask = mask.reshape(batch_size, sequence_length, *([1] * (first.ndim - 2)))
    expanded_mask = expanded_mask.expand_as(first)
    for name, value in tensors.items():
        if not torch.isfinite(value.detach()).masked_select(expanded_mask).all().item():
            raise ValueError(f"{name} must be finite at valid observations")
    return batch_size, sequence_length


def _energy_stat(value: Tensor, expanded_mask: Tensor) -> DiagnosticStat:
    selected = value.masked_select(expanded_mask)
    return DiagnosticStat(selected.square().sum(), expanded_mask.sum())


def _cross_stat(
    left: Tensor, right: Tensor, expanded_mask: Tensor
) -> DiagnosticStat:
    selected = (left * right).masked_select(expanded_mask)
    return DiagnosticStat(selected.sum(), expanded_mask.sum())


def _centered(
    value: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size, sequence_length = value.shape[:2]
    feature_shape = value.shape[2:]
    time_mask = mask.reshape(batch_size, sequence_length, *([1] * len(feature_shape)))
    safe = torch.where(time_mask, value, torch.zeros_like(value))
    valid_times = mask.sum(dim=1).reshape(batch_size, *([1] * len(feature_shape)))
    mean = safe.sum(dim=1) / valid_times.clamp_min(1)
    centered = torch.where(time_mask, value - mean.unsqueeze(1), torch.zeros_like(value))
    expanded_mask = time_mask.expand_as(value)
    per_sample_count = (
        mask.sum(dim=1).to(torch.float64) * math.prod(feature_shape)
    )
    per_sample_energy = centered.square().sum(
        dim=tuple(range(1, centered.ndim))
    ) / per_sample_count.clamp_min(1)
    return centered, expanded_mask, per_sample_energy


def _per_sample_energy(value: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    batch_size = value.shape[0]
    feature_count = math.prod(value.shape[2:])
    expanded = mask.reshape(batch_size, mask.shape[1], *([1] * (value.ndim - 2)))
    safe = torch.where(expanded, value, torch.zeros_like(value))
    count = mask.sum(dim=1).to(torch.float64) * feature_count
    energy = safe.square().sum(dim=tuple(range(1, value.ndim))) / count.clamp_min(1)
    return energy, count > 0


def _cosine_stat(
    left: Tensor, right: Tensor, mask: Tensor, eps: float
) -> DiagnosticStat:
    left_energy, observation_valid = _per_sample_energy(left, mask)
    right_energy, _ = _per_sample_energy(right, mask)
    batch_size = left.shape[0]
    feature_count = math.prod(left.shape[2:])
    expanded = mask.reshape(batch_size, mask.shape[1], *([1] * (left.ndim - 2)))
    cross = torch.where(expanded, left * right, torch.zeros_like(left)).sum(
        dim=tuple(range(1, left.ndim))
    ) / (mask.sum(dim=1).to(torch.float64) * feature_count).clamp_min(1)
    valid = observation_valid & (left_energy > eps) & (right_energy > eps)
    cosine = cross / torch.sqrt(left_energy * right_energy).clamp_min(eps)
    return DiagnosticStat(cosine.masked_select(valid).sum(), valid.sum())


def _roughness_stats(
    value: Tensor,
    centered_energy: Tensor,
    timestamps: Tensor,
    mask: Tensor,
    eps: float,
) -> tuple[DiagnosticStat, DiagnosticStat]:
    delta_time = timestamps[:, 1:] - timestamps[:, :-1]
    interval_valid = (
        mask[:, 1:]
        & mask[:, :-1]
        & torch.isfinite(timestamps[:, 1:])
        & torch.isfinite(timestamps[:, :-1])
        & (delta_time > 0)
    )
    safe_delta_time = torch.where(
        interval_valid, delta_time, torch.ones_like(delta_time)
    )
    difference = value[:, 1:] - value[:, :-1]
    mean_square_difference = difference.square().mean(
        dim=tuple(range(2, difference.ndim))
    )
    numerator = torch.where(
        interval_valid,
        mean_square_difference / safe_delta_time,
        torch.zeros_like(mean_square_difference),
    ).sum(dim=1)
    duration = torch.where(
        interval_valid, delta_time, torch.zeros_like(delta_time)
    ).sum(dim=1)
    valid = interval_valid.any(dim=1)
    roughness = numerator / (duration + eps)
    roughness_stat = DiagnosticStat(
        roughness.masked_select(valid).sum(), valid.sum()
    )
    normalized_valid = valid & (centered_energy > eps)
    normalized = roughness / (centered_energy + eps)
    normalized_stat = DiagnosticStat(
        normalized.masked_select(normalized_valid).sum(), normalized_valid.sum()
    )
    return roughness_stat, normalized_stat


@torch.no_grad()
def compute_decomposition_diagnostics(
    H: Tensor,
    T: Tensor,
    D: Tensor,
    R: Tensor,
    timestamps: Tensor,
    mask: Tensor,
    *,
    eps: float = 1e-8,
) -> DecompositionDiagnostics:
    """Build composable diagnostics from already-computed decomposition tensors."""

    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    with torch.no_grad():
        tensors = {name: value.detach() for name, value in {"H": H, "T": T, "D": D, "R": R}.items()}
        timestamps = timestamps.detach()
        mask = mask.detach()
        batch_size, sequence_length = _validate_decomposition_inputs(
            tensors, timestamps, mask
        )
        tensors = {name: value.to(torch.float64) for name, value in tensors.items()}
        timestamps = timestamps.to(torch.float64)
        expanded_mask = mask.reshape(
            batch_size, sequence_length, *([1] * (H.ndim - 2))
        ).expand_as(tensors["H"])
        energies = {
            name: _energy_stat(value, expanded_mask)
            for name, value in tensors.items()
        }
        centered = {}
        centered_per_sample = {}
        centered_energies = {}
        for name, value in tensors.items():
            centered_value, component_mask, sample_energy = _centered(value, mask)
            centered[name] = centered_value
            centered_per_sample[name] = sample_energy
            centered_energies[name] = _energy_stat(centered_value, component_mask)
        cross_terms = {
            "TD": _cross_stat(tensors["T"], tensors["D"], expanded_mask),
            "TR": _cross_stat(tensors["T"], tensors["R"], expanded_mask),
            "DR": _cross_stat(tensors["D"], tensors["R"], expanded_mask),
        }
        reconstruction = tensors["H"] - tensors["T"] - tensors["D"] - tensors["R"]
        cosines = {
            "TD": _cosine_stat(tensors["T"], tensors["D"], mask, eps),
            "TR": _cosine_stat(tensors["T"], tensors["R"], mask, eps),
            "DR": _cosine_stat(tensors["D"], tensors["R"], mask, eps),
        }
        roughness = {}
        normalized_roughness = {}
        for name, value in tensors.items():
            roughness[name], normalized_roughness[name] = _roughness_stats(
                value,
                centered_per_sample[name],
                timestamps,
                mask,
                eps,
            )
        return DecompositionDiagnostics(
            energies=energies,
            centered_energies=centered_energies,
            cross_terms=cross_terms,
            reconstruction_energy=_energy_stat(reconstruction, expanded_mask),
            cosines=cosines,
            roughness=roughness,
            normalized_roughness=normalized_roughness,
            sample_count=torch.tensor(batch_size, device=H.device, dtype=torch.long),
            eps=float(eps),
        )


@torch.no_grad()
def merge_decomposition_diagnostics(
    left: DecompositionDiagnostics, right: DecompositionDiagnostics
) -> DecompositionDiagnostics:
    if left.eps != right.eps:
        raise ValueError("diagnostic eps values must match")
    with torch.no_grad():
        return DecompositionDiagnostics(
            energies=_merge_mapping(left.energies, right.energies, _merge_stat),
            centered_energies=_merge_mapping(
                left.centered_energies, right.centered_energies, _merge_stat
            ),
            cross_terms=_merge_mapping(left.cross_terms, right.cross_terms, _merge_stat),
            reconstruction_energy=_merge_stat(
                left.reconstruction_energy, right.reconstruction_energy
            ),
            cosines=_merge_mapping(left.cosines, right.cosines, _merge_stat),
            roughness=_merge_mapping(left.roughness, right.roughness, _merge_stat),
            normalized_roughness=_merge_mapping(
                left.normalized_roughness, right.normalized_roughness, _merge_stat
            ),
            sample_count=left.sample_count + right.sample_count,
            eps=left.eps,
        )


@torch.no_grad()
def summarize_decomposition_diagnostics(
    diagnostics: DecompositionDiagnostics,
) -> dict[str, Tensor]:
    with torch.no_grad():
        eps = diagnostics.eps
        energy = {name: _safe_mean(stat) for name, stat in diagnostics.energies.items()}
        rms = {name: torch.sqrt(value.clamp_min(0)) for name, value in energy.items()}
        centered = {
            name: _safe_mean(stat)
            for name, stat in diagnostics.centered_energies.items()
        }
        result = {f"rms_{name}": value for name, value in rms.items()}
        for numerator, denominator in (("D", "H"), ("R", "H"), ("D", "T"), ("R", "T")):
            name = f"{numerator}_over_{denominator}"
            valid = rms[denominator] > eps
            result[name] = torch.where(
                valid,
                rms[numerator] / rms[denominator].clamp_min(eps),
                torch.zeros_like(rms[numerator]),
            )
            result[f"{name}_valid_rate"] = valid.to(torch.float64)
        result.update({f"centered_energy_{name}": value for name, value in centered.items()})
        component_centered_total = centered["T"] + centered["D"] + centered["R"]
        for name in ("T", "D", "R"):
            result[f"centered_fraction_{name}"] = torch.where(
                component_centered_total > eps,
                centered[name] / component_centered_total.clamp_min(eps),
                torch.zeros_like(centered[name]),
            )
        reconstruction_rms = torch.sqrt(
            _safe_mean(diagnostics.reconstruction_energy).clamp_min(0)
        )
        result["reconstruction_relative_error"] = torch.where(
            rms["H"] > eps,
            reconstruction_rms / rms["H"].clamp_min(eps),
            torch.zeros_like(reconstruction_rms),
        )
        cross = {
            name: _safe_mean(stat) for name, stat in diagnostics.cross_terms.items()
        }
        rhs = (
            energy["T"] + energy["D"] + energy["R"]
            + 2.0 * cross["TD"] + 2.0 * cross["TR"] + 2.0 * cross["DR"]
        )
        result["energy_closure_relative_error"] = torch.where(
            energy["H"] > eps,
            (energy["H"] - rhs).abs() / energy["H"].clamp_min(eps),
            torch.zeros_like(energy["H"]),
        )
        for name, stat in diagnostics.cosines.items():
            result[f"cos_{name}"] = _safe_mean(stat)
            result[f"cos_{name}_valid_rate"] = _safe_rate(
                stat.count, diagnostics.sample_count
            )
        for name, stat in diagnostics.roughness.items():
            result[f"roughness_{name}"] = _safe_mean(stat)
            result[f"roughness_valid_rate_{name}"] = _safe_rate(
                stat.count, diagnostics.sample_count
            )
        for name, stat in diagnostics.normalized_roughness.items():
            result[f"normalized_roughness_{name}"] = _safe_mean(stat)
            result[f"normalized_roughness_valid_rate_{name}"] = _safe_rate(
                stat.count, diagnostics.sample_count
            )
        return result


def _validate_contribution_inputs(values: Mapping[str, Tensor]) -> int:
    coefficients = (
        "alpha_T", "alpha_D", "alpha_R", "beta_T_temporal",
        "beta_D_temporal",
    )
    first = values["alpha_T"]
    if not isinstance(first, Tensor) or first.ndim != 1:
        raise ValueError("quality coefficients must have shape [B]")
    batch_size = first.shape[0]
    for name in coefficients:
        value = values[name]
        if not isinstance(value, Tensor) or value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [B]")
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain finite floating values")
    for name in (
        "temporal_T", "temporal_D", "raw_fusion", "temporal_fusion",
    ):
        value = values[name]
        if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [B, D]")
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain finite floating values")
    for name in (
        "temporal_T_valid", "temporal_D_valid"
    ):
        value = values[name]
        if not isinstance(value, Tensor) or value.dtype != torch.bool or value.shape != (batch_size,):
            raise ValueError(f"{name} must be boolean with shape [B]")
    devices = {value.device for value in values.values()}
    if len(devices) != 1:
        raise ValueError("all contribution inputs must share a device")
    return batch_size


def _moments(value: Tensor) -> DiagnosticMoments:
    return DiagnosticMoments(value.sum(), value.square().sum(), value.new_tensor(value.numel(), dtype=torch.long))


@torch.no_grad()
def compute_structure_contribution_diagnostics(
    *,
    alpha_T: Tensor,
    alpha_D: Tensor,
    alpha_R: Tensor,
    beta_T_temporal: Tensor,
    beta_D_temporal: Tensor,
    temporal_T: Tensor,
    temporal_D: Tensor,
    raw_fusion: Tensor,
    temporal_fusion: Tensor,
    temporal_T_valid: Tensor,
    temporal_D_valid: Tensor,
    eps: float = 1e-8,
) -> ContributionDiagnostics:
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    with torch.no_grad():
        values = {
            name: value.detach()
            for name, value in locals().items()
            if isinstance(value, Tensor)
        }
        batch_size = _validate_contribution_inputs(values)
        floating = {
            name: value.to(torch.float64)
            for name, value in values.items()
            if value.is_floating_point()
        }
        gates = {
            "T_temporal": floating["alpha_T"] * floating["beta_T_temporal"],
            "D_temporal": floating["alpha_D"] * floating["beta_D_temporal"],
        }
        vectors = {
            "T_temporal": floating["temporal_T"],
            "D_temporal": floating["temporal_D"],
        }
        validity = {
            "T_temporal": values["temporal_T_valid"],
            "D_temporal": values["temporal_D_valid"],
        }
        effective_norms = {}
        for name in gates:
            norm = torch.linalg.vector_norm(
                gates[name].unsqueeze(-1) * vectors[name], dim=-1
            )
            valid = validity[name]
            effective_norms[name] = DiagnosticStat(
                norm.masked_select(valid).sum(), valid.sum()
            )
        fusion_energy = {
            name: floating[f"{name}_fusion"].square().sum(dim=-1)
            for name in ("raw", "temporal")
        }
        fusion_total = sum(fusion_energy.values())
        fusion_valid = fusion_total > eps
        fusion_shares = {
            name: DiagnosticStat(
                (energy / fusion_total.clamp_min(eps)).masked_select(fusion_valid).sum(),
                fusion_valid.sum(),
            )
            for name, energy in fusion_energy.items()
        }
        return ContributionDiagnostics(
            gates={name: _moments(value) for name, value in gates.items()},
            effective_norms=effective_norms,
            fusion_shares=fusion_shares,
            sample_count=torch.tensor(batch_size, device=alpha_T.device, dtype=torch.long),
            eps=float(eps),
        )


@torch.no_grad()
def merge_contribution_diagnostics(
    left: ContributionDiagnostics, right: ContributionDiagnostics
) -> ContributionDiagnostics:
    if left.eps != right.eps:
        raise ValueError("diagnostic eps values must match")
    with torch.no_grad():
        return ContributionDiagnostics(
            gates=_merge_mapping(left.gates, right.gates, _merge_moments),
            effective_norms=_merge_mapping(
                left.effective_norms, right.effective_norms, _merge_stat
            ),
            fusion_shares=_merge_mapping(
                left.fusion_shares, right.fusion_shares, _merge_stat
            ),
            sample_count=left.sample_count + right.sample_count,
            eps=left.eps,
        )


@torch.no_grad()
def summarize_contribution_diagnostics(
    diagnostics: ContributionDiagnostics,
) -> dict[str, Tensor]:
    with torch.no_grad():
        result = {}
        for name, moments in diagnostics.gates.items():
            mean = torch.where(
                moments.count > 0,
                moments.sum / moments.count.clamp_min(1),
                torch.zeros_like(moments.sum),
            )
            variance = torch.where(
                moments.count > 0,
                moments.sum_sq / moments.count.clamp_min(1) - mean.square(),
                torch.zeros_like(moments.sum),
            ).clamp_min(0)
            result[f"gate_{name}_mean"] = mean
            result[f"gate_{name}_std"] = torch.sqrt(variance)
            result[f"gate_{name}_count"] = moments.count.to(torch.float64)
        for name, stat in diagnostics.effective_norms.items():
            result[f"effective_{name}_norm"] = _safe_mean(stat)
            result[f"effective_{name}_norm_valid_rate"] = _safe_rate(
                stat.count, diagnostics.sample_count
            )
        for name, stat in diagnostics.fusion_shares.items():
            result[f"fusion_share_{name}"] = _safe_mean(stat)
        representative = next(iter(diagnostics.fusion_shares.values()))
        result["fusion_share_valid_rate"] = _safe_rate(
            representative.count, diagnostics.sample_count
        )
        return result


__all__ = [
    "ContributionDiagnostics",
    "DecompositionDiagnostics",
    "DiagnosticMoments",
    "DiagnosticStat",
    "compute_decomposition_diagnostics",
    "compute_structure_contribution_diagnostics",
    "merge_contribution_diagnostics",
    "merge_decomposition_diagnostics",
    "summarize_contribution_diagnostics",
    "summarize_decomposition_diagnostics",
]
