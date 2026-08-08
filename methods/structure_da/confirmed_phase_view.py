"""Domain-Phase calibrated target views produced by frozen geometry and EMA teacher."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .domain_phase_state import DomainPhaseState, PhaseGroup, PhaseGroupStatus
from .phase_registration import resample_gamma, warp_q_gamma, warp_support_gamma
from .temporal_registration import invert_monotone_warp


IDENTITY_PHASE_GROUP_ID = -1


@dataclass(frozen=True)
class ConfirmedPhaseView:
    sample_ids: Tensor
    group_id: int
    member_classes: tuple[int, ...]
    center_gamma: Tensor
    aligned_positions: Tensor
    logits: Tensor
    probabilities: Tensor
    fused_repr: Tensor
    trend_repr: Tensor
    structure_repr: Tensor
    aligned_q_shape: Tensor
    aligned_q_support: Tensor
    q_valid: Tensor


def build_confirmed_class_to_group_map(
    phase_state: DomainPhaseState,
) -> dict[int, PhaseGroup]:
    if phase_state.m == 0:
        return {}
    mapping: dict[int, PhaseGroup] = {}
    for group in phase_state.groups:
        if group.status is not PhaseGroupStatus.CONFIRMED:
            continue
        for class_id in group.member_classes:
            if class_id in mapping:
                raise ValueError(
                    f"class {class_id} belongs to more than one confirmed group"
                )
            mapping[class_id] = group
    return mapping


def align_target_positions_to_source(
    positions: Tensor,
    mask: Tensor,
    center_gamma: Tensor,
) -> Tensor:
    """Map target coordinates through ``center_gamma^{-1}`` on valid dates."""
    if not isinstance(positions, Tensor) or positions.ndim != 2:
        raise ValueError("positions must have shape [B,L]")
    if not positions.is_floating_point():
        raise ValueError("positions must use a floating-point dtype")
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != positions.shape:
        raise ValueError("mask must be a boolean tensor with shape [B,L]")
    valid_positions = positions[mask]
    if not torch.isfinite(valid_positions).all().item():
        raise ValueError("valid positions must be finite")
    tolerance = 1e-6
    if valid_positions.numel() and (
        torch.any(valid_positions < -tolerance).item()
        or torch.any(valid_positions > 1.0 + tolerance).item()
    ):
        raise ValueError("valid positions must lie in [0,1]")
    safe_positions = torch.where(mask, positions.clamp(0.0, 1.0), torch.zeros_like(positions))
    gamma = center_gamma.detach().to(device=positions.device, dtype=positions.dtype)
    gamma = gamma.unsqueeze(0).expand(positions.shape[0], -1)
    aligned = invert_monotone_warp(gamma, safe_positions)
    aligned = torch.where(mask, aligned, torch.zeros_like(aligned))
    for sample_index in range(positions.shape[0]):
        valid = aligned[sample_index, mask[sample_index]]
        if valid.numel() > 1 and not torch.all(valid[1:] > valid[:-1]).item():
            raise ValueError("aligned valid positions must remain strictly increasing")
    return aligned.to(device=positions.device, dtype=positions.dtype).detach()


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _batch_tensor(batch: dict, name: str, device: torch.device):
    value = batch.get(name)
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise ValueError(f"batch[{name!r}] must be a tensor")
    return value.to(device=device)


@torch.no_grad()
def build_phase_calibrated_view(
    *,
    model: nn.Module,
    batch: dict,
    sample_ids: Tensor,
    group_id: int,
    member_classes: tuple[int, ...],
    center_gamma: Tensor,
) -> ConfirmedPhaseView:
    """Build a target view using one confirmed domain-level Phase transform.

    ``center_gamma`` is always the *domain-level* calibration applied to the
    target view.  It may be a confirmed non-identity group center or the
    explicit identity transform when ``PhaseDecisionStatus.IDENTITY_CONFIRMED``
    has already been established.  Individual sample/class gammas are never
    used to build this view.
    """
    if not isinstance(center_gamma, Tensor) or center_gamma.ndim != 1:
        raise ValueError("center_gamma must have shape [K_gamma]")
    gamma_cpu = center_gamma.detach().to(device="cpu", dtype=torch.float64)
    if gamma_cpu.numel() < 2 or not torch.isfinite(gamma_cpu).all().item():
        raise ValueError("center_gamma must be finite and contain at least two points")
    if not torch.all(gamma_cpu[1:] > gamma_cpu[:-1]).item():
        raise ValueError("center_gamma must be strictly increasing")
    if abs(float(gamma_cpu[0])) > 1e-6 or abs(float(gamma_cpu[-1]) - 1.0) > 1e-6:
        raise ValueError("center_gamma must preserve [0,1] endpoints")

    device = _model_device(model)
    pixels = _batch_tensor(batch, "pixels", device)
    valid_pixels = _batch_tensor(batch, "valid_pixels", device)
    positions = _batch_tensor(batch, "positions", device)
    extra = _batch_tensor(batch, "extra", device)
    time_mask = _batch_tensor(batch, "time_mask", device)
    if pixels is None or valid_pixels is None or positions is None:
        raise ValueError("batch must contain pixels, valid_pixels and positions")
    model.eval()
    backbone = model.forward_backbone(
        pixels,
        valid_pixels,
        positions,
        extra,
        time_mask=time_mask,
    )
    aligned_positions = align_target_positions_to_source(
        backbone.normalized_positions,
        backbone.time_mask,
        center_gamma,
    )
    output = model.forward_from_backbone(
        backbone,
        positions,
        extra,
        temporal_positions_override=aligned_positions,
        return_geometry=True,
    )
    if output.geometry is None:
        raise RuntimeError("phase-calibrated view requires functional geometry")
    shape_grid = output.geometry.canonical_grid
    reg_grid = torch.linspace(
        0.0,
        1.0,
        center_gamma.numel(),
        device=shape_grid.device,
        dtype=shape_grid.dtype,
    )
    gamma_shape = resample_gamma(
        center_gamma.to(device=shape_grid.device, dtype=shape_grid.dtype),
        reg_grid,
        shape_grid,
    )
    aligned_q = torch.stack(
        [
            warp_q_gamma(q_row, gamma_shape).squeeze(0)
            for q_row in output.geometry.structure_srvf
        ]
    )
    aligned_support = torch.stack(
        [
            warp_support_gamma(support_row, gamma_shape, shape_grid)
            for support_row in output.geometry.structure_support
        ]
    )
    probabilities = torch.softmax(output.logits, dim=-1)

    def detached(value: Tensor) -> Tensor:
        return value.detach()

    return ConfirmedPhaseView(
        sample_ids=sample_ids.detach().to(device="cpu", dtype=torch.long),
        group_id=int(group_id),
        member_classes=tuple(int(item) for item in member_classes),
        center_gamma=gamma_cpu,
        aligned_positions=detached(aligned_positions),
        logits=detached(output.logits),
        probabilities=detached(probabilities),
        fused_repr=detached(output.fused_repr),
        trend_repr=detached(output.trend_repr),
        structure_repr=detached(output.structure_repr),
        aligned_q_shape=detached(aligned_q),
        aligned_q_support=detached(aligned_support),
        q_valid=detached(output.geometry.structure_valid),
    )


@torch.no_grad()
def build_confirmed_phase_view(
    *,
    model: nn.Module,
    batch: dict,
    sample_ids: Tensor,
    group: PhaseGroup,
) -> ConfirmedPhaseView:
    if group.status is not PhaseGroupStatus.CONFIRMED:
        raise ValueError("only confirmed phase groups can produce target views")
    return build_phase_calibrated_view(
        model=model,
        batch=batch,
        sample_ids=sample_ids,
        group_id=group.group_id,
        member_classes=group.member_classes,
        center_gamma=group.center_gamma,
    )
