from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from methods.structure_da.domain_phase_state import (
    DomainPhaseState,
    PhaseGroup,
    PhaseGroupStatus,
)
from tests.structure_da.test_stage2_parameter_policy import _model


def _group(
    group_id: int = 0,
    members: tuple[int, ...] = (0, 1),
    status: PhaseGroupStatus = PhaseGroupStatus.CONFIRMED,
    *,
    power: float = 1.5,
) -> PhaseGroup:
    gamma = torch.linspace(0.0, 1.0, 128, dtype=torch.float64).pow(power)
    return PhaseGroup(
        group_id=group_id,
        member_classes=members,
        center_gamma=gamma,
        within_dispersion=0.0,
        diameter=0.0,
        core_radius=0.0,
        sample_evidence_count=4.0,
        class_count=len(members),
        center_drift=0.0,
        status=status,
        confirmation_age=2,
    )


def _state(*groups: PhaseGroup) -> DomainPhaseState:
    return DomainPhaseState(
        scan_index=1,
        m=len(groups),
        class_centers=(),
        valid_phase_classes=tuple(sorted({c for group in groups for c in group.member_classes})),
        groups=groups,
        rejected_classes=(),
    )


def test_confirmed_lookup_excludes_provisional_rejected_and_g0() -> None:
    from methods.structure_da.confirmed_phase_view import build_confirmed_class_to_group_map

    confirmed = _group(0, (0, 1), PhaseGroupStatus.CONFIRMED)
    provisional = _group(1, (2, 3), PhaseGroupStatus.PROVISIONAL)
    rejected = _group(2, (4, 5), PhaseGroupStatus.REJECTED)
    mapping = build_confirmed_class_to_group_map(_state(confirmed, provisional, rejected))
    assert mapping == {0: confirmed, 1: confirmed}

    inconsistent_m0 = replace(_state(confirmed), m=0)
    assert build_confirmed_class_to_group_map(inconsistent_m0) == {}

    duplicate = replace(provisional, member_classes=(1, 2), status=PhaseGroupStatus.CONFIRMED)
    with pytest.raises(ValueError, match="more than one confirmed group"):
        build_confirmed_class_to_group_map(_state(confirmed, duplicate))


def test_align_target_positions_uses_inverse_warp_and_ignores_padding() -> None:
    from methods.structure_da.confirmed_phase_view import align_target_positions_to_source

    source = torch.tensor([[0.0, 0.2, 0.5, 0.8, 1.0]], dtype=torch.float32)
    delta = torch.linspace(0.0, 1.0, 128, dtype=torch.float32).square()
    target = source.square()
    mask = torch.tensor([[True, True, True, True, False]])
    target[0, -1] = float("nan")
    aligned = align_target_positions_to_source(target, mask, delta)

    torch.testing.assert_close(aligned[0, :-1], source[0, :-1], atol=2e-4, rtol=0)
    assert aligned[0, -1].item() == 0.0
    assert torch.all(aligned[0, 1:4] > aligned[0, :3])
    assert aligned.dtype == target.dtype
    assert aligned.device == target.device


def test_identity_phase_does_not_change_valid_positions() -> None:
    from methods.structure_da.confirmed_phase_view import align_target_positions_to_source

    positions = torch.tensor([[0.0, 0.25, 0.8, 1.0]])
    mask = torch.ones_like(positions, dtype=torch.bool)
    identity = torch.linspace(0.0, 1.0, 128)
    torch.testing.assert_close(
        align_target_positions_to_source(positions, mask, identity),
        positions,
        atol=1e-6,
        rtol=0,
    )


def test_raw_position_override_leaves_functional_geometry_on_original_positions(
    monkeypatch,
) -> None:
    from methods.structure_da.confirmed_phase_view import align_target_positions_to_source

    torch.manual_seed(44)
    model = _model().eval()
    pixels = torch.randn(2, 5, 2, 4)
    valid = torch.ones(2, 5, 4, dtype=torch.bool)
    physical = torch.tensor([0.0, 80.0, 170.0, 270.0, 365.0])
    positions_seen = []
    original_forward = model.temporal_module.raw_encoder.forward

    def capture_positions(*args, **kwargs):
        positions_seen.append(kwargs["positions"].detach().clone())
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.temporal_module.raw_encoder, "forward", capture_positions)
    baseline = model(pixels, valid, physical, return_geometry=True)
    delta = torch.linspace(0.0, 1.0, 128).pow(1.5)
    aligned = align_target_positions_to_source(
        baseline.positions, baseline.mask, delta
    )
    shifted = model(
        pixels,
        valid,
        physical,
        temporal_positions_override=aligned,
        return_geometry=True,
    )

    torch.testing.assert_close(shifted.geometry.trend_srvf, baseline.geometry.trend_srvf)
    torch.testing.assert_close(shifted.geometry.structure_srvf, baseline.geometry.structure_srvf)
    torch.testing.assert_close(positions_seen[0], baseline.positions)
    torch.testing.assert_close(positions_seen[1], aligned)
    assert not torch.allclose(positions_seen[1], positions_seen[0])


def test_confirmed_phase_view_uses_group_center_and_returns_detached_outputs() -> None:
    from methods.structure_da.confirmed_phase_view import build_confirmed_phase_view

    torch.manual_seed(45)
    model = _model().eval()
    batch = {
        "pixels": torch.randn(2, 5, 2, 4),
        "valid_pixels": torch.ones(2, 5, 4, dtype=torch.bool),
        "positions": torch.tensor([0.0, 80.0, 170.0, 270.0, 365.0]),
    }
    group = _group()
    view = build_confirmed_phase_view(
        model=model,
        batch=batch,
        sample_ids=torch.tensor([7, 9]),
        group=group,
    )

    assert view.sample_ids.tolist() == [7, 9]
    assert view.group_id == group.group_id
    assert view.aligned_q_shape.shape == (2, 5, 4)
    assert view.aligned_q_support.shape == (2, 5)
    assert view.q_valid.shape == (2,)
    for value in view.__dict__.values():
        if isinstance(value, torch.Tensor):
            assert value.requires_grad is False
