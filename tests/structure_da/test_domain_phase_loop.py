from __future__ import annotations

import torch

from methods.structure_da.domain_phase_loop import (
    PHASE_ARMS,
    actual_phase_for_arm,
    build_shared_domain_phase,
    identity_phase,
    map_source_batch_positions,
    map_target_batch_positions,
    phase_distance_value,
)
from methods.structure_da.residual_phase_diagnostic import inverse_phase


def _gamma(power: float, k: int = 65) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, k, dtype=torch.float64)
    out = grid.pow(power)
    out[0] = 0.0
    out[-1] = 1.0
    return out


def test_identity_phase_preserves_source_and_target_coordinates():
    positions = torch.tensor([[0.0, 0.2, 0.5, 1.0]], dtype=torch.float64)
    mask = torch.ones_like(positions, dtype=torch.bool)
    gamma = identity_phase(65)
    assert torch.allclose(map_source_batch_positions(positions, mask, gamma), positions, atol=1e-12)
    assert torch.allclose(map_target_batch_positions(positions, mask, gamma), positions, atol=1e-12)


def test_phase_direction_source_forward_target_inverse():
    gamma = _gamma(1.25)
    source = torch.tensor([[0.1, 0.3, 0.7, 0.9]], dtype=torch.float64)
    mask = torch.ones_like(source, dtype=torch.bool)
    target = map_source_batch_positions(source, mask, gamma)
    recovered = map_target_batch_positions(target, mask, gamma)
    assert torch.allclose(recovered, source, atol=2e-3)
    assert torch.allclose(inverse_phase(gamma)[0::16], _gamma(1 / 1.25)[0::16], atol=3e-2)


def test_shared_phase_is_two_stage_class_equal_frechet_center():
    gammas = {
        0: [_gamma(0.90), _gamma(0.92), _gamma(0.94), _gamma(0.96)],
        1: [_gamma(1.20)],
    }
    estimate = build_shared_domain_phase(
        gammas, accepted_count_by_class={0: 4, 1: 1}, num_classes=3, k_reg=65
    )
    assert estimate.valid
    assert estimate.participating_classes == (0, 1)
    # Duplicating class-0 observations changes its within-class center only
    # negligibly; it must not give class 0 four times the shared weight.
    duplicated = build_shared_domain_phase(
        {0: gammas[0] * 10, 1: gammas[1]},
        accepted_count_by_class={0: 40, 1: 1}, num_classes=3, k_reg=65,
    )
    assert phase_distance_value(estimate.gamma, duplicated.gamma) < 2e-3


def test_single_class_phase_is_invalid_and_identity_proposal():
    estimate = build_shared_domain_phase(
        {2: [_gamma(1.1)]}, accepted_count_by_class={2: 8}, num_classes=4, k_reg=65
    )
    assert not estimate.valid
    assert estimate.invalid_reason == "fewer_than_two_nonempty_class_phase_centers"
    assert torch.equal(estimate.gamma, identity_phase(65))


def test_no_phase_always_identity():
    identity = identity_phase(65)
    proposal = build_shared_domain_phase(
        {0: [_gamma(0.9)], 1: [_gamma(1.2)]},
        accepted_count_by_class={0: 1, 1: 1}, num_classes=2, k_reg=65,
    )
    out = actual_phase_for_arm(
        "NO_PHASE", identity=identity, first_phase=None,
        previous_phase=_gamma(1.1), proposal=proposal, phase_epoch=5,
    )
    assert torch.equal(out, identity)


def test_static_phase_freezes_first_valid_phase():
    identity = identity_phase(65)
    first = _gamma(1.1)
    proposal = build_shared_domain_phase(
        {0: [_gamma(0.8)], 1: [_gamma(1.3)]},
        accepted_count_by_class={0: 1, 1: 1}, num_classes=2, k_reg=65,
    )
    out = actual_phase_for_arm(
        "STATIC_DOMAIN_PHASE", identity=identity, first_phase=first,
        previous_phase=first, proposal=proposal, phase_epoch=7,
    )
    assert torch.equal(out, first)


def test_iterative_phase_updates_only_from_valid_boundary_proposal():
    identity = identity_phase(65)
    valid = build_shared_domain_phase(
        {0: [_gamma(0.85)], 1: [_gamma(1.15)]},
        accepted_count_by_class={0: 1, 1: 1}, num_classes=2, k_reg=65,
    )
    out = actual_phase_for_arm(
        "ITERATIVE_DOMAIN_PHASE", identity=identity, first_phase=None,
        previous_phase=identity, proposal=valid, phase_epoch=2,
    )
    assert phase_distance_value(out, valid.gamma) < 1e-12
    invalid = build_shared_domain_phase(
        {0: [_gamma(0.85)]}, accepted_count_by_class={0: 1}, num_classes=2, k_reg=65,
    )
    held = actual_phase_for_arm(
        "ITERATIVE_DOMAIN_PHASE", identity=identity, first_phase=None,
        previous_phase=out, proposal=invalid, phase_epoch=3,
    )
    assert torch.equal(held, out)


def test_phase_arm_names_are_frozen():
    assert PHASE_ARMS == ("NO_PHASE", "STATIC_DOMAIN_PHASE", "ITERATIVE_DOMAIN_PHASE")


def _script_text() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parents[2] / "scripts" / "diagnose_stage2_shared_domain_phase_loop_14.py").read_text(encoding="utf-8")


def test_training_script_uses_inverse_phase_for_teacher_and_native_target_for_student():
    text = _script_text()
    assert "mapped = map_target_batch_positions" in text
    assert "teacher_out=_forward_target_teacher_phase" in text
    assert "target_out=_raw_forward(model,target)" in text


def test_training_script_uses_forward_phase_for_source_supervision():
    text = _script_text()
    assert "mapped = map_source_batch_positions" in text
    assert "source_out=_forward_source_phase(model,source,gamma)" in text


def test_training_script_has_one_common_warmup_before_arm_fork():
    text = _script_text()
    warmup = text.index("# One common warm-up")
    arm_loop = text.index("for arm in PHASE_ARMS")
    assert warmup < arm_loop
    assert "warmup_state=" in text
    assert "model.load_state_dict(warmup_state[\"student\"]" in text


def test_training_script_keeps_geometry_frozen_and_hash_checked():
    text = _script_text()
    assert "geometry_model.eval()" in text
    assert "p.requires_grad_(False) for p in geometry_model.parameters()" in text
    assert "if geometry_hash_after!=geometry_hash_before" in text


def test_training_script_does_not_restore_old_stable_label_or_alignment_stack():
    text = _script_text()
    assert "from methods.structure_da.stable_target_labels" not in text
    assert "scan_stable_target_labels" not in text
    assert "StableLabelConfig" not in text
    assert "MMD" not in text
    assert "DANN" not in text
    assert "CORAL" not in text
