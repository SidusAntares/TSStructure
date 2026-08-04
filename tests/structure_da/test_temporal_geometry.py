from __future__ import annotations

import pytest
import torch

from methods.structure_da import PhaseTangentOutput, warp_to_identity_tangent

def _widths_from_tangent(tangent: torch.Tensor, angle: float) -> torch.Tensor:
    tangent = tangent / tangent.square().mean().sqrt()
    psi = torch.cos(torch.tensor(angle, dtype=tangent.dtype)) + (
        torch.sin(torch.tensor(angle, dtype=tangent.dtype)) * tangent
    )
    return psi.square() / tangent.numel()


def test_identity_warp_maps_to_zero_identity_tangent() -> None:
    widths = torch.full((3, 6), 1.0 / 6)

    output = warp_to_identity_tangent(widths)

    assert isinstance(output, PhaseTangentOutput)
    torch.testing.assert_close(output.interval_speed, torch.ones_like(widths))
    torch.testing.assert_close(output.warp_srvf, torch.ones_like(widths))
    torch.testing.assert_close(output.tangent, torch.zeros_like(widths))
    torch.testing.assert_close(output.magnitude, torch.zeros(3))


def test_nonidentity_phase_tangent_is_orthogonal_and_differentiable() -> None:
    widths = torch.tensor(
        [[0.08, 0.17, 0.29, 0.31, 0.15]], requires_grad=True
    )

    output = warp_to_identity_tangent(widths)
    expected_angle = torch.acos(output.warp_srvf.mean(dim=-1))

    assert torch.isfinite(output.tangent).all()
    torch.testing.assert_close(
        output.tangent.mean(dim=-1), torch.zeros(1), atol=2e-6, rtol=0
    )
    torch.testing.assert_close(output.magnitude, expected_angle, atol=2e-6, rtol=1e-5)
    (output.tangent.square().sum() + output.magnitude.sum()).backward()
    assert widths.grad is not None and torch.isfinite(widths.grad).all()


@pytest.mark.parametrize(
    "widths,match",
    [
        (torch.ones(4), "shape"),
        (torch.ones(2, 3, dtype=torch.long), "floating"),
        (torch.tensor([[0.5, float("nan")]]), "finite"),
        (torch.tensor([[0.5, float("inf")]]), "finite"),
        (torch.tensor([[0.0, 1.0]]), "positive"),
        (torch.tensor([[-0.1, 1.1]]), "positive"),
        (torch.tensor([[0.2, 0.2]]), "sum"),
    ],
)
def test_phase_tangent_rejects_invalid_widths(widths, match) -> None:
    with pytest.raises(ValueError, match=match):
        warp_to_identity_tangent(widths)
