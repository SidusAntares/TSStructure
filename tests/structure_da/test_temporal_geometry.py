from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from methods.structure_da import (
    PhaseTangentOutput,
    TemporalGeometryLossOutput,
    TemporalGeometryObjective,
    TemporalRegistrationOutput,
    TemporalSRVFRegistration,
    warp_to_identity_tangent,
)


def _widths_from_tangent(tangent: torch.Tensor, angle: float) -> torch.Tensor:
    tangent = tangent / tangent.square().mean().sqrt()
    psi = torch.cos(torch.tensor(angle, dtype=tangent.dtype)) + (
        torch.sin(torch.tensor(angle, dtype=tangent.dtype)) * tangent
    )
    return psi.square() / tangent.numel()


def _registration_output(
    *,
    registered_srvf: torch.Tensor | None = None,
    template_srvf: torch.Tensor | None = None,
    registered_support: torch.Tensor | None = None,
    template_support: torch.Tensor | None = None,
    interval_widths: torch.Tensor | None = None,
    registration_valid: torch.Tensor | None = None,
    batch_size: int = 2,
    grid_size: int = 5,
    feature_dim: int = 2,
    dtype: torch.dtype = torch.float32,
) -> TemporalRegistrationOutput:
    if registered_srvf is None:
        if template_srvf is not None:
            batch_size, grid_size, feature_dim = template_srvf.shape
        elif registered_support is not None:
            batch_size, grid_size = registered_support.shape
        elif template_support is not None:
            batch_size, grid_size = template_support.shape
        elif interval_widths is not None:
            batch_size = interval_widths.shape[0]
            grid_size = interval_widths.shape[1] + 1
    if registered_srvf is None:
        registered_srvf = torch.zeros(
            batch_size, grid_size, feature_dim, dtype=dtype
        )
    batch_size, grid_size, feature_dim = registered_srvf.shape
    if template_srvf is None:
        template_srvf = torch.zeros_like(registered_srvf)
    if registered_support is None:
        registered_support = torch.ones(
            batch_size, grid_size, dtype=registered_srvf.dtype
        )
    if template_support is None:
        template_support = torch.ones_like(registered_support)
    if interval_widths is None:
        interval_widths = torch.full(
            (batch_size, grid_size - 1),
            1.0 / (grid_size - 1),
            dtype=registered_srvf.dtype,
            device=registered_srvf.device,
        )
    if registration_valid is None:
        registration_valid = torch.ones(
            batch_size, dtype=torch.bool, device=registered_srvf.device
        )
    cumulative = interval_widths.cumsum(dim=-1)
    warp = torch.cat(
        [
            torch.zeros_like(interval_widths[:, :1]),
            cumulative[:, :-1],
            torch.ones_like(interval_widths[:, :1]),
        ],
        dim=-1,
    )
    interval_speed = interval_widths * (grid_size - 1)
    warp_derivative = torch.cat(
        [
            interval_speed[:, :1],
            0.5 * (interval_speed[:, :-1] + interval_speed[:, 1:]),
            interval_speed[:, -1:],
        ],
        dim=-1,
    )
    return TemporalRegistrationOutput(
        srvf_output=None,
        template_srvf=template_srvf,
        template_support=template_support,
        template_initialized=torch.tensor(True, device=registered_srvf.device),
        template_mean_support=template_support.mean(),
        interval_logits=torch.zeros_like(interval_widths),
        interval_widths=interval_widths,
        warp=warp,
        warp_derivative=warp_derivative,
        registered_srvf=registered_srvf,
        registered_support=registered_support,
        registration_valid=registration_valid,
    )


def _objective(grid_size: int = 5, **kwargs) -> TemporalGeometryObjective:
    return TemporalGeometryObjective(canonical_grid_size=grid_size, **kwargs)


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


def test_objective_registers_normalized_integration_buffers() -> None:
    objective = _objective(grid_size=6)

    assert set(dict(objective.named_buffers())) == {
        "grid_integration_weights",
        "interval_integration_weights",
    }
    assert not dict(objective.named_parameters())
    torch.testing.assert_close(
        objective.grid_integration_weights,
        torch.tensor([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]),
    )
    torch.testing.assert_close(
        objective.interval_integration_weights, torch.full((5,), 0.2)
    )


def test_equal_registered_and_template_srvf_has_zero_alignment() -> None:
    srvf = torch.randn(2, 5, 3)
    output = _objective()(
        _registration_output(registered_srvf=srvf, template_srvf=srvf.clone()),
        torch.tensor([True, True]),
    )

    assert isinstance(output, TemporalGeometryLossOutput)
    torch.testing.assert_close(output.alignment_loss, torch.zeros(()))
    torch.testing.assert_close(output.per_sample_alignment_error, torch.zeros(2))


def test_alignment_matches_manual_support_weighted_formula() -> None:
    registered = torch.zeros(1, 5, 2)
    registered[:, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    sample_support = torch.tensor([[1.0, 0.5, 0.0, 0.5, 1.0]])
    template_support = torch.tensor([[0.5, 1.0, 1.0, 0.5, 0.25]])
    result = _objective()(
        _registration_output(
            registered_srvf=registered,
            registered_support=sample_support,
            template_support=template_support,
        ),
        torch.tensor([True]),
    )
    grid_weights = torch.tensor([0.125, 0.25, 0.25, 0.25, 0.125])
    support = sample_support * template_support
    squared_error = registered.square().sum(dim=-1)
    expected = (grid_weights * support * squared_error).sum() / (
        grid_weights * support
    ).sum()

    torch.testing.assert_close(result.alignment_loss, expected)
    torch.testing.assert_close(result.per_sample_alignment_error[0], expected)


def test_zero_support_hides_arbitrarily_large_alignment_error() -> None:
    registered = torch.zeros(1, 5, 1)
    registered[0, 2, 0] = 1e6
    support = torch.ones(1, 5)
    support[0, 2] = 0.0

    result = _objective()(
        _registration_output(
            registered_srvf=registered,
            registered_support=support,
        ),
        torch.tensor([True]),
    )

    torch.testing.assert_close(result.alignment_loss, torch.zeros(()))


@pytest.mark.parametrize(
    "source_mask,registration_valid",
    [
        (torch.tensor([True, False]), torch.tensor([True, True])),
        (torch.tensor([True, True]), torch.tensor([True, False])),
    ],
)
def test_inactive_samples_do_not_contribute_to_alignment(
    source_mask, registration_valid
) -> None:
    registered = torch.stack([torch.ones(5, 1), torch.full((5, 1), 100.0)])
    result = _objective()(
        _registration_output(
            registered_srvf=registered,
            registration_valid=registration_valid,
        ),
        source_mask,
    )

    torch.testing.assert_close(result.alignment_loss, torch.tensor(1.0))
    torch.testing.assert_close(
        result.per_sample_alignment_error, torch.tensor([1.0, 0.0])
    )


def test_alignment_aggregates_total_numerator_over_total_support() -> None:
    registered = torch.stack([torch.ones(5, 1), torch.full((5, 1), 3.0)])
    support = torch.stack([torch.ones(5), torch.full((5,), 0.1)])
    result = _objective()(
        _registration_output(
            registered_srvf=registered,
            registered_support=support,
        ),
        torch.tensor([True, True]),
    )

    expected = torch.tensor((1.0 + 0.9) / (1.0 + 0.1))
    torch.testing.assert_close(result.alignment_loss, expected)
    assert not torch.isclose(result.alignment_loss, torch.tensor(5.0))


def test_identity_speed_has_zero_roughness() -> None:
    result = _objective()(
        _registration_output(), torch.tensor([True, True])
    )

    torch.testing.assert_close(result.roughness_loss, torch.zeros(()))
    torch.testing.assert_close(result.per_sample_warp_roughness, torch.zeros(2))


def test_piecewise_speed_has_positive_roughness() -> None:
    widths = torch.tensor([[0.1, 0.2, 0.4, 0.3]])
    result = _objective()(
        _registration_output(interval_widths=widths), torch.tensor([True])
    )

    assert result.roughness_loss.item() > 0
    assert result.per_sample_warp_roughness.item() > 0


def test_low_support_reduces_roughness_at_speed_change() -> None:
    widths = torch.tensor([[0.1, 0.1, 0.7, 0.1]])
    full = _objective()(
        _registration_output(interval_widths=widths), torch.tensor([True])
    )
    support = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0]])
    low = _objective()(
        _registration_output(
            interval_widths=widths, registered_support=support
        ),
        torch.tensor([True]),
    )

    assert low.roughness_loss < full.roughness_loss


def test_two_point_grid_has_stable_zero_roughness() -> None:
    result = _objective(grid_size=2)(
        _registration_output(grid_size=2), torch.tensor([True, True])
    )

    torch.testing.assert_close(result.roughness_loss, torch.zeros(()))
    torch.testing.assert_close(result.per_sample_warp_roughness, torch.zeros(2))


def test_identity_warp_has_zero_unsupported_loss() -> None:
    support = torch.zeros(2, 5)
    result = _objective()(
        _registration_output(registered_support=support),
        torch.tensor([True, True]),
    )

    torch.testing.assert_close(result.unsupported_loss, torch.zeros(()))


def test_nonidentity_speed_in_low_support_region_is_penalized() -> None:
    widths = torch.tensor([[0.6, 0.2, 0.1, 0.1]])
    support = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])
    result = _objective()(
        _registration_output(
            interval_widths=widths, registered_support=support
        ),
        torch.tensor([True]),
    )

    assert result.unsupported_loss.item() > 0


def test_full_support_disables_unsupported_loss() -> None:
    widths = torch.tensor([[0.6, 0.2, 0.1, 0.1]])
    result = _objective()(
        _registration_output(interval_widths=widths), torch.tensor([True])
    )

    torch.testing.assert_close(result.unsupported_loss, torch.zeros(()))


def test_unsupported_loss_depends_on_location_of_low_support() -> None:
    widths = torch.tensor([[0.6, 0.3, 0.1]])
    low_first = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    low_last = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    first = _objective(grid_size=4)(
        _registration_output(
            grid_size=4,
            interval_widths=widths,
            registered_support=low_first,
        ),
        torch.tensor([True]),
    )
    last = _objective(grid_size=4)(
        _registration_output(
            grid_size=4,
            interval_widths=widths,
            registered_support=low_last,
        ),
        torch.tensor([True]),
    )

    assert last.unsupported_loss > first.unsupported_loss


def test_identity_phase_center_is_zero() -> None:
    result = _objective()(
        _registration_output(batch_size=3), torch.tensor([True, True, True])
    )

    torch.testing.assert_close(result.center_loss, torch.zeros(()))


def test_same_phase_directions_have_positive_center_loss() -> None:
    widths = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).expand(3, -1).clone()
    result = _objective()(
        _registration_output(interval_widths=widths),
        torch.tensor([True, True, True]),
    )

    assert result.center_loss.item() > 0


def test_opposite_phase_tangents_center_better_than_same_direction() -> None:
    base = torch.tensor([1.0, -1.0, 1.0, -1.0])
    positive = _widths_from_tangent(base, 0.2)
    negative = _widths_from_tangent(-base, 0.2)
    opposite_widths = torch.stack([positive, negative])
    same_widths = torch.stack([positive, positive])
    objective = _objective()

    opposite = objective(
        _registration_output(interval_widths=opposite_widths),
        torch.tensor([True, True]),
    )
    same = objective(
        _registration_output(interval_widths=same_widths),
        torch.tensor([True, True]),
    )

    assert opposite.center_loss < same.center_loss * 0.01


def test_center_excludes_non_source_and_invalid_samples() -> None:
    widths = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]]
    )
    result = _objective()(
        _registration_output(
            interval_widths=widths,
            registration_valid=torch.tensor([True, True, False]),
        ),
        torch.tensor([True, False, True]),
    )
    expected = warp_to_identity_tangent(widths[:1]).tangent.square().mean()

    torch.testing.assert_close(result.center_loss, expected)
    assert result.active_mask.tolist() == [True, False, False]
    assert result.active_count.item() == 1


def test_low_mean_support_reduces_sample_influence_on_phase_center() -> None:
    first = torch.tensor([0.1, 0.2, 0.3, 0.4])
    second = torch.tensor([0.4, 0.3, 0.2, 0.1])
    widths = torch.stack([first, second])
    equal_support = torch.ones(2, 5)
    low_second = equal_support.clone()
    low_second[1] = 0.01
    objective = _objective()
    equal = objective(
        _registration_output(
            interval_widths=widths, registered_support=equal_support
        ),
        torch.tensor([True, True]),
    )
    weighted = objective(
        _registration_output(
            interval_widths=widths, registered_support=low_second
        ),
        torch.tensor([True, True]),
    )
    first_center = warp_to_identity_tangent(first.unsqueeze(0)).tangent.square().mean()

    assert torch.abs(weighted.center_loss - first_center) < torch.abs(
        equal.center_loss - first_center
    )


def test_center_returns_zero_when_total_sample_weight_does_not_exceed_eps() -> None:
    widths = torch.tensor([[0.1, 0.2, 0.3, 0.4]], requires_grad=True)
    support = torch.full((1, 5), 1e-10)
    result = _objective(eps=1e-8)(
        _registration_output(
            interval_widths=widths,
            registered_support=support,
        ),
        torch.tensor([True]),
    )

    torch.testing.assert_close(
        result.center_loss, torch.zeros(()), atol=0, rtol=0
    )
    assert result.center_loss.requires_grad


def test_total_loss_is_weighted_sum_and_zero_weight_keeps_diagnostic() -> None:
    registered = torch.ones(1, 5, 1)
    widths = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    support = torch.zeros(1, 5)
    objective = _objective(
        alignment_weight=2.0,
        roughness_weight=3.0,
        unsupported_weight=0.0,
        center_weight=5.0,
    )
    result = objective(
        _registration_output(
            registered_srvf=registered,
            registered_support=support,
            template_support=torch.ones_like(support),
            interval_widths=widths,
        ),
        torch.tensor([True]),
    )
    expected = (
        2.0 * result.alignment_loss
        + 3.0 * result.roughness_loss
        + 5.0 * result.center_loss
    )

    torch.testing.assert_close(result.total_loss, expected)
    assert result.unsupported_loss.item() > 0


def test_no_active_source_returns_graph_connected_zeros_and_diagnostics() -> None:
    registered = torch.randn(2, 5, 2, requires_grad=True)
    widths = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        requires_grad=True,
    )
    result = _objective()(
        _registration_output(
            registered_srvf=registered,
            interval_widths=widths,
            registration_valid=torch.tensor([True, True]),
        ),
        torch.tensor([False, False]),
    )

    for loss in (
        result.alignment_loss,
        result.roughness_loss,
        result.unsupported_loss,
        result.center_loss,
        result.total_loss,
    ):
        torch.testing.assert_close(loss, torch.zeros_like(loss))
        assert loss.requires_grad
    torch.testing.assert_close(result.per_sample_alignment_error, torch.zeros(2))
    torch.testing.assert_close(result.per_sample_warp_roughness, torch.zeros(2))
    torch.testing.assert_close(result.per_sample_unsupported_error, torch.zeros(2))
    assert result.phase_tangent.abs().sum().item() > 0
    assert result.phase_magnitude.sum().item() > 0
    assert result.active_count.item() == 0
    result.total_loss.backward()
    assert registered.grad is not None
    assert widths.grad is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_objective_preserves_floating_dtype(dtype: torch.dtype) -> None:
    widths = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=dtype)
    objective = _objective().to(dtype=dtype)
    result = objective(
        _registration_output(dtype=dtype, batch_size=1, interval_widths=widths),
        torch.tensor([True]),
    )

    for value in (
        result.total_loss,
        result.alignment_loss,
        result.roughness_loss,
        result.unsupported_loss,
        result.center_loss,
        result.per_sample_alignment_error,
        result.per_sample_warp_roughness,
        result.per_sample_unsupported_error,
        result.interval_support,
        result.phase_tangent,
        result.phase_magnitude,
    ):
        assert value.dtype == dtype
        assert torch.isfinite(value).all()


def test_objective_promotes_half_precision_inputs_to_float32() -> None:
    result = _objective()(
        _registration_output(dtype=torch.float16),
        torch.tensor([True, True]),
    )

    assert result.total_loss.dtype == torch.float32
    assert torch.isfinite(result.total_loss)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"canonical_grid_size": 1},
        {"canonical_grid_size": 5, "alignment_weight": -1.0},
        {"canonical_grid_size": 5, "roughness_weight": float("nan")},
        {"canonical_grid_size": 5, "unsupported_weight": float("inf")},
        {"canonical_grid_size": 5, "center_weight": -0.1},
        {"canonical_grid_size": 5, "eps": 0.0},
    ],
)
def test_objective_rejects_invalid_constructor_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        TemporalGeometryObjective(**kwargs)


def test_objective_rejects_wrong_registration_output_type() -> None:
    with pytest.raises(ValueError, match="TemporalRegistrationOutput"):
        _objective()(object(), torch.tensor([True]))


@pytest.mark.parametrize(
    "source_mask,match",
    [
        (torch.ones(2), "boolean"),
        (torch.ones(3, dtype=torch.bool), "shape"),
    ],
)
def test_objective_rejects_invalid_source_mask(source_mask, match) -> None:
    with pytest.raises(ValueError, match=match):
        _objective()(_registration_output(), source_mask)


def test_objective_rejects_grid_size_mismatch() -> None:
    with pytest.raises(ValueError, match="canonical grid"):
        _objective(grid_size=6)(
            _registration_output(grid_size=5), torch.tensor([True, True])
        )


@pytest.mark.parametrize("field", ["registered_support", "template_support"])
def test_objective_rejects_support_outside_unit_interval(field) -> None:
    output = _registration_output()
    bad = getattr(output, field).clone()
    bad[0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _objective()(replace(output, **{field: bad}), torch.tensor([True, True]))


def test_objective_rejects_invalid_interval_widths() -> None:
    output = _registration_output()
    with pytest.raises(ValueError, match="positive"):
        _objective()(
            replace(output, interval_widths=torch.tensor([[0.0, 0.3, 0.3, 0.4]]).expand(2, -1)),
            torch.tensor([True, True]),
        )


@pytest.mark.parametrize("field", ["registered_srvf", "template_srvf"])
def test_objective_rejects_nonfinite_srvf(field) -> None:
    output = _registration_output()
    bad = getattr(output, field).clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _objective()(replace(output, **{field: bad}), torch.tensor([True, True]))


def test_objective_rejects_mismatched_batch_dimensions() -> None:
    output = _registration_output()
    with pytest.raises(ValueError, match="shape"):
        _objective()(
            replace(output, template_support=torch.ones(1, 5)),
            torch.tensor([True, True]),
        )


def _make_registration(dtype: torch.dtype = torch.float32):
    registration = TemporalSRVFRegistration(
        num_channels=1,
        channel_feature_dim=2,
        num_basis=6,
        canonical_grid_size=8,
        roughness_grid_size=64,
        min_mean_support=0.0,
        min_dynamic_energy=0.0,
        min_template_mean_support=0.0,
        warp_hidden_dim=8,
        warp_kernel_size=3,
    ).to(dtype=dtype)
    torch.manual_seed(73)
    tokens = torch.randn(2, 6, 1, 2, dtype=dtype)
    positions = torch.tensor(
        [0.0, 39.0, 92.0, 157.0, 244.0, 345.0], dtype=dtype
    )
    mask = torch.tensor(
        [[True, True, True, True, True, True],
         [True, False, True, True, False, True]]
    )
    return registration, tokens, positions, mask


def test_real_registration_geometry_backward_preserves_expected_gradients() -> None:
    registration, tokens, positions, mask = _make_registration()
    registration.update_source_statistics(tokens, mask)
    initial = registration(tokens, positions, mask)
    registration.update_source_support_scale(initial.srvf_output.functional)
    bootstrap = registration(tokens, positions, mask)
    registration.update_source_template(bootstrap)
    differentiable = (tokens + 0.1 * torch.randn_like(tokens)).requires_grad_()
    output = registration(differentiable, positions, mask)
    objective = TemporalGeometryObjective(canonical_grid_size=8)

    geometry = objective(output, torch.tensor([True, True]))
    geometry.total_loss.backward()

    assert output.registration_valid.all().item()
    assert differentiable.grad is not None and torch.isfinite(differentiable.grad).all()
    assert differentiable.grad[mask].abs().sum().item() > 0
    torch.testing.assert_close(
        differentiable.grad[~mask],
        torch.zeros_like(differentiable.grad[~mask]),
        atol=0,
        rtol=0,
    )
    last = registration.warp_estimator.network[-1]
    assert last.weight.grad is not None and torch.isfinite(last.weight.grad).all()
    assert last.bias.grad is not None and torch.isfinite(last.bias.grad).all()
    assert not dict(registration.source_template.named_parameters())
    assert not dict(objective.named_parameters())
    for value in registration.source_template.buffers():
        assert value.grad is None
