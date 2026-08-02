import pytest
import torch

from methods.structure_da import StructureBackbone, StructureBackboneOutput
from models.pse import PixelSetEncoder


def _make_inputs(
    batch_size: int = 2,
    sequence_length: int = 5,
    num_channels: int = 4,
    num_pixels: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(41)
    pixels = torch.randn(batch_size, sequence_length, num_channels, num_pixels)
    valid_pixels = torch.ones(
        batch_size, sequence_length, num_pixels, dtype=torch.bool
    )
    positions = torch.tensor([0, 4, 11, 23, 47])[:sequence_length]
    return pixels, valid_pixels, positions


def _make_backbone(input_dim: int = 4) -> StructureBackbone:
    return StructureBackbone(
        input_dim=input_dim,
        mlp1=[input_dim, 7, 8],
        pooling="mean_std",
        mlp2=[16, 8],
        with_extra=False,
        extra_size=4,
    )


def test_default_mask_shapes_and_reconstruction() -> None:
    pixels, valid_pixels, positions = _make_inputs()
    backbone = _make_backbone()

    output = backbone(pixels, valid_pixels, positions, extra=None)

    assert isinstance(output, StructureBackboneOutput)
    assert output.tokens.shape == (2, 5, 8)
    assert backbone.feature_dim == 8
    assert output.time_mask.shape == (2, 5)
    assert output.time_mask.dtype == torch.bool
    assert output.time_mask.device == pixels.device
    assert output.time_mask.all()
    for component in (
        output.decomposition.trend,
        output.decomposition.dynamics,
        output.decomposition.residual,
    ):
        assert component.shape == (2, 5, 8)
    torch.testing.assert_close(
        output.decomposition.trend
        + output.decomposition.dynamics
        + output.decomposition.residual,
        output.tokens,
        atol=1e-6,
        rtol=1e-5,
    )


def test_one_dimensional_numeric_time_mask_expands_to_batch() -> None:
    pixels, valid_pixels, positions = _make_inputs()
    time_mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])

    output = _make_backbone()(
        pixels, valid_pixels, positions, extra=None, time_mask=time_mask
    )

    expected = time_mask.bool().expand(2, -1)
    torch.testing.assert_close(output.time_mask, expected)


def test_partial_time_mask_only_zeros_decomposition_outputs() -> None:
    pixels, valid_pixels, positions = _make_inputs()
    backbone = _make_backbone()
    time_mask = torch.tensor(
        [[True, False, True, True, False], [True, True, False, True, True]]
    )
    expected_tokens = backbone.pixel_set_encoder(
        pixels, valid_pixels, extra=None
    )

    output = backbone(
        pixels, valid_pixels, positions, extra=None, time_mask=time_mask
    )

    torch.testing.assert_close(output.tokens, expected_tokens)
    for component in (
        output.decomposition.trend,
        output.decomposition.dynamics,
        output.decomposition.residual,
    ):
        torch.testing.assert_close(
            component[~time_mask], torch.zeros_like(component[~time_mask])
        )
    reconstruction = (
        output.decomposition.trend
        + output.decomposition.dynamics
        + output.decomposition.residual
    )
    torch.testing.assert_close(
        reconstruction[time_mask], output.tokens[time_mask]
    )


def test_gradients_reach_pse_and_both_kernel_scales() -> None:
    pixels, valid_pixels, positions = _make_inputs()
    backbone = _make_backbone()

    output = backbone(pixels, valid_pixels, positions, extra=None)
    loss = (
        output.decomposition.trend.square().mean()
        + output.decomposition.dynamics.square().mean()
        + output.decomposition.residual.square().mean()
    )
    loss.backward()

    pse_parameters = list(backbone.pixel_set_encoder.parameters())
    assert pse_parameters
    for parameter in pse_parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for parameter in (
        backbone.decomposition._tau_fast_unconstrained,
        backbone.decomposition._tau_gap_unconstrained,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().item() > 0


def test_backbone_uses_original_pixel_set_encoder() -> None:
    backbone = _make_backbone()

    assert isinstance(backbone.pixel_set_encoder, PixelSetEncoder)


@pytest.mark.parametrize(
    "time_mask",
    [
        torch.ones(1, 1, 5, dtype=torch.bool),
        torch.ones(4, dtype=torch.bool),
        torch.ones(3, 5, dtype=torch.bool),
    ],
)
def test_invalid_time_mask_shapes_raise_value_error(
    time_mask: torch.Tensor,
) -> None:
    pixels, valid_pixels, positions = _make_inputs()

    with pytest.raises(ValueError, match="time_mask"):
        _make_backbone()(
            pixels,
            valid_pixels,
            positions,
            extra=None,
            time_mask=time_mask,
        )


@pytest.mark.parametrize(
    "invalid_value", [2.0, float("nan"), float("inf")]
)
def test_non_binary_or_nonfinite_time_mask_raises_value_error(
    invalid_value: float,
) -> None:
    pixels, valid_pixels, positions = _make_inputs()
    time_mask = torch.ones(2, 5)
    time_mask[0, 2] = invalid_value

    with pytest.raises(
        ValueError, match="time_mask must contain only finite 0/1 values"
    ):
        _make_backbone()(
            pixels,
            valid_pixels,
            positions,
            extra=None,
            time_mask=time_mask,
        )


def test_input_channel_count_must_match_configured_input_dim() -> None:
    pixels, valid_pixels, positions = _make_inputs(num_channels=3)

    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        _make_backbone(input_dim=4)(
            pixels, valid_pixels, positions, extra=None
        )
