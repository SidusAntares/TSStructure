import pytest
import torch

from methods.structure_da import StructureBackbone, StructureBackboneOutput
from models.pse import ChannelPreservingPixelSetEncoder, PixelSetEncoder


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


def _make_backbone(num_channels: int = 4) -> StructureBackbone:
    return StructureBackbone(
        num_channels=num_channels,
        channel_feature_dim=8,
        pixel_hidden_dim=7,
    )


def test_default_mask_shapes_and_reconstruction() -> None:
    pixels, valid_pixels, positions = _make_inputs()
    backbone = _make_backbone()

    output = backbone(pixels, valid_pixels, positions)

    assert isinstance(output, StructureBackboneOutput)
    assert output.channel_tokens.shape == (2, 5, 4, 8)
    assert output.time_mask.shape == (2, 5)
    assert output.time_mask.dtype == torch.bool
    assert output.time_mask.device == pixels.device
    assert output.time_mask.all()
    for component in (
        output.decomposition.trend,
        output.decomposition.dynamics,
        output.decomposition.residual,
    ):
        assert component.shape == (2, 5, 4, 8)
    torch.testing.assert_close(
        output.decomposition.trend
        + output.decomposition.dynamics
        + output.decomposition.residual,
        output.channel_tokens,
        atol=1e-6,
        rtol=1e-5,
    )


def test_one_dimensional_numeric_time_mask_expands_to_batch() -> None:
    pixels, valid_pixels, positions = _make_inputs()
    time_mask = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])

    output = _make_backbone()(pixels, valid_pixels, positions, time_mask)

    expected = time_mask.bool().expand(2, -1)
    torch.testing.assert_close(output.time_mask, expected)


def test_partial_time_mask_only_zeros_decomposition_outputs() -> None:
    pixels, valid_pixels, positions = _make_inputs()
    backbone = _make_backbone()
    time_mask = torch.tensor(
        [[True, False, True, True, False], [True, True, False, True, True]]
    )
    expected_tokens = backbone.pixel_set_encoder(pixels, valid_pixels)

    output = backbone(pixels, valid_pixels, positions, time_mask)

    torch.testing.assert_close(output.channel_tokens, expected_tokens)
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
        reconstruction[time_mask], output.channel_tokens[time_mask]
    )


def test_changing_one_physical_channel_does_not_change_other_tokens() -> None:
    pixels, valid_pixels, positions = _make_inputs()
    changed = pixels.clone()
    changed[:, :, 2, :] += torch.linspace(-2.0, 3.0, pixels.shape[-1])
    backbone = _make_backbone().double().eval()

    original = backbone(
        pixels.double(), valid_pixels, positions
    ).channel_tokens
    actual = backbone(
        changed.double(), valid_pixels, positions
    ).channel_tokens

    unchanged_channels = torch.tensor([True, True, False, True])
    torch.testing.assert_close(
        actual[:, :, unchanged_channels],
        original[:, :, unchanged_channels],
        rtol=0,
        atol=1e-12,
    )
    assert not torch.allclose(actual[:, :, 2], original[:, :, 2])


def test_gradients_reach_pse_and_both_kernel_scales() -> None:
    pixels, valid_pixels, positions = _make_inputs()
    backbone = _make_backbone()

    output = backbone(pixels, valid_pixels, positions)
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


def test_backbone_uses_only_channel_preserving_pse() -> None:
    backbone = _make_backbone()

    assert isinstance(
        backbone.pixel_set_encoder, ChannelPreservingPixelSetEncoder
    )
    assert not any(
        isinstance(module, PixelSetEncoder) for module in backbone.modules()
    )


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
        _make_backbone()(pixels, valid_pixels, positions, time_mask)


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
        _make_backbone()(pixels, valid_pixels, positions, time_mask)


def test_input_channel_count_must_match_configured_num_channels() -> None:
    pixels, valid_pixels, positions = _make_inputs(num_channels=3)

    with pytest.raises(
        ValueError, match="pixels channel dimension must equal num_channels=4"
    ):
        _make_backbone(num_channels=4)(pixels, valid_pixels, positions)
