import pytest
import torch
from torch import nn

from models.pse import (
    ChannelPreservingPixelSetEncoder,
    PixelSetEncoder,
    _masked_mean_and_sample_std,
)


def _make_encoder(num_channels=10, channel_feature_dim=12):
    torch.manual_seed(0)
    return ChannelPreservingPixelSetEncoder(
        num_channels=num_channels,
        channel_feature_dim=channel_feature_dim,
        pixel_hidden_dim=9,
    )


def test_channel_preserving_pse_output_shape():
    encoder = _make_encoder(channel_feature_dim=13)
    pixels = torch.randn(2, 5, 10, 7)
    valid_pixels = torch.ones(2, 5, 7, dtype=torch.bool)

    output = encoder(pixels, valid_pixels)

    assert output.shape == (2, 5, 10, 13)
    assert encoder.num_channels == 10
    assert encoder.channel_feature_dim == 13
    assert encoder.output_dim == 13


def test_pixel_permutation_does_not_change_output():
    encoder = _make_encoder().eval()
    pixels = torch.randn(2, 3, 10, 7)
    valid_pixels = torch.rand(2, 3, 7) > 0.3
    valid_pixels[..., 0] = True
    permutation = torch.randperm(7)

    expected = encoder(pixels, valid_pixels)
    actual = encoder(pixels[..., permutation], valid_pixels[..., permutation])

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_masked_pixel_values_do_not_change_output():
    encoder = _make_encoder().eval()
    pixels = torch.randn(2, 3, 10, 7)
    valid_pixels = torch.tensor(
        [[[True, True, False, False, True, False, True]]] * 3 * 2
    ).reshape(2, 3, 7)
    changed = pixels.clone()
    masked = ~valid_pixels.unsqueeze(2).expand_as(pixels)
    changed[masked] = torch.randn_like(changed[masked]) * 1e6

    expected = encoder(pixels, valid_pixels)
    actual = encoder(changed, valid_pixels.float())

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_nonfinite_masked_pixel_values_do_not_change_output():
    encoder = _make_encoder(num_channels=3).eval()
    pixels = torch.randn(1, 2, 3, 4)
    valid_pixels = torch.tensor(
        [[[True, False, True, False], [False, True, False, True]]]
    )
    changed = pixels.clone()
    masked = ~valid_pixels.unsqueeze(2).expand_as(pixels)
    changed[masked] = float("nan")
    changed[0, 0, 0, 1] = float("inf")

    expected = encoder(pixels, valid_pixels)
    actual = encoder(changed, valid_pixels)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_single_valid_pixel_has_zero_std_and_finite_output():
    tokens = torch.randn(1, 2, 4, 3, 5)
    valid_pixels = torch.tensor([[[True, False, False, False], [True] * 4]])

    _, std = _masked_mean_and_sample_std(tokens, valid_pixels, eps=1e-8)

    assert torch.equal(std[:, 0], torch.zeros_like(std[:, 0]))

    encoder = _make_encoder(num_channels=3)
    pixels = torch.randn(1, 2, 3, 4)
    output = encoder(pixels, valid_pixels)
    assert torch.isfinite(output).all()


def test_all_invalid_date_raises_value_error():
    encoder = _make_encoder(num_channels=3)
    pixels = torch.randn(2, 3, 3, 4)
    valid_pixels = torch.ones(2, 3, 4, dtype=torch.bool)
    valid_pixels[1, 2] = False

    with pytest.raises(ValueError, match="at least one valid pixel"):
        encoder(pixels, valid_pixels)


def test_batch_items_are_independent():
    encoder = _make_encoder(num_channels=3).eval()
    pixels = torch.randn(2, 3, 3, 5)
    valid_pixels = torch.ones(2, 3, 5, dtype=torch.bool)
    expected = encoder(pixels, valid_pixels)[0].clone()
    changed = pixels.clone()
    changed[1] = torch.randn_like(changed[1]) * 1e5

    actual = encoder(changed, valid_pixels)[0]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not any(isinstance(module, nn.BatchNorm1d) for module in encoder.modules())
    assert not any(isinstance(module, nn.Dropout) for module in encoder.modules())


def test_channel_identity_shape_matches_output_features():
    encoder = _make_encoder(num_channels=4)

    assert encoder.channel_embedding.num_embeddings == 4
    assert encoder.channel_embedding.embedding_dim == encoder.channel_feature_dim


def test_selected_channel_output_depends_only_on_its_own_values():
    encoder = _make_encoder(num_channels=3)
    pixels = torch.randn(1, 2, 3, 4, requires_grad=True)
    valid_pixels = torch.ones(1, 2, 4, dtype=torch.bool)

    encoder(pixels, valid_pixels)[0, 1, 2].sum().backward()

    assert torch.count_nonzero(pixels.grad[0, 1, 2]).item() > 0
    assert torch.count_nonzero(pixels.grad[0, 1, :2]).item() == 0


def test_changing_another_channel_does_not_change_selected_token():
    encoder = _make_encoder(num_channels=3).eval()
    pixels = torch.randn(2, 3, 3, 4)
    valid_pixels = torch.ones(2, 3, 4, dtype=torch.bool)
    expected = encoder(pixels, valid_pixels)[:, :, 1].clone()
    changed = pixels.clone()
    changed[:, :, 2] = torch.randn_like(changed[:, :, 2]) * 1e6

    actual = encoder(changed, valid_pixels)[:, :, 1]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_pixel_encoder_matches_channels_when_identifiers_are_zero():
    encoder = _make_encoder(num_channels=4).eval()
    common_values = torch.randn(2, 3, 1, 5)
    pixels = common_values.expand(-1, -1, 4, -1).clone()
    valid_pixels = torch.ones(2, 3, 5, dtype=torch.bool)
    with torch.no_grad():
        encoder.channel_embedding.weight.zero_()

    output = encoder(pixels, valid_pixels)

    for channel in range(1, 4):
        torch.testing.assert_close(output[:, :, channel], output[:, :, 0])


def test_channel_identifier_is_added_exactly_after_pooling():
    encoder = _make_encoder(num_channels=3).eval()
    pixels = torch.randn(2, 3, 3, 5)
    valid_pixels = torch.ones(2, 3, 5, dtype=torch.bool)
    delta = torch.linspace(-0.3, 0.4, encoder.channel_feature_dim)
    with torch.no_grad():
        encoder.channel_embedding.weight.zero_()
    base_output = encoder(pixels, valid_pixels)
    with torch.no_grad():
        encoder.channel_embedding.weight[1].copy_(delta)

    identified_output = encoder(pixels, valid_pixels)

    torch.testing.assert_close(
        identified_output[:, :, 1] - base_output[:, :, 1],
        delta.reshape(1, 1, -1).expand(2, 3, -1),
    )
    torch.testing.assert_close(identified_output[:, :, 0], base_output[:, :, 0])
    torch.testing.assert_close(identified_output[:, :, 2], base_output[:, :, 2])


def test_obsolete_multivariate_context_modules_are_absent():
    encoder = _make_encoder(num_channels=3)

    assert not hasattr(encoder, "spectral_context_encoder")
    assert not hasattr(encoder, "pixel_token_encoder")


def test_legacy_pixel_set_encoder_output_shape_is_unchanged():
    encoder = PixelSetEncoder(
        input_dim=3,
        mlp1=[3, 4],
        pooling="mean_std",
        mlp2=[8, 6],
        with_extra=False,
    )
    pixels = torch.randn(2, 3, 3, 5)
    valid_pixels = torch.ones(2, 3, 5)

    output = encoder(pixels, valid_pixels, extra=None)

    assert output.shape == (2, 3, 6)


@pytest.mark.parametrize(
    ("pixels", "valid_pixels", "message"),
    [
        (torch.randn(2, 3, 4), torch.ones(2, 3, 4), "pixels must be 4D"),
        (torch.randn(2, 3, 4, 5), torch.ones(2, 3, 1, 5), "valid_pixels must be 3D"),
        (torch.randn(2, 3, 4, 5), torch.ones(2, 4, 5), "batch, time, and pixel"),
        (torch.randn(2, 3, 3, 5), torch.ones(2, 3, 5), "num_channels"),
        (
            torch.ones(2, 3, 4, 5, dtype=torch.int64),
            torch.ones(2, 3, 5),
            "floating point",
        ),
        (
            torch.randn(2, 3, 4, 5),
            torch.ones(2, 3, 5, dtype=torch.int64),
            "boolean or floating point",
        ),
    ],
)
def test_input_validation_rejects_invalid_shapes_and_dtypes(
    pixels, valid_pixels, message
):
    encoder = _make_encoder(num_channels=4)

    with pytest.raises((TypeError, ValueError), match=message):
        encoder(pixels, valid_pixels)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_valid_pixel_value_raises_value_error(nonfinite):
    encoder = _make_encoder(num_channels=3)
    pixels = torch.randn(1, 2, 3, 4)
    valid_pixels = torch.ones(1, 2, 4, dtype=torch.bool)
    pixels[0, 1, 2, 3] = nonfinite

    with pytest.raises(ValueError, match="valid pixel values must be finite"):
        encoder(pixels, valid_pixels)


@pytest.mark.parametrize(
    "invalid_mask_value",
    [0.5, 2.0, -1.0, float("nan"), float("inf")],
)
def test_float_mask_must_contain_only_finite_binary_values(invalid_mask_value):
    encoder = _make_encoder(num_channels=3)
    pixels = torch.randn(1, 2, 3, 4)
    valid_pixels = torch.ones(1, 2, 4)
    valid_pixels[0, 1, 2] = invalid_mask_value

    with pytest.raises(ValueError, match="finite 0/1 values"):
        encoder(pixels, valid_pixels)


def test_bool_and_equivalent_float_masks_match():
    encoder = _make_encoder(num_channels=3).eval()
    pixels = torch.randn(1, 2, 3, 4)
    bool_mask = torch.tensor(
        [[[True, False, True, True], [False, True, True, False]]]
    )

    bool_output = encoder(pixels, bool_mask)
    float_output = encoder(pixels, bool_mask.float())

    torch.testing.assert_close(float_output, bool_output, rtol=0, atol=0)
