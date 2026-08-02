import torch

from models.pse import PixelSetEncoder


def _encoder(*, with_extra: bool) -> PixelSetEncoder:
    mlp2 = [128, 128]
    if with_extra:
        mlp2 = [mlp2[0] + 4, mlp2[1]]
    return PixelSetEncoder(
        input_dim=10,
        mlp1=[10, 32, 64],
        pooling="mean_std",
        mlp2=mlp2,
        with_extra=with_extra,
        extra_size=4,
    )


def _inputs():
    torch.manual_seed(11)
    pixels = torch.randn(2, 5, 10, 7)
    valid_pixels = torch.tensor(
        [
            [[1, 1, 1, 0, 0, 1, 0]] * 5,
            [[1, 0, 1, 1, 0, 0, 1]] * 5,
        ],
        dtype=torch.float32,
    )
    extra = torch.randn(2, 4)
    return pixels, valid_pixels, extra


def test_original_pse_default_configuration_outputs_128_features() -> None:
    pixels, valid_pixels, _ = _inputs()
    encoder = _encoder(with_extra=False)

    output = encoder(pixels, valid_pixels, extra=None)

    assert output.shape == (2, 5, 128)
    assert encoder.output_dim == 128
    assert encoder.mlp1_dim == [10, 32, 64]
    assert encoder.mlp2_dim == [128, 128]
    assert encoder.pooling == "mean_std"


def test_original_pse_valid_pixel_mask_excludes_masked_values() -> None:
    pixels, valid_pixels, _ = _inputs()
    changed = pixels.clone()
    invalid = ~valid_pixels.bool().unsqueeze(2).expand_as(changed)
    changed[invalid] = torch.randn_like(changed[invalid]) * 1e6
    encoder = _encoder(with_extra=False).eval()

    expected = encoder(pixels, valid_pixels, extra=None)
    actual = encoder(changed, valid_pixels, extra=None)

    torch.testing.assert_close(actual, expected)


def test_original_pse_with_extra_uses_extra_features() -> None:
    pixels, valid_pixels, extra = _inputs()
    encoder = _encoder(with_extra=True).eval()

    first = encoder(pixels, valid_pixels, extra)
    second = encoder(pixels, valid_pixels, extra + 3.0)

    assert first.shape == (2, 5, 128)
    assert encoder.mlp2_dim == [132, 128]
    assert not torch.allclose(first, second)
