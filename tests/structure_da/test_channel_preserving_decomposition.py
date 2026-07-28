import torch

from methods.structure_da import SymmetricTimeKernelDecomposition
from models.pse import ChannelPreservingPixelSetEncoder


def test_channel_preserving_pse_output_decomposes_without_flattening() -> None:
    torch.manual_seed(15)
    batch, time, channels, pixels_per_parcel, feature_dim = 2, 4, 3, 5, 6
    pixels = torch.randn(batch, time, channels, pixels_per_parcel)
    valid_pixels = torch.tensor(
        [
            [[True, True, False, True, False]] * time,
            [[True, False, True, True, True]] * time,
        ]
    )
    positions = torch.tensor([0, 4, 13, 31])
    encoder = ChannelPreservingPixelSetEncoder(
        num_channels=channels,
        channel_feature_dim=feature_dim,
    )
    decomposition = SymmetricTimeKernelDecomposition()

    encoded = encoder(pixels, valid_pixels)
    components = decomposition(encoded, positions)
    reconstruction = components.trend + components.dynamics + components.residual

    assert encoded.shape == (batch, time, channels, feature_dim)
    assert components.trend.shape == encoded.shape
    assert components.dynamics.shape == encoded.shape
    assert components.residual.shape == encoded.shape
    torch.testing.assert_close(reconstruction, encoded, atol=1e-6, rtol=1e-5)
