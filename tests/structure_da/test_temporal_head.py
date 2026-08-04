from __future__ import annotations

import torch
from torch import nn

from methods.structure_da.temporal_head import ShapeFeatureEncoder, ShapeFeatureOutput

def test_shape_feature_encoder_has_exact_required_network_and_no_phase_encoder() -> None:
    encoder = ShapeFeatureEncoder(
        num_shape_basis=4,
        attribute_projection_dim=3,
        output_dim=7,
        hidden_dim=5,
        dropout=0.2,
    )

    assert [type(layer) for layer in encoder.network] == [
        nn.Linear,
        nn.GELU,
        nn.Dropout,
        nn.Linear,
    ]
    assert encoder.network[0].in_features == 12
    assert encoder.network[-1].out_features == 7
    assert not hasattr(encoder, "phase_encoder")


def test_shape_feature_encoder_zeroes_invalid_rows_after_biases() -> None:
    encoder = ShapeFeatureEncoder(4, 3, output_dim=7, hidden_dim=5, dropout=0.0)
    with torch.no_grad():
        encoder.network[0].bias.fill_(3.0)
        encoder.network[-1].bias.fill_(2.0)
    coordinates = torch.randn(2, 4, 3)
    valid = torch.tensor([True, False])

    output = encoder(coordinates, valid)

    assert isinstance(output, ShapeFeatureOutput)
    assert output.feature.shape == (2, 7)
    assert output.valid is valid
    torch.testing.assert_close(output.feature[1], torch.zeros(7), atol=0, rtol=0)
    assert not hasattr(output, "phase_embedding")
    assert not hasattr(output, "joint_embedding")


def test_shape_feature_encoder_backpropagates_to_coordinates_and_all_parameters() -> None:
    encoder = ShapeFeatureEncoder(4, 3, output_dim=7, hidden_dim=5, dropout=0.0)
    coordinates = torch.randn(2, 4, 3, requires_grad=True)

    output = encoder(coordinates, torch.tensor([True, True]))
    output.feature.square().sum().backward()

    assert coordinates.grad is not None and torch.isfinite(coordinates.grad).all()
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
