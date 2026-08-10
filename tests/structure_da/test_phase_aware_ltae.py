from __future__ import annotations

import pytest
import torch

from models.ltae import ContinuousTime2Vec


def test_continuous_time2vec_shape_mask_dtype_and_fractional_positions() -> None:
    encoder = ContinuousTime2Vec(
        6, time_reference=0.0, time_scale=10.0, max_initial_frequency=4.0
    ).double()
    positions = torch.tensor([[0.0, 0.25, 5.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    output = encoder(positions, time_mask=mask)

    assert output.shape == (2, 3, 6)
    assert output.dtype == torch.float64
    assert output.device == positions.device
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[0, 2]) == 0
    assert not torch.equal(output[0, 0], output[0, 1])


def test_continuous_time2vec_parameters_receive_gradients() -> None:
    encoder = ContinuousTime2Vec(5, time_reference=0.0, time_scale=1.0)
    output = encoder(torch.tensor([[0.25, 0.5, 0.975]])).sum()
    output.backward()

    for name in ("linear_weight", "linear_bias", "frequencies", "phase"):
        gradient = getattr(encoder, name).grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


def test_continuous_time2vec_supports_cpu_autocast() -> None:
    encoder = ContinuousTime2Vec(5, time_reference=0.0, time_scale=1.0)
    positions = torch.tensor([[0.25, 0.5, 0.975]])

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = encoder(positions)

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_dim": 1},
        {"output_dim": 4, "time_reference": float("nan")},
        {"output_dim": 4, "time_scale": 0.0},
        {"output_dim": 4, "max_initial_frequency": 0.5},
    ],
)
def test_continuous_time2vec_rejects_invalid_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        ContinuousTime2Vec(**kwargs)


def test_continuous_time2vec_rejects_valid_positions_outside_normalized_range() -> None:
    encoder = ContinuousTime2Vec(4, time_reference=10.0, time_scale=20.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        encoder(torch.tensor([[9.0, 15.0]]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        encoder(torch.tensor([[15.0, 31.0]]))


def test_continuous_time2vec_is_continuous_without_integer_lookup() -> None:
    encoder = ContinuousTime2Vec(
        4, time_reference=0.0, time_scale=1.0, max_initial_frequency=3.0
    )
    positions = torch.tensor([[0.5000, 0.5001]])

    output = encoder(positions)

    assert torch.isfinite(output).all()
    assert not torch.equal(output[:, 0], output[:, 1])
    assert (output[:, 1] - output[:, 0]).abs().max() < 1e-2
    expected_periodic = torch.sin(
        positions.unsqueeze(-1) * encoder.frequencies + encoder.phase
    )
    torch.testing.assert_close(output[..., 1:], expected_periodic)



def test_removed_dual_trend_structure_ltae_symbol() -> None:
    import models.ltae as module
    assert not hasattr(module, "TrendStructureSharedLTAE")
