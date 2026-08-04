from __future__ import annotations

import pytest
import torch
from torch import nn

from models.ltae import ContinuousTime2Vec, TrendStructureSharedLTAE


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
    encoder = ContinuousTime2Vec(5, time_scale=10.0)
    output = encoder(torch.tensor([[0.25, 3.5, 9.75]])).sum()
    output.backward()

    for name in ("linear_weight", "linear_bias", "frequencies", "phase"):
        gradient = getattr(encoder, name).grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


def test_continuous_time2vec_supports_cpu_autocast() -> None:
    encoder = ContinuousTime2Vec(5, time_scale=10.0)
    positions = torch.tensor([[0.25, 3.5, 9.75]])

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


def _shared_ltae() -> TrendStructureSharedLTAE:
    return TrendStructureSharedLTAE(
        in_channels=4,
        n_head=2,
        d_k=2,
        n_neurons=(8, 5),
        dropout=0.0,
        d_model=8,
        time_reference=0.0,
        time_scale=10.0,
        max_initial_frequency=4.0,
    )


def test_shared_ltae_has_private_stems_norms_and_one_shared_body() -> None:
    model = _shared_ltae()

    assert model.component_names == ("trend", "structure")
    assert set(model.stems) == {"trend", "structure"}
    assert model.stems["trend"] is not model.stems["structure"]
    assert set(model.output_norms) == {"trend", "structure"}
    assert model.output_norms["trend"] is not model.output_norms["structure"]
    assert isinstance(model.time_encoder, ContinuousTime2Vec)
    assert hasattr(model, "attention_heads")
    assert hasattr(model, "shared_projection")
    assert not any(isinstance(module, nn.Embedding) for module in model.modules())
    assert not hasattr(model, "residual_stem")
    assert not hasattr(model, "dynamics_stem")


def test_shared_ltae_outputs_expected_shape_and_zero_for_empty_sample() -> None:
    torch.manual_seed(3)
    model = _shared_ltae().eval()
    trend = torch.randn(2, 4, 4)
    structure = torch.randn(2, 4, 4)
    positions = torch.tensor([[0.25, 2.5, 5.0, 9.75], [0.0, 1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, True, True, True], [False, False, False, False]])

    trend_output, structure_output = model(
        trend, structure, positions, time_mask=mask
    )

    assert trend_output.shape == structure_output.shape == (2, 5)
    assert torch.count_nonzero(trend_output[1]) == 0
    assert torch.count_nonzero(structure_output[1]) == 0
    assert torch.isfinite(trend_output).all()
    assert torch.isfinite(structure_output).all()


def test_shared_ltae_computes_one_identical_time_encoding_for_both_branches(
    monkeypatch,
) -> None:
    model = _shared_ltae().eval()
    trend = torch.randn(2, 4, 4)
    structure = torch.randn(2, 4, 4)
    positions = torch.tensor([0.0, 2.5, 5.0, 10.0])
    calls = []
    original_forward = model.time_encoder.forward

    def capture(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(model.time_encoder, "forward", capture)
    model(trend, structure, positions)

    assert len(calls) == 1


def test_structure_stem_result_does_not_depend_on_trend_input() -> None:
    model = _shared_ltae().eval()
    structure = torch.randn(2, 4, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    safe_structure = torch.where(mask.unsqueeze(-1), structure, torch.zeros_like(structure))

    before = model.stems["structure"](safe_structure)
    _ = model.stems["trend"](torch.randn(2, 4, 4) * 1000.0)
    after = model.stems["structure"](safe_structure)

    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


def test_shared_ltae_uses_exactly_one_attention_and_projection_module() -> None:
    model = _shared_ltae()
    attention_modules = [
        module for module in model.modules() if module.__class__.__name__ == "MultiHeadAttention"
    ]
    assert attention_modules == [model.attention_heads]
    assert isinstance(model.shared_projection, nn.Sequential)


def test_shared_ltae_rejects_mismatched_components() -> None:
    model = _shared_ltae()
    with pytest.raises(ValueError, match="identical shape"):
        model(
            torch.randn(2, 4, 4),
            torch.randn(2, 3, 4),
            torch.linspace(0.0, 10.0, 4),
        )
