from __future__ import annotations

import pytest
import torch

from methods.structure_da import RawTemporalRepresentation, SharedTrendStructureLTAE


def _ltae(**overrides) -> SharedTrendStructureLTAE:
    values = dict(
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
    values.update(overrides)
    return SharedTrendStructureLTAE(**values)


def _inputs():
    torch.manual_seed(3)
    trend = torch.randn(2, 4, 4)
    structure = torch.randn(2, 4, 4)
    positions = torch.tensor([[0.25, 2.5, 5.0, 9.75], [0.0, 1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, True, True, True], [False, False, False, False]])
    return trend, structure, positions, mask


def test_shared_ltae_returns_raw_representation_with_exact_concat() -> None:
    ltae = _ltae().eval()
    trend, structure, positions, mask = _inputs()

    raw = ltae(trend, structure, positions, mask)

    assert isinstance(raw, RawTemporalRepresentation)
    assert raw.trend_repr.shape == (2, 5)
    assert raw.structure_repr.shape == (2, 5)
    assert raw.fused_repr.shape == (2, 10)
    assert raw.positions_used.shape == (2, 4)
    torch.testing.assert_close(
        raw.fused_repr,
        torch.cat([raw.trend_repr, raw.structure_repr], dim=-1),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(raw.trend_repr[1]) == 0


def test_shared_ltae_uses_shared_projection_and_private_norms() -> None:
    ltae = _ltae()
    shared = ltae.shared_ltae
    assert shared.trend_input_projection is shared.structure_input_projection
    assert shared.trend_input_norm is not shared.structure_input_norm
    assert shared.trend_output_norm is not shared.structure_output_norm
    assert hasattr(shared, "shared_time_encoder")
    assert hasattr(shared, "attention_heads")
    assert not hasattr(shared, "stems")


def test_shared_ltae_rejects_gamma_and_phase_inputs() -> None:
    ltae = _ltae()
    trend, structure, positions, mask = _inputs()
    with pytest.raises(TypeError):
        ltae(trend, structure, positions, mask, gamma=torch.linspace(0, 1, 4))
    with pytest.raises(TypeError):
        ltae(trend, structure, positions, mask, phase_valid=torch.ones(2, dtype=torch.bool))
