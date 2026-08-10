from __future__ import annotations

import pytest
import torch

from methods.structure_da import LatentTemporalLTAE, RawTemporalRepresentation


def _ltae(**overrides) -> LatentTemporalLTAE:
    values = dict(
        in_channels=4,
        n_head=2,
        d_k=2,
        n_neurons=(8, 5),
        dropout=0.0,
        d_model=8,
        max_initial_frequency=4.0,
    )
    values.update(overrides)
    return LatentTemporalLTAE(**values)


def _inputs():
    torch.manual_seed(3)
    latent = torch.randn(2, 4, 4)
    positions = torch.tensor([[0.025, 0.25, 0.5, 0.975], [0.0, 0.1, 0.2, 0.3]])
    mask = torch.tensor([[True, True, True, True], [False, False, False, False]])
    return latent, positions, mask


def test_single_ltae_returns_one_task_embedding() -> None:
    ltae = _ltae().eval()
    latent, positions, mask = _inputs()
    raw = ltae(latent, positions, mask)
    assert isinstance(raw, RawTemporalRepresentation)
    assert raw.fused_repr.shape == (2, 5)
    assert raw.positions_used.shape == (2, 4)
    assert torch.count_nonzero(raw.fused_repr[1]) == 0


def test_single_ltae_has_one_projection_and_norm_chain() -> None:
    ltae = _ltae()
    assert hasattr(ltae, "input_projection")
    assert hasattr(ltae, "input_norm")
    assert hasattr(ltae, "time_encoder")
    assert hasattr(ltae, "attention_heads")
    assert hasattr(ltae, "projection")
    assert hasattr(ltae, "output_norm")
    assert not hasattr(ltae, "trend_input_norm")
    assert not hasattr(ltae, "structure_input_norm")


def test_single_ltae_rejects_removed_ts_arguments() -> None:
    ltae = _ltae()
    latent, positions, mask = _inputs()
    with pytest.raises(TypeError):
        ltae(latent, positions, mask, gamma=torch.linspace(0, 1, 4))
