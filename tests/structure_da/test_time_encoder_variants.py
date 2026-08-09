from __future__ import annotations

import torch

from models.layers import get_positional_encoding
from models.ltae import (
    TimeMatchFixedSinusoidal,
    TrendStructureSharedLTAE,
)
from methods.structure_da.full_model import TSStructureModel


def test_timematch_fixed_sinusoidal_matches_official_integer_day_table() -> None:
    d_model = 8
    encoder = TimeMatchFixedSinusoidal(
        d_model,
        time_reference=0.0,
        time_scale=1.0,
        calendar_scale_days=365.0,
        position_offset_days=100.0,
        period=1000.0,
    ).eval()
    day = torch.tensor([0.0, 1.0, 100.0, 364.0])
    positions = (day / 365.0).unsqueeze(0)
    actual = encoder(positions)

    official = get_positional_encoding(565, d_model, T=1000.0)
    expected = official[(day.long() + 100)].unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_timematch_fixed_sinusoidal_is_fixed_and_supports_fractional_days() -> None:
    encoder = TimeMatchFixedSinusoidal(
        8,
        time_scale=1.0,
        calendar_scale_days=365.0,
        position_offset_days=100.0,
        period=1000.0,
    ).eval()
    assert list(encoder.parameters()) == []

    positions = torch.tensor([[-21.5 / 365.0, 120.25 / 365.0, 1.0 + 12.0 / 365.0]])
    output = encoder(positions)
    assert output.shape == (1, 3, 8)
    assert torch.isfinite(output).all().item()


def test_shared_ltae_can_select_timematch_fixed_sinusoidal() -> None:
    module = TrendStructureSharedLTAE(
        in_channels=4,
        n_head=2,
        d_k=2,
        n_neurons=(8, 5),
        dropout=0.0,
        d_model=8,
        time_reference=0.0,
        time_scale=1.0,
        time_encoder_type="timematch_fixed_sinusoidal",
        timematch_pe_period=1000.0,
        timematch_pe_max_shift=100.0,
        calendar_scale_days=365.0,
    ).eval()
    assert module.time_encoder_type == "timematch_fixed_sinusoidal"
    assert isinstance(module.shared_time_encoder, TimeMatchFixedSinusoidal)

    trend = torch.randn(2, 4, 4)
    structure = torch.randn(2, 4, 4)
    positions = torch.tensor(
        [[0.0, 0.25, 0.5, 0.75], [0.1, 0.2, 0.3, 0.4]],
        dtype=torch.float32,
    )
    mask = torch.ones(2, 4, dtype=torch.bool)
    trend_repr, structure_repr = module(
        trend, structure, positions, time_mask=mask
    )
    assert trend_repr.shape == (2, 5)
    assert structure_repr.shape == (2, 5)


def test_tsstructure_model_propagates_fixed_time_encoder_choice() -> None:
    model = TSStructureModel(
        num_classes=3,
        input_dim=2,
        mlp1=(2, 4, 4),
        mlp2=(8, 4),
        time_reference=0.0,
        time_scale=365.0,
        trend_num_basis=4,
        structure_num_basis=4,
        canonical_grid_size=5,
        roughness_grid_size=64,
        n_head=1,
        d_k=2,
        d_model=8,
        ltae_mlp=(8, 4),
        dropout=0.0,
        classifier_hidden=(4,),
        time_encoder_type="timematch_fixed_sinusoidal",
        timematch_pe_period=1000.0,
        timematch_pe_max_shift=100.0,
    )
    encoder = model.temporal_module.raw_encoder.shared_ltae.shared_time_encoder
    assert isinstance(encoder, TimeMatchFixedSinusoidal)
    assert list(encoder.parameters()) == []
