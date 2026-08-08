from __future__ import annotations

import pytest
import torch

from methods.structure_da import (
    RegistrationGeometryOutput,
    SourceRegistrationPrototypeBank,
    TargetGeometryCache,
    evaluate_registration_geometry,
)
from methods.structure_da.temporal_srvf import TemporalSRVFExtractor


def _reg_extractor() -> TemporalSRVFExtractor:
    return TemporalSRVFExtractor(
        feature_dim=4,
        num_basis=6,
        canonical_grid_size=128,
        roughness_grid_size=256,
        smoothing_weight=1e-2,
        time_reference=0.0,
        time_scale=1.0,
        min_mean_support=0.0,
        min_dynamic_energy=0.0,
    )


def _inputs(batch: int = 2, length: int = 7):
    torch.manual_seed(3)
    trend = torch.randn(batch, length, 4)
    positions = torch.linspace(0, 1, length).unsqueeze(0).expand(batch, -1)
    mask = torch.ones(batch, length, dtype=torch.bool)
    return trend, positions, mask


def test_registration_geometry_has_k_reg_grid() -> None:
    extractor = _reg_extractor()
    trend, positions, mask = _inputs()
    output = evaluate_registration_geometry(trend, positions, mask, extractor)

    assert isinstance(output, RegistrationGeometryOutput)
    assert output.trend_srvf.shape == (2, 128, 4)
    assert output.trend_support.shape == (2, 128)
    assert output.trend_valid.shape == (2,)
    assert output.registration_grid.shape == (128,)
    assert output.registration_grid[0].item() == pytest.approx(0.0)
    assert output.registration_grid[-1].item() == pytest.approx(1.0)
    assert torch.isfinite(output.trend_srvf).all()


def test_registration_srvf_is_not_linear_interpolation_of_k_shape() -> None:
    # Same B-spline functional fit, evaluated at K_shape=64 vs K_reg=128: the
    # 128 grid must produce a genuinely different, finer curve.
    extractor_64 = TemporalSRVFExtractor(
        feature_dim=4,
        num_basis=6,
        canonical_grid_size=64,
        roughness_grid_size=256,
        smoothing_weight=1e-2,
        time_reference=0.0,
        time_scale=1.0,
        min_mean_support=0.0,
        min_dynamic_energy=0.0,
    )
    extractor_128 = _reg_extractor()
    trend, positions, mask = _inputs(batch=1)
    out64 = evaluate_registration_geometry(trend, positions, mask, extractor_64)
    out128 = evaluate_registration_geometry(trend, positions, mask, extractor_128)
    # A linear interpolation of the 64-point curve would lie exactly on the
    # spline sampled at those points; the 128 evaluation re-fits and re-samples.
    interp = torch.nn.functional.interpolate(
        out64.trend_srvf.transpose(1, 2), size=128, mode="linear", align_corners=True
    ).transpose(1, 2)
    # The re-evaluated curve is a valid smooth signal and differs from a naive
    # linear interpolation where the spline curvature is non-linear.
    assert not torch.allclose(out128.trend_srvf, interp, atol=1e-3)


def test_source_registration_prototype_bank_shapes() -> None:
    bank = SourceRegistrationPrototypeBank(
        trend_srvf=torch.zeros(3, 128, 4),
        trend_support=torch.ones(3, 128),
        class_counts=torch.tensor([5, 5, 5]),
        ready=torch.tensor([True, True, False]),
        registration_grid=torch.linspace(0, 1, 128),
    )
    assert bank.trend_srvf.shape == (3, 128, 4)
    assert bank.trend_support.shape == (3, 128)
    assert bank.ready_classes() == [0, 1]
    with pytest.raises(ValueError, match="trend_support"):
        SourceRegistrationPrototypeBank(
            trend_srvf=torch.zeros(3, 128, 4),
            trend_support=torch.ones(2, 128),
            class_counts=torch.tensor([5, 5, 5]),
            ready=torch.tensor([True, True, True]),
            registration_grid=torch.linspace(0, 1, 128),
        )


def test_target_geometry_cache_holds_no_labels() -> None:
    cache = TargetGeometryCache(
        sample_ids=torch.tensor([0, 1]),
        trend_srvf_reg=torch.zeros(2, 128, 4),
        trend_support_reg=torch.zeros(2, 128),
        trend_valid=torch.ones(2, dtype=torch.bool),
        structure_srvf_shape=torch.zeros(2, 64, 4),
        structure_support_shape=torch.zeros(2, 64),
        structure_valid=torch.ones(2, dtype=torch.bool),
        registration_grid=torch.linspace(0, 1, 128),
        shape_grid=torch.linspace(0, 1, 64),
    )
    fields = set(cache.__dataclass_fields__)
    assert not (fields & {"labels", "logits", "classifier_logits", "pseudo_label", "z_shape"})
