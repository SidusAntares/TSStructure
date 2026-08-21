import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from models.pse import PixelSetEncoder

from models.decomposition_benchmark import (
    CeemdanSeDecomposition,
    CeemdanSeDecompositionClassifier,
    DwtDecomposition,
    DwtDecompositionClassifier,
    DLinearDecompositionClassifier,
    DLinearSeriesDecomposition,
    IMPLEMENTED_BENCHMARK_MODELS,
    MicnDecompositionClassifier,
    MicnMultiScaleHybridDecomposition,
    PLANNED_BENCHMARK_MODELS,
    StlDecomposition,
    VmdDecomposition,
    VmdDecompositionClassifier,
    StlDecompositionClassifier,
    XPatchEmaDecomposition,
    XPatchEmaDecompositionClassifier,
    build_benchmark_model,
    component_normalization_dataset_name,
    is_benchmark_model,
    sample_entropy,
)


def _small_model():
    return DLinearDecompositionClassifier(
        kernel_size=3,
        input_dim=3,
        mlp1=[3, 4],
        pooling="mean_std",
        mlp2=[8, 8],
        with_extra=False,
        n_head=2,
        d_k=2,
        d_model=8,
        mlp3=[8, 4],
        dropout=0.0,
        mlp4=[4],
        num_classes=5,
        max_temporal_shift=10,
    )


def _batch(batch_size=4, length=7, channels=3, pixels_per_date=5):
    pixels = torch.randn(batch_size, length, channels, pixels_per_date)
    mask = torch.ones(batch_size, length, pixels_per_date)
    positions = torch.arange(length).unsqueeze(0).repeat(batch_size, 1)
    return {
        "pixels": pixels,
        "valid_pixels": mask,
        "positions": positions,
        "extra": torch.zeros(batch_size, 4),
    }


SMALL_MODEL_KWARGS = {
    "input_dim": 3,
    "mlp1": [3, 4],
    "pooling": "mean_std",
    "mlp2": [8, 8],
    "with_extra": False,
    "n_head": 2,
    "d_k": 2,
    "d_model": 8,
    "mlp3": [8, 4],
    "dropout": 0.0,
    "mlp4": [4],
    "num_classes": 5,
    "max_temporal_shift": 10,
}


def _assert_shared_pse_and_independent_ltaes(model, component_names):
    assert tuple(model.temporal_encoders.keys()) == tuple(component_names)
    assert len({id(encoder) for encoder in model.temporal_encoders.values()}) == len(component_names)
    assert sum(isinstance(module, PixelSetEncoder) for module in model.modules()) == 1


def _assert_shift_equivalence(model, batch, shifts=(-2, 0, 3)):
    model.fit_component_normalizers([batch], device="cpu")
    model.eval()
    with torch.no_grad():
        expected = torch.stack(
            [
                model(
                    batch["pixels"],
                    batch["valid_pixels"],
                    batch["positions"] + shift,
                    batch["extra"],
                )
                for shift in shifts
            ],
            dim=1,
        )
        actual = model.forward_shift_candidates(
            batch["pixels"],
            batch["valid_pixels"],
            batch["positions"],
            batch["extra"],
            shifts,
        )
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_frozen_benchmark_plan_contains_thirteen_configs():
    assert len(PLANNED_BENCHMARK_MODELS) == 13
    assert PLANNED_BENCHMARK_MODELS[0] == "raw_timematch"
    assert "dlinear_decomp" in PLANNED_BENCHMARK_MODELS
    assert "ceemdan_se" in PLANNED_BENCHMARK_MODELS
    assert "etsformer" in PLANNED_BENCHMARK_MODELS


def test_dlinear_decomposition_reconstructs_without_changing_time_axis():
    batch = _batch()
    decomposition = DLinearSeriesDecomposition(kernel_size=3)
    components = decomposition(
        batch["pixels"], batch["valid_pixels"], batch["positions"]
    )

    assert [component.name for component in components] == ["trend", "seasonal"]
    trend, seasonal = components
    assert trend.pixels.shape == batch["pixels"].shape
    assert seasonal.pixels.shape == batch["pixels"].shape
    assert torch.equal(trend.positions, batch["positions"])
    assert torch.equal(seasonal.positions, batch["positions"])
    assert torch.allclose(trend.pixels + seasonal.pixels, batch["pixels"], atol=1e-6)


def test_dlinear_uses_one_shared_pse_but_independent_ltaes():
    model = _small_model()
    _assert_shared_pse_and_independent_ltaes(model, ("trend", "seasonal"))


def test_benchmark_component_normalization_dataset_is_always_source():
    model = _small_model()
    assert component_normalization_dataset_name(
        model, source_name="source/train", training_name="target/train"
    ) == "source/train"
    assert component_normalization_dataset_name(
        nn.Linear(2, 2), source_name="source/train", training_name="target/train"
    ) == "target/train"


def test_target_forward_cannot_update_source_fitted_component_statistics():
    model = _small_model()
    source = _batch()
    target = _batch()
    target["pixels"] = 100.0 * target["pixels"] + 500.0
    model.fit_component_normalizers([source], device="cpu")
    source_stats = {
        name: (normalizer.mean.clone(), normalizer.std.clone())
        for name, normalizer in model.component_normalizers.items()
    }
    model.eval()
    with torch.no_grad():
        model(
            target["pixels"],
            target["valid_pixels"],
            target["positions"],
            target["extra"],
        )
    for name, normalizer in model.component_normalizers.items():
        assert torch.equal(normalizer.mean, source_stats[name][0])
        assert torch.equal(normalizer.std, source_stats[name][1])


def test_component_stats_are_fitted_then_forward_runs():
    model = _small_model()
    batch = _batch()
    model.fit_component_normalizers([batch], device="cpu")

    assert model.component_normalizers_fitted
    logits = model(
        batch["pixels"],
        batch["valid_pixels"],
        batch["positions"],
        batch["extra"],
    )
    assert logits.shape == (batch["pixels"].shape[0], 5)


def test_shift_candidate_protocol_matches_repeated_forward_in_eval_mode():
    torch.manual_seed(7)
    model = _small_model()
    batch = _batch()
    model.fit_component_normalizers([batch], device="cpu")
    model.eval()
    shifts = [-2, 0, 3]

    with torch.no_grad():
        expected = torch.stack(
            [
                model(
                    batch["pixels"],
                    batch["valid_pixels"],
                    batch["positions"] + shift,
                    batch["extra"],
                )
                for shift in shifts
            ],
            dim=1,
        )
        actual = model.forward_shift_candidates(
            batch["pixels"],
            batch["valid_pixels"],
            batch["positions"],
            batch["extra"],
            shifts,
        )

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_registry_round2_implemented_and_future_methods_remain_planned_only():
    expected_implemented = (
        "raw_timematch",
        "dlinear_decomp",
        "micn_decomp",
        "xpatch_ema",
        "stl",
        "vmd",
        "dwt",
        "ceemdan_se",
    )
    future = {
        "timekan",
        "sscnn",
        "timemixer",
        "amd",
        "etsformer",
    }
    assert IMPLEMENTED_BENCHMARK_MODELS == expected_implemented
    assert all(is_benchmark_model(name) for name in expected_implemented)
    assert future.issubset(PLANNED_BENCHMARK_MODELS)
    assert not any(is_benchmark_model(name) for name in future)


def test_build_benchmark_model_constructs_all_round2_models():
    configs = {
        "micn_decomp": {"micn_conv_kernels": [3, 5]},
        "xpatch_ema": {"xpatch_ema_alpha": 0.3},
        "stl": {"stl_period": 3},
        "dwt": {"dwt_wavelet": "haar", "dwt_level": 2},
        "vmd": {"vmd_num_modes": 3},
        "ceemdan_se": {"ceemdan_trials": 2},
    }
    for name, kwargs in configs.items():
        model = build_benchmark_model(
            name,
            input_dim=3,
            num_classes=5,
            with_extra=False,
            **kwargs,
        )
        assert isinstance(model, nn.Module)


def test_micn_multi_scale_hybrid_decomposition_preserves_raw_components():
    batch = _batch(length=9)
    decomposition = MicnMultiScaleHybridDecomposition(conv_kernels=[2, 5])
    components = decomposition(batch["pixels"], batch["valid_pixels"], batch["positions"])

    assert decomposition.decomposition_kernels == (3, 5)
    assert [component.name for component in components] == ["trend", "seasonal"]
    trend, seasonal = components
    assert trend.pixels.shape == batch["pixels"].shape
    assert seasonal.pixels.shape == batch["pixels"].shape
    assert torch.equal(trend.mask, batch["valid_pixels"])
    assert torch.equal(seasonal.mask, batch["valid_pixels"])
    assert torch.equal(trend.positions, batch["positions"])
    assert torch.equal(seasonal.positions, batch["positions"])
    assert torch.allclose(trend.pixels + seasonal.pixels, batch["pixels"], atol=1e-6)


def test_micn_classifier_uses_additive_logits_and_shift_protocol():
    model = MicnDecompositionClassifier(conv_kernels=[3, 5], **SMALL_MODEL_KWARGS)
    _assert_shared_pse_and_independent_ltaes(model, ("trend", "seasonal"))
    assert model.fusion_mode == "additive_logits"
    assert set(model.decoders.keys()) == {"trend", "seasonal"}
    assert model.final_decoder is None
    _assert_shift_equivalence(model, _batch(length=9))


def test_xpatch_ema_uses_official_alpha_recurrence_and_reconstructs():
    pixels = torch.tensor([0.0, 10.0, 10.0]).reshape(1, 3, 1, 1)
    mask = torch.ones(1, 3, 1)
    positions = torch.tensor([[2, 9, 20]])
    decomposition = XPatchEmaDecomposition(alpha=0.25)
    trend, seasonal = decomposition(pixels, mask, positions)

    assert [trend.name, seasonal.name] == ["trend", "seasonal"]
    assert torch.allclose(trend.pixels.flatten(), torch.tensor([0.0, 2.5, 4.375]))
    assert torch.allclose(trend.pixels + seasonal.pixels, pixels, atol=1e-6)
    assert torch.equal(trend.positions, positions)
    assert torch.equal(seasonal.positions, positions)


def test_xpatch_classifier_concats_embeddings_before_one_classifier():
    model = XPatchEmaDecompositionClassifier(alpha=0.3, **SMALL_MODEL_KWARGS)
    batch = _batch(length=9)
    model.fit_component_normalizers([batch], device="cpu")
    model.eval()
    logits, features = model(
        batch["pixels"],
        batch["valid_pixels"],
        batch["positions"],
        batch["extra"],
        return_feats=True,
    )
    _assert_shared_pse_and_independent_ltaes(model, ("trend", "seasonal"))
    assert model.fusion_mode == "concat_embeddings"
    assert len(model.decoders) == 0
    assert model.final_decoder is not None
    assert features.shape == (batch["pixels"].shape[0], 2 * SMALL_MODEL_KWARGS["mlp3"][-1])
    assert logits.shape == (batch["pixels"].shape[0], 5)
    _assert_shift_equivalence(model, batch)


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels is not installed")
def test_stl_reconstructs_valid_trajectories_and_preserves_metadata():
    batch = _batch(batch_size=2, length=12, channels=2, pixels_per_date=3)
    batch["valid_pixels"][:, :, -1] = 0
    decomposition = StlDecomposition(period=3)
    components = decomposition(batch["pixels"], batch["valid_pixels"], batch["positions"])

    assert [component.name for component in components] == ["trend", "seasonal", "residual"]
    assert all(component.pixels.shape == batch["pixels"].shape for component in components)
    assert all(torch.equal(component.mask, batch["valid_pixels"]) for component in components)
    assert all(torch.equal(component.positions, batch["positions"]) for component in components)
    reconstructed = sum(component.pixels for component in components)
    assert torch.allclose(reconstructed, batch["pixels"], atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("statsmodels") is None, reason="statsmodels is not installed")
def test_stl_classifier_has_three_independent_branches_and_concat_fusion():
    model = StlDecompositionClassifier(period=3, **SMALL_MODEL_KWARGS)
    batch = _batch(length=12)
    names = ("trend", "seasonal", "residual")
    _assert_shared_pse_and_independent_ltaes(model, names)
    assert len(model.component_normalizers) == 3
    assert model.fusion_mode == "concat_embeddings"
    assert len(model.decoders) == 0
    model.fit_component_normalizers([batch], device="cpu")
    _, features = model(
        batch["pixels"], batch["valid_pixels"], batch["positions"], batch["extra"], return_feats=True
    )
    assert features.shape[1] == 3 * SMALL_MODEL_KWARGS["mlp3"][-1]
    _assert_shift_equivalence(model, batch)


@pytest.mark.skipif(importlib.util.find_spec("pywt") is None, reason="PyWavelets is not installed")
def test_dwt_has_stable_multiscale_order_lengths_and_real_support_positions():
    batch = _batch(batch_size=2, length=9, channels=2, pixels_per_date=3)
    batch["positions"] = torch.tensor(
        [[1, 4, 8, 15, 16, 23, 42, 60, 61], [3, 9, 10, 18, 33, 34, 49, 70, 90]]
    )
    decomposition = DwtDecomposition(wavelet="haar", level=2)
    components = decomposition(batch["pixels"], batch["valid_pixels"], batch["positions"])

    assert [component.name for component in components] == ["approximation_2", "detail_2", "detail_1"]
    assert len({component.pixels.shape[1] for component in components}) > 1
    for component in components:
        assert component.pixels.shape[0] == batch["pixels"].shape[0]
        assert component.pixels.shape[2:] == batch["pixels"].shape[2:]
        assert component.positions.shape == component.pixels.shape[:2]
        assert component.mask.shape == (
            component.pixels.shape[0], component.pixels.shape[1], component.pixels.shape[3]
        )
        assert torch.all(component.positions[:, 1:] >= component.positions[:, :-1])
        assert torch.all(component.positions >= batch["positions"].min(dim=1, keepdim=True).values)
        assert torch.all(component.positions <= batch["positions"].max(dim=1, keepdim=True).values)

    reconstructed = decomposition.reconstruct(components)
    assert reconstructed.shape == batch["pixels"].shape
    assert torch.allclose(reconstructed, batch["pixels"], atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("pywt") is None, reason="PyWavelets is not installed")
def test_dwt_classifier_has_one_ltae_per_scale_and_concat_shift_protocol():
    model = DwtDecompositionClassifier(wavelet="haar", level=2, **SMALL_MODEL_KWARGS)
    batch = _batch(length=12)
    names = ("approximation_2", "detail_2", "detail_1")
    _assert_shared_pse_and_independent_ltaes(model, names)
    assert model.fusion_mode == "concat_embeddings"
    assert len(model.decoders) == 0
    model.fit_component_normalizers([batch], device="cpu")
    _, features = model(
        batch["pixels"], batch["valid_pixels"], batch["positions"], batch["extra"], return_feats=True
    )
    assert features.shape[1] == 3 * SMALL_MODEL_KWARGS["mlp3"][-1]
    _assert_shift_equivalence(model, batch)


@pytest.mark.skipif(importlib.util.find_spec("pywt") is None, reason="PyWavelets is not installed")
def test_dwt_non_haar_wavelet_keeps_support_positions_monotonic_and_reconstructs():
    batch = _batch(batch_size=1, length=32, channels=1, pixels_per_date=2)
    batch["positions"] = torch.cumsum(
        torch.arange(1, 33, dtype=torch.long).unsqueeze(0), dim=1
    )
    decomposition = DwtDecomposition(wavelet="db2", level=2)
    components = decomposition(
        batch["pixels"], batch["valid_pixels"], batch["positions"]
    )
    assert all(
        torch.all(component.positions[:, 1:] >= component.positions[:, :-1])
        for component in components
    )
    assert torch.allclose(
        decomposition.reconstruct(components), batch["pixels"], atol=1e-5, rtol=1e-5
    )


def test_train_cli_exposes_round2_models_and_decomposition_parameters():
    help_text = Path("train.py").read_text(encoding="utf-8")
    for model_name in ("micn_decomp", "xpatch_ema", "stl", "vmd", "dwt", "ceemdan_se"):
        assert model_name in help_text
    for argument in (
        "--micn_conv_kernels",
        "--xpatch_ema_alpha",
        "--stl_period",
        "--dwt_wavelet",
        "--dwt_level",
        "--vmd_num_modes",
        "--vmd_alpha",
        "--ceemdan_trials",
        "--ceemdan_epsilon",
        "--ceemdan_sampen_m",
        "--ceemdan_sampen_r_ratio",
        "--ceemdan_se_threshold_factor",
    ):
        assert argument in help_text


def test_round2_smoke_script_has_fail_fast_source_only_runs_and_dependency_checks():
    script_path = Path("scripts/smoke_decomposition_benchmark_round2.sh")
    script = script_path.read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert "statsmodels" in script
    assert "pywt" in script
    for model_name in (
        "dlinear_decomp",
        "micn_decomp",
        "xpatch_ema",
        "stl",
        "dwt",
    ):
        assert model_name in script
    assert "--epochs 1" in script
    assert "--progress_bar off" in script


def _fake_vmd(signal, alpha, tau, K, DC, init, tol):
    import numpy as np

    signal = np.asarray(signal, dtype=np.float64)
    # Deliberately return frequencies out of order. The adapter must sort them.
    weights = np.arange(1, K + 1, dtype=np.float64)
    weights = weights / weights.sum()
    modes = np.stack([weight * signal for weight in weights], axis=0)
    omega = np.stack(
        [np.linspace(0.0, 1.0, K), np.arange(K, 0, -1, dtype=np.float64)],
        axis=0,
    )
    return modes, np.zeros_like(modes), omega


def test_vmd_preserves_time_axis_sorts_modes_and_reconstructs(monkeypatch):
    import numpy as np

    batch = _batch(batch_size=2, length=9, channels=2, pixels_per_date=3)
    batch["positions"] = torch.tensor(
        [[1, 4, 8, 15, 16, 23, 42, 60, 61], [3, 9, 10, 18, 33, 34, 49, 70, 90]]
    )
    decomposition = VmdDecomposition(num_modes=3)
    monkeypatch.setattr(decomposition, "_vmd_function", lambda: _fake_vmd)
    components = decomposition(
        batch["pixels"], batch["valid_pixels"], batch["positions"]
    )

    assert [component.name for component in components] == ["mode_1", "mode_2", "mode_3"]
    assert all(component.pixels.shape == batch["pixels"].shape for component in components)
    assert all(torch.equal(component.positions, batch["positions"]) for component in components)
    assert all(torch.equal(component.mask, batch["valid_pixels"]) for component in components)
    assert torch.allclose(decomposition.reconstruct(components), batch["pixels"], atol=1e-6)

    # Fake final frequencies are descending, so the first returned branch must be
    # the original highest-index (lowest final-frequency after sorting) weighted mode.
    expected_first_weight = 3.0 / 6.0
    assert torch.allclose(
        components[0].pixels, expected_first_weight * batch["pixels"], atol=1e-6
    )


def test_vmd_classifier_uses_independent_ltaes_additive_logits_and_shift_protocol(monkeypatch):
    model = VmdDecompositionClassifier(num_modes=3, **SMALL_MODEL_KWARGS)
    monkeypatch.setattr(model.decomposer, "_vmd_function", lambda: _fake_vmd)
    names = ("mode_1", "mode_2", "mode_3")
    _assert_shared_pse_and_independent_ltaes(model, names)
    assert model.fusion_mode == "additive_logits"
    assert set(model.decoders.keys()) == set(names)
    assert model.final_decoder is None
    _assert_shift_equivalence(model, _batch(length=9))


def test_sample_entropy_handles_constant_and_irregular_sequences():
    import numpy as np

    assert sample_entropy(np.ones(12), m=2, r_ratio=0.2) == 0.0
    value = sample_entropy(
        np.array([0.0, 1.0, 0.2, 1.4, -0.3, 0.8, -1.1, 0.7, 0.0, 1.2]),
        m=2,
        r_ratio=0.2,
    )
    assert np.isfinite(value) or np.isinf(value)
    assert value >= 0.0


def test_ceemdan_se_uses_adaptive_entropy_reconstruction_without_fixed_imf_slots(monkeypatch):
    import numpy as np
    import models.decomposition_benchmark.ceemdan as ceemdan_module

    decomposition = CeemdanSeDecomposition(
        trials=2,
        sampen_m=2,
        sampen_r_ratio=0.2,
        threshold_factor=0.5,
    )
    original = np.linspace(-1.0, 1.0, 9)
    components = np.stack(
        [0.5 * original, 0.3 * original, 0.2 * original], axis=0
    )
    entropy_values = iter([1.0, 0.1])
    monkeypatch.setattr(
        ceemdan_module,
        "sample_entropy",
        lambda sequence, m, r_ratio: next(entropy_values),
    )
    high, low = decomposition._group_components(components, original)

    # mean entropy = .55, threshold = .275: IMF1 is high, IMF2 and residue are low.
    assert np.allclose(high, 0.5 * original)
    assert np.allclose(low, 0.5 * original)
    assert np.allclose(high + low, original)
    assert decomposition.component_names == ("high_frequency", "low_frequency")


class _FakeCeemdan:
    def __init__(self):
        self.seed = None

    def noise_seed(self, seed):
        self.seed = seed

    def ceemdan(self, signal, max_imf=-1, progress=False):
        import numpy as np

        signal = np.asarray(signal, dtype=np.float64)
        return np.stack([0.6 * signal, 0.25 * signal, 0.15 * signal], axis=0)


def test_ceemdan_se_preserves_time_axis_reconstructs_and_uses_two_fixed_groups(monkeypatch):
    import models.decomposition_benchmark.ceemdan as ceemdan_module

    batch = _batch(batch_size=2, length=12, channels=2, pixels_per_date=3)
    decomposition = CeemdanSeDecomposition(trials=2, noise_seed=7)
    monkeypatch.setattr(decomposition, "_make_ceemdan", lambda: _FakeCeemdan())
    # Keep the test independent of finite-length SampEn edge cases; grouping behavior
    # itself is tested separately above.
    monkeypatch.setattr(
        ceemdan_module,
        "sample_entropy",
        lambda sequence, m, r_ratio: 1.0,
    )
    components = decomposition(
        batch["pixels"], batch["valid_pixels"], batch["positions"]
    )

    assert [component.name for component in components] == ["high_frequency", "low_frequency"]
    assert all(component.pixels.shape == batch["pixels"].shape for component in components)
    assert all(torch.equal(component.positions, batch["positions"]) for component in components)
    assert all(torch.equal(component.mask, batch["valid_pixels"]) for component in components)
    assert torch.allclose(decomposition.reconstruct(components), batch["pixels"], atol=1e-6)


def test_ceemdan_classifier_uses_independent_ltaes_additive_logits_and_shift_protocol(monkeypatch):
    import models.decomposition_benchmark.ceemdan as ceemdan_module

    model = CeemdanSeDecompositionClassifier(trials=2, **SMALL_MODEL_KWARGS)
    monkeypatch.setattr(model.decomposer, "_make_ceemdan", lambda: _FakeCeemdan())
    monkeypatch.setattr(
        ceemdan_module,
        "sample_entropy",
        lambda sequence, m, r_ratio: 1.0,
    )
    names = ("high_frequency", "low_frequency")
    _assert_shared_pse_and_independent_ltaes(model, names)
    assert model.fusion_mode == "additive_logits"
    assert set(model.decoders.keys()) == set(names)
    assert model.final_decoder is None
    _assert_shift_equivalence(model, _batch(length=12))
