import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "scripts" / "summarize_fourier_recon_modes_at1_dk1.py"
LAUNCHER_PATH = ROOT / "scripts" / "run_fourier_recon_modes_at1_dk1_seed1.sh"
MODES = (9, 11, 13, 15, 17, 19)


def _load_summary_module():
    spec = importlib.util.spec_from_file_location("fourier_mode_summary", SUMMARY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("num_modes", MODES)
def test_mode_controls_coefficient_count_and_keeps_original_length(num_modes):
    from models.fourier_reconstruction import (
        BatchedDirectFourierAnalyzer,
        BatchedDirectFourierSynthesizer,
    )

    features = torch.randn(2, 23, 64)
    positions = torch.tensor(
        [
            list(range(3, 325, 14))[:23],
            list(range(7, 352, 15))[:23],
        ],
        dtype=torch.float32,
    )
    analyzer = BatchedDirectFourierAnalyzer(num_modes, reg=0.001)
    synthesizer = BatchedDirectFourierSynthesizer(num_modes)
    coefficients, _ = analyzer(features, positions)
    reconstructed = synthesizer(coefficients, positions)

    assert coefficients.shape == (2, num_modes, 64)
    assert reconstructed.shape == features.shape


def test_model_synthesis_receives_the_original_acquisition_positions():
    from models.stclassifier import PseFourierReconLTae

    model = PseFourierReconLTae(
        input_dim=3,
        mlp1=[3, 4],
        pooling="mean_std",
        mlp2=[8, 8],
        with_extra=False,
        n_head=1,
        d_k=2,
        d_model=8,
        mlp3=[8, 4],
        mlp4=[4, 3],
        num_classes=2,
        dropout=0.0,
        fourier_num_modes=9,
    )
    captured = {}

    class Analyzer(nn.Module):
        def forward(self, features, positions):
            return torch.zeros(
                features.shape[0], 9, features.shape[2], dtype=torch.complex64
            ), {}

    class Synthesizer(nn.Module):
        def forward(self, coefficients, positions):
            captured["positions"] = positions
            return torch.zeros(
                positions.shape[0], positions.shape[1], coefficients.shape[2]
            )

    model.fourier_analyzer = Analyzer()
    model.fourier_synthesizer = Synthesizer()
    spatial_features = torch.randn(2, 6, 8)
    positions = torch.tensor(
        [[4, 19, 71, 142, 219, 331], [8, 31, 88, 157, 246, 349]],
        dtype=torch.float32,
    )
    reconstructed = model.prepare_temporal_features(spatial_features, positions)

    assert captured["positions"] is positions
    assert reconstructed.shape == spatial_features.shape


def test_launcher_is_a_mode_only_sweep_with_the_frozen_protocol():
    source = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert 'MODE_QUEUE_GPU0=(9 17)' in source
    assert 'MODE_QUEUE_GPU1=(11 19)' in source
    assert 'MODE_QUEUE_GPU2=(13)' in source
    assert 'MODE_QUEUE_GPU3=(15)' in source
    assert 'MODES=(9 11 13 15 17 19)' in source
    assert '--fourier_num_modes "$mode"' in source
    assert '--model psefourierreconltae' in source
    assert '--fourier_solver dense_direct' in source
    assert '--fourier_reg 0.001' in source
    assert '--fourier_period_days 365' in source
    assert '--num_pixels 64' in source
    assert '--seq_length 30' in source
    assert '--seed "$SEED"' in source
    assert '--num_folds 1' in source
    assert '--progress_bar off' in source
    assert 'timematch --weights "$source_weights"' in source
    assert source.count("--eval") == 1
    forbidden = ("git ", "pip install", "conda install", "curl ", "wget ")
    assert all(token not in source for token in forbidden)


def test_launcher_does_not_expand_a_local_variable_in_its_own_declaration():
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert not any(
        "source_root=" in line and "source_weights=" in line
        for line in source.splitlines()
    )


def test_summary_parser_extracts_all_core_metrics(monkeypatch):
    module = _load_summary_module()
    logs = {
        "source.log": "Test result for fourier_recon_m09_AT1_source_seed1: accuracy=0.9100, f1=0.8200\n",
        "source_on_target.log": "SOURCE_ON_TARGET|source=AT1|target=DK1|mode=9|accuracy=0.7100|macro_f1=0.6200\n",
        "da.log": (
            "INITIAL_SHIFT|source=AT1|target=DK1|mode=9|shift_days=-12\n"
            "Validation result: loss=0.4, acc=70.0, f1=0.6100\n"
            "Best AM Score shift -10 with accuracy 0.500\n"
            "Validation result: loss=0.3, acc=75.0, f1=0.6800\n"
            "Best AM Score shift -8 with accuracy 0.510\n"
            "Test result for fourier_recon_m09_AT1_DK1_timematch_seed1: accuracy=0.7600, f1=0.6900\n"
        ),
    }
    monkeypatch.setattr(module, "_read", lambda path: logs[path.name])
    result = module.parse_mode_result(Path("unused"), 9, seed=1)

    assert result["source_test_accuracy"] == pytest.approx(0.91)
    assert result["source_test_macro_f1"] == pytest.approx(0.82)
    assert result["source_on_target_accuracy"] == pytest.approx(0.71)
    assert result["source_on_target_macro_f1"] == pytest.approx(0.62)
    assert result["initial_shift"] == -12
    assert result["best_val_macro_f1"] == pytest.approx(0.68)
    assert result["best_epoch"] == 2
    assert result["test_macro_f1"] == pytest.approx(0.69)
    assert result["final_shift"] == -8
    assert result["delta_original"] == pytest.approx(0.69 - 0.8437)

def test_summary_reports_missing_required_metrics(monkeypatch):
    module = _load_summary_module()
    monkeypatch.setattr(module, "_read", lambda path: "")
    result = module.parse_mode_result(Path("unused"), 9, seed=1)
    assert "MISSING_SOURCE_TEST" in result["status"]
