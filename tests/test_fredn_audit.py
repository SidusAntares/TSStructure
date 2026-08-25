import csv
import math

import pytest
import torch
from torch import nn

from models.fredn.disentangler import FrequencyDisentangler
from models.fredn.nufft import (
    BatchedDirectFourierAnalyzer,
    BatchedDirectFourierSynthesizer,
    DenseFourierBackend,
    IrregularFourierAnalyzer,
    IrregularFourierSynthesizer,
)
from scripts.audit_fredn_reconstruction import (
    REQUIRED_SAMPLE_FIELDS,
    audit_fourier_batch,
    build_sweep,
    describe_modes,
    encode_pse,
    load_spatial_encoder_checkpoint,
    run_correctness_smoke,
    summarize_frequency_mask,
    summarize_rows,
    write_sample_csv,
)
from scripts.report_temporal_positions import audit_domain_metadata


class RecordingPSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.observed_training = None
        self.observed_grad_enabled = None

    def forward(self, pixels, mask, extra):
        self.observed_training = self.training
        self.observed_grad_enabled = torch.is_grad_enabled()
        return pixels.mean(dim=-1) * self.scale


def test_audit_statistics_calculation_uses_distribution_quantiles():
    rows = [{"reconstruction_error": value} for value in [1.0, 2.0, 3.0, 4.0]]

    stats = summarize_rows(rows, "reconstruction_error")

    assert stats["mean"] == pytest.approx(2.5)
    assert stats["std"] == pytest.approx(1.11803398875)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["p75"] == pytest.approx(3.25)
    assert stats["p90"] == pytest.approx(3.7)
    assert stats["p95"] == pytest.approx(3.85)
    assert stats["max"] == pytest.approx(4.0)


def test_reconstruction_and_additivity_errors_are_not_conflated():
    backend = DenseFourierBackend()
    analyzer = IrregularFourierAnalyzer(
        num_modes=1,
        period_days=365.0,
        reg=1e-6,
        tol=1e-8,
        max_iter=20,
        backend=backend,
    )
    synthesizer = IrregularFourierSynthesizer(
        num_modes=1,
        period_days=365.0,
        backend=backend,
    )
    disentangler = FrequencyDisentangler(num_modes=1, channels=1)
    features = torch.tensor([[[1.0], [2.0], [4.0], [8.0], [16.0]]])
    positions = torch.tensor([[0.0, 30.0, 80.0, 160.0, 300.0]])

    rows, _ = audit_fourier_batch(
        features,
        positions,
        analyzer,
        synthesizer,
        disentangler,
    )

    assert rows[0]["reconstruction_error"] > 0.5
    assert rows[0]["additivity_error"] < 1e-6


def test_checkpoint_loaded_pse_is_frozen_eval_and_encoded_without_grad(tmp_path):
    source = RecordingPSE()
    source.scale.data.fill_(3.0)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"state_dict": {"spatial_encoder.scale": source.scale.detach().clone()}},
        checkpoint,
    )
    target = RecordingPSE()

    load_spatial_encoder_checkpoint(target, checkpoint, torch.device("cpu"))
    output = encode_pse(
        target,
        torch.ones(1, 2, 1, 3),
        torch.ones(1, 2, 3),
        torch.zeros(1, 4),
    )

    assert target.scale.item() == pytest.approx(3.0)
    assert target.training is False
    assert target.scale.requires_grad is False
    assert target.observed_training is False
    assert target.observed_grad_enabled is False
    assert output.requires_grad is False


def test_per_sample_csv_contains_required_fields(tmp_path):
    path = tmp_path / "reconstruction_DK1.csv"
    row = {
        "domain": "DK1",
        "pse_state": "random_or_untrained_pse",
        "num_modes": 9,
        "regularization": 1e-4,
        "sample_index": 7,
        "sequence_length": 30,
        "position_span": 290.0,
        "cg_iterations": 8,
        "cg_converged": True,
        "reconstruction_error": 0.12,
        "additivity_error": 1e-7,
    }

    write_sample_csv(path, [row])

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        saved = list(reader)
    assert set(REQUIRED_SAMPLE_FIELDS).issubset(reader.fieldnames)
    assert saved[0]["domain"] == "DK1"
    assert float(saved[0]["reconstruction_error"]) == pytest.approx(0.12)


def test_sweep_configuration_is_deterministic_and_rejects_even_modes():
    expected = [
        (9, 1e-6),
        (9, 1e-4),
        (17, 1e-6),
        (17, 1e-4),
    ]

    assert build_sweep([17, 9, 17], [1e-4, 1e-6]) == expected
    assert build_sweep([17, 9, 17], [1e-4, 1e-6]) == expected
    with pytest.raises(ValueError, match="odd"):
        build_sweep([8], [1e-4])


def test_analyzer_exposes_per_sample_solver_diagnostics():
    analyzer = IrregularFourierAnalyzer(
        num_modes=3,
        period_days=365.0,
        reg=1e-3,
        tol=1e-6,
        max_iter=20,
        backend=DenseFourierBackend(),
    )
    features = torch.randn(2, 5, 2)
    positions = torch.tensor(
        [[0.0, 20.0, 70.0, 160.0, 280.0], [2.0, 22.0, 72.0, 162.0, 282.0]]
    )

    _, diagnostics = analyzer(features, positions)

    assert len(diagnostics["per_sample_solver_iterations"]) == 2
    assert len(diagnostics["per_sample_solver_converged"]) == 2
    assert len(diagnostics["per_sample_solver_residual"]) == 2


def test_dense_direct_per_sample_solver_diagnostics_are_opt_in_and_detached():
    analyzer = BatchedDirectFourierAnalyzer(
        num_modes=3,
        period_days=365.0,
        reg=1e-3,
    )
    features = torch.randn(2, 5, 2, requires_grad=True)
    positions = torch.tensor(
        [[0.0, 20.0, 70.0, 160.0, 280.0], [2.0, 22.0, 72.0, 162.0, 282.0]]
    )

    _, normal_diagnostics = analyzer(features, positions)
    _, audit_diagnostics = analyzer(
        features,
        positions,
        collect_diagnostics=True,
    )

    assert "per_sample_solver_residual" not in normal_diagnostics
    assert "per_sample_solver_converged" not in normal_diagnostics
    residuals = audit_diagnostics["per_sample_solver_residual"]
    converged = audit_diagnostics["per_sample_solver_converged"]
    assert residuals.shape == (2,)
    assert converged.shape == (2,)
    assert residuals.requires_grad is False
    assert converged.requires_grad is False
    assert torch.isfinite(residuals).all()
    assert converged.dtype == torch.bool
    assert converged.all()


def test_dense_direct_audit_reports_finite_residual_and_per_sample_convergence():
    torch.manual_seed(43)
    analyzer = BatchedDirectFourierAnalyzer(
        num_modes=3,
        period_days=365.0,
        reg=1e-3,
    )
    synthesizer = BatchedDirectFourierSynthesizer(
        num_modes=3,
        period_days=365.0,
    )
    disentangler = FrequencyDisentangler(num_modes=3, channels=2)
    features = torch.randn(2, 5, 2)
    positions = torch.tensor(
        [[0.0, 20.0, 70.0, 160.0, 280.0], [2.0, 22.0, 72.0, 162.0, 282.0]]
    )

    rows, diagnostics = audit_fourier_batch(
        features,
        positions,
        analyzer,
        synthesizer,
        disentangler,
    )

    assert len(rows) == 2
    for row in rows:
        assert math.isfinite(row["cg_residual"])
        assert math.isfinite(row["reconstruction_error"])
        assert math.isfinite(row["additivity_error"])
        assert row["cg_converged"] is True
        assert row["cg_iterations"] == 0
    assert math.isfinite(diagnostics["imaginary_residual"])


def test_shared_points_solver_keeps_per_sample_iteration_counts():
    analyzer = IrregularFourierAnalyzer(
        num_modes=3,
        period_days=365.0,
        reg=1e-3,
        tol=1e-6,
        max_iter=20,
        backend=DenseFourierBackend(),
    )
    features = torch.stack([torch.zeros(5, 1), torch.randn(5, 1)])
    shared_positions = torch.tensor([[0.0, 20.0, 70.0, 160.0, 280.0]]).repeat(2, 1)

    _, diagnostics = analyzer(features, shared_positions)

    assert diagnostics["per_sample_solver_iterations"][0] == 0
    assert diagnostics["per_sample_solver_iterations"][1] > 0
    assert diagnostics["per_sample_solver_converged"][0] is True


def test_timestamp_audit_reports_abs_direction_and_domain_statistics():
    metadata = {
        "start_date": 20200105,
        "dates": [20200101, 20200105, 20200110],
        "parcels": [{"label": 1}, {"label": 2}],
    }

    result = audit_domain_metadata("DK1", metadata)

    assert result["domain"] == "DK1"
    assert result["start_date"] == "20200105"
    assert result["start_date_fixed"] is True
    assert result["sample_specific_start_date"] is False
    assert result["sample_count"] == 2
    assert result["observation_count"] == 6
    assert result["position_min"] == pytest.approx(0.0)
    assert result["position_max"] == pytest.approx(5.0)
    assert result["sample_span_median"] == pytest.approx(5.0)
    assert result["sequence_length_min"] == 3
    assert result["sequence_length_max"] == 3
    assert result["date_lt_start_date_count"] == 2
    assert result["date_eq_start_date_count"] == 2
    assert result["date_gt_start_date_count"] == 2
    assert result["date_lt_start_date_examples"][0] == {
        "sample_index": 0,
        "date": "20200101",
        "start_date": "20200105",
    }
    assert result["position_unit"] == "days"


def test_correctness_smoke_reports_all_nufft_numerical_checks():
    result = run_correctness_smoke(
        DenseFourierBackend(),
        torch.device("cpu"),
        num_modes=5,
        period_days=365.0,
        regularization=1e-8,
        tolerance=1e-9,
        max_iterations=50,
    )

    assert result["adjoint_relative_error"] < 1e-6
    assert result["synthetic_coefficient_error"] < 1e-5
    assert result["synthetic_reconstruction_error"] < 1e-5
    assert result["additivity_error"] < 1e-6
    assert result["imaginary_residual"] < 1e-6


def test_mode_and_mask_summaries_report_actual_pairing():
    assert describe_modes(9) == {
        "mode_index_min": -4,
        "mode_index_max": 4,
        "complex_coefficient_count": 9,
    }
    summary = summarize_frequency_mask([0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2])
    assert summary["mask_pair_max_abs_diff"] == pytest.approx(0.0)
    assert summary["mask_low_to_high_monotonic"] is True
