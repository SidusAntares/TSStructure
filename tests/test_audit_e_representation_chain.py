import numpy as np
import pytest
import torch

from analysis.audit_e_representation_chain import (
    STAGES,
    _encoded_layers,
    calendar_bin_features,
    compare_sample_manifests,
    confusion_metrics,
    fit_probe_metrics,
    fourier_fidelity_metrics,
    require_protocol_gate,
    validate_checkpoint_role,
)
from models.structure_da.discriminative_structure import FourierStructureExposer
from models.stclassifier import PseStructureProtoLTae


def test_sample_manifest_detects_missing_duplicate_and_label_mismatch():
    official = [("target", 10, 0), ("target", 11, 1), ("target", 12, 1)]
    diagnostic = [("target", 10, 0), ("target", 11, 0), ("target", 11, 0)]
    rows, passed = compare_sample_manifests(official, diagnostic, "target_test")
    assert not passed
    reasons = {row["reason"] for row in rows}
    assert {"duplicate_diagnostic", "missing_diagnostic", "label_mismatch"} <= reasons


def test_confusion_metrics_distinguishes_dynamic_and_fixed_macro():
    cm = np.asarray([[4, 0, 0], [0, 2, 0], [0, 0, 0]])
    result = confusion_metrics(cm)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["fixed_class_macro_f1"] == pytest.approx(2.0 / 3.0)
    assert result["support"].tolist() == [4, 2, 0]


def test_calendar_bins_are_fixed_and_include_occupancy_mask():
    values = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]])
    positions = torch.tensor([[0, 10, 190, 360]])
    features = calendar_bin_features(values, positions, bins=2, period_days=365.0)
    # means [2, 6], then the two occupancy indicators
    assert features.shape == (1, 4)
    assert features[0].tolist() == pytest.approx([2.0, 6.0, 1.0, 1.0])


def test_fourier_fidelity_metrics_known_values():
    truth = torch.tensor([[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]])
    same = fourier_fidelity_metrics(truth, truth)
    assert same["normalized_mse"][0] == pytest.approx(0.0)
    assert same["r2"][0] == pytest.approx(1.0)
    assert same["channel_time_correlation"][0] == pytest.approx(1.0)


def test_fourier_reconstructs_at_original_observation_positions():
    exposer = FourierStructureExposer(num_modes=5, grid_points=64, reg=1e-8)
    positions = torch.tensor([[0.0, 23.0, 71.0, 130.0, 211.0, 310.0]])
    phase = 2.0 * torch.pi * positions / 365.0
    values = torch.stack((
        0.3 + torch.sin(phase),
        -0.2 + 0.5 * torch.cos(2.0 * phase),
    ), dim=-1)
    coefficients = exposer.analyze(values, positions)
    reconstructed = exposer.synthesizer(coefficients, positions)
    assert torch.allclose(reconstructed, values, atol=2e-4, rtol=2e-4)


def test_probe_scaler_is_fit_on_source_train_only():
    source_train = np.asarray([[-1.0], [1.0], [-2.0], [2.0]])
    labels = np.asarray([0, 1, 0, 1])
    source_val = np.asarray([[-1.5], [1.5]])
    target_val = np.asarray([[100.0], [200.0]])
    result = fit_probe_metrics(
        source_train, labels, source_val, np.asarray([0, 1]),
        target_val, np.asarray([0, 1]), np.asarray([0, 1]), seed=7,
    )
    assert result["estimator"].named_steps["scaler"].mean_[0] == pytest.approx(0.0)


def test_protocol_gate_and_checkpoint_roles():
    require_protocol_gate(True)
    with pytest.raises(RuntimeError, match="PROTOCOL_GATE_FAILED"):
        require_protocol_gate(False)
    validate_checkpoint_role(
        "checkpoint_last.pt", {"epoch": 19}, {"epochs": 20}, "final",
    )
    with pytest.raises(ValueError, match="final epoch"):
        validate_checkpoint_role(
            "checkpoint_last.pt", {"epoch": 18}, {"epochs": 20}, "final",
        )
    with pytest.raises(ValueError, match="checkpoint_last"):
        validate_checkpoint_role("model.pt", {"epoch": 19}, {"epochs": 20}, "final")


def test_phase_moment_layer_extraction_uses_real_model_outputs():
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=8, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, fourier_num_modes=5,
        shape_representation="phase_moment", shape_injection="current_query",
    ).eval()
    spatial = torch.randn(2, 16, 8)
    positions = torch.arange(16).repeat(2, 1) * 20
    _, _, structure, layers = _encoded_layers(model, spatial, positions)
    assert set(layers) == set(STAGES)
    assert layers["shape_token"].shape == (2, 8, 8)
    assert layers["anchor_similarity"].shape == (2, 8, 16)
    assert layers["phase_moment"].shape == (2, 64)
    assert layers["shape_response"].shape == (2, 96)
    torch.testing.assert_close(layers["shape_response"], structure["shapelet_response"])
