from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
import torch

import analysis.structure_circular_alignment_validity_audit as audit
from analysis.structure_relative_transition_audit import transition_from_similarity


def test_circular_alignment_audit_files_exist():
    assert Path("analysis/structure_circular_alignment_validity_audit.py").is_file()
    assert Path(
        "scripts/run_structure_circular_alignment_validity_audit_4tasks_4gpu_seed1.sh"
    ).is_file()


def test_script_is_directly_executable():
    result = subprocess.run(
        [sys.executable, "analysis/structure_circular_alignment_validity_audit.py", "--help"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_transfer_split_restores_target_train_without_overlap_or_test_dataset():
    calls = []

    def fold_creator(datasets, num_folds, eligible, val_ratio, test_ratio):
        calls.append(tuple(datasets))
        return [{
            "source": {"train": {0, 1}, "val": {2}, "test": {3}},
            "target": {"train": {10, 11}, "val": {12}, "test": {13}},
        }]

    split = audit.replay_transfer_split(
        "source", "target", {"source": range(4), "target": range(10, 14)},
        seed=1, val_ratio=.1, test_ratio=.2, fold_creator=fold_creator,
    )
    assert calls == [("source", "target")]
    assert split["target_train"] == {10, 11}
    assert split["target_val"] == {12}
    assert split["target_test"] == {13}
    assert split["target_train"].isdisjoint(split["target_val"])
    assert split["target_train"].isdisjoint(split["target_test"])


def test_shift_orchestration_uses_target_train_loader_and_no_gt_arguments():
    loader = object()
    seen = []

    def initialize(model, received_loader, device, config):
        seen.append(("IS", received_loader, config.shift_estimator))
        return -7, np.array([.4, .6]), {"selected_shift": -7}

    def reestimate(
        model, received_loader, device, config, initial_shift,
        distribution, diagnostics, epoch,
    ):
        seen.append(("AM", received_loader, initial_shift, epoch))
        return -4

    result = audit.estimate_target_to_source_shift(
        object(), loader, "cpu", source="s", target="t", num_classes=2,
        initialize_fn=initialize, reestimate_fn=reestimate,
    )
    assert result == {"initial_is_shift": -7, "am_shift": -4}
    assert seen == [("IS", loader, "AM"), ("AM", loader, -7, 0)]


def test_wrap_statistics_and_modulo_positions_are_exact():
    positions = torch.tensor([[0, 20, 360], [5, 100, 364]])
    labels = torch.tensor([0, 1])
    summary, rows = audit.wrap_statistics(
        positions, labels, ["a", "b"], shift=10, period=365,
    )
    assert summary["num_observations"] == 6
    assert summary["num_wrapped"] == 2
    assert summary["wrapped_fraction"] == pytest.approx(2 / 6)
    assert summary["num_samples_with_wrap"] == 2
    assert rows[0]["wrapped_fraction"] == pytest.approx(1 / 3)
    wrapped = audit.circular_positions(positions, 10, 365)
    assert int(wrapped.min()) >= 0
    assert int(wrapped.max()) < 365


def test_no_wrap_linear_and_circular_positions_and_logits_are_identical():
    class FakeModel:
        def __init__(self):
            self.encoded_positions = []

        def _encode_instance(self, spatial, positions, structure):
            self.encoded_positions.append(positions.clone())
            return positions.float().sum(1, keepdim=True)

        def decoder(self, instance):
            return torch.cat((instance, -instance), dim=1)

    model = FakeModel()
    positions = torch.tensor([[20, 30], [100, 120]])
    outputs = audit.linear_circular_logits(
        model, torch.zeros(2, 2, 3), positions, object(), shift=5, period=365,
    )
    torch.testing.assert_close(outputs["linear_positions"], outputs["circular_positions"])
    torch.testing.assert_close(outputs["linear_logits"], outputs["circular_logits"])
    assert len(model.encoded_positions) == 2


def test_endpoint_continuity_uses_observed_pre_fourier_features_and_support():
    features = torch.tensor([[
        [1., 0.], [0., 1.], [1., 0.],
    ], [
        [0., 1.], [1., 0.], [0., 1.],
    ]])
    positions = torch.tensor([[5, 100, 330], [10, 200, 350]])
    rows = audit.endpoint_continuity_rows(
        "AT1", features, positions, torch.tensor([0, 1]), ["a", "b"],
        boundary_width=45, period=365,
    )
    assert rows[0]["start_observation_support"] == 1
    assert rows[0]["end_observation_support"] == 1
    assert rows[0]["start_sample_support"] == 1
    assert rows[0]["end_sample_support"] == 1
    assert rows[0]["endpoint_cosine"] == pytest.approx(1.)
    assert rows[1]["endpoint_cosine"] == pytest.approx(1.)


def test_phase_uses_time_softmax_real_centers_and_shift_only_for_aligned_target():
    similarity = torch.randn(3, 8, 16)
    centers = torch.arange(8, dtype=torch.float32) * 8 + 11.5
    raw = audit.anchor_phase_feature(
        similarity, beta=5., center_indices=centers, phase_shift=0, period=365,
        grid_points=64,
    )
    aligned_zero = audit.anchor_phase_feature(
        similarity, beta=5., center_indices=centers, phase_shift=0, period=365,
        grid_points=64,
    )
    aligned = audit.anchor_phase_feature(
        similarity, beta=5., center_indices=centers, phase_shift=20, period=365,
        grid_points=64,
    )
    assert raw.shape == (3, 32)
    torch.testing.assert_close(raw, aligned_zero)
    assert not torch.allclose(raw, aligned)
    occurrence = audit.occurrence_weights(similarity, beta=5.)
    torch.testing.assert_close(occurrence.sum(1), torch.ones(3, 16))


def test_circular_moment_is_invariant_to_two_pi():
    occurrence = torch.softmax(torch.randn(2, 8, 16), dim=1)
    theta = torch.linspace(0, 2 * torch.pi, 8)
    first = audit.circular_moment(occurrence, theta)
    second = audit.circular_moment(occurrence, theta + 2 * torch.pi)
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


def test_phase_representations_use_real_current_and_reused_transition_helper(monkeypatch):
    current = torch.randn(4, 32)
    similarity = torch.randn(4, 8, 16)
    centers = torch.arange(8, dtype=torch.float32)
    calls = []

    def recording_transition(values, beta, lag):
        calls.append(lag)
        return transition_from_similarity(values, beta, lag)

    monkeypatch.setattr(audit, "transition_from_similarity", recording_transition)
    representations = audit.compose_phase_representations(
        current, similarity, beta=5., center_indices=centers,
        phase_shift=12, period=365, grid_points=64,
    )
    assert representations["current"] is current
    assert calls == [1, 2]
    assert representations["current_plus_phase_raw"].shape == (4, 64)
    assert representations["current_plus_phase_aligned"].shape == (4, 64)
    assert representations["current_plus_transition"].shape == (4, 544)
    assert representations["current_plus_phase_aligned_plus_transition"].shape == (4, 576)


@pytest.mark.parametrize("name,probe_dim", [
    ("current", 32),
    ("current_plus_phase_raw", 64),
    ("current_plus_phase_aligned", 64),
    ("current_plus_transition", 64),
    ("current_plus_phase_aligned_plus_transition", 96),
])
def test_block_preprocessor_has_frozen_probe_dimensions(name, probe_dim):
    raw_dim = {
        "current": 32,
        "current_plus_phase_raw": 64,
        "current_plus_phase_aligned": 64,
        "current_plus_transition": 544,
        "current_plus_phase_aligned_plus_transition": 576,
    }[name]
    values = np.random.default_rng(3).normal(size=(48, raw_dim))
    preprocessor = audit.PhaseProbePreprocessor(name).fit(values)
    assert preprocessor.transform(values).shape == (48, probe_dim)


def test_source_probe_preprocessing_never_fits_target():
    rng = np.random.default_rng(5)
    source = rng.normal(size=(48, 576))
    target = rng.normal(loc=100., size=(12, 576))
    labels = np.tile([0, 1, 2], 16)
    result = audit.fit_source_phase_probe(
        "current_plus_phase_aligned_plus_transition",
        source, labels, source[:12], labels[:12], target, labels[:12],
        class_ids=np.array([0, 1, 2]),
    )
    np.testing.assert_allclose(
        result["preprocessor"].current_scaler.mean_, source[:, :32].mean(0),
    )
    assert not np.allclose(result["preprocessor"].current_scaler.mean_, 100.)


def test_target_oracle_preprocessing_is_refit_inside_each_training_fold(monkeypatch):
    rng = np.random.default_rng(7)
    features = rng.normal(size=(60, 64))
    labels = np.repeat([0, 1, 2], 20)
    fitted_sizes = []
    original_fit = audit.PhaseProbePreprocessor.fit

    def recording_fit(self, values):
        fitted_sizes.append(len(values))
        return original_fit(self, values)

    monkeypatch.setattr(audit.PhaseProbePreprocessor, "fit", recording_fit)
    result = audit.target_oracle_phase_probe(
        "current_plus_phase_raw", features, labels,
        class_ids=np.array([0, 1, 2]), seed=1,
    )
    assert result["folds"] == 5
    assert fitted_sizes == [48] * 5


def test_extraction_is_one_forward_read_only_and_preserves_model_state(monkeypatch):
    class Spatial(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, pixels, valid, extra):
            self.calls += 1
            return pixels

    class Dictionary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.beta = 5.

    class Branch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shapelet_dictionary = Dictionary()

        def window_extractor(self, curve, return_centers=False):
            centers = torch.arange(8, device=curve.device, dtype=curve.dtype) * 8 + 11.5
            return [curve[:, :24]] * 8, torch.full((8,), 24), centers

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial_encoder = Spatial()
            self.structure_branch = Branch()
            self.marker = torch.nn.Parameter(torch.tensor(3.))
            self.structure_calls = 0

        def prepare_structure(self, spatial, positions):
            self.structure_calls += 1
            batch = spatial.shape[0]
            return {
                "shapelet_similarity": torch.zeros(batch, 8, 16),
                "shapelet_response": torch.zeros(batch, 32),
                "exposed_curve": torch.zeros(batch, 64, spatial.shape[-1]),
            }

    batch = {
        "pixels": torch.zeros(2, 3, 4),
        "valid_pixels": torch.ones(2, 3, 4),
        "positions": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "extra": torch.zeros(2, 4),
        "label": torch.tensor([0, 1]),
    }
    monkeypatch.setattr(audit, "deterministic_loader", lambda *args, **kwargs: [batch])
    model = Model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = audit.extract_dataset(
        model, object(), 2, 0, torch.device("cpu"), 128, phase_shift=4,
    )
    assert model.spatial_encoder.calls == 1
    assert model.structure_calls == 1
    assert result["representations"]["current"].shape == (2, 32)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_launcher_is_four_task_read_only_and_merges_once():
    launcher = Path(
        "scripts/run_structure_circular_alignment_validity_audit_4tasks_4gpu_seed1.sh"
    ).read_text(encoding="utf-8")
    script = Path("analysis/structure_circular_alignment_validity_audit.py").read_text(
        encoding="utf-8",
    )
    for line in (
        'run_task "$GPU0" AT1_DK1',
        'run_task "$GPU1" FR1_FR2',
        'run_task "$GPU2" FR2_DK1',
        'run_task "$GPU3" DK1_AT1',
    ):
        assert line in launcher
    assert "--merge-only" in launcher
    assert "outputs/structure_proto_v2clean_4tasks_seed1/source" in launcher
    assert "outputs/structure_circular_alignment_validity_audit_seed1" in launcher
    assert "train.py" not in launcher
    assert "timematch.py" not in launcher
    assert ".backward(" not in script
    assert "torch.optim" not in script
    # Test indices are retained only to assert split disjointness; no test Dataset is built.
    assert 'audit_datasets["target_test"]' not in script
    assert 'audit_datasets["source_test"]' not in script
    assert "target_test_dataset_instantiated=false" in script
