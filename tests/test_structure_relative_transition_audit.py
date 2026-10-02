from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import analysis.structure_relative_transition_audit as audit


def test_relative_transition_audit_module_exists():
    assert Path("analysis/structure_relative_transition_audit.py").is_file()


def test_audit_script_is_directly_executable_from_repository_root():
    completed = subprocess.run(
        [sys.executable, "analysis/structure_relative_transition_audit.py", "--help"],
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _directional_similarity():
    similarity = torch.full((1, 8, 16), -6.)
    for window, anchor in enumerate((0, 1, 2, 0, 1, 2, 0, 1)):
        similarity[0, window, anchor] = 6.
    return similarity


def test_occurrence_softmax_is_over_time_and_transitions_are_row_normalized():
    similarity = torch.randn(3, 8, 16)
    occurrence = audit.occurrence_weights(similarity, beta=5.)
    torch.testing.assert_close(
        occurrence.sum(dim=1), torch.ones(3, 16), atol=1e-6, rtol=1e-6,
    )
    first = audit.relative_anchor_transition(occurrence, lag=1)
    second = audit.relative_anchor_transition(occurrence, lag=2)
    assert first.shape == second.shape == (3, 16, 16)
    torch.testing.assert_close(
        first.sum(dim=-1), torch.ones(3, 16), atol=1e-5, rtol=1e-5,
    )


def test_transition_is_circular_shift_invariant_but_direction_sensitive():
    similarity = _directional_similarity()
    original = audit.transition_from_similarity(similarity, beta=5., lag=1)
    shifted = audit.transition_from_similarity(
        torch.roll(similarity, shifts=3, dims=1), beta=5., lag=1,
    )
    reversed_time = audit.transition_from_similarity(
        torch.flip(similarity, dims=(1,)), beta=5., lag=1,
    )
    torch.testing.assert_close(original, shifted, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(original, reversed_time, atol=1e-3, rtol=1e-3)


def test_current_pooling_is_invariant_to_window_order():
    similarity = _directional_similarity()
    original = audit.current_from_similarity(similarity, beta=5.)
    permutation = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    for transformed in (
        torch.roll(similarity, shifts=3, dims=1),
        similarity[:, permutation],
        torch.flip(similarity, dims=(1,)),
    ):
        torch.testing.assert_close(
            original, audit.current_from_similarity(transformed, beta=5.),
            atol=1e-6, rtol=1e-6,
        )


def test_fixed_permutation_changes_directional_transition():
    similarity = _directional_similarity()
    permutation = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    original = audit.transition_from_similarity(similarity, beta=5., lag=1)
    permuted = audit.transition_from_similarity(
        similarity[:, permutation], beta=5., lag=1,
    )
    assert not torch.allclose(original, permuted, atol=1e-3, rtol=1e-3)


def test_five_representations_use_real_current_and_have_frozen_raw_dimensions():
    current = torch.randn(4, 32)
    similarity = torch.randn(4, 8, 16)
    representations = audit.compose_transition_representations(
        current, similarity, beta=5.,
    )
    assert tuple(representations) == (
        "current", "transition_lag1", "transition_lag12",
        "current_plus_lag1", "current_plus_lag12",
    )
    assert representations["current"] is current
    assert representations["transition_lag1"].shape == (4, 256)
    assert representations["transition_lag12"].shape == (4, 512)
    assert representations["current_plus_lag1"].shape == (4, 288)
    assert representations["current_plus_lag12"].shape == (4, 544)


@pytest.mark.parametrize("name,expected_dim", [
    ("current", 32),
    ("transition_lag1", 32),
    ("transition_lag12", 32),
    ("current_plus_lag1", 64),
    ("current_plus_lag12", 64),
])
def test_probe_preprocessing_has_fixed_output_dimensions(name, expected_dim):
    rng = np.random.default_rng(4)
    raw_dim = {
        "current": 32, "transition_lag1": 256, "transition_lag12": 512,
        "current_plus_lag1": 288, "current_plus_lag12": 544,
    }[name]
    values = rng.normal(size=(48, raw_dim))
    transformer = audit.ProbePreprocessor(name, pca_dim=32).fit(values)
    assert transformer.transform(values).shape == (48, expected_dim)


def test_combined_probe_scales_current_and_transition_blocks_separately():
    rng = np.random.default_rng(8)
    current = rng.normal(loc=10., size=(48, 32))
    transition = rng.normal(loc=-20., size=(48, 256))
    values = np.concatenate((current, transition), axis=1)
    transformer = audit.ProbePreprocessor(
        "current_plus_lag1", pca_dim=32,
    ).fit(values)
    np.testing.assert_allclose(transformer.current_scaler.mean_, current.mean(0))
    np.testing.assert_allclose(
        transformer.transition_pipeline.named_steps["scaler"].mean_,
        transition.mean(0),
    )
    assert transformer.current_scaler is not transformer.transition_pipeline.named_steps["scaler"]


def test_source_to_target_probe_fits_preprocessing_on_source_train_only():
    rng = np.random.default_rng(9)
    source_train = rng.normal(loc=0., size=(48, 288))
    source_labels = np.tile([0, 1], 24)
    source_val = rng.normal(loc=2., size=(8, 288))
    target = rng.normal(loc=100., size=(8, 288))
    result = audit.fit_source_transition_probe(
        "current_plus_lag1", source_train, source_labels,
        source_val, np.tile([0, 1], 4), target, np.tile([0, 1], 4),
        class_ids=np.array([0, 1]),
    )
    np.testing.assert_allclose(
        result["preprocessor"].current_scaler.mean_, source_train[:, :32].mean(0),
    )
    assert not np.allclose(result["preprocessor"].current_scaler.mean_, 100.)


def test_target_oracle_refits_preprocessing_inside_every_fold(monkeypatch):
    rng = np.random.default_rng(10)
    features = rng.normal(size=(60, 256))
    labels = np.tile([0, 1, 2], 20)
    seen_sizes = []
    original = audit.ProbePreprocessor.fit

    def recording_fit(self, values):
        seen_sizes.append(len(values))
        return original(self, values)

    monkeypatch.setattr(audit.ProbePreprocessor, "fit", recording_fit)
    result = audit.target_oracle_transition_probe(
        "transition_lag1", features, labels,
        class_ids=np.array([0, 1, 2]), seed=1,
    )
    assert result["folds"] == 5
    assert seen_sizes == [48] * 5
    assert all(size < len(features) for size in seen_sizes)


def test_launcher_and_audit_are_source_only_read_only_and_four_task():
    launcher = Path(
        "scripts/run_structure_relative_transition_audit_4tasks_4gpu_seed1.sh"
    ).read_text(encoding="utf-8")
    script = Path("analysis/structure_relative_transition_audit.py").read_text(
        encoding="utf-8",
    )
    assert all(f"GPU{index}" in launcher for index in range(4))
    assert all(task in launcher for task in audit.TASKS)
    assert "outputs/structure_proto_v2clean_4tasks_seed1/source" in launcher
    assert "outputs/structure_relative_transition_audit_seed1" in launcher
    assert "--merge" in launcher
    for line in (
        'run_task "$GPU0" AT1_DK1',
        'run_task "$GPU1" FR1_FR2',
        'run_task "$GPU2" FR2_DK1',
        'run_task "$GPU3" DK1_AT1',
    ):
        assert line in launcher
    assert "target_test" not in script
    assert "source_test" not in script
    assert ".backward(" not in script
    assert "torch.optim" not in script
    assert "train.py" not in launcher
    assert "timematch.py" not in launcher.lower()


def test_model_state_is_not_mutated_and_one_structure_forward_supplies_both_inputs():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.scale = torch.nn.Parameter(torch.tensor(1.))
            self.structure_branch = type("Branch", (), {
                "shapelet_dictionary": type("Dictionary", (), {"beta": 5.})(),
            })()

        def spatial_encoder(self, pixels, valid, extra):
            return pixels * self.scale

        def prepare_structure(self, spatial, positions):
            self.calls += 1
            return {
                "shapelet_similarity": torch.randn(2, 8, 16),
                "shapelet_response": torch.randn(2, 32),
            }

    model = FakeModel()
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    result = audit.extract_transition_batch(
        model, torch.randn(2, 8, 3), torch.ones(2, 8),
        torch.arange(8).repeat(2, 1), torch.zeros(2, 4),
    )
    assert model.calls == 1
    assert result["current"].shape == (2, 32)
    assert result["similarity"].shape == (2, 8, 16)
    assert state_before.keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name])


def test_output_schemas_are_exact():
    assert audit.SUMMARY_FIELDS == (
        "task", "representation", "raw_dim", "probe_dim", "effective_rank",
        "source_val_macro_f1", "target_oracle_macro_f1",
        "source_to_target_macro_f1",
    )
    assert audit.PER_CLASS_FIELDS == (
        "task", "representation", "class", "source_support", "target_support",
        "source_val_f1", "target_oracle_f1", "source_to_target_f1",
    )
    assert audit.INVARIANCE_FIELDS == (
        "task", "representation", "shift_cos", "permutation_cos", "reverse_cos",
        "order_gap", "reverse_gap",
    )
