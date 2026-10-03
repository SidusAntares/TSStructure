import copy
import inspect

import numpy as np
import torch


def test_phase_representations_use_real_model_outputs_with_fixed_dimensions():
    from analysis.structure_phase_dependency_audit import phase_representations

    strength = torch.randn(3, 16)
    concentration = torch.randn(3, 16)
    phase = torch.randn(3, 64)
    structure = {
        "shapelet_strength": strength,
        "shapelet_concentration": concentration,
        "shapelet_phase_moments": phase,
        "shapelet_response": torch.cat((strength, concentration, phase), -1),
    }
    values = phase_representations(structure)
    assert values["Current"].shape == (3, 32)
    assert values["Phase"].shape == (3, 64)
    assert values["Current+Phase"].shape == (3, 96)
    torch.testing.assert_close(values["Current+Phase"], structure["shapelet_response"])


def test_rotate_structure_phase_changes_only_phase_and_response():
    from analysis.structure_phase_dependency_audit import rotate_structure_phase

    structure = {
        "shapelet_strength": torch.randn(2, 16),
        "shapelet_concentration": torch.randn(2, 16),
        "shapelet_phase_moments": torch.randn(2, 64),
        "shapelet_response": torch.randn(2, 96),
        "shapelet_similarity": torch.randn(2, 8, 16),
        "shape_tokens": torch.randn(2, 8, 128),
        "exposed_grid": torch.arange(64),
    }
    original = {key: value.clone() for key, value in structure.items()}
    rotated = rotate_structure_phase(structure, 30)
    for key in ("shapelet_strength", "shapelet_concentration", "shapelet_similarity", "shape_tokens", "exposed_grid"):
        torch.testing.assert_close(rotated[key], original[key])
    assert not torch.allclose(rotated["shapelet_phase_moments"], original["shapelet_phase_moments"])
    torch.testing.assert_close(
        rotated["shapelet_response"],
        torch.cat((original["shapelet_strength"], original["shapelet_concentration"], rotated["shapelet_phase_moments"]), -1),
    )


def test_delta_zero_intervention_matches_normal_phase_model_forward():
    from analysis.structure_phase_dependency_audit import phase_intervention_forward
    from tests.test_structure_phase_equivariance import _model

    torch.manual_seed(5)
    model = _model().eval()
    pixels = torch.randn(3, 10, 3, 4)
    mask = torch.ones(3, 10, 4)
    positions = torch.arange(10).repeat(3, 1) * 30
    extra = torch.zeros(3, 4)
    with torch.no_grad():
        normal = model(pixels, mask, positions, extra, return_dict=True)
        intervened, base = phase_intervention_forward(
            model, pixels, mask, positions, extra, delta=0,
        )
    torch.testing.assert_close(intervened["logits"], normal["logits"])
    torch.testing.assert_close(base["shapelet_similarity"], normal["shapelet_similarity"])


def test_phase_probe_fits_source_only_and_oracle_refits_each_fold():
    from analysis.structure_phase_dependency_audit import evaluate_probe

    rng = np.random.default_rng(7)
    labels = np.repeat([0, 1], 12)
    source_train = rng.normal(size=(24, 4))
    source_val = rng.normal(size=(24, 4))
    target_val = rng.normal(loc=100, size=(24, 4))
    result = evaluate_probe(
        source_train, labels, source_val, labels, target_val, labels,
        class_ids=np.array([0, 1]), seed=1,
    )
    np.testing.assert_allclose(
        result["source_probe"].named_steps["scaler"].mean_, source_train.mean(0),
    )
    assert result["target_oracle_folds"] >= 2


def test_audit_scope_is_read_only_and_excludes_train_and_test_splits():
    import analysis.structure_phase_dependency_audit as audit

    assert audit.AUDIT_SPLITS == ("source_train", "source_val", "target_val")
    assert tuple(audit.TASKS) == ("AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1")
    source = inspect.getsource(audit)
    assert "target_train" not in source
    assert "target_test" not in source
    assert "source_test" not in source
    assert ".backward(" not in source
    assert "optimizer" not in source.lower()
    assert "rotate_phase_moments" in source


def test_phase_intervention_does_not_mutate_model_state():
    from analysis.structure_phase_dependency_audit import phase_intervention_forward
    from tests.test_structure_phase_equivariance import _model

    model = _model().eval()
    before = copy.deepcopy(model.state_dict())
    pixels = torch.randn(2, 10, 3, 4)
    mask = torch.ones(2, 10, 4)
    positions = torch.arange(10).repeat(2, 1) * 30
    extra = torch.zeros(2, 4)
    with torch.no_grad():
        phase_intervention_forward(model, pixels, mask, positions, extra, 60)
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value)


def test_phase_audit_launcher_maps_four_tasks_to_four_gpus_and_merges():
    from pathlib import Path

    launcher = Path(
        "scripts/run_structure_phase_dependency_audit_4tasks_4gpu_seed1.sh"
    ).read_text(encoding="utf-8")
    assert "run_task 0 AT1_DK1" in launcher
    assert "run_task 1 FR1_FR2" in launcher
    assert "run_task 2 FR2_DK1" in launcher
    assert "run_task 3 DK1_AT1" in launcher
    assert "--merge" in launcher
