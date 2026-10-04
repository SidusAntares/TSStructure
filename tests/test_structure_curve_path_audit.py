import importlib

import numpy as np
import torch


def audit():
    return importlib.import_module("analysis.structure_curve_path_audit")


def _unit(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def test_calendar_and_dtw_identity_similarity_are_one():
    module = audit()
    curve = _unit([[1, 0], [1, 1], [0, 1]])
    assert np.isclose(module.curve_similarity(curve, curve), 1.0)
    similarity, path_length = module.dtw_similarity(curve, curve)
    assert np.isclose(similarity, 1.0)
    assert path_length == len(curve)


def test_global_shift_recovers_known_non_wrapped_shift_and_one_task_delta():
    module = audit()
    grid = np.arange(0, 121, dtype=np.float64)
    source = _unit(np.stack((
        np.cos(grid / 13), np.sin(grid / 13), np.cos(grid / 29),
    ), axis=1))
    # target(t + 30) equals source(t), so the specified target(t+delta)
    # convention must recover +30 without circular wrapping.
    target_grid = grid + 30
    target = _unit(np.stack((
        np.cos((target_grid - 30) / 13),
        np.sin((target_grid - 30) / 13),
        np.cos((target_grid - 30) / 29),
    ), axis=1))
    # Express target on the audit grid as a delayed source trajectory.
    target = np.vstack((np.repeat(source[:1], 30, axis=0), source[:-30]))
    delta, score = module.find_global_shift(
        {0: source, 1: source[:, ::-1]},
        {0: target, 1: target[:, ::-1]}, grid, shifts=range(-60, 61),
    )
    assert delta == 30
    assert score > 0.999
    shifted, source_common = module.compare_at_shift(source, target, grid, delta)
    assert shifted.shape[0] == source_common.shape[0] == len(grid) - 30
    # There is no December-to-January wrapping: endpoints are discarded.
    assert shifted.shape[0] < len(grid)


def test_arc_coordinates_have_safe_constant_fallback():
    module = audit()
    curve = np.repeat([[1.0, 0.0]], 5, axis=0)
    q = module.arc_length_coordinates(curve)
    assert np.allclose(q, np.linspace(0, 1, 5))
    assert q[0] == 0 and q[-1] == 1
    assert np.all(np.diff(q) >= 0)


def test_arc_resampling_improves_same_path_with_different_speed():
    module = audit()
    theta_source = np.linspace(0, np.pi, 64)
    theta_target = np.linspace(0, np.sqrt(np.pi), 64) ** 2
    source = np.stack((np.cos(theta_source), np.sin(theta_source)), axis=1)
    target = np.stack((np.cos(theta_target), np.sin(theta_target)), axis=1)
    before = module.curve_similarity(source, target)
    after = module.curve_similarity(
        module.resample_by_arc_length(source, 64),
        module.resample_by_arc_length(target, 64),
    )
    assert after > before + 0.05
    assert after > 0.999


def test_arc_class_prototype_is_sample_first_then_average():
    module = audit()
    samples = np.stack((
        _unit([[1, 0], [1, 1], [0, 1], [-1, 1]]),
        _unit([[1, 0], [0.9, 0.1], [0.8, 0.2], [-1, 1]]),
    ))
    result = module.arc_class_prototypes(samples, np.array([2, 2]), [2], 8)[2]
    expected = module.normalize_curve(np.mean([
        module.resample_by_arc_length(sample, 8) for sample in samples
    ], axis=0))
    assert np.allclose(result, expected)


def test_dtw_beats_calendar_for_monotonic_warp():
    module = audit()
    theta = np.linspace(0, np.pi, 64)
    source = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    warped_theta = np.linspace(0, np.sqrt(np.pi), 64) ** 2
    target = np.stack((np.cos(warped_theta), np.sin(warped_theta)), axis=1)
    dtw, _ = module.dtw_similarity(source, target)
    assert dtw > module.curve_similarity(source, target)


def test_class_summary_margin_is_same_minus_best_wrong():
    module = audit()
    matrix = {(0, 0): 0.8, (0, 1): 0.7, (0, 2): 0.1}
    row = module.class_summary_row("X_Y", "calendar", 0, matrix, None)
    assert row["nearest_wrong_class"] == 1
    assert np.isclose(row["same_vs_wrong_margin"], 0.1)


def test_task_evaluation_is_deterministic_and_emits_all_four_methods():
    module = audit()
    grid = np.arange(8, dtype=np.float64)
    first = _unit(np.stack((np.cos(grid / 3), np.sin(grid / 3)), axis=1))
    second = _unit(np.stack((np.cos(grid / 3 + 1), np.sin(grid / 3 + 1)), axis=1))
    curves = np.stack((first, first, second, second))
    labels = np.array([0, 0, 1, 1])
    outputs = module._evaluate_task(
        "X_Y", curves, labels, curves, labels, grid, ["a", "b"],
    )
    repeated = module._evaluate_task(
        "X_Y", curves, labels, curves, labels, grid, ["a", "b"],
    )
    assert outputs == repeated
    assert {row["method"] for row in outputs[2]} == set(module.METHODS)
    assert all(
        row["dtw_path_length"] != ""
        for row in outputs[0] if row["method"] == "dtw"
    )


class _Exposer:
    def __init__(self):
        self.analyze_calls = 0
        self.synthesize_calls = 0

    def analyze(self, prepared, positions):
        self.analyze_calls += 1
        return prepared

    def synthesize_shifted(self, coefficients, shift):
        self.synthesize_calls += 1
        return coefficients, torch.arange(coefficients.shape[1])


class _Branch:
    def __init__(self):
        self.exposer = _Exposer()

    def __call__(self, *args, **kwargs):
        raise AssertionError("window/token/anchor path must not run")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.structure_branch = _Branch()

    def spatial_encoder(self, pixels, valid, extra):
        return pixels * self.weight

    def prepare_temporal_features(self, spatial, positions):
        return spatial

    def prepare_structure_context(self, prepared, positions):
        return {"coefficients": self.structure_branch.exposer.analyze(prepared, positions)}


def test_extraction_stops_after_fourier_and_does_not_mutate_or_backward():
    module = audit()
    model = _Model().eval()
    before = model.weight.detach().clone()
    batch = {
        "pixels": torch.randn(3, 8, 4),
        "valid_pixels": torch.ones(3, 8, 4, dtype=torch.bool),
        "positions": torch.arange(8).repeat(3, 1),
        "extra": torch.zeros(3, 1),
    }
    curves, grid = module.extract_fourier_batch(model, batch)
    assert curves.shape == (3, 8, 4)
    assert grid.shape == (8,)
    assert model.structure_branch.exposer.analyze_calls == 1
    assert model.structure_branch.exposer.synthesize_calls == 1
    assert model.weight.grad is None
    assert torch.equal(model.weight, before)


def test_audit_contract_uses_only_validation_and_p_source():
    module = audit()
    assert module.DATASET_ROLES == ("source_val", "target_val")
    assert module.CHECKPOINT_ROLE == "P_source"
    assert module.MANIFEST_FLAGS == {
        "training": False,
        "target_train_accessed": False,
        "test_accessed": False,
        "uses_validation_labels": True,
    }
    assert set(module.TASKS) == {"AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1"}
