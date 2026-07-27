import numpy as np

from analysis.structure_da.checkpoint_analysis import (
    _sample_pixels,
    component_energy_ratios,
    component_similarities,
    diversity_diagnostics,
    fit_pca_2d,
    normalize_extra_features,
    quality_scores_long_form,
)


def test_component_energy_ratios_and_cosines():
    trend = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    dynamics = np.array([[[0.0, 2.0], [0.0, 2.0]]])
    residual = np.array([[[1.0, 0.0], [1.0, 0.0]]])

    ratios = component_energy_ratios(trend, dynamics, residual)
    similarities = component_similarities(trend, dynamics, residual)

    assert np.allclose(ratios[0], [1 / 6, 4 / 6, 1 / 6])
    assert np.allclose(similarities["T_D"], 0)
    assert np.allclose(similarities["T_R"], 1)


def test_quality_long_form_and_population_diversity_cv():
    frame = quality_scores_long_form(
        "run", np.array(["source", "source"]), np.array(["a", "b"]),
        {"T_component": {"transferability": np.array([0.2, 0.4]), "diversity": np.array([0.1, 0.3])}},
    )
    class_means, summary = diversity_diagnostics(
        {"T_component": np.array([0.1, 0.3])}, np.array([0, 1]), ("a", "b")
    )

    assert set(frame["metric"]) == {"transferability", "diversity"}
    assert list(class_means["mean"]) == [0.1, 0.3]
    assert np.isclose(summary.iloc[0]["std_of_class_means"], 0.1)
    assert np.isclose(summary.iloc[0]["cv"], 0.5)


def test_pca_helper_is_deterministic():
    features = np.arange(30, dtype=np.float64).reshape(10, 3)

    first, first_variance = fit_pca_2d(features)
    second, second_variance = fit_pca_2d(features)

    assert np.allclose(first, second)
    assert np.allclose(first_variance, second_variance)


def test_pixel_sampling_matches_training_padding_and_valid_mask():
    pixels = np.arange(12, dtype=np.float32).reshape(2, 3, 2)

    sampled, valid = _sample_pixels(pixels, count=4, seed=7)

    assert sampled.shape == (2, 3, 4)
    assert np.allclose(sampled[..., :2], pixels / 65535.0)
    assert np.allclose(sampled[..., 2:], sampled[..., :1])
    assert np.array_equal(valid, np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float32))


def test_extra_features_use_training_normalization_constants():
    extra = np.array([40000.0, 1e8, 40000.0, 1.0], dtype=np.float32)

    assert np.array_equal(normalize_extra_features(extra), np.ones(4, dtype=np.float32))
