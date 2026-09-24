from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_script_resolves_repository_imports_outside_repo_working_directory(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "visualize_shapelet_pse_alignment.py"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--data-root", str(tmp_path),
            "--source", "france/30TXT/2017",
            "--target", "france/31TCJ/2017",
            "--start-checkpoint", str(tmp_path / "missing-start.pt"),
            "--end-checkpoint", str(tmp_path / "missing-end.pt"),
            "--output-dir", str(tmp_path / "output"),
            "--device", "cpu",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "checkpoint not found" in result.stderr
    assert "ModuleNotFoundError: No module named 'train'" not in result.stderr


def test_shared_pca_is_fit_once_and_transforms_all_four_groups():
    from scripts.visualize_shapelet_pse_alignment import project_feature_groups

    instances = []

    class SpyPCA:
        def __init__(self, n_components):
            self.n_components = n_components
            self.fit_calls = 0
            self.transform_calls = 0
            self.explained_variance_ratio_ = np.array([.6, .3])
            instances.append(self)

        def fit(self, values):
            self.fit_calls += 1
            self.fit_values = values.copy()
            return self

        def transform(self, values):
            self.transform_calls += 1
            return values[:, :2]

    groups = {
        "start_source": np.arange(24, dtype=float).reshape(2, 3, 4),
        "start_target": np.arange(24, 48, dtype=float).reshape(2, 3, 4),
        "end_source": np.arange(48, 72, dtype=float).reshape(2, 3, 4),
        "end_target": np.arange(72, 96, dtype=float).reshape(2, 3, 4),
    }
    pca, projected = project_feature_groups(
        groups, seed=7, max_points_per_domain=50_000, pca_factory=SpyPCA,
    )

    assert pca is instances[0]
    assert pca.fit_calls == 1
    assert pca.transform_calls == 4
    assert pca.fit_values.shape == (12, 4)
    assert set(projected) == set(groups)
    assert projected["end_target"].shape == (2, 3, 2)


def test_fixed_class_sampling_is_reproducible_without_replacement():
    from scripts.visualize_shapelet_pse_alignment import fixed_class_sample_indices

    labels = np.array([0] * 10 + [1] * 3 + [2] * 8)
    first = fixed_class_sample_indices(labels, class_ids=[0, 1, 2], per_class=5, seed=2)
    second = fixed_class_sample_indices(labels, class_ids=[0, 1, 2], per_class=5, seed=2)

    assert np.array_equal(first, second)
    assert len(np.unique(first)) == len(first)
    assert np.bincount(labels[first], minlength=3).tolist() == [5, 3, 5]


def test_temporal_bin_means_are_correct_and_empty_bins_are_nan():
    from scripts.visualize_shapelet_pse_alignment import temporal_bin_means

    positions = np.array([0., 10., 100., 364.])
    scores = np.array([1., 3., 8., 10.])
    centers, means = temporal_bin_means(positions, scores, num_bins=4)

    assert np.allclose(centers, [45.625, 136.875, 228.125, 319.375])
    assert means[0] == pytest.approx(2.)
    assert means[1] == pytest.approx(8.)
    assert np.isnan(means[2])
    assert means[3] == pytest.approx(10.)


def test_alignment_gap_uses_only_jointly_observed_bins():
    from scripts.visualize_shapelet_pse_alignment import alignment_gap

    source = np.array([1., np.nan, 5., 9.])
    target = np.array([3., 100., np.nan, 8.])
    assert alignment_gap(source, target) == pytest.approx(1.5)


def test_plot_smoke_writes_class_figures_and_overview(tmp_path):
    pytest.importorskip("matplotlib")
    from scripts.visualize_shapelet_pse_alignment import render_task_plots

    centers = np.linspace(7.5, 357.5, 24)
    curves = {}
    for stage in ("start", "end"):
        for domain in ("source", "target"):
            for class_id in (0, 1):
                offset = class_id + (0.2 if domain == "target" else 0.)
                if stage == "end" and domain == "target":
                    offset -= .1
                curves[(stage, domain, class_id)] = np.sin(centers / 50.) + offset

    gaps = render_task_plots(
        output_dir=tmp_path,
        task_name="FR1_FR2",
        class_names={0: "corn", 1: "meadow"},
        bin_centers=centers,
        curves=curves,
    )

    assert (tmp_path / "corn.png").is_file()
    assert (tmp_path / "meadow.png").is_file()
    assert (tmp_path / "overview.png").is_file()
    assert set(gaps) == {0, 1}


def test_final_student_checkpoint_contract_rejects_best_checkpoint():
    from scripts.visualize_shapelet_pse_alignment import validate_end_checkpoint_packet

    packet = {"epoch": 19, "config": {"epochs": 20, "output_student": True}}
    validate_end_checkpoint_packet(Path("checkpoint_last.pt"), packet)
    with pytest.raises(ValueError, match="checkpoint_last.pt"):
        validate_end_checkpoint_packet(Path("model.pt"), packet)
    with pytest.raises(ValueError, match="output_student=true"):
        validate_end_checkpoint_packet(
            Path("checkpoint_last.pt"),
            {"epoch": 19, "config": {"epochs": 20, "output_student": False}},
        )
