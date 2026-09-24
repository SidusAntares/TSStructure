import numpy as np
import pytest
from types import SimpleNamespace


def test_effective_rank_distinguishes_collapsed_and_independent_responses():
    from analysis.structure_efficiency_metrics import effective_rank

    collapsed = np.ones((64, 8))
    independent = np.eye(8)
    assert effective_rank(collapsed) == pytest.approx(1.0, abs=1e-6)
    assert effective_rank(independent) > 7.9


def test_candidate_effective_number_matches_one_hot_and_uniform_weights():
    from analysis.structure_efficiency_metrics import candidate_effective_number

    one_hot = np.zeros((2, 48, 3))
    one_hot[:, 0, :] = 1
    uniform = np.full((2, 48, 3), 1 / 48)
    assert np.allclose(candidate_effective_number(one_hot), 1)
    assert np.allclose(candidate_effective_number(uniform), 48)


def test_checkpoint_resolution_prints_full_missing_path_and_continues(tmp_path, capsys):
    from scripts.diagnose_structure_efficiency import resolve_task_checkpoint

    result = resolve_task_checkpoint(tmp_path, "AT1", "DK1")
    assert result is None
    output = capsys.readouterr().out
    expected = tmp_path / "uda" / "AT1_DK1_seed1" / "fold_0" / "model.pt"
    assert f"MISSING|task=AT1_DK1|checkpoint={expected}" in output


def test_all_missing_checkpoints_still_emit_complete_empty_report(tmp_path, capsys):
    from scripts.diagnose_structure_efficiency import run

    output = tmp_path / "diagnostics"
    run(SimpleNamespace(
        checkpoint_root=tmp_path / "checkpoints", data_root=tmp_path / "data",
        output_dir=output, log_root=tmp_path / "logs", device="cpu", seed=1,
        batch_size=2, samples_per_class=2, max_candidates_per_class=8,
    ))
    text = capsys.readouterr().out
    assert text.count("MISSING|task=") == 4
    for name in (
        "summary.csv", "scale_ablation.csv", "anchor_diagnostics.csv",
        "complexity_coverage.csv", "fourier_reconstruction.csv",
        "timestamp_reuse.csv", "source_convergence.csv", "report.md",
    ):
        assert (output / name).is_file(), name


def test_synthetic_domain_collection_covers_all_ablation_and_representation_levels():
    import torch
    from models.stclassifier import PseStructureProtoLTae
    from scripts.diagnose_structure_efficiency import _collect_domain

    torch.manual_seed(53)
    model = PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=2, shape_dim=6, shapelet_count=3, fourier_num_modes=5,
    ).eval()
    sample = {
        "pixels": torch.randn(2, 10, 3, 5),
        "valid_pixels": torch.ones(2, 10, 5),
        "positions": torch.arange(10).repeat(2, 1) * 20,
        "extra": torch.zeros(2, 4),
        "label": torch.tensor([0, 1]),
    }
    result = _collect_domain(model, [sample], "S", [0, 1], max_candidates=48)
    assert set(result["predictions"]) == {
        "FULL", "ONLY_Q8", "ONLY_Q16", "ONLY_Q24",
        "REMOVE_Q8", "REMOVE_Q16", "REMOVE_Q24", "STRIDE_8", "STRIDE_16",
    }
    assert result["candidate_counts"]["FULL"] == 24
    assert result["candidate_counts"]["STRIDE_8"] == 24
    assert result["candidate_counts"]["STRIDE_16"] == 12
    assert set(result["representations"]) == {
        "normalized_morphology", "first_difference", "mean", "std", "shape_token",
    }
    assert result["responses"].shape == (2, 3)
    assert len(result["anchor_removed"]) == 3
    assert all(np.isfinite(values).all() for values in result["reconstruction"].values())
