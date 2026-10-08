import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.stclassifier import PseStructureProtoLTae
from models.structure_da.discriminative_structure import DiscriminativeStructureBranch
from models.structure_da.discriminative_structure import MultiScaleWindowExtractor


def _branch():
    return DiscriminativeStructureBranch(
        3, shape_dim=128, num_modes=5, grid_points=64,
        window_scales=(24,), window_stride=1, shapelet_count=16,
        shapelet_beta=5., shape_resample_length=7,
        shape_representation="state_org",
        state_org_readout="reliable_composition",
    )


def _model():
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=128, shape_window_scales=(24,),
        shape_window_stride=1, shapelet_count=16, shape_resample_length=24,
        fourier_num_modes=5, shape_representation="state_org",
        state_org_readout="reliable_composition",
        shape_injection="direct_response_query", structure_shift_mode="none",
        dropout=0.,
    )


def _batch(batch=2):
    pixels = torch.randn(batch, 10, 3, 4)
    mask = torch.ones(batch, 10, 4)
    positions = (torch.arange(10) * 30).repeat(batch, 1).long()
    extra = torch.zeros(batch, 4)
    return pixels, mask, positions, extra


def test_reliable_composition_shapes_normalization_and_unmatched_mass():
    branch = _branch().eval()
    similarity = torch.randn(3, 64, 16)
    result = branch.compose_state_org_response(similarity)
    assert result["presence"].shape == (3, 16)
    assert result["state_distribution"].shape == (3, 64, 16)
    assert result["reliable_composition"].shape == (3, 16)
    assert result["organization"].shape == (3, 32)
    assert result["shapelet_response"].shape == (3, 48)
    assert result["unmatched_composition"].shape == (3,)
    torch.testing.assert_close(
        result["state_distribution"].sum(-1), torch.ones(3, 64),
    )
    expected = 1. - result["reliable_composition"].sum(-1)
    torch.testing.assert_close(result["unmatched_composition"], expected)
    assert (result["unmatched_composition"] >= -1e-6).all()
    assert (result["unmatched_composition"] <= 1. + 1e-6).all()


def test_reliable_presence_uses_normalized_group_logmeanexp_then_softmax_pooling():
    branch = _branch().eval()
    similarity = torch.full((2, 64, 16), .25)
    result = branch.compose_state_org_response(similarity)
    torch.testing.assert_close(result["presence"], torch.full((2, 16), .25))
    assert result["local_presence"].shape == (2, 8, 16)
    assert result["local_presence_weights"].shape == (2, 8, 16)
    torch.testing.assert_close(
        result["local_presence_weights"].sum(1), torch.ones(2, 16),
    )


def test_source_calibration_buffers_save_load_and_freeze_for_uda():
    source = _branch().train()
    assert torch.count_nonzero(source.reliable_tau) == 0
    torch.testing.assert_close(source.reliable_scale, torch.ones(16))
    assert int(source.reliable_calibration_updates) == 0
    source.compose_state_org_response(torch.linspace(-1, 1, 4 * 64 * 16).reshape(4, 64, 16))
    assert int(source.reliable_calibration_updates) == 1
    assert torch.count_nonzero(source.reliable_tau) > 0
    packet = source.state_dict()

    uda = _branch()
    uda.load_state_dict(packet, strict=True)
    before_tau = uda.reliable_tau.clone()
    before_scale = uda.reliable_scale.clone()
    before_updates = uda.reliable_calibration_updates.clone()
    uda.set_reliable_calibration_updates(False)
    uda.train()
    uda.compose_state_org_response(torch.randn(3, 64, 16) + 10.)
    torch.testing.assert_close(uda.reliable_tau, before_tau)
    torch.testing.assert_close(uda.reliable_scale, before_scale)
    torch.testing.assert_close(uda.reliable_calibration_updates, before_updates)
    assert not any(parameter.requires_grad for parameter in (
        uda.reliable_tau, uda.reliable_scale, uda.reliable_calibration_updates,
    ))


def test_calibration_uses_one_joint_quantile_and_matches_legacy_rule(monkeypatch):
    branch = _branch().train()
    values = torch.linspace(-1.25, 1.75, 3 * 64 * 16).reshape(3, 64, 16)
    flat = values.reshape(-1, 16).float()
    legacy_q25 = torch.quantile(flat, .25, dim=0)
    legacy_q75 = torch.quantile(flat, .75, dim=0)
    expected_tau = .1 * legacy_q75
    expected_scale = .9 * torch.ones(16) + .1 * (
        legacy_q75 - legacy_q25
    ).clamp_min(.05)

    original_quantile = torch.quantile
    calls = []

    def recording_quantile(input_tensor, q, *args, **kwargs):
        calls.append((input_tensor.shape, torch.as_tensor(q).clone()))
        return original_quantile(input_tensor, q, *args, **kwargs)

    monkeypatch.setattr(torch, "quantile", recording_quantile)
    branch.update_reliable_calibration(values)
    assert len(calls) == 1
    assert calls[0][0] == (3 * 64, 16)
    torch.testing.assert_close(calls[0][1], torch.tensor([.25, .75]))
    torch.testing.assert_close(branch.reliable_tau, expected_tau)
    torch.testing.assert_close(branch.reliable_scale, expected_scale)
    branch.eval()
    result = branch.compose_state_org_response(values)
    expected_distribution = torch.softmax(5. * values, dim=-1)
    expected_gate = torch.sigmoid(
        (values - expected_tau) / expected_scale,
    )
    expected_composition = (expected_gate * expected_distribution).mean(1)
    torch.testing.assert_close(
        result["reliable_composition"], expected_composition,
    )


def test_pseudo_audit_tensor_mode_keeps_step_statistics_on_device():
    from timematch import accepted_pseudo_statistics

    pseudo = torch.tensor([0, 1, 2, 1])
    target = torch.tensor([0, 2, 2, 1])
    mask = torch.tensor([True, True, False, True])
    statistics = accepted_pseudo_statistics(
        pseudo, target, mask, num_classes=3, return_tensors=True,
    )
    assert all(torch.is_tensor(value) for value in statistics.values())
    assert statistics["accepted_count"].item() == 3
    assert statistics["correct_count"].item() == 2
    torch.testing.assert_close(
        statistics["class_accepted_count"], torch.tensor([1, 1, 1]),
    )


def test_window_extractor_does_not_rebuild_scale_or_center_metadata(monkeypatch):
    extractor = MultiScaleWindowExtractor((24,), stride=1, grid_points=64)
    curve = torch.randn(2, 64, 8)

    def unexpected_arange(*args, **kwargs):
        raise AssertionError("window metadata must be cached at construction")

    monkeypatch.setattr(torch, "arange", unexpected_arange)
    windows, scales = extractor(curve)
    assert windows[0].shape == (2, 64, 24, 8)
    assert scales.shape == (64,)
    _, cached_scales, centers = extractor(curve, return_centers=True)
    torch.testing.assert_close(cached_scales, scales)
    assert centers.shape == (64,)


def test_timematch_freezes_loaded_source_calibration_for_all_uda_forwards():
    from timematch import freeze_reliable_composition_calibration

    model = _model().train()
    model.structure_branch.compose_state_org_response(torch.randn(2, 64, 16))
    diagnostics = freeze_reliable_composition_calibration(model)
    assert diagnostics["updates"] == 1
    before = model.structure_branch.reliable_tau.clone()
    model.structure_branch.compose_state_org_response(torch.randn(2, 64, 16) + 8.)
    torch.testing.assert_close(model.structure_branch.reliable_tau, before)


def test_reliable_composition_forward_has_64_windows_and_required_diagnostics():
    model = _model().eval()
    with torch.no_grad():
        output = model(*_batch(), return_dict=True)
    assert output["shapelet_similarity"].shape == (2, 64, 16)
    assert output["shapelet_response"].shape == (2, 48)
    assert output["reliable_composition"].shape == (2, 16)
    assert output["unmatched_composition"].shape == (2,)


def test_staged_detach_schedule_is_first_five_zero_based_epochs_only():
    from timematch import target_structure_detached_for_epoch

    config_a = type("Config", (), {
        "detach_target_structure": False, "target_structure_detach_epochs": 0,
    })()
    config_b = type("Config", (), {
        "detach_target_structure": False, "target_structure_detach_epochs": 5,
    })()
    assert not any(target_structure_detached_for_epoch(config_a, epoch) for epoch in range(20))
    assert [target_structure_detached_for_epoch(config_b, epoch) for epoch in range(20)] == (
        [True] * 5 + [False] * 15
    )


def _gradient_snapshot(detach):
    torch.manual_seed(7)
    model = _model().train()
    model.structure_branch.set_reliable_calibration_updates(False)
    projection = model.temporal_encoder.attention_heads.external_query_projection
    with torch.no_grad():
        projection.weight.normal_(0., .05)
    output = model.forward_with_temporal_shift(
        *_batch(), return_dict=True, detach_structure_query=detach,
    )
    output["logits"].square().mean().backward()
    return {
        "pse": next(model.spatial_encoder.parameters()).grad,
        "token": next(model.structure_branch.token_generator.parameters()).grad,
        "anchor": model.structure_branch.shapelet_dictionary.anchors.grad,
        "composition": next(model.structure_branch.reliable_composition_encoder.parameters()).grad,
        "projection": projection.weight.grad,
    }


def test_target_query_detach_cuts_structure_path_but_preserves_main_pse_path():
    detached = _gradient_snapshot(True)
    open_path = _gradient_snapshot(False)
    assert detached["pse"] is not None and detached["pse"].abs().sum() > 0
    for name in ("token", "anchor", "composition", "projection"):
        assert detached[name] is None or detached[name].abs().sum() == 0, name
        assert open_path[name] is not None and open_path[name].abs().sum() > 0, name


def test_launcher_is_seed1_only_and_reuses_one_source_checkpoint_for_a_and_b():
    text = Path("scripts/run_reliable_composition_ab_seed1.sh").read_text()
    for task in ("AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1"):
        assert task in text
    assert "seed=1" in text
    assert "seed2" not in text and "seed3" not in text
    assert "--state-org-readout reliable_composition" in text
    assert "--shape-window-stride 1" in text
    assert '--target-structure-detach-epochs "$detach_epochs"' in text
    assert 'run_uda "$gpu" A 0' in text
    assert 'run_uda "$gpu" B 5' in text
    assert "--structure-basis-mode adaptive" in text
    assert "--shape-align-weight 0" in text
    assert "--uda-shape-class-weight 0" in text
    assert "--freeze-structure-specific false" in text
    assert "--freeze-state-org-query false" in text
    assert "DRY_RUN" in text and "SKIP_SOURCE" in text
    assert 'if [[ "$DRY_RUN" == "1" && "$SKIP_SOURCE" == "1" ]]' in text
    assert "if (( status != 3 )); then" in text
    assert text.count('ensure_source "$gpu"') == 1
    assert text.index("train_source") < text.index("run_uda \"$gpu\" A")
    assert text.index("run_uda \"$gpu\" A") < text.index("run_uda \"$gpu\" B")


def test_summary_reports_seed1_absolute_best_and_final_without_mean_or_gain():
    from analysis import summarize_reliable_composition_ab as summary

    rows = [
        {"variant": variant, "task": "AT1_DK1", "seed": 1,
         "best_test_macro_f1": best, "final_test_macro_f1": final}
        for variant, best, final in (("A", .7, .6), ("B", .8, .75))
    ]
    result = summary.seed1_results(rows)
    assert summary.np.arange(3).tolist() == [0, 1, 2]
    assert len(result) == 2
    assert {row["seed"] for row in result} == {1}
    assert all(
        "mean" not in key.lower() and "std" not in key.lower()
        and "gain" not in key.lower()
        for row in result for key in row
    )


def test_checkpoint_status_distinguishes_missing_incomplete_and_complete(tmp_path):
    from analysis.summarize_reliable_composition_ab import checkpoint_status

    fold = tmp_path / "missing_run" / "fold_0"
    assert checkpoint_status(fold, "uda") == "missing"
    fold.mkdir(parents=True)
    torch.save({"epoch": 19, "config": SimpleNamespace(epochs=20)}, fold / "checkpoint_last.pt")
    assert checkpoint_status(fold, "uda") == "incomplete"
    torch.save({"epoch": 7}, fold / "checkpoint_best.pt")
    assert checkpoint_status(fold, "uda") == "complete"

    source = tmp_path / "source_fold"
    source.mkdir()
    torch.save({"epoch": 42, "config": SimpleNamespace(epochs=100)}, source / "model.pt")
    assert checkpoint_status(source, "source") == "incomplete"
    torch.save({"epoch": 98, "config": SimpleNamespace(epochs=100)}, source / "checkpoint_last.pt")
    assert checkpoint_status(source, "source") == "incomplete"
    torch.save({"epoch": 99, "config": SimpleNamespace(epochs=100)}, source / "checkpoint_last.pt")
    assert checkpoint_status(source, "source") == "complete"
    (source / "checkpoint_last.pt").write_bytes(b"not a checkpoint")
    assert checkpoint_status(source, "source") == "incomplete"
