import json
from pathlib import Path

import pytest
import torch

from models.stclassifier import PseStructureProtoLTae
from models.structure_da.discriminative_structure import DiscriminativeStructureBranch


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


def test_launcher_reuses_one_source_checkpoint_for_a_and_b_and_has_fixed_matrix():
    text = Path("scripts/run_reliable_composition_ab_4tasks_3seeds.sh").read_text()
    for task in ("AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1"):
        assert task in text
    assert "for seed in 1 2 3" in text
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
    assert text.count('train_source "$gpu"') == 1
    assert text.index("train_source") < text.index("run_uda \"$gpu\" A")
    assert text.index("run_uda \"$gpu\" A") < text.index("run_uda \"$gpu\" B")


def test_summary_reports_absolute_best_and_final_mean_std_without_historical_gain():
    from analysis.summarize_reliable_composition_ab import aggregate_results

    rows = [
        {"variant": "A", "task": "AT1_DK1", "seed": seed,
         "best_test_macro_f1": best, "final_test_macro_f1": final}
        for seed, best, final in ((1, .6, .5), (2, .7, .6), (3, .8, .7))
    ]
    summary = aggregate_results(rows)
    assert len(summary) == 2
    by_stage = {row["stage"]: row for row in summary}
    assert by_stage["best"]["test_macro_f1_mean"] == pytest.approx(.7)
    assert by_stage["best"]["test_macro_f1_std"] == pytest.approx(torch.tensor([.6, .7, .8]).std(unbiased=False).item())
    assert by_stage["final"]["test_macro_f1_mean"] == pytest.approx(.6)
    assert all("gain" not in key.lower() for row in summary for key in row)
