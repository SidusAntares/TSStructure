from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from models.stclassifier import FrozenStateOrgReference, PseStructureProtoLTae


def _model(readout="presence"):
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=128, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, shape_resample_length=24,
        fourier_num_modes=5, shape_representation="state_org",
        state_org_readout=readout, shape_injection="direct_response_query",
        structure_shift_mode="none", dropout=0.,
    )


def _batch(batch=3):
    return (
        torch.randn(batch, 10, 3, 4), torch.ones(batch, 10, 4),
        torch.arange(10).repeat(batch, 1) * 30, torch.zeros(batch, 4),
    )


def test_default_state_org_does_not_shift_but_explicit_context_shift_does():
    torch.manual_seed(501)
    model = _model().eval()
    pixels, mask, positions, extra = _batch()
    spatial = model.spatial_encoder(pixels, mask, extra)
    default_zero = model.prepare_structure(spatial, positions, 0)
    default_shift = model.prepare_structure(spatial, positions, 30)
    context = model.prepare_structure_context(spatial, positions)
    explicit_shift = model.prepare_structure_from_context(
        context, temporal_shift=30,
    )
    torch.testing.assert_close(
        default_zero["shapelet_similarity"], default_shift["shapelet_similarity"],
    )
    assert not torch.allclose(
        default_zero["shapelet_similarity"], explicit_shift["shapelet_similarity"],
    )


def test_circular_grid_shift_has_exact_inverse_roll_semantics():
    torch.manual_seed(503)
    exposer = _model().structure_branch.exposer
    coefficients = torch.randn(
        2, exposer.synthesizer.num_modes, 8, dtype=torch.complex64,
    )
    zero, _ = exposer.synthesize_shifted(coefficients, 0)
    grid_step = exposer.period_days / exposer.grid_points
    positive, _ = exposer.synthesize_shifted(coefficients, grid_step)
    negative, _ = exposer.synthesize_shifted(coefficients, -grid_step)
    torch.testing.assert_close(torch.roll(positive, -1, 1), zero, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(torch.roll(negative, 1, 1), zero, atol=2e-5, rtol=2e-5)


def test_oracle_shift_selector_uses_only_provided_validation_scores():
    from analysis.state_org_shift_sensitivity_audit import (
        select_oracle_class_shifts, shift_plan,
    )

    labels = np.array([0, 0, 1, 1])
    candidates = np.array([-1, 0, 1])
    log_prob = np.full((4, 3, 2), -9., dtype=np.float64)
    log_prob[:2, 0, 0] = 0.
    log_prob[2:, 2, 1] = 0.
    selected = select_oracle_class_shifts(log_prob, labels, candidates, 2)
    assert selected == {0: -1, 1: 1}
    plan = shift_plan(17, selected)
    assert plan["source"] == 0
    assert plan["target"] == {
        "raw": 0, "timematch_global": 17,
        "oracle_class_shift": {0: -1, 1: 1},
    }


def test_shifted_ltae_positions_remain_integer_indices():
    from analysis.state_org_shift_sensitivity_audit import _shift_positions

    positions = torch.tensor([[3, 8], [10, 20]], dtype=torch.long)
    shifted = _shift_positions(positions, torch.tensor([2., -4.]))
    assert shifted.dtype == torch.long
    torch.testing.assert_close(
        shifted, torch.tensor([[5, 10], [6, 16]], dtype=torch.long),
    )


def test_raw_and_soft_assignment_are_exact_and_presence_is_shared():
    from analysis.state_org_assignment_audit import assignment_views

    torch.manual_seed(509)
    similarity = torch.randn(4, 8, 16)
    presence = torch.randn(4, 16)
    views = assignment_views(similarity, presence, beta=5.)
    torch.testing.assert_close(views["raw"], similarity)
    torch.testing.assert_close(views["soft"], torch.softmax(5. * similarity, -1))
    torch.testing.assert_close(views["presence_soft"], presence)
    torch.testing.assert_close(views["presence_raw"], presence)
    assert torch.equal(views["raw"].argmax(-1), views["soft"].argmax(-1))


def test_geometric_update_modes_are_label_free_normalized_and_balanced():
    from timematch import balanced_geometric_token_banks, geometric_anchor_update

    torch.manual_seed(521)
    anchors = F.normalize(torch.randn(4, 6), dim=-1)
    source = F.normalize(torch.randn(13, 6), dim=-1)
    target = F.normalize(torch.randn(9, 6), dim=-1)
    balanced_source, balanced_target = balanced_geometric_token_banks(
        source, target, limit=10, seed=1,
    )
    assert balanced_source.shape[0] == balanced_target.shape[0] == 9

    fixed, fixed_diag = geometric_anchor_update(
        anchors, target, mode="fixed", step=.1, beta=5., source_tokens=source,
    )
    torch.testing.assert_close(fixed, anchors)
    target_updated, _ = geometric_anchor_update(
        anchors, target, mode="target_ema", step=.1, beta=5., source_tokens=source,
    )
    shared_updated, shared_diag = geometric_anchor_update(
        anchors, balanced_target, mode="shared_ema", step=.1, beta=5.,
        source_tokens=balanced_source,
    )
    torch.testing.assert_close(target_updated.norm(dim=-1), torch.ones(4))
    torch.testing.assert_close(shared_updated.norm(dim=-1), torch.ones(4))
    assert fixed_diag["mean_anchor_step_cosine"] == 1.
    assert shared_diag["min_anchor_effective_support"] > 0

    repeated = 3. * anchors.clone()
    fixed_source = repeated.clone()
    for _ in range(20):
        repeated, _ = geometric_anchor_update(
            repeated, target, mode="fixed", step=.1, beta=5.,
            source_tokens=source,
        )
    torch.testing.assert_close(repeated, fixed_source)

    different_source = F.normalize(torch.randn_like(source), dim=-1)
    target_only, _ = geometric_anchor_update(
        anchors, target, mode="target_ema", step=.1, beta=5.,
        source_tokens=different_source,
    )
    torch.testing.assert_close(target_only, target_updated)


def test_geometric_reference_is_frozen_outside_optimizer_and_teacher_syncs():
    from timematch import sync_reference_anchors

    student = _model("composition")
    reference = FrozenStateOrgReference.from_source_model(student)
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    optimizer_ids = {id(parameter) for parameter in student.parameters() if parameter.requires_grad}
    assert id(reference.structure_branch.shapelet_dictionary.anchors) not in optimizer_ids
    teacher = deepcopy(reference)
    new_anchor = F.normalize(torch.randn_like(
        reference.structure_branch.shapelet_dictionary.anchors,
    ), dim=-1)
    sync_reference_anchors(reference, teacher, new_anchor)
    torch.testing.assert_close(
        reference.structure_branch.shapelet_dictionary.anchors, new_anchor,
    )
    torch.testing.assert_close(
        teacher.structure_branch.shapelet_dictionary.anchors, new_anchor,
    )


def test_main_path_stays_trainable_with_frozen_geometric_reference():
    torch.manual_seed(523)
    model = _model("composition").train()
    reference = FrozenStateOrgReference.from_source_model(model).eval()
    model.configure_state_org_query("full", 1., reference)
    output = model.forward_with_temporal_shift(*_batch(), return_dict=True)
    F.cross_entropy(output["logits"], torch.tensor([0, 1, 2])).backward()
    assert model.spatial_encoder.mlp1[0].linear.weight.grad is not None
    assert model.temporal_encoder.attention_heads.key.weight.grad is not None
    assert next(model.decoder.parameters()).grad is not None
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_next_audit_launcher_has_three_tasks_and_nine_geometric_runs():
    text = Path("scripts/run_state_org_next_audits_seed1.sh").read_text()
    for task in ("AT1", "FR2", "DK1"):
        assert task in text
    assert "state_org_shift_sensitivity_audit.py" in text
    assert "state_org_assignment_audit.py" in text
    assert "for mode in fixed target_ema shared_ema" in text
    assert "--epochs 20 --steps_per_epoch 500" in text
    assert "--anchor-geometric-step 0.1" in text
    assert "--anchor-geometric-token-limit 50000" in text


def test_final_epoch_protocol_remains_checkpoint_last():
    from train import protocol_manifest

    manifest = protocol_manifest("timematch")
    assert manifest["uda_test_checkpoint"] == "checkpoint_last.pt"
    assert manifest["target_validation_used_for_test_selection"] is False


def test_final_evaluation_restores_frozen_state_org_reference():
    from train import restore_state_org_reference_for_evaluation

    model = _model("composition")
    reference = FrozenStateOrgReference.from_source_model(model)
    with torch.no_grad():
        reference.structure_branch.shapelet_dictionary.anchors.fill_(.125)
    packet = {
        "config": {
            "state_org_query_view": "full", "shape_query_scale": 1.,
            "output_student": True,
        },
        "state_org_reference_state_dict": reference.state_dict(),
    }
    restored = restore_state_org_reference_for_evaluation(model, packet)
    assert restored is not None
    torch.testing.assert_close(
        restored.structure_branch.shapelet_dictionary.anchors,
        reference.structure_branch.shapelet_dictionary.anchors,
    )
    assert model._frozen_state_org_reference is restored


def test_summary_script_is_directly_executable():
    result = subprocess.run(
        [sys.executable, "analysis/summarize_state_org_next_audits.py", "--help"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
