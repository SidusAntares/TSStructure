from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models.stclassifier import PseStructureProtoLTae


def _model():
    return PseStructureProtoLTae(
        input_dim=3, mlp1=[3, 4], mlp2=[8, 8], with_extra=False,
        n_head=2, d_k=4, d_model=8, mlp3=[8, 6], mlp4=[6],
        num_classes=3, shape_dim=128, shape_window_scales=(24,),
        shape_window_stride=8, shapelet_count=16, shape_resample_length=24,
        fourier_num_modes=5, shape_representation="state_org",
        shape_injection="direct_response_query", structure_shift_mode="none",
        dropout=0.,
    )


def _batch(batch=3):
    return (
        torch.randn(batch, 10, 3, 4), torch.ones(batch, 10, 4),
        torch.arange(10).repeat(batch, 1) * 30, torch.zeros(batch, 4),
    )


def test_frozen_reference_is_immutable_while_student_main_path_trains():
    from models.stclassifier import FrozenStateOrgReference

    torch.manual_seed(701)
    student = _model().train()
    reference = FrozenStateOrgReference.from_source_model(student).eval()
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    student.configure_state_org_query("full", 1., reference)
    student_parameter_ids = {id(parameter) for parameter in student.parameters()}
    assert all(id(parameter) not in student_parameter_ids for parameter in reference.parameters())
    batch = _batch()
    before = reference(*batch).detach().clone()
    with torch.no_grad():
        spatial_before = student.spatial_encoder(batch[0], batch[1], batch[3])
        adaptive_before = student._shape_evidence(
            student.prepare_structure(spatial_before, batch[2], 0)
        ).clone()
    evidence = reference(*batch)
    output = student.forward_with_external_shape_evidence(
        *batch, shape_evidence=evidence, query_view="full", query_scale=1.,
    )
    F.cross_entropy(output["logits"], torch.tensor([0, 1, 2])).backward()
    assert next(student.spatial_encoder.parameters()).grad is not None
    assert student.temporal_encoder.attention_heads.key.weight.grad is not None
    assert next(student.decoder.parameters()).grad is not None
    optimizer = torch.optim.SGD(student.parameters(), lr=.01)
    optimizer.step()
    torch.testing.assert_close(reference(*batch), before)
    with torch.no_grad():
        spatial_after = student.spatial_encoder(batch[0], batch[1], batch[3])
        adaptive_after = student._shape_evidence(
            student.prepare_structure(spatial_after, batch[2], 0)
        )
    assert not torch.allclose(adaptive_before, adaptive_after)


def test_presence_mask_is_after_shared_norm_and_query_scale_is_post_projection():
    torch.manual_seed(703)
    model = _model().eval()
    response = torch.randn(3, 48)
    normalized = model.shape_response_norm(response)
    presence = model.select_state_org_query(normalized, "presence")
    torch.testing.assert_close(presence[:, :16], normalized[:, :16])
    assert torch.count_nonzero(presence[:, 16:]) == 0
    projection = model.temporal_encoder.attention_heads.external_query_projection
    with torch.no_grad():
        projection.weight.normal_(std=.1)
    x = torch.randn(3, 10, 8); positions = torch.arange(10).repeat(3, 1)
    master = model.temporal_encoder(x, positions, external_query=None)
    zero = model.temporal_encoder(x, positions, external_query=normalized, query_scale=0.)
    one = model.temporal_encoder(x, positions, external_query=normalized, query_scale=1.)
    legacy = model.temporal_encoder(x, positions, external_query=normalized)
    torch.testing.assert_close(zero, master)
    torch.testing.assert_close(one, legacy)
    full = projection(normalized)
    parts = projection(torch.cat((normalized[:, :16], torch.zeros_like(normalized[:, 16:])), 1))
    parts = parts + projection(torch.cat((torch.zeros_like(normalized[:, :16]), normalized[:, 16:]), 1))
    torch.testing.assert_close(full, parts)


def test_circular_t1_t2_are_roll_invariant_and_reverse_transposes_t1():
    from analysis.state_org_anchor_basis_audit import organization_statistics

    torch.manual_seed(709)
    q = torch.softmax(torch.randn(4, 8, 5), -1)
    base = organization_statistics(q)
    rolled = organization_statistics(torch.roll(q, 3, 1))
    reversed_stats = organization_statistics(torch.flip(q, (1,)))
    torch.testing.assert_close(base["composition"], rolled["composition"])
    torch.testing.assert_close(base["t1"], rolled["t1"])
    torch.testing.assert_close(base["t2"], rolled["t2"])
    torch.testing.assert_close(reversed_stats["t1"], base["t1"].transpose(1, 2))


def test_counterfactual_differences_report_feature_query_logit_and_prediction():
    from analysis.state_org_anchor_basis_audit import counterfactual_differences

    original = {
        "organization": torch.zeros(2, 3), "query": torch.zeros(2, 4),
        "logits": torch.tensor([[2., 0.], [0., 2.]]),
    }
    changed = {
        "organization": torch.ones(2, 3), "query": torch.ones(2, 4),
        "logits": torch.tensor([[0., 2.], [0., 2.]]),
    }
    row = counterfactual_differences(original, changed)
    assert row["organization_feature_l2_diff"] > 0
    assert row["query_correction_l2_diff"] > 0
    assert row["logit_l2_diff"] > 0
    assert row["prediction_change_rate"] == .5


def test_peak_final_drop_uses_same_scale_validation_history():
    from analysis.summarize_state_org_hierarchical import validation_history_stats

    result = validation_history_stats([.71, .76, .73])
    assert result["best_val"] == .76
    assert result["final_val"] == .73
    assert result["peak_final_drop"] == pytest.approx(.03)


def test_hierarchical_launcher_contract():
    text = Path("scripts/run_state_org_hierarchical_audit_seed1.sh").read_text()
    for task in ("AT1", "FR2", "DK1"):
        assert task in text
    for variant in ("adaptive_presence", "frozen_full", "frozen_presence"):
        assert variant in text
    assert "outputs/state_org_feasibility/variants/no_shape_aux" in text
    assert "--epochs 8" in text and "--steps_per_epoch 500" in text
    assert "--uda-shape-class-weight 0" in text
    assert "analysis/state_org_anchor_basis_audit.py" in text
    assert "analysis/summarize_state_org_hierarchical.py" in text
