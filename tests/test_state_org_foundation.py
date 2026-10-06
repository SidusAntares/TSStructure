from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from models.stclassifier import FrozenStateOrgReference, PseStructureProtoLTae
from models.structure_da.discriminative_structure import DiscriminativeStructureBranch


def _branch(readout="full"):
    return DiscriminativeStructureBranch(
        channels=4, shape_dim=12, num_modes=3, grid_points=64,
        window_scales=(24,), window_stride=8, shapelet_count=16,
        shape_representation="state_org", state_org_readout=readout,
    ).eval()


def _model(readout="full"):
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


@pytest.mark.parametrize("readout", ("full", "composition", "presence"))
def test_state_org_readouts_are_dimension_matched(readout):
    branch = _branch(readout)
    result = branch.compose_state_org_response(torch.randn(3, 8, 16))
    assert result["presence"].shape == (3, 16)
    assert result["organization"].shape == (3, 32)
    assert result["shapelet_response"].shape == (3, 48)


def test_presence_has_no_organization_encoder_and_composition_is_permutation_invariant():
    presence = _branch("presence")
    assert not hasattr(presence, "organization_encoder")
    assert not hasattr(presence, "composition_encoder")
    similarity = torch.randn(3, 8, 16)
    assert torch.count_nonzero(
        presence.compose_state_org_response(similarity)["organization"]
    ) == 0

    composition = _branch("composition")
    first = composition.compose_state_org_response(similarity)
    permuted = composition.compose_state_org_response(similarity[:, [3, 0, 7, 2, 5, 1, 6, 4]])
    torch.testing.assert_close(first["shapelet_response"], permuted["shapelet_response"])


def test_full_is_circular_roll_invariant_but_order_sensitive():
    torch.manual_seed(91)
    branch = _branch("full")
    similarity = torch.randn(3, 8, 16)
    original = branch.compose_state_org_response(similarity)["shapelet_response"]
    rolled = branch.compose_state_org_response(torch.roll(similarity, 3, 1))["shapelet_response"]
    permuted = branch.compose_state_org_response(similarity[:, [3, 0, 7, 2, 5, 1, 6, 4]])[
        "shapelet_response"
    ]
    torch.testing.assert_close(original, rolled, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(original, permuted)


def test_presence_basis_semantic_features_do_not_require_organization_encoder():
    from analysis.state_org_foundation_audit import semantic_features

    branch = _branch("presence")
    details = branch.compose_state_org_response(torch.randn(2, 8, 16))
    features = semantic_features(details["presence"], details["state_distribution"])
    assert set(features) == {"P", "P_C", "P_T1", "P_T1_T2"}
    assert features["P"].shape == (2, 16)
    assert features["P_T1_T2"].shape == (2, 16 + 2 * 16 * 16)


@pytest.mark.parametrize(
    "mode,source_expected,target_expected",
    (("fixed", False, False), ("source", True, False),
     ("target", False, True), ("shared", True, True)),
)
def test_anchor_gradient_semantics(mode, source_expected, target_expected):
    from timematch import anchor_gradient_enabled

    model = _model().eval()
    with torch.no_grad():
        model.temporal_encoder.attention_heads.external_query_projection.weight.normal_()
    reference = FrozenStateOrgReference.from_source_model(
        model, trainable_anchors=True,
    )
    model.configure_state_org_query("full", 1., reference)
    batch = _batch()
    for domain, expected in (("source", source_expected), ("target", target_expected)):
        reference.structure_branch.shapelet_dictionary.anchors.grad = None
        output = model.forward_with_temporal_shift(
            *batch, return_dict=True,
            structure_anchor_grad=anchor_gradient_enabled(mode, domain),
        )
        output["logits"].sum().backward()
        gradient = reference.structure_branch.shapelet_dictionary.anchors.grad
        has_gradient = gradient is not None and bool(torch.count_nonzero(gradient).item())
        assert has_gradient is expected
        assert all(
            parameter.grad is None
            for name, parameter in reference.named_parameters()
            if name != "structure_branch.shapelet_dictionary.anchors"
        )


def test_teacher_anchor_reference_is_distinct_and_ema_updated():
    from timematch import update_reference_anchor_ema

    student = FrozenStateOrgReference.from_source_model(_model(), trainable_anchors=True)
    teacher = deepcopy(student)
    assert student.structure_branch.shapelet_dictionary.anchors.data_ptr() != (
        teacher.structure_branch.shapelet_dictionary.anchors.data_ptr()
    )
    before = teacher.structure_branch.shapelet_dictionary.anchors.detach().clone()
    with torch.no_grad():
        student.structure_branch.shapelet_dictionary.anchors.add_(1.)
    update_reference_anchor_ema(student, teacher, .5)
    expected = .5 * before + .5 * student.structure_branch.shapelet_dictionary.anchors
    torch.testing.assert_close(teacher.structure_branch.shapelet_dictionary.anchors, expected)


def test_norm_controlled_query_cases():
    from analysis.state_org_query_role_audit import compose_query_case

    torch.manual_seed(97)
    master = torch.randn(4, 8)
    correction = torch.randn(3, 4, 8)
    master_only, _ = compose_query_case(master, correction, "rho", rho=0.)
    bounded, ratio = compose_query_case(master, correction, "rho", rho=1.)
    raw, _ = compose_query_case(master, correction, "raw")
    shape_only, _ = compose_query_case(master, correction, "shape_only")
    torch.testing.assert_close(master_only, master[None].expand(3, -1, -1))
    torch.testing.assert_close(raw, master[None] + correction)
    torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=1e-5, rtol=1e-5)
    assert not torch.allclose(shape_only, master[None])
    torch.testing.assert_close(
        shape_only.norm(dim=-1), master.norm(dim=-1)[None].expand(3, -1),
        atol=1e-5, rtol=1e-5,
    )


def test_raw_explicit_query_matches_normal_ltae_output():
    from analysis.state_org_query_role_audit import compose_query_case

    torch.manual_seed(101)
    model = _model().eval()
    with torch.no_grad():
        model.temporal_encoder.attention_heads.external_query_projection.weight.normal_()
    pixels, mask, positions, extra = _batch()
    spatial = model.spatial_encoder(pixels, mask, extra)
    structure = model.prepare_structure(spatial, positions, 0)
    evidence = model.shape_response_norm(structure["shapelet_response"])
    normal, normal_attention = model.temporal_encoder(
        spatial, positions, external_query=evidence, return_att=True,
    )
    heads = model.temporal_encoder.attention_heads
    correction = heads.external_query_projection(evidence).reshape(
        evidence.shape[0], heads.n_head, heads.d_k,
    )
    query, _ = compose_query_case(heads.query, correction, "raw")
    explicit, explicit_attention = model.temporal_encoder.forward_with_explicit_queries(
        spatial, positions, query[:, None], return_att=True,
    )
    torch.testing.assert_close(normal, explicit[:, 0])
    torch.testing.assert_close(normal_attention, explicit_attention)


def test_frozen_external_evidence_survives_complete_optimizer_step_and_health_snapshot():
    from methods.structure_da.prototype_losses import shape_health_snapshot

    student = _model().train()
    reference = FrozenStateOrgReference.from_source_model(student).eval()
    student.configure_state_org_query("full", 1., reference)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    source = student.forward_with_temporal_shift(*_batch(), return_dict=True)
    target = student.forward_with_temporal_shift(*_batch(), return_dict=True)
    loss = F.cross_entropy(source["logits"], torch.tensor([0, 1, 2]))
    loss = loss + F.cross_entropy(target["logits"], torch.tensor([1, 2, 0]))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    snapshot = shape_health_snapshot(student, source)
    assert "shape_response_effective_rank" in snapshot


def test_foundation_launcher_contract_and_safe_local_binding():
    text = Path("scripts/run_state_org_foundation_audit_seed1.sh").read_text()
    for readout in ("full", "composition", "presence"):
        assert f'for readout in full composition presence' in text
        assert '--state-org-readout "$readout"' in text
    for mode in ("fixed", "source", "target", "shared"):
        assert mode in text
    assert "--epochs 20 --steps_per_epoch 500" in text
    assert "--freeze-state-org-query true" in text
    assert "--shapelet-diversity-weight 0" in text
    assert "--uda-shape-class-weight 0" in text
    assert 'local task="${src}_${tgt}"' in text
    assert 'tgt_data="$5" task="${src}_${tgt}"' not in text
