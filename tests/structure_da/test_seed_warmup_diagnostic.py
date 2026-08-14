import numpy as np
import pytest
import torch
from torch.nn import functional as F

from methods.structure_da.full_model import TSStructureModel
from methods.structure_da.seed_warmup_diagnostic import (
    BOOTSTRAP_ARMS,
    LABEL_FREE_BOOTSTRAP_ARMS,
    class_balanced_seed_cross_entropy,
    confidence_geometry_veto_bootstrap,
    configure_13b_semantic_student,
    dapl_bootstrap,
    diagonal_gaussian_conformity_percentile,
    ipl_bootstrap,
    normalize_seed_manifest_rows,
    seed_manifest_fingerprint,
    tfda_nn_bootstrap,
    timematch_confidence_bootstrap,
    validate_checkpoint_epochs,
)


def _small_model():
    return TSStructureModel(
        num_classes=3, input_dim=10, mlp1=[10,8,8], mlp2=[16,8], n_head=2, d_k=2, d_model=8,
        ltae_mlp=(8,6), classifier_hidden=(4,), trend_num_basis=4, structure_num_basis=4,
        canonical_grid_size=8, roughness_grid_size=64,
    )


def test_13b_arm_matrix_is_frozen_and_excludes_transpl():
    assert BOOTSTRAP_ARMS == (
        "CTRL_SOURCE","PL_TIMEMATCH_CONF","PL_DAPL_BOOT","PL_IPL_BOOT","PL_TFDA_NN","PL_CONF_GEOM","CTRL_ORACLE"
    )
    assert LABEL_FREE_BOOTSTRAP_ARMS == BOOTSTRAP_ARMS[1:6]
    assert all("TRANSPL" not in arm for arm in BOOTSTRAP_ARMS)


def test_13b_parameter_policy_trains_pse_raw_encoder_classifier_only():
    model=_small_model(); policy=configure_13b_semantic_student(model); trainable=set(policy.trainable_parameter_names)
    assert any(n.startswith("backbone.pixel_set_encoder.") for n in trainable)
    assert any(n.startswith("temporal_module.raw_encoder.") for n in trainable)
    assert any(n.startswith("classifier.") for n in trainable)
    assert not any(n.startswith("backbone.decomposition.") for n in trainable)
    assert not any(n.startswith("temporal_module.trend_geometry.") for n in trainable)
    assert not any(n.startswith("temporal_module.structure_geometry.") for n in trainable)


def test_class_balanced_seed_ce_is_mean_of_present_class_means():
    logits=torch.tensor([[3.,0.,0.],[2.,0.,0.],[0.,2.,0.],[0.,1.5,0.],[0.,1.,0.]])
    labels=torch.tensor([0,0,1,1,1],dtype=torch.long); per=F.cross_entropy(logits,labels,reduction="none")
    torch.testing.assert_close(class_balanced_seed_cross_entropy(logits,labels),0.5*(per[:2].mean()+per[2:].mean()))


def test_class_balanced_seed_ce_does_not_weight_by_seed_count():
    base_logits=torch.tensor([[2.,0.],[0.,2.]]); base_labels=torch.tensor([0,1],dtype=torch.long)
    expanded_logits=torch.tensor([[2.,0.],[0.,2.],[0.,2.],[0.,2.]]); expanded_labels=torch.tensor([0,1,1,1],dtype=torch.long)
    torch.testing.assert_close(class_balanced_seed_cross_entropy(base_logits,base_labels),class_balanced_seed_cross_entropy(expanded_logits,expanded_labels))


def test_seed_manifest_is_label_free_and_raw_candidate_match_is_optional_only_when_explicit():
    rows=[{"sample_id":"10","pseudo_label":"2","selection_evidence":"fixed bootstrap"}]
    records=normalize_seed_manifest_rows(rows,valid_sample_ids=[10,11],initial_candidates={10:2,11:1},num_classes=3)
    assert [(r.sample_id,r.pseudo_label) for r in records]==[(10,2)]
    with pytest.raises(ValueError,match="relabels Stage-1 candidate"):
        normalize_seed_manifest_rows([{"sample_id":10,"pseudo_label":1}],valid_sample_ids=[10],initial_candidates={10:2},num_classes=3)
    relabeled=normalize_seed_manifest_rows([{"sample_id":10,"pseudo_label":1}],valid_sample_ids=[10],initial_candidates={10:2},num_classes=3,require_initial_candidate_match=False)
    assert relabeled[0].pseudo_label==1
    with pytest.raises(ValueError,match="forbidden oracle fields"):
        normalize_seed_manifest_rows([{"sample_id":10,"pseudo_label":2,"true_label":2}],valid_sample_ids=[10],initial_candidates={10:2},num_classes=3)


def test_seed_manifest_fingerprint_is_deterministic():
    a=normalize_seed_manifest_rows([{"sample_id":2,"pseudo_label":1},{"sample_id":1,"pseudo_label":0}],valid_sample_ids=[1,2],initial_candidates={1:0,2:1},num_classes=3)
    assert seed_manifest_fingerprint(a)==seed_manifest_fingerprint(tuple(reversed(a)))


def test_timematch_confidence_uses_fixed_raw_top1_threshold():
    ids=np.array([10,11,12]); post=np.array([[.91,.09],[.9,.1],[.2,.8]])
    rec=timematch_confidence_bootstrap(ids,post,threshold=.9)
    assert [(r.sample_id,r.pseudo_label) for r in rec]==[(10,0)]  # strict > like TimeMatch


def test_dapl_bootstrap_intersects_confidence_and_source_conformity():
    ids=np.array([1,2,3]); post=np.array([[.95,.05],[.96,.04],[.2,.8]]); pct=np.array([.1,.99,.2])
    rec=dapl_bootstrap(ids,post,pct,confidence_threshold=.9,conformity_threshold=.95)
    assert [(r.sample_id,r.pseudo_label) for r in rec]==[(1,0)]


def test_diagonal_gaussian_conformity_uses_source_only_and_returns_percentile():
    sids=np.arange(12); labels=np.array([0]*6+[1]*6)
    sx=np.vstack([np.c_[np.linspace(0,1,6),np.linspace(0,1,6)],np.c_[np.linspace(5,6,6),np.linspace(5,6,6)]])
    tx=np.array([[.5,.5],[5.5,5.5],[10.,10.]])
    cand=np.array([0,1,1])
    score,pct=diagonal_gaussian_conformity_percentile(sids,sx,labels,tx,cand,num_classes=2,folds=3)
    assert score.shape==pct.shape==(3,); assert np.all((pct>=0)&(pct<=1)); assert pct[2]>=pct[1]


def test_ipl_bootstrap_requires_classifier_prototype_agreement_and_knn_support():
    ids=np.array([0,1,2,3]); post=np.array([[.9,.1],[.8,.2],[.1,.9],[.2,.8]])
    proto=np.array([0,1,1,1])
    # self is assumed already excluded upstream; rows are just neighbor indices.
    knn=np.array([[1,2],[0,3],[3,1],[2,1]])
    rec=ipl_bootstrap(ids,post,proto,knn,k=2,support_threshold=.5)
    assert [(r.sample_id,r.pseudo_label) for r in rec]==[(0,0),(2,1),(3,1)]


def test_tfda_nn_can_relabel_raw_candidate_from_neighbor_mean_posterior():
    ids=np.array([0,1,2]); post=np.array([[.9,.1],[.1,.9],[.2,.8]])
    knn=np.array([[1,2],[0,2],[0,1]])
    rec=tfda_nn_bootstrap(ids,post,knn,k=2)
    by={r.sample_id:r.pseudo_label for r in rec}
    assert by[0]==1  # raw top1 was class 0; neighborhood changes bootstrap label.
    assert len(rec)==3


def test_geometry_veto_never_relabels_and_only_removes_strong_conflict():
    ids=np.array([0,1,2]); post=np.array([[.95,.05],[.96,.04],[.2,.8]])
    t=np.array([.2,.99,.2]); s=np.array([.2,.2,.2])
    rec=confidence_geometry_veto_bootstrap(ids,post,t,s,confidence_threshold=.9,conflict_percentile=.95)
    assert [(r.sample_id,r.pseudo_label) for r in rec]==[(0,0)]


def test_13b_checkpoint_schedule_requires_initial_early_middle_end():
    assert validate_checkpoint_epochs(5,(0,1,3,5))==(0,1,3,5)
    with pytest.raises(ValueError): validate_checkpoint_epochs(5,(0,5))
    with pytest.raises(ValueError): validate_checkpoint_epochs(5,(0,1,3,4))
