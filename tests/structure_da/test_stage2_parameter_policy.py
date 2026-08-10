from __future__ import annotations

from methods.structure_da import TSStructureModel, configure_stage2_parameter_policy


def _model() -> TSStructureModel:
    return TSStructureModel(
        num_classes=3,input_dim=2,mlp1=(2,4,4),mlp2=(8,4),
        trend_num_basis=4,structure_num_basis=4,canonical_grid_size=5,
        roughness_grid_size=64,n_head=1,d_k=2,d_model=8,ltae_mlp=(8,4),
        dropout=0.0,classifier_hidden=(4,),max_initial_frequency=4.0,
    )


def test_stage2_policy_partitions_every_parameter() -> None:
    model=_model(); policy=configure_stage2_parameter_policy(model)
    all_names={n for n,_ in model.named_parameters()}; train=set(policy.trainable_parameter_names); frozen=set(policy.frozen_parameter_names)
    assert train.isdisjoint(frozen) and train|frozen==all_names


def test_stage2_policy_freezes_pse_decomposition_and_geometry() -> None:
    model=_model(); policy=configure_stage2_parameter_policy(model)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(not p.requires_grad for p in model.temporal_module.trend_geometry.parameters())
    assert all(not p.requires_grad for p in model.temporal_module.structure_geometry.parameters())
    raw=model.temporal_module.raw_encoder
    expected=(raw.time_encoder,raw.input_projection,raw.input_norm,raw.attention_heads,raw.projection,raw.output_norm,model.classifier)
    expected_ids={id(p) for m in expected for p in m.parameters()}
    trainable_ids={id(p) for n,p in model.named_parameters() if n in policy.trainable_parameter_names}
    assert trainable_ids==expected_ids
