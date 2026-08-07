from __future__ import annotations

from methods.structure_da import TSStructureModel


def _model() -> TSStructureModel:
    return TSStructureModel(
        num_classes=3,
        input_dim=2,
        mlp1=(2, 4, 4),
        mlp2=(8, 4),
        trend_num_basis=4,
        structure_num_basis=4,
        canonical_grid_size=5,
        roughness_grid_size=64,
        n_head=1,
        d_k=2,
        d_model=8,
        ltae_mlp=(8, 4),
        dropout=0.0,
        classifier_hidden=(4,),
        max_initial_frequency=4.0,
    )


def test_stage2_policy_partitions_every_parameter_by_module_ownership() -> None:
    from methods.structure_da.stage2_parameter_policy import (
        configure_stage2_parameter_policy,
    )

    model = _model()
    policy = configure_stage2_parameter_policy(model)
    all_names = {name for name, _ in model.named_parameters()}
    trainable = set(policy.trainable_parameter_names)
    frozen = set(policy.frozen_parameter_names)

    assert trainable.isdisjoint(frozen)
    assert trainable | frozen == all_names
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == trainable


def test_stage2_policy_freezes_geometry_and_trains_only_raw_representation_and_classifier() -> None:
    from methods.structure_da.stage2_parameter_policy import (
        configure_stage2_parameter_policy,
    )

    model = _model()
    policy = configure_stage2_parameter_policy(model)
    trainable_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name in policy.trainable_parameter_names
    }

    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in model.temporal_module.trend_geometry.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.temporal_module.structure_geometry.parameters()
    )

    shared = model.temporal_module.raw_encoder.shared_ltae
    expected_modules = (
        shared.shared_time_encoder,
        shared.shared_input_projection,
        shared.attention_heads,
        shared.shared_projection,
        shared.trend_input_norm,
        shared.structure_input_norm,
        shared.trend_output_norm,
        shared.structure_output_norm,
        model.classifier,
    )
    expected_ids = {
        id(parameter)
        for module in expected_modules
        for parameter in module.parameters()
    }
    assert trainable_ids == expected_ids
    assert all(parameter.requires_grad for module in expected_modules for parameter in module.parameters())
