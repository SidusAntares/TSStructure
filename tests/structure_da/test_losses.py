"""Numerical and real-module gradient tests for Structure DA losses."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from methods.structure_da import (
    ComponentStructureClassifier,
    JointStructuralSpaceBuilder,
    LossWeights,
    StructuralAdversarialAdapter,
    StructuralAdversarialOutput,
    classification_loss,
    component_diversity_loss,
    compose_total_loss,
    quality_classification_loss,
    quality_domain_loss,
    structural_adversarial_loss,
)


def _scores(domain_logits, class_logits):
    return SimpleNamespace(
        scores=SimpleNamespace(
            domain_logits=domain_logits,
            class_logits=class_logits,
        )
    )


def _fake_output(batch_size=4, num_classes=3, valid=None):
    if valid is None:
        valid = torch.ones(batch_size, dtype=torch.bool)
    torch.manual_seed(batch_size * 10 + num_classes)
    structural = [
        _scores(
            torch.randn(batch_size, 2, requires_grad=True),
            torch.randn(batch_size, num_classes, requires_grad=True),
        )
        for _ in range(3)
    ]
    components = []
    for _ in range(3):
        quality = _scores(
            torch.randn(batch_size, 2, requires_grad=True),
            torch.randn(batch_size, num_classes, requires_grad=True),
        )
        quality.diversity = torch.rand(batch_size, requires_grad=True)
        components.append(quality)
    return SimpleNamespace(
        logits=torch.randn(batch_size, num_classes, requires_grad=True),
        trend_temporal=SimpleNamespace(valid=valid.clone()),
        dynamics_temporal=SimpleNamespace(valid=valid.clone()),
        dynamics_channel=SimpleNamespace(valid=valid.clone()),
        structural_quality=SimpleNamespace(
            trend_temporal=structural[0],
            dynamics_temporal=structural[1],
            dynamics_channel=structural[2],
        ),
        component_quality=SimpleNamespace(
            trend=components[0],
            dynamics=components[1],
            residual=components[2],
        ),
    )


def _structural_pairs(output):
    return (
        (output.structural_quality.trend_temporal, output.trend_temporal.valid),
        (
            output.structural_quality.dynamics_temporal,
            output.dynamics_temporal.valid,
        ),
        (
            output.structural_quality.dynamics_channel,
            output.dynamics_channel.valid,
        ),
    )


def _component_qualities(output):
    return (
        output.component_quality.trend,
        output.component_quality.dynamics,
        output.component_quality.residual,
    )


def test_classification_loss_matches_mean_cross_entropy():
    output = _fake_output()
    labels = torch.tensor([0, 2, 1, 0])

    actual = classification_loss(output, labels)

    torch.testing.assert_close(actual, F.cross_entropy(output.logits, labels))


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([0.0, 1.0, 2.0, 0.0]),
        torch.tensor([True, False, True, False]),
        torch.tensor([[0], [1], [2], [0]]),
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([-1, 1, 2, 0]),
    ],
)
def test_classification_loss_rejects_invalid_source_labels(labels):
    with pytest.raises(ValueError, match="source_labels"):
        classification_loss(_fake_output(), labels)


def test_quality_domain_loss_matches_six_branch_domain_targets_and_validity():
    source_valid = torch.tensor([True, False, True, False])
    target_valid = torch.tensor([False, True, True, False, True])
    source = _fake_output(4, valid=source_valid)
    target = _fake_output(5, valid=target_valid)
    expected = source.logits.sum() * 0.0
    for (source_quality, source_mask), (target_quality, target_mask) in zip(
        _structural_pairs(source), _structural_pairs(target)
    ):
        expected = expected + F.cross_entropy(
            source_quality.scores.domain_logits[source_mask],
            torch.ones(source_mask.sum(), dtype=torch.long),
        )
        expected = expected + F.cross_entropy(
            target_quality.scores.domain_logits[target_mask],
            torch.zeros(target_mask.sum(), dtype=torch.long),
        )
    for source_quality, target_quality in zip(
        _component_qualities(source), _component_qualities(target)
    ):
        expected = expected + F.cross_entropy(
            source_quality.scores.domain_logits,
            torch.ones(4, dtype=torch.long),
        )
        expected = expected + F.cross_entropy(
            target_quality.scores.domain_logits,
            torch.zeros(5, dtype=torch.long),
        )

    actual = quality_domain_loss(source, target)

    torch.testing.assert_close(actual, expected)


def test_quality_domain_loss_does_not_read_final_classifier_logits():
    source = _fake_output(3)
    target = _fake_output(5)
    expected = quality_domain_loss(source, target)
    source.logits = None
    target.logits = None

    torch.testing.assert_close(quality_domain_loss(source, target), expected)


@pytest.mark.parametrize("empty_side", ["source", "target"])
def test_quality_domain_loss_skips_structural_branch_missing_either_domain(
    empty_side,
):
    source = _fake_output(3)
    target = _fake_output(5)
    branch_source = source.structural_quality.trend_temporal
    branch_target = target.structural_quality.trend_temporal
    if empty_side == "source":
        source.trend_temporal.valid.zero_()
    else:
        target.trend_temporal.valid.zero_()

    loss = quality_domain_loss(source, target)
    loss.backward()

    torch.testing.assert_close(
        branch_source.scores.domain_logits.grad,
        torch.zeros_like(branch_source.scores.domain_logits),
    )
    torch.testing.assert_close(
        branch_target.scores.domain_logits.grad,
        torch.zeros_like(branch_target.scores.domain_logits),
    )


def test_component_quality_domain_loss_ignores_structural_validity_masks():
    source = _fake_output(3, valid=torch.zeros(3, dtype=torch.bool))
    target = _fake_output(5, valid=torch.zeros(5, dtype=torch.bool))

    quality_domain_loss(source, target).backward()

    for quality in _component_qualities(source) + _component_qualities(target):
        assert quality.scores.domain_logits.grad is not None
        assert torch.count_nonzero(quality.scores.domain_logits.grad) > 0


def test_quality_classification_loss_is_source_only_valid_filtered_six_head_ce():
    valid = torch.tensor([True, False, True, True])
    source = _fake_output(4, valid=valid)
    labels = torch.tensor([0, 2, 1, 0])
    expected = source.logits.sum() * 0.0
    for quality, mask in _structural_pairs(source):
        expected = expected + F.cross_entropy(
            quality.scores.class_logits[mask], labels[mask]
        )
    for quality in _component_qualities(source):
        expected = expected + F.cross_entropy(
            quality.scores.class_logits, labels
        )

    actual = quality_classification_loss(source, labels)

    torch.testing.assert_close(actual, expected)
    source.logits = None
    torch.testing.assert_close(
        quality_classification_loss(source, labels), actual
    )


def test_quality_classification_empty_structural_branch_is_differentiable_zero():
    source = _fake_output(4)
    source.trend_temporal.valid.zero_()
    branch_logits = source.structural_quality.trend_temporal.scores.class_logits

    quality_classification_loss(source, torch.tensor([0, 1, 2, 0])).backward()

    torch.testing.assert_close(branch_logits.grad, torch.zeros_like(branch_logits))


def test_component_diversity_loss_matches_exact_present_class_cv_formula():
    source = _fake_output(5, num_classes=4)
    labels = torch.tensor([0, 0, 2, 2, 2])
    scores = (
        torch.tensor([0.1, 0.3, 0.4, 0.6, 0.8], requires_grad=True),
        torch.tensor([0.7, 0.5, 0.2, 0.4, 0.6], requires_grad=True),
        torch.tensor([0.2, 0.4, 0.9, 0.7, 0.5], requires_grad=True),
    )
    for quality, values in zip(_component_qualities(source), scores):
        quality.diversity = values
    expected = source.logits.sum() * 0.0
    for values in scores:
        class_means = torch.stack((values[:2].mean(), values[2:].mean()))
        expected = expected - class_means.std(unbiased=False) / (
            class_means.mean() + 1e-8
        )

    actual = component_diversity_loss(source, labels)

    torch.testing.assert_close(actual, expected)


def test_component_diversity_single_class_returns_graph_connected_zero():
    source = _fake_output(4)
    labels = torch.zeros(4, dtype=torch.long)

    loss = component_diversity_loss(source, labels)
    loss.backward()

    assert loss.item() == 0.0
    for quality in _component_qualities(source):
        assert quality.diversity.grad is not None
        torch.testing.assert_close(
            quality.diversity.grad, torch.zeros_like(quality.diversity)
        )


def test_component_diversity_does_not_use_structural_quality_values():
    source = _fake_output(4)
    labels = torch.tensor([0, 0, 1, 1])
    original = component_diversity_loss(source, labels)
    source.structural_quality.trend_temporal.scores.class_logits = (
        source.structural_quality.trend_temporal.scores.class_logits + 1000
    )

    torch.testing.assert_close(component_diversity_loss(source, labels), original)


@pytest.mark.parametrize("eps", [0, -1, float("nan"), float("inf"), True])
def test_component_diversity_rejects_invalid_eps(eps):
    with pytest.raises(ValueError, match="eps"):
        component_diversity_loss(
            _fake_output(), torch.tensor([0, 0, 1, 1]), eps=eps
        )


def test_structural_adversarial_loss_matches_positive_bce_for_unequal_batches():
    source_logits = torch.tensor([0.2, -0.4, 1.0], requires_grad=True)
    target_logits = torch.tensor([-0.1, 0.6, -1.2, 0.3, 0.9], requires_grad=True)
    output = StructuralAdversarialOutput(
        source_logits=source_logits,
        target_logits=target_logits,
        source_joint=torch.randn(3, 2),
        target_joint=torch.randn(5, 2),
    )
    expected = F.binary_cross_entropy_with_logits(
        source_logits, torch.ones_like(source_logits)
    ) + F.binary_cross_entropy_with_logits(
        target_logits, torch.zeros_like(target_logits)
    )

    actual = structural_adversarial_loss(output)

    torch.testing.assert_close(actual, expected)
    assert actual.item() > 0


def test_loss_weights_defaults_and_total_match_exact_weighted_formula():
    values = tuple(torch.tensor(value) for value in (2.0, 3.0, 4.0, -0.5, 6.0))
    defaults = LossWeights()
    assert defaults == LossWeights(qdom=1.0, qcls=1.0, diversity=1.0, sda=1.0)
    weights = LossWeights(qdom=0.2, qcls=0.3, diversity=0.4, sda=0.5)

    losses = compose_total_loss(*values, weights)

    expected = values[0] + 0.2 * values[1] + 0.3 * values[2]
    expected = expected + 0.4 * values[3] + 0.5 * values[4]
    torch.testing.assert_close(losses.total, expected)
    assert losses.classification is values[0]
    assert losses.quality_domain is values[1]
    assert losses.quality_classification is values[2]
    assert losses.diversity is values[3]
    assert losses.structural_adversarial is values[4]


def test_zero_auxiliary_weights_make_total_exactly_classification():
    classification = torch.tensor(2.0)
    auxiliaries = tuple(torch.tensor(value) for value in (3.0, 4.0, -5.0, 6.0))

    losses = compose_total_loss(
        classification,
        *auxiliaries,
        LossWeights(qdom=0, qcls=0, diversity=0, sda=0),
    )

    torch.testing.assert_close(losses.total, classification)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_loss_weights_reject_invalid_values(value):
    with pytest.raises(ValueError):
        LossWeights(qdom=value)


@pytest.mark.parametrize(
    "replacement",
    [torch.ones(1), torch.tensor(1), torch.tensor(float("nan")), 1.0],
)
def test_compose_total_loss_rejects_invalid_scalar_losses(replacement):
    values = [torch.tensor(1.0) for _ in range(5)]
    values[2] = replacement
    with pytest.raises(ValueError):
        compose_total_loss(*values, LossWeights())


def test_compose_total_loss_rejects_mismatched_dtype():
    values = [torch.tensor(1.0) for _ in range(5)]
    values[-1] = values[-1].double()
    with pytest.raises(ValueError, match="dtype"):
        compose_total_loss(*values, LossWeights())


def _make_model():
    return ComponentStructureClassifier(
        feature_dim=4,
        num_classes=3,
        n_head=2,
        d_k=2,
        d_model=8,
        ltae_mlp=(8, 6),
        dropout=0.0,
        max_position=64,
        max_temporal_shift=4,
        classifier_hidden=(5,),
        quality_hidden_cap=5,
        quality_eta=0.2,
    )


def _real_inputs(batch_size):
    torch.manual_seed(100 + batch_size)
    return (
        torch.randn(batch_size, 4, 4, requires_grad=True),
        torch.tensor([0, 2, 7, 12]),
    )


def _quality_modules(model):
    return (
        model.trend_temporal_quality,
        model.dynamics_temporal_quality,
        model.dynamics_channel_quality,
        model.trend_component_quality,
        model.dynamics_component_quality,
        model.residual_component_quality,
    )


def _assert_all_parameter_grads(module):
    parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


def _assert_no_parameter_grads(module):
    assert all(parameter.grad is None for parameter in module.parameters())


def _build_joint(builder, output):
    return builder(
        output.trend_temporal.statistic,
        output.dynamics_temporal.statistic,
        output.dynamics_channel.statistic,
        output.effective_gates.beta_trend_temporal,
        output.effective_gates.beta_dynamics_temporal,
        output.effective_gates.beta_dynamics_channel,
    ).joint


def test_real_classification_loss_gradient_route():
    model = _make_model()
    source_h, positions = _real_inputs(4)
    output = model(source_h, positions)

    classification_loss(output, torch.tensor([0, 1, 2, 0])).backward()

    assert source_h.grad is not None and torch.isfinite(source_h.grad).all()
    _assert_all_parameter_grads(model.decomposition)
    _assert_all_parameter_grads(model.shared_ltae)
    _assert_all_parameter_grads(model.classifier)
    for module in _quality_modules(model):
        _assert_no_parameter_grads(module)


def test_real_quality_domain_loss_gradient_route():
    model = _make_model()
    source_h, positions = _real_inputs(3)
    target_h, _ = _real_inputs(5)
    source = model(source_h, positions)
    target = model(target_h, positions)

    quality_domain_loss(source, target).backward()

    assert source_h.grad is None and target_h.grad is None
    _assert_no_parameter_grads(model.decomposition)
    _assert_no_parameter_grads(model.shared_ltae)
    _assert_no_parameter_grads(model.classifier)
    for module in _quality_modules(model):
        _assert_all_parameter_grads(module.transferability)
        _assert_no_parameter_grads(module.discriminability)
        if hasattr(module, "diversity"):
            _assert_no_parameter_grads(module.diversity)


def test_real_quality_classification_loss_gradient_route():
    model = _make_model()
    source_h, positions = _real_inputs(4)
    source = model(source_h, positions)

    quality_classification_loss(source, torch.tensor([0, 1, 2, 0])).backward()

    assert source_h.grad is None
    _assert_no_parameter_grads(model.decomposition)
    _assert_no_parameter_grads(model.shared_ltae)
    _assert_no_parameter_grads(model.classifier)
    for module in _quality_modules(model):
        _assert_no_parameter_grads(module.transferability)
        _assert_all_parameter_grads(module.discriminability)
        if hasattr(module, "diversity"):
            _assert_no_parameter_grads(module.diversity)


def test_real_component_diversity_loss_gradient_route():
    model = _make_model()
    source_h, positions = _real_inputs(4)
    source = model(source_h, positions)

    component_diversity_loss(source, torch.tensor([0, 0, 1, 1])).backward()

    assert source_h.grad is None
    _assert_no_parameter_grads(model.decomposition)
    _assert_no_parameter_grads(model.shared_ltae)
    _assert_no_parameter_grads(model.classifier)
    for module in _quality_modules(model):
        _assert_no_parameter_grads(module.transferability)
        _assert_no_parameter_grads(module.discriminability)
    for module in (
        model.trend_component_quality.diversity,
        model.dynamics_component_quality.diversity,
        model.residual_component_quality.diversity,
    ):
        _assert_all_parameter_grads(module)


@pytest.mark.parametrize("coefficient", [1.0, 0.0])
def test_real_structural_adversarial_loss_gradient_route(coefficient):
    model = _make_model()
    builder = JointStructuralSpaceBuilder(4)
    adapter = StructuralAdversarialAdapter(builder.joint_dim, hidden_dim=8)
    source_h, positions = _real_inputs(3)
    target_h, _ = _real_inputs(5)
    source = model(source_h, positions)
    target = model(target_h, positions)
    adaptation = adapter(
        _build_joint(builder, source),
        _build_joint(builder, target),
        grl_coefficient=coefficient,
    )

    structural_adversarial_loss(adaptation).backward()

    _assert_all_parameter_grads(adapter.discriminator)
    _assert_no_parameter_grads(model.shared_ltae)
    _assert_no_parameter_grads(model.classifier)
    for module in _quality_modules(model):
        _assert_no_parameter_grads(module)
    if coefficient == 1.0:
        assert source_h.grad is not None and torch.isfinite(source_h.grad).all()
        assert target_h.grad is not None and torch.isfinite(target_h.grad).all()
        assert torch.count_nonzero(source_h.grad) > 0
        assert torch.count_nonzero(target_h.grad) > 0
        _assert_all_parameter_grads(model.decomposition)
        assert any(
            torch.count_nonzero(parameter.grad) > 0
            for parameter in model.decomposition.parameters()
        )
    else:
        if source_h.grad is not None:
            assert torch.count_nonzero(source_h.grad) == 0
        if target_h.grad is not None:
            assert torch.count_nonzero(target_h.grad) == 0
        assert all(
            parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
            for parameter in model.decomposition.parameters()
        )


def test_combined_total_loss_reaches_every_intended_parameter_group():
    model = _make_model()
    builder = JointStructuralSpaceBuilder(4)
    adapter = StructuralAdversarialAdapter(builder.joint_dim, hidden_dim=8)
    source_h, positions = _real_inputs(4)
    target_h, _ = _real_inputs(5)
    labels = torch.tensor([0, 0, 1, 2])
    source = model(source_h, positions)
    target = model(target_h, positions)
    adaptation = adapter(
        _build_joint(builder, source),
        _build_joint(builder, target),
        grl_coefficient=0.7,
    )
    losses = compose_total_loss(
        classification_loss(source, labels),
        quality_domain_loss(source, target),
        quality_classification_loss(source, labels),
        component_diversity_loss(source, labels),
        structural_adversarial_loss(adaptation),
        LossWeights(qdom=0.5, qcls=0.5, diversity=0.5, sda=0.5),
    )

    losses.total.backward()

    assert source_h.grad is not None and torch.isfinite(source_h.grad).all()
    assert target_h.grad is not None and torch.isfinite(target_h.grad).all()
    assert torch.count_nonzero(source_h.grad) > 0
    assert torch.count_nonzero(target_h.grad) > 0
    _assert_all_parameter_grads(model.decomposition)
    _assert_all_parameter_grads(model.shared_ltae)
    _assert_all_parameter_grads(model.classifier)
    for module in _quality_modules(model):
        _assert_all_parameter_grads(module)
    _assert_all_parameter_grads(adapter.discriminator)
