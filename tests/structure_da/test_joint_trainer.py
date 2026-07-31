from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from methods.structure_da.joint_trainer import (
    JointStructureDATrainingConfig,
    create_joint_structure_da_train_loaders,
    joint_structure_da_train_step,
    train_joint_structure_da,
    validation_structure_contributions,
)
from methods.structure_da import joint_trainer as joint_trainer_module
from methods.structure_da.quality_fusion import (
    HierarchicalQualityObjective,
    concatenate_hierarchical_quality_outputs,
)
from tests.structure_da.test_full_model import _model


def _sample(batch: int, length: int, *, labels: bool = True):
    torch.manual_seed(1000 + length)
    sample = {
        "pixels": torch.randn(batch, length, 2, 4),
        "valid_pixels": torch.ones(batch, length, 4, dtype=torch.bool),
        "positions": torch.linspace(0, 300, length).round().long().expand(batch, -1),
    }
    if labels:
        sample["label"] = torch.arange(batch) % 3
    return sample


def _config(**kwargs):
    values = dict(
        epochs=1,
        steps_per_epoch=2,
        lr=1e-3,
        weight_decay=0.0,
        task_weight=1.2,
        geometry_weight=0.7,
        alignment_weight=0.4,
        structural_classification_weight=0.8,
        structural_domain_weight=0.9,
        component_classification_weight=1.1,
        component_domain_weight=1.3,
        log_step=1,
        progress_bar="off",
        classes=("a", "b", "c"),
    )
    values.update(kwargs)
    return JointStructureDATrainingConfig(**values)


def _objective(config):
    return HierarchicalQualityObjective(
        structural_classification_weight=config.structural_classification_weight,
        structural_domain_weight=config.structural_domain_weight,
        component_classification_weight=config.component_classification_weight,
        component_domain_weight=config.component_domain_weight,
    )


def _optimizers(model, config):
    return (
        torch.optim.Adam(model.task_parameters(), lr=config.lr),
        torch.optim.Adam(model.geometry_parameters(), lr=config.lr),
    )


def _counts(model):
    temporal = model.temporal_operator.extractor.registration
    channel = model.channel_operator.extractor
    return (
        int(temporal.srvf_extractor.functional_lift.standardizer.num_updates.item()),
        int(temporal.srvf_extractor.support_scale.num_updates.item()),
        int(temporal.source_template.num_updates.item()),
        int(channel.attribute_standardizer.num_updates.item()),
        int(channel.energy_scale.num_updates.item()),
    )


def _isolated_losses(model, config):
    model.alignment.grl.iteration.fill_(3)
    source = _sample(2, 5)
    target = _sample(2, 7, labels=False)
    source_tensors = (
        source["pixels"], source["valid_pixels"], source["positions"]
    )
    target_tensors = (
        target["pixels"], target["valid_pixels"], target["positions"]
    )
    source_backbone = model.forward_backbone(*source_tensors)
    target_backbone = model.forward_backbone(*target_tensors)
    model.update_source_state_from_backbone(
        model.detach_backbone_for_state(source_backbone), source_tensors[2]
    )
    source_output = model.forward_from_backbone(source_backbone, source_tensors[2])
    target_output = model.forward_from_backbone(target_backbone, target_tensors[2])
    merged_quality = concatenate_hierarchical_quality_outputs(
        source_output.representation.quality,
        target_output.representation.quality,
    )
    source_labels = source["label"]
    domain_labels = torch.tensor([1, 1, 0, 0])
    class_labels = torch.cat([source_labels, torch.zeros(2, dtype=torch.long)])
    return {
        "geometry": model.forward_source_geometry(
            source_output, source_tensors[2]
        ).total_loss,
        "task": F.cross_entropy(source_output.representation.logits, source_labels),
        "quality": _objective(config)(
            merged_quality,
            class_labels,
            domain_labels,
            domain_labels == 1,
        ).total_loss,
        "alignment": model.align(source_output, target_output).loss,
    }


def _has_gradient(parameters) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in parameters
    )


@pytest.mark.parametrize(
    "loss_name,expected",
    [
        ("geometry", (True, False, False, False, False)),
        ("task", (False, True, False, True, False)),
        ("quality", (False, False, True, False, False)),
        ("alignment", (False, True, False, False, True)),
    ],
)
def test_each_loss_has_the_intended_gradient_route(loss_name, expected) -> None:
    model = _model()
    config = _config()
    losses = _isolated_losses(model, config)

    model.zero_grad(set_to_none=True)
    losses[loss_name].backward()

    actual = (
        _has_gradient(model.geometry_parameters()),
        _has_gradient(
            parameter
            for module in (
                model.backbone,
                model.temporal_operator.extractor,
                model.channel_operator,
                model.representation.component_ltae,
            )
            for parameter in module.parameters()
            if id(parameter)
            not in {id(warp) for warp in model.geometry_parameters()}
        ),
        _has_gradient(model.representation.quality_fusion.parameters()),
        _has_gradient(model.representation.classifier.parameters()),
        _has_gradient(model.alignment.discriminator.parameters()),
    )
    assert actual == expected


def test_training_config_rejects_invalid_values_and_has_no_legacy_fields() -> None:
    config = _config()
    assert config.quality_domain_score_warmup_epochs == 5
    assert config.amp is False
    assert config.amp_dtype == "float16"
    for field in (
        "task_weight", "geometry_weight", "alignment_weight",
        "structural_classification_weight", "structural_domain_weight",
        "component_classification_weight", "component_domain_weight",
    ):
        with pytest.raises(ValueError):
            _config(**{field: -1.0})
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError, match="quality_domain_score_warmup_epochs"):
            _config(quality_domain_score_warmup_epochs=invalid)
    with pytest.raises(ValueError, match="amp"):
        _config(amp="true")
    with pytest.raises(ValueError, match="amp_dtype"):
        _config(amp_dtype="float32")


def test_default_joint_epoch_uses_source_loader_length() -> None:
    config = _config(steps_per_epoch=None)

    assert joint_trainer_module._resolve_steps(config, [1, 2, 3], [1]) == 3
    assert joint_trainer_module._resolve_steps(config, [1, 2], [1, 2, 3, 4]) == 2


def test_grl_warmup_resolution_uses_fraction_override_and_conflict_rules() -> None:
    resolve = joint_trainer_module.resolve_grl_warmup_max_iters

    assert resolve(epochs=100, steps_per_epoch=37) == 740
    assert resolve(epochs=1, steps_per_epoch=1, fraction=0.01) == 1
    assert resolve(
        epochs=100,
        steps_per_epoch=37,
        fraction=None,
        override=123,
    ) == 123
    with pytest.raises(ValueError, match="both"):
        resolve(
            epochs=100,
            steps_per_epoch=37,
            fraction=0.2,
            override=123,
        )


@pytest.mark.parametrize(
    "warmup_epochs,expected",
    [
        (0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        (1, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        (5, [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]),
    ],
)
def test_resolve_domain_score_weight(
    warmup_epochs: int, expected: list[float]
) -> None:
    actual = [
        joint_trainer_module.resolve_domain_score_weight(epoch, warmup_epochs)
        for epoch in range(6)
    ]

    assert actual == expected


def test_joint_step_explicitly_forwards_domain_score_weight(monkeypatch) -> None:
    model = _model()
    config = _config()
    task_optimizer, geometry_optimizer = _optimizers(model, config)
    received = []
    original = model.forward_from_backbone

    def recording_forward(*args, **kwargs):
        received.append(kwargs.get("domain_score_weight"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward_from_backbone", recording_forward)
    joint_structure_da_train_step(
        model,
        _sample(2, 5),
        _sample(2, 7, labels=False),
        task_optimizer,
        geometry_optimizer,
        _objective(config),
        config,
        torch.device("cpu"),
        domain_score_weight=0.25,
    )

    assert received == [0.25, 0.25]


def test_joint_step_supports_unequal_lengths_and_never_reads_target_label() -> None:
    model = _model()
    model.alignment.grl.iteration.fill_(2)
    config = _config()
    task_optimizer, geometry_optimizer = _optimizers(model, config)
    step_calls = 0
    original_step = task_optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    task_optimizer.step = counted_step
    result = joint_structure_da_train_step(
        model,
        _sample(2, 5),
        _sample(3, 7, labels=False),
        task_optimizer,
        geometry_optimizer,
        _objective(config),
        config,
        torch.device("cpu"),
    )

    expected_total = (
        config.task_weight * result.losses.task_loss
        + result.losses.quality_loss.total_loss
        + config.geometry_weight * result.losses.geometry_loss
        + config.alignment_weight * result.losses.alignment_loss
    )
    torch.testing.assert_close(result.losses.total_loss, expected_total)
    assert result.source_batch_size == 2
    assert result.target_batch_size == 3
    assert torch.equal(result.alignment.labels, torch.tensor([1, 1, 0, 0, 0]))
    assert _counts(model) == (1, 1, 1, 1, 1)
    assert model.alignment.grl.iteration.item() == 3
    assert step_calls == 1
    assert all(torch.isfinite(value) for value in (
        result.losses.total_loss,
        result.losses.task_loss,
        result.losses.geometry_loss,
        result.losses.alignment_loss,
    ))
    groups = (
        model.backbone.pixel_set_encoder,
        model.backbone.decomposition,
        model.temporal_operator,
        model.channel_operator,
        model.representation.component_ltae,
        model.representation.quality_fusion,
        model.representation.classifier,
    )
    for group in groups:
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in group.parameters()
        )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.temporal_operator.extractor.warp_parameters()
    )


def test_amp_requested_on_cpu_falls_back_to_finite_full_precision_step() -> None:
    model = _model()
    config = _config(amp=True, amp_dtype="float16")
    task_optimizer, geometry_optimizer = _optimizers(model, config)

    result = joint_structure_da_train_step(
        model,
        _sample(2, 5),
        _sample(2, 7, labels=False),
        task_optimizer,
        geometry_optimizer,
        _objective(config),
        config,
        torch.device("cpu"),
    )

    assert all(
        torch.isfinite(value).item()
        for value in (
            result.losses.total_loss,
            result.losses.task_loss,
            result.losses.quality_loss.total_loss,
            result.losses.geometry_loss,
            result.losses.alignment_loss,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_matches_full_precision_with_finite_gradients() -> None:
    def snapshot(amp: bool) -> dict[str, torch.Tensor]:
        torch.manual_seed(1729)
        model = _model().cuda().train()
        source = _sample(2, 5)
        source_tensors = tuple(
            source[name].cuda() for name in ("pixels", "valid_pixels", "positions")
        )
        labels = source["label"].cuda()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            backbone = model.forward_backbone(*source_tensors)
            model.update_source_state_from_backbone(
                model.detach_backbone_for_state(backbone), source_tensors[2]
            )
            output = model.forward_from_backbone(backbone, source_tensors[2])
            task_loss = F.cross_entropy(output.representation.logits, labels)
            alignment_loss = model.align(output, output).loss
        with torch.autocast("cuda", enabled=False):
            geometry_loss = model.forward_source_geometry(
                output, source_tensors[2]
            ).total_loss.float()
        geometry_loss.backward()
        (task_loss + alignment_loss).backward()
        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert trainable_gradients
        assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)
        quality = output.representation.quality
        reconstruction = (
            backbone.decomposition.trend
            + backbone.decomposition.dynamics
            + backbone.decomposition.residual
        )
        return {
            "logits": output.representation.logits.detach().float().cpu(),
            "task_loss": task_loss.detach().float().cpu(),
            "geometry_loss": geometry_loss.detach().float().cpu(),
            "alignment_loss": alignment_loss.detach().float().cpu(),
            "alpha": torch.stack(
                [quality.alpha_trend, quality.alpha_dynamics, quality.alpha_residual],
                dim=-1,
            ).detach().float().cpu(),
            "beta": torch.stack(
                [
                    quality.beta_trend_temporal,
                    quality.beta_dynamics_temporal,
                    quality.beta_trend_channel,
                    quality.beta_dynamics_channel,
                ],
                dim=-1,
            ).detach().float().cpu(),
            "reconstruction": reconstruction.detach().float().cpu(),
            "tokens": backbone.channel_tokens.detach().float().cpu(),
        }

    full_precision = snapshot(False)
    mixed_precision = snapshot(True)
    for name in (
        "logits", "task_loss", "geometry_loss", "alignment_loss", "alpha", "beta"
    ):
        assert torch.isfinite(mixed_precision[name]).all(), name
        torch.testing.assert_close(
            mixed_precision[name], full_precision[name], rtol=5e-2, atol=5e-3
        )
    for result in (full_precision, mixed_precision):
        torch.testing.assert_close(
            result["reconstruction"], result["tokens"], rtol=1e-5, atol=1e-6
        )


def test_joint_step_uses_separate_optimizers_and_schedulers() -> None:
    model = _model()
    config = _config()
    task_optimizer = torch.optim.Adam(model.task_parameters(), lr=config.lr)
    geometry_optimizer = torch.optim.Adam(
        model.geometry_parameters(), lr=config.lr
    )
    task_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        task_optimizer, T_max=2
    )
    geometry_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        geometry_optimizer, T_max=2
    )
    counts = {"task": 0, "geometry": 0}
    original_task_step = task_optimizer.step
    original_geometry_step = geometry_optimizer.step

    def task_step(*args, **kwargs):
        counts["task"] += 1
        return original_task_step(*args, **kwargs)

    def geometry_step(*args, **kwargs):
        counts["geometry"] += 1
        return original_geometry_step(*args, **kwargs)

    task_optimizer.step = task_step
    geometry_optimizer.step = geometry_step
    joint_structure_da_train_step(
        model,
        _sample(2, 5),
        _sample(2, 7, labels=False),
        task_optimizer,
        geometry_optimizer,
        _objective(config),
        config,
        torch.device("cpu"),
        task_scheduler=task_scheduler,
        geometry_scheduler=geometry_scheduler,
    )

    assert counts == {"task": 1, "geometry": 1}
    assert task_scheduler.last_epoch == 1
    assert geometry_scheduler.last_epoch == 1


def test_joint_step_runs_each_domain_backbone_once_and_updates_only_source_state(
    monkeypatch,
) -> None:
    model = _model()
    config = _config()
    task_optimizer, geometry_optimizer = _optimizers(model, config)
    backbone_calls = []
    state_inputs = []
    original_backbone = model.forward_backbone
    original_update = model.update_source_state_from_backbone

    def counted_backbone(*args, **kwargs):
        output = original_backbone(*args, **kwargs)
        backbone_calls.append(output)
        return output

    def counted_update(backbone, *args, **kwargs):
        state_inputs.append(backbone)
        return original_update(backbone, *args, **kwargs)

    monkeypatch.setattr(model, "forward_backbone", counted_backbone)
    monkeypatch.setattr(model, "update_source_state_from_backbone", counted_update)
    joint_structure_da_train_step(
        model,
        _sample(2, 5),
        _sample(3, 7, labels=False),
        task_optimizer,
        geometry_optimizer,
        _objective(config),
        config,
        torch.device("cpu"),
    )

    assert len(backbone_calls) == 2
    assert len(state_inputs) == 1
    assert not state_inputs[0].channel_tokens.requires_grad
    assert state_inputs[0].channel_tokens.shape[0] == 2


def test_joint_step_exposes_finite_source_target_diagnostics(monkeypatch) -> None:
    model = _model()
    config = _config()
    task_optimizer, geometry_optimizer = _optimizers(model, config)
    recorded_outputs = []
    original_forward = model.forward_from_backbone

    def recording_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        recorded_outputs.append(output)
        return output

    monkeypatch.setattr(model, "forward_from_backbone", recording_forward)
    result = joint_structure_da_train_step(
        model,
        _sample(2, 5),
        _sample(3, 7, labels=False),
        task_optimizer,
        geometry_optimizer,
        _objective(config),
        config,
        torch.device("cpu"),
    )
    scalars = result.diagnostics.scalars

    required = {
        "source_train_accuracy", "domain_accuracy", "grl_coefficient",
        "loss_total", "loss_task", "loss_quality_total",
        "loss_quality_structural_cls", "loss_quality_structural_domain",
        "loss_quality_component_cls", "loss_quality_component_domain",
        "loss_geometry_total", "loss_alignment",
        "geometry_T_alignment", "geometry_T_roughness",
        "geometry_T_unsupported", "geometry_T_phase_center",
        "geometry_D_alignment", "geometry_D_roughness",
        "geometry_D_unsupported", "geometry_D_phase_center",
        "tau_fast", "tau_slow", "tau_gap",
    }
    for domain in ("source", "target"):
        required.update(
            f"{domain}_{name}"
            for name in (
                "trend_energy_fraction", "dynamics_energy_fraction",
                "residual_energy_fraction", "reconstruction_relative_error",
                "temporal_T_valid_rate", "temporal_D_valid_rate",
                "channel_T_valid_rate", "channel_D_valid_rate",
                "raw_T_norm", "raw_D_norm", "raw_R_norm",
                "temporal_T_norm", "temporal_D_norm",
                "channel_T_norm", "channel_D_norm",
                "raw_fusion_norm", "temporal_fusion_norm",
                "channel_fusion_norm", "fused_feature_norm",
                "channel_T_relation_mass", "channel_D_relation_mass",
                "channel_T_state_reliability_mean",
                "channel_D_state_reliability_mean",
                "channel_T_evolution_reliability_mean",
                "channel_D_evolution_reliability_mean",
            )
        )
        for coefficient in (
            "alpha_T", "alpha_D", "alpha_R", "beta_T_temporal",
            "beta_D_temporal", "beta_T_channel", "beta_D_channel",
        ):
            required.add(f"{domain}_{coefficient}_mean")
            required.add(f"{domain}_{coefficient}_std")

    assert required <= scalars.keys()
    assert all(
        isinstance(value, torch.Tensor)
        and value.ndim == 0
        and torch.isfinite(value).item()
        for value in scalars.values()
    )
    for domain in ("source", "target"):
        fraction_sum = sum(
            scalars[f"{domain}_{component}_energy_fraction"]
            for component in ("trend", "dynamics", "residual")
        )
        torch.testing.assert_close(fraction_sum, torch.ones_like(fraction_sum))
        assert scalars[f"{domain}_reconstruction_relative_error"] < 1e-10

    source_quality = recorded_outputs[0].representation.quality
    target_quality = recorded_outputs[1].representation.quality
    torch.testing.assert_close(
        scalars["source_alpha_T_mean"], source_quality.alpha_trend.mean()
    )
    torch.testing.assert_close(
        scalars["target_alpha_T_mean"], target_quality.alpha_trend.mean()
    )
    torch.testing.assert_close(
        scalars["source_alpha_T_std"],
        source_quality.alpha_trend.std(unbiased=False),
    )
    torch.testing.assert_close(
        scalars["target_alpha_T_std"],
        target_quality.alpha_trend.std(unbiased=False),
    )


def test_nonfinite_loss_raises_before_optimizer_step(monkeypatch) -> None:
    import methods.structure_da.joint_trainer as module

    model = _model()
    config = _config()
    task_optimizer, geometry_optimizer = _optimizers(model, config)
    original_cross_entropy = module.F.cross_entropy
    call_count = 0

    def first_nan_cross_entropy(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return args[0].sum() * float("nan")
        return original_cross_entropy(*args, **kwargs)

    monkeypatch.setattr(module.F, "cross_entropy", first_nan_cross_entropy)
    for optimizer in (task_optimizer, geometry_optimizer):
        monkeypatch.setattr(
            optimizer,
            "step",
            lambda *args, **kwargs: pytest.fail("optimizer.step must not run"),
        )

    with pytest.raises(FloatingPointError) as error:
        joint_structure_da_train_step(
            model, _sample(2, 5), _sample(2, 7, labels=False),
            task_optimizer, geometry_optimizer, _objective(config), config,
            torch.device("cpu"),
        )

    message = str(error.value)
    for name in (
        "task_loss", "quality_loss", "geometry_loss", "alignment_loss",
        "total_loss",
    ):
        assert name in message


@pytest.mark.parametrize("field,value", [("alpha_trend", 1.1), ("beta_trend_channel", -0.1)])
def test_invalid_quality_coefficient_raises_before_optimizer_step(
    monkeypatch, field, value
) -> None:
    model = _model()
    config = _config()
    task_optimizer, geometry_optimizer = _optimizers(model, config)
    original_forward = model.forward_from_backbone

    def invalid_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        quality = output.representation.quality
        invalid = getattr(quality, field).clone()
        invalid[0] = value
        quality = replace(quality, **{field: invalid})
        return replace(
            output,
            representation=replace(output.representation, quality=quality),
        )

    monkeypatch.setattr(model, "forward_from_backbone", invalid_forward)
    for optimizer in (task_optimizer, geometry_optimizer):
        monkeypatch.setattr(
            optimizer,
            "step",
            lambda *args, **kwargs: pytest.fail("optimizer.step must not run"),
        )

    with pytest.raises(FloatingPointError, match=field):
        joint_structure_da_train_step(
            model, _sample(2, 5), _sample(2, 7, labels=False),
            task_optimizer, geometry_optimizer, _objective(config), config,
            torch.device("cpu"),
        )


def test_invalid_grl_coefficient_raises_before_optimizer_step(monkeypatch) -> None:
    model = _model()
    config = _config()
    task_optimizer, geometry_optimizer = _optimizers(model, config)
    original_align = model.align

    def invalid_align(*args, **kwargs):
        output = original_align(*args, **kwargs)
        return replace(output, coefficient=output.coefficient.new_tensor(1.1))

    monkeypatch.setattr(model, "align", invalid_align)
    for optimizer in (task_optimizer, geometry_optimizer):
        monkeypatch.setattr(
            optimizer,
            "step",
            lambda *args, **kwargs: pytest.fail("optimizer.step must not run"),
        )

    with pytest.raises(FloatingPointError, match="GRL coefficient"):
        joint_structure_da_train_step(
            model, _sample(2, 5), _sample(2, 7, labels=False),
            task_optimizer, geometry_optimizer, _objective(config), config,
            torch.device("cpu"),
        )


def test_loader_factory_forces_no_extra_and_independent_train_loaders(monkeypatch) -> None:
    import methods.structure_da.joint_trainer as module

    datasets = []
    loaders = []

    class FakeDataset:
        def __init__(self, **kwargs):
            datasets.append(kwargs)

    def fake_loader(dataset, batch_size, num_workers):
        loader = object()
        loaders.append((loader, dataset, batch_size, num_workers))
        return loader

    monkeypatch.setattr(module, "PixelSetData", FakeDataset)
    monkeypatch.setattr(module, "create_train_loader", fake_loader)
    config = type("Config", (), {
        "num_pixels": 4, "data_root": "root", "classes": ["a"],
        "closed_set": True, "combine_spring_and_winter": False,
        "source": "source", "target": "target", "batch_size": 2,
        "num_workers": 0,
    })()
    splits = {"source": {"train": {1, 2}}, "target": {"train": {3, 4}}}

    source_loader, target_loader = create_joint_structure_da_train_loaders(config, splits)

    assert source_loader is loaders[0][0]
    assert target_loader is loaders[1][0]
    assert source_loader is not target_loader
    assert all(dataset["with_extra"] is False for dataset in datasets)
    assert datasets[0]["indices"] == {1, 2}
    assert datasets[1]["indices"] == {3, 4}


class _Writer:
    def __init__(self):
        self.tags = set()

    def add_scalar(self, tag, value, global_step):
        self.tags.add(tag)


def test_counterfactual_validation_reuses_one_backbone_forward_per_batch(
    monkeypatch,
) -> None:
    model = _model()
    loader = [_sample(2, 5), _sample(1, 7)]
    backbone_calls = 0
    classifier_calls = 0
    original_backbone = model.forward_backbone
    original_classifier = model.representation.classifier.forward

    def counted_backbone(*args, **kwargs):
        nonlocal backbone_calls
        backbone_calls += 1
        return original_backbone(*args, **kwargs)

    def counted_classifier(*args, **kwargs):
        nonlocal classifier_calls
        classifier_calls += 1
        return original_classifier(*args, **kwargs)

    monkeypatch.setattr(model, "forward_backbone", counted_backbone)
    monkeypatch.setattr(model.representation.classifier, "forward", counted_classifier)

    metrics = validation_structure_contributions(
        model,
        loader,
        torch.device("cpu"),
        classes=("a", "b", "c"),
    )

    assert backbone_calls == len(loader)
    assert classifier_calls == 4 * len(loader)
    assert set(metrics) == {
        "full_loss", "no_temporal_loss", "no_channel_loss", "raw_only_loss",
        "full_f1", "no_temporal_f1", "no_channel_f1", "raw_only_f1",
        "delta_temporal", "delta_channel", "delta_structure",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["delta_temporal"] == pytest.approx(
        metrics["full_f1"] - metrics["no_temporal_f1"]
    )
    assert metrics["delta_channel"] == pytest.approx(
        metrics["full_f1"] - metrics["no_channel_f1"]
    )
    assert metrics["delta_structure"] == pytest.approx(
        metrics["full_f1"] - metrics["raw_only_f1"]
    )


def test_counterfactual_validation_aggregates_metrics_over_the_full_dataset() -> None:
    model = _model()
    loader = [_sample(1, 5), _sample(2, 7)]

    metrics = validation_structure_contributions(
        model,
        loader,
        torch.device("cpu"),
        classes=("a", "b", "c"),
    )

    with torch.inference_mode():
        labels = []
        logits = []
        for sample in loader:
            output = model.forward_details(
                sample["pixels"], sample["valid_pixels"], sample["positions"], None
            )
            labels.append(sample["label"])
            logits.append(model.representation.classifier(output.representation.quality.fused_feature))
        labels = torch.cat(labels)
        logits = torch.cat(logits)
        predictions = logits.argmax(dim=-1)
        expected_ce = F.cross_entropy(logits, labels).item()
        from sklearn.metrics import f1_score

        expected_f1 = f1_score(
            labels.numpy(), predictions.numpy(), average="macro", zero_division=0
        )

    assert metrics["full_loss"] == pytest.approx(expected_ce)
    assert metrics["full_f1"] == pytest.approx(expected_f1)


def test_full_loop_runs_two_steps_saves_and_restores_state(tmp_path, monkeypatch, capsys) -> None:
    import methods.structure_da.joint_trainer as module

    model = _model()
    config = _config()
    writer = _Writer()
    path = tmp_path / "model.pt"
    recorded = []
    original_train_step = module.joint_structure_da_train_step

    def recording_train_step(*args, **kwargs):
        result = original_train_step(*args, **kwargs)
        recorded.append(result)
        return result

    monkeypatch.setattr(module, "joint_structure_da_train_step", recording_train_step)
    monkeypatch.setattr(
        module,
        "tqdm",
        lambda *args, **kwargs: pytest.fail(
            "progress_bar=off must not construct tqdm"
        ),
    )
    train_joint_structure_da(
        model,
        [_sample(2, 5)],
        [_sample(2, 7, labels=False)],
        [_sample(2, 5)],
        config,
        writer,
        torch.device("cpu"),
        path,
    )

    checkpoint = torch.load(path, weights_only=False)
    restored = _model()
    restored.load_state_dict(checkpoint["state_dict"])
    assert torch.isfinite(torch.tensor(checkpoint["best_f1"]))
    assert _counts(restored) == (2, 2, 2, 2, 2)
    assert restored.alignment.grl.iteration.item() == 2
    epoch_log = capsys.readouterr().out
    assert "TRAIN_STEP|" in epoch_log
    assert "TRAIN_EPOCH|" in epoch_log
    assert "STRUCTURE_EPOCH|" in epoch_log
    assert "QUALITY_EPOCH|" in epoch_log
    assert "GEOMETRY_EPOCH|" in epoch_log
    assert epoch_log.count("DECOMP_EPOCH|") == 2
    assert epoch_log.count("CONTRIBUTION_EPOCH|") == 2
    assert "VAL_CONTRIBUTION|" in epoch_log
    train_step = next(
        line for line in epoch_log.splitlines() if line.startswith("TRAIN_STEP|")
    )
    for name in (
        "total", "task", "q_total", "geometry", "alignment",
        "train_acc", "domain_acc", "grl", "q_dom_w", "lr",
    ):
        assert f"|{name}=" in train_step
    train_epoch = next(
        line for line in epoch_log.splitlines() if line.startswith("TRAIN_EPOCH|")
    )
    for name in (
        "total", "task", "q_total", "q_struct_cls", "q_struct_dom",
        "q_comp_cls", "q_comp_dom", "geometry", "alignment",
        "train_acc", "domain_acc", "grl", "q_dom_w", "lr",
    ):
        assert f"|{name}=" in train_epoch
    structure_epoch = next(
        line for line in epoch_log.splitlines()
        if line.startswith("STRUCTURE_EPOCH|")
    )
    for name in (
        "tau_fast", "tau_slow", "tau_gap", "energy_T_s", "energy_T_t",
        "reconstruction_s", "reconstruction_t", "temporal_T_valid_s",
        "channel_T_valid_t", "raw_fusion_norm_s", "temporal_fusion_norm_t",
        "channel_fusion_norm_s", "channel_T_relation_mass_s",
    ):
        assert f"|{name}=" in structure_epoch
    quality_epoch = next(
        line for line in epoch_log.splitlines() if line.startswith("QUALITY_EPOCH|")
    )
    for name in (
        "alpha_T_s", "alpha_T_t", "alpha_D_s", "alpha_D_t",
        "alpha_R_s", "alpha_R_t", "beta_T_temporal_s",
        "beta_T_temporal_t", "beta_D_channel_s", "beta_D_channel_t",
        "domain_score_weight",
    ):
        assert f"|{name}=" in quality_epoch
    geometry_epoch = next(
        line for line in epoch_log.splitlines() if line.startswith("GEOMETRY_EPOCH|")
    )
    for name in (
        "T_align", "T_rough", "T_unsupported", "T_center",
        "D_align", "D_rough", "D_unsupported", "D_center",
    ):
        assert f"|{name}=" in geometry_epoch
    required = {
        "train/loss_total", "train/loss_task", "train/loss_quality_total",
        "train/loss_quality_structural_cls", "train/loss_quality_structural_domain",
        "train/loss_quality_component_cls", "train/loss_quality_component_domain",
        "train/loss_geometry", "train/loss_alignment", "train/domain_accuracy",
        "train/grl_coefficient", "train/alpha_trend", "train/alpha_dynamics",
        "train/alpha_residual", "train/beta_trend_temporal",
        "train/beta_dynamics_temporal", "train/beta_trend_channel",
        "train/beta_dynamics_channel", "train/lr",
        "train/quality/domain_score_weight",
        "train/source/energy_fraction_trend",
        "train/target/energy_fraction_trend",
        "train/source/temporal_valid_trend",
        "train/target/temporal_valid_trend",
        "train/source/fusion_norm_raw",
        "train/target/fusion_norm_raw",
        "train/source/alpha_trend_mean",
        "train/source/alpha_trend_std",
        "train/target/alpha_trend_mean",
        "train/target/alpha_trend_std",
        "train/geometry/trend_alignment",
        "train/geometry/dynamics_alignment",
        "train/diagnostics/source/decomposition/energy_closure_relative_error",
        "train/diagnostics/target/decomposition/roughness_D",
        "train/diagnostics/source/contribution/effective_T_temporal_norm",
        "train/diagnostics/target/contribution/fusion_share_channel",
        "val/counterfactual/full_f1",
        "val/counterfactual/delta_structure",
    }
    assert required <= writer.tags
