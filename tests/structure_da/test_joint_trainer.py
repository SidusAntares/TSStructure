from __future__ import annotations

import pytest
import torch

from methods.structure_da.joint_trainer import (
    JointStructureDATrainingConfig,
    create_joint_structure_da_train_loaders,
    joint_structure_da_train_step,
    train_joint_structure_da,
)
from methods.structure_da.quality_fusion import HierarchicalQualityObjective
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


def test_training_config_rejects_invalid_values_and_has_no_legacy_fields() -> None:
    config = _config()
    assert not hasattr(config, "quality_warmup_steps")
    assert not hasattr(config, "quality_progress")
    assert not hasattr(config, "diversity_weight")
    assert not hasattr(config, "coefficient_stop_gradient")
    for field in (
        "task_weight", "geometry_weight", "alignment_weight",
        "structural_classification_weight", "structural_domain_weight",
        "component_classification_weight", "component_domain_weight",
    ):
        with pytest.raises(ValueError):
            _config(**{field: -1.0})


def test_joint_step_supports_unequal_lengths_and_never_reads_target_label() -> None:
    model = _model()
    model.alignment.grl.iteration.fill_(2)
    config = _config()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    step_calls = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    result = joint_structure_da_train_step(
        model,
        _sample(2, 5),
        _sample(3, 7, labels=False),
        optimizer,
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
        model.representation.shared_ltae,
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


def test_full_loop_runs_two_steps_saves_and_restores_state(tmp_path, monkeypatch, capsys) -> None:
    import methods.structure_da.joint_trainer as module

    model = _model()
    config = _config()
    writer = _Writer()
    path = tmp_path / "model.pt"

    # Exercise the standard four-argument forward without CUDA-bound evaluation.
    def fake_validation(best_f1, best_model_path, training_config, criterion,
                        device, epoch, validation_model, val_loader, validation_writer):
        sample = _sample(2, 5)
        validation_model(
            sample["pixels"], sample["valid_pixels"], sample["positions"], None
        )
        torch.save(
            {"epoch": epoch, "state_dict": validation_model.state_dict(), "best_f1": 0.5},
            best_model_path,
        )
        return 0.5

    monkeypatch.setattr(module, "validation", fake_validation)
    train_joint_structure_da(
        model,
        [_sample(2, 5)],
        [_sample(2, 7, labels=False)],
        object(),
        config,
        writer,
        torch.device("cpu"),
        path,
    )

    checkpoint = torch.load(path, weights_only=False)
    restored = _model()
    restored.load_state_dict(checkpoint["state_dict"])
    assert checkpoint["best_f1"] == 0.5
    assert _counts(restored) == (2, 2, 2, 2, 2)
    assert restored.alignment.grl.iteration.item() == 2
    assert "TRAIN_EPOCH|" in capsys.readouterr().out
    required = {
        "train/loss_total", "train/loss_task", "train/loss_quality_total",
        "train/loss_quality_structural_cls", "train/loss_quality_structural_domain",
        "train/loss_quality_component_cls", "train/loss_quality_component_domain",
        "train/loss_geometry", "train/loss_alignment", "train/domain_accuracy",
        "train/grl_coefficient", "train/alpha_trend", "train/alpha_dynamics",
        "train/alpha_residual", "train/beta_trend_temporal",
        "train/beta_dynamics_temporal", "train/beta_trend_channel",
        "train/beta_dynamics_channel", "train/lr",
    }
    assert required <= writer.tags
