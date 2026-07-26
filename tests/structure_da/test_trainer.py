"""Regression tests for the formal Structure DA training integration."""

from types import SimpleNamespace

import pytest
import torch

from methods.structure_da import LossWeights, StructureDAModel
from methods.structure_da import trainer


def config(**updates):
    values = dict(
        epochs=2, steps_per_epoch=None, lr=1e-3, weight_decay=0.0,
        quality_warmup_steps=None, grl_warmup_steps=None, grl_gamma=10.0,
        loss_weights=LossWeights(), log_step=1, progress_bar="off",
        classes=("a", "b", "c"),
    )
    values.update(updates)
    return trainer.StructureDATrainingConfig(**values)


class Loader:
    def __init__(self, size, values=None):
        self.size = size
        self.values = list(range(size)) if values is None else values
    def __len__(self):
        return self.size
    def __iter__(self):
        return iter(self.values)


@pytest.mark.parametrize("field,value", [
    ("epochs", 0), ("epochs", True), ("steps_per_epoch", 0),
    ("quality_warmup_steps", -1), ("grl_warmup_steps", False),
    ("lr", 0), ("lr", float("inf")), ("weight_decay", -1),
    ("grl_gamma", 0), ("log_step", 0),
])
def test_training_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        config(**{field: value})


def test_auto_steps_use_longer_loader_and_default_warmups():
    result = trainer.resolve_structure_da_training(config(), Loader(2), Loader(3))
    assert result == trainer.ResolvedStructureDATraining(3, 3, 3)


def test_custom_steps_and_warmups_are_preserved():
    result = trainer.resolve_structure_da_training(
        config(steps_per_epoch=5, quality_warmup_steps=7, grl_warmup_steps=9),
        Loader(2), Loader(3),
    )
    assert result == trainer.ResolvedStructureDATraining(5, 7, 9)


@pytest.mark.parametrize("source,target,name", [(0, 2, "source"), (2, 0, "target")])
def test_empty_loader_rejected(source, target, name):
    with pytest.raises(ValueError, match=f"{name}.*batch_size"):
        trainer.resolve_structure_da_training(config(), Loader(source), Loader(target))


def test_train_transforms_are_exact_and_loaders_are_independent(monkeypatch):
    datasets = []
    class Dataset:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            datasets.append(self)
    monkeypatch.setattr(trainer, "PixelSetData", Dataset)
    monkeypatch.setattr(trainer, "create_train_loader", lambda ds, *args: object())
    args = SimpleNamespace(
        num_pixels=4, data_root="root", source="s", target="t",
        classes=["a"], closed_set=True, combine_spring_and_winter=False,
        batch_size=2, num_workers=0, with_extra=False,
    )
    source, target = trainer.create_structure_da_train_loaders(
        args, {"s": {"train": {1}}, "t": {"train": {2}}}
    )
    assert source is not target
    assert [[type(x).__name__ for x in ds.transform.transforms] for ds in datasets] == [
        ["RandomSamplePixels", "Normalize", "ToTensor"],
        ["RandomSamplePixels", "Normalize", "ToTensor"],
    ]


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.forward_calls = []
        self.adapt_calls = []
    def forward_details(self, *args, quality_progress):
        self.forward_calls.append((args, quality_progress))
        return SimpleNamespace(component=object())
    def adapt(self, source, target, grl_coefficient):
        self.adapt_calls.append(grl_coefficient)
        return object()


def sample(batch=2, length=4, label=True):
    result = {
        "pixels": torch.randn(batch, length, 3, 5),
        "valid_pixels": torch.ones(batch, length, 5),
        "positions": torch.arange(length),
    }
    if label:
        result["label"] = torch.arange(batch) % 3
    return result


def patch_losses(monkeypatch, model, values=(1, 2, 3, -4, 5)):
    names = (
        "classification_loss", "quality_domain_loss",
        "quality_classification_loss", "component_diversity_loss",
        "structural_adversarial_loss",
    )
    calls = []
    for name, value in zip(names, values):
        monkeypatch.setattr(
            trainer, name,
            lambda *args, _name=name, _value=value: (
                calls.append(_name) or model.parameter * 0 + float(_value)
            ),
        )
    return calls


def test_step_needs_source_label():
    model = FakeModel()
    with pytest.raises(KeyError, match="label"):
        trainer.structure_da_train_step(
            model, sample(label=False), sample(label=False),
            torch.optim.SGD(model.parameters(), lr=0.1), "cpu", 0, 2, 3, 10,
            LossWeights(),
        )


def test_step_does_not_need_or_read_target_label(monkeypatch):
    model = FakeModel()
    calls = patch_losses(monkeypatch, model)
    result = trainer.structure_da_train_step(
        model, sample(batch=2), sample(batch=3, length=5, label=False),
        torch.optim.SGD(model.parameters(), lr=0.1), "cpu", 0, 2, 4, 10,
        LossWeights(qdom=2, qcls=3, diversity=4, sda=5),
    )
    assert result.source_batch_size == 2 and result.target_batch_size == 3
    assert result.losses.total.item() == pytest.approx(1 + 2*2 + 3*3 + 4*(-4) + 5*5)
    assert len(model.forward_calls) == 2 and len(model.adapt_calls) == 1
    assert model.forward_calls[0][1] == model.forward_calls[1][1] == 0
    assert model.adapt_calls == [0]
    assert len(calls) == 5
    assert model.forward_calls[0][0][0].shape[1] == 4
    assert model.forward_calls[1][0][0].shape[1] == 5


def test_independent_midpoint_schedules_are_forwarded(monkeypatch):
    model = FakeModel()
    patch_losses(monkeypatch, model)
    result = trainer.structure_da_train_step(
        model, sample(), sample(label=False),
        torch.optim.SGD(model.parameters(), lr=0.1), "cpu", 5, 10, 20, 10,
        LossWeights(),
    )
    assert result.quality_progress == pytest.approx(0.5)
    assert result.grl_coefficient == pytest.approx(trainer.grl_coefficient(5, 20, 10))


def make_model():
    return StructureDAModel(
        num_classes=3, input_dim=3, pse_mlp1=(3, 4), pse_pooling="mean_std",
        pse_mlp2=(8, 6), n_head=2, d_k=2, d_model=8, ltae_mlp=(8, 6),
        dropout=0.0, max_position=64, max_temporal_shift=4,
        classifier_hidden=(5,), quality_hidden_cap=5, sda_hidden_dim=8,
    )


def test_true_cpu_train_step_is_finite_and_updates_model():
    torch.manual_seed(4)
    model = make_model()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    result = trainer.structure_da_train_step(
        model, sample(batch=3, length=4), sample(batch=2, length=5, label=False),
        torch.optim.Adam(model.parameters(), lr=1e-3), "cpu", 1, 1, 1, 10,
        LossWeights(),
    )
    assert torch.isfinite(result.losses.total)
    changed = {name for name, value in model.named_parameters() if not torch.equal(value, before[name])}
    assert any(name.startswith("spatial_encoder") for name in changed)
    assert any("decomposition" in name for name in changed)
    assert any("ltae" in name for name in changed)
    assert any("classifier" in name for name in changed)
    assert any("transferability" in name for name in changed)
    assert any("discriminability" in name for name in changed)
    assert any("diversity" in name for name in changed)
    assert any("adversarial_adapter" in name for name in changed)


class Writer:
    def add_scalar(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("source_size,target_size,steps", [(2, 3, 3), (4, 2, 4)])
def test_high_level_cycles_shorter_loader_and_validates_once(
    monkeypatch, source_size, target_size, steps
):
    seen = []
    model = torch.nn.Linear(1, 1)
    losses = trainer.StructureDALosses(*(torch.tensor(1.0) for _ in range(6)))
    def fake_step(model, source, target, optimizer, *args):
        seen.append((source, target))
        optimizer.zero_grad(set_to_none=True)
        (model.weight.sum() * 0).backward()
        optimizer.step()
        return trainer.StructureDATrainStepOutput(losses, 0.0, 0.0, 1, 1)
    validations = []
    monkeypatch.setattr(trainer, "structure_da_train_step", fake_step)
    monkeypatch.setattr(trainer, "validation", lambda *args: validations.append(args) or 0.5)
    trainer.train_structure_da(
        model, Loader(source_size), Loader(target_size), None,
        config(epochs=1), Writer(), "cpu", "best.pt",
    )
    assert len(seen) == steps
    assert len(validations) == 1 and validations[0][1] == "best.pt"
    assert validations[0][0] == float("-inf")


def test_custom_step_budget_is_exact(monkeypatch):
    count = []
    model = torch.nn.Linear(1, 1)
    losses = trainer.StructureDALosses(*(torch.tensor(1.0) for _ in range(6)))
    def fake_step(model, source, target, optimizer, *args):
        count.append(1); optimizer.zero_grad(); (model.weight.sum()*0).backward(); optimizer.step()
        return trainer.StructureDATrainStepOutput(losses, 0, 0, 1, 1)
    monkeypatch.setattr(trainer, "structure_da_train_step", fake_step)
    monkeypatch.setattr(trainer, "validation", lambda *args: 0)
    trainer.train_structure_da(model, Loader(2), Loader(3), None, config(epochs=1, steps_per_epoch=5), Writer(), "cpu", None)
    assert len(count) == 5


def test_scheduler_steps_after_optimizer(monkeypatch):
    events = []
    model = torch.nn.Linear(1, 1)
    losses = trainer.StructureDALosses(*(torch.tensor(1.0) for _ in range(6)))
    def fake_step(model, source, target, optimizer, *args):
        events.append("optimizer")
        return trainer.StructureDATrainStepOutput(losses, 0, 0, 1, 1)
    class Scheduler:
        def __init__(self, *args, **kwargs): pass
        def step(self): events.append("scheduler")
    monkeypatch.setattr(trainer, "structure_da_train_step", fake_step)
    monkeypatch.setattr(trainer.torch.optim.lr_scheduler, "CosineAnnealingLR", Scheduler)
    monkeypatch.setattr(trainer, "validation", lambda *args: 0)
    trainer.train_structure_da(model, Loader(1), Loader(1), None, config(epochs=1), Writer(), "cpu", None)
    assert events == ["optimizer", "scheduler"]


def test_trainer_auto_progress_falls_back_for_redirected_stderr(monkeypatch):
    captured = []
    model = torch.nn.Linear(1, 1)
    losses = trainer.StructureDALosses(*(torch.tensor(1.0) for _ in range(6)))
    training_config = config(epochs=1, progress_bar="auto")
    monkeypatch.setattr(
        trainer, "progress_bar_disabled",
        lambda mode: captured.append(mode) or True,
    )
    monkeypatch.setattr(
        trainer, "structure_da_train_step",
        lambda model, source, target, optimizer, *args: (
            optimizer.step()
            or trainer.StructureDATrainStepOutput(losses, 0, 0, 1, 1)
        ),
    )
    monkeypatch.setattr(trainer, "validation", lambda *args: 0)
    trainer.train_structure_da(model, Loader(1), Loader(1), None, training_config, Writer(), "cpu", None)
    assert captured == ["auto"]


def test_target_label_values_do_not_change_loss(monkeypatch):
    totals = []
    for target_label in (torch.tensor([0, 0]), torch.tensor([2, 1])):
        model = FakeModel()
        patch_losses(monkeypatch, model)
        target = sample(batch=2)
        target["label"] = target_label
        result = trainer.structure_da_train_step(
            model, sample(batch=2), target,
            torch.optim.SGD(model.parameters(), lr=0.1), "cpu", 0, 2, 2, 10,
            LossWeights(),
        )
        totals.append(result.losses.total.item())
    assert totals[0] == totals[1]
