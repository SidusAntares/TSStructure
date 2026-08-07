from __future__ import annotations

import pytest
import torch

from methods.structure_da import SourceClassificationTrainer, TSStructureModel


def _model(**overrides) -> TSStructureModel:
    options = dict(
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
    options.update(overrides)
    return TSStructureModel(**options)


def _batch(batch_size: int = 3, length: int = 5) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    return {
        "pixels": torch.randn(batch_size, length, 2, 4),
        "valid_pixels": torch.ones(batch_size, length, 4, dtype=torch.bool),
        "positions": torch.linspace(0, 300, length).round().long().expand(batch_size, -1),
        "label": torch.tensor([0, 1, 2]),
    }


def test_source_trainer_step_updates_parameters_without_target() -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = SourceClassificationTrainer(
        model, optimizer, device=torch.device("cpu"), amp_enabled=False
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}

    metrics = trainer.train_step(_batch(), warmup=True)

    assert set(metrics) == {
        "loss", "classification_loss", "accuracy",
        "q_proto_loss", "f_proto_loss", "q_to_cls_loss",
        "q_valid_count", "f_valid_count", "consistency_valid_count",
    }
    assert torch.isfinite(torch.tensor(metrics["loss"])).item()
    assert metrics["q_proto_loss"] == 0.0
    assert metrics["f_proto_loss"] == 0.0
    assert metrics["q_to_cls_loss"] == 0.0
    after = model.state_dict()
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert changed, "no parameter was updated by the source CE step"


def test_source_trainer_skips_functional_fit(monkeypatch) -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = SourceClassificationTrainer(
        model, optimizer, device=torch.device("cpu"), amp_enabled=False
    )
    calls = 0
    original = model.temporal_module.trend_geometry.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.temporal_module.trend_geometry, "forward", counted)
    trainer.train_step(_batch(), warmup=True)
    assert calls == 0


def test_source_trainer_cpu_smoke_without_amp() -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = SourceClassificationTrainer(
        model, optimizer, device=torch.device("cpu"), amp_enabled=False
    )
    for _ in range(3):
        metrics = trainer.train_step(_batch(), warmup=True)
        assert torch.isfinite(torch.tensor(metrics["loss"])).item()
    assert model.training


def test_source_trainer_requires_model_type() -> None:
    with pytest.raises(ValueError, match="TSStructureModel"):
        SourceClassificationTrainer(
            torch.nn.Linear(2, 2),
            torch.optim.Adam(torch.nn.Linear(2, 2).parameters()),
            device=torch.device("cpu"),
            amp_enabled=False,
        )
