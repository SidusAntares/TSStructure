from __future__ import annotations

import inspect

import torch

from methods.structure_da import (
    SourceClassificationTrainer,
    build_source_prototype_bank,
)
from torch.utils.data import DataLoader

from tests.structure_da.test_stage1_training_helpers import (
    TinySourceDataset,
    _bank,
    _batch,
    _model,
    _objective,
)


def _trainer(model, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return SourceClassificationTrainer(
        model,
        optimizer,
        device=torch.device("cpu"),
        amp_enabled=False,
        objective=_objective(),
        **kwargs,
    )


def test_warmup_step_uses_ce_only_and_zero_prototype_losses() -> None:
    model = _model()
    trainer = _trainer(model)
    metrics = trainer.train_step(_batch(), warmup=True, bank=None)
    assert metrics["q_proto_loss"] == 0.0
    assert metrics["f_proto_loss"] == 0.0
    assert metrics["q_to_cls_loss"] == 0.0
    assert metrics["loss"] == metrics["classification_loss"]


def test_warmup_skips_geometry(monkeypatch) -> None:
    model = _model()
    trainer = _trainer(model)
    calls = 0
    original = model.temporal_module.trend_geometry.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.temporal_module.trend_geometry, "forward", counted)
    trainer.train_step(_batch(), warmup=True)
    assert calls == 0


def test_post_warmup_step_runs_geometry_and_full_loss() -> None:
    model = _model()
    trainer = _trainer(model)
    bank = _bank()
    metrics = trainer.train_step(_batch(), warmup=False, bank=bank)
    assert "q_proto_loss" in metrics
    assert metrics["loss"] != metrics["classification_loss"] or metrics["q_proto_loss"] > 0


def test_bank_version_fixed_within_epoch() -> None:
    model = _model()
    trainer = _trainer(model)
    bank = _bank()
    # Two steps against the same bank must never mutate it.
    metrics1 = trainer.train_step(_batch(), warmup=False, bank=bank)
    version_before = bank.version
    metrics2 = trainer.train_step(_batch(), warmup=False, bank=bank)
    assert bank.version == version_before
    assert torch.equal(bank.fused, _bank().fused)


def test_full_source_scan_counts_each_sample_once() -> None:
    model = _model().eval()
    dataset = TinySourceDataset(n=24)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False)
    bank = build_source_prototype_bank(model, loader, 3, device=torch.device("cpu"))
    # 24 samples / 3 classes = 8 each.
    assert bank.class_counts.tolist() == [8, 8, 8]
    assert bank.ready.tolist() == [True, True, True]


def test_trainer_requires_no_target_loader() -> None:
    model = _model()
    trainer = _trainer(model)
    # The trainer's API takes only a source batch; passing a target dict must
    # not be part of the signature at all.
    import inspect

    signature = inspect.signature(SourceClassificationTrainer.train_step)
    assert "target" not in signature.parameters
    assert "target_batch" not in signature.parameters
