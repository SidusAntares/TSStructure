from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
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
    torch.manual_seed(0)
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


class _RecordingWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        del args, kwargs


def test_train_source_classification_builds_bank_at_warmup_boundary(
    monkeypatch, tmp_path
) -> None:
    # train.py imports the real dataset stack; zarr is absent in the lightweight
    # sandbox but present in the formal project environment.
    pytest.importorskip("zarr")
    import train as train_module

    torch.manual_seed(0)
    model = _model()
    dataset = TinySourceDataset(n=24)
    source_loader = DataLoader(dataset, batch_size=6, shuffle=False, drop_last=False)
    source_scan_loader = DataLoader(
        dataset, batch_size=6, shuffle=False, drop_last=False
    )

    config = SimpleNamespace(
        stage1_epochs=3,
        source_warmup_epochs=1,
        steps_per_epoch=None,
        lr=1e-3,
        weight_decay=0.0,
        num_classes=3,
        lambda_q=0.1,
        lambda_f=0.1,
        lambda_q_to_cls=0.1,
        margin_q=0.1,
        margin_f=0.1,
        tau_q=0.1,
        fold_dir=str(tmp_path),
        log_step=1000,
        amp=False,
        amp_dtype="float16",
        progress_bar="off",
    )

    events: list[tuple] = []
    scanned_banks = []
    objective_calls = []
    non_warmup_metrics = []

    original_scan = train_module.build_source_prototype_bank

    def tracked_scan(*args, **kwargs):
        bank = original_scan(*args, **kwargs)
        assert torch.all(bank.ready).item()
        scanned_banks.append(bank)
        events.append(("scan", id(bank)))
        return bank

    monkeypatch.setattr(train_module, "build_source_prototype_bank", tracked_scan)

    original_train_step = SourceClassificationTrainer.train_step

    def tracked_train_step(self, batch, *, warmup=False, bank=None):
        events.append(("step", warmup, None if bank is None else id(bank)))
        metrics = original_train_step(self, batch, warmup=warmup, bank=bank)
        if not warmup:
            non_warmup_metrics.append(metrics)
        return metrics

    monkeypatch.setattr(
        SourceClassificationTrainer, "train_step", tracked_train_step
    )

    original_objective_forward = train_module.Stage1Objective.forward

    def tracked_objective_forward(self, **kwargs):
        if not kwargs.get("warmup", False):
            for name in (
                "q",
                "q_support",
                "q_valid",
                "integration_weights",
                "bank",
            ):
                assert kwargs[name] is not None
            objective_calls.append(
                (
                    id(kwargs["bank"]),
                    kwargs["q"].shape,
                    kwargs["q_support"].shape,
                    kwargs["q_valid"].shape,
                    kwargs["integration_weights"].shape,
                )
            )
        return original_objective_forward(self, **kwargs)

    monkeypatch.setattr(
        train_module.Stage1Objective, "forward", tracked_objective_forward
    )

    def fake_source_validation(
        best_f1, best_model_path, cfg, device, epoch, model, val_loader, writer
    ):
        del best_model_path, cfg, device, model, val_loader, writer
        macro_f1 = 0.5 + 0.01 * epoch
        return max(best_f1, macro_f1), {
            "accuracy": 0.5,
            "macro_f1": macro_f1,
        }

    monkeypatch.setattr(train_module, "_source_validation", fake_source_validation)
    monkeypatch.setattr(
        train_module, "_finalize_stage1_checkpoints", lambda *args, **kwargs: None
    )

    train_module.train_source_classification(
        model,
        source_loader,
        source_loader,
        config,
        _RecordingWriter(),
        torch.device("cpu"),
        None,
        source_scan_loader=source_scan_loader,
    )

    steps_per_epoch = len(source_loader)
    assert len(scanned_banks) == 2
    assert scanned_banks[0] is not scanned_banks[1]

    # Epoch 1 is CE-only. Its boundary scan builds the bank used by every
    # mini-batch in epoch 2; the epoch-2 boundary scan builds the epoch-3 bank.
    assert events[:steps_per_epoch] == [
        ("step", True, None)
    ] * steps_per_epoch
    assert events[steps_per_epoch] == ("scan", id(scanned_banks[0]))

    epoch2_start = steps_per_epoch + 1
    epoch2_end = epoch2_start + steps_per_epoch
    assert events[epoch2_start:epoch2_end] == [
        ("step", False, id(scanned_banks[0]))
    ] * steps_per_epoch
    assert events[epoch2_end] == ("scan", id(scanned_banks[1]))
    assert events[epoch2_end + 1 :] == [
        ("step", False, id(scanned_banks[1]))
    ] * steps_per_epoch

    assert len(objective_calls) == 2 * steps_per_epoch
    assert len(non_warmup_metrics) == 2 * steps_per_epoch
    for metrics in non_warmup_metrics:
        for name in ("q_proto_loss", "f_proto_loss", "q_to_cls_loss"):
            assert torch.isfinite(torch.tensor(metrics[name])).item()
    assert {call[0] for call in objective_calls[:steps_per_epoch]} == {
        id(scanned_banks[0])
    }
    assert {call[0] for call in objective_calls[steps_per_epoch:]} == {
        id(scanned_banks[1])
    }
    for _, q_shape, support_shape, valid_shape, weight_shape in objective_calls:
        assert q_shape[:2] == (6, 5)
        assert support_shape == (6, 5)
        assert valid_shape == (6,)
        assert weight_shape == (5,)


def test_full_scan_loaders_group_variable_pixel_widths_before_collate(monkeypatch) -> None:
    """Full source/target scans must not stack parcels with different pixel widths."""

    pytest.importorskip("zarr")
    import numpy as np
    import train as train_module
    from dataset import PixelSetData as RealPixelSetData

    class VariableWidthPixelSetData(RealPixelSetData):
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.widths = (3, 4, 3, 4, 5)

        def __len__(self) -> int:
            return len(self.widths)

        def get_shapes(self):
            return [(5, 2, width) for width in self.widths]

        def __getitem__(self, index: int):
            width = self.widths[index]
            return {
                "index": index,
                "parcel_index": index,
                "pixels": torch.full((5, 2, width), float(index)),
                "valid_pixels": torch.ones(5, width),
                "positions": torch.arange(5, dtype=torch.float32),
                "extra": torch.zeros(4),
                "label": index % 3,
            }

        def get_labels(self):
            return np.asarray([index % 3 for index in range(len(self))])

    monkeypatch.setattr(train_module, "PixelSetData", VariableWidthPixelSetData)

    config = SimpleNamespace(
        data_root="unused",
        source="source/domain/2017",
        target="target/domain/2017",
        classes=("a", "b", "c"),
        closed_set=True,
        combine_spring_and_winter=False,
        time_coordinate_mode="canonical_day_of_year",
        eval_batch_size=4,
        batch_size=4,
        num_workers=0,
    )
    splits = {
        config.source: {"train": list(range(5))},
        config.target: {"train": list(range(5))},
    }

    for loader in (
        train_module.create_source_scan_loader(config, splits),
        train_module.create_target_statistics_loader(config, splits),
    ):
        seen = []
        for batch in loader:
            # Reaching this point is the regression check: default_collate can
            # stack the batch only because every parcel has the same width.
            seen.extend(batch["parcel_index"].tolist())
            assert batch["pixels"].shape[-1] in {3, 4, 5}
        assert sorted(seen) == list(range(5))
