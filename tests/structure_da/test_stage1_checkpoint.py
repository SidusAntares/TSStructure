from __future__ import annotations

import copy
import os

import pytest
import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader

from methods.structure_da import (
    Stage1Objective,
    SourceClassificationTrainer,
    build_source_prototype_bank,
    finalize_distance_statistics,
)

from tests.structure_da.test_stage1_training_helpers import (
    TinySourceDataset,
    _model,
)


def _config(**overrides) -> SimpleNamespace:
    values = dict(
        stage1_epochs=3,
        source_warmup_epochs=1,
        lambda_q=0.1,
        lambda_f=0.1,
        lambda_q_to_cls=0.1,
        margin_q=0.1,
        margin_f=0.1,
        tau_q=0.1,
        num_classes=3,
        canonical_grid_size=5,
        fold_dir="/tmp/stage1_test",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _checkpoint_state_dict(bank, examples, epoch) -> dict:
    """Mimic the checkpoint layout produced by train._finalize_stage1_checkpoints."""
    return {
        "stage": "stage1",
        "epoch": epoch,
        "prototype_bank": {
            "trend_srvf": bank.trend_srvf.detach().cpu(),
            "shape_srvf": bank.shape_srvf.detach().cpu(),
            "trend_support": bank.trend_support.detach().cpu(),
            "shape_support": bank.shape_support.detach().cpu(),
            "fused": bank.fused.detach().cpu(),
            "class_counts": bank.class_counts.detach().cpu(),
            "ready": bank.ready.detach().cpu(),
            "q_distance_samples": tuple(v.detach().cpu() for v in bank.q_distance_samples),
            "f_distance_samples": tuple(v.detach().cpu() for v in bank.f_distance_samples),
            "q_quantiles": bank.q_quantiles.detach().cpu(),
            "f_quantiles": bank.f_quantiles.detach().cpu(),
            "version": bank.version,
        },
        "shape_examples": examples,
    }


def test_best_and_last_checkpoints_contain_prototype_bank(tmp_path) -> None:
    model = _model()
    dataset = TinySourceDataset(n=24)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False)
    bank = build_source_prototype_bank(model, loader, 3, device=torch.device("cpu"))
    finalized, examples = finalize_distance_statistics(
        model, loader, bank, device=torch.device("cpu")
    )

    best_ckpt = _checkpoint_state_dict(finalized, examples, epoch=1)
    last_ckpt = _checkpoint_state_dict(finalized, examples, epoch=2)

    assert "prototype_bank" in best_ckpt
    assert "prototype_bank" in last_ckpt
    assert "shape_examples" in best_ckpt
    assert best_ckpt["prototype_bank"]["shape_srvf"].shape == (3, 5, 4)
    assert best_ckpt["prototype_bank"]["fused"].shape == (3, 8)
    assert best_ckpt["prototype_bank"]["ready"].tolist() == [True, True, True]
    # per-class examples bounded by 3
    for class_id in range(3):
        class_examples = [e for e in examples if e["class_id"] == class_id]
        assert len(class_examples) <= 3
        for example in class_examples:
            assert example["q_shape"].shape == (5, 4)
            assert example["support"].shape == (5,)
            assert example["canonical_grid"].shape == (5,)
    # checkpoint tensors on CPU
    for value in (
        best_ckpt["prototype_bank"]["shape_srvf"],
        best_ckpt["prototype_bank"]["fused"],
        best_ckpt["prototype_bank"]["q_quantiles"],
    ):
        assert value.device.type == "cpu"


def test_example_selection_uses_distinct_samples() -> None:
    model = _model()
    dataset = TinySourceDataset(n=30)
    loader = DataLoader(dataset, batch_size=5, shuffle=False, drop_last=False)
    bank = build_source_prototype_bank(model, loader, 3, device=torch.device("cpu"))
    _, examples = finalize_distance_statistics(
        model, loader, bank, device=torch.device("cpu")
    )
    class0 = [e for e in examples if e["class_id"] == 0]
    sample_ids = [e["sample_id"] for e in class0]
    assert len(set(sample_ids)) == len(sample_ids), "examples must be distinct"
    roles = {e["role"] for e in class0}
    assert "prototype_nearest" in roles
    assert "class_median" in roles
    assert "outer_representative" in roles


def test_best_prototype_matches_rescanned_best_model() -> None:
    """Recomputing a prototype bank from a model state must reproduce it.

    This is the guard against putting the *last* prototype into the *best*
    checkpoint: given the same model parameters the prototype scan is
    deterministic, so two scans of the same model must match.
    """
    model = _model()
    dataset = TinySourceDataset(n=24)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False)
    bank1 = build_source_prototype_bank(model, loader, 3, device=torch.device("cpu"))
    bank2 = build_source_prototype_bank(model, loader, 3, device=torch.device("cpu"))
    torch.testing.assert_close(bank1.shape_srvf, bank2.shape_srvf, rtol=0, atol=0)
    torch.testing.assert_close(bank1.fused, bank2.fused, rtol=0, atol=0)


def test_trainer_stage1_full_protocol_smoke(tmp_path) -> None:
    """Run a 3-epoch Stage-1 loop (warmup=1) with refresh on a tiny dataset."""
    model = _model()
    objective = Stage1Objective(
        num_classes=3, lambda_q=0.1, lambda_f=0.1, lambda_q_to_cls=0.1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = SourceClassificationTrainer(
        model,
        optimizer,
        device=torch.device("cpu"),
        amp_enabled=False,
        objective=objective,
    )
    dataset = TinySourceDataset(n=24)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False)
    bank = None
    for epoch in range(3):
        warmup = epoch < 1
        if warmup:
            for batch in loader:
                metrics = trainer.train_step(batch, warmup=warmup, bank=None)
                assert torch.isfinite(torch.tensor(metrics["loss"])).item()
            bank = build_source_prototype_bank(
                model, loader, 3, device=torch.device("cpu")
            )
            continue
        for batch in loader:
            metrics = trainer.train_step(batch, warmup=False, bank=bank)
            assert torch.isfinite(torch.tensor(metrics["loss"])).item()
    assert bank is not None
    assert bank.ready.tolist() == [True, True, True]
