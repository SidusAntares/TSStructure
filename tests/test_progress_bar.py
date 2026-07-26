import inspect
import io
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tqdm import tqdm

import evaluation
from utils import train_utils

import train


def test_auto_enables_progress_bar_for_interactive_stderr(monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    assert train_utils.progress_bar_disabled("auto") is False


def test_auto_disables_progress_bar_for_redirected_stderr(monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)

    assert train_utils.progress_bar_disabled("auto") is True


def test_explicit_progress_bar_modes_ignore_tty(monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    assert train_utils.progress_bar_disabled("off") is True

    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    assert train_utils.progress_bar_disabled("on") is False


def test_unknown_progress_bar_mode_raises_value_error():
    with pytest.raises(ValueError, match="Unknown progress bar mode"):
        train_utils.progress_bar_disabled("sometimes")


def test_disabled_tqdm_writes_no_dynamic_output():
    stream = io.StringIO()

    for _ in tqdm(
        range(3),
        disable=train_utils.progress_bar_disabled("off"),
        file=stream,
    ):
        pass

    assert stream.getvalue() == ""


def test_evaluation_progress_bar_parameter_remains_optional():
    signature = inspect.signature(evaluation.evaluation)

    assert signature.parameters["progress_bar"].default == "auto"


class _EmptyProgress:
    def __iter__(self):
        return iter(())

    def close(self):
        pass


class _EmptyDataset:
    def __len__(self):
        return 0


def test_structure_da_training_config_defaults_progress_bar_to_auto():
    config = train.StructureDATrainingConfig(
        epochs=1, steps_per_epoch=1, lr=0.001, weight_decay=0.0,
        quality_warmup_steps=None, grl_warmup_steps=None, grl_gamma=10.0,
        loss_weights=train.LossWeights(), log_step=1,
    )

    assert config.progress_bar == "auto"


def test_final_evaluation_defaults_missing_progress_bar_to_auto(monkeypatch):
    captured = []
    target_val = object()
    target_test = object()
    evaluation_domains = []

    class Model:
        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            pass

    monkeypatch.setattr(train, "prepare_data_protocol", lambda config: ({}, None))
    monkeypatch.setattr(train, "create_train_val_test_folds", lambda *args: [{}])
    monkeypatch.setattr(
        train,
        "create_evaluation_loaders",
        lambda dataset_name, *args: (
            evaluation_domains.append(dataset_name)
            or (target_val, target_test)
        ),
    )
    monkeypatch.setattr(train, "StructureDAModel", lambda **kwargs: Model())
    monkeypatch.setattr(train.torch, "load", lambda *args, **kwargs: {"state_dict": {}})
    monkeypatch.setattr(
        train,
        "evaluation",
        lambda *args, **kwargs: (
            captured.append((args[1], kwargs["progress_bar"]))
            or {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "classification_report": "",
                "confusion_matrix": np.zeros((1, 1)),
            }
        ),
    )
    monkeypatch.setattr(train, "save_results", lambda *args: None)
    monkeypatch.setattr(train, "overall_performance", lambda *args: None)
    config = SimpleNamespace(
        seed=1,
        device="cpu",
        source="source",
        target="target",
        num_folds=1,
        val_ratio=0.1,
        test_ratio=0.2,
        overall=False,
        closed_set=False,
        output_dir="outputs",
        sample_pixels_val=False,
        eval=True,
        temporal_shift=False,
        model="structure_da",
        input_dim=10,
        num_classes=1,
        with_extra=False,
        classes=["crop"],
        experiment_name="test",
        time_scale=365.0,
        tau_fast_init=0.05,
        tau_slow_init=0.2,
        tau_min=1e-4,
        delta_tau_min=1e-4,
        quality_hidden_cap=128,
        quality_eta=0.1,
        sda_hidden_dim=128,
    )

    train.main(config)

    assert evaluation_domains == ["target"]
    assert captured == [(target_test, "auto")]


def test_training_selects_checkpoint_on_source_validation_and_tests_target(
    monkeypatch,
):
    source_val = object()
    source_unused_test = object()
    target_val = object()
    target_test = object()
    source_train = object()
    target_train = object()
    evaluation_domains = []
    trainer_validation_loaders = []
    final_test_loaders = []

    class Model:
        def to(self, device):
            return self

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            pass

        def parameters(self):
            return iter(())

    class Writer:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(train, "prepare_data_protocol", lambda config: ({}, None))
    monkeypatch.setattr(train, "create_train_val_test_folds", lambda *args: [{}])

    def create_evaluation_loaders(dataset_name, *args):
        evaluation_domains.append(dataset_name)
        if dataset_name == "source":
            return source_val, source_unused_test
        return target_val, target_test

    monkeypatch.setattr(train, "create_evaluation_loaders", create_evaluation_loaders)
    monkeypatch.setattr(train, "StructureDAModel", lambda **kwargs: Model())
    monkeypatch.setattr(
        train,
        "create_structure_da_train_loaders",
        lambda *args: (source_train, target_train),
    )
    monkeypatch.setattr(
        train,
        "train_structure_da",
        lambda model, source, target, validation_loader, *args: (
            trainer_validation_loaders.append(validation_loader)
        ),
    )
    monkeypatch.setattr(train.torch, "load", lambda *args, **kwargs: {"state_dict": {}})
    monkeypatch.setattr(
        train,
        "evaluation",
        lambda model, loader, *args, **kwargs: (
            final_test_loaders.append(loader)
            or {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "classification_report": "",
                "confusion_matrix": np.zeros((1, 1)),
            }
        ),
    )
    monkeypatch.setattr(train, "save_results", lambda *args: None)
    monkeypatch.setattr(train, "overall_performance", lambda *args: None)
    monkeypatch.setitem(
        sys.modules,
        "torch.utils.tensorboard",
        SimpleNamespace(SummaryWriter=Writer),
    )
    config = SimpleNamespace(
        seed=1, device="cpu", source="source", target="target", num_folds=1,
        val_ratio=0.1, test_ratio=0.2, overall=False, closed_set=False,
        output_dir="outputs", sample_pixels_val=False, eval=False,
        model="structure_da", input_dim=10, num_classes=1, with_extra=False,
        classes=["crop"], experiment_name="test", time_scale=365.0,
        tau_fast_init=0.05, tau_slow_init=0.2, tau_min=1e-4,
        delta_tau_min=1e-4, quality_hidden_cap=128, quality_eta=0.1,
        sda_hidden_dim=128, tensorboard_log_dir="runs", epochs=1,
        steps_per_epoch=1, lr=1e-3, weight_decay=0.0,
        quality_warmup_steps=None, grl_warmup_steps=None, grl_gamma=10.0,
        lambda_qdom=1.0, lambda_qcls=1.0, lambda_div=1.0,
        lambda_sda=1.0, log_step=1, progress_bar="off",
    )

    train.main(config)

    assert evaluation_domains == ["source", "target"]
    assert trainer_validation_loaders == [source_val]
    assert target_val not in trainer_validation_loaders
    assert final_test_loaders == [target_test]
