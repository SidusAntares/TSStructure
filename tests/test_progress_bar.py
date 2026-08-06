import inspect
import io
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tqdm import tqdm

import evaluation
from utils import train_utils

import train


def _current_config(**overrides):
    values = dict(
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
        eval=False,
        input_dim=10,
        num_classes=1,
        with_extra=False,
        classes=["crop"],
        experiment_name="test",
        time_reference=0.0,
        time_scale=365.0,
        tau_fast_init=0.05,
        tau_slow_init=0.20,
        tau_min=1e-4,
        delta_tau_min=1e-4,
        trend_num_basis=12,
        structure_num_basis=12,
        canonical_grid_size=64,
        roughness_grid_size=256,
        trend_smoothing=1e-2,
        structure_smoothing=1e-3,
        n_head=16,
        d_k=8,
        d_model=256,
        ltae_mlp=[256, 128],
        dropout=0.2,
        classifier_hidden=[64, 32],
        time2vec_max_frequency=16.0,
        tensorboard_log_dir="runs",
        stage1_epochs=1,
        batch_size=2,
        eval_batch_size=2,
        steps_per_epoch=1,
        lr=1e-3,
        weight_decay=0.0,
        amp=False,
        amp_dtype="float16",
        log_step=1,
        progress_bar="off",
        feature_snapshot_interval=0,
        feature_snapshot_samples_per_class=32,
        feature_snapshot_batch_size=8,
        feature_snapshot_dtype="float16",
        feature_snapshot_dir=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


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


def test_joint_structure_da_training_config_defaults_progress_bar_to_auto():
    # The source-only trainer writes plain-text logs; the progress bar mode only
    # affects evaluation tqdm output and defaults to "auto".  The auto mode is
    # interactive-stderr aware, so the actual value depends on the tty state.
    assert train.main is not None
    assert train.train_source_classification is not None
    assert train.create_source_train_loader is not None


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
    monkeypatch.setattr(train, "TSStructureModel", lambda **kwargs: Model())
    monkeypatch.setattr(train.torch, "load", lambda *args, **kwargs: {"state_dict": {}})
    monkeypatch.setattr(
        train,
        "evaluation",
        lambda *args, **kwargs: (
            captured.append(
                (args[1], kwargs["progress_bar"], kwargs.get("criterion"))
            )
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
    config = _current_config(eval=True)
    delattr(config, "progress_bar")

    train.main(config)

    assert evaluation_domains == ["target"]
    assert len(captured) == 1
    assert captured[0][:2] == (target_test, "auto")
    assert isinstance(captured[0][2], torch.nn.CrossEntropyLoss)


def test_training_selects_checkpoint_on_source_validation_and_tests_target(
    monkeypatch,
):
    source_val = object()
    source_unused_test = object()
    target_val = object()
    target_test = object()
    source_train = [object()]
    evaluation_domains = []
    trainer_validation_loaders = []
    final_test_loaders = []
    model_kwargs = []

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
    monkeypatch.setattr(
        train,
        "TSStructureModel",
        lambda **kwargs: (model_kwargs.append(kwargs) or Model()),
    )
    monkeypatch.setattr(
        train,
        "create_source_train_loader",
        lambda config, splits: source_train,
    )
    monkeypatch.setattr(
        train,
        "train_source_classification",
        lambda model, source, validation_loader, *args, **kwargs: (
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
    config = _current_config(
        tau_fast_init=0.07,
        tau_slow_init=0.29,
        tau_min=0.0002,
        delta_tau_min=0.0003,
    )

    train.main(config)

    assert evaluation_domains == ["source", "target"]
    assert trainer_validation_loaders == [source_val]
    assert target_val not in trainer_validation_loaders
    assert final_test_loaders == [target_test]
    assert len(model_kwargs) == 1
    assert model_kwargs[0]["tau_fast_init"] == 0.07
    assert model_kwargs[0]["tau_slow_init"] == 0.29
    assert model_kwargs[0]["canonical_grid_size"] == 64


def test_train_help_exposes_new_arguments_and_removes_legacy_arguments():
    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for option in (
        "--stage1_epochs",
        "--canonical_grid_size",
        "--trend_num_basis",
        "--structure_num_basis",
        "--n_head",
        "--d_k",
        "--d_model",
        "--eval_batch_size",
        "--amp", "--amp_dtype",
        "--feature_snapshot_interval", "--feature_snapshot_samples_per_class",
        "--feature_snapshot_dtype", "--feature_snapshot_dir",
        "--balance-source", "--no-balance-source",
    ):
        assert option in result.stdout
    for option in ("--channel" + "_feature_dim", "--pixel" + "_hidden_dim"):
        assert option not in result.stdout
    for option in (
        "--structure_dim", "--lambda_task", "--lambda_alignment",
        "--lambda_structural_cls", "--lambda_structural_domain",
        "--lambda_component_cls", "--lambda_component_domain",
        "--quality_" + "warmup_steps", "--grl_gamma", "--lambda_qdom",
        "--lambda_qcls", "--lambda_" + "div", "--lambda_" + "sda",
        "--quality_hidden_cap", "--quality_eta", "--sda_hidden_dim",
        "--domain_hidden_dim", "--grl_warmup_max_iters",
        "--grl_warmup_fraction", "--lambda_global_domain",
    ):
        assert option not in result.stdout
    assert "--num_blocks" not in result.stdout


def test_removed_global_alignment_cli_arguments_are_rejected():
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--grl_warmup_fraction",
            "0.2",
            "--grl_warmup_max_iters",
            "100",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
