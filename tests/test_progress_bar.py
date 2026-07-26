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


def test_supervised_training_defaults_missing_progress_bar_to_auto(
    monkeypatch,
):
    captured = []
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    monkeypatch.setattr(
        train, "PixelSetData", lambda *args, **kwargs: _EmptyDataset()
    )
    monkeypatch.setattr(train, "create_train_loader", lambda *args, **kwargs: [])
    monkeypatch.setattr(train, "validation", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        train,
        "tqdm",
        lambda *args, **kwargs: (
            captured.append(kwargs["disable"]) or _EmptyProgress()
        ),
    )
    config = SimpleNamespace(
        lr=0.001,
        weight_decay=0.0,
        num_pixels=2,
        seq_length=3,
        max_shift_aug=5,
        shift_aug_p=1.0,
        with_shift_aug=False,
        source="source",
        target="target",
        train_on_target=False,
        data_root="data",
        classes=["crop"],
        closed_set=True,
        combine_spring_and_winter=False,
        batch_size=2,
        num_workers=0,
        focal_loss_gamma=1.0,
        epochs=1,
    )

    train.train_supervised(
        torch.nn.Linear(1, 1),
        config,
        writer=None,
        splits={"source": {"train": {1}}},
        val_loader=None,
        device="cpu",
        best_model_path="unused.pt",
    )

    assert captured == [True]


def test_final_evaluation_defaults_missing_progress_bar_to_auto(monkeypatch):
    captured = []

    class Model:
        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            pass

    monkeypatch.setattr(train, "prepare_data_protocol", lambda config: ({}, None))
    monkeypatch.setattr(train, "create_train_val_test_folds", lambda *args: [{}])
    monkeypatch.setattr(train, "create_evaluation_loaders", lambda *args: ([], []))
    monkeypatch.setattr(train, "PseLTae", lambda **kwargs: Model())
    monkeypatch.setattr(train.torch, "load", lambda *args, **kwargs: {"state_dict": {}})
    monkeypatch.setattr(
        train,
        "evaluation",
        lambda *args, **kwargs: (
            captured.append(kwargs["progress_bar"])
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
        model="pseltae",
        input_dim=10,
        num_classes=1,
        with_extra=False,
        classes=["crop"],
        experiment_name="test",
    )

    train.main(config)

    assert captured == ["auto"]
