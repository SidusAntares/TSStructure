from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


def test_unified_evaluation_script_is_directly_executable():
    result = subprocess.run(
        [sys.executable, "analysis/summarize_reliable_composition_ab.py", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "recheck" in result.stdout


def test_formal_train_test_explicitly_uses_zero_shift():
    source = Path("train.py").read_text(encoding="utf-8")
    test_call = source[source.index("test_metrics = evaluation("):]
    test_call = test_call[:test_call.index(")\n\n        print")]
    assert "mode='test'" in test_call
    assert "temporal_shift=0" in test_call


def test_fixed_test_macro_f1_counts_zero_support_protocol_classes():
    from evaluation import classification_metrics

    result = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 0]),
        class_names=("a", "b", "zero-support"),
        mode="test",
    )
    # F1(a)=.8, F1(b)=2/3, F1(zero-support)=0.
    assert result["macro_f1"] == pytest.approx((.8 + 2. / 3.) / 3.)
    assert result["support"] == [2, 2, 0]


def test_validation_macro_f1_keeps_historical_observed_class_semantics():
    from evaluation import classification_metrics

    result = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 0]),
        class_names=("a", "b", "zero-support"),
        mode="val",
    )
    assert result["macro_f1"] == pytest.approx((.8 + 2. / 3.) / 2.)


def test_unified_checkpoint_evaluation_forces_zero_shift(monkeypatch):
    from analysis import summarize_reliable_composition_ab as summary

    captured = {}

    def fake_evaluation(model, loader, device, classes, **kwargs):
        captured.update(kwargs)
        return {
            "macro_f1": .5,
            "accuracy": .5,
            "confusion_matrix": np.eye(2, dtype=int),
            "per_class_precision": np.array([.5, .5]),
            "per_class_recall": np.array([.5, .5]),
            "per_class_f1": np.array([.5, .5]),
            "support": np.array([1, 1]),
        }

    monkeypatch.setattr(summary, "official_evaluation", fake_evaluation)
    result = summary.evaluate_model_on_official_test(
        object(), object(), SimpleNamespace(classes=["a", "b"]), "cpu",
    )
    assert captured["mode"] == "test"
    assert captured["temporal_shift"] == 0
    assert result["inference_shift"] == 0


def test_official_test_loader_reuses_train_split_and_dataset_factory(monkeypatch):
    from analysis import summarize_reliable_composition_ab as summary

    config = SimpleNamespace(
        seed=7, source="source", target="target", num_folds=1,
        val_ratio=.1, test_ratio=.2, sample_pixels_val=False,
        data_root="old", batch_size=128, classes=["a", "b"],
    )
    split = {"target": {"test": {11, 12}}}
    monkeypatch.setattr(summary, "official_prepare_data_protocol", lambda cfg: ({"source": 3, "target": 4}, None))
    split_factory = Mock(return_value=[split])
    monkeypatch.setattr(summary, "official_create_splits", split_factory)
    loader_factory = Mock(return_value=("val-loader", "test-loader"))
    monkeypatch.setattr(summary, "official_create_evaluation_loaders", loader_factory)

    loader = summary.build_official_test_loader(config, "/dataset", batch_size=64)
    assert loader == "test-loader"
    assert config.data_root == "/dataset"
    assert config.batch_size == 64
    split_factory.assert_called_once()
    loader_factory.assert_called_once_with(
        "target", split, config, False,
    )


def test_best_and_final_roles_are_resolved_without_test_selection(tmp_path):
    from analysis.summarize_reliable_composition_ab import checkpoint_roles

    fold = tmp_path / "fold_0"
    fold.mkdir()
    (fold / "checkpoint_best.pt").touch()
    (fold / "checkpoint_last.pt").touch()
    roles = checkpoint_roles(fold)
    assert roles == {
        "best": fold / "checkpoint_best.pt",
        "final": fold / "checkpoint_last.pt",
    }
    assert {path.name for path in roles.values()} == {
        "checkpoint_best.pt", "checkpoint_last.pt",
    }


def test_verified_confusion_requires_final_protocol_manifest(tmp_path):
    from analysis.summarize_reliable_composition_ab import verified_final_artifacts

    fold = tmp_path / "fold_0"
    fold.mkdir()
    confusion = fold / "conf_mat_final_target.pkl"
    metrics = fold / "test_metrics_final_target.json"
    confusion.touch(); metrics.touch()
    assert verified_final_artifacts(fold, "target") is None
    (fold / "manifest.json").write_text(
        '{"uda_test_checkpoint":"checkpoint_last.pt",'
        '"target_validation_used_for_test_selection":false}',
        encoding="utf-8",
    )
    assert verified_final_artifacts(fold, "target") == (metrics, confusion)
