from pathlib import Path
from unittest.mock import patch

import torch


def test_timematch_test_prefers_last_and_falls_back_to_legacy_model():
    from train import resolve_test_checkpoint

    root = Path("protocol-fixture")
    with patch.object(Path, "is_file", return_value=False):
        path, policy = resolve_test_checkpoint("timematch", root)
        assert path == root / "model.pt"
        assert policy == "final_epoch_legacy_fallback"
    with patch.object(Path, "is_file", return_value=True):
        path, policy = resolve_test_checkpoint("timematch", root)
        assert path == root / "checkpoint_last.pt"
        assert policy == "final_epoch"


def test_supervised_test_selection_remains_validation_selected():
    from train import resolve_test_checkpoint

    root = Path("protocol-fixture")
    path, policy = resolve_test_checkpoint(None, root)
    assert path == root / "model.pt"
    assert policy == "validation_selected"


def test_timematch_final_checkpoint_overwrites_model_but_not_best():
    from timematch import save_timematch_checkpoint

    root = Path("protocol-fixture")
    best_path = root / "model.pt"
    first = {"epoch": 0, "state_dict": {"weight": torch.tensor([1.])}}
    final = {"epoch": 19, "state_dict": {"weight": torch.tensor([3.])}}
    calls = []
    with patch("timematch.torch.save", side_effect=lambda packet, path: calls.append((packet["epoch"], Path(path)))):
        save_timematch_checkpoint(first, root, best_path, validation_best=True)
        save_timematch_checkpoint(final, root, best_path, validation_best=False)
        save_timematch_checkpoint(final, root, best_path, final_epoch=True)
    assert (0, root / "checkpoint_best.pt") in calls
    assert (19, root / "checkpoint_last.pt") in calls
    assert calls[-1] == (19, best_path)


def test_timematch_manifest_and_result_names_are_explicit():
    from train import protocol_manifest, result_artifact_names

    assert protocol_manifest("timematch") == {
        "uda_model_selection": "final_epoch",
        "uda_test_checkpoint": "checkpoint_last.pt",
        "validation_best_checkpoint": "checkpoint_best.pt",
        "target_validation_used_for_test_selection": False,
    }
    names = result_artifact_names("timematch", "denmark_32VNH_2017")
    assert names == {
        "metrics": "test_metrics_final_denmark_32VNH_2017.json",
        "report": "class_report_final_denmark_32VNH_2017.txt",
        "confusion": "conf_mat_final_denmark_32VNH_2017.pkl",
    }


def test_training_source_has_no_ambiguous_best_restore_message():
    source = Path("train.py").read_text(encoding="utf-8")
    assert "Restoring best model weights for testing" not in source
    assert "TEST_CHECKPOINT|method=" in source


def test_timematch_best_selection_checks_best_artifact_not_final_model():
    source = Path("timematch.py").read_text(encoding="utf-8")
    assert "or not os.path.isfile(best_model_path)" not in source
    assert source.count('checkpoint_best.pt') >= 1


def test_final_protocol_launcher_is_eval_only_and_preserves_old_results():
    launcher = Path(
        "scripts/run_structure_phase_final_protocol_4tasks_4gpu_seed1.sh"
    ).read_text(encoding="utf-8")
    assert "--eval" in launcher
    assert "--epochs" not in launcher
    assert "--steps_per_epoch" not in launcher
    assert "summarize_structure_phase_final_protocol.py" in launcher
    assert "outputs/structure_phase_final_protocol_seed1.csv" in launcher


def test_phase_final_summary_has_four_tasks_and_required_columns():
    from scripts.summarize_structure_phase_final_protocol import FIELDS, TASKS

    assert tuple(TASKS) == ("AT1_DK1", "FR1_FR2", "FR2_DK1", "DK1_AT1")
    assert FIELDS == (
        "task", "P_final_test_macro_f1", "E_final_test_macro_f1",
        "P_old_bestval_test_macro_f1", "E_old_bestval_test_macro_f1",
    )


def test_v2clean_e_rerun_preflights_phase_and_keeps_master_summary_only():
    launcher = Path(
        "scripts/run_v2clean_e_final_4tasks_4gpu_seed1.sh"
    ).read_text(encoding="utf-8")
    assert "PREFLIGHT_FAILED|missing_cli=" in launcher
    assert 'train.py --help' in launcher
    assert 'train.py timematch --help' in launcher
    assert "--shape-representation" in launcher
    assert "--shape-equivariance-weight" in launcher
    assert "RUN_ROUND=SOURCE" in launcher
    assert "RUN_ROUND=E" in launcher
    assert "RUN_ROUND=SOURCE_E" not in launcher
    assert launcher.count("> /dev/null 2>&1") == 3
    assert "ROUND_FAILED|round=V2CLEAN|logs=$V2_LOG" in launcher
    assert "ROUND_FAILED|round=E_SOURCE|logs=$E_LOG" in launcher
    assert "ROUND_FAILED|round=E_UDA|logs=$E_LOG" in launcher

