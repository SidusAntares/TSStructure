from pathlib import Path
from unittest.mock import patch

import torch


def test_r16_and_r24_resample_inputs_but_preserve_shape_token_dimension():
    from models.structure_da.discriminative_structure import ShapeTokenGenerator

    windows = torch.randn(2, 3, 24, 5)
    r16 = ShapeTokenGenerator(5, shape_dim=32, resample_length=16).eval()
    r24 = ShapeTokenGenerator(5, shape_dim=32, resample_length=24).eval()
    with torch.no_grad():
        parts16 = r16.components(windows)
        parts24 = r24.components(windows)
        token16 = r16(windows)
        token24 = r24(windows)
    assert parts16["normalized"].shape == (2, 3, 16, 5)
    assert parts24["normalized"].shape == (2, 3, 24, 5)
    assert token16.shape == token24.shape == (2, 3, 32)


def test_timematch_test_checkpoint_is_final_epoch_last_checkpoint():
    from train import protocol_manifest, resolve_test_checkpoint

    root = Path("protocol-fixture")
    with patch.object(Path, "is_file", return_value=True):
        path, policy = resolve_test_checkpoint("timematch", root)
    assert path == root / "checkpoint_last.pt"
    assert policy == "final_epoch"
    assert protocol_manifest("timematch") == {
        "uda_model_selection": "final_epoch",
        "uda_test_checkpoint": "checkpoint_last.pt",
        "validation_best_checkpoint": "checkpoint_best.pt",
        "target_validation_used_for_test_selection": False,
    }


def test_timematch_checkpoint_roles_keep_best_separate_from_final_model():
    from timematch import save_timematch_checkpoint

    root = Path("protocol-fixture")
    model_path = root / "model.pt"
    calls = []
    packet = {"epoch": 19, "state_dict": {"weight": torch.tensor([1.0])}}
    with patch(
        "timematch.torch.save",
        side_effect=lambda value, path: calls.append((value["epoch"], Path(path))),
    ):
        save_timematch_checkpoint(
            packet, root, model_path, validation_best=True, final_epoch=True,
        )
    assert calls == [
        (19, root / "checkpoint_last.pt"),
        (19, root / "checkpoint_best.pt"),
        (19, model_path),
    ]


def test_launcher_runs_r16_then_r24_with_only_resample_length_changed():
    launcher = Path(
        "scripts/run_v2_r16_r24_4tasks_4gpu_seed1.sh"
    ).read_text(encoding="utf-8")
    assert 'run_variant "R16" 16' in launcher
    assert 'run_variant "R24" 24' in launcher
    assert launcher.index('run_variant "R16" 16') < launcher.index(
        'run_variant "R24" 24'
    )
    assert '--shape-window-scales 24' in launcher
    assert '--shape-resample-length "$resample_length"' in launcher
    assert 'outputs/v2_r16_seed1' in launcher
    assert 'outputs/v2_r24_seed1' in launcher
    assert 'test_checkpoint=checkpoint_last.pt' in launcher
    for mapping in (
        'launch_task "$GPU0" AT1 "$AT1" DK1 "$DK1"',
        'launch_task "$GPU1" FR1 "$FR1" FR2 "$FR2"',
        'launch_task "$GPU2" FR2 "$FR2" DK1 "$DK1"',
        'launch_task "$GPU3" DK1 "$DK1" AT1 "$AT1"',
    ):
        assert mapping in launcher


def test_v2_loss_path_remains_present():
    source = Path("timematch.py").read_text(encoding="utf-8")
    for required in (
        "loss_shape_target",
        "shape_alignment",
        "stats_alignment",
        "target_ins",
        "loss_shaping",
        "loss_diversity",
    ):
        assert required in source
