from pathlib import Path


SCRIPT = Path("scripts/run_timematch_original_4tasks_4gpu_seed1.sh")


def test_launcher_assigns_one_original_timematch_task_per_gpu():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'run_task "$GPU0" AT1 "$AT1" DK1 "$DK1"' in text
    assert 'run_task "$GPU1" FR1 "$FR1" FR2 "$FR2"' in text
    assert 'run_task "$GPU2" FR2 "$FR2" DK1 "$DK1"' in text
    assert 'run_task "$GPU3" DK1 "$DK1" AT1 "$AT1"' in text


def test_launcher_trains_raw_pseltae_source_then_standard_timematch():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--model pseltae" in text
    assert "--epochs 100" in text
    assert "timematch --weights" in text
    assert "--epochs 20 --steps_per_epoch 500" in text
    assert "--pseudo_threshold 0.9" in text
    assert "--output_student true" in text
    assert "psestructureprotoltae" not in text
    assert "fourier_num_modes" not in text


def test_launcher_verifies_final_student_and_runs_attention_after_all_workers():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "checkpoint_last.pt" in text
    assert "validate_final_checkpoint" in text
    assert 'wait "$PID0"' in text
    assert 'wait "$PID3"' in text
    wait_position = text.index('wait "$PID3"')
    visualization_position = text.index("run_timematch_ltae_attention_vis.sh")
    assert visualization_position > wait_position

