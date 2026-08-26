from pathlib import Path


LAUNCHER = Path(
    "closedset_scripts/run_reimts_mtan_openset_AT1_3tasks_3gpu.sh"
)


def _script():
    assert LAUNCHER.is_file()
    return LAUNCHER.read_text()


def test_openset_launcher_trains_two_sources_once_on_gpus_one_and_two():
    script = _script()

    assert 'SOURCE_ALIASES=("DK1" "FR1")' in script
    assert 'run_source 1 DK1 &' in script
    assert 'run_source 2 FR1 &' in script
    assert script.count('run_source 1 DK1 &') == 1
    assert script.count('run_source 2 FR1 &') == 1
    assert 'source_checkpoint "$source_alias"' in script
    assert 'source_DK1/fold_0/model.pt' in script
    assert 'source_FR1/fold_0/model.pt' in script


def test_openset_launcher_runs_exact_original_five_tasks_on_three_gpus():
    script = _script()

    expected_workers = (
        'run_da_worker 1 "DK1:FR1" "FR1:DK1" &',
        'run_da_worker 2 "DK1:FR2" "FR1:FR2" &',
        'run_da_worker 3 "DK1:AT1" &',
    )
    for worker in expected_workers:
        assert worker in script

    assert 'DK1:FR1' in script
    assert 'DK1:FR2' in script
    assert 'DK1:AT1' in script
    assert 'FR1:DK1' in script
    assert 'FR1:FR2' in script
    assert 'FR1:AT1' not in script
    assert 'FR2:' not in script
    assert 'AT1:' not in script


def test_openset_launcher_preserves_original_timematch_protocol_and_reimts_setup():
    script = _script()

    assert script.count("--closed_set false") == 2
    assert "--closed_set true" not in script
    assert "--data_root" not in script
    for option in (
        "--combine_spring_and_winter false",
        "--with_shift_aug false",
        "--model psereimtsmtanltae",
        "--reimts_levels 3",
        "--reimts_scale_factor 2",
        "--reimts_period 365",
        "--mtan_num_ref_points 8",
        "--mtan_latent_dim 128",
        "--mtan_heads 1",
        "--reimts_loss_mode sample",
        "--num_folds 1",
        "--progress_bar off",
    ):
        assert script.count(option) >= 2
    assert "--epochs 100" in script
    assert "--epochs 20" in script
    assert "--steps_per_epoch 500" in script
    assert "--sample_size 1" not in script
    assert "--batch_size 4" not in script


def test_openset_launcher_reuses_matching_source_checkpoint_and_task_logs():
    script = _script()

    assert (
        'EXPERIMENT_NAME="${EXPERIMENT_NAME:-'
        'reimts_mtan_openset_timematch5_seed1}"'
    ) in script
    assert 'LOG_ROOT="logs/$EXPERIMENT_NAME"' in script
    assert 'OUTPUT_ROOT="outputs/$EXPERIMENT_NAME"' in script
    assert 'RUN_ROOT="runs/$EXPERIMENT_NAME"' in script
    assert '"$LOG_ROOT/$task.log"' in script
    assert '--weights "$OUTPUT_ROOT/source_$source_alias"' in script
    assert 'local task="source_$source_alias"' in script
    assert 'local task="${source_alias}_to_${target_alias}"' in script


def test_openset_launcher_blocks_all_da_if_a_source_fails_and_isolates_da_failures():
    script = _script()

    assert "one or more source runs failed; DA tasks were not started" in script
    assert "source checkpoint missing" in script
    assert 'wait "$source_pid1"' in script
    assert 'wait "$source_pid2"' in script
    assert 'wait "$da_pid1"' in script
    assert 'wait "$da_pid2"' in script
    assert 'wait "$da_pid3"' in script
    assert 'echo "$task" > "$STATE_ROOT/$task.failed"' in script
    assert "[RUN_START]" in script
    assert "[RUN_END]" in script
    assert "runtime_seconds=" in script
