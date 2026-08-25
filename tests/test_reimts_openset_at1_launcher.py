from pathlib import Path


LAUNCHER = Path(
    "closedset_scripts/run_reimts_mtan_openset_AT1_3tasks_3gpu.sh"
)


def _script():
    assert LAUNCHER.is_file()
    return LAUNCHER.read_text()


def test_openset_launcher_trains_at1_source_once_then_maps_three_da_gpus():
    script = _script()

    assert 'SOURCE_ALIAS="AT1"' in script
    assert 'SOURCE="austria/33UVP/2017"' in script
    assert 'run_source 1' in script
    assert script.count('run_source 1') == 1
    assert 'run_da 1 DK1 "denmark/32VNH/2017" &' in script
    assert 'run_da 2 FR1 "france/30TXT/2017" &' in script
    assert 'run_da 3 FR2 "france/31TCJ/2017" &' in script
    assert 'source_checkpoint="$OUTPUT_ROOT/source_AT1/fold_0/model.pt"' in script


def test_openset_launcher_changes_only_protocol_flag_from_formal_reimts_setup():
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


def test_openset_launcher_uses_isolated_experiment_logs_outputs_and_weights():
    script = _script()

    assert 'EXPERIMENT_NAME="${EXPERIMENT_NAME:-reimts_mtan_openset_AT1_3tasks_seed1}"' in script
    assert 'LOG_ROOT="logs/$EXPERIMENT_NAME"' in script
    assert 'OUTPUT_ROOT="outputs/$EXPERIMENT_NAME"' in script
    assert 'RUN_ROOT="runs/$EXPERIMENT_NAME"' in script
    assert '"$LOG_ROOT/$task.log"' in script
    assert '--weights "$OUTPUT_ROOT/source_AT1"' in script
    assert 'local task="source_AT1"' in script
    assert 'local task="AT1_to_${target_alias}"' in script


def test_openset_launcher_blocks_on_source_failure_and_isolates_da_failures():
    script = _script()

    assert 'source checkpoint missing' in script
    assert 'exit 1' in script
    assert 'wait "$pid1"' in script
    assert 'wait "$pid2"' in script
    assert 'wait "$pid3"' in script
    assert 'failed_tasks+=("$task")' in script
    assert "[RUN_START]" in script
    assert "[RUN_END]" in script
    assert "runtime_seconds=" in script
