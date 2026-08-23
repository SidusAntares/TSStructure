from pathlib import Path
import re


LAUNCHER = Path("closedset_scripts/run_reimts_mtan_closedset_12tasks_4gpu.sh")
ALIASES = ["DK1", "FR1", "FR2", "AT1"]
DOMAINS = [
    "denmark/32VNH/2017",
    "france/30TXT/2017",
    "france/31TCJ/2017",
    "austria/33UVP/2017",
]


def _script():
    assert LAUNCHER.is_file()
    return LAUNCHER.read_text()


def test_launcher_declares_fixed_four_gpu_source_mapping_and_twelve_da_plan():
    script = _script()

    assert 'DOMAIN_ALIASES=("DK1" "FR1" "FR2" "AT1")' in script
    for domain in DOMAINS:
        assert f'"{domain}"' in script
    for gpu, alias in enumerate(ALIASES):
        assert f'run_source_worker {gpu} {alias} &' in script
    assert 'if [[ "$target_alias" == "$source_alias" ]]' in script


def test_launcher_commands_freeze_formal_reimts_and_timematch_arguments():
    script = _script()

    assert 'SEED="${SEED:-1}"' in script
    assert 'PYTHON_BIN="${PYTHON_BIN:-python}"' in script
    assert '--data_root' not in script
    for option in (
        "--closed_set true", "--combine_spring_and_winter false",
        "--with_shift_aug false", "--model psereimtsmtanltae",
        "--reimts_levels 3", "--reimts_scale_factor 2",
        "--reimts_period 365", "--mtan_num_ref_points 8",
        "--mtan_latent_dim 128", "--mtan_heads 1",
        "--reimts_loss_mode patch", "--num_folds 1",
        "--progress_bar off",
    ):
        assert script.count(option) >= 2
    assert "--epochs 100" in script
    assert "--epochs 20" in script
    assert "--steps_per_epoch 500" in script
    assert "--sample_size 1" not in script
    assert "--batch_size 4" not in script


def test_launcher_paths_weights_and_task_logs_follow_experiment_layout():
    script = _script()

    assert 'EXPERIMENT_NAME="${EXPERIMENT_NAME:-reimts_mtan_closedset_12tasks_seed1}"' in script
    assert 'LOG_ROOT="logs/$EXPERIMENT_NAME"' in script
    assert 'OUTPUT_ROOT="outputs/$EXPERIMENT_NAME"' in script
    assert 'RUN_ROOT="runs/$EXPERIMENT_NAME"' in script
    assert '"$LOG_ROOT/$task.log"' in script
    assert '--output_dir "$OUTPUT_ROOT"' in script
    assert '--tensorboard_log_dir "$RUN_ROOT"' in script
    assert '--weights "$OUTPUT_ROOT/source_$source_alias"' in script
    assert 'source_checkpoint="$OUTPUT_ROOT/source_$source_alias/fold_0/model.pt"' in script


def test_launcher_failure_handling_stops_failed_source_only_and_continues_da():
    script = _script()

    assert 'source checkpoint missing' in script
    assert 'return 1' in script
    assert 'if ! run_task "$gpu_id" "$task"' in script
    assert 'failed_tasks+=("$task")' in script
    assert 'continue' in script
    assert '[RUN_START]' in script
    assert '[RUN_END]' in script
    assert 'runtime_seconds=' in script
    assert 'exit_code=' in script
    assert 'return "$exit_code"' in script
    assert 'exit "$exit_code"' not in script
    assert re.search(r'for target_alias in "\$\{DOMAIN_ALIASES\[@\]\}"', script)
