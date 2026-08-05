from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
COMMON = SCRIPTS / "structure_da_v3_common.sh"
ENV_CHECK = SCRIPTS / "check_server_env.sh"
SMOKE = SCRIPTS / "smoke_structure_da_v3.sh"
SMOKE_CHECKER = SCRIPTS / "check_structure_da_smoke.py"
DIAGNOSTIC = SCRIPTS / "run_structure_da_diagnostic_at1_dk1.sh"
DIAGNOSTIC_ANALYZER = SCRIPTS / "analyze_structure_da_diagnostic.py"
PILOT4 = SCRIPTS / "run_structure_da_pilot4_4gpu.sh"
PILOT4_ANALYZER = SCRIPTS / "analyze_structure_da_pilot4.py"
FORMAL = SCRIPTS / "run_structure_da_12tasks_4gpu_3seeds.sh"
README = SCRIPTS / "README_structure_da_v3.md"
SHELL_SCRIPTS = [COMMON, ENV_CHECK, SMOKE, DIAGNOSTIC, PILOT4, FORMAL]
EXPECTED_VERSION = "structure_da_v3_compact_snapshots_no_fused_alignment_v1"
OBSOLETE_FLAGS = {
    "--structure_dim",
    "--lambda_task",
    "--lambda_alignment",
    "--lambda_structural_cls",
    "--lambda_structural_domain",
    "--lambda_component_cls",
    "--lambda_component_domain",
    "--num_blocks",
}


def _bash_executable() -> str:
    for candidate in (
        Path("D:/Git/bin/bash.exe"),
        Path("C:/Program Files/Git/bin/bash.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("bash") or "bash"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _training_flags(source: str) -> set[str]:
    return set(re.findall(r"(?m)^\s*(--[a-zA-Z0-9_-]+)(?:\s|$)", source))


def _train_help_flags() -> set[str]:
    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(re.findall(r"--[a-zA-Z0-9_-]+", result.stdout))


def test_all_v3_server_artifacts_exist() -> None:
    for path in [
        *SHELL_SCRIPTS,
        SMOKE_CHECKER,
        DIAGNOSTIC_ANALYZER,
        PILOT4_ANALYZER,
        README,
    ]:
        assert path.is_file(), path


@pytest.mark.parametrize("script", SHELL_SCRIPTS)
def test_shell_scripts_are_strict_offline_and_git_free(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in source
    forbidden = re.compile(
        r"(?im)^\s*(?:git\s+|curl(?:\s|$)|wget(?:\s|$)|"
        r"pip\s+install|conda\s+install|apt(?:-get)?(?:\s|$))"
    )
    assert forbidden.search(source) is None
    assert "http://" not in source
    assert "https://" not in source
    assert not (OBSOLETE_FLAGS & _training_flags(source))


def test_common_configuration_uses_current_v3_cli_and_fixed_version() -> None:
    source = COMMON.read_text(encoding="utf-8")
    assert EXPECTED_VERSION in source
    for name in (
        "PROJECT_ROOT", "DATA_ROOT", "OUTPUT_ROOT", "LOG_ROOT",
        "PYTHON_BIN", "NUM_WORKERS", "CODE_VERSION", "V3_COMMON_ARGS",
    ):
        assert name in source
    for function in (
        "require_command", "require_file", "require_directory",
        "make_run_directory", "run_training",
    ):
        assert re.search(rf"(?m)^{function}\(\)", source)
    assert "CONDA_ENV" not in source
    assert "activate_environment" not in source
    assert "conda activate" not in source
    for required in (
        "--balance-source", "--amp true", "--time_coordinate_mode",
        "--warp_num_candidates", "--shape_dim", "--lambda_cls",
        "--lambda_quality",
        "--lambda_target_semantic", "--progress_bar auto",
    ):
        assert required in source
    for removed in (
        "--domain_hidden_dim", "--grl_warmup_max_iters",
        "--grl_warmup_fraction", "--lambda_global_domain",
    ):
        assert removed not in source
    assert "CMD=(" in source
    assert "printf '%q '" in source
    assert 'DATA_ROOT="${DATA_ROOT:-}"' in source
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"' in source
    assert 'LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs}"' in source
    assert "${PROJECT_ROOT}/data" not in source
    assert "${PROJECT_ROOT}/outputs/structure_da_v3" not in source
    assert "--progress_bar off" not in source


def test_data_root_is_only_appended_when_explicitly_set() -> None:
    source = COMMON.read_text(encoding="utf-8")
    command_block = source.split("CMD=(", 1)[1].split(")", 1)[0]
    assert "--data_root" not in command_block
    assert re.search(
        r'if \[\[ -n "\$\{DATA_ROOT\}" \]\]; then\s*'
        r'CMD\+=\(--data_root "\$\{DATA_ROOT\}"\)\s*fi',
        source,
    )
    assert '> "${task_log_file}" 2>&1' in source
    for legacy in ("stderr.log", "command.txt", "train.pid", "exit_code.txt"):
        assert legacy not in source
    for marker in ("RUN_START|", "RUN_COMMAND|", "RUN_END|"):
        assert marker in source


def test_every_common_training_flag_exists_in_train_help() -> None:
    common_flags = _training_flags(COMMON.read_text(encoding="utf-8"))
    assert common_flags
    assert common_flags <= _train_help_flags()


def test_environment_check_covers_data_dependencies_cuda_and_storage() -> None:
    source = ENV_CHECK.read_text(encoding="utf-8")
    for fragment in (
        "torch", "numpy", "sklearn", "train.py --help", "cuda",
        "austria/33UVP/2017", "denmark/32VNH/2017",
        "france/30TXT/2017", "france/31TCJ/2017",
        "SERVER_ENV_CHECK=PASS", "SERVER_ENV_CHECK=FAIL", "df -P",
    ):
        assert fragment in source
    assert "if ! run_training" not in source


def test_smoke_is_short_fixed_and_checked() -> None:
    source = SMOKE.read_text(encoding="utf-8")
    for fragment in (
        'SOURCE_DOMAIN="${SOURCE_DOMAIN:-austria/33UVP/2017}"',
        'TARGET_DOMAIN="${TARGET_DOMAIN:-denmark/32VNH/2017}"',
        'SEED="${SEED:-1}"', 'CUDA_DEVICE="${CUDA_DEVICE:-0}"',
        'SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"',
        'SMOKE_STEPS_PER_EPOCH="${SMOKE_STEPS_PER_EPOCH:-2}"',
        'RUN_OUTPUT_DIRECTORY="${OUTPUT_ROOT}/smoke_structure_da_v3"',
        'SMOKE_LOG_ROOT="${LOG_ROOT}/smoke_structure_da_v3"',
        'SMOKE_TRAIN_LOG_DIRECTORY="${SMOKE_LOG_ROOT}/train_logs"',
        'SMOKE_SNAPSHOT_DIRECTORY="${SMOKE_LOG_ROOT}/snapshots"',
        'SMOKE_LOG_FILE="${SMOKE_TRAIN_LOG_DIRECTORY}/smoke.log"',
        "check_structure_da_smoke.py", "--log-directory",
        "--feature_snapshot_interval 0",
        "SMOKE_RESULT|status=SUCCESS", "SMOKE_RESULT|status=FAILED",
    ):
        assert fragment in source
    assert "SMOKE_SUCCESS" not in source
    assert "SMOKE_FAILED" not in source


def test_smoke_checker_supports_separate_output_and_log_directories() -> None:
    source = SMOKE_CHECKER.read_text(encoding="utf-8")
    assert 'parser.add_argument("--log-directory"' in source
    assert "log_directory / \"smoke.log\"" in source
    assert "run_directory / \"fold_0\" / \"model.pt\"" in source


EXPECTED_FORMAL_TASKS = (
    "AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017",
    "AT1|austria/33UVP/2017|FR1|france/31TCJ/2017",
    "AT1|austria/33UVP/2017|FR2|france/30TXT/2017",
    "DK1|denmark/32VNH/2017|AT1|austria/33UVP/2017",
    "DK1|denmark/32VNH/2017|FR1|france/31TCJ/2017",
    "DK1|denmark/32VNH/2017|FR2|france/30TXT/2017",
    "FR1|france/31TCJ/2017|AT1|austria/33UVP/2017",
    "FR1|france/31TCJ/2017|DK1|denmark/32VNH/2017",
    "FR1|france/31TCJ/2017|FR2|france/30TXT/2017",
    "FR2|france/30TXT/2017|AT1|austria/33UVP/2017",
    "FR2|france/30TXT/2017|DK1|denmark/32VNH/2017",
    "FR2|france/30TXT/2017|FR1|france/31TCJ/2017",
)


def test_formal_launcher_defines_exactly_12_tasks_and_36_runs() -> None:
    source = FORMAL.read_text(encoding="utf-8")
    for task in EXPECTED_FORMAL_TASKS:
        assert f'"{task}"' in source
    assert "SEEDS=(1 2 3)" in source
    assert "EXPECTED_RUNS=36" in source
    assert "--epochs" not in source
    assert "--steps_per_epoch" not in source


def test_formal_launcher_requires_run_group_and_uses_four_sequential_workers() -> None:
    source = FORMAL.read_text(encoding="utf-8")
    assert 'RUN_GROUP="${RUN_GROUP:-}"' in source
    assert "RUN_GROUP must contain only letters, digits, underscores, and hyphens" in source
    for gpu in range(4):
        assert f'GPU{gpu}="${{GPU{gpu}:-{gpu}}}"' in source
        assert f'run_worker {gpu} "${{GPU{gpu}}}" &' in source
    assert "job_index += 4" in source
    assert 'wait "${worker_pid}"' in source
    assert 'CUDA_VISIBLE_DEVICES="${physical_gpu}"' in COMMON.read_text(encoding="utf-8")


def test_formal_launcher_separates_logs_outputs_and_status_files() -> None:
    source = FORMAL.read_text(encoding="utf-8")
    for fragment in (
        'GROUP_LOG_ROOT="${LOG_ROOT}/${RUN_GROUP}"',
        'GROUP_TRAIN_LOG_DIRECTORY="${GROUP_LOG_ROOT}/train_logs"',
        'GROUP_SNAPSHOT_DIRECTORY="${GROUP_LOG_ROOT}/snapshots"',
        'GROUP_OUTPUT_DIRECTORY="${OUTPUT_ROOT}/${RUN_GROUP}"',
        "launcher.pid", "manifest.tsv", "completed.tsv", "failed.tsv",
        "experiment_status.tsv",
        'TASK_LOG_FILE="${GROUP_TRAIN_LOG_DIRECTORY}/${run_name}.log"',
        'SNAPSHOT_DIRECTORY="${GROUP_SNAPSHOT_DIRECTORY}/${run_name}"',
        'RUN_OUTPUT_DIRECTORY="${GROUP_OUTPUT_DIRECTORY}/${run_name}"',
        "--feature_snapshot_interval 25",
        "--feature_snapshot_samples_per_class 32",
        "--feature_snapshot_dtype float16",
    ):
        assert fragment in source
    for legacy in (
        '"${RUN_LOG_DIRECTORY}/TASK_DONE"',
        '"${RUN_LOG_DIRECTORY}/TASK_FAILED"',
        "EXPERIMENT_DONE", "EXPERIMENT_FAILED", "stderr.log", "command.txt",
        "train.pid", "exit_code.txt",
    ):
        assert legacy not in source
    assert "run_structure_da_pilot4_4gpu.sh" not in source


def test_readme_has_standard_nohup_workflow_without_pilot_gate() -> None:
    source = README.read_text(encoding="utf-8")
    for fragment in (
        "logs/smoke_structure_da_v3/train_logs/nohup.log",
        "logs/smoke_structure_da_v3/train_logs/smoke.log",
        'RUN_GROUP="structure_da_v3_12tasks_3seeds_$(date +%Y%m%d_%H%M%S)"',
        "run_structure_da_12tasks_4gpu_3seeds.sh",
        'logs/${RUN_GROUP}/train_logs/AT1_DK1_seed1.log',
        "Killing only the launcher PID may leave already-started training children",
    ):
        assert fragment in source
    assert "PASS_FOR_PILOT4" not in source
    assert "READY_FOR_FULL_EXPERIMENT" not in source


def test_diagnostic_is_fixed_task_with_formal_common_parameters() -> None:
    source = DIAGNOSTIC.read_text(encoding="utf-8")
    for fragment in (
        'SOURCE_DOMAIN="austria/33UVP/2017"',
        'TARGET_DOMAIN="denmark/32VNH/2017"', 'SEED="1"',
        'DIAGNOSTIC_EPOCHS="${DIAGNOSTIC_EPOCHS:-25}"',
        "这是诊断 pilot，不是最终报告结果", "run_training",
        "analyze_structure_da_diagnostic.py",
    ):
        assert fragment in source


def test_pilot4_has_exact_tasks_gpu_overrides_and_independent_outputs() -> None:
    source = PILOT4.read_text(encoding="utf-8")
    for fragment in (
        'GPU0="${GPU0:-0}"', 'GPU1="${GPU1:-1}"',
        'GPU2="${GPU2:-2}"', 'GPU3="${GPU3:-3}"',
        '"AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|1|${GPU0}"',
        '"AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|2|${GPU1}"',
        '"AT1|austria/33UVP/2017|DK1|denmark/32VNH/2017|3|${GPU2}"',
        '"DK1|denmark/32VNH/2017|FR2|france/30TXT/2017|1|${GPU3}"',
        'CUDA_VISIBLE_DEVICES="${physical_gpu}"', "run_training",
        'PIDS+=("$!")', 'wait "${pid}"', "pilot4_manifest.json",
        "pilot4_summary.json", "pilot4_summary.md",
    ):
        assert fragment in source
    assert "FR1" not in source


def test_paths_are_quoted_and_existing_runs_require_overwrite() -> None:
    common = COMMON.read_text(encoding="utf-8")
    for fragment in (
        'cd "${PROJECT_ROOT}"', '"${CMD[@]}"', 'OVERWRITE',
        '"${run_directory}"', '"${OUTPUT_ROOT}"',
    ):
        assert fragment in common


def test_diagnostic_analyzer_uses_only_pseudo_target_class_labels(tmp_path: Path) -> None:
    analyzer = _load_module(DIAGNOSTIC_ANALYZER, "diagnostic_analyzer_test")
    log = tmp_path / "train.log"
    log.write_text(
        "PHASE_CLASS_EPOCH|epoch=1|split=target|"
        "label_source=target_true|class_id=0|sample_count=2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target_geometry_pseudo"):
        analyzer.analyze_run(tmp_path)


def test_diagnostic_analyzer_parses_synthetic_log(tmp_path: Path) -> None:
    analyzer = _load_module(DIAGNOSTIC_ANALYZER, "diagnostic_analyzer_synthetic")
    (tmp_path / "fold_0").mkdir()
    (tmp_path / "fold_0" / "model.pt").write_bytes(b"checkpoint")
    (tmp_path / "fold_0" / "test_metrics_denmark_32VNH_2017.json").write_text(
        json.dumps({"macro_f1": 0.42, "accuracy": 0.5}), encoding="utf-8"
    )
    (tmp_path / "train.log").write_text(
        "TRAIN_EPOCH|epoch=1/2|task_step_success_rate=1|"
        "task_step_skip_rate=0|source_state_update_rate=1\n"
        "Validation result: loss=1.2, acc=0.40, f1=0.35\n"
        "PHASE_EPOCH|epoch=1|domain=source|valid_identity_rate=0.4|"
        "valid_nonidentity_rate=0.5|failure_rate=0.1|"
        "candidate_collapse_rate=0.2|candidate_trainable_rate=0.8|"
        "candidate_acceptable_rate=0.7|T_phase_magnitude_mean=0.03|"
        "T_phase_magnitude_p95=0.08|S_shape_valid_rate=0.6\n"
        "PHASE_CLASS_EPOCH|epoch=1|split=source|label_source=source_true|"
        "class_id=0|sample_count=8|failure_rate=0.1\n"
        "PHASE_CLASS_EPOCH|epoch=1|split=target|"
        "label_source=target_geometry_pseudo|class_id=0|sample_count=4|"
        "failure_rate=0.2\n",
        encoding="utf-8",
    )
    summary = analyzer.analyze_run(tmp_path)
    assert summary["status"] in {"PASS_FOR_PILOT4", "REVIEW_REQUIRED"}
    assert summary["classification"]["target_macro_f1"] == pytest.approx(0.42)
    assert summary["phase_by_class"]["target"]["0"]["sample_count"] == 4
    assert summary["x1"]["relative_identity_gain"] == "NOT_AVAILABLE_IN_CURRENT_LOGS"


def test_pilot4_analyzer_reports_missing_x1_metrics(tmp_path: Path) -> None:
    analyzer = _load_module(PILOT4_ANALYZER, "pilot4_analyzer_test")
    summary = analyzer.summarize_pilot4(
        tmp_path,
        [
            {
                "run_name": "AT1_DK1_seed1",
                "source": "AT1", "target": "DK1", "seed": 1,
                "exit_code": 0,
                "diagnostic": {
                    "status": "PASS_FOR_PILOT4",
                    "classification": {"source_validation_best_macro_f1": 0.5,
                                       "target_macro_f1": 0.4},
                    "phase": {}, "optimization": {}, "quality": {},
                    "x1": {},
                },
            }
        ],
    )
    assert summary["x1"]["relative_identity_gain"] == "NOT_AVAILABLE_IN_CURRENT_LOGS"
    assert summary["readiness"] != "READY_FOR_FULL_EXPERIMENT"


@pytest.mark.parametrize("script", SHELL_SCRIPTS)
def test_shell_scripts_parse_with_bash_when_available(script: Path) -> None:
    relative = script.relative_to(REPO_ROOT).as_posix()
    try:
        result = subprocess.run(
            [_bash_executable(), "-n", relative], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, PermissionError) as error:
        pytest.skip(f"bash unavailable in this environment: {error}")
    combined = result.stdout + result.stderr
    normalized = combined.replace("\x00", "")
    if "CreateInstance" in normalized and "E_ACCESSDENIED" in normalized:
        pytest.skip("Windows/WSL CreateInstance E_ACCESSDENIED")
    assert result.returncode == 0, combined
