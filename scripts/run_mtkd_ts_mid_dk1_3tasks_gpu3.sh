#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mtkd_ts_mid_dk1_3tasks}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-runs/mtkd_ts_mid_dk1_3tasks}"
LOG_ROOT="${LOG_ROOT:-logs/mtkd_ts_mid_dk1_3tasks}"
DRY_RUN="${DRY_RUN:-0}"
SEED=1
NUM_FOLDS=1

GPU_IDS=(3)
SOURCE_DOMAINS=(DK1)

declare -A DOMAIN_PATHS=(
    [AT1]="austria/33UVP/2017"
    [DK1]="denmark/32VNH/2017"
    [FR1]="france/30TXT/2017"
    [FR2]="france/31TCJ/2017"
)

set_target_domains() {
    local source="$1"
    case "$source" in
        DK1) TARGET_DOMAINS=(AT1 FR1 FR2) ;;
        *)
            printf '[ERROR] Unsupported source domain: %s\n' "$source" >&2
            return 1
            ;;
    esac
}

source_experiment_name() {
    printf 'mtkd_ts_mid_%s_source_seed%s' "$1" "$SEED"
}

adaptation_experiment_name() {
    printf 'mtkd_ts_mid_%s_%s_seed%s' "$1" "$2" "$SEED"
}

build_training_command() {
    local stage="$1"
    local source="$2"
    local target="$3"
    local experiment="$4"
    local weights_path="${5:-}"

    TRAINING_COMMAND=(
        python -u train.py
        --data_root "$DATA_ROOT"
        --source "${DOMAIN_PATHS[$source]}"
        --target "${DOMAIN_PATHS[$target]}"
        --model psemtkdmidltae
        --seed "$SEED"
        --num_folds "$NUM_FOLDS"
        --device cuda
        --progress_bar off
        --closed_set true
        --combine_spring_and_winter false
        --with_shift_aug false
        --output_dir "$OUTPUT_ROOT"
        --tensorboard_log_dir "$TENSORBOARD_ROOT"
        --experiment_name "$experiment"
    )

    if [[ "$stage" == "timematch" ]]; then
        TRAINING_COMMAND+=(timematch --weights "$weights_path")
    fi
}

print_task_boundary() {
    local title="$1"
    local source="$2"
    local target="$3"
    local gpu_id="$4"
    local status="${5:-}"

    printf '\n\n################################################################\n'
    printf '%s\n' "$title"
    printf 'source : %s\n' "$source"
    printf 'target : %s\n' "$target"
    printf 'gpu    : %s\n' "$gpu_id"
    printf 'seed   : %s\n' "$SEED"
    if [[ -n "$status" ]]; then
        printf 'status : %s\n' "$status"
    fi
    printf '################################################################\n\n\n'
}

run_training_task() {
    local stage="$1"
    local gpu_id="$2"
    local source="$3"
    local target="$4"
    local run_root="$5"
    local source_weights="${6:-}"
    local experiment task_log exit_code

    if [[ "$stage" == "source" ]]; then
        experiment="$(source_experiment_name "$source")"
        task_log="$run_root/source_${source}.log"
    else
        experiment="$(adaptation_experiment_name "$source" "$target")"
        task_log="$run_root/${source}_to_${target}.log"
    fi

    if [[ -e "$OUTPUT_ROOT/$experiment" ]]; then
        print_task_boundary "FAILED TASK" "$source" "$target" "$gpu_id" \
            "OUTPUT ALREADY EXISTS: $OUTPUT_ROOT/$experiment"
        return 1
    fi

    build_training_command "$stage" "$source" "$target" "$experiment" "$source_weights"
    print_task_boundary "START TASK" "$source" "$target" "$gpu_id"
    printf 'experiment log : %s\n' "$task_log"
    printf 'output path    : %s/%s\n\n' "$OUTPUT_ROOT" "$experiment"

    if CUDA_VISIBLE_DEVICES="$gpu_id" "${TRAINING_COMMAND[@]}" > "$task_log" 2>&1; then
        print_task_boundary "FINISHED TASK" "$source" "$target" "$gpu_id" "SUCCESS"
        return 0
    else
        exit_code=$?
        print_task_boundary "FAILED TASK" "$source" "$target" "$gpu_id" \
            "FAILED (exit_code=$exit_code; log=$task_log)"
        return "$exit_code"
    fi
}

run_worker() {
    local gpu_id="$1"
    local source="$2"
    local run_root="$3"
    local source_experiment source_weights source_checkpoint target failures=0

    set_target_domains "$source" || return 1
    source_experiment="$(source_experiment_name "$source")"
    source_weights="$OUTPUT_ROOT/$source_experiment"
    source_checkpoint="$source_weights/fold_0/model.pt"

    printf '[WORKER] GPU %s handles source %s and targets: %s\n' \
        "$gpu_id" "$source" "${TARGET_DOMAINS[*]}"

    if ! run_training_task source "$gpu_id" "$source" "$source" "$run_root"; then
        printf '[ERROR] Source training failed; stopping GPU %s / %s worker.\n' \
            "$gpu_id" "$source" >&2
        return 1
    fi

    if [[ ! -f "$source_checkpoint" ]]; then
        printf '[ERROR] Source checkpoint does not exist: %s\n' \
            "$source_checkpoint" >&2
        return 1
    fi

    printf '[WORKER] Source checkpoint verified: %s\n' "$source_checkpoint"
    for target in "${TARGET_DOMAINS[@]}"; do
        if ! run_training_task timematch "$gpu_id" "$source" "$target" \
            "$run_root" "$source_weights"; then
            failures=$((failures + 1))
            printf '[WARN] Continuing with remaining targets for GPU %s / %s.\n' \
                "$gpu_id" "$source" >&2
        fi
    done

    if [[ "$failures" -ne 0 ]]; then
        printf '[WORKER] GPU %s / %s finished with %s failed DA task(s).\n' \
            "$gpu_id" "$source" "$failures" >&2
        return 1
    fi

    printf '[WORKER] GPU %s / %s completed all three DA tasks.\n' \
        "$gpu_id" "$source"
}

check_fresh_outputs() {
    local source target experiment

    for source in "${SOURCE_DOMAINS[@]}"; do
        experiment="$(source_experiment_name "$source")"
        if [[ -e "$OUTPUT_ROOT/$experiment" ]]; then
            printf '[ERROR] Refusing to overwrite existing output: %s/%s\n' \
                "$OUTPUT_ROOT" "$experiment" >&2
            return 1
        fi

        set_target_domains "$source" || return 1
        for target in "${TARGET_DOMAINS[@]}"; do
            experiment="$(adaptation_experiment_name "$source" "$target")"
            if [[ -e "$OUTPUT_ROOT/$experiment" ]]; then
                printf '[ERROR] Refusing to overwrite existing output: %s/%s\n' \
                    "$OUTPUT_ROOT" "$experiment" >&2
                return 1
            fi
        done
    done
}

print_dry_run_command() {
    local stage="$1"
    local gpu_id="$2"
    local source="$3"
    local target="$4"
    local run_root="$5"
    local source_weights="${6:-}"
    local experiment task_log

    if [[ "$stage" == "source" ]]; then
        experiment="$(source_experiment_name "$source")"
        task_log="$run_root/source_${source}.log"
    else
        experiment="$(adaptation_experiment_name "$source" "$target")"
        task_log="$run_root/${source}_to_${target}.log"
    fi

    build_training_command "$stage" "$source" "$target" "$experiment" "$source_weights"
    printf '[DRY-RUN] stage=%s gpu=%s source=%s target=%s\n' \
        "$stage" "$gpu_id" "$source" "$target"
    printf '  log path    : %s\n' "$task_log"
    printf '  output path : %s/%s\n' "$OUTPUT_ROOT" "$experiment"
    if [[ "$stage" == "timematch" ]]; then
        printf '  weights path: %s\n' "$source_weights"
    else
        printf '  checkpoint  : %s/%s/fold_0/model.pt\n' "$OUTPUT_ROOT" "$experiment"
    fi
    printf '  command     : CUDA_VISIBLE_DEVICES=%q ' "$gpu_id"
    printf '%q ' "${TRAINING_COMMAND[@]}"
    printf '\n\n'
}

run_dry_run() {
    local run_root="$1"
    local index gpu_id source target source_weights

    printf 'Run root: %s\n' "$run_root"
    printf 'Output root: %s\n' "$OUTPUT_ROOT"
    printf 'TensorBoard root: %s\n\n' "$TENSORBOARD_ROOT"

    for index in "${!SOURCE_DOMAINS[@]}"; do
        gpu_id="${GPU_IDS[$index]}"
        source="${SOURCE_DOMAINS[$index]}"
        set_target_domains "$source" || return 1
        source_weights="$OUTPUT_ROOT/$(source_experiment_name "$source")"

        printf 'GPU %s / %s worker -> %s\n\n' \
            "$gpu_id" "$source" "${TARGET_DOMAINS[*]}"
        print_dry_run_command source "$gpu_id" "$source" "$source" "$run_root"
        for target in "${TARGET_DOMAINS[@]}"; do
            print_dry_run_command timematch "$gpu_id" "$source" "$target" \
                "$run_root" "$source_weights"
        done
    done

    printf '[DRY-RUN] 1 source command + 3 TimeMatch DA commands; no training launched.\n'
    printf '\nDRY-RUN SUMMARY\n'
    printf 'source commands:       1\n'
    printf 'TimeMatch commands:    3\n'
    printf 'source:                 DK1\n'
    printf 'DA:                     DK1 -> AT1, DK1 -> FR1, DK1 -> FR2\n'
    printf 'GPU assignment:         DK1 = GPU3\n'
    printf 'GPU0 used:              false\n'
    printf 'GPU1 used:              false\n'
    printf 'GPU2 used:              false\n'
    printf 'GPU3 used:              true\n'
    printf 'model:                  psemtkdmidltae\n'
    printf 'seed:                   %s\n' "$SEED"
    printf 'num_folds:              %s\n' "$NUM_FOLDS"
    printf 'progress_bar:           off\n'
}

launch_workers() {
    local run_root="$1"
    local launcher_log="$run_root/launcher.log"
    local script_path="$PROJECT_DIR/scripts/run_mtkd_ts_mid_dk1_3tasks_gpu3.sh"
    local index gpu_id source worker_log pid

    mkdir -p "$run_root" || return 1
    printf 'Run root: %s\n' "$run_root" | tee -a "$launcher_log"

    for index in "${!SOURCE_DOMAINS[@]}"; do
        gpu_id="${GPU_IDS[$index]}"
        source="${SOURCE_DOMAINS[$index]}"
        worker_log="$run_root/worker_gpu${gpu_id}_${source}.log"

        nohup bash "$script_path" --worker "$gpu_id" "$source" "$run_root" \
            > "$worker_log" 2>&1 < /dev/null &
        pid=$!
        printf 'GPU%s / %s worker PID: %s\n' "$gpu_id" "$source" "$pid" \
            | tee -a "$launcher_log"
        printf '  tail -f %s\n' "$worker_log" | tee -a "$launcher_log"
        printf '  ps -fp %s\n\n' "$pid" | tee -a "$launcher_log"
    done

    printf 'Outputs: %s\nTensorBoard: %s\n' "$OUTPUT_ROOT" "$TENSORBOARD_ROOT" \
        | tee -a "$launcher_log"
}

if [[ "${1:-}" == "--worker" ]]; then
    if [[ "$#" -ne 4 ]]; then
        printf '[ERROR] Worker usage: %s --worker GPU_ID SOURCE RUN_ROOT\n' "$0" >&2
        exit 2
    fi
    run_worker "$2" "$3" "$4"
    exit $?
fi

if [[ "$#" -ne 0 ]]; then
    printf '[ERROR] Usage: bash %s\n' "$0" >&2
    exit 2
fi

if ! command -v python >/dev/null 2>&1; then
    printf '[ERROR] python is not available in the current shell environment.\n' >&2
    exit 1
fi
python --version

RUN_ROOT="$LOG_ROOT/$(date +%Y%m%d_%H%M%S)"
check_fresh_outputs || exit 1

if [[ "$DRY_RUN" == "1" ]]; then
    run_dry_run "$RUN_ROOT"
    exit $?
fi

if [[ -e "$RUN_ROOT" ]]; then
    printf '[ERROR] Refusing to reuse existing run log directory: %s\n' "$RUN_ROOT" >&2
    exit 1
fi

launch_workers "$RUN_ROOT"
