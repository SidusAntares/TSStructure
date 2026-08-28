#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mtkd_tq_single_tau_sweep}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-runs/mtkd_tq_single_tau_sweep}"
LOG_ROOT="${LOG_ROOT:-logs/mtkd_tq_single_tau_sweep}"
DRY_RUN="${DRY_RUN:-0}"
SEED=1
NUM_FOLDS=1

GPU_IDS=(0 1 2)
TAU_VALUES=(45 60 75)
SOURCE_DOMAINS=(FR1 FR2)

declare -A DOMAIN_PATHS=(
    [AT1]="austria/33UVP/2017"
    [DK1]="denmark/32VNH/2017"
    [FR1]="france/30TXT/2017"
    [FR2]="france/31TCJ/2017"
)

set_target_domains() {
    local source="$1"
    case "$source" in
        FR1) TARGET_DOMAINS=(AT1 DK1 FR2) ;;
        FR2) TARGET_DOMAINS=(AT1 DK1 FR1) ;;
        *)
            printf '[ERROR] Unsupported source domain: %s\n' "$source" >&2
            return 1
            ;;
    esac
}

source_experiment_name() {
    printf 'mtkd_tq_single_tau%s_%s_source_seed%s' "$1" "$2" "$SEED"
}

adaptation_experiment_name() {
    printf 'mtkd_tq_single_tau%s_%s_%s_seed%s' "$1" "$2" "$3" "$SEED"
}

build_training_command() {
    local stage="$1"
    local source="$2"
    local target="$3"
    local experiment="$4"
    local tau_days="$5"
    local weights_path="${6:-}"
    local output_root="$OUTPUT_ROOT/tau${tau_days}"
    local tensorboard_root="$TENSORBOARD_ROOT/tau${tau_days}"

    TRAINING_COMMAND=(
        python -u train.py
        --data_root "$DATA_ROOT"
        --source "${DOMAIN_PATHS[$source]}"
        --target "${DOMAIN_PATHS[$target]}"
        --model psemtkdtqsingleltae
        --tq_tau_days "$tau_days"
        --seed "$SEED"
        --num_folds "$NUM_FOLDS"
        --device cuda
        --progress_bar off
        --closed_set true
        --combine_spring_and_winter false
        --with_shift_aug false
        --output_dir "$output_root"
        --tensorboard_log_dir "$tensorboard_root"
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
    local tau_days="$3"
    local source="$4"
    local target="$5"
    local run_root="$6"
    local source_weights="${7:-}"
    local output_root="$OUTPUT_ROOT/tau${tau_days}"
    local experiment task_log exit_code

    if [[ "$stage" == "source" ]]; then
        experiment="$(source_experiment_name "$tau_days" "$source")"
        task_log="$run_root/source_${source}.log"
    else
        experiment="$(adaptation_experiment_name "$tau_days" "$source" "$target")"
        task_log="$run_root/${source}_to_${target}.log"
    fi

    if [[ -e "$output_root/$experiment" ]]; then
        print_task_boundary "FAILED TASK" "$source" "$target" "$gpu_id" \
            "OUTPUT ALREADY EXISTS: $output_root/$experiment"
        return 1
    fi

    build_training_command "$stage" "$source" "$target" "$experiment" \
        "$tau_days" "$source_weights"
    print_task_boundary "START TASK" "$source" "$target" "$gpu_id"
    printf 'tau days       : %s\n' "$tau_days"
    printf 'experiment log : %s\n' "$task_log"
    printf 'output path    : %s/%s\n\n' "$output_root" "$experiment"

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

run_source_group() {
    local gpu_id="$1"
    local tau_days="$2"
    local source="$3"
    local run_root="$4"
    local output_root="$OUTPUT_ROOT/tau${tau_days}"
    local source_experiment source_weights source_checkpoint target failures=0

    set_target_domains "$source" || return 1
    source_experiment="$(source_experiment_name "$tau_days" "$source")"
    source_weights="$output_root/$source_experiment"
    source_checkpoint="$source_weights/fold_0/model.pt"

    printf '[WORKER] GPU %s / tau%s handles source %s and targets: %s\n' \
        "$gpu_id" "$tau_days" "$source" "${TARGET_DOMAINS[*]}"

    if ! run_training_task source "$gpu_id" "$tau_days" "$source" "$source" \
        "$run_root"; then
        printf '[ERROR] Source training failed; stopping GPU %s / tau%s worker.\n' \
            "$gpu_id" "$tau_days" >&2
        return 2
    fi

    if [[ ! -f "$source_checkpoint" ]]; then
        printf '[ERROR] Source checkpoint does not exist: %s\n' \
            "$source_checkpoint" >&2
        return 2
    fi

    printf '[WORKER] Source checkpoint verified: %s\n' "$source_checkpoint"
    for target in "${TARGET_DOMAINS[@]}"; do
        if ! run_training_task timematch "$gpu_id" "$tau_days" "$source" \
            "$target" "$run_root" "$source_weights"; then
            failures=$((failures + 1))
            printf '[WARN] Continuing with remaining targets for GPU %s / tau%s / %s.\n' \
                "$gpu_id" "$tau_days" "$source" >&2
        fi
    done

    if [[ "$failures" -ne 0 ]]; then
        printf '[WORKER] GPU %s / tau%s / %s finished with %s failed DA task(s).\n' \
            "$gpu_id" "$tau_days" "$source" "$failures" >&2
        return 1
    fi
}

run_worker() {
    local gpu_id="$1"
    local tau_days="$2"
    local run_root="$3"
    local source group_status da_failures=0

    for source in "${SOURCE_DOMAINS[@]}"; do
        run_source_group "$gpu_id" "$tau_days" "$source" "$run_root"
        group_status=$?
        if [[ "$group_status" -eq 2 ]]; then
            return 1
        fi
        if [[ "$group_status" -ne 0 ]]; then
            da_failures=$((da_failures + 1))
            printf '[WARN] Continuing with the next source group for GPU %s / tau%s.\n' \
                "$gpu_id" "$tau_days" >&2
        fi
    done
    if [[ "$da_failures" -ne 0 ]]; then
        return 1
    fi
    printf '[WORKER] GPU %s / tau%s completed FR1 and FR2 groups.\n' \
        "$gpu_id" "$tau_days"
}

check_fresh_outputs() {
    local tau_days source target experiment output_root

    for tau_days in "${TAU_VALUES[@]}"; do
        output_root="$OUTPUT_ROOT/tau${tau_days}"
        for source in "${SOURCE_DOMAINS[@]}"; do
            experiment="$(source_experiment_name "$tau_days" "$source")"
            if [[ -e "$output_root/$experiment" ]]; then
                printf '[ERROR] Refusing to overwrite existing output: %s/%s\n' \
                    "$output_root" "$experiment" >&2
                return 1
            fi

            set_target_domains "$source" || return 1
            for target in "${TARGET_DOMAINS[@]}"; do
                experiment="$(adaptation_experiment_name \
                    "$tau_days" "$source" "$target")"
                if [[ -e "$output_root/$experiment" ]]; then
                    printf '[ERROR] Refusing to overwrite existing output: %s/%s\n' \
                        "$output_root" "$experiment" >&2
                    return 1
                fi
            done
        done
    done
}

print_dry_run_command() {
    local stage="$1"
    local gpu_id="$2"
    local tau_days="$3"
    local source="$4"
    local target="$5"
    local run_root="$6"
    local source_weights="${7:-}"
    local output_root="$OUTPUT_ROOT/tau${tau_days}"
    local experiment task_log

    if [[ "$stage" == "source" ]]; then
        experiment="$(source_experiment_name "$tau_days" "$source")"
        task_log="$run_root/source_${source}.log"
    else
        experiment="$(adaptation_experiment_name "$tau_days" "$source" "$target")"
        task_log="$run_root/${source}_to_${target}.log"
    fi

    build_training_command "$stage" "$source" "$target" "$experiment" \
        "$tau_days" "$source_weights"
    printf '[DRY-RUN] stage=%s gpu=%s tau=%s source=%s target=%s\n' \
        "$stage" "$gpu_id" "$tau_days" "$source" "$target"
    printf '  log path    : %s\n' "$task_log"
    printf '  output path : %s/%s\n' "$output_root" "$experiment"
    if [[ "$stage" == "timematch" ]]; then
        printf '  weights path: %s\n' "$source_weights"
    else
        printf '  checkpoint  : %s/%s/fold_0/model.pt\n' "$output_root" "$experiment"
    fi
    printf '  command     : CUDA_VISIBLE_DEVICES=%q ' "$gpu_id"
    printf '%q ' "${TRAINING_COMMAND[@]}"
    printf '\n\n'
}

run_dry_run() {
    local run_root="$1"
    local index gpu_id tau_days tau_run_root output_root source target source_weights

    printf 'Run root: %s\n' "$run_root"
    printf 'Output root: %s\n' "$OUTPUT_ROOT"
    printf 'TensorBoard root: %s\n\n' "$TENSORBOARD_ROOT"

    for index in "${!TAU_VALUES[@]}"; do
        gpu_id="${GPU_IDS[$index]}"
        tau_days="${TAU_VALUES[$index]}"
        tau_run_root="$run_root/tau${tau_days}"
        output_root="$OUTPUT_ROOT/tau${tau_days}"
        printf 'GPU %s worker -> tau%s\n\n' "$gpu_id" "$tau_days"

        for source in "${SOURCE_DOMAINS[@]}"; do
            set_target_domains "$source" || return 1
            source_weights="$output_root/$(source_experiment_name \
                "$tau_days" "$source")"
            print_dry_run_command source "$gpu_id" "$tau_days" "$source" \
                "$source" "$tau_run_root"
            for target in "${TARGET_DOMAINS[@]}"; do
                print_dry_run_command timematch "$gpu_id" "$tau_days" "$source" \
                    "$target" "$tau_run_root" "$source_weights"
            done
        done
    done

    printf '[DRY-RUN] 6 source commands + 18 TimeMatch DA commands; no training launched.\n'
    printf '\nDRY-RUN SUMMARY\n'
    printf 'tau values:            45, 60, 75\n'
    printf 'sources:               FR1, FR2\n'
    printf 'source commands:       6\n'
    printf 'TimeMatch commands:    18\n'
    printf 'total training commands: 24\n'
    printf 'GPU0:                  tau45\n'
    printf 'GPU1:                  tau60\n'
    printf 'GPU2:                  tau75\n'
    printf 'GPU3:                  unused\n'
    printf 'model:                 psemtkdtqsingleltae\n'
    printf 'seed:                  %s\n' "$SEED"
    printf 'num_folds:             %s\n' "$NUM_FOLDS"
    printf 'progress_bar:          off\n'
}

launch_workers() {
    local run_root="$1"
    local launcher_log="$run_root/launcher.log"
    local script_path="$PROJECT_DIR/scripts/run_mtkd_tq_single_tau_sweep_fr1_fr2.sh"
    local index gpu_id tau_days tau_run_root worker_log pid

    mkdir -p "$run_root" || return 1
    printf 'Run root: %s\n' "$run_root" | tee -a "$launcher_log"

    for index in "${!TAU_VALUES[@]}"; do
        gpu_id="${GPU_IDS[$index]}"
        tau_days="${TAU_VALUES[$index]}"
        tau_run_root="$run_root/tau${tau_days}"
        mkdir -p "$tau_run_root" || return 1
        worker_log="$tau_run_root/worker_gpu${gpu_id}_tau${tau_days}.log"

        nohup bash "$script_path" --worker "$gpu_id" "$tau_days" "$tau_run_root" \
            > "$worker_log" 2>&1 < /dev/null &
        pid=$!
        printf 'GPU%s / tau%s worker PID: %s\n' "$gpu_id" "$tau_days" "$pid" \
            | tee -a "$launcher_log"
        printf '  tail -f %s\n' "$worker_log" | tee -a "$launcher_log"
        printf '  ps -fp %s\n\n' "$pid" | tee -a "$launcher_log"
    done

    printf 'Outputs: %s\nTensorBoard: %s\n' "$OUTPUT_ROOT" "$TENSORBOARD_ROOT" \
        | tee -a "$launcher_log"
}

if [[ "${1:-}" == "--worker" ]]; then
    if [[ "$#" -ne 4 ]]; then
        printf '[ERROR] Worker usage: %s --worker GPU_ID TAU_DAYS RUN_ROOT\n' "$0" >&2
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
