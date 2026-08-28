#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mtkd_tq_fixed_tau75_90_12tasks}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-runs/mtkd_tq_fixed_tau75_90_12tasks}"
LOG_ROOT="${LOG_ROOT:-logs/mtkd_tq_fixed_tau75_90_12tasks}"
DRY_RUN="${DRY_RUN:-0}"
SEED=1
NUM_FOLDS=1

GPU_IDS=(0 1 2)
WORKER_TAUS=(75 90 90)
WORKER_SOURCE_SETS=("AT1 DK1" "AT1 DK1" "FR1 FR2")

declare -A DOMAIN_PATHS=(
    [AT1]="austria/33UVP/2017"
    [DK1]="denmark/32VNH/2017"
    [FR1]="france/30TXT/2017"
    [FR2]="france/31TCJ/2017"
)

set_target_domains() {
    case "$1" in
        AT1) TARGET_DOMAINS=(DK1 FR1 FR2) ;;
        DK1) TARGET_DOMAINS=(AT1 FR1 FR2) ;;
        FR1) TARGET_DOMAINS=(AT1 DK1 FR2) ;;
        FR2) TARGET_DOMAINS=(AT1 DK1 FR1) ;;
        *) printf '[ERROR] unsupported source: %s\n' "$1" >&2; return 1 ;;
    esac
}

source_experiment_name() {
    printf 'mtkd_tq_fixed_tau%s_%s_source_seed%s' "$1" "$2" "$SEED"
}

adaptation_experiment_name() {
    printf 'mtkd_tq_fixed_tau%s_%s_%s_seed%s' "$1" "$2" "$3" "$SEED"
}

build_training_command() {
    local stage="$1" tau_days="$2" source="$3" target="$4" experiment="$5"
    local weights="${6:-}"
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
        --output_dir "$OUTPUT_ROOT/tau${tau_days}"
        --tensorboard_log_dir "$TENSORBOARD_ROOT/tau${tau_days}"
        --experiment_name "$experiment"
    )
    if [[ "$stage" == timematch ]]; then
        TRAINING_COMMAND+=(timematch --weights "$weights")
    fi
}

validate_worker_matrix() {
    local index tau source_set source
    for index in "${!GPU_IDS[@]}"; do
        tau="${WORKER_TAUS[$index]}"
        source_set="${WORKER_SOURCE_SETS[$index]}"
        read -r -a sources <<<"$source_set"
        [[ "${#sources[@]}" -eq 2 ]] || return 1
        for source in "${sources[@]}"; do
            if [[ "$tau" == 75 && "$source" != AT1 && "$source" != DK1 ]]; then
                printf '[ERROR] tau75 must not schedule completed source %s\n' "$source" >&2
                return 1
            fi
        done
    done
    [[ "${WORKER_TAUS[*]}" == "75 90 90" ]] || return 1
    [[ "${WORKER_SOURCE_SETS[0]}" == "AT1 DK1" ]] || return 1
    [[ "${WORKER_SOURCE_SETS[1]}" == "AT1 DK1" ]] || return 1
    [[ "${WORKER_SOURCE_SETS[2]}" == "FR1 FR2" ]] || return 1
}

run_training_task() {
    local stage="$1" gpu="$2" tau_days="$3" source="$4" target="$5" run_dir="$6"
    local weights="${7:-}" experiment output_path log_file exit_code
    if [[ "$stage" == source ]]; then
        experiment="$(source_experiment_name "$tau_days" "$source")"
        log_file="$run_dir/source_${source}.log"
    else
        experiment="$(adaptation_experiment_name "$tau_days" "$source" "$target")"
        log_file="$run_dir/${source}_to_${target}.log"
    fi
    output_path="$OUTPUT_ROOT/tau${tau_days}/$experiment"
    if [[ -e "$output_path" ]]; then
        printf '[FAILED] output already exists: %s\n' "$output_path" >&2
        return 1
    fi
    build_training_command "$stage" "$tau_days" "$source" "$target" \
        "$experiment" "$weights"
    printf '[START] gpu=%s tau=%s stage=%s source=%s target=%s\n' \
        "$gpu" "$tau_days" "$stage" "$source" "$target"
    if CUDA_VISIBLE_DEVICES="$gpu" "${TRAINING_COMMAND[@]}" >"$log_file" 2>&1; then
        printf '[DONE] gpu=%s tau=%s stage=%s source=%s target=%s\n' \
            "$gpu" "$tau_days" "$stage" "$source" "$target"
        return 0
    fi
    exit_code=$?
    printf '[FAILED] exit=%s log=%s\n' "$exit_code" "$log_file" >&2
    return "$exit_code"
}

run_source_group() {
    local gpu="$1" tau_days="$2" source="$3" run_dir="$4"
    local source_experiment weights checkpoint target failures=0
    set_target_domains "$source" || return 2
    source_experiment="$(source_experiment_name "$tau_days" "$source")"
    weights="$OUTPUT_ROOT/tau${tau_days}/$source_experiment"
    checkpoint="$weights/fold_0/model.pt"

    if ! run_training_task source "$gpu" "$tau_days" "$source" "$source" "$run_dir"; then
        printf '[FAILED] tau%s %s source; skipping its three DA tasks.\n' \
            "$tau_days" "$source" >&2
        return 2
    fi
    if [[ ! -f "$checkpoint" ]]; then
        printf '[FAILED] source checkpoint missing: %s\n' "$checkpoint" >&2
        return 2
    fi
    for target in "${TARGET_DOMAINS[@]}"; do
        if ! run_training_task timematch "$gpu" "$tau_days" "$source" \
            "$target" "$run_dir" "$weights"; then
            failures=$((failures + 1))
            printf '[WARN] continuing after failed DA: tau%s %s -> %s\n' \
                "$tau_days" "$source" "$target" >&2
        fi
    done
    [[ "$failures" -eq 0 ]]
}

run_worker() {
    local gpu="$1" tau_days="$2" run_dir="$3"
    shift 3
    local source status failures=0
    mkdir -p "$run_dir" || return 1
    for source in "$@"; do
        run_source_group "$gpu" "$tau_days" "$source" "$run_dir"
        status=$?
        if [[ "$status" -ne 0 ]]; then
            failures=$((failures + 1))
            printf '[WARN] continuing to next source on GPU%s tau%s.\n' \
                "$gpu" "$tau_days" >&2
        fi
    done
    [[ "$failures" -eq 0 ]]
}

check_fresh_outputs() {
    local index tau source_set source target experiment
    for index in "${!GPU_IDS[@]}"; do
        tau="${WORKER_TAUS[$index]}"
        source_set="${WORKER_SOURCE_SETS[$index]}"
        read -r -a sources <<<"$source_set"
        for source in "${sources[@]}"; do
            experiment="$(source_experiment_name "$tau" "$source")"
            if [[ -e "$OUTPUT_ROOT/tau${tau}/$experiment" ]]; then
                printf '[ERROR] refusing to overwrite: %s\n' \
                    "$OUTPUT_ROOT/tau${tau}/$experiment" >&2
                return 1
            fi
            set_target_domains "$source" || return 1
            for target in "${TARGET_DOMAINS[@]}"; do
                experiment="$(adaptation_experiment_name "$tau" "$source" "$target")"
                if [[ -e "$OUTPUT_ROOT/tau${tau}/$experiment" ]]; then
                    printf '[ERROR] refusing to overwrite: %s\n' \
                        "$OUTPUT_ROOT/tau${tau}/$experiment" >&2
                    return 1
                fi
            done
        done
    done
}

print_dry_command() {
    local stage="$1" gpu="$2" tau="$3" source="$4" target="$5" run_dir="$6"
    local weights="${7:-}" experiment
    if [[ "$stage" == source ]]; then
        experiment="$(source_experiment_name "$tau" "$source")"
    else
        experiment="$(adaptation_experiment_name "$tau" "$source" "$target")"
    fi
    build_training_command "$stage" "$tau" "$source" "$target" \
        "$experiment" "$weights"
    printf '[DRY-RUN] gpu=%s tau=%s stage=%s source=%s target=%s\n  command: ' \
        "$gpu" "$tau" "$stage" "$source" "$target"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${TRAINING_COMMAND[@]}"
    printf '\n'
}

run_dry_run() {
    local run_root="$1" index gpu tau source_set source target weights
    local tau75_sources=0 tau75_da=0 tau90_sources=0 tau90_da=0
    for index in "${!GPU_IDS[@]}"; do
        gpu="${GPU_IDS[$index]}"
        tau="${WORKER_TAUS[$index]}"
        source_set="${WORKER_SOURCE_SETS[$index]}"
        read -r -a sources <<<"$source_set"
        for source in "${sources[@]}"; do
            set_target_domains "$source" || return 1
            weights="$OUTPUT_ROOT/tau${tau}/$(source_experiment_name "$tau" "$source")"
            print_dry_command source "$gpu" "$tau" "$source" "$source" "$run_root"
            if [[ "$tau" == 75 ]]; then
                tau75_sources=$((tau75_sources + 1))
            else
                tau90_sources=$((tau90_sources + 1))
            fi
            for target in "${TARGET_DOMAINS[@]}"; do
                print_dry_command timematch "$gpu" "$tau" "$source" "$target" \
                    "$run_root" "$weights"
                if [[ "$tau" == 75 ]]; then
                    tau75_da=$((tau75_da + 1))
                else
                    tau90_da=$((tau90_da + 1))
                fi
            done
        done
    done
    [[ "$tau75_sources" -eq 2 && "$tau75_da" -eq 6 ]] || return 1
    [[ "$tau90_sources" -eq 4 && "$tau90_da" -eq 12 ]] || return 1

    printf '\nDRY-RUN SUMMARY\n'
    printf 'new tau75 source commands = 2\n'
    printf 'new tau75 DA commands     = 6\n'
    printf 'new tau90 source commands = 4\n'
    printf 'new tau90 DA commands     = 12\n'
    printf 'total source commands = 6\n'
    printf 'total DA commands     = 18\n'
    printf 'total commands        = 24\n'
    printf 'tau75 FR1 source commands = 0\n'
    printf 'tau75 FR2 source commands = 0\n'
    printf 'tau75 FR1 DA commands = 0\n'
    printf 'tau75 FR2 DA commands = 0\n'
    printf 'GPU0 = tau75 AT1+DK1\n'
    printf 'GPU1 = tau90 AT1+DK1\n'
    printf 'GPU2 = tau90 FR1+FR2\n'
    printf 'GPU3 = unused\n'
    printf 'model = psemtkdtqsingleltae\n'
    printf 'seed = 1; num_folds = 1; progress_bar = off\n'
}

launch_workers() {
    local run_root="$1"
    local script_path="$PROJECT_DIR/scripts/run_mtkd_tq_fixed_tau75_90_remaining_3gpu.sh"
    local index gpu tau source_set worker_dir worker_log pid
    mkdir -p "$run_root" || return 1
    export DATA_ROOT OUTPUT_ROOT TENSORBOARD_ROOT LOG_ROOT
    for index in "${!GPU_IDS[@]}"; do
        gpu="${GPU_IDS[$index]}"
        tau="${WORKER_TAUS[$index]}"
        source_set="${WORKER_SOURCE_SETS[$index]}"
        read -r -a sources <<<"$source_set"
        worker_dir="$run_root/gpu${gpu}_tau${tau}"
        worker_log="$worker_dir/worker.log"
        mkdir -p "$worker_dir" || return 1
        nohup bash "$script_path" --worker "$gpu" "$tau" "$worker_dir" \
            "${sources[@]}" >"$worker_log" 2>&1 </dev/null &
        pid=$!
        printf 'GPU%s tau%s sources=%s worker PID=%s log=%s\n' \
            "$gpu" "$tau" "$source_set" "$pid" "$worker_log"
    done
    printf 'GPU3 unused. Outer nohup is not required.\n'
}

if [[ "${1:-}" == --worker ]]; then
    [[ "$#" -eq 6 ]] || { printf 'worker usage error\n' >&2; exit 2; }
    run_worker "$2" "$3" "$4" "$5" "$6"
    exit $?
fi
[[ "$#" -eq 0 ]] || { printf 'usage: bash %s\n' "$0" >&2; exit 2; }

validate_worker_matrix || { printf '[ERROR] invalid worker matrix\n' >&2; exit 1; }
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$LOG_ROOT/$RUN_TIMESTAMP"
if [[ "$DRY_RUN" == 1 ]]; then
    run_dry_run "$RUN_ROOT"
    exit $?
fi
check_fresh_outputs || exit 1
launch_workers "$RUN_ROOT"
