#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-/data/user/dataset/timematch_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mtkd_tq_loo_3variants}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-runs/mtkd_tq_loo_3variants}"
LOG_ROOT="${LOG_ROOT:-logs/mtkd_tq_loo_3variants}"
DRY_RUN="${DRY_RUN:-0}"
SEED=1
NUM_FOLDS=1

VARIANTS=(loo_tau_only pse_loo_fixed75 split_loo_tau_pse)
GPU_IDS=(0 1 2)
SOURCE_DOMAINS=(FR1 FR2)

declare -A DOMAIN_PATHS=(
    [AT1]="austria/33UVP/2017"
    [DK1]="denmark/32VNH/2017"
    [FR1]="france/30TXT/2017"
    [FR2]="france/31TCJ/2017"
)

set_target_domains() {
    case "$1" in
        FR1) TARGET_DOMAINS=(AT1 DK1 FR2) ;;
        FR2) TARGET_DOMAINS=(AT1 DK1 FR1) ;;
        *) printf '[ERROR] unsupported source: %s\n' "$1" >&2; return 1 ;;
    esac
}

source_experiment_name() {
    printf 'mtkd_tq_loo_%s_%s_source_seed%s' "$1" "$2" "$SEED"
}

adaptation_experiment_name() {
    printf 'mtkd_tq_loo_%s_%s_%s_seed%s' "$1" "$2" "$3" "$SEED"
}

source_weights_path() {
    printf '%s/%s/%s' "$OUTPUT_ROOT" "$1" \
        "$(source_experiment_name "$1" "$2")"
}

build_timematch_command() {
    local variant="$1" source="$2" target="$3" experiment="$4" weights="$5"
    TRAINING_COMMAND=(
        python -u train.py
        --data_root "$DATA_ROOT"
        --source "${DOMAIN_PATHS[$source]}"
        --target "${DOMAIN_PATHS[$target]}"
        --model psemtkdtqlooltae
        --tq_loo_variant "$variant"
        --tq_tau_init_days 75
        --tq_tau_min_days 1
        --tq_loo_pse_weight 0.1
        --seed "$SEED"
        --num_folds "$NUM_FOLDS"
        --device cuda
        --progress_bar off
        --closed_set true
        --combine_spring_and_winter false
        --with_shift_aug false
        --output_dir "$OUTPUT_ROOT/$variant"
        --tensorboard_log_dir "$TENSORBOARD_ROOT/$variant"
        --experiment_name "$experiment"
        timematch
        --weights "$weights"
    )
}

verify_source_checkpoints() {
    local variant source weights checkpoint missing=0
    for variant in "${VARIANTS[@]}"; do
        for source in "${SOURCE_DOMAINS[@]}"; do
            weights="$(source_weights_path "$variant" "$source")"
            checkpoint="$weights/fold_0/model.pt"
            if [[ -f "$checkpoint" ]]; then
                printf '[CHECKPOINT] reuse %s\n' "$checkpoint"
            else
                printf '[ERROR] source checkpoint missing: %s\n' "$checkpoint" >&2
                missing=$((missing + 1))
            fi
        done
    done
    if [[ "$missing" -ne 0 ]]; then
        printf '[ERROR] %s source checkpoint(s) missing; no output was removed.\n' \
            "$missing" >&2
        return 1
    fi
}

clear_da_outputs() {
    local variant source target experiment output_path tensorboard_path
    for variant in "${VARIANTS[@]}"; do
        for source in "${SOURCE_DOMAINS[@]}"; do
            set_target_domains "$source" || return 1
            for target in "${TARGET_DOMAINS[@]}"; do
                experiment="$(adaptation_experiment_name "$variant" "$source" "$target")"
                output_path="$OUTPUT_ROOT/$variant/$experiment"
                tensorboard_path="$TENSORBOARD_ROOT/$variant/${experiment}_fold0"
                case "$output_path" in
                    "$OUTPUT_ROOT/$variant/mtkd_tq_loo_${variant}_${source}_${target}_seed${SEED}") ;;
                    *) printf '[ERROR] unsafe cleanup path: %s\n' "$output_path" >&2; return 1 ;;
                esac
                printf '[CLEAN] %s\n' "$output_path"
                printf '[CLEAN] %s\n' "$tensorboard_path"
                if ! rm -rf -- "$output_path" "$tensorboard_path"; then
                    printf '[ERROR] failed to clear DA paths for %s\n' "$experiment" >&2
                    return 1
                fi
                if [[ -e "$output_path" || -e "$tensorboard_path" ]]; then
                    printf '[ERROR] DA path still exists after cleanup: %s\n' \
                        "$experiment" >&2
                    return 1
                fi
            done
        done
    done
}

run_da_task() {
    local gpu="$1" variant="$2" source="$3" target="$4" run_dir="$5" weights="$6"
    local experiment output_path log_file exit_code
    experiment="$(adaptation_experiment_name "$variant" "$source" "$target")"
    output_path="$OUTPUT_ROOT/$variant/$experiment"
    log_file="$run_dir/${source}_to_${target}.log"
    if [[ -e "$output_path" ]]; then
        printf '[ERROR] DA output still exists after cleanup: %s\n' "$output_path" >&2
        return 1
    fi
    build_timematch_command "$variant" "$source" "$target" "$experiment" "$weights"
    printf '[START] gpu=%s variant=%s source=%s target=%s weights=%s\n' \
        "$gpu" "$variant" "$source" "$target" "$weights"
    if CUDA_VISIBLE_DEVICES="$gpu" "${TRAINING_COMMAND[@]}" >"$log_file" 2>&1; then
        printf '[DONE] gpu=%s variant=%s source=%s target=%s\n' \
            "$gpu" "$variant" "$source" "$target"
        return 0
    fi
    exit_code=$?
    printf '[FAILED] exit=%s log=%s\n' "$exit_code" "$log_file" >&2
    return "$exit_code"
}

run_source_group() {
    local gpu="$1" variant="$2" source="$3" run_dir="$4"
    local weights checkpoint target failures=0
    set_target_domains "$source" || return 1
    weights="$(source_weights_path "$variant" "$source")"
    checkpoint="$weights/fold_0/model.pt"
    if [[ ! -f "$checkpoint" ]]; then
        printf '[ERROR] source checkpoint disappeared: %s\n' "$checkpoint" >&2
        return 1
    fi
    for target in "${TARGET_DOMAINS[@]}"; do
        if ! run_da_task "$gpu" "$variant" "$source" "$target" "$run_dir" "$weights"; then
            failures=$((failures + 1))
        fi
    done
    [[ "$failures" -eq 0 ]]
}

run_worker() {
    local gpu="$1" variant="$2" run_dir="$3" source failures=0
    mkdir -p "$run_dir" || return 1
    for source in "${SOURCE_DOMAINS[@]}"; do
        if ! run_source_group "$gpu" "$variant" "$source" "$run_dir"; then
            failures=$((failures + 1))
            printf '[WARN] continuing to next source for %s\n' "$variant" >&2
        fi
    done
    [[ "$failures" -eq 0 ]]
}

print_dry_run() {
    local index gpu variant source target experiment weights output_path tensorboard_path
    for index in "${!VARIANTS[@]}"; do
        gpu="${GPU_IDS[$index]}"
        variant="${VARIANTS[$index]}"
        for source in "${SOURCE_DOMAINS[@]}"; do
            set_target_domains "$source" || return 1
            weights="$(source_weights_path "$variant" "$source")"
            printf '[DRY-RUN] reuse checkpoint: %s/fold_0/model.pt\n' "$weights"
            for target in "${TARGET_DOMAINS[@]}"; do
                experiment="$(adaptation_experiment_name "$variant" "$source" "$target")"
                output_path="$OUTPUT_ROOT/$variant/$experiment"
                tensorboard_path="$TENSORBOARD_ROOT/$variant/${experiment}_fold0"
                printf '[DRY-RUN] would remove: %s\n' "$output_path"
                printf '[DRY-RUN] would remove: %s\n' "$tensorboard_path"
                build_timematch_command "$variant" "$source" "$target" "$experiment" "$weights"
                printf '[DRY-RUN] gpu=%s variant=%s source=%s target=%s\n  command: ' \
                    "$gpu" "$variant" "$source" "$target"
                printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
                printf '%q ' "${TRAINING_COMMAND[@]}"
                printf '\n'
            done
        done
    done
    printf '\nDRY-RUN SUMMARY\n'
    printf 'variants:              3\n'
    printf 'sources:               FR1, FR2\n'
    printf 'source commands:       0\n'
    printf 'reused source checkpoints: 6\n'
    printf 'TimeMatch commands:    18\n'
    printf 'total training commands: 18\n'
    printf 'GPU0:                  loo_tau_only\n'
    printf 'GPU1:                  pse_loo_fixed75\n'
    printf 'GPU2:                  split_loo_tau_pse\n'
    printf 'GPU3:                  unused\n'
    printf 'model:                 psemtkdtqlooltae\n'
    printf 'seed:                  1\n'
    printf 'num_folds:             1\n'
    printf 'progress_bar:          off\n'
    printf 'cleanup:               dry-run only; nothing removed\n'
}

launch_workers() {
    local run_root="$1" script_path="$PROJECT_DIR/scripts/run_mtkd_tq_loo_3variants_fr1_fr2.sh"
    local index gpu variant worker_dir worker_log pid
    mkdir -p "$run_root" || return 1
    export DATA_ROOT OUTPUT_ROOT TENSORBOARD_ROOT LOG_ROOT
    for index in "${!VARIANTS[@]}"; do
        gpu="${GPU_IDS[$index]}"
        variant="${VARIANTS[$index]}"
        worker_dir="$run_root/$variant"
        worker_log="$worker_dir/worker_gpu${gpu}.log"
        mkdir -p "$worker_dir" || return 1
        nohup bash "$script_path" --worker "$gpu" "$variant" "$worker_dir" \
            >"$worker_log" 2>&1 </dev/null &
        pid=$!
        printf 'GPU%s %s worker PID=%s log=%s\n' "$gpu" "$variant" "$pid" "$worker_log"
    done
    printf 'GPU3 unused. Outer nohup is not required.\n'
}

if [[ "${1:-}" == --worker ]]; then
    [[ "$#" -eq 4 ]] || { printf 'worker usage error\n' >&2; exit 2; }
    run_worker "$2" "$3" "$4"
    exit $?
fi
[[ "$#" -eq 0 ]] || { printf 'usage: bash %s\n' "$0" >&2; exit 2; }

if [[ "$DRY_RUN" == 1 ]]; then
    print_dry_run
    exit $?
fi

verify_source_checkpoints || exit 1
clear_da_outputs || exit 1
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
launch_workers "$LOG_ROOT/$RUN_TIMESTAMP"
