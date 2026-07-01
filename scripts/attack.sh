#!/usr/bin/env bash
set -uo pipefail

if [ -x ".venv/bin/python" ]; then
    PYTHON="${PYTHON:-.venv/bin/python}"
else
    PYTHON="${PYTHON:-python3}"
fi

DETECTORS=(
    ojha_sd14
    drct_clip_sd14
    corvi_sd14
    drct_convb_sd14
)

DATASETS=(
    synthbuster_500
    chameleon_500
    gpt4o_500
)

ATTACKS=(
    pgd
    ensemble
    diverse
    universal
    query
)

EPSILON=8
OUTPUT_DIR="data/attack_results/attack_results_${EPSILON}_255"
BATCH_SIZE=1
NUM_WORKERS=1
CONFIG_PATH="configs/attacks/attack_configs_eps_${EPSILON}.json"
REMAINING_ARGS=()
FAILED=0

run_attack() {
    local detector=$1
    local dataset=$2
    local attack_type=$3
    local data_path="data/dataset/${dataset}"
    local start
    local end
    local duration
    local status

    start=$(date +%s)
    
    echo "---- Running ${attack_type} attack on ${detector} with ${dataset} dataset ----"
    echo "Started: $(date)" | tee -a "${LOG_FILE}"
    
    if "${PYTHON}" src/run_attack.py \
        --detector_name "${detector}" \
        --dataset_name "${dataset}" \
        --data_path "${data_path}" \
        --output_dir "${OUTPUT_DIR}" \
        --attack_type "${attack_type}" \
        --config_path "${CONFIG_PATH}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        "${REMAINING_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    then
        status="Success"
    else
        status="Failed"
        FAILED=1
    fi
    
    end=$(date +%s)
    duration=$((end - start))
    echo "Finished in ${duration}s: ${status}"
}

MODE="loop"
DETECTOR=""
DATASET=""
ATTACK_TYPE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --detector)
            DETECTOR="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --attack)
            ATTACK_TYPE="$2"
            shift 2
            ;;
        --eps)
            EPSILON="$2"
            OUTPUT_DIR="data/attack_results/attack_results_${EPSILON}_255"
            CONFIG_PATH="configs/attacks/attack_configs_eps_${EPSILON}.json"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/attack.log"
: > "${LOG_FILE}"

echo "Starting attacks at $(date)"
echo "Mode: ${MODE}"
echo "Attack type: ${ATTACK_TYPE}"

if [ "$MODE" = "test" ]; then
    if [ -z "$DETECTOR" ] || [ -z "$DATASET" ]; then
        echo "Error: test mode requires --detector and --dataset arguments"
        exit 1
    fi
    if [ "$ATTACK_TYPE" = "all" ]; then
        for attack in "${ATTACKS[@]}"; do
            run_attack "$DETECTOR" "$DATASET" "$attack"
        done
    else
        run_attack "$DETECTOR" "$DATASET" "$ATTACK_TYPE"
    fi
else
    echo "Detectors: ${DETECTORS[*]}"
    echo "Datasets: ${DATASETS[*]}"
    echo "Attacks: ${ATTACKS[*]}"
    for detector in "${DETECTORS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            if [ "$ATTACK_TYPE" = "all" ]; then
                for attack in "${ATTACKS[@]}"; do
                    run_attack "$detector" "$dataset" "$attack"
                done
            else
                run_attack "$detector" "$dataset" "$ATTACK_TYPE"
            fi
        done
    done
fi

echo
echo "Attack evaluation completed. Full log: ${LOG_FILE}"
exit "$FAILED"
