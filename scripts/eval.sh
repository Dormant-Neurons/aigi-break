#!/usr/bin/env bash
set -uo pipefail

if [ -x ".venv/bin/python" ]; then
    PYTHON="${PYTHON:-.venv/bin/python}"
else
    PYTHON="${PYTHON:-python3}"
fi

DETECTORS=(
    ojha_sd14
    corvi_sd14
    drct_clip_sd14
    drct_convb_sd14

    ojha_sd14_r2
    ojha_sd14_r4
)

DATASETS=(
    synthbuster_500
    chameleon_500
    gpt4o_500
)

OUTPUT_DIR="data/results/baseline"
BATCH_SIZE=1
NUM_WORKERS=16
PREPROCESSING=false
JPEG_QUALITY=20
RESIZE_FRACTION=0.5
BLUR_STD=1.0
NOISE_STD=0.1
BLUR_KERNEL=5
REMAINING_ARGS=()
FAILED=0

evaluate_single() {
    local detector=$1
    local dataset=$2
    local data_path="data/dataset/${dataset}"
    local start
    local end
    local duration
    local status
    local output_subdir
    local preprocess_args=()

    start=$(date +%s)
    
    echo "---- Evaluating ${detector} on ${dataset} ----"
    echo "Started: $(date)" | tee -a "${LOG_FILE}"
    
    if [ "$PREPROCESSING" = true ]; then
        echo "Preprocessing parameters:" | tee -a "${LOG_FILE}"
        echo "- JPEG quality: ${JPEG_QUALITY}" | tee -a "${LOG_FILE}"
        echo "- Scale factor: ${RESIZE_FRACTION}" | tee -a "${LOG_FILE}"
        echo "- Blur: kernel=${BLUR_KERNEL}, sigma=${BLUR_STD}" | tee -a "${LOG_FILE}"
        echo "- Noise std: ${NOISE_STD}" | tee -a "${LOG_FILE}"
        
        preprocess_args=(
            --preprocessing
            --jpeg_quality "${JPEG_QUALITY}"
            --resize_fraction "${RESIZE_FRACTION}"
            --blur_std "${BLUR_STD}"
            --noise_std "${NOISE_STD}"
            --blur_kernel "${BLUR_KERNEL}"
        )
        output_subdir="preprocessed_${JPEG_QUALITY}"
    else
        output_subdir="baseline"
    fi
    
    if PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" src/evaluate.py \
        --detector_name "${detector}" \
        --dataset_name "${dataset}" \
        --data_path "${data_path}" \
        --output_dir "${OUTPUT_DIR}/${output_subdir}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        "${preprocess_args[@]}" \
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
        --preprocessing)
            PREPROCESSING=true
            shift
            ;;
        --jpeg_quality)
            JPEG_QUALITY="$2"
            shift 2
            ;;
        --resize_fraction)
            RESIZE_FRACTION="$2"
            shift 2
            ;;
        --blur_std)
            BLUR_STD="$2"
            shift 2
            ;;
        --noise_std)
            NOISE_STD="$2"
            shift 2
            ;;
        --blur_kernel)
            BLUR_KERNEL="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/eval.log"
: > "${LOG_FILE}"

echo "Starting evaluation at $(date)"
echo "Mode: ${MODE}"
if [ "$PREPROCESSING" = true ]; then
    echo "Preprocessing enabled"
fi

if [ "$MODE" = "test" ]; then
    if [ -z "$DETECTOR" ] || [ -z "$DATASET" ]; then
        echo "Error: test mode requires --detector and --dataset arguments"
        exit 1
    fi
    evaluate_single "$DETECTOR" "$DATASET"
else
    echo "Detectors: ${DETECTORS[*]}"
    echo "Datasets: ${DATASETS[*]}"
    for detector in "${DETECTORS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            evaluate_single "$detector" "$dataset"
        done
    done
fi

echo
echo "Evaluation completed. Full log: ${LOG_FILE}"
exit "$FAILED"
