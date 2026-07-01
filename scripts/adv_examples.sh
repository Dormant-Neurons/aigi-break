#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
    cat <<'USAGE'
Usage: ./scripts/adv_examples.sh [options] [-- extra attack args]

Generate adversarial examples with one detector, dataset, attack, and epsilon.

Options:
  --detector NAME     Detector name. Default: ojha_sd14.
  --dataset NAME      Dataset name under data/dataset/. Default: synthbuster_10.
  --data-path PATH    Dataset path. Default: data/dataset/<dataset>.
  --attack NAME       Attack: pgd, ensemble, diverse, universal, or query. Default: pgd.
  --eps N             Attack config suffix in 255-scale units. Default: 4.
  --samples N         Number of samples. Default: all.
  --output PATH       Output directory. Default: data/attack_results/examples_eps_<eps>.
  --checkpoint-dir PATH
                      Directory with detector checkpoints. Default: data/checkpoints.
  --checkpoint DETECTOR=PATH
                      Override one detector checkpoint path.
  --batch-size N      Batch size. Default: 2.
  --workers N         Data loader workers. Default: 0.
  --skip-images       Save only CSV metrics, not example PNGs.
  -h, --help          Show this message.

Example:
  ./scripts/adv_examples.sh --detector ojha_sd14 --dataset synthbuster_10 --attack pgd --eps 4 --samples 1
USAGE
}

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PYTHON=".venv/bin/python"
    else
        PYTHON="python3"
    fi
fi

DETECTOR="ojha_sd14"
DATASET="synthbuster_10"
DATA_PATH=""
ATTACK="pgd"
EPS=4
SAMPLES=""
OUTPUT=""
CHECKPOINT_DIR=""
CHECKPOINTS=()
BATCH_SIZE=2
WORKERS=0
SKIP_IMAGES=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --detector|--detector-name|--detector_name)
            DETECTOR="${2:?missing value for --detector}"
            shift 2
            ;;
        --dataset|--dataset-name|--dataset_name)
            DATASET="${2:?missing value for --dataset}"
            shift 2
            ;;
        --data-path|--data_path)
            DATA_PATH="${2:?missing value for --data-path}"
            shift 2
            ;;
        --attack|--attack-type|--attack_type)
            ATTACK="${2:?missing value for --attack}"
            shift 2
            ;;
        --eps)
            EPS="${2:?missing value for --eps}"
            shift 2
            ;;
        --samples|--num-samples|--num_samples)
            SAMPLES="${2:?missing value for --samples}"
            shift 2
            ;;
        --output|--output-dir|--output_dir)
            OUTPUT="${2:?missing value for --output}"
            shift 2
            ;;
        --checkpoint-dir|--checkpoint_dir)
            CHECKPOINT_DIR="${2:?missing value for --checkpoint-dir}"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINTS+=("${2:?missing value for --checkpoint}")
            shift 2
            ;;
        --batch-size|--batch_size)
            BATCH_SIZE="${2:?missing value for --batch-size}"
            shift 2
            ;;
        --workers|--num-workers|--num_workers)
            WORKERS="${2:?missing value for --workers}"
            shift 2
            ;;
        --skip-images|--skip_images)
            SKIP_IMAGES=true
            shift
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

DATA_PATH="${DATA_PATH:-data/dataset/$DATASET}"
OUTPUT="${OUTPUT:-data/attack_results/examples_eps_$EPS}"
CONFIG="configs/attacks/attack_configs_eps_${EPS}.json"

if [[ ! -d "$DATA_PATH" ]]; then
    echo "Dataset not found: $DATA_PATH" >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Attack config not found: $CONFIG" >&2
    exit 1
fi

SAMPLE_ARGS=()
if [[ -n "$SAMPLES" ]]; then
    SAMPLE_ARGS=(--num_samples "$SAMPLES")
fi

IMAGE_ARGS=()
if [[ "$SKIP_IMAGES" == true ]]; then
    IMAGE_ARGS=(--skip_images)
fi

CHECKPOINT_ARGS=()
if [[ -n "$CHECKPOINT_DIR" ]]; then
    CHECKPOINT_ARGS=(--checkpoint_dir "$CHECKPOINT_DIR")
fi
for checkpoint in "${CHECKPOINTS[@]}"; do
    CHECKPOINT_ARGS+=(--checkpoint "$checkpoint")
done

"$PYTHON" src/run_attack.py \
    --detector_name "$DETECTOR" \
    --dataset_name "$DATASET" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT" \
    --attack_type "$ATTACK" \
    --config_path "$CONFIG" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$WORKERS" \
    "${CHECKPOINT_ARGS[@]}" \
    "${SAMPLE_ARGS[@]}" \
    "${IMAGE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
