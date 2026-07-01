#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
    cat <<'USAGE'
Usage: ./scripts/train.sh --detector ojha|corvi [options] [-- extra trainer args]

Options:
  --detector NAME     Detector training code to run: ojha or corvi.
  --data PATH         Training split root.
                      Default: data/dataset/genimage/train
  --out PATH          Checkpoint root. Default: data/models
  --name NAME         Run name. Defaults to clip_vitl14 or corvi_genimage.
  --gpu ID            CUDA_VISIBLE_DEVICES value. Default: current value or 0.
  --epochs N          Epoch count. Default: 20.
  --batch-size N      Batch size. Defaults: 256 for ojha, 1024 for corvi.
  --workers N         Data loader workers for corvi. Default: 8.
  -h, --help          Show this message.

Examples:
  ./scripts/train.sh --detector ojha --data data/dataset/genimage/train
  ./scripts/train.sh --detector corvi --data data/dataset/genimage/train --epochs 20
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

DETECTOR=""
DATA="data/dataset/genimage/train"
OUT="data/models"
NAME=""
GPU="${CUDA_VISIBLE_DEVICES:-0}"
EPOCHS=20
BATCH_SIZE=""
WORKERS=8
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --detector)
            DETECTOR="${2:?missing value for --detector}"
            shift 2
            ;;
        --data)
            DATA="${2:?missing value for --data}"
            shift 2
            ;;
        --out)
            OUT="${2:?missing value for --out}"
            shift 2
            ;;
        --name)
            NAME="${2:?missing value for --name}"
            shift 2
            ;;
        --gpu)
            GPU="${2:?missing value for --gpu}"
            shift 2
            ;;
        --epochs)
            EPOCHS="${2:?missing value for --epochs}"
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

if [[ -z "$DETECTOR" ]]; then
    echo "--detector is required." >&2
    usage >&2
    exit 2
fi

if [[ ! -d "$DATA" ]]; then
    echo "Training data not found: $DATA" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"

case "$DETECTOR" in
    ojha)
        if [[ ! -d "$DATA/real" || ! -d "$DATA/fake" ]]; then
            echo "Ojha data must contain real/ and fake/: $DATA" >&2
            exit 1
        fi
        NAME="${NAME:-clip_vitl14}"
        BATCH_SIZE="${BATCH_SIZE:-256}"
        mkdir -p "$OUT/ojha"
        "$PYTHON" src/external/ojha/train.py \
            --name="$NAME" \
            --data_mode=ours \
            --arch=CLIP:ViT-L/14 \
            --fix_backbone \
            --real_list_path "$DATA/real" \
            --fake_list_path "$DATA/fake" \
            --loss_freq 20 \
            --batch_size "$BATCH_SIZE" \
            --checkpoints_dir "$OUT/ojha" \
            --niter "$EPOCHS" \
            "${EXTRA_ARGS[@]}"
        ;;
    corvi)
        REAL_DIR="$(find "$DATA" -maxdepth 2 -type d -name 0_real -print -quit)"
        FAKE_DIR="$(find "$DATA" -maxdepth 2 -type d -name 1_fake -print -quit)"
        if [[ -z "$REAL_DIR" || -z "$FAKE_DIR" ]]; then
            echo "Corvi data must contain 0_real/ and 1_fake/ folders, directly or one level under: $DATA" >&2
            exit 1
        fi
        NAME="${NAME:-corvi_genimage}"
        BATCH_SIZE="${BATCH_SIZE:-1024}"
        mkdir -p "$OUT/corvi"
        "$PYTHON" src/external/corvi/training_code/train.py \
            --name "$NAME" \
            --dataroot "$DATA" \
            --batch_size "$BATCH_SIZE" \
            --num_threads "$WORKERS" \
            --arch res50nodown \
            --cropSize 96 \
            --norm_type resnet \
            --resize_size 256 \
            --resize_ratio 0.75 \
            --blur_sig 0.0,3.0 \
            --cmp_method cv2,pil \
            --cmp_qual 30,100 \
            --resize_prob 0.2 \
            --jitter_prob 0.8 \
            --colordist_prob 0.2 \
            --cutout_prob -0.2 \
            --noise_prob 0.2 \
            --blur_prob 0.5 \
            --cmp_prob 0.5 \
            --rot90_prob 1.0 \
            --lr 0.00001 \
            --num_epoches "$EPOCHS" \
            --checkpoints_dir "$OUT/corvi" \
            "${EXTRA_ARGS[@]}"
        ;;
    *)
        echo "Unknown detector: $DETECTOR" >&2
        echo "Use ojha or corvi." >&2
        exit 2
        ;;
esac
