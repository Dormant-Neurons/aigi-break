#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
    cat <<'USAGE'
Usage: ./scripts/setup.sh [--smoke]

Create the Python environment and the local data directories.

Options:
  --smoke    Run a one-sample eval and attack after setup.
  -h, --help Show this message.
USAGE
}

RUN_SMOKE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            RUN_SMOKE=true
            shift
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

UV="${UV:-}"
if [[ -z "$UV" ]]; then
    if command -v uv >/dev/null 2>&1; then
        UV="$(command -v uv)"
    elif [[ -x "$HOME/.local/bin/uv" ]]; then
        UV="$HOME/.local/bin/uv"
    elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
        UV="$HOME/.cargo/bin/uv"
    else
        UV="uv"
    fi
fi

if ! "$UV" --version >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/ and rerun this script." >&2
    exit 1
fi

if [[ ! -d .venv ]]; then
    "$UV" venv
fi

"$UV" sync

mkdir -p \
    data/dataset \
    data/checkpoints \
    data/models \
    data/results \
    data/attack_results \
    data/paper_data

echo "Setup complete."

if [[ "$RUN_SMOKE" == true ]]; then
    ./scripts/eval.sh --mode test --detector ojha_sd14 --dataset synthbuster_10 --num_samples 1 --batch_size 4 --num_workers 0
    ./scripts/adv_examples.sh --detector ojha_sd14 --dataset synthbuster_10 --attack pgd --eps 4 --samples 1 --batch-size 2 --workers 0 --skip-images
fi
