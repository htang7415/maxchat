#!/bin/bash
set -Eeuo pipefail

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

log() { echo "[speedrun] $*"; }
die() { echo "[speedrun] ERROR: $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

check_writable_dir() {
    local dir="$1"
    mkdir -p "$dir"
    local probe="$dir/.write_test_$$"
    : > "$probe" || die "Cannot write to $dir (check permissions/quota)"
    rm -f "$probe"
}

check_free_space_gb() {
    local dir="$1"
    local min_gb="$2"
    local free_kb
    free_kb=$(df -Pk "$dir" | awk 'NR==2 {print $4}')
    local min_kb=$((min_gb * 1024 * 1024))
    if (( free_kb < min_kb )); then
        die "Not enough free disk in $dir: $((free_kb / 1024 / 1024))GB available, need at least ${min_gb}GB"
    fi
    log "Free disk in $dir: $((free_kb / 1024 / 1024))GB"
}

check_cuda_gpus() {
    require_cmd nvidia-smi
    local visible
    visible=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    if (( visible < NPROC_PER_NODE )); then
        die "Need at least NPROC_PER_NODE=${NPROC_PER_NODE} visible GPUs, found ${visible}. Run this on a GPU compute node."
    fi
    log "Detected ${visible} visible GPU(s)"
}

check_pytorch_cuda() {
    python - "$NPROC_PER_NODE" <<'PY'
import sys
import torch
need = int(sys.argv[1])
count = torch.cuda.device_count() if torch.cuda.is_available() else 0
if count < need:
    raise SystemExit(f"PyTorch CUDA check failed: need >= {need} GPUs, found {count}.")
print(f"[speedrun] PyTorch CUDA OK ({count} GPU(s) visible)")
PY
}

check_checkpoint_dir() {
    local ckpt_dir="$1"
    local stage_name="$2"
    if [[ ! -d "$ckpt_dir" ]]; then
        die "${stage_name} checkpoint directory is missing: $ckpt_dir"
    fi
    shopt -s nullglob
    local model_files=("$ckpt_dir"/model_*.pt)
    shopt -u nullglob
    if (( ${#model_files[@]} == 0 )); then
        die "${stage_name} checkpoint directory has no model_*.pt files: $ckpt_dir"
    fi
    log "${stage_name} checkpoints verified: $ckpt_dir"
}

download_with_retry() {
    local url="$1"
    local out="$2"
    curl --fail --location --retry 5 --retry-delay 2 --retry-all-errors --continue-at - --output "$out" "$url"
}

DATASET_DOWNLOAD_PID=""
cleanup_background_jobs() {
    if [[ -n "${DATASET_DOWNLOAD_PID:-}" ]] && kill -0 "$DATASET_DOWNLOAD_PID" 2>/dev/null; then
        log "Stopping background dataset download (pid ${DATASET_DOWNLOAD_PID})"
        kill "$DATASET_DOWNLOAD_PID" || true
        wait "$DATASET_DOWNLOAD_PID" 2>/dev/null || true
    fi
}
trap cleanup_background_jobs EXIT

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BASE_MODEL_TAG="${BASE_MODEL_TAG:-d26}"
NANOCHAT_MIN_FREE_GB="${NANOCHAT_MIN_FREE_GB:-80}"

[[ -f "pyproject.toml" ]] || die "Run this script from the repository root (missing pyproject.toml)"
check_writable_dir "$NANOCHAT_BASE_DIR"
check_free_space_gb "$NANOCHAT_BASE_DIR" "$NANOCHAT_MIN_FREE_GB"
check_cuda_gpus

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
if ! command -v uv >/dev/null 2>&1; then
    require_cmd curl
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate
check_pytorch_cuda

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "${WANDB_RUN:-}" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 350 shards are needed for 10B tokens of data for pretraining.
# The maximum total number of shards available in the entire dataset is 1822.
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"
DATASET_DOWNLOAD_PID=""

# d26 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8.25)
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- --depth=26 --model-tag="$BASE_MODEL_TAG" --target-param-data-ratio=8.25 --device-batch-size=16 --fp8 --run="$WANDB_RUN"
BASE_CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$BASE_MODEL_TAG"
check_checkpoint_dir "$BASE_CKPT_DIR" "Base training"
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- --model-tag="$BASE_MODEL_TAG" --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
check_free_space_gb "$NANOCHAT_BASE_DIR" 1
download_with_retry "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl" "$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
[[ -s "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]] || die "identity_conversations.jsonl is missing or empty after download"

# run SFT and eval the model
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- --model-tag="$BASE_MODEL_TAG" --device-batch-size=16 --run="$WANDB_RUN"
SFT_CKPT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$BASE_MODEL_TAG"
check_checkpoint_dir "$SFT_CKPT_DIR" "Chat SFT"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft -g "$BASE_MODEL_TAG"

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
