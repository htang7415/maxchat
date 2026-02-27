#!/bin/bash
set -euo pipefail

# Multi-node speedrun launcher.
# Intended default: 2 nodes x 4 GPUs/node = 8 GPUs total.
#
# Launch on node 0:
#   NODE_RANK=0 MASTER_ADDR=<node0-hostname> bash runs/speedrun_multinode.sh
# Launch on node 1:
#   NODE_RANK=1 MASTER_ADDR=<node0-hostname> bash runs/speedrun_multinode.sh
#
# Optional overrides:
#   NNODES=2 NPROC_PER_NODE=4 MASTER_PORT=29500 WANDB_RUN=myrun bash runs/speedrun_multinode.sh

log() {
    echo "[speedrun-mn][$(date '+%Y-%m-%d %H:%M:%S')][node_rank=${NODE_RANK:-?}] $*"
}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Prefer scratch for large temporary caches when available.
if [ -z "${NANOCHAT_BASE_DIR:-}" ]; then
    if [ -n "${SCRATCH:-}" ] && [ -w "${SCRATCH}" ]; then
        NANOCHAT_BASE_DIR="${SCRATCH%/}/nanochat"
    elif [ -w "/kfs3/scratch/$USER" ]; then
        NANOCHAT_BASE_DIR="/kfs3/scratch/$USER/nanochat"
    else
        NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
    fi
fi
export NANOCHAT_BASE_DIR
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NANOCHAT_BASE_DIR/uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$NANOCHAT_BASE_DIR/uv_python}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$NANOCHAT_BASE_DIR/xdg_cache}"
if [ -z "${UV_PROJECT_ENVIRONMENT:-}" ]; then
    UV_PROJECT_ENVIRONMENT="$(pwd)/.venv"
fi
export UV_PROJECT_ENVIRONMENT
mkdir -p "$NANOCHAT_BASE_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME"

# Distributed launch config.
export NNODES="${NNODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
if [ -z "${NODE_RANK:-}" ]; then
    if [ -n "${SLURM_NODEID:-}" ]; then
        NODE_RANK="$SLURM_NODEID"
    else
        echo "ERROR: NODE_RANK is not set. Use NODE_RANK=0 on node0 and NODE_RANK=1 on node1."
        exit 1
    fi
fi
export NODE_RANK
export MASTER_PORT="${MASTER_PORT:-29500}"
if [ -z "${MASTER_ADDR:-}" ]; then
    if [ -n "${SLURM_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then
        MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
    else
        echo "ERROR: MASTER_ADDR is not set."
        echo "Set MASTER_ADDR to node0 hostname/IP, e.g. MASTER_ADDR=node0.example.org"
        exit 1
    fi
fi
export MASTER_ADDR

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi
export WANDB_RUN

log "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
log "NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# install uv (if not already installed)
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# One-time prep and synchronization between nodes.
SYNC_DIR="$NANOCHAT_BASE_DIR/multinode_sync"
PREP_DONE_FILE="$SYNC_DIR/prep.done"
PREP_FAIL_FILE="$SYNC_DIR/prep.failed"
IDENTITY_DONE_FILE="$SYNC_DIR/identity.done"
IDENTITY_FAIL_FILE="$SYNC_DIR/identity.failed"
mkdir -p "$SYNC_DIR"

if [ "$NODE_RANK" = "0" ]; then
    rm -f "$PREP_DONE_FILE" "$PREP_FAIL_FILE" "$IDENTITY_DONE_FILE" "$IDENTITY_FAIL_FILE"
    trap 'touch "$PREP_FAIL_FILE"' ERR
    log "Running one-time setup on node_rank=0 (venv/deps/dataset/tokenizer)"

    [ -d "$UV_PROJECT_ENVIRONMENT" ] || uv venv "$UV_PROJECT_ENVIRONMENT"
    uv sync --extra gpu
    # shellcheck source=/dev/null
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"

    python -m nanochat.report reset
    python -m nanochat.dataset -n 8
    python -m nanochat.dataset -n 370 &
    DATASET_DOWNLOAD_PID=$!
    python -m scripts.tok_train
    python -m scripts.tok_eval
    log "Waiting for background dataset download to complete..."
    wait "$DATASET_DOWNLOAD_PID"

    touch "$PREP_DONE_FILE"
    trap - ERR
else
    log "Waiting for node_rank=0 setup to complete..."
    while [ ! -f "$PREP_DONE_FILE" ]; do
        if [ -f "$PREP_FAIL_FILE" ]; then
            log "Detected setup failure on node_rank=0; exiting."
            exit 1
        fi
        sleep 10
    done

    [ -d "$UV_PROJECT_ENVIRONMENT" ] || {
        log "Missing virtual environment at $UV_PROJECT_ENVIRONMENT"
        exit 1
    }
    # shellcheck source=/dev/null
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
fi

TR_ARGS=(
    --nnodes="$NNODES"
    --nproc_per_node="$NPROC_PER_NODE"
    --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
)

# Base model pretraining + eval.
torchrun "${TR_ARGS[@]}" -m scripts.base_train -- --depth=26 --target-param-data-ratio=8.25 --device-batch-size=16 --fp8 --run="$WANDB_RUN"
torchrun "${TR_ARGS[@]}" -m scripts.base_eval -- --device-batch-size=16

# Download identity conversations once.
if [ "$NODE_RANK" = "0" ]; then
    trap 'touch "$IDENTITY_FAIL_FILE"' ERR
    curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    touch "$IDENTITY_DONE_FILE"
    trap - ERR
else
    while [ ! -f "$IDENTITY_DONE_FILE" ]; do
        if [ -f "$IDENTITY_FAIL_FILE" ]; then
            log "Detected identity dataset download failure on node_rank=0; exiting."
            exit 1
        fi
        sleep 5
    done
fi

# SFT + eval.
torchrun "${TR_ARGS[@]}" -m scripts.chat_sft -- --device-batch-size=16 --run="$WANDB_RUN"
torchrun "${TR_ARGS[@]}" -m scripts.chat_eval -- -i sft

# Generate report once.
if [ "$NODE_RANK" = "0" ]; then
    python -m nanochat.report generate
fi

