#!/bin/bash
#SBATCH --account=nawimem
#SBATCH --time=1-00:00:00
#SBATCH --job-name=maxchat
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=256G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4

# NREL Slurm submit script for nanochat 2-node speedrun.
# Usage:
#   sbatch runs/submit_speedrun_multinode_nrel.sh
# Optional overrides at submit time:
#   sbatch --partition=gpu-h100 --qos=normal \
#     --export=ALL,WANDB_RUN=myrun,MASTER_PORT=29500,NANOCHAT_BASE_DIR=/kfs3/scratch/$USER/nanochat \
#     runs/submit_speedrun_multinode_nrel.sh

set -euo pipefail

JOB_START_TS=$(date +%s)
echo "Job Start Time: $(date)"

finish() {
    local status=$?
    local end_ts
    end_ts=$(date +%s)
    echo "Job End Time: $(date)"
    echo "Job Duration Seconds: $((end_ts - JOB_START_TS))"
    echo "Job Exit Status: ${status}"
}
trap finish EXIT

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"
mkdir -p logs

NNODES="${SLURM_NNODES:-2}"
export NNODES
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WANDB_RUN="${WANDB_RUN:-speedrun_${SLURM_JOB_ID:-manual}}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/kfs3/scratch/$USER/nanochat}"

echo "Workdir: $WORKDIR"
echo "DDP config: nnodes=${NNODES}, nproc_per_node=${NPROC_PER_NODE}, total_gpus=$((NNODES * NPROC_PER_NODE))"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"
echo "WANDB_RUN=${WANDB_RUN}"
echo "NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR}"

# Launch one controller process per node.
# runs/speedrun_multinode.sh will derive NODE_RANK from SLURM_NODEID and run torchrun.
srun \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  --chdir="$WORKDIR" \
  bash runs/speedrun_multinode.sh
