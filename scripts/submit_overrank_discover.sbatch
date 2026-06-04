#!/bin/bash
#SBATCH --job-name=fast-matmul-or-disc
#SBATCH --partition=pi_dbertsim
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs batch_results

echo "========== Job Info =========="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "PWD: $PWD"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "================================"

module load miniforge
conda activate fast-matmul-search

export OVER_RANK_MAX_WALL_HOURS="${OVER_RANK_MAX_WALL_HOURS:-5.5}"
export OVER_RANK_MAX_TIER="${OVER_RANK_MAX_TIER:-3}"
export OVER_RANK_RESTARTS="${OVER_RANK_RESTARTS:-300}"
# Same total FLOPs as flops_matched; 80/15/5 main/refine/snap (see overrank_search.py).
export OVER_RANK_BUDGET_MODE="${OVER_RANK_BUDGET_MODE:-discover}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python -c "import socket, torch; assert torch.cuda.is_available(), 'CUDA not available on ' + socket.gethostname()"

python -u -m src.run_experiments --overrank --overrank-queue discover

echo "End: $(date)"
