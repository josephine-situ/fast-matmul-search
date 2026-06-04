#!/bin/bash
#SBATCH --job-name=fast-matmul-or-val
#SBATCH --partition=mit_normal_gpu
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
echo "================================"

module load miniforge
conda activate fast-matmul-search

export OVER_RANK_MAX_WALL_HOURS="${OVER_RANK_MAX_WALL_HOURS:-5.5}"
export OVER_RANK_MAX_TIER="${OVER_RANK_MAX_TIER:-1}"
export OVER_RANK_RESTARTS="${OVER_RANK_RESTARTS:-100}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python -u -m src.run_experiments --overrank --overrank-queue validate

echo "End: $(date)"
