#!/bin/bash
#SBATCH --job-name=fast-matmul-or-val
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Over-rank VALIDATION: cases where batch_results/experiment_log.json already
# found exact decompositions (gradient / other methods). Re-runs Formulation B'
# to compare hit rate vs baseline.
#
#   sbatch scripts/submit_overrank_validate.sbatch
#   sbatch --export=OVER_RANK_MAX_TIER=1,OVER_RANK_RESTARTS=80 scripts/submit_overrank_validate.sbatch
#
# Logs:
#   batch_results/overrank_restarts_validate.jsonl
#   batch_results/overrank_experiment_log_validate.json

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs batch_results

export OVER_RANK_QUEUE=validate
export OVER_RANK_MAX_WALL_HOURS="${OVER_RANK_MAX_WALL_HOURS:-5.5}"
# Tier 0–1 validate only (6 cases) fits ~6h; set OVER_RANK_MAX_TIER=2 to add sparsity tier.
export OVER_RANK_MAX_TIER="${OVER_RANK_MAX_TIER:-1}"
export OVER_RANK_RESTARTS="${OVER_RANK_RESTARTS:-100}"

echo "========== Over-rank VALIDATION =========="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "OVER_RANK_MAX_TIER=${OVER_RANK_MAX_TIER}"
echo "OVER_RANK_RESTARTS=${OVER_RANK_RESTARTS}"
echo "OVER_RANK_MAX_WALL_HOURS=${OVER_RANK_MAX_WALL_HOURS}"
echo "=========================================="

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif [[ -f .venv/Scripts/activate ]]; then
  source .venv/Scripts/activate
else
  module load miniforge
  conda activate fast-matmul-search
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python -m pip install -e . -q
python -u -m run_experiments --overrank --overrank-queue validate

echo "End: $(date)"
