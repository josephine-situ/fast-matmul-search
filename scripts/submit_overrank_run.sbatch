#!/bin/bash
#SBATCH --job-name=fast-matmul-overrank
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Formulation B' batch (define_experiments tiers 0→3).
#
#   sbatch scripts/submit_overrank_run.sbatch
#   sbatch --export=MODE=quick scripts/submit_overrank_run.sbatch
#   sbatch --export=MODE=steps_matched scripts/submit_overrank_run.sbatch
#
# Logs: batch_results/overrank_restarts.jsonl, overrank_experiment_log.json
# Baseline comparison: batch_results/experiment_log.json

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs batch_results

echo "========== Job Info =========="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "PWD: $PWD"
echo "MODE: ${MODE:-full}"
echo "================================"

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

ARGS=(--overrank)
case "${MODE:-full}" in
  quick)          ARGS+=(--quick) ;;
  steps_matched)  ARGS+=(--steps-matched) ;;
  full)           ;;
  *) echo "Unknown MODE=${MODE}; use full, quick, or steps_matched" >&2; exit 1 ;;
esac

python -u -m run_experiments "${ARGS[@]}"

echo "End: $(date)"
