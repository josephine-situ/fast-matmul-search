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
# Legacy wrapper — prefer the 6h split jobs (validation + discover):
#
#   sbatch scripts/submit_overrank_validate.sbatch
#   sbatch scripts/submit_overrank_discover.sbatch
#
# Quick local smoke test:
#   sbatch --export=MODE=quick scripts/submit_overrank_run.sbatch
#
# Preview queues without GPU:
#   python -m run_experiments --overrank --list-queue --overrank-queue validate
#   python -m run_experiments --overrank --list-queue --overrank-queue discover

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs batch_results

echo "========== Job Info =========="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "MODE: ${MODE:-validate}"
echo "Use submit_overrank_validate.sbatch + submit_overrank_discover.sbatch for production."
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

case "${MODE:-validate}" in
  quick)
    python -u -m run_experiments --overrank --quick --overrank-queue validate
    ;;
  discover)
    python -u -m run_experiments --overrank --overrank-queue discover
    ;;
  validate|full)
    python -u -m run_experiments --overrank --overrank-queue validate
    ;;
  *)
    echo "Unknown MODE=${MODE}; use validate, discover, or quick" >&2
    exit 1
    ;;
esac

echo "End: $(date)"
