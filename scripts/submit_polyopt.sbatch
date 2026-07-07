#!/bin/bash
#SBATCH --job-name=polyopt-certify
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Certified lower bounds for tensor decompositions (CPU-only; MOSEK).
#
# Usage (all polyopt-certify flags pass through):
#   sbatch scripts/submit_polyopt.sbatch --case karatsuba --rank 2 --method cutting-plane
#   sbatch scripts/submit_polyopt.sbatch --case 2,2,2 --rank 6 --method cutting-plane --cp-max-iters 100
#
# One-time setup on the cluster:
#   1. conda activate fast-matmul-search && pip install -e ".[mosek]"
#   2. copy your MOSEK license to $HOME/mosek/mosek.lic
#      (personal academic licenses are user-locked, not machine-locked;
#      on Windows it lives at %USERPROFILE%\mosek\mosek.lic)
#   3. verify:  python -c "import mosek; mosek.Env().checkoutlicense(mosek.feature.pton); print('license OK')"

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results/certificates

echo "========== Job Info =========="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Args: $*"
echo "================================"

module load miniforge
conda activate fast-matmul-search

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1   # progress lines reach the log immediately

python -c "import mosek; mosek.Env().checkoutlicense(mosek.feature.pton); print('MOSEK license OK')"

# console script installed by `pip install -e .` (see pyproject [project.scripts])
polyopt-certify "$@" --out "results/certificates/${SLURM_JOB_ID}.json"

echo "End: $(date)"
