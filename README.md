# fast-matmul-search

Search and **certified bounds** for fast matrix multiplication tensor decompositions
`⟨m,p,n⟩`: finding rank-R decompositions `T = Σ_r u_r ⊗ v_r ⊗ w_r` of the matmul
structure tensor, and proving when they cannot exist.

Two engines:

1. **Continuous search** (`src/continuous_search.py`, `src/run_experiments.py`) —
   gradient descent / ALS over factor entries, with flip-graph hybrid moves
   (`src/flip_graph_hybrid.py`) and finite-field lifting (`src/finite_field_search.py`).
   Finds candidate decompositions; proves nothing.

2. **Exact polynomial optimization** (`src/polyopt/`) — the decomposition loss
   `||T − Σ_r u_r⊗v_r⊗w_r||²` is a degree-6 polynomial in the factor entries.
   Following Bertsimas–den Hertog–Koukouvinos (arXiv:2507.02120, see `docs/`),
   we compute the *best sum-of-linear-times-convex (SLC) decomposition* of that
   polynomial via an SDP dual, giving a **certified lower bound** on the global
   minimum over a box `[−B, B]^n`. A strictly positive bound proves
   "no rank-R decomposition with entries in [−B, B] exists". Spatial branch &
   bound (with the continuous search as upper-bound oracle) closes gaps and
   recovers exact solutions.

## Usage

Continuous search (unchanged):

```bash
python -m src.run_experiments --quick     # smoke run
python -m src.run_experiments             # full batch (see batch_results/)
```

Certified bounds:

```bash
polyopt-validate --stage deg3             # validation ladder (deg3/deg4/deg6/...)
polyopt-certify --case 2,2,2 --rank 6 --box 1.0   # headline nonachievability run
polyopt-bb --case 2,2,2 --rank 7          # branch & bound search
```

Solvers: open-source Clarabel/SCS work out of the box (via cvxpy); MOSEK
(`pip install .[mosek]` + academic license) is strongly recommended at scale.

Two solve methods: `--method dual` builds the monolithic dual SDP (tightest,
memory-hungry); `--method cutting-plane` alternates a moment-space master
with an adversarial pessimization over decompositions (paper Remark 4) —
far lighter in memory, and its bound is valid even when stopped before
convergence.

## Running on the SLURM cluster

MOSEK is pip-installable (no module needed) and personal academic licenses
are user-locked, not machine-locked, so your laptop license works on the
cluster. One-time setup:

```bash
# on the cluster, inside the repo
module load miniforge && conda activate fast-matmul-search
pip install -e ".[mosek]"
# copy your license (from %USERPROFILE%\mosek\mosek.lic on Windows):
mkdir -p ~/mosek && scp <laptop>:mosek/mosek.lic ~/mosek/mosek.lic
python -c "import mosek; mosek.Env().checkoutlicense(mosek.feature.pton); print('license OK')"
```

Then submit (flags pass through to `polyopt-certify`):

```bash
sbatch scripts/submit_polyopt.sbatch --case karatsuba --rank 2 --method cutting-plane
sbatch scripts/submit_polyopt.sbatch --case 2,2,2 --rank 6 --method cutting-plane --cp-max-iters 100
```

Results land in `results/certificates/<jobid>.json`; logs in `logs/`.
If no MOSEK license is available, `--solver CLARABEL --method dual` runs
on open-source solvers (much slower, smaller cases only).

## Structure

- `src/`: continuous-search modules (flat) and the `polyopt/` package
- `scripts/`: runnable helpers and analysis entry points
- `batch_results/`, `results/`: recorded outputs
- `docs/`: the polynomial optimization paper, degree-6 theorem note, and the
  original (reference-only) Julia implementation for degrees 3–4
- `tests/`: pytest suites (`tests/polyopt/` for the certification engine)
