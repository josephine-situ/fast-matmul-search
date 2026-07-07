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

## Structure

- `src/`: continuous-search modules (flat) and the `polyopt/` package
- `scripts/`: runnable helpers and analysis entry points
- `batch_results/`, `results/`: recorded outputs
- `docs/`: the polynomial optimization paper, degree-6 theorem note, and the
  original (reference-only) Julia implementation for degrees 3–4
- `tests/`: pytest suites (`tests/polyopt/` for the certification engine)
