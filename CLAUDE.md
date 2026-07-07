# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

Search and certified bounds for fast matrix multiplication tensor decompositions `⟨m,p,n⟩`: finding rank-R decompositions `T = Σ_r u_r ⊗ v_r ⊗ w_r`, and proving when they cannot exist. Two independent engines:

1. **Continuous search** (flat modules in `src/`): gradient descent / ALS over factor entries (`continuous_search.py`), flip-graph hybrid moves (`flip_graph_hybrid.py`), finite-field lifting (`finite_field_search.py`). Finds candidates; proves nothing.
2. **Exact polynomial optimization** (`src/polyopt/` package): the decomposition loss is a degree-6 polynomial; the engine computes the best sum-of-linear-times-convex (SLC) decomposition via an SDP dual (Bertsimas–den Hertog–Koukouvinos, arXiv:2507.02120, paper in `docs/`), giving a **certified lower bound** on the loss over a box. A strictly positive bound proves no rank-R decomposition with entries in `[−B, B]` exists.

## Commands

Environment is uv-managed (Python ≥3.12, `.venv` in repo). The venv lives inside OneDrive and file locks sometimes break syncs (`Access denied` on dist-info) — prefer `uv run --no-sync` once synced; retry if a sync fails.

```bash
# Tests (pytest; certificate tests marked "slow"). Must be `python -m pytest`
# from the repo root: several test modules import `from tests.polyopt ...`,
# so the cwd has to be on sys.path — bare `pytest` fails collection.
uv run --no-sync python -m pytest                       # full suite
uv run --no-sync python -m pytest -m "not slow"         # skip long-running certificates
uv run --no-sync python -m pytest tests/polyopt/test_slc_degree3.py -k <name>   # single test

# Certification engine (console scripts from pyproject)
uv run --no-sync polyopt-validate --stage deg3      # validation ladder: deg3/deg4/deg6
uv run --no-sync polyopt-certify --case 2,2,2 --rank 6 --box 1.0
uv run --no-sync polyopt-certify --case karatsuba --rank 2 --method cutting-plane
# polyopt-bb exists as an entry point but is NOT implemented (raises SystemExit)

# Continuous search
uv run --no-sync python -m src.run_experiments --quick    # smoke run
uv run --no-sync python -m src.run_experiments            # overnight batch -> batch_results/

# SLURM cluster (flags pass through to polyopt-certify; results -> results/certificates/<jobid>.json)
sbatch scripts/submit_polyopt.sbatch --case 2,2,2 --rank 6 --method cutting-plane --cp-max-iters 300
```

There is no linter or formatter configured.

## Solver facts (critical)

- **cvxpy cannot drive MOSEK 11** (`module 'mosek' has no attribute 'conetype'`). `solver="MOSEK"` in polyopt routes to the direct Fusion backend in `src/polyopt/mosek_backend.py` — never through cvxpy. cvxpy+CLARABEL is the open-source path, used for small problems and tests; Fusion is ~100–300× faster on large SLC matching systems.
- MOSEK needs an academic license (`~/mosek/mosek.lic`, user-locked, works on the cluster too). MOSEK out-of-memory shows up as `rescode.err_space(1051)`.

## Architecture of the certification pipeline (`src/polyopt/`)

Data flows one way:

```
matmul_poly.py      build the loss ||T − Σ u⊗v⊗w||² as a SparsePolynomial (sparse_poly.py)
multipliers.py      choose the (I,J)-multiset multiplier family (support_driven_family / full_family)
slc_constraints.py  coefficient-matching linear system + Slater feasibility_check
relaxation.py       monolithic dual SDP  (method="dual")
cutting_plane.py    level-bundle adversarial loop (method="cutting-plane", MOSEK only)
certify.py          orchestrates: shift → family → Slater → solve → CertifyResult
cli.py              polyopt-certify / polyopt-validate / polyopt-bb entry points
```

Domain invariants that must not be regressed (each was learned the hard way; see `tests/polyopt/`):

- **A certified bound is only valid if the matching system is strictly feasible (Slater)** — `feasibility_check` must return γ > 0. Skipping it (`--no-slater`) makes bounds unsound.
- **`sym_box=True` (native `[−1,1]` box) is essential for tensor losses**: the `[0,1]` shift destroys sign structure and makes restricted families infeasible.
- **All complement splits are required for bound strength**: `support_driven_family(..., complement_splits=True)` with unlimited flips ≈ full-family strength at ~1/3 the pairs; capping `max_flips=1` produces useless bounds.
- Cutting-plane mode's lower bound is **anytime-valid** (valid even on early stop). The level-bundle method is mandatory — plain Kelley zigzags and never converges.

Family knobs (`--family support|full`, `--supp monomial|full`) trade bound strength against memory; `full/full` is cluster-sized. Bound strength decays sharply with rank count — large cases (e.g. ⟨2,2,2⟩ rank 6, n=72) need the cluster and/or cutting-plane mode.

## Layout notes

- `pyproject.toml` maps `package-dir = {"" = "src"}`: the flat search modules are top-level modules (`import tensor_utils`, not `import src.tensor_utils`), and `polyopt` is a package. `polyopt-validate` imports from `tests/`, so run it from the repo root.
- `docs/` holds the reference paper (`poly_opt_paper.pdf`), the degree-6 theorem note (`general_thm.md`), and the original Julia degree-3/4 implementation (reference only — do not port from it blindly).
- `batch_results/`, `results/` are recorded experiment outputs; treat as data, not code.
