"""Command-line entry points for the polyopt certification engine."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def _build_case(case: str, rank: int):
    """Return (loss polynomial, variables, human name) for a named case."""
    from polyopt.matmul_poly import (
        build_decomposition_loss,
        build_matmul_loss,
        build_polymult_tensor,
    )

    if case == "karatsuba":
        poly, var = build_decomposition_loss(build_polymult_tensor(2, 2), rank)
        return poly, var, f"polymult<2,2> rank {rank}"
    dims = tuple(int(t) for t in case.split(","))
    if len(dims) != 3:
        raise SystemExit(f"--case must be 'm,p,n' or 'karatsuba', got {case}")
    poly, var = build_matmul_loss(*dims, rank)
    return poly, var, f"<{dims[0]},{dims[1]},{dims[2]}> rank {rank}"


def main_certify(argv=None):
    from polyopt.certify import certify_box_minimum
    from polyopt.multipliers import (
        MultiplierPair,
        full_family,
        support_driven_family,
    )

    ap = argparse.ArgumentParser(
        description="Certified lower bound for a tensor decomposition loss"
    )
    ap.add_argument("--case", required=True,
                    help="'m,p,n' for matmul or 'karatsuba'")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--box", type=float, default=1.0,
                    help="factor entries constrained to [-box, box]")
    ap.add_argument("--solver", default="MOSEK")
    ap.add_argument("--method", choices=["dual", "cutting-plane"],
                    default="dual",
                    help="monolithic dual SDP or memory-light "
                         "adversarial loop (bound valid on early stop)")
    ap.add_argument("--cp-max-iters", type=int, default=100)
    ap.add_argument("--cp-tol", type=float, default=1e-4)
    ap.add_argument("--cp-rho", type=float, default=10.0,
                    help="coefficient bound for pessimization scenarios "
                         "(smaller = faster convergence, weaker limit)")
    ap.add_argument("--max-flips", type=int, default=None,
                    help="cap complement flips in the multiplier family "
                         "(default: all flips - needed for strong bounds)")
    ap.add_argument("--family", choices=["support", "full"],
                    default="support",
                    help="'support': pairs drawn from the loss monomials "
                         "(scalable); 'full': every (I,J) pair of order "
                         "degree-2 (strongest known bounds, O(n^4) pairs "
                         "- cluster-sized memory)")
    ap.add_argument("--supp", choices=["monomial", "full"],
                    default="monomial",
                    help="quadratic support per pair: variables of the "
                         "originating monomial, or all n variables "
                         "(wider = tighter and much more memory)")
    ap.add_argument("--no-slater", action="store_true",
                    help="skip the strict-feasibility check (not recommended)")
    ap.add_argument("--no-upper", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the problem and report sizes without "
                         "solving - use to dimension cluster jobs")
    ap.add_argument("--threads", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)),
                    help="MOSEK thread cap (default: SLURM_CPUS_PER_TASK "
                         "if set, else all cores; MOSEK sees physical "
                         "cores, not the cgroup limit)")
    ap.add_argument("--mosek-param", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="extra MOSEK parameter for the dual solve, "
                         "repeatable (e.g. intpntSolveForm=dual, "
                         "presolveUse=off)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    poly, var, name = _build_case(args.case, args.rank)
    print(f"case {name}: {var.n_vars} variables, "
          f"{len(poly)} monomials, degree {poly.degree}")

    if args.family == "full":
        pairs = full_family(var.n_vars, max(poly.degree - 2, 0))
    else:
        pairs = support_driven_family(
            list(poly.coeffs.keys()), complement_splits=True,
            max_flips=args.max_flips,
        )
        if args.supp == "full":
            full_supp = tuple(range(var.n_vars))
            pairs = [
                MultiplierPair(I, J, full_supp)
                for (I, J) in sorted({(p.I, p.J) for p in pairs})
            ]
    max_supp = max(len(t.supp) for t in pairs)
    print(f"multiplier family: {args.family}/{args.supp}, "
          f"{len(pairs)} pairs, max support {max_supp} "
          f"(largest LMI {max_supp + 1}x{max_supp + 1})")

    if args.dry_run:
        from polyopt.certify import estimate_problem_size

        est = estimate_problem_size(
            poly, var.n_vars, pairs, box=args.box, sym_box=True
        )
        print("dry run - problem sizes (no solve):")
        for k, v in est.items():
            print(f"  {k}: {v}")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(
                {"case": name, "rank": args.rank, "dry_run": True, **est},
                indent=2))
            print(f"wrote {args.out}")
        return

    cp_options = {"max_iters": args.cp_max_iters, "tol": args.cp_tol,
                  "rho": args.cp_rho}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cp_options["checkpoint_path"] = str(
            args.out.with_suffix(".progress.json")
        )

    t0 = time.time()
    res = certify_box_minimum(
        poly, n_vars=var.n_vars, box=args.box, pairs=pairs,
        solver=args.solver, sym_box=True,
        check_slater=not args.no_slater,
        compute_upper=not args.no_upper,
        method=args.method,
        cp_options=cp_options,
        threads=args.threads,
        solver_params=dict(p.split("=", 1) for p in args.mosek_param),
        # cutting-plane runs are long: always show iteration progress
        verbose=args.verbose or args.method == "cutting-plane",
    )
    elapsed = time.time() - t0

    print(f"status: {res.status}   slater gamma: {res.slater_gamma}")
    print(f"CERTIFIED LOWER BOUND: {res.bound}")
    if res.upper_bound is not None:
        print(f"best upper bound found: {res.upper_bound}")
    if res.bound is not None and res.bound > 0:
        print(f"==> no rank-{args.rank} decomposition with entries in "
              f"[-{args.box}, {args.box}] exists (loss >= {res.bound:.6g})")
    print(f"sizes: pairs={res.n_pairs} lambda={res.n_lambda} "
          f"moments={res.n_moments}   elapsed {elapsed:.1f}s")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "case": name, "rank": args.rank, "box": args.box,
            "bound": res.bound, "upper_bound": res.upper_bound,
            "status": res.status, "slater_gamma": res.slater_gamma,
            "n_pairs": res.n_pairs, "n_lambda": res.n_lambda,
            "n_moments": res.n_moments, "times": res.times,
            "solver": args.solver, "method": args.method,
            "family": args.family, "supp": args.supp,
            "max_flips": args.max_flips,
        }, indent=2))
        print(f"wrote {args.out}")


def main_validate(argv=None):
    from polyopt.certify import certify_box_minimum

    ap = argparse.ArgumentParser(description="Run the validation ladder")
    ap.add_argument("--stage", choices=["deg3", "deg4", "deg6"],
                    default="deg3")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--solver", default="MOSEK")
    args = ap.parse_args(argv)

    degree = {"deg3": 3, "deg4": 4, "deg6": 6}[args.stage]
    n = args.n or {"deg3": 8, "deg4": 5, "deg6": 3}[args.stage]
    rng = np.random.default_rng(0)

    from tests.polyopt.test_slc_degree3 import random_dense_poly

    print(f"{'trial':>5} {'LB':>14} {'UB':>14} {'gap':>10} {'time':>7}")
    for trial in range(args.trials):
        poly = random_dense_poly(rng, n, degree)
        t0 = time.time()
        res = certify_box_minimum(
            poly, n_vars=n, box=1.0, solver=args.solver
        )
        print(f"{trial:>5} {res.bound:>14.6f} {res.upper_bound:>14.6f} "
              f"{res.gap:>10.2e} {time.time()-t0:>6.1f}s")


def main_bb(argv=None):
    from polyopt.branch_bound import branch_and_bound

    ap = argparse.ArgumentParser(
        description="Spatial branch & bound with certified node bounds"
    )
    ap.add_argument("--case", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--box", type=float, default=1.0)
    ap.add_argument("--target", type=float, default=0.0,
                    help="stop as soon as the global LB exceeds this "
                         "(0.0 = nonachievability certificate); pass nan "
                         "to run to gap closure")
    ap.add_argument("--max-nodes", type=int, default=200)
    ap.add_argument("--gap", type=float, default=1e-6)
    ap.add_argument("--solver", default="MOSEK")
    ap.add_argument("--threads", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    poly, var, name = _build_case(args.case, args.rank)
    print(f"case {name}: {var.n_vars} variables, "
          f"{len(poly)} monomials, degree {poly.degree}", flush=True)

    target = None if np.isnan(args.target) else args.target
    checkpoint = None
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = str(args.out.with_suffix(".progress.json"))

    t0 = time.time()
    res = branch_and_bound(
        poly, n_vars=var.n_vars, box=args.box, target=target,
        gap_tol=args.gap, max_nodes=args.max_nodes, solver=args.solver,
        threads=args.threads, checkpoint_path=checkpoint, verbose=True,
    )
    elapsed = time.time() - t0

    print(f"status: {res['status']}")
    print(f"certified LB: {res['lb']}   best UB: {res['ub']}")
    print(f"nodes: {res['nodes_solved']} solved, {res['open_nodes']} open"
          f"   elapsed {elapsed:.1f}s")
    if res["lb"] is not None and target is not None and res["lb"] > target:
        print(f"==> no rank-{args.rank} decomposition with entries in "
              f"[-{args.box}, {args.box}] exists (loss >= {res['lb']:.6g})")

    if args.out:
        out = {k: v for k, v in res.items() if k not in ("x_best",)}
        out["history"] = [[float(a), float(b)] for a, b in out["history"]]
        out.update({"case": name, "rank": args.rank, "box": args.box})
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
