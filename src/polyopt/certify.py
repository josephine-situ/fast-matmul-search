"""Root-node certification driver.

Pipeline: polynomial over [-B, B]^n  ->  shift to [0,1]^n  ->  multiplier
family  ->  (optional) strict-feasibility check  ->  dual best-SLC SDP  ->
certified lower bound. The bound is invariant under the reparametrization,
so it applies directly to the original box.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from polyopt.multipliers import MultiplierPair, full_family
from polyopt.relaxation import build_relaxation, solve_relaxation_cvxpy
from polyopt.slc_constraints import build_matching, feasibility_check
from polyopt.sparse_poly import SparsePolynomial
from polyopt.upper_bounds import candidate_points, multistart_upper_bound


@dataclass
class CertifyResult:
    bound: float | None            # certified lower bound of p over the box
    status: str
    slater_gamma: float | None     # strict-feasibility margin (None = skipped)
    upper_bound: float | None
    n_pairs: int
    n_lambda: int
    n_moments: int
    times: dict = field(default_factory=dict)
    x_box: np.ndarray | None = None   # relaxation first moment, original coords

    @property
    def gap(self) -> float | None:
        if self.bound is None or self.upper_bound is None:
            return None
        return self.upper_bound - self.bound


def estimate_problem_size(
    poly: SparsePolynomial,
    n_vars: int,
    pairs: list[MultiplierPair],
    box: float = 1.0,
    sym_box: bool = True,
    top_lmi: bool = True,
) -> dict:
    """Build the matching + relaxation data WITHOUT solving and report
    sizes, so cluster jobs can be dimensioned before submission."""
    from polyopt.relaxation import build_relaxation
    import time as _time

    t0 = _time.perf_counter()
    shifted = (poly.substitute_affine(box, 0.0) if sym_box
               else poly.substitute_affine(2.0 * box, -box))
    matching = build_matching(pairs, shifted, sym_box=sym_box)
    t_match = _time.perf_counter() - t0
    t0 = _time.perf_counter()
    data = build_relaxation(matching, n_vars, top_lmi=top_lmi)
    t_assemble = _time.perf_counter() - t0

    supp_sizes = [len(t.supp) for t in pairs]
    # dual method: per pair one (k+1)x(k+1) PSD block + one kxk dual-PSD
    psd_scalars = sum(
        (k + 1) * (k + 2) // 2 + k * (k + 1) // 2 for k in supp_sizes
    )
    return {
        "n_pairs": len(pairs),
        "max_supp": max(supp_sizes),
        "n_lambda": matching.n_constraints,
        "n_moments": len(data.moments),
        "n_rlt_rows": int(data.rlt_rows.shape[0]),
        "dual_psd_scalars": psd_scalars,
        "build_seconds": round(t_match + t_assemble, 1),
    }


def certify_box_minimum(
    poly: SparsePolynomial,
    n_vars: int,
    box: float = 1.0,
    pairs: list[MultiplierPair] | None = None,
    check_slater: bool = True,
    compute_upper: bool = True,
    solver: str = "CLARABEL",
    top_lmi: bool = True,
    sym_box: bool = False,
    extra_cuts: list | None = None,
    method: str = "dual",
    cp_options: dict | None = None,
    threads: int = 0,
    solver_params: dict | None = None,
    verbose: bool = False,
) -> CertifyResult:
    """Certified lower bound for min p(x) over x in [-box, box]^n.

    `poly` is expressed in the original coordinates. If `pairs` is None the
    full family of order (degree - 2) is used - only viable for small n.
    With sym_box=True the engine works natively on [-1,1]^n with
    multipliers (1+y_i), (1-y_j) - this preserves the polynomial's sign
    structure (no coefficient explosion from shifting), which matters for
    restricted families on structured polynomials.

    method="dual" solves the monolithic dual SDP; method="cutting-plane"
    runs the memory-light adversarial loop (MOSEK only; the bound is
    valid even when stopped before convergence). cp_options are passed
    to solve_cutting_plane (max_iters, tol, rho, threads).
    """
    times: dict[str, float] = {}
    t0 = time.perf_counter()
    if sym_box:
        shifted = poly.substitute_affine(box, 0.0)      # q(y) = p(B y)
        lo, hi = -1.0, 1.0
    else:
        shifted = poly.substitute_affine(2.0 * box, -box)  # q(y) = p(2B y - B)
        lo, hi = 0.0, 1.0
    times["shift"] = time.perf_counter() - t0

    degree = shifted.degree
    if pairs is None:
        pairs = full_family(n_vars, max(degree - 2, 0))

    if verbose:
        print(f"building coefficient matching for {len(pairs)} pairs...",
              flush=True)
    t0 = time.perf_counter()
    matching = build_matching(pairs, shifted, sym_box=sym_box)
    times["matching"] = time.perf_counter() - t0
    if verbose:
        print(f"  {matching.n_constraints} matching equalities "
              f"({times['matching']:.1f}s)", flush=True)

    use_mosek = solver.upper() == "MOSEK"
    gamma = None
    if method == "cutting-plane":
        check_slater = False  # the cutting-plane loop checks it itself
    if check_slater:
        t0 = time.perf_counter()
        if use_mosek:
            from polyopt.mosek_backend import feasibility_check_mosek

            gamma = feasibility_check_mosek(matching)
        else:
            gamma, _ = feasibility_check(matching, solver=solver)
        times["slater"] = time.perf_counter() - t0
        if not np.isfinite(gamma) or gamma <= 0:
            return CertifyResult(
                bound=None,
                status=f"slater_failed (gamma={gamma})",
                slater_gamma=gamma,
                upper_bound=None,
                n_pairs=len(pairs),
                n_lambda=matching.n_constraints,
                n_moments=0,
                times=times,
            )

    if verbose:
        print("assembling relaxation data...", flush=True)
    t0 = time.perf_counter()
    data = build_relaxation(matching, n_vars, top_lmi=top_lmi)
    times["assemble"] = time.perf_counter() - t0
    if verbose:
        print(f"  {len(data.moments)} moments, "
              f"{data.rlt_rows.shape[0]} RLT rows "
              f"({times['assemble']:.1f}s); solving ({method})...",
              flush=True)

    extra_ineq = None
    if extra_cuts:
        from polyopt.symmetry import cuts_to_rows

        extra_ineq = cuts_to_rows(extra_cuts, data.moments)

    t0 = time.perf_counter()
    if method == "cutting-plane":
        if not use_mosek:
            raise ValueError("cutting-plane mode requires solver='MOSEK'")
        from polyopt.cutting_plane import solve_cutting_plane

        cp_kwargs = dict(cp_options or {})
        cp_kwargs.setdefault("threads", threads)
        sol = solve_cutting_plane(
            data, top_lmi=top_lmi, extra_ineq=extra_ineq,
            verbose=verbose, **cp_kwargs,
        )
    elif use_mosek:
        from polyopt.mosek_backend import solve_relaxation_mosek

        sol = solve_relaxation_mosek(
            data, top_lmi=top_lmi, verbose=verbose, extra_ineq=extra_ineq,
            threads=threads, solver_params=solver_params,
        )
    else:
        sol = solve_relaxation_cvxpy(
            data, solver=solver, top_lmi=top_lmi, verbose=verbose,
            extra_ineq=extra_ineq,
        )
    times["solve"] = time.perf_counter() - t0
    if "n_moments" not in sol:
        sol["n_moments"] = len(data.moments)

    upper = None
    x_box = None
    if compute_upper:
        t0 = time.perf_counter()
        x0_list = []
        if "x" in sol and "X" in sol:
            x0_list = candidate_points(sol["x"], sol["X"])
        upper, y_best = multistart_upper_bound(
            shifted, n_vars, lo, hi, x0_list=x0_list
        )
        times["upper"] = time.perf_counter() - t0
        if y_best is not None:
            x_box = box * y_best if sym_box else 2.0 * box * y_best - box
    elif "x" in sol:
        x_box = box * sol["x"] if sym_box else 2.0 * box * sol["x"] - box

    return CertifyResult(
        bound=sol["bound"],
        status=sol["status"],
        slater_gamma=sol.get("slater_gamma", gamma),
        upper_bound=upper,
        n_pairs=len(pairs),
        n_lambda=matching.n_constraints,
        n_moments=sol["n_moments"],
        times=times,
        x_box=x_box,
    )
