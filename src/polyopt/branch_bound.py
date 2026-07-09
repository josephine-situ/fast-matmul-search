"""Spatial branch & bound over the box, using the best-SLC root SDP as
the node bound.

Motivation: on tensor losses the root relaxation is loose because the
moment variables average over the many symmetric global minimizers
(rank-term permutations x sign flips). Splitting the box separates the
symmetric copies, so child relaxations tighten sharply - this is the
lever the multiplier-family axis cannot provide (measured: full family
vs support family differ by ~8%, while the rank-1 -> rank-2 collapse is
~15 units).

Design: best-first on node lower bounds; axis-aligned bisection on the
variable with the largest second-moment inconsistency X_ii - x_i^2 at
the node relaxation optimum; each node's polynomial is the original loss
affinely mapped from the node box onto [-1,1]^n, re-certified with a
fresh support-driven family. If a node's matching loses strict
feasibility, the node inherits its parent bound (valid: the parent bound
holds on a superset). The global lower bound is the minimum over open
and pruned leaves and is valid at every iteration; for nonachievability
runs the loop stops as soon as it exceeds `target`.
"""

from __future__ import annotations

import heapq
import itertools
import json
import time
from dataclasses import dataclass, field

import numpy as np

from polyopt.certify import certify_box_minimum
from polyopt.multipliers import support_driven_family
from polyopt.sparse_poly import SparsePolynomial
from polyopt.upper_bounds import poly_gradient


@dataclass(order=True)
class _Node:
    lb: float
    tiebreak: int
    lo: np.ndarray = field(compare=False)
    hi: np.ndarray = field(compare=False)
    depth: int = field(compare=False, default=0)
    x_hint: np.ndarray | None = field(compare=False, default=None)
    branch_scores: np.ndarray | None = field(compare=False, default=None)


def _certify_node(poly, n_vars, lo, hi, solver, check_slater, threads):
    """Map the node box onto [-1,1]^n and run the root certification.
    Returns (bound or None, x in original coords or None, X_diag or None).
    """
    a = (hi - lo) / 2.0
    b = (hi + lo) / 2.0
    node_poly = poly.substitute_affine_per_var(a, b)
    if not node_poly.coeffs or node_poly.degree < 3:
        return None, None, None   # degenerate node: caller inherits parent LB
    pairs = support_driven_family(
        list(node_poly.coeffs.keys()), complement_splits=True
    )
    res = certify_box_minimum(
        node_poly, n_vars=n_vars, box=1.0, pairs=pairs, solver=solver,
        sym_box=True, check_slater=check_slater, compute_upper=False,
        threads=threads,
    )
    if res.bound is None:
        return None, None, None
    x_node = res.x_box            # relaxation first moment in [-1,1]^n
    x_orig = a * x_node + b if x_node is not None else None
    scores = None
    if res.X_diag is not None and x_node is not None:
        scores = res.X_diag - x_node ** 2   # moment inconsistency per var
    return res.bound, x_orig, scores


def branch_and_bound(
    poly: SparsePolynomial,
    n_vars: int,
    box: float = 1.0,
    target: float | None = None,
    gap_tol: float = 1e-6,
    max_nodes: int = 200,
    solver: str = "MOSEK",
    check_slater: bool = True,
    threads: int = 0,
    ub_starts: int = 12,
    checkpoint_path: str | None = None,
    verbose: bool = False,
) -> dict:
    """Minimize poly over [-box, box]^n with certified global bounds.

    Stops when UB - LB <= gap_tol, when LB > target (nonachievability
    certified), or at max_nodes; the returned LB is valid in all cases.
    """
    t0 = time.perf_counter()
    rng_seed = itertools.count()

    def node_ub_direct(lo, hi, x_hint=None):
        best_val, best_x = np.inf, None
        rng = np.random.default_rng(next(rng_seed))
        starts = [x_hint] if x_hint is not None else []
        starts += [rng.uniform(lo, hi) for _ in range(ub_starts)]
        from scipy.optimize import minimize

        for x0 in starts:
            r = minimize(
                lambda x: poly.eval(x),
                np.clip(np.asarray(x0, float), lo, hi),
                jac=lambda x: poly_gradient(poly, x),
                method="L-BFGS-B",
                bounds=list(zip(lo, hi)),
            )
            if r.fun < best_val:
                best_val, best_x = float(r.fun), r.x
        return best_val, best_x

    lo0 = -box * np.ones(n_vars)
    hi0 = box * np.ones(n_vars)

    UB, x_best = node_ub_direct(lo0, hi0)
    lb0, x0, xdiag0 = _certify_node(
        poly, n_vars, lo0, hi0, solver, check_slater, threads
    )
    if lb0 is None:
        return {"status": "root_failed", "lb": None, "ub": UB}

    counter = itertools.count()
    heap = [_Node(lb0, next(counter), lo0, hi0, 0, x0, xdiag0)]
    pruned_lb = np.inf   # min LB over pruned/closed leaves
    n_solved = 1
    history = [(lb0, UB)]
    status = "max_nodes"

    def global_lb():
        open_lb = heap[0].lb if heap else np.inf
        return min(open_lb, pruned_lb)

    def checkpoint(status_now):
        if checkpoint_path is None:
            return
        with open(checkpoint_path, "w") as f:
            json.dump({
                "lb": global_lb(), "ub": UB, "nodes_solved": n_solved,
                "open_nodes": len(heap), "status": status_now,
                "elapsed": time.perf_counter() - t0,
            }, f, indent=2)

    while heap:
        LB = global_lb()
        if verbose:
            print(f"  bb: {n_solved} nodes, LB {LB:.6f}, UB {UB:.6f}, "
                  f"open {len(heap)}", flush=True)
        checkpoint("running")
        if target is not None and LB > target:
            status = "target_certified"
            break
        if UB - LB <= gap_tol * max(1.0, abs(UB)):
            status = "gap_closed"
            break
        if n_solved >= max_nodes:
            break

        node = heapq.heappop(heap)
        if node.lb >= UB - gap_tol * max(1.0, abs(UB)):
            pruned_lb = min(pruned_lb, node.lb)
            continue

        # branch on the most moment-inconsistent variable
        if node.branch_scores is not None:
            j = int(np.argmax(node.branch_scores))
        else:
            j = int(np.argmax(node.hi - node.lo))
        mid = 0.5 * (node.lo[j] + node.hi[j])

        for side in (0, 1):
            lo = node.lo.copy()
            hi = node.hi.copy()
            if side == 0:
                hi[j] = mid
            else:
                lo[j] = mid
            lb_c, x_c, xdiag_c = _certify_node(
                poly, n_vars, lo, hi, solver, check_slater, threads
            )
            if lb_c is None:
                lb_c, x_c, xdiag_c = node.lb, None, None  # inherit (valid)
            lb_c = max(lb_c, node.lb)  # child bound never below parent's
            n_solved += 1
            ub_c, x_ub = node_ub_direct(lo, hi, x_c)
            if ub_c < UB:
                UB, x_best = ub_c, x_ub
            if lb_c < UB - gap_tol * max(1.0, abs(UB)):
                heapq.heappush(
                    heap,
                    _Node(lb_c, next(counter), lo, hi, node.depth + 1,
                          x_c, xdiag_c),
                )
            else:
                pruned_lb = min(pruned_lb, lb_c)
        history.append((global_lb(), UB))

    if status == "max_nodes":
        final_lb = global_lb()
        if UB - final_lb <= gap_tol * max(1.0, abs(UB)):
            status = "gap_closed"
        elif target is not None and final_lb > target:
            status = "target_certified"
        elif not heap:
            status = "tree_exhausted"
    result = {
        "status": status,
        "lb": global_lb(),
        "ub": UB,
        "x_best": x_best,
        "nodes_solved": n_solved,
        "open_nodes": len(heap),
        "history": history,
        "elapsed": time.perf_counter() - t0,
    }
    checkpoint(status)
    return result
