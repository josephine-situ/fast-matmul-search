"""Certified-pruning exhaustive search for discrete-alphabet decompositions.

Searches for rank-R decompositions T = sum_r u_r (x) v_r (x) w_r with U, V
entries in a finite alphabet (default {-1, 0, 1}) and W free (solved
exactly by least squares at the leaves). Novelty vs. heuristic searches
(gradient descent, flip graphs, RL) and GF(2) SAT enumeration: subtrees
are eliminated by CERTIFIED lower bounds, so a completed run is a
theorem about the whole alphabet, not just a set of found solutions.

Why this escapes the certification wall measured at the symmetric root:
fixing entries destroys the continuous scaling gauge and the sign orbit,
shrinks the variable count, and drops the loss degree (U fixed -> degree
4; U,V fixed -> W is linear least squares). The SLC engine is strong
exactly on such reduced, low-symmetry problems.

Pruning ladder (cheap -> expensive):
  1. canonical ordering: per-term leading sign fixed (+1 first nonzero of
     u_r and of v_r, absorbable into w_r), terms lex-nondecreasing -
     eliminates the 2^2R R! discrete symmetry group up front;
  2. unfolding bound: once U is complete, min_G ||T - U G||_F^2 over
     UNSTRUCTURED G (a least-squares projection) lower-bounds the loss;
  3. SDP bound: substitute the partial assignment, compact variables and
     certify the reduced polynomial over [-1,1]^m (valid for the alphabet
     since it is contained in the box); bound > eps prunes the subtree.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field

import numpy as np

from polyopt.matmul_poly import MatmulVariables, build_decomposition_loss
from polyopt.multipliers import support_driven_family
from polyopt.certify import certify_box_minimum
from polyopt.sparse_poly import SparsePolynomial


def _canonical_vector(vec: tuple) -> bool:
    """First nonzero entry must be positive (leading-sign gauge)."""
    for x in vec:
        if x != 0:
            return x > 0
    return True  # all-zero is canonical


@dataclass
class SearchStats:
    nodes: int = 0
    leaves: int = 0
    pruned_canonical: int = 0
    pruned_unfolding: int = 0
    pruned_sdp: int = 0
    sdp_calls: int = 0
    solutions: list = field(default_factory=list)


def _leaf_solve(T: np.ndarray, U: np.ndarray, V: np.ndarray,
                tol: float) -> np.ndarray | None:
    """Solve min_W ||T - sum u_r v_r w_r||_F^2 exactly; return W if the
    residual vanishes (an exact decomposition), else None."""
    d1, d2, d3 = T.shape
    R = U.shape[1]
    A = np.zeros((d1 * d2, R))
    for r in range(R):
        A[:, r] = np.outer(U[:, r], V[:, r]).ravel()
    t_mat = T.reshape(d1 * d2, d3)
    W_t, res, _, _ = np.linalg.lstsq(A, t_mat, rcond=None)
    if np.linalg.norm(A @ W_t - t_mat) < tol:
        return W_t.T  # (d3, R)
    return None


def _unfolding_bound(T: np.ndarray, U: np.ndarray) -> float:
    """min over unstructured G of ||T_(1) - U G||_F^2: a valid lower
    bound on the loss for any completion (relaxes v_r w_r^T to free)."""
    d1 = T.shape[0]
    T1 = T.reshape(d1, -1)
    G, _, _, _ = np.linalg.lstsq(U, T1, rcond=None)
    return float(np.linalg.norm(U @ G - T1) ** 2)


def _sdp_bound(poly: SparsePolynomial, fixed: dict[int, float],
               solver: str, threads: int) -> float | None:
    """Certified lower bound of the loss over [-1,1] for the free
    variables, given the partial assignment. None if unavailable."""
    reduced = poly.substitute_values(fixed)
    free = sorted(reduced.variables)
    if not free or reduced.degree < 3:
        return None
    mapping = {v: i for i, v in enumerate(free)}
    compact = reduced.rename_vars(mapping)
    pairs = support_driven_family(
        list(compact.coeffs.keys()), complement_splits=True
    )
    try:
        res = certify_box_minimum(
            compact, n_vars=len(free), box=1.0, pairs=pairs, solver=solver,
            sym_box=True, check_slater=False, compute_upper=False,
            threads=threads,
        )
    except Exception:
        return None
    return res.bound


def certified_ternary_search(
    T: np.ndarray,
    rank: int,
    alphabet: tuple = (-1.0, 0.0, 1.0),
    exact_tol: float = 1e-8,
    prune_eps: float = 1e-6,
    sdp_prune: str = "after-U",   # "off" | "after-U" | "each-V-term"
    solver: str = "MOSEK",
    threads: int = 0,
    max_leaves: int | None = None,
    find_first: bool = True,
    checkpoint_path: str | None = None,
    verbose: bool = False,
) -> dict:
    """Exhaustive search over U, V entries in `alphabet` (W free/exact).

    Enumerates canonical per-term (u_r, v_r) columns depth-first. If it
    terminates without max_leaves, the outcome is exhaustive for the
    alphabet modulo the eliminated symmetries.
    """
    t0 = time.perf_counter()
    d1, d2, d3 = T.shape
    poly, var = build_decomposition_loss(T, rank)
    stats = SearchStats()

    # canonical column candidates (leading sign fixed)
    u_cands = [c for c in itertools.product(alphabet, repeat=d1)
               if _canonical_vector(c)]
    v_cands = [c for c in itertools.product(alphabet, repeat=d2)
               if _canonical_vector(c)]

    def checkpoint(status):
        if checkpoint_path is None:
            return
        with open(checkpoint_path, "w") as f:
            json.dump({
                "status": status, "nodes": stats.nodes,
                "leaves": stats.leaves,
                "pruned": {"canonical": stats.pruned_canonical,
                           "unfolding": stats.pruned_unfolding,
                           "sdp": stats.pruned_sdp},
                "sdp_calls": stats.sdp_calls,
                "n_solutions": len(stats.solutions),
                "elapsed": time.perf_counter() - t0,
            }, f, indent=2)

    def fixed_dict(U_cols, V_cols):
        fixed = {}
        for r, col in enumerate(U_cols):
            for a, val in enumerate(col):
                fixed[var.u(r, a)] = float(val)
        for r, col in enumerate(V_cols):
            for b, val in enumerate(col):
                fixed[var.v(r, b)] = float(val)
        return fixed

    done = False

    def recurse_V(U_cols, V_cols):
        nonlocal done
        if done:
            return
        r = len(V_cols)
        if max_leaves is not None and stats.leaves >= max_leaves:
            done = True
            return
        if r == rank:
            stats.leaves += 1
            U = np.array(U_cols, dtype=float).T
            V = np.array(V_cols, dtype=float).T
            W = _leaf_solve(T, U, V, exact_tol)
            if W is not None:
                stats.solutions.append((U, V, W))
                if verbose:
                    print(f"  FOUND exact decomposition "
                          f"(leaf {stats.leaves})", flush=True)
                if find_first:
                    done = True
            return
        for v_col in v_cands:
            stats.nodes += 1
            # lex order among terms with equal u-columns
            if r > 0 and U_cols[r - 1] == U_cols[r] and \
                    tuple(v_col) < tuple(V_cols[r - 1]):
                stats.pruned_canonical += 1
                continue
            if sdp_prune == "each-V-term" and r < rank - 1:
                stats.sdp_calls += 1
                fixed = fixed_dict(U_cols, V_cols + [v_col])
                bound = _sdp_bound(poly, fixed, solver, threads)
                if bound is not None and bound > prune_eps:
                    stats.pruned_sdp += 1
                    continue
            recurse_V(U_cols, V_cols + [v_col])
            if done:
                return

    def recurse_U(U_cols):
        nonlocal done
        if done:
            return
        r = len(U_cols)
        if r == rank:
            U = np.array(U_cols, dtype=float).T
            if _unfolding_bound(T, U) > prune_eps:
                stats.pruned_unfolding += 1
                return
            if sdp_prune in ("after-U", "each-V-term"):
                stats.sdp_calls += 1
                bound = _sdp_bound(poly, fixed_dict(U_cols, []),
                                   solver, threads)
                if bound is not None and bound > prune_eps:
                    stats.pruned_sdp += 1
                    return
            recurse_V(U_cols, [])
            return
        for u_col in u_cands:
            stats.nodes += 1
            # terms lex-nondecreasing in u
            if r > 0 and tuple(u_col) < tuple(U_cols[r - 1]):
                stats.pruned_canonical += 1
                continue
            recurse_U(U_cols + [u_col])
            if done:
                return
            if stats.nodes % 5000 == 0:
                checkpoint("running")
                if verbose:
                    print(f"  search: {stats.nodes} nodes, "
                          f"{stats.leaves} leaves, "
                          f"{len(stats.solutions)} found", flush=True)

    recurse_U([])

    status = ("found" if stats.solutions and find_first else
              "budget" if done else "exhausted")
    checkpoint(status)
    return {
        "status": status,
        "solutions": stats.solutions,
        "nodes": stats.nodes,
        "leaves": stats.leaves,
        "pruned_canonical": stats.pruned_canonical,
        "pruned_unfolding": stats.pruned_unfolding,
        "pruned_sdp": stats.pruned_sdp,
        "sdp_calls": stats.sdp_calls,
        "elapsed": time.perf_counter() - t0,
    }
