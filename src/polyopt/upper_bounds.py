"""Upper-bound oracles: local minimization of the polynomial over a box."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from polyopt.sparse_poly import SparsePolynomial


def poly_gradient(poly: SparsePolynomial, x: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(x)
    for m, c in poly.coeffs.items():
        if not m:
            continue
        for v in set(m):
            mult = m.count(v)
            term = c * mult * x[v] ** (mult - 1)
            for u in set(m):
                if u != v:
                    term *= x[u] ** m.count(u)
            grad[v] += term
    return grad


def multistart_upper_bound(
    poly: SparsePolynomial,
    n_vars: int,
    lower: float = 0.0,
    upper: float = 1.0,
    n_starts: int = 20,
    seed: int = 0,
    x0_list: list[np.ndarray] | None = None,
) -> tuple[float, np.ndarray]:
    """Best local minimum of poly over the box [lower, upper]^n from random
    (plus optional user-provided) starting points."""
    rng = np.random.default_rng(seed)
    starts = list(x0_list or [])
    starts.extend(
        rng.uniform(lower, upper, size=n_vars) for _ in range(n_starts)
    )
    best_val, best_x = np.inf, None
    bounds = [(lower, upper)] * n_vars
    for x0 in starts:
        res = minimize(
            lambda x: poly.eval(x),
            np.clip(np.asarray(x0, dtype=float), lower, upper),
            jac=lambda x: poly_gradient(poly, x),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if res.fun < best_val:
            best_val, best_x = float(res.fun), res.x
    return best_val, best_x


def candidate_points(x: np.ndarray, X: np.ndarray) -> list[np.ndarray]:
    """Candidate solutions extracted from relaxation moments: the first
    moment itself plus normalized columns of the second-moment matrix
    (port of calculate_candidate_vectors in docs/Polynomial_Degree3.jl)."""
    candidates = [x]
    for j in range(X.shape[1]):
        if abs(x[j]) > 1e-9:
            candidates.append(X[:, j] / x[j])
    return candidates
