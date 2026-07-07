"""Degree-3 validation gates for the SLC/RPT engine (paper Section 2)."""

import numpy as np
import pytest

from polyopt.certify import certify_box_minimum
from polyopt.multipliers import full_family
from polyopt.slc_constraints import (
    build_matching,
    feasibility_check,
    reconstruct_polynomial,
)
from polyopt.sparse_poly import SparsePolynomial, mono
from polyopt.upper_bounds import multistart_upper_bound


def random_dense_poly(rng, n, degree, coeff_range=5.0, density=0.5):
    """Random polynomial as in the paper's experiments (Section 6)."""
    p = SparsePolynomial()
    import itertools

    for k in range(1, degree + 1):
        for m in itertools.combinations_with_replacement(range(n), k):
            if rng.uniform() < density:
                p.add_term(m, float(rng.uniform(-coeff_range, coeff_range)))
    return p


def test_full_family_matching_is_strictly_feasible_and_reconstructs():
    rng = np.random.default_rng(3)
    n = 4
    poly = random_dense_poly(rng, n, 3)
    pairs = full_family(n, 1)
    matching = build_matching(pairs, poly)
    gamma, Z = feasibility_check(matching)
    assert gamma > 1e-6  # Slater point exists (Theorem 1 construction)
    recon = reconstruct_polynomial(matching, Z)
    diff = recon - poly
    assert max((abs(c) for c in diff.coeffs.values()), default=0.0) < 1e-5


def test_univariate_cubic_bound():
    # p(x) = x^3 - x on [-1, 1]: global min -2/(3 sqrt 3) at x = 1/sqrt(3)
    p = SparsePolynomial({(0, 0, 0): 1.0, (0,): -1.0})
    res = certify_box_minimum(p, n_vars=1, box=1.0)
    true_min = -2.0 / (3.0 * np.sqrt(3.0))
    assert res.slater_gamma > 0
    assert res.bound <= true_min + 1e-6
    assert res.bound >= true_min - 5e-3  # root relaxation is tight here


def test_random_degree3_bounds_below_upper():
    rng = np.random.default_rng(7)
    n = 5
    tight = 0
    for trial in range(4):
        poly = random_dense_poly(rng, n, 3)
        res = certify_box_minimum(poly, n_vars=n, box=1.0)
        assert res.status in ("optimal", "optimal_inaccurate"), res.status
        assert res.bound is not None and res.upper_bound is not None
        assert res.bound <= res.upper_bound + 1e-5
        if res.gap < 1e-3 * max(1.0, abs(res.upper_bound)):
            tight += 1
    # paper Table 1: best-SLC root bound equals the optimum on most
    # box-constrained instances
    assert tight >= 2
