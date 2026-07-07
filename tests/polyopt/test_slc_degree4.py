"""Degree-4 validation gates (paper Section 3, Theorem 5 parity)."""

import numpy as np
import pytest

from polyopt.certify import certify_box_minimum
from polyopt.multipliers import full_family
from polyopt.slc_constraints import (
    build_matching,
    feasibility_check,
    reconstruct_polynomial,
)
from polyopt.sparse_poly import SparsePolynomial

from tests.polyopt.test_slc_degree3 import random_dense_poly


def test_family_counts_match_paper():
    # Theorem 4 family: {1} + {y_i, 1-y_i} + {y_i y_j (i<=j), y_i(1-y_j)
    # (all i,j), (1-y_i)(1-y_j) (i<=j)}
    n = 6
    pairs = full_family(n, 2)
    expected = 1 + 2 * n + (n * (n + 1) // 2) * 2 + n * n
    assert len(pairs) == expected


def test_degree4_reconstruction():
    rng = np.random.default_rng(11)
    n = 3
    poly = random_dense_poly(rng, n, 4)
    matching = build_matching(full_family(n, 2), poly)
    gamma, Z = feasibility_check(matching)
    assert gamma > 1e-6
    recon = reconstruct_polynomial(matching, Z)
    diff = recon - poly
    assert max((abs(c) for c in diff.coeffs.values()), default=0.0) < 1e-5


def test_double_well_quartic():
    # p(x) = x^4 - x^2 on [-1, 1]: global min -1/4 at x = +-1/sqrt(2).
    # The symmetric interior minima leave a genuine root-relaxation gap
    # (bound -2(3-2*sqrt(2)) ~ -0.3431); spatial branching closes it.
    p = SparsePolynomial({(0, 0, 0, 0): 1.0, (0, 0): -1.0})
    res = certify_box_minimum(p, n_vars=1, box=1.0)
    assert res.slater_gamma > 0
    assert res.bound <= -0.25 + 1e-6   # valid lower bound
    assert res.bound >= -0.35          # and not vacuous


def test_random_degree4_bounds_below_upper():
    rng = np.random.default_rng(13)
    n = 4
    tight = 0
    for trial in range(3):
        poly = random_dense_poly(rng, n, 4, density=0.4)
        res = certify_box_minimum(poly, n_vars=n, box=1.0)
        assert res.status in ("optimal", "optimal_inaccurate"), res.status
        assert res.bound <= res.upper_bound + 1e-5
        if res.gap < 1e-3 * max(1.0, abs(res.upper_bound)):
            tight += 1
    assert tight >= 2
