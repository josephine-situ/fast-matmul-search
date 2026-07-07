"""Degree-6 validation gates (Theorem 6 / docs/general_thm.md)."""

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


def test_univariate_sextic_bound():
    # p(x) = x^6 - x^2 on [-1,1]: min = -(2/3)*3^(-1/2) at x^2 = 3^(-1/2)
    p = SparsePolynomial({(0,) * 6: 1.0, (0, 0): -1.0})
    res = certify_box_minimum(p, n_vars=1, box=1.0)
    true_min = -(2.0 / 3.0) * 3 ** (-0.5)
    assert res.slater_gamma > 0
    assert res.bound <= true_min + 1e-6
    assert res.bound >= true_min - 0.2  # root bound within a sane margin


def test_positive_sextic_certificate():
    # p(x) = (x^3 - x)^2 + 0.1 has global min 0.1 > 0: the root bound
    # must certify strict positivity - the shape of a nonachievability
    # proof (loss > epsilon).
    p = SparsePolynomial(
        {(0,) * 6: 1.0, (0,) * 4: -2.0, (0, 0): 1.0, (): 0.1}
    )
    res = certify_box_minimum(p, n_vars=1, box=1.0)
    assert res.bound > 0.0
    assert res.bound <= 0.1 + 1e-6


def test_degree6_reconstruction_bivariate():
    rng = np.random.default_rng(17)
    n = 2
    poly = random_dense_poly(rng, n, 6, density=0.6)
    matching = build_matching(full_family(n, 4), poly)
    gamma, Z = feasibility_check(matching)
    assert gamma > 1e-6
    recon = reconstruct_polynomial(matching, Z)
    diff = recon - poly
    assert max((abs(c) for c in diff.coeffs.values()), default=0.0) < 1e-4


def test_random_degree6_bounds_below_upper():
    rng = np.random.default_rng(19)
    n = 3
    for trial in range(2):
        poly = random_dense_poly(rng, n, 6, density=0.3)
        res = certify_box_minimum(poly, n_vars=n, box=1.0)
        assert res.status in ("optimal", "optimal_inaccurate"), res.status
        # solver-tolerance slack; rigorous deflation is done by
        # verify_certificate for real certificates
        slack = 1e-3 * max(1.0, abs(res.upper_bound))
        assert res.bound <= res.upper_bound + slack
