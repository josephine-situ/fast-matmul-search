"""Cutting-plane mode must reproduce dual bounds (requires MOSEK)."""

import numpy as np
import pytest

from polyopt.certify import certify_box_minimum
from polyopt.sparse_poly import SparsePolynomial

from tests.polyopt.test_slc_degree3 import random_dense_poly

mosek = pytest.importorskip("mosek")


def test_cutting_plane_positive_sextic():
    p = SparsePolynomial(
        {(0,) * 6: 1.0, (0,) * 4: -2.0, (0, 0): 1.0, (): 0.1}
    )
    res = certify_box_minimum(
        p, n_vars=1, box=1.0, solver="MOSEK", sym_box=True,
        compute_upper=False, method="cutting-plane",
    )
    assert res.status == "converged"
    assert res.bound == pytest.approx(0.1, abs=1e-5)


@pytest.mark.parametrize("degree,n", [(3, 3), (4, 2)])
def test_cutting_plane_matches_dual(degree, n):
    rng = np.random.default_rng(200 + degree)
    poly = random_dense_poly(rng, n, degree, density=0.6)
    r_dual = certify_box_minimum(
        poly, n_vars=n, box=1.0, solver="MOSEK", sym_box=True,
        compute_upper=False, check_slater=False,
    )
    r_cp = certify_box_minimum(
        poly, n_vars=n, box=1.0, solver="MOSEK", sym_box=True,
        compute_upper=False, method="cutting-plane",
        cp_options={"max_iters": 60, "tol": 1e-5},
    )
    # cutting-plane converges to the dual value from below; allow the
    # loop to stop slightly short
    assert r_cp.bound <= r_dual.bound + 1e-4
    assert r_cp.bound >= r_dual.bound - 5e-3 * max(1.0, abs(r_dual.bound))