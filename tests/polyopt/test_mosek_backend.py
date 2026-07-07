"""Backend parity: MOSEK Fusion must reproduce the cvxpy/Clarabel bounds."""

import numpy as np
import pytest

from polyopt.certify import certify_box_minimum
from polyopt.sparse_poly import SparsePolynomial

from tests.polyopt.test_slc_degree3 import random_dense_poly

mosek = pytest.importorskip("mosek")


@pytest.mark.parametrize("degree,n", [(3, 4), (4, 3), (6, 2)])
def test_backend_parity(degree, n):
    rng = np.random.default_rng(100 + degree)
    poly = random_dense_poly(rng, n, degree, density=0.5)
    res_cl = certify_box_minimum(
        poly, n_vars=n, box=1.0, solver="CLARABEL", compute_upper=False,
        check_slater=False,
    )
    res_mo = certify_box_minimum(
        poly, n_vars=n, box=1.0, solver="MOSEK", compute_upper=False,
        check_slater=False,
    )
    assert res_mo.bound == pytest.approx(res_cl.bound, rel=1e-5, abs=1e-6)


def test_mosek_positive_certificate():
    p = SparsePolynomial(
        {(0,) * 6: 1.0, (0,) * 4: -2.0, (0, 0): 1.0, (): 0.1}
    )
    res = certify_box_minimum(p, n_vars=1, box=1.0, solver="MOSEK")
    assert res.bound > 0.0
    assert res.bound <= 0.1 + 1e-6
