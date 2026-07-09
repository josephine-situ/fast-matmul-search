"""Spatial branch & bound: closes root gaps that the relaxation leaves."""

import numpy as np
import pytest

from polyopt.branch_bound import branch_and_bound
from polyopt.sparse_poly import SparsePolynomial

mosek = pytest.importorskip("mosek")


def test_substitute_affine_per_var():
    rng = np.random.default_rng(5)
    p = SparsePolynomial({(0, 0, 1): 2.0, (1, 1): -1.0, (0,): 0.5, (): 1.0})
    a = np.array([0.5, 2.0])
    b = np.array([-0.25, 1.0])
    q = p.substitute_affine_per_var(a, b)
    for _ in range(10):
        y = rng.normal(size=2)
        assert q.eval(y) == pytest.approx(p.eval(a * y + b), rel=1e-12)


def test_double_well_gap_closed_by_branching():
    # root relaxation gives ~-0.3431 (symmetric interior minima averaged);
    # branching separates the wells and must close to the true -0.25
    p = SparsePolynomial({(0, 0, 0, 0): 1.0, (0, 0): -1.0})
    res = branch_and_bound(
        p, n_vars=1, box=1.0, target=None, gap_tol=1e-4, max_nodes=30,
    )
    assert res["status"] == "gap_closed"
    assert res["lb"] == pytest.approx(-0.25, abs=2e-3)
    assert res["ub"] == pytest.approx(-0.25, abs=1e-6)


def test_positive_target_stops_early():
    # p >= 0.1 on the box; the root already certifies > 0, so the loop
    # should stop immediately with the target met
    p = SparsePolynomial(
        {(0,) * 6: 1.0, (0,) * 4: -2.0, (0, 0): 1.0, (): 0.1}
    )
    res = branch_and_bound(
        p, n_vars=1, box=1.0, target=0.0, max_nodes=10,
    )
    assert res["status"] == "target_certified"
    assert res["lb"] > 0.0
    assert res["nodes_solved"] == 1
