"""Tensor nonachievability certificates (slow; requires MOSEK)."""

import numpy as np
import pytest

from polyopt.certify import certify_box_minimum
from polyopt.matmul_poly import build_decomposition_loss, build_polymult_tensor
from polyopt.multipliers import support_driven_family

mosek = pytest.importorskip("mosek")


@pytest.mark.slow
def test_polymult_rank1_certificate():
    """The polymult<2,2> (Karatsuba) tensor has rank 3; rank 1 is
    impossible and the root bound must certify it (loss > 0 on the box).
    Support-driven family with all complement splits reaches ~+0.44
    (full family: ~+0.47; true box minimum: 2.5)."""
    T = build_polymult_tensor(2, 2)
    poly, var = build_decomposition_loss(T, 1)
    pairs = support_driven_family(
        list(poly.coeffs.keys()), complement_splits=True
    )
    res = certify_box_minimum(
        poly, n_vars=var.n_vars, box=1.0, pairs=pairs,
        solver="MOSEK", sym_box=True,
    )
    assert res.slater_gamma > 0
    assert res.bound > 0.3          # certified: rank-1 impossible on box
    assert res.upper_bound == pytest.approx(2.5, abs=1e-6)


@pytest.mark.slow
def test_matvec_121_certificates():
    """<1,2,1> (inner product): rank 2 exists, rank 1 does not.
    The engine must certify rank-1 impossibility (positive bound) and
    must NOT produce a positive bound at rank 2 (negative control)."""
    from polyopt.matmul_poly import build_matmul_loss

    poly1, var1 = build_matmul_loss(1, 2, 1, 1)
    pairs1 = support_driven_family(
        list(poly1.coeffs.keys()), complement_splits=True
    )
    res1 = certify_box_minimum(
        poly1, n_vars=var1.n_vars, box=1.0, pairs=pairs1,
        solver="MOSEK", sym_box=True,
    )
    assert res1.slater_gamma > 0
    assert res1.bound > 0.0          # rank 1 certified impossible

    poly2, var2 = build_matmul_loss(1, 2, 1, 2)
    pairs2 = support_driven_family(
        list(poly2.coeffs.keys()), complement_splits=True
    )
    res2 = certify_box_minimum(
        poly2, n_vars=var2.n_vars, box=1.0, pairs=pairs2,
        solver="MOSEK", sym_box=True, check_slater=False,
        compute_upper=True,
    )
    assert res2.bound <= 1e-6        # no false positive
    assert res2.upper_bound == pytest.approx(0.0, abs=1e-6)
