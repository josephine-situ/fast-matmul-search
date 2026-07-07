"""Decomposition-loss polynomial vs. the numeric Frobenius loss."""

import numpy as np
import pytest

from polyopt.matmul_poly import (
    build_decomposition_loss,
    build_matmul_loss,
    build_polymult_tensor,
)
from tensor_utils import build_mult_tensor, frobenius_loss


@pytest.mark.parametrize(
    "case, rank",
    [((2, 2, 2), 6), ((2, 2, 2), 7), ((2, 2, 1), 3), ((1, 2, 2), 2)],
)
def test_matmul_loss_matches_frobenius(case, rank):
    m, p_, n = case
    poly, var = build_matmul_loss(m, p_, n, rank)
    T = build_mult_tensor(m, p_, n)
    rng = np.random.default_rng(42)
    for _ in range(5):
        U = rng.normal(size=(m * p_, rank))
        V = rng.normal(size=(p_ * n, rank))
        W = rng.normal(size=(m * n, rank))
        x = var.pack(U, V, W)
        assert poly.eval(x) == pytest.approx(
            frobenius_loss(T, U, V, W), rel=1e-10
        )


def test_pack_unpack_roundtrip():
    _, var = build_matmul_loss(2, 2, 2, 6)
    rng = np.random.default_rng(0)
    U, V, W = rng.normal(size=(4, 6)), rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    U2, V2, W2 = var.unpack(var.pack(U, V, W))
    assert np.allclose(U, U2) and np.allclose(V, V2) and np.allclose(W, W2)


def test_polymult_tensor_loss():
    T = build_polymult_tensor(2, 2)
    assert T.shape == (2, 2, 3)
    poly, var = build_decomposition_loss(T, 3)
    # Karatsuba-style exact rank-3 decomposition:
    #   c0 = a0 b0, c2 = a1 b1, c1 = (a0+a1)(b0+b1) - a0 b0 - a1 b1
    U = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    V = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    # W[c, r] = contribution of product r to output coefficient c
    W = np.array([[1, 0, 0], [-1, -1, 1], [0, 1, 0]], dtype=float)
    x = var.pack(U, V, W)
    assert poly.eval(x) == pytest.approx(0.0, abs=1e-12)


def test_loss_sizes_222_rank6():
    poly, var = build_matmul_loss(2, 2, 2, 6)
    assert var.n_vars == 72
    assert poly.degree == 6
    # 1 constant + 8 nonzeros * 6 ranks deg-3 + deg-6 terms
    assert len(poly) < 3000
