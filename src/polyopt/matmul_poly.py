"""Decomposition-loss polynomials for tensors.

The squared Frobenius loss ||T - sum_r u_r (x) v_r (x) w_r||^2 is a
degree-6 polynomial in the factor entries. This module builds it sparsely
and defines the global variable indexing shared by the whole polyopt
pipeline: all U entries first (r-major), then V, then W.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polyopt.sparse_poly import SparsePolynomial


@dataclass(frozen=True)
class MatmulVariables:
    """Variable indexing for a rank-R decomposition of a (d1, d2, d3) tensor."""

    d1: int
    d2: int
    d3: int
    rank: int

    @property
    def n_vars(self) -> int:
        return self.rank * (self.d1 + self.d2 + self.d3)

    def u(self, r: int, a: int) -> int:
        return r * self.d1 + a

    def v(self, r: int, b: int) -> int:
        return self.rank * self.d1 + r * self.d2 + b

    def w(self, r: int, c: int) -> int:
        return self.rank * (self.d1 + self.d2) + r * self.d3 + c

    def unpack(self, x) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flat variable vector -> (U, V, W) with shapes (d1,R), (d2,R), (d3,R)
        matching tensor_utils conventions (mode-dim first, rank second)."""
        x = np.asarray(x, dtype=np.float64)
        R = self.rank
        U = x[: R * self.d1].reshape(R, self.d1).T
        V = x[R * self.d1 : R * (self.d1 + self.d2)].reshape(R, self.d2).T
        W = x[R * (self.d1 + self.d2) :].reshape(R, self.d3).T
        return U, V, W

    def pack(self, U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
        return np.concatenate([U.T.ravel(), V.T.ravel(), W.T.ravel()])


def build_decomposition_loss(
    T: np.ndarray, rank: int
) -> tuple[SparsePolynomial, MatmulVariables]:
    """Sparse polynomial ||T - sum_r u_r (x) v_r (x) w_r||_F^2.

    Expansion:
      ||T||^2  -  2 sum_{abc} T_abc sum_r u_ra v_rb w_rc
               +  sum_{r,s} (u_r . u_s)(v_r . v_s)(w_r . w_s)
    """
    d1, d2, d3 = T.shape
    var = MatmulVariables(d1, d2, d3, rank)
    p = SparsePolynomial()

    p.add_term((), float(np.sum(T * T)))

    for a, b, c in zip(*np.nonzero(T)):
        coeff = -2.0 * float(T[a, b, c])
        for r in range(rank):
            p.add_term((var.u(r, int(a)), var.v(r, int(b)), var.w(r, int(c))), coeff)

    for r in range(rank):
        for s in range(rank):
            for a in range(d1):
                for b in range(d2):
                    for c in range(d3):
                        p.add_term(
                            (
                                var.u(r, a), var.u(s, a),
                                var.v(r, b), var.v(s, b),
                                var.w(r, c), var.w(s, c),
                            ),
                            1.0,
                        )
    return p, var


def build_matmul_loss(
    m: int, p: int, n: int, rank: int
) -> tuple[SparsePolynomial, MatmulVariables]:
    """Loss polynomial for the <m,p,n> matrix multiplication tensor."""
    from tensor_utils import build_mult_tensor

    return build_decomposition_loss(build_mult_tensor(m, p, n), rank)


def build_polymult_tensor(k1: int = 2, k2: int = 2) -> np.ndarray:
    """Structure tensor of univariate polynomial multiplication:
    (a_0 + ... + a_{k1-1} x^{k1-1}) * (b_0 + ... + b_{k2-1} x^{k2-1}).

    Shape (k1, k2, k1+k2-1); T[i,j,i+j] = 1. For k1=k2=2 this is the
    Karatsuba tensor with rank 3 (and rank 2 impossible).
    """
    T = np.zeros((k1, k2, k1 + k2 - 1))
    for i in range(k1):
        for j in range(k2):
            T[i, j, i + j] = 1.0
    return T
