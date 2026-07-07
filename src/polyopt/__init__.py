"""Exact polynomial optimization for certified tensor-rank lower bounds.

Implements the best-SLC-decomposition method of Bertsimas, den Hertog &
Koukouvinos (arXiv:2507.02120), generalized to arbitrary degree via
(I, J)-multiset multipliers, for polynomials over a box.
"""

from polyopt.sparse_poly import SparsePolynomial
from polyopt.matmul_poly import (
    MatmulVariables,
    build_decomposition_loss,
    build_matmul_loss,
)

__all__ = [
    "SparsePolynomial",
    "MatmulVariables",
    "build_decomposition_loss",
    "build_matmul_loss",
]
