"""Sparse multivariate polynomials keyed by monomial multisets.

A monomial is a sorted tuple of variable indices with multiplicity, e.g.
x0^2 * x3 is (0, 0, 3) and the constant monomial is (). This canonical
form makes coefficient accumulation a plain dict merge and keeps degree-6
polynomials in 72 variables tractable (dense coefficient tensors are not).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Iterator

Monomial = tuple[int, ...]


def mono(*indices: int) -> Monomial:
    """Canonicalize variable indices into a monomial key."""
    return tuple(sorted(indices))


def mono_mul(a: Monomial, b: Monomial) -> Monomial:
    """Product of two monomials."""
    return tuple(sorted(a + b))


def mono_divisors(m: Monomial) -> Iterator[Monomial]:
    """All distinct divisor monomials of m (including () and m itself)."""
    counts = Counter(m)
    variables = sorted(counts)
    ranges = [range(counts[v] + 1) for v in variables]
    for choice in itertools.product(*ranges):
        yield tuple(
            v for v, k in zip(variables, choice) for _ in range(k)
        )


class SparsePolynomial:
    """Immutable-by-convention sparse polynomial over real coefficients."""

    __slots__ = ("coeffs",)

    def __init__(self, coeffs: dict[Monomial, float] | None = None):
        self.coeffs: dict[Monomial, float] = {}
        if coeffs:
            for m, c in coeffs.items():
                if c != 0.0:
                    self.coeffs[tuple(sorted(m))] = (
                        self.coeffs.get(tuple(sorted(m)), 0.0) + c
                    )

    # ---------------------------------------------------------------- basics

    @property
    def degree(self) -> int:
        return max((len(m) for m in self.coeffs), default=0)

    @property
    def variables(self) -> set[int]:
        return {v for m in self.coeffs for v in m}

    def __len__(self) -> int:
        return len(self.coeffs)

    def __getitem__(self, m: Monomial) -> float:
        return self.coeffs.get(tuple(sorted(m)), 0.0)

    def __repr__(self) -> str:
        return (
            f"SparsePolynomial(degree={self.degree}, "
            f"terms={len(self.coeffs)}, vars={len(self.variables)})"
        )

    def copy(self) -> "SparsePolynomial":
        p = SparsePolynomial()
        p.coeffs = dict(self.coeffs)
        return p

    # --------------------------------------------------------------- algebra

    def add_term(self, m: Monomial, c: float) -> None:
        """In-place accumulation (used by builders before freezing)."""
        key = tuple(sorted(m))
        new = self.coeffs.get(key, 0.0) + c
        if new == 0.0:
            self.coeffs.pop(key, None)
        else:
            self.coeffs[key] = new

    def __add__(self, other: "SparsePolynomial") -> "SparsePolynomial":
        p = self.copy()
        for m, c in other.coeffs.items():
            p.add_term(m, c)
        return p

    def __sub__(self, other: "SparsePolynomial") -> "SparsePolynomial":
        return self + other * (-1.0)

    def __mul__(self, other: "SparsePolynomial | float | int"):
        if isinstance(other, (int, float)):
            if other == 0:
                return SparsePolynomial()
            p = SparsePolynomial()
            p.coeffs = {m: c * other for m, c in self.coeffs.items()}
            return p
        p = SparsePolynomial()
        for ma, ca in self.coeffs.items():
            for mb, cb in other.coeffs.items():
                p.add_term(mono_mul(ma, mb), ca * cb)
        return p

    __rmul__ = __mul__

    # ------------------------------------------------------------ evaluation

    def eval(self, x) -> float:
        """Evaluate at a point (indexable by variable id)."""
        total = 0.0
        for m, c in self.coeffs.items():
            term = c
            for v in m:
                term *= x[v]
            total += term
        return total

    # ---------------------------------------------------- affine substitution

    def substitute_affine(self, a: float, b: float) -> "SparsePolynomial":
        """Return q(y) = p(a*y + b), substituting x_i = a*y_i + b for all i.

        Used to map a polynomial over the box [-B, B]^n to one over
        [0, 1]^n via x = 2B*y - B (a=2B, b=-B). Each monomial expands
        into its divisor monomials only, so sparsity is preserved.
        """
        q = SparsePolynomial()
        for m, c in self.coeffs.items():
            counts = Counter(m)
            variables = sorted(counts)
            ranges = [range(counts[v] + 1) for v in variables]
            for choice in itertools.product(*ranges):
                coeff = c
                sub = []
                for v, j in zip(variables, choice):
                    k = counts[v]
                    coeff *= math.comb(k, j) * (a ** j) * (b ** (k - j))
                    sub.extend([v] * j)
                if coeff != 0.0:
                    q.add_term(tuple(sub), coeff)
        return q

    def substitute_affine_per_var(self, a, b) -> "SparsePolynomial":
        """Return q(y) = p(a*y + b) with per-variable coefficients:
        x_i = a[i]*y_i + b[i]. Used by branch & bound to map an arbitrary
        node box prod [lo_i, hi_i] onto [-1,1]^n."""
        q = SparsePolynomial()
        for m, c in self.coeffs.items():
            counts = Counter(m)
            variables = sorted(counts)
            ranges = [range(counts[v] + 1) for v in variables]
            for choice in itertools.product(*ranges):
                coeff = c
                sub = []
                for v, j in zip(variables, choice):
                    k = counts[v]
                    coeff *= math.comb(k, j) * (a[v] ** j) * (b[v] ** (k - j))
                    sub.extend([v] * j)
                if coeff != 0.0:
                    q.add_term(tuple(sub), coeff)
        return q

    # ------------------------------------------------------------- utilities

    def support_closure(self) -> set[Monomial]:
        """All divisor monomials of the support (needed for coefficient
        matching: multiplier expansions can only produce divisors when the
        multiplier indices are drawn from the monomials themselves)."""
        closure: set[Monomial] = set()
        for m in self.coeffs:
            closure.update(mono_divisors(m))
        return closure
