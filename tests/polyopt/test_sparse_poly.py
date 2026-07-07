"""Sparse polynomial algebra and substitution."""

import numpy as np
import pytest

from polyopt.sparse_poly import SparsePolynomial, mono, mono_divisors


def random_poly(rng, n_vars, degree, n_terms):
    p = SparsePolynomial()
    for _ in range(n_terms):
        d = rng.integers(0, degree + 1)
        m = mono(*rng.integers(0, n_vars, size=d))
        p.add_term(m, float(rng.normal()))
    return p


def test_canonicalization_and_accumulation():
    p = SparsePolynomial()
    p.add_term((3, 0, 0), 2.0)
    p.add_term((0, 3, 0), 1.5)   # same monomial, unsorted
    assert p[(0, 0, 3)] == pytest.approx(3.5)
    p.add_term((0, 0, 3), -3.5)  # cancel exactly -> term removed
    assert len(p) == 0


def test_mul_matches_numeric():
    rng = np.random.default_rng(0)
    for _ in range(10):
        a = random_poly(rng, 4, 3, 6)
        b = random_poly(rng, 4, 2, 5)
        x = rng.normal(size=4)
        assert (a * b).eval(x) == pytest.approx(a.eval(x) * b.eval(x), rel=1e-12)
        assert (a + b).eval(x) == pytest.approx(a.eval(x) + b.eval(x), rel=1e-12)
        assert (a - b).eval(x) == pytest.approx(a.eval(x) - b.eval(x), rel=1e-12)
        assert (2.5 * a).eval(x) == pytest.approx(2.5 * a.eval(x), rel=1e-12)


def test_substitute_affine_roundtrip():
    rng = np.random.default_rng(1)
    B = 1.0
    for _ in range(10):
        p = random_poly(rng, 5, 6, 12)
        q = p.substitute_affine(2 * B, -B)  # q(y) = p(2B*y - B)
        y = rng.uniform(0, 1, size=5)
        x = 2 * B * y - B
        assert q.eval(y) == pytest.approx(p.eval(x), rel=1e-10, abs=1e-12)
        assert q.degree == p.degree


def test_divisors():
    divisors = set(mono_divisors((0, 0, 2)))
    assert divisors == {(), (0,), (2,), (0, 0), (0, 2), (0, 0, 2)}
