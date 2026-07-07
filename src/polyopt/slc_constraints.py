"""Coefficient-matching constraints for SLC decompositions.

Every valid SLC decomposition p(y) = sum_t f_t(y) q_t(y) is characterized
by linear equalities on the quadratic coefficients (P_t, r_t, w_t): one
equality per monomial in the *closure* (all monomials producible by any
f_t * q_t), with right-hand side the coefficient of that monomial in p
(zero if absent). This module enumerates those constraints sparsely; they
are consumed both by the feasibility/reconstruction checks here and by the
dual "best decomposition" SDP in relaxation.py.

Expansion identity used throughout:
    f_{I,J}(y) = sum_{S subseteq J} (-1)^{|S|} C(J,S) y^{I+S}
where S ranges over sub-multisets of J and C(J,S) = prod_j binom(J_j, S_j).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from polyopt.multipliers import MultiplierPair
from polyopt.sparse_poly import Monomial, SparsePolynomial


def expand_box_product(plus: Monomial, minus: Monomial, sym_box: bool
                       ) -> list[tuple[Monomial, float]]:
    """Expand a product of box factors into [(monomial, coefficient)].

    On [0,1]^n (sym_box=False): prod_{i in plus} y_i * prod_{j in minus} (1-y_j).
    On [-1,1]^n (sym_box=True): prod_{i in plus} (1+y_i) * prod_{j in minus} (1-y_j).
    Both are nonnegative on their box, which is what RLT and SLC
    multipliers require.
    """
    base: list[tuple[list[int], float]] = [([], 1.0)]

    def convolve(factors: Counter, sign: float):
        nonlocal base
        for v in sorted(factors):
            k = factors[v]
            new: list[tuple[list[int], float]] = []
            for mono_part, coeff in base:
                for s in range(k + 1):
                    c = coeff * math.comb(k, s) * (sign ** s)
                    new.append((mono_part + [v] * s, c))
            base = new

    if sym_box:
        convolve(Counter(plus), 1.0)      # (1 + y)^k
    else:
        # y^k contributes a single term
        base = [(list(plus), 1.0)]
    convolve(Counter(minus), -1.0)        # (1 - y)^k

    merged: dict[Monomial, float] = {}
    for mono_part, coeff in base:
        key = tuple(sorted(mono_part))
        merged[key] = merged.get(key, 0.0) + coeff
    return [(m, c) for m, c in merged.items() if c != 0.0]


def multiplier_expansion(pair: MultiplierPair, sym_box: bool = False
                         ) -> list[tuple[Monomial, float]]:
    """Expand f_{I,J} into [(base monomial, signed coefficient)]."""
    return expand_box_product(pair.I, pair.J, sym_box)


@dataclass
class PairContrib:
    """Sparse contribution of one pair's coefficients to the matching
    equalities, in local support coordinates (a, b index into pair.supp).

    Conventions: P_t is a symmetric matrix; a matching equality picks up
    sigma * (P[a,b] + P[b,a]) for a != b and sigma * P[a,a] on the
    diagonal. Entries here store raw sigma with a <= b; consumers fold
    the symmetry factor.
    """

    pair: MultiplierPair
    # (monomial index, a, b, sigma) with a <= b local indices
    P_entries: list[tuple[int, int, int, float]] = field(default_factory=list)
    # (monomial index, a, sigma)
    r_entries: list[tuple[int, int, float]] = field(default_factory=list)
    # (monomial index, sigma)
    w_entries: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class MatchingData:
    """Constraint system describing all valid SLC decompositions."""

    pairs: list[MultiplierPair]
    closure: list[Monomial]                  # sorted; equality index = position
    closure_index: dict[Monomial, int]
    contribs: list[PairContrib]              # parallel to pairs
    rhs: np.ndarray                          # coefficient of p per closure monomial
    sym_box: bool = False                    # False: [0,1]^n, True: [-1,1]^n

    @property
    def n_constraints(self) -> int:
        return len(self.closure)


def build_matching(pairs: list[MultiplierPair], poly: SparsePolynomial,
                   sym_box: bool = False) -> MatchingData:
    """Enumerate the closure and each pair's sparse contribution data."""
    expansions = [multiplier_expansion(t, sym_box) for t in pairs]

    closure: set[Monomial] = set()
    for pair, expansion in zip(pairs, expansions):
        for mu, _ in expansion:
            closure.add(mu)
            for ai in range(len(pair.supp)):
                a = pair.supp[ai]
                closure.add(tuple(sorted(mu + (a,))))
                for b in pair.supp[ai:]:
                    closure.add(tuple(sorted(mu + (a, b))))

    missing = [m for m in poly.coeffs if m not in closure]
    if missing:
        raise ValueError(
            f"multiplier family cannot produce {len(missing)} monomials of p, "
            f"e.g. {missing[:3]}; matching would be infeasible"
        )

    closure_list = sorted(closure, key=lambda m: (len(m), m))
    index = {m: i for i, m in enumerate(closure_list)}

    contribs = []
    for pair, expansion in zip(pairs, expansions):
        contrib = PairContrib(pair)
        k = len(pair.supp)
        for mu, sigma in expansion:
            contrib.w_entries.append((index[mu], sigma))
            for ai in range(k):
                a = pair.supp[ai]
                contrib.r_entries.append(
                    (index[tuple(sorted(mu + (a,)))], ai, sigma)
                )
                for bi in range(ai, k):
                    b = pair.supp[bi]
                    contrib.P_entries.append(
                        (index[tuple(sorted(mu + (a, b)))], ai, bi, sigma)
                    )
        contribs.append(contrib)

    rhs = np.zeros(len(closure_list))
    for m, c in poly.coeffs.items():
        rhs[index[m]] = c

    return MatchingData(list(pairs), closure_list, index, contribs, rhs, sym_box)


def reconstruct_polynomial(data: MatchingData, Z: list[tuple]) -> SparsePolynomial:
    """Expand sum_t f_t * q_t symbolically from concrete coefficients
    Z = [(P_t, r_t, w_t)]. Used to verify matching correctness in tests."""
    total = SparsePolynomial()
    for pair, (P, r, w) in zip(data.pairs, Z):
        k = len(pair.supp)
        quad = SparsePolynomial()
        quad.add_term((), float(w))
        for ai in range(k):
            quad.add_term((pair.supp[ai],), float(r[ai]))
            for bi in range(k):
                quad.add_term(
                    (pair.supp[ai], pair.supp[bi]), float(P[ai, bi])
                )
        f = SparsePolynomial()
        for mu, sigma in multiplier_expansion(pair, data.sym_box):
            f.add_term(mu, sigma)
        total = total + f * quad
    return total


def feasibility_check(data: MatchingData, solver: str = "CLARABEL"
                      ) -> tuple[float, list[tuple] | None]:
    """Maximize the Slater margin gamma s.t. coefficient matching holds
    with P_t >= gamma*I for all pairs.

    gamma > 0 certifies strict feasibility of the decomposition set Z,
    which is required for the dual SDP bound to be valid (strong duality
    of the inner maximization). Returns (gamma, Z sample or None).
    """
    import cvxpy as cp

    gamma = cp.Variable()
    P_vars, r_vars, w_vars = [], [], []
    exprs: dict[int, list] = {}
    for contrib in data.contribs:
        k = len(contrib.pair.supp)
        P = cp.Variable((k, k), symmetric=True)
        r = cp.Variable(k)
        w = cp.Variable()
        P_vars.append(P), r_vars.append(r), w_vars.append(w)
        for m_idx, a, b, sigma in contrib.P_entries:
            fold = 1.0 if a == b else 2.0
            exprs.setdefault(m_idx, []).append(fold * sigma * P[a, b])
        for m_idx, a, sigma in contrib.r_entries:
            exprs.setdefault(m_idx, []).append(sigma * r[a])
        for m_idx, sigma in contrib.w_entries:
            exprs.setdefault(m_idx, []).append(sigma * w)

    constraints = []
    for m_idx in range(data.n_constraints):
        terms = exprs.get(m_idx)
        target = data.rhs[m_idx]
        if terms is None:
            if target != 0.0:
                return -np.inf, None
            continue
        constraints.append(cp.sum(cp.hstack(terms)) == target)
    for P in P_vars:
        constraints.append(P - gamma * np.eye(P.shape[0]) >> 0)
    # keep gamma bounded so the problem is never unbounded
    constraints.append(gamma <= 1.0)

    prob = cp.Problem(cp.Maximize(gamma), constraints)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return -np.inf, None
    Z = [
        (np.atleast_2d(P.value), np.atleast_1d(r.value), float(w.value))
        for P, r, w in zip(P_vars, r_vars, w_vars)
    ]
    return float(gamma.value), Z
