"""Multiplier families for SLC decompositions.

A multiplier pair (I, J, supp) represents the linear-part factor
f_{I,J}(y) = prod_{i in I} y_i * prod_{j in J} (1 - y_j) over [0,1]^n,
where I and J are multisets (repeats allowed, e.g. y_i^2), paired with a
convex quadratic q(y) supported on the variables in `supp`.

The full family of order d-2 realizes the paper's Theorems 1/4/6 and is
only viable for small n; restricted families (smaller pair sets, smaller
quadratic supports) still give valid lower bounds provided coefficient
matching stays (strictly) feasible.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from polyopt.sparse_poly import Monomial


@dataclass(frozen=True)
class MultiplierPair:
    I: tuple[int, ...]        # multiset of plain variables  (y_i)
    J: tuple[int, ...]        # multiset of complements      (1 - y_j)
    supp: tuple[int, ...]     # quadratic support (distinct, sorted)

    @property
    def order(self) -> int:
        return len(self.I) + len(self.J)

    def __repr__(self) -> str:
        return f"Pair(I={self.I}, J={self.J}, |supp|={len(self.supp)})"


def full_family(n_vars: int, order: int, supp: tuple[int, ...] | None = None
                ) -> list[MultiplierPair]:
    """All multiplier pairs of order <= `order` with full quadratic support.

    Matches the paper's families: order 1 gives {1, y_i, 1-y_i} (degree 3,
    Theorem 1); order 2 adds y_i y_j, y_i (1-y_j), (1-y_i)(1-y_j)
    (degree 4, Theorem 4); order 4 covers degree 6 (Theorem 6).
    Non-disjoint I/J (e.g. y_i(1-y_i)) are included, as in the paper's
    degree-4 family. Size grows as O(n^order) - small n only.
    """
    if supp is None:
        supp = tuple(range(n_vars))
    pairs = []
    for k1 in range(order + 1):
        for k2 in range(order + 1 - k1):
            for I in itertools.combinations_with_replacement(range(n_vars), k1):
                for J in itertools.combinations_with_replacement(range(n_vars), k2):
                    pairs.append(MultiplierPair(I, J, supp))
    return pairs


def _sub_multisets(m: Monomial, max_size: int) -> set[Monomial]:
    """All distinct sub-multisets of m with size <= max_size."""
    out: set[Monomial] = set()
    for k in range(min(max_size, len(m)) + 1):
        for sub in itertools.combinations(m, k):
            out.add(tuple(sorted(sub)))
    return out


def support_driven_family(
    monomials: list[Monomial],
    order: int = 4,
    include_complements: bool = False,
    complement_splits: bool = False,
    max_flips: int | None = None,
) -> list[MultiplierPair]:
    """Family tailored to a sparse polynomial's monomial support (given in
    ORIGINAL, pre-shift coordinates - the affine box shift only produces
    divisor monomials, which this family covers by construction).

    For each monomial m with |m| >= 3, every sub-multiset I of m with
    |I| <= min(order, |m| - 2) becomes a pair (I, emptyset) whose quadratic
    is supported on vars(m); a global order-0 pair over all active
    variables covers degree <= 2 terms. Pairs with equal (I, J) are merged
    by taking the union of their supports. With `include_complements`,
    matching complement pairs (emptyset, J) are added, enriching the
    decomposition space (stronger bounds, bigger SDP).
    """
    supp_by_key: dict[tuple[Monomial, Monomial], set[int]] = {}
    active: set[int] = set()

    def add(I: Monomial, J: Monomial, supp: set[int]):
        supp_by_key.setdefault((I, J), set()).update(supp)

    for m in monomials:
        active.update(m)
        if len(m) < 3:
            continue
        vars_m = set(m)
        for I in _sub_multisets(m, min(order, len(m) - 2)):
            add(I, (), vars_m)
            if include_complements and I:
                add((), I, vars_m)
            if complement_splits and I:
                # all ways to flip a subset of I into (1 - y) factors,
                # mirroring the mixed pairs of Theorems 4/6; max_flips
                # caps the flipped-subset size to control family growth
                flip_cap = len(I) if max_flips is None else max_flips
                for k in range(len(I) + 1):
                    if len(I) - k > flip_cap:
                        continue
                    for keep in itertools.combinations(range(len(I)), k):
                        keep_set = set(keep)
                        I1 = tuple(I[t] for t in range(len(I)) if t in keep_set)
                        J1 = tuple(
                            I[t] for t in range(len(I)) if t not in keep_set
                        )
                        add(tuple(sorted(I1)), tuple(sorted(J1)), vars_m)

    add((), (), active)
    for i in sorted(active):
        add((i,), (), active)
        add((), (i,), active)

    pairs = [
        MultiplierPair(I, J, tuple(sorted(supp)))
        for (I, J), supp in supp_by_key.items()
    ]
    return sorted(pairs, key=lambda t: (t.order, t.I, t.J))
