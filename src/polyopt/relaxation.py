"""The dual "best SLC decomposition" SDP over [0,1]^n.

Realizes Theorem 2 (degree 3), Theorem 5 (degree 4) and the degree-6
statement in docs/general_thm.md with one general assembly driven by the
matching data: for each multiplier pair t with quadratic support size k,

    dual-PSD:    -Y_t - sum_m lambda_m A_m^t  >= 0        (k x k)
    moment LMI:  G_t = [[Y_t, t_t], [t_t^T, s_t]] >= 0    (k+1 x k+1)
    ties:        t_t[a] + sum_m lambda_m c_m^t[a] = 0
                 s_t    + sum_m lambda_m mu_m^t   = 0

where s_t / t_t[a] are the linearized multiplier moments
    s_t = sum_S sigma_S M[I+S],   t_t[a] = sum_S sigma_S M[I+S+(a)],
plus box-RLT inequalities and the top-level moment LMI [[X, x],[x^T, 1]].
The optimal value  -sum_m lambda_m p_m  is a certified lower bound on
min p over [0,1]^n (given strict feasibility of the matching system).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from polyopt.sparse_poly import Monomial, mono_divisors
from polyopt.slc_constraints import MatchingData, multiplier_expansion


class MomentIndex:
    """Maps lifted monomials (degree >= 1) to columns of the moment vector.

    The empty monomial is the constant 1 and is handled separately.
    The index is closed under divisors so RLT chains always resolve.
    """

    def __init__(self):
        self.index: dict[Monomial, int] = {}

    def add(self, m: Monomial) -> None:
        for d in mono_divisors(m):
            if d and d not in self.index:
                self.index[d] = len(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, m: Monomial) -> bool:
        return m in self.index

    def __getitem__(self, m: Monomial) -> int:
        return self.index[m]


@dataclass
class RelaxationData:
    """Backend-neutral description of the assembled SDP."""

    matching: MatchingData
    moments: MomentIndex
    n_vars: int
    # per pair: sparse maps (see assemble())
    P_maps: list[sp.csr_matrix]      # (k*k, L)  lambda -> vec(sum lambda A)
    C_maps: list[sp.csr_matrix]      # (k, L)    lambda -> tie residual for t_t
    Mu_maps: list[sp.csr_matrix]     # (1, L)    lambda -> tie residual for s_t
    T_maps: list[sp.csr_matrix]      # (k, nM)   moments -> t_t
    S_maps: list[sp.csr_matrix]      # (1, nM)   moments -> s_t (linear part)
    S_consts: list[float]            # constant part of s_t
    rlt_rows: sp.csr_matrix          # (nR, nM)  rows: R @ M + rlt_const >= 0
    rlt_const: np.ndarray


def build_relaxation(matching: MatchingData, n_vars: int,
                     top_lmi: bool = True) -> RelaxationData:
    moments = MomentIndex()
    expansions = [
        multiplier_expansion(t, matching.sym_box) for t in matching.pairs
    ]
    for pair, expansion in zip(matching.pairs, expansions):
        for mu, _ in expansion:
            if mu:
                moments.add(mu)
            for a in pair.supp:
                moments.add(tuple(sorted(mu + (a,))))
    if top_lmi:
        for i in range(n_vars):
            for j in range(i, n_vars):
                moments.add((i, j))

    nM = len(moments)
    L = matching.n_constraints

    P_maps, C_maps, Mu_maps, T_maps, S_maps, S_consts = [], [], [], [], [], []
    for pair, contrib, expansion in zip(
        matching.pairs, matching.contribs, expansions
    ):
        k = len(pair.supp)
        rows, cols, vals = [], [], []
        for m_idx, a, b, sigma in contrib.P_entries:
            rows.append(a * k + b), cols.append(m_idx), vals.append(sigma)
            if a != b:
                rows.append(b * k + a), cols.append(m_idx), vals.append(sigma)
        P_maps.append(sp.csr_matrix((vals, (rows, cols)), shape=(k * k, L)))

        rows, cols, vals = [], [], []
        for m_idx, a, sigma in contrib.r_entries:
            rows.append(a), cols.append(m_idx), vals.append(sigma)
        C_maps.append(sp.csr_matrix((vals, (rows, cols)), shape=(k, L)))

        rows, cols, vals = [], [], []
        for m_idx, sigma in contrib.w_entries:
            rows.append(0), cols.append(m_idx), vals.append(sigma)
        Mu_maps.append(sp.csr_matrix((vals, (rows, cols)), shape=(1, L)))

        t_rows, t_cols, t_vals = [], [], []
        s_cols, s_vals, s_const = [], [], 0.0
        for mu, sigma in expansion:
            if mu:
                s_cols.append(moments[mu]), s_vals.append(sigma)
            else:
                s_const += sigma
            for ai, a in enumerate(pair.supp):
                t_rows.append(ai)
                t_cols.append(moments[tuple(sorted(mu + (a,)))])
                t_vals.append(sigma)
        T_maps.append(sp.csr_matrix((t_vals, (t_rows, t_cols)), shape=(k, nM)))
        S_maps.append(
            sp.csr_matrix((s_vals, ([0] * len(s_cols), s_cols)), shape=(1, nM))
        )
        S_consts.append(s_const)

    rlt = _build_rlt(moments, nM, matching.sym_box)

    return RelaxationData(
        matching, moments, n_vars,
        P_maps, C_maps, Mu_maps, T_maps, S_maps, S_consts,
        rlt[0], rlt[1],
    )


def _build_rlt(moments: MomentIndex, nM: int, sym_box: bool = False
               ) -> tuple[sp.csr_matrix, np.ndarray]:
    """Box RLT rows over the moment vector: each row r with constant c
    encodes r @ M + c >= 0.

    For every indexed monomial m and every split m = alpha + beta into a
    plus part and a complement part, emit the linearization of
        [0,1]:   y^alpha * prod_{b in beta} (1 - y_b)      >= 0
        [-1,1]:  prod_{a in alpha} (1+y_a) * prod (1-y_b)  >= 0
    This is the complete set of box-constraint products realizable over
    the moment index (it subsumes nonnegativity/monotonicity and the
    pairwise level-2 rows). All rows are valid for true moments."""
    import itertools
    from collections import Counter

    from polyopt.slc_constraints import expand_box_product

    rows, cols, vals, consts = [], [], [], []
    row = 0

    def emit(entries: list[tuple[Monomial, float]]):
        nonlocal row
        const = 0.0
        for m_key, v in entries:
            if m_key:
                rows.append(row), cols.append(moments[m_key]), vals.append(v)
            else:
                const += v
        consts.append(const)
        row += 1

    for m in list(moments.index):
        counts = Counter(m)
        variables = sorted(counts)
        # each split assigns j of the k copies of each variable to beta
        for choice in itertools.product(
            *(range(counts[v] + 1) for v in variables)
        ):
            alpha, beta = [], []
            for v, j in zip(variables, choice):
                beta.extend([v] * j)
                alpha.extend([v] * (counts[v] - j))
            emit(expand_box_product(tuple(alpha), tuple(beta), sym_box))

    mat = sp.csr_matrix((vals, (rows, cols)), shape=(row, nM))
    return mat, np.asarray(consts)


def solve_relaxation_cvxpy(
    data: RelaxationData,
    solver: str = "CLARABEL",
    top_lmi: bool = True,
    verbose: bool = False,
    extra_ineq: tuple[sp.csr_matrix, np.ndarray] | None = None,
) -> dict:
    """Assemble and solve the dual SDP with cvxpy. Returns a dict with
    the certified lower bound, first/second moments, and diagnostics."""
    import cvxpy as cp

    nM = len(data.moments)
    L = data.matching.n_constraints
    M = cp.Variable(nM)
    lam = cp.Variable(L)

    constraints = []
    for k_pair in range(len(data.matching.pairs)):
        k = len(data.matching.pairs[k_pair].supp)
        G = cp.Variable((k + 1, k + 1), PSD=True)
        dualA = cp.reshape(data.P_maps[k_pair] @ lam, (k, k), order="C")
        constraints.append(-G[:k, :k] - dualA >> 0)
        t_expr = data.T_maps[k_pair] @ M
        s_expr = data.S_maps[k_pair] @ M + data.S_consts[k_pair]
        constraints.append(G[k, :k] == t_expr)
        constraints.append(G[k, k] == s_expr[0])
        constraints.append(t_expr + data.C_maps[k_pair] @ lam == 0)
        constraints.append(s_expr + data.Mu_maps[k_pair] @ lam == 0)

    constraints.append(data.rlt_rows @ M + data.rlt_const >= 0)
    if extra_ineq is not None:
        A_extra, b_extra = extra_ineq
        constraints.append(A_extra @ M + b_extra >= 0)

    if top_lmi:
        n = data.n_vars
        rows, cols, vals = [], [], []
        for i in range(n):
            for j in range(i, n):
                idx = data.moments[(i, j)]
                rows.append(i * n + j), cols.append(idx), vals.append(1.0)
                if i != j:
                    rows.append(j * n + i), cols.append(idx), vals.append(1.0)
        Q = sp.csr_matrix((vals, (rows, cols)), shape=(n * n, nM))
        X = cp.reshape(Q @ M, (n, n), order="C")
        rows = [i for i in range(n)]
        P1 = sp.csr_matrix(
            (np.ones(n), (rows, [data.moments[(i,)] for i in range(n)])),
            shape=(n, nM),
        )
        x = P1 @ M
        top = cp.bmat([[X, cp.reshape(x, (n, 1), order="C")],
                       [cp.reshape(x, (1, n), order="C"), np.ones((1, 1))]])
        constraints.append(top >> 0)

    objective = cp.Minimize(-(data.matching.rhs @ lam))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    result = {
        "status": prob.status,
        "bound": float(prob.value) if prob.value is not None else None,
        "n_moments": nM,
        "n_lambda": L,
        "n_pairs": len(data.matching.pairs),
        "solve_time": getattr(prob.solver_stats, "solve_time", None),
    }
    if M.value is not None:
        result["x"] = np.array(
            [
                M.value[data.moments[(i,)]] if (i,) in data.moments else 0.0
                for i in range(data.n_vars)
            ]
        )
        n = data.n_vars
        X_val = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                key = (i, j)
                if key in data.moments:
                    X_val[i, j] = X_val[j, i] = M.value[data.moments[key]]
        result["X"] = X_val
    return result
