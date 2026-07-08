"""MOSEK Fusion backend for the dual best-SLC SDP.

Consumes the same RelaxationData as the cvxpy backend and must produce
the same optimal value (tests enforce parity). Preferred at scale: no
cvxpy compilation overhead and much better SDP performance.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from polyopt.relaxation import RelaxationData


def _fusion_sparse(mat: sp.spmatrix):
    from mosek.fusion import Matrix

    coo = mat.tocoo()
    return Matrix.sparse(
        int(coo.shape[0]),
        int(coo.shape[1]),
        [int(i) for i in coo.row],
        [int(j) for j in coo.col],
        [float(v) for v in coo.data],
    )


def feasibility_check_mosek(data, tol: float = 1e-8, return_Z: bool = False):
    """Fusion version of slc_constraints.feasibility_check (Slater margin).

    Maximizes gamma s.t. the coefficient-matching equalities hold with
    P_t >= gamma*I; returns gamma (or -inf if infeasible). Much faster
    than the cvxpy path on large families. With return_Z=True returns
    (gamma, [(P_t, r_t, w_t)]) - a strictly feasible decomposition usable
    as the initial cutting-plane scenario.
    """
    import numpy as np
    from mosek.fusion import Domain, Expr, Model, ObjectiveSense
    from mosek.fusion import SolutionStatus

    L = data.n_constraints
    with Model("slc-feasibility") as model:
        gamma = model.variable(1, Domain.lessThan(1.0))
        exprs = []
        P_vars, r_vars, w_vars = [], [], []
        for contrib in data.contribs:
            k = len(contrib.pair.supp)
            P = model.variable(Domain.inPSDCone(k))
            r = model.variable(k, Domain.unbounded())
            w = model.variable(1, Domain.unbounded())
            P_vars.append(P), r_vars.append(r), w_vars.append(w)
            rows, cols, vals = [], [], []
            g_rows, g_vals = [], []
            for m_idx, a, b, sigma in contrib.P_entries:
                rows.append(m_idx), cols.append(a * k + b), vals.append(sigma)
                if a != b:
                    rows.append(m_idx), cols.append(b * k + a), vals.append(sigma)
                else:
                    g_rows.append(m_idx), g_vals.append(sigma)
            EP = sp.csr_matrix((vals, (rows, cols)), shape=(L, k * k))
            exprs.append(Expr.mul(_fusion_sparse(EP), Expr.flatten(P)))
            rows, cols, vals = [], [], []
            for m_idx, a, sigma in contrib.r_entries:
                rows.append(m_idx), cols.append(a), vals.append(sigma)
            Er = sp.csr_matrix((vals, (rows, cols)), shape=(L, k))
            exprs.append(Expr.mul(_fusion_sparse(Er), r))
            rows, vals = [], []
            for m_idx, sigma in contrib.w_entries:
                rows.append(m_idx), vals.append(sigma)
            Ew = sp.csr_matrix(
                (vals, (rows, [0] * len(rows))), shape=(L, 1)
            )
            exprs.append(Expr.mul(_fusion_sparse(Ew), w))
            if g_rows:
                Eg = sp.csr_matrix(
                    (g_vals, (g_rows, [0] * len(g_rows))), shape=(L, 1)
                )
                exprs.append(Expr.mul(_fusion_sparse(Eg), gamma))

        model.constraint(Expr.add(exprs), Domain.equalsTo(list(data.rhs)))
        model.objective(ObjectiveSense.Maximize, gamma.index(0))
        model.solve()
        if model.getPrimalSolutionStatus() != SolutionStatus.Optimal:
            return (float("-inf"), None) if return_Z else float("-inf")
        g = float(gamma.level()[0])
        if not return_Z:
            return g
        Z = []
        for contrib, P, r, w in zip(data.contribs, P_vars, r_vars, w_vars):
            k = len(contrib.pair.supp)
            # the matched quadratic is P_tilde + gamma*I
            P_val = np.array(P.level()).reshape(k, k) + g * np.eye(k)
            Z.append((P_val, np.array(r.level()), float(w.level()[0])))
        return g, Z


def solve_relaxation_mosek(
    data: RelaxationData,
    top_lmi: bool = True,
    verbose: bool = False,
    threads: int = 0,
    tol: float = 1e-9,
    extra_ineq: tuple[sp.csr_matrix, np.ndarray] | None = None,
    solver_params: dict | None = None,
) -> dict:
    """Assemble and solve with MOSEK Fusion. Returns the same result dict
    shape as solve_relaxation_cvxpy, plus primal/dual objective values."""
    from mosek.fusion import Domain, Expr, Model, ObjectiveSense
    from mosek.fusion import SolutionStatus

    nM = len(data.moments)
    L = data.matching.n_constraints
    n = data.n_vars

    with Model("best-slc") as model:
        if verbose:
            import sys

            model.setLogHandler(sys.stdout)
        if threads:
            model.setSolverParam("numThreads", threads)
        model.setSolverParam("intpntCoTolRelGap", tol)
        for key, val in (solver_params or {}).items():
            model.setSolverParam(key, val)

        Mvar = model.variable("M", nM, Domain.unbounded())
        lam = model.variable("lam", L, Domain.unbounded())

        for k_pair, pair in enumerate(data.matching.pairs):
            k = len(pair.supp)
            G = model.variable(Domain.inPSDCone(k + 1))
            Gtop = G.slice([0, 0], [k, k])
            dualA = Expr.reshape(
                Expr.mul(_fusion_sparse(data.P_maps[k_pair]), lam), [k, k]
            )
            model.constraint(
                Expr.neg(Expr.add(Gtop, dualA)), Domain.inPSDCone(k)
            )
            t_expr = Expr.mul(_fusion_sparse(data.T_maps[k_pair]), Mvar)
            t_row = Expr.flatten(G.slice([k, 0], [k + 1, k]))
            model.constraint(Expr.sub(t_row, t_expr), Domain.equalsTo(0.0))
            model.constraint(
                Expr.add(t_expr, Expr.mul(_fusion_sparse(data.C_maps[k_pair]), lam)),
                Domain.equalsTo(0.0),
            )
            s_lin = Expr.mul(_fusion_sparse(data.S_maps[k_pair]), Mvar)
            s_const = data.S_consts[k_pair]
            model.constraint(
                Expr.sub(G.index([k, k]), Expr.flatten(s_lin)),
                Domain.equalsTo(s_const),
            )
            model.constraint(
                Expr.add(
                    Expr.flatten(s_lin),
                    Expr.flatten(
                        Expr.mul(_fusion_sparse(data.Mu_maps[k_pair]), lam)
                    ),
                ),
                Domain.equalsTo(-s_const),
            )

        model.constraint(
            Expr.mul(_fusion_sparse(data.rlt_rows), Mvar),
            Domain.greaterThan([-c for c in data.rlt_const]),
        )
        if extra_ineq is not None:
            A_extra, b_extra = extra_ineq
            if A_extra.shape[0]:
                model.constraint(
                    Expr.mul(_fusion_sparse(A_extra), Mvar),
                    Domain.greaterThan([-c for c in b_extra]),
                )

        if top_lmi:
            rows, cols, vals = [], [], []
            for i in range(n):
                for j in range(i, n):
                    idx = data.moments[(i, j)]
                    rows.append(i * n + j), cols.append(idx), vals.append(1.0)
                    if i != j:
                        rows.append(j * n + i), cols.append(idx), vals.append(1.0)
            Q = sp.csr_matrix((vals, (rows, cols)), shape=(n * n, nM))
            P1 = sp.csr_matrix(
                (
                    np.ones(n),
                    (list(range(n)), [data.moments[(i,)] for i in range(n)]),
                ),
                shape=(n, nM),
            )
            X = Expr.reshape(Expr.mul(_fusion_sparse(Q), Mvar), [n, n])
            xcol = Expr.reshape(Expr.mul(_fusion_sparse(P1), Mvar), [n, 1])
            xrow = Expr.reshape(Expr.mul(_fusion_sparse(P1), Mvar), [1, n])
            one = Expr.constTerm(np.ones((1, 1)))
            top = Expr.vstack(Expr.hstack(X, xcol), Expr.hstack(xrow, one))
            model.constraint(top, Domain.inPSDCone(n + 1))

        model.objective(
            ObjectiveSense.Minimize,
            Expr.neg(Expr.dot(list(data.matching.rhs), lam)),
        )
        model.solve()

        status = model.getPrimalSolutionStatus()
        result = {
            "status": str(status),
            "bound": None,
            "n_moments": nM,
            "n_lambda": L,
            "n_pairs": len(data.matching.pairs),
            "solve_time": model.getSolverDoubleInfo("optimizerTime"),
        }
        if status in (SolutionStatus.Optimal,):
            primal = model.primalObjValue()
            dual = model.dualObjValue()
            # our SDP is a minimization whose optimal value is <= min p;
            # the solver's dual objective is a rigorous lower bound on the
            # SDP optimum, so it is the conservative value to report
            result["bound"] = float(min(primal, dual))
            result["primal_obj"] = float(primal)
            result["dual_obj"] = float(dual)
            Mval = np.array(Mvar.level())
            result["x"] = np.array(
                [
                    Mval[data.moments[(i,)]] if (i,) in data.moments else 0.0
                    for i in range(n)
                ]
            )
            X_val = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    if (i, j) in data.moments:
                        X_val[i, j] = X_val[j, i] = Mval[data.moments[(i, j)]]
            result["X"] = X_val
        return result
