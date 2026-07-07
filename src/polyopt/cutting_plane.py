"""Cutting-plane (adversarial) mode for the best-SLC bound (paper Remark 4).

Instead of dualizing the inner maximization over all decompositions Z
(which yields one PSD block pair per multiplier and a lambda variable per
closure monomial - the memory bottleneck at scale), alternate between:

  master:  min_{M in X_bar} tau  s.t.  tau >= g(M, Z_i) for scenarios Z_i
           - one rotated second-order cone per (scenario, active pair);
           no lambda vector, no dual-PSD blocks.
  pessimization:  Z_{k+1} = argmax_{Z in Z_rho} g(M_hat, Z)
           - a block-diagonal SDP of the same size as the feasibility
           check (matching equalities + PSD cones + trace bounds).

Every master value is a certified lower bound on min p over the box
(the finite scenario set is a subset of Z, so the master underestimates
the full min-max, which underestimates the true minimum). The loop stops
when pessimization cannot raise g above tau by more than `tol`, or at
`max_iters`; either way the last master value is reported and valid.

The trace/entry bound rho keeps pessimization bounded; it restricts Z to
Z_rho (a subset), so validity is unaffected - only the achievable
tightness, which increases with rho.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from polyopt.relaxation import RelaxationData

Scenario = list[tuple[np.ndarray, np.ndarray, float]]  # per pair (P, r, w)

_EIG_TOL = 1e-9
# s_t >= 0 is enforced in the master via RLT (f_t >= 0 linearization is a
# split of the indexed monomial I+J), so s_hat is only negative by solver
# tolerance; the clamp guards the 1/s_hat scale in the pessimization.
_S_CLAMP = 1e-6


def _scenario_factors(Z: Scenario) -> list[np.ndarray | None]:
    """Cholesky-like factors L with P = L L^T (eigendecomposition with
    clipped negatives; None when the quadratic part is numerically zero)."""
    factors = []
    for P, _, _ in Z:
        vals, vecs = np.linalg.eigh(P)
        keep = vals > _EIG_TOL
        if not np.any(keep):
            factors.append(None)
            continue
        factors.append(vecs[:, keep] * np.sqrt(vals[keep]))
    return factors


def _build_master_base(model, data: RelaxationData, top_lmi, extra_ineq):
    """Moment variable + X_bar constraints (RLT, top LMI, extra cuts)."""
    from mosek.fusion import Domain, Expr

    from polyopt.mosek_backend import _fusion_sparse

    nM = len(data.moments)
    n = data.n_vars
    M = model.variable("M", nM, Domain.unbounded())
    model.constraint(
        Expr.mul(_fusion_sparse(data.rlt_rows), M),
        Domain.greaterThan([-c for c in data.rlt_const]),
    )
    if extra_ineq is not None and extra_ineq[0].shape[0]:
        model.constraint(
            Expr.mul(_fusion_sparse(extra_ineq[0]), M),
            Domain.greaterThan([-c for c in extra_ineq[1]]),
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
            (np.ones(n),
             (list(range(n)), [data.moments[(i,)] for i in range(n)])),
            shape=(n, nM),
        )
        X = Expr.reshape(Expr.mul(_fusion_sparse(Q), M), [n, n])
        xcol = Expr.reshape(Expr.mul(_fusion_sparse(P1), M), [n, 1])
        xrow = Expr.reshape(Expr.mul(_fusion_sparse(P1), M), [1, n])
        top = Expr.vstack(
            Expr.hstack(X, xcol),
            Expr.hstack(xrow, Expr.constTerm(np.ones((1, 1)))),
        )
        model.constraint(top, Domain.inPSDCone(n + 1))
    return M


def _add_scenario_epigraphs(model, data: RelaxationData, M,
                            scenarios: list[Scenario]):
    """Return one affine+cone expression g_i(M) per scenario, as a list of
    Fusion scalar expressions (each g_i is exact, via rotated cones)."""
    from mosek.fusion import Domain, Expr

    from polyopt.mosek_backend import _fusion_sparse

    nM = len(data.moments)
    n_pairs = len(data.matching.pairs)
    g_exprs = []
    for Z in scenarios:
        factors = _scenario_factors(Z)
        lin_row = sp.csr_matrix((1, nM))
        lin_const = 0.0
        quad_terms = []
        for t in range(n_pairs):
            P, r, w = Z[t]
            lin_row = lin_row + sp.csr_matrix(r) @ data.T_maps[t] \
                + w * data.S_maps[t]
            lin_const += w * data.S_consts[t]
            Lf = factors[t]
            if Lf is None:
                continue
            u = model.variable(1, Domain.greaterThan(0.0))
            s_expr = Expr.add(
                Expr.flatten(Expr.mul(_fusion_sparse(data.S_maps[t]), M)),
                Expr.constTerm([data.S_consts[t]]),
            )
            z_expr = Expr.mul(
                _fusion_sparse(sp.csr_matrix(Lf.T) @ data.T_maps[t]), M
            )
            model.constraint(
                Expr.vstack(u, s_expr, z_expr),
                Domain.inRotatedQCone(2 + Lf.shape[1]),
            )
            quad_terms.append(Expr.mul(2.0, u.index(0)))
        g_expr = Expr.add(
            Expr.flatten(Expr.mul(_fusion_sparse(lin_row.tocsr()), M)),
            Expr.constTerm([lin_const]),
        )
        if quad_terms:
            g_expr = Expr.add(g_expr, Expr.add(quad_terms)
                              if len(quad_terms) > 1 else quad_terms[0])
        g_exprs.append(g_expr)
    return g_exprs


def solve_master(
    data: RelaxationData,
    scenarios: list[Scenario],
    top_lmi: bool = True,
    extra_ineq: tuple[sp.csr_matrix, np.ndarray] | None = None,
    threads: int = 0,
    level: float | None = None,
    center: np.ndarray | None = None,
) -> dict:
    """Scenario master problem over the moment vector.

    level=None: Kelley step - minimize tau s.t. tau >= g_i(M); the optimal
    value is a valid lower bound on the min-max (hence on min p).
    level=l with a center: level-bundle step - minimize ||M - center||
    s.t. g_i(M) <= l. Primal infeasibility proves min-max >= l, which the
    caller uses to raise its lower bound.
    """
    from mosek.fusion import Domain, Expr, Model, ObjectiveSense
    from mosek.fusion import ProblemStatus, SolutionStatus

    with Model("slc-master") as model:
        if threads:
            model.setSolverParam("numThreads", threads)
        M = _build_master_base(model, data, top_lmi, extra_ineq)
        g_exprs = _add_scenario_epigraphs(model, data, M, scenarios)

        if level is None:
            tau = model.variable("tau", 1, Domain.unbounded())
            for g in g_exprs:
                model.constraint(
                    Expr.sub(tau.index(0), g), Domain.greaterThan(0.0)
                )
            model.objective(ObjectiveSense.Minimize, tau.index(0))
        else:
            for g in g_exprs:
                model.constraint(g, Domain.lessThan(level))
            dist = model.variable("dist", 1, Domain.greaterThan(0.0))
            model.constraint(
                Expr.vstack(dist.index(0), Expr.sub(M, list(center))),
                Domain.inQCone(1 + len(center)),
            )
            model.objective(ObjectiveSense.Minimize, dist.index(0))

        model.solve()
        status = model.getPrimalSolutionStatus()
        if status != SolutionStatus.Optimal:
            infeasible = model.getProblemStatus() in (
                ProblemStatus.PrimalInfeasible,
            )
            return {"status": str(status), "bound": None, "M": None,
                    "infeasible": infeasible}
        result = {
            "status": str(status),
            "M": np.array(M.level()),
            "infeasible": False,
        }
        if level is None:
            result["bound"] = float(
                min(model.primalObjValue(), model.dualObjValue())
            )
        return result


def pessimize(
    data: RelaxationData,
    M_hat: np.ndarray,
    rho: float = 100.0,
    threads: int = 0,
) -> tuple[float, Scenario]:
    """max_{Z in Z_rho} g(M_hat, Z): linear SDP over the decomposition
    coefficients with matching equalities, P_t >= 0, and trace/entry
    bounds rho for boundedness."""
    from mosek.fusion import Domain, Expr, Model, ObjectiveSense
    from mosek.fusion import SolutionStatus

    from polyopt.mosek_backend import _fusion_sparse

    matching = data.matching
    L = matching.n_constraints
    with Model("slc-pessimize") as model:
        if threads:
            model.setSolverParam("numThreads", threads)
        eq_exprs = []
        obj_terms = []
        P_vars, r_vars, w_vars = [], [], []
        for t, contrib in enumerate(matching.contribs):
            k = len(contrib.pair.supp)
            P = model.variable(Domain.inPSDCone(k))
            r = model.variable(k, Domain.inRange(-rho, rho))
            w = model.variable(1, Domain.inRange(-rho, rho))
            P_vars.append(P), r_vars.append(r), w_vars.append(w)
            model.constraint(
                Expr.sum(P.diag()), Domain.lessThan(rho)
            )

            s_hat = float((data.S_maps[t] @ M_hat)[0] + data.S_consts[t])
            t_hat = np.asarray(data.T_maps[t] @ M_hat).ravel()
            W = np.outer(t_hat, t_hat) / max(s_hat, _S_CLAMP)
            obj_terms.append(Expr.dot(W, P))
            obj_terms.append(Expr.dot(list(t_hat), r))
            obj_terms.append(Expr.mul(max(s_hat, 0.0), w.index(0)))

            rows, cols, vals = [], [], []
            for m_idx, a, b, sigma in contrib.P_entries:
                rows.append(m_idx), cols.append(a * k + b), vals.append(sigma)
                if a != b:
                    rows.append(m_idx), cols.append(b * k + a), vals.append(sigma)
            EP = sp.csr_matrix((vals, (rows, cols)), shape=(L, k * k))
            eq_exprs.append(Expr.mul(_fusion_sparse(EP), Expr.flatten(P)))
            rows, cols, vals = [], [], []
            for m_idx, a, sigma in contrib.r_entries:
                rows.append(m_idx), cols.append(a), vals.append(sigma)
            Er = sp.csr_matrix((vals, (rows, cols)), shape=(L, k))
            eq_exprs.append(Expr.mul(_fusion_sparse(Er), r))
            rows, vals = [], []
            for m_idx, sigma in contrib.w_entries:
                rows.append(m_idx), vals.append(sigma)
            Ew = sp.csr_matrix((vals, (rows, [0] * len(rows))), shape=(L, 1))
            eq_exprs.append(Expr.mul(_fusion_sparse(Ew), w))

        model.constraint(
            Expr.add(eq_exprs), Domain.equalsTo(list(matching.rhs))
        )
        model.objective(ObjectiveSense.Maximize, Expr.add(obj_terms))
        model.solve()
        if model.getPrimalSolutionStatus() != SolutionStatus.Optimal:
            return float("-inf"), None
        Z = []
        for contrib, P, r, w in zip(
            matching.contribs, P_vars, r_vars, w_vars
        ):
            k = len(contrib.pair.supp)
            Z.append((
                np.array(P.level()).reshape(k, k),
                np.array(r.level()),
                float(w.level()[0]),
            ))
        return float(model.primalObjValue()), Z


def solve_cutting_plane(
    data: RelaxationData,
    top_lmi: bool = True,
    extra_ineq: tuple[sp.csr_matrix, np.ndarray] | None = None,
    max_iters: int = 30,
    tol: float = 1e-4,
    rho: float = 100.0,
    threads: int = 0,
    verbose: bool = False,
) -> dict:
    """Level-bundle master/pessimization loop.

    Invariants: LB is always a valid lower bound on min p over the box
    (Kelley master value, or a level proven infeasible); V = min over
    evaluated pessimization values is an upper bound on the min-max
    target, so the loop stops when V - LB <= tol. `bound`=LB is valid
    even on early stop. kappa in (0,1) sets the level between LB and V.
    """
    from polyopt.mosek_backend import feasibility_check_mosek

    kappa = 0.4
    t0 = time.perf_counter()
    gamma, Z0 = feasibility_check_mosek(data.matching, return_Z=True)
    if Z0 is None or gamma <= 0:
        return {"status": f"slater_failed (gamma={gamma})", "bound": None}

    scenarios: list[Scenario] = [Z0]
    history = []
    status = "max_iters"

    # initial Kelley step: valid LB + starting center
    master = solve_master(data, scenarios, top_lmi=top_lmi,
                          extra_ineq=extra_ineq, threads=threads)
    if master.get("bound") is None:
        return {"status": f"master_failed ({master['status']})",
                "bound": None}
    LB, center, M_hat = master["bound"], master["M"], master["M"]
    v, Z_new = pessimize(data, center, rho=rho, threads=threads)
    V = v
    if Z_new is not None:
        scenarios.append(Z_new)
    history.append((LB, v))
    if verbose:
        print(f"  cp init: LB {LB:.6f}, pessimization {v:.6f}")

    for it in range(max_iters):
        if V - LB <= tol * max(1.0, abs(LB)):
            status = "converged"
            break
        level = LB + kappa * (V - LB)
        step = solve_master(
            data, scenarios, top_lmi=top_lmi, extra_ineq=extra_ineq,
            threads=threads, level=level, center=center,
        )
        if step["infeasible"]:
            # no moment point keeps all scenarios below the level, so
            # the min-max (and hence min p) is at least `level`
            LB = level
            history.append((LB, None))
            if verbose:
                print(f"  cp iter {it}: level {level:.6f} infeasible "
                      f"-> LB {LB:.6f}")
            continue
        if step["M"] is None:
            # numerically undecided near the boundary: refresh LB with a
            # plain Kelley master over all accumulated scenarios
            kelley = solve_master(data, scenarios, top_lmi=top_lmi,
                                  extra_ineq=extra_ineq, threads=threads)
            if kelley.get("bound") is None:
                status = f"master_failed ({step['status']})"
                break
            LB = max(LB, kelley["bound"])
            M_hat = kelley["M"]
            if verbose:
                print(f"  cp iter {it}: level step undecided, Kelley "
                      f"refresh -> LB {LB:.6f}")
        else:
            M_hat = step["M"]
        v, Z_new = pessimize(data, M_hat, rho=rho, threads=threads)
        history.append((LB, v))
        if verbose:
            print(f"  cp iter {it}: LB {LB:.6f}, level {level:.6f}, "
                  f"pessimization {v:.6f}")
        if Z_new is None:
            status = "pessimization_failed"
            break
        if v < V:
            V = v
            center = M_hat      # serious step: recentre at the improver
        scenarios.append(Z_new)
    bound = LB

    result = {
        "status": status,
        "bound": bound,
        "n_moments": len(data.moments),
        "n_lambda": data.matching.n_constraints,
        "n_pairs": len(data.matching.pairs),
        "n_scenarios": len(scenarios),
        "history": history,
        "solve_time": time.perf_counter() - t0,
        "slater_gamma": gamma,
    }
    if M_hat is not None:
        n = data.n_vars
        result["x"] = np.array(
            [M_hat[data.moments[(i,)]] if (i,) in data.moments else 0.0
             for i in range(n)]
        )
        X_val = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if (i, j) in data.moments:
                    X_val[i, j] = X_val[j, i] = M_hat[data.moments[(i, j)]]
        result["X"] = X_val
    return result
