"""
Naive Theorem-6 Best-SLC + RPT + spatial branch-and-bound for matmul Frobenius loss.

Minimal experiment scaffold for <2,2,2> rank 7. Mosek via cvxpy.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import sympy as sp

from tensor_utils import build_mult_tensor, verify_decomposition

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

_START: float = 0.0
_MAX_TIME: float = 3600.0


def _log(msg: str) -> None:
    elapsed = time.perf_counter() - _START
    remaining = max(0.0, _MAX_TIME - elapsed)
    print(f"[t={elapsed:.1f}s left={remaining:.1f}s] {msg}", flush=True)


def _time_up() -> bool:
    return (time.perf_counter() - _START) >= _MAX_TIME


# ---------------------------------------------------------------------------
# Polynomial builder
# ---------------------------------------------------------------------------

Monomial = Tuple[int, ...]
SLCTerm = Tuple[FrozenSet[int], FrozenSet[int]]


def _factor_var_names(m: int, p: int, n: int, rank: int) -> List[str]:
    d1, d2, d3 = m * p, p * n, m * n
    names: List[str] = []
    for r in range(rank):
        for prefix, dim in (("u", d1), ("v", d2), ("w", d3)):
            for i in range(dim):
                names.append(f"{prefix}{r}_{i}")
    return names


def build_matmul_loss_coeff_dict(
    m: int, p: int, n_dim: int, rank: int
) -> Tuple[Dict[Monomial, float], List[str], int, int]:
    """Build Frobenius loss ||T - sum u_r v_r w_r||^2 as sparse coeff dict."""
    T = build_mult_tensor(m, p, n_dim)
    d1, d2, d3 = T.shape
    names = _factor_var_names(m, p, n_dim, rank)
    syms = sp.symbols(names)
    n = len(syms)

    blocks: List[List] = []
    off = 0
    for _ in range(rank):
        u = syms[off : off + d1]
        off += d1
        v = syms[off : off + d2]
        off += d2
        w = syms[off : off + d3]
        off += d3
        blocks.append((u, v, w))

    t0 = time.perf_counter()
    expr = sp.Integer(0)
    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                tval = float(T[i, j, k])
                if tval == 0.0:
                    continue
                recon = sum(u[i] * v[j] * w[k] for u, v, w in blocks)
                diff = sp.Float(tval) - recon
                expr += diff**2

    poly = sp.Poly(sp.expand(expr), syms)
    coeff_dict: Dict[Monomial, float] = {}
    for exp, coeff in zip(poly.monoms(), poly.coeffs()):
        coeff_dict[tuple(int(e) for e in exp)] = float(coeff)

    degree = max(sum(e) for e in coeff_dict) if coeff_dict else 0
    _log(
        f"polynomial: n_vars={n} degree={degree} "
        f"nnz_monomials={len(coeff_dict)} build_s={time.perf_counter()-t0:.2f}"
    )
    return coeff_dict, names, n, degree


def unflatten_factors(
    x: np.ndarray, m: int, p: int, n_dim: int, rank: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d1, d2, d3 = m * p, p * n_dim, m * n_dim
    U = np.zeros((d1, rank))
    V = np.zeros((d2, rank))
    W = np.zeros((d3, rank))
    off = 0
    for r in range(rank):
        U[:, r] = x[off : off + d1]
        off += d1
        V[:, r] = x[off : off + d2]
        off += d2
        W[:, r] = x[off : off + d3]
        off += d3
    return U, V, W


def eval_poly(coeff_dict: Dict[Monomial, float], x: np.ndarray) -> float:
    val = 0.0
    for exp, c in coeff_dict.items():
        term = c
        for i, e in enumerate(exp):
            if e:
                term *= x[i] ** e
        val += term
    return val


# ---------------------------------------------------------------------------
# Box map: original [-2,2] <-> unit box [0,1]
# ---------------------------------------------------------------------------

ORIG_LO = -2.0
ORIG_HI = 2.0


def to_unit(x_orig: np.ndarray) -> np.ndarray:
    return (x_orig - ORIG_LO) / (ORIG_HI - ORIG_LO)


def from_unit(x_unit: np.ndarray) -> np.ndarray:
    return ORIG_LO + (ORIG_HI - ORIG_LO) * x_unit


# ---------------------------------------------------------------------------
# Theorem 6: SLC basis enumeration
# ---------------------------------------------------------------------------


def estimate_slc_term_count(n: int, max_k: int) -> int:
    return sum(math.comb(n, k) * (2**k) for k in range(max_k + 1))


def enumerate_slc_terms(n: int, max_k: int) -> Iterator[SLCTerm]:
    """All (I, J) disjoint with |I union J| <= max_k."""
    for k in range(max_k + 1):
        for idxs in itertools.combinations(range(n), k):
            idx_set = set(idxs)
            for mask in range(1 << k):
                I: set[int] = set()
                J: set[int] = set()
                for bit, idx in enumerate(idxs):
                    if mask & (1 << bit):
                        I.add(idx)
                    else:
                        J.add(idx)
                yield frozenset(I), frozenset(J)


def expand_f_monomials(I: FrozenSet[int], J: FrozenSet[int], n: int) -> Dict[Monomial, float]:
    """Expand f_{I,J}(x) = prod_I x_i prod_J (1-x_j) into monomial coefficients."""
    # Recursive expansion over J
    if not J:
        exp = [0] * n
        for i in I:
            exp[i] += 1
        return {tuple(exp): 1.0}

    j0 = min(J)
    Jrest = J - {j0}
    out: Dict[Monomial, float] = {}
    for sub, csub in expand_f_monomials(I, Jrest, n).items():
        # multiply by (1 - x_{j0})
        out[sub] = out.get(sub, 0.0) + csub
        sub2 = list(sub)
        sub2[j0] += 1
        key2 = tuple(sub2)
        out[key2] = out.get(key2, 0.0) - csub
    return out


def term_contrib_to_monomial(
    I: FrozenSet[int], J: FrozenSet[int], beta: Monomial, n: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Coefficient of monomial x^beta in f_{I,J}(x) * (x^T P x + r^T x + w).

    Returns (vec_P, vec_r, coeff_w) such that contribution =
        sum P_flat @ vec_P + r @ vec_r + coeff_w * w
    """
    f_mono = expand_f_monomials(I, J, n)
    vec_P = np.zeros(n * n)
    vec_r = np.zeros(n)
    coeff_w = 0.0

    for alpha, c_alpha in f_mono.items():
        # P part: alpha + e_k + e_l = beta
        for k in range(n):
            for l in range(n):
                gamma = list(alpha)
                gamma[k] += 1
                gamma[l] += 1
                if tuple(gamma) == beta:
                    vec_P[k * n + l] += c_alpha
                    if k != l:
                        vec_P[l * n + k] += c_alpha
        # r part: alpha + e_k = beta
        for k in range(n):
            gamma = list(alpha)
            gamma[k] += 1
            if tuple(gamma) == beta:
                vec_r[k] += c_alpha
        # w part: alpha = beta
        if alpha == beta:
            coeff_w += c_alpha

    return vec_P, vec_r, coeff_w


# ---------------------------------------------------------------------------
# Coefficient matching system
# ---------------------------------------------------------------------------


@dataclass
class MatchingSystem:
    monomials: List[Monomial]
    slc_terms: List[SLCTerm]
    rhs: np.ndarray  # target coefficients
    # rows[t][m_idx] = (vec_P, vec_r, coeff_w) for term t, monomial m_idx
    rows: List[List[Optional[Tuple[np.ndarray, np.ndarray, float]]]] = field(
        default_factory=list
    )


def build_matching_system(
    coeff_dict: Dict[Monomial, float],
    slc_terms: List[SLCTerm],
    n: int,
) -> MatchingSystem:
    monomials = sorted(coeff_dict.keys())
    m_idx = {m: i for i, m in enumerate(monomials)}
    rhs = np.array([coeff_dict[m] for m in monomials])
    rows: List[List[Optional[Tuple[np.ndarray, np.ndarray, float]]]] = []

    t0 = time.perf_counter()
    for t_idx, (I, J) in enumerate(slc_terms):
        if t_idx % 50000 == 0 and t_idx > 0:
            _log(f"matching: {t_idx}/{len(slc_terms)} terms ({time.perf_counter()-t0:.1f}s)")
            if _time_up():
                break
        row: List[Optional[Tuple[np.ndarray, np.ndarray, float]]] = []
        for beta in monomials:
            vp, vr, cw = term_contrib_to_monomial(I, J, beta, n)
            if np.any(vp) or np.any(vr) or abs(cw) > 0:
                row.append((vp, vr, cw))
            else:
                row.append(None)
        rows.append(row)

    _log(
        f"matching done: {len(rows)} terms x {len(monomials)} monomials "
        f"({time.perf_counter()-t0:.1f}s)"
    )
    return MatchingSystem(monomials=monomials, slc_terms=slc_terms, rhs=rhs, rows=rows)


# ---------------------------------------------------------------------------
# McCormick helpers on [l, u]
# ---------------------------------------------------------------------------


def mccormick_product(
    constraints: list,
    a,
    b,
    w,
    l_a: float,
    u_a: float,
    l_b: float,
    u_b: float,
):
    """Enforce w = a*b on box [l_a,u_a] x [l_b,u_b]."""
    constraints += [
        w >= l_a * b + l_b * a - l_a * l_b,
        w >= u_a * b + u_b * a - u_a * u_b,
        w <= u_a * b + l_b * a - u_a * l_b,
        w <= l_a * b + u_b * a - l_a * u_b,
    ]


# ---------------------------------------------------------------------------
# RPT Best-SLC dual SDP at one BnB node
# ---------------------------------------------------------------------------


def solve_node_sdp(
    matching: MatchingSystem,
    n: int,
    x_lo: np.ndarray,
    x_hi: np.ndarray,
    node_id: int,
    depth: int,
    branch_var: int,
    branch_mid: float,
) -> Tuple[float, str, float, Optional[np.ndarray]]:
    """
    Build and solve naive Theorem-6 dual SDP relaxation at a node.

    Unified perspective template per SLC term (I,J):
      s_{I,J} ~= f_{I,J}(x),  W_k = s * x_k,  Schur [[Y, W], [W^T, s]] >= 0
    """
    t0 = time.perf_counter()
    n_terms = len(matching.slc_terms)
    n_mono = len(matching.monomials)

    x = cp.Variable(n)
    lam = cp.Variable(n_mono)

    constraints: list = [x >= x_lo, x <= x_hi]

    # xx^T McCormick (optional linking; unit-box RLT)
    U = cp.Variable((n, n), symmetric=True)
    for i in range(n):
        mccormick_product(constraints, x[i], x[i], U[i, i], x_lo[i], x_hi[i], x_lo[i], x_hi[i])
        for j in range(i + 1, n):
            mccormick_product(constraints, x[i], x[j], U[i, j], x_lo[i], x_hi[i], x_lo[j], x_hi[j])
            constraints.append(U[i, j] == U[j, i])

    # Per-SLC-term perspective variables
    s_vars = []
    W_vars = []
    Y_vars = []
    for t_idx, (I, J) in enumerate(matching.slc_terms):
        s = cp.Variable(nonneg=True, name=f"s_{t_idx}")
        W = cp.Variable(n, name=f"W_{t_idx}")
        Y = cp.Variable((n, n), symmetric=True, name=f"Y_{t_idx}")
        s_vars.append(s)
        W_vars.append(W)
        Y_vars.append(Y)

        # W_k = s * x_k
        for k in range(n):
            mccormick_product(constraints, s, x[k], W[k], 0.0, 1.0, x_lo[k], x_hi[k])

        # Schur complement for perspective
        constraints.append(
            cp.bmat([[Y, W.reshape(n, 1)], [W.reshape(1, n), s.reshape(1, 1)]]) >> 0
        )

        # Link s = f_{I,J}(x) via sequential McCormick
        cur_expr: object = 1.0
        cur_l, cur_u = 1.0, 1.0
        if not I and not J:
            constraints.append(s == 1)
        else:
            for i in sorted(I):
                nxt = cp.Variable(nonneg=True)
                base = cur_expr if not isinstance(cur_expr, float) else cp.Constant(1.0)
                bl, bu = (cur_l, cur_u) if not isinstance(cur_expr, float) else (1.0, 1.0)
                mccormick_product(
                    constraints, base, x[i], nxt, bl, bu, x_lo[i], x_hi[i]
                )
                cur_l, cur_u = bl * x_lo[i], bu * x_hi[i]
                cur_expr = nxt
            for j in sorted(J):
                one_minus = 1 - x[j]
                oml, omu = 1 - x_hi[j], 1 - x_lo[j]
                nxt = cp.Variable(nonneg=True)
                base = cur_expr if not isinstance(cur_expr, float) else cp.Constant(1.0)
                bl, bu = (cur_l, cur_u) if not isinstance(cur_expr, float) else (1.0, 1.0)
                mccormick_product(
                    constraints, base, one_minus, nxt, bl, bu, oml, omu
                )
                cur_l, cur_u = bl * oml, bu * omu
                cur_expr = nxt
            constraints.append(s == cur_expr)

        # Dual PSD: Y + sum_j lam_j A_{t,j} <= 0  (matching contributions)
        dual_lmi = Y
        for m_idx in range(n_mono):
            row = matching.rows[t_idx][m_idx] if t_idx < len(matching.rows) else None
            if row is None:
                continue
            vp, vr, cw = row
            if np.any(vp):
                P_mat = vp.reshape(n, n)
                dual_lmi = dual_lmi + lam[m_idx] * (P_mat + P_mat.T) / 2
            if np.any(vr):
                # r part couples to W in perspective; approximate via W
                for k in range(n):
                    if vr[k] != 0:
                        dual_lmi = dual_lmi  # linear in r handled in eq constraints below
            if abs(cw) > 0:
                dual_lmi = dual_lmi  # w part linear in s
        constraints.append(dual_lmi << 0)

        # Linear dual equalities from perspective (Theorem 2 style)
        for m_idx in range(n_mono):
            row = matching.rows[t_idx][m_idx] if t_idx < len(matching.rows) else None
            if row is None:
                continue
            vp, vr, cw = row
            if np.any(vr):
                constraints.append(W @ vr == -lam[m_idx] * vr)  # simplified linkage
            if abs(cw) > 0:
                constraints.append(cw * s == -lam[m_idx] * cw)

    obj = -lam @ matching.rhs
    prob = cp.Problem(cp.Minimize(obj), constraints)

    remaining = max(1.0, _MAX_TIME - (time.perf_counter() - _START))
    prob.solve(
        solver=cp.MOSEK,
        verbose=False,
        mosek_params={"MSK_DPAR_OPTIMIZER_MAX_TIME": remaining},
    )

    elapsed = time.perf_counter() - t0
    lb = float(prob.value) if prob.value is not None else float("nan")
    status = prob.status or "unknown"
    x_val = x.value
    if x_val is not None:
        x_val = np.asarray(x_val, dtype=np.float64).ravel()

    _log(
        f"[node {node_id} d={depth}] branch x{branch_var} @ {branch_mid:.4f} | "
        f"SDP status={status} node_LB={lb:.4g} ({elapsed:.2f}s)"
    )
    return lb, status, elapsed, x_val


# ---------------------------------------------------------------------------
# Branch-and-bound
# ---------------------------------------------------------------------------


@dataclass
class BnBState:
    x_lo: np.ndarray
    x_hi: np.ndarray
    depth: int
    parent: int
    branch_var: int
    branch_mid: float


def branch_and_bound(
    matching: MatchingSystem,
    coeff_dict: Dict[Monomial, float],
    n: int,
    max_time: float,
) -> dict:
    global _START, _MAX_TIME
    _START = time.perf_counter()
    _MAX_TIME = max_time

    queue: List[BnBState] = [
        BnBState(
            x_lo=np.zeros(n),
            x_hi=np.ones(n),
            depth=0,
            parent=-1,
            branch_var=-1,
            branch_mid=0.5,
        )
    ]

    global_lb = -float("inf")
    global_ub = float("inf")
    best_x: Optional[np.ndarray] = None
    nodes = 0
    stop_reason = "time_limit"

    while queue and not _time_up():
        state = queue.pop(0)
        nodes += 1

        # Pick widest variable for branching
        widths = state.x_hi - state.x_lo
        bv = int(np.argmax(widths))
        mid = 0.5 * (state.x_lo[bv] + state.x_hi[bv])

        lb, status, _, x_val = solve_node_sdp(
            matching,
            n,
            state.x_lo,
            state.x_hi,
            nodes,
            state.depth,
            bv,
            mid,
        )
        global_lb = max(global_lb, lb)

        if x_val is not None:
            x_val = np.clip(x_val, state.x_lo, state.x_hi)
            ub = eval_poly(coeff_dict, x_val)
            if ub < global_ub:
                global_ub = ub
                best_x = x_val.copy()
                _log(f"incumbent: node {nodes} loss={ub:.6g}")

        gap_str = f"{global_ub - global_lb:.4g}" if global_ub < float("inf") else "-"
        _log(
            f"[node {nodes} d={state.depth}] global LB={global_lb:.4g} "
            f"UB={global_ub:.4g} gap={gap_str} queue={len(queue)}"
        )

        if global_ub - global_lb < 1e-6 and global_ub < float("inf"):
            stop_reason = "optimal"
            break

        if mid - state.x_lo[bv] > 1e-8:
            lo_state = BnBState(
                x_lo=state.x_lo.copy(),
                x_hi=state.x_hi.copy(),
                depth=state.depth + 1,
                parent=nodes,
                branch_var=bv,
                branch_mid=mid,
            )
            lo_state.x_hi[bv] = mid
            queue.append(lo_state)

        if state.x_hi[bv] - mid > 1e-8:
            hi_state = BnBState(
                x_lo=state.x_lo.copy(),
                x_hi=state.x_hi.copy(),
                depth=state.depth + 1,
                parent=nodes,
                branch_var=bv,
                branch_mid=mid,
            )
            hi_state.x_lo[bv] = mid
            queue.append(hi_state)

    wall = time.perf_counter() - _START
    _log(
        f"stop={stop_reason} nodes={nodes} wall_s={wall:.1f} "
        f"final_LB={global_lb:.6g} final_UB={global_ub:.6g}"
    )
    return {
        "stop_reason": stop_reason,
        "nodes": nodes,
        "wall_seconds": wall,
        "lower_bound": global_lb,
        "upper_bound": global_ub,
        "best_x_unit": best_x,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def solve_matmul_222(max_time: float = 3600.0) -> dict:
    global _START, _MAX_TIME
    _START = time.perf_counter()
    _MAX_TIME = max_time

    _log("=== RPT-SLC exact matmul <2,2,2> rank 7 ===")
    _log(f"box map: [{ORIG_LO},{ORIG_HI}] -> [0,1] per variable")

    coeff_dict, names, n, degree = build_matmul_loss_coeff_dict(2, 2, 2, 7)
    max_k = degree - 2
    est = estimate_slc_term_count(n, max_k)
    _log(f"SLC basis: max_k={max_k} estimated_terms={est}")

    t_enum = time.perf_counter()
    slc_terms: List[SLCTerm] = []
    for idx, term in enumerate(enumerate_slc_terms(n, max_k)):
        slc_terms.append(term)
        if idx > 0 and idx % 500000 == 0:
            _log(f"SLC enum: {idx} terms ({time.perf_counter()-t_enum:.1f}s)")
            if _time_up():
                break
    _log(f"SLC enum done: {len(slc_terms)} terms ({time.perf_counter()-t_enum:.1f}s)")

    if _time_up() or len(slc_terms) == 0:
        return {"stop_reason": "build_failed", "wall_seconds": time.perf_counter() - _START}

    matching = build_matching_system(coeff_dict, slc_terms, n)
    if _time_up() or len(matching.rows) < len(slc_terms):
        return {"stop_reason": "build_failed", "wall_seconds": time.perf_counter() - _START}

    _log(
        f"SDP size estimate: n={n} slc_terms={len(slc_terms)} "
        f"matching_rows={len(matching.monomials)}"
    )

    result = branch_and_bound(matching, coeff_dict, n, max_time)

    if result.get("best_x_unit") is not None:
        x_orig = from_unit(result["best_x_unit"])
        result["best_x_orig"] = x_orig
        U, V, W = unflatten_factors(x_orig, 2, 2, 2, 7)
        result["linf_error"] = verify_decomposition(
            build_mult_tensor(2, 2, 2), U, V, W
        )
        result["loss"] = eval_poly(coeff_dict, result["best_x_unit"])
        _log(f"verify: linf={result['linf_error']:.4g} loss={result['loss']:.6g}")

    return result
