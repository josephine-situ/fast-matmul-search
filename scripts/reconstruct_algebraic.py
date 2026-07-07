#!/usr/bin/env python3
"""
LLL / PSLQ / identify reconstruction on continuous best_attempt factors.

Strict PSLQ searches for exact integer relations among {1, sqrt(d), x}.
mpmath.identify is a separate heuristic (tolerance-based) ℚ(√2) fit.

Usage:
    PYTHONPATH=src python scripts/reconstruct_algebraic.py batch_results/3_3_3_rank23/best_attempt.npz
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import mpmath
import numpy as np
from mpmath import mp
from sympy import sympify, N, sqrt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensor_utils import build_mult_tensor, frobenius_loss, verify_decomposition


@dataclass
class FieldSpec:
    name: str
    basis: Sequence[mpmath.mpf]
    constants: list[str]


FIELDS = [
    FieldSpec("Q", [mpmath.mpf(1)], []),
    FieldSpec("Q(sqrt2)", [mpmath.mpf(1), mpmath.sqrt(2)], ["sqrt(2)"]),
    FieldSpec("Q(sqrt3)", [mpmath.mpf(1), mpmath.sqrt(3)], ["sqrt(3)"]),
    FieldSpec(
        "Q(sqrt2,sqrt3)",
        [mpmath.mpf(1), mpmath.sqrt(2), mpmath.sqrt(3)],
        ["sqrt(2)", "sqrt(3)"],
    ),
]


def pslq_relation(
    x: float,
    basis: Sequence[mpmath.mpf],
    maxcoeff: int = 1000,
    dps: int = 80,
) -> Optional[Tuple[int, ...]]:
    """Return c with sum_i c[i]*basis[i] + c[-1]*x = 0, or None."""
    old_dps = mp.dps
    try:
        mp.dps = dps
        xv = mpmath.mpf(float(x))
        rel = mpmath.pslq(list(basis) + [xv], maxcoeff=maxcoeff, maxsteps=5000)
        if rel is None:
            return None
        coeffs = tuple(int(v) for v in rel)
        if coeffs[-1] == 0:
            return None
        return coeffs
    finally:
        mp.dps = old_dps


def relation_value(coeffs: Sequence[int], basis: Sequence[mpmath.mpf]) -> float:
    body = sum(c * float(b) for c, b in zip(coeffs[:-1], basis))
    return -body / coeffs[-1]


def snap_array_rational_bounded(
    arr: np.ndarray, max_den: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Best rational with |den| <= max_den (continued-fraction / LLL-equivalent)."""
    out = np.zeros_like(arr, dtype=np.float64)
    errs = np.zeros_like(arr, dtype=np.float64)
    for idx, x in np.ndenumerate(arr):
        val = float(Fraction(float(x)).limit_denominator(max_den))
        out[idx] = val
        errs[idx] = abs(val - float(x))
    return out, errs


def identify_value(
    x: float,
    constants: list[str],
    tol: float,
    dps: int = 50,
) -> Tuple[float, bool, Optional[str]]:
    old_dps = mp.dps
    try:
        mp.dps = dps
        s = mpmath.identify(float(x), constants, tol=tol)
        if s is None:
            return float(x), False, None
        val = float(N(sympify(s), dps))
        return val, True, s
    finally:
        mp.dps = old_dps


def snap_array_pslq(
    arr: np.ndarray,
    field: FieldSpec,
    maxcoeff: int,
    dps: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    out = np.zeros_like(arr, dtype=np.float64)
    errs = np.full(arr.shape, np.nan, dtype=np.float64)
    ok = 0
    for idx, x in np.ndenumerate(arr):
        rel = pslq_relation(float(x), field.basis, maxcoeff=maxcoeff, dps=dps)
        if rel is None:
            out[idx] = float(x)
            continue
        val = relation_value(rel, field.basis)
        out[idx] = val
        errs[idx] = abs(val - float(x))
        ok += 1
    return out, errs, ok


def snap_array_lll_rational(arr: np.ndarray, max_den: int) -> Tuple[np.ndarray, np.ndarray, int]:
    out, errs = snap_array_rational_bounded(arr, max_den)
    return out, errs, arr.size


def snap_array_identify(
    arr: np.ndarray,
    constants: list[str],
    tol: float,
) -> Tuple[np.ndarray, int]:
    out = np.zeros_like(arr, dtype=np.float64)
    ok = 0
    for idx, x in np.ndenumerate(arr):
        val, found, _ = identify_value(float(x), constants, tol=tol)
        out[idx] = val
        ok += int(found)
    return out, ok


def algebraic_degree(
    x: float,
    maxdeg: int = 4,
    maxcoeff: int = 2000,
    dps: int = 80,
) -> Optional[Tuple[int, Tuple[int, ...]]]:
    old_dps = mp.dps
    try:
        mp.dps = dps
        xv = mpmath.mpf(float(x))
        for deg in range(1, maxdeg + 1):
            rel = mpmath.pslq([xv ** k for k in range(deg + 1)], maxcoeff=maxcoeff)
            if rel is not None:
                return deg, tuple(int(v) for v in rel)
        return None
    finally:
        mp.dps = old_dps


def report(label: str, U: np.ndarray, V: np.ndarray, W: np.ndarray, T: np.ndarray) -> None:
    mx = verify_decomposition(T, U, V, W)
    f2 = frobenius_loss(T, U, V, W)
    print(f"  {label}")
    print(f"    max tensor error: {mx:.6e}")
    print(f"    Frobenius^2:      {f2:.6e}")


def err_stats(errs: np.ndarray) -> str:
    errs = errs[~np.isnan(errs)]
    if errs.size == 0:
        return "no snaps"
    return f"median={np.median(errs):.3e} max={np.max(errs):.3e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "npz",
        nargs="?",
        type=Path,
        default=Path("batch_results/3_3_3_rank23/best_attempt.npz"),
    )
    parser.add_argument("--maxcoeff", type=int, default=2000)
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--identify-tol", type=float, default=1e-6)
    args = parser.parse_args()

    data = np.load(args.npz)
    U0, V0, W0 = data["U"], data["V"], data["W"]
    m, p, n = int(data["m"]), int(data["p"]), int(data["n"])
    T = build_mult_tensor(m, p, n)
    n_coeff = U0.size + V0.size + W0.size

    print(f"Loaded {args.npz}")
    print(f"  U,V,W shape = {U0.shape}, n_coeff = {n_coeff}")
    print()
    report("Continuous reference", U0, V0, W0, T)
    print()

    # Strict PSLQ per field
    for field in FIELDS:
        print(f"=== Strict PSLQ — {field.name} (maxcoeff={args.maxcoeff}) ===")
        Us, eu, ou = snap_array_pslq(U0, field, args.maxcoeff, args.dps)
        Vs, ev, ov = snap_array_pslq(V0, field, args.maxcoeff, args.dps)
        Ws, ew, ow = snap_array_pslq(W0, field, args.maxcoeff, args.dps)
        errs = np.concatenate([eu.ravel(), ev.ravel(), ew.ravel()])
        print(f"  relations found: {ou + ov + ow}/{n_coeff}")
        print(f"  per-entry snap:  {err_stats(errs)}")
        report("Joint strict PSLQ", Us, Vs, Ws, T)
        print()

    # Bounded-denominator rationals (continued-fraction / standard LLL recon)
    print(f"=== Bounded-denominator rationals (den <= {args.maxcoeff}) ===")
    Us, eu, ou = snap_array_lll_rational(U0, max_den=args.maxcoeff)
    Vs, ev, ov = snap_array_lll_rational(V0, max_den=args.maxcoeff)
    Ws, ew, ow = snap_array_lll_rational(W0, max_den=args.maxcoeff)
    errs = np.concatenate([eu.ravel(), ev.ravel(), ew.ravel()])
    print(f"  all entries snapped")
    print(f"  per-entry snap:  {err_stats(errs)}")
    report("Joint bounded-rational snap", Us, Vs, Ws, T)
    print()

    # Heuristic identify (tolerance-based, not strict PSLQ)
    for field in [FIELDS[1], FIELDS[3]]:
        print(
            f"=== mpmath.identify — {field.name} "
            f"(tol={args.identify_tol:.0e}, heuristic) ==="
        )
        Us, ou = snap_array_identify(U0, field.constants, args.identify_tol)
        Vs, ov = snap_array_identify(V0, field.constants, args.identify_tol)
        Ws, ow = snap_array_identify(W0, field.constants, args.identify_tol)
        print(f"  formulas found: {ou + ov + ow}/{n_coeff}")
        report("Joint identify snap", Us, Vs, Ws, T)
        print()

    # Algebraic degree spot-check
    print("=== Algebraic degree (strict PSLQ on [1,x,...,x^d]) — first 20 coeffs ===")
    allc = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
    found = 0
    for i, x in enumerate(allc[:20]):
        res = algebraic_degree(float(x), maxdeg=4, maxcoeff=args.maxcoeff, dps=args.dps)
        if res is None:
            print(f"  [{i:3d}] {float(x):+.8f}  no relation (deg <= 4)")
        else:
            found += 1
            deg, rel = res
            print(f"  [{i:3d}] {float(x):+.8f}  degree {deg}  relation {rel}")
    print(f"  ({found}/20 had low-degree relations)")
    print()

    # Sample identify strings
    print("=== Sample identify formulas (Q(sqrt2), tol=1e-6) ===")
    for x in allc[:8]:
        val, ok, s = identify_value(float(x), ["sqrt(2)"], tol=1e-6)
        if ok:
            print(f"  {float(x):+.8f}  =  {s}  (err {abs(val - float(x)):.2e})")
        else:
            print(f"  {float(x):+.8f}  ->  no formula")


if __name__ == "__main__":
    main()
