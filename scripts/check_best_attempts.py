"""Verify best_attempt.npz: continuous vs rounded reconstruction."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensor_utils import build_mult_tensor, frobenius_loss, verify_decomposition, wrong_entries
from validation import verify_exact_integer, verify_all
from tensor_utils import make_continuous_result


def main():
    root = Path(__file__).resolve().parents[1]
    log = json.loads((root / "batch_results/experiment_log.json").read_text())
    new = [e for e in log if e.get("timestamp", "").startswith("2026-06-18")]

    header = (
        f"{'case':<12} {'R':>3} {'F2_logged':>12} {'F2_recomp':>12} "
        f"{'max_err_cont':>12} {'max_err_rnd':>12} {'wrong':>9} {'int_alg':>8}"
    )
    print(header)
    print("-" * len(header))

    for e in new:
        m, p, n = e["case"]
        R = e["target_rank"]
        path = root / f"batch_results/{m}_{p}_{n}_rank{R}/best_attempt.npz"
        if not path.exists():
            print(f"<{m},{p},{n}> R={R}: missing {path}")
            continue

        d = np.load(path)
        U, V, W = d["U"], d["V"], d["W"]
        T = build_mult_tensor(m, p, n)
        f2 = frobenius_loss(T, U, V, W)
        logged = float(e.get("best_loss", float("nan")))
        max_cont = verify_decomposition(T, U, V, W)
        U_r = np.round(U).astype(np.int64)
        V_r = np.round(V).astype(np.int64)
        W_r = np.round(W).astype(np.int64)
        max_rnd = verify_decomposition(
            T, U_r.astype(float), V_r.astype(float), W_r.astype(float)
        )
        n_wrong, n_total = wrong_entries(T, U_r, V_r, W_r)
        int_exact = max_rnd < 1e-10 and n_wrong == 0

        case = f"<{m},{p},{n}>"
        print(
            f"{case:<12} {R:3d} {logged:12.3e} {f2:12.3e} "
            f"{max_cont:12.3e} {max_rnd:12.3e} {n_wrong:4d}/{n_total:<4d} "
            f"{int_exact!s:>8}"
        )

    print("\n=== Near-zero continuous cases (F2 < 1e-6) ===\n")
    for e in new:
        m, p, n = e["case"]
        R = e["target_rank"]
        path = root / f"batch_results/{m}_{p}_{n}_rank{R}/best_attempt.npz"
        d = np.load(path)
        U, V, W = d["U"], d["V"], d["W"]
        T = build_mult_tensor(m, p, n)
        f2 = frobenius_loss(T, U, V, W)
        if f2 > 1e-6:
            continue

        U_r = np.round(U).astype(np.int64)
        V_r = np.round(V).astype(np.int64)
        W_r = np.round(W).astype(np.int64)
        max_cont = verify_decomposition(T, U, V, W)
        max_rnd = verify_decomposition(
            T, U_r.astype(float), V_r.astype(float), W_r.astype(float)
        )
        n_wrong, n_total = wrong_entries(T, U_r, V_r, W_r)
        max_dev = max(
            np.max(np.abs(U - U_r)),
            np.max(np.abs(V - V_r)),
            np.max(np.abs(W - W_r)),
        )
        frac_near = np.mean(
            np.concatenate(
                [
                    (np.abs(U - np.round(U)) < 0.15).ravel(),
                    (np.abs(V - np.round(V)) < 0.15).ravel(),
                    (np.abs(W - np.round(W)) < 0.15).ravel(),
                ]
            )
        )

        result = make_continuous_result(U, V, W, m, p, n, "gradient")
        rnd_result = make_continuous_result(U_r, V_r, W_r, m, p, n, "rounded")
        rnd_result.field = "Z"
        int_ok = verify_exact_integer(rnd_result)

        print(f"<{m},{p},{n}> rank {R}")
        print(f"  Frobenius^2 (continuous): {f2:.6e}")
        print(f"  max |T - T_hat| (continuous): {max_cont:.6e}")
        print(f"  is_exact flag (F2 < 1e-10): {result.is_exact}")
        print(f"  max |T - T_hat| (rounded):   {max_rnd:.6e}")
        print(f"  wrong entries after round:   {n_wrong}/{n_total}")
        print(f"  max |x - round(x)|:          {max_dev:.6f}")
        print(f"  frac entries within 0.15 of int: {frac_near:.1%}")
        print(f"  verify_exact_integer (rounded): {int_ok}")
        print(f"  valid exact INTEGER algorithm? {int_ok}")
        print()


if __name__ == "__main__":
    main()
