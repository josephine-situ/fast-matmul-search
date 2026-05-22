"""
Hybrid search: gradient descent with flip-graph escape moves.

When standard gradient search stalls, this module:
  1. Takes the iterate with lowest **float reconstruction error** over all steps
     (not the final or best-rounded iterate) as the escape anchor
  2. Rounds that point to integers for flip seeds; uses the float iterate for snap
  3. Applies exact-preserving flip moves (on exact schemes) or
     flip-inspired entry perturbations (on near-exact schemes)
  4. Warm-starts gradient descent from each escape point

Does not modify ContinuousSearch or other existing modules.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from continuous_search import ContinuousSearch
from hardcode_known import strassen_222
from tensor_utils import (
    DecompositionResult,
    build_mult_tensor,
    make_result,
    verify_decomposition,
)


EXACT_TOL = 1e-10


@dataclass
class StallState:
    """
    State captured when a gradient run stalls without an exact solution.

    Flip escapes anchor on the iterate with minimum reconstruction loss across
    all steps (U_float/V_float/W_float). Integer seeds are round() of that point.
    """

    recon_error: float
    rounded_error: float
    U_int: np.ndarray
    V_int: np.ndarray
    W_int: np.ndarray
    U_float: np.ndarray
    V_float: np.ndarray
    W_float: np.ndarray
    step_best_recon: int = -1
    step_best_rounded: int = -1


# Backward-compatible alias
RoundedBestState = StallState


@dataclass
class FlipHybridConfig:
    """Knobs for flip-graph escape attempts after gradient stalls."""

    near_exact_threshold: float = 1.0
    n_flip_walks: int = 8
    n_moves_per_walk: int = 3
    n_entry_flips: int = 4
    escape_n_steps: int = 12000
    escape_lr: float = 0.003
    snap_n_steps: int = 5000
    try_snap_before_flips: bool = True
    only_flip_when_near_exact: bool = True


# ---------------------------------------------------------------------------
# Decomposition helpers
# ---------------------------------------------------------------------------

def reconstruction_error(
    m: int,
    p: int,
    n: int,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
) -> float:
    T = build_mult_tensor(m, p, n)
    return verify_decomposition(
        T,
        np.asarray(U, dtype=np.float64),
        np.asarray(V, dtype=np.float64),
        np.asarray(W, dtype=np.float64),
    )


def is_exact(
    m: int,
    p: int,
    n: int,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    tol: float = EXACT_TOL,
) -> bool:
    return reconstruction_error(m, p, n, U, V, W) <= tol


def round_factors(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.round(U).astype(np.int64),
        np.round(V).astype(np.int64),
        np.round(W).astype(np.int64),
    )


# ---------------------------------------------------------------------------
# Exact-preserving flip moves (valid on exact integer decompositions)
# ---------------------------------------------------------------------------

def swap_rank_columns(
    U: np.ndarray, V: np.ndarray, W: np.ndarray, r: int, s: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Swap two rank components in U, V, W simultaneously."""
    U2, V2, W2 = U.copy(), V.copy(), W.copy()
    cols = [r, s]
    U2[:, cols] = U[:, cols[::-1]]
    V2[:, cols] = V[:, cols[::-1]]
    W2[:, cols] = W[:, cols[::-1]]
    return U2, V2, W2


def scale_rank_with_signs(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    r: int,
    su: int,
    sv: int,
    sw: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale rank-r columns by (su, sv, sw) with su*sv*sw == 1 so the rank-1 term is unchanged.
    """
    if su * sv * sw != 1:
        raise ValueError("su * sv * sw must be 1")
    U2, V2, W2 = U.copy(), V.copy(), W.copy()
    U2[:, r] = su * U[:, r]
    V2[:, r] = sv * V[:, r]
    W2[:, r] = sw * W[:, r]
    return U2, V2, W2


def random_rank_sign_flip(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    r: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick random signs (±1) on rank r with product +1."""
    su = int(rng.choice([-1, 1]))
    sv = int(rng.choice([-1, 1]))
    sw = int(su * sv)  # ensures su * sv * sw == 1
    return scale_rank_with_signs(U, V, W, r, su, sv, sw)


def random_rank_permutation(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = U.shape[1]
    perm = rng.permutation(R)
    return U[:, perm].copy(), V[:, perm].copy(), W[:, perm].copy()


def apply_random_exact_move(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One random move that preserves exactness (when input is exact)."""
    R = U.shape[1]
    move = rng.choice(["swap", "sign", "permute"])
    if move == "swap" and R >= 2:
        r, s = rng.choice(R, size=2, replace=False)
        return swap_rank_columns(U, V, W, int(r), int(s))
    if move == "sign":
        r = int(rng.integers(0, R))
        return random_rank_sign_flip(U, V, W, r, rng)
    return random_rank_permutation(U, V, W, rng)


def random_exact_walk(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    n_moves: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cur = (U.copy(), V.copy(), W.copy())
    for _ in range(n_moves):
        cur = apply_random_exact_move(cur[0], cur[1], cur[2], rng)
    return cur


# ---------------------------------------------------------------------------
# Flip-inspired moves (may break exactness; repaired by gradient)
# ---------------------------------------------------------------------------

def flip_entry_sign(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    matrix: str,
    row: int,
    col: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Negate one coefficient; intended as a discrete hop before GD repair."""
    U2, V2, W2 = U.copy(), V.copy(), W.copy()
    target = {"U": U2, "V": V2, "W": W2}[matrix]
    target[row, col] = -target[row, col]
    return U2, V2, W2


def random_entry_sign_flip(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mats = []
    for name, M in [("U", U), ("V", V), ("W", W)]:
        nz = np.argwhere(M != 0)
        if len(nz) > 0:
            mats.append((name, M.shape, nz))
    if not mats:
        idx = int(rng.integers(0, U.shape[0]))
        return flip_entry_sign(U, V, W, "U", idx, int(rng.integers(0, U.shape[1])))
    name, shape, nz = mats[int(rng.integers(0, len(mats)))]
    i, j = nz[int(rng.integers(0, len(nz)))]
    return flip_entry_sign(U, V, W, name, int(i), int(j))


def generate_flip_escape_seeds(
    m: int,
    p: int,
    n: int,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    config: FlipHybridConfig,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build integer seeds for warm-started gradient runs.

    Uses exact-preserving walks when the rounded point is exact; otherwise
    entry sign flips when within near_exact_threshold.
    """
    U_i, V_i, W_i = round_factors(U, V, W)
    err = reconstruction_error(m, p, n, U_i, V_i, W_i)
    seeds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = [(U_i, V_i, W_i)]

    if is_exact(m, p, n, U_i, V_i, W_i):
        for _ in range(config.n_flip_walks):
            uw = random_exact_walk(
                U_i, V_i, W_i, config.n_moves_per_walk, rng
            )
            seeds.append(uw)
        return seeds

    if config.only_flip_when_near_exact and err > config.near_exact_threshold:
        return seeds

    for _ in range(config.n_entry_flips):
        seeds.append(random_entry_sign_flip(U_i, V_i, W_i, rng))
    for _ in range(max(0, config.n_flip_walks - config.n_entry_flips)):
        seeds.append(
            apply_random_exact_move(U_i, V_i, W_i, rng)
        )
    return seeds


# ---------------------------------------------------------------------------
# Gradient with state (subclass in this file only — does not edit continuous_search)
# ---------------------------------------------------------------------------

class ContinuousSearchWithState(ContinuousSearch):
    """ContinuousSearch that tracks best-recon and best-rounded iterates."""

    def search_single_with_best(
        self,
        R: int,
        n_steps: int = 20000,
        lr: float = 0.003,
        init_method: str = "gaussian",
        verbose: bool = False,
    ) -> Tuple[Optional[DecompositionResult], Optional[StallState]]:
        U, V, W = self._init_factors(R, method=init_method)

        optimizer = torch.optim.Adam([U, V, W], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=n_steps // 4, T_mult=1, eta_min=lr * 0.01
        )

        phase1_end = int(n_steps * 0.4)
        phase2_end = int(n_steps * 0.7)

        best_recon_error = float("inf")
        best_recon_U = best_recon_V = best_recon_W = None
        step_best_recon = -1

        best_rounded_error = float("inf")
        best_rounded_U = best_rounded_V = best_rounded_W = None
        step_best_rounded = -1

        for step in range(n_steps):
            optimizer.zero_grad()
            recon = self._recon_loss(U, V, W)
            loss = recon

            with torch.no_grad():
                recon_val = recon.item()
                if recon_val < best_recon_error:
                    best_recon_error = recon_val
                    step_best_recon = step
                    best_recon_U = U.detach().cpu().numpy().copy()
                    best_recon_V = V.detach().cpu().numpy().copy()
                    best_recon_W = W.detach().cpu().numpy().copy()

            if step < phase1_end:
                loss = loss + 0.01 * self._balance_loss(U, V, W)
            elif step < phase2_end:
                t = (step - phase1_end) / (phase2_end - phase1_end)
                int_w = 0.3 * t ** 2
                sparse_w = 0.05 * t
                mag_w = 0.1 * t
                loss = (
                    loss
                    + int_w * self._integrality_loss(U, V, W)
                    + sparse_w * self._sparsity_loss(U, V, W)
                    + mag_w * self._magnitude_loss(U, V, W)
                    + 0.01 * self._balance_loss(U, V, W)
                )
            else:
                t = (step - phase2_end) / (n_steps - phase2_end)
                int_w = 0.3 + 2.0 * t
                sparse_w = 0.05 + 0.2 * t
                mag_w = 0.1 + 0.5 * t
                loss = (
                    loss
                    + int_w * self._integrality_loss(U, V, W)
                    + sparse_w * self._sparsity_loss(U, V, W)
                    + mag_w * self._magnitude_loss(U, V, W)
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
            optimizer.step()
            scheduler.step()

            if step % 1000 == 0 or step == n_steps - 1:
                with torch.no_grad():
                    U_r = torch.round(U)
                    V_r = torch.round(V)
                    W_r = torch.round(W)
                    rounded_error = self._recon_loss(U_r, V_r, W_r).item()

                    if rounded_error < best_rounded_error:
                        best_rounded_error = rounded_error
                        step_best_rounded = step
                        best_rounded_U = U_r.cpu().numpy()
                        best_rounded_V = V_r.cpu().numpy()
                        best_rounded_W = W_r.cpu().numpy()

                    if verbose and step % 5000 == 0:
                        print(
                            f"  step {step:>6d}: recon={recon_val:.6f} "
                            f"best_recon={best_recon_error:.6f}@step{step_best_recon} "
                            f"rounded_err={rounded_error:.6f} "
                            f"best_rounded={best_rounded_error:.6f}"
                        )

                    if rounded_error < EXACT_TOL:
                        if verbose:
                            print(f"  EXACT at step {step}!")
                        res = make_result(
                            best_rounded_U, best_rounded_V, best_rounded_W,
                            self.m, self.p, self.n,
                            "gradient", "Z",
                        )
                        return res, None

        if best_recon_U is None:
            return None, None

        U_snap = torch.tensor(
            best_recon_U, dtype=torch.float64, device=self.device
        )
        V_snap = torch.tensor(
            best_recon_V, dtype=torch.float64, device=self.device
        )
        W_snap = torch.tensor(
            best_recon_W, dtype=torch.float64, device=self.device
        )
        result = self._snap_and_refine(U_snap, V_snap, W_snap)
        if result is not None:
            return result, None

        U_i, V_i, W_i = round_factors(best_recon_U, best_recon_V, best_recon_W)
        rounded_at_recon = reconstruction_error(
            self.m, self.p, self.n, U_i, V_i, W_i
        )
        if rounded_at_recon < EXACT_TOL:
            res = make_result(
                U_i, V_i, W_i,
                self.m, self.p, self.n,
                "gradient", "Z",
            )
            return res, None

        if best_rounded_error < EXACT_TOL and best_rounded_U is not None:
            res = make_result(
                best_rounded_U, best_rounded_V, best_rounded_W,
                self.m, self.p, self.n,
                "gradient", "Z",
            )
            return res, None

        state = StallState(
            recon_error=best_recon_error,
            rounded_error=rounded_at_recon,
            U_int=U_i,
            V_int=V_i,
            W_int=W_i,
            U_float=best_recon_U,
            V_float=best_recon_V,
            W_float=best_recon_W,
            step_best_recon=step_best_recon,
            step_best_rounded=step_best_rounded,
        )
        return None, state


def refine_from_integer_seed(
    searcher: ContinuousSearch,
    U_seed: np.ndarray,
    V_seed: np.ndarray,
    W_seed: np.ndarray,
    n_steps: int,
    lr: float,
    method_label: str = "flip_hybrid",
    verbose: bool = False,
) -> Optional[DecompositionResult]:
    """Warm-start gradient descent from integer (or near-integer) factors."""
    U = torch.tensor(U_seed, dtype=torch.float64, device=searcher.device)
    V = torch.tensor(V_seed, dtype=torch.float64, device=searcher.device)
    W = torch.tensor(W_seed, dtype=torch.float64, device=searcher.device)
    U.requires_grad_(True)
    V.requires_grad_(True)
    W.requires_grad_(True)

    optimizer = torch.optim.Adam([U, V, W], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.01
    )

    phase1_end = int(n_steps * 0.35)
    phase2_end = int(n_steps * 0.65)

    for step in range(n_steps):
        optimizer.zero_grad()
        recon = searcher._recon_loss(U, V, W)
        loss = recon

        if step < phase1_end:
            loss = loss + 0.01 * searcher._balance_loss(U, V, W)
        elif step < phase2_end:
            t = (step - phase1_end) / max(1, phase2_end - phase1_end)
            loss = loss + (0.2 * t ** 2) * searcher._integrality_loss(U, V, W)
        else:
            t = (step - phase2_end) / max(1, n_steps - phase2_end)
            loss = loss + (0.5 + 1.5 * t) * searcher._integrality_loss(U, V, W)

        loss.backward()
        torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
        optimizer.step()
        scheduler.step()

        if step % 1000 == 0 or step == n_steps - 1:
            with torch.no_grad():
                err = searcher._recon_loss(
                    torch.round(U), torch.round(V), torch.round(W)
                ).item()
                if err < EXACT_TOL:
                    if verbose:
                        print(f"    escape GD exact at step {step}")
                    return make_result(
                        torch.round(U).cpu().numpy(),
                        torch.round(V).cpu().numpy(),
                        torch.round(W).cpu().numpy(),
                        searcher.m, searcher.p, searcher.n,
                        method_label, "Z",
                    )

    with torch.no_grad():
        err = searcher._recon_loss(
            torch.round(U), torch.round(V), torch.round(W)
        ).item()
        if err < EXACT_TOL:
            return make_result(
                torch.round(U).cpu().numpy(),
                torch.round(V).cpu().numpy(),
                torch.round(W).cpu().numpy(),
                searcher.m, searcher.p, searcher.n,
                method_label, "Z",
            )

    snap = searcher._snap_and_refine(U.detach(), V.detach(), W.detach())
    if snap is not None:
        snap.method = method_label
    return snap


def try_flip_escapes(
    searcher: ContinuousSearchWithState,
    R: int,
    state: StallState,
    config: FlipHybridConfig,
    rng: np.random.Generator,
    verbose: bool = False,
) -> List[DecompositionResult]:
    """Run snap + flip-graph seeds + GD from the best-reconstruction iterate."""
    m, p, n = searcher.m, searcher.p, searcher.n
    found: List[DecompositionResult] = []

    if config.try_snap_before_flips:
        U_t = torch.tensor(
            state.U_float, dtype=torch.float64, device=searcher.device
        )
        V_t = torch.tensor(
            state.V_float, dtype=torch.float64, device=searcher.device
        )
        W_t = torch.tensor(
            state.W_float, dtype=torch.float64, device=searcher.device
        )
        snap = searcher._snap_and_refine(
            U_t, V_t, W_t, n_steps=config.snap_n_steps
        )
        if snap is not None:
            snap.method = "flip_hybrid+snap"
            if verbose:
                print(f"    snap refine: FOUND — {snap.summary()}")
            return [snap]

    seeds = generate_flip_escape_seeds(
        m, p, n,
        state.U_float, state.V_float, state.W_float,
        config, rng,
    )

    if verbose:
        print(
            f"    flip escapes: {len(seeds)} seeds "
            f"(best recon={state.recon_error:.4e} @ step {state.step_best_recon}, "
            f"rounded@recon={state.rounded_error:.4e})"
        )

    seen = set()
    for i, (U_s, V_s, W_s) in enumerate(seeds):
        key = (U_s.tobytes(), V_s.tobytes(), W_s.tobytes())
        if key in seen:
            continue
        seen.add(key)

        res = refine_from_integer_seed(
            searcher,
            U_s, V_s, W_s,
            n_steps=config.escape_n_steps,
            lr=config.escape_lr,
            method_label="flip_hybrid",
            verbose=verbose and i == 0,
        )
        if res is not None and res.is_exact:
            found.append(res)
            if verbose:
                print(f"    seed {i}: FOUND — {res.summary()}")

    return found


# ---------------------------------------------------------------------------
# Main hybrid search API
# ---------------------------------------------------------------------------

class FlipGraphHybridSearch:
    """
    Gradient search with optional flip-graph escapes on stalled restarts.

    Uses ContinuousSearch for the primary pass; does not replace it.
    """

    def __init__(
        self,
        m: int,
        p: int,
        n: int,
        device: str = "cpu",
        flip_config: Optional[FlipHybridConfig] = None,
    ):
        self.m, self.p, self.n = m, p, n
        self.device = device
        self.flip_config = flip_config or FlipHybridConfig()
        self._searcher = ContinuousSearchWithState(m, p, n, device=device)

    def search_single_with_escapes(
        self,
        R: int,
        n_steps: int = 20000,
        lr: float = 0.003,
        init_method: str = "gaussian",
        verbose: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> List[DecompositionResult]:
        """One restart: gradient, then flip escapes if needed."""
        rng = rng or np.random.default_rng()
        results: List[DecompositionResult] = []

        direct, state = self._searcher.search_single_with_best(
            R, n_steps=n_steps, lr=lr, init_method=init_method, verbose=verbose
        )
        if direct is not None:
            return [direct]

        if state is None:
            return []

        if verbose:
            print(
                f"  gradient stalled (best recon={state.recon_error:.4e} "
                f"@ step {state.step_best_recon}, "
                f"rounded@recon={state.rounded_error:.4e}), "
                f"trying flip escapes..."
            )

        escapes = try_flip_escapes(
            self._searcher, R, state, self.flip_config, rng, verbose=verbose
        )
        results.extend(escapes)
        return results

    def search(
        self,
        R: int,
        n_restarts: int = 200,
        n_steps: int = 20000,
        lr: float = 0.003,
        verbose: bool = True,
        seed: Optional[int] = None,
    ) -> List[DecompositionResult]:
        """
        Multi-restart search: standard gradient per restart, then flip escapes
        when gradient does not yield an exact decomposition.
        """
        rng = np.random.default_rng(seed)
        all_results: List[DecompositionResult] = []
        init_methods = ["gaussian", "sparse", "uniform"]

        if verbose:
            print(
                f"\nFlip-graph hybrid search for <{self.m},{self.p},{self.n}> rank {R}"
            )
            print(f"  Restarts: {n_restarts}, steps/restart: {n_steps}")
            print(
                f"  Escape config: near_exact<={self.flip_config.near_exact_threshold}, "
                f"walks={self.flip_config.n_flip_walks}"
            )

        t0 = time.time()
        for restart in range(n_restarts):
            init = init_methods[restart % len(init_methods)]
            found = self.search_single_with_escapes(
                R,
                n_steps=n_steps,
                lr=lr,
                init_method=init,
                verbose=verbose and restart < 3,
                rng=rng,
            )
            for res in found:
                if res.is_exact:
                    all_results.append(res)
                    if verbose:
                        elapsed = time.time() - t0
                        print(
                            f"  restart {restart:>4d} [{elapsed:>7.1f}s]: "
                            f"FOUND — {res.summary()}"
                        )

            if verbose and restart % 50 == 0 and restart > 0:
                elapsed = time.time() - t0
                print(
                    f"  restart {restart:>4d} [{elapsed:>7.1f}s]: "
                    f"{len(all_results)} exact so far"
                )

        if verbose:
            elapsed = time.time() - t0
            print(
                f"\nHybrid search done in {elapsed:.1f}s. "
                f"Exact solutions: {len(all_results)}"
            )

        return all_results


def search_with_flip_escapes(
    m: int,
    p: int,
    n: int,
    target_rank: int,
    n_restarts: int = 200,
    n_steps: int = 20000,
    device: str = "cpu",
    flip_config: Optional[FlipHybridConfig] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> List[DecompositionResult]:
    """Functional entry point for flip-graph hybrid search."""
    hybrid = FlipGraphHybridSearch(
        m, p, n, device=device, flip_config=flip_config
    )
    return hybrid.search(
        target_rank,
        n_restarts=n_restarts,
        n_steps=n_steps,
        verbose=verbose,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Matrix multiplication search with flip-graph gradient escapes"
    )
    parser.add_argument("--case", type=str, required=True, help='e.g. "2,2,2"')
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--restarts", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    m, p, n = (int(x) for x in args.case.split(","))
    device = "cpu"
    if args.use_cuda:
        import torch as _torch
        if _torch.cuda.is_available():
            device = "cuda"

    flip_config = FlipHybridConfig()
    n_restarts = args.restarts
    n_steps = args.steps
    if args.quick:
        n_restarts = 20
        n_steps = 8000
        flip_config.n_flip_walks = 4
        flip_config.n_entry_flips = 2
        flip_config.escape_n_steps = 6000

    results = search_with_flip_escapes(
        m, p, n,
        target_rank=args.rank,
        n_restarts=n_restarts,
        n_steps=n_steps,
        device=device,
        flip_config=flip_config,
        verbose=True,
        seed=args.seed,
    )

    print(f"\nFound {len(results)} exact decomposition(s).")
    for r in results[:5]:
        print(f"  {r.summary()}")


if __name__ == "__main__":
    main()
