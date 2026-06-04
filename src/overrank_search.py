"""
Formulation B': over-parameterized rank-N·R search with column pruning.

N=2, FLOP-matched budget, LOO pruning, cancellation penalty.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hardcode_known import min_hamming_to_known
from tensor_utils import (
    DecompositionResult,
    build_mult_tensor,
    make_result,
    verify_decomposition,
    wrong_entries,
)

OVER_RANK_N = 2
MASS_EPS = 1e-4
PENALTY_GRAD_CAP = 5.0
GLOBAL_GRAD_CAP = 5.0
LOSS_SAMPLE_EVERY = 500

FLOP_SHARE_MAIN = 0.90
FLOP_SHARE_REFINE = 0.07
FLOP_SHARE_SNAP = 0.03

INIT_GAUSSIAN_FRAC = 2.0 / 3.0


def compute_overrank_step_budget(
    baseline_steps: int = 25000,
    budget_mode: str = "flops_matched",
) -> Tuple[int, int, int]:
    """Return (n_steps_main, n_steps_refine, n_steps_snap) for N=2."""
    if budget_mode == "steps_matched":
        total = baseline_steps / FLOP_SHARE_MAIN
    else:
        total = baseline_steps / OVER_RANK_N
    n_main = int(total * FLOP_SHARE_MAIN)
    n_refine = int(total * FLOP_SHARE_REFINE)
    n_snap = int(total * FLOP_SHARE_SNAP)
    return n_main, n_refine, n_snap


def compute_overrank_restarts(
    baseline_restarts: int = 300,
    baseline_steps: int = 25000,
    budget_mode: str = "flops_matched",
) -> int:
    """Match total FLOPs: B_base·S_base ≈ B_or·S_or·N."""
    n_main, n_refine, n_snap = compute_overrank_step_budget(baseline_steps, budget_mode)
    s_or = n_main + n_refine + n_snap
    return max(1, int(baseline_restarts * baseline_steps / (s_or * OVER_RANK_N)))


def _column_masses(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return U.norm(dim=0) * V.norm(dim=0) * W.norm(dim=0)


def _column_mass_penalty(
    U: torch.Tensor, V: torch.Tensor, W: torch.Tensor, mu: float
) -> torch.Tensor:
    if mu <= 0:
        return torch.zeros((), dtype=U.dtype, device=U.device)
    m = _column_masses(U, V, W)
    return mu * ((m + MASS_EPS) ** (2.0 / 3.0)).sum()


def _cancellation_penalty(
    U: torch.Tensor, V: torch.Tensor, W: torch.Tensor, lam: float
) -> torch.Tensor:
    if lam <= 0:
        return torch.zeros((), dtype=U.dtype, device=U.device)
    R = U.shape[1]
    loss = torch.zeros((), dtype=U.dtype, device=U.device)
    for r in range(R):
        ur, vr, wr = U[:, r], V[:, r], W[:, r]
        for s in range(r + 1, R):
            ip = (ur @ U[:, s]) * (vr @ V[:, s]) * (wr @ W[:, s])
            loss = loss + F.relu(-ip)
    return lam * loss


def _effective_rank(m: torch.Tensor) -> float:
    s2 = (m ** 2).sum()
    s4 = (m ** 4).sum()
    return float((s2 ** 2 / (s4 + 1e-30)).item())


def _loo_deltas(
    T: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    """Δ_r = m_r² + 2⟨ρ, u_r⊗v_r⊗w_r⟩."""
    T_hat = torch.einsum("ir,jr,kr->ijk", U, V, W)
    rho = T - T_hat
    R = U.shape[1]
    masses = _column_masses(U, V, W)
    deltas = []
    for r in range(R):
        term = (
            U[:, r].unsqueeze(1).unsqueeze(2)
            * V[:, r].unsqueeze(0).unsqueeze(2)
            * W[:, r].unsqueeze(0).unsqueeze(1)
        )
        inner = (rho * term).sum()
        deltas.append(masses[r] ** 2 + 2.0 * inner)
    return torch.stack(deltas)


def _max_consecutive_gap_ratio(sorted_vals: np.ndarray) -> float:
    """Largest ratio sorted[i+1]/sorted[i] over the full ascending mass vector."""
    if len(sorted_vals) < 2:
        return float("nan")
    best = 0.0
    for i in range(len(sorted_vals) - 1):
        denom = sorted_vals[i]
        if denom <= 1e-30:
            ratio = float("inf")
        else:
            ratio = float(sorted_vals[i + 1] / denom)
        if np.isfinite(ratio):
            best = max(best, ratio)
        else:
            return float("inf")
    return best if best > 0 else float("nan")


def _init_method_for_restart(restart_id: int, n_restarts: int) -> str:
    """~2/3 gaussian; sparse and uniform are smaller ablation controls."""
    n_gaussian = int(n_restarts * INIT_GAUSSIAN_FRAC)
    n_sparse = (n_restarts - n_gaussian) // 2
    if restart_id < n_gaussian:
        return "gaussian"
    if restart_id < n_gaussian + n_sparse:
        return "sparse"
    return "uniform"


def _frobenius_recon_error(T: np.ndarray, U: np.ndarray, V: np.ndarray, W: np.ndarray) -> float:
    T_recon = np.einsum("ir,jr,kr->ijk", U, V, W)
    return float(np.sum((T - T_recon) ** 2))


def _try_one_flip_exact(
    T: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    R: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Flip one factor entry; return updated factors if reconstruction becomes exact."""
    U_i = np.round(U).astype(np.int64)
    V_i = np.round(V).astype(np.int64)
    W_i = np.round(W).astype(np.int64)

    if verify_decomposition(T, U_i.astype(float), V_i.astype(float), W_i.astype(float)) < 1e-10:
        return U_i, V_i, W_i, False

    candidates = (-1, 0, 1)
    for name, M in (("U", U_i), ("V", V_i), ("W", W_i)):
        for i in range(M.shape[0]):
            for r in range(R):
                orig = int(M[i, r])
                for new_val in candidates:
                    if new_val == orig:
                        continue
                    U_try, V_try, W_try = U_i.copy(), V_i.copy(), W_i.copy()
                    if name == "U":
                        U_try[i, r] = new_val
                    elif name == "V":
                        V_try[i, r] = new_val
                    else:
                        W_try[i, r] = new_val
                    err = verify_decomposition(
                        T,
                        U_try.astype(float),
                        V_try.astype(float),
                        W_try.astype(float),
                    )
                    if err < 1e-10:
                        return U_try, V_try, W_try, True
    return U_i, V_i, W_i, False


def _save_exact_decomposition(
    output_dir: str,
    m: int,
    p: int,
    n: int,
    R: int,
    res: DecompositionResult,
    tag: str,
) -> str:
    save_path = os.path.join(output_dir, f"{m}_{p}_{n}_rank{R}_overrank")
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, f"{tag}.npz")
    np.savez(
        filepath,
        U=res.U,
        V=res.V,
        W=res.W,
        method=res.method,
        additions=res.num_additions,
        max_coeff=res.max_coefficient,
        reconstruction_error=res.reconstruction_error,
    )
    return filepath


def _min_pair_inner(
    U: torch.Tensor, V: torch.Tensor, W: torch.Tensor, indices: torch.Tensor
) -> float:
    idx = indices.tolist()
    if len(idx) < 2:
        return 0.0
    worst = 0.0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            r, s = idx[i], idx[j]
            ip = float((U[:, r] @ U[:, s]) * (V[:, r] @ V[:, s]) * (W[:, r] @ W[:, s]))
            worst = min(worst, ip)
    return worst


def _max_cosine_survivor_dead(
    U: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
    keep_idx: torch.Tensor,
    drop_idx: torch.Tensor,
) -> float:
    if len(keep_idx) == 0 or len(drop_idx) == 0:
        return 0.0
    best = 0.0
    for ki in keep_idx.tolist():
        for di in drop_idx.tolist():
            for M in (U, V, W):
                a, b = M[:, ki], M[:, di]
                na, nb = a.norm(), b.norm()
                if na > 1e-12 and nb > 1e-12:
                    best = max(best, float((a @ b / (na * nb)).abs().item()))
    return best


def _mass_penalty_grads_clipped(
    U: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
    mass_pen: torch.Tensor,
    R_eff: int,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return per-column capped gradients from mass penalty only."""
    if not mass_pen.requires_grad or mass_pen.item() == 0:
        return {}
    mass_pen.backward(retain_graph=True)
    out: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for r in range(R_eff):
        if U.grad is None:
            break
        gu = U.grad[:, r].clone()
        gv = V.grad[:, r].clone()
        gw = W.grad[:, r].clone()
        gn = (gu.norm() ** 2 + gv.norm() ** 2 + gw.norm() ** 2).sqrt()
        if gn > PENALTY_GRAD_CAP:
            s = PENALTY_GRAD_CAP / gn
            gu, gv, gw = gu * s, gv * s, gw * s
        out[r] = (gu, gv, gw)
    return out


def _add_mass_grads(
    U: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
    mass_grads: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    for r, (gu, gv, gw) in mass_grads.items():
        U.grad[:, r] = U.grad[:, r] + gu
        V.grad[:, r] = V.grad[:, r] + gv
        W.grad[:, r] = W.grad[:, r] + gw


def _slice_adam_optimizer(
    optimizer: torch.optim.Adam,
    U: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
    keep_idx: torch.Tensor,
    lr: float,
) -> Tuple[torch.optim.Adam, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = keep_idx.to(U.device)
    U_new = U[:, idx].detach().clone().requires_grad_(True)
    V_new = V[:, idx].detach().clone().requires_grad_(True)
    W_new = W[:, idx].detach().clone().requires_grad_(True)

    old_params = [U, V, W]
    new_params = [U_new, V_new, W_new]
    new_opt = torch.optim.Adam(new_params, lr=lr)

    for old_p, new_p in zip(old_params, new_params):
        if old_p not in optimizer.state:
            continue
        st = optimizer.state[old_p]
        new_st: Dict[str, Any] = {}
        if "step" in st:
            new_st["step"] = st["step"]
        if "exp_avg" in st:
            new_st["exp_avg"] = st["exp_avg"].index_select(1, idx).clone()
            new_st["exp_avg_sq"] = st["exp_avg_sq"].index_select(1, idx).clone()
        new_opt.state[new_p] = new_st

    return new_opt, U_new, V_new, W_new


def _rounded_key(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> str:
    return (
        np.round(U).astype(np.int16).tobytes()
        + np.round(V).astype(np.int16).tobytes()
        + np.round(W).astype(np.int16).tobytes()
    ).hex()


class OverRankSearchMixin:
    """Formulation B' methods mixed into ContinuousSearch."""

    def _init_overrank_factors(
        self,
        R_target: int,
        method: str = "gaussian",
        extra_scale: float = 0.25,
        N: int = OVER_RANK_N,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        R_eff = N * R_target
        U_p, V_p, W_p = self._init_factors(R_target, method=method)
        U_p = U_p.detach()
        V_p = V_p.detach()
        W_p = W_p.detach()

        scale_base = (self.T_norm / R_target) ** (1.0 / 3.0) * 0.3
        extra_mult = extra_scale * scale_base * (N ** (-1.0 / 3.0))

        n_extra = R_eff - R_target
        if method == "uniform":
            U_e = (torch.rand(self.d1, n_extra, dtype=torch.float64, device=self.device) - 0.5) * 4
            V_e = (torch.rand(self.d2, n_extra, dtype=torch.float64, device=self.device) - 0.5) * 4
            W_e = (torch.rand(self.d3, n_extra, dtype=torch.float64, device=self.device) - 0.5) * 4
        else:
            U_e = torch.randn(self.d1, n_extra, dtype=torch.float64, device=self.device)
            V_e = torch.randn(self.d2, n_extra, dtype=torch.float64, device=self.device)
            W_e = torch.randn(self.d3, n_extra, dtype=torch.float64, device=self.device)
        U_e = U_e * extra_mult
        V_e = V_e * extra_mult
        W_e = W_e * extra_mult

        U = torch.cat([U_p, U_e], dim=1).requires_grad_(True)
        V = torch.cat([V_p, V_e], dim=1).requires_grad_(True)
        W = torch.cat([W_p, W_e], dim=1).requires_grad_(True)
        return U, V, W

    def _phase_weights(self, step: int, n_steps: int) -> Dict[str, float]:
        phase1_end = int(n_steps * 0.4)
        phase2_end = int(n_steps * 0.7)
        if step < phase1_end:
            return {
                "mu": 0.0,
                "lam_pair": 0.0,
                "int_w": 0.0,
                "sparse_w": 0.0,
                "mag_w": 0.0,
                "bal_w": 0.01,
            }
        if step < phase2_end:
            t = (step - phase1_end) / max(1, phase2_end - phase1_end)
            mu = 0.05 * t ** 2
            return {
                "mu": mu,
                "lam_pair": 0.1 * mu,
                "int_w": 0.3 * t ** 2,
                "sparse_w": 0.05 * t,
                "mag_w": 0.1 * t,
                "bal_w": 0.01,
            }
        t = (step - phase2_end) / max(1, n_steps - phase2_end)
        mu = 0.05 + 0.5 * t
        return {
            "mu": mu,
            "lam_pair": 0.1 * mu,
            "int_w": 0.3 + 2.0 * t,
            "sparse_w": 0.05 + 0.2 * t,
            "mag_w": 0.1 + 0.5 * t,
            "bal_w": 0.0,
        }

    def _main_overrank_step(
        self,
        optimizer: torch.optim.Optimizer,
        U: torch.Tensor,
        V: torch.Tensor,
        W: torch.Tensor,
        step: int,
        n_steps: int,
    ) -> Dict[str, float]:
        w = self._phase_weights(step, n_steps)
        optimizer.zero_grad()

        recon = self._recon_loss(U, V, W)
        loss_core = recon
        loss_core = loss_core + w["bal_w"] * self._balance_loss(U, V, W)
        loss_core = loss_core + w["int_w"] * self._integrality_loss(U, V, W)
        loss_core = loss_core + w["sparse_w"] * self._sparsity_loss(U, V, W)
        loss_core = loss_core + w["mag_w"] * self._magnitude_loss(U, V, W)
        cancel = _cancellation_penalty(U, V, W, w["lam_pair"])
        mass = _column_mass_penalty(U, V, W, w["mu"])
        loss_core = loss_core + cancel

        mass_grads = _mass_penalty_grads_clipped(U, V, W, mass, U.shape[1])
        optimizer.zero_grad()
        loss_core.backward()
        if mass_grads:
            _add_mass_grads(U, V, W, mass_grads)
        torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=GLOBAL_GRAD_CAP)
        optimizer.step()

        with torch.no_grad():
            m = _column_masses(U, V, W)
            eff = _effective_rank(m)
        return {
            "recon": float(recon.item()),
            "mass_pen": float(mass.item()),
            "cancel_pen": float(cancel.item()),
            "mu": w["mu"],
            "eff_rank": eff,
        }

    def _run_overrank_single(
        self,
        R: int,
        restart_id: int,
        seed: int,
        n_steps_main: int,
        n_steps_refine: int,
        n_steps_snap: int,
        lr: float = 0.003,
        init_method: str = "gaussian",
        extra_scale: float = 0.25,
        budget_mode: str = "flops_matched",
        save_dir: Optional[str] = None,
    ) -> Tuple[Optional[DecompositionResult], Dict[str, Any]]:
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 31))

        R_eff = OVER_RANK_N * R
        t0 = time.time()
        ttfe_steps: Optional[int] = None

        U, V, W = self._init_overrank_factors(R, method=init_method, extra_scale=extra_scale)
        optimizer = torch.optim.Adam([U, V, W], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, n_steps_main // 4), T_mult=1, eta_min=lr * 0.01
        )

        loss_samples: List[Dict[str, float]] = []
        exact_hit = False
        best_exact: Optional[DecompositionResult] = None

        for step in range(n_steps_main):
            comps = self._main_overrank_step(optimizer, U, V, W, step, n_steps_main)
            scheduler.step()
            if step % LOSS_SAMPLE_EVERY == 0:
                loss_samples.append(comps)

            if step % 1000 == 0 or step == n_steps_main - 1:
                with torch.no_grad():
                    U_r = torch.round(U)
                    V_r = torch.round(V)
                    W_r = torch.round(W)
                    err = self._recon_loss(U_r, V_r, W_r).item()
                    if err < 1e-10 and not exact_hit:
                        exact_hit = True
                        ttfe_steps = step
                        best_exact = make_result(
                            U_r.cpu().numpy(),
                            V_r.cpu().numpy(),
                            W_r.cpu().numpy(),
                            self.m,
                            self.p,
                            self.n,
                            "overrank",
                            "Z",
                        )

        with torch.no_grad():
            recon_final = float(self._recon_loss(U, V, W).item())
            masses = _column_masses(U, V, W).cpu().numpy()
            deltas = _loo_deltas(self.T, U, V, W).cpu().numpy()
            sort_mass = np.sort(masses)
            sort_loo = np.sort(deltas)
            gap_mass = _max_consecutive_gap_ratio(sort_mass)
            gap_loo = _max_consecutive_gap_ratio(sort_loo)

            loo_order = np.argsort(-deltas)
            keep_idx = torch.tensor(loo_order[:R], device=self.device, dtype=torch.long)
            drop_idx = torch.tensor(loo_order[R:], device=self.device, dtype=torch.long)

            min_inner_top = _min_pair_inner(U, V, W, keep_idx)
            min_inner_bottom = _min_pair_inner(U, V, W, drop_idx)
            max_cos = _max_cosine_survivor_dead(U, V, W, keep_idx, drop_idx)

            U_pr = U[:, keep_idx]
            V_pr = V[:, keep_idx]
            W_pr = W[:, keep_idx]
            recon_pruned = float(self._recon_loss(U_pr, V_pr, W_pr).item())

        optimizer, U, V, W = _slice_adam_optimizer(optimizer, U, V, W, keep_idx, lr)

        refine_hit_cap = False
        refine_steps_done = 0
        prev_ref = None
        for step in range(n_steps_refine):
            refine_steps_done = step + 1
            w = self._phase_weights(step, n_steps_refine)
            optimizer.zero_grad()
            recon = self._recon_loss(U, V, W)
            loss = (
                recon
                + w["int_w"] * self._integrality_loss(U, V, W)
                + w["sparse_w"] * self._sparsity_loss(U, V, W)
                + w["mag_w"] * self._magnitude_loss(U, V, W)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=GLOBAL_GRAD_CAP)
            optimizer.step()
            cur = float(recon.item())
            if prev_ref is not None and abs(cur - prev_ref) < 1e-12:
                break
            prev_ref = cur
        else:
            if n_steps_refine > 0:
                refine_hit_cap = True

        with torch.no_grad():
            recon_refined = float(self._recon_loss(U, V, W).item())

        snap_result = self._snap_and_refine(
            U.detach(), V.detach(), W.detach(), n_steps=n_steps_snap
        )
        recon_snapped = recon_refined
        if snap_result is not None:
            recon_snapped = float(snap_result.reconstruction_error)
            if snap_result.is_exact:
                exact_hit = True
                best_exact = snap_result
                if ttfe_steps is None:
                    ttfe_steps = n_steps_main + refine_steps_done

        wall = time.time() - t0
        U_np = U.detach().cpu().numpy()
        V_np = V.detach().cpu().numpy()
        W_np = W.detach().cpu().numpy()
        if snap_result is not None:
            U_np = snap_result.U.astype(float)
            V_np = snap_result.V.astype(float)
            W_np = snap_result.W.astype(float)

        T_np = build_mult_tensor(self.m, self.p, self.n)
        T_norm_sq = float(np.sum(T_np ** 2))

        U_r = np.round(U_np).astype(np.int64)
        V_r = np.round(V_np).astype(np.int64)
        W_r = np.round(W_np).astype(np.int64)

        recon_frobenius_snapped = _frobenius_recon_error(T_np, U_r, V_r, W_r)
        recon_max_snapped = float(
            verify_decomposition(
                T_np, U_r.astype(float), V_r.astype(float), W_r.astype(float)
            )
        )
        one_flip_applied = False
        if recon_max_snapped >= 1e-10:
            U_r, V_r, W_r, one_flip_applied = _try_one_flip_exact(
                T_np, U_r, V_r, W_r, R
            )
            recon_frobenius_snapped = _frobenius_recon_error(T_np, U_r, V_r, W_r)
            recon_max_snapped = float(
                verify_decomposition(
                    T_np, U_r.astype(float), V_r.astype(float), W_r.astype(float)
                )
            )

        exact_hit_verified = recon_max_snapped < 1e-10
        exact_hit = exact_hit_verified
        n_wrong, n_total = wrong_entries(T_np, U_r, V_r, W_r)
        hamming_known = min_hamming_to_known(self.m, self.p, self.n, U_r, V_r, W_r)

        if exact_hit_verified:
            best_exact = make_result(
                U_r.astype(float),
                V_r.astype(float),
                W_r.astype(float),
                self.m,
                self.p,
                self.n,
                "overrank",
                "Z",
            )
            if save_dir:
                tag = f"restart_{restart_id}_seed_{seed}"
                _save_exact_decomposition(
                    save_dir, self.m, self.p, self.n, R, best_exact, tag
                )

        record: Dict[str, Any] = {
            "case": [self.m, self.p, self.n],
            "restart_id": restart_id,
            "seed": seed,
            "N": OVER_RANK_N,
            "R_eff": R_eff,
            "budget_mode": budget_mode,
            "hardware": str(self.device),
            "init_method": init_method,
            "extra_scale": extra_scale,
            "n_steps_main": n_steps_main,
            "n_steps_refine": n_steps_refine,
            "n_steps_snap": n_steps_snap,
            "ttfe_steps": ttfe_steps,
            "ttfe_wall_clock": wall if exact_hit else None,
            "recon_loss_final": recon_final,
            "recon_loss_pruned": recon_pruned,
            "recon_loss_refined": recon_refined,
            "recon_loss_refined_rel": recon_refined / T_norm_sq,
            "recon_loss_snapped": recon_snapped,
            "recon_frobenius_snapped": recon_frobenius_snapped,
            "recon_frobenius_snapped_rel": recon_frobenius_snapped / T_norm_sq,
            "recon_max_snapped": recon_max_snapped,
            "exact_hit_verified": exact_hit_verified,
            "one_flip_applied": one_flip_applied,
            "column_masses": sort_mass.tolist(),
            "column_loo_delta": sort_loo.tolist(),
            "gap_ratio_mass": gap_mass,
            "gap_ratio_loo": gap_loo,
            "min_pair_inner_top": min_inner_top,
            "min_pair_inner_bottom": min_inner_bottom,
            "max_cosine_survivor_dead": max_cos,
            "rounded_addition_count": int(
                make_result(U_r, V_r, W_r, self.m, self.p, self.n, "overrank", "Z").num_additions
            ),
            "rounded_max_coeff": int(np.max(np.abs(np.round(np.concatenate([U_np, V_np, W_np]))))),
            "exact_hit": exact_hit,
            "near_miss_K1": int(n_wrong <= 1),
            "near_miss_K2": int(n_wrong <= 2),
            "near_miss_K3": int(n_wrong <= 3),
            "n_wrong_entries": n_wrong,
            "hamming_to_known": hamming_known,
            "effective_rank_samples": [s.get("eff_rank", float("nan")) for s in loss_samples],
            "loss_components_samples": loss_samples,
            "refine_steps": refine_steps_done,
            "refine_hit_cap": refine_hit_cap,
            "rounded_key": _rounded_key(
                U_r.astype(float), V_r.astype(float), W_r.astype(float)
            ),
            "cancel_pen_final": (
                loss_samples[-1].get("cancel_pen") if loss_samples else None
            ),
            "mass_pen_final": (
                loss_samples[-1].get("mass_pen") if loss_samples else None
            ),
            "eff_rank_final": (
                loss_samples[-1].get("eff_rank") if loss_samples else None
            ),
        }

        return best_exact if exact_hit_verified else None, record

    def search_overrank(
        self,
        R: int,
        n_restarts: Optional[int] = None,
        baseline_steps: int = 25000,
        baseline_restarts: int = 300,
        budget_mode: str = "flops_matched",
        lr: float = 0.003,
        extra_scale: float = 0.25,
        verbose: bool = True,
        restart_log_path: Optional[str] = None,
    ) -> Tuple[List[DecompositionResult], Dict[str, Any]]:
        """Formulation B' search with FLOP-matched restarts and ~90/7/3 step split."""
        if n_restarts is None:
            n_restarts = compute_overrank_restarts(
                baseline_restarts, baseline_steps, budget_mode
            )
        n_main, n_refine, n_snap = compute_overrank_step_budget(baseline_steps, budget_mode)

        results: List[DecompositionResult] = []
        restart_records: List[Dict[str, Any]] = []
        unique_rounded: set = set()
        unique_exact_keys: set = set()
        t_start = time.time()
        save_dir = (
            os.path.dirname(os.path.abspath(restart_log_path))
            if restart_log_path
            else None
        )

        if verbose:
            print(
                f"\nOver-rank search <{self.m},{self.p},{self.n}> rank {R} "
                f"N={OVER_RANK_N} budget={budget_mode}"
            )
            n_gauss = int(n_restarts * INIT_GAUSSIAN_FRAC)
            n_sparse = (n_restarts - n_gauss) // 2
            n_uniform = n_restarts - n_gauss - n_sparse
            print(
                f"  restarts={n_restarts} (gaussian={n_gauss}, sparse={n_sparse}, "
                f"uniform={n_uniform}) steps=({n_main}/{n_refine}/{n_snap}) "
                f"extra_scale={extra_scale}"
            )

        log_f = open(restart_log_path, "a", encoding="utf-8") if restart_log_path else None

        for rid in range(n_restarts):
            seed = int(np.random.randint(0, 2 ** 31 - 1))
            method = _init_method_for_restart(rid, n_restarts)
            res, rec = self._run_overrank_single(
                R=R,
                restart_id=rid,
                seed=seed,
                n_steps_main=n_main,
                n_steps_refine=n_refine,
                n_steps_snap=n_snap,
                lr=lr,
                init_method=method,
                extra_scale=extra_scale,
                budget_mode=budget_mode,
                save_dir=save_dir,
            )
            restart_records.append(rec)
            unique_rounded.add(rec["rounded_key"])
            if log_f:
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
            if res is not None and res.is_exact:
                exact_key = _rounded_key(res.U, res.V, res.W)
                if exact_key not in unique_exact_keys:
                    unique_exact_keys.add(exact_key)
                    results.append(res)
                if verbose:
                    print(f"  restart {rid}: EXACT — {res.summary()}")

        if log_f:
            log_f.close()

        gaps_loo = [
            r["gap_ratio_loo"] for r in restart_records if np.isfinite(r["gap_ratio_loo"])
        ]
        summary = {
            "n_found": len(results),
            "n_restarts": n_restarts,
            "budget_mode": budget_mode,
            "mean_gap_ratio_loo": float(np.mean(gaps_loo)) if gaps_loo else None,
            "unique_rounded_candidates": len(unique_rounded),
            "elapsed_seconds": time.time() - t_start,
            "restart_records_count": len(restart_records),
        }

        if verbose:
            print(
                f"  Done: {len(results)} exact, "
                f"unique_rounded={len(unique_rounded)}, "
                f"mean_gap_loo={summary['mean_gap_ratio_loo']}"
            )

        return results, summary
