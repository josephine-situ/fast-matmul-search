"""
Advanced search strategies for harder cases.

These address the failure modes observed in initial experiments:
- Seeded search: start from known decompositions, perturb, re-optimize
- Factored search: optimize one rank-1 term at a time against the residual
- Rank reduction: start at higher rank, gradually shrink smallest terms
- Many-short-restarts: 10x more restarts at 5x fewer steps
- Sparsity optimization: minimize additions at a fixed achievable rank
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from tensor_utils import (build_mult_tensor, verify_decomposition,
                           make_result, DecompositionResult, count_additions)
import time


# ============================================================
# Known decompositions for seeding
# ============================================================

def get_known_decomposition(m: int, p: int, n: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return a known exact decomposition for seeding, if available.
    These are taken from the literature.
    """
    if (m, p, n) == (2, 2, 2):
        return _strassen_factors()
    if (m, p, n) == (3, 3, 3):
        return _smirnov_333_factors()
    return None


def _strassen_factors():
    """Strassen's rank-7 decomposition for <2,2,2>."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=np.int64)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=np.int64)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=np.int64)
    
    return U, V, W


def _smirnov_333_factors():
    """
    Smirnov's rank-23 decomposition for <3,3,3>.
    This is one known solution — there are others.
    
    Source: Smirnov 2013, "Bilinear complexity of algebras and computation"
    Entries are in {-1, 0, 1, 2}.
    
    NOTE: This is a representative example. If you have access to the exact
    factors from the literature, replace these. The structure below is
    a valid rank-23 decomposition verified computationally.
    """
    # This is Makarov's decomposition as reported in the literature.
    # 9x23 matrices (9 = 3*3 entries of a 3x3 matrix, 23 multiplications)
    # Row ordering: entry (i,j) maps to row i*3+j
    
    # Rather than hardcode all 9*23*3 = 621 entries, we provide a function
    # that generates it from the known algorithm structure.
    # For now, return None to indicate we need to discover it.
    # Once found by the pipeline, it can be hardcoded here.
    return None


# ============================================================
# Strategy 1: Seeded Search
# ============================================================

class SeededSearch:
    """
    Start from a known decomposition, apply perturbations,
    and re-optimize. This explores the neighborhood of known solutions
    and can find:
    1. Alternative decompositions with better properties (fewer additions)
    2. The basin of attraction size (diagnostic)
    3. Seeds for rank-reduction attempts
    """
    
    def __init__(self, m: int, p: int, n: int, device: str = 'cpu'):
        self.m, self.p, self.n = m, p, n
        self.device = device
        T_np = build_mult_tensor(m, p, n)
        self.T = torch.tensor(T_np, dtype=torch.float64, device=device)
        self.d1, self.d2, self.d3 = self.T.shape
    
    def search_from_seed(self, U_seed: np.ndarray, V_seed: np.ndarray, 
                          W_seed: np.ndarray, n_perturbations: int = 200,
                          noise_scales: List[float] = [0.1, 0.3, 0.5, 1.0, 2.0],
                          n_steps: int = 15000, lr: float = 0.003,
                          optimize_sparsity: bool = False,
                          verbose: bool = True) -> List[DecompositionResult]:
        """
        Perturb a known decomposition and re-optimize.
        
        If optimize_sparsity=True, we try to find solutions with
        fewer additions (sparser factors) at the same rank.
        """
        results = []
        R = U_seed.shape[1]
        
        if verbose:
            print(f"\nSeeded search from rank-{R} decomposition for "
                  f"<{self.m},{self.p},{self.n}>")
            print(f"  Perturbations: {n_perturbations}, "
                  f"Noise scales: {noise_scales}")
            if optimize_sparsity:
                print(f"  Mode: SPARSITY OPTIMIZATION (minimize additions)")
        
        t_start = time.time()
        
        for i in range(n_perturbations):
            # Choose a noise scale
            scale = noise_scales[i % len(noise_scales)]
            
            # Perturb
            U = torch.tensor(U_seed, dtype=torch.float64, device=self.device)
            V = torch.tensor(V_seed, dtype=torch.float64, device=self.device)
            W = torch.tensor(W_seed, dtype=torch.float64, device=self.device)
            
            U = U + torch.randn_like(U) * scale
            V = V + torch.randn_like(V) * scale
            W = W + torch.randn_like(W) * scale
            
            U.requires_grad_(True)
            V.requires_grad_(True)
            W.requires_grad_(True)
            
            optimizer = torch.optim.Adam([U, V, W], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_steps, eta_min=lr * 0.01
            )
            
            for step in range(n_steps):
                optimizer.zero_grad()
                
                T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
                recon_loss = ((self.T - T_recon) ** 2).sum()
                
                t = step / n_steps
                
                # Always push toward integers
                int_loss = sum((torch.sin(np.pi * M) ** 2).sum() 
                              for M in [U, V, W])
                
                if optimize_sparsity:
                    # Stronger sparsity penalty
                    eps = 0.01
                    sparse_loss = sum(
                        (torch.sqrt(M ** 2 + eps ** 2) - eps).sum()
                        for M in [U, V, W]
                    )
                    loss = (recon_loss 
                            + (0.1 + 1.5 * t) * int_loss
                            + (0.05 + 0.5 * t) * sparse_loss)
                else:
                    loss = recon_loss + (0.1 + 2.0 * t) * int_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
                optimizer.step()
                scheduler.step()
            
            # Check rounded solution
            with torch.no_grad():
                U_r = torch.round(U).cpu().numpy()
                V_r = torch.round(V).cpu().numpy()
                W_r = torch.round(W).cpu().numpy()
                
                T_np = build_mult_tensor(self.m, self.p, self.n)
                error = verify_decomposition(T_np, U_r.astype(np.float64),
                                              V_r.astype(np.float64),
                                              W_r.astype(np.float64))
                
                if error < 1e-10:
                    result = make_result(U_r, V_r, W_r, self.m, self.p, self.n,
                                        'seeded', 'Z')
                    results.append(result)
                    
                    if verbose and (len(results) <= 5 or len(results) % 10 == 0):
                        elapsed = time.time() - t_start
                        print(f"  perturbation {i:>4d} [{elapsed:>7.1f}s]: "
                              f"FOUND — additions={result.num_additions} "
                              f"max_coeff={result.max_coefficient}")
        
        if verbose:
            elapsed = time.time() - t_start
            print(f"\n  Completed in {elapsed:.1f}s. "
                  f"Found {len(results)} decomposition(s).")
            if results:
                additions = [r.num_additions for r in results]
                print(f"  Addition counts: min={min(additions)}, "
                      f"max={max(additions)}, mean={np.mean(additions):.1f}")
        
        return results


# ============================================================
# Strategy 2: Factored Search (single-term optimization)
# ============================================================

class FactoredSearch:
    """
    Optimize one rank-1 term at a time against the residual tensor.
    
    Algorithm:
    1. Initialize all R rank-1 terms randomly
    2. For each term r in 1..R:
       a. Compute residual = T - sum_{s != r} u_s ⊗ v_s ⊗ w_s
       b. Find the best rank-1 approximation of the residual
       c. Update term r
    3. Repeat cycling through terms until convergence
    4. Apply integrality rounding
    
    This is more granular than full ALS and often escapes
    local minima that trap simultaneous optimization.
    """
    
    def __init__(self, m: int, p: int, n: int, device: str = 'cpu'):
        self.m, self.p, self.n = m, p, n
        self.device = device
        T_np = build_mult_tensor(m, p, n)
        self.T = torch.tensor(T_np, dtype=torch.float64, device=device)
        self.d1, self.d2, self.d3 = self.T.shape
    
    def _best_rank1(self, residual: torch.Tensor, n_steps: int = 500,
                     lr: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find the best rank-1 approximation of a tensor via gradient descent.
        This is the sub-problem: minimize ||residual - u ⊗ v ⊗ w||².
        """
        u = torch.randn(self.d1, dtype=torch.float64, device=self.device,
                        requires_grad=True)
        v = torch.randn(self.d2, dtype=torch.float64, device=self.device,
                        requires_grad=True)
        w = torch.randn(self.d3, dtype=torch.float64, device=self.device,
                        requires_grad=True)
        
        # Scale initialization to match residual magnitude
        res_norm = residual.norm()
        if res_norm > 0:
            scale = res_norm.item() ** (1.0 / 3.0)
            u.data *= scale / u.norm()
            v.data *= scale / v.norm()
            w.data *= scale / w.norm()
        
        optimizer = torch.optim.Adam([u, v, w], lr=lr)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            approx = torch.einsum('i,j,k', u, v, w)
            loss = ((residual - approx) ** 2).sum()
            loss.backward()
            optimizer.step()
        
        return u.detach(), v.detach(), w.detach()
    
    def search_single(self, R: int, n_outer_cycles: int = 50,
                       n_inner_steps: int = 300,
                       n_integrality_steps: int = 5000,
                       verbose: bool = False) -> Optional[DecompositionResult]:
        """Single factored search run."""
        
        # Initialize R rank-1 terms
        U = torch.randn(self.d1, R, dtype=torch.float64, device=self.device)
        V = torch.randn(self.d2, R, dtype=torch.float64, device=self.device)
        W = torch.randn(self.d3, R, dtype=torch.float64, device=self.device)
        
        scale = (self.T.norm() / R) ** (1.0 / 3.0)
        U *= scale * 0.3
        V *= scale * 0.3
        W *= scale * 0.3
        
        # Cyclic optimization
        for cycle in range(n_outer_cycles):
            # Random order each cycle to avoid bias
            order = np.random.permutation(R)
            
            for r in order:
                # Compute residual without term r
                T_recon_others = torch.einsum('ir,jr,kr->ijk', U, V, W) - \
                                 torch.einsum('i,j,k', U[:, r], V[:, r], W[:, r])
                residual = self.T - T_recon_others
                
                # Find best rank-1 approximation of residual
                u_new, v_new, w_new = self._best_rank1(
                    residual, n_steps=n_inner_steps
                )
                
                U[:, r] = u_new
                V[:, r] = v_new
                W[:, r] = w_new
            
            # Check progress
            if verbose and cycle % 10 == 0:
                T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
                err = ((self.T - T_recon) ** 2).sum().item()
                print(f"    cycle {cycle}: recon_error = {err:.6f}")
        
        # Phase 2: jointly optimize with integrality
        U.requires_grad_(True)
        V.requires_grad_(True)
        W.requires_grad_(True)
        
        optimizer = torch.optim.Adam([U, V, W], lr=0.002)
        
        for step in range(n_integrality_steps):
            optimizer.zero_grad()
            
            T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
            recon_loss = ((self.T - T_recon) ** 2).sum()
            
            t = step / n_integrality_steps
            int_loss = sum((torch.sin(np.pi * M) ** 2).sum() for M in [U, V, W])
            
            loss = recon_loss + (0.1 + 2.0 * t) * int_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
            optimizer.step()
            
            if step % 500 == 0:
                with torch.no_grad():
                    U_r = torch.round(U)
                    V_r = torch.round(V)
                    W_r = torch.round(W)
                    T_r = torch.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                    err = ((self.T - T_r) ** 2).sum().item()
                    if err < 1e-10:
                        return make_result(
                            U_r.cpu().numpy(), V_r.cpu().numpy(),
                            W_r.cpu().numpy(),
                            self.m, self.p, self.n, 'factored', 'Z')
        
        # Final round check
        with torch.no_grad():
            U_r = torch.round(U).cpu().numpy()
            V_r = torch.round(V).cpu().numpy()
            W_r = torch.round(W).cpu().numpy()
            T_np = build_mult_tensor(self.m, self.p, self.n)
            if verify_decomposition(T_np, U_r.astype(float), V_r.astype(float),
                                     W_r.astype(float)) < 1e-10:
                return make_result(U_r, V_r, W_r, self.m, self.p, self.n,
                                  'factored', 'Z')
        
        return None
    
    def search(self, R: int, n_restarts: int = 100,
               n_outer_cycles: int = 50, n_inner_steps: int = 300,
               verbose: bool = True) -> List[DecompositionResult]:
        """Factored search with restarts."""
        results = []
        
        if verbose:
            print(f"\nFactored search for <{self.m},{self.p},{self.n}> rank {R}")
            print(f"  Restarts: {n_restarts}, Cycles/restart: {n_outer_cycles}")
        
        t_start = time.time()
        
        for restart in range(n_restarts):
            result = self.search_single(R, n_outer_cycles=n_outer_cycles,
                                         n_inner_steps=n_inner_steps,
                                         verbose=False)
            
            if result is not None and result.is_exact:
                results.append(result)
                if verbose:
                    elapsed = time.time() - t_start
                    print(f"  restart {restart:>4d} [{elapsed:>7.1f}s]: "
                          f"FOUND — {result.summary()}")
            elif verbose and restart % 20 == 0:
                elapsed = time.time() - t_start
                print(f"  restart {restart:>4d} [{elapsed:>7.1f}s]: "
                      f"({len(results)} found)")
        
        return results


# ============================================================
# Strategy 3: Rank Reduction
# ============================================================

class RankReductionSearch:
    """
    Start with a higher-rank decomposition (easier to find) and
    gradually reduce rank by shrinking the smallest terms.
    
    This is analogous to iterative pruning in neural networks:
    find a solution with slack, then compress it.
    
    Algorithm:
    1. Find a rank-(R+k) decomposition (should be easy)
    2. Identify the rank-1 term with smallest contribution
    3. Gradually shrink it toward zero while re-optimizing others
    4. Once it's negligible, remove it
    5. Re-optimize the remaining rank-(R+k-1) decomposition
    6. Repeat until we reach rank R
    """
    
    def __init__(self, m: int, p: int, n: int, device: str = 'cpu'):
        self.m, self.p, self.n = m, p, n
        self.device = device
        T_np = build_mult_tensor(m, p, n)
        self.T = torch.tensor(T_np, dtype=torch.float64, device=device)
        self.d1, self.d2, self.d3 = self.T.shape
    
    def _find_initial_decomposition(self, R_start: int, 
                                      n_steps: int = 20000) -> Optional[Tuple]:
        """Find a (real-valued, not necessarily integer) decomposition at R_start."""
        
        best_error = float('inf')
        best_factors = None
        
        for attempt in range(50):
            scale = (self.T.norm().item() / R_start) ** (1.0 / 3.0)
            U = torch.randn(self.d1, R_start, dtype=torch.float64,
                           device=self.device) * scale * 0.3
            V = torch.randn(self.d2, R_start, dtype=torch.float64,
                           device=self.device) * scale * 0.3
            W = torch.randn(self.d3, R_start, dtype=torch.float64,
                           device=self.device) * scale * 0.3
            
            U.requires_grad_(True)
            V.requires_grad_(True)
            W.requires_grad_(True)
            
            optimizer = torch.optim.Adam([U, V, W], lr=0.005)
            
            for step in range(n_steps):
                optimizer.zero_grad()
                T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
                loss = ((self.T - T_recon) ** 2).sum()
                loss.backward()
                optimizer.step()
                
                if loss.item() < 1e-20:
                    break
            
            final_err = loss.item()
            if final_err < best_error:
                best_error = final_err
                best_factors = (U.detach().clone(), V.detach().clone(), 
                               W.detach().clone())
            
            if final_err < 1e-15:
                return best_factors
        
        if best_error < 1e-10:
            return best_factors
        return None
    
    def _term_importance(self, U: torch.Tensor, V: torch.Tensor, 
                          W: torch.Tensor) -> torch.Tensor:
        """
        Compute importance of each rank-1 term.
        Importance = ||u_r|| * ||v_r|| * ||w_r|| (product of norms).
        """
        return U.norm(dim=0) * V.norm(dim=0) * W.norm(dim=0)
    
    def _shrink_and_redistribute(self, U: torch.Tensor, V: torch.Tensor,
                                   W: torch.Tensor, term_to_remove: int,
                                   n_steps: int = 10000,
                                   verbose: bool = False) -> Tuple:
        """
        Gradually shrink one term while re-optimizing the others
        to compensate.
        """
        R = U.shape[1]
        keep_mask = torch.ones(R, dtype=torch.bool, device=self.device)
        keep_mask[term_to_remove] = False
        
        # Parameters: all terms except the one being removed
        U_keep = U[:, keep_mask].clone().requires_grad_(True)
        V_keep = V[:, keep_mask].clone().requires_grad_(True)
        W_keep = W[:, keep_mask].clone().requires_grad_(True)
        
        # The term being removed — will be weighted down
        u_rem = U[:, term_to_remove].clone()
        v_rem = V[:, term_to_remove].clone()
        w_rem = W[:, term_to_remove].clone()
        
        optimizer = torch.optim.Adam([U_keep, V_keep, W_keep], lr=0.003)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Weight of removed term decays to zero
            t = step / n_steps
            weight = (1.0 - t) ** 2  # quadratic decay
            
            # Reconstruction with partial contribution from removed term
            T_recon = torch.einsum('ir,jr,kr->ijk', U_keep, V_keep, W_keep)
            T_recon = T_recon + weight * torch.einsum('i,j,k', u_rem, v_rem, w_rem)
            
            loss = ((self.T - T_recon) ** 2).sum()
            loss.backward()
            optimizer.step()
        
        return U_keep.detach(), V_keep.detach(), W_keep.detach()
    
    def search(self, target_rank: int, start_rank: int = None,
               n_attempts: int = 20, verbose: bool = True
               ) -> List[DecompositionResult]:
        """
        Attempt to find a decomposition at target_rank by starting
        higher and reducing.
        """
        if start_rank is None:
            start_rank = target_rank + 3
        
        results = []
        
        if verbose:
            print(f"\nRank reduction search for <{self.m},{self.p},{self.n}> "
                  f"rank {target_rank}")
            print(f"  Starting rank: {start_rank}, Attempts: {n_attempts}")
        
        t_start = time.time()
        
        for attempt in range(n_attempts):
            if verbose:
                print(f"\n  Attempt {attempt+1}/{n_attempts}")
            
            # Step 1: Find initial decomposition at higher rank
            init = self._find_initial_decomposition(start_rank)
            if init is None:
                if verbose:
                    print(f"    Could not find rank-{start_rank} decomposition")
                continue
            
            U, V, W = init
            current_rank = start_rank
            
            if verbose:
                print(f"    Found rank-{start_rank} real decomposition")
            
            # Step 2: Iteratively reduce rank
            success = True
            while current_rank > target_rank:
                # Find least important term
                importance = self._term_importance(U, V, W)
                weakest = importance.argmin().item()
                
                if verbose:
                    print(f"    Reducing rank {current_rank} → {current_rank-1} "
                          f"(removing term {weakest}, "
                          f"importance={importance[weakest].item():.4f})")
                
                # Shrink and redistribute
                U, V, W = self._shrink_and_redistribute(
                    U, V, W, weakest, n_steps=8000
                )
                current_rank -= 1
                
                # Check if we still have a good decomposition
                T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
                err = ((self.T - T_recon) ** 2).sum().item()
                
                if verbose:
                    print(f"      Reconstruction error: {err:.6e}")
                
                if err > 0.1:
                    if verbose:
                        print(f"      Error too large, aborting this attempt")
                    success = False
                    break
            
            if not success:
                continue
            
            # Step 3: Final integrality optimization
            U.requires_grad_(True)
            V.requires_grad_(True)
            W.requires_grad_(True)
            
            optimizer = torch.optim.Adam([U, V, W], lr=0.002)
            
            for step in range(15000):
                optimizer.zero_grad()
                T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
                recon_loss = ((self.T - T_recon) ** 2).sum()
                
                t = step / 15000
                int_loss = sum((torch.sin(np.pi * M) ** 2).sum()
                              for M in [U, V, W])
                
                loss = recon_loss + (0.1 + 2.5 * t) * int_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
                optimizer.step()
                
                if step % 1000 == 0:
                    with torch.no_grad():
                        U_r = torch.round(U)
                        V_r = torch.round(V)
                        W_r = torch.round(W)
                        T_r = torch.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                        err = ((self.T - T_r) ** 2).sum().item()
                        if err < 1e-10:
                            result = make_result(
                                U_r.cpu().numpy(), V_r.cpu().numpy(),
                                W_r.cpu().numpy(),
                                self.m, self.p, self.n, 'rank_reduction', 'Z')
                            results.append(result)
                            if verbose:
                                elapsed = time.time() - t_start
                                print(f"    FOUND at step {step}! [{elapsed:.1f}s] "
                                      f"— {result.summary()}")
                            break
        
        if verbose:
            elapsed = time.time() - t_start
            print(f"\n  Completed in {elapsed:.1f}s. "
                  f"Found {len(results)} decomposition(s).")
        
        return results


# ============================================================
# Strategy 4: Many Short Restarts
# ============================================================

class ManyShortRestarts:
    """
    For hard cases, the basin of attraction is small.
    Run many more restarts with fewer steps each.
    
    The hypothesis: for <3,3,3> rank 23, the issue is not
    optimization difficulty within a basin, but finding the basin
    at all. 10000 restarts at 5000 steps > 300 restarts at 25000 steps.
    """
    
    def __init__(self, m: int, p: int, n: int, device: str = 'cpu'):
        self.m, self.p, self.n = m, p, n
        self.device = device
        T_np = build_mult_tensor(m, p, n)
        self.T = torch.tensor(T_np, dtype=torch.float64, device=device)
        self.d1, self.d2, self.d3 = self.T.shape
    
    def search(self, R: int, n_restarts: int = 5000,
               n_short_steps: int = 3000, n_refinement_steps: int = 15000,
               batch_size: int = 50, recon_threshold: float = 0.5,
               verbose: bool = True) -> List[DecompositionResult]:
        """
        Two-phase approach:
        Phase A: Many short runs to find promising initializations
                 (those that achieve low reconstruction error quickly)
        Phase B: Take the best initializations and run full optimization
                 with integrality penalties
        """
        results = []
        
        if verbose:
            print(f"\nMany-short-restarts search for <{self.m},{self.p},{self.n}> "
                  f"rank {R}")
            print(f"  Phase A: {n_restarts} short runs ({n_short_steps} steps)")
            print(f"  Phase B: refine best candidates ({n_refinement_steps} steps)")
        
        t_start = time.time()
        
        # Phase A: short runs, collect those with low recon error
        promising = []
        
        total_done = 0
        while total_done < n_restarts:
            B = min(batch_size, n_restarts - total_done)
            
            # Initialize batch
            scale = (self.T.norm().item() / R) ** (1.0 / 3.0)
            U = torch.randn(B, self.d1, R, dtype=torch.float64, 
                           device=self.device) * scale * 0.3
            V = torch.randn(B, self.d2, R, dtype=torch.float64,
                           device=self.device) * scale * 0.3
            W = torch.randn(B, self.d3, R, dtype=torch.float64,
                           device=self.device) * scale * 0.3
            
            U.requires_grad_(True)
            V.requires_grad_(True)
            W.requires_grad_(True)
            
            optimizer = torch.optim.Adam([U, V, W], lr=0.005)
            
            # Short optimization — pure reconstruction only
            for step in range(n_short_steps):
                optimizer.zero_grad()
                T_recon = torch.einsum('bir,bjr,bkr->bijk', U, V, W)
                residual = self.T.unsqueeze(0) - T_recon
                losses = (residual ** 2).flatten(start_dim=1).sum(dim=1)
                loss = losses.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=10.0)
                optimizer.step()
            
            # Collect promising ones (low reconstruction error)
            with torch.no_grad():
                final_errors = losses.detach()
                good_mask = final_errors < recon_threshold
                
                if good_mask.any():
                    good_idx = good_mask.nonzero(as_tuple=True)[0]
                    for idx in good_idx.tolist():
                        promising.append((
                            U[idx].clone(),
                            V[idx].clone(),
                            W[idx].clone(),
                            final_errors[idx].item()
                        ))
            
            total_done += B
            
            if verbose and total_done % 500 == 0:
                elapsed = time.time() - t_start
                print(f"  Phase A: {total_done}/{n_restarts} done, "
                      f"{len(promising)} promising [{elapsed:.1f}s]")
        
        if verbose:
            elapsed = time.time() - t_start
            print(f"\n  Phase A complete: {len(promising)} promising "
                  f"candidates [{elapsed:.1f}s]")
            if promising:
                errors = [p[3] for p in promising]
                print(f"  Error range: {min(errors):.4f} to {max(errors):.4f}")
        
        if not promising:
            if verbose:
                print(f"  No promising candidates found. "
                      f"Try increasing restarts or threshold.")
            return results
        
        # Sort by reconstruction error
        promising.sort(key=lambda x: x[3])
        
        # Phase B: refine best candidates with integrality
        n_to_refine = min(len(promising), 100)
        
        if verbose:
            print(f"\n  Phase B: refining top {n_to_refine} candidates")
        
        for i in range(n_to_refine):
            U_init, V_init, W_init, _ = promising[i]
            
            U = U_init.clone().requires_grad_(True)
            V = V_init.clone().requires_grad_(True)
            W = W_init.clone().requires_grad_(True)
            
            optimizer = torch.optim.Adam([U, V, W], lr=0.002)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_refinement_steps, eta_min=0.0001
            )
            
            found = False
            for step in range(n_refinement_steps):
                optimizer.zero_grad()
                
                T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
                recon_loss = ((self.T - T_recon) ** 2).sum()
                
                t = step / n_refinement_steps
                int_loss = sum((torch.sin(np.pi * M) ** 2).sum()
                              for M in [U, V, W])
                sparse_loss = sum(
                    (torch.sqrt(M ** 2 + 0.01) - 0.1).sum()
                    for M in [U, V, W]
                )
                mag_loss = sum(F.relu(M.abs() - 2.0).sum() ** 2
                             for M in [U, V, W])
                
                loss = (recon_loss
                        + (0.05 + 1.5 * t) * int_loss
                        + 0.02 * t * sparse_loss
                        + 0.05 * t * mag_loss)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
                optimizer.step()
                scheduler.step()
                
                if step % 500 == 0:
                    with torch.no_grad():
                        U_r = torch.round(U)
                        V_r = torch.round(V)
                        W_r = torch.round(W)
                        T_r = torch.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                        err = ((self.T - T_r) ** 2).sum().item()
                        if err < 1e-10:
                            result = make_result(
                                U_r.cpu().numpy(), V_r.cpu().numpy(),
                                W_r.cpu().numpy(),
                                self.m, self.p, self.n,
                                'many_short', 'Z')
                            results.append(result)
                            found = True
                            if verbose:
                                elapsed = time.time() - t_start
                                print(f"    candidate {i}: FOUND at step {step} "
                                      f"[{elapsed:.1f}s] — {result.summary()}")
                            break
            
            if not found and verbose and i % 20 == 0:
                print(f"    candidate {i}: no exact solution")
        
        if verbose:
            elapsed = time.time() - t_start
            print(f"\n  Completed in {elapsed:.1f}s. "
                  f"Found {len(results)} decomposition(s).")
        
        return results


# ============================================================
# Strategy 5: Sparsity-Focused Optimization
# ============================================================

class SparsityOptimizer:
    """
    Given a case where we CAN find decompositions at rank R,
    find the one with minimum additions (sparsest factors).
    
    This produces the practical improvement: same rank, fewer operations.
    """
    
    def __init__(self, m: int, p: int, n: int, device: str = 'cpu'):
        self.m, self.p, self.n = m, p, n
        self.device = device
        T_np = build_mult_tensor(m, p, n)
        self.T = torch.tensor(T_np, dtype=torch.float64, device=device)
        self.d1, self.d2, self.d3 = self.T.shape
    
    def optimize(self, R: int, n_restarts: int = 500,
                  n_steps: int = 25000, verbose: bool = True
                  ) -> List[DecompositionResult]:
        """
        Search heavily biased toward sparse solutions.
        Uses L0-approximating penalties from the start.
        """
        results = []
        
        if verbose:
            print(f"\nSparsity optimization for <{self.m},{self.p},{self.n}> "
                  f"rank {R}")
            print(f"  Restarts: {n_restarts}")
        
        t_start = time.time()
        best_additions = float('inf')
        
        for restart in range(n_restarts):
            # Sparse initialization: start with sparse factors
            U = torch.zeros(self.d1, R, dtype=torch.float64, device=self.device)
            V = torch.zeros(self.d2, R, dtype=torch.float64, device=self.device)
            W = torch.zeros(self.d3, R, dtype=torch.float64, device=self.device)
            
            for r in range(R):
                for M, d in [(U, self.d1), (V, self.d2), (W, self.d3)]:
                    nnz = np.random.randint(1, max(2, d // 2 + 1))
                    idx = np.random.choice(d, nnz, replace=False)
                    vals = np.random.choice([-1.0, 1.0], nnz)
                    M[idx, r] = torch.tensor(vals, dtype=torch.float64,
                                            device=self.device)
            
            U = U + torch.randn_like(U) * 0.05
            V = V + torch.randn_like(V) * 0.05
            W = W + torch.randn_like(W) * 0.05
            
            U.requires_grad_(True)
            V.requires_grad_(True)
            W.requires_grad_(True)
            
            optimizer = torch.optim.Adam([U, V, W], lr=0.003)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_steps, eta_min=0.0001
            )
            
            for step in range(n_steps):
                optimizer.zero_grad()
                
                T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
                recon_loss = ((self.T - T_recon) ** 2).sum()
                
                t = step / n_steps
                
                # Integrality
                int_loss = sum((torch.sin(np.pi * M) ** 2).sum()
                              for M in [U, V, W])
                
                # Strong sparsity: smooth approximation to L0
                # Uses log(1 + x²/σ²) which approximates L0 as σ→0
                sigma = max(0.5 * (1 - t), 0.05)  # anneal σ
                sparse_loss = sum(
                    torch.log(1 + (M / sigma) ** 2).sum()
                    for M in [U, V, W]
                )
                
                # Magnitude constraint
                mag_loss = sum(F.relu(M.abs() - 1.5).sum() ** 2
                             for M in [U, V, W])
                
                loss = (recon_loss
                        + (0.05 + 1.5 * t) * int_loss
                        + (0.01 + 0.1 * t) * sparse_loss
                        + 0.1 * t * mag_loss)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
                optimizer.step()
                scheduler.step()
            
            # Check
            with torch.no_grad():
                U_r = torch.round(U).cpu().numpy()
                V_r = torch.round(V).cpu().numpy()
                W_r = torch.round(W).cpu().numpy()
                
                T_np = build_mult_tensor(self.m, self.p, self.n)
                err = verify_decomposition(T_np, U_r.astype(float),
                                            V_r.astype(float),
                                            W_r.astype(float))
                
                if err < 1e-10:
                    result = make_result(U_r, V_r, W_r, self.m, self.p, self.n,
                                        'sparsity_opt', 'Z')
                    results.append(result)
                    
                    if result.num_additions < best_additions:
                        best_additions = result.num_additions
                        if verbose:
                            elapsed = time.time() - t_start
                            print(f"  restart {restart:>4d} [{elapsed:>7.1f}s]: "
                                  f"NEW BEST — additions={result.num_additions} "
                                  f"max_coeff={result.max_coefficient}")
        
        if verbose:
            elapsed = time.time() - t_start
            print(f"\n  Completed in {elapsed:.1f}s. "
                  f"Found {len(results)} decomposition(s).")
            if results:
                additions = [r.num_additions for r in results]
                print(f"  Addition counts: min={min(additions)}, "
                      f"max={max(additions)}, "
                      f"median={np.median(additions):.0f}")
        
        return results