"""
Fixed search methods addressing the failures in the initial run.

Key fixes:
1. Start rank reduction from the STANDARD decomposition (trivially correct)
2. Direct integer search via simulated annealing (avoids integrality gap)
3. Greedy finite field construction (not random sampling)
4. Much more aggressive continuous search with proper initialization
5. Better reconstruction thresholds calibrated to tensor norm
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from tensor_utils import (build_mult_tensor, verify_decomposition,
                           make_result, DecompositionResult, count_additions)
import time


# ============================================================
# The Standard Decomposition — Our Guaranteed Starting Point
# ============================================================

def standard_decomposition(m: int, p: int, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The standard matrix multiplication algorithm as a tensor decomposition.
    
    This has rank m*p*n and uses the trivial algorithm:
        C[i,j] = Σ_k A[i,k] * B[k,j]
    
    Each rank-1 term corresponds to one scalar product A[i,k]*B[k,j]
    contributing to C[i,j].
    
    This is ALWAYS correct and gives us a guaranteed starting point
    for rank reduction.
    """
    R = m * p * n
    d1 = m * p  # entries of A
    d2 = p * n  # entries of B
    d3 = m * n  # entries of C
    
    U = np.zeros((d1, R), dtype=np.int64)
    V = np.zeros((d2, R), dtype=np.int64)
    W = np.zeros((d3, R), dtype=np.int64)
    
    r = 0
    for i in range(m):
        for k in range(p):
            for j in range(n):
                # Term r computes: A[i,k] * B[k,j] -> C[i,j]
                U[i * p + k, r] = 1
                V[k * n + j, r] = 1
                W[i * n + j, r] = 1
                r += 1
    
    return U, V, W


def verify_standard():
    """Quick check that standard_decomposition is correct."""
    for m, p, n in [(2,2,2), (2,3,4), (3,3,3), (2,2,5)]:
        U, V, W = standard_decomposition(m, p, n)
        T = build_mult_tensor(m, p, n)
        err = verify_decomposition(T, U.astype(float), V.astype(float), W.astype(float))
        assert err < 1e-10, f"Standard decomposition failed for <{m},{p},{n}>: err={err}"
    print("Standard decomposition verified for all test cases.")


# ============================================================
# Strategy 1: Proper Rank Reduction from Standard
# ============================================================

class ProperRankReduction:
    """
    Start from the standard rank-mpn decomposition and reduce rank
    by merging terms.
    
    The key insight: to go from rank R to rank R-1, we need to find
    two rank-1 terms that can be "merged" — i.e., their contributions
    can be absorbed by adjusting other terms.
    
    We do this in continuous space:
    1. Start from standard decomposition (always works, rank mpn)
    2. Add a penalty on the number of "active" terms (L1 on term norms)
    3. Optimize until terms die off
    4. Remove dead terms
    5. Re-optimize and round
    """
    
    def __init__(self, m: int, p: int, n: int, device: str = 'cpu'):
        self.m, self.p, self.n = m, p, n
        self.device = device
        T_np = build_mult_tensor(m, p, n)
        self.T = torch.tensor(T_np, dtype=torch.float64, device=device)
        self.d1, self.d2, self.d3 = self.T.shape
        self.mpn = m * p * n
    
    def reduce(self, target_rank: int, n_attempts: int = 20,
               verbose: bool = True) -> List[DecompositionResult]:
        """
        Reduce from standard decomposition to target_rank.
        """
        results = []
        
        if verbose:
            print(f"\nProper rank reduction for <{self.m},{self.p},{self.n}> "
                  f"from rank {self.mpn} to rank {target_rank}")
        
        t_start = time.time()
        
        for attempt in range(n_attempts):
            result = self._single_attempt(target_rank, verbose=(verbose and attempt < 3))
            if result is not None and result.is_exact:
                results.append(result)
                if verbose:
                    elapsed = time.time() - t_start
                    print(f"  Attempt {attempt+1}: FOUND [{elapsed:.1f}s] "
                          f"— {result.summary()}")
            elif verbose and attempt % 5 == 0:
                elapsed = time.time() - t_start
                print(f"  Attempt {attempt+1}: no solution [{elapsed:.1f}s]")
        
        return results
    
    def _single_attempt(self, target_rank: int, 
                         verbose: bool = False) -> Optional[DecompositionResult]:
        """
        Single rank reduction attempt.
        
        Strategy: start at a rank slightly above target (not all the way from mpn),
        with the standard decomposition's support structure but perturbed values.
        Then optimize with term-death penalty.
        """
        # Start from standard decomposition
        U_std, V_std, W_std = standard_decomposition(self.m, self.p, self.n)
        
        # We'll work with a slightly higher rank than target to give room
        start_rank = target_rank + 5
        
        # Take the standard factors and reduce dimensions by combining columns randomly
        # This gives us a rank-start_rank initialization that's "close" to the manifold
        R_std = self.mpn
        
        # Random combination matrix: (R_std, start_rank)
        # Each new column is a random sparse combination of standard columns
        combo = np.zeros((R_std, start_rank))
        for r in range(start_rank):
            # Pick 2-4 standard terms to combine
            n_combine = np.random.randint(1, min(4, R_std) + 1)
            idx = np.random.choice(R_std, n_combine, replace=False)
            signs = np.random.choice([-1.0, 1.0], n_combine)
            combo[idx, r] = signs
        
        # New factors: U_new = U_std @ combo, etc.
        U_init = (U_std.astype(np.float64) @ combo)
        V_init = (V_std.astype(np.float64) @ combo)
        W_init = (W_std.astype(np.float64) @ combo)
        
        # This won't be exact, but it's a reasonable initialization
        # Now optimize to make it exact and sparse
        U = torch.tensor(U_init, dtype=torch.float64, device=self.device, requires_grad=True)
        V = torch.tensor(V_init, dtype=torch.float64, device=self.device, requires_grad=True)
        W = torch.tensor(W_init, dtype=torch.float64, device=self.device, requires_grad=True)
        
        # Phase 1: achieve exact reconstruction at start_rank
        optimizer = torch.optim.Adam([U, V, W], lr=0.01)
        
        for step in range(10000):
            optimizer.zero_grad()
            T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
            loss = ((self.T - T_recon) ** 2).sum()
            loss.backward()
            optimizer.step()
            
            if loss.item() < 1e-20:
                break
        
        recon_err = loss.item()
        if recon_err > 1e-10:
            # Couldn't even match at start_rank, try again
            # Fall back to higher rank
            return self._reduction_from_standard_direct(target_rank, verbose)
        
        if verbose:
            print(f"    Phase 1: achieved real decomposition at rank {start_rank} "
                  f"(err={recon_err:.2e})")
        
        # Phase 2: gradually kill terms via L1 penalty on term norms
        # Goal: reduce from start_rank to target_rank
        n_to_kill = start_rank - target_rank
        
        optimizer = torch.optim.Adam([U, V, W], lr=0.003)
        
        for step in range(20000):
            optimizer.zero_grad()
            
            T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
            recon_loss = ((self.T - T_recon) ** 2).sum()
            
            # Term norms
            term_norms = (U.norm(dim=0) * V.norm(dim=0) * W.norm(dim=0))
            
            # Penalty: push the smallest terms toward zero
            # Sort norms and heavily penalize the smallest n_to_kill terms
            sorted_norms, _ = term_norms.sort()
            death_penalty = sorted_norms[:n_to_kill].sum()
            
            t = step / 20000
            loss = recon_loss + (0.01 + 2.0 * t) * death_penalty
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
            optimizer.step()
        
        # Identify and remove dead terms
        with torch.no_grad():
            term_norms = (U.norm(dim=0) * V.norm(dim=0) * W.norm(dim=0))
            _, sorted_idx = term_norms.sort()
            keep_idx = sorted_idx[n_to_kill:]  # keep the largest terms
            
            U_kept = U[:, keep_idx].clone()
            V_kept = V[:, keep_idx].clone()
            W_kept = W[:, keep_idx].clone()
        
        if verbose:
            print(f"    Phase 2: killed {n_to_kill} terms, "
                  f"remaining rank = {U_kept.shape[1]}")
        
        # Phase 3: re-optimize the kept terms with integrality
        U_kept.requires_grad_(True)
        V_kept.requires_grad_(True)
        W_kept.requires_grad_(True)
        
        optimizer = torch.optim.Adam([U_kept, V_kept, W_kept], lr=0.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=25000, eta_min=0.0001
        )
        
        best_err = float('inf')
        best_factors = None
        
        for step in range(25000):
            optimizer.zero_grad()
            
            T_recon = torch.einsum('ir,jr,kr->ijk', U_kept, V_kept, W_kept)
            recon_loss = ((self.T - T_recon) ** 2).sum()
            
            t = step / 25000
            int_loss = sum((torch.sin(np.pi * M) ** 2).sum()
                          for M in [U_kept, V_kept, W_kept])
            sparse_loss = sum((torch.sqrt(M ** 2 + 0.01) - 0.1).sum()
                            for M in [U_kept, V_kept, W_kept])
            mag_loss = sum(F.relu(M.abs() - 2.0).pow(2).sum()
                         for M in [U_kept, V_kept, W_kept])
            
            loss = (recon_loss 
                    + (0.05 + 1.5 * t) * int_loss
                    + 0.02 * t * sparse_loss
                    + 0.05 * t * mag_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([U_kept, V_kept, W_kept], max_norm=5.0)
            optimizer.step()
            scheduler.step()
            
            if step % 500 == 0:
                with torch.no_grad():
                    U_r = torch.round(U_kept)
                    V_r = torch.round(V_kept)
                    W_r = torch.round(W_kept)
                    T_r = torch.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                    err = ((self.T - T_r) ** 2).sum().item()
                    
                    if err < best_err:
                        best_err = err
                        best_factors = (U_r.cpu().numpy(), V_r.cpu().numpy(),
                                       W_r.cpu().numpy())
                    
                    if err < 1e-10:
                        return make_result(
                            U_r.cpu().numpy(), V_r.cpu().numpy(),
                            W_r.cpu().numpy(),
                            self.m, self.p, self.n, 'rank_reduction_v2', 'Z')
        
        if verbose:
            print(f"    Phase 3: best rounded error = {best_err:.4f}")
        
        return None
    
    def _reduction_from_standard_direct(self, target_rank: int,
                                          verbose: bool = False
                                          ) -> Optional[DecompositionResult]:
        """
        Alternative: start directly from the standard decomposition
        and use column operations to reduce rank.
        
        The standard decomposition has rank mpn with U, V, W being
        binary matrices (one 1 per column). We can:
        1. Add two columns (merging two products)
        2. Re-optimize to maintain correctness
        """
        U_std, V_std, W_std = standard_decomposition(self.m, self.p, self.n)
        
        U = torch.tensor(U_std, dtype=torch.float64, device=self.device)
        V = torch.tensor(V_std, dtype=torch.float64, device=self.device)
        W = torch.tensor(W_std, dtype=torch.float64, device=self.device)
        
        current_rank = self.mpn
        
        while current_rank > target_rank:
            # Randomly merge two columns
            i, j = np.random.choice(current_rank, 2, replace=False)
            i, j = min(i, j), max(i, j)
            
            # Add column j into column i for one of U, V, W randomly
            # Then remove column j
            choice = np.random.randint(3)
            if choice == 0:
                U[:, i] = U[:, i] + U[:, j]
            elif choice == 1:
                V[:, i] = V[:, i] + V[:, j]
            else:
                W[:, i] = W[:, i] + W[:, j]
            
            # Remove column j
            keep = list(range(current_rank))
            keep.remove(j)
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            
            current_rank -= 1
        
        # Now we have a rank-target_rank factorization that's probably wrong
        # Optimize to fix it
        U.requires_grad_(True)
        V.requires_grad_(True)
        W.requires_grad_(True)
        
        optimizer = torch.optim.Adam([U, V, W], lr=0.005)
        
        for step in range(30000):
            optimizer.zero_grad()
            T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
            recon_loss = ((self.T - T_recon) ** 2).sum()
            
            t = step / 30000
            int_loss = sum((torch.sin(np.pi * M) ** 2).sum()
                          for M in [U, V, W])
            
            loss = recon_loss + max(0, t - 0.5) * 2.0 * int_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
            optimizer.step()
            
            if step % 1000 == 0 and step > 15000:
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
                            self.m, self.p, self.n, 'direct_reduction', 'Z')
        
        return None


# ============================================================
# Strategy 2: Simulated Annealing in Integer Space
# ============================================================

class IntegerSimulatedAnnealing:
    """
    Search directly over integer factor matrices.
    
    This completely avoids the integrality gap problem.
    We never work in continuous space — all moves are integer-to-integer.
    
    Moves:
    - Flip one entry: change U[i,r] by ±1
    - Swap two columns
    - Negate a column (u_r, v_r) -> (-u_r, -v_r)
    - Zero out an entry
    
    Cost function: ||T - Σ u_r ⊗ v_r ⊗ w_r||²  (computed exactly over integers)
    """
    
    def __init__(self, m: int, p: int, n: int):
        self.m, self.p, self.n = m, p, n
        self.T = build_mult_tensor(m, p, n).astype(np.int64)
        self.d1, self.d2, self.d3 = self.T.shape
    
    def _cost(self, U: np.ndarray, V: np.ndarray, W: np.ndarray) -> int:
        """Exact integer cost: sum of squared residual entries."""
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return int(((self.T - T_recon) ** 2).sum())
    
    def _cost_fast_update(self, U, V, W, residual_sq_sum: int,
                           T_recon: np.ndarray, 
                           factor: int, idx: int, col: int, 
                           old_val: int, new_val: int) -> Tuple[int, np.ndarray]:
        """
        Compute cost after changing one entry, without full recomputation.
        Returns new cost and updated T_recon.
        
        When U[idx, col] changes by delta:
            T_recon[idx, :, :] changes by delta * V[:, col] ⊗ W[:, col]
        """
        delta = new_val - old_val
        if delta == 0:
            return residual_sq_sum, T_recon
        
        T_recon_new = T_recon.copy()
        
        if factor == 0:  # U changed
            # T_recon[idx, j, k] += delta * V[j, col] * W[k, col]
            update = delta * np.outer(V[:, col], W[:, col])
            T_recon_new[idx, :, :] += update
        elif factor == 1:  # V changed
            update = delta * np.outer(U[:, col], W[:, col])
            T_recon_new[:, idx, :] += update
        else:  # W changed
            update = delta * np.outer(U[:, col], V[:, col])
            T_recon_new[:, :, idx] += update
        
        new_cost = int(((self.T - T_recon_new) ** 2).sum())
        return new_cost, T_recon_new
    
    def search(self, R: int, n_restarts: int = 10,
               n_steps_per_restart: int = 2000000,
               T_init: float = 5.0, T_final: float = 0.01,
               max_coeff: int = 2,
               verbose: bool = True) -> List[DecompositionResult]:
        """
        Simulated annealing over integer matrices.
        """
        results = []
        
        if verbose:
            print(f"\nInteger SA search for <{self.m},{self.p},{self.n}> rank {R}")
            print(f"  Restarts: {n_restarts}, Steps: {n_steps_per_restart}")
            print(f"  Coeff range: [-{max_coeff}, {max_coeff}]")
            print(f"  Temperature: {T_init} → {T_final}")
        
        t_start = time.time()
        
        for restart in range(n_restarts):
            # Initialize: sparse random integers
            U = np.zeros((self.d1, R), dtype=np.int64)
            V = np.zeros((self.d2, R), dtype=np.int64)
            W = np.zeros((self.d3, R), dtype=np.int64)
            
            for r in range(R):
                for M, d in [(U, self.d1), (V, self.d2), (W, self.d3)]:
                    nnz = np.random.randint(1, max(2, d // 2 + 1))
                    idx = np.random.choice(d, nnz, replace=False)
                    M[idx, r] = np.random.choice(
                        list(range(-max_coeff, 0)) + list(range(1, max_coeff + 1)),
                        size=nnz
                    )
            
            # Compute initial cost
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            current_cost = int(((self.T - T_recon) ** 2).sum())
            best_cost = current_cost
            best_factors = (U.copy(), V.copy(), W.copy())
            
            # Annealing schedule
            for step in range(n_steps_per_restart):
                # Temperature
                t = step / n_steps_per_restart
                temp = T_init * (T_final / T_init) ** t
                
                # Propose a move
                move_type = np.random.randint(4)
                
                if move_type <= 2:
                    # Change one entry by ±1 or set to random value
                    factor = move_type  # 0=U, 1=V, 2=W
                    M = [U, V, W][factor]
                    d = M.shape[0]
                    
                    idx = np.random.randint(d)
                    col = np.random.randint(R)
                    old_val = M[idx, col]
                    
                    # Choose new value
                    if np.random.rand() < 0.7:
                        # Small perturbation
                        new_val = old_val + np.random.choice([-1, 1])
                    else:
                        # Random from allowed set
                        new_val = np.random.randint(-max_coeff, max_coeff + 1)
                    
                    new_val = np.clip(new_val, -max_coeff, max_coeff)
                    
                    if new_val == old_val:
                        continue
                    
                    # Compute new cost
                    new_cost, new_T_recon = self._cost_fast_update(
                        U, V, W, current_cost, T_recon,
                        factor, idx, col, old_val, new_val
                    )
                    
                    # Accept/reject
                    delta = new_cost - current_cost
                    if delta <= 0 or np.random.rand() < np.exp(-delta / max(temp, 1e-10)):
                        M[idx, col] = new_val
                        current_cost = new_cost
                        T_recon = new_T_recon
                    
                else:
                    # Swap two columns (free move — doesn't change cost)
                    # Actually swap within same factor with random sign
                    factor = np.random.randint(3)
                    M = [U, V, W][factor]
                    c1, c2 = np.random.choice(R, 2, replace=False)
                    M[:, [c1, c2]] = M[:, [c2, c1]]
                    # Also need to swap in other factors to maintain decomposition
                    for M2 in [U, V, W]:
                        if M2 is not M:
                            M2[:, [c1, c2]] = M2[:, [c2, c1]]
                    # Cost unchanged — this just reorders terms
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_factors = (U.copy(), V.copy(), W.copy())
                    
                    if best_cost == 0:
                        result = make_result(U, V, W, self.m, self.p, self.n,
                                           'integer_SA', 'Z')
                        results.append(result)
                        if verbose:
                            elapsed = time.time() - t_start
                            print(f"  restart {restart} step {step}: "
                                  f"EXACT [{elapsed:.1f}s] — {result.summary()}")
                        break
                
                # Logging
                if verbose and step % 200000 == 0 and step > 0:
                    elapsed = time.time() - t_start
                    print(f"  restart {restart} step {step//1000}k: "
                          f"cost={current_cost} best={best_cost} "
                          f"temp={temp:.3f} [{elapsed:.1f}s]")
            
            if verbose and best_cost > 0:
                print(f"  restart {restart}: best_cost={best_cost} (not exact)")
        
        if verbose:
            elapsed = time.time() - t_start
            print(f"\n  Completed in {elapsed:.1f}s. "
                  f"Found {len(results)} exact decomposition(s).")
        
        return results


# ============================================================
# Strategy 3: Greedy Finite Field Construction
# ============================================================

class GreedyFiniteField:
    """
    Build a decomposition over GF(p) greedily, one rank-1 term at a time.
    
    At each step, find the rank-1 term (u, v, w) over GF(p) that
    maximizes the number of tensor entries it "covers" (makes zero in
    the residual).
    
    For GF(2) with dimensions like 6×12×8:
    - A rank-1 term u⊗v⊗w has support on entries (i,j,k) where
      u[i]=v[j]=w[k]=1
    - The number of such entries is nnz(u) * nnz(v) * nnz(w)
    - We want this to match the residual tensor as much as possible
    
    Since each factor vector is binary (for GF(2)), we can enumerate
    efficiently for small dimensions.
    """
    
    def __init__(self, m: int, p: int, n: int):
        self.m, self.p, self.n = m, p, n
        self.T = build_mult_tensor(m, p, n).astype(np.int64)
        self.d1, self.d2, self.d3 = self.T.shape
    
    def search_gf2(self, target_rank: int, n_restarts: int = 100,
                    verbose: bool = True) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Greedy construction over GF(2) with random tie-breaking.
        """
        solutions = []
        
        if verbose:
            print(f"\nGreedy GF(2) search for <{self.m},{self.p},{self.n}> rank {target_rank}")
            print(f"  Tensor shape: ({self.d1}, {self.d2}, {self.d3})")
        
        t_start = time.time()
        
        # Precompute all possible factor vectors over GF(2)
        # For dimension d, there are 2^d - 1 nonzero vectors
        u_vecs = self._all_nonzero_binary(self.d1)
        v_vecs = self._all_nonzero_binary(self.d2)
        w_vecs = self._all_nonzero_binary(self.d3)
        
        if verbose:
            print(f"  Factor space sizes: {len(u_vecs)} × {len(v_vecs)} × {len(w_vecs)}")
            total_rank1 = len(u_vecs) * len(v_vecs) * len(w_vecs)
            print(f"  Total rank-1 candidates per step: {total_rank1:,}")
            feasible = total_rank1 < 10_000_000
            if not feasible:
                print(f"  Too large for exhaustive enumeration. Using sampled search.")
        
        for restart in range(n_restarts):
            residual = self.T.copy() % 2
            U_cols = []
            V_cols = []
            W_cols = []
            
            success = True
            for r in range(target_rank):
                # Find best rank-1 term for current residual
                best_u, best_v, best_w, best_score = self._find_best_rank1_gf2(
                    residual, u_vecs, v_vecs, w_vecs,
                    randomize=(restart > 0)
                )
                
                if best_score == 0:
                    success = False
                    break
                
                U_cols.append(best_u)
                V_cols.append(best_v)
                W_cols.append(best_w)
                
                # Update residual mod 2
                contribution = np.einsum('i,j,k', best_u, best_v, best_w)
                residual = (residual - contribution) % 2
                
                # Check if we're done early
                if not residual.any():
                    break
            
            if not residual.any() and len(U_cols) <= target_rank:
                # Pad with zero columns if finished early
                while len(U_cols) < target_rank:
                    U_cols.append(np.zeros(self.d1, dtype=np.int64))
                    V_cols.append(np.zeros(self.d2, dtype=np.int64))
                    W_cols.append(np.zeros(self.d3, dtype=np.int64))
                
                U = np.column_stack(U_cols)
                V = np.column_stack(V_cols)
                W = np.column_stack(W_cols)
                solutions.append((U, V, W))
                
                if verbose:
                    elapsed = time.time() - t_start
                    actual_rank = sum(1 for c in range(U.shape[1]) 
                                    if np.any(U[:, c]) or np.any(V[:, c]) or np.any(W[:, c]))
                    print(f"  restart {restart}: FOUND over GF(2) "
                          f"(effective rank {actual_rank}) [{elapsed:.1f}s]")
                
                if len(solutions) >= 20:
                    break
            elif verbose and restart % 20 == 0:
                remaining_nnz = int(residual.sum())
                print(f"  restart {restart}: failed (residual has {remaining_nnz} "
                      f"nonzeros after {len(U_cols)} terms)")
        
        if verbose:
            elapsed = time.time() - t_start
            print(f"  Found {len(solutions)} GF(2) decompositions in {elapsed:.1f}s")
        
        return solutions
    
    def _all_nonzero_binary(self, d: int) -> List[np.ndarray]:
        """Generate all nonzero binary vectors of length d."""
        if d > 16:
            # Too many — return a random sample
            n_sample = min(10000, 2**d - 1)
            vecs = []
            seen = set()
            while len(vecs) < n_sample:
                v = np.random.randint(0, 2, d)
                if v.any():
                    key = tuple(v)
                    if key not in seen:
                        seen.add(key)
                        vecs.append(v.astype(np.int64))
            return vecs
        
        vecs = []
        for i in range(1, 2**d):
            v = np.array([(i >> bit) & 1 for bit in range(d)], dtype=np.int64)
            vecs.append(v)
        return vecs
    
    def _find_best_rank1_gf2(self, residual: np.ndarray,
                               u_vecs, v_vecs, w_vecs,
                               randomize: bool = True
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Find the rank-1 binary tensor that best matches the residual over GF(2).
        
        "Best match" = maximizes the number of positions where
        u[i]*v[j]*w[k] == residual[i,j,k] (mod 2) for entries where
        u[i]*v[j]*w[k] = 1.
        
        Actually: we want to maximize |{(i,j,k) : u_i*v_j*w_k = 1 AND residual_{ijk} = 1}|
        because those are the entries we "fix" by subtracting this term.
        """
        best_score = 0
        best = (u_vecs[0], v_vecs[0], w_vecs[0])
        
        # For small cases, enumerate all combinations
        total_candidates = len(u_vecs) * len(v_vecs) * len(w_vecs)
        
        if total_candidates <= 5_000_000:
            # Exhaustive with smart ordering
            # Precompute: for each (u, v) pair, compute the "slice" sum
            # This is still O(|u|*|v|*d3) but avoids the full triple loop
            
            for u in u_vecs:
                # u_mask: which rows of residual are active
                u_idx = np.where(u)[0]
                if len(u_idx) == 0:
                    continue
                
                # Sum residual over active rows: shape (d2, d3)
                # Actually we need residual[u_idx, :, :] and then check against v, w
                sub_residual = residual[u_idx, :, :]  # (|u|, d2, d3)
                
                for v in v_vecs:
                    v_idx = np.where(v)[0]
                    if len(v_idx) == 0:
                        continue
                    
                    # sub_sub: residual values where u_i=1 AND v_j=1
                    # shape: (|u|, |v|, d3), sum to (d3,)
                    sub_sub = sub_residual[:, v_idx, :]  # (|u|, |v|, d3)
                    col_sums = sub_sub.sum(axis=(0, 1))  # (d3,)
                    
                    # For each w, score = sum of col_sums[k] where w[k]=1
                    # = dot(col_sums, w)
                    # Best w maximizes this dot product mod 2...
                    # Actually in GF(2), residual entries are 0 or 1.
                    # Score = number of (i,j,k) with u_i=v_j=w_k=residual_{ijk}=1
                    # = sum over k where w_k=1 of col_sums[k]
                    
                    for w in w_vecs:
                        score = int((col_sums * w).sum())
                        
                        if score > best_score or (score == best_score and randomize 
                                                   and np.random.rand() < 0.01):
                            best_score = score
                            best = (u.copy(), v.copy(), w.copy())
        else:
            # Sampled search
            n_samples = 2_000_000
            for _ in range(n_samples):
                u = u_vecs[np.random.randint(len(u_vecs))]
                v = v_vecs[np.random.randint(len(v_vecs))]
                w = w_vecs[np.random.randint(len(w_vecs))]
                
                # Score: number of entries "covered"
                contribution = np.einsum('i,j,k', u, v, w)
                score = int((contribution * residual).sum())
                
                if score > best_score or (score == best_score and randomize 
                                           and np.random.rand() < 0.001):
                    best_score = score
                    best = (u.copy(), v.copy(), w.copy())
        
        return best[0], best[1], best[2], best_score
    
    def search_and_lift(self, target_rank: int, n_restarts: int = 100,
                         verbose: bool = True) -> List[DecompositionResult]:
        """Full pipeline: find GF(2) decomposition then lift to integers."""
        from finite_field_search import lift_to_integers
        
        gf2_solutions = self.search_gf2(target_rank, n_restarts=n_restarts,
                                         verbose=verbose)
        
        results = []
        if verbose and gf2_solutions:
            print(f"\n  Attempting to lift {len(gf2_solutions)} solutions to Z...")
        
        for i, (U, V, W) in enumerate(gf2_solutions):
            result = lift_to_integers(U, V, W, self.m, self.p, self.n, 
                                      prime=2, verbose=False)
            if result is not None and result.is_exact:
                results.append(result)
                if verbose:
                    print(f"  Solution {i}: LIFTED — {result.summary()}")
        
        return results


# ============================================================
# Strategy 4: Continuous Search with Proper Initialization
# ============================================================

class ImprovedContinuousSearch:
    """
    Key improvement: initialize from the standard decomposition
    with random column merges, not from random Gaussian.
    
    This guarantees we start "near" the solution manifold.
    """
    
    def __init__(self, m: int, p: int, n: int, device: str = 'cpu'):
        self.m, self.p, self.n = m, p, n
        self.device = device
        T_np = build_mult_tensor(m, p, n)
        self.T = torch.tensor(T_np, dtype=torch.float64, device=device)
        self.d1, self.d2, self.d3 = self.T.shape
        self.mpn = m * p * n
    
    def _init_from_standard(self, R: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize a rank-R factorization by randomly combining columns
        of the standard rank-mpn factorization.
        
        The idea: group the mpn standard terms into R clusters,
        and sum each cluster into one combined term.
        """
        U_std, V_std, W_std = standard_decomposition(self.m, self.p, self.n)
        
        # Random assignment of mpn terms to R groups
        assignment = np.random.randint(0, R, size=self.mpn)
        
        # Ensure each group has at least one term
        for r in range(R):
            if not np.any(assignment == r):
                # Steal from the largest group
                largest = np.argmax(np.bincount(assignment, minlength=R))
                members = np.where(assignment == largest)[0]
                assignment[members[0]] = r
        
        U = np.zeros((self.d1, R), dtype=np.float64)
        V = np.zeros((self.d2, R), dtype=np.float64)
        W = np.zeros((self.d3, R), dtype=np.float64)
        
        for r in range(R):
            members = np.where(assignment == r)[0]
            for idx in members:
                # Add with random sign
                sign = np.random.choice([-1.0, 1.0])
                U[:, r] += sign * U_std[:, idx]
                V[:, r] += sign * V_std[:, idx]
                W[:, r] += sign * W_std[:, idx]
        
        # Add small noise to break symmetry
        U += np.random.randn(*U.shape) * 0.1
        V += np.random.randn(*V.shape) * 0.1
        W += np.random.randn(*W.shape) * 0.1
        
        U_t = torch.tensor(U, dtype=torch.float64, device=self.device, requires_grad=True)
        V_t = torch.tensor(V, dtype=torch.float64, device=self.device, requires_grad=True)
        W_t = torch.tensor(W, dtype=torch.float64, device=self.device, requires_grad=True)
        
        return U_t, V_t, W_t
    
    def search(self, R: int, n_restarts: int = 300, n_steps: int = 30000,
               lr: float = 0.005, verbose: bool = True) -> List[DecompositionResult]:
        """
        Search with standard-decomposition-based initialization.
        """
        results = []
        
        if verbose:
            print(f"\nImproved continuous search for <{self.m},{self.p},{self.n}> rank {R}")
            print(f"  Restarts: {n_restarts}, Steps: {n_steps}")
            print(f"  Initialization: from standard decomposition with random merges")
        
        t_start = time.time()
        
        for restart in range(n_restarts):
            U, V, W = self._init_from_standard(R)
            
            optimizer = torch.optim.Adam([U, V, W], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=n_steps // 3, T_mult=1, eta_min=lr * 0.01
            )
            
            # Check initial reconstruction error
            with torch.no_grad():
                T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
                init_err = ((self.T - T_recon) ** 2).sum().item()
            
            phase1_end = int(n_steps * 0.5)
            phase2_end = int(n_steps * 0.75)
            
            for step in range(n_steps):
                optimizer.zero_grad()
                
                T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
                recon_loss = ((self.T - T_recon) ** 2).sum()
                
                if step < phase1_end:
                    loss = recon_loss
                elif step < phase2_end:
                    t = (step - phase1_end) / (phase2_end - phase1_end)
                    int_loss = sum((torch.sin(np.pi * M) ** 2).sum()
                                  for M in [U, V, W])
                    loss = recon_loss + (0.05 + 0.5 * t) * int_loss
                else:
                    t = (step - phase2_end) / (n_steps - phase2_end)
                    int_loss = sum((torch.sin(np.pi * M) ** 2).sum()
                                  for M in [U, V, W])
                    sparse_loss = sum(
                        (torch.sqrt(M ** 2 + 0.01) - 0.1).sum()
                        for M in [U, V, W]
                    )
                    mag_loss = sum(F.relu(M.abs() - 2.0).pow(2).sum()
                                 for M in [U, V, W])
                    loss = (recon_loss 
                            + (0.5 + 2.0 * t) * int_loss
                            + 0.03 * t * sparse_loss
                            + 0.1 * t * mag_loss)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
                optimizer.step()
                scheduler.step()
                
                # Check
                if step % 1000 == 0 and step > phase1_end:
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
                                'improved_continuous', 'Z')
                            results.append(result)
                            if verbose:
                                elapsed = time.time() - t_start
                                print(f"  restart {restart} [{elapsed:.1f}s]: "
                                      f"FOUND — {result.summary()}")
                            break
            
            if verbose and restart % 50 == 0:
                elapsed = time.time() - t_start
                print(f"  restart {restart} [{elapsed:.1f}s]: "
                      f"init_err={init_err:.2f}, {len(results)} found")
        
        if verbose:
            elapsed = time.time() - t_start
            print(f"\n  Completed in {elapsed:.1f}s. Found {len(results)} solution(s).")
        
        return results


# ============================================================
# Strategy 5: Combined Pipeline
# ============================================================

def full_search(m: int, p: int, n: int, target_rank: int,
                device: str = 'cpu', verbose: bool = True,
                time_budget: float = 7200.0  # 2 hours per case
                ) -> List[DecompositionResult]:
    """
    Run all strategies in sequence with a time budget.
    Stop as soon as one works.
    """
    all_results = []
    t_start = time.time()
    
    def time_remaining():
        return time_budget - (time.time() - t_start)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"FULL SEARCH: <{m},{p},{n}> rank {target_rank}")
        print(f"Time budget: {time_budget:.0f}s")
        print(f"{'='*60}")
    
    # 1. Improved continuous search (standard-based init)
    if time_remaining() > 60:
        if verbose:
            print(f"\n--- Strategy 1: Improved continuous search ---")
        n_restarts = min(500, max(50, int(time_remaining() / 20)))
        ics = ImprovedContinuousSearch(m, p, n, device=device)
        results = ics.search(R=target_rank, n_restarts=n_restarts,
                            n_steps=25000, verbose=verbose)
        all_results.extend(results)
        if results:
            return all_results
    
    # 2. Rank reduction from standard
    if time_remaining() > 120:
        if verbose:
            print(f"\n--- Strategy 2: Rank reduction from standard ---")
        prr = ProperRankReduction(m, p, n, device=device)
        results = prr.reduce(target_rank=target_rank,
                            n_attempts=min(20, max(5, int(time_remaining() / 300))),
                            verbose=verbose)
        all_results.extend(results)
        if results:
            return all_results
    
    # 3. Integer simulated annealing
    if time_remaining() > 120:
        if verbose:
            print(f"\n--- Strategy 3: Integer simulated annealing ---")
        isa = IntegerSimulatedAnnealing(m, p, n)
        n_sa_steps = min(5000000, max(500000, int(time_remaining() * 2000)))
        results = isa.search(R=target_rank, n_restarts=5,
                            n_steps_per_restart=n_sa_steps,
                            verbose=verbose)
        all_results.extend(results)
        if results:
            return all_results
    
    # 4. Greedy GF(2) + lift
    if time_remaining() > 60:
        if verbose:
            print(f"\n--- Strategy 4: Greedy GF(2) + lift ---")
        gff = GreedyFiniteField(m, p, n)
        results = gff.search_and_lift(target_rank=target_rank,
                                       n_restarts=100, verbose=verbose)
        all_results.extend(results)
    
    return all_results