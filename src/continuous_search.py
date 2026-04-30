"""
Continuous optimization search for tensor decompositions.

The core idea: express the decomposition problem as minimizing
    ||T - Σ_r u_r ⊗ v_r ⊗ w_r||²
with regularization that encourages integer solutions.

Key design decisions:
  - Phase 1: pure reconstruction (find the right basin)
  - Phase 2: staged integrality + sparsity penalties (crystallize to integers)
  - Multiple random restarts (the landscape is highly non-convex)
  - Track ALL solutions found (different decompositions have different properties)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from tensor_utils import (build_mult_tensor, verify_decomposition, 
                           make_result, DecompositionResult)
import time


class ContinuousSearch:
    """
    Gradient-based search for integer tensor decompositions.
    """
    
    def __init__(self, m: int, p: int, n: int, device: str = 'cpu'):
        self.m, self.p, self.n = m, p, n
        self.device = device
        
        T_np = build_mult_tensor(m, p, n)
        self.T = torch.tensor(T_np, dtype=torch.float64, device=device)
        self.d1, self.d2, self.d3 = self.T.shape
        
        # Precompute tensor norm for relative error reporting
        self.T_norm = self.T.norm().item()
        
        # Precompute unfoldings for ALS
        self.T1 = self.T.reshape(self.d1, -1)
        self.T2 = self.T.permute(1, 0, 2).reshape(self.d2, -1)
        self.T3 = self.T.permute(2, 0, 1).reshape(self.d3, -1)
    
    def _init_factors(self, R: int, method: str = 'gaussian'):
        """
        Initialize factor matrices. Good initialization matters enormously.
        """
        if method == 'gaussian':
            # Scale so that each rank-1 term contributes roughly equally
            # to matching the tensor norm
            scale = (self.T_norm / R) ** (1.0 / 3.0)
            U = torch.randn(self.d1, R, dtype=torch.float64, 
                           device=self.device) * scale * 0.3
            V = torch.randn(self.d2, R, dtype=torch.float64, 
                           device=self.device) * scale * 0.3
            W = torch.randn(self.d3, R, dtype=torch.float64, 
                           device=self.device) * scale * 0.3
            
        elif method == 'sparse':
            # Initialize with sparse factors — biased toward simple algorithms
            U = torch.zeros(self.d1, R, dtype=torch.float64, device=self.device)
            V = torch.zeros(self.d2, R, dtype=torch.float64, device=self.device)
            W = torch.zeros(self.d3, R, dtype=torch.float64, device=self.device)
            
            for r in range(R):
                # Each factor gets 1-3 nonzero entries from {-1, 0, 1}
                for M, d in [(U, self.d1), (V, self.d2), (W, self.d3)]:
                    nnz = np.random.randint(1, min(4, d + 1))
                    idx = np.random.choice(d, nnz, replace=False)
                    vals = np.random.choice([-1.0, 1.0], nnz)
                    M[idx, r] = torch.tensor(vals, dtype=torch.float64, device=self.device)
            
            # Add small noise to break symmetry
            U += torch.randn_like(U) * 0.1
            V += torch.randn_like(V) * 0.1
            W += torch.randn_like(W) * 0.1
            
        elif method == 'uniform':
            # Uniform in [-2, 2] — biased toward small integer solutions
            U = (torch.rand(self.d1, R, dtype=torch.float64, 
                           device=self.device) - 0.5) * 4
            V = (torch.rand(self.d2, R, dtype=torch.float64, 
                           device=self.device) - 0.5) * 4
            W = (torch.rand(self.d3, R, dtype=torch.float64, 
                           device=self.device) - 0.5) * 4
        
        U.requires_grad_(True)
        V.requires_grad_(True)
        W.requires_grad_(True)
        return U, V, W
    
    def _reconstruct_batched(self, U, V, W):
        """T_approx[b,i,j,k] = Σ_r U[b,i,r] V[b,j,r] W[b,k,r]"""
        return torch.einsum('bir,bjr,bkr->bijk', U, V, W)
    
    def _recon_loss_batched(self, U, V, W):
        """Squared Frobenius norm of residual per batch element."""
        residual = self.T.unsqueeze(0) - self._reconstruct_batched(U, V, W)
        return (residual ** 2).flatten(start_dim=1).sum(dim=1)
    
    def _integrality_loss_batched(self, U, V, W):
        loss = 0.0
        for M in [U, V, W]:
            loss = loss + (torch.sin(np.pi * M) ** 2).flatten(start_dim=1).sum(dim=1)
        return loss

    def _sparsity_loss_batched(self, U, V, W):
        eps = 1e-4
        loss = 0.0
        for M in [U, V, W]:
            loss = loss + (torch.sqrt(M ** 2 + eps ** 2) - eps).flatten(start_dim=1).sum(dim=1)
        return loss

    def _magnitude_loss_batched(self, U, V, W):
        loss = 0.0
        for M in [U, V, W]:
            loss = loss + (F.relu(M.abs() - 2.0) ** 2).flatten(start_dim=1).sum(dim=1)
        return loss

    def _balance_loss_batched(self, U, V, W):
        normU = (U ** 2).sum(dim=1)  # (B, R)
        normV = (V ** 2).sum(dim=1)
        normW = (W ** 2).sum(dim=1)
        gmean = (normU * normV * normW).clamp(min=1e-8) ** (1.0 / 3.0)
        
        loss = ((normU / gmean - 1) ** 2).sum(dim=1) + \
               ((normV / gmean - 1) ** 2).sum(dim=1) + \
               ((normW / gmean - 1) ** 2).sum(dim=1)
        return loss

    def _reconstruct(self, U, V, W):
        """T_approx[i,j,k] = Σ_r U[i,r] V[j,r] W[k,r]"""
        return torch.einsum('ir,jr,kr->ijk', U, V, W)
    
    def _recon_loss(self, U, V, W):
        """Squared Frobenius norm of residual."""
        residual = self.T - self._reconstruct(U, V, W)
        return (residual ** 2).sum()
    
    def _integrality_loss(self, U, V, W):
        """
        sin²(πx) is zero at all integers, smooth, and periodic.
        This gently pushes all entries toward the nearest integer
        without creating hard boundaries.
        """
        loss = 0.0
        for M in [U, V, W]:
            loss = loss + (torch.sin(np.pi * M) ** 2).sum()
        return loss
    
    def _sparsity_loss(self, U, V, W):
        """
        Smooth L1 penalty: encourages zeros in the factors.
        Sparser factors = fewer additions = simpler algorithm.
        """
        eps = 0.05
        loss = 0.0
        for M in [U, V, W]:
            loss = loss + (torch.sqrt(M ** 2 + eps ** 2) - eps).sum()
        return loss
    
    def _magnitude_loss(self, U, V, W):
        """
        Penalize large coefficients. Practical algorithms use
        small integers. Soft penalty beyond magnitude 2.
        """
        loss = 0.0
        for M in [U, V, W]:
            loss = loss + (F.relu(M.abs() - 2.0) ** 2).sum()
        return loss
    
    def _balance_loss(self, U, V, W):
        """
        Encourage the three factors to have similar norms.
        Prevents degenerate solutions where one factor is huge
        and another is tiny (which hurts rounding).
        """
        nu = U.norm(dim=0)  # (R,)
        nv = V.norm(dim=0)
        nw = W.norm(dim=0)
        
        # Geometric mean per component
        gmean = (nu * nv * nw) ** (1.0 / 3.0) + 1e-10
        
        loss = ((nu / gmean - 1) ** 2).sum() + \
               ((nv / gmean - 1) ** 2).sum() + \
               ((nw / gmean - 1) ** 2).sum()
        return loss
    
    def search_batched(self, R: int, B: int, n_steps: int = 20000,
                       lr: float = 0.003, verbose: bool = False) -> List[DecompositionResult]:
        """
        Batched optimization run solving B restarts simultaneously.
        Dramatically reduces overhead.
        """
        # We manually initialize batch. We can vary initialization method per batch element.
        U = torch.zeros(B, self.d1, R, dtype=torch.float64, device=self.device)
        V = torch.zeros(B, self.d2, R, dtype=torch.float64, device=self.device)
        W = torch.zeros(B, self.d3, R, dtype=torch.float64, device=self.device)
        methods = ['gaussian', 'sparse', 'uniform']
        
        for b in range(B):
            u_b, v_b, w_b = self._init_factors(R, method=methods[b % len(methods)])
            U[b] = u_b.detach()
            V[b] = v_b.detach()
            W[b] = w_b.detach()
        
        U.requires_grad_(True)
        V.requires_grad_(True)
        W.requires_grad_(True)

        optimizer = torch.optim.Adam([U, V, W], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=n_steps // 4, T_mult=1, eta_min=lr * 0.01
        )
        
        phase1_end = int(n_steps * 0.4)
        phase2_end = int(n_steps * 0.7)
        
        best_rounded_error = torch.full((B,), float('inf'), device=self.device, dtype=torch.float64)
        best_U = U.detach().clone()
        best_V = V.detach().clone()
        best_W = W.detach().clone()
        
        exact_masks = torch.zeros(B, dtype=torch.bool, device=self.device)
        results = []
        
        for step in range(n_steps):
            if exact_masks.all():
                break # All found
            
            optimizer.zero_grad()
            
            recon = self._recon_loss_batched(U, V, W)
            loss_batch = recon.clone()
            
            if step < phase1_end:
                loss_batch += 0.01 * self._balance_loss_batched(U, V, W)
            elif step < phase2_end:
                t = (step - phase1_end) / (phase2_end - phase1_end)
                int_w = 0.3 * t ** 2
                sparse_w = 0.05 * t
                mag_w = 0.1 * t
                loss_batch += (int_w * self._integrality_loss_batched(U, V, W) +
                               sparse_w * self._sparsity_loss_batched(U, V, W) +
                               mag_w * self._magnitude_loss_batched(U, V, W) +
                               0.01 * self._balance_loss_batched(U, V, W))
            else:
                t = (step - phase2_end) / (n_steps - phase2_end)
                int_w = 0.3 + 2.0 * t
                sparse_w = 0.05 + 0.2 * t
                mag_w = 0.1 + 0.5 * t
                loss_batch += (int_w * self._integrality_loss_batched(U, V, W) +
                               sparse_w * self._sparsity_loss_batched(U, V, W) +
                               mag_w * self._magnitude_loss_batched(U, V, W))
            
            # Mask out already converged gradients
            loss = (loss_batch * (~exact_masks)).sum()
            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
                optimizer.step()
                scheduler.step()
            
            if step % 1000 == 0 or step == n_steps - 1:
                with torch.no_grad():
                    U_r = torch.round(U)
                    V_r = torch.round(V)
                    W_r = torch.round(W)
                    rounded_errs = self._recon_loss_batched(U_r, V_r, W_r)
                    
                    improved = rounded_errs < best_rounded_error
                    if improved.any():
                        best_rounded_error[improved] = rounded_errs[improved]
                        best_U[improved] = U_r[improved].detach()
                        best_V[improved] = V_r[improved].detach()
                        best_W[improved] = W_r[improved].detach()
                    
                    newly_exact = (best_rounded_error < 1e-10) & (~exact_masks)
                    if newly_exact.any():
                        idx = newly_exact.nonzero(as_tuple=True)[0]
                        for i in idx.tolist():
                            if verbose:
                                print(f"  EXACT in batch element {i} at step {step}!")
                            results.append(make_result(best_U[i].cpu().numpy(), 
                                                       best_V[i].cpu().numpy(), 
                                                       best_W[i].cpu().numpy(),
                                                       self.m, self.p, self.n,
                                                       'gradient', 'Z'))
                        exact_masks |= newly_exact
                        
        # Now try to snap and refine those that are close but not exact
        not_exact = (~exact_masks).nonzero(as_tuple=True)[0].tolist()
        for b in not_exact:
            result = self._snap_and_refine(U[b].detach(), V[b].detach(), W[b].detach())
            if result is not None:
                results.append(result)
            elif best_rounded_error[b] < 1e-10:
                results.append(make_result(best_U[b].cpu().numpy(), 
                                           best_V[b].cpu().numpy(), 
                                           best_W[b].cpu().numpy(),
                                           self.m, self.p, self.n,
                                           'gradient', 'Z'))
        return results

    def search_single(self, R: int, n_steps: int = 20000,
                       lr: float = 0.003, init_method: str = 'gaussian',
                       verbose: bool = False) -> Optional[DecompositionResult]:
        """
        Single optimization run. Returns a DecompositionResult if an
        exact integer decomposition is found, None otherwise.
        """
        U, V, W = self._init_factors(R, method=init_method)
        
        optimizer = torch.optim.Adam([U, V, W], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=n_steps // 4, T_mult=1, eta_min=lr * 0.01
        )
        
        # Phase boundaries
        phase1_end = int(n_steps * 0.4)   # pure reconstruction
        phase2_end = int(n_steps * 0.7)   # add integrality + sparsity
        # phase 3: strong integrality + reduce lr
        
        best_rounded_error = float('inf')
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Reconstruction loss — always present
            recon = self._recon_loss(U, V, W)
            loss = recon
            
            if step < phase1_end:
                # Phase 1: pure reconstruction + mild balance
                loss = loss + 0.01 * self._balance_loss(U, V, W)
                
            elif step < phase2_end:
                # Phase 2: gradually introduce integrality
                t = (step - phase1_end) / (phase2_end - phase1_end)
                int_w = 0.3 * t ** 2
                sparse_w = 0.05 * t
                mag_w = 0.1 * t
                
                loss = (loss 
                        + int_w * self._integrality_loss(U, V, W)
                        + sparse_w * self._sparsity_loss(U, V, W)
                        + mag_w * self._magnitude_loss(U, V, W)
                        + 0.01 * self._balance_loss(U, V, W))
            else:
                # Phase 3: strong integrality pressure
                t = (step - phase2_end) / (n_steps - phase2_end)
                int_w = 0.3 + 2.0 * t
                sparse_w = 0.05 + 0.2 * t
                mag_w = 0.1 + 0.5 * t
                
                loss = (loss
                        + int_w * self._integrality_loss(U, V, W)
                        + sparse_w * self._sparsity_loss(U, V, W)
                        + mag_w * self._magnitude_loss(U, V, W))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
            optimizer.step()
            scheduler.step()
            
            # Periodically check rounded solution
            if step % 1000 == 0 or step == n_steps - 1:
                with torch.no_grad():
                    U_r = torch.round(U)
                    V_r = torch.round(V)
                    W_r = torch.round(W)
                    rounded_error = self._recon_loss(U_r, V_r, W_r).item()
                    
                    if rounded_error < best_rounded_error:
                        best_rounded_error = rounded_error
                        best_U = U_r.cpu().numpy()
                        best_V = V_r.cpu().numpy()
                        best_W = W_r.cpu().numpy()
                    
                    if verbose and step % 5000 == 0:
                        print(f"  step {step:>6d}: recon={recon.item():.6f} "
                              f"rounded_err={rounded_error:.6f} "
                              f"best_rounded={best_rounded_error:.6f}")
                    
                    if rounded_error < 1e-10:
                        if verbose:
                            print(f"  EXACT at step {step}!")
                        return make_result(best_U, best_V, best_W,
                                          self.m, self.p, self.n,
                                          'gradient', 'Z')
        
        # Final: try snapping and re-optimizing unfixed entries
        result = self._snap_and_refine(U.detach(), V.detach(), W.detach())
        if result is not None:
            return result
        
        # Return best rounded if close enough
        if best_rounded_error < 1e-10:
            return make_result(best_U, best_V, best_W,
                              self.m, self.p, self.n, 'gradient', 'Z')
        
        return None
    
    def _snap_and_refine(self, U: torch.Tensor, V: torch.Tensor,
                        W: torch.Tensor, n_steps: int = 5000
                        ) -> Optional[DecompositionResult]:
        """
        Post-processing: snap near-integer entries and re-optimize
        the remaining entries with the snapped ones fixed.

        The key difficulty: we need the computation graph to flow
        through the free (non-snapped) parameters. We achieve this
        by storing fixed values as constants and free values as
        Parameters, then reconstructing full matrices each step.
        """
        threshold = 0.15

        # Identify which entries are close to integers
        factors = [U.clone(), V.clone(), W.clone()]
        fixed_masks = []
        fixed_vals = []
        free_vals_list = []

        for M in factors:
            near_int = (M - M.round()).abs() < threshold
            fixed_masks.append(near_int)
            fixed_vals.append(M.round().clone())  # integer values for fixed entries

            # Extract free (non-snapped) values as optimizable parameters
            if (~near_int).any():
                fv = M[~near_int].clone().detach().requires_grad_(True)
            else:
                fv = None
            free_vals_list.append(fv)

        # Check if everything got snapped
        if all(fv is None for fv in free_vals_list):
            U_r = fixed_vals[0]
            V_r = fixed_vals[1]
            W_r = fixed_vals[2]
            error = self._recon_loss(U_r, V_r, W_r).item()
            if error < 1e-10:
                return make_result(U_r.cpu().numpy(), V_r.cpu().numpy(),
                                W_r.cpu().numpy(), self.m, self.p, self.n,
                                'gradient+snap', 'Z')
            return None

        # Collect optimizable parameters
        opt_params = [fv for fv in free_vals_list if fv is not None]
        optimizer = torch.optim.Adam(opt_params, lr=0.001)

        def build_matrix(fixed_val, fixed_mask, free_val):
            """
            Reconstruct full matrix from fixed and free parts.
            This preserves the computation graph through free_val.
            """
            M = fixed_val.clone()
            if free_val is not None:
                M[~fixed_mask] = free_val
            return M

        for step in range(n_steps):
            optimizer.zero_grad()

            # Reconstruct full matrices (differentiable through free_vals)
            U_full = build_matrix(fixed_vals[0], fixed_masks[0], free_vals_list[0])
            V_full = build_matrix(fixed_vals[1], fixed_masks[1], free_vals_list[1])
            W_full = build_matrix(fixed_vals[2], fixed_masks[2], free_vals_list[2])

            recon = self._recon_loss(U_full, V_full, W_full)
            int_loss = self._integrality_loss(U_full, V_full, W_full)

            t = step / n_steps
            loss = recon + (0.1 + 2.0 * t) * int_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=5.0)
            optimizer.step()

            if step % 500 == 0:
                with torch.no_grad():
                    U_r = build_matrix(fixed_vals[0], fixed_masks[0], 
                                    free_vals_list[0]).round()
                    V_r = build_matrix(fixed_vals[1], fixed_masks[1], 
                                    free_vals_list[1]).round()
                    W_r = build_matrix(fixed_vals[2], fixed_masks[2], 
                                    free_vals_list[2]).round()
                    err = self._recon_loss(U_r, V_r, W_r).item()

                    if err < 1e-10:
                        return make_result(
                            U_r.cpu().numpy(), V_r.cpu().numpy(),
                            W_r.cpu().numpy(), self.m, self.p, self.n,
                            'gradient+snap', 'Z')

        return None
    
    def search(self, R: int, n_restarts: int = 200, n_steps: int = 20000,
               lr: float = 0.003, verbose: bool = True
               ) -> List[DecompositionResult]:
        """
        Main search: run many random restarts in batches.
        Returns ALL exact decompositions found.
        """
        results = []
        batch_size = min(30, n_restarts)
        
        if verbose:
            print(f"\nSearching for <{self.m},{self.p},{self.n}> "
                  f"rank-{R} decomposition")
            print(f"  Tensor shape: {self.T.shape}")
            print(f"  Standard rank: {self.m * self.p * self.n}")
            print(f"  Restarts: {n_restarts}, Steps/restart: {n_steps}, Batch Size: {batch_size}")
            print()
            
        t_start = time.time()
        
        restarts_done = 0
        while restarts_done < n_restarts:
            B = min(batch_size, n_restarts - restarts_done)
            
            batched_results = self.search_batched(
                R, B=B, n_steps=n_steps, lr=lr, verbose=False
            )
            
            for res in batched_results:
                if res.is_exact:
                    results.append(res)
                    if verbose:
                        elapsed = time.time() - t_start
                        print(f"  Found exact solution! [{elapsed:>7.1f}s]: "
                              f"FOUND — {res.summary()}")
            
            restarts_done += B
            if verbose:
                elapsed = time.time() - t_start
                print(f"  Completed {restarts_done}/{n_restarts} restarts [{elapsed:>7.1f}s] "
                      f"({len(results)} found so far)")
                      
        elapsed = time.time() - t_start
        if verbose:
            print(f"\nCompleted in {elapsed:.1f}s. "
                  f"Total exact solutions found: {len(results)}")
            
        return results


class ALSSearch:
    """
    Alternating Least Squares search for tensor decompositions.
    
    ALS fixes two factors and solves for the third as a linear
    least squares problem. Each sub-problem has a closed-form solution.
    Often converges faster than gradient descent for tensor problems.
    """
    
    def __init__(self, m: int, p: int, n: int):
        self.m, self.p, self.n = m, p, n
        self.T = build_mult_tensor(m, p, n)
        self.d1, self.d2, self.d3 = self.T.shape
        
        # Precompute unfoldings
        self.T1 = self.T.reshape(self.d1, -1)                            # (d1, d2*d3)
        self.T2 = self.T.transpose(1, 0, 2).reshape(self.d2, -1)        # (d2, d1*d3)
        self.T3 = self.T.transpose(2, 0, 1).reshape(self.d3, -1)        # (d3, d1*d2)
    
    def _khatri_rao(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Khatri-Rao product (column-wise Kronecker).
        A: (I, R), B: (J, R) -> result: (I*J, R)
        """
        I, R = A.shape
        J = B.shape[0]
        result = np.zeros((I * J, R))
        for r in range(R):
            result[:, r] = np.kron(A[:, r], B[:, r])
        return result
    
    def search_single(self, R: int, n_steps: int = 3000,
                       progressive_round: bool = True
                       ) -> Optional[DecompositionResult]:
        """Single ALS run."""
        
        # Initialize
        scale = (np.linalg.norm(self.T) / R) ** (1./3.)
        U = np.random.randn(self.d1, R) * scale * 0.3
        V = np.random.randn(self.d2, R) * scale * 0.3
        W = np.random.randn(self.d3, R) * scale * 0.3
        
        for step in range(n_steps):
            # Solve for U: T1 ≈ U @ KR(W, V)^T
            KR_WV = self._khatri_rao(W, V)  # (d3*d2, R)
            U, _, _, _ = np.linalg.lstsq(KR_WV, self.T1.T, rcond=None)
            U = U.T  # (d1, R)
            
            # Solve for V: T2 ≈ V @ KR(W, U)^T
            KR_WU = self._khatri_rao(W, U)
            V, _, _, _ = np.linalg.lstsq(KR_WU, self.T2.T, rcond=None)
            V = V.T
            
            # Solve for W: T3 ≈ W @ KR(V, U)^T
            KR_VU = self._khatri_rao(V, U)
            W, _, _, _ = np.linalg.lstsq(KR_VU, self.T3.T, rcond=None)
            W = W.T
            
            # Progressive rounding in later iterations
            if progressive_round and step > n_steps * 0.6:
                threshold = 0.2 * (1.0 - step / n_steps)
                for M in [U, V, W]:
                    close = np.abs(M - np.round(M)) < threshold
                    M[close] = np.round(M[close])
        
        # Final rounding
        U_r = np.round(U).astype(np.int64)
        V_r = np.round(V).astype(np.int64)
        W_r = np.round(W).astype(np.int64)
        
        error = verify_decomposition(self.T, U_r.astype(np.float64),
                                      V_r.astype(np.float64),
                                      W_r.astype(np.float64))
        
        if error < 1e-10:
            return make_result(U_r, V_r, W_r, self.m, self.p, self.n,
                              'ALS', 'Z')
        return None
    
    def search(self, R: int, n_restarts: int = 500, n_steps: int = 3000,
               verbose: bool = True) -> List[DecompositionResult]:
        """ALS search with random restarts."""
        results = []
        
        if verbose:
            print(f"\nALS search for <{self.m},{self.p},{self.n}> rank {R}")
        
        t_start = time.time()
        
        for restart in range(n_restarts):
            result = self.search_single(R, n_steps=n_steps)
            
            if result is not None and result.is_exact:
                results.append(result)
                if verbose:
                    elapsed = time.time() - t_start
                    print(f"  restart {restart:>4d} [{elapsed:>7.1f}s]: "
                          f"FOUND — {result.summary()}")
            elif verbose and restart % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  restart {restart:>4d} [{elapsed:>7.1f}s]: "
                      f"({len(results)} found)")
        
        return results