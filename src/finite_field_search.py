"""
Search for tensor decompositions over finite fields, then lift to integers.

Key insight: the search space over GF(p) is finite, so we can use
combinatorial methods. A decomposition over GF(p) can often be lifted
to one over Z, giving a practical algorithm.

Strategy:
  1. Random search over GF(p) for p in {2, 3, 5}
  2. For each solution found, attempt to lift to integers
  3. Lifting uses: direct sign adjustment, then gradient refinement
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from tensor_utils import (build_mult_tensor, verify_decomposition,
                           verify_decomposition_modular, make_result,
                           DecompositionResult)
import time


def search_gf(m: int, p_dim: int, n: int, target_rank: int, 
              prime: int, n_attempts: int = 2000000,
              verbose: bool = True) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Random search for decompositions over GF(prime).
    
    Each attempt generates random U, V, W over GF(prime) and checks
    if they form a valid decomposition.
    
    For very small tensors and small primes, this has a nonzero
    (though small) hit rate.
    """
    T = build_mult_tensor(m, p_dim, n)
    T_mod = T.astype(np.int64) % prime
    d1, d2, d3 = T.shape
    R = target_rank
    
    found = []
    t_start = time.time()
    
    if verbose:
        print(f"\nGF({prime}) random search for <{m},{p_dim},{n}> rank {R}")
        print(f"  Tensor shape: ({d1}, {d2}, {d3})")
        search_space = prime ** (R * (d1 + d2 + d3))
        print(f"  Search space size: {prime}^{R*(d1+d2+d3)} ≈ 10^{np.log10(search_space):.0f}")
        print(f"  Attempts: {n_attempts}")
    
    for attempt in range(n_attempts):
        U = np.random.randint(0, prime, size=(d1, R))
        V = np.random.randint(0, prime, size=(d2, R))
        W = np.random.randint(0, prime, size=(d3, R))
        
        # Fast check: reconstruct and compare mod prime
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W) % prime
        
        if np.array_equal(T_recon, T_mod):
            found.append((U.copy(), V.copy(), W.copy()))
            if verbose:
                elapsed = time.time() - t_start
                print(f"  attempt {attempt:>8d} [{elapsed:>7.1f}s]: "
                      f"FOUND over GF({prime})! ({len(found)} total)")
            
            if len(found) >= 20:
                break
        
        if verbose and attempt % 500000 == 0 and attempt > 0:
            elapsed = time.time() - t_start
            rate = attempt / elapsed
            print(f"  attempt {attempt:>8d} [{elapsed:>7.1f}s]: "
                  f"{rate:.0f} attempts/sec, {len(found)} found")
    
    if verbose:
        elapsed = time.time() - t_start
        print(f"  Completed in {elapsed:.1f}s. Found {len(found)} solution(s).")
    
    return found


def search_gf_structured(m: int, p_dim: int, n: int, target_rank: int,
                          prime: int, n_attempts: int = 500000,
                          verbose: bool = True
                          ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Smarter GF search: instead of fully random, bias toward
    sparse factors (which correspond to simpler, more liftable algorithms).
    
    For each rank-1 term, randomly decide a sparsity level, then
    randomly place nonzeros.
    """
    T = build_mult_tensor(m, p_dim, n)
    T_mod = T.astype(np.int64) % prime
    d1, d2, d3 = T.shape
    R = target_rank
    
    found = []
    t_start = time.time()
    
    if verbose:
        print(f"\nGF({prime}) structured search for <{m},{p_dim},{n}> rank {R}")
    
    for attempt in range(n_attempts):
        U = np.zeros((d1, R), dtype=np.int64)
        V = np.zeros((d2, R), dtype=np.int64)
        W = np.zeros((d3, R), dtype=np.int64)
        
        for r in range(R):
            # Sparse: each factor vector gets 1 to d//2+1 nonzeros
            for M, d in [(U, d1), (V, d2), (W, d3)]:
                max_nnz = max(2, d // 2 + 1)
                nnz = np.random.randint(1, max_nnz + 1)
                idx = np.random.choice(d, nnz, replace=False)
                M[idx, r] = np.random.randint(1, prime, size=nnz)
        
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W) % prime
        
        if np.array_equal(T_recon, T_mod):
            found.append((U.copy(), V.copy(), W.copy()))
            if verbose:
                elapsed = time.time() - t_start
                print(f"  attempt {attempt:>8d} [{elapsed:>7.1f}s]: "
                      f"FOUND (sparse) over GF({prime})!")
            
            if len(found) >= 20:
                break
    
    return found


def lift_to_integers(U_mod: np.ndarray, V_mod: np.ndarray, W_mod: np.ndarray,
                     m: int, p_dim: int, n: int, prime: int,
                     verbose: bool = False) -> Optional[DecompositionResult]:
    """
    Attempt to lift a GF(prime) decomposition to integers.
    
    Step 1: Adjust signs. Over GF(p), the value (p-1) often corresponds
            to -1 over Z. Try all reasonable sign patterns.
    Step 2: If direct lift works, done.
    Step 3: Otherwise, use gradient refinement starting from the lift.
    """
    T = build_mult_tensor(m, p_dim, n)
    R = U_mod.shape[1]
    
    # Step 1: Generate candidate integer lifts
    # For each entry x in GF(p), possible integer values are:
    # x, x-p, x+p, x-2p, x+2p, ...
    # For small coefficients, we try values in [-3, 3]
    
    def lift_candidates(val, prime):
        """Integer values that reduce to val mod prime."""
        candidates = []
        for c in range(-3, 4):
            if c % prime == val % prime:
                candidates.append(c)
        return candidates
    
    # First try: the simplest lift (adjust p-1 -> -1, etc.)
    def simple_lift(M_mod, prime):
        M = M_mod.astype(np.int64).copy()
        # Map values > p//2 to negative
        M[M > prime // 2] -= prime
        return M
    
    U_int = simple_lift(U_mod, prime)
    V_int = simple_lift(V_mod, prime)
    W_int = simple_lift(W_mod, prime)
    
    error = verify_decomposition(T, U_int.astype(np.float64),
                                  V_int.astype(np.float64),
                                  W_int.astype(np.float64))
    
    if error < 1e-10:
        if verbose:
            print(f"    Direct lift successful!")
        return make_result(U_int, V_int, W_int, m, p_dim, n,
                          f'GF({prime})+lift', 'Z')
    
    # Step 2: Try a few random sign-flip patterns
    # For each factor column, we can independently flip signs
    # (multiply u_r by -1 and w_r by -1, or v_r by -1 and w_r by -1)
    for trial in range(100):
        U_try = U_int.copy()
        V_try = V_int.copy()
        W_try = W_int.copy()
        
        for r in range(R):
            # Randomly flip some entries by ±prime
            for M in [U_try, V_try, W_try]:
                for idx in range(M.shape[0]):
                    if np.random.rand() < 0.3 and M[idx, r] != 0:
                        # Try shifting by ±prime
                        shift = np.random.choice([-prime, 0, prime])
                        M[idx, r] += shift
        
        error = verify_decomposition(T, U_try.astype(np.float64),
                                      V_try.astype(np.float64),
                                      W_try.astype(np.float64))
        
        if error < 1e-10:
            if verbose:
                print(f"    Lift with sign adjustment successful (trial {trial})!")
            return make_result(U_try, V_try, W_try, m, p_dim, n,
                              f'GF({prime})+signflip', 'Z')
    
    # Step 3: Gradient refinement from the lift
    return _gradient_refine_lift(U_int, V_int, W_int, T, m, p_dim, n, 
                                  prime, verbose)


def _gradient_refine_lift(U_init: np.ndarray, V_init: np.ndarray,
                           W_init: np.ndarray, T: np.ndarray,
                           m: int, p_dim: int, n: int, prime: int,
                           verbose: bool = False
                           ) -> Optional[DecompositionResult]:
    """
    Gradient refinement starting from a finite field lift.
    """
    try:
        import torch
    except ImportError:
        return None

    T_torch = torch.tensor(T, dtype=torch.float64)

    # These are the optimizable parameters
    U = torch.tensor(U_init, dtype=torch.float64, requires_grad=True)
    V = torch.tensor(V_init, dtype=torch.float64, requires_grad=True)
    W = torch.tensor(W_init, dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.Adam([U, V, W], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5000, eta_min=1e-4
    )

    for step in range(5000):
        optimizer.zero_grad()

        T_recon = torch.einsum('ir,jr,kr->ijk', U, V, W)
        recon_loss = ((T_torch - T_recon) ** 2).sum()

        # Strong integrality from the start (we're already near integers)
        int_loss = ((torch.sin(np.pi * U) ** 2).sum() +
                    (torch.sin(np.pi * V) ** 2).sum() +
                    (torch.sin(np.pi * W) ** 2).sum())

        t = step / 5000
        loss = recon_loss + (0.3 + 1.5 * t) * int_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_([U, V, W], max_norm=5.0)
        optimizer.step()
        scheduler.step()

        if step % 500 == 0:
            with torch.no_grad():
                U_r = torch.round(U)
                V_r = torch.round(V)
                W_r = torch.round(W)
                T_check = torch.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                err = ((T_torch - T_check) ** 2).sum().item()

                if err < 1e-10:
                    if verbose:
                        print(f"    Gradient refinement succeeded at step {step}!")
                    return make_result(
                        U_r.numpy(), V_r.numpy(), W_r.numpy(),
                        m, p_dim, n, f'GF({prime})+gradient', 'Z')

    return None


def search_all_fields(m: int, p_dim: int, n: int, target_rank: int,
                       primes: List[int] = [2, 3, 5],
                       n_attempts_per_prime: int = 1000000,
                       verbose: bool = True) -> List[DecompositionResult]:
    """
    Search over multiple finite fields and attempt to lift each solution.
    """
    all_results = []
    
    for prime in primes:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Searching GF({prime}) for <{m},{p_dim},{n}> rank {target_rank}")
            print(f"{'='*60}")
        
        # Try both random and structured search
        gf_solutions = search_gf(m, p_dim, n, target_rank, prime,
                                  n_attempts=n_attempts_per_prime // 2,
                                  verbose=verbose)
        gf_solutions += search_gf_structured(m, p_dim, n, target_rank, prime,
                                              n_attempts=n_attempts_per_prime // 2,
                                              verbose=verbose)
        
        if verbose:
            print(f"\n  Attempting to lift {len(gf_solutions)} "
                  f"GF({prime}) solutions to Z...")
        
        for i, (U, V, W) in enumerate(gf_solutions):
            result = lift_to_integers(U, V, W, m, p_dim, n, prime,
                                      verbose=verbose)
            if result is not None and result.is_exact:
                all_results.append(result)
                if verbose:
                    print(f"  Solution {i}: LIFTED — {result.summary()}")
    
    return all_results