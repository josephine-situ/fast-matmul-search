"""
Core utilities for matrix multiplication tensor decomposition search.

The matrix multiplication tensor T for <m,p,n> encodes the bilinear map
    (m×p matrix) × (p×n matrix) → (m×n matrix)
as a 3-way tensor of shape (mp, pn, mn).

A rank-R decomposition T = Σ_r u_r ⊗ v_r ⊗ w_r corresponds to an
algorithm using R scalar multiplications (instead of the standard mpn).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict


@dataclass
class DecompositionResult:
    """Stores a found tensor decomposition with metadata."""
    U: np.ndarray          # shape (d1, R) — A-side factors
    V: np.ndarray          # shape (d2, R) — B-side factors
    W: np.ndarray          # shape (d3, R) — C-side factors
    rank: int              # R = number of scalar multiplications
    reconstruction_error: float
    m: int
    p: int
    n: int
    method: str            # how it was found
    field: str             # 'Z' for integers, 'GF2', 'GF3', 'GF5', 'R' for reals
    max_coefficient: int   # largest absolute value in factors
    num_additions: int     # total nonzeros in U, V, W minus 3R
    
    @property
    def is_exact(self):
        return self.reconstruction_error < 1e-10
    
    @property
    def omega_bound(self):
        """Upper bound on ω from this single algorithm via τ-theorem."""
        if self.m == self.p == self.n:
            return np.log(self.rank) / np.log(self.m)
        else:
            return compute_omega_single(self.m, self.p, self.n, self.rank)
    
    def summary(self):
        status = "EXACT" if self.is_exact else f"error={self.reconstruction_error:.2e}"
        return (f"<{self.m},{self.p},{self.n}> rank {self.rank} [{status}] "
                f"max_coeff={self.max_coefficient} additions={self.num_additions} "
                f"ω≤{self.omega_bound:.4f} ({self.method}, {self.field})")


def build_mult_tensor(m: int, p: int, n: int) -> np.ndarray:
    """
    Build the structure tensor for (m×p) × (p×n) matrix multiplication.
    
    T[i*p+k, k*n+j, i*n+j] = 1 encodes: C[i,j] += A[i,k] * B[k,j]
    
    Returns tensor of shape (m*p, p*n, m*n).
    """
    T = np.zeros((m * p, p * n, m * n), dtype=np.float64)
    for i in range(m):
        for k in range(p):
            for j in range(n):
                T[i * p + k, k * n + j, i * n + j] = 1.0
    return T


def verify_decomposition(T: np.ndarray, U: np.ndarray, V: np.ndarray, 
                          W: np.ndarray) -> float:
    """
    Check ||T - Σ_r u_r ⊗ v_r ⊗ w_r||_max.
    Returns 0 for an exact decomposition.
    """
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.max(np.abs(T - T_recon))


def verify_decomposition_modular(T: np.ndarray, U: np.ndarray, V: np.ndarray,
                                  W: np.ndarray, prime: int) -> bool:
    """Check decomposition over GF(prime)."""
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.all((T - T_recon) % prime == 0)


def count_additions(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> int:
    """
    Count the number of additions implied by a decomposition.
    Each rank-1 term requires: (nnz(u_r)-1) + (nnz(v_r)-1) additions
    for computing the two linear combinations, plus (nnz(w_r)) additions
    for accumulating into C entries.
    """
    R = U.shape[1]
    total = 0
    for r in range(R):
        total += max(0, np.count_nonzero(U[:, r]) - 1)  # A-side additions
        total += max(0, np.count_nonzero(V[:, r]) - 1)  # B-side additions
        total += np.count_nonzero(W[:, r])                # C-side accumulations
    return total


def make_result(U, V, W, m, p, n, method, field) -> DecompositionResult:
    """Package arrays into a DecompositionResult."""
    T = build_mult_tensor(m, p, n)
    U_int = np.round(U).astype(np.int64)
    V_int = np.round(V).astype(np.int64)
    W_int = np.round(W).astype(np.int64)
    
    error = verify_decomposition(T, U_int.astype(np.float64), 
                                  V_int.astype(np.float64), 
                                  W_int.astype(np.float64))
    
    return DecompositionResult(
        U=U_int, V=V_int, W=W_int,
        rank=U.shape[1],
        reconstruction_error=error,
        m=m, p=p, n=n,
        method=method, field=field,
        max_coefficient=max(np.max(np.abs(U_int)), np.max(np.abs(V_int)), 
                           np.max(np.abs(W_int))),
        num_additions=count_additions(U_int, V_int, W_int)
    )


def compute_omega_single(m: int, p: int, n: int, R: int) -> float:
    """
    Compute the ω bound from a single <m,p,n> algorithm with rank R.
    
    We need to find: min s+t+u  subject to  m^s * p^t * n^u <= R, s,t,u >= 0
    
    For m=p=n=k, the optimum is s=t=u and ω = 3*log(R)/log(k^3) = log_k(R).
    For rectangular cases, we solve numerically.
    """
    from scipy.optimize import minimize
    
    if R <= 0:
        return float('inf')
    
    log_m, log_p, log_n, log_R = np.log(m), np.log(p), np.log(n), np.log(R)
    
    def objective(x):
        return x[0] + x[1] + x[2]
    
    def constraint(x):
        return log_R - (x[0] * log_m + x[1] * log_p + x[2] * log_n)
    
    from scipy.optimize import minimize
    best = float('inf')
    
    # Try multiple starting points
    for s0 in [1.0, 0.5, 0.8]:
        for t0 in [1.0, 0.5, 0.8]:
            u0 = max(0, (log_R - s0 * log_m - t0 * log_p) / log_n) if log_n > 0 else 1.0
            
            result = minimize(
                objective, [s0, t0, u0],
                method='SLSQP',
                bounds=[(0, None), (0, None), (0, None)],
                constraints={'type': 'ineq', 'fun': constraint}
            )
            
            if result.success:
                best = min(best, result.fun)
    
    return best


# ============================================================
# Known best results — our reference table
# ============================================================

# Format: (m, p, n) -> (best_known_rank, source)
# Sorted by mpn for readability
# Only including cases where m <= p <= n (others follow by symmetry)

KNOWN_LOWER_BOUNDS: Dict[Tuple[int, int, int], int] = {
    (2, 2, 2): 7,
    (2, 2, 3): 11,
    (2, 2, 4): 14,
    (2, 3, 3): 15,
    (2, 3, 4): 18,
    (3, 3, 3): 19,
    (3, 3, 4): 24,
    (3, 4, 4): 28,
}

def get_lower_bound(m: int, p: int, n: int, field: str = "arbitrary") -> int:
    """Get the mathematically known lower bound for the tensor rank."""
    dims = tuple(sorted([m, p, n]))
    
    # Specific known bounds for F2
    if field.upper() == "GF2" or field.upper() == "F2":
        special_gf2 = {
            (2, 3, 4): 19,
            (3, 3, 3): 20,
            (3, 3, 4): 25,
            (3, 4, 4): 29,
        }
        if dims in special_gf2:
            return special_gf2[dims]
            
    if dims in KNOWN_LOWER_BOUNDS:
        return KNOWN_LOWER_BOUNDS[dims]
        
    # Asymptotic exact rank lower bound for square matrices (Blaser 1999)
    if dims[0] == dims[1] == dims[2]:
        m = dims[0]
        return int(np.ceil(2.5 * (m**2) - 3 * m))
    
    # Mathematical absolute minimum (cannot use fewer mults than the dimensions)
    # Actually, a simple lower bound is max(m*p, p*n, m*n).
    bound = max(dims[0]*dims[1], dims[1]*dims[2], dims[0]*dims[2])
    
    # Hopcroft-Kerr bound for <2,2,n>
    if dims[0] == 2 and dims[1] == 2:
        bound = max(bound, int(np.ceil(7 * dims[2] / 2)))
        
    return bound

KNOWN_RANKS: Dict[Tuple[int, int, int], Tuple[int, str]] = {
    # mpn = 8
    (2, 2, 2): (7, "Strassen 1969"),
    
    # mpn = 12
    (2, 2, 3): (11, "Hopcroft-Kerr 1971"),
    
    # mpn = 16
    (2, 2, 4): (14, "Hopcroft-Kerr 1971 / 2x Strassen"),  
    (2, 4, 2): (14, "symmetry of above"),
    
    # mpn = 18
    (2, 3, 3): (15, "Smirnov / Pan"),
    
    # mpn = 20
    (2, 2, 5): (18, "various"),
    
    # mpn = 24
    (2, 3, 4): (26, "Smirnov"),
    (2, 2, 6): (21, "3x Strassen"),
    
    # mpn = 27
    (3, 3, 3): (23, "Makarov-Smirnov / AlphaTensor"),
    
    # mpn = 30
    (2, 3, 5): (30, "Smirnov — possibly improvable"),
    
    # mpn = 32
    (2, 4, 4): (36, "Smirnov — needs verification"),
    (2, 2, 8): (28, "4x Strassen"),
    
    # mpn = 36
    (3, 3, 4): (29, "AlphaTensor 2022"),
    (2, 3, 6): (33, "Smirnov"),
    
    # mpn = 40  
    (2, 4, 5): (36, "Smirnov — limited prior work"),
    (2, 2, 10): (35, "Strassen recursive"),
    
    # mpn = 45
    (3, 3, 5): (38, "limited prior work"),
    
    # mpn = 48
    (3, 4, 4): (38, "limited prior work"),
    (2, 4, 6): (40, "Smirnov"),
    
    # mpn = 54
    (3, 3, 6): (40, "Smirnov"),
    
    # mpn = 60
    (3, 4, 5): (47, "limited prior work"),
    (2, 5, 6): (47, "limited prior work"),
    (2, 3, 10): (52, "limited prior work"),
    
    # mpn = 64
    (4, 4, 4): (49, "Strassen recursive; AlphaTensor got 47 over GF(2)"),
    
    # mpn = 72
    (3, 4, 6): (54, "limited prior work"),
    
    # mpn = 75
    (3, 5, 5): (55, "limited prior work"),
    
    # mpn = 80
    (4, 4, 5): (58, "limited prior work"),
}


def get_sorted_targets():
    """
    Return all known cases sorted by potential research value.
    Cases with 'limited prior work' and reasonable tensor sizes
    are the most promising targets.
    """
    targets = []
    for (m, p, n), (R, source) in KNOWN_RANKS.items():
        if m > p or p > n:
            continue  # skip duplicates from symmetry
            
        mpn = m * p * n
        tensor_size = (m*p) * (p*n) * (m*n)
        omega_current = compute_omega_single(m, p, n, R)
        omega_improved = compute_omega_single(m, p, n, R - 1)
        marginal = omega_current - omega_improved
        
        targets.append({
            'case': (m, p, n),
            'mpn': mpn,
            'tensor_shape': (m*p, p*n, m*n),
            'tensor_entries': tensor_size,
            'best_rank': R,
            'source': source,
            'omega_current': omega_current,
            'omega_if_improved': omega_improved,
            'marginal_omega': marginal,
            'limited_prior_work': 'limited' in source.lower(),
            'standard_rank': mpn,
            'savings_pct': 100 * (1 - R / mpn),
        })
    
    return targets