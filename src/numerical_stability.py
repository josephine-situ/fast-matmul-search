"""
Numerical stability analysis for found algorithms.

Two algorithms with the same rank can differ enormously in how
they amplify floating-point errors. This module quantifies that.
"""

import numpy as np
from tensor_utils import DecompositionResult


def error_amplification(result: DecompositionResult, 
                         n_trials: int = 10000,
                         condition_range: tuple = (1, 1e6)
                         ) -> dict:
    """
    Measure how the algorithm amplifies input perturbations.
    
    For each trial:
    1. Generate a random matrix pair (A, B)
    2. Compute C = A @ B via standard and via the algorithm, both in float64
    3. Compare the relative error
    
    The relative error characterizes numerical stability.
    """
    m, p, n = result.m, result.p, result.n
    R = result.rank
    U, V, W = result.U, result.V, result.W
    
    relative_errors = []
    
    for trial in range(n_trials):
        # Generate matrices with varying condition numbers
        A = np.random.randn(m, p)
        B = np.random.randn(p, n)
        
        # Scale some entries to create ill-conditioning
        scale = np.random.uniform(1, condition_range[1] ** 0.5)
        if np.random.rand() < 0.3:
            A[0, :] *= scale
            B[:, 0] *= scale
        
        C_standard = A @ B
        
        # Compute via algorithm
        C_algo = np.zeros((m, n), dtype=np.float64)
        for r in range(R):
            a_combo = 0.0
            for i in range(m):
                for k in range(p):
                    coeff = float(U[i * p + k, r])
                    if coeff != 0:
                        a_combo += coeff * A[i, k]
            
            b_combo = 0.0
            for k in range(p):
                for j in range(n):
                    coeff = float(V[k * n + j, r])
                    if coeff != 0:
                        b_combo += coeff * B[k, j]
            
            product = a_combo * b_combo
            
            for i in range(m):
                for j in range(n):
                    coeff = float(W[i * n + j, r])
                    if coeff != 0:
                        C_algo[i, j] += coeff * product
        
        # Relative error
        norm_C = np.linalg.norm(C_standard)
        if norm_C > 1e-15:
            rel_error = np.linalg.norm(C_algo - C_standard) / norm_C
            relative_errors.append(rel_error)
    
    errors = np.array(relative_errors)
    
    return {
        'mean_relative_error': np.mean(errors),
        'max_relative_error': np.max(errors),
        'median_relative_error': np.median(errors),
        'p95_relative_error': np.percentile(errors, 95),
        'p99_relative_error': np.percentile(errors, 99),
        'n_trials': n_trials,
    }


def compare_stability(results: list, n_trials: int = 10000):
    """
    Compare numerical stability across multiple algorithms for the same case.
    """
    if not results:
        return
    
    m, p, n = results[0].m, results[0].p, results[0].n
    
    print(f"\nNUMERICAL STABILITY COMPARISON for <{m},{p},{n}>")
    print(f"  {n_trials} random matrix pairs per algorithm")
    print()
    print(f"  {'Method':<25} {'Rank':>4} {'MaxCoeff':>8} "
          f"{'Mean RelErr':>12} {'Max RelErr':>12} {'P99 RelErr':>12}")
    print(f"  {'-'*85}")
    
    for r in results:
        stats = error_amplification(r, n_trials=n_trials)
        print(f"  {r.method:<25} {r.rank:>4} {r.max_coefficient:>8} "
              f"{stats['mean_relative_error']:>12.2e} "
              f"{stats['max_relative_error']:>12.2e} "
              f"{stats['p99_relative_error']:>12.2e}")


def stability_vs_standard(result: DecompositionResult, n_trials: int = 50000):
    """
    Compare the algorithm's stability against the theoretical best
    (standard multiplication), expressing the ratio.
    
    A ratio of 1.0 means equally stable.
    Strassen typically shows ratios of 2-5 for well-conditioned inputs.
    """
    m, p, n = result.m, result.p, result.n
    R = result.rank
    U, V, W = result.U, result.V, result.W
    
    # Use float32 to make errors more visible
    ratios = []
    
    for _ in range(n_trials):
        A = np.random.randn(m, p).astype(np.float32)
        B = np.random.randn(p, n).astype(np.float32)
        
        # "True" answer in float64
        C_true = (A.astype(np.float64) @ B.astype(np.float64))
        
        # Standard float32
        C_std = (A @ B).astype(np.float64)
        
        # Algorithm in float32
        C_algo = np.zeros((m, n), dtype=np.float32)
        for r in range(R):
            a_combo = np.float32(0.0)
            for i in range(m):
                for k in range(p):
                    coeff = np.float32(U[i * p + k, r])
                    if coeff != 0:
                        a_combo += coeff * A[i, k]
            
            b_combo = np.float32(0.0)
            for k in range(p):
                for j in range(n):
                    coeff = np.float32(V[k * n + j, r])
                    if coeff != 0:
                        b_combo += coeff * B[k, j]
            
            product = a_combo * b_combo
            
            for i in range(m):
                for j in range(n):
                    coeff = np.float32(W[i * n + j, r])
                    if coeff != 0:
                        C_algo[i, j] += coeff * product
        
        C_algo = C_algo.astype(np.float64)
        
        err_std = np.linalg.norm(C_std - C_true)
        err_algo = np.linalg.norm(C_algo - C_true)
        
        if err_std > 1e-15:
            ratios.append(err_algo / err_std)
    
    ratios = np.array(ratios)
    
    print(f"\n  Stability ratio (algorithm / standard multiplication):")
    print(f"    Mean:   {np.mean(ratios):.2f}x")
    print(f"    Median: {np.median(ratios):.2f}x")
    print(f"    P95:    {np.percentile(ratios, 95):.2f}x")
    print(f"    P99:    {np.percentile(ratios, 99):.2f}x")
    print(f"    (1.0 = equally stable, higher = worse)")
    
    return {
        'mean_ratio': np.mean(ratios),
        'median_ratio': np.median(ratios),
        'p95_ratio': np.percentile(ratios, 95),
        'p99_ratio': np.percentile(ratios, 99),
    }