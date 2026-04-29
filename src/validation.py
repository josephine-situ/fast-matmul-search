"""
Rigorous verification of found decompositions.

A decomposition is useless if it's wrong. This module provides
multiple independent verification methods to ensure correctness.
"""

import numpy as np
from fractions import Fraction
from typing import Tuple
from tensor_utils import build_mult_tensor, DecompositionResult


def verify_exact_integer(result: DecompositionResult) -> bool:
    """
    Verify using exact integer arithmetic (no floating point).
    This is the gold standard — if this passes, the decomposition
    is mathematically correct.
    """
    m, p, n = result.m, result.p, result.n
    T = build_mult_tensor(m, p, n).astype(np.int64)
    
    U = result.U.astype(np.int64)
    V = result.V.astype(np.int64)
    W = result.W.astype(np.int64)
    
    # Reconstruct using integer arithmetic
    d1, d2, d3 = T.shape
    R = result.rank
    
    T_recon = np.zeros((d1, d2, d3), dtype=np.int64)
    for r in range(R):
        for i in range(d1):
            if U[i, r] == 0:
                continue
            for j in range(d2):
                if V[j, r] == 0:
                    continue
                for k in range(d3):
                    if W[k, r] == 0:
                        continue
                    T_recon[i, j, k] += U[i, r] * V[j, r] * W[k, r]
    
    return np.array_equal(T, T_recon)


def verify_by_random_matrices(result: DecompositionResult, 
                               n_tests: int = 10000,
                               use_integers: bool = True) -> Tuple[bool, float]:
    """
    Verify by applying the algorithm to random matrices and comparing
    against standard multiplication.
    
    If use_integers=True, tests with random integer matrices to avoid
    any floating point ambiguity.
    """
    m, p, n = result.m, result.p, result.n
    R = result.rank
    U, V, W = result.U, result.V, result.W
    
    max_error = 0.0
    
    for _ in range(n_tests):
        if use_integers:
            A = np.random.randint(-10, 11, size=(m, p))
            B = np.random.randint(-10, 11, size=(p, n))
        else:
            A = np.random.randn(m, p)
            B = np.random.randn(p, n)
        
        # Standard multiplication
        C_expected = A @ B
        
        # Algorithm multiplication
        C_algo = np.zeros((m, n), dtype=A.dtype)
        
        for r in range(R):
            # Compute linear combination of A entries
            a_combo = 0
            for i in range(m):
                for k in range(p):
                    coeff = U[i * p + k, r]
                    if coeff != 0:
                        a_combo += coeff * A[i, k]
            
            # Compute linear combination of B entries
            b_combo = 0
            for k in range(p):
                for j in range(n):
                    coeff = V[k * n + j, r]
                    if coeff != 0:
                        b_combo += coeff * B[k, j]
            
            # Multiply and accumulate
            product = a_combo * b_combo
            
            for i in range(m):
                for j in range(n):
                    coeff = W[i * n + j, r]
                    if coeff != 0:
                        C_algo[i, j] += coeff * product
        
        error = np.max(np.abs(C_expected - C_algo))
        max_error = max(max_error, error)
        
        if use_integers and error != 0:
            return False, error
    
    passed = max_error < 1e-10 if not use_integers else max_error == 0
    return passed, max_error


def verify_all(result: DecompositionResult) -> dict:
    """Run all verification methods and return a report."""
    
    report = {
        'case': f"<{result.m},{result.p},{result.n}>",
        'rank': result.rank,
        'method': result.method,
    }
    
    # Method 1: Exact integer tensor reconstruction
    report['integer_tensor_check'] = verify_exact_integer(result)
    
    # Method 2: Random integer matrix tests
    passed, max_err = verify_by_random_matrices(result, n_tests=10000, 
                                                 use_integers=True)
    report['random_integer_check'] = passed
    report['random_integer_max_error'] = max_err
    
    # Method 3: Random float matrix tests (checks numerical stability)
    passed_f, max_err_f = verify_by_random_matrices(result, n_tests=10000,
                                                     use_integers=False)
    report['random_float_check'] = passed_f
    report['random_float_max_error'] = max_err_f
    
    # Overall
    report['verified'] = (report['integer_tensor_check'] and 
                          report['random_integer_check'])
    
    return report


def full_verification_report(result: DecompositionResult):
    """Print a detailed verification report."""
    
    report = verify_all(result)
    
    print(f"\nVERIFICATION REPORT")
    print(f"  Case: {report['case']} rank {report['rank']}")
    print(f"  Method: {report['method']}")
    print(f"  Integer tensor reconstruction: "
          f"{'PASS' if report['integer_tensor_check'] else 'FAIL'}")
    print(f"  Random integer matrices (10000 tests): "
          f"{'PASS' if report['random_integer_check'] else 'FAIL'} "
          f"(max error: {report['random_integer_max_error']})")
    print(f"  Random float matrices (10000 tests): "
          f"{'PASS' if report['random_float_check'] else 'FAIL'} "
          f"(max error: {report['random_float_max_error']:.2e})")
    print(f"  Overall: {'VERIFIED' if report['verified'] else 'FAILED'}")
    
    return report