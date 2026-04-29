"""
Validate the entire pipeline by recovering Strassen's algorithm.

If this doesn't work, nothing else will. Run this first.
"""

import numpy as np
import sys
from tensor_utils import build_mult_tensor, verify_decomposition, make_result
from continuous_search import ContinuousSearch, ALSSearch
from finite_field_search import search_all_fields
from validation import full_verification_report, verify_exact_integer
from numerical_stability import stability_vs_standard


def strassen_factors():
    """Strassen's algorithm as explicit factor matrices."""
    
    # 7 multiplications for 2x2 matrix multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B01 - B11)
    # M4 = (A11)(B10 - B00)
    # M5 = (A00 + A01)(B11)
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U: (4, 7) — coefficients for A entries
    # A entries ordered: A00, A01, A10, A11 (row-major of 2x2)
    U = np.array([
        # M1  M2  M3  M4  M5  M6  M7
        [ 1,  0,  1,  0,  1, -1,  0],  # A00
        [ 0,  0,  0,  0,  1,  0,  1],  # A01
        [ 0,  1,  0,  0,  0,  1,  0],  # A10
        [ 1,  1,  0,  1,  0,  0, -1],  # A11
    ], dtype=np.int64)
    
    # V: (4, 7) — coefficients for B entries
    # B entries ordered: B00, B01, B10, B11
    V = np.array([
        # M1  M2  M3  M4  M5  M6  M7
        [ 1,  1,  0, -1,  0,  1,  0],  # B00
        [ 0,  0,  1,  0,  0,  1,  0],  # B01
        [ 0,  0,  0,  1,  0,  0,  1],  # B10
        [ 1,  0, -1,  0,  1,  0,  1],  # B11
    ], dtype=np.int64)
    
    # W: (4, 7) — accumulation into C entries
    # C entries ordered: C00, C01, C10, C11
    W = np.array([
        # M1  M2  M3  M4  M5  M6  M7
        [ 1,  0,  0,  1, -1,  0,  1],  # C00
        [ 0,  0,  1,  0,  1,  0,  0],  # C01
        [ 0,  1,  0,  1,  0,  0,  0],  # C10
        [ 1, -1,  1,  0,  0,  1,  0],  # C11
    ], dtype=np.int64)
    
    return U, V, W


def validate_framework():
    """
    Full validation: verify that our framework correctly handles
    Strassen's algorithm, then try to rediscover it.
    """
    print("=" * 70)
    print("FRAMEWORK VALIDATION — STRASSEN'S ALGORITHM")
    print("=" * 70)
    
    # Step 1: Verify the known Strassen decomposition
    print("\n1. Verifying known Strassen factors...")
    
    T = build_mult_tensor(2, 2, 2)
    U, V, W = strassen_factors()
    
    error = verify_decomposition(T, U.astype(np.float64), 
                                  V.astype(np.float64), 
                                  W.astype(np.float64))
    print(f"   Reconstruction error: {error}")
    assert error < 1e-10, f"Strassen verification failed! Error: {error}"
    print("   PASSED")
    
    # Step 2: Full verification
    print("\n2. Full verification of Strassen...")
    result = make_result(U, V, W, 2, 2, 2, 'known_strassen', 'Z')
    report = full_verification_report(result)
    assert report['verified'], "Full verification failed!"
    
    # Step 3: Numerical stability baseline
    print("\n3. Numerical stability of Strassen...")
    stability_vs_standard(result)
    
    # Step 4: Try to rediscover Strassen via gradient descent
    print("\n4. Attempting to rediscover rank-7 for <2,2,2> via gradient...")
    
    searcher = ContinuousSearch(2, 2, 2)
    results = searcher.search(R=7, n_restarts=50, n_steps=15000, verbose=True)
    
    if results:
        print(f"\n   Found {len(results)} rank-7 decomposition(s)!")
        for r in results[:3]:
            print(f"   {r.summary()}")
            full_verification_report(r)
    else:
        print("\n   WARNING: Could not rediscover rank 7.")
        print("   Try increasing restarts or steps.")
    
    # Step 5: Try ALS
    print("\n5. Attempting via ALS...")
    
    als = ALSSearch(2, 2, 2)
    als_results = als.search(R=7, n_restarts=200, n_steps=2000, verbose=True)
    
    if als_results:
        print(f"\n   ALS found {len(als_results)} decomposition(s)!")
    
    # Step 6: Verify rank 6 is impossible (should fail)
    print("\n6. Confirming rank 6 is NOT achievable (should find nothing)...")
    
    searcher6 = ContinuousSearch(2, 2, 2)
    results6 = searcher6.search(R=6, n_restarts=30, n_steps=10000, verbose=True)
    
    if results6:
        print("   WARNING: Found rank-6 decomposition! This would be a")
        print("   major result (contradicts known lower bound). Verify carefully.")
    else:
        print("   Correctly failed to find rank 6 (known to be impossible).")
    
    # Step 7: Finite field search
    print("\n7. Finite field search for <2,2,2> rank 7...")
    
    ff_results = search_all_fields(2, 2, 2, target_rank=7,
                                    primes=[2, 3, 5],
                                    n_attempts_per_prime=500000,
                                    verbose=True)
    
    if ff_results:
        print(f"\n   Found {len(ff_results)} via finite field + lifting!")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    total = len(results) + len(als_results) + len(ff_results)
    print(f"\n  Total rank-7 decompositions found: {total}")
    
    if total > 0:
        print("  Framework is working correctly.")
        print("  Ready to search for new results.")
    else:
        print("  WARNING: No decompositions found. Check configuration.")
        print("  Possible issues:")
        print("    - Learning rate too high/low")
        print("    - Not enough restarts")
        print("    - Numerical issues")
    
    return total > 0


if __name__ == "__main__":
    success = validate_framework()
    sys.exit(0 if success else 1)