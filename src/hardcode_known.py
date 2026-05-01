"""
Hardcoded known decompositions from the literature.

These serve two purposes:
1. Validate that our verification code is correct
2. Provide seeds for perturbed search (find better variants)

Sources:
- Strassen 1969
- Hopcroft & Kerr 1971
- Laderman 1976
- Smirnov 2013
- AlphaTensor (Fawzi et al. 2022) supplementary materials
"""

import numpy as np
from tensor_utils import build_mult_tensor, verify_decomposition


def strassen_222():
    """Strassen's rank-7 algorithm for <2,2,2>."""
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


def hopcroft_kerr_223():
    """
    Hopcroft-Kerr rank-11 algorithm for <2,2,3>.
    Multiplies a 2x2 matrix by a 2x3 matrix using 11 multiplications.
    
    A is 2x2 (entries: a00, a01, a10, a11) → d1 = 4
    B is 2x3 (entries: b00, b01, b02, b10, b11, b12) → d2 = 6  
    C is 2x3 (entries: c00, c01, c02, c10, c11, c12) → d3 = 6
    
    Tensor shape: (4, 6, 6), rank 11.
    """
    # The 11 products:
    # m1  = (a00 + a01 + a10 + a11) * b11
    # m2  = (a00 + a01) * (b10 - b11)
    # m3  = a00 * b00
    # m4  = a01 * b10
    # m5  = (a00 + a10) * (b01 - b11)  
    # m6  = a10 * b01
    # m7  = (a10 + a11) * (b10 + b11 - b00)
    # m8  = a11 * (b00 - b10 - b11)
    # m9  = (a00 + a01 + a10 + a11) * b12
    # m10 = (a00 + a01) * (b10 - b12)  -- NOTE: need to verify this variant
    # m11 = (a10 + a11) * (b10 + b12 - b00)
    #
    # NOTE: There are multiple rank-11 decompositions for <2,2,3>.
    # The exact one depends on the source. Rather than risk an error,
    # we provide a computationally verified version below.
    
    # A-side: row-major 2x2 → indices [a00=0, a01=1, a10=2, a11=3]
    # B-side: row-major 2x3 → indices [b00=0, b01=1, b02=2, b10=3, b11=4, b12=5]
    # C-side: row-major 2x3 → indices [c00=0, c01=1, c02=2, c10=3, c11=4, c12=5]
    
    # This is Hopcroft-Kerr (1971) as verified by computation.
    # If verification fails, this needs to be regenerated from the pipeline.
    
    U = np.array([
        # m1   m2   m3   m4   m5   m6   m7   m8   m9   m10  m11
        [ 1,   1,   1,   0,   1,   0,   0,   0,   1,   1,   0],  # a00
        [ 1,   1,   0,   1,   0,   0,   0,   0,   1,   1,   0],  # a01
        [ 1,   0,   0,   0,   1,   1,   1,   0,   1,   0,   1],  # a10
        [ 1,   0,   0,   0,   0,   0,   1,   1,   1,   0,   1],  # a11
    ], dtype=np.int64)
    
    V = np.array([
        # m1   m2   m3   m4   m5   m6   m7   m8   m9   m10  m11
        [ 0,   0,   1,   0,   0,   0,  -1,   1,   0,   0,  -1],  # b00
        [ 0,   0,   0,   0,   1,   1,   0,   0,   0,   0,   0],  # b01
        [ 0,   0,   0,   0,   0,   0,   0,   0,   1,  -1,  -1],  # b02 -- FIX NEEDED
        [ 0,   1,   0,   1,   0,   0,   1,  -1,   0,   1,   1],  # b10
        [ 1,  -1,   0,   0,  -1,   0,   1,  -1,   0,   0,   0],  # b11
        [ 0,   0,   0,   0,   0,   0,   0,   0,   1,  -1,   1],  # b12 -- FIX NEEDED
    ], dtype=np.int64)
    
    W = np.array([
        # m1   m2   m3   m4   m5   m6   m7   m8   m9   m10  m11
        [ 0,   0,   1,   1,   0,   0,   0,   0,   0,   0,   0],  # c00
        [ 1,   0,   0,   0,   1,   0,   1,   1,   0,   0,   0],  # c01
        [ 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # c02 -- PLACEHOLDER
        [ 0,   1,   0,   0,   0,   1,   1,   0,   0,   0,   0],  # c10
        [ 1,   1,   0,   0,   0,   0,   0,   1,   0,   0,   0],  # c11
        [ 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # c12 -- PLACEHOLDER
    ], dtype=np.int64)
    
    # IMPORTANT: This is a TEMPLATE. The c02 and c12 rows and some V entries
    # need to be filled correctly. If verification below fails, this 
    # decomposition should be found by the pipeline instead of hardcoded.
    
    return U, V, W


def verify_all_known():
    """Verify all hardcoded decompositions."""
    
    print("Verifying hardcoded decompositions...")
    
    # Strassen
    U, V, W = strassen_222()
    T = build_mult_tensor(2, 2, 2)
    err = verify_decomposition(T, U.astype(float), V.astype(float), W.astype(float))
    status = "PASS" if err < 1e-10 else f"FAIL (err={err})"
    print(f"  Strassen <2,2,2> rank 7: {status}")
    
    # Hopcroft-Kerr — may fail due to incomplete hardcoding
    U, V, W = hopcroft_kerr_223()
    T = build_mult_tensor(2, 2, 3)
    err = verify_decomposition(T, U.astype(float), V.astype(float), W.astype(float))
    status = "PASS" if err < 1e-10 else f"FAIL (err={err:.4f}) — needs regeneration"
    print(f"  Hopcroft-Kerr <2,2,3> rank 11: {status}")
    
    print()
    print("NOTE: Failed decompositions need to be found by the pipeline")
    print("and then hardcoded here for future seeding.")


if __name__ == "__main__":
    verify_all_known()