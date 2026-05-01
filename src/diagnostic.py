"""
Diagnostic script: verify the search methods work on trivial cases
before wasting compute on hard ones.

This should take < 5 minutes and tells you exactly what's working.
"""

import numpy as np
import torch
import time
import sys

from tensor_utils import build_mult_tensor, verify_decomposition, make_result
from fixed_search import (standard_decomposition, verify_standard,
                           ProperRankReduction, IntegerSimulatedAnnealing,
                           GreedyFiniteField, ImprovedContinuousSearch)
from validation import verify_all


def run_diagnostics():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()
    
    all_pass = True
    
    # ---- Test 1: Standard decomposition ----
    print("=" * 60)
    print("TEST 1: Standard decomposition correctness")
    print("=" * 60)
    verify_standard()
    print("PASS\n")
    
    # ---- Test 2: Can we find Strassen (rank 7 for <2,2,2>)? ----
    print("=" * 60)
    print("TEST 2: Recover Strassen via improved continuous search")
    print("=" * 60)
    
    ics = ImprovedContinuousSearch(2, 2, 2, device=device)
    t = time.time()
    results = ics.search(R=7, n_restarts=50, n_steps=15000, verbose=True)
    elapsed = time.time() - t
    
    if results:
        print(f"PASS — found {len(results)} solutions in {elapsed:.1f}s")
        # Verify the best one
        report = verify_all(results[0])
        print(f"  Verification: {'PASS' if report['verified'] else 'FAIL'}")
        if not report['verified']:
            all_pass = False
    else:
        print(f"FAIL — could not find Strassen in {elapsed:.1f}s")
        all_pass = False
    print()
    
    # ---- Test 3: Rank reduction on <2,2,2> ----
    print("=" * 60)
    print("TEST 3: Rank reduction from standard (should find rank 7)")
    print("=" * 60)
    
    prr = ProperRankReduction(2, 2, 2, device=device)
    t = time.time()
    results = prr.reduce(target_rank=7, n_attempts=10, verbose=True)
    elapsed = time.time() - t
    
    if results:
        print(f"PASS — found {len(results)} solutions in {elapsed:.1f}s")
    else:
        print(f"FAIL — rank reduction didn't work in {elapsed:.1f}s")
        all_pass = False
    print()
    
    # ---- Test 4: Integer SA on <2,2,2> ----
    print("=" * 60)
    print("TEST 4: Integer simulated annealing (Strassen)")
    print("=" * 60)
    
    isa = IntegerSimulatedAnnealing(2, 2, 2)
    t = time.time()
    results = isa.search(R=7, n_restarts=3, n_steps_per_restart=1000000,
                         verbose=True)
    elapsed = time.time() - t
    
    if results:
        print(f"PASS — found {len(results)} solutions in {elapsed:.1f}s")
    else:
        print(f"WARN — SA didn't find Strassen in {elapsed:.1f}s "
              f"(may need more steps)")
    print()
    
    # ---- Test 5: Greedy GF(2) on <2,2,2> ----
    print("=" * 60)
    print("TEST 5: Greedy GF(2) construction")
    print("=" * 60)
    
    gff = GreedyFiniteField(2, 2, 2)
    t = time.time()
    results = gff.search_and_lift(target_rank=7, n_restarts=50, verbose=True)
    elapsed = time.time() - t
    
    if results:
        print(f"PASS — found {len(results)} solutions in {elapsed:.1f}s")
    else:
        print(f"WARN — GF(2) greedy didn't find liftable solution in {elapsed:.1f}s")
    print()
    
    # ---- Test 6: Scale test — <2,2,3> rank 11 ----
    print("=" * 60)
    print("TEST 6: Scale test — <2,2,3> rank 11 (known achievable)")
    print("=" * 60)
    
    ics = ImprovedContinuousSearch(2, 2, 3, device=device)
    t = time.time()
    results = ics.search(R=11, n_restarts=100, n_steps=20000, verbose=True)
    elapsed = time.time() - t
    
    if results:
        print(f"PASS — found {len(results)} solutions in {elapsed:.1f}s")
    else:
        print(f"FAIL — could not find rank 11 for <2,2,3> in {elapsed:.1f}s")
        all_pass = False
    print()
    
    # ---- Test 7: Scale test — <2,3,3> rank 15 ----
    print("=" * 60)
    print("TEST 7: Scale test — <2,3,3> rank 15 (known achievable)")
    print("=" * 60)
    
    ics = ImprovedContinuousSearch(2, 3, 3, device=device)
    t = time.time()
    results = ics.search(R=15, n_restarts=100, n_steps=20000, verbose=True)
    elapsed = time.time() - t
    
    if results:
        print(f"PASS — found {len(results)} solutions in {elapsed:.1f}s")
    else:
        # Try rank reduction as backup
        print(f"  Continuous search failed, trying rank reduction...")
        prr = ProperRankReduction(2, 3, 3, device=device)
        results = prr.reduce(target_rank=15, n_attempts=10, verbose=True)
        if results:
            print(f"PASS (via rank reduction) — found in {time.time()-t:.1f}s")
        else:
            print(f"FAIL — could not find rank 15 for <2,3,3>")
            all_pass = False
    print()
    
    # ---- Test 8: The critical test — <3,3,3> rank 23 ----
    print("=" * 60)
    print("TEST 8: Critical — <3,3,3> rank 23 (MUST reproduce)")
    print("=" * 60)
    print("This is the key test. If this fails, we cannot attempt rank 22.")
    print()
    
    # Try all methods with generous budgets
    found_333 = False
    
    # Method A: Improved continuous
    print("  Method A: Improved continuous search...")
    ics = ImprovedContinuousSearch(3, 3, 3, device=device)
    t = time.time()
    results = ics.search(R=23, n_restarts=200, n_steps=30000, verbose=True)
    elapsed = time.time() - t
    if results:
        print(f"  PASS via continuous — {elapsed:.1f}s")
        found_333 = True
    else:
        print(f"  Failed via continuous ({elapsed:.1f}s)")
    
    # Method B: Rank reduction
    if not found_333:
        print("\n  Method B: Rank reduction from standard...")
        prr = ProperRankReduction(3, 3, 3, device=device)
        t = time.time()
        results = prr.reduce(target_rank=23, n_attempts=20, verbose=True)
        elapsed = time.time() - t
        if results:
            print(f"  PASS via rank reduction — {elapsed:.1f}s")
            found_333 = True
        else:
            print(f"  Failed via rank reduction ({elapsed:.1f}s)")
    
    # Method C: Integer SA (longer)
    if not found_333:
        print("\n  Method C: Integer simulated annealing...")
        isa = IntegerSimulatedAnnealing(3, 3, 3)
        t = time.time()
        results = isa.search(R=23, n_restarts=3, n_steps_per_restart=3000000,
                            verbose=True)
        elapsed = time.time() - t
        if results:
            print(f"  PASS via integer SA — {elapsed:.1f}s")
            found_333 = True
        else:
            print(f"  Failed via integer SA ({elapsed:.1f}s)")
    
    if not found_333:
        print(f"\n  FAIL — Could not reproduce <3,3,3> rank 23.")
        print(f"  This means our methods are still insufficient for this case.")
        print(f"  Do NOT attempt rank 22 until this is resolved.")
        all_pass = False
    print()
    
    # ---- Test 9: <2,3,4> rank 20 ----
    print("=" * 60)
    print("TEST 9: <2,3,4> rank 20 (AlphaTensor result)")
    print("=" * 60)
    
    ics = ImprovedContinuousSearch(2, 3, 4, device=device)
    t = time.time()
    results = ics.search(R=20, n_restarts=200, n_steps=30000, verbose=True)
    elapsed = time.time() - t
    
    if results:
        print(f"PASS — found {len(results)} solutions in {elapsed:.1f}s")
    else:
        # Try rank reduction
        print(f"  Continuous failed, trying rank reduction...")
        prr = ProperRankReduction(2, 3, 4, device=device)
        results = prr.reduce(target_rank=20, n_attempts=15, verbose=True)
        if results:
            print(f"PASS (via rank reduction) — found in {time.time()-t:.1f}s")
        else:
            print(f"FAIL — could not reproduce <2,3,4> rank 20")
            all_pass = False
    print()
    
    # ---- Summary ----
    print("=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if all_pass:
        print("All critical tests PASSED.")
        print("Safe to proceed with improvement attempts.")
    else:
        print("Some tests FAILED.")
        print("Do NOT attempt improvement experiments until all pass.")
        print("\nRecommended actions:")
        print("  1. Check that initialization from standard decomposition works")
        print("  2. Increase restarts/steps for failing cases")
        print("  3. Consider if the problem needs fundamentally different approach")
    
    return all_pass


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)