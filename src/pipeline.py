"""
Main pipeline: target selection → search → analysis.

Usage:
    python pipeline.py                    # full pipeline
    python pipeline.py --case 2,3,4      # specific case
    python pipeline.py --analyze-only     # just print target analysis
"""

import argparse
import json
import os
import time
from typing import List, Dict
import numpy as np

from tensor_utils import (DecompositionResult, KNOWN_RANKS, 
                           build_mult_tensor, verify_decomposition,
                           get_lower_bound)
from omega_analysis import analyze_targets, select_experiments
from continuous_search import ContinuousSearch, ALSSearch
from finite_field_search import search_all_fields


def run_search_for_case(m: int, p: int, n: int, target_rank: int,
                         config: Dict) -> List[DecompositionResult]:
    """
    Run all Tier 1 search methods for a single case.
    Returns all exact decompositions found.
    """
    all_results = []
    
    print(f"\n{'#'*70}")
    print(f"# TARGET: <{m},{p},{n}> rank {target_rank}")
    print(f"# Standard rank: {m*p*n}, "
          f"known best: {KNOWN_RANKS.get((m,p,n), (m*p*n, 'unknown'))[0]}")
    print(f"{'#'*70}")
    
    lb_arbitrary = get_lower_bound(m, p, n, field="arbitrary")
    if target_rank < lb_arbitrary:
        print(f"Skipping pursuit of rank {target_rank} because it is below the known mathematically proven lower bound of {lb_arbitrary}.")
        return all_results
    
    # ---- Method 1: Continuous optimization (gradient descent) ----
    print(f"\n--- Method 1: Gradient descent ---")
    
    device = 'cuda' if config.get('use_cuda', False) else 'cpu'
    
    try:
        import torch
        if config.get('use_cuda') and torch.cuda.is_available():
            device = 'cuda'
            print(f"  Using CUDA")
        
        searcher = ContinuousSearch(m, p, n, device=device)
        gradient_results = searcher.search(
            R=target_rank,
            n_restarts=config.get('gradient_restarts', 200),
            n_steps=config.get('gradient_steps', 20000),
            lr=config.get('gradient_lr', 0.003),
            verbose=True
        )
        all_results.extend(gradient_results)
        
    except ImportError:
        print("  PyTorch not available, skipping gradient search")
    
    # ---- Method 2: Alternating Least Squares ----
    print(f"\n--- Method 2: Alternating Least Squares ---")
    
    als_searcher = ALSSearch(m, p, n)
    als_results = als_searcher.search(
        R=target_rank,
        n_restarts=config.get('als_restarts', 500),
        n_steps=config.get('als_steps', 3000),
        verbose=True
    )
    all_results.extend(als_results)
    
    # ---- Method 3: Finite field search + lifting ----
    print(f"\n--- Method 3: Finite field search + lifting ---")
    
    # Only feasible for smallish tensors
    tensor_size = (m * p) * (p * n) * (m * n)
    if tensor_size <= 2000:
        ff_primes = config.get('primes', [2, 3, 5])
        for prime in ff_primes:
            lb_ff = get_lower_bound(m, p, n, field=f"GF{prime}")
            if target_rank < lb_ff:
                print(f"  Skipping exhaustive search for prime {prime} (target {target_rank} < lower bound {lb_ff})")
                continue
                
            ff_results = search_all_fields(
                m, p, n, target_rank,
                primes=[prime],
                n_attempts_per_prime=config.get('ff_attempts', 1000000),
                verbose=True
            )
            all_results.extend(ff_results)
    else:
        print(f"  Tensor too large ({tensor_size} entries) for "
              f"exhaustive finite field search")
        print(f"  Running structured search only...")
        
        from finite_field_search import search_gf_structured, lift_to_integers
        for prime in config.get('primes', [2, 3, 5]):
            lb_ff = get_lower_bound(m, p, n, field=f"GF{prime}")
            if target_rank < lb_ff:
                print(f"  Skipping structured search for prime {prime} (target {target_rank} < lower bound {lb_ff})")
                continue
                
            solutions = search_gf_structured(
                m, p, n, target_rank, prime,
                n_attempts=config.get('ff_attempts', 500000) // 2,
                verbose=True
            )
            for U, V, W in solutions:
                result = lift_to_integers(U, V, W, m, p, n, prime, verbose=True)
                if result is not None and result.is_exact:
                    all_results.append(result)
    
    return all_results


def deduplicate_results(results: List[DecompositionResult]
                         ) -> List[DecompositionResult]:
    """
    Remove duplicate decompositions (same algorithm found by different methods).
    
    Two decompositions are considered equivalent if they have the same
    set of (u_r, v_r, w_r) triples up to permutation of the R terms.
    """
    unique = []
    seen_signatures = set()
    
    for r in results:
        # Create a canonical signature: sort columns and hash
        cols = []
        for j in range(r.rank):
            col = (tuple(r.U[:, j].flat), 
                   tuple(r.V[:, j].flat), 
                   tuple(r.W[:, j].flat))
            cols.append(col)
        cols.sort()
        sig = str(cols)
        
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique.append(r)
    
    return unique


def analyze_results(results: List[DecompositionResult]):
    """Print detailed analysis of found decompositions."""
    
    if not results:
        print("\nNo exact decompositions found.")
        return
    
    results = deduplicate_results(results)
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(results)} unique exact decomposition(s)")
    print(f"{'='*70}")
    
    # Sort by quality: prefer smaller max_coefficient, then fewer additions
    results.sort(key=lambda r: (r.max_coefficient, r.num_additions))
    
    for i, r in enumerate(results):
        print(f"\n  Solution {i+1}:")
        print(f"    {r.summary()}")
        print(f"    U (A-side factors):")
        print(f"      {r.U.T}")
        print(f"    V (B-side factors):")
        print(f"      {r.V.T}")
        print(f"    W (C-side factors):")
        print(f"      {r.W.T}")
    
    # Best solution
    best = results[0]
    print(f"\n  BEST (by coefficient size): {best.summary()}")
    
    # Check if this improves on known results
    key = (best.m, best.p, best.n)
    if key in KNOWN_RANKS:
        known_rank, source = KNOWN_RANKS[key]
        if best.rank < known_rank:
            print(f"\n  *** NEW RESULT: rank {best.rank} improves on "
                  f"known best {known_rank} ({source})! ***")
        elif best.rank == known_rank:
            print(f"\n  Matches known best rank {known_rank} ({source})")
        else:
            print(f"\n  Known best is rank {known_rank} ({source})")


def save_results(results: List[DecompositionResult], output_dir: str):
    """Save results to disk for later analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, r in enumerate(results):
        case_str = f"{r.m}_{r.p}_{r.n}"
        filename = f"{case_str}_rank{r.rank}_{r.method}_{i}.npz"
        filepath = os.path.join(output_dir, filename)
        
        np.savez(filepath,
                 U=r.U, V=r.V, W=r.W,
                 m=r.m, p=r.p, n=r.n,
                 rank=r.rank,
                 method=r.method,
                 field=r.field,
                 error=r.reconstruction_error)
    
    # Also save a summary
    summary = []
    for r in results:
        summary.append({
            'case': f"<{r.m},{r.p},{r.n}>",
            'rank': r.rank,
            'method': r.method,
            'field': r.field,
            'max_coefficient': int(r.max_coefficient),
            'num_additions': int(r.num_additions),
            'omega_bound': float(r.omega_bound),
            'is_exact': r.is_exact,
        })
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def generate_algorithm_code(result: DecompositionResult) -> str:
    """
    Convert a found decomposition into readable pseudocode / Python.
    """
    m, p, n = result.m, result.p, result.n
    R = result.rank
    U, V, W = result.U, result.V, result.W
    
    lines = []
    lines.append(f"def matmul_{m}{p}{n}(A, B):")
    lines.append(f"    \"\"\"")
    lines.append(f"    Multiply ({m}x{p}) @ ({p}x{n}) using "
                 f"{R} multiplications (standard: {m*p*n}).")
    lines.append(f"    Found by: {result.method}")
    lines.append(f"    Max coefficient: {result.max_coefficient}")
    lines.append(f"    \"\"\"")
    lines.append(f"    import numpy as np")
    lines.append(f"")
    
    # Generate intermediate products
    for r in range(R):
        a_terms = []
        for i in range(m):
            for k in range(p):
                coeff = int(U[i * p + k, r])
                if coeff == 0:
                    continue
                term = f"A[{i},{k}]"
                if coeff == 1:
                    a_terms.append(f"+ {term}")
                elif coeff == -1:
                    a_terms.append(f"- {term}")
                elif coeff > 0:
                    a_terms.append(f"+ {coeff}*{term}")
                else:
                    a_terms.append(f"- {-coeff}*{term}")
        
        b_terms = []
        for k in range(p):
            for j in range(n):
                coeff = int(V[k * n + j, r])
                if coeff == 0:
                    continue
                term = f"B[{k},{j}]"
                if coeff == 1:
                    b_terms.append(f"+ {term}")
                elif coeff == -1:
                    b_terms.append(f"- {term}")
                elif coeff > 0:
                    b_terms.append(f"+ {coeff}*{term}")
                else:
                    b_terms.append(f"- {-coeff}*{term}")
        
        a_expr = " ".join(a_terms).strip()
        if a_expr.startswith("+ "):
            a_expr = a_expr[2:]
        b_expr = " ".join(b_terms).strip()
        if b_expr.startswith("+ "):
            b_expr = b_expr[2:]
        
        lines.append(f"    m{r} = ({a_expr}) * ({b_expr})")
    
    lines.append(f"")
    lines.append(f"    C = np.zeros(({m}, {n}), dtype=A.dtype)")
    
    for i in range(m):
        for j in range(n):
            c_terms = []
            for r in range(R):
                coeff = int(W[i * n + j, r])
                if coeff == 0:
                    continue
                if coeff == 1:
                    c_terms.append(f"+ m{r}")
                elif coeff == -1:
                    c_terms.append(f"- m{r}")
                elif coeff > 0:
                    c_terms.append(f"+ {coeff}*m{r}")
                else:
                    c_terms.append(f"- {-coeff}*m{r}")
            
            c_expr = " ".join(c_terms).strip()
            if c_expr.startswith("+ "):
                c_expr = c_expr[2:]
            if c_expr:
                lines.append(f"    C[{i},{j}] = {c_expr}")
    
    lines.append(f"")
    lines.append(f"    return C")
    
    return "\n".join(lines)


def validate_algorithm(result: DecompositionResult, n_tests: int = 1000):
    """
    Validate a found algorithm by testing it on random matrices.
    """
    m, p, n = result.m, result.p, result.n
    
    code = generate_algorithm_code(result)
    
    # Create the function
    local_ns = {}
    exec(code, {'np': np, 'numpy': np}, local_ns)
    func_name = f"matmul_{m}{p}{n}"
    fast_func = local_ns[func_name]
    
    max_error = 0.0
    for _ in range(n_tests):
        A = np.random.randn(m, p)
        B = np.random.randn(p, n)
        
        C_standard = A @ B
        C_fast = fast_func(A, B)
        
        error = np.max(np.abs(C_standard - C_fast))
        max_error = max(max_error, error)
    
    return max_error


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Matrix multiplication algorithm search pipeline'
    )
    parser.add_argument('--case', type=str, default=None,
                        help='Specific case to search, e.g., "2,3,4"')
    parser.add_argument('--rank', type=int, default=None,
                        help='Target rank (default: known best)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only print target analysis')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--gradient-restarts', type=int, default=200)
    parser.add_argument('--gradient-steps', type=int, default=20000)
    parser.add_argument('--als-restarts', type=int, default=500)
    parser.add_argument('--ff-attempts', type=int, default=1000000)
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with reduced parameters')
    
    args = parser.parse_args()
    
    # Target analysis
    print("\n" + "=" * 70)
    print("MATRIX MULTIPLICATION ALGORITHM SEARCH — TIER 1 PIPELINE")
    print("=" * 70)
    
    targets = analyze_targets()
    
    if args.analyze_only:
        return
    
# Build config
    config = {
        'gradient_restarts': args.gradient_restarts,
        'gradient_steps': args.gradient_steps,
        'gradient_lr': 0.003,
        'als_restarts': args.als_restarts,
        'als_steps': 3000,
        'ff_attempts': args.ff_attempts,
        'primes': [2, 3, 5],
        'use_cuda': args.use_cuda,
    }
    
    if args.quick:
        config.update({
            'gradient_restarts': 30,
            'gradient_steps': 8000,
            'als_restarts': 100,
            'ff_attempts': 200000,
        })
    
    # Determine what to search
    if args.case:
        parts = [int(x) for x in args.case.split(',')]
        assert len(parts) == 3, "Case must be m,p,n"
        m, p, n = parts
        
        if args.rank:
            target_rank = args.rank
        elif (m, p, n) in KNOWN_RANKS:
            target_rank = KNOWN_RANKS[(m, p, n)][0]
        else:
            target_rank = m * p * n - 1
        
        all_results = run_search_for_case(m, p, n, target_rank, config)
        analyze_results(all_results)
        
        if all_results:
            save_results(all_results, args.output_dir)
            
            # Generate and validate algorithm code
            best = sorted(all_results, 
                         key=lambda r: (r.max_coefficient, r.num_additions))[0]
            
            print(f"\n{'='*70}")
            print("GENERATED ALGORITHM")
            print(f"{'='*70}")
            code = generate_algorithm_code(best)
            print(code)
            
            print(f"\n{'='*70}")
            print("VALIDATION")
            print(f"{'='*70}")
            max_err = validate_algorithm(best, n_tests=10000)
            print(f"  Max error over 10000 random tests: {max_err:.2e}")
            if max_err < 1e-10:
                print(f"  PASSED — algorithm is correct")
            else:
                print(f"  WARNING — numerical error detected")
    
    else:
        # Run the full pipeline on prioritized targets
        experiments = select_experiments(max_tensor_entries=3000, n_targets=5)
        
        all_found = []
        
        for exp in experiments:
            m, p, n = exp['case']
            target_rank = exp['target_rank']
            purpose = exp['purpose']
            
            print(f"\n\n{'*'*70}")
            print(f"* EXPERIMENT: <{m},{p},{n}> rank {target_rank} ({purpose})")
            print(f"{'*'*70}")
            
            results = run_search_for_case(m, p, n, target_rank, config)
            
            if results:
                all_found.extend(results)
                analyze_results(results)
                save_results(results, 
                           os.path.join(args.output_dir, 
                                       f"{m}_{p}_{n}_rank{target_rank}"))
                
                # If we're trying to improve and succeeded, this is big
                if purpose == 'improve':
                    print(f"\n  !!! POTENTIAL NEW RESULT: "
                          f"<{m},{p},{n}> achievable at rank {target_rank} !!!")
                    print(f"  !!! Verify carefully before claiming !!!")
            else:
                if purpose == 'validate':
                    print(f"\n  Could not reproduce known rank {target_rank} "
                          f"for <{m},{p},{n}>")
                    print(f"  Consider increasing restarts or steps")
                else:
                    print(f"\n  No improvement found for <{m},{p},{n}> "
                          f"at rank {target_rank}")
                    print(f"  (This is expected — most attempts won't succeed)")
        
        # Final summary
        print(f"\n\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"  Total experiments: {len(experiments)}")
        print(f"  Total decompositions found: {len(all_found)}")
        
        if all_found:
            save_results(all_found, args.output_dir)
            
            improvements = [r for r in all_found 
                          if (r.m, r.p, r.n) in KNOWN_RANKS 
                          and r.rank < KNOWN_RANKS[(r.m, r.p, r.n)][0]]
            
            if improvements:
                print(f"\n  *** IMPROVEMENTS FOUND: ***")
                for r in improvements:
                    known = KNOWN_RANKS[(r.m, r.p, r.n)][0]
                    print(f"    <{r.m},{r.p},{r.n}>: rank {r.rank} "
                          f"(previously {known})")
            else:
                print(f"\n  No improvements over known best ranks.")
                print(f"  Matched known results: "
                      f"{sum(1 for r in all_found if (r.m,r.p,r.n) in KNOWN_RANKS and r.rank == KNOWN_RANKS[(r.m,r.p,r.n)][0])}")


if __name__ == "__main__":
    main()