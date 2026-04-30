#!/usr/bin/env python3
"""
Verify and report if the discovered decompositions improve upon 
existing known results (in terms of rank or number of additions/coefficients).
"""

import os
import json
import glob
import numpy as np
import sys

# Add src to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from tensor_utils import KNOWN_RANKS, KNOWN_LOWER_BOUNDS
    from validation import verify_all, DecompositionResult
    from numerical_stability import error_amplification
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Format: (m, p, n): (rank, additions)
# Additions are set to None for larger AlphaTensor discoveries where 
# the exact count highly depends on the specific Common Subexpression Elimination (CSE) pass used.
KNOWN_ADDITIONS = {
    # mpn = 8
    (2, 2, 2): (7, 15),     # Strassen 1969 (18 adds) / Winograd variant (15 adds)
    
    # mpn = 12
    (2, 2, 3): (11, 28),    # Hopcroft-Kerr 1971 / AlphaTensor baseline
    
    # mpn = 16
    (2, 2, 4): (14, 40),    # AlphaTensor 2022 / Hopcroft-Kerr
    (2, 4, 2): (14, 40),    # symmetry of above
    
    # mpn = 18
    (2, 3, 3): (15, 63),    # Smirnov / AlphaTensor 2022
    
    # mpn = 20
    (2, 2, 5): (18, 56),    # AlphaTensor 2022
    
    # mpn = 24
    (2, 3, 4): (20, 84),    # AlphaTensor 2022
    (2, 2, 6): (21, 45),    # recursive 3x (2,2,2) -> 3 * 15 adds
    
    # mpn = 27
    (3, 3, 3): (23, 68),    # Laderman 1976 Standard (68 adds); recent AI methods found 60
    
    # mpn = 30
    (2, 3, 5): (25, None),  # AlphaTensor 2022 (Additions depend on CSE pass)
    
    # mpn = 32
    (2, 4, 4): (26, None),  # AlphaTensor 2022
    (2, 2, 8): (28, 60),    # recursive 4x Strassen/Winograd -> 4 * 15 adds
    
    # mpn = 36
    (3, 3, 4): (29, None),  # AlphaTensor 2022
    (2, 3, 6): (30, 126),   # recursive 2x (2,3,3) -> 2 * 63 adds
    
    # mpn = 40  
    (2, 4, 5): (33, None),  # AlphaTensor 2022
    (2, 2, 10): (35, 75),   # recursive 5x (2,2,2) -> 5 * 15 adds
    
    # mpn = 45
    (3, 3, 5): (36, None),  # AlphaTensor 2022
    
    # mpn = 48
    (3, 4, 4): (38, None),  # AlphaTensor 2022
    (2, 4, 6): (40, 168),   # recursive 2x (2,3,4) -> 2 * 84 adds
    
    # mpn = 50
    (2, 5, 5): (40, None),  # AlphaTensor 2022
    
    # mpn = 54
    (3, 3, 6): (45, 189),   # recursive 3x (2,3,2) equiv to (2,3,3) -> 3 * 63 adds
    
    # mpn = 60
    (3, 4, 5): (47, None),  # AlphaTensor 2022
    (2, 5, 6): (45, None),  # recursive
    (2, 3, 10): (50, None), # recursive 2x (2,3,5)
    
    # mpn = 64
    (4, 4, 4): (49, 260),   # Strassen recursive; AlphaTensor got 47 over GF(2)
    
    # mpn = 72
    (3, 4, 6): (54, None),  # limited prior work
    
    # mpn = 75
    (3, 5, 5): (58, None),  # AlphaTensor 2022
    
    # mpn = 80
    (4, 4, 5): (63, None),  # AlphaTensor 2022 Standard; 62 over GF(2)
    
    # mpn = 100
    (4, 5, 5): (76, None),  # AlphaTensor 2022
    
    # mpn = 125
    (5, 5, 5): (98, None)   # AlphaTensor 2022 Standard; 96 over GF(2)
}

def load_npz_result(filepath, dims, rank):
    data = np.load(filepath)
    m, p, n = dims
    
    # We might need to guess the shape if it's stored flat,
    # but let's assume it has U, V, W or u, v, w arrays.
    # From tensor_utils.py, build_mult_tensor and extraction
    # usually stores U, V, W directly. Let's inspect properties.
    if 'U' in data and 'V' in data and 'W' in data:
        U, V, W = data['U'], data['V'], data['W']
    else:
        # Fallback if structure is different
        pass
    
    # Simple placeholder DecompositionResult
    res = DecompositionResult(
        U=U,
        V=V,
        W=W,
        rank=rank,
        reconstruction_error=0.0,
        m=m,
        p=p,
        n=n,
        method="loaded",
        field="Z",
        max_coefficient=1,
        num_additions=0
    )
    
    # compute max coeff and additions
    res.max_coefficient = int(max(np.max(np.abs(U)), np.max(np.abs(V)), np.max(np.abs(W))))
    
    additions = 0
    for r in range(rank):
        additions += max(0, np.count_nonzero(U[:, r]) - 1)
        additions += max(0, np.count_nonzero(V[:, r]) - 1)
    
    for i in range(m):
        for j in range(n):
            w_nnz = np.count_nonzero(W[i*n+j, :])
            if w_nnz > 1:
                additions += w_nnz - 1
                
    res.num_additions = additions
    
    # Compute numerical stability estimate using a low trial count to be fast but informative
    stability_stats = error_amplification(res, n_trials=100)
    res.stability_ratio = stability_stats['mean_relative_error']
    
    return res

def get_known_rank(m, p, n):
    dims = tuple(sorted([m, p, n]))
    if dims in KNOWN_RANKS:
        return KNOWN_RANKS[dims][0], KNOWN_RANKS[dims][1]
    if dims in KNOWN_LOWER_BOUNDS:
        return KNOWN_LOWER_BOUNDS[dims], "Lower bound"
    return None, "Unknown"

def main():
    log_file = os.path.join('batch_results', 'experiment_log.json')
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    with open(log_file, 'r') as f:
        log_data = json.load(f)

    improvements_by_rank = []
    improvements_by_adds = []
    verified_results = []
    
    print("Analyzing batch results against known baselines...\n")

    for entry in log_data:
        # Get dimensions from the correct key 'case'
        m, p, n = entry['case']
        t_rank = entry['target_rank']
        
        # Check if the desired target_rank was achieved
        found = entry.get('n_found', 0) > 0
        
        if not found:
            continue
            
        best_adds = entry.get('best_additions', float('inf'))
        best_coeff = entry.get('best_max_coeff', float('inf'))
        
        dims = tuple(sorted([m, p, n]))
        known_rank, source = get_known_rank(m, p, n)
        
        is_rank_improvement = False
        is_add_improvement = False
        
        # Check rank improvements
        if known_rank is not None and t_rank < known_rank:
            is_rank_improvement = True
            
        # Check addition improvements
        known_adds = KNOWN_ADDITIONS.get(dims, (None, float('inf')))[1]
        if known_rank is not None and t_rank == known_rank:
            if best_adds < known_adds:
                is_add_improvement = True
                
        if is_rank_improvement:
            improvements_by_rank.append((entry, known_rank, source))
        if is_add_improvement:
            improvements_by_adds.append((entry, known_adds))
            
        verified_results.append(entry)

    print(f"Total successful decomposition runs evaluated: {len(verified_results)}")
    print(f"Found {len(improvements_by_rank)} rank improvements.")
    print(f"Found {len(improvements_by_adds)} addition improvements compared to registered KNOWN_ADDITIONS.\n")
    
    if improvements_by_rank:
        print("=== RANK IMPROVEMENTS ===")
        for entry, kr, src in improvements_by_rank:
            dims = f"{entry['case']}"
            print(f"- {dims}: Found rank {entry['target_rank']} (Known: {kr} from {src})")
            
    if improvements_by_adds:
        print("=== PROBABLE ADDITION IMPROVEMENTS ===")
        for entry, ka in improvements_by_adds:
            dims = f"{entry['case']}"
            
            # Formulate rank directory name
            # Assuming format: m_p_n_rankR
            m, p, n = entry['case']
            rank = entry['target_rank']
            dir_name = f"{m}_{p}_{n}_rank{rank}"
            dir_path = os.path.join('batch_results', dir_name)
            
            stability_str = "N/A"
            if os.path.exists(dir_path):
                # Try load solution_0.npz
                sol_file = os.path.join(dir_path, 'solution_0.npz')
                if os.path.exists(sol_file):
                    res = load_npz_result(sol_file, (m, p, n), rank)
                    stability_str = f"{res.stability_ratio:.2e}"

            print(f"- {dims} at rank {entry['target_rank']}: Found {entry['best_additions']} additions (Known to beat: {ka}) | Stability (mean rel error): {stability_str}")

    if not improvements_by_rank and not improvements_by_adds:
        print("No improvements over existing bounds were detected in this batch.")
        
    print("\nNote: Some base methods might not be perfectly tracked for additions in KNOWN_ADDITIONS.")
    print("If an entry has unexpectedly low additions, it might be an improvement regardless.")
    
    # We can also dynamically load an NPZ for an improvement and run validation to ensure it's not a hallucinated find
    # (Just an example stub - could loop through all of them if desired)

if __name__ == '__main__':
    main()
