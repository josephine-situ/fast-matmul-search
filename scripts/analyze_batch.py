"""
Analyze batch results, compare against known bounds,
and identify improvements.
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List, Tuple
from tensor_utils import KNOWN_RANKS, count_additions


# Known addition counts from literature (where available)
KNOWN_ADDITIONS: Dict[Tuple[int, int, int], Dict[int, Tuple[int, str]]] = {
    # Format: (m,p,n) -> {rank: (best_known_additions, source)}
    (2, 2, 2): {7: (18, "Strassen — U has 10 nnz, V has 10 nnz, W has 12 nnz")},
    (2, 2, 3): {11: (28, "Hopcroft-Kerr — approximate, verify")},
    (2, 2, 4): {14: (38, "approximate from literature")},
    (2, 2, 5): {18: (56, "approximate")},
    (2, 3, 3): {15: (42, "approximate from literature")},
}


def load_results(output_dir: str) -> List[Dict]:
    """Load experiment log."""
    log_file = os.path.join(output_dir, 'experiment_log.json')
    if not os.path.exists(log_file):
        print(f"No log file found at {log_file}")
        return []
    
    with open(log_file, 'r') as f:
        return json.load(f)


def load_decompositions(output_dir: str) -> Dict[str, List]:
    """Load all saved decompositions."""
    decomps = {}
    
    for dirname in os.listdir(output_dir):
        dirpath = os.path.join(output_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
        
        for filename in os.listdir(dirpath):
            if filename.endswith('.npz'):
                filepath = os.path.join(dirpath, filename)
                data = np.load(filepath, allow_pickle=True)
                
                key = dirname
                if key not in decomps:
                    decomps[key] = []
                
                decomps[key].append({
                    'U': data['U'],
                    'V': data['V'],
                    'W': data['W'],
                    'method': str(data.get('method', 'unknown')),
                    'additions': int(data.get('additions', -1)),
                    'max_coeff': int(data.get('max_coeff', -1)),
                })
    
    return decomps


def full_analysis(output_dir: str = 'batch_results_v2'):
    """Complete analysis of all results."""
    
    log = load_results(output_dir)
    if not log:
        return
    
    print(f"{'='*70}")
    print(f"FULL ANALYSIS OF RESULTS")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Total experiments in log: {len(log)}")
    print()
    
    # Categorize results
    validations = [e for e in log if e['purpose'] == 'validate']
    improvements = [e for e in log if e['purpose'] == 'improve']
    sparsity = [e for e in log if e['purpose'] == 'sparsity']
    explorations = [e for e in log if e['purpose'] == 'explore']
    
    # ---- Validation summary ----
    print(f"VALIDATION RESULTS ({len(validations)} experiments)")
    print(f"-" * 50)
    for e in validations:
        case_str = f"<{e['case'][0]},{e['case'][1]},{e['case'][2]}>"
        status = "PASS" if e['n_found'] > 0 else "FAIL"
        detail = ""
        if e['n_found'] > 0:
            detail = f" [additions={e.get('best_additions', '?')}, method={e.get('best_method', '?')}]"
        print(f"  {case_str} rank {e['target_rank']}: {status}{detail}")
    
    # ---- Improvement attempts ----
    print(f"\nIMPROVEMENT ATTEMPTS ({len(improvements)} experiments)")
    print(f"-" * 50)
    for e in improvements:
        case_str = f"<{e['case'][0]},{e['case'][1]},{e['case'][2]}>"
        known = KNOWN_RANKS.get(tuple(e['case']))
        known_rank = known[0] if known else '?'
        
        if e['n_found'] > 0:
            print(f"  {case_str} rank {e['target_rank']} "
                  f"(known: {known_rank}): *** SUCCESS ***")
        else:
            print(f"  {case_str} rank {e['target_rank']} "
                  f"(known: {known_rank}): not found")
    
    # ---- Sparsity results ----
    print(f"\nSPARSITY OPTIMIZATION ({len(sparsity)} experiments)")
    print(f"-" * 50)
    for e in sparsity:
        if e['n_found'] == 0:
            continue
        case_str = f"<{e['case'][0]},{e['case'][1]},{e['case'][2]}>"
        case_key = tuple(e['case'])
        rank = e['target_rank']
        
        # Compare against known
        known_add = KNOWN_ADDITIONS.get(case_key, {}).get(rank)
        improvement_str = ""
        if known_add:
            known_val, source = known_add
            if e['best_additions'] < known_val:
                improvement_str = f" *** IMPROVEMENT (was {known_val}) ***"
            else:
                improvement_str = f" (known: {known_val})"
        
        print(f"  {case_str} rank {rank}: "
              f"best_additions={e['best_additions']}"
              f"{improvement_str}")
        if 'all_addition_counts' in e:
            print(f"    All found: {e['all_addition_counts']}")
    
    # ---- Explorations ----
    if explorations:
        print(f"\nEXPLORATIONS ({len(explorations)} experiments)")
        print(f"-" * 50)
        for e in explorations:
            case_str = f"<{e['case'][0]},{e['case'][1]},{e['case'][2]}>"
            if e['n_found'] > 0:
                print(f"  {case_str} rank {e['target_rank']}: "
                      f"FOUND ({e['n_found']} solutions)")
            else:
                print(f"  {case_str} rank {e['target_rank']}: not found")
    
    # ---- Method effectiveness ----
    print(f"\nMETHOD EFFECTIVENESS")
    print(f"-" * 50)
    method_counts = {}
    for e in log:
        if e['n_found'] > 0:
            method = e.get('best_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
    
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count} successful experiments")
    
    # ---- Timing analysis ----
    print(f"\nTIMING")
    print(f"-" * 50)
    total_time = sum(e.get('elapsed_seconds', 0) for e in log)
    print(f"  Total compute time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    
    successful_times = [e['elapsed_seconds'] for e in log if e['n_found'] > 0]
    failed_times = [e['elapsed_seconds'] for e in log if e['n_found'] == 0]
    
    if successful_times:
        print(f"  Successful experiments: mean {np.mean(successful_times):.0f}s, "
              f"median {np.median(successful_times):.0f}s")
    if failed_times:
        print(f"  Failed experiments: mean {np.mean(failed_times):.0f}s, "
              f"median {np.median(failed_times):.0f}s")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'batch_results_v2'
    full_analysis(output_dir)