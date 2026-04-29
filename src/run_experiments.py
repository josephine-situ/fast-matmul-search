"""
Run a systematic batch of experiments on promising targets.

This is the script you run overnight or over a weekend to
generate results across many cases.
"""

import time
import json
import os
import sys
import numpy as np
from typing import List, Dict

from tensor_utils import KNOWN_RANKS, DecompositionResult
from omega_analysis import select_experiments
from continuous_search import ContinuousSearch, ALSSearch
from finite_field_search import search_all_fields
from validation import verify_all
from numerical_stability import error_amplification


def run_single_experiment(m: int, p: int, n: int, target_rank: int,
                           config: Dict) -> List[DecompositionResult]:
    """Run all methods on a single case, return verified results."""
    
    all_results = []
    
    # Method 1: Gradient
    try:
        import torch
        searcher = ContinuousSearch(m, p, n, device=config.get('device', 'cpu'))
        grad_results = searcher.search(
            R=target_rank,
            n_restarts=config['gradient_restarts'],
            n_steps=config['gradient_steps'],
            verbose=False
        )
        all_results.extend(grad_results)
    except ImportError:
        pass
    
    # Method 2: ALS
    # als = ALSSearch(m, p, n)
    # als_results = als.search(
    #     R=target_rank,
    #     n_restarts=config['als_restarts'],
    #     n_steps=config['als_steps'],
    #     verbose=False
    # )
    # all_results.extend(als_results)
    
    # Method 3: Finite field (only for small cases)
    tensor_entries = (m * p) * (p * n) * (m * n)
    if tensor_entries <= 2000:
        ff_results = search_all_fields(
            m, p, n, target_rank,
            primes=config['primes'],
            n_attempts_per_prime=config['ff_attempts'],
            verbose=False
        )
        all_results.extend(ff_results)
    
    # Verify all results
    verified = []
    for r in all_results:
        report = verify_all(r)
        if report['verified']:
            verified.append(r)
    
    return verified


def batch_run(config: Dict = None):
    """
    Run experiments on all prioritized targets.
    Saves results incrementally.
    """
    if config is None:
        config = {
            'gradient_restarts': 300,
            'gradient_steps': 25000,
            'gradient_lr': 0.003,
            'als_restarts': 500,
            'als_steps': 3000,
            'ff_attempts': 1000000,
            'primes': [2, 3, 5],
            'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
            'output_dir': 'batch_results',
        }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Get prioritized experiments
    experiments = select_experiments(max_tensor_entries=5000, n_targets=8)
    
    log_file = os.path.join(config['output_dir'], 'experiment_log.json')
    log = []
    completed_cases = set()
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log = json.load(f)
            for e in log:
                completed_cases.add((tuple(e['case']), e['target_rank']))
        except Exception as e:
            print(f"Warning: could not load existing log: {e}")

    total_start = time.time()
    
    print(f"Starting batch run: {len(experiments)} experiments")
    print(f"Config: {json.dumps({k:v for k,v in config.items() if k != 'output_dir'}, indent=2)}")
    print()
    
    for i, exp in enumerate(experiments):
        m, p, n = exp['case']
        target_rank = exp['target_rank']
        purpose = exp['purpose']
        
        if ((m, p, n), target_rank) in completed_cases:
            print(f"[{i+1}/{len(experiments)}] <{m},{p},{n}> rank {target_rank} ({purpose}) - SKIPPING (already ran)")
            continue
        
        print(f"[{i+1}/{len(experiments)}] <{m},{p},{n}> rank {target_rank} ({purpose})")
        
        t_start = time.time()
        results = run_single_experiment(m, p, n, target_rank, config)
        elapsed = time.time() - t_start
        
        entry = {
            'case': [m, p, n],
            'target_rank': target_rank,
            'purpose': purpose,
            'n_found': len(results),
            'elapsed_seconds': elapsed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        if results:
            best = min(results, key=lambda r: (r.max_coefficient, r.num_additions))
            entry['best_max_coeff'] = int(best.max_coefficient)
            entry['best_additions'] = int(best.num_additions)
            entry['best_method'] = best.method
            
            # Save decomposition
            save_path = os.path.join(config['output_dir'],
                                      f"{m}_{p}_{n}_rank{target_rank}")
            os.makedirs(save_path, exist_ok=True)
            
            for j, r in enumerate(results):
                np.savez(os.path.join(save_path, f"solution_{j}.npz"),
                         U=r.U, V=r.V, W=r.W)
            
            # Check if this is an improvement
            known = KNOWN_RANKS.get((m, p, n))
            if known and target_rank < known[0]:
                entry['is_improvement'] = True
                print(f"  *** IMPROVEMENT: rank {target_rank} < known {known[0]} ***")
            
            print(f"  Found {len(results)} solution(s) in {elapsed:.1f}s "
                  f"[best: max_coeff={best.max_coefficient}, "
                  f"method={best.method}]")
        else:
            print(f"  No solutions in {elapsed:.1f}s")
        
        log.append(entry)
        
        # Save log incrementally
        with open(os.path.join(config['output_dir'], 'experiment_log.json'), 'w') as f:
            json.dump(log, f, indent=2)
    
    # Final summary
    total_elapsed = time.time() - total_start
    n_success = sum(1 for e in log if e['n_found'] > 0)
    improvements = [e for e in log if e.get('is_improvement', False)]
    
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print(f"  Experiments: {len(log)}")
    print(f"  Successful: {n_success}")
    print(f"  Improvements over known: {len(improvements)}")
    
    if improvements:
        print(f"\n  IMPROVEMENTS:")
        for e in improvements:
            print(f"    <{e['case'][0]},{e['case'][1]},{e['case'][2]}> "
                  f"rank {e['target_rank']}")
    
    return log


if __name__ == "__main__":
    # Quick test mode
    if '--quick' in sys.argv:
        config = {
            'gradient_restarts': 30,
            'gradient_steps': 8000,
            'als_restarts': 100,
            'als_steps': 2000,
            'ff_attempts': 200000,
            'primes': [2, 3],
            'device': 'cpu',
            'output_dir': 'quick_results',
        }
    else:
        config = None  # use defaults
    
    batch_run(config)