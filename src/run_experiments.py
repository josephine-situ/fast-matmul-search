"""
Run a systematic batch of experiments on promising targets.

Updated with:
- Advanced search strategies for harder cases
- Sparsity optimization for cases where we can find solutions
- Better experiment selection prioritizing tractable improvements
- Seeded search from known decompositions
"""

import time
import json
import os
import sys
import numpy as np
from typing import List, Dict

from tensor_utils import KNOWN_RANKS, DecompositionResult
from omega_analysis import select_experiments
from continuous_search import ContinuousSearch
from finite_field_search import search_all_fields
from advanced_search import (SeededSearch, FactoredSearch, 
                              RankReductionSearch, ManyShortRestarts,
                              SparsityOptimizer, get_known_decomposition)
from validation import verify_all
from numerical_stability import error_amplification


def run_single_experiment(m: int, p: int, n: int, target_rank: int,
                           config: Dict) -> List[DecompositionResult]:
    """Run all methods on a single case, return verified results."""
    
    all_results = []
    device = config.get('device', 'cpu')
    tensor_entries = (m * p) * (p * n) * (m * n)
    
    # Determine difficulty tier
    is_hard = tensor_entries > 300  # <3,3,3> and above
    
    # ---- Method 1: Standard gradient (works for easier cases) ----
    if not is_hard:
        try:
            import torch
            searcher = ContinuousSearch(m, p, n, device=device)
            grad_results = searcher.search(
                R=target_rank,
                n_restarts=config['gradient_restarts'],
                n_steps=config['gradient_steps'],
                verbose=config.get('verbose', False)
            )
            all_results.extend(grad_results)
        except ImportError:
            pass
    
    # ---- Method 2: Many short restarts (for harder cases) ----
    if is_hard or not all_results:
        try:
            msr = ManyShortRestarts(m, p, n, device=device)
            msr_results = msr.search(
                R=target_rank,
                n_restarts=config.get('msr_restarts', 5000),
                n_short_steps=config.get('msr_short_steps', 3000),
                n_refinement_steps=config.get('msr_refine_steps', 15000),
                batch_size=config.get('msr_batch_size', 50),
                recon_threshold=config.get('msr_threshold', 1.0),
                verbose=config.get('verbose', False)
            )
            all_results.extend(msr_results)
        except Exception as e:
            if config.get('verbose', False):
                print(f"  Many-short-restarts failed: {e}")
    
    # ---- Method 3: Factored search (for harder cases) ----
    if is_hard or not all_results:
        try:
            fs = FactoredSearch(m, p, n, device=device)
            fs_results = fs.search(
                R=target_rank,
                n_restarts=config.get('factored_restarts', 50),
                n_outer_cycles=config.get('factored_cycles', 40),
                verbose=config.get('verbose', False)
            )
            all_results.extend(fs_results)
        except Exception as e:
            if config.get('verbose', False):
                print(f"  Factored search failed: {e}")
    
    # ---- Method 4: Rank reduction (for improvement attempts) ----
    known = KNOWN_RANKS.get((m, p, n))
    if known and target_rank < known[0]:
        try:
            rr = RankReductionSearch(m, p, n, device=device)
            rr_results = rr.search(
                target_rank=target_rank,
                start_rank=target_rank + 3,
                n_attempts=config.get('rank_reduction_attempts', 15),
                verbose=config.get('verbose', False)
            )
            all_results.extend(rr_results)
        except Exception as e:
            if config.get('verbose', False):
                print(f"  Rank reduction failed: {e}")
    
    # ---- Method 5: Seeded search (if we have a known decomposition) ----
    seed = get_known_decomposition(m, p, n)
    if seed is not None:
        U_seed, V_seed, W_seed = seed
        # Only use seeded search if target rank matches seed rank
        if U_seed.shape[1] == target_rank:
            try:
                ss = SeededSearch(m, p, n, device=device)
                ss_results = ss.search_from_seed(
                    U_seed, V_seed, W_seed,
                    n_perturbations=config.get('seeded_perturbations', 100),
                    verbose=config.get('verbose', False)
                )
                all_results.extend(ss_results)
            except Exception as e:
                if config.get('verbose', False):
                    print(f"  Seeded search failed: {e}")
    
    # ---- Method 6: Finite field (only for small cases) ----
    if tensor_entries <= 2000:
        ff_results = search_all_fields(
            m, p, n, target_rank,
            primes=config['primes'],
            n_attempts_per_prime=config['ff_attempts'],
            verbose=config.get('verbose', False)
        )
        all_results.extend(ff_results)
    
    # Verify all results
    verified = []
    for r in all_results:
        report = verify_all(r)
        if report['verified']:
            verified.append(r)
    
    return verified


def run_sparsity_experiment(m: int, p: int, n: int, rank: int,
                             config: Dict) -> List[DecompositionResult]:
    """
    Dedicated sparsity optimization: find solutions with minimum additions.
    Only run on cases where we already know rank is achievable.
    """
    device = config.get('device', 'cpu')
    all_results = []
    
    # Method A: Sparsity optimizer
    so = SparsityOptimizer(m, p, n, device=device)
    so_results = so.optimize(
        R=rank,
        n_restarts=config.get('sparsity_restarts', 500),
        n_steps=config.get('sparsity_steps', 25000),
        verbose=config.get('verbose', False)
    )
    all_results.extend(so_results)
    
    # Method B: Seeded search with sparsity focus
    seed = get_known_decomposition(m, p, n)
    if seed is not None and seed[0].shape[1] == rank:
        ss = SeededSearch(m, p, n, device=device)
        ss_results = ss.search_from_seed(
            seed[0], seed[1], seed[2],
            n_perturbations=config.get('seeded_perturbations', 200),
            optimize_sparsity=True,
            verbose=config.get('verbose', False)
        )
        all_results.extend(ss_results)
    
    # Verify
    verified = []
    for r in all_results:
        from validation import verify_all
        report = verify_all(r)
        if report['verified']:
            verified.append(r)
    
    return verified


def define_experiments(config: Dict) -> List[Dict]:
    """
    Define the full experiment list with priorities.
    
    Three types of experiments:
    1. VALIDATE: reproduce known best rank (confirms methods work)
    2. IMPROVE: try to beat known best rank (the research goal)
    3. SPARSITY: minimize additions at known rank (practical improvement)
    """
    experiments = []
    
    # ---- Track A: Cases where pipeline already works ----
    # These are the <2,2,k> family and small rectangular cases
    easy_cases = [
        # (m, p, n, known_rank, try_rank)
        (2, 2, 2, 7, 7),       # Strassen — validation
        (2, 2, 3, 11, 11),     # known — validation + sparsity
        (2, 2, 4, 14, 14),     # known — validation + sparsity
        (2, 3, 3, 15, 15),     # known — validation + sparsity
        (2, 2, 5, 18, 18),     # known — validation + sparsity
        (2, 2, 5, 18, 17),     # IMPROVEMENT ATTEMPT
        (2, 2, 6, 21, 21),     # known — validation + sparsity
        (2, 2, 6, 21, 20),     # IMPROVEMENT ATTEMPT
        (2, 2, 7, None, 24),   # explore — no well-established bound
        (2, 2, 8, 28, 27),     # IMPROVEMENT ATTEMPT
    ]
    
    for m, p, n, known_rank, try_rank in easy_cases:
        if known_rank is not None and try_rank == known_rank:
            # Validation + sparsity
            experiments.append({
                'case': (m, p, n),
                'target_rank': try_rank,
                'purpose': 'validate',
                'priority': 3.0,
                'methods': ['gradient', 'ff'],
            })
            experiments.append({
                'case': (m, p, n),
                'target_rank': try_rank,
                'purpose': 'sparsity',
                'priority': 4.0,  # higher priority — likely to produce results
                'methods': ['sparsity'],
            })
        elif known_rank is not None and try_rank < known_rank:
            # Improvement attempt
            experiments.append({
                'case': (m, p, n),
                'target_rank': try_rank,
                'purpose': 'improve',
                'priority': 5.0,
                'methods': ['gradient', 'msr', 'rank_reduction', 'ff'],
            })
        else:
            # Exploration
            experiments.append({
                'case': (m, p, n),
                'target_rank': try_rank,
                'purpose': 'explore',
                'priority': 2.0,
                'methods': ['gradient', 'msr', 'ff'],
            })
    
    # ---- Track B: Harder cases that need new methods ----
    hard_cases = [
        (2, 3, 4, 20, 20, 'validate'),   # AlphaTensor result
        (2, 3, 4, 20, 19, 'improve'),
        (3, 3, 3, 23, 23, 'validate'),   # Critical to reproduce
        (3, 3, 3, 23, 22, 'improve'),    # Open problem
        (2, 3, 5, 25, 25, 'validate'),
        (2, 3, 5, 25, 24, 'improve'),
        (2, 4, 4, 26, 26, 'validate'),
        (2, 4, 4, 26, 25, 'improve'),
    ]
    
    for entry in hard_cases:
        m, p, n, known_rank, try_rank, purpose = entry
        experiments.append({
            'case': (m, p, n),
            'target_rank': try_rank,
            'purpose': purpose,
            'priority': 4.5 if purpose == 'validate' else 5.5,
            'methods': ['msr', 'factored', 'rank_reduction', 'seeded', 'ff'],
        })
    
    # Sort by priority (higher = do first)
    experiments.sort(key=lambda e: e['priority'], reverse=True)
    
    return experiments


def batch_run(config: Dict = None):
    """
    Run experiments on all prioritized targets.
    Saves results incrementally.
    """
    if config is None:
        config = {
            # Standard gradient search
            'gradient_restarts': 300,
            'gradient_steps': 20000,
            'gradient_lr': 0.003,
            
            # Many short restarts (for hard cases)
            'msr_restarts': 5000,
            'msr_short_steps': 3000,
            'msr_refine_steps': 15000,
            'msr_batch_size': 50,
            'msr_threshold': 1.0,
            
            # Factored search
            'factored_restarts': 50,
            'factored_cycles': 40,
            
            # Rank reduction
            'rank_reduction_attempts': 15,
            
            # Seeded search
            'seeded_perturbations': 200,
            
            # Sparsity optimization
            'sparsity_restarts': 500,
            'sparsity_steps': 25000,
            
            # Finite field
            'ff_attempts': 1000000,
            'primes': [2, 3, 5],
            
            # General
            'device': 'cuda' if _cuda_available() else 'cpu',
            'output_dir': 'batch_results_v2',
            'verbose': True,
        }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Define experiments
    experiments = define_experiments(config)
    
    # Load existing progress
    log_file = os.path.join(config['output_dir'], 'experiment_log.json')
    log = []
    completed_cases = set()
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log = json.load(f)
            for e in log:
                key = (tuple(e['case']), e['target_rank'], e['purpose'])
                completed_cases.add(key)
        except Exception as e:
            print(f"Warning: could not load existing log: {e}")
    
    total_start = time.time()
    
    print(f"{'='*70}")
    print(f"BATCH RUN v2 — Advanced Search Strategies")
    print(f"{'='*70}")
    print(f"Experiments: {len(experiments)}")
    print(f"Already completed: {len(completed_cases)}")
    print(f"Device: {config['device']}")
    print(f"Config: {json.dumps({k:v for k,v in config.items() if k not in ['output_dir', 'verbose']}, indent=2)}")
    print()
    
    for i, exp in enumerate(experiments):
        m, p, n = exp['case']
        target_rank = exp['target_rank']
        purpose = exp['purpose']
        
        key = ((m, p, n), target_rank, purpose)
        if key in completed_cases:
            print(f"[{i+1}/{len(experiments)}] <{m},{p},{n}> rank {target_rank} "
                  f"({purpose}) - SKIPPING (already ran)")
            continue
        
        print(f"\n[{i+1}/{len(experiments)}] <{m},{p},{n}> rank {target_rank} "
              f"({purpose}) [priority={exp['priority']:.1f}]")
        
        t_start = time.time()
        
        if purpose == 'sparsity':
            results = run_sparsity_experiment(m, p, n, target_rank, config)
        else:
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
            
            # Collect all unique addition counts
            additions_set = sorted(set(r.num_additions for r in results))
            entry['all_addition_counts'] = additions_set
            
            # Save decompositions
            save_path = os.path.join(config['output_dir'],
                                      f"{m}_{p}_{n}_rank{target_rank}_{purpose}")
            os.makedirs(save_path, exist_ok=True)
            
            for j, r in enumerate(results):
                np.savez(os.path.join(save_path, f"solution_{j}.npz"),
                         U=r.U, V=r.V, W=r.W,
                         method=r.method,
                         additions=r.num_additions,
                         max_coeff=r.max_coefficient)
            
            # Check if this is a rank improvement
            known = KNOWN_RANKS.get((m, p, n))
            if known and target_rank < known[0]:
                entry['is_rank_improvement'] = True
                print(f"  *** RANK IMPROVEMENT: rank {target_rank} < "
                      f"known {known[0]} ***")
            
            print(f"  Found {len(results)} solution(s) in {elapsed:.1f}s")
            print(f"  Best: additions={best.num_additions}, "
                  f"max_coeff={best.max_coefficient}, "
                  f"method={best.method}")
            if len(additions_set) > 1:
                print(f"  Addition count range: {additions_set[0]} to "
                      f"{additions_set[-1]}")
        else:
            print(f"  No solutions in {elapsed:.1f}s")
        
        log.append(entry)
        
        # Save log incrementally
        with open(log_file, 'w') as f:
            json.dump(log, f, indent=2)
    
    # ---- Final summary ----
    total_elapsed = time.time() - total_start
    n_success = sum(1 for e in log if e['n_found'] > 0)
    rank_improvements = [e for e in log if e.get('is_rank_improvement', False)]
    sparsity_results = [e for e in log if e['purpose'] == 'sparsity' and e['n_found'] > 0]
    
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print(f"  Total experiments: {len(log)}")
    print(f"  Successful: {n_success}")
    print(f"  Rank improvements: {len(rank_improvements)}")
    print(f"  Sparsity optimized: {len(sparsity_results)}")
    
    if rank_improvements:
        print(f"\n  RANK IMPROVEMENTS:")
        for e in rank_improvements:
            print(f"    <{e['case'][0]},{e['case'][1]},{e['case'][2]}> "
                  f"rank {e['target_rank']} (method: {e.get('best_method', '?')})")
    
    if sparsity_results:
        print(f"\n  SPARSITY RESULTS (best addition counts):")
        for e in sparsity_results:
            print(f"    <{e['case'][0]},{e['case'][1]},{e['case'][2]}> "
                  f"rank {e['target_rank']}: {e['best_additions']} additions "
                  f"(range: {e.get('all_addition_counts', ['?'])})")
    
    return log


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    if '--quick' in sys.argv:
        config = {
            'gradient_restarts': 30,
            'gradient_steps': 8000,
            'gradient_lr': 0.003,
            'msr_restarts': 500,
            'msr_short_steps': 2000,
            'msr_refine_steps': 8000,
            'msr_batch_size': 30,
            'msr_threshold': 1.5,
            'factored_restarts': 10,
            'factored_cycles': 20,
            'rank_reduction_attempts': 5,
            'seeded_perturbations': 50,
            'sparsity_restarts': 100,
            'sparsity_steps': 15000,
            'ff_attempts': 200000,
            'primes': [2, 3],
            'device': 'cuda' if _cuda_available() else 'cpu',
            'output_dir': 'quick_results_v2',
            'verbose': True,
        }
    elif '--medium' in sys.argv:
        config = {
            'gradient_restarts': 200,
            'gradient_steps': 15000,
            'gradient_lr': 0.003,
            'msr_restarts': 3000,
            'msr_short_steps': 2500,
            'msr_refine_steps': 12000,
            'msr_batch_size': 40,
            'msr_threshold': 1.0,
            'factored_restarts': 30,
            'factored_cycles': 30,
            'rank_reduction_attempts': 10,
            'seeded_perturbations': 150,
            'sparsity_restarts': 300,
            'sparsity_steps': 20000,
            'ff_attempts': 500000,
            'primes': [2, 3, 5],
            'device': 'cuda' if _cuda_available() else 'cpu',
            'output_dir': 'medium_results_v2',
            'verbose': True,
        }
    else:
        config = None  # use defaults
    
    batch_run(config)