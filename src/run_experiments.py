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

from tensor_utils import KNOWN_RANKS, DecompositionResult, build_mult_tensor, wrong_entries
from continuous_search import ContinuousSearch
from validation import verify_all


def define_experiments() -> List[Dict]:
    """
    Curated experiment targets organized by tier.

    Tier 0: Must-solve validation (proves methods work)
    Tier 1: Harder validations (proves methods scale)
    Tier 2: Sparsity optimization (practical improvements)
    Tier 3: Rank improvement attempts (the research goal)
    """
    experiments = []

    # ---- Tier 0: Must-solve validation ----
    tier0 = [
        (2, 2, 2, 7, "Strassen"),
        (2, 2, 3, 11, "Hopcroft-Kerr"),
        (2, 2, 4, 14, "known"),
        (2, 3, 3, 15, "known"),
        (2, 2, 5, 18, "known"),
    ]
    for m, p, n, R, name in tier0:
        experiments.append({
            'case': (m, p, n),
            'target_rank': R,
            'purpose': 'validate',
            'tier': 0,
            'name': name,
        })

    # ---- Tier 1: Harder validations ----
    tier1 = [
        (2, 2, 6, 21, "recursive Strassen"),
        (2, 3, 4, 20, "AlphaTensor"),
        (3, 3, 3, 23, "Smirnov/Laderman"),
        (2, 3, 5, 25, "AlphaTensor"),
        (2, 4, 4, 26, "AlphaTensor"),
    ]
    for m, p, n, R, name in tier1:
        experiments.append({
            'case': (m, p, n),
            'target_rank': R,
            'purpose': 'validate',
            'tier': 1,
            'name': name,
        })

    # ---- Tier 2: Sparsity optimization ----
    tier2 = [
        (2, 2, 2, 7),
        (2, 2, 3, 11),
        (2, 2, 4, 14),
        (2, 3, 3, 15),
        (2, 2, 5, 18),
        (2, 2, 6, 21),
    ]
    for m, p, n, R in tier2:
        experiments.append({
            'case': (m, p, n),
            'target_rank': R,
            'purpose': 'sparsity',
            'tier': 2,
            'name': f'minimize additions at rank {R}',
        })

    # ---- Tier 3: Rank improvement attempts ----
    tier3 = [
        (2, 2, 5, 17, "from known 18"),
        (2, 2, 6, 20, "from known 21"),
        (2, 2, 7, 24, "explore"),
        (2, 3, 4, 19, "from AlphaTensor 20"),
        (3, 3, 3, 22, "from known 23 -- MAJOR if found"),
    ]
    for m, p, n, R, name in tier3:
        experiments.append({
            'case': (m, p, n),
            'target_rank': R,
            'purpose': 'improve',
            'tier': 3,
            'name': name,
        })

    return experiments


def run_single_experiment(m: int, p: int, n: int, target_rank: int,
                           config: Dict):
    """Run all methods on a single case.

    Returns (verified_results, near_miss_info) where near_miss_info is a dict
    with reconstruction-error diagnostics from the best non-exact attempt
    (useful for deciding whether flip-graph search is viable).
    """
    all_results = []
    best_near_miss = None

    # Method 1: Gradient search
    try:
        import torch
        searcher = ContinuousSearch(m, p, n, device=config.get('device', 'cpu'))
        grad_results, grad_near_miss = searcher.search(
            R=target_rank,
            n_restarts=config['gradient_restarts'],
            n_steps=config['gradient_steps'],
            verbose=False
        )
        all_results.extend(grad_results)
        if grad_near_miss is not None:
            if (best_near_miss is None
                    or grad_near_miss.reconstruction_error
                    < best_near_miss.reconstruction_error):
                best_near_miss = grad_near_miss
    except ImportError:
        pass

    # Verify all results
    verified = []
    for r in all_results:
        report = verify_all(r)
        if report['verified']:
            verified.append(r)

    # Build near-miss diagnostics
    near_miss_info = None
    if best_near_miss is not None and not best_near_miss.is_exact:
        T = build_mult_tensor(m, p, n)
        n_wrong, n_total = wrong_entries(T, best_near_miss.U, best_near_miss.V,
                                         best_near_miss.W)
        near_miss_info = {
            'recon_error': float(best_near_miss.reconstruction_error),
            'n_wrong_entries': n_wrong,
            'total_entries': n_total,
            'pct_correct': 100.0 * (1 - n_wrong / n_total),
            'flip_graph_candidate': n_wrong <= max(3, int(0.1 * n_total)),
        }

    return verified, near_miss_info


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
            'ff_attempts': 1000000,
            'primes': [2, 3, 5],
            'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
            'output_dir': 'batch_results',
        }

    os.makedirs(config['output_dir'], exist_ok=True)

    experiments = define_experiments()

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
        results, near_miss_info = run_single_experiment(m, p, n, target_rank, config)
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

            save_path = os.path.join(config['output_dir'],
                                      f"{m}_{p}_{n}_rank{target_rank}")
            os.makedirs(save_path, exist_ok=True)

            for j, r in enumerate(results):
                np.savez(os.path.join(save_path, f"solution_{j}.npz"),
                         U=r.U, V=r.V, W=r.W)

            known = KNOWN_RANKS.get((m, p, n))
            if known and target_rank < known[0]:
                entry['is_improvement'] = True
                print(f"  *** IMPROVEMENT: rank {target_rank} < known {known[0]} ***")

            print(f"  Found {len(results)} solution(s) in {elapsed:.1f}s "
                  f"[best: max_coeff={best.max_coefficient}, "
                  f"method={best.method}]")
        else:
            if near_miss_info is not None:
                entry['near_miss'] = near_miss_info
                nm = near_miss_info
                tag = " *** FLIP-GRAPH CANDIDATE ***" if nm['flip_graph_candidate'] else ""
                print(f"  No exact solution in {elapsed:.1f}s  |  "
                      f"near-miss: {nm['n_wrong_entries']}/{nm['total_entries']} "
                      f"wrong ({nm['pct_correct']:.1f}% correct), "
                      f"max_err={nm['recon_error']:.4f}{tag}")
            else:
                print(f"  No solutions in {elapsed:.1f}s")

        log.append(entry)

        with open(os.path.join(config['output_dir'], 'experiment_log.json'), 'w') as f:
            json.dump(log, f, indent=2)

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

    flip_candidates = [e for e in log
                       if e.get('near_miss', {}).get('flip_graph_candidate')]
    if flip_candidates:
        print(f"\n  FLIP-GRAPH CANDIDATES ({len(flip_candidates)}):")
        for e in flip_candidates:
            nm = e['near_miss']
            print(f"    <{e['case'][0]},{e['case'][1]},{e['case'][2]}> "
                  f"rank {e['target_rank']}  "
                  f"{nm['n_wrong_entries']}/{nm['total_entries']} wrong "
                  f"({nm['pct_correct']:.1f}% correct)")

    return log


if __name__ == "__main__":
    if '--quick' in sys.argv:
        config = {
            'gradient_restarts': 30,
            'gradient_steps': 8000,
            'ff_attempts': 200000,
            'primes': [2, 3],
            'device': 'cpu',
            'output_dir': 'quick_results',
        }
    else:
        config = None  # use defaults

    batch_run(config)