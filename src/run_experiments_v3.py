"""
Experiment runner v3 — uses the fixed search methods.

Key changes from v2:
- Starts with validation (must reproduce known results before attempting improvements)
- Uses proper rank reduction from standard decomposition
- Adds integer simulated annealing
- Greedy finite field construction
- Better initialization from standard decomposition
- Progressive: only attempts harder problems after easier ones succeed
"""

import time
import json
import os
import sys
import numpy as np
from typing import List, Dict

from tensor_utils import KNOWN_RANKS, DecompositionResult
from fixed_search import (full_search, standard_decomposition, verify_standard,
                           ProperRankReduction, IntegerSimulatedAnnealing,
                           GreedyFiniteField, ImprovedContinuousSearch)
from validation import verify_all


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def define_progressive_experiments() -> List[Dict]:
    """
    Define experiments in a progressive order:
    1. First validate on cases we MUST solve (Strassen, known results)
    2. Then attempt sparsity improvements on validated cases
    3. Then attempt rank improvements
    
    CRITICAL: We gate harder experiments on success of easier ones.
    """
    experiments = []
    
    # ---- Tier 0: Must-solve validation (proves methods work) ----
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
            'purpose': 'validate_easy',
            'tier': 0,
            'name': name,
            'time_budget': 1800,  # 30 min max
        })
    
    # ---- Tier 1: Harder validations (proves methods scale) ----
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
            'purpose': 'validate_hard',
            'tier': 1,
            'name': name,
            'time_budget': 7200,  # 2 hour max
        })
    
    # ---- Tier 2: Sparsity optimization (practical improvements) ----
    tier2_cases = [
        (2, 2, 2, 7),
        (2, 2, 3, 11),
        (2, 2, 4, 14),
        (2, 3, 3, 15),
        (2, 2, 5, 18),
        (2, 2, 6, 21),
    ]
    for m, p, n, R in tier2_cases:
        experiments.append({
            'case': (m, p, n),
            'target_rank': R,
            'purpose': 'sparsity',
            'tier': 2,
            'name': f'minimize additions at rank {R}',
            'time_budget': 3600,
        })
    
    # ---- Tier 3: Rank improvement attempts ----
    tier3 = [
        (2, 2, 5, 17, "from known 18"),
        (2, 2, 6, 20, "from known 21"),
        (2, 2, 7, 24, "explore"),
        (2, 3, 4, 19, "from AlphaTensor 20"),
        (3, 3, 3, 22, "from known 23 — MAJOR if found"),
    ]
    for m, p, n, R, name in tier3:
        experiments.append({
            'case': (m, p, n),
            'target_rank': R,
            'purpose': 'improve',
            'tier': 3,
            'name': name,
            'time_budget': 14400,  # 4 hours
        })
    
    return experiments


def run_experiment(exp: Dict, config: Dict) -> Dict:
    """Run a single experiment and return a log entry."""
    m, p, n = exp['case']
    target_rank = exp['target_rank']
    purpose = exp['purpose']
    device = config.get('device', 'cpu')
    
    t_start = time.time()
    
    if purpose == 'sparsity':
        # For sparsity, use ImprovedContinuousSearch with sparsity bias
        # and also standard continuous search
        ics = ImprovedContinuousSearch(m, p, n, device=device)
        results = ics.search(R=target_rank, n_restarts=500,
                            n_steps=25000, verbose=config.get('verbose', True))
    else:
        # Use full multi-strategy search
        results = full_search(m, p, n, target_rank, device=device,
                             verbose=config.get('verbose', True),
                             time_budget=exp.get('time_budget', 7200))
    
    elapsed = time.time() - t_start
    
    # Verify results
    verified = []
    for r in results:
        report = verify_all(r)
        if report['verified']:
            verified.append(r)
    
    entry = {
        'case': [m, p, n],
        'target_rank': target_rank,
        'purpose': purpose,
        'tier': exp['tier'],
        'name': exp.get('name', ''),
        'n_found': len(verified),
        'elapsed_seconds': elapsed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    if verified:
        best = min(verified, key=lambda r: (r.max_coefficient, r.num_additions))
        entry['best_max_coeff'] = int(best.max_coefficient)
        entry['best_additions'] = int(best.num_additions)
        entry['best_method'] = best.method
        entry['all_additions'] = sorted(set(r.num_additions for r in verified))
        entry['all_methods'] = list(set(r.method for r in verified))
    
    return entry, verified


def batch_run_v3(config: Dict = None):
    """Progressive batch run with gating."""
    
    if config is None:
        config = {
            'device': 'cuda' if _cuda_available() else 'cpu',
            'output_dir': 'results_v3',
            'verbose': True,
        }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    experiments = define_progressive_experiments()
    
    # Load existing progress
    log_file = os.path.join(config['output_dir'], 'experiment_log.json')
    log = []
    completed = set()
    tier_success = {0: set(), 1: set(), 2: set(), 3: set()}
    
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log = json.load(f)
            for e in log:
                key = (tuple(e['case']), e['target_rank'], e['purpose'])
                completed.add(key)
                if e['n_found'] > 0:
                    tier_success[e['tier']].add(tuple(e['case']))
        except Exception:
            pass
    
    print(f"{'='*70}")
    print(f"BATCH RUN v3 — Fixed Search Methods")
    print(f"{'='*70}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Already completed: {len(completed)}")
    print(f"Device: {config['device']}")
    print()
    
    # First, verify standard decomposition works
    print("Verifying standard decomposition...")
    verify_standard()
    print()
    
    total_start = time.time()
    
    for i, exp in enumerate(experiments):
        m, p, n = exp['case']
        target_rank = exp['target_rank']
        purpose = exp['purpose']
        tier = exp['tier']
        
        key = ((m, p, n), target_rank, purpose)
        
        if key in completed:
            print(f"[{i+1}/{len(experiments)}] <{m},{p},{n}> rank {target_rank} "
                  f"({purpose}) — SKIPPING (done)")
            continue
        
        # Gating: don't attempt tier N+1 unless tier N has some successes
        if tier >= 1 and len(tier_success[tier - 1]) == 0:
            print(f"[{i+1}/{len(experiments)}] <{m},{p},{n}> rank {target_rank} "
                  f"({purpose}) — SKIPPING (tier {tier-1} has no successes yet)")
            continue
        
        # For improvements, require that validation of the same case succeeded
        if purpose == 'improve':
            known = KNOWN_RANKS.get((m, p, n))
            if known:
                val_key = ((m, p, n), known[0], 'validate_easy')
                val_key2 = ((m, p, n), known[0], 'validate_hard')
                if val_key not in completed and val_key2 not in completed:
                    # Check if we successfully validated
                    if (m, p, n) not in tier_success[0] and (m, p, n) not in tier_success[1]:
                        print(f"[{i+1}/{len(experiments)}] <{m},{p},{n}> rank {target_rank} "
                              f"({purpose}) — SKIPPING (validate first)")
                        continue
        
        print(f"\n[{i+1}/{len(experiments)}] <{m},{p},{n}> rank {target_rank} "
              f"({purpose}) [tier {tier}: {exp.get('name', '')}]")
        
        entry, verified = run_experiment(exp, config)
        
        if entry['n_found'] > 0:
            tier_success[tier].add((m, p, n))
            
            # Save decompositions
            save_dir = os.path.join(config['output_dir'],
                                     f"{m}_{p}_{n}_rank{target_rank}_{purpose}")
            os.makedirs(save_dir, exist_ok=True)
            for j, r in enumerate(verified):
                np.savez(os.path.join(save_dir, f"solution_{j}.npz"),
                         U=r.U, V=r.V, W=r.W,
                         method=r.method,
                         additions=r.num_additions,
                         max_coeff=r.max_coefficient)
            
            print(f"  ✓ Found {len(verified)} solution(s) in {entry['elapsed_seconds']:.1f}s")
            print(f"    Best: additions={entry['best_additions']}, "
                  f"method={entry['best_method']}")
            
            # Special: check for rank improvement
            known = KNOWN_RANKS.get((m, p, n))
            if known and target_rank < known[0]:
                print(f"  *** RANK IMPROVEMENT: {target_rank} < known {known[0]} ***")
                entry['is_rank_improvement'] = True
        else:
            print(f"  ✗ No solutions in {entry['elapsed_seconds']:.1f}s")
        
        log.append(entry)
        completed.add(key)
        
        # Save incrementally
        with open(log_file, 'w') as f:
            json.dump(log, f, indent=2)
    
    # Final summary
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print(f"Results by tier:")
    for tier in range(4):
        tier_exps = [e for e in log if e['tier'] == tier]
        tier_found = [e for e in tier_exps if e['n_found'] > 0]
        print(f"  Tier {tier}: {len(tier_found)}/{len(tier_exps)} successful")
    
    improvements = [e for e in log if e.get('is_rank_improvement', False)]
    if improvements:
        print(f"\nRANK IMPROVEMENTS FOUND:")
        for e in improvements:
            print(f"  <{e['case'][0]},{e['case'][1]},{e['case'][2]}> → rank {e['target_rank']}")


if __name__ == "__main__":
    config = {
        'device': 'cuda' if _cuda_available() else 'cpu',
        'output_dir': 'results_v3',
        'verbose': True,
    }
    
    if '--quick' in sys.argv:
        # Override time budgets for testing
        config['output_dir'] = 'results_v3_quick'
        # We'll still use the gating logic but with shorter budgets
    
    batch_run_v3(config)