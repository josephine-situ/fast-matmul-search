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
from typing import Dict, List, Optional, Tuple

from tensor_utils import (
    KNOWN_RANKS,
    DecompositionResult,
    build_mult_tensor,
    wrong_entries,
)
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


def _case_output_dir(output_dir: str, m: int, p: int, n: int,
                     target_rank: int) -> str:
    return os.path.join(output_dir, f"{m}_{p}_{n}_rank{target_rank}")


def _build_best_attempt_info(m: int, p: int, n: int,
                             best: DecompositionResult) -> Dict:
    """Summarize the best continuous attempt for experiment_log.json."""
    T = build_mult_tensor(m, p, n)
    t_norm_sq = float(np.sum(T ** 2))
    info = {
        'recon_error': float(best.reconstruction_error),
        'recon_error_rel': (
            float(best.reconstruction_error / t_norm_sq) if t_norm_sq > 0 else 0.0
        ),
        'is_exact': bool(best.is_exact),
        'method': best.method,
        'field': best.field,
    }
    if best.field == 'R':
        U_r = np.round(best.U)
        V_r = np.round(best.V)
        W_r = np.round(best.W)
        n_wrong, n_total = wrong_entries(T, U_r, V_r, W_r)
        info['rounded_n_wrong_entries'] = n_wrong
        info['rounded_total_entries'] = n_total
        info['rounded_pct_correct'] = 100.0 * (1 - n_wrong / n_total)
    else:
        n_wrong, n_total = wrong_entries(T, best.U, best.V, best.W)
        info['n_wrong_entries'] = n_wrong
        info['total_entries'] = n_total
        info['pct_correct'] = 100.0 * (1 - n_wrong / n_total)
        info['max_coefficient'] = int(best.max_coefficient)
        info['num_additions'] = int(best.num_additions)
    return info


def _save_best_attempt(save_path: str, best: DecompositionResult) -> str:
    """Persist the best U,V,W (continuous by default) and recon loss."""
    os.makedirs(save_path, exist_ok=True)
    filename = 'best_attempt.npz'
    filepath = os.path.join(save_path, filename)
    np.savez(
        filepath,
        U=best.U,
        V=best.V,
        W=best.W,
        reconstruction_error=float(best.reconstruction_error),
        field=best.field,
        method=best.method,
        m=best.m,
        p=best.p,
        n=best.n,
        rank=best.rank,
    )
    return filename


def run_single_experiment(m: int, p: int, n: int, target_rank: int,
                           config: Dict
                           ) -> Tuple[List[DecompositionResult],
                                      Optional[DecompositionResult]]:
    """Run all methods on a single case.

    Returns (verified_results, best_attempt) where best_attempt is the
    lowest Frobenius reconstruction loss seen on continuous factors.
    """
    all_results = []
    best_attempt = None
    integer_penalties = config.get('integer_penalties', False)

    # Method 1: Gradient search
    try:
        import torch
        searcher = ContinuousSearch(m, p, n, device=config.get('device', 'cpu'))
        grad_results, grad_best = searcher.search(
            R=target_rank,
            n_restarts=config['gradient_restarts'],
            n_steps=config['gradient_steps'],
            verbose=False,
            integer_penalties=integer_penalties,
        )
        all_results.extend(grad_results)
        if grad_best is not None:
            if (best_attempt is None
                    or grad_best.reconstruction_error
                    < best_attempt.reconstruction_error):
                best_attempt = grad_best
    except ImportError:
        pass

    verified = []
    for r in all_results:
        report = verify_all(r)
        if report['verified']:
            verified.append(r)
    return verified, best_attempt


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
            'integer_penalties': False,
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
        results, best_attempt = run_single_experiment(m, p, n, target_rank, config)
        elapsed = time.time() - t_start

        entry = {
            'case': [m, p, n],
            'target_rank': target_rank,
            'purpose': purpose,
            'n_found': len(results),
            'elapsed_seconds': elapsed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        save_path = _case_output_dir(config['output_dir'], m, p, n, target_rank)

        if best_attempt is not None:
            attempt_info = _build_best_attempt_info(m, p, n, best_attempt)
            entry['best_loss'] = attempt_info['recon_error']
            entry['best_attempt'] = attempt_info
            attempt_file = _save_best_attempt(save_path, best_attempt)
            entry['best_attempt_file'] = attempt_file

        if results:
            best = min(results, key=lambda r: (r.max_coefficient, r.num_additions))
            entry['best_max_coeff'] = int(best.max_coefficient)
            entry['best_additions'] = int(best.num_additions)
            entry['best_method'] = best.method

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
            if best_attempt is not None:
                ba = entry['best_attempt']
                rel = ba.get('recon_error_rel')
                rel_str = f", rel={rel:.3g}" if rel is not None else ""
                print(f"  No verified exact solution in {elapsed:.1f}s  |  "
                      f"best loss={entry['best_loss']:.6g}{rel_str}")
                if 'rounded_n_wrong_entries' in ba:
                    print(f"  (if rounded: {ba['rounded_n_wrong_entries']}/"
                          f"{ba['rounded_total_entries']} wrong, "
                          f"{ba['rounded_pct_correct']:.1f}% correct)")
                print(f"  Saved best attempt -> {save_path}/{entry['best_attempt_file']}")
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

    return log


if __name__ == "__main__":
    if '--quick' in sys.argv:
        config = {
            'gradient_restarts': 30,
            'gradient_steps': 8000,
            'integer_penalties': False,
            'ff_attempts': 200000,
            'primes': [2, 3],
            'device': 'cpu',
            'output_dir': 'quick_results',
        }
    elif '--integer' in sys.argv:
        config = {
            'gradient_restarts': 300,
            'gradient_steps': 25000,
            'integer_penalties': True,
            'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
            'output_dir': 'batch_results',
        }
    else:
        config = None  # use defaults

    batch_run(config)