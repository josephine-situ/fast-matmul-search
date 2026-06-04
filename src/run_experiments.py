"""
Run a systematic batch of experiments on promising targets.

This is the script you run overnight or over a weekend to
generate results across many cases.

Use --overrank for Formulation B' batch (see overrank_search.py).
"""

import time
import json
import os
import sys
import numpy as np
from typing import List, Dict, Optional

from tensor_utils import KNOWN_RANKS, DecompositionResult, build_mult_tensor, wrong_entries
from continuous_search import ContinuousSearch
from validation import verify_all
from overrank_search import compute_overrank_step_budget


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


def _load_baseline_log(output_dir: str) -> Dict[tuple, Dict]:
    path = os.path.join(output_dir, "experiment_log.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    out = {}
    for e in entries:
        key = (tuple(e["case"]), e["target_rank"], e.get("purpose", ""))
        out[key] = e
    return out


def run_overrank_experiment(
    m: int,
    p: int,
    n: int,
    target_rank: int,
    config: Dict,
    exp_meta: Dict,
) -> Dict:
    """Run Formulation B' on one case; return case-level log entry."""
    import torch

    baseline_log = _load_baseline_log(config.get("baseline_dir", "batch_results"))
    bkey = ((m, p, n), target_rank, exp_meta.get("purpose", ""))
    baseline_entry = baseline_log.get(bkey) or baseline_log.get(((m, p, n), target_rank, "validate"))
    n_found_baseline = baseline_entry["n_found"] if baseline_entry else None

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    restart_log = os.path.join(output_dir, "overrank_restarts.jsonl")

    searcher = ContinuousSearch(m, p, n, device=config.get("device", "cpu"))
    budget_mode = config.get("budget_mode", "flops_matched")
    extra_scale = config.get("extra_scale", 0.25)

    t_start = time.time()
    results, summary = searcher.search_overrank(
        R=target_rank,
        n_restarts=config.get("overrank_restarts"),
        baseline_steps=config.get("gradient_steps", 25000),
        baseline_restarts=config.get("gradient_restarts", 300),
        budget_mode=budget_mode,
        lr=config.get("gradient_lr", 0.003),
        extra_scale=extra_scale,
        verbose=config.get("verbose", True),
        restart_log_path=restart_log,
    )
    elapsed = time.time() - t_start

    verified = [r for r in results if verify_all(r)["verified"]]
    n_main, n_refine, n_snap = compute_overrank_step_budget(
        config.get("gradient_steps", 25000), budget_mode
    )

    delta = None
    if n_found_baseline is not None:
        delta = len(verified) - n_found_baseline

    entry = {
        "case": [m, p, n],
        "target_rank": target_rank,
        "purpose": exp_meta.get("purpose"),
        "tier": exp_meta.get("tier"),
        "search_mode": "overrank",
        "budget_mode": budget_mode,
        "extra_scale": extra_scale,
        "n_found": len(verified),
        "n_found_baseline": n_found_baseline,
        "delta_n_found": delta,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_steps_main": n_main,
        "n_steps_refine": n_refine,
        "n_steps_snap": n_snap,
        "n_restarts": summary.get("n_restarts"),
        "mean_gap_ratio_loo": summary.get("mean_gap_ratio_loo"),
        "unique_rounded_candidates": summary.get("unique_rounded_candidates"),
        "hardware": config.get("device", "cpu"),
    }

    tier = exp_meta.get("tier")
    if tier == 1 and delta is not None:
        entry["tier1_win"] = delta > 0
    if tier == 3 and delta is not None:
        entry["tier3_win"] = delta > 0

    if verified:
        best = min(verified, key=lambda r: (r.max_coefficient, r.num_additions))
        entry["best_max_coeff"] = int(best.max_coefficient)
        entry["best_additions"] = int(best.num_additions)
        entry["best_method"] = best.method

        save_path = os.path.join(output_dir, f"{m}_{p}_{n}_rank{target_rank}_overrank")
        os.makedirs(save_path, exist_ok=True)
        for j, r in enumerate(verified):
            np.savez(
                os.path.join(save_path, f"solution_{j}.npz"),
                U=r.U,
                V=r.V,
                W=r.W,
            )

    return entry


def batch_run_overrank(config: Dict = None):
    """Run define_experiments() queue with Formulation B'."""
    if config is None:
        config = {
            "gradient_restarts": 300,
            "gradient_steps": 25000,
            "gradient_lr": 0.003,
            "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
            "output_dir": "batch_results",
            "baseline_dir": "batch_results",
            "budget_mode": "flops_matched",
            "extra_scale": 0.25,
            "verbose": True,
        }

    os.makedirs(config["output_dir"], exist_ok=True)
    experiments = define_experiments()
    log_file = os.path.join(config["output_dir"], "overrank_experiment_log.json")

    log: List[Dict] = []
    completed = set()
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log = json.load(f)
            for e in log:
                completed.add((tuple(e["case"]), e["target_rank"], e.get("purpose")))
        except Exception as exc:
            print(f"Warning: could not load {log_file}: {exc}")

    tier0_regressions = 0
    for i, exp in enumerate(experiments):
        m, p, n = exp["case"]
        target_rank = exp["target_rank"]
        purpose = exp["purpose"]
        key = ((m, p, n), target_rank, purpose)
        if key in completed:
            print(f"[{i+1}/{len(experiments)}] <{m},{p},{n}> rank {target_rank} ({purpose}) - SKIP")
            continue

        print(f"[{i+1}/{len(experiments)}] OVER-RANK <{m},{p},{n}> rank {target_rank} tier {exp['tier']} ({purpose})")
        entry = run_overrank_experiment(m, p, n, target_rank, config, exp)
        log.append(entry)

        if exp["tier"] == 0 and entry.get("delta_n_found") is not None and entry["delta_n_found"] < 0:
            if entry.get("n_found_baseline", 0) > 0 and entry["n_found"] == 0:
                tier0_regressions += 1
                print(f"  *** Tier 0 regression ({tier0_regressions}) ***")

        delta = entry.get("delta_n_found")
        print(
            f"  n_found={entry['n_found']} baseline={entry.get('n_found_baseline')} "
            f"delta={delta} gap_loo={entry.get('mean_gap_ratio_loo')}"
        )

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

        if tier0_regressions > 1 and exp["tier"] == 0:
            print("Aborting: >1 Tier-0 regressions unresolved.")
            break

    return log


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
    if "--overrank" in sys.argv:
        if "--quick" in sys.argv:
            config = {
                "gradient_restarts": 30,
                "gradient_steps": 8000,
                "overrank_restarts": 5,
                "device": "cpu",
                "output_dir": "quick_results",
                "baseline_dir": "batch_results",
                "budget_mode": "flops_matched",
                "extra_scale": 0.25,
                "verbose": True,
            }
        elif "--steps-matched" in sys.argv:
            config = {
                "gradient_restarts": 300,
                "gradient_steps": 25000,
                "device": "cpu",
                "output_dir": "batch_results",
                "baseline_dir": "batch_results",
                "budget_mode": "steps_matched",
                "extra_scale": 0.25,
                "verbose": True,
            }
        else:
            config = None
        batch_run_overrank(config)
    elif "--quick" in sys.argv:
        config = {
            "gradient_restarts": 30,
            "gradient_steps": 8000,
            "ff_attempts": 200000,
            "primes": [2, 3],
            "device": "cpu",
            "output_dir": "quick_results",
        }
    else:
        config = None  # use defaults

    if "--overrank" not in sys.argv:
        batch_run(config)