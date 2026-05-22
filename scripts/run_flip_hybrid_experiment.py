#!/usr/bin/env python3
"""
Compare baseline ContinuousSearch vs FlipGraphHybridSearch on the same case.

Usage:
    uv run python scripts/run_flip_hybrid_experiment.py
    uv run python scripts/run_flip_hybrid_experiment.py --case 2,2,3 --rank 11
"""

import argparse
import json
import os
import time

from continuous_search import ContinuousSearch
from flip_graph_hybrid import FlipGraphHybridSearch, FlipHybridConfig


def run_experiment(
    m: int,
    p: int,
    n: int,
    rank: int,
    n_restarts: int,
    n_steps: int,
    seed: int,
    output_dir: str,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    tensor_shape = (m * p, p * n, m * n)
    # Rough CPU timings (M-series / modern laptop): ~0.5s per 1k steps at this size.
    est_baseline_s = n_restarts * (n_steps / 1000) * 0.55
    est_hybrid_s = n_restarts * (n_steps / 1000) * 0.55 + n_restarts * 15

    print("=" * 70)
    print(f"Experiment: <{m},{p},{n}> rank {rank}")
    print(f"  Tensor shape: {tensor_shape}, standard rank = {m*p*n}")
    print(f"  restarts={n_restarts}, steps/restart={n_steps}, seed={seed}")
    print(f"  Estimated wall time (CPU): baseline ~{est_baseline_s/60:.1f} min, "
          f"hybrid ~{est_hybrid_s/60:.1f} min (if most restarts need escapes)")
    print("  GPU is optional; tensor is small — CPU is usually enough.")
    print("=" * 70)

    # --- Baseline (unchanged ContinuousSearch) ---
    print("\n[1/2] Baseline gradient search (ContinuousSearch)...")
    t0 = time.time()
    baseline = ContinuousSearch(m, p, n, device="cpu")
    baseline_results = baseline.search(
        rank, n_restarts=n_restarts, n_steps=n_steps, verbose=True
    )
    baseline_elapsed = time.time() - t0
    baseline_exact = [r for r in baseline_results if r.is_exact]

    # --- Hybrid ---
    print("\n[2/2] Flip-graph hybrid search...")
    flip_config = FlipHybridConfig(
        near_exact_threshold=1.0,
        n_flip_walks=6,
        n_moves_per_walk=3,
        n_entry_flips=4,
        escape_n_steps=max(6000, n_steps // 2),
        try_snap_before_flips=True,
    )
    t0 = time.time()
    hybrid = FlipGraphHybridSearch(m, p, n, device="cpu", flip_config=flip_config)
    hybrid_results = hybrid.search(
        rank,
        n_restarts=n_restarts,
        n_steps=n_steps,
        verbose=True,
        seed=seed,
    )
    hybrid_elapsed = time.time() - t0
    hybrid_exact = [r for r in hybrid_results if r.is_exact]

    summary = {
        "case": [m, p, n],
        "rank": rank,
        "n_restarts": n_restarts,
        "n_steps": n_steps,
        "seed": seed,
        "baseline": {
            "elapsed_seconds": round(baseline_elapsed, 2),
            "n_exact": len(baseline_exact),
            "methods": list({r.method for r in baseline_exact}),
            "best_additions": (
                min(r.num_additions for r in baseline_exact)
                if baseline_exact
                else None
            ),
        },
        "hybrid": {
            "elapsed_seconds": round(hybrid_elapsed, 2),
            "n_exact": len(hybrid_exact),
            "methods": list({r.method for r in hybrid_exact}),
            "best_additions": (
                min(r.num_additions for r in hybrid_exact)
                if hybrid_exact
                else None
            ),
            "flip_config": {
                "near_exact_threshold": flip_config.near_exact_threshold,
                "n_flip_walks": flip_config.n_flip_walks,
                "n_entry_flips": flip_config.n_entry_flips,
                "escape_n_steps": flip_config.escape_n_steps,
            },
        },
    }

    out_path = os.path.join(output_dir, "comparison.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline: {summary['baseline']['n_exact']} exact in "
          f"{summary['baseline']['elapsed_seconds']}s")
    print(f"  Hybrid:   {summary['hybrid']['n_exact']} exact in "
          f"{summary['hybrid']['elapsed_seconds']}s")
    print(f"  Saved: {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="2,3,4")
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=25)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/flip_hybrid_experiment",
    )
    args = parser.parse_args()
    m, p, n = (int(x) for x in args.case.split(","))
    run_experiment(
        m, p, n, args.rank,
        args.restarts, args.steps, args.seed, args.output_dir,
    )


if __name__ == "__main__":
    main()
