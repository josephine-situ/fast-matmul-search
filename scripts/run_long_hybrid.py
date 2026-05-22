#!/usr/bin/env python3
"""Long flip-graph hybrid run with incremental JSON logging."""

import argparse
import json
import os
import time

import numpy as np

from flip_graph_hybrid import FlipGraphHybridSearch, FlipHybridConfig
from tensor_utils import make_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="2,3,4")
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=220)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/flip_hybrid_2_3_4_long",
    )
    parser.add_argument(
        "--near-exact-threshold",
        type=float,
        default=100.0,
        help="Allow entry-flip escapes when rounded error is below this",
    )
    args = parser.parse_args()

    m, p, n = (int(x) for x in args.case.split(","))
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "progress.json")
    text_log = os.path.join(args.output_dir, "run.log")

    def log_line(msg: str) -> None:
        print(msg, flush=True)
        with open(text_log, "a") as lf:
            lf.write(msg + "\n")
            lf.flush()

    flip_config = FlipHybridConfig(
        near_exact_threshold=args.near_exact_threshold,
        n_flip_walks=8,
        n_moves_per_walk=3,
        n_entry_flips=6,
        escape_n_steps=12000,
        try_snap_before_flips=True,
        only_flip_when_near_exact=True,
    )

    est_s = args.restarts * (args.steps / 1000 * 0.55 + 18)
    log_line(f"Long hybrid: <{m},{p},{n}> rank {args.rank}")
    log_line(f"  {args.restarts} restarts × {args.steps} steps")
    log_line(f"  Estimated wall time (CPU): ~{est_s/3600:.1f} h")
    log_line(f"  Output: {args.output_dir}")
    log_line("  Watch progress.json for live updates (run.log appended each restart)")

    hybrid = FlipGraphHybridSearch(m, p, n, device="cpu", flip_config=flip_config)
    rng = np.random.default_rng(args.seed)
    init_methods = ["gaussian", "sparse", "uniform"]

    all_exact = []
    t0 = time.time()
    progress = {
        "case": [m, p, n],
        "rank": args.rank,
        "n_restarts": args.restarts,
        "n_steps": args.steps,
        "seed": args.seed,
        "flip_config": {
            "near_exact_threshold": flip_config.near_exact_threshold,
            "n_flip_walks": flip_config.n_flip_walks,
            "n_entry_flips": flip_config.n_entry_flips,
            "escape_n_steps": flip_config.escape_n_steps,
        },
        "restarts_done": 0,
        "n_exact": 0,
        "elapsed_seconds": 0,
        "findings": [],
        "last_best_recon_error": None,
        "last_best_rounded_error": None,
        "last_best_recon_step": None,
    }

    for restart in range(args.restarts):
        init = init_methods[restart % len(init_methods)]
        t_restart = time.time()
        direct, state = hybrid._searcher.search_single_with_best(
            args.rank,
            n_steps=args.steps,
            init_method=init,
            verbose=False,
        )
        found = []
        if direct is not None:
            found = [direct]
        elif state is not None:
            progress["last_best_recon_error"] = state.recon_error
            progress["last_best_rounded_error"] = state.rounded_error
            progress["last_best_recon_step"] = state.step_best_recon
            from flip_graph_hybrid import try_flip_escapes
            found = try_flip_escapes(
                hybrid._searcher, args.rank, state, flip_config, rng, verbose=False
            )
        for res in found:
            if res.is_exact:
                all_exact.append(res)
                entry = {
                    "restart": restart,
                    "summary": res.summary(),
                    "method": res.method,
                    "num_additions": res.num_additions,
                    "max_coefficient": int(res.max_coefficient),
                }
                progress["findings"].append(entry)
                log_line(f"*** FOUND at restart {restart}: {res.summary()}")

        progress["restarts_done"] = restart + 1
        progress["n_exact"] = len(all_exact)
        progress["elapsed_seconds"] = round(time.time() - t0, 1)
        with open(log_path, "w") as f:
            json.dump(progress, f, indent=2)

        restart_s = time.time() - t_restart
        recon_str = (
            f"{progress['last_best_recon_error']:.4g}@step{progress['last_best_recon_step']}"
            if progress.get("last_best_recon_error") is not None
            else "n/a"
        )
        rounded_str = (
            f"{progress['last_best_rounded_error']:.4g}"
            if progress.get("last_best_rounded_error") is not None
            else "n/a"
        )
        log_line(
            f"restart {restart + 1}/{args.restarts} done in {restart_s:.0f}s, "
            f"exact={len(all_exact)}, best_recon={recon_str}, "
            f"rounded@recon={rounded_str}"
        )

        if restart > 0 and restart % 25 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (restart + 1)
            remaining = rate * (args.restarts - restart - 1)
            log_line(
                f"  milestone [{restart + 1}/{args.restarts}] "
                f"{len(all_exact)} exact, "
                f"{elapsed/60:.1f} min elapsed, "
                f"~{remaining/60:.1f} min left"
            )

    progress["finished"] = True
    progress["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(log_path, "w") as f:
        json.dump(progress, f, indent=2)

    if all_exact:
        best = min(all_exact, key=lambda r: (r.num_additions, r.max_coefficient))
        np.savez(
            os.path.join(args.output_dir, "best_decomposition.npz"),
            U=best.U, V=best.V, W=best.W,
            m=m, p=p, n=n, rank=args.rank,
        )

    log_line(f"Done in {progress['elapsed_seconds']}s. Exact: {len(all_exact)}")
    log_line(f"Progress: {log_path}")


if __name__ == "__main__":
    main()
