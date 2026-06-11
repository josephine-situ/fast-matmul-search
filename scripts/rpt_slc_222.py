#!/usr/bin/env python3
"""Run naive RPT-SLC exact solver on <2,2,2> rank 7."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rpt_slc.solve import solve_matmul_222


def main() -> None:
    parser = argparse.ArgumentParser(description="RPT-SLC exact matmul <2,2,2> rank 7")
    parser.add_argument(
        "--max-time",
        type=float,
        default=3600.0,
        help="Wall-clock limit in seconds (default 3600)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Diagnostics are always printed; flag kept for CLI compatibility",
    )
    args = parser.parse_args()

    if args.verbose:
        print("verbose: diagnostic prints enabled", flush=True)

    result = solve_matmul_222(max_time=args.max_time)

    print("\n=== summary ===", flush=True)
    for k, v in result.items():
        if k == "best_x_unit" or k == "best_x_orig":
            if v is not None:
                print(f"{k}: shape=({len(v)},)", flush=True)
            else:
                print(f"{k}: None", flush=True)
        else:
            print(f"{k}: {v}", flush=True)


if __name__ == "__main__":
    main()
