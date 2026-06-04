#!/usr/bin/env python3
"""Summarize over-rank <2,2,2> restart logs for correctness and diversity checks."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hardcode_known import min_hamming_to_known, strassen_222
from tensor_utils import build_mult_tensor, verify_decomposition, wrong_entries


def load_restarts(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if r.get("case") == [2, 2, 2]]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "batch_results/overrank_restarts.jsonl"
    if not os.path.exists(path):
        print(f"No log at {path}")
        return

    rows = load_restarts(path)
    print(f"Loaded {len(rows)} restarts for <2,2,2> from {path}\n")

    baseline_path = os.path.join("batch_results", "experiment_log.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            log = json.load(f)
        for e in log:
            if e.get("case") == [2, 2, 2] and e.get("target_rank") == 7 and e.get("purpose") == "validate":
                print(f"Baseline gradient validate rank-7: n_found={e.get('n_found')}")
                break
    print()

    exact = [r for r in rows if r.get("exact_hit_verified") or r.get("exact_hit")]
    print(f"Exact (verified or legacy flag): {len(exact)}")
    if exact:
        frob = [
            r.get("recon_frobenius_snapped", r.get("recon_loss_refined"))
            for r in exact
            if r.get("recon_frobenius_snapped", r.get("recon_loss_refined")) is not None
        ]
        rel = [
            r.get("recon_frobenius_snapped_rel", r.get("recon_loss_refined_rel"))
            for r in exact
            if r.get("recon_frobenius_snapped_rel", r.get("recon_loss_refined_rel")) is not None
        ]
        if frob:
            print(f"  recon_frobenius_snapped: min={min(frob):.2e} max={max(frob):.2e}")
        if rel:
            print(f"  recon_frobenius_rel:     min={min(rel):.2e} max={max(rel):.2e}")
        else:
            print(
                "  recon_frobenius_rel: (not in log — re-run with updated overrank_search)"
            )
        hams = [r.get("hamming_to_known") for r in exact if r.get("hamming_to_known") is not None]
        if hams:
            print(f"  hamming_to_known: unique={sorted(set(hams))}")
        keys = [r.get("rounded_key") for r in exact]
        print(f"  unique rounded_key among exact: {len(set(keys))}")

    legacy_false = [
        r for r in rows
        if r.get("exact_hit") and not r.get("exact_hit_verified", r.get("exact_hit"))
    ]
    if legacy_false:
        print(f"\nLegacy exact_hit without verified reconstruction: {len(legacy_false)}")

    near = [0, 6, 21, 34]
    print("\nNear-miss restarts (requested ids):")
    T = build_mult_tensor(2, 2, 2)
    U_s, V_s, W_s = strassen_222()
    for rid in near:
        rec = next((r for r in rows if r["restart_id"] == rid), None)
        if not rec:
            print(f"  restart {rid}: missing")
            continue
        print(
            f"  restart {rid} ({rec.get('init_method')}): "
            f"n_wrong={rec.get('n_wrong_entries')} "
            f"hamming_known={rec.get('hamming_to_known')} "
            f"recon_max={rec.get('recon_max_snapped', '?')}"
        )

    r15 = next((r for r in rows if r["restart_id"] == 15), None)
    if r15:
        print("\nRestart 15 (rounded_max_coeff=2):")
        print(f"  exact_hit={r15.get('exact_hit')} verified={r15.get('exact_hit_verified')}")
        print(f"  recon_max_snapped={r15.get('recon_max_snapped')}")
        print(f"  hamming_to_known={r15.get('hamming_to_known')}")

    uniform = [r for r in rows if r.get("init_method") == "uniform" and not r.get("exact_hit_verified")]
    if uniform:
        eff = [r.get("eff_rank_final") for r in uniform if r.get("eff_rank_final") is not None]
        cancel = [r.get("cancel_pen_final") for r in uniform if r.get("cancel_pen_final") is not None]
        mass = [r.get("mass_pen_final") for r in uniform if r.get("mass_pen_final") is not None]
        print(f"\nUniform failures ({len(uniform)}): eff_rank_final mean={np.mean(eff):.2f}")
        if cancel:
            print(f"  cancel_pen_final mean={np.mean(cancel):.4f} max={np.max(cancel):.4f}")
        if mass:
            print(f"  mass_pen_final mean={np.mean(mass):.4f} max={np.max(mass):.4f}")


if __name__ == "__main__":
    main()
