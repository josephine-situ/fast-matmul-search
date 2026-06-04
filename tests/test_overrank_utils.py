"""Tests for over-rank search helpers."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hardcode_known import min_hamming_to_known, strassen_222
from overrank_search import (
    _init_method_for_restart,
    _max_consecutive_gap_ratio,
    _try_one_flip_exact,
    compute_overrank_step_budget,
)
from tensor_utils import build_mult_tensor, verify_decomposition


def test_max_consecutive_gap_ratio_full_vector():
    vals = np.array([1e-6, 1e-3, 0.5, 0.9, 2.0])
    # Largest jump: 1e-3 / 1e-6 = 1000
    assert _max_consecutive_gap_ratio(vals) == 1000.0


def test_strassen_hamming_zero():
    U, V, W = strassen_222()
    assert min_hamming_to_known(2, 2, 2, U, V, W) == 0


def test_init_method_two_thirds_gaussian():
    methods = [_init_method_for_restart(i, 30) for i in range(30)]
    assert methods.count("gaussian") == 20
    assert methods.count("sparse") == 5
    assert methods.count("uniform") == 5


def test_discover_budget_allocates_more_refine_than_flops_matched():
    flops = compute_overrank_step_budget(25000, "flops_matched")
    discover = compute_overrank_step_budget(25000, "discover")
    assert discover[0] + discover[1] + discover[2] == flops[0] + flops[1] + flops[2]
    assert discover[1] > flops[1]
    assert discover[2] > flops[2]
    assert discover[0] < flops[0]


def test_one_flip_can_fix_single_entry_error():
    U, V, W = strassen_222()
    U_bad = U.copy()
    U_bad[0, 0] = 0
    T = build_mult_tensor(2, 2, 2)
    U_fix, V_fix, W_fix, ok = _try_one_flip_exact(T, U_bad, V, W, 7)
    assert ok
    assert verify_decomposition(T, U_fix.astype(float), V_fix.astype(float), W_fix.astype(float)) < 1e-10
