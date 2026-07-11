"""Certified-pruning discrete search: recovery and exhaustion semantics."""

import numpy as np
import pytest

from polyopt.discrete_search import certified_ternary_search
from polyopt.matmul_poly import build_polymult_tensor
from tensor_utils import build_mult_tensor, verify_decomposition


def test_karatsuba_rank3_recovered():
    T = build_polymult_tensor(2, 2)
    res = certified_ternary_search(T, rank=3, sdp_prune="off")
    assert res["status"] == "found"
    U, V, W = res["solutions"][0]
    assert verify_decomposition(T, U, V, W) < 1e-8


def test_karatsuba_rank2_exhausted_no_solution():
    # rank(T) = 3, so the exhaustive ternary search must terminate with
    # zero solutions - a (re)proved theorem over the alphabet
    T = build_polymult_tensor(2, 2)
    res = certified_ternary_search(T, rank=2, sdp_prune="off",
                                   find_first=False)
    assert res["status"] == "exhausted"
    assert len(res["solutions"]) == 0


def test_matvec_rank2_recovered():
    T = build_mult_tensor(1, 2, 1)
    res = certified_ternary_search(T, rank=2, sdp_prune="off")
    assert res["status"] == "found"
    U, V, W = res["solutions"][0]
    assert verify_decomposition(T, U, V, W) < 1e-8
