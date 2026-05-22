"""Tests for flip-graph hybrid search helpers."""

import numpy as np

from flip_graph_hybrid import (
    FlipHybridConfig,
    apply_random_exact_move,
    generate_flip_escape_seeds,
    is_exact,
    random_exact_walk,
    reconstruction_error,
    round_factors,
    swap_rank_columns,
)
from hardcode_known import strassen_222


def test_exact_moves_preserve_strassen():
    U, V, W = strassen_222()
    m, p, n = 2, 2, 2
    assert is_exact(m, p, n, U, V, W)

    rng = np.random.default_rng(0)
    for _ in range(20):
        U2, V2, W2 = apply_random_exact_move(U, V, W, rng)
        assert is_exact(m, p, n, U2, V2, W2)

    U3, V3, W3 = random_exact_walk(U, V, W, n_moves=5, rng=rng)
    assert is_exact(m, p, n, U3, V3, W3)

    U4, V4, W4 = swap_rank_columns(U, V, W, 0, 1)
    assert is_exact(m, p, n, U4, V4, W4)


def test_generate_seeds_from_exact_includes_walks():
    U, V, W = strassen_222()
    cfg = FlipHybridConfig(n_flip_walks=3, n_entry_flips=0)
    seeds = generate_flip_escape_seeds(2, 2, 2, U, V, W, cfg, np.random.default_rng(1))
    assert len(seeds) == 1 + cfg.n_flip_walks
    for U_s, V_s, W_s in seeds:
        assert is_exact(2, 2, 2, U_s, V_s, W_s)


def test_near_exact_seeds_include_entry_flips():
    U, V, W = strassen_222()
    U_bad = U.copy()
    U_bad[0, 0] += 1
    cfg = FlipHybridConfig(
        near_exact_threshold=10.0,
        only_flip_when_near_exact=True,
        n_flip_walks=2,
        n_entry_flips=3,
    )
    U_r, V_r, W_r = round_factors(U_bad, V, W)
    err = reconstruction_error(2, 2, 2, U_r, V_r, W_r)
    assert err > 0
    seeds = generate_flip_escape_seeds(
        2, 2, 2, U_bad, V, W, cfg, np.random.default_rng(2)
    )
    assert len(seeds) >= 1 + cfg.n_entry_flips


def test_refine_from_exact_flip_seed():
    from flip_graph_hybrid import ContinuousSearchWithState, refine_from_integer_seed

    U, V, W = strassen_222()
    rng = np.random.default_rng(3)
    U2, V2, W2 = random_exact_walk(U, V, W, n_moves=2, rng=rng)
    searcher = ContinuousSearchWithState(2, 2, 2, device="cpu")
    res = refine_from_integer_seed(
        searcher, U2, V2, W2, n_steps=2000, lr=0.003, verbose=False
    )
    assert res is not None
    assert res.is_exact
    assert res.rank == 7


def test_search_tracks_best_recon_iterate():
    from flip_graph_hybrid import ContinuousSearchWithState

    searcher = ContinuousSearchWithState(2, 2, 2, device="cpu")
    result, state = searcher.search_single_with_best(
        R=7, n_steps=3000, init_method="gaussian", verbose=False
    )
    if result is not None:
        return
    assert state is not None
    assert state.step_best_recon >= 0
    assert state.recon_error < float("inf")
    assert state.U_float.shape == state.U_int.shape


def test_try_flip_escapes_from_stalled_state():
    from flip_graph_hybrid import (
        ContinuousSearchWithState,
        StallState,
        try_flip_escapes,
    )

    U, V, W = strassen_222()
    U_bad = U.astype(np.float64) + 0.05 * np.random.default_rng(0).standard_normal(U.shape)
    U_i, V_i, W_i = round_factors(U_bad, V, W)
    searcher = ContinuousSearchWithState(2, 2, 2, device="cpu")
    state = StallState(
        recon_error=float(np.sum(U_bad ** 2)),
        rounded_error=reconstruction_error(2, 2, 2, U_i, V_i, W_i),
        U_int=U_i,
        V_int=V_i,
        W_int=W_i,
        U_float=U_bad,
        V_float=V.astype(float),
        W_float=W.astype(float),
        step_best_recon=100,
    )
    cfg = FlipHybridConfig(n_flip_walks=3, escape_n_steps=4000, near_exact_threshold=5.0)
    found = try_flip_escapes(searcher, 7, state, cfg, np.random.default_rng(4), verbose=False)
    assert isinstance(found, list)
