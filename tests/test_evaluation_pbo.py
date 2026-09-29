"""Tests for markout.evaluation.pbo (CSCV, Bailey, Borwein, López de Prado & Zhu 2017).

The PBO paper has no worked example with published data to reproduce, so the checks are:
a hand-computed case, agreement with a naive loop-over-combinations reference, and the
two limiting behaviours (pure noise -> ~0.5, one real edge -> ~0).
"""

import itertools
import math
import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from markout.evaluation.dsr import sharpe
from markout.evaluation.pbo import pbo


def _nanmean(v: np.ndarray) -> float:
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def _reference_pbo(X: np.ndarray, S: int, metric) -> dict:
    """Direct, loop-based CSCV, written independently of the vectorized implementation."""
    T, N = X.shape
    sizes = [T // S + (1 if i < T % S else 0) for i in range(S)]
    edges = np.cumsum([0] + sizes)
    blocks = [list(range(edges[i], edges[i + 1])) for i in range(S)]
    omegas, is_m, oos_m = [], [], []
    for combo in itertools.combinations(range(S), S // 2):
        is_rows = sum((blocks[b] for b in combo), [])
        oos_rows = sum((blocks[b] for b in range(S) if b not in combo), [])
        R = np.array([metric(X[is_rows, j]) for j in range(N)])
        Rb = np.array([metric(X[oos_rows, j]) for j in range(N)])
        best = np.nanmax(R)
        ok = ~np.isnan(Rb)
        ranks = np.full(N, np.nan)
        ranks[ok] = stats.rankdata(Rb[ok])
        winners = [j for j in range(N) if R[j] == best and ok[j]]
        if not winners:
            continue
        omegas.append(np.mean([ranks[j] / (ok.sum() + 1) for j in winners]))
        is_m.append(best)
        oos_m.append(np.mean([Rb[j] for j in winners]))
    om = np.array(omegas)
    return {"pbo": float(np.mean(np.log(om / (1 - om)) <= 0)), "oos_rank": om,
            "is_metric": np.array(is_m), "oos_metric": np.array(oos_m)}


def test_hand_computed_two_block_example():
    # Blocks: rows 0-1 (IS in combo 1) and rows 2-3. Metric = mean.
    X = np.array([[1, 0, 2],
                  [1, 0, 2],
                  [0, 3, 1],
                  [0, 3, 1]], dtype=float)
    r = pbo(X, n_blocks=2, metric="mean")
    # combo 1: IS means (1, 0, 2) -> col 2 wins; OOS means (0, 3, 1) -> rank 2 of 3 -> omega 2/4
    # combo 2: IS means (0, 3, 1) -> col 1 wins; OOS means (1, 0, 2) -> rank 1 of 3 -> omega 1/4
    np.testing.assert_allclose(r["oos_rank"], [0.5, 0.25])
    np.testing.assert_allclose(r["logits"], [0.0, math.log(1 / 3)])
    assert r["pbo"] == 1.0                           # lambda <= 0 in both combinations
    np.testing.assert_allclose(r["is_metric"], [2.0, 3.0])
    np.testing.assert_allclose(r["oos_metric"], [1.0, 0.0])
    assert r["degradation_slope"] == pytest.approx(-1.0)
    assert r["degradation_intercept"] == pytest.approx(3.0)
    assert r["prob_oos_loss"] == 0.0                 # OOS means 1 and 0, neither < 0
    np.testing.assert_allclose(r["selection_freq"].to_numpy(), [0.0, 0.5, 0.5])
    assert r["n_combinations"] == 2 and r["n_used"] == 2


def _messy_matrix(seed: int, T: int = 124, N: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, N)) + 0.04 * np.arange(N)
    X[rng.random(X.shape) < 0.08] = np.nan           # scattered missing days
    X[: T // 2, 3] = 0.0                             # flat for the whole first half...
    X[T // 2:, 5] = np.nan                           # ...and no data for a whole half
    X[: T // 2, 6] = 0.25                            # constant non-zero for a whole half
    return X


@pytest.mark.parametrize("S", [2, 4, 8])
@pytest.mark.parametrize("metric", ["sharpe", "mean"])
def test_vectorized_matches_naive_reference(S, metric):
    X = _messy_matrix(seed=S)
    fn = sharpe if metric == "sharpe" else _nanmean
    ref = _reference_pbo(X, S, fn)
    got = pbo(X, n_blocks=S, metric=sharpe if metric == "sharpe" else "mean")
    assert got["pbo"] == ref["pbo"]
    np.testing.assert_allclose(got["oos_rank"], ref["oos_rank"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["is_metric"], ref["is_metric"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got["oos_metric"], ref["oos_metric"], rtol=1e-9, atol=1e-12)


def test_generic_callable_path_matches_fast_path():
    X = _messy_matrix(seed=11)
    fast = pbo(X, n_blocks=4)                                   # dsr.sharpe -> fast path
    slow = pbo(X, n_blocks=4, metric=lambda v: sharpe(v))       # opaque callable
    np.testing.assert_allclose(slow["oos_rank"], fast["oos_rank"], atol=1e-12)
    np.testing.assert_allclose(slow["is_metric"], fast["is_metric"], rtol=1e-9)
    assert slow["pbo"] == fast["pbo"]
    total = pbo(X, n_blocks=4, metric="sum")
    total_ref = _reference_pbo(X, 4, lambda v: np.nansum(v) if (~np.isnan(v)).any() else np.nan)
    np.testing.assert_allclose(total["oos_rank"], total_ref["oos_rank"], atol=1e-12)


def test_pure_noise_gives_pbo_near_one_half():
    """Selecting among 50 zero-skill strategies is a coin flip OOS. A single PBO is very
    noisy (sd ~0.17 across seeds, measured over 200 seeds), so average 30 independent
    matrices: SE ~0.03, and the tolerance of +/-0.12 is ~4 SE."""
    vals, ranks = [], []
    for seed in range(30):
        X = np.random.default_rng(seed).standard_normal((240, 50))
        r = pbo(X, n_blocks=16)
        assert r["n_used"] == r["n_combinations"] == 12_870
        vals.append(r["pbo"])
        ranks.append(r["oos_rank"].mean())
    assert abs(np.mean(vals) - 0.5) < 0.12
    assert abs(np.mean(ranks) - 0.5) < 0.06          # OOS rank of the IS winner ~ uniform


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_real_edge_among_noise_gives_pbo_near_zero(seed):
    rng = np.random.default_rng(100 + seed)
    X = rng.standard_normal((240, 50))
    X[:, 17] += 0.5                                 # daily Sharpe 0.5: a real, strong edge
    r = pbo(pd.DataFrame(X, columns=[f"trial{j}" for j in range(50)]), n_blocks=16)
    assert r["pbo"] < 0.05
    assert r["selection_freq"]["trial17"] > 0.95
    assert r["prob_oos_loss"] < 0.05
    assert r["selection_freq"].sum() == pytest.approx(1.0)


def test_sixteen_blocks_is_fast():
    X = np.random.default_rng(0).standard_normal((240, 50))
    t0 = time.perf_counter()
    r = pbo(X, n_blocks=16)
    elapsed = time.perf_counter() - t0
    assert r["n_combinations"] == math.comb(16, 8) == 12_870
    assert r["block_sizes"] == [15] * 16
    assert elapsed < 10.0


def test_ties_and_duplicates():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((240, 10))
    X[:, 2] += 0.6
    X = np.column_stack([X, X[:, 2]])               # column 10 duplicates column 2
    r = pbo(X, n_blocks=8)
    # tied IS winners share credit instead of the lower column index taking it all
    assert r["selection_freq"].iloc[2] == pytest.approx(0.5)
    assert r["selection_freq"].iloc[10] == pytest.approx(0.5)
    # the tied pair holds the top two OOS ranks: average rank 10.5 of 11 -> omega 10.5/12
    np.testing.assert_allclose(r["oos_rank"], 10.5 / 12)
    assert r["pbo"] == 0.0

    # All strategies identical: selection adds nothing, omega = 0.5, counted as overfit.
    same = np.tile(rng.standard_normal((80, 1)), (1, 5))
    r = pbo(same, n_blocks=4)
    np.testing.assert_allclose(r["oos_rank"], 0.5)
    assert r["pbo"] == 1.0 and np.all(r["logits"] == 0.0)


def test_all_nan_column_changes_nothing_but_the_count():
    X = np.random.default_rng(8).standard_normal((120, 6))
    with_nan = np.column_stack([X, np.full(120, np.nan)])
    a, b = pbo(X, n_blocks=8), pbo(with_nan, n_blocks=8)
    np.testing.assert_allclose(a["oos_rank"], b["oos_rank"])
    assert a["pbo"] == b["pbo"] and b["selection_freq"].iloc[-1] == 0.0


def test_is_winner_that_goes_flat_oos_is_a_flop_not_a_drop():
    rng = np.random.default_rng(6)
    X = 0.5 + rng.standard_normal((100, 5))          # the others: steady, clearly positive
    X[:50, 0] = 2.0 + 0.1 * rng.standard_normal(50)   # superb in the first half...
    X[50:, 0] = 0.0                                   # ...then stops trading
    r = pbo(X, n_blocks=2)
    assert r["n_used"] == 2                           # the flat OOS half was ranked, not dropped
    assert r["oos_rank"][0] == pytest.approx(1 / 6)   # Sharpe 0 ranks last among 5
    assert r["oos_metric"][0] == 0.0


def test_input_validation():
    X = np.random.default_rng(0).standard_normal((40, 4))
    with pytest.raises(ValueError):
        pbo(X, n_blocks=5)                            # odd
    with pytest.raises(ValueError):
        pbo(X[:3], n_blocks=4)                        # fewer rows than blocks
    with pytest.raises(ValueError):
        pbo(X[:, :1], n_blocks=4)                     # nothing to select among
    bad = X.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        pbo(bad, n_blocks=4)
    with pytest.raises(ValueError):
        pbo(X, n_blocks=4, metric="sortino")
