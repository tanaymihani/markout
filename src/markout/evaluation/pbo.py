"""Probability of Backtest Overfitting (PBO) by combinatorially symmetric cross-validation.

Reference
---------
Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2017). "The Probability
of Backtest Overfitting." *Journal of Computational Finance* 20(4), 39-69. SSRN 2326253.
Algorithm and section numbers refer to the author-hosted preprint
(davidhbailey.com/dhbpapers/backtest-prob.pdf).

Why this exists
---------------
DSR asks whether the *winner* is better than luck. PBO asks a different question about
the *selection procedure*: if I pick the best of my N configurations on one half of the
history, how often is that pick no better than the median configuration on the other
half? If picking the in-sample best is no better than picking at random, the search
has learned noise, however good the in-sample winner looks. It needs no distributional
assumptions, only the T x N matrix of per-period PnL of every trial, which is why the
registry keeps each trial's daily PnL.

Algorithm (CSCV, Algorithm 2.3)
-------------------------------
1. Split the T rows of M (T x N) into S contiguous blocks.
2. For each of the C(S, S/2) ways to choose S/2 blocks as in-sample (IS, "training set
   J"), the other S/2 blocks are out-of-sample (OOS, "testing set J-bar").
3. Compute the metric of every column on IS (vector R) and on OOS (R-bar), and let
   n* = argmax R.
4. OOS relative rank of the IS winner: omega = rank(R-bar[n*]) / (N + 1) in (0, 1),
   with rank 1 = worst and N = best. Logit: lambda = ln(omega / (1 - omega)).
5. PBO = share of combinations with lambda <= 0, i.e. the IS winner lands at or below
   the OOS median (Section 3.1, phi = integral of f(lambda) up to 0).

Also returned (Section 3.2): performance degradation, the OLS slope of R-bar[n*] on
R[n*] across combinations (negative slopes are typical of overfit searches), and the
probability of OOS loss, P[R-bar[n*] < 0].

Choices this implementation makes (the paper leaves them open)
--------------------------------------------------------------
* Unequal blocks: when S doesn't divide T, block sizes differ by at most one row (as in
  numpy.array_split) rather than dropping data.
* Ties in the OOS ranking use average ranks, so a strategy tied with k others gets the
  mean of their positions. Identical columns therefore share a rank, and N identical
  columns give omega = 0.5, which counts as overfit: selection added nothing.
* Ties for the IS maximum (e.g. duplicate trials) don't fall back on column order:
  omega, R[n*] and R-bar[n*] are averaged over all tied winners. That equals the
  expected outcome of breaking the tie at random.
* NaN entries mean "no observation": each metric is computed on the non-NaN rows of each
  column. A column whose metric is NaN on the IS half can't be selected, and one whose
  metric is NaN on the OOS half is left out of that combination's ranking (N then counts
  only the ranked columns). If every IS winner is NaN out of sample, the combination is
  dropped and counted in `n_dropped`. If NaN really means "flat that day", pass
  `pnl.fillna(0)` (or `Registry.pnl_matrix(fill_value=0)`) instead.
* A flat half (all zeros) has Sharpe 0 (see `dsr.sharpe`), so an IS winner that stops
  trading OOS ranks as "earned nothing" instead of disappearing.
* For odd N, the exact OOS median has lambda = 0 and counts as overfit. Under pure noise
  E[PBO] is exactly 1/2 for even N and 1/2 + 1/(2N) for odd N.

Caveats for reading the numbers
-------------------------------
* The C(S, S/2) combinations overlap heavily (each block is IS in half of them), so
  the PBO estimate is far noisier than 1/sqrt(#combinations) suggests. On pure noise
  with T = 240, N = 50 and S = 16, single-sample PBOs averaged 0.49 over 200 seeds, but
  individual ones ranged from 0.12 to 0.88 (sd 0.17). One PBO number is one draw.
* The degradation slope is negative partly by construction. IS and OOS are
  complementary halves of one fixed sample, so for any single column a good IS half
  implies a worse OOS half. With one dominant column selected in every split, the
  slope is about -1 even though nothing is overfit (PBO ~ 0). Read it next to PBO,
  never on its own.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from markout.evaluation.dsr import sharpe

Metric = Callable[[np.ndarray], float] | str
_FAST_METRICS = ("sharpe", "mean", "sum")
_TIE_RTOL = 1e-12  # relative tolerance for "tied for the IS maximum"
_CHUNK_ELEMS = 2_000_000  # bound on (combinations x strategies) per chunk, ~16 MB/array


def _resolve_metric(metric: Metric) -> str | Callable[[np.ndarray], float]:
    if metric is sharpe:
        return "sharpe"
    if isinstance(metric, str):
        if metric not in _FAST_METRICS:
            raise ValueError(f"metric must be one of {_FAST_METRICS} or a callable, got {metric!r}")
        return metric
    if callable(metric):
        return metric
    raise TypeError(f"metric must be a callable or a string, got {type(metric).__name__}")


def _block_edges(T: int, S: int) -> np.ndarray:
    """Row boundaries of S contiguous blocks whose sizes differ by at most one."""
    sizes = np.full(S, T // S)
    sizes[: T % S] += 1
    return np.concatenate([[0], np.cumsum(sizes)])


def _combinations(S: int) -> np.ndarray:
    """Boolean (C(S, S/2) x S) membership matrix: row c marks the IS blocks of combo c."""
    combos = np.array(list(itertools.combinations(range(S), S // 2)))
    A = np.zeros((len(combos), S), dtype=bool)
    A[np.repeat(np.arange(len(combos)), S // 2), combos.ravel()] = True
    return A


class _BlockStats:
    """Per-block sufficient statistics that give any union of blocks' metrics exactly.

    Sums are taken on column-centered values (x - column mean) so that the variance
    formula (sum(z^2) - sum(z)^2 / n) / (n - 1) doesn't suffer cancellation when a
    column's mean is large relative to its spread. Block minima and maxima let a
    constant half be detected exactly, the same way `dsr.sharpe` does it.
    """

    def __init__(self, X: np.ndarray, edges: np.ndarray) -> None:
        valid = ~np.isnan(X)
        has_any = valid.any(axis=0)
        center = np.zeros(X.shape[1])
        center[has_any] = np.nanmean(X[:, has_any], axis=0)
        Z = np.where(valid, X - center, 0.0)
        starts = edges[:-1]
        self.center = center
        self.cnt = np.add.reduceat(valid.astype(float), starts, axis=0)  # S x N
        self.s1 = np.add.reduceat(Z, starts, axis=0)
        self.s2 = np.add.reduceat(Z * Z, starts, axis=0)
        self.bmin = np.minimum.reduceat(np.where(valid, X, np.inf), starts, axis=0)
        self.bmax = np.maximum.reduceat(np.where(valid, X, -np.inf), starts, axis=0)

    def metric(self, A: np.ndarray, kind: str) -> np.ndarray:
        """Metric of every column on the union of blocks marked in each row of A."""
        Af = A.astype(float)
        n = Af @ self.cnt
        s1 = Af @ self.s1
        with np.errstate(invalid="ignore", divide="ignore"):
            if kind == "sum":
                return np.where(n > 0, s1 + self.center * n, np.nan)
            mean = s1 / n + self.center
            if kind == "mean":
                return np.where(n > 0, mean, np.nan)
            # Sharpe, with the degenerate cases of dsr.sharpe decided exactly.
            s2 = Af @ self.s2
            var = (s2 - s1 * s1 / n) / (n - 1)
            hmin = np.where(A[:, :, None], self.bmin[None], np.inf).min(axis=1)
            hmax = np.where(A[:, :, None], self.bmax[None], -np.inf).max(axis=1)
            const = hmax == hmin
            out = mean / np.sqrt(var)
            out = np.where(var > 0, out, np.nan)
            out = np.where(const, np.where(hmax == 0.0, 0.0, np.nan), out)
            return np.where(n >= 2, out, np.nan)


def _generic_metric(X: np.ndarray, rows: np.ndarray, metric: Callable[[np.ndarray], float]) -> np.ndarray:
    """Evaluate a user metric on the given rows of every column (NaNs dropped first)."""
    sub = X[rows]
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        col = sub[:, j]
        col = col[~np.isnan(col)]
        if col.size:
            out[j] = float(metric(col))
    return out


def _select_and_rank(R_is: np.ndarray, R_oos: np.ndarray) -> dict[str, np.ndarray]:
    """Steps 3-4 of CSCV for a chunk of combinations (rows)."""
    is_ok = ~np.isnan(R_is)
    m = np.where(is_ok, R_is, -np.inf).max(axis=1)
    tol = np.where(np.isfinite(m), _TIE_RTOL * np.abs(m), 0.0)
    winners = is_ok & (R_is >= (m - tol)[:, None])

    oos_ok = ~np.isnan(R_oos)
    ranks = stats.rankdata(R_oos, method="average", axis=1, nan_policy="omit")
    rel = ranks / (oos_ok.sum(axis=1, keepdims=True) + 1.0)

    w = winners & oos_ok
    k = w.sum(axis=1)
    ok = k > 0
    kk = np.maximum(k, 1)
    omega = np.where(ok, np.where(w, rel, 0.0).sum(axis=1) / kk, np.nan)
    oos_sel = np.where(ok, np.where(w, R_oos, 0.0).sum(axis=1) / kk, np.nan)
    is_sel = np.where(ok, m, np.nan)
    credit = np.where(w, 1.0 / kk[:, None], 0.0)  # tied winners share one selection
    return {"omega": omega, "is_sel": is_sel, "oos_sel": oos_sel, "credit": credit.sum(axis=0)}


def pbo(pnl: pd.DataFrame | np.ndarray, n_blocks: int = 16, metric: Metric = sharpe) -> dict[str, Any]:
    """Probability of Backtest Overfitting of "pick the best column", by CSCV.

    Parameters
    ----------
    pnl : T x N matrix of per-period PnL (rows in time order, one column per trial),
        e.g. ``Registry.pnl_matrix(fill_value=0)``. NaN = no observation (see module doc).
    n_blocks : S, the even number of contiguous blocks. The default 16 gives
        C(16, 8) = 12,870 IS/OOS splits; T = 240 days gives 15-day blocks.
    metric : selection criterion, where higher is better. The default `dsr.sharpe`, and
        the strings "sharpe", "mean" and "sum", use an exact vectorized path (per-block
        sufficient statistics), which handles S = 16 with hundreds of trials in about a
        second. Any other callable f(1-D array) -> float is evaluated column by column,
        which is correct but about 1000x slower.

    Returns
    -------
    dict with
        pbo                share of combinations with logit <= 0
        logits             lambda_c for each used combination (in itertools order)
        oos_rank           omega_c, the OOS relative rank of the IS winner, in (0, 1)
        is_metric, oos_metric   the selected strategy's metric IS and OOS per combination
        degradation_slope, degradation_intercept   OLS of oos_metric on is_metric
        prob_oos_loss      share of combinations with oos_metric < 0
        selection_freq     pd.Series: how often each column is the IS winner (sums to 1)
        n_combinations, n_used, n_dropped, n_blocks, block_sizes, n_strategies, T
    """
    if isinstance(pnl, pd.DataFrame):
        columns = list(pnl.columns)
        X = pnl.to_numpy(dtype=float, na_value=np.nan)
    else:
        X = np.asarray(pnl, dtype=float)
        columns = list(range(X.shape[1])) if X.ndim == 2 else []
    if X.ndim != 2:
        raise ValueError(f"pnl must be a 2-D (T x N) matrix, got shape {X.shape}")
    T, N = X.shape
    S = int(n_blocks)
    if S < 2 or S % 2:
        raise ValueError(f"n_blocks must be an even integer >= 2, got {n_blocks}")
    if T < S:
        raise ValueError(f"need at least one row per block: T={T} < n_blocks={S}")
    if N < 2:
        raise ValueError(f"PBO needs at least 2 strategies to select among, got {N}")
    if np.isinf(X).any():
        raise ValueError("pnl contains +/-inf")

    kind = _resolve_metric(metric)
    edges = _block_edges(T, S)
    A = _combinations(S)
    C = A.shape[0]
    chunk = max(1, _CHUNK_ELEMS // max(N * (S if kind == "sharpe" else 1), 1))

    omega = np.empty(C)
    is_sel = np.empty(C)
    oos_sel = np.empty(C)
    credit = np.zeros(N)
    if isinstance(kind, str):
        bs = _BlockStats(X, edges)
    else:
        block_rows = [np.arange(edges[b], edges[b + 1]) for b in range(S)]

    for lo in range(0, C, chunk):
        Ac = A[lo:lo + chunk]
        if isinstance(kind, str):
            R_is, R_oos = bs.metric(Ac, kind), bs.metric(~Ac, kind)
        else:
            R_is = np.empty((len(Ac), N))
            R_oos = np.empty((len(Ac), N))
            for i, row in enumerate(Ac):
                is_rows = np.concatenate([block_rows[b] for b in np.flatnonzero(row)])
                oos_rows = np.concatenate([block_rows[b] for b in np.flatnonzero(~row)])
                R_is[i] = _generic_metric(X, is_rows, kind)
                R_oos[i] = _generic_metric(X, oos_rows, kind)
        res = _select_and_rank(R_is, R_oos)
        omega[lo:lo + chunk] = res["omega"]
        is_sel[lo:lo + chunk] = res["is_sel"]
        oos_sel[lo:lo + chunk] = res["oos_sel"]
        credit += res["credit"]

    used = ~np.isnan(omega)
    omega, is_sel, oos_sel = omega[used], is_sel[used], oos_sel[used]
    # Averaging tied ranks can land a hair off 0.5; snap so that logit == 0 exactly.
    omega = np.where(np.abs(omega - 0.5) <= 1e-12, 0.5, omega)
    logits = np.log(omega / (1.0 - omega))

    n_used = int(used.sum())
    slope = intercept = float("nan")
    if n_used >= 2 and np.ptp(is_sel) > 0:
        slope, intercept = (float(v) for v in np.polyfit(is_sel, oos_sel, 1))

    freq = credit / credit.sum() if credit.sum() > 0 else credit
    return {
        "pbo": float(np.mean(logits <= 0)) if n_used else float("nan"),
        "logits": logits,
        "oos_rank": omega,
        "is_metric": is_sel,
        "oos_metric": oos_sel,
        "degradation_slope": slope,
        "degradation_intercept": intercept,
        "prob_oos_loss": float(np.mean(oos_sel < 0)) if n_used else float("nan"),
        "selection_freq": pd.Series(freq, index=columns, name="selection_freq"),
        "n_combinations": int(C),
        "n_used": n_used,
        "n_dropped": int(C - n_used),
        "n_blocks": S,
        "block_sizes": np.diff(edges).tolist(),
        "n_strategies": int(N),
        "T": int(T),
    }
