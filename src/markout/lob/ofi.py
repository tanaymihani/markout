"""Order-flow imbalance (Cont, Kukanov & Stoikov 2014) and what it does and does not predict.

CKS define, for consecutive best-quote states n-1 -> n,

    e_n = qB_n 1{PB_n >= PB_{n-1}} - qB_{n-1} 1{PB_n <= PB_{n-1}}
        - qA_n 1{PA_n <= PA_{n-1}} + qA_{n-1} 1{PA_n >= PA_{n-1}},

so bid-side arrivals and ask-side cancels or executions count as buying pressure. Summed
over a 10 s bucket this is OFI_k. Their headline result is *contemporaneous*: within a
30-minute window, dP_k = alpha + beta OFI_k + eps_k explains a large share of the mid
change over the *same* bucket, and beta is inversely proportional to the depth at the
best quotes (log beta = c - lambda log D with lambda ~ 1).

A high contemporaneous R^2 is not a forecast. Knowing OFI_k requires waiting until the
end of bucket k, by which time dP_k has already happened. The honest predictive test
regresses the *next* bucket's dP_{k+1} on OFI_k, fitted before 12:45 and scored after.
Both are computed here so the gap can be reported side by side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm

from markout.lob.lobster import OPEN, SESSION, SPLIT, TICK, Day, state_index, time_average

BUCKET = 10.0      # seconds
WINDOW = 1800.0    # seconds: CKS estimate beta in 30-minute windows


def ofi_events(bid_price: np.ndarray, bid_size: np.ndarray,
               ask_price: np.ndarray, ask_size: np.ndarray) -> np.ndarray:
    """Per-event OFI e_n (shares) from the best quotes; e_0 = 0 (no previous state)."""
    pb, qb = bid_price, bid_size.astype(np.float64)
    pa, qa = ask_price, ask_size.astype(np.float64)
    e = np.zeros(len(pb))
    e[1:] = (qb[1:] * (pb[1:] >= pb[:-1]) - qb[:-1] * (pb[1:] <= pb[:-1])
             - qa[1:] * (pa[1:] <= pa[:-1]) + qa[:-1] * (pa[1:] >= pa[:-1]))
    return e


@dataclass(frozen=True)
class Buckets:
    """Per-bucket OFI and mid change on a regular clock grid."""

    start: np.ndarray       # bucket start times (s after midnight)
    ofi: np.ndarray         # shares
    dmid: np.ndarray        # mid change over the bucket, ticks
    depth: np.ndarray       # time-weighted mean of (qB + qA)/2 at the best, shares

    def __len__(self) -> int:
        return len(self.start)


def make_buckets(day: Day, width: float = BUCKET,
                 session: tuple[float, float] = SESSION) -> Buckets:
    """Sum e_n over (t_k, t_k + width] and take the mid change between the states in
    effect at the two edges, so the events counted are exactly those that moved the
    state from one edge to the next."""
    m, b = day.messages, day.book
    edges = np.arange(session[0], session[1] + width / 2, width)
    e = ofi_events(b.bid_price[:, 0], b.bid_size[:, 0], b.ask_price[:, 0], b.ask_size[:, 0])
    csum = np.concatenate([[0.0], np.cumsum(e)])
    pos = np.searchsorted(m.time, edges, side="right")      # messages with time <= edge
    ofi = csum[pos[1:]] - csum[pos[:-1]]
    mid = b.mid() / TICK
    at = state_index(m.time, edges)
    dmid = mid[at[1:]] - mid[at[:-1]]
    q = 0.5 * (b.bid_size[:, 0] + b.ask_size[:, 0])
    depth = time_average(m.time, q, edges[:-1], edges[1:])
    return Buckets(edges[:-1], ofi, dmid, depth)


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(alpha, beta, R^2) of y = alpha + beta x."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    tss = float(np.sum((y - y.mean()) ** 2))
    return float(coef[0]), float(coef[1]), 1.0 - float(resid @ resid) / tss if tss > 0 else np.nan


def window_regressions(bk: Buckets, window: float = WINDOW, origin: float = OPEN) -> list[dict]:
    """CKS regression dP = alpha + beta OFI in each clock half-hour (09:30-10:00, ...).

    The first and last windows lose the trimmed 5 minutes, so they hold 25 minutes.
    Depth is the time-weighted mean over the window (mean of the bucket depths, which
    all have equal length)."""
    wid = np.floor((bk.start - origin) / window).astype(int)
    rows = []
    for w in np.unique(wid):
        s = wid == w
        a, beta, r2 = ols(bk.ofi[s], bk.dmid[s])
        rows.append({"window": int(w), "start": float(origin + w * window), "n": int(s.sum()),
                     "alpha": a, "beta": beta, "r2": r2, "depth": float(np.nanmean(bk.depth[s]))})
    return rows


@dataclass(frozen=True)
class DepthSlope:
    slope: float            # d log beta / d log depth (CKS: about -1)
    ci: tuple[float, float] # 95%, HC3 heteroskedasticity-robust
    intercept: float
    r2: float
    n: int
    n_dropped: int          # windows with beta <= 0 (log undefined)


def beta_depth_slope(betas: np.ndarray, depths: np.ndarray,
                     groups: np.ndarray | None = None) -> DepthSlope:
    """OLS of log beta on log mean depth; with `groups`, adds one intercept per group
    (a within-ticker slope that ignores level differences between stocks)."""
    ok = (betas > 0) & (depths > 0)
    y, x = np.log(betas[ok]), np.log(depths[ok])
    if groups is None:
        X = sm.add_constant(x)
    else:
        g = np.asarray(groups)[ok]
        dummies = (g[:, None] == np.unique(g)[None, :]).astype(np.float64)
        X = np.column_stack([x, dummies])
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    j = 1 if groups is None else 0
    lo, hi = fit.conf_int(0.05)[j]
    return DepthSlope(float(fit.params[j]), (float(lo), float(hi)),
                      float(fit.params[0]) if groups is None else float("nan"),
                      float(fit.rsquared), int(ok.sum()), int((~ok).sum()))


@dataclass(frozen=True)
class OutOfSample:
    r2_oos: float           # 1 - SSE(model) / SSE(training-mean forecast), test set
    r2_oos_zero: float      # same against a zero forecast
    beta: float             # fitted on the training set
    alpha: float
    n_train: int
    n_test: int


def oos_r2(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> OutOfSample:
    a, beta, _ = ols(x[train], y[train])
    pred = a + beta * x[test]
    sse = float(np.sum((y[test] - pred) ** 2))
    sse_mean = float(np.sum((y[test] - y[train].mean()) ** 2))
    sse_zero = float(np.sum(y[test] ** 2))
    return OutOfSample(1 - sse / sse_mean, 1 - sse / sse_zero, beta, a,
                       int(train.sum()), int(test.sum()))


def contemporaneous_vs_lagged(bk: Buckets, split: float = SPLIT) -> dict[str, OutOfSample]:
    """Fit before `split`, score after: dP_k on OFI_k (contemporaneous, not tradable)
    against dP_{k+1} on OFI_k (predictive). Pairs that straddle the split are dropped."""
    end = bk.start + BUCKET
    contemp = oos_r2(bk.ofi, bk.dmid, end <= split, bk.start >= split)
    x, y = bk.ofi[:-1], bk.dmid[1:]
    lagged = oos_r2(x, y, end[1:] <= split, bk.start[:-1] >= split)
    return {"contemporaneous": contemp, "lagged": lagged}


def full_day_beta(bk: Buckets) -> tuple[float, float]:
    """Pooled CKS beta over the whole trimmed day (ticks per share) and its R^2."""
    _, beta, r2 = ols(bk.ofi, bk.dmid)
    return beta, r2
