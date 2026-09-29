"""Stationary block bootstrap confidence intervals for the Sharpe ratio and the mean.

References
----------
Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap." *Journal of the
American Statistical Association* 89(428), 1303-1313.
Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection for the Dependent
Bootstrap." *Econometric Reviews* 23(1), 53-70; correction by Patton, Politis & White
(2009), *Econometric Reviews* 28(4), 372-375.

Why a block bootstrap
---------------------
Daily strategy PnL is serially dependent (volatility clustering, regime persistence,
and positions that carry signal across days). An iid bootstrap shuffles days
independently, destroys that dependence, and understates the sampling variance of the
mean and the Sharpe ratio when autocorrelation is positive, so its intervals are too
narrow. The stationary bootstrap resamples blocks of random, geometrically distributed
length (mean `block`) starting at random days, wrapping around the end of the sample.
That keeps short-range dependence inside blocks, and unlike fixed-length blocks the
resampled series is itself stationary. When `block` is None, the expected block length
is chosen from the data by the Politis-White rule.

Resampling indices come from `arch.bootstrap.StationaryBootstrap`. The statistic is
evaluated vectorized over all replications, and the interval is the percentile interval
[q(alpha/2), q(1 - alpha/2)] of the bootstrap distribution.

How accurate the intervals are (measured, not assumed)
------------------------------------------------------
Coverage of the nominal-95% Sharpe interval in simulations with a known true Sharpe of
0.1 per period, Politis-White block, n_boot = 1000:

    iid normal, T = 240 ........................ 94.3%   (2000 replications)
    iid normal, T = 60 (holdout length) ........ 92.3%   (1000 replications)
    AR(1) phi = 0.3, T = 240 ................... ~90-91% (600-1000 replications)
    AR(1) phi = 0.5, T = 240 ................... 88.5-90% (400-1000 replications)
    AR(1) phi = 0.5, T = 1000 .................. 93.7%   (300 replications)
    AR(1) phi = 0.5, T = 240, iid bootstrap .... 69%     (600 replications)

So the intervals are close to nominal for weakly dependent PnL, and they get narrower
than nominal when dependence is strong and the sample is short. Block bootstrap
variance is biased down in finite samples, and no block-length rule we tried (2x
Politis-White, or fixed blocks of 10 or 20) removed that. Ignoring dependence
altogether is far worse. Read a 95% interval from 60 days as roughly a 92% one. The
tests in tests/test_evaluation_bootstrap.py check these properties.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from arch.bootstrap import StationaryBootstrap, optimal_block_length

from markout.evaluation.dsr import ArrayLike1D, _as_1d, sharpe

_CHUNK_ELEMS = 4_000_000  # bound on (replications x T) indices held at once, ~32 MB


def auto_block_length(r: ArrayLike1D) -> float:
    """Politis-White (2004, corrected 2009) expected block length for the stationary
    bootstrap, via `arch.bootstrap.optimal_block_length`, floored at 1 (which is the iid
    bootstrap)."""
    x = _as_1d(r)
    b = float(optimal_block_length(x)["stationary"].iloc[0])
    return b if np.isfinite(b) and b >= 1.0 else 1.0


def _rows_sharpe(Y: np.ndarray) -> np.ndarray:
    """Row-wise `dsr.sharpe`, with the same exact degenerate-case rules."""
    hi, lo = Y.max(axis=1), Y.min(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = Y.mean(axis=1) / Y.std(axis=1, ddof=1)
    const = hi == lo
    return np.where(const, np.where(hi == 0.0, 0.0, np.nan), out)


def _rows_mean(Y: np.ndarray) -> np.ndarray:
    return Y.mean(axis=1)


def _bootstrap(r: ArrayLike1D, point_fn: Callable[[np.ndarray], float],
               rows_fn: Callable[[np.ndarray], np.ndarray], level: float, n_boot: int,
               block: float | None, seed: int) -> dict[str, Any]:
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    x = _as_1d(r)
    T = x.size
    if T < 3:
        raise ValueError(f"need at least 3 observations to bootstrap, got {T}")
    block = auto_block_length(x) if block is None else float(block)
    if block < 1:
        raise ValueError(f"expected block length must be >= 1, got {block}")

    bs = StationaryBootstrap(block, x, seed=seed)
    draws = np.empty(n_boot)
    chunk = max(1, _CHUNK_ELEMS // T)
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = np.stack([bs.update_indices() for _ in range(m)])
        draws[start:start + m] = rows_fn(x[idx])

    ok = draws[~np.isnan(draws)]
    alpha = 1.0 - level
    lo, hi = (np.quantile(ok, [alpha / 2, 1 - alpha / 2]) if ok.size
              else (np.nan, np.nan))
    return {
        "point": float(point_fn(x)),
        "lo": float(lo),
        "hi": float(hi),
        "block": float(block),
        "level": float(level),
        "n_boot": int(n_boot),
        "n_valid": int(ok.size),
        "se": float(ok.std(ddof=1)) if ok.size > 1 else float("nan"),
        "T": int(T),
    }


def sharpe_ci(r: ArrayLike1D, level: float = 0.95, n_boot: int = 5000,
              block: float | None = None, seed: int = 0) -> dict[str, Any]:
    """Stationary-bootstrap percentile CI for the per-period Sharpe ratio (`dsr.sharpe`).

    Parameters
    ----------
    r : per-period returns or PnL. NaNs are dropped, which closes the gaps, so pass
        zeros for flat days if that is what they are.
    level : two-sided confidence level.
    n_boot : bootstrap replications.
    block : expected block length in periods. None picks it with Politis-White.
    seed : RNG seed, so the interval is reproducible.

    Returns
    -------
    dict with point (sample Sharpe), lo, hi, block (the expected length used), level,
    n_boot, n_valid (replications with a defined Sharpe), se (bootstrap sd) and T.
    Everything is per period: multiply point, lo and hi by sqrt(252) to annualize daily
    values for display.
    """
    return _bootstrap(r, sharpe, _rows_sharpe, level, n_boot, block, seed)


def mean_ci(r: ArrayLike1D, level: float = 0.95, n_boot: int = 5000,
            block: float | None = None, seed: int = 0) -> dict[str, Any]:
    """Stationary-bootstrap percentile CI for the mean per-period PnL.

    Same parameters and return keys as `sharpe_ci`.
    """
    return _bootstrap(r, lambda v: float(np.mean(v)), _rows_mean, level, n_boot, block, seed)
