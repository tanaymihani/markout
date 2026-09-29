"""Diebold-Mariano test of equal predictive accuracy.

References
----------
Diebold, F. X. & Mariano, R. S. (1995). "Comparing Predictive Accuracy." *Journal of
Business & Economic Statistics* 13(3), 253-263.
Harvey, D., Leybourne, S. & Newbold, P. (1997). "Testing the Equality of Prediction Mean
Squared Errors." *International Journal of Forecasting* 13(2), 281-291.
Newey, W. K. & West, K. D. (1987; lag rule 1994). Heteroskedasticity- and
autocorrelation-consistent covariance estimation.

Why this exists
---------------
"The model's MAE is 2% lower than the baseline's" means nothing without a standard
error, and the per-day loss differences aren't independent. DM tests whether the mean
loss differential d_t = L(a)_t - L(b)_t is zero, using a long-run (autocorrelation-
robust) variance for d. It is a test about the forecasts, not the models, so it applies
to any loss (MAE, MSE, negative IC).

Measured size (nominal 5%, T = 240, equally accurate forecasts, 3000 simulations):
iid differentials 5.4%. AR(1) differentials with phi = 0.3: 6.6% with the default
Newey-West lag vs 10.6% with DM's h - 1 = 0 window. With phi = 0.5: 8.3% vs 17.2%. The
robust default roughly halves the over-rejection under persistence but doesn't remove
it, so treat p-values near 0.05 on persistent daily loss differentials with suspicion.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from markout.evaluation.dsr import ArrayLike1D


def newey_west_lrv(d: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance: g0 + 2 * sum_{k=1..L} (1 - k/(L+1)) * g_k.

    Here g_k = (1/T) * sum_t (d_t - mean)(d_{t-k} - mean) is the sample autocovariance.
    The Bartlett weights guarantee a non-negative estimate, which the unweighted DM
    window does not. lag = 0 gives the plain (1/T) sample variance.
    """
    x = np.asarray(d, dtype=float) - np.mean(d)
    T = x.size
    lrv = float(x @ x) / T
    for k in range(1, min(int(lag), T - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * float(x[k:] @ x[:-k]) / T
    return lrv


def _nw_rule_of_thumb(T: int) -> int:
    """Newey-West (1994) plug-in lag floor(4 * (T/100)^(2/9)); 4 for T = 240."""
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def _paired(loss_a: ArrayLike1D, loss_b: ArrayLike1D) -> np.ndarray:
    """d = loss_a - loss_b, aligned on the index when both are Series, NaN pairs dropped."""
    if isinstance(loss_a, pd.Series) and isinstance(loss_b, pd.Series):
        loss_a, loss_b = loss_a.align(loss_b, join="inner")
    a = np.asarray(loss_a, dtype=float).ravel()
    b = np.asarray(loss_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"loss series differ in length: {a.size} vs {b.size}")
    d = a - b
    if np.isinf(d).any():
        raise ValueError("losses contain +/-inf")
    return d[~np.isnan(d)]


def diebold_mariano(loss_a: ArrayLike1D, loss_b: ArrayLike1D, h: int = 1,
                    lag: int | None = None,
                    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
                    ) -> dict[str, Any]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

        d_t   = loss_a_t - loss_b_t
        DM    = mean(d) / sqrt(LRV_NW(d) / T)
        stat  = DM * sqrt((T + 1 - 2h + h(h - 1)/T) / T)      (HLN 1997)
        p     from Student-t with T - 1 degrees of freedom      (HLN 1997)

    Parameters
    ----------
    loss_a, loss_b : per-period losses of forecasts A and B (e.g. daily MAE). Two pandas
        Series are aligned on their index; arrays must have equal length. Pairs with a
        NaN on either side are dropped.
    h : forecast horizon in periods. Under optimal h-step forecasts, d_t is MA(h - 1).
    lag : Newey-West truncation lag. None uses max(h - 1, floor(4 (T/100)^(2/9))). The
        classic DM window stops at h - 1, but daily loss differentials often carry extra
        serial dependence (volatility regimes), and a too-short window overstates
        significance. With h = 1 and lag = 0, the statistic equals the paired t-test.
    alternative : "two-sided" (H1: equal accuracy fails), "less" (H1: A has lower
        expected loss, so A is better), or "greater" (H1: A has higher expected loss).

    Returns
    -------
    dict with stat (HLN-corrected), p_value, mean_diff (mean of loss_a - loss_b, where
    negative means A is better), dm (uncorrected statistic), lrv, lag, h, T and
    alternative.

    Degenerate case: if d has no variation, mean_diff == 0 gives stat 0 and p 1, and a
    non-zero constant gives stat +/-inf and p 0 (in the direction of the sign).
    """
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(f"unknown alternative {alternative!r}")
    d = _paired(loss_a, loss_b)
    T = int(d.size)
    if T < 3:
        raise ValueError(f"need at least 3 paired observations, got {T}")
    L = max(h - 1, _nw_rule_of_thumb(T)) if lag is None else int(lag)
    if L < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")

    dbar = float(d.mean())
    lrv = newey_west_lrv(d, L) if np.ptp(d) > 0 else 0.0
    if lrv > 0:
        dm = dbar / math.sqrt(lrv / T)
    else:
        dm = 0.0 if dbar == 0 else math.copysign(math.inf, dbar)
    hln = math.sqrt(max(T + 1 - 2 * h + h * (h - 1) / T, 0.0) / T)
    stat = dm * hln

    t = stats.t(df=T - 1)
    if alternative == "two-sided":
        p = 2.0 * float(t.sf(abs(stat)))
    elif alternative == "less":
        p = float(t.cdf(stat))
    else:
        p = float(t.sf(stat))
    return {
        "stat": float(stat),
        "p_value": min(p, 1.0),
        "mean_diff": dbar,
        "dm": float(dm),
        "lrv": float(lrv),
        "lag": int(L),
        "h": int(h),
        "T": T,
        "alternative": alternative,
    }
