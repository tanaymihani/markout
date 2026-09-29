"""Probabilistic and Deflated Sharpe ratios.

Reference
---------
Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality." *Journal of Portfolio
Management* 40(5), 94-107. SSRN 2460551. Equation and page numbers below refer to the
author-hosted version of July 31, 2014 (davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

Why this exists
---------------
If you try N strategy variants and report the best one, its Sharpe ratio is the maximum
of N noisy estimates, so it is biased upward even when every variant has zero true skill
(the winner's curse). DSR asks: given how many independent things I tried and how
dispersed their Sharpe ratios were, how likely is it that the winner's true Sharpe is
above zero? It also corrects for short samples and for non-normal returns, since
negative skew and fat tails make a given sample Sharpe less trustworthy.

Conventions
-----------
Everything is per period and NOT annualized: daily PnL gives a daily Sharpe, and T
counts days. Kurtosis is the plain (non-excess) fourth standardized moment, which is 3
for a normal distribution. Mixing annualized Sharpe ratios with a per-period T silently
breaks every formula here, so nothing inside this module annualizes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

EULER_GAMMA: float = float(np.euler_gamma)  # 0.5772156649...

ArrayLike1D = Sequence[float] | np.ndarray | Any  # lists, arrays, pandas Series


def _as_1d(r: ArrayLike1D) -> np.ndarray:
    """Float 1-D array with NaNs dropped. Infinite values are an error, not data."""
    x = np.asarray(r, dtype=float)
    if x.ndim != 1:
        x = np.squeeze(x)
        if x.ndim != 1:
            raise ValueError(f"expected a 1-D series of returns, got shape {np.shape(r)}")
    if np.isinf(x).any():
        raise ValueError("returns contain +/-inf; fix the PnL before computing statistics")
    return x[~np.isnan(x)]


def sharpe(r: ArrayLike1D) -> float:
    """Per-period (non-annualized) Sharpe ratio: mean(r) / std(r, ddof=1).

    NaNs are dropped. No risk-free rate is subtracted: the inputs are PnL of
    self-financing long/short positions, whose benchmark is zero.

    Degenerate cases, decided exactly rather than with a tolerance:
      * fewer than 2 observations -> NaN;
      * all observations exactly 0 (the strategy never traded) -> 0.0. It earned
        nothing, and ranking it as "earned nothing" matters in PBO: an in-sample winner
        that stops trading out of sample must count as a flop, not silently drop out;
      * constant but non-zero -> NaN (zero risk with non-zero return has no finite Sharpe).

    To annualize a daily Sharpe for display, multiply by sqrt(252), but never feed an
    annualized value back into `psr` or `deflated_sharpe`.
    """
    x = _as_1d(r)
    if x.size < 2:
        return float("nan")
    hi, lo = float(x.max()), float(x.min())
    if hi == lo:
        return 0.0 if hi == 0.0 else float("nan")
    return float(x.mean() / x.std(ddof=1))


def _skew_kurt(x: np.ndarray) -> tuple[float, float]:
    """Sample skewness and NON-excess kurtosis, using the plain moment estimators.

    The plain (biased) estimators are deliberate. They are the moments of the empirical
    distribution, so they always satisfy kurt >= 1 + skew^2 (Pearson), which keeps the
    PSR variance term positive. The small-sample "unbiased" versions don't guarantee it.
    """
    if x.size < 2 or not np.std(x) > 0:
        return float("nan"), float("nan")
    return (float(stats.skew(x, bias=True)),
            float(stats.kurtosis(x, fisher=False, bias=True)))


def psr(sr: float, sr_star: float, T: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio: the probability that the true Sharpe exceeds sr_star.

        PSR(SR*) = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

    with g3 the skewness and g4 the (non-excess) kurtosis of the returns (Bailey & López
    de Prado 2014, eq. 2; the PSR itself is from Bailey & López de Prado 2012, "The Sharpe
    Ratio Efficient Frontier", Journal of Risk 15(2)).

    The denominator is the standard error of the sample Sharpe under non-normal iid
    returns (Mertens 2002), scaled by sqrt(T - 1). When SR > 0, negative skew and fat
    tails (kurt > 3) both widen it, so the same observed Sharpe earns less confidence.
    That is the non-normality correction. (When SR < SR*, a wider denominator pulls PSR
    toward 0.5 instead, which is also correct: less evidence either way.)

    Parameters
    ----------
    sr : observed per-period Sharpe ratio (not annualized).
    sr_star : benchmark Sharpe, per period. 0 gives "probability of any skill", and
        `expected_max_sr` gives the Deflated Sharpe Ratio.
    T : number of return observations behind `sr`.
    skew : sample skewness of the returns.
    kurt : sample kurtosis of the returns, NON-excess (3 for a normal).

    Returns NaN if any input is NaN. Raises ValueError if T < 2 or the moments are
    inconsistent: every real distribution has kurt >= 1 + skew^2, which keeps the
    variance term positive, so a non-positive term usually means excess kurtosis was
    passed by mistake.
    """
    sr, sr_star, skew, kurt = (float(v) for v in (sr, sr_star, skew, kurt))
    if any(math.isnan(v) for v in (sr, sr_star, skew, kurt)):
        return float("nan")
    if T < 2:
        raise ValueError(f"PSR needs T >= 2 observations, got T={T}")
    radicand = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if not radicand > 0:
        raise ValueError(
            f"non-positive variance term 1 - skew*sr + (kurt-1)/4*sr^2 = {radicand:.3g}; "
            "check that kurt is NON-excess (normal = 3) and that kurt >= 1 + skew^2")
    z = (sr - sr_star) * math.sqrt(T - 1) / math.sqrt(radicand)
    return float(stats.norm.cdf(z))


def expected_max_sr(var_sr: float, n_trials: float) -> float:
    """Expected maximum Sharpe ratio of n_trials independent zero-skill trials.

        E[max SR_n] ~= sqrt(V[SR_n]) * ( (1 - gamma) * Phi^-1(1 - 1/N)
                                         + gamma * Phi^-1(1 - 1/(N*e)) )

    with gamma the Euler-Mascheroni constant (Bailey & López de Prado 2014, eq. 1 and
    Appendix A.1, with E[SR_n] = 0 under the null of no skill). This is the rejection
    threshold SR_0 behind the DSR: the Sharpe ratio that luck alone is expected to
    produce once you have looked at N independent variants.

    Parameters
    ----------
    var_sr : variance of the trials' Sharpe ratios, in the same per-period units as the
        Sharpe being tested. Without a trial log, 1/T is the variance of a zero-skill
        sample Sharpe from T iid normal observations.
    n_trials : number of INDEPENDENT trials (see `Registry.effective_n`). May be
        fractional, e.g. an average-correlation estimate.

    Notes
    -----
    The formula is an asymptotic (extreme-value) approximation. Against the exact
    expected maximum of N iid standard normals it overstates by about 0.036 sd at
    N = 10, 0.023 sd at N = 100 and 0.014 sd at N = 1000, so it errs slightly
    conservative. n_trials == 1 returns 0 (no selection, so no deflation). The formula
    diverges to -inf as N -> 1 from above, so results are floored at 0: the expected
    maximum of N >= 1 zero-mean draws is never negative.
    """
    var_sr, n_trials = float(var_sr), float(n_trials)
    if not var_sr >= 0:
        raise ValueError(f"var_sr must be >= 0, got {var_sr}")
    if not n_trials >= 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_trials == 1:
        return 0.0
    z = ((1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n_trials)
         + EULER_GAMMA * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return max(0.0, math.sqrt(var_sr) * float(z))


def deflated_sharpe(returns: ArrayLike1D,
                    trial_srs: ArrayLike1D | None = None,
                    var_sr: float | None = None,
                    n_trials: float | None = None) -> dict[str, float]:
    """Deflated Sharpe Ratio of a selected strategy: DSR = PSR(SR_0), SR_0 = E[max SR].

    Parameters
    ----------
    returns : per-period returns (e.g. daily PnL) of the SELECTED strategy. NaNs are
        dropped, and T is the count of what remains.
    trial_srs : per-period Sharpe ratios of EVERY trial in the search, including the
        failures and the selected one. They supply var_sr (sample variance, ddof=1) and,
        if n_trials isn't given, N = their count.
    var_sr : variance of the trials' Sharpe ratios. Overrides the one from trial_srs.
    n_trials : number of independent trials. Pass `Registry.effective_n()` here, because
        the raw count double-counts near-duplicate variants.

    Returns
    -------
    dict with keys
        sr       observed per-period Sharpe
        sr_star  SR_0, the expected maximum Sharpe under no skill
        psr_0    PSR against 0: evidence of skill before any multiple-testing correction
        dsr      PSR against SR_0: the deflated answer
        T, skew, kurt (non-excess), n_trials, var_sr

    Why the trial count is mandatory: it is the input that's easiest to "forget", so this
    function refuses to guess it. Pass trial_srs or n_trials. If this really was the only
    thing you tried, say so with n_trials=1, and DSR then equals PSR(0).

    Worked example from the paper (pp. 9-10). N = 100, V[SR] = 1/2 annualized, T = 1250
    daily observations, skew = -3, kurt = 10, and SR = 2.5 annualized:

        sr_star = expected_max_sr(0.5 / 250, 100)        # 0.1132 (per day)
        psr(2.5 / 250**0.5, sr_star, 1250, -3, 10)       # 0.9004 < 0.95: not significant
    """
    x = _as_1d(returns)
    T = int(x.size)
    sr = sharpe(x)
    skew, kurt = _skew_kurt(x)

    if trial_srs is not None:
        s = _as_1d(trial_srs)
        if var_sr is None:
            var_sr = float(s.var(ddof=1)) if s.size >= 2 else float("nan")
        if n_trials is None:
            n_trials = int(s.size)
    if n_trials is None:
        raise ValueError(
            "pass trial_srs or n_trials: the number of trials drives the deflation. "
            "If this really was the only strategy tried, pass n_trials=1.")
    if n_trials == 1:
        sr_star = 0.0
    else:
        if var_sr is None or not np.isfinite(var_sr):
            raise ValueError("n_trials > 1 needs var_sr, or trial_srs with >= 2 finite values")
        sr_star = expected_max_sr(float(var_sr), float(n_trials))

    if T >= 2 and np.isfinite(sr):
        psr_0 = psr(sr, 0.0, T, skew, kurt)
        dsr = psr(sr, sr_star, T, skew, kurt)
    else:
        psr_0 = dsr = float("nan")
    return {
        "sr": sr,
        "sr_star": float(sr_star),
        "psr_0": psr_0,
        "dsr": dsr,
        "T": T,
        "skew": skew,
        "kurt": kurt,
        "n_trials": n_trials,
        "var_sr": float(var_sr) if var_sr is not None else float("nan"),
    }
