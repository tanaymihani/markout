"""Tests for markout.evaluation.dsr: PSR, expected max Sharpe and the Deflated Sharpe Ratio.

The paper's own worked example is reproduced to its printed precision. The rest are
Monte Carlo checks of the properties the formulas claim, all with fixed seeds.
"""

import math

import numpy as np
import pytest
from scipy import integrate, stats

from markout.evaluation.dsr import deflated_sharpe, expected_max_sr, psr, sharpe


def _row_sharpe(x: np.ndarray, axis: int) -> np.ndarray:
    return x.mean(axis=axis) / x.std(axis=axis, ddof=1)


# ---------------------------------------------------------------------------------------
# (a) The paper's worked example
# ---------------------------------------------------------------------------------------
# Source: Bailey, D. H. & López de Prado, M. (2014), "The Deflated Sharpe Ratio:
# Correcting for Selection Bias, Backtest Overfitting and Non-Normality", Journal of
# Portfolio Management 40(5); SSRN 2460551. Section "A NUMERICAL EXAMPLE", pp. 9-10 of
# the author-hosted version dated July 31, 2014
# (https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), read directly from the
# PDF. The strategist reports N = 100 independent trials, V[SR_n] = 1/2 (annualized),
# T = 1250 daily observations, skew = -3, kurtosis = 10, and a best annualized SR of
# 2.5, with 250 observations per year. The paper prints:
#   SR_0 = sqrt(1/(2*250)) * ((1-g) Z^-1[1 - 1/100] + g Z^-1[1 - 1/100 e^-1]) ~= 0.1132
#   DSR ~= 0.9004 < 0.95
#   "after running only N=46 independent trials ... DSR would have been 0.9505"
#   "If the strategy had exhibited Normal returns (g3 = 0, g4 = 3), DSR = 0.9505 after
#    N=88 independent trials."
PAPER_SR = 2.5 / math.sqrt(250)   # annualized 2.5 -> per day
PAPER_VAR = 0.5 / 250             # annualized variance 1/2 -> per day
PAPER_T = 1250


def test_paper_worked_example_sr0_and_dsr():
    sr0 = expected_max_sr(PAPER_VAR, 100)
    assert round(sr0, 4) == 0.1132
    assert round(psr(PAPER_SR, sr0, PAPER_T, -3.0, 10.0), 4) == 0.9004


def test_paper_worked_example_n46_and_normal_n88():
    dsr_46 = psr(PAPER_SR, expected_max_sr(PAPER_VAR, 46), PAPER_T, -3.0, 10.0)
    dsr_88_normal = psr(PAPER_SR, expected_max_sr(PAPER_VAR, 88), PAPER_T, 0.0, 3.0)
    assert round(dsr_46, 4) == 0.9505
    assert round(dsr_88_normal, 4) == 0.9505


def test_deflated_sharpe_wires_the_same_formulas():
    rng = np.random.default_rng(0)
    x = 0.05 + rng.standard_t(5, size=500)
    out = deflated_sharpe(x, var_sr=PAPER_VAR, n_trials=100)
    skew = stats.skew(x, bias=True)
    kurt = stats.kurtosis(x, fisher=False, bias=True)
    assert out["T"] == 500
    assert out["sr"] == pytest.approx(x.mean() / x.std(ddof=1), rel=1e-12)
    assert out["skew"] == pytest.approx(skew, rel=1e-12)
    assert out["kurt"] == pytest.approx(kurt, rel=1e-12)
    assert out["sr_star"] == pytest.approx(expected_max_sr(PAPER_VAR, 100), rel=1e-12)
    assert out["dsr"] == pytest.approx(psr(out["sr"], out["sr_star"], 500, skew, kurt), rel=1e-12)
    assert out["psr_0"] == pytest.approx(psr(out["sr"], 0.0, 500, skew, kurt), rel=1e-12)
    assert out["dsr"] < out["psr_0"]


# ---------------------------------------------------------------------------------------
# (b) E[max SR] against Monte Carlo and against the exact order statistic
# ---------------------------------------------------------------------------------------

def test_expected_max_sr_matches_monte_carlo_of_max_sample_sharpe():
    """N iid zero-mean normal strategies with T observations: the average of the max
    sample Sharpe should match expected_max_sr(var_sr=1/T, N).

    Expected gap: the formula overstates the exact E[max of 100 normals] by ~0.9%, and
    the sample Sharpe's fatter-than-normal tails (it is t-distributed with T-1 df) pull
    the other way by ~0.3%. The Monte Carlo standard error is ~0.4% of the mean, so a
    2% tolerance is > 3.5 standard errors from the expected gap.
    """
    N, T, reps = 100, 250, 2000
    rng = np.random.default_rng(12345)
    maxima = []
    for _ in range(reps // 200):
        r = rng.standard_normal((200, T, N))
        maxima.append(_row_sharpe(r, axis=1).max(axis=1))
    mc_mean = float(np.concatenate(maxima).mean())
    assert mc_mean == pytest.approx(expected_max_sr(1.0 / T, N), rel=0.02)


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_expected_max_sr_close_to_exact_and_conservative(n):
    """Against E[max of n iid N(0,1)] computed by quadrature, the approximation is within
    2.5% and errs high (conservative) for n >= 10."""
    exact, _ = integrate.quad(lambda x: x * n * stats.norm.pdf(x) * stats.norm.cdf(x) ** (n - 1),
                              -12, 12, limit=400)
    approx = expected_max_sr(1.0, n)
    assert approx >= exact
    assert approx == pytest.approx(exact, rel=0.025)


def test_expected_max_sr_edge_cases_and_scaling():
    assert expected_max_sr(0.3, 1) == 0.0                      # no selection, no deflation
    assert expected_max_sr(4.0, 50) == pytest.approx(2 * expected_max_sr(1.0, 50))
    vals = [expected_max_sr(1.0, n) for n in (2, 5, 10, 100, 1000)]
    assert all(b > a for a, b in zip(vals, vals[1:]))          # more trials, higher bar
    assert expected_max_sr(1.0, 1.05) >= 0.0                   # floored near N = 1
    with pytest.raises(ValueError):
        expected_max_sr(-1.0, 10)
    with pytest.raises(ValueError):
        expected_max_sr(1.0, 0.5)


# ---------------------------------------------------------------------------------------
# (c) Under the null, PSR(0) is Uniform(0, 1)
# ---------------------------------------------------------------------------------------

def _psr_many(x: np.ndarray, sr_star: float, *, normal: bool = False) -> np.ndarray:
    T = x.shape[1]
    sr = _row_sharpe(x, axis=1)
    sk = np.zeros(len(x)) if normal else stats.skew(x, axis=1, bias=True)
    ku = np.full(len(x), 3.0) if normal else stats.kurtosis(x, axis=1, fisher=False, bias=True)
    return np.array([psr(a, sr_star, T, b, c) for a, b, c in zip(sr, sk, ku)])


def test_psr_is_uniform_under_the_null():
    """4000 zero-skill normal strategies with T = 250. The rejection rate of PSR(0) > 0.95
    is ~5% (SE 0.34%, tolerance +/-1.5%), and the PSRs pass a KS test for uniformity."""
    x = np.random.default_rng(2024).standard_normal((4000, 250))
    p = _psr_many(x, 0.0)
    assert abs(np.mean(p > 0.95) - 0.05) < 0.015
    assert abs(np.mean(p < 0.05) - 0.05) < 0.015
    assert stats.kstest(p, "uniform").pvalue > 1e-3


def test_psr_null_calibration_survives_fat_tails():
    """Student-t(4) returns, zero mean: still ~5% false positives at the 95% level."""
    x = np.random.default_rng(99).standard_t(4, size=(4000, 250))
    p = _psr_many(x, 0.0)
    assert abs(np.mean(p > 0.95) - 0.05) < 0.015


# ---------------------------------------------------------------------------------------
# (d) Non-normality lowers PSR for the same Sharpe, and by the right amount
# ---------------------------------------------------------------------------------------

def test_negative_skew_and_fat_tails_lower_psr_for_the_same_sr():
    sr, T = 0.1, 250
    base = psr(sr, 0.0, T, 0.0, 3.0)
    assert psr(sr, 0.0, T, -1.0, 3.0) < base          # negative skew alone
    assert psr(sr, 0.0, T, 0.0, 8.0) < base           # fat tails alone
    assert psr(sr, 0.0, T, -2.0, 10.0) < psr(sr, 0.0, T, -1.0, 6.0) < base
    assert psr(sr, 0.0, T, +1.0, 3.0) > base          # positive skew helps a positive SR
    assert psr(sr, sr, T, -2.0, 10.0) == pytest.approx(0.5)


def test_same_mean_and_std_but_skewed_data_gets_lower_psr():
    """Two return series standardized to the SAME mean and std (so the same Sharpe): the
    negatively skewed, fat-tailed one earns less confidence."""
    rng = np.random.default_rng(5)
    T, mu = 500, 0.08
    normal = rng.standard_normal(T)
    skewed = -rng.lognormal(0.0, 0.8, T)             # long left tail
    std = lambda v: (v - v.mean()) / v.std(ddof=1)   # noqa: E731
    a, b = mu + std(normal), mu + std(skewed)
    ra = deflated_sharpe(a, n_trials=1)
    rb = deflated_sharpe(b, n_trials=1)
    assert ra["sr"] == pytest.approx(rb["sr"], rel=1e-12)
    assert rb["skew"] < -1.5 and rb["kurt"] > 6.0
    assert rb["psr_0"] < ra["psr_0"]


def test_non_normality_adjustment_is_calibrated_not_just_directional():
    """Negatively skewed returns (skew ~ -1.75, kurt ~ 8.9) with a TRUE Sharpe of 0.1.
    PSR(0.1) with the sample moments rejects at ~5% (5.0% here), while pretending the
    returns are normal over-rejects (7.6%), overstating confidence. Measured offline:
    the gap widens with stronger skew (6.1% vs 10.7% at skew -3.3)."""
    M, T, sr_true, s = 4000, 250, 0.1, 0.5
    z = np.random.default_rng(7).lognormal(0.0, s, size=(M, T))
    eps = -(z - np.exp(s**2 / 2)) / np.sqrt((np.exp(s**2) - 1) * np.exp(s**2))
    x = sr_true + eps
    rej_adj = np.mean(_psr_many(x, sr_true) > 0.95)
    rej_norm = np.mean(_psr_many(x, sr_true, normal=True) > 0.95)
    assert abs(rej_adj - 0.05) < 0.015
    assert rej_norm > rej_adj + 0.01


# ---------------------------------------------------------------------------------------
# plumbing: sharpe(), argument checks
# ---------------------------------------------------------------------------------------

def test_sharpe_basics_and_degenerate_cases():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert sharpe(x) == pytest.approx(x.mean() / x.std(ddof=1))
    assert sharpe([np.nan, 1.0, 2.0, 3.0, 4.0]) == pytest.approx(sharpe(x))  # NaNs dropped
    assert sharpe(np.zeros(10)) == 0.0          # never traded: earned nothing
    assert math.isnan(sharpe(np.full(10, 0.3)))  # riskless non-zero return: undefined
    assert math.isnan(sharpe([1.0]))
    with pytest.raises(ValueError):
        sharpe([1.0, np.inf, 2.0])
    with pytest.raises(ValueError):
        sharpe(np.ones((3, 3)))


def test_deflated_sharpe_demands_the_trial_count():
    x = np.random.default_rng(1).standard_normal(100) + 0.1
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe(x)
    with pytest.raises(ValueError):
        deflated_sharpe(x, n_trials=10)          # N > 1 but no dispersion given
    one = deflated_sharpe(x, n_trials=1)
    assert one["sr_star"] == 0.0 and one["dsr"] == one["psr_0"]


def test_deflated_sharpe_from_trial_srs():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(240) + 0.15
    srs = rng.normal(0.0, 0.07, size=30)
    out = deflated_sharpe(x, trial_srs=srs)
    assert out["n_trials"] == 30
    assert out["var_sr"] == pytest.approx(np.var(srs, ddof=1))
    assert out["sr_star"] == pytest.approx(expected_max_sr(np.var(srs, ddof=1), 30))
    # an explicit (e.g. clustered) N overrides the raw count, and fewer trials deflate less
    fewer = deflated_sharpe(x, trial_srs=srs, n_trials=5)
    assert fewer["n_trials"] == 5 and fewer["dsr"] > out["dsr"]


def test_psr_rejects_inconsistent_moments():
    # A classic slip: passing EXCESS kurtosis (normal = 0) with a large Sharpe.
    with pytest.raises(ValueError, match="NON-excess"):
        psr(3.0, 0.0, 100, 1.0, 0.0)
    assert math.isnan(psr(float("nan"), 0.0, 100, 0.0, 3.0))
    with pytest.raises(ValueError):
        psr(0.1, 0.0, 1, 0.0, 3.0)
