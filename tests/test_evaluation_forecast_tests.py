"""Tests for markout.evaluation.forecast_tests (Diebold-Mariano with HLN correction)."""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from scipy.signal import lfilter
from statsmodels.stats.sandwich_covariance import cov_hac

from markout.evaluation.forecast_tests import diebold_mariano, newey_west_lrv


def _equal_accuracy_losses(rng, T=240, phi=0.0):
    """Two equally accurate forecasts whose loss differential is AR(1) with parameter phi."""
    d = lfilter([1.0], [1.0, -phi], rng.standard_normal(T)) * math.sqrt(1 - phi**2)
    la = 1.0 + 0.5 * d + np.abs(rng.standard_normal(T))
    lb = 1.0 - 0.5 * d + np.abs(rng.standard_normal(T))
    return la, lb


def test_identical_losses_do_not_reject():
    loss = np.abs(np.random.default_rng(0).standard_normal(240))
    r = diebold_mariano(loss, loss.copy())
    assert r["stat"] == 0.0 and r["p_value"] == 1.0 and r["mean_diff"] == 0.0


def test_clearly_better_forecast_is_detected():
    rng = np.random.default_rng(1)
    truth = rng.standard_normal(240)
    good = truth + 0.5 * rng.standard_normal(240)
    bad = truth + 1.0 * rng.standard_normal(240)
    la, lb = np.abs(truth - good), np.abs(truth - bad)       # absolute errors, daily
    r = diebold_mariano(la, lb)
    assert r["mean_diff"] < 0 and r["stat"] < 0 and r["p_value"] < 1e-4
    assert diebold_mariano(la, lb, alternative="less")["p_value"] < 1e-4
    assert diebold_mariano(la, lb, alternative="greater")["p_value"] > 0.99
    assert diebold_mariano(lb, la)["stat"] == pytest.approx(-r["stat"])  # antisymmetric


def test_size_is_nominal_for_equally_good_forecasts():
    """1000 pairs of equally accurate forecasts: ~5% false rejections (SE 0.7%).
    Measured offline over 3000 reps: 5.4%."""
    rej = [diebold_mariano(*_equal_accuracy_losses(np.random.default_rng(i)))["p_value"] < 0.05
           for i in range(1000)]
    assert 0.03 <= np.mean(rej) <= 0.075


def test_newey_west_default_resists_autocorrelated_differentials():
    """Loss differentials with AR(1) phi = 0.5. The classic lag-0 variance (DM's window for
    h = 1) over-rejects badly (17.2% offline over 3000 reps), and the Newey-West default
    lag (4 at T = 240) cuts that to 8.3%. That's better but still above 5%, which is why
    the lag is exposed."""
    nw, lag0 = [], []
    for i in range(1000):
        la, lb = _equal_accuracy_losses(np.random.default_rng(50_000 + i), phi=0.5)
        nw.append(diebold_mariano(la, lb)["p_value"] < 0.05)
        lag0.append(diebold_mariano(la, lb, lag=0)["p_value"] < 0.05)
    assert np.mean(nw) < 0.12
    assert np.mean(lag0) > np.mean(nw) + 0.04


def test_h1_lag0_equals_paired_t_test():
    """With h = 1 the HLN factor is sqrt((T-1)/T) and the reference is t(T-1), which is
    exactly the paired t-test."""
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(60) ** 2, rng.standard_normal(60) ** 2 + 0.2
    r = diebold_mariano(a, b, h=1, lag=0)
    t = stats.ttest_rel(a, b)
    assert r["stat"] == pytest.approx(t.statistic, rel=1e-10)
    assert r["p_value"] == pytest.approx(t.pvalue, rel=1e-10)


def test_hln_correction_and_default_lag_for_multi_step():
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal(100) ** 2, rng.standard_normal(100) ** 2
    T, h = 100, 6
    r = diebold_mariano(a, b, h=h)
    assert r["lag"] == max(h - 1, math.floor(4 * (T / 100) ** (2 / 9)))
    assert r["stat"] == pytest.approx(r["dm"] * math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T))
    assert r["p_value"] == pytest.approx(2 * stats.t(df=T - 1).sf(abs(r["stat"])))
    assert diebold_mariano(a, b)["lag"] == 4                           # NW rule at T = 100


@pytest.mark.parametrize("lag", [0, 1, 4, 10])
def test_newey_west_matches_statsmodels_hac(lag):
    d = lfilter([1.0], [1.0, -0.4], np.random.default_rng(lag).standard_normal(300))
    res = sm.OLS(d, np.ones(len(d))).fit()
    hac_var_of_mean = cov_hac(res, nlags=lag, use_correction=False)[0, 0]
    assert newey_west_lrv(d, lag) / len(d) == pytest.approx(hac_var_of_mean, rel=1e-10)


def test_alignment_nans_and_degenerate_inputs():
    rng = np.random.default_rng(5)
    idx = pd.RangeIndex(181, 421, name="date_id")
    a = pd.Series(rng.standard_normal(240) ** 2, index=idx)
    b = pd.Series(rng.standard_normal(240) ** 2, index=idx)
    shuffled = b.sample(frac=1.0, random_state=0)                     # same days, other order
    assert diebold_mariano(a, shuffled)["stat"] == pytest.approx(diebold_mariano(a, b)["stat"])
    a_gap = a.copy()
    a_gap.iloc[:10] = np.nan
    assert diebold_mariano(a_gap, b)["T"] == 230
    with pytest.raises(ValueError):
        diebold_mariano(a.to_numpy(), b.to_numpy()[:-1])
    quarters = rng.integers(0, 40, 240) / 4.0                          # exact in binary, so
    const = diebold_mariano(quarters + 1.0, quarters)                  # d == 1.0 exactly
    assert const["stat"] == math.inf and const["p_value"] == 0.0
    with pytest.raises(ValueError):
        diebold_mariano(a, b, h=0)
