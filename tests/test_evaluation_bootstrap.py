"""Tests for markout.evaluation.bootstrap: coverage of stationary-bootstrap CIs.

Coverage is the property that matters: across repeated samples, a 95% interval should
contain the true Sharpe about 95% of the time. The tolerances below are set from the
coverage measured offline with 300-2000 replications (quoted in each test) and the
binomial standard error of the replication count used here, never from the fixed-seed
outcome.
"""

import math

import numpy as np
import pytest
from scipy.signal import lfilter

from markout.evaluation.bootstrap import auto_block_length, mean_ci, sharpe_ci
from markout.evaluation.dsr import sharpe

TRUE_SR = 0.1  # per period (~1.6 annualized for daily data)


def _ar1(rng: np.random.Generator, T: int, phi: float, mu: float = TRUE_SR) -> np.ndarray:
    """mu + stationary AR(1) noise with unit variance, so the true Sharpe is exactly mu."""
    u = rng.standard_normal(T) * math.sqrt(1.0 - phi**2)
    u[0] = rng.standard_normal()                   # start in the stationary distribution
    return mu + lfilter([1.0], [1.0, -phi], u)


def _coverage(fn, T: int, phi: float, reps: int, seed: int, truth: float = TRUE_SR, **kw) -> float:
    hits = 0
    for i in range(reps):
        x = _ar1(np.random.default_rng(seed + i), T, phi)
        ci = fn(x, seed=i, **kw)
        hits += ci["lo"] <= truth <= ci["hi"]
    return hits / reps


def test_sharpe_ci_coverage_iid():
    """iid normal, T = 240. Measured offline: 94.3% for the 95% CI (2000 reps), 88.9% for
    the 90% CI (1000 reps).
    With 400 reps the binomial SE is ~1.2% (95%) and ~1.6% (90%); bounds sit ~3 SE out."""
    cov95 = _coverage(sharpe_ci, T=240, phi=0.0, reps=400, seed=10_000, n_boot=1000, level=0.95)
    cov90 = _coverage(sharpe_ci, T=240, phi=0.0, reps=400, seed=10_000, n_boot=1000, level=0.90)
    assert 0.905 <= cov95 <= 0.985
    assert 0.84 <= cov90 <= 0.945
    assert cov95 > cov90


def test_mean_ci_coverage_iid():
    """Same design for the mean PnL (true mean = 0.1). Measured offline: 94.2% over 2000 reps."""
    cov = _coverage(mean_ci, T=240, phi=0.0, reps=300, seed=20_000, n_boot=1000)
    assert 0.90 <= cov <= 0.99


def test_sharpe_ci_coverage_ar1_approaches_nominal_with_sample_size():
    """AR(1) with phi = 0.5 and T = 1000: blocks capture the dependence, and coverage is
    close to nominal (93.7% measured offline over 300 reps; SE here ~1.7%)."""
    cov = _coverage(sharpe_ci, T=1000, phi=0.5, reps=200, seed=30_000, n_boot=1000)
    assert 0.88 <= cov <= 0.99


def test_block_bootstrap_beats_iid_bootstrap_under_autocorrelation():
    """AR(1), phi = 0.5, T = 240 (the research-period length). Offline over 600 reps, the
    stationary bootstrap covered 88.7% and an iid bootstrap (block = 1) only 69.3%.
    Ignoring dependence makes the interval far too narrow. The honest small-sample
    shortfall of the block bootstrap is documented in the module docstring."""
    block = _coverage(sharpe_ci, T=240, phi=0.5, reps=200, seed=40_000, n_boot=1000)
    iid = _coverage(sharpe_ci, T=240, phi=0.5, reps=200, seed=40_000, n_boot=1000, block=1.0)
    assert block >= 0.82
    assert block - iid >= 0.10


def test_interval_properties_and_reproducibility():
    x = _ar1(np.random.default_rng(1), 240, 0.2)
    a = sharpe_ci(x, n_boot=2000, seed=3)
    assert a["point"] == pytest.approx(sharpe(x))
    assert a["lo"] < a["point"] < a["hi"]
    assert a["n_valid"] == a["n_boot"] == 2000 and a["T"] == 240
    assert a["block"] == pytest.approx(auto_block_length(x))
    assert sharpe_ci(x, n_boot=2000, seed=3) == a                    # same seed, same answer
    assert sharpe_ci(x, n_boot=2000, seed=4)["lo"] != a["lo"]
    narrow = sharpe_ci(x, n_boot=2000, seed=3, level=0.80)
    assert a["lo"] <= narrow["lo"] < narrow["hi"] <= a["hi"]          # same draws, nested
    fixed = sharpe_ci(x, n_boot=500, block=7.0)
    assert fixed["block"] == 7.0
    m = mean_ci(x, n_boot=2000)
    assert m["point"] == pytest.approx(x.mean()) and m["lo"] < m["point"] < m["hi"]


def test_auto_block_length_grows_with_dependence():
    rng = np.random.default_rng(0)
    iid = auto_block_length(_ar1(rng, 1000, 0.0))
    dep = auto_block_length(_ar1(rng, 1000, 0.7))
    assert 1.0 <= iid < dep


def test_nans_dropped_and_bad_arguments_rejected():
    x = _ar1(np.random.default_rng(2), 100, 0.0)
    with_gaps = np.concatenate([x[:50], [np.nan] * 5, x[50:]])
    assert sharpe_ci(with_gaps, n_boot=300)["point"] == pytest.approx(sharpe(x))
    with pytest.raises(ValueError):
        sharpe_ci(x[:2])
    with pytest.raises(ValueError):
        sharpe_ci(x, level=1.5)
    with pytest.raises(ValueError):
        sharpe_ci(x, block=0.5)
