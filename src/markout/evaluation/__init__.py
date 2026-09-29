"""Module C, honest evaluation: the statistics that keep a backtest from fooling us.

    registry        append-only log of every trial (config, git SHA, daily PnL)
    dsr             Probabilistic and Deflated Sharpe ratios (Bailey & López de Prado 2014)
    pbo             Probability of Backtest Overfitting via CSCV (Bailey et al. 2017)
    bootstrap       stationary block bootstrap CIs for the Sharpe ratio and the mean
    holdout         run-once guard around the sealed holdout
    forecast_tests  Diebold-Mariano test with the HLN small-sample correction

Conventions: Sharpe ratios are per period (daily) and never annualized internally, and
kurtosis is non-excess (normal = 3).
"""

from markout.evaluation.bootstrap import auto_block_length, mean_ci, sharpe_ci
from markout.evaluation.dsr import deflated_sharpe, expected_max_sr, psr, sharpe
from markout.evaluation.forecast_tests import diebold_mariano, newey_west_lrv
from markout.evaluation.holdout import HoldoutAlreadyUsed, HoldoutGuard
from markout.evaluation.pbo import pbo
from markout.evaluation.registry import Registry, cluster_trials, config_hash, git_state

__all__ = [
    "Registry", "cluster_trials", "config_hash", "git_state",
    "sharpe", "psr", "expected_max_sr", "deflated_sharpe",
    "pbo",
    "sharpe_ci", "mean_ci", "auto_block_length",
    "HoldoutGuard", "HoldoutAlreadyUsed",
    "diebold_mariano", "newey_west_lrv",
]
