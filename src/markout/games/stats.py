"""Small statistics helpers shared by the games modules.

Every Monte Carlo number in module E is reported with a 95% confidence interval
computed across independent seeds (runs). Two estimators are needed:

- a plain mean, ``x̄ ± z·s/√n``;
- a ratio of totals, ``R = Σy / Σx`` (for example PnL per trade = total PnL /
  total trades). Its standard error comes from the delta method:
  ``Var(R) ≈ Var(y − R·x) / (n·x̄²)``, which treats each seed as one draw of the
  pair (y, x).
"""

from __future__ import annotations

import math

import numpy as np

Z95 = 1.959963984540054  # Φ⁻¹(0.975)


def mean_ci(x: np.ndarray, z: float = Z95) -> dict[str, float]:
    """Mean of independent draws with a normal-approximation CI.

    Returns ``{"mean", "se", "lo", "hi", "n"}``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    m = float(x.mean()) if n else math.nan
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return {"mean": m, "se": se, "lo": m - z * se, "hi": m + z * se, "n": n}


def ratio_ci(y: np.ndarray, x: np.ndarray, z: float = Z95) -> dict[str, float]:
    """Ratio of totals ``Σy/Σx`` over seeds, with a delta-method CI.

    Seeds with ``x = 0`` still count (they contribute to both totals). If the
    denominator total is zero the ratio is undefined and NaN is returned.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    n = y.size
    xbar = x.mean() if n else 0.0
    if n == 0 or xbar == 0:
        return {"mean": math.nan, "se": math.nan, "lo": math.nan, "hi": math.nan, "n": n}
    r = float(y.sum() / x.sum())
    resid = y - r * x
    se = float(resid.std(ddof=1) / (math.sqrt(n) * abs(xbar))) if n > 1 else math.nan
    return {"mean": r, "se": se, "lo": r - z * se, "hi": r + z * se, "n": n}


def ci_contains(ci: dict[str, float], value: float) -> bool:
    """True when ``value`` lies inside the interval ``[lo, hi]``."""
    return bool(ci["lo"] <= value <= ci["hi"])
