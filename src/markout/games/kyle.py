"""One-period Kyle (1985) model, checked three ways, plus the bridge to module D.

Kyle, A. S. (1985), "Continuous auctions and insider trading", *Econometrica*
53(6), 1315–1335 (the single-auction equilibrium of section 2).

Setup
-----
The liquidation value is v ~ N(p₀, Σ₀). An insider who knows v submits a market
order x; noise traders submit u ~ N(0, σ_u²) independently. A competitive market
maker sees only total order flow y = x + u and sets p = E[v | y].

Linear equilibrium
------------------
Guess x = β(v − p₀) and p = p₀ + λy.

- Insider, given λ: maximize E[x(v − p) | v] = x(v − p₀) − λx², so x = (v − p₀)/(2λ),
  i.e. the best response is β = 1/(2λ).
- Maker, given β: p is the regression of v on y, so
  λ = Cov(v, y)/Var(y) = βΣ₀ / (β²Σ₀ + σ_u²).

Solving both: β* = σ_u/√Σ₀ and λ* = √Σ₀/(2σ_u) (so λ*β* = ½). The insider's
expected profit is β(1 − λβ)Σ₀, which is ½σ_u√Σ₀ at the equilibrium, and the
price reveals half of the insider's information: Var(v | y) = Σ₀/2.

Composing the two best responses gives the map used for the iteration check,
λ ← 2λΣ₀ / (Σ₀ + 4λ²σ_u²); λ* is its fixed point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from markout import results


@dataclass(frozen=True)
class KyleParams:
    """Prior mean p₀, prior variance Σ₀ of v, and noise-flow s.d. σ_u."""

    p0: float = 100.0
    sigma0_sq: float = 4.0
    sigma_u: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma0_sq <= 0 or self.sigma_u <= 0:
            raise ValueError("need Σ₀ > 0 and σ_u > 0")


@dataclass(frozen=True)
class KyleEquilibrium:
    beta: float             # insider trading intensity β* = σ_u/√Σ₀
    lam: float              # price impact λ* = √Σ₀/(2σ_u)
    insider_profit: float   # E[x(v − p)] = ½σ_u√Σ₀
    posterior_var: float    # Var(v | y) = Σ₀/2


def equilibrium(k: KyleParams) -> KyleEquilibrium:
    """Closed-form linear equilibrium of the one-period Kyle model."""
    s0 = math.sqrt(k.sigma0_sq)
    return KyleEquilibrium(beta=k.sigma_u / s0, lam=s0 / (2 * k.sigma_u),
                           insider_profit=0.5 * k.sigma_u * s0,
                           posterior_var=k.sigma0_sq / 2)


def insider_best_response(lam: float) -> float:
    """β that maximizes x(v − p₀) − λx² for every v: β = 1/(2λ)."""
    return 1.0 / (2.0 * lam)


def maker_best_response(beta: float, k: KyleParams) -> float:
    """Zero-profit (regression) λ given β: λ = βΣ₀ / (β²Σ₀ + σ_u²)."""
    return beta * k.sigma0_sq / (beta * beta * k.sigma0_sq + k.sigma_u ** 2)


def lambda_map(lam, k: KyleParams):
    """One round of best responses: λ ← 2λΣ₀ / (Σ₀ + 4λ²σ_u²)."""
    return 2 * lam * k.sigma0_sq / (k.sigma0_sq + 4 * lam * lam * k.sigma_u ** 2)


def lambda_iteration(k: KyleParams, lam0: float, n_iter: int = 60) -> np.ndarray:
    """Iterate :func:`lambda_map` from λ₀; returns λ₀, λ₁, …, λ_n.

    The map is maximized at λ* where it equals λ*, so after one step λ ≤ λ*;
    below λ* it is increasing with λ_(n+1) > λ_n, so the sequence converges to λ*
    from any λ₀ > 0 (quadratically near λ*, where the map's slope is zero).
    """
    out = np.empty(n_iter + 1)
    out[0] = lam0
    for i in range(n_iter):
        out[i + 1] = lambda_map(out[i], k)
    return out


def expected_insider_profit(beta, lam: float, k: KyleParams):
    """E[x(v − p)] with x = β(v − p₀): β(1 − λβ)Σ₀ (maximized at β = 1/(2λ))."""
    return beta * (1 - lam * beta) * k.sigma0_sq


def mc_insider_profit(betas: np.ndarray, lam: float, k: KyleParams, n: int,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo insider profit for each β with λ held fixed.

    Uses common random numbers (the same v and u for every β), so the curve is
    smooth in β and its argmax is estimated much more precisely than any single
    point. Returns (mean, standard error) arrays.
    """
    d = rng.normal(0.0, math.sqrt(k.sigma0_sq), n)       # v − p₀
    u = rng.normal(0.0, k.sigma_u, n)
    betas = np.asarray(betas, dtype=float)
    means = np.empty(betas.size)
    ses = np.empty(betas.size)
    for i, b in enumerate(betas):
        x = b * d
        profit = x * (d - lam * (x + u))                 # x·(v − p), p = p₀ + λ(x + u)
        means[i] = profit.mean()
        ses[i] = profit.std(ddof=1) / math.sqrt(n)
    return means, ses


def mc_market_maker_check(k: KyleParams, n: int, rng: np.random.Generator) -> dict[str, float]:
    """With the insider at β*, regress v on y by OLS.

    The slope estimates the maker's zero-profit λ and the residual variance
    estimates Var(v | y); they should match λ* and Σ₀/2.
    """
    eq = equilibrium(k)
    d = rng.normal(0.0, math.sqrt(k.sigma0_sq), n)
    y = eq.beta * d + rng.normal(0.0, k.sigma_u, n)
    slope = float(np.dot(y, d) / np.dot(y, y))   # E[y] = 0, so no intercept is needed
    resid = d - slope * y
    return {"lambda_hat": slope, "posterior_var_hat": float(resid.var()),
            "lambda_star": eq.lam, "posterior_var_star": eq.posterior_var}


# ---------------------------------------------------------------------------
# Bridge to module D (CKS price impact)
# ---------------------------------------------------------------------------

_BETA_KEYS = ("beta_ticks_per_1000_shares", "beta_ticks_per_1000", "ofi_beta", "cks_beta",
              "beta", "impact", "lambda", "price_impact")


def _as_rows(block: Any) -> list[dict]:
    """Normalize module D's ``kyle_bridge`` block to a list of row dicts.

    Accepts ``{ticker: number}``, ``{ticker: {field: value}}`` (what module D
    writes), a list of row dicts, or a dict with a ``rows``/``stocks`` list.
    """
    if isinstance(block, dict):
        for key in ("rows", "stocks", "by_stock", "tickers"):
            if isinstance(block.get(key), list):
                return _as_rows(block[key])
        rows = []
        for name, val in block.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                rows.append({"ticker": str(name), "cks_beta": float(val)})
            elif isinstance(val, dict):
                rows.append({"ticker": str(name), **val})
        return rows
    if isinstance(block, list):
        return [dict(r) for r in block if isinstance(r, dict)]
    return []


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _first(row: dict, keys: tuple[str, ...], positive: bool = False) -> float | None:
    for key in keys:
        v = row.get(key)
        if _is_num(v) and (not positive or v > 0):
            return float(v)
    return None


def kyle_bridge(micro: dict | None = None) -> dict[str, Any]:
    """Compare module D's CKS price-impact coefficients with Kyle's λ.

    Reads ``reports/results/microstructure.json`` (written by module D) unless
    ``micro`` is given, and uses its ``kyle_bridge`` block: per stock, the CKS
    slope β of the 10 s mid change (ticks) on order-flow imbalance (OFI), in
    ticks per 1,000 shares, with its R² and the mean depth at the best quotes.

    Derived per stock, when the inputs exist:

    - ``kyle_ratio`` = β/√R². In Kyle's model the price is an exact linear
      function of order flow (R² = 1), so the OLS slope equals the equilibrium
      λ = s.d.(Δp)/s.d.(y). In data the OLS slope is ρ·s.d.(Δp)/s.d.(y) with
      ρ = √R², so β/√R² is the Kyle-style ratio for the same dispersions.
    - ``beta_times_depth`` = β × depth (ticks): CKS's stylized book predicts
      about ½ tick, because OFI equal to the depth at the best moves the mid by
      half a tick.
    - ``kyle_lambda_trades`` = s.d.(Δmid)/s.d.(net traded volume), only if module
      D also reports those dispersions (``sd_dmid_ticks``,
      ``sd_trade_imbalance_k``): Kyle's λ proper, since in Kyle every order is a
      trade.

    Returns ``{"available": bool, "rows": [...], "note": str, ...}``.
    """
    if micro is None:
        micro = results.load("microstructure")
    if not micro or "kyle_bridge" not in micro:
        return {"available": False, "rows": [], "has_kyle_lambda": False,
                "note": "needs module D: run `python -m markout.lob.report` first, "
                        "which writes reports/results/microstructure.json with a "
                        "`kyle_bridge` block"}
    rows = []
    for r in _as_rows(micro["kyle_bridge"]):
        name = str(r.get("ticker") or r.get("stock") or r.get("symbol") or r.get("name") or "?")
        beta = _first(r, _BETA_KEYS)
        if beta is None:
            beta = next((float(v) for k, v in r.items() if "beta" in k.lower() and _is_num(v)), None)
        if beta is None:
            continue
        row: dict[str, Any] = {"ticker": name, "cks_beta": beta}
        r2 = _first(r, ("r2", "R2", "cks_r2"), positive=True)
        depth = _first(r, ("mean_depth_best", "depth", "avg_depth", "depth_best"), positive=True)
        price = _first(r, ("mean_price_usd", "price", "mean_price"), positive=True)
        if r2 is not None:
            row["r2"] = r2
            row["kyle_ratio"] = beta / math.sqrt(r2)
        if depth is not None:
            row["depth"] = depth
            row["beta_times_depth"] = beta * depth / 1000.0
        if price is not None:
            row["price"] = price
        for key in ("beta_bps_per_1000_shares", "beta_usd_per_share",
                    "median_window_beta_ticks_per_1000_shares"):
            if _is_num(r.get(key)):
                row[key] = float(r[key])
        sd_dp = _first(r, ("sd_dmid_ticks", "sd_dmid", "dmid_sd"), positive=True)
        sd_ti = _first(r, ("sd_trade_imbalance_k", "sd_trade_imbalance", "sd_ti"), positive=True)
        if sd_dp and sd_ti:
            row["kyle_lambda_trades"] = sd_dp / sd_ti
            row["ratio_cks_to_kyle"] = beta / row["kyle_lambda_trades"]
        rows.append(row)
    if not rows:
        return {"available": False, "rows": [], "has_kyle_lambda": False,
                "note": "module D's `kyle_bridge` block has no numeric CKS coefficient "
                        f"this parser recognizes (looked for {', '.join(_BETA_KEYS)})"}
    has_kyle = any("kyle_lambda_trades" in r for r in rows)
    note = ("Kyle's λ computed from module D's s.d. of Δmid and of net traded volume"
            if has_kyle else
            "module D reports CKS β per share of OFI; a trade-based Kyle λ would also need the "
            "per-bucket s.d. of Δmid and of net traded volume")
    return {"available": True, "rows": rows, "note": note, "has_kyle_lambda": has_kyle}
