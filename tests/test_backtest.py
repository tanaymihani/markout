import numpy as np
import polars as pl
import pytest

from markout.backtest.costs import CostModel
from markout.backtest.engine import Policy, backtest, daily, decision_frame, kelly_table, trades
from markout.decision import calibration as cal
from markout.decision import voi


def dec_frame(edges, targets, spread=2.0, bid_size=1e6, ask_size=1e6, date=0, sec=0):
    n = len(edges)
    return pl.DataFrame({
        "stock_id": np.arange(n, dtype=np.int16), "date_id": np.full(n, date, np.int16),
        "seconds_in_bucket": np.full(n, sec, np.int16), "edge": np.asarray(edges, float),
        "target": np.asarray(targets, float), "bid_size": np.full(n, bid_size), "ask_size": np.full(n, ask_size),
        "spread_entry_bps": np.full(n, spread), "spread_exit_bps": np.full(n, spread),
    })


def test_cost_is_one_spread_plus_fees_plus_hedge_scaled():
    d = dec_frame([0.0], [0.0], spread=3.0).with_columns(spread_exit_bps=pl.lit(5.0))
    c = CostModel(fee_bps_per_side=0.5, hedge_bps_round_trip=0.25)
    assert d.select(c.expr(1.0)).item() == pytest.approx(4.0 + 1.0 + 0.25)
    assert d.select(c.expr(2.0)).item() == pytest.approx(2 * 5.25)


def test_threshold_side_size_and_pnl():
    costs = CostModel(0.5, 0.5)  # with spread 2: cost = 2 + 1 + 0.5 = 3.5 bps
    d = dec_frame([5.0, -5.0, 3.0, 10.0], [4.0, -6.0, 9.0, -2.0], bid_size=5e5, ask_size=2e5)
    tr = trades(d, Policy(threshold=1.0, max_notional=1e6, depth_frac=0.1, max_gross=1e9), costs)
    assert set(tr["stock_id"].to_list()) == {0, 1, 3}  # |3| < 3.5 is skipped
    row = {r["stock_id"]: r for r in tr.iter_rows(named=True)}
    assert row[0]["side"] == 1 and row[0]["notional"] == pytest.approx(0.1 * 2e5)  # buy -> ask size
    assert row[1]["side"] == -1 and row[1]["notional"] == pytest.approx(0.1 * 5e5)  # sell -> bid size
    assert row[0]["pnl_usd"] == pytest.approx(2e4 * (4.0 - 3.5) / 1e4)
    assert row[1]["pnl_usd"] == pytest.approx(5e4 * (6.0 - 3.5) / 1e4)
    assert row[3]["pnl_usd"] == pytest.approx(2e4 * (-2.0 - 3.5) / 1e4)


def test_ramp_sizing_and_gross_cap():
    costs = CostModel(0.5, 0.5)
    d = dec_frame([3.5 * 1.5, 3.5 * 3.0], [0.0, 0.0])
    tr = trades(d, Policy(threshold=1.0, sizing="ramp", ramp=1.0, max_notional=100.0, depth_frac=1.0,
                          max_gross=1e9), costs)
    sizes = dict(zip(tr["stock_id"].to_list(), tr["notional"].to_list()))
    assert sizes[0] == pytest.approx(50.0) and sizes[1] == pytest.approx(100.0)
    capped = trades(d, Policy(threshold=1.0, max_notional=100.0, depth_frac=1.0, max_gross=100.0), costs)
    assert capped["notional"].sum() == pytest.approx(100.0)


def test_decide_vs_pay_multipliers():
    costs = CostModel(0.5, 0.5)
    d = dec_frame([5.0], [5.0])
    assert trades(d, Policy(), costs, decide_mult=2.0).height == 0  # 5 < 7: adapted rule skips it
    tr = trades(d, Policy(), costs, decide_mult=1.0, pay_mult=2.0)  # unadapted: trades, pays 7
    assert tr["cost_bps"].item() == pytest.approx(7.0)


def test_daily_fills_idle_days_and_drawdown():
    d = pl.concat([dec_frame([9.0], [1.0], date=0), dec_frame([9.0], [20.0], date=2)])
    tr = trades(d, Policy(max_gross=1e12), CostModel(0.5, 0.5))
    day = daily(tr, (0, 3))
    assert day["date_id"].to_list() == [0, 1, 2, 3]
    assert day["pnl_usd"].to_list()[1] == 0 and day["pnl_usd"].to_list()[3] == 0
    assert (day["drawdown_usd"] <= 0).all()


def test_decision_frame_uses_exit_spread_60s_later():
    rows = []
    for sec in (0, 10, 60, 540):
        rows.append({"stock_id": 1, "date_id": 0, "seconds_in_bucket": sec, "target": 1.0,
                     "bid_price": 1.0, "ask_price": 1.0 + (sec + 10) * 1e-5, "bid_size": 1.0, "ask_size": 1.0})
    market = pl.DataFrame(rows).with_columns(pl.col(["stock_id", "date_id", "seconds_in_bucket"]).cast(pl.Int16))
    edges = market.select(["stock_id", "date_id", "seconds_in_bucket"]).with_columns(edge=pl.lit(1.0))
    d = decision_frame(edges, market)
    assert d["seconds_in_bucket"].to_list() == [0, 60, 540]
    r0 = d.row(0, named=True)
    assert r0["spread_exit_bps"] == pytest.approx(d.row(1, named=True)["spread_entry_bps"])
    r540 = d.row(2, named=True)
    assert r540["spread_exit_bps"] == pytest.approx(r540["spread_entry_bps"])


def test_calibration_slope_and_expanding_uses_only_the_past():
    assert cal.slope_through_origin(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(2.0)
    rng = np.random.default_rng(0)
    p = rng.normal(size=3000)
    folds = np.repeat([0, 1, 2], 1000)
    y = np.where(folds == 0, 0.5 * p, np.where(folds == 1, 1.5 * p, -9.0 * p))
    s = cal.expanding_slopes(pl.DataFrame({"pred": p, "target": y, "fold": folds}))
    assert s[0] == 0.0
    assert s[1] == pytest.approx(0.5)
    past = folds < 2  # pooled folds 0-1; fold 2's own data (slope -9) is never used
    assert s[2] == pytest.approx((p[past] * y[past]).sum() / (p[past] ** 2).sum())
    assert 0.5 < s[2] < 1.5


def test_synthetic_forecasts_hit_their_ic_and_break_even():
    rng = np.random.default_rng(1)
    d = dec_frame(np.zeros(20000), rng.standard_t(5, 20000) * 8)
    for rho in (0.1, 0.5):
        f = voi.with_synthetic(d, rho, seed=2)["edge"].to_numpy()
        assert np.corrcoef(f, d["target"].to_numpy())[0, 1] == pytest.approx(rho, abs=0.02)
    assert voi.break_even([0, 0.1, 0.2], [-2.0, -1.0, 1.0]) == pytest.approx(0.15)
    assert voi.break_even([0, 0.1], [-2.0, -1.0]) is None
    assert voi.with_oracle(d)["edge"].to_list() == d["target"].to_list()


def test_kelly_growth_peaks_near_full_kelly():
    r = np.random.default_rng(3).normal(0.001, 0.01, 2000) * 1e6
    rows = {k["kelly_fraction"]: k for k in kelly_table(r, capital=1e6, fractions=(0.5, 1.0, 2.0), n_boot=50)}
    g = {f: rows[f]["growth_daily_in_sample"] for f in rows}
    assert g[1.0] > g[0.5] and g[1.0] > g[2.0]
    assert rows[2.0]["max_drawdown_median"] < rows[0.5]["max_drawdown_median"]


def test_backtest_summary_on_a_known_stream():
    d = pl.concat([dec_frame([10.0], [8.0], date=i) for i in range(5)])
    _, day, s = backtest(d, Policy(max_notional=1e4, depth_frac=1.0), CostModel(0.5, 0.5), (0, 4))
    assert s["trades"] == 5 and s["net_bps_per_trade"] == pytest.approx(8.0 - 3.5)
    assert s["hit_rate"] == 1.0
