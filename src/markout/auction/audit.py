"""Data audit: what the closing-auction data contains, checked before any modelling.

Besides missingness, coverage and distributions, the audit *reconstructs the
target*: target = (60 s WAP return of the stock) - (60 s return of a fixed-weight
index). If that holds, then `ret60 - target` is the same number for every stock at
a given (date, second). Regressing it on all stocks' ret60 recovers the index
weights, which the features and the post-processing then use.
"""

from __future__ import annotations

import numpy as np
import polars as pl

GROUP = ["stock_id", "date_id"]
XS = ["date_id", "seconds_in_bucket"]
HORIZON_STEPS = 6  # 60 s in 10 s snapshots
STEPS_PER_DAY = 55


def with_forward_return(df: pl.DataFrame) -> pl.DataFrame:
    """Add ret60 = 60 s forward WAP return in bps within each stock-day.

    Null for the last six snapshots (their t+60 s lies after 540 s, outside the data).
    Assumes complete, sorted stock-days, which `coverage` verifies.
    """
    fwd = pl.col("wap").shift(-HORIZON_STEPS).over(GROUP)
    return df.with_columns(((fwd / pl.col("wap") - 1.0) * 1e4).cast(pl.Float64).alias("ret60"))


def missingness(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Share of nulls per column at each seconds_in_bucket."""
    return (df.group_by("seconds_in_bucket")
              .agg([pl.col(c).is_null().mean().alias(c) for c in columns])
              .sort("seconds_in_bucket"))


def coverage(df: pl.DataFrame) -> dict:
    per_day = df.filter(pl.col("seconds_in_bucket") == 0).group_by("date_id").agg(pl.len().alias("n")).sort("date_id")
    rows = df.group_by(GROUP).agg(pl.len().alias("n"))
    return {
        "n_rows": df.height,
        "n_stocks": df["stock_id"].n_unique(),
        "n_days": df["date_id"].n_unique(),
        "stocks_per_day_min": int(per_day["n"].min()),
        "stocks_per_day_median": float(per_day["n"].median()),
        "stocks_per_day_max": int(per_day["n"].max()),
        "incomplete_stock_days": int((rows["n"] != STEPS_PER_DAY).sum()),
        "stock_days": rows.height,
        "per_day": per_day,
    }


def distributions(df: pl.DataFrame) -> dict:
    spread = ((pl.col("ask_price") - pl.col("bid_price"))
              / ((pl.col("ask_price") + pl.col("bid_price")) / 2) * 1e4)
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    s = df.select(spread.alias("spread_bps"))["spread_bps"].drop_nulls().to_numpy()
    t = df["target"].drop_nulls().to_numpy().astype(np.float64)
    tc = t - t.mean()
    return {
        "spread_bps_quantiles": dict(zip([f"q{int(100 * p):02d}" for p in q], np.quantile(s, q))),
        "target_mean": float(t.mean()),
        "target_std": float(t.std()),
        "target_mae_vs_zero": float(np.abs(t).mean()),
        "target_quantiles": dict(zip([f"q{int(100 * p):02d}" for p in q], np.quantile(t, q))),
        "target_excess_kurtosis": float((tc ** 4).mean() / (tc ** 2).mean() ** 2 - 3.0),
        "target_share_abs_gt_20bps": float((np.abs(t) > 20).mean()),
        "target_null_share": float(df["target"].is_null().mean()),
    }


def reconstruction(df: pl.DataFrame) -> dict:
    """Check that ret60 - target is common to all stocks at each (date, second)."""
    x = with_forward_return(df).with_columns((pl.col("ret60") - pl.col("target")).alias("implied_index"))
    per = (x.drop_nulls(["implied_index"])
             .group_by(XS)
             .agg(pl.col("implied_index").std().alias("implied_std"),
                  pl.col("ret60").std().alias("ret_std"),
                  pl.len().alias("n"))
             .filter(pl.col("n") >= 5))
    return {
        "median_cross_stock_std_implied_index_bps": float(per["implied_std"].median()),
        "median_cross_stock_std_ret60_bps": float(per["ret_std"].median()),
        "ratio": float(per["implied_std"].median() / per["ret_std"].median()),
        "n_timestamps": per.height,
    }


def index_weights(df: pl.DataFrame, days: tuple[int, int]) -> tuple[pl.DataFrame, dict]:
    """Least-squares index weights from days[0]..days[1] (inclusive).

    y(d, t) = median over stocks of (ret60 - target) is the index move; X(d, t, i)
    is stock i's ret60. Only timestamps where every stock is present are used, so
    no missing return is silently treated as zero.
    """
    x = with_forward_return(df.filter(pl.col("date_id").is_between(*days)))
    x = x.with_columns((pl.col("ret60") - pl.col("target")).alias("implied_index")).drop_nulls(["ret60"])
    stocks = np.sort(x["stock_id"].unique().to_numpy())
    wide = x.pivot(on="stock_id", index=XS, values="ret60").sort(XS)
    y = (x.drop_nulls(["implied_index"]).group_by(XS)
          .agg(pl.col("implied_index").median().alias("y")))
    wide = wide.join(y, on=XS, how="inner").drop_nulls()
    cols = [str(s) for s in stocks]
    X = wide.select(cols).to_numpy().astype(np.float64)
    yv = wide["y"].to_numpy().astype(np.float64)
    info = {"n_timestamps": int(X.shape[0]), "n_stocks": int(X.shape[1]), "days": list(days)}
    if X.shape[0] < 2 * X.shape[1]:
        w = np.full(len(stocks), 1.0 / len(stocks))
        info.update(method="equal (too few complete timestamps)", r2=float("nan"))
    else:
        w, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ w
        info.update(method="least squares", r2=float(1 - resid.var() / yv.var()),
                    resid_std_bps=float(resid.std()), weight_sum=float(w.sum()),
                    min_weight=float(w.min()), max_weight=float(w.max()))
    return pl.DataFrame({"stock_id": stocks.astype(np.int16), "index_weight": w.astype(np.float64)}), info


def run(df: pl.DataFrame, weight_days: tuple[int, int]) -> dict:
    """Everything the report's audit section needs, as plain data."""
    cols = ["imbalance_size", "reference_price", "matched_size", "far_price", "near_price",
            "bid_price", "ask_price", "wap", "target"]
    miss = missingness(df, cols)
    cov = coverage(df)
    per_day = cov.pop("per_day")
    weights, winfo = index_weights(df, weight_days)
    first_far = miss.filter(pl.col("far_price") < 0.5)["seconds_in_bucket"]
    return {
        "coverage": cov,
        "stocks_per_day": per_day,
        "missingness": miss,
        "far_near_first_second": int(first_far.min()) if first_far.len() else None,
        "far_null_share_before_300": float(miss.filter(pl.col("seconds_in_bucket") < 300)["far_price"].mean()),
        "far_null_share_after_300": float(miss.filter(pl.col("seconds_in_bucket") >= 300)["far_price"].mean()),
        "distributions": distributions(df),
        "reconstruction": reconstruction(df),
        "weights": weights,
        "weights_info": winfo,
    }
