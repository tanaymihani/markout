"""Causal features for the closing-auction forecast.

The rule every feature obeys: the value at (stock s, day d, second t) uses only
rows (any stock, day d, second <= t) and days < d. That is what a live system would
know at that instant. Labels (targets) of day d are never used on day d.
`tests/test_auction_features.py` enforces this with a perturbation test: corrupt
every future row, rebuild, and require the past features to be identical.

Groups:
- book:     continuous-market book state (spread, book imbalance, WAP vs mid)
- auction:  auction state (imbalance vs matched size, reference/near/far vs WAP)
- dynamics: within-day changes of the above over 1-6 snapshots
- xs:       cross-sectional context at the same instant (ranks, index-relative moves)
- hist:     trailing multi-day per-stock statistics, lagged one full day
"""

from __future__ import annotations

import polars as pl

GROUP = ["stock_id", "date_id"]
XS = ["date_id", "seconds_in_bucket"]
EPS = 1e-9

FEATURES: dict[str, list[str]] = {
    "book": ["spread_bps", "book_imb", "wap_vs_mid_bps", "log_depth"],
    "auction": ["imb_ratio", "imb_to_depth", "log_matched", "ref_vs_wap_bps", "near_vs_wap_bps",
                "far_vs_wap_bps", "near_minus_far_bps", "imb_flag", "is_post300", "seconds_f"],
    "dynamics": ["wap_ret_1", "wap_ret_3", "wap_ret_6", "wap_from_open_bps", "imb_ratio_d1",
                 "imb_ratio_d3", "matched_chg_1", "book_imb_d1"],
    "xs": ["rel_ret_1", "rel_ret_6", "rel_from_open_bps", "rank_imb_ratio", "rank_book_imb",
           "rank_ref_vs_wap", "z_rel_ret_6"],
    "hist": ["stock_absret_20d", "stock_spread_20d"],
}


def feature_list(groups: list[str]) -> list[str]:
    return [f for g in groups for f in FEATURES[g]]


def _clip(e: pl.Expr, lim: float) -> pl.Expr:
    return e.clip(-lim, lim)


def _book_and_auction(df: pl.DataFrame) -> pl.DataFrame:
    mid = (pl.col("ask_price") + pl.col("bid_price")) / 2
    depth = pl.col("bid_size") + pl.col("ask_size")
    signed_imb = pl.col("imbalance_buy_sell_flag").cast(pl.Float32) * pl.col("imbalance_size")
    matched = pl.when(pl.col("matched_size") > 0).then(pl.col("matched_size"))
    return df.with_columns(
        spread_bps=(pl.col("ask_price") - pl.col("bid_price")) / mid * 1e4,
        book_imb=(pl.col("bid_size") - pl.col("ask_size")) / (depth + EPS),
        wap_vs_mid_bps=(pl.col("wap") / mid - 1) * 1e4,
        log_depth=depth.log1p(),
        imb_ratio=_clip(signed_imb / matched, 10.0),
        imb_to_depth=pl.col("imbalance_buy_sell_flag").cast(pl.Float32)
        * (pl.col("imbalance_size") / (depth + EPS)).log1p(),
        log_matched=pl.col("matched_size").log1p(),
        ref_vs_wap_bps=_clip((pl.col("reference_price") / pl.col("wap") - 1) * 1e4, 2000),
        near_vs_wap_bps=_clip((pl.col("near_price") / pl.col("wap") - 1) * 1e4, 2000),
        far_vs_wap_bps=_clip((pl.col("far_price") / pl.col("wap") - 1) * 1e4, 2000),
        near_minus_far_bps=_clip((pl.col("near_price") - pl.col("far_price")) * 1e4, 2000),
        imb_flag=pl.col("imbalance_buy_sell_flag").cast(pl.Float32),
        is_post300=(pl.col("seconds_in_bucket") >= 300).cast(pl.Float32),
        seconds_f=pl.col("seconds_in_bucket").cast(pl.Float32),
        wap_from_open_bps=(pl.col("wap") - 1) * 1e4,
    )


def _dynamics(df: pl.DataFrame) -> pl.DataFrame:
    wap = pl.col("wap")
    lag = lambda c, k: pl.col(c).shift(k).over(GROUP)  # noqa: E731  (backward shift only)
    return df.with_columns(
        wap_ret_1=(wap / lag("wap", 1) - 1) * 1e4,
        wap_ret_3=(wap / lag("wap", 3) - 1) * 1e4,
        wap_ret_6=(wap / lag("wap", 6) - 1) * 1e4,
        imb_ratio_d1=pl.col("imb_ratio") - lag("imb_ratio", 1),
        imb_ratio_d3=pl.col("imb_ratio") - lag("imb_ratio", 3),
        matched_chg_1=_clip(pl.col("matched_size") / (lag("matched_size", 1) + EPS) - 1, 10.0),
        book_imb_d1=pl.col("book_imb") - lag("book_imb", 1),
    )


def _cross_section(df: pl.DataFrame, weights: pl.DataFrame) -> pl.DataFrame:
    """Index-relative moves use the reconstructed index weights, renormalized over the
    stocks present at each instant (all of their prices are known at that instant)."""
    df = df.join(weights, on="stock_id", how="left").with_columns(pl.col("index_weight").fill_null(0.0))

    def index_of(c: str) -> pl.Expr:
        w = pl.col("index_weight") * pl.col(c).is_not_null().cast(pl.Float64)
        return (pl.col(c).fill_null(0.0) * w).sum().over(XS) / (w.sum().over(XS) + EPS)

    n = pl.len().over(XS)
    df = df.with_columns(
        rel_ret_1=pl.col("wap_ret_1") - index_of("wap_ret_1"),
        rel_ret_6=pl.col("wap_ret_6") - index_of("wap_ret_6"),
        rel_from_open_bps=pl.col("wap_from_open_bps") - index_of("wap_from_open_bps"),
        rank_imb_ratio=pl.col("imb_ratio").rank().over(XS) / n,
        rank_book_imb=pl.col("book_imb").rank().over(XS) / n,
        rank_ref_vs_wap=pl.col("ref_vs_wap_bps").rank().over(XS) / n,
    )
    return df.with_columns(
        z_rel_ret_6=pl.col("rel_ret_6") / (pl.col("rel_ret_6").std().over(XS) + EPS),
    ).drop("index_weight")


def _history(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Trailing per-stock statistics over the previous `window` days (day d uses d-window..d-1).

    Targets of earlier days are fully realized by day d, so using them is causal.
    """
    daily = (df.group_by(GROUP)
               .agg(pl.col("target").abs().mean().alias("_absret"),
                    pl.col("spread_bps").mean().alias("_spread"))
               .sort(GROUP))
    daily = daily.with_columns(
        stock_absret_20d=pl.col("_absret").shift(1).rolling_mean(window, min_samples=5).over("stock_id"),
        stock_spread_20d=pl.col("_spread").shift(1).rolling_mean(window, min_samples=5).over("stock_id"),
    ).select(GROUP + ["stock_absret_20d", "stock_spread_20d"])
    return df.join(daily, on=GROUP, how="left")


def build(df: pl.DataFrame, weights: pl.DataFrame) -> pl.DataFrame:
    """Add every feature in FEATURES (float32). `df` must be sorted by stock, day, second."""
    out = _book_and_auction(df)
    out = _dynamics(out)
    out = _cross_section(out, weights)
    out = _history(out)
    feats = feature_list(list(FEATURES))
    return out.with_columns([pl.col(f).cast(pl.Float32) for f in feats]).sort(
        ["stock_id", "date_id", "seconds_in_bucket"])
