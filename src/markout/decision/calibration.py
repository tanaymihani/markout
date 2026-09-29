"""Forecast calibration: turn a model's raw prediction into an expected edge.

A model's largest predictions are, disproportionately, its largest errors: when you
act only where the forecast looks biggest, you select on noise (the optimizer's
curse, Smith & Winkler 2006). Regressing realized targets on predictions and
shrinking by the slope corrects the average overstatement.

The slope for evaluation block k uses only out-of-sample predictions from blocks
< k (block 0 exists only to seed this), so calibration never sees its own test data.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def slope_through_origin(pred: np.ndarray, target: np.ndarray) -> float:
    """OLS slope of target on pred without intercept (both are index-relative)."""
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    ok = np.isfinite(pred) & np.isfinite(target)
    den = float((pred[ok] ** 2).sum())
    return float((pred[ok] * target[ok]).sum() / den) if den > 0 else 0.0


def expanding_slopes(preds: pl.DataFrame, max_slope: float = 5.0) -> dict[int, float]:
    """Calibration slope per fold k, fitted on folds < k. Needs `pred`, `target`, `fold`."""
    out = {}
    folds = sorted(preds["fold"].unique().to_list())
    for k in folds:
        past = preds.filter(pl.col("fold") < k).drop_nulls(["target"])
        b = slope_through_origin(past["pred"].to_numpy(), past["target"].to_numpy()) if past.height else 0.0
        out[k] = float(np.clip(b, 0.0, max_slope))
    return out


def apply(preds: pl.DataFrame, slopes: dict[int, float]) -> pl.DataFrame:
    """edge = slope[fold] * pred."""
    s = pl.DataFrame({"fold": list(slopes), "_b": list(slopes.values())},
                     schema={"fold": preds.schema["fold"], "_b": pl.Float64})
    return (preds.join(s, on="fold", how="left")
                 .with_columns((pl.col("pred") * pl.col("_b").fill_null(0.0)).alias("edge"))
                 .drop("_b"))


def decile_table(pred: np.ndarray, target: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Mean prediction vs mean realized target by prediction decile."""
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    ok = np.isfinite(pred) & np.isfinite(target)
    pred, target = pred[ok], target[ok]
    if pred.size == 0 or np.all(pred == pred[0]):
        return []
    edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, n_bins - 1)
    return [{"decile": int(i + 1), "mean_pred_bps": float(pred[idx == i].mean()),
             "mean_target_bps": float(target[idx == i].mean()), "n": int((idx == i).sum())}
            for i in range(n_bins) if (idx == i).any()]
