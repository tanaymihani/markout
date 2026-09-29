"""Walk-forward prediction: fit on each fold's training days, predict its test days.

Predictions are cached per (model spec, dataset) because retraining LightGBM on
millions of rows is slow on a laptop. The cache key includes a data fingerprint,
so synthetic and real data can never be mixed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from markout.auction import features as F
from markout.auction.cv import Design, Fold
from markout.auction.models import make

ID = ["stock_id", "date_id", "seconds_in_bucket"]
XS = ["date_id", "seconds_in_bucket"]


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    groups: tuple[str, ...] = ("book", "auction", "dynamics", "xs", "hist")
    params: dict = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "groups": list(self.groups), "params": dict(sorted(self.params.items()))}

    @property
    def key(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return f"{self.kind}-{hashlib.sha256(blob).hexdigest()[:10]}"

    @property
    def features(self) -> list[str]:
        return F.feature_list(list(self.groups))

    def build(self):
        return make(self.kind, **self.params)


def fingerprint(df: pl.DataFrame) -> str:
    s = df.select(pl.len().alias("n"), pl.col("target").sum().alias("t"), pl.col("wap").sum().alias("w"))
    return hashlib.sha256(json.dumps(s.row(0), default=float).encode()).hexdigest()[:10]


def neutralize(preds: pl.DataFrame, weights: pl.DataFrame, col: str = "pred") -> pl.DataFrame:
    """Subtract the index-weighted mean prediction at each instant.

    Targets are index-relative, so their index-weighted mean is ~0 at every instant;
    forcing the forecasts to satisfy the same identity removes a common error."""
    w = preds.join(weights, on="stock_id", how="left")["index_weight"].fill_null(0.0)
    p = preds.with_columns(_w=w)
    wmean = (pl.col(col) * pl.col("_w")).sum().over(XS) / (pl.col("_w").sum().over(XS) + 1e-12)
    return p.with_columns((pl.col(col) - wmean).cast(pl.Float32).alias(col)).drop("_w")


def predict_fold(feat: pl.DataFrame, spec: ModelSpec, fold: Fold) -> tuple[pl.DataFrame, dict]:
    train = feat.filter(pl.col("date_id").is_between(*fold.train) & pl.col("target").is_not_null())
    test = feat.filter(pl.col("date_id").is_between(*fold.test))
    model = spec.build().fit(train, spec.features)
    info = {"fold": fold.k, "role": fold.role, "train": fold.train, "test": fold.test, "n_train": train.height}
    if hasattr(model, "best_iteration"):
        info["best_iteration"] = model.best_iteration
    if hasattr(model, "importance"):
        info["importance"] = model.importance()
    pred = model.predict(test)
    del train, model
    out = test.select(ID).with_columns(pl.Series("pred", pred, dtype=pl.Float32), pl.lit(fold.k).alias("fold"))
    return out, info


def walk_forward(feat: pl.DataFrame, spec: ModelSpec, design: Design, weights: pl.DataFrame,
                 cache_dir: Path | None = None, data_fp: str | None = None,
                 folds: list[Fold] | None = None, tag: str = "wf") -> tuple[pl.DataFrame, list[dict]]:
    """Out-of-sample predictions for every test row of every fold (neutralized)."""
    folds = folds if folds is not None else design.folds()
    path = meta_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{tag}_{spec.key}_{data_fp or 'nofp'}"
        path, meta_path = cache_dir / f"{stem}.parquet", cache_dir / f"{stem}.json"
        if path.exists() and meta_path.exists():
            return pl.read_parquet(path), json.loads(meta_path.read_text())
    parts, infos = [], []
    for fold in folds:
        p, info = predict_fold(feat, spec, fold)
        parts.append(p)
        infos.append(info)
    preds = neutralize(pl.concat(parts), weights)
    if path is not None:
        preds.write_parquet(path, compression="zstd")
        meta_path.write_text(json.dumps(infos, default=float))
    return preds, infos


def forecast_metrics(preds: pl.DataFrame, truth: pl.DataFrame) -> dict:
    """MAE and information coefficients of neutralized forecasts vs realized targets."""
    j = preds.join(truth.select(ID + ["target"]), on=ID, how="inner").drop_nulls(["target"])
    err = (pl.col("pred") - pl.col("target")).abs()
    by_day = j.group_by("date_id").agg(err.mean().alias("mae")).sort("date_id")
    ic = (j.group_by(XS)
           .agg(pl.corr("pred", "target", method="spearman").alias("ic"))
           .drop_nulls("ic").filter(pl.col("ic").is_not_nan()))
    ic_day = ic.group_by("date_id").agg(pl.col("ic").mean()).sort("date_id")["ic"].to_numpy()
    pearson = float(np.corrcoef(j["pred"].to_numpy(), j["target"].to_numpy())[0, 1]) if j["pred"].std() > 0 else 0.0
    return {
        "mae": float(j.select(err.mean()).item()),
        "n_rows": j.height,
        "ic_spearman_mean": float(ic["ic"].mean()) if ic.height else 0.0,
        "ic_day_tstat": float(ic_day.mean() / (ic_day.std(ddof=1) / np.sqrt(len(ic_day))))
        if len(ic_day) > 2 and ic_day.std() > 0 else 0.0,
        "ic_pearson_pooled": pearson,
        "mae_by_day": by_day,
        "mae_by_second": j.group_by("seconds_in_bucket").agg(err.mean().alias("mae")).sort("seconds_in_bucket"),
        "mae_by_stock": j.group_by("stock_id").agg(err.mean().alias("mae")).sort("stock_id"),
    }
