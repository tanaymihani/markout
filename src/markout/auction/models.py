"""Forecast models, from trivial baselines to gradient boosting.

All models predict the target in bps from causal features. The competition metric
is MAE, so the baselines are MAE-optimal constants (zero, per-stock median), the
linear model is fit on a winsorized target (least squares is not robust to the
fat tails here), and LightGBM optimizes L1 directly.
"""

from __future__ import annotations

import numpy as np
import polars as pl


class Model:
    name = "base"

    def fit(self, train: pl.DataFrame, features: list[str]) -> "Model":
        raise NotImplementedError

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError


class Zero(Model):
    """Predict no relative move: the MAE baseline everything must beat."""

    name = "zero"

    def fit(self, train, features):
        return self

    def predict(self, df):
        return np.zeros(df.height, dtype=np.float32)


class StockMedian(Model):
    """Per-stock median of the training target (MAE-optimal per-stock constant)."""

    name = "median"

    def fit(self, train, features):
        self.medians = train.group_by("stock_id").agg(pl.col("target").median().alias("m"))
        return self

    def predict(self, df):
        out = df.select("stock_id").join(self.medians, on="stock_id", how="left", maintain_order="left")
        return out["m"].fill_null(0.0).to_numpy().astype(np.float32)


class Ridge(Model):
    """Standardized ridge regression on winsorized features and target."""

    name = "ridge"

    def __init__(self, alpha: float = 10.0, winsor: float = 0.005):
        self.alpha, self.winsor = alpha, winsor

    def _x(self, df: pl.DataFrame) -> np.ndarray:
        X = df.select(self.features).to_numpy().astype(np.float64)
        X = np.clip(X, self.lo, self.hi)
        X = (X - self.mu) / self.sd
        return np.nan_to_num(X, nan=0.0)  # missing -> the training mean

    def fit(self, train, features):
        from sklearn.linear_model import Ridge as SkRidge

        self.features = list(features)
        X = train.select(self.features).to_numpy().astype(np.float64)
        self.lo = np.nanquantile(X, self.winsor, axis=0)
        self.hi = np.nanquantile(X, 1 - self.winsor, axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu = np.nanmean(Xc, axis=0)
        self.sd = np.nanstd(Xc, axis=0) + 1e-9
        y = train["target"].to_numpy().astype(np.float64)
        y = np.clip(y, *np.quantile(y, [0.01, 0.99]))
        self.model = SkRidge(alpha=self.alpha).fit(self._x(train), y)
        return self

    def predict(self, df):
        return self.model.predict(self._x(df)).astype(np.float32)


class LGBM(Model):
    """LightGBM with an L1 objective; early stopping on the last `valid_days` of the
    training window (never on test days)."""

    name = "lgbm"

    def __init__(self, num_leaves: int = 63, min_data_in_leaf: int = 500, learning_rate: float = 0.05,
                 feature_fraction: float = 0.8, bagging_fraction: float = 0.7, lambda_l2: float = 1.0,
                 max_rounds: int = 1500, valid_days: int = 30, threads: int = 4, seed: int = 0):
        self.params = {
            "objective": "l1", "learning_rate": learning_rate, "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf, "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction, "bagging_freq": 1, "lambda_l2": lambda_l2,
            "max_bin": 63, "num_threads": threads, "seed": seed, "deterministic": True,
            "force_row_wise": True, "verbose": -1,
        }
        self.max_rounds, self.valid_days = max_rounds, valid_days

    def fit(self, train, features):
        import lightgbm as lgb

        self.features = list(features)
        last = int(train["date_id"].max())
        is_valid = pl.col("date_id") > last - self.valid_days
        tr, va = train.filter(~is_valid), train.filter(is_valid)
        dtr = lgb.Dataset(tr.select(self.features).to_numpy(), tr["target"].to_numpy(),
                          feature_name=self.features, free_raw_data=True)
        dva = lgb.Dataset(va.select(self.features).to_numpy(), va["target"].to_numpy(), reference=dtr)
        self.booster = lgb.train(self.params, dtr, num_boost_round=self.max_rounds, valid_sets=[dva],
                                 callbacks=[lgb.early_stopping(100, verbose=False)])
        self.best_iteration = int(self.booster.best_iteration or self.max_rounds)
        return self

    def predict(self, df):
        X = df.select(self.features).to_numpy()
        return self.booster.predict(X, num_iteration=self.best_iteration).astype(np.float32)

    def importance(self) -> dict[str, float]:
        gain = self.booster.feature_importance("gain", iteration=self.best_iteration)
        return dict(sorted(zip(self.features, (gain / gain.sum()).tolist()), key=lambda kv: -kv[1]))


def make(kind: str, **params) -> Model:
    return {"zero": Zero, "median": StockMedian, "ridge": Ridge, "lgbm": LGBM}[kind](**params)
