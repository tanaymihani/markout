import numpy as np
import polars as pl
import pytest

from markout.auction import audit
from markout.auction.cv import Design
from markout.auction.pipeline import ModelSpec, forecast_metrics, neutralize, walk_forward
from markout.auction import features as F
from markout.auction.synth import make_synthetic


@pytest.fixture(scope="module")
def synth():
    return make_synthetic(n_stocks=30, n_days=120, seed=1)


def test_audit_recovers_index_weights_and_target(synth):
    df, w_true = synth
    out = audit.run(df, (0, 30))
    w = out["weights"].sort("stock_id")["index_weight"].to_numpy()
    assert np.abs(w - w_true).max() < 1e-4
    assert out["reconstruction"]["ratio"] < 1e-3
    assert out["far_near_first_second"] == 300
    assert out["far_null_share_before_300"] == 1.0
    assert out["coverage"]["incomplete_stock_days"] == 0


@pytest.mark.parametrize("design", [Design(), Design.scaled(120), Design.scaled(200, embargo=3)])
def test_walk_forward_folds_are_causal_and_tile_the_research_period(design):
    folds = design.folds()
    assert folds[0].role == "calibration" and all(f.role == "evaluation" for f in folds[1:])
    for f in folds:
        assert f.train[0] == 0
        assert f.train[1] == f.test[0] - 1 - design.embargo  # training ends before the embargo gap
    evals = [f.test for f in folds[1:]]
    assert evals[0][0] == design.first_eval and evals[-1][1] == design.research_end
    for a, b in zip(evals, evals[1:]):
        assert b[0] == a[1] + 1  # contiguous, non-overlapping
    lo, hi = design.holdout
    assert lo == design.research_end + 1 and hi == design.n_days - 1
    assert design.final_fold().train[1] < lo
    assert design.weight_days[1] < folds[0].test[0]  # index weights never see a test day


def test_real_design_matches_plan():
    d = Design()
    assert [f.test for f in d.folds()] == [(121, 180), (181, 240), (241, 300), (301, 360), (361, 420)]
    assert d.holdout == (421, 480)


def test_models_find_the_planted_signal(synth):
    df, w_true = synth
    design = Design.scaled(120)
    weights = pl.DataFrame({"stock_id": np.arange(30, dtype=np.int16), "index_weight": w_true})
    feat = F.build(df, weights)
    zero, _ = walk_forward(feat, ModelSpec("zero"), design, weights)
    ridge, _ = walk_forward(feat, ModelSpec("ridge"), design, weights)
    mz = forecast_metrics(zero.filter(pl.col("fold") >= 1), df)
    mr = forecast_metrics(ridge.filter(pl.col("fold") >= 1), df)
    assert mr["mae"] < mz["mae"]
    assert mr["ic_spearman_mean"] > 0.1
    assert mz["ic_spearman_mean"] == 0.0


def test_neutralize_makes_index_weighted_mean_zero(synth):
    df, w_true = synth
    weights = pl.DataFrame({"stock_id": np.arange(30, dtype=np.int16), "index_weight": w_true})
    p = df.select(["stock_id", "date_id", "seconds_in_bucket"]).with_columns(
        pred=pl.Series(np.random.default_rng(0).normal(3, 5, df.height), dtype=pl.Float32))
    n = neutralize(p, weights).join(weights, on="stock_id")
    wm = n.group_by(["date_id", "seconds_in_bucket"]).agg(
        ((pl.col("pred") * pl.col("index_weight")).sum() / pl.col("index_weight").sum()).alias("wm"))
    assert wm["wm"].abs().max() < 1e-4


def test_model_spec_key_is_stable():
    a = ModelSpec("lgbm", params={"num_leaves": 63, "learning_rate": 0.05})
    b = ModelSpec("lgbm", params={"learning_rate": 0.05, "num_leaves": 63})
    assert a.key == b.key and a.key != ModelSpec("lgbm").key
