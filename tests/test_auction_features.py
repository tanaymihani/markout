"""The no-lookahead test: corrupt the future, rebuild, and the past must not move.

For a cut (d*, t*), every non-key column of every row after (d*, t*) is replaced by
noise, and every target from day d* on is replaced too (labels of the current day
are never known). Features at or before (d*, t*) must be identical to the
uncorrupted build. A negative control shows the test can catch a leak.
"""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from markout.auction import features as F
from markout.auction.synth import make_synthetic

KEYS = ["stock_id", "date_id", "seconds_in_bucket"]


@pytest.fixture(scope="module")
def data():
    df, w = make_synthetic(n_stocks=12, n_days=30, seed=3)
    weights = pl.DataFrame({"stock_id": np.arange(12, dtype=np.int16), "index_weight": w})
    return df, weights


def corrupt_future(df: pl.DataFrame, d_cut: int, t_cut: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    future = (pl.col("date_id") > d_cut) | ((pl.col("date_id") == d_cut) & (pl.col("seconds_in_bucket") > t_cut))
    cols = [c for c in df.columns if c not in KEYS]
    out = df.with_columns([
        pl.when(future).then(pl.Series(rng.normal(1.0, 0.5, df.height)).cast(df.schema[c])).otherwise(pl.col(c)).alias(c)
        for c in cols if c != "target"
    ])
    return out.with_columns(
        pl.when(pl.col("date_id") >= d_cut).then(pl.Series(rng.normal(0, 50, df.height)).cast(pl.Float32))
        .otherwise(pl.col("target")).alias("target"))


def past_rows(frame: pl.DataFrame, d_cut: int, t_cut: int, cols: list[str]) -> pl.DataFrame:
    past = (pl.col("date_id") < d_cut) | ((pl.col("date_id") == d_cut) & (pl.col("seconds_in_bucket") <= t_cut))
    return frame.filter(past).select(KEYS + cols).sort(KEYS)


@pytest.mark.parametrize("d_cut,t_cut", [(10, 250), (20, 0), (15, 300), (29, 540)])
def test_features_only_use_the_past(data, d_cut, t_cut):
    df, weights = data
    feats = F.feature_list(list(F.FEATURES))
    clean = past_rows(F.build(df, weights), d_cut, t_cut, feats)
    dirty = past_rows(F.build(corrupt_future(df, d_cut, t_cut), weights), d_cut, t_cut, feats)
    assert clean.height > 0
    assert_frame_equal(clean, dirty, check_exact=True)


def test_perturbation_test_catches_a_leak(data):
    """Negative control: a feature built with a forward shift must fail the check."""
    df, weights = data

    def leaky_build(frame):
        out = F.build(frame, weights)
        return out.with_columns(leak=pl.col("wap").shift(-1).over(["stock_id", "date_id"]))

    d_cut, t_cut = 10, 250
    clean = past_rows(leaky_build(df), d_cut, t_cut, ["leak"])
    dirty = past_rows(leaky_build(corrupt_future(df, d_cut, t_cut)), d_cut, t_cut, ["leak"])
    with pytest.raises(AssertionError):
        assert_frame_equal(clean, dirty, check_exact=True)


def test_feature_groups_are_disjoint_and_built(data):
    df, weights = data
    names = F.feature_list(list(F.FEATURES))
    assert len(names) == len(set(names))
    built = F.build(df, weights)
    assert set(names) <= set(built.columns)
    assert built.schema["seconds_in_bucket"] == df.schema["seconds_in_bucket"]
