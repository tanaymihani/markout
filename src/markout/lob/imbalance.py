"""Queue imbalance as a predictor of the next mid-price move (Gould & Bonart 2016).

    I = (qB - qA) / (qB + qA)   at the best quotes, in [-1, 1].

A long bid queue facing a short ask queue means the ask is more likely to be depleted
first, so the next mid-price change is more likely to be up. Gould & Bonart show that a
logistic regression of 1{next move up} on I is strongly predictive for large-tick stocks,
whose queues are long and slow to deplete, and weak for small-tick stocks, where the
best level is thin and constantly replaced.

Sampling. We sample the book on a 1 s clock grid (every second weighs the same,
however many messages it contained) and label each sample with the direction of the
first mid-price change after it. The logistic model is fitted on samples before 12:45
whose label is also resolved before 12:45 (no label straddles the split), and scored by
out-of-sample AUC on samples after 12:45.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from markout.lob.lobster import SESSION, SPLIT, Day, state_index

STEP = 1.0   # seconds between samples


def queue_imbalance(bid_size: np.ndarray, ask_size: np.ndarray) -> np.ndarray:
    qb, qa = bid_size.astype(np.float64), ask_size.astype(np.float64)
    tot = qb + qa
    return np.divide(qb - qa, tot, out=np.zeros_like(tot), where=tot > 0)


def next_mid_change(mid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """For each state index i, the index j > i of the first state whose mid differs from
    mid[i] (len(mid) if the mid never changes again)."""
    change = np.flatnonzero(mid[1:] != mid[:-1]) + 1
    k = np.searchsorted(change, idx, side="right")
    out = np.full(len(idx), len(mid), dtype=np.int64)
    ok = k < len(change)
    out[ok] = change[k[ok]]
    return out


@dataclass(frozen=True)
class Samples:
    time: np.ndarray        # sample times
    imbalance: np.ndarray   # I at the state in effect
    up: np.ndarray          # 1 if the next mid change is up, 0 if down
    resolve: np.ndarray     # time at which the label is known (the next mid change)


def make_samples(day: Day, step: float = STEP,
                 session: tuple[float, float] = SESSION) -> Samples:
    m, b = day.messages, day.book
    t = np.arange(session[0], session[1], step)
    i = state_index(m.time, t)
    mid = b.mid()
    j = next_mid_change(mid, i)
    ok = (j < len(mid))
    ok[ok] &= m.time[j[ok]] < session[1]        # label must resolve inside the session
    i, j, t = i[ok], j[ok], t[ok]
    imb = queue_imbalance(b.bid_size[i, 0], b.ask_size[i, 0])
    return Samples(t, imb, (mid[j] > mid[i]).astype(np.int8), m.time[j])


@dataclass(frozen=True)
class ImbalanceFit:
    auc: float              # out-of-sample
    auc_train: float
    coef: float             # logit slope on I
    intercept: float
    n_train: int
    n_test: int
    base_rate_test: float   # share of up moves in the test set
    brier_test: float
    brier_base: float       # Brier score of always predicting the training base rate
    curve: list[dict]       # binned P(up | I) on the test set


def fit_imbalance(s: Samples, split: float = SPLIT, bins: int = 10) -> ImbalanceFit:
    train = (s.time < split) & (s.resolve < split)
    test = s.time >= split
    X = s.imbalance[:, None]
    model = LogisticRegression(C=1e6, max_iter=1000).fit(X[train], s.up[train])
    p = model.predict_proba(X[test])[:, 1]
    y = s.up[test]
    base = float(s.up[train].mean())
    edges = np.linspace(-1, 1, bins + 1)
    which = np.clip(np.digitize(s.imbalance[test], edges[1:-1]), 0, bins - 1)
    curve = [{"lo": float(edges[k]), "hi": float(edges[k + 1]), "n": int(np.sum(which == k)),
              "p_up": float(y[which == k].mean()) if np.any(which == k) else None,
              "p_model": float(p[which == k].mean()) if np.any(which == k) else None}
             for k in range(bins)]
    return ImbalanceFit(
        auc=float(roc_auc_score(y, p)),
        auc_train=float(roc_auc_score(s.up[train], model.predict_proba(X[train])[:, 1])),
        coef=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
        n_train=int(train.sum()),
        n_test=int(test.sum()),
        base_rate_test=float(y.mean()),
        brier_test=float(np.mean((p - y) ** 2)),
        brier_base=float(np.mean((base - y) ** 2)),
        curve=curve,
    )
