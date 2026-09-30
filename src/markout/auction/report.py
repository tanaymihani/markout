"""Modules A-C report: `python -m markout.auction.report` regenerates reports/01_auction.md.

Pipeline: audit -> causal features -> walk-forward forecasts for every model ->
decision policies (each model x policy pair is a registry trial) -> selection at
1x cost on the research period -> robustness (cost multiples, oracle bound,
value of information, calibration, leverage) -> honest evaluation (trial count,
effective N, Deflated Sharpe, PBO, bootstrap CIs) -> the holdout, run once.

Flags:
  --no-holdout   stop before the holdout (for iterating on the research period)
  --synthetic    run the identical pipeline on synthetic data; writes dev_* files
                 that are never committed (for development and tests)
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from markout import plotting as mp
from markout import reporting as R
from markout import results
from markout.auction import audit, features as F
from markout.auction.cv import Design
from markout.auction.experiments import COSTS, MODELS, Trial, model_label, run_policies
from markout.auction.pipeline import ID, ModelSpec, fingerprint, forecast_metrics, neutralize, predict_fold, walk_forward
from markout.backtest.engine import Policy, backtest, decision_frame
from markout.decision import calibration as cal
from markout.decision import voi
from markout.paths import PROCESSED, REGISTRY, RESULTS

RHOS = np.array([0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])


@dataclass
class Ctx:
    synthetic: bool
    df: pl.DataFrame
    design: Design
    prefix: str
    cache: Path
    registry_path: Path
    lock_path: Path
    audit_path: Path
    data_fp: str = ""
    out: dict = field(default_factory=dict)

    def name(self, s: str) -> str:
        return f"{self.prefix}{s}"


def make_ctx(synthetic: bool) -> Ctx:
    if synthetic:
        from markout.auction.synth import make_synthetic

        df, _ = make_synthetic(n_stocks=40, n_days=160, seed=7)
        cache = PROCESSED / "dev_preds"
        shutil.rmtree(cache, ignore_errors=True)
        for p in REGISTRY.glob("dev_*"):
            p.unlink()
        ctx = Ctx(True, df, Design.scaled(160), "dev_", cache, REGISTRY / "dev_trials.jsonl",
                  REGISTRY / "dev_holdout.lock", REGISTRY / "dev_holdout_audit.jsonl")
    else:
        from markout.auction.load import load

        df = load()
        ctx = Ctx(False, df, Design(), "", PROCESSED / "optiver" / "preds", REGISTRY / "trials.jsonl",
                  REGISTRY / "holdout.lock", REGISTRY / "holdout_audit.jsonl")
    REGISTRY.mkdir(parents=True, exist_ok=True)
    ctx.data_fp = fingerprint(ctx.df)
    return ctx


# ----------------------------------------------------------------------------- A
def run_audit(ctx: Ctx) -> dict:
    a = audit.run(ctx.df, ctx.design.weight_days)
    ctx.out["audit"] = {k: v for k, v in a.items() if k not in ("stocks_per_day", "missingness", "weights")}
    ctx.out["audit"]["top_index_weights"] = (a["weights"].sort("index_weight", descending=True).head(5)
                                             .to_dicts())
    miss = a["missingness"]

    def draw():
        fig, ax = mp.subplots(h=3.4)
        x = miss["seconds_in_bucket"].to_numpy()
        for i, (c, lab) in enumerate([("far_price", "far price"), ("near_price", "near price")]):
            ax.plot(x, 100 * miss[c].to_numpy(), color=mp.series(i), label=lab, drawstyle="steps-post")
        ax.set_xlabel("seconds into the closing auction")
        ax.set_ylabel("% of rows missing")
        ax.set_ylim(-3, 103)
        mp.legend(ax, loc="center right")
        mp.title(ax, "Indicative auction prices only exist after 300 s",
                 "Share of rows with a null far/near price, by snapshot")
        return fig

    mp.render(ctx.name("a_missing_by_second"), draw)
    return a


# ----------------------------------------------------------------------------- A
def run_forecasts(ctx: Ctx, weights: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    feat = F.build(ctx.df, weights)
    preds_by_model, infos, metrics = {}, {}, {}
    eval_lo = ctx.design.first_eval
    for spec in MODELS:
        t0 = time.time()
        p, info = walk_forward(feat, spec, ctx.design, weights, cache_dir=ctx.cache, data_fp=ctx.data_fp)
        preds_by_model[spec.key] = p
        infos[spec.key] = info
        m = forecast_metrics(p.filter(pl.col("date_id") >= eval_lo), ctx.df)
        metrics[spec.key] = m
        print(f"  {model_label(spec):32s} MAE {m['mae']:.4f}  IC {m['ic_spearman_mean']:+.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    ctx.out["feature_frame_rows"] = feat.height
    del feat
    return preds_by_model, {"infos": infos, "metrics": metrics}


def dm_vs(metrics_a: dict, metrics_b: dict) -> dict:
    from markout.evaluation.forecast_tests import diebold_mariano

    a = metrics_a["mae_by_day"].rename({"mae": "a"}).join(metrics_b["mae_by_day"].rename({"mae": "b"}), on="date_id")
    res = diebold_mariano(a["a"].to_numpy(), a["b"].to_numpy(), h=1)
    p = res.get("p_value", res.get("pvalue"))
    return {"stat": float(res.get("stat", np.nan)), "p_value": float(p) if p is not None else None,
            "mean_diff": float(np.mean(a["a"].to_numpy() - a["b"].to_numpy())),
            "share_days_better": float(np.mean(a["a"].to_numpy() < a["b"].to_numpy()))}


# ----------------------------------------------------------------------------- B
def edges_for(ctx: Ctx, preds: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    truth = ctx.df.select(ID + ["target"])
    pt = preds.join(truth, on=ID, how="left")
    slopes = cal.expanding_slopes(pt)
    e = cal.apply(pt.drop("target"), slopes).filter(pl.col("fold") >= 1)
    return decision_frame(e, ctx.df), slopes


def run_trials(ctx: Ctx, preds_by_model: dict) -> tuple[list[Trial], dict]:
    from markout.evaluation.registry import Registry

    reg = Registry(path=ctx.registry_path)
    trials, decs, slopes = [], {}, {}
    for spec in MODELS:
        if spec.kind in ("zero", "median"):
            continue  # constant forecasts never trade after neutralization
        dec, s = edges_for(ctx, preds_by_model[spec.key])
        decs[spec.key], slopes[spec.key] = dec, s
        trials += run_policies(spec, dec, ctx.design.eval_days, registry=_Dedup(reg, ctx.data_fp))
    return trials, {"decision_frames": decs, "slopes": slopes, "registry": reg}


class _Dedup:
    """Registry wrapper: re-running the report must not inflate the trial count.
    A trial whose config (incl. data fingerprint) is already logged is not re-logged."""

    def __init__(self, reg, data_fp: str):
        self.reg, self.data_fp = reg, data_fp
        try:
            df = reg.load()
            self.seen = {} if df is None or len(df) == 0 else dict(zip(df["config_hash"], df["trial_id"]))
        except FileNotFoundError:
            self.seen = {}

    def log(self, config, metrics, pnl, tags=None):
        from markout.evaluation.registry import config_hash

        config = {**config, "data": self.data_fp}
        h = config_hash(config)
        if h in self.seen:
            return self.seen[h]
        tid = self.reg.log(config, metrics, pnl, tags={**(tags or {}), "stage": "research", "data": self.data_fp})
        self.seen[h] = tid
        return tid


def select(trials: list[Trial]) -> Trial:
    return max(trials, key=lambda t: (t.summary["sharpe_daily"], t.summary["total_pnl_usd"]))


def robustness(ctx: Ctx, best: Trial, dec: pl.DataFrame, model_ic: float) -> dict:
    days = ctx.design.eval_days
    rows = []
    for m in (1.0, 2.0, 3.0):
        _, _, adapted = backtest(dec, best.policy, COSTS, days, decide_mult=m)
        _, _, unadapted = backtest(dec, best.policy, COSTS, days, decide_mult=1.0, pay_mult=m)
        _, _, orc = backtest(voi.with_oracle(dec), best.policy, COSTS, days, decide_mult=m)
        rows.append({"cost_mult": m, "adapted": adapted, "unadapted": unadapted, "oracle": orc})
    curve = voi.curve(dec, best.policy, COSTS, days, RHOS)
    be = {m: voi.break_even([r["rho"] for r in curve if r["cost_mult"] == m],
                            [r["blind_net_bps"] for r in curve if r["cost_mult"] == m]) for m in (1.0, 2.0, 3.0)}
    deciles = cal.decile_table(dec["pred"].to_numpy(), dec["target"].to_numpy())
    tr, day, _ = backtest(dec, best.policy, COSTS, days)
    return {"cost_rows": rows, "voi": curve, "break_even_ic": be, "model_ic": model_ic, "deciles": deciles,
            "capacity": capacity(tr, day, best.policy)}


def capacity(tr: pl.DataFrame, day: pl.DataFrame, policy: Policy) -> dict:
    """Is sizing limited by risk or by liquidity?

    Kelly leverage on the daily PnL relative to the peak capital actually deployed is
    L* = mu / sigma^2. For 60 s round trips it is enormous, because capital is at risk
    for a minute per trade. The binding constraint is then the displayed depth, so the
    decision that matters is which trades to take (the threshold), not the leverage.
    """
    if tr.height == 0:
        return {}
    peak = float(tr.group_by(["date_id", "seconds_in_bucket"]).agg(pl.col("notional").sum())["notional"].max())
    r = day["pnl_usd"].to_numpy() / peak
    touch = np.where(tr["side"].to_numpy() > 0, tr["ask_size"].to_numpy(), tr["bid_size"].to_numpy())
    depth_cap = policy.depth_frac * touch
    return {"peak_gross_usd": peak, "kelly_leverage_on_peak": float(r.mean() / r.var(ddof=1)) if r.var() > 0 else None,
            "share_trades_depth_capped": float(np.mean(depth_cap < policy.max_notional)),
            "mean_notional_usd": float(tr["notional"].mean()), "max_notional_usd": policy.max_notional,
            "median_touch_usd": float(np.median(touch))}


# ----------------------------------------------------------------------------- C
def honest(ctx: Ctx, trials: list[Trial], best: Trial, reg) -> dict:
    from markout.evaluation.bootstrap import sharpe_ci
    from markout.evaluation.dsr import deflated_sharpe
    from markout.evaluation.pbo import pbo

    ids = {t.trial_id for t in trials}
    active_ids = {t.trial_id for t in trials if t.summary["trades"] > 0}
    in_run = lambda rec: rec["trial_id"] in ids  # noqa: E731
    in_active = lambda rec: rec["trial_id"] in active_ids  # noqa: E731
    mat = reg.pnl_matrix(where=in_active, fill_value=0.0)  # a day without trades is a 0-PnL day
    srs = reg.trial_sharpes(where=in_run, fill_value=0.0).to_numpy()
    n_raw = len(trials)
    # never-trading variants are all the same "do nothing" strategy: one extra cluster at most
    n_eff = (reg.effective_n(where=in_active, fill_value=0.0) if len(active_ids) > 1 else len(active_ids)) \
        + (1 if len(active_ids) < n_raw else 0)
    r = best.daily["pnl_usd"].to_numpy().astype(float)
    out = {"n_trials": n_raw, "n_trading_trials": len(active_ids), "n_effective": int(n_eff),
           "trial_sharpes_daily": srs.tolist()}
    if np.std(r) > 0 and len(active_ids) > 1:
        out["dsr"] = deflated_sharpe(r, trial_srs=srs, n_trials=max(int(n_eff), 1))
        pb = pbo(mat, n_blocks=16)
        out["pbo"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in pb.items() if k in ("pbo", "prob_oos_loss", "degradation_slope", "n_used")}
        out["pbo"]["logits"] = np.asarray(pb.get("logits", []), float).tolist()
    if np.std(r) > 0:
        out["sharpe_ci"] = sharpe_ci(r, level=0.95, n_boot=5000, seed=0)
    return out


# ----------------------------------------------------------------------------- holdout
def run_holdout(ctx: Ctx, best: Trial, weights: pl.DataFrame, research_preds: pl.DataFrame,
                median_spec: ModelSpec) -> dict:
    from markout.evaluation.holdout import HoldoutAlreadyUsed, HoldoutGuard

    guard = HoldoutGuard(lock_path=ctx.lock_path, audit_path=ctx.audit_path)
    frozen = {**best.config, "data": ctx.data_fp, "holdout": list(ctx.design.holdout)}

    def score() -> dict:
        feat = F.build(ctx.df, weights)
        fold = ctx.design.final_fold()
        p, info = predict_fold(feat, best.spec, fold)
        pm, _ = predict_fold(feat, median_spec, fold)
        del feat
        p, pm = neutralize(p, weights), neutralize(pm, weights)
        truth = ctx.df.select(ID + ["target"])
        rp = research_preds.join(truth, on=ID, how="left").drop_nulls(["target"])
        slope = float(np.clip(cal.slope_through_origin(rp["pred"].to_numpy(), rp["target"].to_numpy()), 0, 5))
        e = p.with_columns((pl.col("pred") * slope).alias("edge"))
        dec = decision_frame(e, ctx.df)
        days = ctx.design.holdout
        out = {"slope": slope, "best_iteration": info.get("best_iteration")}
        for m in (1.0, 2.0, 3.0):
            _, day, s = backtest(dec, best.policy, COSTS, days, decide_mult=m)
            _, _, u = backtest(dec, best.policy, COSTS, days, decide_mult=1.0, pay_mult=m)
            out[f"adapted_{m:g}x"], out[f"unadapted_{m:g}x"] = s, u
            if m == 1.0:
                out["daily_pnl"] = day["pnl_usd"].to_list()
                out["daily_trades"] = day["n_trades"].to_list()
                out["days"] = day["date_id"].to_list()
        fm, fb = forecast_metrics(p, ctx.df), forecast_metrics(pm, ctx.df)
        out["mae_model"], out["mae_median"] = fm["mae"], fb["mae"]
        out["ic_model"] = fm["ic_spearman_mean"]
        return out

    full = ctx.lock_path.with_name(ctx.lock_path.stem + "_result.json")

    def score_and_keep() -> dict:
        res = score()
        results_json = results._clean(res)
        full.write_text(__import__("json").dumps(results_json, indent=1))
        return res

    try:
        res = guard.run(score_and_keep, frozen)
    except HoldoutAlreadyUsed:
        if not full.exists():
            raise SystemExit(f"holdout already used but {full} is missing; refusing to re-run it")
        res = __import__("json").loads(full.read_text())
    res = dict(res or {})
    res["accessed"] = guard.accessed_count()
    from markout.evaluation.bootstrap import sharpe_ci

    if res.get("daily_pnl") and np.std(res["daily_pnl"]) > 0:
        res["sharpe_ci"] = sharpe_ci(np.asarray(res["daily_pnl"], float), level=0.95, n_boot=5000, seed=0)
    return res


# ----------------------------------------------------------------------------- figures
def figures(ctx: Ctx, fc: dict, best: Trial, median_key: str, rob: dict, hon: dict, hold: dict | None) -> None:
    m_best, m_med = fc["metrics"][best.spec.key], fc["metrics"][median_key]

    def mae_by_second():
        fig, ax = mp.subplots(h=3.4)
        for i, (m, lab) in enumerate([(m_med, "per-stock median"), (m_best, model_label(best.spec))]):
            s = m["mae_by_second"]
            ax.plot(s["seconds_in_bucket"].to_numpy(), s["mae"].to_numpy(), color=mp.series(i), label=lab)
        ax.axvline(300, color=mp.baseline_color(), linewidth=0.9, zorder=1)
        ax.set_xlabel("seconds into the closing auction")
        ax.set_ylabel("MAE, bps")
        mp.legend(ax)
        mp.title(ax, "Out-of-sample error by snapshot", "Research period, walk-forward; lower is better")
        return fig

    mp.render(ctx.name("a_mae_by_second"), mae_by_second)

    imp = next((i.get("importance") for i in reversed(fc["infos"][best.spec.key]) if i.get("importance")), None)
    if imp:
        top = list(imp.items())[:12][::-1]

        def importance():
            fig, ax = mp.subplots(h=3.8)
            mp.bars(ax, [k for k, _ in top], [100 * v for _, v in top], horizontal=True)
            ax.grid(axis="x"), ax.grid(axis="y", visible=False)
            ax.set_xlabel("% of total split gain")
            mp.title(ax, "What the model uses", f"{model_label(best.spec)}, last walk-forward fold")
            return fig

        mp.render(ctx.name("a_importance"), importance)

    if rob["deciles"]:
        dx = np.array([d["mean_pred_bps"] for d in rob["deciles"]])
        dy = np.array([d["mean_target_bps"] for d in rob["deciles"]])

        def calibration():
            fig, ax = mp.subplots(w=5.2, h=4.0)
            lim = float(max(np.abs(dx).max(), np.abs(dy).max()) * 1.15)
            ax.plot([-lim, lim], [-lim, lim], color=mp.baseline_color(), linewidth=0.9, zorder=1)
            ax.plot(dx, dy, "o", color=mp.series(0))
            mp.end_label(ax, dx[-1], dy[-1], "top decile")
            ax.set_xlabel("mean prediction, bps")
            ax.set_ylabel("mean realized target, bps")
            mp.title(ax, "Calibration by prediction decile", "On the 45° line = calibrated")
            return fig

        mp.render(ctx.name("b_calibration"), calibration)

    cr = rob["cost_rows"]

    def cost():
        fig, ax = mp.subplots(w=5.6, h=3.6)
        x = [r["cost_mult"] for r in cr]
        ax.plot(x, [r["adapted"]["sharpe_annualized"] for r in cr], "o-", color=mp.series(0),
                label="rule knows the true cost")
        ax.plot(x, [r["unadapted"]["sharpe_annualized"] for r in cr], "o-", color=mp.series(1),
                label="rule assumes 1x cost")
        mp.zero_line(ax)
        ax.set_xticks(x, [f"{v:g}x" for v in x])
        ax.set_xlabel("true cost as a multiple of the base cost model")
        ax.set_ylabel("annualized Sharpe")
        mp.legend(ax)
        mp.title(ax, "Cost sensitivity of the selected policy", "Research period, net of costs")
        return fig

    mp.render(ctx.name("b_cost"), cost)

    curve = rob["voi"]

    def voi_fig():
        fig, (a1, a2) = mp.subplots(1, 2, w=9.0, h=3.8)
        for i, m in enumerate((1.0, 2.0, 3.0)):
            rs = [r for r in curve if r["cost_mult"] == m]
            a1.plot([r["rho"] for r in rs], [np.nan if r["blind_net_bps"] is None else r["blind_net_bps"] for r in rs],
                    color=mp.series(i), label=f"{m:g}x cost")
        mp.zero_line(a1)
        rs = [r for r in curve if r["cost_mult"] == 1.0]
        a2.plot([r["rho"] for r in rs], [r["threshold_sharpe_ann"] for r in rs], color=mp.series(0))
        mp.zero_line(a2)
        for ax in (a1, a2):
            ax.axvline(rob["model_ic"], color=mp.ink("muted"), linewidth=0.9, zorder=1)
            ax.set_xlabel("forecast correlation with the target (IC)")
        a1.annotate("this model", xy=(rob["model_ic"], 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=8.5, color=mp.ink("secondary"))
        a1.set_ylabel("net bps per trade")
        a2.set_ylabel("annualized Sharpe")
        mp.legend(a1, loc="lower right")
        mp.title(a1, "Cost-blind: trade the top decile", "Synthetic forecasts of known IC")
        mp.title(a2, "Cost-aware threshold rule, 1x cost", "Same forecasts, perfectly calibrated")
        return fig

    mp.render(ctx.name("b_voi"), voi_fig)

    def equity():
        fig, ax = mp.subplots(h=3.4)
        d = best.daily
        ax.plot(d["date_id"].to_numpy(), d["cum_pnl_usd"].to_numpy() / 1e3, color=mp.series(0), label="research")
        if hold and hold.get("daily_pnl"):
            start = float(d["cum_pnl_usd"].to_numpy()[-1])
            hx = np.asarray(hold["days"])
            hy = start + np.cumsum(hold["daily_pnl"])
            ax.plot(np.r_[d["date_id"].to_numpy()[-1], hx], np.r_[start, hy] / 1e3, color=mp.series(1),
                    label="holdout (run once)")
        mp.zero_line(ax)
        ax.set_xlabel("day")
        ax.set_ylabel("cumulative net PnL, $ thousands")
        mp.legend(ax, loc="upper left")
        mp.title(ax, "Selected policy, net of costs", f"{model_label(best.spec)}, "
                 f"threshold {best.policy.threshold:g}x cost, {best.policy.sizing} sizing")
        return fig

    mp.render(ctx.name("b_equity"), equity)

    srs = np.sort(np.asarray(hon.get("trial_sharpes_daily", []), float))
    if len(srs) > 1 and "dsr" in hon:
        sr_star = hon["dsr"]["sr_star"]

        def trials_fig():
            fig, ax = mp.subplots(h=3.4)
            ax.plot(np.arange(1, len(srs) + 1), srs * np.sqrt(252), "o", color=mp.series(0), label="trials")
            ax.axhline(sr_star * np.sqrt(252), color=mp.series(1), linewidth=1.5,
                       label="expected best of N by luck alone (SR*)")
            mp.zero_line(ax)
            ax.set_xlabel("trial, sorted by research-period Sharpe")
            ax.set_ylabel("annualized Sharpe")
            mp.legend(ax, loc="upper left")
            mp.title(ax, "Every trial that was run", "The winner is compared with what N tries would find by luck")
            return fig

        mp.render(ctx.name("c_trials"), trials_fig)


# ----------------------------------------------------------------------------- text
PUBLIC_BENCH = {"median": 5.815, "gbm": 5.708,
                "url": "https://github.com/andrew-somerset/optiver-trading-at-the-close"}


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b if b else float("nan")


def write_report(ctx: Ctx, a: dict, fc: dict, best: Trial, trials: list[Trial], rob: dict, hon: dict,
                 hold: dict | None, slopes: dict) -> None:
    n = R.num
    au, dist, rec, wi = ctx.out["audit"], ctx.out["audit"]["distributions"], ctx.out["audit"]["reconstruction"], \
        ctx.out["audit"]["weights_info"]
    cov = au["coverage"]
    med_spec = next(s for s in MODELS if s.kind == "median")
    m_med = fc["metrics"][med_spec.key]
    lines = []
    rows = []
    for spec in MODELS:
        m = fc["metrics"][spec.key]
        rows.append({"model": model_label(spec), "MAE (bps)": m["mae"],
                     "vs median": f"{pct(m['mae'], m_med['mae']):+.2f}%",
                     "IC (Spearman)": m["ic_spearman_mean"], "IC t-stat (daily)": m["ic_day_tstat"]})
    best_fc = fc["metrics"][best.spec.key]
    dm = fc.get("dm", {})
    s = best.summary
    ci = hon.get("sharpe_ci", {})
    dsr = hon.get("dsr", {})
    pb = hon.get("pbo", {})
    cr = {r["cost_mult"]: r for r in rob["cost_rows"]}
    be = rob["break_even_ic"]

    survives = s["total_pnl_usd"] > 0 and ci.get("lo", -1) > 0
    verdict_research = ("positive and its 95% bootstrap CI excludes zero" if survives else
                        "not distinguishable from zero" if s["total_pnl_usd"] > 0 else "negative")
    title = "Closing-auction alpha: does it survive costs?"
    head = [f"# 01 · {title}", "",
            ("> **Synthetic data.** This is a development run on generated data; none of these numbers "
             "describe the real market.\n" if ctx.synthetic else ""),
            "**Question.** A model can forecast 60-second index-relative returns in the NASDAQ closing auction "
            "better than a baseline. Is that edge still there after paying the spread, fees and the index hedge, "
            "and after correcting for how many variants were tried?", ""]
    ans = [f"**Answer.** The best model ({model_label(best.spec)}) cuts out-of-sample MAE by "
           f"{n(-pct(best_fc['mae'], m_med['mae']))}% versus the per-stock median "
           f"(IC {n(best_fc['ic_spearman_mean'])}"
           + (f"; Diebold–Mariano p {R.prob(dm['p_value']) if R.prob(dm['p_value']).startswith(('<', '>')) else '= ' + R.prob(dm['p_value'])}"
              if dm.get("p_value") is not None else "") + "). "
           f"Turned into trades at 1x cost, the selected policy's research-period PnL is {verdict_research}: "
           f"annualized Sharpe {n(s['sharpe_annualized'])}"
           + (f" (95% CI {n(ci['lo'] * np.sqrt(252))} to {n(ci['hi'] * np.sqrt(252))})" if ci else "")
           + f", {n(s['trades_per_day'])} trades a day, {n(s['net_bps_per_trade'])} bps net per trade."]
    a2 = cr.get(2.0, {}).get("adapted", {})
    if a2:
        ans.append(f" The edge is about one spread wide: at 2x cost the same rule's Sharpe is "
                   f"{n(a2['sharpe_annualized'])}" + (" and it stops paying." if a2["sharpe_annualized"] <= 0 else "."))
    cap = rob.get("capacity") or {}
    if cap:
        ans.append(f" It is also small: the displayed-depth cap binds on {n(100 * cap['share_trades_depth_capped'])}% "
                   f"of trades, the average trade is {R.usd(cap['mean_notional_usd'])}, and the research period nets "
                   f"{R.usd(s['total_pnl_usd'])} in total.")
    if dsr:
        strong = dsr["dsr"] >= 0.95
        ans.append(f" Deflated for {hon['n_effective']} effective trials (of {hon['n_trials']}), the Deflated "
                   f"Sharpe Ratio is {R.prob(dsr['dsr'])}" + (f" and PBO is {R.prob(pb.get('pbo'))}" if pb else "")
                   + (": the selection survives the multiple-testing correction." if strong else
                      ": suggestive, but below the usual 0.95 bar once the search is accounted for."))
    if hold and hold.get("adapted_1x"):
        h = hold["adapted_1x"]
        ans.append(f" On the holdout (days {ctx.design.holdout[0]}–{ctx.design.holdout[1]}, scored once), the "
                   f"same frozen policy nets {R.usd(h['total_pnl_usd'])} over {h['days']} days "
                   f"(annualized Sharpe {n(h['sharpe_annualized'])}).")
    lines += head + ["".join(ans), ""]

    lines += ["## 1. Data audit", "",
              R.bullets([
                  f"{cov['n_rows']:,} rows: {cov['n_stocks']} stocks × {cov['n_days']} days × 55 snapshots "
                  f"(0–540 s). Stocks per day: {cov['stocks_per_day_min']}–{cov['stocks_per_day_max']}; "
                  f"incomplete stock-days: {cov['incomplete_stock_days']}.",
                  f"Far/near indicative prices are null in {n(100 * au['far_null_share_before_300'])}% of rows "
                  f"before 300 s and {n(100 * au['far_null_share_after_300'])}% after: Nasdaq publishes them "
                  "from 3:55 pm, so the auction changes regime halfway through.",
                  f"Target: std {n(dist['target_std'])} bps, mean |target| {n(dist['target_mae_vs_zero'])} bps "
                  f"(the MAE of predicting zero), excess kurtosis {n(dist['target_excess_kurtosis'])}; "
                  f"{n(100 * dist['target_share_abs_gt_20bps'])}% of targets exceed 20 bps in size.",
                  f"Quoted spread: median {n(dist['spread_bps_quantiles']['q50'])} bps "
                  f"(5th–95th percentile {n(dist['spread_bps_quantiles']['q05'])}–"
                  f"{n(dist['spread_bps_quantiles']['q95'])}). A 60 s round trip pays about one full spread.",
                  f"Target reconstruction: stock WAP return − target is the same number for every stock at an "
                  f"instant (median cross-stock std {n(rec['median_cross_stock_std_implied_index_bps'])} bps vs "
                  f"{n(rec['median_cross_stock_std_ret60_bps'])} bps for raw returns), so the target really is "
                  "\"stock minus a fixed index\".",
                  f"Index weights recovered by least squares on days {wi['days'][0]}–{wi['days'][1]} "
                  f"({wi['method']}, R² {R.prob(wi.get('r2'))}, weights sum to {n(wi.get('weight_sum'))})"
                  + (". Stocks " + ", ".join(f"{sid} (first trades on day {day})" for sid, day in wi["absent_stocks"])
                     + " only appear after the weight window, so their weights are unknown and set to zero "
                     "rather than estimated from test days." if wi.get("absent_stocks") else "."),
              ]), "",
              mp.picture(ctx.name("a_missing_by_second"), "Share of null far/near prices by second"), ""]

    lines += ["## 2. Forecasts (walk-forward, research period)", "",
              f"Four expanding walk-forward blocks of {ctx.design.block} days cover days "
              f"{ctx.design.eval_days[0]}–{ctx.design.eval_days[1]} ({ctx.design.eval_days[1] - ctx.design.eval_days[0] + 1} "
              f"out-of-sample days); a fifth, earlier block exists only to calibrate the first. Every feature passes a "
              "perturbation test (corrupt the future, the past must not change). "
              "Predictions are neutralized to an index-weighted mean of zero at each instant.", "",
              R.table(rows, formats={"MAE (bps)": "{:.4f}", "IC (Spearman)": "{:+.4f}", "IC t-stat (daily)": "{:.1f}"}),
              ""]
    if dm:
        lines += [f"Diebold–Mariano test on daily MAE, {model_label(best.spec)} vs median: statistic "
                  f"{n(dm['stat'])}, p {R.prob(dm['p_value']) if R.prob(dm['p_value']).startswith(('<', '>')) else '= ' + R.prob(dm['p_value'])}; the model has lower MAE on "
                  f"{n(100 * dm['share_days_better'])}% of days.", ""]
    mbs = m_med["mae_by_second"]
    peak = mbs.sort("mae", descending=True).row(0, named=True)
    typical = float(mbs.filter((pl.col("seconds_in_bucket") < 200) | (pl.col("seconds_in_bucket") > 320))["mae"].median())
    lines += [f"The error is not uniform: the baseline's MAE peaks at {n(peak['mae'])} bps at {peak['seconds_in_bucket']} s, "
              f"{n(peak['mae'] / typical)}× its typical level, for snapshots whose 60 s window crosses 300 s, when "
              "Nasdaq starts publishing the indicative prices.", "",
              mp.picture(ctx.name("a_mae_by_second"), "MAE by second, model vs baseline"), ""]
    if (Path(ctx.cache).exists() and fc["infos"][best.spec.key][-1].get("importance")):
        lines += [mp.picture(ctx.name("a_importance"), "Feature importance"), ""]

    pol = best.policy
    lines += ["## 3. From forecasts to trades", "",
              f"Every 60 s (0, 60, …, 540 s) the rule trades a stock only if its calibrated edge exceeds "
              f"{pol.threshold:g}× the round-trip cost, holds it 60 s, and pays half the entry spread plus half the "
              f"exit spread (both from the data), {n(COSTS.fee_bps_per_side)} bp fees per side and "
              f"{n(COSTS.hedge_bps_round_trip)} bp for the index hedge. Size is "
              + ("flat" if pol.sizing == "flat" else f"a ramp reaching full size when the edge is {1 + pol.ramp:g}× cost")
              + f", capped at ${n(pol.max_notional)} and {n(100 * pol.depth_frac)}% of the displayed touch; gross is "
              f"capped at ${n(pol.max_gross)} per instant. The {len(trials)} (model, policy) trials below were all "
              "logged; this one had the best research-period Sharpe at 1× cost.", "",
              R.table([{"metric": k, "value": v} for k, v in [
                  ("annualized Sharpe", n(s["sharpe_annualized"])),
                  ("total net PnL", R.usd(s["total_pnl_usd"])),
                  ("trades per day", n(s["trades_per_day"])),
                  ("hit rate (net > 0)", n(100 * s["hit_rate"]) + "%"),
                  ("predicted edge, bps (notional-weighted)", n(s["edge_pred_bps"])),
                  ("realized edge before costs, bps", n(s["edge_realized_bps"])),
                  ("net per trade, bps", n(s["net_bps_per_trade"])),
                  ("costs as % of gross PnL", n(100 * s["cost_share_of_gross"]) + "%" if s["cost_share_of_gross"] else "n/a"),
                  ("max drawdown", R.usd(s["max_drawdown_usd"])),
              ]]), "",
              "**Calibration.** Expanding-window slopes of realized on predicted target, by block: "
              + ", ".join(f"{k}: {n(v)}" for k, v in slopes.items() if k >= 1)
              + ". A slope below 1 means raw forecasts overstate the edge.", ""]
    if rob["deciles"]:
        lines += [mp.picture(ctx.name("b_calibration"), "Calibration by decile"), ""]
    lines += ["**Cost sensitivity.** \"Adapted\" re-decides with the true cost; \"unadapted\" keeps deciding with the "
              "1× estimate while paying more, which is what an underestimated cost model does to you. The oracle "
              "knows every target: it is the ceiling for any model under this cost model.", "",
              R.table([{"cost": f"{m:g}x",
                        "adapted Sharpe": cr[m]["adapted"]["sharpe_annualized"],
                        "adapted PnL $": cr[m]["adapted"]["total_pnl_usd"],
                        "unadapted Sharpe": cr[m]["unadapted"]["sharpe_annualized"],
                        "unadapted PnL $": cr[m]["unadapted"]["total_pnl_usd"],
                        "oracle net bps/trade": cr[m]["oracle"]["net_bps_per_trade"]} for m in (1.0, 2.0, 3.0)],
                      formats={"adapted Sharpe": "{:.2f}", "unadapted Sharpe": "{:.2f}", "adapted PnL $": "{:,.0f}",
                               "unadapted PnL $": "{:,.0f}", "oracle net bps/trade": "{:.2f}"}), "",
              mp.picture(ctx.name("b_cost"), "Sharpe by cost multiple"), "",
              "**Value of information.** Synthetic forecasts with a chosen IC (perfectly calibrated) go through the "
              "same rule. A cost-blind "
              "rule (trade the top decile every instant) breaks even at IC "
              + ", ".join(f"{n(be[m]) if be[m] is not None else f'> {RHOS.max():g}'} ({m:g}x)" for m in (1.0, 2.0, 3.0))
              + f"; this model's IC at decision times is {n(rob['model_ic'])}.", "",
              mp.picture(ctx.name("b_voi"), "Value of information curves"), ""]
    cap = rob.get("capacity") or {}
    if cap:
        lines += [f"**Sizing: risk or liquidity?** Kelly leverage on this PnL stream, relative to the peak capital "
                  f"actually deployed ({R.usd(cap['peak_gross_usd'])} at one instant), is "
                  f"{n(cap['kelly_leverage_on_peak'])}×. Capital is at risk for only 60 s per trade, so risk never binds. "
                  f"Liquidity does: the depth cap ({n(100 * pol.depth_frac)}% of the displayed touch, median touch "
                  f"{R.usd(cap['median_touch_usd'])}) is below the {R.usd(cap['max_notional_usd'])} per-trade cap on "
                  f"{n(100 * cap['share_trades_depth_capped'])}% of trades, and the average trade is "
                  f"{R.usd(cap['mean_notional_usd'])}. The decision that matters is which trades to take, not how "
                  "much leverage to use.", ""]

    lines += ["## 4. Honest evaluation", ""]
    if dsr:
        lines += [R.bullets([
            f"Trials run: {hon['n_trials']} ({hon['n_trading_trials']} ever traded); effective independent trials "
            f"after clustering near-duplicates: {hon['n_effective']}.",
            f"Selected daily Sharpe {n(dsr['sr'])} over T = {dsr['T']} days (skew {n(dsr['skew'])}, kurtosis "
            f"{n(dsr['kurt'])}); PSR vs 0: {R.prob(dsr['psr_0'])}.",
            f"Expected best daily Sharpe from {hon['n_effective']} tries of pure luck: SR* = {n(dsr['sr_star'])}. "
            f"Deflated Sharpe Ratio: {R.prob(dsr['dsr'])}, the probability that the true Sharpe beats SR*.",
            f"PBO (CSCV, 16 blocks) = {n(pb.get('pbo'))}: how often the in-sample winner lands in the bottom half "
            "out of sample.",
            f"Stationary-bootstrap 95% CI for the annualized Sharpe: {n(ci.get('lo', np.nan) * np.sqrt(252))} to "
            f"{n(ci.get('hi', np.nan) * np.sqrt(252))} (block length {n(ci.get('block'))} days).",
        ]), "", mp.picture(ctx.name("c_trials"), "All trials vs the luck benchmark"), ""]
    else:
        lines += ["Too few trading trials for a Deflated Sharpe or PBO: nothing to deflate.", ""]

    lines += [f"## 5. Holdout: days {ctx.design.holdout[0]}–{ctx.design.holdout[1]}", ""]
    if hold and hold.get("adapted_1x"):
        h = hold["adapted_1x"]
        hci = hold.get("sharpe_ci", {})
        lines += [f"Scored once through `HoldoutGuard` (accesses recorded: {hold['accessed']}), with the frozen "
                  f"model, policy and a calibration slope of {n(hold['slope'])} fitted on research predictions only.",
                  "",
                  R.table([{"cost": f"{m:g}x", "adapted Sharpe": hold[f'adapted_{m:g}x']['sharpe_annualized'],
                            "adapted PnL $": hold[f'adapted_{m:g}x']['total_pnl_usd'],
                            "unadapted PnL $": hold[f'unadapted_{m:g}x']['total_pnl_usd'],
                            "trades/day": hold[f'adapted_{m:g}x']['trades_per_day']} for m in (1.0, 2.0, 3.0)],
                          formats={"adapted Sharpe": "{:.2f}", "adapted PnL $": "{:,.0f}",
                                   "unadapted PnL $": "{:,.0f}", "trades/day": "{:.1f}"}), "",
                  f"Holdout MAE: model {n(hold['mae_model'])} vs median {n(hold['mae_median'])} bps "
                  f"({pct(hold['mae_model'], hold['mae_median']):+.2f}%). For reference, a public repo using the same "
                  f"holdout days reports {PUBLIC_BENCH['median']} (median) and {PUBLIC_BENCH['gbm']} (gradient "
                  f"boosting) ([source]({PUBLIC_BENCH['url']}))"
                  + ("; the matching baseline is an independent check that the data and the split agree."
                     if abs(hold["mae_median"] - PUBLIC_BENCH["median"]) < 0.01 else ".")
                  + (" At 2x and 3x cost the adapted rule barely trades "
                     f"({n(hold['adapted_2x']['trades_per_day'])} and {n(hold['adapted_3x']['trades_per_day'])} a day), "
                     "so its Sharpe there rests on a handful of trades; the unadapted column is the informative one."
                     if hold.get("adapted_2x", {}).get("trades_per_day", 99) < 2 else "")
                  + (f" Bootstrap 95% CI for the holdout annualized Sharpe: {n(hci['lo'] * np.sqrt(252))} to "
                     f"{n(hci['hi'] * np.sqrt(252))}." if hci else ""), "",
                  mp.picture(ctx.name("b_equity"), "Cumulative PnL, research and holdout"), ""]
    else:
        lines += ["Not run yet (`--no-holdout`).", "", mp.picture(ctx.name("b_equity"), "Cumulative PnL"), ""]

    losers = [t for t in trials if t.summary["total_pnl_usd"] < 0]
    idle = [t for t in trials if t.summary["trades"] == 0]
    ramp = [t for t in trials if t.policy.sizing == "ramp"]
    flat = [t for t in trials if t.policy.sizing == "flat"]
    lines += ["## 6. What didn't work", "", R.bullets([
        f"{len(losers)} of {len(trials)} trials lost money after costs; {len(idle)} never traded because no "
        "calibrated edge cleared their threshold.",
        f"Ramp vs flat sizing: mean annualized Sharpe {n(np.mean([t.summary['sharpe_annualized'] for t in ramp]))} vs "
        f"{n(np.mean([t.summary['sharpe_annualized'] for t in flat]))}.",
        "Models that did not beat the per-stock median on MAE: "
        + (", ".join(model_label(sp) for sp in MODELS if sp.kind not in ("median",)
                     and fc["metrics"][sp.key]["mae"] >= m_med["mae"]) or "none") + ".",
    ]), ""]
    lines += ["## 7. Limitations", "", R.bullets([
        "The target is relative to a synthetic index, so PnL assumes a hedge whose cost is an assumption here.",
        "Fills are at the touch with no queue, no latency and no market impact from our own orders; depth caps "
        "limit size but do not model impact. Module D measures what realistic fills do to a signal.",
        "The closing cross itself (4:00 pm) is not traded: positions opened at 540 s are marked to the "
        "600 s WAP that the target uses.",
        f"The holdout is {ctx.design.holdout[1] - ctx.design.holdout[0] + 1} days, so its Sharpe CI is wide.",
        "The data is anonymized (no tickers), so no sector, event or corporate-action checks are possible.",
        "Stocks that first trade after the index-weight window get zero index weight, a small error in the "
        "index-relative features and in the neutralization on the days they trade.",
    ]), ""]
    lines += ["## Reproduce", "", "```bash", "make optiver                        # download + convert (Kaggle login needed)",
              "python -m markout.auction.report    # this report (cached forecasts make re-runs fast)",
              "python -m markout.backtest.report   # the daily PnL report of the selected policy", "```"]
    R.write(ctx.name("01_auction"), "\n".join(lines))


def summary_json(ctx, fc, best, trials, rob, hon, hold, slopes) -> dict:
    return {
        "synthetic": ctx.synthetic, "data_fingerprint": ctx.data_fp, "design": ctx.design.__dict__,
        "audit": ctx.out["audit"],
        "forecasts": {model_label(s): {k: v for k, v in fc["metrics"][s.key].items() if not isinstance(v, pl.DataFrame)}
                      for s in MODELS},
        "diebold_mariano": fc.get("dm"),
        "selected": {"model": model_label(best.spec), "spec": best.spec.to_dict(), "policy": best.policy.to_dict(),
                     "trial_id": best.trial_id, "summary": best.summary},
        "trials": [{"trial_id": t.trial_id, "model": model_label(t.spec), "policy": t.policy.to_dict(),
                    **{k: t.summary[k] for k in ("sharpe_annualized", "total_pnl_usd", "trades_per_day",
                                                 "net_bps_per_trade")}} for t in trials],
        "calibration_slopes": slopes,
        "robustness": {k: v for k, v in rob.items()},
        "honest": hon,
        "holdout": hold,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-holdout", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    ctx = make_ctx(args.synthetic)
    print(f"data: {ctx.df.height:,} rows, fingerprint {ctx.data_fp}", flush=True)
    a = run_audit(ctx)
    weights = a["weights"]
    print("forecasts:", flush=True)
    preds, fc = run_forecasts(ctx, weights)
    trials, tinfo = run_trials(ctx, preds)
    best = select(trials)
    med_spec = next(s for s in MODELS if s.kind == "median")
    fc["dm"] = dm_vs(fc["metrics"][best.spec.key], fc["metrics"][med_spec.key])
    dec = tinfo["decision_frames"][best.spec.key]
    model_ic = float(np.corrcoef(dec["pred"].to_numpy(), dec["target"].to_numpy())[0, 1])
    rob = robustness(ctx, best, dec, model_ic)
    hon = honest(ctx, trials, best, tinfo["registry"])
    hold = None
    if not args.no_holdout:
        print("holdout:", flush=True)
        hold = run_holdout(ctx, best, weights, preds[best.spec.key], med_spec)
    figures(ctx, fc, best, med_spec.key, rob, hon, hold)
    slopes = tinfo["slopes"][best.spec.key]
    write_report(ctx, a, fc, best, trials, rob, hon, hold, slopes)
    results.save(ctx.name("auction"), summary_json(ctx, fc, best, trials, rob, hon, hold, slopes))
    best.daily.write_csv(RESULTS / f"{ctx.name('auction_daily_research')}.csv")
    if hold and hold.get("daily_pnl"):
        pl.DataFrame({"date_id": hold["days"], "n_trades": hold["daily_trades"], "pnl_usd": hold["daily_pnl"]}
                     ).write_csv(RESULTS / f"{ctx.name('auction_daily_holdout')}.csv")
    print(f"done in {time.time() - t0:.0f}s -> reports/{ctx.name('01_auction')}.md", flush=True)


if __name__ == "__main__":
    main()
