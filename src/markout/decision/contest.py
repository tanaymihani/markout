"""Pre-registered sizing study for the SIG Predictions Cup:  python -m markout.decision.contest

Only the top three final bankrolls win a prize, so the payoff is convex in rank.
Kelly sizing maximises expected log wealth, which is the median-wealth-maximising
choice over many bets. It is not what maximises P(1st) when 1st place is all that
pays. This module simulates a contest to measure the gap, for a player with a
modest real edge, under explicit and adjustable assumptions (the real rules, field
size and market count are unknown).

Model
-----
* Markets: binary events in `n_rounds` rounds of `markets_per_round` independent
  markets, each resolving before the next round. The truth is logit(pi) ~ N(0, truth_sd^2)
  and the outcome is Y ~ Bernoulli(pi).
* Prices: logit(q) = logit(pi) + N(0, market_noise^2), clipped to [c, 1 - c]. Players
  are price-takers: no slippage, fees or position limits.
* Beliefs: every player sees the truth plus independent noise, logit(p) = logit(pi) +
  N(0, s^2). A few field players are sharper than the market (s < market_noise); most
  are noisier. The user's noise is `user_noise_ratio` x market_noise.
* Sizing: each player stakes c x Kelly (kelly.kelly_binary, on their own belief) on every
  market in a round, scaled down so the round's stakes never exceed the bankroll.
  "All-in" players put the whole bankroll on their single largest Kelly fraction.
  Bankrolls start equal (1.0), with no refills.
* Prizes: 1st/2nd/3rd by final bankroll (ties split evenly, a bankrupt player wins nothing).

Every user policy is evaluated against the same simulated fields and markets
(common random numbers), so differences between policies are not sampling noise
between fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from markout.decision import kelly

CONTEST_START_ET = "2026-10-01T12:00:00-04:00"   # registration opens at noon ET
CONTEST_END_ET = "2026-11-04T12:00:00-05:00"     # closes at noon ET


@dataclass(frozen=True)
class ContestConfig:
    n_players: int = 1000                  # including the user
    n_rounds: int = 10
    markets_per_round: int = 4
    truth_sd: float = 1.0                  # sd of logit(pi) across markets
    market_noise: float = 0.35             # sd of logit(q) - logit(pi)
    price_clip: float = 0.02
    user_noise_ratio: float = 0.7          # user's belief noise / market noise
    sharp_share: float = 0.05              # field players sharper than the market ...
    sharp_noise_ratio: float = 0.5         # ... with this noise ratio
    noisy_ratio_range: tuple = (1.0, 3.0)  # the rest: uniform noise ratio in this range
    field_allin_share: float = 0.10        # field players who go all-in on their best edge
    field_multipliers: tuple = ((0.5, 0.30), (1.0, 0.35), (2.0, 0.20), (4.0, 0.15))  # (c, share) for the rest
    prizes: tuple = (30000.0, 5000.0, 2500.0)
    n_sims: int = 4000
    seed: int = 20261001

    @property
    def n_markets(self) -> int:
        return self.n_rounds * self.markets_per_round

    def with_(self, **kw) -> "ContestConfig":
        return replace(self, **kw)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


N_PAID = len(ContestConfig().prizes)     # paid places in the default (real) prize list, used in labels
LOSS_THRESHOLD = 0.5                     # "lose more than half the bankroll"
WINNER_Q = (0.1, 0.9)                    # quantiles quoted for the winning bankroll


@dataclass(frozen=True)
class Policy:
    """User sizing rule. kind: 'kelly' (c x Kelly), 'allin' (whole bankroll on the best
    edge) or 'calibrated' (c x Kelly on a belief shrunk to the Bayesian posterior).
    Optionally escalate for the last `late_share` of the rounds (rounded to whole rounds),
    re-checked before every round and only while outside the top 3 if `only_if_behind`
    (the leaderboard is assumed visible)."""
    name: str
    kind: str = "kelly"
    mult: float = 1.0
    late_share: float = 0.0
    late_kind: str = "allin"
    late_mult: float = 1.0
    only_if_behind: bool = True


def spec_policies() -> list[Policy]:
    return [Policy(f"{c:g}x Kelly", "kelly", c) for c in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)] + \
           [Policy("all-in on best edge", "allin")]


def grid_policies(mults: Sequence[float]) -> list[Policy]:
    return [Policy(f"grid {c:g}x", "kelly", c) for c in mults]


def _window(share: float) -> str:
    return "whenever" if share >= 1 else f"in the last {share:.0%} of rounds if"


def adaptive_policies(base: float, late: Sequence[float]) -> list[Policy]:
    """Start at base x Kelly; escalate (all-in, or 3x Kelly) late in the contest while outside the top 3."""
    out = []
    for sh in late:
        out.append(Policy(f"{base:g}x Kelly, all-in {_window(sh)} outside top {N_PAID}", "kelly", base, sh, "allin"))
        out.append(Policy(f"{base:g}x Kelly, 3x Kelly {_window(sh)} outside top {N_PAID}", "kelly", base, sh, "kelly", 3.0))
    out.append(Policy(f"{base:g}x Kelly, all-in in the last {late[0]:.0%} of rounds regardless", "kelly", base, late[0],
                      "allin", only_if_behind=False))
    return out


def late_rounds(p: Policy, n_rounds: int) -> int:
    return int(round(p.late_share * n_rounds))


# ------------------------------------------------------------------ simulation core

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _round_gain(f: np.ndarray, ret: np.ndarray, mult: np.ndarray | float, allin: np.ndarray | bool) -> np.ndarray:
    """Bankroll multiplier minus one for a round of simultaneous markets (last axis).

    c x Kelly stakes are scaled down pro rata when they sum above the bankroll;
    all-in stakes everything on the largest Kelly fraction (nothing if every f is 0).
    """
    stake = np.multiply(np.expand_dims(mult, -1), f)
    tot = stake.sum(-1, keepdims=True)
    stake = np.where(tot > 1.0, stake / np.maximum(tot, 1e-300), stake)
    gain = (stake * ret).sum(-1)
    best = f.argmax(-1)
    g_all = np.take_along_axis(ret, best[..., None], -1)[..., 0]
    g_all = np.where(np.take_along_axis(f, best[..., None], -1)[..., 0] > 0, g_all, 0.0)
    return np.where(allin, g_all, gain)


def _posterior_prob(logit_q: np.ndarray, logit_p: np.ndarray, cfg: ContestConfig) -> np.ndarray:
    """E[pi | market price, own belief] for Gaussian logit noise, via the probit approximation
    E[sigmoid(x)] ~= sigmoid(m / sqrt(1 + pi v / 8)) for x ~ N(m, v)."""
    s_u = cfg.user_noise_ratio * cfg.market_noise
    prec = 1 / cfg.truth_sd**2 + 1 / cfg.market_noise**2 + 1 / s_u**2
    m = (logit_q / cfg.market_noise**2 + logit_p / s_u**2) / prec
    return _sigmoid(m / np.sqrt(1 + math.pi / (8 * prec)))


def _field_composition(rng, B: int, M: int, cfg: ContestConfig):
    sharp = rng.random((B, M)) < cfg.sharp_share
    lo, hi = cfg.noisy_ratio_range
    ratio = np.where(sharp, cfg.sharp_noise_ratio, rng.uniform(lo, hi, (B, M)))
    allin = rng.random((B, M)) < cfg.field_allin_share
    cs, ws = zip(*cfg.field_multipliers)
    mult = rng.choice(np.array(cs), size=(B, M), p=np.array(ws) / sum(ws))
    return (ratio * cfg.market_noise).astype(np.float32), allin, mult.astype(np.float32), sharp


def simulate(cfg: ContestConfig, policies: Sequence[Policy], chunk_elems: int = 1_500_000) -> dict:
    """Simulate `cfg.n_sims` contests; return per-policy final bankrolls and prize outcomes."""
    rng = np.random.default_rng(cfg.seed)
    M, K, R = cfg.n_players - 1, cfg.markets_per_round, cfg.n_rounds
    chunk = max(1, chunk_elems // (M * K))
    prizes = np.array(cfg.prizes, float)
    n_paid = len(cfg.prizes)
    out = {p.name: {"wealth": [], "p1": [], "top3": [], "prize": []} for p in policies}
    fld = {"winner_wealth": [], "third_wealth": [], "winner_allin": [], "winner_sharp": [], "median_wealth": []}
    user_stats = {"brier_user": 0.0, "brier_market": 0.0, "n": 0, "eq_ret": 0.0, "eq_perceived": 0.0,
                  "kw_ret": 0.0, "kw_perceived": 0.0, "kw_stake": 0.0}
    done = 0
    while done < cfg.n_sims:
        B = min(chunk, cfg.n_sims - done)
        noise, allin, mult, sharp = _field_composition(rng, B, M, cfg)
        W = np.ones((B, M), np.float32)
        Wu = {p.name: np.ones(B) for p in policies}
        for r in range(R):
            L = cfg.truth_sd * rng.standard_normal((B, K))
            y = (rng.random((B, K)) < _sigmoid(L)).astype(float)
            lq = L + cfg.market_noise * rng.standard_normal((B, K))
            q = np.clip(_sigmoid(lq), cfg.price_clip, 1 - cfg.price_clip)
            lq = np.log(q / (1 - q))
            # field: beliefs, Kelly fractions and payoffs for every player x market
            pf = _sigmoid(L[:, None, :].astype(np.float32)
                          + noise[:, :, None] * rng.standard_normal((B, M, K), dtype=np.float32))
            qf = q[:, None, :].astype(np.float32)
            yes, f = kelly.kelly_binary(pf, qf)
            ret = kelly.payoff_per_unit(yes, qf, y[:, None, :].astype(np.float32))
            k = M - n_paid                  # the field's n_paid-th best bankroll marks the last paid place
            leaders = np.partition(W, k, axis=1)[:, k] if k >= 0 else np.zeros(B)
            W = W * (1.0 + _round_gain(f.astype(np.float32), ret.astype(np.float32), mult, allin)).astype(np.float32)
            # the user
            lp = L + cfg.user_noise_ratio * cfg.market_noise * rng.standard_normal((B, K))
            pu = _sigmoid(lp)
            yes_u, fu = kelly.kelly_binary(pu, q)
            ret_u = kelly.payoff_per_unit(yes_u, q, y)
            yes_c, fc = kelly.kelly_binary(_posterior_prob(lq, lp, cfg), q)
            ret_c = kelly.payoff_per_unit(yes_c, q, y)
            perceived = np.where(yes_u, (pu - q) / q, (q - pu) / (1 - q))   # E[return] under your own belief
            user_stats["brier_user"] += float(((pu - y) ** 2).sum())
            user_stats["brier_market"] += float(((q - y) ** 2).sum())
            user_stats["n"] += y.size
            user_stats["eq_ret"] += float(ret_u.sum())
            user_stats["eq_perceived"] += float(perceived.sum())
            user_stats["kw_ret"] += float((fu * ret_u).sum())
            user_stats["kw_perceived"] += float((fu * perceived).sum())
            user_stats["kw_stake"] += float(fu.sum())
            for p in policies:
                w = Wu[p.name]
                n_late = late_rounds(p, R)
                late = n_late > 0 and r >= R - n_late
                escalate = np.full(B, bool(late))
                if late and p.only_if_behind:
                    # not strictly ahead of the field's 3rd-best bankroll: at the start, when everyone is
                    # level, nobody holds a top-3 place yet, so a "whenever behind" rule escalates at once
                    escalate = w <= leaders
                kind = np.where(escalate, p.late_kind, p.kind)
                c = np.where(escalate, p.late_mult, p.mult)
                g_k = _round_gain(fu, ret_u, c, kind == "allin")
                g = np.where(kind == "calibrated", _round_gain(fc, ret_c, c, False), g_k)
                Wu[p.name] = w * (1.0 + g)
        top = np.sort(W, axis=1)
        fld["winner_wealth"].append(top[:, -1])
        fld["third_wealth"].append(top[:, -n_paid] if M >= n_paid else np.zeros(B))
        fld["median_wealth"].append(np.median(W, axis=1))
        idx = W.argmax(1)
        fld["winner_allin"].append(allin[np.arange(B), idx])
        fld["winner_sharp"].append(sharp[np.arange(B), idx])
        for p in policies:
            w = Wu[p.name]
            above = (W > w[:, None]).sum(1)
            ties = (W == w[:, None]).sum(1)
            p1, t3, prize = _prize_share(above, ties, prizes)
            alive = w > 0
            o = out[p.name]
            o["wealth"].append(w)
            o["p1"].append(p1 * alive)
            o["top3"].append(t3 * alive)
            o["prize"].append(prize * alive)
        done += B
    res = {name: {k: np.concatenate(v) for k, v in o.items()} for name, o in out.items()}
    return {"policies": res, "field": {k: np.concatenate(v) for k, v in fld.items()}, "user": user_stats}


def _prize_share(above: np.ndarray, ties: np.ndarray, prizes: np.ndarray):
    """P(1st), P(top 3) and expected prize for a player with `above` players strictly ahead
    and `ties` players level (positions above+1 .. above+ties+1, equally likely)."""
    n = ties + 1
    first = np.where(above == 0, 1.0 / n, 0.0)
    top3 = np.clip(len(prizes) - above, 0, n) / n
    prize = np.zeros(above.shape)
    for k, amount in enumerate(prizes):            # position k+1 is shared if above <= k <= above + ties
        prize += np.where((above <= k) & (k <= above + ties), amount / n, 0.0)
    return first, top3, prize


def summarize(sim: dict, cfg: ContestConfig) -> list[dict]:
    """The metrics per policy, with binomial standard errors for the probabilities."""
    rows = []
    n = cfg.n_sims
    for name, o in sim["policies"].items():
        w = o["wealth"]
        p1, t3 = float(o["p1"].mean()), float(o["top3"].mean())
        rows.append({"policy": name, "p_first": p1, "p_first_se": math.sqrt(max(p1 * (1 - p1), 1e-12) / n),
                     "p_top3": t3, "p_top3_se": math.sqrt(max(t3 * (1 - t3), 1e-12) / n),
                     "expected_prize": float(o["prize"].mean()), "expected_prize_se": float(o["prize"].std(ddof=1) / math.sqrt(n)),
                     "median_bankroll": float(np.median(w)), "p90_bankroll": float(np.quantile(w, 0.9)),
                     "p_lose_half": float((w < LOSS_THRESHOLD).mean()), "p_bust": float((w <= 0).mean()),
                     "lift_first": p1 * cfg.n_players})
    return rows


def field_summary(sim: dict, cfg: ContestConfig) -> dict:
    f, u = sim["field"], sim["user"]
    return {"winner_median": float(np.median(f["winner_wealth"])),
            "winner_p10": float(np.quantile(f["winner_wealth"], WINNER_Q[0])),
            "winner_p90": float(np.quantile(f["winner_wealth"], WINNER_Q[1])),
            "third_median": float(np.median(f["third_wealth"])),
            "field_median": float(np.median(f["median_wealth"])),
            "winner_allin_share": float(f["winner_allin"].mean()),
            "winner_sharp_share": float(f["winner_sharp"].mean()),
            "brier_user": u["brier_user"] / u["n"], "brier_market": u["brier_market"] / u["n"],
            "brier_skill": 1 - u["brier_user"] / u["brier_market"],
            "edge_realized": u["eq_ret"] / u["n"], "edge_perceived": u["eq_perceived"] / u["n"],
            "edge_realized_kelly_weighted": u["kw_ret"] / max(u["kw_stake"], 1e-12),
            "edge_perceived_kelly_weighted": u["kw_perceived"] / max(u["kw_stake"], 1e-12)}


# ======================================================================= the study

BASE = ContestConfig(n_sims=10_000)
GRID = (0.1, 0.75, 1.5, 4.0, 7.0, 10.0)       # extra Kelly multipliers for the curves
ADAPT_BASE = 0.5                              # the adaptive policies start at half Kelly
LATE = (0.2, 0.4, 0.6, 0.8, 1.0)              # escalation windows, as a share of the rounds (1 = whenever behind)
CALIBRATED = Policy("1x Kelly on calibrated belief", "calibrated", 1.0)
SENSITIVITY = {
    "n_players": (250, 1000, 4000),
    "n_rounds": (5, 10, 20),
    "user_noise_ratio": (0.5, 0.7, 1.0, 1.5),
    "field_allin_share": (0.0, 0.02, 0.10, 0.25),
}
SENS_LABELS = {"n_players": "players in the field", "n_rounds": "rounds (4 markets each)",
               "user_noise_ratio": "your belief noise ÷ market noise", "field_allin_share": "share of the field going all-in"}
EXTRA_SCENARIOS = {                           # combined changes that one-at-a-time sweeps cannot express
    "cautious field (no all-in, c <= 1)": {"field_allin_share": 0.0,
                                           "field_multipliers": ((0.25, 0.3), (0.5, 0.4), (1.0, 0.3))},
}
SENS_BUDGET = 1.6e8                           # n_sims x players x markets per sensitivity run
SENS_SIMS = (1500, 16000)


def all_policies() -> list[Policy]:
    return spec_policies() + grid_policies(GRID) + [CALIBRATED] + adaptive_policies(ADAPT_BASE, LATE)


def _row(rows: list[dict], name: str) -> dict:
    return next(r for r in rows if r["policy"] == name)


def _kelly_curve(rows: list[dict]) -> list[dict]:
    """(multiplier, metrics) for every plain c x Kelly policy, sorted by c."""
    out = []
    for r in rows:
        name = r["policy"]
        if name.endswith("x Kelly") and "," not in name:
            c = float(name.split("x")[0])
        elif name.startswith("grid "):
            c = float(name[5:-1])
        else:
            continue
        out.append({"mult": c, **r})
    return sorted(out, key=lambda r: r["mult"])


def run_study(base: ContestConfig = BASE) -> tuple[dict, dict]:
    t0 = time.time()
    pols = all_policies()
    sim = simulate(base, pols)
    rows = summarize(sim, base)
    fs = field_summary(sim, base)
    figdata = {"winners": sim["field"]["winner_wealth"], "third": sim["field"]["third_wealth"],
               "user_wealth": {n: sim["policies"][n]["wealth"] for n in ("1x Kelly", "5x Kelly")}}
    sens = {}
    for j, (param, values) in enumerate(SENSITIVITY.items()):
        sens[param] = []
        for i, v in enumerate(values):
            if v == getattr(base, param):
                sens[param].append({"value": v, "n_sims": base.n_sims, "rows": rows, "field": fs, "is_base": True})
                continue
            cfg = base.with_(**{param: v})
            n = int(np.clip(SENS_BUDGET / (cfg.n_players * cfg.n_markets), *SENS_SIMS))
            cfg = cfg.with_(n_sims=n, seed=base.seed + 1000 * (j + 1) + i)
            s = simulate(cfg, pols)
            sens[param].append({"value": v, "n_sims": n, "rows": summarize(s, cfg), "field": field_summary(s, cfg),
                                "is_base": False})
    extra = []
    for i, (label, kw) in enumerate(EXTRA_SCENARIOS.items()):
        cfg = base.with_(**kw)
        n = int(np.clip(SENS_BUDGET / (cfg.n_players * cfg.n_markets), *SENS_SIMS))
        cfg = cfg.with_(n_sims=n, seed=base.seed + 9000 + i)
        s = simulate(cfg, pols)
        extra.append({"label": label, "n_sims": n, "rows": summarize(s, cfg), "field": field_summary(s, cfg)})
    digest = hashlib.sha256(json.dumps([[r["policy"], round(r["expected_prize"], 6), round(r["median_bankroll"], 9)]
                                        for r in rows]).encode()).hexdigest()[:12]
    out = {"config": asdict(base), "config_digest": base.digest(), "result_digest": digest,
           "policies": [asdict(p) for p in pols], "base": rows, "field": fs, "curve": _kelly_curve(rows),
           "sensitivity": sens, "extra": extra, "seconds": time.time() - t0,
           "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}
    return out, figdata


def scenarios(C: dict) -> list[tuple[str, dict]]:
    """Every distinct scenario run: the base case once, each sensitivity value, the extras."""
    out = [("base case", {"rows": C["base"], "field": C["field"], "n_sims": C["config"]["n_sims"]})]
    out += [(f"{SENS_LABELS[k]} = {e['value']:g}", e) for k, v in C["sensitivity"].items() for e in v if not e["is_base"]]
    out += [(e["label"], e) for e in C.get("extra", [])]
    return out


def robustness(C: dict, candidates: Sequence[str]) -> list[dict]:
    """For each policy: expected prize relative to the best policy, in every scenario run."""
    scen = scenarios(C)
    out = []
    for name in candidates:
        ratios, wins = [], 0
        for _, e in scen:
            best = max(r["expected_prize"] for r in e["rows"] if not r["policy"].startswith("grid"))
            val = _row(e["rows"], name)["expected_prize"]
            ratios.append(val / best if best > 0 else float("nan"))
            wins += val == best
        worst = int(np.nanargmin(ratios))
        out.append({"policy": name, "worst_ratio": float(np.nanmin(ratios)), "mean_ratio": float(np.nanmean(ratios)),
                    "worst_scenario": scen[worst][0], "wins": int(wins), "scenarios": len(scen)})
    return sorted(out, key=lambda r: (-r["worst_ratio"], -r["mean_ratio"]))


# ======================================================================== figures

def _figures(C: dict, fd: dict) -> None:
    from markout import plotting as mp

    rows, cfg = C["base"], C["config"]
    curve = C["curve"]
    allin = _row(rows, "all-in on best edge")
    calib = _row(rows, CALIBRATED.name)
    M = cfg["n_players"]

    def sizing():
        fig, axes = mp.subplots(1, 3, w=10.4, h=3.9)
        x = [r["mult"] for r in curve]
        a1, a2, a3 = axes
        a1.plot(x, [r["median_bankroll"] for r in curve], "o-", color=mp.series(0), label="c × Kelly on your belief")
        a1.axhline(calib["median_bankroll"], color=mp.series(2), linestyle="--", linewidth=1.2,
                   label="1 × Kelly on a calibrated belief")
        a1.axhline(1.0, color=mp.baseline_color(), linewidth=0.9)
        a1.set_ylabel("median final bankroll (start = 1)")
        mp.legend(a1, loc="lower left", fontsize=8)
        mp.title(a1, "Growth peaks below full Kelly", "median over simulated contests")
        a2.plot(x, [100 * r["p_first"] for r in curve], "o-", color=mp.series(0), label="P(1st)")
        a2.plot(x, [100 * r["p_top3"] for r in curve], "o-", color=mp.series(1), label=f"P(top {N_PAID})")
        a2.axhline(100 * allin["p_first"], color=mp.series(0), linestyle="--", linewidth=1.2, label="all-in: P(1st)")
        a2.axhline(100 * allin["p_top3"], color=mp.series(1), linestyle="--", linewidth=1.2, label=f"all-in: P(top {N_PAID})")
        a2.axhline(100 / M, color=mp.ink("muted"), linestyle=":", linewidth=1.2, label="a random player's P(1st)")
        a2.set_ylabel("probability (%)")
        mp.legend(a2, loc="upper left", fontsize=8)
        mp.title(a2, "Winning needs far more risk", f"{M:,} players, top {len(cfg['prizes'])} paid")
        a3.plot(x, [r["expected_prize"] for r in curve], "o-", color=mp.series(0), label="c × Kelly")
        a3.axhline(allin["expected_prize"], color=mp.series(3), linestyle="--", linewidth=1.2, label="all-in on best edge")
        a3.set_ylabel("expected prize ($)")
        mp.legend(a3, loc="upper left", fontsize=8)
        mp.title(a3, "Expected prize", "prizes $" + " / $".join(f"{p:,.0f}" for p in cfg["prizes"]))
        a2.set_ylim(0, 1.75 * max(100 * allin["p_top3"], max(100 * r["p_top3"] for r in curve)))
        a3.set_ylim(0, 1.45 * max(allin["expected_prize"], max(r["expected_prize"] for r in curve)))
        ticks = [0.1, 0.25, 0.5, 1, 2, 5, 10]
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.minorticks_off()
            ax.set_xlabel("Kelly multiplier c")
        return fig

    def winner():
        fig, ax = mp.subplots()
        w = np.log10(np.maximum(fd["winners"], 1e-3))
        ax.hist(w, bins=60, color=mp.series(0), alpha=0.85, edgecolor=mp.surface(), linewidth=0.4,
                label="winning bankroll")
        for i, (name, wealth) in enumerate(fd["user_wealth"].items()):
            med = float(np.median(wealth))
            if med > 0:
                ax.axvline(math.log10(med), color=mp.series(i + 1), linewidth=1.6, label=f"your median at {name}")
        ax.axvline(math.log10(float(np.median(fd["third"]))), color=mp.ink("muted"), linestyle="--", linewidth=1.2,
                   label="median 3rd-place bankroll")
        meds = [math.log10(float(np.median(v))) for v in fd["user_wealth"].values() if float(np.median(v)) > 0]
        ticks = np.arange(math.floor(min([w.min()] + meds)), math.ceil(w.max()) + 1)
        ax.set_xticks(ticks, [f"{10.0 ** t:g}×" for t in ticks])
        ax.set_xlabel("final bankroll (multiple of the start, log scale)")
        ax.set_ylabel("simulated contests")
        mp.legend(ax, loc="upper right", fontsize=8)
        mp.title(ax, "What it takes to win", f"{cfg['n_players']:,} players, {cfg['n_rounds'] * cfg['markets_per_round']} markets")
        return fig

    def adaptive():
        fig, (a1, a2) = mp.subplots(1, 2, w=9.0, h=3.8)
        pol = {p.name: p for p in adaptive_policies(ADAPT_BASE, LATE)}
        xs = [100 * sh for sh in LATE]
        for i, (kind, mult, lab) in enumerate((("allin", 1.0, "all-in"), ("kelly", 3.0, "3× Kelly"))):
            names = [next(n for n, p in pol.items() if p.late_kind == kind and p.late_mult == mult and p.only_if_behind
                          and p.late_share == sh) for sh in LATE]
            pts = [_row(rows, n) for n in names]
            a1.plot(xs, [p["expected_prize"] for p in pts], "o-", color=mp.series(i), label=f"escalate to {lab} while outside top {N_PAID}")
            a2.plot(xs, [p["median_bankroll"] for p in pts], "o-", color=mp.series(i), label=f"escalate to {lab} while outside top {N_PAID}")
        base_row = _row(rows, f"{ADAPT_BASE:g}x Kelly")
        for ax, key in ((a1, "expected_prize"), (a2, "median_bankroll")):
            ax.axhline(base_row[key], color=mp.ink("muted"), linestyle=":", linewidth=1.2, label=f"{ADAPT_BASE:g}× Kelly throughout")
            ax.axhline(allin[key], color=mp.series(3), linestyle="--", linewidth=1.2, label="all-in throughout")
            ax.set_xlabel(f"escalation window: last share of the {cfg['n_rounds']} rounds (%)")
            ax.set_xticks(xs)
        a1.set_ylabel("expected prize ($)")
        a2.set_ylabel("median final bankroll")
        mp.legend(a1, loc="upper left", fontsize=8)
        mp.title(a1, "When to increase risk", "expected prize")
        mp.title(a2, "What it costs", "median final bankroll")
        return fig

    def sensitivity():
        fig, axes = mp.subplots(2, 2, w=9.4, h=6.6)
        adapt = [p.name for p in adaptive_policies(ADAPT_BASE, LATE)]
        best_adapt = max((r for r in rows if r["policy"] in adapt), key=lambda r: r["expected_prize"])["policy"]
        focus = [f"{ADAPT_BASE:g}x Kelly", "1x Kelly", "3x Kelly", "5x Kelly", "all-in on best edge", best_adapt]
        for ax, (param, entries) in zip(axes.ravel(), C["sensitivity"].items()):
            xs = np.arange(len(entries))
            for i, name in enumerate(focus):
                lift = [_row(e["rows"], name)["p_first"] * (e["value"] if param == "n_players" else M) for e in entries]
                ax.plot(xs, lift, "o-", color=mp.series(i), label=name)
            ax.axhline(1.0, color=mp.ink("muted"), linestyle=":", linewidth=1.1)
            ax.set_xticks(xs, [f"{e['value']:g}" + (" (base)" if e["is_base"] else "") for e in entries])
            ax.set_xlabel(SENS_LABELS[param])
            ax.set_ylabel("P(1st) ÷ random player's")
        mp.title(axes[0, 0], "Sensitivity of P(1st)", "dotted line: a random player")
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="outside lower center", ncol=3, fontsize=8)
        return fig

    mp.render("p_sizing", sizing)
    mp.render("p_winner", winner)
    mp.render("p_adaptive", adaptive)
    mp.render("p_sensitivity", sensitivity)


# ======================================================================= markdown

def _pc(x: float, d: int = 1) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{100 * x:.{d}f}%"


def _escalation_date(late: int, rounds: int) -> str:
    start, end = datetime.fromisoformat(CONTEST_START_ET), datetime.fromisoformat(CONTEST_END_ET)
    return (start + (end - start) * (rounds - late) / rounds).strftime("%b %d").replace(" 0", " ")


def part1_markdown(C: dict) -> str:
    from markout import plotting as mp
    from markout import reporting as rp

    cfg, rows, fs, curve = C["config"], C["base"], C["field"], C["curve"]
    M, R, K = cfg["n_players"], cfg["n_rounds"], cfg["markets_per_round"]
    N = R * K
    prizes = cfg["prizes"]
    n_paid = len(prizes)
    last_place = "third place" if n_paid == 3 else f"place {n_paid}"
    main_names = [p.name for p in spec_policies()] + [CALIBRATED.name]
    adapt_names = [p.name for p in adaptive_policies(ADAPT_BASE, LATE)]
    k1, allin, calib = _row(rows, "1x Kelly"), _row(rows, "all-in on best edge"), _row(rows, CALIBRATED.name)
    base_half = _row(rows, f"{ADAPT_BASE:g}x Kelly")
    growth = max(curve, key=lambda r: r["median_bankroll"])
    cands = [r for r in rows if not r["policy"].startswith("grid")]
    best_p1 = max(cands, key=lambda r: r["p_first"])
    best_prize = max(cands, key=lambda r: r["expected_prize"])
    best_adapt = max((r for r in rows if r["policy"] in adapt_names), key=lambda r: r["expected_prize"])
    rob = robustness(C, main_names + adapt_names)
    robust = rob[0]
    pol = {p.name: p for p in all_policies()}
    scen = scenarios(C)

    def share(p: float) -> str:
        return f"none of the {cfg['n_sims']:,} simulated contests" if p == 0 else f"{_pc(p, 2)} of contests"

    first = datetime.fromisoformat(C["first_generated_utc"])
    start = datetime.fromisoformat(CONTEST_START_ET)
    prereg = (f"First generated {first:%Y-%m-%d %H:%M} UTC, before the contest opens ({start:%b %d, %H:%M} ET), "
              "so this is the pre-registered analysis. Commit it before the contest starts, so the timestamp can be checked."
              if first < start else
              f"Generated {first:%Y-%m-%d %H:%M} UTC, after the contest opened, so it is not a pre-registration.")

    def prow(r):
        return {"policy": r["policy"], "P(1st)": f"{_pc(r['p_first'], 2)} ± {_pc(r['p_first_se'], 2)}",
                f"P(top {n_paid})": _pc(r["p_top3"], 2), "expected prize": f"${r['expected_prize']:,.0f} ± {r['expected_prize_se']:,.0f}",
                "median bankroll": f"{r['median_bankroll']:.3f}", f"P(lose > {LOSS_THRESHOLD:.0%})": _pc(r["p_lose_half"], 1),
                "P(bust)": _pc(r["p_bust"], 1)}

    mults = ", ".join(f"{c:g} ({w:.0%})" for c, w in cfg["field_multipliers"])
    lo, hi = cfg["noisy_ratio_range"]
    assumptions = [
        {"assumption": "players (including you)", "value": f"{M:,}"},
        {"assumption": "markets", "value": f"{N}: {R} rounds of {K} independent binary markets; each round resolves before the next"},
        {"assumption": "truth", "value": f"logit(π) ~ N(0, {cfg['truth_sd']:g}²); outcome Y ~ Bernoulli(π)"},
        {"assumption": "market price", "value": f"logit(q) = logit(π) + N(0, {cfg['market_noise']:g}²), clipped to "
                                               f"[{cfg['price_clip']:g}, {1 - cfg['price_clip']:g}]; everyone trades at q, "
                                               "with no fees, slippage or position limits"},
        {"assumption": "beliefs", "value": "logit(p) = logit(π) + N(0, s²), independent across players and of the price"},
        {"assumption": "field accuracy", "value": f"{cfg['sharp_share']:.0%} sharper than the market (s = "
                                                 f"{cfg['sharp_noise_ratio']:g} × market noise); the rest s = U({lo:g}, {hi:g}) × market noise"},
        {"assumption": "field sizing", "value": f"{cfg['field_allin_share']:.0%} all-in on their best edge every round; the rest "
                                               f"c × Kelly on their own belief with c (share) = {mults}"},
        {"assumption": "you", "value": f"s = {cfg['user_noise_ratio']:g} × market noise"},
        {"assumption": "bankroll", "value": "equal start, no refills; a round's stakes are scaled down to at most the bankroll"},
        {"assumption": "prizes", "value": f"top {n_paid} final bankrolls: " + " / ".join(f"${p:,.0f}" for p in prizes)
                                         + "; ties split; a bankrupt player wins nothing"},
        {"assumption": "leaderboard", "value": "visible (the late-escalation rules use it)"},
        {"assumption": "simulated contests", "value": f"{cfg['n_sims']:,} (seed {cfg['seed']})"},
    ]

    edge_txt = (f"Your edge in this model: your Brier score beats the market's by {_pc(fs['brier_skill'], 1)}. An equal stake "
                f"on the side your belief favours returns {fs['edge_realized']:+.1%} per unit on average, while your own "
                f"belief expects {fs['edge_perceived']:+.1%}; weighted by Kelly stakes it is "
                f"{fs['edge_realized_kelly_weighted']:+.1%} realized against {fs['edge_perceived_kelly_weighted']:+.1%} "
                f"believed. The ratio ({fs['edge_realized'] / fs['edge_perceived']:.2f}) is the optimizer's curse: you act "
                "where your own error is largest. One caveat on the model: because each belief is the truth plus "
                "*independent* noise, every disagreement with the price carries some information, even from players "
                "noisier than the market. Real players' errors overlap with the market's, so real edges are smaller "
                "than the model's.")

    growth_txt = (f"The median final bankroll peaks at {growth['mult']:g}× Kelly ({growth['median_bankroll']:.3f} times the "
                  f"start), against {k1['median_bankroll']:.3f} at full Kelly and {calib['median_bankroll']:.3f} for full "
                  f"Kelly on a calibrated belief (your belief shrunk to the Bayesian posterior given the price).")
    if growth["mult"] < 1:
        growth_txt += (" Full Kelly on a raw belief overbets, because part of every disagreement with the price is your own "
                       "noise. That is the optimizer's curse again: a fraction below one on a noisy belief does the work of "
                       "full Kelly on a calibrated one.")
    bp = pol[best_p1["policy"]]
    above = bp.kind == "allin" or bp.mult > 1 or bp.late_share > 0
    p1_txt = (f"The highest P(1st) is {best_p1['policy']}: {_pc(best_p1['p_first'], 2)}, {best_p1['lift_first']:.1f}× a random "
              f"player's {_pc(1 / M, 2)}, against {_pc(k1['p_first'], 2)} at 1× Kelly. ")
    p1_txt += ("So the sizing that maximises P(1st) is far above Kelly, as the convex payoff predicts. "
               if above else "Here the sizing that maximises P(1st) is not above Kelly, contrary to the convexity argument. ")
    p1_txt += (f"It pays for that with the median: a median bankroll of {best_p1['median_bankroll']:.3f}, and more than half "
               f"the bankroll lost {_pc(best_p1['p_lose_half'], 0)} of the time. The best expected prize comes from "
               f"\"{best_prize['policy']}\": ${best_prize['expected_prize']:,.0f}, against ${k1['expected_prize']:,.0f} at "
               f"1× Kelly. For scale, the prize pool split evenly would pay ${sum(prizes) / M:,.2f} per player.")

    win_txt = (f"The winning bankroll has a median of {fs['winner_median']:,.0f}× the start "
               f"({100 * WINNER_Q[0]:.0f}th–{100 * WINNER_Q[1]:.0f}th percentile "
               f"{fs['winner_p10']:,.0f}×–{fs['winner_p90']:,.0f}×), and {last_place} needs a median {fs['third_median']:,.0f}×. "
               f"{_pc(fs['winner_allin_share'], 0)} of contests were won by a field player going all-in every round, and "
               f"{_pc(fs['winner_sharp_share'], 0)} by one of the sharp players. Your median at {growth['mult']:g}× Kelly is "
               f"{growth['median_bankroll']:.2f}×, and at 1× Kelly you reach the top {n_paid} in {share(k1['p_top3'])}. In a "
               f"{M:,}-player field over {N} markets, forecasting skill does not move you up the ranking; variance does.")

    doublings = math.log2(fs["third_median"] / base_half["median_bankroll"])
    arows = [prow(_row(rows, n)) for n in adapt_names] + [prow(base_half), prow(allin)]
    adapt_txt = (f"The best escalation rule by expected prize is \"{best_adapt['policy']}\": ${best_adapt['expected_prize']:,.0f} "
                 f"and P(top {n_paid}) {_pc(best_adapt['p_top3'], 2)}, against ${base_half['expected_prize']:,.0f} for "
                 f"{ADAPT_BASE:g}× Kelly throughout and ${allin['expected_prize']:,.0f} for all-in throughout. ")
    L_best = pol[best_adapt["policy"]].late_share
    adapt_txt += (f"Short windows do not work. {last_place.capitalize()} needs a median {fs['third_median']:,.0f}×, and from the "
                  f"{base_half['median_bankroll']:.2f}× that {ADAPT_BASE:g}× Kelly typically reaches that is "
                  f"{doublings:.1f} doublings at even-money prices, so the gamble has to start with several rounds left. ")
    adapt_txt += (f"Escalating only while outside the top {n_paid} also beats going all-in from the start: a player who gets "
                  "lucky early stops gambling and keeps the place. "
                  if best_adapt["expected_prize"] > allin["expected_prize"] else
                  "Going all-in from the start beats every escalation rule here. ")
    if L_best >= 1:
        adapt_txt += (f"The best window is the whole contest: gamble whenever you are behind, and protect a top-{n_paid} place "
                      "when you have one. ")
    elif L_best == max(LATE):
        adapt_txt += f"The best window is the longest one tried (the last {L_best:.0%} of rounds). "
    adapt_txt += (f"The price is the median bankroll, which falls from {base_half['median_bankroll']:.3f} to "
                  f"{best_adapt['median_bankroll']:.3f}, with a bust in {_pc(best_adapt['p_bust'], 0)} of contests.")

    srows = []
    for label, e in scen:
        rr = e["rows"]
        cc = [r for r in rr if not r["policy"].startswith("grid")]
        b1 = max(cc, key=lambda r: r["p_first"])
        bpz = max(cc, key=lambda r: r["expected_prize"])
        g = max(_kelly_curve(rr), key=lambda r: r["median_bankroll"])
        srows.append({"scenario": label, "contests": f"{e['n_sims']:,}", "your Brier skill": _pc(e["field"]["brier_skill"], 1),
                      "your true edge": f"{e['field']['edge_realized']:+.1%}",
                      "best P(1st)": f"{b1['policy']} ({_pc(b1['p_first'], 2)})",
                      "best expected prize": f"{bpz['policy']} (${bpz['expected_prize']:,.0f})",
                      "growth-optimal c": f"{g['mult']:g}", "median winner": f"{e['field']['winner_median']:,.0f}×",
                      "_above": pol[b1["policy"]].kind == "allin" or pol[b1["policy"]].mult > 1 or pol[b1["policy"]].late_share > 0})
    n_above = sum(r.pop("_above") for r in srows)
    edge_scen = [e for k, v in C["sensitivity"].items() if k == "user_noise_ratio" for e in v]
    prize_by_edge = [_row(e["rows"], robust["policy"])["expected_prize"] for e in edge_scen]
    skill_by_edge = [e["field"]["brier_skill"] for e in edge_scen]
    g_by_edge = [max(_kelly_curve(e["rows"]), key=lambda r: r["median_bankroll"])["mult"] for e in edge_scen]
    sens_txt = f"In {n_above} of {len(srows)} scenarios the P(1st)-maximising policy is more aggressive than 1× Kelly."
    if len(edge_scen) >= 2:
        flat = max(prize_by_edge) < 1.5 * min(prize_by_edge) if min(prize_by_edge) > 0 else False
        sens_txt += (f" Your edge moves the growth-optimal multiplier, from {g_by_edge[0]:g}× at a Brier skill of "
                     f"{_pc(skill_by_edge[0], 1)} to {g_by_edge[-1]:g}× at {_pc(skill_by_edge[-1], 1)}. Across the same "
                     f"range the expected prize of \"{robust['policy']}\" runs from ${min(prize_by_edge):,.0f} to "
                     f"${max(prize_by_edge):,.0f}"
                     + (": the prize barely depends on skill, so this contest format rewards variance far more than "
                        "forecasting." if flat else "."))
    rob_rows = [{"policy": r["policy"], "worst case (share of the best)": _pc(r["worst_ratio"], 0),
                 "worst scenario": r["worst_scenario"], "average share": _pc(r["mean_ratio"], 0),
                 "best in": f"{r['wins']} of {r['scenarios']}"} for r in rob[:6]]

    # ---- the pre-registered policy, chosen from the results
    rp_ = pol[robust["policy"]]
    steps = [f"Log every trade before placing it (`python -m markout.decision.journal add ...`), with the reason. "
             "Part 2 scores the beliefs, not the rank."]
    if rp_.late_share:
        n_late = late_rounds(rp_, R)
        esc = "all-in on your single largest Kelly fraction" if rp_.late_kind == "allin" else f"{rp_.late_mult:g}× Kelly"
        cond = (f"unless you hold a top-{n_paid} place, go {esc}; while you hold one, bet {rp_.mult:g}× the journal's Kelly "
                f"fraction on every market you trade" if rp_.only_if_behind else f"go {esc} regardless of rank")
        if n_late >= R:
            steps.append(f"Sizing, re-checked against the leaderboard before each round: {cond}. At the start nobody "
                         f"holds a top-{n_paid} place, so the first round is already {esc.split(' on ')[0]}. (For comparison, "
                         f"the growth-optimal multiplier on raw beliefs here is {growth['mult']:g}× Kelly.)")
        else:
            steps.append(f"Default size: {rp_.mult:g}× the journal's Kelly fraction on every market you trade. The "
                         f"growth-optimal multiplier on raw beliefs here is {growth['mult']:g}×.")
            steps.append(f"Escalate for the last {n_late} of {R} rounds (the last {rp_.late_share:.0%} of markets, from "
                         f"about {_escalation_date(n_late, R)}). Before each round: {cond}.")
    elif rp_.kind == "allin":
        steps.append("Size: all-in on your single largest Kelly fraction every round, from the start.")
    else:
        steps.append(f"Size: {rp_.mult:g}× the journal's Kelly fraction throughout"
                     + (" on a calibrated belief (shrink toward the price first)." if rp_.kind == "calibrated" else "."))
    runner = rob[1]
    rr_base, ru_base = _row(rows, robust["policy"]), _row(rows, runner["policy"])
    close = abs(rr_base["expected_prize"] - ru_base["expected_prize"]) < 2 * math.hypot(rr_base["expected_prize_se"],
                                                                                        ru_base["expected_prize_se"])
    steps.append(f"Why this rule: it is the best policy in {robust['wins']} of the {robust['scenarios']} simulated scenarios "
                 f"and keeps at least {_pc(robust['worst_ratio'], 0)} of the best policy's expected prize in all of them; "
                 f"its worst case is \"{robust['worst_scenario']}\". "
                 + (f"The runner-up, \"{runner['policy']}\", is within sampling error of it in the base case "
                    f"(${ru_base['expected_prize']:,.0f} ± {ru_base['expected_prize_se']:,.0f} against "
                    f"${rr_base['expected_prize']:,.0f} ± {rr_base['expected_prize_se']:,.0f}). "
                    if close else "")
                 + "The objective is prize money, since SUSQies are worth nothing after the close.")
    steps.append(f"It busts in {_pc(_row(rows, robust['policy'])['p_bust'], 0)} of simulated contests. After a bust, keep "
                 "logging your belief on markets you would have traded, with `--stake 0`, so Part 2 still has a full "
                 "sample of forecasts to score.")
    steps.append("Deviations are allowed but logged: start the journal's reason with \"DEVIATION:\" and say why.")
    steps.append("What would change it: position limits (all-in becomes impossible, so escalate to the largest stake "
                 "allowed), a very different field size or market count (see the table), or rules that differ from the "
                 "assumptions above.")

    return f"""## Part 1: Pre-registered sizing study (`python -m markout.decision.contest`)

*{prereg}*

**The question.** Only the top {len(prizes)} final bankrolls win a prize
({" / ".join(f"${p:,.0f}" for p in prizes)}), so the payoff is convex in rank. Kelly sizing
maximises expected log wealth, which is the growth rate and so the median bankroll over many bets.
It does not maximise P(1st). How much more risk should a player with a modest real edge take?

**Kelly for a binary contract** (Kelly 1956; Thorp 2006). Buying YES at price q with belief p and a
fraction f of the bankroll grows it at g(f) = p·ln(1 + f(1 − q)/q) + (1 − p)·ln(1 − f), which is
maximised at f* = (p − q)/(1 − q). For NO it is (q − p)/q. Fractional Kelly bets c·f*; for small
edges g(c·f*) ≈ (2c − c²)·g(f*), so half Kelly keeps about {_pc(2 * 0.5 - 0.5 ** 2, 0)} of the growth, and 2× Kelly
{_pc(2 * 2 - 2 ** 2, 0)}. The continuous analogue is f* = μ/σ².

**Assumptions.** The real contest's rules, field size and number of markets are not known. Each
number below is a choice, and the sensitivity section varies the ones that matter most.

{rp.table(assumptions)}

{edge_txt}

### Base case

{rp.table([prow(_row(rows, n)) for n in main_names])}

{mp.picture('p_sizing', 'Median bankroll, P(1st) and expected prize against the Kelly multiplier')}

The curves behind the figure (every c × Kelly policy simulated):

{rp.table([{"c": f"{r['mult']:g}", "median bankroll": f"{r['median_bankroll']:.3f}", "P(1st)": _pc(r['p_first'], 2),
            f"P(top {n_paid})": _pc(r['p_top3'], 2), "expected prize": f"${r['expected_prize']:,.0f}",
            f"P(lose > {LOSS_THRESHOLD:.0%})": _pc(r['p_lose_half'], 1)} for r in curve])}

{growth_txt}

{p1_txt}

### What it takes to win

{mp.picture('p_winner', 'Distribution of the winning bankroll with the user median bankrolls marked')}

{win_txt}

### When to increase risk

Each adaptive rule bets {ADAPT_BASE:g}× Kelly, then escalates in the last part of the contest (a share
of the rounds, so it means the same thing when the number of markets changes), re-checked before every
round.

{rp.table(arows)}

{mp.picture('p_adaptive', 'Expected prize and median bankroll against the length of the late escalation window')}

{adapt_txt}

### Sensitivity

The sweeps vary one assumption at a time, plus one combined scenario. The number of simulated
contests keeps the work per scenario roughly constant, so the probabilities are noisier for large fields.

{mp.picture('p_sensitivity', 'P(1st) relative to a random player across field size, markets, your edge and field aggressiveness')}

{rp.table(srows)}

{sens_txt}

Robustness of expected prize across all {robust['scenarios']} scenarios (top six policies):

{rp.table(rob_rows)}

### The pre-registered policy

{rp.bullets(f"**{i + 1}.** {s}" for i, s in enumerate(steps))}
"""


def _rel(path) -> str:
    """Repo-relative path for the committed JSON (never a local absolute path)."""
    from markout.paths import ROOT

    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def write_report(journal_path=None, C: dict | None = None):
    """Regenerate reports/05_predictions_cup.md: Part 1 from the study, Part 2 from the journal."""
    from markout import reporting as rp
    from markout import results
    from markout.decision import journal

    stored = results.load("predictions_cup") or {}
    C = C if C is not None else stored.get("contest")
    jp = journal.JOURNAL if journal_path is None else journal_path
    scores, n_logged = None, 0
    if jp.exists():
        df = journal.load(jp)
        n_logged = len(df)
        scores = journal.score(df)
        if scores is not None:
            journal.figures(df, scores)
    now = datetime.now(timezone.utc)
    status = "pending" if scores is None else ("final" if now >= datetime.fromisoformat(CONTEST_END_ET) else "interim")
    results.save("predictions_cup", {"contest": C, "journal": {"status": status, "n_logged": n_logged, "scores": scores,
                                                               "journal_file": _rel(jp)}})
    head = f"""# 05 · SIG Predictions Cup: sizing for a winner-take-most contest

*Part 1 is generated by `python -m markout.decision.contest` (fixed seeds). Part 2 is generated by
`python -m markout.decision.journal score` from `data/journal.csv`. Numbers are in
`reports/results/predictions_cup.json`; none is typed by hand.*
"""
    part1 = part1_markdown(C) if C else "## Part 1\n\nNot generated yet: run `python -m markout.decision.contest`.\n"
    part2 = journal.part2_markdown(scores, n_logged, status, C["field"]["brier_skill"] if C else None)
    return rp.write("05_predictions_cup", "\n".join([head, part1, part2]))


def main() -> None:
    from markout import results

    C, fd = run_study()
    old = (results.load("predictions_cup") or {}).get("contest") or {}
    same = old.get("config_digest") == C["config_digest"] and old.get("result_digest") == C["result_digest"]
    C["first_generated_utc"] = old.get("first_generated_utc", C["generated_utc"]) if same else C["generated_utc"]
    _figures(C, fd)
    path = write_report(C=C)
    print(f"contest study: {C['seconds']:.0f} s -> {path}")


if __name__ == "__main__":
    main()
