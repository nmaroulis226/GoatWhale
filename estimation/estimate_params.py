"""
Estimate the model's data-driven parameters from the historical panel
(data/panel.parquet, built by estimation/panel.py):

  * CV curves  log c = a + b log mu   (QB passing, WR receiving, TE receiving)
  * per-game injury hazard h by position
  * absence-length distributions (Kaplan-Meier, right-censored at season end):
      - all injury absences that follow a played game (used for simulated injuries)
      - absences that began with a Questionable designation
      - IR stints (2022-2025, the 4-game-minimum era), conditional on >= 4 games
  * p_q = P(plays | listed Questionable), by position
  * an out-of-sample Gamma vs lognormal comparison (held-out 2024-2025)

Writes estimates/estimates.json and prints a report.  Every definition here
is documented in ASSUMPTIONS.md.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "panel.parquet"
OUT = ROOT / "estimates"

POS = ["QB", "WR", "TE"]
REL_SHARE_FULL = 0.70          # game counts as "full" if snap share >= 70% of the player-season median
QB_STARTER_MEDIAN_PCT = 0.75   # QB regular: player-season median snap share >= 75%
REG_MEDIAN_PCT = 0.50          # WR/TE regular: median snap share >= 50%
MIN_HEALTHY_GAMES = 8
YPG_BAND = {"WR": (35.0, 90.0), "TE": (25.0, 65.0)}
INJURY_DESIGNATIONS = {"Out", "Doubtful", "Questionable"}
IR_SEASONS = (2022, 2023, 2024, 2025)
IR_MIN_GAMES = 4
ABSENCE_SUPPORT = 18           # 1..17 games + season-ending cell
HOLDOUT_SEASONS = (2024, 2025)


# ----------------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PANEL)
    played = p[p.played]
    med = played.groupby(["season", "gsis_id"]).offense_pct.median().rename("med_pct")
    p = p.merge(med, on=["season", "gsis_id"], how="left")
    p["rel_share"] = p.offense_pct / p.med_pct
    p["full_game"] = p.played & (p.rel_share >= REL_SHARE_FULL)
    p["regular"] = np.where(p.position == "QB", p.med_pct >= QB_STARTER_MEDIAN_PCT, p.med_pct >= REG_MEDIAN_PCT)
    p["injury_miss"] = (~p.played) & (p.report_status.isin(INJURY_DESIGNATIONS) | p.on_ir)
    return p.sort_values(["season", "gsis_id", "week"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# CV curves
# ----------------------------------------------------------------------------
def player_season_table(p: pd.DataFrame) -> pd.DataFrame:
    hg = p[p.full_game]
    t = hg.groupby(["season", "gsis_id", "position"]).agg(
        n=("yards", "size"), mu=("yards", "mean"), sd=("yards", lambda s: s.std(ddof=1)),
        name=("full_name", "first"), med_pct=("med_pct", "first"), regular=("regular", "first")).reset_index()
    t["cv"] = t.sd / t.mu
    return t


def cv_population(t: pd.DataFrame, pos: str) -> pd.DataFrame:
    d = t[(t.position == pos) & (t.n >= MIN_HEALTHY_GAMES) & (t.mu > 0)]
    if pos == "QB":
        d = d[d.regular]
    else:
        lo, hi = YPG_BAND[pos]
        d = d[(d.mu >= lo) & (d.mu <= hi)]
    return d[d.cv > 0]


def fit_cv(d: pd.DataFrame) -> dict:
    x, y, w = np.log(d.mu.values), np.log(d.cv.values), d.n.values.astype(float)
    X = np.column_stack([np.ones_like(x), x])
    W = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    resid = y - X @ beta
    return {"a": float(beta[0]), "b": float(beta[1]), "n_player_seasons": int(len(d)),
            "n_games": int(d.n.sum()), "resid_sd_logcv": float(np.sqrt(np.average(resid ** 2, weights=w))),
            "mean_mu": float(d.mu.mean()), "mean_cv": float(d.cv.mean()),
            "cv_at": {str(m): float(np.exp(beta[0] + beta[1] * np.log(m))) for m in (25, 40, 55, 70, 90, 150, 220, 260)}}


# ----------------------------------------------------------------------------
# Gamma vs lognormal, held-out seasons
# ----------------------------------------------------------------------------
def _interval_ll_gamma(y, mu, c):
    alpha = 1 / c ** 2
    scale = mu / alpha
    hi = np.maximum(y, 0) + 0.5
    lo = np.maximum(y - 0.5, 0)
    pr = special.gammainc(alpha, hi / scale) - special.gammainc(alpha, lo / scale)
    return np.log(np.maximum(pr, 1e-300))


def _interval_ll_lognormal(y, mu, c):
    s2 = np.log1p(c ** 2)
    m = np.log(mu) - s2 / 2
    s = np.sqrt(s2)
    hi = np.maximum(y, 0) + 0.5
    lo = np.maximum(y - 0.5, 0)
    F = lambda v: np.where(v <= 0, 0.0, special.ndtr((np.log(np.maximum(v, 1e-12)) - m) / s))
    return np.log(np.maximum(F(hi) - F(lo), 1e-300))


def compare_distributions(p: pd.DataFrame, t: pd.DataFrame) -> dict:
    out = {}
    for pos in POS:
        train = cv_population(t[t.season < min(HOLDOUT_SEASONS)], pos)
        fit = fit_cv(train)
        test = cv_population(t[t.season.isin(HOLDOUT_SEASONS)], pos)
        hg = p[p.full_game & (p.position == pos)].merge(test[["season", "gsis_id", "mu"]], on=["season", "gsis_id"])
        c = np.exp(fit["a"] + fit["b"] * np.log(hg.mu.values))
        llg = _interval_ll_gamma(hg.yards.values, hg.mu.values, c).sum()
        lll = _interval_ll_lognormal(hg.yards.values, hg.mu.values, c).sum()
        out[pos] = {"n_games": int(len(hg)), "loglik_gamma": float(llg), "loglik_lognormal": float(lll),
                    "gamma_minus_lognormal_per_game": float((llg - lll) / len(hg))}
    return out


# ----------------------------------------------------------------------------
# Absences: identify starts and lengths
# ----------------------------------------------------------------------------
def absence_table(p: pd.DataFrame) -> pd.DataFrame:
    """One row per missed-game spell of a regular player.  Length in games; censored if the
    season ran out before he played again."""
    rows = []
    for (season, gid), g in p[p.regular].groupby(["season", "gsis_id"], sort=False):
        played = g.played.values
        i, n = 0, len(g)
        while i < n:
            if played[i]:
                i += 1
                continue
            j = i
            while j < n and not played[j]:
                j += 1
            length = j - i
            censored = j >= n
            first = g.iloc[i]
            ir_idx = next((k for k in range(i, j) if g.on_ir.values[k]), None)
            rows.append({
                "season": season, "gsis_id": gid, "position": first.position, "start_week": first.week,
                "after_played": i > 0 and played[i - 1],
                "prev_rel_share": g.rel_share.values[i - 1] if i > 0 else np.nan,
                "injury_start": bool(first.injury_miss), "start_designation": first.report_status,
                "length": length, "censored": censored,
                "ir_length": (j - ir_idx) if ir_idx is not None else None,
            })
            i = j
    return pd.DataFrame(rows)


KM_MIN_AT_RISK = 5


def kaplan_meier_pmf(lengths: np.ndarray, censored: np.ndarray, support: int = ABSENCE_SUPPORT) -> dict:
    """Kaplan-Meier estimate of P(L = k), k = 1..support-1, with the remaining mass P(L >= support)
    (and any unestimable tail) put in the last cell = season-ending.  Once fewer than
    KM_MIN_AT_RISK absences remain at risk the curve is frozen and the rest of the mass is
    treated as season-ending (the tail is otherwise dominated by 1-2 observations)."""
    lengths = np.asarray(lengths, dtype=int)
    censored = np.asarray(censored, dtype=bool)
    S = 1.0
    surv = []
    for k in range(1, support):
        at_risk = np.sum(lengths >= k)
        d = np.sum((lengths == k) & ~censored)
        if at_risk >= KM_MIN_AT_RISK:
            S *= 1 - d / at_risk
        surv.append(S)
    surv = np.array(surv)
    pmf = np.empty(support)
    pmf[0] = 1 - surv[0]
    pmf[1:support - 1] = surv[:-1] - surv[1:]
    pmf[support - 1] = surv[-1]
    pmf = np.maximum(pmf, 0)
    pmf /= pmf.sum()
    return {"pmf": pmf.tolist(), "n": int(len(lengths)), "n_censored": int(censored.sum()),
            "mean_games_capped_17": float(np.sum(pmf * np.r_[np.arange(1, support), 17]))}


# ----------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    p = load_panel()
    t = player_season_table(p)
    est = {"pulled": str(date.today()), "seasons": [int(s) for s in sorted(p.season.unique())],
           "definitions": {
               "full_game": f"played and snap share >= {REL_SHARE_FULL} x player-season median snap share",
               "regular_QB": f"player-season median snap share >= {QB_STARTER_MEDIAN_PCT}",
               "regular_WR_TE": f"player-season median snap share >= {REG_MEDIAN_PCT}",
               "cv_population": {"QB": f"regular QBs with >= {MIN_HEALTHY_GAMES} full games",
                                 "WR": f">= {MIN_HEALTHY_GAMES} full games and full-game mean {YPG_BAND['WR']} ypg",
                                 "TE": f">= {MIN_HEALTHY_GAMES} full games and full-game mean {YPG_BAND['TE']} ypg"},
               "injury_miss": "did not play and (Out/Doubtful/Questionable on the injury report or on an injury reserve list: R01/R48/R04/R05)",
               "hazard": "events = played games followed by an injury-caused missed game; exposure = played games of regulars with a later team game that season",
               "absence_length": "consecutive missed games from the first missed game until the next game played; censored if the season ended first (Kaplan-Meier)",
               "ir_stint": f"games missed from the first week on an injury reserve list, seasons {IR_SEASONS}, conditional on >= {IR_MIN_GAMES}",
           }}

    # CV
    est["cv"] = {}
    for pos in POS:
        d = cv_population(t, pos)
        est["cv"][pos] = fit_cv(d)
    est["distribution_comparison_holdout"] = compare_distributions(p, t)

    # hazard and p_q
    reg = p[p.regular].copy()
    reg["next_injury_miss"] = reg.groupby(["season", "gsis_id"]).injury_miss.shift(-1)
    reg["next_played"] = reg.groupby(["season", "gsis_id"]).played.shift(-1)
    exp_ = reg[reg.played & reg.next_injury_miss.notna()]
    est["hazard"] = {}
    for pos in POS:
        e = exp_[exp_.position == pos]
        ev = e.next_injury_miss.astype(bool)
        in_game = ev & (e.rel_share < REL_SHARE_FULL)
        est["hazard"][pos] = {"h": float(ev.mean()), "n_exposure_games": int(len(e)), "n_events": int(ev.sum()),
                              "share_events_in_game": float(in_game.sum() / max(ev.sum(), 1)),
                              "mean_rel_share_in_game_injury": float(e.rel_share[in_game].mean()) if in_game.any() else None}
    pooled = exp_.next_injury_miss.astype(bool)
    est["hazard"]["pooled"] = {"h": float(pooled.mean()), "n_exposure_games": int(len(exp_)), "n_events": int(pooled.sum())}

    est["p_questionable_plays"] = {}
    q = reg[reg.report_status == "Questionable"]
    for pos in POS:
        d = q[q.position == pos]
        est["p_questionable_plays"][pos] = {"p": float(d.played.mean()), "n": int(len(d))}
    est["p_questionable_plays"]["pooled"] = {"p": float(q.played.mean()), "n": int(len(q))}
    d = reg[reg.report_status == "Doubtful"]
    est["p_doubtful_plays_info"] = {"p": float(d.played.mean()), "n": int(len(d))}

    # absences
    ab = absence_table(p)
    inj = ab[ab.injury_start & ab.after_played]
    est["absence"] = {}
    for pos in POS:
        d = inj[inj.position == pos]
        est["absence"][pos] = kaplan_meier_pmf(d.length.values, d.censored.values)
    est["absence"]["pooled"] = kaplan_meier_pmf(inj.length.values, inj.censored.values)
    # raw (uncensored-only) histogram for the report
    est["absence"]["raw_uncensored_hist_pooled"] = {int(k): int(v) for k, v in
                                                     inj[~inj.censored].length.value_counts().sort_index().items()}

    qa = ab[ab.injury_start & (ab.start_designation == "Questionable")]
    est["q_absence"] = {"pooled": kaplan_meier_pmf(qa.length.values, qa.censored.values)}
    for pos in POS:
        d = qa[qa.position == pos]
        est["q_absence"][pos] = kaplan_meier_pmf(d.length.values, d.censored.values)

    ir = ab[ab.ir_length.notna() & ab.season.isin(IR_SEASONS)].copy()
    ir["ir_length"] = ir.ir_length.astype(int)
    km = kaplan_meier_pmf(ir.ir_length.values, ir.censored.values)
    pmf = np.array(km["pmf"])
    pmf[:IR_MIN_GAMES - 1] = 0.0
    pmf /= pmf.sum()
    km_cond = dict(km, pmf=pmf.tolist(),
                   mean_games_capped_17=float(np.sum(pmf * np.r_[np.arange(1, ABSENCE_SUPPORT), 17])),
                   raw_hist_uncensored={int(k): int(v) for k, v in
                                        ir[~ir.censored].ir_length.value_counts().sort_index().items()},
                   n_censored_ir=int(ir.censored.sum()))
    est["ir_stint"] = {"pooled_conditional_ge4": km_cond}

    # expected availability (for the report)
    est["availability_summary"] = {pos: {"games_played_share_regulars": float(reg[reg.position == pos].played.mean())}
                                   for pos in POS}

    with open(OUT / "estimates.json", "w") as f:
        json.dump(est, f, indent=2)
    print(json.dumps({k: est[k] for k in ("cv", "hazard", "p_questionable_plays", "distribution_comparison_holdout")}, indent=1))
    for key in ("absence", "q_absence"):
        for pos in POS + ["pooled"]:
            e = est[key][pos]
            print(f"{key} {pos}: n={e['n']} censored={e['n_censored']} mean(cap17)={e['mean_games_capped_17']:.2f} "
                  f"pmf1-6={np.round(e['pmf'][:6], 3).tolist()} season_end={e['pmf'][-1]:.3f}")
    e = est["ir_stint"]["pooled_conditional_ge4"]
    print(f"ir: n={e['n']} censored={e['n_censored']} mean(cap17)={e['mean_games_capped_17']:.2f} pmf4-10={np.round(e['pmf'][3:10], 3).tolist()} season_end={e['pmf'][-1]:.3f}")
    print("raw absence hist:", est["absence"]["raw_uncensored_hist_pooled"])
    print("raw IR hist:", e["raw_hist_uncensored"])


if __name__ == "__main__":
    main()
