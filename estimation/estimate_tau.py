"""
Estimate tau (prior log-SD of the market's healthy-rate estimate) from archived
preseason FantasyPros consensus projections (Wayback Machine captures), as the
substitute for historical season-long sportsbook lines (which are not available).

  r_i = log(actual full-game ypg_i) - log(projected season yards_i)
  r_i demeaned within (position, season)            <- removes the projection source's bias
  tau^2 = Var(r) - mean_i( c(mu_i)^2 / n_i )         <- subtract game-to-game sampling noise

Players are selected on the *projection* (ex ante), not on the outcome, to avoid
truncating the error distribution.  Writes estimates/tau.json.
"""
from __future__ import annotations

import io
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from estimation.estimate_params import load_panel, player_season_table, MIN_HEALTHY_GAMES

ROOT = Path(__file__).resolve().parent.parent
FP = ROOT / "data" / "raw" / "fantasypros"
EST = json.load(open(ROOT / "estimates" / "estimates.json"))

# Valid preseason capture window for season Y: May 1 .. Sep 10 of year Y.
WINDOW = ("0501", "0910")
# projection-based population bands (season yards / 15 games ~ per-game rate)
PROJ_BAND = {"QB": (3000.0, 1e9), "WR": (35 * 15.0, 90 * 15.0), "TE": (25 * 15.0, 65 * 15.0)}
TEAM_ALIAS = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS", "SD": "LAC", "STL": "LA", "OAK": "OAK", "FA": None}


def norm_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    s = re.sub(r"[^a-z ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def capture_ts(html: str):
    m = re.search(r"web/(\d{14})", html)
    return m.group(1) if m else None


def parse_page(path: Path, pos: str):
    html = open(path, encoding="utf-8", errors="ignore").read()
    ts = capture_ts(html)
    if ts is None:
        return None, None
    title = re.search(r"<title>(.*?)</title>", html, re.S)
    title = title.group(1).strip() if title else ""
    # only season-long ("draft") projection pages, e.g. "2023 QB Projections - ..."; skip weekly pages
    if not re.match(r"^\d{4} (QB|WR|TE) Projections", title):
        return ts, None
    t = pd.read_html(io.StringIO(html))[0]
    t.columns = [" ".join(c).strip() if isinstance(c, tuple) else str(c) for c in t.columns]
    t = t.rename(columns={t.columns[0]: "Player"})
    ycol = "PASSING YDS" if pos == "QB" else "RECEIVING YDS"
    if ycol not in t.columns:
        return ts, None
    rows = []
    for raw, yds in zip(t.Player.astype(str), t[ycol]):
        m = re.match(r"^(.*?)\s+([A-Z]{2,3})$", raw.strip())
        name, team = (m.group(1), m.group(2)) if m else (raw.strip(), None)
        rows.append({"fp_name": name, "name_key": norm_name(name), "fp_team": TEAM_ALIAS.get(team, team),
                     "proj_yards": float(yds)})
    return ts, pd.DataFrame(rows)


def collect_pages():
    """Every captured page, keyed by (position, season of the capture)."""
    pages = {}
    for f in sorted(FP.rglob("*.html")):
        pos = f.name.split("_")[0].replace("wrvar", "wr").upper()
        if pos not in ("QB", "WR", "TE"):
            continue
        ts, df = parse_page(f, pos)
        if ts is None or df is None or len(df) < 20:
            continue
        season = int(ts[:4])
        if not (ts[4:8] >= WINDOW[0] and ts[4:8] <= WINDOW[1]):
            continue
        key = (pos, season)
        # prefer the capture closest to (but before) kickoff: latest timestamp in the window
        if key not in pages or ts > pages[key]["ts"]:
            pages[key] = {"ts": ts, "file": str(f.relative_to(ROOT)), "df": df}
    return pages


def main():
    panel = load_panel()
    ps = player_season_table(panel)
    team = panel.groupby(["season", "gsis_id"]).team.agg(lambda s: s.mode().iloc[0]).rename("team")
    ps = ps.merge(team, on=["season", "gsis_id"])
    ps["name_key"] = ps.name.map(norm_name)
    pages = collect_pages()
    out = {"pulled": str(date.today()), "method": __doc__.strip(), "captures": {}, "positions": {}}
    for pos in ("QB", "WR", "TE"):
        a, b = EST["cv"][pos]["a"], EST["cv"][pos]["b"]
        frames = []
        for (p, season), pg in sorted(pages.items()):
            if p != pos:
                continue
            out["captures"][f"{pos}_{season}"] = {"wayback_timestamp": pg["ts"], "file": pg["file"], "n_rows": int(len(pg["df"]))}
            lo, hi = PROJ_BAND[pos]
            proj = pg["df"][(pg["df"].proj_yards >= lo) & (pg["df"].proj_yards <= hi)]
            cand = ps[(ps.season == season) & (ps.position == pos)]
            m = proj.merge(cand, on="name_key", how="inner")
            # disambiguate duplicate names by team
            dup = m.groupby("name_key").gsis_id.transform("nunique") > 1
            m = m[~dup | (m.fp_team == m.team)]
            m = m.drop_duplicates("name_key", keep=False)
            frames.append(m.assign(season=season))
        if not frames:
            out["positions"][pos] = {"tau": None, "reason": "no valid preseason captures"}
            continue
        d = pd.concat(frames)
        d = d[(d.n >= MIN_HEALTHY_GAMES) & (d.mu > 0)]
        d["r"] = np.log(d.mu) - np.log(d.proj_yards)
        d["r_dm"] = d.r - d.groupby("season").r.transform("mean")
        d["noise"] = np.exp(a + b * np.log(d.mu)) ** 2 / d.n
        var_dm = float(d.r_dm.var(ddof=1))
        var_raw = float(d.r.var(ddof=1))
        noise = float(d.noise.mean())
        tau_dm = float(np.sqrt(max(var_dm - noise, 0)))
        tau_raw = float(np.sqrt(max(var_raw - noise, 0)))
        out["positions"][pos] = {
            "tau": round(tau_dm, 4), "tau_without_season_demeaning": round(tau_raw, 4),
            "var_log_error_demeaned": var_dm, "var_log_error_raw": var_raw, "mean_sampling_noise_var": noise,
            "n_player_seasons": int(len(d)), "seasons": sorted(int(s) for s in d.season.unique()),
            "n_by_season": {int(k): int(v) for k, v in d.groupby("season").size().items()},
            "mean_log_ratio_by_season": {int(k): round(float(v), 4) for k, v in d.groupby("season").r.mean().items()},
            "placeholder_from_prompt": {"QB": 0.11, "WR": 0.20, "TE": 0.22}[pos],
        }
        print(f"{pos}: tau={tau_dm:.4f} (raw {tau_raw:.4f}); var_dm={var_dm:.4f} noise={noise:.4f} n={len(d)} seasons={out['positions'][pos]['seasons']}")
    json.dump(out, open(ROOT / "estimates" / "tau.json", "w"), indent=2)
    print("captures used:", {k: v["wayback_timestamp"] for k, v in out["captures"].items()})


if __name__ == "__main__":
    main()
