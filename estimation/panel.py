"""
Build a player-game panel from locally cached nflverse files (data/raw/).

The nflverse release files are the same ones `nfl_data_py` downloads
(import_weekly_data -> stats_player_week_{season}.parquet,
 import_snap_counts -> snap_counts_{season}.parquet,
 import_injuries    -> injuries_{season}.parquet,
 import_weekly_rosters -> roster_weekly_{season}.parquet,
 import_schedules   -> games.csv).
`nfl_data_py` itself could not be installed here (it pins pandas<2.0 and the
source build of pandas 1.x fails on this machine), so the files are read
directly with pandas/pyarrow. See ASSUMPTIONS.md.

Panel row = one (player, season, team game).  Columns:
  season, week, team, gsis_id, name, position, game_idx (0-based order of the
  team's regular-season games), n_team_games, played, offense_snaps,
  offense_pct, yards (passing yards for QB, receiving yards for WR/TE),
  report_status (Out / Doubtful / Questionable / None), on_ir (roster status RES
  with an injured-reserve designation), roster_status.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
POSITIONS = ("QB", "WR", "TE")
SEASONS = tuple(range(2018, 2026))

# Roster status codes (nflverse status_description_abbr):
#   R01 = Reserve/Injured, R04 = Reserve/PUP, R05 = Reserve/NFI (non-football
#   injury), R48 = Reserve/Injured; designated for return, R27 = Reserve/Suspended
#   R30 = Reserve/COVID-19, R40 = Reserve/Retired, R02 = Reserve/Retired, ...
IR_CODES = {"R01", "R04", "R05", "R48", "R47", "R30", "R37"}  # injury-type reserve lists


def _norm_name(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower()
    s = s.str.replace(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", regex=True)
    s = s.str.replace(r"[^a-z ]", "", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def load_schedule(seasons=SEASONS) -> pd.DataFrame:
    g = pd.read_csv(RAW / "games.csv")
    g = g[(g.game_type == "REG") & g.season.isin(seasons)]
    home = g[["season", "week", "game_id", "home_team"]].rename(columns={"home_team": "team"})
    away = g[["season", "week", "game_id", "away_team"]].rename(columns={"away_team": "team"})
    tg = pd.concat([home, away]).sort_values(["season", "team", "week"]).reset_index(drop=True)
    tg["game_idx"] = tg.groupby(["season", "team"]).cumcount()
    tg["n_team_games"] = tg.groupby(["season", "team"])["week"].transform("size")
    return tg


def load_rosters(seasons=SEASONS) -> pd.DataFrame:
    frames = []
    for s in seasons:
        r = pd.read_parquet(RAW / f"roster_weekly_{s}.parquet",
                            columns=["season", "week", "team", "position", "full_name", "gsis_id",
                                     "pfr_id", "status", "status_description_abbr", "game_type"])
        frames.append(r[(r.game_type == "REG") & r.position.isin(POSITIONS)])
    r = pd.concat(frames, ignore_index=True)
    r = r.dropna(subset=["gsis_id"])
    # A player can appear twice in a team-week (e.g. status change); keep the last row.
    r = r.drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")
    r["name_key"] = _norm_name(r.full_name)
    r["on_ir"] = (r.status == "RES") & r.status_description_abbr.isin(IR_CODES)
    return r


def load_stats(seasons=SEASONS) -> pd.DataFrame:
    frames = []
    for s in seasons:
        st = pd.read_parquet(RAW / f"stats_player_week_{s}.parquet",
                             columns=["player_id", "season", "week", "season_type", "team", "position",
                                      "passing_yards", "receiving_yards", "attempts", "targets", "receptions"])
        frames.append(st[st.season_type == "REG"])
    st = pd.concat(frames, ignore_index=True).rename(columns={"player_id": "gsis_id"})
    return st.drop_duplicates(["season", "week", "team", "gsis_id"])


def load_snaps(seasons=SEASONS) -> pd.DataFrame:
    frames = []
    for s in seasons:
        sc = pd.read_parquet(RAW / f"snap_counts_{s}.parquet",
                             columns=["season", "game_type", "week", "player", "pfr_player_id", "team",
                                      "offense_snaps", "offense_pct"])
        frames.append(sc[sc.game_type == "REG"])
    sc = pd.concat(frames, ignore_index=True).rename(columns={"pfr_player_id": "pfr_id"})
    sc["name_key"] = _norm_name(sc.player)
    return sc.drop_duplicates(["season", "week", "team", "pfr_id"])


def load_injuries(seasons=SEASONS) -> pd.DataFrame:
    frames = []
    for s in seasons:
        inj = pd.read_parquet(RAW / f"injuries_{s}.parquet",
                              columns=["season", "game_type", "team", "week", "gsis_id", "report_status",
                                       "practice_status", "report_primary_injury"])
        frames.append(inj[inj.game_type == "REG"])
    inj = pd.concat(frames, ignore_index=True).dropna(subset=["gsis_id"])
    # keep the most severe designation if duplicated
    sev = {"Out": 3, "Doubtful": 2, "Questionable": 1}
    inj["sev"] = inj.report_status.map(sev).fillna(0)
    inj = inj.sort_values("sev").drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")
    return inj.drop(columns="sev")


def build_panel(seasons=SEASONS) -> pd.DataFrame:
    tg = load_schedule(seasons)
    ro = load_rosters(seasons)
    st = load_stats(seasons)
    sc = load_snaps(seasons)
    inj = load_injuries(seasons)

    # roster rows only for weeks in which the team actually plays (drops bye weeks)
    p = ro.merge(tg, on=["season", "week", "team"], how="inner")

    # stats (yards by position)
    p = p.merge(st[["season", "week", "team", "gsis_id", "passing_yards", "receiving_yards",
                    "attempts", "targets", "receptions"]],
                on=["season", "week", "team", "gsis_id"], how="left")
    p["yards"] = np.where(p.position == "QB", p.passing_yards, p.receiving_yards)

    # snap counts: join on pfr_id first, then fall back to normalized name within team-week
    s1 = sc[["season", "week", "team", "pfr_id", "offense_snaps", "offense_pct"]].dropna(subset=["pfr_id"])
    p = p.merge(s1, on=["season", "week", "team", "pfr_id"], how="left")
    s2 = sc[["season", "week", "team", "name_key", "offense_snaps", "offense_pct"]].drop_duplicates(
        ["season", "week", "team", "name_key"]).rename(
        columns={"offense_snaps": "snaps_nm", "offense_pct": "pct_nm"})
    p = p.merge(s2, on=["season", "week", "team", "name_key"], how="left")
    p["offense_snaps"] = p.offense_snaps.fillna(p.snaps_nm)
    p["offense_pct"] = p.offense_pct.fillna(p.pct_nm)
    p = p.drop(columns=["snaps_nm", "pct_nm"])

    # injury report
    p = p.merge(inj[["season", "week", "team", "gsis_id", "report_status", "practice_status",
                     "report_primary_injury"]],
                on=["season", "week", "team", "gsis_id"], how="left")

    has_stat = p[["attempts", "targets", "receptions"]].fillna(0).sum(axis=1) > 0
    p["played"] = (p.offense_snaps.fillna(0) > 0) | has_stat
    p["yards"] = p.yards.fillna(0).astype(float)
    p["has_snap_record"] = p.offense_snaps.notna()
    p = p.sort_values(["season", "gsis_id", "game_idx"]).reset_index(drop=True)
    return p[["season", "week", "team", "gsis_id", "full_name", "position", "game_idx", "n_team_games",
              "played", "has_snap_record", "offense_snaps", "offense_pct", "yards", "attempts", "targets",
              "report_status", "practice_status", "report_primary_injury", "status", "on_ir"]]


if __name__ == "__main__":
    panel = build_panel()
    out = Path(__file__).resolve().parent.parent / "data" / "panel.parquet"
    panel.to_parquet(out)
    print(panel.shape, "->", out)
