"""Confirm the four players' 2026 teams from nflverse weekly rosters and check none are teammates;
also list the head-to-head games between their teams from the 2026 schedule."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["Kyler Murray", "Christian Watson", "DeVonta Smith", "Juwan Johnson"]

r = pd.read_parquet(ROOT / "data/raw/roster_weekly_2026.parquet")
r = r[r.full_name.isin(NAMES)].sort_values(["full_name", "week"])
latest = r.groupby("full_name").tail(1)[["full_name", "team", "position", "gsis_id", "week", "status"]]
teams = dict(zip(latest.full_name, latest.team))
g = pd.read_csv(ROOT / "data/raw/games.csv")
g = g[(g.season == 2026) & (g.game_type == "REG")]
tv = set(teams.values())
h2h = g[g.home_team.isin(tv) & g.away_team.isin(tv)][["week", "away_team", "home_team"]]
out = {"teams": teams, "positions": dict(zip(latest.full_name, latest.position)),
       "all_different_teams": len(tv) == len(teams),
       "head_to_head_games_2026": h2h.to_dict("records"),
       "source": "nflverse weekly_rosters roster_weekly_2026.parquet and schedules games.csv"}
json.dump(out, open(ROOT / "estimates" / "roster_check.json", "w"), indent=2)
print(json.dumps(out, indent=1))
