"""Reading config.yaml / games.csv / status.csv, writing history.csv and calibration.json."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from .model import (AvailabilityParams, PlayerSpec, PlayerState, PositionParams, SEASON_GAMES,
                    parse_status)

ROOT = Path(__file__).resolve().parent.parent
FLAGS = ("full", "injured", "dnp")


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    bet = cfg["bet"]
    cfg["payout_decimal"] = (bet["stake"] + bet["profit"]) / bet["stake"]
    return cfg


def config_hash(cfg: dict) -> str:
    """Hash of everything calibration depends on (players, parameters, sims, seed)."""
    keep = {k: cfg[k] for k in ("players", "positions", "simulation", "season_games") if k in cfg}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=float).encode()).hexdigest()[:16]


def position_params(cfg: dict) -> Tuple[Dict[str, PositionParams], Dict[str, AvailabilityParams]]:
    pos, av = {}, {}
    for name, p in cfg["positions"].items():
        pos[name] = PositionParams(cv_log_a=float(p["cv_log_a"]), cv_log_b=float(p["cv_log_b"]), tau=float(p["tau"]))
        av[name] = AvailabilityParams(hazard=float(p["hazard"]),
                                      p_questionable_plays=float(p["p_questionable_plays"]),
                                      absence_pmf=np.array(p["absence_pmf"], dtype=float),
                                      q_absence_pmf=np.array(p["q_absence_pmf"], dtype=float),
                                      ir_pmf=np.array(p["ir_pmf"], dtype=float))
    return pos, av


def player_specs(cfg: dict, calibration: dict | None = None) -> List[PlayerSpec]:
    specs = []
    for p in cfg["players"]:
        mu0 = None
        if calibration is not None:
            mu0 = calibration["players"][p["name"]]["mu0"]
        specs.append(PlayerSpec(name=p["name"], position=p["position"], line=float(p["line"]),
                                team=p.get("team", ""), mu0=mu0))
    return specs


def read_games(path: Path, player_names: List[str]) -> Dict[str, List[dict]]:
    """games.csv rows grouped by player, validated."""
    games: Dict[str, List[dict]] = {n: [] for n in player_names}
    if not Path(path).exists():
        return games
    seen = set()
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            name = row["player"].strip()
            if name not in games:
                raise ValueError(f"games.csv line {i}: unknown player {name!r}")
            flag = row["flag"].strip().lower()
            if flag not in FLAGS:
                raise ValueError(f"games.csv line {i}: flag must be one of {FLAGS}, got {flag!r}")
            week = int(row["week"])
            yards = float(row["yards"]) if str(row["yards"]).strip() != "" else 0.0
            if flag == "dnp" and yards != 0:
                raise ValueError(f"games.csv line {i}: dnp games must have 0 yards")
            if (name, week) in seen:
                raise ValueError(f"games.csv line {i}: duplicate week {week} for {name}")
            seen.add((name, week))
            games[name].append({"week": week, "yards": yards, "flag": flag})
    for n in games:
        games[n].sort(key=lambda r: r["week"])
        if len(games[n]) > SEASON_GAMES:
            raise ValueError(f"{n} has more than {SEASON_GAMES} games entered")
    return games


def read_status(path: Path, player_names: List[str]) -> Dict[str, str]:
    status = {n: "healthy" for n in player_names}
    if not Path(path).exists():
        return status
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            name = row["player"].strip()
            if name not in status:
                raise ValueError(f"status.csv line {i}: unknown player {name!r}")
            s = row["status"].strip().lower()
            parse_status(s)  # validates
            status[name] = s
    return status


def build_states(games: Dict[str, List[dict]], status: Dict[str, str]) -> Dict[str, PlayerState]:
    """Games used = every row (full, injured, dnp).  Only `full` games update the rate.
    Yards from `full` and `injured` games count toward the season total."""
    states = {}
    for name, rows in games.items():
        st = PlayerState(status=status[name])
        for r in rows:
            st.games_used += 1
            st.yards_so_far += r["yards"]
            if r["flag"] == "full":
                st.full_game_yards.append(r["yards"])
        states[name] = st
    return states


def load_calibration(path: Path = ROOT / "calibration.json") -> dict | None:
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_calibration(cal: dict, path: Path = ROOT / "calibration.json") -> None:
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)


def append_history(result: dict, week: int, path: Path = ROOT / "history.csv") -> None:
    new = not Path(path).exists()
    names = [r["player"] for r in result["players"]]
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_time_utc", "week"] + [f"p_{n}" for n in names] +
                       ["p_all4", "p_product", "unit_ev", "ev_example_stake", "n_sims"])
        w.writerow([datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), week] +
                   [f"{r['p_hit']:.5f}" for r in result["players"]] +
                   [f"{result['p_all']:.5f}", f"{result['p_product']:.5f}", f"{result['unit_ev']:.5f}",
                    f"{result['example_ev']:.2f}", result["n_sims"]])
