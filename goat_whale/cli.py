"""
GOAT WHALE bet tracker (command-line logic; `python goat_whale.py` calls main()).

  python goat_whale.py               # run an update from games.csv / status.csv, append history.csv
  python goat_whale.py calibrate     # solve the kickoff prior medians (mu0) once, write calibration.json
  python goat_whale.py --no-history  # run without appending to history.csv
  python goat_whale.py --sims 200000 # override the number of simulated seasons
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from . import io as gio
from .model import Draws, calibrate_mu0, run_model
from .report import format_result

ROOT = Path(__file__).resolve().parent.parent


def do_calibrate(cfg: dict, force: bool = False, quiet: bool = False) -> dict:
    games = gio.read_games(ROOT / "games.csv", [p["name"] for p in cfg["players"]])
    if any(games.values()) and not force:
        sys.exit("games.csv already has games entered; calibration is meant to run once before the season.\n"
                 "Use `python goat_whale.py calibrate --force` if you really want to re-solve mu0.")
    pos, av = gio.position_params(cfg)
    specs = gio.player_specs(cfg)
    sim = cfg["simulation"]
    draws = Draws(int(sim["n_sims"]), len(specs), int(sim["seed"]))
    cal = {"date": str(date.today()), "config_hash": gio.config_hash(cfg), "n_sims": int(sim["n_sims"]),
           "seed": int(sim["seed"]), "players": {}}
    if not quiet:
        print("Calibrating kickoff priors so that P(total > line) = 50% with 17 games left and status healthy:")
    for i, s in enumerate(specs):
        r = calibrate_mu0(s, pos[s.position], av[s.position], draws, i, n_grid=int(sim.get("n_grid", 300)))
        cal["players"][s.name] = r
        if not quiet:
            print(f"  {s.name:18s} {s.position}  line {s.line:7.1f}  mu0 = {r['mu0']:7.2f} ypg  "
                  f"(line/17 = {s.line / 17:6.2f})  E[games played] = {r['expected_games']:.2f}  "
                  f"P(hit) at mu0 = {100 * r['p_hit']:.2f}%")
    gio.save_calibration(cal, ROOT / "calibration.json")
    if not quiet:
        print(f"Saved calibration.json (config hash {cal['config_hash']}).")
    return cal


def do_update(cfg: dict, n_sims: int | None = None, write_history: bool = True) -> dict:
    cal = gio.load_calibration(ROOT / "calibration.json")
    names = [p["name"] for p in cfg["players"]]
    games = gio.read_games(ROOT / "games.csv", names)
    if cal is None:
        if any(games.values()):
            sys.exit("calibration.json is missing and games.csv already has rows. Run "
                     "`python goat_whale.py calibrate --force` (re-solves mu0 at kickoff settings).")
        cal = do_calibrate(cfg)
    elif cal.get("config_hash") != gio.config_hash(cfg):
        print("WARNING: config.yaml changed since calibration.json was written (players, parameters, sims or seed). "
              "The stored mu0 values are still used; run `python goat_whale.py calibrate --force` to re-solve.",
              file=sys.stderr)
    status = gio.read_status(ROOT / "status.csv", names)
    states = gio.build_states(games, status)
    pos, av = gio.position_params(cfg)
    specs = gio.player_specs(cfg, cal)
    sim = cfg["simulation"]
    result = run_model(specs, [states[s.name] for s in specs], pos, av,
                       n_sims=int(n_sims or sim["n_sims"]), seed=int(sim["seed"]),
                       payout_decimal=cfg["payout_decimal"], example_stake=float(cfg["bet"]["example_stake"]),
                       n_grid=int(sim.get("n_grid", 300)))
    week = max((r["week"] for rows in games.values() for r in rows), default=0)
    result["week"] = week
    if write_history:
        gio.append_history(result, week, ROOT / "history.csv")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="GOAT WHALE season-long parlay tracker")
    ap.add_argument("command", nargs="?", default="update", choices=["update", "calibrate"])
    ap.add_argument("--force", action="store_true", help="re-run calibration even if games are entered")
    ap.add_argument("--sims", type=int, default=None, help="override number of simulated seasons")
    ap.add_argument("--no-history", action="store_true", help="do not append to history.csv")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args(argv)
    cfg = gio.load_config(Path(args.config))
    if args.command == "calibrate":
        do_calibrate(cfg, force=args.force)
        return
    result = do_update(cfg, n_sims=args.sims, write_history=not args.no_history)
    print(f"GOAT WHALE update — through week {result['week']} — {date.today()}\n")
    print(format_result(result))
    if not args.no_history:
        print("\nAppended to history.csv")


if __name__ == "__main__":
    main()
