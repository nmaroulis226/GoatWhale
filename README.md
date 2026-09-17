# GOAT WHALE model

Prices the 4-leg season-long yardage parlay (Kyler Murray 3049.5 passing, Christian Watson
799.5 receiving, DeVonta Smith 1099.5 receiving, Juwan Johnson 599.5 receiving; $1,750 wins
$23,245.79) and updates it week by week. Bayesian grid posterior for each player's healthy
yards-per-game rate + Monte Carlo simulation of the rest of the season including injuries.
All modeling choices and data sources: `ASSUMPTIONS.md`. Test results: `TEST_RESULTS.md`.

## Setup

```
pip install -r requirements.txt      # numpy, scipy, pandas, pyyaml (+ pyarrow/lxml/pytest for estimation & tests)
```

Python 3.10+. The model itself only needs numpy, scipy and pyyaml.

## Weekly update (the normal workflow)

1. **Add one row per player to `games.csv`** for the week just played:

   ```
   week,player,yards,flag
   2,Kyler Murray,245,full
   2,Christian Watson,0,dnp
   2,DeVonta Smith,88,full
   2,Juwan Johnson,31,injured
   ```

   - `full` — a normal game. Updates the rate posterior and counts the yards.
   - `injured` — he got hurt during the game. Yards count, game is used up, rate is **not** updated.
   - `dnp` — he did not play (yards must be 0). Uses up a game, no rate update.
   - Do not enter bye weeks. Games left = 17 − rows entered for that player.

2. **Set each player's state for his next game in `status.csv`:**

   | status | meaning |
   |---|---|
   | `healthy` | available |
   | `questionable` | listed Questionable for the next game |
   | `out:N` | ruled out for exactly N games (e.g. `out:2`) |
   | `ir` | placed on IR (misses at least 4 more games) |
   | `season_over` | done for the year |

   Use `out:1` for a Doubtful player.

3. **Run the update:**

   ```
   python goat_whale.py
   ```

   Prints the table (yards so far, line, games left, status, posterior μ median with 80%
   interval, projected final total with 10th–90th percentiles, P(hit)), then P(all 4 hit) with
   the product-of-legs check, unit EV and the EV of a $250 bet, and appends a row to
   `history.csv`. Options: `--no-history` (don't append), `--sims 200000` (more simulated
   seasons; default 100,000 from `config.yaml`).

## Calibration (already done; run once)

`calibration.json` holds the kickoff prior medians μ₀ that make every leg exactly 50% with 17
games left and status healthy. It is never re-solved during the season. If you change
`config.yaml` (parameters, seed, sims) you will see a warning; re-solve only if you mean to:

```
python goat_whale.py calibrate --force
```

## Files

| File | Purpose |
|---|---|
| `goat_whale.py` | entry point (`update` default, `calibrate`) |
| `goat_whale/model.py` | Gamma likelihood, grid posterior, availability walk, simulation, calibration |
| `goat_whale/io.py`, `report.py`, `cli.py` | input files, output table, history, command line |
| `config.yaml` | players, lines, bet, sims/seed, every model parameter with its source |
| `games.csv`, `status.csv` | your weekly inputs (ship with Week 1 entered) |
| `calibration.json` | stored μ₀ per player |
| `history.csv` | one row per run |
| `ASSUMPTIONS.md` | all assumptions, data sources, placeholders, omissions |
| `TEST_RESULTS.md` | test summary with the kickoff and Week 1 scenario tables |
| `tests/` | automated tests: `python -m pytest -s -q tests/` |
| `estimation/` | offline scripts that built the parameters from nflverse (`panel.py` → `estimate_params.py` → `estimate_tau.py` → `write_config.py`) |
| `estimates/` | their outputs (`estimates.json`, `tau.json`, `roster_check.json`) |
| `data/raw/` | cached nflverse release files and archived FantasyPros projection pages |

## Re-estimating parameters (optional)

```
python estimation/panel.py                 # rebuild the player-game panel from data/raw/
python estimation/estimate_params.py       # CV curves, hazard, absence distributions, p_q, IR
python -m estimation.estimate_tau          # tau from archived preseason projections
python -m estimation.write_config          # regenerate config.yaml (then recalibrate)
```

`config.yaml` is plain YAML and can be hand-edited instead (e.g. to try the prompt's placeholder
τ values); recalibrate afterwards.

## Web page for the group

A shareable what-if page lives at https://claude.ai/artifact/N3matY1z179F65JnBVkzSj (source in
`web/`: `model.js` is a JavaScript port of `goat_whale/model.py`, validated against the Python
results by `web/validate.js`; `page.template.html` is the page). Friends enter yards and statuses
and see the odds instantly; their edits stay in their own browser and can be shared as a link.
Only the page owner sees "Save as official", which stores the week inside the page for everyone.
The Python model remains the source of truth for `history.csv`.
