# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Bayesian + Monte Carlo pricer for the **GOAT WHALE**: a 4-leg 2026 NFL season-long yardage parlay
(Kyler Murray/MIN passing 3049.5, Christian Watson/GB receiving 799.5, DeVonta Smith/PHI receiving
1099.5, Juwan Johnson/NO TE receiving 599.5; $1,750 wins $23,245.79, decimal payout 14.283309,
break-even P(all 4) = 7.00%). The original spec is `goat_whale_claude_code_prompt.md`; every
modeling choice and data source is in `ASSUMPTIONS.md`; test output is in `TEST_RESULTS.md`.
There are two implementations of the same model that must stay in agreement: the Python package
(source of truth) and a JavaScript port inside a published web page for the owner's friends.

## Commands

```bash
pip install -r requirements.txt

python goat_whale.py                       # weekly update from games.csv + status.csv; appends history.csv
python goat_whale.py --no-history          # same, without touching history.csv
python goat_whale.py --sims 200000         # override simulation count
python goat_whale.py calibrate --force     # re-solve kickoff priors (mu0) -> calibration.json (see invariant below)

python -m pytest -s -q tests/                                   # all tests; -s prints the kickoff and Week 1 tables
python -m pytest -q tests/test_goat_whale.py::test_2_kyler_status_scenarios   # one test
python -m pytest -s -q tests/ > out.txt    # TEST_RESULTS.md was built from this output

# offline parameter estimation (only when re-deriving parameters from nflverse data)
python estimation/panel.py                 # data/raw/*.parquet -> data/panel.parquet
python estimation/estimate_params.py       # CV curves, hazard, absence pmfs, p_q, IR -> estimates/estimates.json
python -m estimation.estimate_tau          # tau from archived FantasyPros projections -> estimates/tau.json
python estimation/check_rosters.py         # teams / head-to-head check -> estimates/roster_check.json
python -m estimation.write_config          # regenerate config.yaml from estimates (then recalibrate)

# web page
python web/build.py                        # config.yaml + calibration.json + web/official.json + template -> web/index.html
node web/validate.js web/params.json       # JS model vs Python numbers (should match TEST_RESULTS.md within ~0.5 pt)
```

## Architecture

**Data flow for an update** (`goat_whale/cli.py:do_update`):
`config.yaml` + `calibration.json` → `io.py` reads `games.csv`/`status.csv` into `PlayerState`s →
`model.run_model` builds a `GridPosterior` per player, applies every `full` game, walks the
remaining games with `simulate_availability`, draws season yards with `simulate_player`, and
combines legs → `report.py` prints the table → `io.append_history`.

**model.py is the whole model** (~360 lines) and is deliberately self-contained:
- Per-game yards ~ Gamma(mean μ, CV c(μ)) with `log c = a + b·log μ` per position; the likelihood
  of an entered game is interval-censored (`F(y+0.5) − F(y−0.5)`, `y ≤ 0 → F(0.5)`), with a
  log-density fallback when that underflows.
- Belief about μ is a 300-point log-spaced grid over μ₀·e^(±4τ) with a lognormal prior. Only
  `full` games update it. `drift()` is an intentional no-op hook for a future random walk.
- Availability states: `healthy | questionable | out:N | ir | season_over`. Absence lengths are
  empirical pmfs of length 18 (index 17 = season-ending, encoded as `SEASON_ENDING = 99`).
- All randomness comes from one `Draws` object (common random numbers). Calibration bisects on μ₀
  reusing the same draws, and season totals are drawn with `gammaincinv` on fixed uniforms so the
  bisection is monotone and scenario comparisons are stable. Keep this property if you touch the
  simulation.

**Input semantics** (in `io.build_states`): one `games.csv` row per team game; byes are not
entered; `games_left = 17 − rows`. `injured` rows count yards but do not update μ; `dnp` rows use
up a game with 0 yards and no update. `status.csv` is the state going into the next game.

**Estimation is separate from the model.** `estimation/` reads cached nflverse release files from
`data/raw/` (the exact files `nfl_data_py` would download; the package itself cannot be installed
here because it pins pandas<2.0). Its outputs in `estimates/*.json` are rendered into
`config.yaml` by `write_config.py`; `config.yaml` is then the single source of truth and is
hand-editable. `data/` and `estimates/` are inputs to that step, not to the model.

**Web page** (`web/`): `model.js` is a line-for-line port of `model.py` (same grid, same
availability walk; Gamma draws use Marsaglia-Tsang with a per-simulation seeded stream instead of
the inverse CDF). `page.template.html` has three placeholders (`__OFFICIAL_JSON__`,
`__PARAMS_JSON__`, `__MODEL_JS__`); `web/build.py` inlines `model.js`, a params JSON made from
`config.yaml` + `calibration.json`, and `web/official.json`, producing the single-file
`web/index.html` that is published as the claude.ai artifact
`https://claude.ai/artifact/N3matY1z179F65JnBVkzSj` (republish with the Artifact tool, passing
that URL). The
page declares only the `artifact` capability: official results live in
`<script id="official-data">` inside the page and "Save as official" republishes the page via
`artifact.publish` with new JSON. `db` was deliberately not used (it makes the artifact
organization-internal, which would lock out the friends). Friends' what-ifs are localStorage +
`#s=` hash links.

## Invariants and gotchas

- **Calibration runs once.** `calibration.json` holds μ₀ per player and a hash of the calibration
  inputs. Never re-solve after games are entered unless the user asks; the CLI refuses without
  `--force`. Re-running with `--force` after a cosmetic `config.yaml` change is fine (μ₀ comes out
  identical if parameters are unchanged) and refreshes the hash warning.
- **Quote `team: "NO"` in `config.yaml`.** Bare `NO` parses as YAML `false` (this bit us once).
- **Kickoff test (Test 1) expects exactly 50% per leg** only because calibration and updates share
  seed and sim count. Changing `simulation.seed`/`n_sims` without recalibrating shifts it by noise.
- **Test 3.1** checks "a 0-yard game is more informative for a QB" in prior-SD units and at equal
  τ, not as a raw ratio of medians: with the estimated τ (QB 0.083) the QB posterior is pinned at
  the ±4τ grid edge. See `TEST_RESULTS.md` and ASSUMPTIONS A7 before "fixing" it.
- **τ comes from archived FantasyPros preseason projections** (Wayback captures in
  `data/raw/fantasypros/`), not sportsbook lines; WR/TE τ (0.29/0.31) are above the prompt's
  placeholders and are flagged in ASSUMPTIONS C1. Mid-season captures of the "draft" page are not
  frozen and must not be used.
- **Republishing the page:** someone may have pressed "Save as official" since the last publish.
  A publish is then refused with the saved version's path; read it, copy its `official-data`
  JSON into `web/official.json`, rebuild with `web/build.py`, and publish again. Never overwrite
  a save, and keep `web/official.json` in sync with the live page.
- **Keep Python and JS in agreement.** After any change to `model.py`, mirror it in `web/model.js`
  and run `web/validate.js`; per-leg probabilities should match `TEST_RESULTS.md` within ~0.5 pt
  and posterior medians exactly. The Python model remains the source of truth for `history.csv`.
- `ASSUMPTIONS.md` must be updated whenever a parameter, filter, or data source changes; the
  spec requires it to list every outside number with its source and pull date.
