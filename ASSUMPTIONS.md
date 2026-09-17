# ASSUMPTIONS.md — GOAT WHALE model

Everything the model assumes, every outside number and where it came from, and every
placeholder or judgment call. Data pull date for all nflverse files and Wayback captures:
**2026-09-16** (local time; see `estimates/estimates.json` and `estimates/tau.json`, which
the estimation scripts regenerate).

Sections: 1 Bet · 2 Inputs & counting · 3 Per-game distribution · 4 Rate uncertainty (prior, τ)
· 5 Availability (injuries) · 6 Injured-game handling · 7 Calibration · 8 Simulation & independence
· 9 Data pipeline · 10 Placeholders & judgment calls · 11 Considered but not included
· 12 Index of outside-data numbers

---

## 1. The bet

| Item | Value | Source |
|---|---|---|
| Stake / profit on the reference ticket | $1,750 / $23,245.79 | bet slip (prompt) |
| Decimal payout | (1750 + 23245.79) / 1750 = 14.283309 | arithmetic |
| Break-even P(all 4 hit) | 1 / 14.283309 = 7.00% | arithmetic |
| Unit EV | P(all 4) × 14.283309 − 1 | definition |
| Example bet | $250 → EV = 250 × unit EV | prompt |

**Lines** (stored as half-points; a leg hits if season total **> line**):
Kyler Murray passing 3049.5 · Christian Watson receiving 799.5 · DeVonta Smith receiving 1099.5
· Juwan Johnson receiving 599.5.

- **Assumption A1.** Watson's line is confirmed as 799.5. The other three are *assumed* to follow the
  same half-point convention (Kyler hits at 3050+, Smith at 1100+, Johnson at 600+). No pushes.
- **Assumption A2.** The preseason lines were 50/50 with no vig, so the kickoff prior for each leg is
  calibrated to exactly 50% (§7).
- **Assumption A3.** "Season" = the 17-game 2026 regular season. Playoff games do not count.

## 2. Inputs and counting conventions

- `games.csv` has one row per **team game** for each player: `flag=full` (normal game),
  `injured` (hurt during the game), or `dnp` (did not play, yards 0). Bye weeks are not entered.
- **Games left = 17 − number of rows entered for that player** (full + injured + dnp). This is how
  "games the team has already played" is counted; it relies on you entering a `dnp` row for every
  game the player missed.
- Yards from `full` and `injured` rows count toward the season total. Only `full` rows update the
  rate posterior. `dnp` rows use up a game but carry no information about the rate.
- `status.csv` is the availability state going into the player's **next** game.
- **Placeholder P1.** `status.csv` ships with Kyler Murray = `questionable`. His real Week 2
  designation was not in the nflverse 2026 injury file at pull time (he left the Week 1 game
  after 11 snaps). Set it to whatever the actual report says before running.
- Week 1 numbers in `games.csv` (Kyler 18 passing yards, Watson 147, Smith 53, Johnson 54) were
  verified against nflverse `stats_player_week_2026.parquet` (pulled 2026-09-16). Kyler's snap share
  in that game was 11 of 65 (17%), consistent with the `injured` flag.
- `history.csv` gets one appended row per run (`week` = the largest week in `games.csv`).

## 3. Per-game yardage distribution

**Model.** Given the true healthy rate μ, yards in a full healthy game ~ Gamma(shape α = 1/c², scale μ/α),
with c = c(μ) depending on the rate and position: **log c = a + b·log μ**.

**Likelihood of an entered game** is interval-censored:
L(y | μ) = F(y + 0.5) − F(y − 0.5), and any y ≤ 0 uses L = F(0.5). If the interval probability
underflows to 0 in floating point, the code falls back to the log Gamma density at max(y, 0.5) so
the posterior stays finite (never triggered at realistic inputs; `is_valid()` is checked after
every update).

- **Assumption A4.** Negative yards are never simulated (Gamma is non-negative). A negative
  actual game is entered as-is: it counts toward the total, and it updates the rate as a 0-yard
  observation (L = F(0.5)).
- **Assumption A5.** The season sum for fixed μ uses the additive property: k full games plus
  partial fractions u_j sum to Gamma(shape α·(k + Σu_j), same scale). One Gamma draw per player
  per simulated season, evaluated with the inverse CDF so the same uniforms can be reused.
- **Assumption A6.** c(μ) is clipped to [0.05, 2.5] (never binding on the grids used here).

### CV curve estimates (nflverse 2018–2025, `stats_player_week` + `snap_counts`)

Definitions used for the fit:
- **Full healthy game** = the player played and his offensive snap share was ≥ 0.70 × his
  player-season median snap share (drops games he left hurt, got benched in, or rested).
  This threshold is a judgment call (J1); 5–9% of starters' games fall below it, mostly
  in-game injuries and blowout pulls.
- **Population:** player-seasons with ≥ 8 full healthy games, and
  - QB "primary starter" = player-season median snap share ≥ 0.75;
  - WR: full-game mean receiving yards between 35 and 90 ypg;
  - TE: full-game mean receiving yards between 25 and 65 ypg.
- Per player-season: μ̂ = mean of full-game yards, ĉ = sample SD (ddof = 1) / μ̂.
- Fit: weighted least squares of log ĉ on log μ̂, weights = number of games. (Judgment J2: no
  errors-in-variables correction for noise in μ̂; with n ≥ 8 the attenuation in b is small.)

| Position | a | b | player-seasons | games | residual SD of log c | implied c at … |
|---|---|---|---|---|---|---|
| QB (passing) | 1.737827 | −0.554330 | 248 | 3,395 | 0.207 | 150: 0.354 · 220: 0.286 · 260: 0.261 |
| WR (receiving) | 1.419412 | −0.480949 | 532 | 7,119 | 0.218 | 40: 0.701 · 55: 0.602 · 70: 0.536 · 90: 0.475 |
| TE (receiving) | 1.329356 | −0.479155 | 223 | 2,994 | 0.214 | 25: 0.808 · 40: 0.645 · 55: 0.554 |

At the calibrated rates the implied CVs are QB 0.30 (μ₀ ≈ 201), Watson 0.60, Smith 0.52,
Johnson 0.64 — close to the prompt's placeholders (0.30 / 0.62 / 0.68), which are therefore
**not** used.

### Distribution comparison (done; Gamma kept)

Fitted the CV curve on 2018–2023 and scored held-out 2024–2025 full games under Gamma vs
lognormal with matched mean and CV (both interval-censored, using each player-season's own
mean, which is equally favourable to both):

| Position | held-out games | log-lik Gamma | log-lik lognormal | Gamma advantage per game |
|---|---|---|---|---|
| QB | 856 | −4787.1 | −4819.8 | +0.038 |
| WR | 1,760 | −8798.3 | −10292.4 | +0.849 |
| TE | 875 | −4086.4 | −4795.7 | +0.811 |

Gamma wins at every position (the lognormal badly under-predicts 0–10 yard WR/TE games), so no
switch. Zero-inflated Gamma and targets × yards-per-target were **not** tested (J3): the
interval-censored Gamma already puts explicit mass on 0-yard games, and the targets model would
need targets entered every week.

## 4. Uncertainty about the true rate: grid prior and τ

- Belief about μ lives on a **300-point log-spaced grid** from μ₀·e^(−4τ) to μ₀·e^(4τ).
- **Prior:** lognormal, median μ₀ (calibrated, §7), log-SD τ by position. Because the grid is
  uniform in log μ, prior weights are Gaussian in log μ.
- **Update:** each `full` game multiplies grid weights by L(y | μ_grid) with c(μ_grid), then
  renormalises (done in log space). No fixed weighting schedule.
- **Assumption A7 (grid truncation).** The posterior cannot move outside ±4τ of μ₀. For the QB
  (τ = 0.083) the grid spans 145–280 ypg; a run of terrible games pins the posterior at the
  lower edge rather than below it. This matters for Test 3.1 (see `TEST_RESULTS.md`) and would
  matter if a player's true rate collapsed (e.g. a QB losing his job) — a known omission (§11).
- Random-walk drift in μ is **not** included; `GridPosterior.drift()` is a no-op hook.

### τ estimates (substitute source: archived FantasyPros preseason consensus projections)

Historical preseason season-long *lines* could not be found in any downloadable source, so —
as the prompt allows — τ was estimated from **archived preseason FantasyPros consensus
projections** (season-total yards) captured by the Wayback Machine. Method:

1. r_i = log(actual full-game ypg_i) − log(projected season yards_i), for player-seasons with
   ≥ 8 full healthy games. (A constant games-played factor drops out of the variance.)
2. Population selected on the **projection**, not the outcome: QB projected ≥ 3000 passing
   yards; WR projected 525–1350 receiving yards (≈ 35–90 ypg × 15); TE 375–975 (≈ 25–65 × 15).
3. r demeaned within (position, season) to remove the projection source's bias; the
   non-demeaned value is reported too.
4. τ² = Var(r) − mean over players of c(μ̂_i)² / n_i (game-to-game sampling noise).

| Position | τ used | τ without season demeaning | Var(r) | mean noise | player-seasons | seasons with a valid preseason capture |
|---|---|---|---|---|---|---|
| QB | **0.0826** | 0.0980 | 0.0123 | 0.0055 | 201 | 2018–2025 (8) |
| WR | **0.2869** | 0.2930 | 0.1183 | 0.0359 | 250 | 2019, 2021, 2022, 2023 (4) |
| TE | **0.3051** | 0.3249 | 0.1347 | 0.0416 | 166 | 2019–2025 (7) |

Wayback captures used (all "YYYY {POS} Projections" season-total pages, captured 1 May–10 Sep of
the season; captures dated mid-season were discarded because the "draft" page is *not* frozen —
July and October 2019 WR values matched on only 4% of players):
QB 20180901, 20190902, 20200818, 20210817, 20220902, 20230906, 20240905, 20250902 ·
WR 20190719, 20210728, 20220825, 20230906 · TE 20190717, 20200811, 20210512, 20220529,
20230906, 20240519, 20250822. No preseason WR capture exists for 2018, 2020, 2024, 2025 (the
bare `wr.php` July 2025 capture is a Week 1 page and was rejected). Raw HTML is in
`data/raw/fantasypros/`.

- **Caveat C1.** Consensus fantasy projections are noisier than a sharp closing line, so these τ
  values probably **overstate** the market's error, especially for WR/TE (0.29–0.31 vs the
  prompt's 0.20–0.22 placeholders). Practical effect: the prior is looser, so each entered game
  moves μ more (Watson's 147-yard game lifts his median from 55 to 71 ypg). To use the
  placeholders instead, edit `tau:` in `config.yaml` and re-run `python goat_whale.py calibrate --force`.
- **Caveat C2.** Some projections build in known suspensions/holdouts; that inflates Var(r) a
  little (in the direction of C1).

## 5. Availability model (injuries)

States going into the next game and how they are simulated:

| State | Simulation |
|---|---|
| `healthy` | Plays. Each game an injury occurs with probability h (by position). |
| `questionable` | Plays the next game with probability p_q (then treated as a full game, hazard applies). Otherwise misses that game plus the rest of an absence whose total length L is drawn from the Questionable-initiated distribution (L counted from the missed game). |
| `out:N` | Misses exactly N games, then `healthy`. |
| `ir` | Misses L games, L drawn from the IR-stint distribution conditional on L ≥ 4, counted from now, then `healthy` if games remain. |
| `season_over` | 0 more yards. |

**Simulated in-game injury:** the player gets a fraction u ~ Uniform(0,1) of that game (yards
Gamma(u·α, scale)), then misses L games from the position's absence-length distribution. A
"season-ending" draw means all remaining games. After any absence he returns `healthy` and the
hazard applies again from the next game.

- **Assumption A8.** The hazard is applied per game *played* (in-game and practice-week injuries
  are pooled; the model treats every one as happening during the game with a random fraction).
  Historically 39% of hazard events showed a low snap share in the injury game (mean 0.38–0.41
  of the player's normal share, consistent with u ~ Uniform).
- **Assumption A9.** A Questionable player who plays gets a full game (no reduced snap share).
- **Assumption A10.** Doubtful is not a separate state; historically only 1.7% of Doubtful
  regulars played (n = 173). Enter `out:1` for a Doubtful player.
- **Assumption A11.** COVID-19 reserve list stints and non-injury absences (suspension, rest,
  personal, benching, healthy scratches) are not injuries and are excluded from all estimates.

### Estimates (nflverse 2018–2025 regular season; `injuries`, `weekly_rosters`, `snap_counts`, `schedules`)

Definitions:
- **Regular** player-season: median offensive snap share over games played ≥ 0.50 (QB ≥ 0.75).
  (J4: this slightly under-counts players hurt in their first game or two; ~2% effect on h.)
- **Injury-caused missed game**: did not play (0 offensive snaps and no passing/receiving stat)
  and either listed Out / Doubtful / Questionable on that week's injury report or on an
  injury reserve list (roster codes R01 Reserve/Injured, R48 IR-designated-for-return,
  R04 PUP, R05 NFI).
- **Hazard event**: a played game followed by an injury-caused missed game at the team's next
  game. **Exposure**: played games of regulars with a later team game that season.
- **Absence length**: consecutive missed team games from the first missed game until the
  next game played; **right-censored** at the season's last team game. Estimated by
  **Kaplan-Meier**; P(L = k) for k = 1…17 with the surviving mass placed in a "season-ending"
  cell. Once fewer than 5 absences remain at risk the curve is frozen and the remainder is
  season-ending (J5; otherwise 1–2 observations dominate the tail).

| Position | h (per game) | events / exposure games | p_q = P(plays \| Questionable) | n Questionable weeks |
|---|---|---|---|---|
| QB | **0.04576** | 184 / 4,021 | **0.4905** | 210 |
| WR | **0.05188** | 560 / 10,795 | **0.7465** | 1,006 |
| TE | **0.04407** | 203 / 4,606 | **0.7632** | 321 |
| pooled | 0.04876 | 947 / 19,422 | 0.7150 | 1,537 |

The prompt's placeholders (h = 0.045, p_q = 0.80) are therefore **not** used. Note the QB
result: quarterbacks tagged Questionable play only about half the time.

Absence-length distributions used (Kaplan-Meier pmf, first six cells and season-ending mass):

| Distribution | n (censored) | P(1) | P(2) | P(3) | P(4) | P(5) | P(6) | season-ending | mean (capped at 17) |
|---|---|---|---|---|---|---|---|---|---|
| Injury absence, QB | 184 (74) | .245 | .190 | .105 | .065 | .059 | .023 | .229 | 6.25 |
| Injury absence, WR | 560 (154) | .361 | .212 | .114 | .064 | .044 | .029 | .139 | 4.43 |
| Injury absence, TE | 203 (61) | .335 | .163 | .119 | .034 | .077 | .059 | .145 | 4.93 |
| Questionable-initiated, pooled QB/WR/TE | 262 (54) | .550 | .197 | .108 | .014 | .017 | .025 | .073 | 2.92 |
| IR stint, pooled, 2022–2025, given L ≥ 4 | 140 (89) | 0 | 0 | 0 | .127 | .132 | .065 | .323 | 10.4 |

Full vectors are in `config.yaml` (`absence_pmf`, `q_absence_pmf`, `ir_pmf`). Raw uncensored
absence counts, pooled: 1 game 315, 2: 159, 3: 83, 4: 37, 5: 31, 6: 17, 7: 5, 8: 6, 9: 4, 10: 1.

- **J6.** The general absence distribution is position-specific (n ≥ 184 each). The
  Questionable-initiated distribution is pooled across positions because the per-position
  samples are small (QB 53, TE 47); note QB Questionable absences run longer (mean 4.8 games
  vs 2.8 for WR/TE), so Kyler's `questionable` state is, if anything, optimistic.
- **J7.** IR stints use 2022–2025 only (the 4-game-minimum era; 2018–2019 had an 8-week
  minimum and 2020–2021 three games) and are pooled across QB/WR/TE. A stint is counted from
  the first week the roster shows an injury-reserve code. 89 of 140 stints were censored
  (never returned that season), so the conditional distribution has 32% season-ending mass.
- **J8.** QB absences include cases where an injured starter was later kept on the bench
  (the absence continues), so part of "benching after injury" risk is in the data even though
  benching per se is not modeled.

## 6. Handling a game where the player gets injured

Enter the row with `flag=injured` and the yards he actually gained. The yards count toward the
total, the game is used up, and **the rate posterior is not updated at all** (no snap counts
needed). Set his state for the next game in `status.csv`. Snap-share exposure weighting was
considered and rejected because it would require entering snap counts weekly.

## 7. Calibration: 50% at kickoff

For each player, bisection on μ₀ finds the prior median such that the simulated
P(season total > line) = 0.50 with 17 games left and status `healthy`, including rate
uncertainty (τ), game-to-game noise (CV curve) and injury risk (h and absence lengths).
Common random numbers: the availability walk and all uniforms are drawn once from the
config seed and reused at every bisection step; the same draws are used by updates, so the
kickoff table shows exactly 50.0% per leg. μ₀ is stored in `calibration.json` and is **never
re-solved after games are entered** (the CLI refuses unless `--force`).

| Player | line | line / 17 | calibrated μ₀ (healthy ypg) | E[games played] | prompt's rough expectation |
|---|---|---|---|---|---|
| Kyler Murray | 3049.5 | 179.4 | **200.8** | 14.43 | ~220–225 |
| Christian Watson | 799.5 | 47.0 | **55.1** | 14.76 | ~54 |
| DeVonta Smith | 1099.5 | 64.7 | **75.5** | 14.74 | ~72 |
| Juwan Johnson | 599.5 | 35.3 | **41.0** | 14.87 | ~39 |

**Sanity note (as requested).** μ₀ should be roughly consistent with public healthy per-game
projections. Watson, Smith and Johnson land on the prompt's expectations. Kyler's 200.8 is
below the 220–225 range: that range implies ~13.7 expected games, while the estimated QB
inputs (h = 0.046, absence mean 6.25 games) give 14.4. If μ₀ looks far off, the injury
inputs (hazard and absence lengths) are the likely cause, not μ₀ itself; the calibration
simply solves for the rate that makes the line a coin flip under those inputs.

## 8. Simulation and combining the legs

- ≥ 50,000 seasons required; the config uses **100,000** (calibration uses the same draws).
- Per simulated season and player: draw μ once from the grid posterior (inverse CDF), walk the
  remaining games from the current state, sum yards (one Gamma draw), add actual yards to
  date, hit if total > line.
- **Independence.** The four legs use independent random streams. Roster check from nflverse
  `roster_weekly_2026.parquet` (2026-09-16): Kyler Murray **MIN**, Christian Watson **GB**,
  DeVonta Smith **PHI**, Juwan Johnson **NO** — no two are teammates
  (`estimates/roster_check.json`).
- **Assumption A12.** Games between their teams create a small real correlation that is not
  modeled. In 2026 there are four: Week 1 GB @ MIN (already played), Week 5 MIN @ NO,
  Week 10 MIN @ GB, Week 13 GB @ NO. League-wide shocks (rule changes, weather) are also a
  common factor that is ignored.
- P(all 4 hit) is taken from the joint simulation; the product of the four individual
  probabilities is printed alongside as a check (they agree within simulation noise).

## 9. Data pipeline (what was actually run)

- **nflverse release files** (the same files `nfl_data_py` downloads), cached in `data/raw/`:
  `stats_player_week_{2018..2026}.parquet` (import_weekly_data), `snap_counts_*` (import_snap_counts),
  `injuries_*` (import_injuries), `roster_weekly_*` (import_weekly_rosters), `games.csv`
  (import_schedules). Regular season only. URL pattern:
  `https://github.com/nflverse/nflverse-data/releases/download/<release>/<file>`.
- **Environment note.** `nfl_data_py` could not be installed here (it pins pandas < 2.0 and the
  from-source build fails on this machine), so `estimation/panel.py` reads the release files
  directly with pandas/pyarrow. Nothing else differs.
- Snap counts are keyed by PFR id; they were joined to the gsis-keyed rosters by `pfr_id`, with
  a normalized-name fallback. Coverage of played games with a snap record: 99.9–100%.
- Scripts: `estimation/panel.py` → `data/panel.parquet`; `estimation/estimate_params.py` →
  `estimates/estimates.json`; `estimation/estimate_tau.py` → `estimates/tau.json`;
  `estimation/check_rosters.py` → `estimates/roster_check.json`;
  `python -m estimation.write_config` → `config.yaml`.

## 10. Placeholders and judgment calls (not estimated from data)

| Label | What | Value | Note |
|---|---|---|---|
| P1 | Kyler Murray's status in `status.csv` | `questionable` | Placeholder; set from the real Week 2 report. |
| J1 | "Full game" snap-share threshold | 0.70 × player-season median | Choice; 0.5–0.8 give similar CV fits. |
| J2 | CV regression method | WLS of log ĉ on log μ̂, weights n, no EIV correction | Choice. |
| J3 | Alternatives not tested | zero-inflated Gamma, targets × YPT | Deliberate; Gamma vs lognormal was tested. |
| J4 | "Regular" definition | median snap share ≥ 0.5 (QB ≥ 0.75) | Choice. |
| J5 | Kaplan-Meier tail rule | freeze when < 5 at risk; remainder = season-ending | Choice. |
| J6 | Q-initiated absence pooling | pooled across positions | Sample size. |
| J7 | IR seasons / pooling | 2022–2025, pooled | Rule-era choice. |
| J9 | Calibration bracket | μ₀ ∈ [0.7, 2.0] × line/17 | Search range only. |
| J10 | Simulation count / seed | 100,000 / 20260916 | Choice (≥ 50,000 required). |
| J11 | In-game injury fraction | u ~ Uniform(0, 1) | Prompt's specification; roughly consistent with data (mean observed share ≈ 0.4). |
| — | Prompt placeholders (CV 0.30/0.62/0.68; τ 0.11/0.20/0.22; h 0.045; p_q 0.80) | **not used** | All replaced by estimates above; kept in `estimation/write_config.py` as fallbacks. |

## 11. Considered but not included (known omissions)

- Player-specific injury hazards from injury history (all players use their position's h).
  Kyler Murray in particular has an extensive injury history that the position hazard ignores.
- Kyler Murray losing his starting job (benching risk), or any player's role changing.
- Stars resting in Week 18.
- A receiver's production dropping when his team's QB is injured.
- Reduced snap share when a Questionable player plays.
- Week-to-week drift (random walk) in μ; older games are never discounted (hook exists).
- Snap-share exposure weighting of partial games.
- Correlation between legs (head-to-head games, league-wide shocks).
- Lognormal (tested, worse), zero-inflated Gamma and targets × yards-per-target (not tested).
- Posterior mass outside ±4τ of the prior median (grid truncation, A7).
- Trades or roster moves during the season (a traded player keeps his position parameters).

## 12. Index of every outside-data number used

| Number | Value | Source | Seasons / filters | How computed | Pulled |
|---|---|---|---|---|---|
| CV curve a, b (QB) | 1.737827, −0.554330 | nflverse stats_player_week + snap_counts | 2018–2025 REG; regular QBs, ≥ 8 full games; 248 player-seasons | WLS log ĉ ~ log μ̂ | 2026-09-16 |
| CV curve a, b (WR) | 1.419412, −0.480949 | same | 35–90 ypg, ≥ 8 full games; 532 player-seasons | same | 2026-09-16 |
| CV curve a, b (TE) | 1.329356, −0.479155 | same | 25–65 ypg, ≥ 8 full games; 223 player-seasons | same | 2026-09-16 |
| τ (QB / WR / TE) | 0.0826 / 0.2869 / 0.3051 | Wayback captures of FantasyPros preseason consensus projections + nflverse actuals | QB 2018–25 (201 ps); WR 2019, 2021–23 (250); TE 2019–25 (166) | Var of demeaned log error minus sampling noise, square root | 2026-09-16 |
| Hazard h (QB / WR / TE) | 0.04576 / 0.05188 / 0.04407 | nflverse injuries, weekly_rosters, snap_counts, schedules | 2018–2025 REG, regulars | events / exposure games | 2026-09-16 |
| p_q (QB / WR / TE) | 0.4905 / 0.7465 / 0.7632 | nflverse injuries + snap_counts | 2018–2025 REG, regulars listed Questionable | played / listed | 2026-09-16 |
| Absence-length pmf (QB / WR / TE) | vectors in config.yaml | same | injury absences after a played game: 184 / 560 / 203 | Kaplan-Meier | 2026-09-16 |
| Questionable-initiated absence pmf | vector in config.yaml | same | 262 absences, pooled | Kaplan-Meier | 2026-09-16 |
| IR stint pmf (given ≥ 4) | vector in config.yaml | same | 2022–2025, 140 stints, pooled | Kaplan-Meier, truncated at 4, renormalised | 2026-09-16 |
| Teams of the four players | MIN, GB, PHI, NO | nflverse roster_weekly_2026 | 2026 weeks 1–2 | lookup | 2026-09-16 |
| Head-to-head games | weeks 1, 5, 10, 13 | nflverse schedules games.csv | 2026 REG | lookup | 2026-09-16 |
| Week 1 yards | 18 / 147 / 53 / 54 | prompt; verified vs nflverse stats_player_week_2026 | 2026 week 1 | lookup | 2026-09-16 |
| Calibrated μ₀ | 200.8 / 55.1 / 75.5 / 41.0 | this model | — | bisection, 100,000 sims, seed 20260916 | 2026-09-16 |
