# Build the GOAT WHALE Model

## What you're building

A Python tool that prices a season-long NFL bet called the **GOAT WHALE** and updates it live as the season goes on. After each week, I enter each player's yards and injury status. The tool then outputs:

1. Each player's probability of going over his season yardage line
2. The probability of winning the whole bet (all 4 players must hit)
3. The EV per $1 staked (unit EV)
4. The EV of a $250 bet

It's a Bayesian model with Monte Carlo simulation. Each player's healthy yards-per-game rate starts from a prior calibrated to the sportsbook line, updates with every game I enter, and gets simulated forward through the rest of the season, including injury risk.

## The bet

- **Season:** 2026 NFL regular season (17 games per team)
- **All 4 legs must hit.** If any one player misses his line, the bet loses.
- **Payout:** a $1,750 bet wins $23,245.79 in profit.
  - Decimal payout = (1750 + 23245.79) / 1750 = **14.283309**
  - Unit EV = P(all 4 hit) × 14.283309 − 1
  - $250 bet EV = 250 × unit EV
  - Break-even P(all 4 hit) = 1 / 14.283309 ≈ 7.00%
- People bet different amounts, which is why the output is unit EV plus a $250 example.

| Player | Position | Stat | Line | Hits if season total is at least |
|---|---|---|---|---|
| Kyler Murray | QB | Passing yards | 3049.5 | 3050 |
| Christian Watson | WR | Receiving yards | 799.5 | 800 |
| DeVonta Smith | WR | Receiving yards | 1099.5 | 1100 |
| Juwan Johnson | TE | Receiving yards | 599.5 | 600 |

**The lines are half-point lines, so there are no pushes.** A player hits if his season total is at least the whole number shown (for example, Watson hits at exactly 800 yards). Watson's line is confirmed as 799.5. The other three are assumed to follow the same convention; list that as an assumption in ASSUMPTIONS.md. Store the half-point lines in the config and use "total > line" everywhere.

**The preseason lines were 50/50** (no vig to remove). At kickoff, before any games, the model must give each player exactly a 50% chance of going over.

## Documentation requirements (important)

Create an `ASSUMPTIONS.md` file and keep it up to date. It must list:

1. **Every assumption you make**, including modeling choices, simplifications, and how you interpreted anything ambiguous in this prompt.
2. **Every number you use that comes from an outside data source**, with:
   - the value
   - the source (dataset, function, URL)
   - the seasons or filters used
   - how it was calculated
   - the date you pulled it
3. **Every number that is a placeholder or judgment call** rather than estimated from data, clearly labeled as such.

Never make up a number and present it as if it came from data. If you can't get the data for something, use the placeholder values given below, label them as placeholders, and tell me.

---

## Model specification

### 1. Per-game yardage distribution: Gamma

Given a player's true healthy rate μ, yards in a full healthy game are distributed **Gamma with mean μ and coefficient of variation c**. The shape is α = 1/c², and the scale is μ/α.

- **Let the CV depend on volume and position.** Lower-volume players are proportionally more volatile. From historical game logs, fit log(c) as a linear function of log(μ), separately for QB passing yards, WR receiving yards, and TE receiving yards.
  - To estimate this, use player-seasons with at least 8 full healthy games.
  - Estimate μ and c from each player-season's healthy games.
  - It's fine to use historical snap counts here to filter out partial games in the historical data (see "Data").
- **Handling zero and negative yards.** The Gamma density can be 0 or infinite at 0 yards, which would break the Bayesian update. So in the likelihood, treat each observed game as interval-censored:
  - L(y | μ) = F(y + 0.5) − F(y − 0.5), where F is the Gamma CDF
  - Any observed y ≤ 0 uses L = F(0.5)
  - Negative yards aren't simulated. Document this as an assumption.
- Summing works cleanly: with a fixed μ, k full games sum to Gamma(shape k·α, same scale). Use this to draw remaining-season yards efficiently.

**Considered but not included:** lognormal, zero-inflated Gamma, and a targets × yards-per-target model. You may optionally compare Gamma against these by out-of-sample log-likelihood on held-out seasons, but only switch if the improvement is meaningful. Document what you did either way.

**Placeholders if data can't be obtained:**

| Position | Per-game CV |
|---|---|
| QB | 0.30 |
| WR | 0.62 (higher for lower-volume WRs) |
| TE | 0.68 |

### 2. Uncertainty about each player's true rate: grid posterior

Each player has an unknown true healthy yards per game, μ. Represent your belief about it as a probability distribution on a grid.

- **Prior:** lognormal, with median μ₀ (from calibration, section 5) and log-SD τ (by position).
- **Grid:** 300 points, log-spaced, from μ₀·e^(−4τ) to μ₀·e^(4τ).
- **Update:** for each full healthy game I enter, multiply each grid point's weight by L(y | μ) from section 1, then renormalize. The CV used at each grid point is c(μ) for that grid point's μ.
- No fixed weighting schedule. The posterior determines how much the prior versus the observed games matter.
- **Estimating τ:** τ is the typical error of the preseason market's rate estimate.
  - Ideally, use historical preseason season-long yardage lines: τ² ≈ Var(log(actual healthy ypg) − log(line-implied healthy ypg)) − (average sampling noise from game-to-game variance).
  - If historical lines are unavailable, use archived preseason consensus projections as a substitute, and document that.
  - If neither is available, use the placeholders below.

**Placeholder τ values:**

| Position | τ |
|---|---|
| QB | 0.11 |
| WR | 0.20 |
| TE | 0.22 |

**Considered but not included:** letting μ drift week to week (a random walk) to discount older games. Leave it out for now, but structure the code so it could be added later.

### 3. Availability model (injuries)

Each remaining game, each player is in one of these states:

| State | Meaning | Simulation |
|---|---|---|
| `healthy` | Available | Plays. A per-game injury hazard h (by position) applies. |
| `questionable` | Listed questionable for the next game | Plays that game with probability p_q. If he doesn't play, he starts an absence whose extra length is drawn from the empirical distribution for absences that began with a Questionable designation. |
| `out:N` | Ruled out for the next N games (I know N) | Misses exactly N games, then becomes `healthy`. |
| `ir` | Placed on IR | Misses at least 4 games. Total length is drawn from the empirical distribution of IR stints, conditional on being at least 4 games. |
| `season_over` | Done for the season | Adds 0 yards for the rest of the season. |

**Simulated injuries:**

- When a simulated injury happens during a game, the player gets a random fraction u ~ Uniform(0, 1) of that game. His yards that game are Gamma(shape u·α, scale μ/α).
- He then misses a number of games drawn from the **empirical distribution of injury absence lengths** (in games). That distribution is lumpy: many 1-game absences, a cluster of 2–4 games, and some season-ending injuries.
- Build that distribution from historical injury reports and game participation.
- Absences still going at season's end are right-censored. Handle this sensibly (for example, a Kaplan-Meier style estimate) and document your method.

**Hazard h:** the per-game probability of an injury that causes at least one missed game, estimated **by position only** from historical data. Placeholder: 0.045 per game for all positions.

**p_q:** the historical share of Questionable-designated players who played that week. Placeholder: 0.80.

**Considered but deliberately NOT included right now** (list these in ASSUMPTIONS.md as known omissions):

- Player-specific injury hazards based on injury history (all players use their position's hazard)
- Kyler Murray losing his starting job (benching risk)
- Stars resting in Week 18
- A receiver's production dropping when his team's QB is injured
- Reduced snap share when a Questionable player plays (he gets a full game)

### 4. Handling a game where the player gets injured

This is deliberately simple. **Snap counts are not entered.**

- If a player gets hurt during a game, I'll mark that game as `injured`. The game is **scratched from the rate update**: it does not change the μ posterior at all.
- The yards he gained in that game **still count toward his season total**, since they count for the bet.
- That game is still used up (it's no longer one of the remaining games).
- I'll set his availability state for the next game separately (`healthy`, `questionable`, `out:N`, `ir`, or `season_over`).

**Considered but not included:** using snap share as an exposure weight, so a partial game informs the rate in proportion to the snaps played. We chose not to, because it would require entering snap counts every week.

### 5. Calibration: 50% at kickoff

For each player, find the prior median μ₀ such that, with the player `healthy` and all 17 games remaining, the simulated P(season total > line) = 0.50.

- **Method:** bisection on μ₀. Use common random numbers (a fixed seed and the same underlying random draws across bisection steps) so the search is stable. The simulation in each step must include everything: rate uncertainty (τ), game-to-game noise (CV), and injury risk (h plus absence lengths).
- Use the half-point lines (e.g., 799.5) here, not the rounded numbers.
- Because injuries drag totals down, μ₀ will be greater than line/17. Rough expectations:
  - Kyler: around 220–225 ypg
  - Watson: around 54
  - Smith: around 72
  - Johnson: around 39
- **Sanity check:** print each calibrated μ₀ and the implied expected games played, E[G]. Add a note in ASSUMPTIONS.md that μ₀ should be roughly consistent with public healthy per-game projections, and that if it's far off, the injury inputs are the likely cause, not μ₀.
- **Calibration runs once.** μ₀ is stored and never re-solved after games are entered.

### 6. Simulation

Use at least 50,000 simulated seasons for each update. For each simulated season and each player:

1. **Draw μ once** from the player's current posterior on the grid (once per season, not once per game).
2. **Walk through the remaining games** week by week, starting from his current availability state, applying the rules in section 3.
3. **Add up the yards** for the games he plays in that simulated season, then add his actual yards to date.
4. **The leg hits if the final total > line**, using the half-point lines (so Watson hits at 800 or more).

**Combining the players:**

- **Assume the four players are independent.** None of them are teammates (verify this against current rosters, and document the check).
- There is a small real correlation when two of them play each other, such as a QB against an opposing receiver in a high-scoring game. This is not modeled; list it as an assumption.
- Compute P(all 4 hit) from the joint simulation: the share of simulated seasons where all four hit. With independence, this should match the product of the four individual probabilities within simulation noise. Print both as a check.

### 7. Inputs

Use simple files I can edit by hand:

**`config.yaml`:**
- players, positions, lines
- bet stake ($1,750) and profit ($23,245.79)
- the example bet size ($250)
- number of simulations and random seed
- all model parameters, with each one's source noted

**`games.csv`:** one row per player per game, with columns `week, player, yards, flag`.
- `flag` is `full` (a normal game) or `injured` (got hurt during the game, so the game is scratched from the rate update but the yards count).
- A game the player missed is entered as `flag=dnp` with yards 0. It uses up a game but doesn't count as a 0-yard performance.

**`status.csv`:** each player's current availability state going into his next game (`healthy`, `questionable`, `out:N`, `ir`, `season_over`).

The number of remaining games = 17 − (number of that player's team's games already played). Base it on the games entered, and document how you count.

### 8. Output

A command like `python goat_whale.py` prints a table:

| Player | Yards so far | Line | Games left | Status | Posterior median μ (and 80% interval) | Projected final total: median (10th–90th percentile) | P(hit) |
|---|---|---|---|---|---|---|---|

Followed by:

- **P(all 4 hit)**, plus the product of the individual probabilities as a check
- **Unit EV** (EV per $1)
- **EV of a $250 bet**

Also append each run to `history.csv` (week, each player's P, combined P, unit EV), so I can track the bet over the season.

---

## Data

Use **nflverse** (`nfl_data_py` in Python), seasons **2018–2025** unless you document a reason for a different range.

| Data | Function | Used for |
|---|---|---|
| Weekly player stats | `import_weekly_data` | Per-game passing and receiving yards, CV fits |
| Snap counts | `import_snap_counts` | Historical data only: filtering out partial games when fitting CV, and identifying in-game injuries when building the hazard |
| Injury reports | `import_injuries` | Hazard h, absence-length distribution, p_q, IR stint lengths |
| Schedules | `import_schedules` | Which games happened, so absences can be identified |
| Rosters | `import_weekly_rosters` / `import_seasonal_rosters` | Positions, and confirming none of the four players are teammates |

**Population filters for fitting:**
- QBs: primary starters
- WRs: player-seasons averaging roughly 35–90 receiving ypg in healthy games
- TEs: player-seasons averaging roughly 25–65 receiving ypg in healthy games

Document the exact filters you use.

**Historical preseason season-long yardage lines** (for τ) are not in nflverse. Try to find a usable source. If you can't, fall back to archived preseason projections or the placeholders, and say which one you used.

If you can't download data at all (no network access, for example), build the model with the placeholders, label them clearly, and make it easy to swap in estimated values later.

---

## Tests (write these as automated tests, and run them)

### Test 1: Kickoff calibration

With no games entered and all players `healthy`:

- Each player's P(hit) = 50% ± 0.5 percentage points
- P(all 4 hit) ≈ 6.25%
- Unit EV ≈ 0.0625 × 14.283309 − 1 = **−0.1073**
- $250 EV ≈ **−$26.82**

### Test 2: Week 1 results

Week 1 actual results:

| Player | Week 1 yards | Flag |
|---|---|---|
| Kyler Murray | 18 | `injured` (he got hurt during the game) |
| Christian Watson | 147 | `full` |
| DeVonta Smith | 53 | `full` |
| Juwan Johnson | 54 | `full` |

Everyone now has 16 games left. Expected behavior:

- **Watson:** μ posterior moves up (147 is far above his ~54 prior), and P(hit) rises substantially above 50%.
- **Johnson:** μ moves up (54 vs. a ~39 prior), and P(hit) rises above 50%.
- **Smith:** μ moves down slightly (53 vs. a ~72 prior), and P(hit) falls a bit below 50%.
- **Kyler:**
  - His μ posterior is **unchanged** from the prior, because the game is scratched.
  - His total is 18 with 16 games left.
- **Kyler status scenarios.** Run his P(hit) under each status and verify it strictly decreases in this order:

  `healthy` > `questionable` > `out:1` > `out:4` > `ir` > `season_over`

  - Even `healthy` should be below 50%, because he lost most of a game's production.
  - `season_over` must give exactly 0%. That makes P(all 4 hit) = 0%, unit EV = −1.00, and $250 EV = −$250.
- **Print the full output table** for each Kyler scenario, so I can look at the results.

### Test 3: Sanity checks

1. **A 0-yard full game** lowers μ, and lowers it proportionally more for a QB than for a TE (QB games are less noisy, so each one is more informative).
2. **A player who has already passed his line** shows 100%.
3. **Hit boundary:** a player finishing on exactly the rounded number (e.g., Watson at 800) counts as a hit; one yard less (799) does not.
4. **A `dnp` game** doesn't change μ.
5. **Simulation noise:** doubling the simulation count changes no probability by more than ~0.5 points.
6. **The posterior** stays a valid distribution (sums to 1, no NaNs) even after extreme inputs like 0 yards or 300 yards.

---

## Deliverables

1. The model code, organized and commented
2. `config.yaml`, `games.csv` (with the Week 1 rows), and `status.csv`
3. `ASSUMPTIONS.md`, containing all assumptions, all outside-data numbers with sources, all placeholders, and the list of things considered but not included
4. The tests, plus a short summary of their results, including the kickoff table and the Week 1 tables for each Kyler status scenario
5. A short `README.md` explaining how to enter a new week and run an update
