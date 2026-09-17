# TEST_RESULTS.md

Run: `python -m pytest -s -q tests/` on 2026-09-16 with the shipped `config.yaml` and
`calibration.json` (100,000 simulated seasons, seed 20260916). **Result: 10 passed, 0 failed.**

## Summary

**Test 1 — kickoff calibration** (no games, all healthy): every leg 50.0%; P(all 4) = 6.29%
(product 6.25%); unit EV −0.1013; $250 EV −$25.32. Targets were 50% ± 0.5, ≈ 6.25%, ≈ −0.1073,
≈ −$26.82; the EV gap is simulation noise on the joint indicator (SE ≈ 0.08 pts on P(all 4)).

**Test 2 — Week 1** (Kyler 18 `injured`, Watson 147, Smith 53, Johnson 54; 16 games left each):
- Watson: μ median 55.1 → 70.6, P(hit) 81.8%.
- Johnson: 41.0 → 44.6, P(hit) 61.9%.
- Smith: 75.5 → 71.3, P(hit) 42.7%.
- Kyler: posterior weights identical to the prior (scratched game), total 18, 16 games left.
- Kyler P(hit) by status: healthy 39.8% > questionable 28.6% > out:1 26.6% > out:4 1.3% > ir 0.2%
  > season_over 0.0% (strictly decreasing; healthy < 50% because most of a game was lost;
  season_over gives P(all 4) = 0, unit EV −1.00, $250 EV −$250). Full tables below.

**Test 3 — sanity checks**
1. A 0-yard full game lowers μ for both a QB and a TE. Informativeness was checked two ways:
   at the model's own τ values the drop in prior-SD units is QB 3.4τ vs TE 1.8τ; with equal
   τ = 0.20 the median ratio is QB 0.475 vs TE 0.733. **Note:** the literal comparison "QB's
   proportional drop is larger" does *not* hold at the estimated τ values (QB ratio 0.753 vs
   TE 0.574), because the QB's tight prior (τ = 0.083) confines the posterior to a ±4τ grid
   whose lower edge is 0.72 × μ₀; the QB likelihood is far steeper but the grid caps the move.
   See ASSUMPTIONS.md A7.
2. A player already past his line shows 100%.
3. Watson at exactly 800 with no games left (or `season_over`) counts as a hit; 799 does not.
4. A `dnp` game leaves μ unchanged.
5. Doubling simulations (100k → 200k) changed no probability by more than 0.10 points
   (max 0.001 for Kyler).
6. The posterior stays valid (sums to 1, no NaNs) after 0-yard, 300-yard and repeated
   extreme games at every position.

Also checked: `games.csv` / `status.csv` parse to the Week 1 inputs.

## Full test output

```

=== TEST 1: kickoff (no games entered, all healthy) ===
| Player           | Yards so far | Line   | Games left | Status  | Posterior mu median (80% int.) | Proj. final median (10th-90th) | P(hit) |
|------------------|--------------|--------|------------|---------|--------------------------------|--------------------------------|--------|
| Kyler Murray     | 0            | 3049.5 | 17         | healthy | 200.8 (180.7-223.3)            | 3049 (1542-3735)               | 50.0%  |
| Christian Watson | 0            | 799.5  | 17         | healthy | 55.1 (38.1-79.6)               | 799 (420-1268)                 | 50.0%  |
| DeVonta Smith    | 0            | 1099.5 | 17         | healthy | 75.5 (52.3-109.0)              | 1099 (582-1728)                | 50.0%  |
| Juwan Johnson    | 0            | 599.5  | 17         | healthy | 41.0 (27.7-60.6)               | 599 (307-976)                  | 50.0%  |

P(all 4 hit)          : 6.29%   (product of individual P: 6.25%; break-even 7.00%)
Unit EV (per $1)      : -0.1013   (decimal payout 14.283309)
EV of $250 bet        : $-25.32
Simulated seasons     : 100,000  (seed 20260916)
..
=== TEST 2: week 1 entered, Kyler status = healthy ===
| Player           | Yards so far | Line   | Games left | Status  | Posterior mu median (80% int.) | Proj. final median (10th-90th) | P(hit) |
|------------------|--------------|--------|------------|---------|--------------------------------|--------------------------------|--------|
| Kyler Murray     | 18           | 3049.5 | 16         | healthy | 200.8 (180.7-223.3)            | 2900 (1480-3548)               | 39.8%  |
| Christian Watson | 147          | 799.5  | 16         | healthy | 70.6 (49.5-99.4)               | 1116 (668-1645)                | 81.8%  |
| DeVonta Smith    | 53           | 1099.5 | 16         | healthy | 71.3 (51.8-96.7)               | 1036 (590-1523)                | 42.7%  |
| Juwan Johnson    | 54           | 599.5  | 16         | healthy | 44.6 (31.1-63.1)               | 671 (377-1021)                 | 61.9%  |

P(all 4 hit)          : 8.59%   (product of individual P: 8.59%; break-even 7.00%)
Unit EV (per $1)      : +0.2267   (decimal payout 14.283309)
EV of $250 bet        : $+56.66
Simulated seasons     : 100,000  (seed 20260916)

=== TEST 2: week 1 entered, Kyler status = questionable ===
| Player           | Yards so far | Line   | Games left | Status       | Posterior mu median (80% int.) | Proj. final median (10th-90th) | P(hit) |
|------------------|--------------|--------|------------|--------------|--------------------------------|--------------------------------|--------|
| Kyler Murray     | 18           | 3049.5 | 16         | questionable | 200.8 (180.7-223.3)            | 2704 (1059-3416)               | 28.6%  |
| Christian Watson | 147          | 799.5  | 16         | healthy      | 70.6 (49.5-99.4)               | 1116 (668-1645)                | 81.8%  |
| DeVonta Smith    | 53           | 1099.5 | 16         | healthy      | 71.3 (51.8-96.7)               | 1036 (590-1523)                | 42.7%  |
| Juwan Johnson    | 54           | 599.5  | 16         | healthy      | 44.6 (31.1-63.1)               | 671 (377-1021)                 | 61.9%  |

P(all 4 hit)          : 6.22%   (product of individual P: 6.17%; break-even 7.00%)
Unit EV (per $1)      : -0.1109   (decimal payout 14.283309)
EV of $250 bet        : $-27.72
Simulated seasons     : 100,000  (seed 20260916)

=== TEST 2: week 1 entered, Kyler status = out:1 ===
| Player           | Yards so far | Line   | Games left | Status  | Posterior mu median (80% int.) | Proj. final median (10th-90th) | P(hit) |
|------------------|--------------|--------|------------|---------|--------------------------------|--------------------------------|--------|
| Kyler Murray     | 18           | 3049.5 | 16         | out:1   | 200.8 (180.7-223.3)            | 2732 (1406-3341)               | 26.6%  |
| Christian Watson | 147          | 799.5  | 16         | healthy | 70.6 (49.5-99.4)               | 1116 (668-1645)                | 81.8%  |
| DeVonta Smith    | 53           | 1099.5 | 16         | healthy | 71.3 (51.8-96.7)               | 1036 (590-1523)                | 42.7%  |
| Juwan Johnson    | 54           | 599.5  | 16         | healthy | 44.6 (31.1-63.1)               | 671 (377-1021)                 | 61.9%  |

P(all 4 hit)          : 5.74%   (product of individual P: 5.74%; break-even 7.00%)
Unit EV (per $1)      : -0.1797   (decimal payout 14.283309)
EV of $250 bet        : $-44.93
Simulated seasons     : 100,000  (seed 20260916)

=== TEST 2: week 1 entered, Kyler status = out:4 ===
| Player           | Yards so far | Line   | Games left | Status  | Posterior mu median (80% int.) | Proj. final median (10th-90th) | P(hit) |
|------------------|--------------|--------|------------|---------|--------------------------------|--------------------------------|--------|
| Kyler Murray     | 18           | 3049.5 | 16         | out:4   | 200.8 (180.7-223.3)            | 2223 (1207-2716)               | 1.3%   |
| Christian Watson | 147          | 799.5  | 16         | healthy | 70.6 (49.5-99.4)               | 1116 (668-1645)                | 81.8%  |
| DeVonta Smith    | 53           | 1099.5 | 16         | healthy | 71.3 (51.8-96.7)               | 1036 (590-1523)                | 42.7%  |
| Juwan Johnson    | 54           | 599.5  | 16         | healthy | 44.6 (31.1-63.1)               | 671 (377-1021)                 | 61.9%  |

P(all 4 hit)          : 0.28%   (product of individual P: 0.28%; break-even 7.00%)
Unit EV (per $1)      : -0.9600   (decimal payout 14.283309)
EV of $250 bet        : $-240.00
Simulated seasons     : 100,000  (seed 20260916)

=== TEST 2: week 1 entered, Kyler status = ir ===
| Player           | Yards so far | Line   | Games left | Status  | Posterior mu median (80% int.) | Proj. final median (10th-90th) | P(hit) |
|------------------|--------------|--------|------------|---------|--------------------------------|--------------------------------|--------|
| Kyler Murray     | 18           | 3049.5 | 16         | ir      | 200.8 (180.7-223.3)            | 1074 (18-2286)                 | 0.2%   |
| Christian Watson | 147          | 799.5  | 16         | healthy | 70.6 (49.5-99.4)               | 1116 (668-1645)                | 81.8%  |
| DeVonta Smith    | 53           | 1099.5 | 16         | healthy | 71.3 (51.8-96.7)               | 1036 (590-1523)                | 42.7%  |
| Juwan Johnson    | 54           | 599.5  | 16         | healthy | 44.6 (31.1-63.1)               | 671 (377-1021)                 | 61.9%  |

P(all 4 hit)          : 0.03%   (product of individual P: 0.04%; break-even 7.00%)
Unit EV (per $1)      : -0.9950   (decimal payout 14.283309)
EV of $250 bet        : $-248.75
Simulated seasons     : 100,000  (seed 20260916)

=== TEST 2: week 1 entered, Kyler status = season_over ===
| Player           | Yards so far | Line   | Games left | Status      | Posterior mu median (80% int.) | Proj. final median (10th-90th) | P(hit) |
|------------------|--------------|--------|------------|-------------|--------------------------------|--------------------------------|--------|
| Kyler Murray     | 18           | 3049.5 | 16         | season_over | 200.8 (180.7-223.3)            | 18 (18-18)                     | 0.0%   |
| Christian Watson | 147          | 799.5  | 16         | healthy     | 70.6 (49.5-99.4)               | 1116 (668-1645)                | 81.8%  |
| DeVonta Smith    | 53           | 1099.5 | 16         | healthy     | 71.3 (51.8-96.7)               | 1036 (590-1523)                | 42.7%  |
| Juwan Johnson    | 54           | 599.5  | 16         | healthy     | 44.6 (31.1-63.1)               | 671 (377-1021)                 | 61.9%  |

P(all 4 hit)          : 0.00%   (product of individual P: 0.00%; break-even 7.00%)
Unit EV (per $1)      : -1.0000   (decimal payout 14.283309)
EV of $250 bet        : $-250.00
Simulated seasons     : 100,000  (seed 20260916)

Kyler P(hit) by status: {'healthy': 0.3975, 'questionable': 0.2859, 'out:1': 0.266, 'out:4': 0.0128, 'ir': 0.0017, 'season_over': 0.0}
.
TEST 3.1: 0-yard game, drop in prior-SD units: {'QB': 3.43, 'TE': 1.82} | median ratio with equal tau=0.20: {'QB': 0.475, 'TE': 0.733}
....
TEST 3.5: |P(N) - P(2N)| by player: {'Kyler Murray': 0.001, 'Christian Watson': 0.0001, 'DeVonta Smith': 0.0005, 'Juwan Johnson': 0.0008, 'all4': 0.0006}
...
10 passed in 7.82s
```
