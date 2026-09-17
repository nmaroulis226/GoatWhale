"""
Core model for the GOAT WHALE bet.

Structure (see ASSUMPTIONS.md for every modeling choice):

  1. Per-game yards | true healthy rate mu  ~ Gamma(mean mu, CV c(mu)),
     c(mu) = exp(a + b*log(mu)) by position.  Likelihood of an observed game is
     interval-censored: L(y|mu) = F(y+0.5) - F(y-0.5), y<=0 -> F(0.5).
  2. Belief about mu: 300-point log-spaced grid with a lognormal prior
     (median mu0, log-SD tau).  `GridPosterior.update(y)` multiplies by L.
     (Hook for a future random-walk drift: `GridPosterior.drift()` is a no-op.)
  3. Availability: healthy / questionable / out:N / ir / season_over, with a
     per-game hazard h and empirical absence-length distributions.
  4. Simulation: draw mu once per season, walk the remaining games, sum a
     single Gamma draw (shape = alpha * (full games + partial fractions)).
     All randomness comes from a `Draws` object (common random numbers), so
     calibration by bisection and scenario comparisons are stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import special

SEASON_GAMES = 17
SEASON_ENDING = 99          # absence length code meaning "rest of season"
ABSENCE_SUPPORT = 18        # pmf index k = 1..17 games, index 18 = season-ending (>=18)


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass
class PositionParams:
    """Per-position parameters."""
    cv_log_a: float          # log c = a + b log mu
    cv_log_b: float
    tau: float               # prior log-SD of mu
    cv_min: float = 0.05
    cv_max: float = 2.5


@dataclass
class AvailabilityParams:
    """Injury / availability parameters (per position)."""
    hazard: float                      # per-game P(injury causing >=1 missed game)
    p_questionable_plays: float        # P(plays | listed questionable)
    absence_pmf: np.ndarray            # P(L = k), k=1..17 at index k-1; index 17 = season-ending
    q_absence_pmf: np.ndarray          # same, for absences that began with a Questionable tag
    ir_pmf: np.ndarray                 # same, IR stints conditional on L >= 4

    def __post_init__(self):
        for name in ("absence_pmf", "q_absence_pmf", "ir_pmf"):
            v = np.asarray(getattr(self, name), dtype=float)
            if v.shape != (ABSENCE_SUPPORT,):
                raise ValueError(f"{name} must have length {ABSENCE_SUPPORT}, got {v.shape}")
            if v.min() < 0 or not np.isclose(v.sum(), 1.0, atol=1e-6):
                raise ValueError(f"{name} must be a probability vector summing to 1")
            setattr(self, name, v / v.sum())
        if not np.allclose(self.ir_pmf[:3], 0.0):
            raise ValueError("ir_pmf must put no mass on absences shorter than 4 games")


@dataclass
class PlayerSpec:
    name: str
    position: str
    line: float              # half-point line, e.g. 799.5
    team: str = ""
    mu0: Optional[float] = None   # calibrated prior median (yards per healthy game)


@dataclass
class PlayerState:
    """What has happened so far this season for one player."""
    yards_so_far: float = 0.0
    games_used: int = 0                       # rows in games.csv (full + injured + dnp)
    full_game_yards: List[float] = field(default_factory=list)   # games that update mu
    status: str = "healthy"                   # healthy | questionable | out:N | ir | season_over

    @property
    def games_left(self) -> int:
        return SEASON_GAMES - self.games_used


# ----------------------------------------------------------------------------
# Per-game distribution
# ----------------------------------------------------------------------------
def cv_of_mu(mu, pp: PositionParams):
    """Coefficient of variation of a full healthy game as a function of the rate mu."""
    mu = np.asarray(mu, dtype=float)
    c = np.exp(pp.cv_log_a + pp.cv_log_b * np.log(np.maximum(mu, 1e-9)))
    return np.clip(c, pp.cv_min, pp.cv_max)


def gamma_shape_scale(mu, pp: PositionParams):
    c = cv_of_mu(mu, pp)
    alpha = 1.0 / c ** 2
    scale = np.asarray(mu, dtype=float) / alpha
    return alpha, scale


def interval_loglik(y: float, mu_grid: np.ndarray, pp: PositionParams) -> np.ndarray:
    """
    log L(y | mu) for each grid point, with the game treated as interval-censored:
      L = F(y + 0.5) - F(y - 0.5);  y <= 0 -> L = F(0.5)
    If the interval probability underflows to 0 we fall back to the log-density
    at y (times a unit-width interval) so extreme observations stay finite.
    """
    alpha, scale = gamma_shape_scale(mu_grid, pp)
    hi = max(y, 0.0) + 0.5
    lo = max(y - 0.5, 0.0)
    p = special.gammainc(alpha, hi / scale) - special.gammainc(alpha, lo / scale)
    with np.errstate(divide="ignore"):
        ll = np.log(p)
    bad = ~np.isfinite(ll)
    if bad.any():
        yy = max(y, 0.5)
        ll_pdf = ((alpha - 1) * np.log(yy) - yy / scale - special.gammaln(alpha) - alpha * np.log(scale))
        ll = np.where(bad, ll_pdf, ll)
    return ll


# ----------------------------------------------------------------------------
# Grid posterior for mu
# ----------------------------------------------------------------------------
class GridPosterior:
    """Discrete belief about a player's true healthy yards-per-game rate."""

    def __init__(self, mu0: float, pp: PositionParams, n_grid: int = 300, half_width_sds: float = 4.0):
        self.mu0 = float(mu0)
        self.pp = pp
        z = np.linspace(-half_width_sds * pp.tau, half_width_sds * pp.tau, n_grid)
        self.log_mu = np.log(mu0) + z
        self.mu = np.exp(self.log_mu)
        # lognormal prior; grid is uniform in log(mu) so density in z is Gaussian
        logw = -0.5 * (z / pp.tau) ** 2
        self.log_w = logw - special.logsumexp(logw)
        self.n_updates = 0

    @property
    def w(self) -> np.ndarray:
        return np.exp(self.log_w)

    def update(self, y: float) -> None:
        """Multiply by the interval-censored likelihood of one full healthy game."""
        self.log_w = self.log_w + interval_loglik(y, self.mu, self.pp)
        self.log_w -= special.logsumexp(self.log_w)
        self.n_updates += 1

    def drift(self) -> None:
        """Hook for a week-to-week random walk in mu (deliberately not implemented)."""
        return None

    def cdf(self) -> np.ndarray:
        return np.cumsum(self.w)

    def quantile(self, q) -> np.ndarray:
        """Quantiles of mu, interpolated in log-space along the grid CDF (mid-point convention,
        so a symmetric prior has median exactly mu0)."""
        q = np.atleast_1d(np.asarray(q, dtype=float))
        w = self.w
        c = np.cumsum(w) - 0.5 * w
        return np.exp(np.interp(q, c, self.log_mu))

    def median(self) -> float:
        return float(self.quantile(0.5)[0])

    def sample(self, u: np.ndarray) -> np.ndarray:
        """Inverse-CDF draw of mu from the grid using uniforms u (common random numbers)."""
        idx = np.searchsorted(self.cdf(), u, side="right")
        return self.mu[np.clip(idx, 0, len(self.mu) - 1)]

    def is_valid(self) -> bool:
        w = self.w
        return bool(np.all(np.isfinite(w)) and np.isclose(w.sum(), 1.0, atol=1e-9) and (w >= 0).all())


# ----------------------------------------------------------------------------
# Common random numbers
# ----------------------------------------------------------------------------
class Draws:
    """All uniforms used by the simulation, generated once from a seed."""

    def __init__(self, n_sims: int, n_players: int, seed: int):
        rng = np.random.default_rng(seed)
        self.n = n_sims
        G = SEASON_GAMES
        self.u_mu = rng.random((n_players, n_sims))
        self.u_total = rng.random((n_players, n_sims))
        self.u_inj = rng.random((n_players, n_sims, G))
        self.u_frac = rng.random((n_players, n_sims, G))
        self.u_len = rng.random((n_players, n_sims, G))
        self.u_q = rng.random((n_players, n_sims))
        self.u_ir = rng.random((n_players, n_sims))


# ----------------------------------------------------------------------------
# Availability simulation
# ----------------------------------------------------------------------------
def _draw_length(pmf: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Absence length (games) via inverse CDF; the last cell means season-ending."""
    cdf = np.cumsum(pmf)
    k = np.searchsorted(cdf, u, side="right") + 1        # 1..18
    return np.where(k >= ABSENCE_SUPPORT, SEASON_ENDING, k).astype(np.int64)


def parse_status(status: str):
    s = status.strip().lower()
    if s in ("healthy", "questionable", "ir", "season_over"):
        return s, 0
    if s.startswith("out:"):
        n = int(s.split(":", 1)[1])
        if n < 1:
            raise ValueError("out:N needs N >= 1")
        return "out", n
    raise ValueError(f"unknown status {status!r} (use healthy, questionable, out:N, ir, season_over)")


@dataclass
class AvailabilityResult:
    full_games: np.ndarray       # number of full games played, per sim
    partial_frac: np.ndarray     # sum of partial-game fractions u, per sim
    games_played: np.ndarray     # full + partial appearances, per sim


def simulate_availability(status: str, games_left: int, ap: AvailabilityParams, draws: Draws,
                          player_idx: int) -> AvailabilityResult:
    """Walk the remaining games for every simulated season (vectorised over sims)."""
    n = draws.n
    kind, n_out = parse_status(status)
    miss_left = np.zeros(n, dtype=np.int64)
    full = np.zeros(n, dtype=np.int64)
    frac = np.zeros(n, dtype=float)
    played = np.zeros(n, dtype=np.int64)

    if games_left <= 0 or kind == "season_over":
        return AvailabilityResult(full, frac, played)
    if kind == "out":
        miss_left[:] = n_out
    elif kind == "ir":
        miss_left[:] = _draw_length(ap.ir_pmf, draws.u_ir[player_idx])
    elif kind == "questionable":
        sits = draws.u_q[player_idx] >= ap.p_questionable_plays
        # He misses this game plus the rest of an absence drawn from the
        # Questionable-initiated distribution (length counted from this game).
        miss_left[sits] = _draw_length(ap.q_absence_pmf, draws.u_len[player_idx, sits, 0])

    u_inj = draws.u_inj[player_idx]
    u_frac = draws.u_frac[player_idx]
    u_len = draws.u_len[player_idx]
    for g in range(games_left):
        out_now = miss_left > 0
        active = ~out_now
        injured = active & (u_inj[:, g] < ap.hazard)
        full += active & ~injured
        frac += np.where(injured, u_frac[:, g], 0.0)
        played += active
        miss_left[out_now] -= 1
        if injured.any():
            miss_left[injured] = _draw_length(ap.absence_pmf, u_len[injured, g])
    return AvailabilityResult(full, frac, played)


# ----------------------------------------------------------------------------
# Yardage simulation
# ----------------------------------------------------------------------------
def simulate_player(post: GridPosterior, avail: AvailabilityResult, draws: Draws, player_idx: int,
                    mu_override: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Remaining-season yards per simulated season.  mu is drawn once per season
    from the grid; all games share it, so the season sum is one Gamma draw with
    shape alpha(mu) * (full games + sum of partial fractions).
    """
    mu = post.sample(draws.u_mu[player_idx]) if mu_override is None else mu_override
    alpha, scale = gamma_shape_scale(mu, post.pp)
    shape = alpha * (avail.full_games + avail.partial_frac)
    out = np.zeros(draws.n)
    pos = shape > 0
    out[pos] = scale[pos] * special.gammaincinv(shape[pos], draws.u_total[player_idx, pos])
    return out


# ----------------------------------------------------------------------------
# Calibration: P(total > line) = 0.5 at kickoff
# ----------------------------------------------------------------------------
def calibrate_mu0(spec: PlayerSpec, pp: PositionParams, ap: AvailabilityParams, draws: Draws,
                  player_idx: int, target: float = 0.5, tol: float = 1e-4, max_iter: int = 60,
                  n_grid: int = 300) -> Dict[str, float]:
    """
    Bisection on the prior median mu0 with common random numbers.  The
    availability walk does not depend on mu0, so it is simulated once; each
    bisection step only re-maps the fixed uniforms through the new prior.
    """
    avail = simulate_availability("healthy", SEASON_GAMES, ap, draws, player_idx)
    # prior draws: z index is fixed, mu = mu0 * exp(z)
    ref = GridPosterior(1.0, pp, n_grid=n_grid)
    z_draw = ref.log_mu[np.clip(np.searchsorted(ref.cdf(), draws.u_mu[player_idx], side="right"),
                                0, n_grid - 1)]

    def p_hit(mu0: float) -> float:
        mu = np.exp(np.log(mu0) + z_draw)
        alpha, scale = gamma_shape_scale(mu, pp)
        shape = alpha * (avail.full_games + avail.partial_frac)
        tot = np.zeros(draws.n)
        pos = shape > 0
        tot[pos] = scale[pos] * special.gammaincinv(shape[pos], draws.u_total[player_idx, pos])
        return float(np.mean(tot > spec.line))

    lo, hi = spec.line / SEASON_GAMES * 0.7, spec.line / SEASON_GAMES * 2.0
    assert p_hit(lo) < target < p_hit(hi), "bisection bracket does not contain the target"
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if p_hit(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    mu0 = 0.5 * (lo + hi)
    return {"mu0": mu0, "p_hit": p_hit(mu0), "expected_games": float(avail.games_played.mean()),
            "expected_full_games": float(avail.full_games.mean())}


# ----------------------------------------------------------------------------
# Full update
# ----------------------------------------------------------------------------
def run_model(specs: Sequence[PlayerSpec], states: Sequence[PlayerState], pos_params: Dict[str, PositionParams],
              avail_params: Dict[str, AvailabilityParams], n_sims: int, seed: int, payout_decimal: float,
              example_stake: float, n_grid: int = 300) -> Dict:
    """Run one update: posteriors, joint simulation, per-leg and parlay probabilities, EV."""
    draws = Draws(n_sims, len(specs), seed)
    rows = []
    hits = np.ones(n_sims, dtype=bool)
    for i, (spec, st) in enumerate(zip(specs, states)):
        pp = pos_params[spec.position]
        ap = avail_params[spec.position]
        post = GridPosterior(spec.mu0, pp, n_grid=n_grid)
        for y in st.full_game_yards:
            post.update(y)
        avail = simulate_availability(st.status, st.games_left, ap, draws, i)
        rem = simulate_player(post, avail, draws, i)
        final = st.yards_so_far + rem
        hit = final > spec.line
        hits &= hit
        q_mu = post.quantile([0.10, 0.50, 0.90])
        rows.append({
            "player": spec.name, "position": spec.position, "team": spec.team,
            "yards_so_far": st.yards_so_far, "line": spec.line, "games_left": st.games_left,
            "status": st.status, "mu_median": float(q_mu[1]), "mu_p10": float(q_mu[0]),
            "mu_p90": float(q_mu[2]), "mu_prior_median": spec.mu0,
            "final_median": float(np.median(final)), "final_p10": float(np.percentile(final, 10)),
            "final_p90": float(np.percentile(final, 90)), "p_hit": float(hit.mean()),
            "exp_games_played": float(avail.games_played.mean()),
            "n_rate_updates": post.n_updates, "posterior_valid": post.is_valid(),
        })
    p_all = float(hits.mean())
    p_prod = float(np.prod([r["p_hit"] for r in rows]))
    unit_ev = p_all * payout_decimal - 1.0
    return {"players": rows, "p_all": p_all, "p_product": p_prod, "unit_ev": unit_ev,
            "example_stake": example_stake, "example_ev": example_stake * unit_ev,
            "payout_decimal": payout_decimal, "n_sims": n_sims, "seed": seed}
