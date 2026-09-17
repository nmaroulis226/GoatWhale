"""
Automated tests for the GOAT WHALE model (run: python -m pytest -s -q tests/).
-s shows the printed kickoff and Week-1 scenario tables.
"""
from pathlib import Path

import numpy as np
import pytest

from goat_whale import io as gio
from goat_whale.cli import do_calibrate
from goat_whale.model import GridPosterior, PlayerState, SEASON_GAMES, run_model
from goat_whale.report import format_result

ROOT = Path(__file__).resolve().parents[1]
PAYOUT = (1750 + 23245.79) / 1750          # 14.283309
NAMES = ["Kyler Murray", "Christian Watson", "DeVonta Smith", "Juwan Johnson"]

WEEK1 = {
    "Kyler Murray": [{"week": 1, "yards": 18.0, "flag": "injured"}],
    "Christian Watson": [{"week": 1, "yards": 147.0, "flag": "full"}],
    "DeVonta Smith": [{"week": 1, "yards": 53.0, "flag": "full"}],
    "Juwan Johnson": [{"week": 1, "yards": 54.0, "flag": "full"}],
}


@pytest.fixture(scope="session")
def cfg():
    return gio.load_config(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def cal(cfg):
    c = gio.load_calibration(ROOT / "calibration.json")
    if c is None or c.get("config_hash") != gio.config_hash(cfg):
        c = do_calibrate(cfg, force=True, quiet=True)
    return c


def run(cfg, cal, games=None, status=None, n_sims=None, seed=None, states=None):
    pos, av = gio.position_params(cfg)
    specs = gio.player_specs(cfg, cal)
    if states is None:
        games = {n: list((games or {}).get(n, [])) for n in NAMES}
        status = {n: (status or {}).get(n, "healthy") for n in NAMES}
        states = gio.build_states(games, status)
    sim = cfg["simulation"]
    return run_model(specs, [states[s.name] for s in specs], pos, av,
                     n_sims=int(n_sims or sim["n_sims"]), seed=int(sim["seed"] if seed is None else seed),
                     payout_decimal=cfg["payout_decimal"], example_stake=float(cfg["bet"]["example_stake"]),
                     n_grid=int(sim["n_grid"]))


def by_player(result):
    return {r["player"]: r for r in result["players"]}


def show(title, result):
    print(f"\n=== {title} ===")
    print(format_result(result))


# ----------------------------------------------------------------------------
# Test 1: kickoff calibration
# ----------------------------------------------------------------------------
def test_1_kickoff_calibration(cfg, cal):
    r = run(cfg, cal)
    show("TEST 1: kickoff (no games entered, all healthy)", r)
    for p in r["players"]:
        assert abs(p["p_hit"] - 0.5) <= 0.005, p
        assert p["games_left"] == SEASON_GAMES and p["yards_so_far"] == 0
    assert abs(r["p_all"] - 0.0625) <= 0.003
    assert abs(r["unit_ev"] - (0.0625 * PAYOUT - 1)) <= 0.045          # -0.1073
    assert abs(r["example_ev"] - 250 * (0.0625 * PAYOUT - 1)) <= 11.5  # -26.82
    assert abs(r["payout_decimal"] - 14.283309) < 1e-6
    assert abs(r["p_all"] - r["p_product"]) <= 0.003                   # independence check


# ----------------------------------------------------------------------------
# Test 2: week 1 results
# ----------------------------------------------------------------------------
def test_2_week1_posteriors(cfg, cal):
    r = run(cfg, cal, WEEK1, {"Kyler Murray": "healthy"})
    b = by_player(r)
    for n in NAMES:
        assert b[n]["games_left"] == 16
    w = b["Christian Watson"]
    assert w["mu_median"] > w["mu_prior_median"] * 1.05 and w["p_hit"] > 0.60
    j = b["Juwan Johnson"]
    assert j["mu_median"] > j["mu_prior_median"] and j["p_hit"] > 0.50
    s = b["DeVonta Smith"]
    assert 0.90 < s["mu_median"] / s["mu_prior_median"] < 1.0 and s["p_hit"] < 0.50
    k = b["Kyler Murray"]
    assert k["n_rate_updates"] == 0
    assert abs(k["mu_median"] - k["mu_prior_median"]) < 1e-9          # scratched game: prior unchanged
    assert k["yards_so_far"] == 18 and k["games_left"] == 16
    # posterior weights are literally the prior weights
    pos, _ = gio.position_params(cfg)
    prior = GridPosterior(k["mu_prior_median"], pos["QB"])
    st = gio.build_states({n: WEEK1[n] for n in NAMES}, {n: "healthy" for n in NAMES})["Kyler Murray"]
    post = GridPosterior(k["mu_prior_median"], pos["QB"])
    for y in st.full_game_yards:
        post.update(y)
    assert np.array_equal(prior.w, post.w)


def test_2_kyler_status_scenarios(cfg, cal):
    order = ["healthy", "questionable", "out:1", "out:4", "ir", "season_over"]
    ps = []
    last = None
    for s in order:
        r = run(cfg, cal, WEEK1, {"Kyler Murray": s})
        show(f"TEST 2: week 1 entered, Kyler status = {s}", r)
        ps.append(by_player(r)["Kyler Murray"]["p_hit"])
        last = r
    print("\nKyler P(hit) by status:", dict(zip(order, [round(p, 4) for p in ps])))
    for a, b in zip(ps, ps[1:]):
        assert a > b, (order, ps)
    assert ps[0] < 0.5                       # healthy but lost most of a game
    assert ps[-1] == 0.0
    assert last["p_all"] == 0.0
    assert last["unit_ev"] == pytest.approx(-1.0)
    assert last["example_ev"] == pytest.approx(-250.0)


# ----------------------------------------------------------------------------
# Test 3: sanity checks
# ----------------------------------------------------------------------------
def test_3_1_zero_yard_game_more_informative_for_qb(cfg, cal):
    """A 0-yard full game lowers mu for both positions and is more informative for a QB.

    'More informative' is checked two ways, because the literal ratio of medians also
    depends on how tight the prior is: with the estimated taus (QB 0.08 vs TE 0.31) the
    QB posterior is pinned near the lower edge of its +/-4 tau grid, so its proportional
    drop is capped at exp(-4 tau) even though its likelihood is far steeper.
      (a) at the model's own taus, the drop in log-median measured in prior-SD units is
          larger for the QB;
      (b) with the same tau for both positions, the proportional drop is larger for the QB.
    """
    from dataclasses import replace
    pos, _ = gio.position_params(cfg)
    mu0 = {"QB": cal["players"]["Kyler Murray"]["mu0"], "TE": cal["players"]["Juwan Johnson"]["mu0"]}

    std_shift, ratios_same_tau = {}, {}
    for p in ("QB", "TE"):
        g = GridPosterior(mu0[p], pos[p])
        m0 = g.median()
        g.update(0.0)
        assert g.is_valid() and g.median() < m0
        std_shift[p] = -np.log(g.median() / m0) / pos[p].tau
        g2 = GridPosterior(mu0[p], replace(pos[p], tau=0.20))
        m0 = g2.median()
        g2.update(0.0)
        ratios_same_tau[p] = g2.median() / m0
        assert ratios_same_tau[p] < 1.0
    print("\nTEST 3.1: 0-yard game, drop in prior-SD units:", {k: round(v, 2) for k, v in std_shift.items()},
          "| median ratio with equal tau=0.20:", {k: round(v, 3) for k, v in ratios_same_tau.items()})
    assert std_shift["QB"] > std_shift["TE"], std_shift
    assert ratios_same_tau["QB"] < ratios_same_tau["TE"], ratios_same_tau


def test_3_2_already_past_line_is_100_percent(cfg, cal):
    st = {n: PlayerState() for n in NAMES}
    st["Christian Watson"] = PlayerState(yards_so_far=900.0, games_used=10, full_game_yards=[90.0] * 10)
    r = run(cfg, cal, states=st)
    assert by_player(r)["Christian Watson"]["p_hit"] == 1.0


def test_3_3_hit_boundary(cfg, cal):
    for total, expected in ((800.0, 1.0), (799.0, 0.0)):
        st = {n: PlayerState() for n in NAMES}
        st["Christian Watson"] = PlayerState(yards_so_far=total, games_used=17, full_game_yards=[])
        r = run(cfg, cal, states=st)
        assert by_player(r)["Christian Watson"]["p_hit"] == expected
        st["Christian Watson"] = PlayerState(yards_so_far=total, games_used=12, status="season_over")
        r = run(cfg, cal, states=st)
        assert by_player(r)["Christian Watson"]["p_hit"] == expected


def test_3_4_dnp_game_does_not_change_mu(cfg, cal):
    games = {"Christian Watson": [{"week": 1, "yards": 0.0, "flag": "dnp"}]}
    r = run(cfg, cal, games)
    w = by_player(r)["Christian Watson"]
    assert w["n_rate_updates"] == 0 and abs(w["mu_median"] - w["mu_prior_median"]) < 1e-9
    assert w["games_left"] == 16 and w["yards_so_far"] == 0


def test_3_5_simulation_noise(cfg, cal):
    n = int(cfg["simulation"]["n_sims"])
    r1 = run(cfg, cal, WEEK1, {"Kyler Murray": "questionable"}, n_sims=n)
    r2 = run(cfg, cal, WEEK1, {"Kyler Murray": "questionable"}, n_sims=2 * n)
    diffs = {a["player"]: abs(a["p_hit"] - b["p_hit"]) for a, b in zip(r1["players"], r2["players"])}
    diffs["all4"] = abs(r1["p_all"] - r2["p_all"])
    print("\nTEST 3.5: |P(N) - P(2N)| by player:", {k: round(v, 4) for k, v in diffs.items()})
    assert max(diffs.values()) <= 0.005


def test_3_6_posterior_valid_after_extreme_inputs(cfg, cal):
    pos, _ = gio.position_params(cfg)
    for name, p in (("Kyler Murray", "QB"), ("Christian Watson", "WR"), ("Juwan Johnson", "TE")):
        for seq in ([0.0], [300.0], [0.0] * 5 + [300.0] * 5, [300.0] * 8, [0.0] * 8):
            g = GridPosterior(cal["players"][name]["mu0"], pos[p])
            for y in seq:
                g.update(y)
                assert g.is_valid(), (name, seq)
            assert np.all(np.isfinite(g.quantile([0.1, 0.5, 0.9])))


# ----------------------------------------------------------------------------
# Input files as delivered
# ----------------------------------------------------------------------------
def test_input_files_parse(cfg):
    games = gio.read_games(ROOT / "games.csv", NAMES)
    status = gio.read_status(ROOT / "status.csv", NAMES)
    assert games["Kyler Murray"] == [{"week": 1, "yards": 18.0, "flag": "injured"}]
    assert games["Christian Watson"][0]["yards"] == 147.0
    assert set(status) == set(NAMES)
