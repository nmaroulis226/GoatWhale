"""GOAT WHALE: Bayesian + Monte Carlo pricing of a 4-leg season-long NFL yardage parlay."""
from .model import (  # noqa: F401
    PositionParams, AvailabilityParams, PlayerSpec, PlayerState, Draws,
    GridPosterior, cv_of_mu, interval_loglik, simulate_availability, simulate_player,
    calibrate_mu0, run_model,
)
