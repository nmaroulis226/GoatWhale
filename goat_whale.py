#!/usr/bin/env python3
"""
GOAT WHALE bet tracker.

  python goat_whale.py               # run an update from games.csv / status.csv, append history.csv
  python goat_whale.py calibrate     # solve the kickoff prior medians (mu0) once, write calibration.json
  python goat_whale.py --no-history  # run without appending to history.csv
  python goat_whale.py --sims 200000 # override the number of simulated seasons
"""
from goat_whale.cli import main

if __name__ == "__main__":
    main()
