"""
Build the GOAT WHALE 2026 web page from the repo's own files.

  python web/build.py            # writes web/params.json and web/index.html
  node web/validate.js web/params.json   # check the JS port against the Python numbers

Inputs: config.yaml + calibration.json (model parameters and calibrated mu0),
web/official.json (the official results embedded in the page; when republishing, take the
latest copy from the live artifact's <script id="official-data"> first), web/page.template.html,
web/model.js.  Publish web/index.html to https://claude.ai/artifact/N3matY1z179F65JnBVkzSj.
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
cal = json.load(open(ROOT / "calibration.json"))
params = {"positions": cfg["positions"],
          "players": [dict(p, mu0=cal["players"][p["name"]]["mu0"]) for p in cfg["players"]],
          "bet": cfg["bet"]}
json.dump(params, open(WEB / "params.json", "w"))
official = json.load(open(WEB / "official.json"))
page = (WEB / "page.template.html").read_text()
page = (page.replace("__OFFICIAL_JSON__", json.dumps(official))
            .replace("__PARAMS_JSON__", json.dumps(params))
            .replace("__MODEL_JS__", (WEB / "model.js").read_text()))
(WEB / "index.html").write_text(page)
print(f"wrote web/params.json and web/index.html ({len(page):,} bytes); official through week {official['week']}")
