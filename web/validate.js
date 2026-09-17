require("./model.js");
const fs = require("fs");
const P = JSON.parse(fs.readFileSync(process.argv[2] || "../params.json", "utf8"));
const N = 100000, SEED = 20260916;
// special-function spot checks vs scipy (values computed in python below)
const checks = [[11, 5.5], [2.4, 0.03], [16, 40], [0.7, 0.2], [187, 150]];
console.log("gammp checks:", checks.map(([a, x]) => GW.gammp(a, x).toPrecision(10)).join(" "));
const wk1 = { "Kyler Murray": { 1: { y: 18, f: "injured" } }, "Christian Watson": { 1: { y: 147, f: "full" } }, "DeVonta Smith": { 1: { y: 53, f: "full" } }, "Juwan Johnson": { 1: { y: 54, f: "full" } } };
function run(games, status) {
  const states = {}; for (const p of P.players) states[p.name] = GW.stateFromGames(games[p.name] || {}, status[p.name] || "healthy");
  return GW.runModel(P, states, { nSims: N, seed: SEED });
}
const t0 = Date.now();
const kick = run({}, {});
console.log(`kickoff (${Date.now() - t0} ms):`, kick.players.map(p => p.name.split(" ")[1] + " " + (100 * p.pHit).toFixed(2)).join(" | "), "all4", (100 * kick.pAll).toFixed(2), "EV", kick.unitEv.toFixed(4));
for (const s of ["healthy", "questionable", "out:1", "out:4", "ir", "season_over"]) {
  const t = Date.now(); const r = run(wk1, { "Kyler Murray": s });
  console.log(`wk1 Kyler=${s.padEnd(12)} (${Date.now() - t} ms):`, r.players.map(p => (100 * p.pHit).toFixed(1)).join(" | "), "all4", (100 * r.pAll).toFixed(2), "EV", r.unitEv.toFixed(4), "mu", r.players.map(p => p.muMedian.toFixed(1)).join("/"));
}
