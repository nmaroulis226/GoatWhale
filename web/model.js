/* GOAT WHALE model — JavaScript port of goat_whale/model.py (same grid posterior, availability
   walk and Gamma season sum; Gamma draws use Marsaglia-Tsang instead of the inverse CDF, with a
   per-simulation seeded stream so scenarios share common random numbers). */
(function (global) {
  "use strict";
  const SEASON_GAMES = 17, SEASON_ENDING = 99, SUPPORT = 18, N_GRID = 300;

  // ---------- special functions ----------
  const LANCZOS = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  function gammln(x) {
    let y = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) ser += LANCZOS[j] / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }
  // regularized lower incomplete gamma P(a, x)
  function gammp(a, x) {
    if (x <= 0) return 0;
    const gln = gammln(a);
    if (x < a + 1) {
      let ap = a, del = 1 / a, sum = del;
      for (let n = 0; n < 2000; n++) {
        ap += 1; del *= x / ap; sum += del;
        if (Math.abs(del) < Math.abs(sum) * 1e-15) break;
      }
      return sum * Math.exp(-x + a * Math.log(x) - gln);
    }
    const FPMIN = 1e-300;
    let b = x + 1 - a, c = 1 / FPMIN, d = 1 / b, h = d;
    for (let i = 1; i < 2000; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = b + an / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return 1 - Math.exp(-x + a * Math.log(x) - gln) * h;
  }
  function logsumexp(arr) {
    let m = -Infinity;
    for (const v of arr) if (v > m) m = v;
    if (m === -Infinity) return m;
    let s = 0;
    for (const v of arr) s += Math.exp(v - m);
    return m + Math.log(s);
  }

  // ---------- per-game distribution ----------
  function cvOfMu(mu, pp) {
    const c = Math.exp(pp.cv_log_a + pp.cv_log_b * Math.log(Math.max(mu, 1e-9)));
    return Math.min(Math.max(c, 0.05), 2.5);
  }
  function shapeScale(mu, pp) {
    const c = cvOfMu(mu, pp), alpha = 1 / (c * c);
    return [alpha, mu / alpha];
  }
  function intervalLoglik(y, mu, pp) {
    const [alpha, scale] = shapeScale(mu, pp);
    const hi = Math.max(y, 0) + 0.5, lo = Math.max(y - 0.5, 0);
    const p = gammp(alpha, hi / scale) - gammp(alpha, lo / scale);
    let ll = Math.log(p);
    if (!isFinite(ll)) {
      const yy = Math.max(y, 0.5);
      ll = (alpha - 1) * Math.log(yy) - yy / scale - gammln(alpha) - alpha * Math.log(scale);
    }
    return ll;
  }

  // ---------- grid posterior ----------
  class GridPosterior {
    constructor(mu0, pp) {
      this.mu0 = mu0; this.pp = pp;
      this.logMu = new Float64Array(N_GRID); this.mu = new Float64Array(N_GRID);
      this.logW = new Float64Array(N_GRID);
      const lo = -4 * pp.tau, hi = 4 * pp.tau;
      for (let i = 0; i < N_GRID; i++) {
        const z = lo + (hi - lo) * i / (N_GRID - 1);
        this.logMu[i] = Math.log(mu0) + z; this.mu[i] = Math.exp(this.logMu[i]);
        this.logW[i] = -0.5 * (z / pp.tau) ** 2;
      }
      this.normalize(); this.nUpdates = 0;
    }
    normalize() { const l = logsumexp(this.logW); for (let i = 0; i < N_GRID; i++) this.logW[i] -= l; }
    update(y) {
      for (let i = 0; i < N_GRID; i++) this.logW[i] += intervalLoglik(y, this.mu[i], this.pp);
      this.normalize(); this.nUpdates++;
    }
    weights() { const w = new Float64Array(N_GRID); for (let i = 0; i < N_GRID; i++) w[i] = Math.exp(this.logW[i]); return w; }
    cdf() { const w = this.weights(), c = new Float64Array(N_GRID); let s = 0; for (let i = 0; i < N_GRID; i++) { s += w[i]; c[i] = s; } return c; }
    quantile(q) {  // mid-point convention, log-space interpolation (matches Python)
      const w = this.weights(), c = new Float64Array(N_GRID); let s = 0;
      for (let i = 0; i < N_GRID; i++) { s += w[i]; c[i] = s - 0.5 * w[i]; }
      if (q <= c[0]) return this.mu[0];
      if (q >= c[N_GRID - 1]) return this.mu[N_GRID - 1];
      let i = 1; while (c[i] < q) i++;
      const t = (q - c[i - 1]) / (c[i] - c[i - 1]);
      return Math.exp(this.logMu[i - 1] + t * (this.logMu[i] - this.logMu[i - 1]));
    }
    isValid() { const w = this.weights(); let s = 0; for (const v of w) { if (!isFinite(v) || v < 0) return false; s += v; } return Math.abs(s - 1) < 1e-9; }
  }

  // ---------- random numbers: one seeded stream per (player, simulation) ----------
  function mix32(h) { h = Math.imul(h ^ (h >>> 16), 0x85ebca6b); h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35); return (h ^ (h >>> 16)) >>> 0; }
  function makeRng(seed, p, i) {
    let a = mix32((seed >>> 0) ^ Math.imul(p + 1, 0x9e3779b9) ^ Math.imul(i + 1, 0x7f4a7c15));
    return function () {   // mulberry32
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function normal(rng) { const u1 = 1 - rng(), u2 = rng(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }
  function gammaRand(a, rng) {  // Marsaglia-Tsang
    if (a < 1) return gammaRand(a + 1, rng) * Math.pow(1 - rng(), 1 / a);
    const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = normal(rng); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = rng();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  function cumsum(p) { const c = new Float64Array(p.length); let s = 0; for (let i = 0; i < p.length; i++) { s += p[i]; c[i] = s; } return c; }
  function drawLen(cdf, u) { let k = 0; while (k < SUPPORT && cdf[k] <= u) k++; k += 1; return k >= SUPPORT ? SEASON_ENDING : k; }
  function searchRight(cdf, u) { let lo = 0, hi = cdf.length; while (lo < hi) { const m = (lo + hi) >> 1; if (cdf[m] <= u) lo = m + 1; else hi = m; } return Math.min(lo, cdf.length - 1); }

  function parseStatus(s) {
    s = String(s || "healthy").trim().toLowerCase();
    if (["healthy", "questionable", "ir", "season_over"].includes(s)) return [s, 0];
    if (s.startsWith("out:")) { const n = parseInt(s.slice(4), 10); if (!(n >= 1)) throw new Error("out:N needs N >= 1"); return ["out", n]; }
    throw new Error("unknown status " + s);
  }

  // ---------- one player ----------
  function simulatePlayer(spec, state, pp, ap, nSims, seed, pIdx) {
    const post = new GridPosterior(spec.mu0, pp);
    for (const y of state.fullGameYards) post.update(y);
    const cdf = post.cdf(), absCdf = cumsum(ap.absence_pmf), qCdf = cumsum(ap.q_absence_pmf), irCdf = cumsum(ap.ir_pmf);
    const [kind, nOut] = parseStatus(state.status);
    const gamesLeft = SEASON_GAMES - state.gamesUsed;
    const remaining = new Float64Array(nSims), hit = new Uint8Array(nSims);
    let playedSum = 0;
    const uInj = new Float64Array(SEASON_GAMES), uFrac = new Float64Array(SEASON_GAMES), uLen = new Float64Array(SEASON_GAMES);
    for (let i = 0; i < nSims; i++) {
      const rng = makeRng(seed, pIdx, i);
      const uMu = rng(), uQ = rng(), uIr = rng();
      for (let g = 0; g < SEASON_GAMES; g++) { uInj[g] = rng(); uFrac[g] = rng(); uLen[g] = rng(); }
      let missLeft = 0, full = 0, frac = 0, played = 0;
      if (gamesLeft > 0 && kind !== "season_over") {
        if (kind === "out") missLeft = nOut;
        else if (kind === "ir") missLeft = drawLen(irCdf, uIr);
        else if (kind === "questionable" && uQ >= ap.p_questionable_plays) missLeft = drawLen(qCdf, uLen[0]);
        for (let g = 0; g < gamesLeft; g++) {
          if (missLeft > 0) { missLeft--; continue; }
          played++;
          if (uInj[g] < ap.hazard) { frac += uFrac[g]; missLeft = drawLen(absCdf, uLen[g]); }
          else full++;
        }
      }
      playedSum += played;
      const mu = post.mu[searchRight(cdf, uMu)];
      const [alpha, scale] = shapeScale(mu, pp);
      const shape = alpha * (full + frac);
      const rem = shape > 0 ? scale * gammaRand(shape, rng) : 0;
      remaining[i] = rem;
      hit[i] = (state.yardsSoFar + rem > spec.line) ? 1 : 0;
    }
    let hits = 0, remSum = 0;
    for (let i = 0; i < nSims; i++) { hits += hit[i]; remSum += remaining[i]; }
    const sorted = Float64Array.from(remaining).sort();
    const pct = (q) => { const x = q * (nSims - 1), lo = Math.floor(x), hi = Math.min(lo + 1, nSims - 1); return sorted[lo] + (sorted[hi] - sorted[lo]) * (x - lo); };
    return {
      hit, pHit: hits / nSims, gamesLeft,
      muMedian: post.quantile(0.5), muP10: post.quantile(0.1), muP90: post.quantile(0.9), muPrior: spec.mu0,
      finalMedian: state.yardsSoFar + pct(0.5), finalP10: state.yardsSoFar + pct(0.1), finalP90: state.yardsSoFar + pct(0.9),
      expRemaining: remSum / nSims, expGames: playedSum / nSims, nUpdates: post.nUpdates, posteriorValid: post.isValid(),
    };
  }

  // ---------- full run ----------
  function runModel(params, states, opts) {
    const nSims = opts.nSims, seed = opts.seed;
    const payout = (params.bet.stake + params.bet.profit) / params.bet.stake;
    const all = new Uint8Array(nSims).fill(1);
    const players = params.players.map((spec, idx) => {
      const st = states[spec.name];
      const r = simulatePlayer(spec, st, params.positions[spec.position], params.positions[spec.position], nSims, seed, idx);
      for (let i = 0; i < nSims; i++) all[i] &= r.hit[i];
      delete r.hit;
      return Object.assign({ name: spec.name, position: spec.position, team: spec.team, line: spec.line,
        yardsSoFar: st.yardsSoFar, gamesUsed: st.gamesUsed, gamesPlayed: st.gamesPlayed, status: st.status }, r);
    });
    let a = 0; for (let i = 0; i < nSims; i++) a += all[i];
    const pAll = a / nSims, pProduct = players.reduce((x, p) => x * p.pHit, 1), unitEv = pAll * payout - 1;
    return { players, pAll, pProduct, unitEv, payout, breakEven: 1 / payout, exampleStake: params.bet.example_stake, exampleEv: params.bet.example_stake * unitEv, nSims, seed };
  }

  /* Build a player's state from a week map {week: {y, f}} (f = full | injured | dnp). */
  function stateFromGames(weeks, status) {
    const st = { yardsSoFar: 0, gamesUsed: 0, gamesPlayed: 0, fullGameYards: [], status: status || "healthy" };
    const keys = Object.keys(weeks || {}).map(Number).sort((x, y) => x - y);
    for (const w of keys) {
      const g = weeks[w]; if (!g) continue;
      st.gamesUsed++;
      if (g.f === "dnp") continue;
      const y = Number(g.y) || 0;
      st.yardsSoFar += y; st.gamesPlayed++;
      if (g.f === "full") st.fullGameYards.push(y);
    }
    return st;
  }

  global.GW = { runModel, stateFromGames, GridPosterior, gammp, gammln, SEASON_GAMES };
})(typeof window !== "undefined" ? window : globalThis);
