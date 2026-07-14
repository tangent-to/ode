/**
 * Adaptive Dormand-Prince RK45 (the same method as scipy's RK45 / MATLAB
 * ode45): a 5th-order step with an embedded 4th-order error estimate,
 * PI step-size control, a 4th-order dense-output interpolant, and
 * bracketed event detection.
 */

import { errorNorm, initialStep, makeResult, normalizeState, wrapRhs } from './_util.js';

// Dormand-Prince coefficients (7 stages, FSAL)
const C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1];
const A = [
  [],
  [1 / 5],
  [3 / 40, 9 / 40],
  [44 / 45, -56 / 15, 32 / 9],
  [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
  [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
];
// 5th-order solution weights (== A[6], FSAL)
const B = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0];
// Error = B - Bhat(4th order)
const E = [
  71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40,
];
const ORDER = 4; // error estimator order used for step control

/**
 * Integrate y' = f(t, y) with adaptive Dormand-Prince RK45.
 *
 * @param {Function} f - (t, y) => dydt; y is Array<number>, returns Array<number> (or scalar)
 * @param {[number, number]} tSpan - [t0, tEnd] (tEnd may be < t0 for backward integration)
 * @param {number|Array<number>} y0 - Initial state
 * @param {Object} [options]
 * @param {number} [options.rtol=1e-6] - Relative tolerance
 * @param {number} [options.atol=1e-9] - Absolute tolerance
 * @param {Array<number>} [options.tEval] - Times at which to report the solution (dense output)
 * @param {number} [options.maxStep=Infinity] - Maximum step size
 * @param {number} [options.firstStep] - Initial step size (auto if omitted)
 * @param {number} [options.maxSteps=100000] - Safety cap on accepted+rejected steps
 * @param {Function|Array<Function>} [options.events] - g(t, y) => number; a root marks an event
 * @returns {{t: number[], y: number[] | number[][], success: boolean, message: string, nfev: number, nsteps: number, events?: Array<{t: number[], y: number[] | number[][]}>}} Solver result
 */
export function rk45(f, tSpan, y0, options = {}) {
  const rtol = options.rtol ?? 1e-6;
  const atol = options.atol ?? 1e-9;
  const maxStep = options.maxStep ?? Infinity;
  const maxSteps = options.maxSteps ?? 100000;

  const [t0, tEnd] = tSpan;
  const direction = tEnd >= t0 ? 1 : -1;
  const { y: yInit, scalar } = normalizeState(y0);
  const n = yInit.length;
  const fn = wrapRhs(f, n);

  const eventFns = options.events
    ? (Array.isArray(options.events) ? options.events : [options.events])
    : [];
  const evalG = (t, y) => eventFns.map((g) => g(t, Array.from(y)));

  const tEval = options.tEval ? options.tEval.slice() : null;
  if (tEval) {
    for (const te of tEval) {
      if (direction * (te - t0) < -1e-12 || direction * (te - tEnd) > 1e-12) {
        throw new Error(`tEval point ${te} is outside tSpan [${t0}, ${tEnd}]`);
      }
    }
  }

  let t = t0;
  let y = Float64Array.from(yInit);
  let fEval = fn(t, y);
  let nfev = 1;

  let h = options.firstStep !== undefined
    ? Math.abs(options.firstStep) * direction
    : initialStep(fn, t, y, fEval, ORDER, atol, rtol, direction);

  const k = Array.from({ length: 7 }, () => new Float64Array(n));

  // Per-step work buffers, allocated once and reused across steps. The
  // accepted state swaps between `y` and `yNext` by reference; everything
  // recorded in the output copies values out first.
  const yi = new Float64Array(n);
  let yNext = new Float64Array(n);
  const err = new Float64Array(n);

  // Output collection
  const outT = [];
  const outY = [];
  let evalIdx = 0;
  if (!tEval) {
    outT.push(t);
    outY.push(Float64Array.from(y));
  }

  const events = eventFns.map(() => []);
  let gPrev = eventFns.length ? evalG(t, y) : null;

  const SAFETY = 0.9;
  const MIN_FACTOR = 0.2;
  const MAX_FACTOR = 10;
  let errPrev = 1e-4;
  let nsteps = 0;

  /**
   * Dense output: the Dormand-Prince quartic continuous extension, evaluated
   * at theta in [0, 1] across the accepted step. RK45_P holds scipy's
   * continuous-extension coefficients (7 stages x 4 powers of theta).
   */
  function interpolate(tOld, yOld, kk, hStep, theta) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let s = 0; s < 7; s++) {
        const p = RK45_P[s];
        const poly = theta * (p[0] + theta * (p[1] + theta * (p[2] + theta * p[3])));
        acc += kk[s][i] * poly;
      }
      out[i] = yOld[i] + hStep * acc;
    }
    return out;
  }

  function recordEvents(tOld, yOld, kk, hStep, tNew, yNew) {
    if (!eventFns.length) return;
    const gNew = evalG(tNew, yNew);
    for (let e = 0; e < eventFns.length; e++) {
      const a = gPrev[e];
      const b = gNew[e];
      if (a === 0) {
        events[e].push({ t: tOld, y: Array.from(yOld) });
      } else if (a < 0 !== b < 0) {
        // Bracket the root of g along the interpolant with bisection
        let lo = 0;
        let hi = 1;
        let gl = a;
        for (let iter = 0; iter < 60; iter++) {
          const mid = 0.5 * (lo + hi);
          const ym = interpolate(tOld, yOld, kk, hStep, mid);
          const gm = eventFns[e](tOld + mid * hStep, Array.from(ym));
          if (gm === 0 || (hi - lo) < 1e-14) { lo = hi = mid; break; }
          if ((gl < 0) !== (gm < 0)) { hi = mid; } else { lo = mid; gl = gm; }
        }
        const theta = 0.5 * (lo + hi);
        const yr = interpolate(tOld, yOld, kk, hStep, theta);
        events[e].push({ t: tOld + theta * hStep, y: Array.from(yr) });
      }
    }
    gPrev = gNew;
  }

  while (direction * (t - tEnd) < 0) {
    if (nsteps++ > maxSteps) {
      return finish(false, `exceeded maxSteps (${maxSteps})`);
    }

    let absH = Math.min(Math.abs(h), maxStep, Math.abs(tEnd - t));
    if (absH < 1e-14 * Math.max(1, Math.abs(t))) {
      return finish(false, 'step size underflow');
    }
    h = absH * direction;

    // Compute the 7 stages
    for (let i = 0; i < n; i++) k[0][i] = fEval[i];
    for (let s = 1; s < 7; s++) {
      for (let i = 0; i < n; i++) {
        let acc = 0;
        for (let j = 0; j < s; j++) acc += A[s][j] * k[j][i];
        yi[i] = y[i] + h * acc;
      }
      const ks = fn(t + C[s] * h, yi);
      nfev++;
      k[s] = ks;
    }

    // 5th-order solution and embedded error
    for (let i = 0; i < n; i++) {
      let sol = 0;
      let e = 0;
      for (let s = 0; s < 7; s++) {
        sol += B[s] * k[s][i];
        e += E[s] * k[s][i];
      }
      yNext[i] = y[i] + h * sol;
      err[i] = h * e;
    }

    const errNorm = errorNorm(err, y, yNext, atol, rtol);

    if (errNorm <= 1) {
      // Accept. PI controller.
      const tOld = t;
      const yOld = y;
      // Stage snapshot for the dense-output interpolant; only needed when
      // tEval points or event functions can look inside the step.
      const kk = tEval || eventFns.length
        ? k.map((ks) => Float64Array.from(ks))
        : null;
      const fNext = fn(t + h, yNext); // FSAL: reused as next k[0]
      nfev++;

      // Dense output for tEval points in (tOld, t+h]
      if (tEval) {
        while (evalIdx < tEval.length &&
               direction * (tEval[evalIdx] - (tOld + h)) <= 1e-12 &&
               direction * (tEval[evalIdx] - tOld) >= -1e-12) {
          const theta = h === 0 ? 0 : (tEval[evalIdx] - tOld) / h;
          outT.push(tEval[evalIdx]);
          outY.push(interpolate(tOld, yOld, kk, h, theta));
          evalIdx++;
        }
      }

      recordEvents(tOld, yOld, kk, h, tOld + h, yNext);

      t = tOld + h;
      y = yNext;
      yNext = yOld; // recycle the outgoing state buffer for the next attempt
      fEval = fNext;

      if (!tEval) {
        outT.push(t);
        outY.push(Float64Array.from(y));
      }

      const factor = errNorm === 0 ? MAX_FACTOR
        : Math.min(MAX_FACTOR,
            SAFETY * errNorm ** (-0.7 / (ORDER + 1)) * errPrev ** (0.4 / (ORDER + 1)));
      errPrev = Math.max(errNorm, 1e-10);
      h = absH * Math.max(MIN_FACTOR, factor) * direction;
    } else {
      // Reject, shrink.
      const factor = Math.max(MIN_FACTOR, SAFETY * errNorm ** (-1 / (ORDER + 1)));
      h = absH * factor * direction;
    }
  }

  return finish(true, 'integration successful');

  function finish(success, message) {
    const extra = { success, message, nfev, nsteps };
    if (eventFns.length) {
      extra.events = events.map((evs) => ({
        t: evs.map((e) => e.t),
        y: scalar ? evs.map((e) => e.y[0]) : evs.map((e) => e.y),
      }));
    }
    return makeResult(outT, outY, scalar, extra);
  }
}

// scipy RK45 continuous-extension coefficients (7 stages x 4 powers).
const RK45_P = [
  [1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432],
  [0, 0, 0, 0],
  [0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
  [0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
  [0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
  [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
  [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
];
