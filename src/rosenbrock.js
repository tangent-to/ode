/**
 * Adaptive Rosenbrock-Wanner solver for stiff systems: the 4-stage,
 * 4th-order Kaps-Rentrop method with Shampine's coefficient set (the
 * classic GRK4 pairing used by Numerical Recipes' stiff integrator),
 * with an embedded 3rd-order error estimate. The method is A-stable
 * with strong damping at infinity (|R(inf)| = 1/3), so step size is
 * limited by accuracy rather than stability on stiff problems.
 *
 * Each step factors W = I/(gamma*h) - J once with lina's LU and back-
 * substitutes the four stage systems against that single factorization.
 * The Jacobian J = df/dy is recomputed at every accepted point by
 * central finite differences (or via options.jac when provided).
 *
 * Dense output (options.tEval) uses cubic Hermite interpolation across
 * accepted steps (the states and derivatives at both step ends), which
 * is 3rd-order accurate inside a step.
 *
 * Event detection is not yet supported for the stiff solver; passing
 * options.events throws. Use rk45 for event detection.
 */

import { errorNorm, initialStep, makeResult, normalizeState, wrapRhs } from './_util.js';
import { lu } from '@tangent.to/lina';

// Kaps-Rentrop 4(3), Shampine's parameter set (gamma = 1/2).
const GAMMA = 1 / 2;
const A21 = 2;
const A31 = 48 / 25;
const A32 = 6 / 25;
const C21 = -8;
const C31 = 372 / 25;
const C32 = 12 / 5;
const C41 = -112 / 125;
const C42 = -54 / 125;
const C43 = -2 / 5;
const B = [19 / 9, 1 / 2, 25 / 108, 125 / 108];
const E = [17 / 54, 7 / 36, 0, 125 / 108];
// Coefficients of the h * df/dt terms and stage abscissae.
const C1X = 1 / 2;
const C2X = -3 / 2;
const C3X = 121 / 50;
const C4X = 29 / 250;
const A2X = 1;
const A3X = 3 / 5;
const ORDER = 3; // embedded error estimator order used for step control

const SQRT_EPS = Math.sqrt(Number.EPSILON);

/**
 * Jacobian df/dy by central finite differences, columnwise, with
 * step h_j = sqrt(eps) * max(1, |y_j|). Costs 2n RHS evaluations.
 *
 * @param {Function} fn - Wrapped RHS
 * @param {number} t - Current time
 * @param {Float64Array} y - Current state
 * @returns {Array<Array<number>>} Nested n x n Jacobian
 */
function fdJacobian(fn, t, y) {
  const n = y.length;
  const J = [];
  for (let i = 0; i < n; i++) J.push(new Array(n).fill(0));
  const yp = Float64Array.from(y);
  for (let j = 0; j < n; j++) {
    const hj = SQRT_EPS * Math.max(1, Math.abs(y[j]));
    yp[j] = y[j] + hj;
    const fPlus = fn(t, yp);
    yp[j] = y[j] - hj;
    const fMinus = fn(t, yp);
    yp[j] = y[j];
    for (let i = 0; i < n; i++) J[i][j] = (fPlus[i] - fMinus[i]) / (2 * hj);
  }
  return J;
}

/**
 * LU-factor a nested square matrix once, for repeated triangular solves.
 * Throws if the matrix is singular (a pivot below tolerance).
 *
 * @param {Array<Array<number>>} W - Nested n x n matrix
 * @returns {{L: Array<Array<number>>, U: Array<Array<number>>, perm: Int32Array}}
 */
function factorize(W) {
  const n = W.length;
  const { L, U, P } = lu(W);
  let maxU = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) maxU = Math.max(maxU, Math.abs(U[i][j]));
  }
  const perm = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    perm[i] = P[i].indexOf(1);
    if (!(Math.abs(U[i][i]) > 1e-13 * maxU)) {
      throw new Error('rosenbrock: W = I/(gamma*h) - J is singular');
    }
  }
  return { L, U, perm };
}

/**
 * Solve L U x = P b using a factorization from factorize().
 *
 * @param {{L: Array<Array<number>>, U: Array<Array<number>>, perm: Int32Array}} fac
 * @param {Float64Array} b - Right-hand side (not modified)
 * @returns {Float64Array}
 */
function luSolve(fac, b) {
  const { L, U, perm } = fac;
  const n = b.length;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = b[perm[i]];
  for (let i = 1; i < n; i++) {
    let s = x[i];
    for (let j = 0; j < i; j++) s -= L[i][j] * x[j];
    x[i] = s;
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= U[i][j] * x[j];
    x[i] = s / U[i][i];
  }
  return x;
}

/**
 * Integrate the stiff system y' = f(t, y) with an adaptive 4(3)
 * Rosenbrock-Wanner method (Kaps-Rentrop with Shampine's coefficients).
 *
 * @param {Function} f - (t, y) => dydt; y is Array<number>, returns Array<number> (or scalar)
 * @param {[number, number]} tSpan - [t0, tEnd] (tEnd may be < t0 for backward integration)
 * @param {number|Array<number>} y0 - Initial state
 * @param {Object} [options]
 * @param {number} [options.rtol=1e-6] - Relative tolerance
 * @param {number} [options.atol=1e-9] - Absolute tolerance
 * @param {Array<number>} [options.tEval] - Times at which to report the solution; dense output
 *   uses cubic Hermite interpolation between accepted steps
 * @param {number} [options.maxStep=Infinity] - Maximum step size
 * @param {number} [options.firstStep] - Initial step size (auto if omitted)
 * @param {number} [options.maxSteps=100000] - Safety cap on accepted+rejected steps
 * @param {Function} [options.jac] - (t, y) => nested n x n Jacobian df/dy; central finite
 *   differences are used when omitted
 * @returns {{t: number[], y: number[] | number[][], success: boolean, message: string, nfev: number, njev: number, nsteps: number}} Solver result
 */
export function rosenbrock(f, tSpan, y0, options = {}) {
  const rtol = options.rtol ?? 1e-6;
  const atol = options.atol ?? 1e-9;
  const maxStep = options.maxStep ?? Infinity;
  const maxSteps = options.maxSteps ?? 100000;

  if (options.events) {
    throw new Error('rosenbrock does not support events yet; use rk45 for event detection');
  }

  const [t0, tEnd] = tSpan;
  const direction = tEnd >= t0 ? 1 : -1;
  const { y: yInit, scalar } = normalizeState(y0);
  const n = yInit.length;
  const fn = wrapRhs(f, n);

  const userJac = options.jac
    ? (t, y) => {
        const J = options.jac(t, Array.from(y));
        if (!Array.isArray(J) || J.length !== n || J.some((row) => row.length !== n)) {
          throw new Error(`options.jac must return an ${n}x${n} nested matrix`);
        }
        return J.map((row) => Array.from(row, Number));
      }
    : null;

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
  let njev = 0;

  let h = options.firstStep !== undefined
    ? Math.abs(options.firstStep) * direction
    : initialStep(fn, t, y, fEval, ORDER, atol, rtol, direction);

  // Output collection
  const outT = [];
  const outY = [];
  let evalIdx = 0;
  if (!tEval) {
    outT.push(t);
    outY.push(Float64Array.from(y));
  }

  const SAFETY = 0.9;
  const MIN_FACTOR = 0.2;
  const MAX_FACTOR = 10;
  let errPrev = 1e-4;
  let nsteps = 0;

  // Jacobian and time derivative at the current point; recomputed lazily
  // after each accepted step, reused across rejected attempts.
  let J = null;
  let dfdt = null;

  /** Cubic Hermite interpolation across the accepted step, theta in [0, 1]. */
  function interpolate(yOld, fOld, yNew, fNew, hStep, theta) {
    const out = new Float64Array(n);
    const t2 = theta * theta;
    const t3 = t2 * theta;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + theta;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;
    for (let i = 0; i < n; i++) {
      out[i] = h00 * yOld[i] + h10 * hStep * fOld[i] + h01 * yNew[i] + h11 * hStep * fNew[i];
    }
    return out;
  }

  while (direction * (t - tEnd) < 0) {
    if (nsteps++ > maxSteps) {
      return finish(false, `exceeded maxSteps (${maxSteps})`);
    }

    const absH = Math.min(Math.abs(h), maxStep, Math.abs(tEnd - t));
    if (absH < 1e-14 * Math.max(1, Math.abs(t))) {
      return finish(false, 'step size underflow');
    }
    h = absH * direction;

    if (J === null) {
      if (userJac) {
        J = userJac(t, y);
      } else {
        J = fdJacobian(fn, t, y);
        nfev += 2 * n;
      }
      njev++;
      const dt = SQRT_EPS * Math.max(Math.abs(t), Math.abs(h)) * direction;
      const ft = fn(t + dt, y);
      nfev++;
      dfdt = new Float64Array(n);
      for (let i = 0; i < n; i++) dfdt[i] = (ft[i] - fEval[i]) / dt;
    }

    // Attempt one step: factor W = I/(gamma*h) - J once, solve 4 stages.
    let yNext = null;
    let err = null;
    try {
      const W = [];
      const diag = 1 / (GAMMA * h);
      for (let i = 0; i < n; i++) {
        const row = new Array(n);
        for (let j = 0; j < n; j++) row[j] = (i === j ? diag : 0) - J[i][j];
        W.push(row);
      }
      const fac = factorize(W);

      const rhs = new Float64Array(n);
      const ys = new Float64Array(n);

      for (let i = 0; i < n; i++) rhs[i] = fEval[i] + h * C1X * dfdt[i];
      const g1 = luSolve(fac, rhs);

      for (let i = 0; i < n; i++) ys[i] = y[i] + A21 * g1[i];
      const f2 = fn(t + A2X * h, ys);
      nfev++;
      for (let i = 0; i < n; i++) rhs[i] = f2[i] + h * C2X * dfdt[i] + (C21 * g1[i]) / h;
      const g2 = luSolve(fac, rhs);

      for (let i = 0; i < n; i++) ys[i] = y[i] + A31 * g1[i] + A32 * g2[i];
      const f3 = fn(t + A3X * h, ys);
      nfev++;
      for (let i = 0; i < n; i++) {
        rhs[i] = f3[i] + h * C3X * dfdt[i] + (C31 * g1[i] + C32 * g2[i]) / h;
      }
      const g3 = luSolve(fac, rhs);

      // Stage 4 reuses f3 (evaluated at the stage-3 point).
      for (let i = 0; i < n; i++) {
        rhs[i] = f3[i] + h * C4X * dfdt[i] + (C41 * g1[i] + C42 * g2[i] + C43 * g3[i]) / h;
      }
      const g4 = luSolve(fac, rhs);

      yNext = new Float64Array(n);
      err = new Float64Array(n);
      let finite = true;
      for (let i = 0; i < n; i++) {
        yNext[i] = y[i] + B[0] * g1[i] + B[1] * g2[i] + B[2] * g3[i] + B[3] * g4[i];
        err[i] = E[0] * g1[i] + E[1] * g2[i] + E[2] * g3[i] + E[3] * g4[i];
        finite = finite && Number.isFinite(yNext[i]) && Number.isFinite(err[i]);
      }
      if (!finite) yNext = null;
    } catch {
      // Singular W or a RHS failure inside the trial step: retry smaller.
      yNext = null;
    }

    if (yNext === null) {
      h = absH * 0.5 * direction;
      continue;
    }

    const errNorm = errorNorm(err, y, yNext, atol, rtol);

    if (errNorm <= 1) {
      // Accept. PI controller in the same style as rk45, with gains tuned
      // for the 3rd-order estimator (steady state near errNorm ~ 0.55).
      const tOld = t;
      const yOld = y;
      const fOld = fEval;
      const fNext = fn(tOld + h, yNext); // reused as f(t, y) of the next step
      nfev++;

      // Dense output for tEval points in [tOld, tOld + h]
      if (tEval) {
        while (evalIdx < tEval.length &&
               direction * (tEval[evalIdx] - (tOld + h)) <= 1e-12 &&
               direction * (tEval[evalIdx] - tOld) >= -1e-12) {
          const theta = h === 0 ? 0 : (tEval[evalIdx] - tOld) / h;
          outT.push(tEval[evalIdx]);
          outY.push(interpolate(yOld, fOld, yNext, fNext, h, theta));
          evalIdx++;
        }
      }

      t = tOld + h;
      y = yNext;
      fEval = fNext;
      J = null; // force a fresh Jacobian at the new point

      if (!tEval) {
        outT.push(t);
        outY.push(Float64Array.from(y));
      }

      const factor = errNorm === 0 ? MAX_FACTOR
        : Math.min(MAX_FACTOR,
            SAFETY * errNorm ** (-0.9 / (ORDER + 1)) * errPrev ** (0.2 / (ORDER + 1)));
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
    return makeResult(outT, outY, scalar, { success, message, nfev, njev, nsteps });
  }
}
