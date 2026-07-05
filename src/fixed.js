/**
 * Fixed-step explicit integrators: forward Euler, midpoint (RK2), and the
 * classic 4th-order Runge-Kutta. No adaptivity, no dense output — useful for
 * teaching, for systems-dynamics models that want a fixed reporting grid, and
 * as simple baselines against the adaptive solvers.
 */

import { makeResult, normalizeState, wrapRhs } from './_util.js';

/**
 * Shared fixed-step driver. Walks from t0 to tEnd in steps of size `step`
 * (or tSpan/nSteps), clamping the last step to land exactly on tEnd, and
 * records the state at every step endpoint.
 *
 * @param {Function} stepper - (fn, t, y, h, n) => [yNext, stageEvals]
 * @param {Function} f - User RHS
 * @param {[number, number]} tSpan - [t0, tEnd] (tEnd may be < t0)
 * @param {number|Array<number>} y0 - Initial state
 * @param {Object} options - {step} or {nSteps}
 * @returns {Object} {t, y, success, message, nfev, nsteps}
 */
function integrate(stepper, f, tSpan, y0, options = {}) {
  const [t0, tEnd] = tSpan;
  const direction = tEnd >= t0 ? 1 : -1;

  let h;
  if (options.step !== undefined) {
    if (!(options.step > 0)) throw new Error('options.step must be a positive number');
    h = options.step;
  } else if (options.nSteps !== undefined) {
    if (!Number.isInteger(options.nSteps) || options.nSteps < 1) {
      throw new Error('options.nSteps must be a positive integer');
    }
    h = Math.abs(tEnd - t0) / options.nSteps;
  } else {
    throw new Error('either options.step or options.nSteps is required');
  }

  const { y: yInit, scalar } = normalizeState(y0);
  const n = yInit.length;
  const fn = wrapRhs(f, n);

  let t = t0;
  let y = Float64Array.from(yInit);
  let nfev = 0;
  let nsteps = 0;

  const outT = [t];
  const outY = [Float64Array.from(y)];

  try {
    while (direction * (tEnd - t) > 0) {
      // Clamp the last step to land exactly on tEnd. The relative slack keeps
      // floating-point drift in t from producing a spurious extra tiny step.
      const remaining = Math.abs(tEnd - t);
      const last = remaining <= h * (1 + 1e-9);
      const absH = last ? remaining : h;
      const [yNext, evals] = stepper(fn, t, y, absH * direction, n);
      nfev += evals;
      nsteps++;
      t = last ? tEnd : t + absH * direction;
      y = yNext;
      outT.push(t);
      outY.push(Float64Array.from(y));
    }
  } catch (err) {
    return makeResult(outT, outY, scalar, {
      success: false,
      message: err.message,
      nfev,
      nsteps,
    });
  }

  return makeResult(outT, outY, scalar, {
    success: true,
    message: 'integration successful',
    nfev,
    nsteps,
  });
}

function eulerStep(fn, t, y, h, n) {
  const k1 = fn(t, y);
  const yNext = new Float64Array(n);
  for (let i = 0; i < n; i++) yNext[i] = y[i] + h * k1[i];
  return [yNext, 1];
}

function rk2Step(fn, t, y, h, n) {
  const k1 = fn(t, y);
  const yMid = new Float64Array(n);
  for (let i = 0; i < n; i++) yMid[i] = y[i] + 0.5 * h * k1[i];
  const k2 = fn(t + 0.5 * h, yMid);
  const yNext = new Float64Array(n);
  for (let i = 0; i < n; i++) yNext[i] = y[i] + h * k2[i];
  return [yNext, 2];
}

function rk4Step(fn, t, y, h, n) {
  const k1 = fn(t, y);
  const y2 = new Float64Array(n);
  for (let i = 0; i < n; i++) y2[i] = y[i] + 0.5 * h * k1[i];
  const k2 = fn(t + 0.5 * h, y2);
  const y3 = new Float64Array(n);
  for (let i = 0; i < n; i++) y3[i] = y[i] + 0.5 * h * k2[i];
  const k3 = fn(t + 0.5 * h, y3);
  const y4 = new Float64Array(n);
  for (let i = 0; i < n; i++) y4[i] = y[i] + h * k3[i];
  const k4 = fn(t + h, y4);
  const yNext = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    yNext[i] = y[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  }
  return [yNext, 4];
}

/**
 * Integrate y' = f(t, y) with the forward Euler method (1st order).
 *
 * @param {Function} f - (t, y) => dydt; y is Array<number>, returns Array<number> (or scalar)
 * @param {[number, number]} tSpan - [t0, tEnd] (tEnd may be < t0 for backward integration)
 * @param {number|Array<number>} y0 - Initial state
 * @param {Object} options
 * @param {number} [options.step] - Fixed step size h > 0 (required unless nSteps given; wins if both)
 * @param {number} [options.nSteps] - Number of equal steps across tSpan
 * @returns {Object} {t, y, success, message, nfev, nsteps}
 */
export function euler(f, tSpan, y0, options) {
  return integrate(eulerStep, f, tSpan, y0, options);
}

/**
 * Integrate y' = f(t, y) with the explicit midpoint method (2nd order).
 *
 * @param {Function} f - (t, y) => dydt; y is Array<number>, returns Array<number> (or scalar)
 * @param {[number, number]} tSpan - [t0, tEnd] (tEnd may be < t0 for backward integration)
 * @param {number|Array<number>} y0 - Initial state
 * @param {Object} options
 * @param {number} [options.step] - Fixed step size h > 0 (required unless nSteps given; wins if both)
 * @param {number} [options.nSteps] - Number of equal steps across tSpan
 * @returns {Object} {t, y, success, message, nfev, nsteps}
 */
export function rk2(f, tSpan, y0, options) {
  return integrate(rk2Step, f, tSpan, y0, options);
}

/**
 * Integrate y' = f(t, y) with the classic 4th-order Runge-Kutta method.
 *
 * @param {Function} f - (t, y) => dydt; y is Array<number>, returns Array<number> (or scalar)
 * @param {[number, number]} tSpan - [t0, tEnd] (tEnd may be < t0 for backward integration)
 * @param {number|Array<number>} y0 - Initial state
 * @param {Object} options
 * @param {number} [options.step] - Fixed step size h > 0 (required unless nSteps given; wins if both)
 * @param {number} [options.nSteps] - Number of equal steps across tSpan
 * @returns {Object} {t, y, success, message, nfev, nsteps}
 */
export function rk4(f, tSpan, y0, options) {
  return integrate(rk4Step, f, tSpan, y0, options);
}
