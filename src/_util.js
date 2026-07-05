/**
 * Shared helpers for the ODE solvers.
 *
 * The system is y' = f(t, y) with y a plain Array<number> (or a scalar,
 * accepted and normalized to a length-1 vector). f returns a new array;
 * solvers never mutate the user's arrays in place.
 */

/** Normalize y0 (number or array) to a Float64Array; record whether it was scalar. */
export function normalizeState(y0) {
  if (typeof y0 === 'number') {
    return { y: Float64Array.of(y0), scalar: true };
  }
  if (Array.isArray(y0) || ArrayBuffer.isView(y0)) {
    if (y0.length === 0) throw new Error('y0 must be a non-empty array or a number');
    return { y: Float64Array.from(y0), scalar: false };
  }
  throw new Error('y0 must be a number or an array of numbers');
}

/**
 * Wrap a user's f(t, y) so it always receives a plain Array and returns a
 * validated Float64Array of the right length.
 *
 * @param {Function} f - (t, y) => dydt
 * @param {number} n - System dimension
 * @returns {Function} (t, yTyped) => Float64Array
 */
export function wrapRhs(f, n) {
  return (t, yTyped) => {
    const dy = f(t, Array.from(yTyped));
    const out = typeof dy === 'number' ? [dy] : dy;
    if (!out || out.length !== n) {
      throw new Error(`f(t, y) must return ${n} value(s), got ${out?.length}`);
    }
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      r[i] = out[i];
      if (!Number.isFinite(r[i])) {
        throw new Error(`f(t, y) returned a non-finite value at component ${i} (t=${t})`);
      }
    }
    return r;
  };
}

/**
 * RMS error norm scaled by atol/rtol (Hairer-Norsett-Wanner).
 * ||e|| over sc_i = atol + rtol * max(|y_i|, |y_next_i|).
 *
 * @param {Float64Array} err - Error estimate
 * @param {Float64Array} y - Current state
 * @param {Float64Array} yNext - Proposed next state
 * @param {number} atol - Absolute tolerance
 * @param {number} rtol - Relative tolerance
 * @returns {number}
 */
export function errorNorm(err, y, yNext, atol, rtol) {
  const n = err.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const sc = atol + rtol * Math.max(Math.abs(y[i]), Math.abs(yNext[i]));
    const e = err[i] / sc;
    sum += e * e;
  }
  return Math.sqrt(sum / n);
}

/**
 * Initial step-size guess (Hairer, Norsett & Wanner, II.4).
 *
 * @param {Function} f - Wrapped RHS
 * @param {number} t0 - Start time
 * @param {Float64Array} y0 - Initial state
 * @param {Float64Array} f0 - f(t0, y0)
 * @param {number} order - Method order
 * @param {number} atol
 * @param {number} rtol
 * @param {number} direction - +1 or -1
 * @returns {number}
 */
export function initialStep(f, t0, y0, f0, order, atol, rtol, direction) {
  const n = y0.length;
  let d0 = 0;
  let d1 = 0;
  for (let i = 0; i < n; i++) {
    const sc = atol + rtol * Math.abs(y0[i]);
    d0 += (y0[i] / sc) ** 2;
    d1 += (f0[i] / sc) ** 2;
  }
  d0 = Math.sqrt(d0 / n);
  d1 = Math.sqrt(d1 / n);
  let h0 = d0 < 1e-5 || d1 < 1e-5 ? 1e-6 : 0.01 * (d0 / d1);

  const y1 = new Float64Array(n);
  for (let i = 0; i < n; i++) y1[i] = y0[i] + direction * h0 * f0[i];
  const f1 = f(t0 + direction * h0, y1);
  let d2 = 0;
  for (let i = 0; i < n; i++) {
    const sc = atol + rtol * Math.abs(y0[i]);
    d2 += ((f1[i] - f0[i]) / sc) ** 2;
  }
  d2 = Math.sqrt(d2 / n) / h0;

  const maxD = Math.max(d1, d2);
  const h1 = maxD <= 1e-15 ? Math.max(1e-6, h0 * 1e-3) : (0.01 / maxD) ** (1 / (order + 1));
  return direction * Math.min(100 * h0, h1);
}

/**
 * Assemble the solver's return object from collected steps.
 *
 * @param {Array<number>} ts - Time points
 * @param {Array<Float64Array>} ys - States at each time point
 * @param {boolean} scalar - Whether the caller supplied a scalar y0
 * @param {Object} extra - Additional fields (success, message, nfev, ...)
 * @returns {Object} {t, y, success, ...}
 */
export function makeResult(ts, ys, scalar, extra) {
  // y is returned component-major: y[i] is the trajectory of component i,
  // matching scipy's solve_ivp .y layout. For scalar systems, y is a flat
  // array of the single component's trajectory.
  const n = ys.length ? ys[0].length : 0;
  const y = [];
  for (let i = 0; i < n; i++) {
    y.push(ts.map((_, k) => ys[k][i]));
  }
  return {
    t: ts.slice(),
    y: scalar ? (y[0] ?? []) : y,
    ...extra,
  };
}
