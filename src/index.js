/**
 * @tangent.to/ode - ODE integration for JavaScript (ESM)
 *
 * Adaptive Dormand-Prince RK45 (non-stiff, with dense output and event
 * detection), an adaptive Rosenbrock method for stiff systems, and the
 * classic fixed-step integrators. Systems are y' = f(t, y) with plain
 * array (or scalar) state. scipy.integrate-validated. MIT leaf of the
 * tangent suite.
 *
 * PDEs are handled by the method of lines: discretize space yourself and
 * hand the resulting ODE system to a solver (see examples/).
 */

export { rk45 } from './rk45.js';
export { rosenbrock } from './rosenbrock.js';
export { euler, rk2, rk4 } from './fixed.js';

import { rk45 } from './rk45.js';
import { rosenbrock } from './rosenbrock.js';
import { euler, rk2, rk4 } from './fixed.js';

/**
 * Solve an initial value problem, dispatching by method name (scipy
 * solve_ivp style). Defaults to adaptive RK45.
 *
 * @param {Function} f - (t, y) => dydt
 * @param {[number, number]} tSpan - [t0, tEnd]
 * @param {number|Array<number>} y0 - Initial state
 * @param {Object} [options] - {method, ...solver options}
 * @param {string} [options.method='rk45'] - 'rk45' | 'rosenbrock' | 'euler' | 'rk2' | 'rk4'
 * @returns {{t: number[], y: number[] | number[][], success: boolean, message: string, nfev: number, nsteps: number, njev?: number, events?: Array<{t: number[], y: number[] | number[][]}>}} Solver result (fields depend on the chosen method)
 */
export function solve(f, tSpan, y0, options = {}) {
  const { method = 'rk45', ...opts } = options;
  const solvers = { rk45, rosenbrock, euler, rk2, rk4 };
  const solver = solvers[String(method).toLowerCase()];
  if (!solver) {
    throw new Error(`solve: unknown method '${method}'. Available: ${Object.keys(solvers).join(', ')}`);
  }
  return solver(f, tSpan, y0, opts);
}

/**
 * Convenience bundle of every solver under one object, so consumers can
 * `import ode from '@tangent.to/ode'` and call `ode.solve(...)`, `ode.rk45(...)`, etc.
 * @type {{solve: typeof solve, rk45: typeof rk45, rosenbrock: typeof rosenbrock, euler: typeof euler, rk2: typeof rk2, rk4: typeof rk4}}
 */
export default { solve, rk45, rosenbrock, euler, rk2, rk4 };
