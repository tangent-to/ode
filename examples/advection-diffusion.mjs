/**
 * 1-D solute transport by the method of lines.
 *
 * The advection-diffusion equation
 *   dc/dt = D d^2c/dx^2 - v dc/dx
 * is discretized in space on a uniform grid (central differences for both
 * derivatives), turning the PDE into a system of ODEs c_i'(t) — one per
 * grid node — which any tangent/ode solver integrates in time. This is the
 * pattern for solute transport, heat conduction, and reaction-diffusion in
 * general: discretize space yourself, hand the resulting stiff ODE system
 * to `rosenbrock` (diffusion makes it stiff on fine grids).
 *
 * Boundary conditions here: fixed concentration c=1 at the inlet (x=0),
 * zero-gradient (outflow) at the outlet.
 */

import { rosenbrock } from '../src/rosenbrock.js';

const D = 0.1; // diffusion coefficient
const v = 1.0; // advection velocity
const L = 10; // domain length
const N = 101; // grid nodes
const dx = L / (N - 1);

/** Right-hand side: dc/dt at every interior node. */
function transport(t, c) {
  const dc = new Array(N).fill(0);
  // Inlet: fixed concentration (Dirichlet) -> derivative pinned to 0
  dc[0] = 0;
  for (let i = 1; i < N - 1; i++) {
    const diffusion = D * (c[i + 1] - 2 * c[i] + c[i - 1]) / (dx * dx);
    const advection = -v * (c[i + 1] - c[i - 1]) / (2 * dx);
    dc[i] = diffusion + advection;
  }
  // Outlet: zero-gradient (c[N-1] = c[N-2]) -> upwind advection, no diffusion
  dc[N - 1] = -v * (c[N - 1] - c[N - 2]) / dx;
  return dc;
}

const c0 = new Array(N).fill(0);
c0[0] = 1; // inlet pulse present from t=0

const t0 = performance.now?.() ?? 0;
const sol = rosenbrock(transport, [0, 8], c0, {
  rtol: 1e-6,
  atol: 1e-9,
  tEval: [0, 2, 4, 6, 8],
});

console.log(`Method-of-lines advection-diffusion: ${N} nodes, stiff solver`);
console.log(`success=${sol.success} steps=${sol.nsteps} nfev=${sol.nfev} njev=${sol.njev}`);
console.log('\nConcentration front position (x where c crosses 0.5):');
for (let k = 0; k < sol.t.length; k++) {
  // sol.y is component-major: sol.y[i][k] = c at node i, time k
  let xFront = L;
  for (let i = 0; i < N - 1; i++) {
    const ci = sol.y[i][k];
    const cip = sol.y[i + 1][k];
    if (ci >= 0.5 && cip < 0.5) {
      xFront = i * dx + dx * (ci - 0.5) / (ci - cip);
      break;
    }
  }
  // Analytic front travels at advection speed v (≈ v*t), spread by diffusion
  console.log(`  t=${sol.t[k]}: front at x≈${xFront.toFixed(2)} (pure advection: ${(v * sol.t[k]).toFixed(2)})`);
}
