#!/usr/bin/env node
/**
 * Helper: integrate a named system with @tangent.to/ode at tEval points
 * supplied by the Python driver; print the solution as JSON.
 */

import { readFileSync } from 'node:fs';
import { rk45 } from '../src/rk45.js';
import { rosenbrock } from '../src/rosenbrock.js';

const spec = JSON.parse(readFileSync(process.argv[2], 'utf8'));

const SYSTEMS = {
  exp: (t, [y]) => [y],
  decay: (t, [y]) => [-2 * y],
  oscillator: (t, [y, v]) => [v, -y],
  lotka: (t, [x, y]) => [1.5 * x - x * y, -3 * y + x * y],
  vanderpol: (t, [y1, y2]) => [y2, spec.mu * (1 - y1 * y1) * y2 - y1],
  robertson: (t, [a, b, c]) => [
    -0.04 * a + 1e4 * b * c,
    0.04 * a - 1e4 * b * c - 3e7 * b * b,
    3e7 * b * b,
  ],
};

const f = SYSTEMS[spec.system];
const solver = spec.method === 'rosenbrock' ? rosenbrock : rk45;
const r = solver(f, spec.tSpan, spec.y0, {
  tEval: spec.tEval,
  rtol: spec.rtol ?? 1e-8,
  atol: spec.atol ?? 1e-10,
});

// Normalize y to component-major nested arrays even for scalar systems
const y = Array.isArray(r.y[0]) ? r.y : [r.y];
process.stdout.write(JSON.stringify({ t: r.t, y, success: r.success, nsteps: r.nsteps }));
