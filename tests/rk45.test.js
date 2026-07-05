import { describe, expect, it } from 'vitest';
import { rk45 } from '../src/rk45.js';

const last = (arr) => arr[arr.length - 1];

describe('rk45 (Dormand-Prince)', () => {
  it('integrates y\' = y to e', () => {
    const r = rk45((t, y) => y, [0, 1], 1, { rtol: 1e-9, atol: 1e-12 });
    expect(r.success).toBe(true);
    expect(last(r.y)).toBeCloseTo(Math.E, 8);
  });

  it('integrates the harmonic oscillator over one period', () => {
    // y'' = -y  ->  [y, v]' = [v, -y]; after 2pi returns to [1, 0]
    const r = rk45((t, [y, v]) => [v, -y], [0, 2 * Math.PI], [1, 0], { rtol: 1e-10, atol: 1e-12 });
    expect(last(r.y[0])).toBeCloseTo(1, 7);
    expect(last(r.y[1])).toBeCloseTo(0, 7);
  });

  it('matches the exact solution of a decaying system', () => {
    // y' = -2y  ->  y = e^{-2t}
    const r = rk45((t, y) => -2 * y, [0, 3], 1, { rtol: 1e-9, atol: 1e-12 });
    expect(last(r.y)).toBeCloseTo(Math.exp(-6), 8);
  });

  it('reports the solution exactly at tEval points via dense output', () => {
    const tEval = [0, 0.5, 1, 1.5, 2];
    const r = rk45((t, y) => y, [0, 2], 1, { tEval, rtol: 1e-10, atol: 1e-12 });
    expect(r.t).toEqual(tEval);
    for (let i = 0; i < tEval.length; i++) {
      expect(r.y[i]).toBeCloseTo(Math.exp(tEval[i]), 7);
    }
  });

  it('integrates backward in time', () => {
    const r = rk45((t, y) => y, [0, -1], 1, { rtol: 1e-9, atol: 1e-12 });
    expect(last(r.y)).toBeCloseTo(Math.exp(-1), 8);
  });

  it('solves Lotka-Volterra and conserves its invariant', () => {
    // x' = a x - b x y ; y' = -c y + d x y
    const [a, b, c, d] = [1.5, 1, 3, 1];
    const f = (t, [x, y]) => [a * x - b * x * y, -c * y + d * x * y];
    const r = rk45(f, [0, 15], [10, 5], { rtol: 1e-10, atol: 1e-12 });
    expect(r.success).toBe(true);
    // Conserved quantity V = d x - c ln x + b y - a ln y
    const V = (x, y) => d * x - c * Math.log(x) + b * y - a * Math.log(y);
    const V0 = V(10, 5);
    for (let k = 0; k < r.t.length; k++) {
      expect(V(r.y[0][k], r.y[1][k])).toBeCloseTo(V0, 4);
    }
    // populations stay positive
    expect(Math.min(...r.y[0])).toBeGreaterThan(0);
    expect(Math.min(...r.y[1])).toBeGreaterThan(0);
  });

  it('detects events by root-finding on g(t, y)', () => {
    // Oscillator crossing y = 0; events at t = pi/2, 3pi/2, ...
    const g = (t, [y]) => y;
    const r = rk45((t, [y, v]) => [v, -y], [0, 3 * Math.PI], [1, 0], {
      events: g, rtol: 1e-11, atol: 1e-13,
    });
    const times = r.events[0].t;
    expect(times.length).toBe(3);
    expect(times[0]).toBeCloseTo(Math.PI / 2, 6);
    expect(times[1]).toBeCloseTo(3 * Math.PI / 2, 6);
    expect(times[2]).toBeCloseTo(5 * Math.PI / 2, 6);
  });

  it('accepts a scalar y0 and returns a flat trajectory', () => {
    const r = rk45((t, y) => -y, [0, 1], 2, { rtol: 1e-9, atol: 1e-12 });
    expect(Array.isArray(r.y)).toBe(true);
    expect(typeof r.y[0]).toBe('number');
    expect(last(r.y)).toBeCloseTo(2 * Math.exp(-1), 8);
  });

  it('takes more steps for tighter tolerances', () => {
    const loose = rk45((t, y) => y, [0, 5], 1, { rtol: 1e-4, atol: 1e-7 });
    const tight = rk45((t, y) => y, [0, 5], 1, { rtol: 1e-10, atol: 1e-13 });
    expect(tight.nsteps).toBeGreaterThan(loose.nsteps);
    expect(last(loose.y)).toBeCloseTo(Math.exp(5), 2);
    expect(last(tight.y)).toBeCloseTo(Math.exp(5), 6);
  });

  it('validates inputs', () => {
    expect(() => rk45((t, y) => y, [0, 1], [])).toThrow(/non-empty/);
    expect(() => rk45((t, y) => [y[0], 0], [0, 1], 1)).toThrow(/must return 1/);
    expect(() => rk45((t, y) => y, [0, 1], 1, { tEval: [0, 2] })).toThrow(/outside tSpan/);
  });
});
