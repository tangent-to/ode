import { describe, expect, it } from 'vitest';
import { euler, rk2, rk4 } from '../src/fixed.js';

/** Endpoint value of a scalar solve (y is a flat array for scalar y0). */
function endpoint(res) {
  return res.y[res.y.length - 1];
}

/** Error at t=1 for y' = y, y0 = 1 (exact answer e), with a given step. */
function expError(method, h) {
  const res = method((t, y) => [y[0]], [0, 1], 1, { step: h });
  expect(res.success).toBe(true);
  return Math.abs(endpoint(res) - Math.E);
}

describe('convergence order', () => {
  const cases = [
    ['euler', euler, 1],
    ['rk2', rk2, 2],
    ['rk4', rk4, 4],
  ];
  for (const [name, method, order] of cases) {
    it(`${name} converges at order ${order}`, () => {
      const h = 0.02;
      const e1 = expError(method, h);
      const e2 = expError(method, h / 2);
      const observed = Math.log2(e1 / e2);
      expect(Math.abs(observed - order)).toBeLessThan(0.3);
    });
  }
});

describe('rk4 accuracy', () => {
  it('hits e to 1e-10 on y\' = y with step 1e-3', () => {
    const res = rk4((t, y) => [y[0]], [0, 1], 1, { step: 1e-3 });
    expect(res.success).toBe(true);
    expect(endpoint(res)).toBeCloseTo(Math.E, 10);
    expect(Math.abs(endpoint(res) - Math.E)).toBeLessThan(1e-10);
  });
});

describe('2D harmonic oscillator', () => {
  it('returns to [1, 0] after one period and conserves energy', () => {
    const f = (t, y) => [y[1], -y[0]];
    const res = rk4(f, [0, 2 * Math.PI], [1, 0], { step: 1e-3 });
    expect(res.success).toBe(true);
    const nPts = res.t.length;
    expect(Math.abs(res.y[0][nPts - 1] - 1)).toBeLessThan(1e-6);
    expect(Math.abs(res.y[1][nPts - 1] - 0)).toBeLessThan(1e-6);
    for (let k = 0; k < nPts; k++) {
      const energy = res.y[0][k] ** 2 + res.y[1][k] ** 2;
      expect(Math.abs(energy - 1)).toBeLessThan(1e-6);
    }
  });
});

describe('negative direction', () => {
  it('integrates y\' = y from t=0 to t=-1 to reach e^-1', () => {
    const res = rk4((t, y) => [y[0]], [0, -1], 1, { step: 1e-3 });
    expect(res.success).toBe(true);
    expect(res.t[res.t.length - 1]).toBe(-1);
    expect(endpoint(res)).toBeCloseTo(Math.exp(-1), 10);
  });
});

describe('nSteps option', () => {
  it('produces exactly nSteps+1 time points, last exactly tEnd', () => {
    for (const method of [euler, rk2, rk4]) {
      const res = method((t, y) => [y[0]], [0, 1], 1, { nSteps: 7 });
      expect(res.success).toBe(true);
      expect(res.t.length).toBe(8);
      expect(res.nsteps).toBe(7);
      expect(res.t[7]).toBe(1);
    }
  });

  it('step wins when both step and nSteps are given', () => {
    const res = euler((t, y) => [y[0]], [0, 1], 1, { step: 0.5, nSteps: 100 });
    expect(res.t.length).toBe(3);
  });
});

describe('non-finite RHS', () => {
  it('returns success:false instead of throwing', () => {
    for (const method of [euler, rk2, rk4]) {
      const res = method(() => [NaN], [0, 1], [1], { step: 0.1 });
      expect(res.success).toBe(false);
      expect(res.message).toMatch(/non-finite/);
    }
  });
});

describe('scalar vs array y0', () => {
  it('scalar y0 returns a flat y array', () => {
    const res = euler((t, y) => [-y[0]], [0, 1], 1, { nSteps: 10 });
    expect(res.success).toBe(true);
    expect(typeof res.y[0]).toBe('number');
    expect(res.y.length).toBe(res.t.length);
  });

  it('array y0 returns component-major y', () => {
    const res = euler((t, y) => [-y[0]], [0, 1], [1], { nSteps: 10 });
    expect(res.success).toBe(true);
    expect(Array.isArray(res.y[0])).toBe(true);
    expect(res.y.length).toBe(1);
    expect(res.y[0].length).toBe(res.t.length);
  });

  it('scalar and array y0 give the same trajectory', () => {
    const fS = (t, y) => [-y[0]];
    const a = rk2(fS, [0, 1], 1, { nSteps: 20 });
    const b = rk2(fS, [0, 1], [1], { nSteps: 20 });
    expect(a.y).toEqual(b.y[0]);
  });
});
