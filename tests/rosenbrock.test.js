import { describe, expect, it } from 'vitest';
import { rosenbrock } from '../src/rosenbrock.js';

describe('rosenbrock', () => {
  it('solves a genuinely stiff tracking problem below the explicit stability floor', () => {
    // y' = -1000 (y - cos t): a fast transient onto the slow solution ~cos(t).
    // An explicit method is stability-limited to h < 2/1000 across the WHOLE
    // interval (> 2500 steps on [0, 5]) even though the solution is smooth.
    // An A-stable solver follows cos(t) with accuracy-limited steps instead.
    const f = (t, y) => [-1000 * (y[0] - Math.cos(t))];
    const res = rosenbrock(f, [0, 5], 0, { rtol: 1e-6, atol: 1e-8 });
    expect(res.success).toBe(true);
    expect(res.nsteps).toBeLessThan(2500); // below the explicit stability floor
    // Tracks cos(t) after the transient (phase lag is O(1/1000))
    for (const tq of [1, 2, 3, 4, 5]) {
      let best = 0;
      let bd = Infinity;
      res.t.forEach((t, k) => {
        if (Math.abs(t - tq) < bd) { bd = Math.abs(t - tq); best = res.y[k]; }
      });
      expect(Math.abs(best - Math.cos(tq))).toBeLessThan(5e-3);
    }
  });

  it('handles the Van der Pol oscillator in the stiff regime (mu = 100)', () => {
    const mu = 100;
    const f = (t, [y1, y2]) => [y2, mu * (1 - y1 * y1) * y2 - y1];
    const res = rosenbrock(f, [0, 200], [2, 0]);
    expect(res.success).toBe(true);
    expect(res.nsteps).toBeLessThan(5000);
    const y1 = res.y[0];
    const lo = y1.reduce((a, v) => Math.min(a, v), Infinity);
    const hi = y1.reduce((a, v) => Math.max(a, v), -Infinity);
    expect(lo).toBeGreaterThanOrEqual(-2.5);
    expect(hi).toBeLessThanOrEqual(2.5);
  });

  it('solves the Robertson chemical kinetics problem', () => {
    const f = (t, [y1, y2, y3]) => [
      -0.04 * y1 + 1e4 * y2 * y3,
      0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 * y2,
      3e7 * y2 * y2,
    ];
    const res = rosenbrock(f, [0, 1e4], [1, 0, 0], { atol: 1e-10 });
    expect(res.success).toBe(true);
    const last = res.t.length - 1;
    const mass = res.y[0][last] + res.y[1][last] + res.y[2][last];
    expect(Math.abs(mass - 1)).toBeLessThan(1e-4);
    // y1 is monotonically decreasing
    for (let k = 1; k < res.t.length; k++) {
      expect(res.y[0][k]).toBeLessThanOrEqual(res.y[0][k - 1] + 1e-10);
    }
  });

  it('solves a non-stiff sanity case y\' = y to high accuracy', () => {
    const res = rosenbrock((t, y) => y[0], [0, 2], 1, { rtol: 1e-8 });
    expect(res.success).toBe(true);
    const yEnd = res.y[res.y.length - 1];
    expect(Math.abs(yEnd / Math.exp(2) - 1)).toBeLessThan(1e-5);
  });

  it('matches finite differences when an analytic Jacobian is supplied', () => {
    // Stiff linear system with eigenvalues -1 and -1000.
    const A = [[-2, 1], [998, -999]];
    const f = (t, [u, v]) => [A[0][0] * u + A[0][1] * v, A[1][0] * u + A[1][1] * v];
    const jac = () => A;
    const opts = { rtol: 1e-8, atol: 1e-12 };
    const fd = rosenbrock(f, [0, 1], [1, 0], opts);
    const an = rosenbrock(f, [0, 1], [1, 0], { ...opts, jac });
    expect(fd.success).toBe(true);
    expect(an.success).toBe(true);
    expect(an.njev).toBeGreaterThan(0);
    const lastFd = fd.t.length - 1;
    const lastAn = an.t.length - 1;
    for (let i = 0; i < 2; i++) {
      expect(Math.abs(fd.y[i][lastFd] - an.y[i][lastAn])).toBeLessThan(1e-6);
    }
  });

  it('returns exactly the requested tEval points', () => {
    const tEval = [0, 0.5, 1, 1.5, 2];
    const res = rosenbrock((t, y) => -y[0], [0, 2], 1, { tEval });
    expect(res.success).toBe(true);
    expect(res.t).toEqual(tEval);
    expect(res.y.length).toBe(tEval.length);
  });

  it('throws when events are requested', () => {
    expect(() => rosenbrock((t, y) => -y[0], [0, 1], 1, { events: () => 1 }))
      .toThrow(/events/);
  });
});
