# tangent/ode

ODE integration for JavaScript (ESM). Browser-first, runs in Node.js and
Deno. The differential-equations leaf of the
[tangent suite](https://github.com/tangent-to) — MIT-licensed.

- **`rk45`** — adaptive Dormand-Prince (scipy's RK45 / MATLAB's ode45):
  5th order with embedded error control, 4th-order dense output, and
  bracketed event detection
- **`rosenbrock`** — adaptive, A-stable solver for **stiff** systems
  (reaction kinetics, diffusion, fast-slow dynamics), built on
  [lina](https://github.com/tangent-to/lina) for the linear solves
- **`euler` / `rk2` / `rk4`** — classic fixed-step integrators
- **`solve`** — one entry point dispatching by method name

Systems are `y' = f(t, y)` with plain array (or scalar) state — no matrix
types to learn.

## Install

```bash
npm install @tangent.to/ode     # npm
deno add jsr:@tangent/ode       # Deno / JSR
```

## Usage

```javascript
import { rk45, solve } from '@tangent.to/ode';

// Lotka-Volterra predator-prey
const [a, b, c, d] = [1.5, 1, 3, 1];
const sol = rk45(
  (t, [x, y]) => [a * x - b * x * y, -c * y + d * x * y],
  [0, 15], [10, 5],
  { tEval: Array.from({ length: 151 }, (_, i) => i * 0.1) },
);
sol.t;        // time points
sol.y[0];     // prey trajectory (component-major, like scipy's .y)
sol.y[1];     // predator trajectory

// Stiff system -> switch method, same interface
solve(robertsonKinetics, [0, 1e4], [1, 0, 0], { method: 'rosenbrock' });
```

### Events

```javascript
// Find every time the pendulum passes through vertical
const sol = rk45(pendulum, [0, 20], [Math.PI / 2, 0], {
  events: (t, [theta]) => theta,   // roots of g mark events
});
sol.events[0].t;   // event times, located by bisection on the interpolant
```

### PDEs by the method of lines

There is no PDE solver — the browser-scale answer is to discretize space
yourself and integrate the resulting ODE system.
`examples/advection-diffusion.mjs` solves 1-D solute transport
(`dc/dt = D c_xx - v c_x`) on a 101-node grid this way, handing the stiff
system to `rosenbrock`.

## Validation against scipy

`tests_compare-to-scipy/` integrates shared problems with both
tangent/ode and `scipy.integrate.solve_ivp`, sampled at identical time
points: non-stiff cases (exp, decay, oscillator, Lotka-Volterra) against
scipy's RK45, stiff cases (Van der Pol mu=100, Robertson kinetics)
against scipy's Radau. Requires [uv](https://docs.astral.sh/uv/) and Node:

```bash
npm run test:scipy
```

## Scope

Explicit non-stiff and Rosenbrock stiff integration for first-order
systems, double precision, sized for the suite's modeling workloads
(systems dynamics, transport, kinetics, ecology). Out of scope for now:
implicit multistep (BDF) beyond Rosenbrock, DAEs, boundary-value problems,
symplectic integrators.

## License

MIT.
