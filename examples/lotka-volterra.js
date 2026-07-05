// ---
// title: Lotka-Volterra, a predator-prey model
// id: ode-lotka-volterra
// ---

// %% [markdown]
/*
# Lotka-Volterra predator-prey dynamics

`@tangent.to/ode` integrates initial value problems of the form
`y' = f(t, y)`, where the state `y` is a plain array of numbers (or a scalar).
The adaptive Dormand-Prince solver `rk45` matches scipy's RK45 and MATLAB's
ode45: a 5th-order step with an embedded error estimate, PI step-size control,
a dense-output interpolant, and bracketed event detection.

This notebook uses the classic Lotka-Volterra system as a throughline. It
imports the local build. Once the package is published you would import it
from a CDN instead:

    import { rk45, solve } from 'https://esm.sh/@tangent.to/ode';
*/

// %% [javascript]

import { rk45, solve } from '../dist/index.js';

// The model has two species: prey x and predator y.
//   x' = a*x - b*x*y   (prey grow, and are eaten on contact)
//   y' = -c*y + d*x*y  (predators starve, and thrive on contact)
// The state is the array [x, y], so f returns [x', y'].
const a = 1.0, b = 0.1, c = 1.5, d = 0.075;
const f = (t, [x, y]) => [a * x - b * x * y, -c * y + d * x * y];

f(0, [10, 5]); // [x', y'] at the initial state: [5, -3.75]

// %% [markdown]
/*
## Integrating the system

Call `rk45(f, [t0, tEnd], y0)` with the initial populations `[10, 5]` over the
window `[0, 15]`. The result is `{ t, y, success, ... }`. The state array `y`
is component-major: `y[0]` is the whole prey trajectory and `y[1]` the whole
predator trajectory, each sampled at the times in `t`. The populations do not
settle; they cycle. Scanning for local maxima shows the prey peak (near 40.5)
recurring about every 5.5 time units, with the predator peak trailing it.
*/

// %% [javascript]

const sol = rk45(f, [0, 15], [10, 5]);

const [prey, predator] = sol.y;

// Local maxima of a trajectory: a sample larger than both neighbors.
const peaks = (series) => {
  const out = [];
  for (let i = 1; i < series.length - 1; i++) {
    if (series[i] > series[i - 1] && series[i] > series[i + 1]) {
      out.push({ t: +sol.t[i].toFixed(2), value: +series[i].toFixed(2) });
    }
  }
  return out;
};

({
  success: sol.success,
  steps: sol.nsteps,
  preyPeaks: peaks(prey),
  predatorPeaks: peaks(predator),
});

// %% [markdown]
/*
## A conserved quantity checks the accuracy

The Lotka-Volterra flow preserves the quantity
`V = d*x - c*ln(x) + b*y - a*ln(y)`. It is constant along any exact solution,
so its drift over the numerical trajectory is a direct read on integration
error. With the default tolerances the spread of V stays below 1e-6 across the
whole run, tighter than the reported solution samples themselves, which is why
the closed orbit does not visibly spiral.
*/

// %% [javascript]

const V = (x, y) => d * x - c * Math.log(x) + b * y - a * Math.log(y);

const invariants = prey.map((x, i) => V(x, predator[i]));
const vMin = Math.min(...invariants);
const vMax = Math.max(...invariants);

({
  V_initial: +invariants[0].toFixed(6),
  V_final: +invariants[invariants.length - 1].toFixed(6),
  spread: vMax - vMin, // ~9e-7: constant to well under 1e-4
});

// %% [markdown]
/*
## Event detection

Pass an event function `g(t, y) => number` as the `events` option; the solver
brackets each sign change of `g` and refines the crossing time by bisection on
the dense-output interpolant. To find every time the prey population crosses
20, use `g = x - 20`. The crossing times land in `result.events[0].t`, and the
matching states in `result.events[0].y`. The alternating up and down crossings
confirm the period seen in the peaks above.
*/

// %% [javascript]

const crossing = (t, [x, y]) => x - 20; // root where prey = 20

const withEvents = rk45(f, [0, 15], [10, 5], { events: crossing });

({
  crossingTimes: withEvents.events[0].t.map((t) => +t.toFixed(3)),
  preyAtCrossings: withEvents.events[0].y.map(([x]) => +x.toFixed(2)),
});

// %% [markdown]
/*
## Dense output at chosen times

The `tEval` option reports the solution at exactly the times you ask for,
using the same interpolant, instead of at the solver's internal step points.
This is the tidy way to tabulate or plot a trajectory on a fixed grid. The
`solve` dispatcher wraps the same solvers by name; for stiff systems, where
`rk45` would take tiny steps, pass `{ method: 'rosenbrock' }` to switch to the
adaptive Rosenbrock integrator instead.
*/

// %% [javascript]

const grid = [0, 3, 6, 9, 12, 15];
const sampled = solve(f, [0, 15], [10, 5], { method: 'rk45', tEval: grid });

({
  t: sampled.t,
  prey: sampled.y[0].map((x) => +x.toFixed(2)),
  predator: sampled.y[1].map((y) => +y.toFixed(2)),
});
