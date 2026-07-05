#!/usr/bin/env python3
"""
Compare @tangent.to/ode against scipy.integrate.solve_ivp on non-stiff
(RK45) and stiff (Rosenbrock vs Radau) problems, sampled at shared time
points. Run from the package root:

    uv run --with scipy python3 tests_compare-to-scipy/compare_with_scipy.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "tests_compare-to-scipy" / "compare_ode.mjs"

FAILURES = []


def check(label, err, tol):
    ok = err < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  (err={err:.2e}, tol={tol:.0e})")
    if not ok:
        FAILURES.append(label)


def run_node(spec):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        path = fh.name
    r = subprocess.run(["node", str(NODE), path], check=True, capture_output=True, text=True, cwd=ROOT)
    return json.loads(r.stdout)


def rhs(system, mu=1.0):
    return {
        "exp": lambda t, y: [y[0]],
        "decay": lambda t, y: [-2 * y[0]],
        "oscillator": lambda t, y: [y[1], -y[0]],
        "lotka": lambda t, y: [1.5 * y[0] - y[0] * y[1], -3 * y[1] + y[0] * y[1]],
        "vanderpol": lambda t, y: [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]],
        "robertson": lambda t, y: [
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2,
        ],
    }[system]


def compare(label, system, y0, tspan, teval, method="rk45", scipy_method="RK45",
            mu=1.0, tol=1e-6):
    js = run_node({"system": system, "y0": y0, "tSpan": tspan, "tEval": teval,
                   "method": method, "mu": mu, "rtol": 1e-9, "atol": 1e-11})
    sol = solve_ivp(rhs(system, mu), tspan, y0, t_eval=teval, method=scipy_method,
                    rtol=1e-9, atol=1e-11)
    js_y = np.array(js["y"])
    err = float(np.max(np.abs(js_y - sol.y)))
    check(label, err, tol)


def main():
    print("scipy.integrate comparison for @tangent.to/ode")

    # --- Non-stiff RK45 vs scipy RK45 ---
    compare("exp growth", "exp", [1.0], [0, 5], list(np.linspace(0, 5, 11)))
    compare("decay", "decay", [1.0], [0, 5], list(np.linspace(0, 5, 11)))
    compare("harmonic oscillator", "oscillator", [1.0, 0.0], [0, 20],
            list(np.linspace(0, 20, 41)))
    compare("Lotka-Volterra", "lotka", [10.0, 5.0], [0, 15],
            list(np.linspace(0, 15, 61)), tol=1e-5)

    # --- Stiff Rosenbrock vs scipy Radau ---
    compare("Van der Pol (mu=100, stiff)", "vanderpol", [2.0, 0.0], [0, 300],
            list(np.linspace(0, 300, 61)), method="rosenbrock", scipy_method="Radau",
            mu=100, tol=5e-3)
    compare("Robertson (stiff kinetics)", "robertson", [1.0, 0.0, 0.0], [0, 1e3],
            [0, 1, 10, 100, 1000], method="rosenbrock", scipy_method="Radau", tol=1e-5)

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nAll comparisons passed.")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
