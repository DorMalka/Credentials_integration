#!/usr/bin/env python3
"""Fit measured FAR/FRR curves with Gaussian CDF/SF by least squares.

Input: an existing whitespace/comma-separated table with T FAR FRR columns.
Additional columns are ignored. No histogram reconstruction or KS fitting.

Example:
  python3 gaussian_least_squares_fit.py measurements.txt

Fits FRR(t)=norm.cdf(t,loc=mu,scale=sigma) and
FAR(t)=norm.sf(t,loc=mu,scale=sigma) independently. Every measured threshold
has equal weight. Prints metrics and copy-ready SciPy parameters; modifies
no input files. Requires NumPy and SciPy.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.optimize import least_squares


def load_measurements(path):
    with path.open(encoding="utf-8-sig") as handle:
        lines = [line.strip() for line in handle
                 if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError("The measurement file is empty.")
    columns = lines[0].replace(",", " ").split()
    if not {"T", "FAR", "FRR"}.issubset(columns):
        raise ValueError("Expected named T FAR FRR columns; additional columns are allowed.")
    rows = [line.replace(",", " ").split() for line in lines[1:]]
    if any(len(row) != len(columns) for row in rows):
        raise ValueError("Header and data column counts differ.")
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or len(values) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("At least three finite measurement rows are required.")
    t, far, frr = [values[:, columns.index(k)] for k in ("T", "FAR", "FRR")]
    order = np.argsort(t)
    t, far, frr = t[order], far[order], frr[order]
    if np.any(np.diff(t) <= 0):
        raise ValueError("Thresholds must be distinct.")
    if np.any((far < 0) | (far > 1)) or np.any((frr < 0) | (frr > 1)):
        raise ValueError("FAR/FRR measurements must be probabilities in [0,1].")
    return t, far, frr


def fit_gaussian(t, measured, kind, starts=36, seed=0):
    """Multi-start nonlinear least squares; return the lowest observed SSE."""
    t, measured = np.asarray(t, float), np.asarray(measured, float)
    if kind not in {"far", "frr"}:
        raise ValueError("kind must be far or frr")
    if t.ndim != 1 or t.shape != measured.shape or len(t) < 3:
        raise ValueError("Provide at least three paired one-dimensional measurements.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(measured)):
        raise ValueError("Measurements must be finite.")
    if np.any((measured < 0) | (measured > 1)):
        raise ValueError("Measured rates must lie in [0,1].")
    if np.ptp(t) == 0 or np.ptp(measured) < 1e-12:
        raise ValueError("A constant curve cannot identify both Gaussian parameters.")
    if starts < 1:
        raise ValueError("starts must be positive")
    origin, span = float(t.min()), float(np.ptp(t))
    z = (t - origin) / span
    model = stats.norm.sf if kind == "far" else stats.norm.cdf
    # Optimize a normalized mean and log standard deviation. Wide bounds
    # permit probability mass outside the observed threshold interval.
    lower, upper = np.array([-10., np.log(1e-5)]), np.array([11., np.log(100.)])
    def residual(p):
        return model(z, loc=p[0], scale=np.exp(p[1])) - measured
    cdf = 1 - measured if kind == "far" else measured
    center = float(z[np.argmin(np.abs(cdf - .5))])
    guesses = [[center, np.log(v)] for v in (.05, .1, .2, .5, 1.)]
    # A probit line provides a data-driven initial estimate where rates are
    # away from 0 and 1. The final objective remains untransformed rate SSE.
    interior = (cdf > .01) & (cdf < .99)
    if interior.sum() >= 2:
        slope, intercept = np.polyfit(z[interior], stats.norm.ppf(cdf[interior]), 1)
        if slope > 0:
            guesses.insert(0, [-intercept / slope, np.log(1 / slope)])
    rng = np.random.default_rng(seed)
    while len(guesses) < starts:
        guesses.append([rng.uniform(-1, 2), rng.uniform(np.log(.01), np.log(3))])
    candidates = []
    for guess in guesses[:starts]:
        result = least_squares(
            residual, np.clip(guess, lower + 1e-9, upper - 1e-9),
            bounds=(lower, upper), loss="linear", max_nfev=5000,
            ftol=1e-12, xtol=1e-12, gtol=1e-12,
        )
        if np.all(np.isfinite(result.fun)):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("No finite Gaussian fit was found.")
    best = min(candidates, key=lambda r: float(np.sum(r.fun ** 2)))
    mu, sigma = origin + span * best.x[0], span * np.exp(best.x[1])
    predicted = model(t, loc=mu, scale=sigma)
    errors = predicted - measured
    sse = float(np.sum(errors ** 2))
    sst = float(np.sum((measured - measured.mean()) ** 2))
    return {
        "mu": float(mu), "sigma": float(sigma), "sse": sse,
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "max_error": float(np.max(np.abs(errors))),
        "r2": float(1 - sse / sst),
        "converged": bool(best.success),
        "boundary": bool(np.any(np.isclose(best.x, lower, atol=1e-5))
                         or np.any(np.isclose(best.x, upper, atol=1e-5))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="existing table with T FAR FRR columns")
    parser.add_argument("--kind", choices=("both", "far", "frr"), default="both")
    parser.add_argument("--starts", type=int, default=36)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.starts < 1 or args.seed < 0:
        parser.error("--starts must be positive and --seed non-negative.")
    try:
        t, far, frr = load_measurements(args.file)
        print(f"Input: {args.file}\nMeasured thresholds: {len(t)}")
        for kind, measurements, label in [("far", far, "Impostor / FAR"),
                                           ("frr", frr, "Genuine / FRR")]:
            if args.kind not in ("both", kind):
                continue
            fit = fit_gaussian(t, measurements, kind, args.starts, args.seed)
            print(f"\n{label}")
            print(f"mu (loc) = {fit['mu']:.10g}")
            print(f"sigma (scale) = {fit['sigma']:.10g}")
            print(f"SSE = {fit['sse']:.10g}")
            print(f"RMSE = {fit['rmse']:.8f} ({100 * fit['rmse']:.4f} percentage points)")
            print(f"MAE = {fit['mae']:.8f}")
            print(f"Maximum absolute error = {fit['max_error']:.8f}")
            print(f"R^2 = {fit['r2']:.8f}; approximation score (100 x R^2) = {100 * fit['r2']:.2f}")
            fn = "sf" if kind == "far" else "cdf"
            print(f'"{kind.upper()}_normal": stats.norm.{fn}(t, '
                  f"loc={fit['mu']:.10g}, scale={fit['sigma']:.10g}),")
            if not fit['converged'] or fit['boundary']:
                print("Fit reached an iteration limit or search boundary; inspect the parameters.")
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print("\nRMSE is minimized; lower is better. R^2=1 is perfect, 0 matches the")
    print("constant-mean baseline, and negative is worse. The score is descriptive,")
    print("not a confidence level. Each supplied threshold has equal weight.")
    print("Multi-start fitting improves reliability but does not prove a global minimum.")


if __name__ == "__main__":
    main()
