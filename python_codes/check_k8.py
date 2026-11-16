#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

REPO_ROOT = Path("/Users/dormalka/Desktop/Dor/Paper")
SPOOF = REPO_ROOT / "figs" / "fig_spoofed_users"
INPUTS = SPOOF / "psafe_sweep_inputs.npz"
RESULTS = SPOOF / "livdet_fixed_prefix_far_frr_success_by_k.txt"

with np.load(INPUTS, allow_pickle=False) as d:
    thr = np.asarray(d["thresholds"], dtype=float)
    fars = np.asarray(d["fars"], dtype=float)
    frrs = np.asarray(d["frrs"], dtype=float)

table = np.loadtxt(RESULTS, skiprows=1)
reported = {int(row[0]): float(row[3]) for row in table}

edges = np.linspace(thr[0], thr[-1], 400)

print(f"{'k':>3} {'best equal':>14} {'joint optimum':>14} {'gap':>14}")
print("-" * 50)
for K in sorted(reported):
    def neg(t):
        far = np.interp(t, thr, fars)
        frr = np.interp(t, thr, frrs)
        return -(1 - frr**K) * (1 - far)**K

    best = max(
        -minimize_scalar(neg, bounds=(lo, hi), method="bounded").fun
        for lo, hi in zip(edges[:-1], edges[1:])
    )
    gap = reported[K] - best
    print(f"{K:>3} {best:>14.10f} {reported[K]:>14.10f} {gap:>+14.10f}")