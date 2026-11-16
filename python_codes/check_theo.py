#!/usr/bin/env python3
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

KS_UNIFORM_GENUINE  = (1.25, 97.5)
KS_UNIFORM_IMPOSTOR = (1.25, 77.5)
KS_QUADRATIC_GENUINE  = (4.0, 44.3902, 54.6395)
KS_QUADRATIC_IMPOSTOR = (4.0, 26.4433, 52.3973)

g = np.linspace(0, 100, 4000)

models = {
    "uniform": (stats.uniform.sf(g, *KS_UNIFORM_IMPOSTOR),
                stats.uniform.cdf(g, *KS_UNIFORM_GENUINE)),
    "quadratic": (stats.rdist.sf(g, *KS_QUADRATIC_IMPOSTOR),
                  stats.rdist.cdf(g, *KS_QUADRATIC_GENUINE)),
}

for name, (fa, fr) in models.items():
    print(f"\n--- {name} ---")
    for K in (2, 3, 8):
        eq = (1 - fr**K) * (1 - fa)**K
        best_eq = eq.max()
        if K == 2:
            S = (1 - fr[:, None]*fr[None, :]) * (1 - fa[:, None]) * (1 - fa[None, :])
            i, j = np.unravel_index(np.argmax(S), S.shape)
            print(f"k=2 global t1={g[i]:.3f} t2={g[j]:.3f} P={S[i,j]:.10f}  "
                  f"equal={best_eq:.10f}  gap={S[i,j]-best_eq:+.2e}")
        else:
            print(f"k={K} best equal={best_eq:.10f}")