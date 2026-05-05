from pathlib import Path
import numpy as np
from scipy.stats import norm

# =========================
# Config
# =========================
OUTPUT_DIR = Path("/Users/dormalka/Desktop/Dor/Paper/figs/fig_2continuous")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "gaussian_far_frr_data.txt"

# Asymmetric Gaussian distributions
# Attacker: lower scores, wider variance
mu_a = 0.35
sigma_a = 0.16

# User: higher scores, narrower variance
mu_u = 0.68
sigma_u = 0.10

# Chosen threshold
T = 0.52

# Plot range
x_min = 0.0
x_max = 1.0
n_points = 1000

x = np.linspace(x_min, x_max, n_points)

# PDFs
f_a = norm.pdf(x, loc=mu_a, scale=sigma_a)
f_u = norm.pdf(x, loc=mu_u, scale=sigma_u)

# Shaded regions:
# FAR = attacker accepted = attacker score >= T
far_area = np.where(x >= T, f_a, 0.0)

# FRR = user rejected = user score < T
frr_area = np.where(x <= T, f_u, 0.0)

# Save data for TikZ/pgfplots
with open(OUTPUT_FILE, "w") as f:
    f.write("x user attacker FAR FRR threshold\n")
    for xi, ui, ai, fari, frri in zip(x, f_u, f_a, far_area, frr_area):
        f.write(f"{xi:.4f} {ui:.4f} {ai:.4f} {fari:.4f} {frri:.4f} {T:.4f}\n")

print(f"[i] Saved TikZ data to: {OUTPUT_FILE}")
print(f"[i] FAR = {1 - norm.cdf(T, loc=mu_a, scale=sigma_a):.4f}")
print(f"[i] FRR = {norm.cdf(T, loc=mu_u, scale=sigma_u):.4f}")