import numpy as np
import matplotlib.pyplot as plt

# --- Parameters (roots) ---
alpha1, alpha2 = 1.0, 3.0
beta1,  beta2  = 2.0, 3.5

# --- Coefficients for the parabolas ---
a = 6 / (alpha1 - alpha2)**3
b = - (6 * (alpha1 + alpha2)) / (alpha1 - alpha2)**3
c = (6 * alpha1 * alpha2) / (alpha1 - alpha2)**3

d = 6 / (beta1 - beta2)**3
e = - (6 * (beta1 + beta2)) / (beta1 - beta2)**3
f = (6 * beta1 * beta2) / (beta1 - beta2)**3

# For plotting
x_min, x_max = 0, 4.5
x = np.linspace(x_min, x_max, 1000)

# --- Define the parabolas ---
y_alpha = a * x**2 + b * x + c
y_beta  = d * x**2 + e * x + f

# Zero outside the roots
y_alpha[(x < alpha1) | (x > alpha2)] = 0
y_beta[(x < beta1)  | (x > beta2)]  = 0

# Peaks
x_alpha_peak = (alpha1 + alpha2) / 2.0
x_beta_peak  = (beta1 + beta2) / 2.0

y_alpha_peak = a * x_alpha_peak**2 + b * x_alpha_peak + c
y_beta_peak  = d * x_beta_peak**2  + e * x_beta_peak  + f

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 4))

# Parabolas
ax.plot(x, y_alpha, label=r"$P_U(x)$", linewidth=2, color="blue")
ax.plot(x, y_beta,  label=r"$P_A(x)$", linewidth=2, color="red")

# Axis line
ax.axhline(0, color="black", linewidth=1)

# Root markers
ax.scatter([alpha1, alpha2], [0, 0], color="blue", zorder=5)
ax.scatter([beta1,  beta2],  [0, 0], color="red", zorder=5)

# Peak points
ax.scatter([x_alpha_peak], [y_alpha_peak], color="blue", zorder=6)
ax.scatter([x_beta_peak],  [y_beta_peak],  color="red",  zorder=6)

# --- Horizontal dashed lines from y-axis to each peak ---
ax.hlines(y_alpha_peak, 0, x_alpha_peak,
          linestyles="--", colors="blue", linewidth=1)

ax.hlines(y_beta_peak, 0, x_beta_peak,
          linestyles="--", colors="red", linewidth=1)


# Symbolic axes labels
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$",labelpad=-40)

# Remove numeric ticks (symbolic only)
ax.set_xticks(
    ticks=[0.0, alpha1, beta1, alpha2, beta2],
    labels=["0", r"$\alpha_1$", r"$\beta_1$", r"$\alpha_2$", r"$\beta_2$"],fontsize = 14)
ax.set_yticks(    
    ticks=[0.0, y_alpha_peak, y_beta_peak],
    labels=["0", r"$\frac{\alpha_1+\alpha_2}{2}$", r"$\frac{\beta_1+\beta_2}{2}$"],
    fontsize = 14)

# FORCE the grid to show even without ticks
ax.grid(True, linestyle=":", zorder=0)

ax.set_xlim(x_min, x_max)
ax.set_ylim(0, max(y_beta_peak, y_alpha_peak) * 1.3)

ax.legend()

plt.tight_layout()
plt.savefig("Parabola_figure.pdf", format="pdf")