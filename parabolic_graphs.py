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


# ----------------------------------------------------------------------
# --- P_success(T) piecewise ---
# ----------------------------------------------------------------------
def P_success(T):
    T = np.array(T)
    P = np.zeros_like(T)

    # Region 1: alpha1 ≤ T ≤ beta1
    idx1 = (T >= alpha1) & (T <= beta1)
    P[idx1] = ((T[idx1] - alpha1)**2 *
               (2*T[idx1] + alpha1 - 3*alpha2)) / ((alpha1 - alpha2)**3)

    # Region 2: beta1 ≤ T ≤ alpha2
    idx2 = (T >= beta1) & (T <= alpha2)
    P[idx2] = -((T[idx2] - alpha1)**2 *
                (T[idx2] - beta2)**2 *
                (2*T[idx2] + alpha1 - 3*alpha2) *
                (2*T[idx2] - 3*beta1 + beta2)) / (
                (alpha1 - alpha2)**3 * (beta1 - beta2)**3)

    # Region 3: alpha2 ≤ T ≤ beta2
    idx3 = (T >= alpha2) & (T <= beta2)
    P[idx3] = -((T[idx3] - beta2)**2 *
                (2*T[idx3] - 3*beta1 + beta2)) / ((beta1 - beta2)**3)

    return P

# Compute P_success array
y_success = P_success(x)



# ----------------------------------------------------------------------
# === PLOT 1: PU(x) and PA(x)
# ----------------------------------------------------------------------
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

# Horizontal peak lines
ax.hlines(y_alpha_peak, 0, x_alpha_peak,
          linestyles="--", colors="blue", linewidth=1)
ax.hlines(y_beta_peak, 0, x_beta_peak,
          linestyles="--", colors="red", linewidth=1)

# Region boundaries
for xc in [alpha1, beta1, alpha2, beta2]:
    ax.axvline(xc, color="gray", linestyle=":", linewidth=0.8)

# Symbolic ticks
ax.set_xticks([0.0, alpha1, beta1, alpha2, beta2])
ax.set_xticklabels(["0", r"$\alpha_1$", r"$\beta_1$", r"$\alpha_2$", r"$\beta_2$"],
                   fontsize=14)

ax.set_yticks([0.0, y_alpha_peak, y_beta_peak])
ax.set_yticklabels(
    ["0",
     r"$\frac{\alpha_1+\alpha_2}{2}$",
     r"$\frac{\beta_1+\beta_2}{2}$"],
    fontsize=14
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$", labelpad=-40)

ax.grid(True, linestyle=":", zorder=0)
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, max(y_beta_peak, y_alpha_peak) * 1.3)

ax.legend()
plt.tight_layout()
plt.savefig("Parabola_figure.pdf", format="pdf")



# ----------------------------------------------------------------------
# === PLOT 2: ONLY P_success(T) + horizontal line to the max point
# ----------------------------------------------------------------------

# Find location of maximum of P_success
idx_max = np.argmax(y_success)
x_opt = x[idx_max]
y_success_max = y_success[idx_max]

fig2, ax2 = plt.subplots(figsize=(10, 4))

# P_success curve
ax2.plot(x, y_success, color="black", linewidth=2.5,
         label=r"$P_{\text{success}}(T)$")

# Horizontal line from T=0 to T_opt at P_success(opt)
ax2.hlines(y_success_max, 0, x_opt,
           linestyles="--", colors="black", linewidth=1.5)

# Mark the optimal point itself (optional but nice)
ax2.scatter([x_opt], [y_success_max], color="black", zorder=5)

# Region boundaries
for xc in [alpha1, beta1, alpha2, beta2]:
    ax2.axvline(xc, color="gray", linestyle=":", linewidth=0.8)

# Symbolic ticks
ax2.set_xticks([0.0, alpha1, beta1, alpha2, beta2])
ax2.set_xticklabels(["0", r"$\alpha_1$", r"$\beta_1$", r"$\alpha_2$", r"$\beta_2$"],
                    fontsize=14)

# y-axis ticks: 0 and P_success(opt)
ax2.set_yticks([0.0, y_success_max])
ax2.set_yticklabels(["0", r"$P_{\text{success}}(\text{opt})$"],
                    fontsize=14)

ax2.set_xlabel(r"$T$")
ax2.set_ylabel(r"$P_{\text{success}}$", labelpad=-40)

ax2.grid(True, linestyle=":", zorder=0)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(0, y_success_max * 1.3)

ax2.legend()
plt.tight_layout()
plt.savefig("fig_success_parabola.pdf", format="pdf")
