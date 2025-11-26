import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# GLOBAL FONT SETTINGS
# ============================================================
FONTSIZE = 16
FONTNAME = "Times New Roman"

plt.rcParams.update({
    "font.size": FONTSIZE,
    "font.family": FONTNAME,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
})

# ============================================
# USER ROOTS
# ============================================
alpha1, alpha2 = 3.0, 4.0  # user parabola roots

# ============================================
# ATTACKER ROOTS (ATTACKER LEFT + OVERLAP)
# beta1 < alpha1 < beta2 < alpha2
# ============================================
attacker_roots = [
    (2.0, 3.5),
    (1.2, 3.5),
    (0.6, 3.5)
]

beta1 = attacker_roots[0][0]  # constant beta1
labels = ["(a)", "(b)", "(c)"]

# ============================================
# PDF FUNCTIONS
# ============================================
def user_pdf(x):
    a = 6 / (alpha1 - alpha2)**3
    b = -6 * (alpha1 + alpha2) / (alpha1 - alpha2)**3
    c = 6 * alpha1 * alpha2 / (alpha1 - alpha2)**3
    y = a * x**2 + b * x + c
    y[(x < alpha1) | (x > alpha2)] = 0
    return y

def attacker_pdf(x, b1, b2):
    d = 6 / (b1 - b2)**3
    e = -6 * (b1 + b2) / (b1 - b2)**3
    f = 6 * b1 * b2 / (b1 - b2)**3
    y = d * x**2 + e * x + f
    y[(x < b1) | (x > b2)] = 0
    return y

# ============================================
# FAR / FRR — ATTACKER LEFT (CANONICAL)
# ============================================
def compute_far_frr(x, PU, PA, b1, b2):
    dx = x[1] - x[0]

    # cumulative integrals
    PA_cum = np.cumsum(PA) * dx
    PU_cum = np.cumsum(PU) * dx

    # indices for bounds
    idx_b1 = np.argmin(np.abs(x - b1))
    idx_b2 = np.argmin(np.abs(x - b2))
    idx_a1 = np.argmin(np.abs(x - alpha1))
    idx_a2 = np.argmin(np.abs(x - alpha2))

    # total mass (should be 1, but use numeric total for robustness)
    total_A = PA_cum[idx_b2] - (PA_cum[idx_b1-1] if idx_b1 > 0 else 0.0)
    total_U = PU_cum[idx_a2] - (PU_cum[idx_a1-1] if idx_a1 > 0 else 0.0)

    def FAR(T):
        T = np.atleast_1d(T)
        out = np.zeros_like(T, dtype=float)
        for i, t in enumerate(T):
            if t <= b1:
                out[i] = 1.0
            elif t >= b2:
                out[i] = 0.0
            else:
                idx = np.argmin(np.abs(x - t))
                num = PA_cum[idx_b2] - PA_cum[idx]
                out[i] = num / total_A
        return out

    def FRR(T):
        T = np.atleast_1d(T)
        out = np.zeros_like(T, dtype=float)
        for i, t in enumerate(T):
            if t <= alpha1:
                out[i] = 0.0
            elif t >= alpha2:
                out[i] = 1.0
            else:
                idx = np.argmin(np.abs(x - t))
                num = PU_cum[idx] - PU_cum[idx_a1]
                out[i] = num / total_U
        return out

    return FAR, FRR

# ============================================
# SIMPLIFIED Y-AXIS HANDLER
# ============================================
def remove_y_ticks(ax):
    ax.set_yticks([0])
    ax.set_yticklabels(["0"])

def set_symbolic_ticks(ax, b1, b2):
    # attacker-left ordering: beta1 < alpha1 < beta2 < alpha2
    ax.set_xticks([0, b1, alpha1, b2, alpha2])
    ax.set_xticklabels([
        "0",
        r"$\beta_1$",
        r"$\alpha_1$",
        r"$\beta_2$",
        r"$\alpha_2$"
    ])
    ax.set_yticks([0])
    ax.set_yticklabels(["0"])

# ============================================
# DOMAIN
# ============================================
x = np.linspace(0, 5.0, 3000)
PU = user_pdf(x)

# ============================================
# PLOT 1 — Parabola_figure.pdf
# ============================================
b1_plot, b2_plot = attacker_roots[0]
PA = attacker_pdf(x, b1_plot, b2_plot)

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(x, PU, color="blue", linewidth=2, label=r"$P_U(s)$")
ax1.plot(x, PA, color="red",  linewidth=2, label=r"$P_A(s)$")

ax1.axhline(0, color="black")
ax1.legend()

set_symbolic_ticks(ax1, b1_plot, b2_plot)
remove_y_ticks(ax1)

ax1.set_xlabel(r"$s$")
ax1.set_ylabel(r"$P(s)$")

ax1.grid(axis='y', linestyle=":")

plt.tight_layout()
plt.savefig("Parabola_figure.pdf", format="pdf")


# ============================================================
# PLOT 2A — PDFs
# ============================================================
fig2A, axes2A = plt.subplots(1, 3, figsize=(24, 8))

for col, (b1, b2) in enumerate(attacker_roots):
    PA = attacker_pdf(x, b1, b2)
    ax = axes2A[col]

    ax.plot(x, PU, color="blue", linewidth=2)
    ax.plot(x, PA, color="red", linewidth=2)

    set_symbolic_ticks(ax, b1, b2)
    remove_y_ticks(ax)

    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$P(s)$")

    ax.text(0.5, -0.28, labels[col],
            transform=ax.transAxes, fontweight="bold",
            ha="center", va="top")

plt.tight_layout()
plt.savefig("fig_distributions_parabola.pdf", format="pdf")


# ============================================================
# PLOT 2B — P_success (from FAR/FRR, attacker-left)
# ============================================================
fig2B, axes2B = plt.subplots(1, 3, figsize=(24, 8))

for col, (b1, b2) in enumerate(attacker_roots):
    PA = attacker_pdf(x, b1, b2)
    FAR, FRR = compute_far_frr(x, PU, PA, b1, b2)

    far_vals = FAR(x)
    frr_vals = FRR(x)

    # P_success(T) = 1 - FAR - FRR + FAR*FRR
    Ps = 1.0 - far_vals - frr_vals + far_vals * frr_vals

    idx_opt = np.argmax(Ps)
    T_opt = x[idx_opt]
    Ps_opt = Ps[idx_opt]

    diff = np.abs(far_vals - frr_vals)
    idx_eer = np.argmin(diff)
    T_eer = x[idx_eer]
    Ps_eer = Ps[idx_eer]

    ax = axes2B[col]
    ax.plot(x, Ps, color="black", linewidth=2)

    ax.scatter([T_opt], [Ps_opt], color="blue")
    ax.annotate("Optimal",
                xy=(T_opt, Ps_opt),
                xytext=(T_opt + 0.05, Ps_opt),
                fontsize=12)

    ax.scatter([T_eer], [Ps_eer], color="red")
    ax.annotate("EER",
                xy=(T_eer, Ps_eer),
                xytext=(T_eer - 0.15, Ps_eer),
                ha="right",
                fontsize=12)

    set_symbolic_ticks(ax, b1, b2)
    remove_y_ticks(ax)

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$P_{\text{success}}(T)$")

    ax.text(0.5, -0.28, labels[col],
            transform=ax.transAxes, fontweight="bold",
            ha="center", va="top")

plt.tight_layout()
plt.savefig("fig_psuccess_parabola.pdf", format="pdf")


# ============================================
# PLOT 3 — FAR vs FRR
# ============================================
fig3, axes3 = plt.subplots(1, 3, figsize=(24, 8))

for col, (b1, b2) in enumerate(attacker_roots):
    PA = attacker_pdf(x, b1, b2)
    FAR, FRR = compute_far_frr(x, PU, PA, b1, b2)

    far_vals = FAR(x)
    frr_vals = FRR(x)

    # success for optimal point
    Ps = 1.0 - far_vals - frr_vals + far_vals * frr_vals
    idx_opt = np.argmax(Ps)
    T_opt = x[idx_opt]
    far_opt = FAR(T_opt)[0]
    frr_opt = FRR(T_opt)[0]

    diff = np.abs(frr_vals - far_vals)
    idx_eer = np.argmin(diff)
    eer = far_vals[idx_eer]

    ax = axes3[col]
    ax.plot(frr_vals, far_vals, color="purple", linewidth=2)

    ax.scatter([frr_opt], [far_opt], color="blue")
    ax.annotate("Optimal",
                xy=(frr_opt, far_opt),
                xytext=(frr_opt + 0.03, far_opt + 0.03),
                fontsize=12)

    ax.scatter([eer], [eer], color="red")
    ax.annotate("EER",
                xy=(eer, eer),
                xytext=(eer + 0.03, eer),
                fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # No numbers
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel("FRR(T)")
    ax.set_ylabel("FAR(T)")

    ax.grid(axis='y', linestyle=":")

    ax.text(0.5, -0.28, labels[col],
            transform=ax.transAxes, fontweight="bold",
            ha="center", va="top")

plt.tight_layout()
plt.savefig("fig_FARvFRR_parabola.pdf", format="pdf")


# ============================================
# PLOT 4 — Gap between EER and Optimal vs Variance
# ============================================

# Sweep attacker beta2 values to vary shape, while keeping:
# beta1 < alpha1 < beta2 < alpha2
beta1_sweep = np.linspace(0.5, 2.5, 120)  # all between 1 and 2

variances = []
gaps = []
beta2 = attacker_roots[0][1]
for beta1 in beta1_sweep:
    PA = attacker_pdf(x, beta1, beta2)

    # variance of attacker
    EX  = np.trapezoid(x * PA, x)
    EX2 = np.trapezoid((x**2) * PA, x)
    variance = EX2 - EX**2
    variances.append(variance)

    FAR, FRR = compute_far_frr(x, PU, PA, beta1, beta2)
    far_vals = FAR(x)
    frr_vals = FRR(x)

    Ps = 1.0 - far_vals - frr_vals + far_vals * frr_vals

    # Optimal T
    idx_opt = np.argmax(Ps)
    T_opt = x[idx_opt]

    # EER T
    diff = np.abs(far_vals - frr_vals)
    idx_eer = np.argmin(diff)
    T_eer = x[idx_eer]

    gaps.append(abs(T_opt - T_eer))

variances = np.array(variances)
gaps = np.array(gaps)

fig_gap, ax_gap = plt.subplots(figsize=(10, 6))

ax_gap.plot(variances, gaps, color="purple", linewidth=2)

ax_gap.set_xlabel(r"Attacker Variance $\mathrm{Var}(s)$")
ax_gap.set_ylabel(r"$|T_{\text{opt}} - T_{\text{EER}}|$")
ax_gap.grid(True, linestyle=":")

plt.tight_layout()
plt.savefig("fig_gap_vs_variance_parabola.pdf", format="pdf")

print("Saved all parabolic figures:")
print("  - Parabola_figure.pdf")
print("  - fig_distributions_parabola.pdf")
print("  - fig_psuccess_parabola.pdf")
print("  - fig_FARvFRR_parabola.pdf")
print("  - fig_gap_vs_variance_parabola.pdf")
