import numpy as np
import matplotlib.pyplot as plt


# ============================================
# USER ROOTS
# ============================================
alpha1, alpha2 = 1.0, 2.0

# ============================================
# 3 ATTACKER ROOTS (3 columns)
# ============================================
attacker_roots = [
    (1.25, 2.4),
    (1.25, 3.0),
    (1.25, 4.5)
]

beta1 = attacker_roots[0][0]

# Subplot labels
labels = ["(a)", "(b)", "(c)"]

# ============================================
# PDF FUNCTIONS
# ============================================
def user_pdf(x):
    a = 6 / (alpha1 - alpha2)**3
    b = -6*(alpha1 + alpha2) / (alpha1 - alpha2)**3
    c = 6*alpha1*alpha2 / (alpha1 - alpha2)**3
    y = a*x**2 + b*x + c
    y[(x < alpha1) | (x > alpha2)] = 0
    return y

def attacker_pdf(x, b1, b2):
    d = 6 / (b1 - b2)**3
    e = -6*(b1 + b2) / (b1 - b2)**3
    f = 6*b1*b2 / (b1 - b2)**3
    y = d*x**2 + e*x + f
    y[(x < b1) | (x > b2)] = 0
    return y

# ============================================
# P_success(T)
# ============================================
def P_success(T, beta1, beta2):
    T = np.array(T)
    P = np.zeros_like(T)

    idx1 = (T >= alpha1) & (T <= beta1)
    P[idx1] = ((T[idx1] - alpha1)**2 *
               (2*T[idx1] + alpha1 - 3*alpha2)) / ((alpha1 - alpha2)**3)

    idx2 = (T >= beta1) & (T <= alpha2)
    P[idx2] = -((T[idx2] - alpha1)**2 *
                (T[idx2] - beta2)**2 *
                (2*T[idx2] + alpha1 - 3*alpha2) *
                (2*T[idx2] - 3*beta1 + beta2)) / (
                (alpha1 - alpha2)**3 * (beta1 - beta2)**3)

    idx3 = (T >= alpha2) & (T <= beta2)
    P[idx3] = -((T[idx3] - beta2)**2 *
                (2*T[idx3] - 3*beta1 + beta2)) / ((beta1 - beta2)**3)

    return P

# ============================================
# FAR / FRR
# ============================================
def compute_far_frr(x, PU, PA, b1, b2):
    dx = x[1] - x[0]

    PA_cum = np.cumsum(PA) * dx
    idx_b1 = np.argmin(np.abs(x - b1))

    def FAR(T):
        T = np.atleast_1d(T)
        out = np.zeros_like(T)
        base = PA_cum[idx_b1]
        total = PA_cum[-1] - base
        for i, t in enumerate(T):
            if t <= b1: out[i] = 0
            elif t >= b2: out[i] = total
            else:
                idx = np.argmin(np.abs(x - t))
                out[i] = PA_cum[idx] - base
        return out

    PU_cum_rev = np.cumsum(PU[::-1]) * dx
    PU_cum = PU_cum_rev[::-1]
    idx_a2 = np.argmin(np.abs(x - alpha2))
    full_seg = PU_cum[0] - PU_cum[idx_a2]

    def FRR(T):
        T = np.atleast_1d(T)
        out = np.zeros_like(T)
        for i, t in enumerate(T):
            if t >= alpha2: out[i] = 0
            elif t <= alpha1: out[i] = full_seg
            else:
                idx = np.argmin(np.abs(x - t))
                out[i] = PU_cum[idx] - PU_cum[idx_a2]
        return out

    return FAR, FRR


# ============================================
# FIXED: REMOVE Y-TICKS GAP
# ============================================
def remove_y_ticks(ax):
    ax.set_yticks([0])
    ax.set_yticklabels(["0"], fontsize=16)

    # Remove padding between tick and axis
    ax.tick_params(axis='y', pad=0)

    # Force left axis spine to be exactly on border
    ax.spines['left'].set_position(('outward', 0))
    ax.spines['left'].set_linewidth(1.5)


def set_symbolic_ticks(ax, beta2):
    ax.set_xticks([0, alpha1, beta1, alpha2, beta2])
    ax.set_xticklabels(
        ["0", r"$\alpha_1$", r"$\beta_1$", r"$\alpha_2$", r"$\beta_2$"],
        fontsize=16
    )
    ax.set_yticks([0.0])
    ax.set_yticklabels(
    ["0"],
    fontsize=16
)


# ============================================
# DOMAIN + USER PDF
# ============================================
x = np.linspace(0, 5.0, 3000)
PU = user_pdf(x)

# ============================================
# ============= PLOT 1 ========================
# Parabola_figure.pdf
# ============================================
b1_plot, b2_plot = attacker_roots[0]
PA = attacker_pdf(x, b1_plot, b2_plot)

fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(x, PU, color="blue", linewidth=2, label=r"$P_U(x)$")
ax1.plot(x, PA, color="red",  linewidth=2, label=r"$P_A(x)$")
ax1.legend(fontsize=16)

ax1.axhline(0, color="black")
for xc in [alpha1, beta1, alpha2, b2_plot]:
    ax1.axvline(xc, color="gray", linestyle=":", linewidth=0.8)

set_symbolic_ticks(ax1, b2_plot)
remove_y_ticks(ax1)

ax1.set_xlabel(r"$x$", fontsize=16)
ax1.set_ylabel(r"$P(x)$", fontsize=16)
ax1.tick_params(axis="both", labelsize=16)
ax1.grid(True, linestyle=":")

plt.tight_layout()
plt.savefig("Parabola_figure.pdf", format="pdf")


# ============================================
# ============= PLOT 2A =======================
# PDFs only (1×3)
# ============================================
fig2A, axes2A = plt.subplots(1, 3, figsize=(24,8))

for col, (b1, b2) in enumerate(attacker_roots):
    PA = attacker_pdf(x, b1, b2)
    ax = axes2A[col]

    ax.plot(x, PU, color="blue", linewidth=2, label=r"$P_U(x)$")
    ax.plot(x, PA, color="red", linewidth=2, label=r"$P_A(x)$")
    ax.legend(fontsize=16)

    ax.axhline(0, color="black")

    for xc in [alpha1, beta1, alpha2, b2]:
        ax.axvline(xc, linestyle=":", color="gray")

    set_symbolic_ticks(ax, b2)

    ax.set_xlabel(r"$x$", fontsize=16)
    ax.set_ylabel(r"$P(x)$", fontsize=16)
    ax.grid(True, linestyle=":", zorder=0)

    ax.text(0.5, -0.28, labels[col],
            transform=ax.transAxes,
            fontsize=16, fontweight="bold",
            ha="center", va="top")

plt.tight_layout()
plt.savefig("fig_distributions_parabola.pdf", format="pdf")


# ============================================
# ============= PLOT 2B =======================
# P_success only (1×3)
# ============================================
fig2B, axes2B = plt.subplots(1, 3, figsize=(24,8))

for col, (b1, b2) in enumerate(attacker_roots):
    PA = attacker_pdf(x, b1, b2)
    Ps = P_success(x, b1, b2)
    FAR, FRR = compute_far_frr(x, PU, PA, b1, b2)

    idx_opt = np.argmax(Ps)
    T_opt = x[idx_opt]
    Ps_opt = Ps[idx_opt]

    diff = np.abs(FAR(x) - FRR(x))
    idx_eer = np.argmin(diff)
    T_eer = x[idx_eer]
    Ps_eer = Ps[idx_eer]

    ax = axes2B[col]
    ax.plot(x, Ps, color="black", linewidth=2, label=r"$P_{\text{success}}$")
    ax.legend(fontsize=16)

    ax.scatter([T_opt], [Ps_opt], color="blue")
    ax.annotate("Optimal", xy=(T_opt, Ps_opt),
                xytext=(T_opt + 0.05, Ps_opt),
                fontsize=14, color="blue")

    ax.scatter([T_eer], [Ps_eer], color="red")
    ax.annotate("EER", xy=(T_eer, Ps_eer),
                xytext=(T_eer - 0.15, Ps_eer),
                fontsize=14, color="red",
                ha="right", va="center")

    for xc in [alpha1, beta1, alpha2, b2]:
        ax.axvline(xc, linestyle=":", color="gray")

    set_symbolic_ticks(ax, b2)
    remove_y_ticks(ax)

    ax.set_xlabel(r"$T$", fontsize=16)
    ax.set_ylabel(r"$P_{\text{success}}(T)$", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, linestyle=":")

    ax.text(0.5, -0.28, labels[col],
            transform=ax.transAxes,
            fontsize=16, fontweight="bold",
            ha="center", va="top")

plt.tight_layout()
plt.savefig("fig_psuccess_parabola.pdf", format="pdf")


# ============================================
# ============= PLOT 3 ========================
# FAR–FRR curves (1×3)
# ============================================
fig3, axes3 = plt.subplots(1, 3, figsize=(24,8))

for col, (b1, b2) in enumerate(attacker_roots):
    PA = attacker_pdf(x, b1, b2)
    Ps = P_success(x, b1, b2)
    FAR, FRR = compute_far_frr(x, PU, PA, b1, b2)

    far_vals = FAR(x)
    frr_vals = FRR(x)

    idx_opt = np.argmax(Ps)
    T_opt = x[idx_opt]
    far_opt = FAR(T_opt)[0]
    frr_opt = FRR(T_opt)[0]

    diff = np.abs(frr_vals - far_vals)
    idx_eer = np.argmin(diff)
    eer = far_vals[idx_eer]

    ax = axes3[col]
    ax.plot(frr_vals, far_vals, color="purple", linewidth=2, label="FAR vs FRR")
    ax.legend(fontsize=16)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(axis="both", labelsize=16)

    ax.scatter([frr_opt], [far_opt], color="blue")
    ax.annotate("Optimal", xy=(frr_opt, far_opt),
                xytext=(frr_opt + 0.03, far_opt + 0.03),
                fontsize=14, color="blue")

    ax.scatter([eer], [eer], color="red")
    ax.annotate("EER",
                xy=(eer, eer),
                xytext=(eer + 0.03, eer),
                fontsize=14, color="red")

    ax.set_xlabel("FRR(T)", fontsize=16)
    ax.set_ylabel("FAR(T)", fontsize=16)

    ax.grid(True, linestyle=":")
    
    ax.text(0.5, -0.28, labels[col],
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="center", va="top")

plt.tight_layout()
plt.savefig("fig_FARvFRR_parabola.pdf", format="pdf")

# ============================================
# NEW PLOT 4: Gap between EER and Optimal vs Overlap Area
# ============================================

# Sweep attacker beta2 values to vary overlap
beta2_sweep = np.linspace(2.4, 4.5, 120)

overlaps = []
gaps = []

for beta2 in beta2_sweep:
    PA = attacker_pdf(x, beta1, beta2)

    # Overlap area = ∫ min(PU,PA)
    overlap = np.trapezoid(np.minimum(PU, PA), x)
    overlaps.append(overlap)

    # Compute FAR/FRR
    FAR, FRR = compute_far_frr(x, PU, PA, beta1, beta2)

    # Success
    Ps = P_success(x, beta1, beta2)

    # Optimal T
    idx_opt = np.argmax(Ps)
    T_opt = x[idx_opt]

    # EER T
    diff = np.abs(FAR(x) - FRR(x))
    idx_eer = np.argmin(diff)
    T_eer = x[idx_eer]

    gaps.append(abs(T_opt - T_eer))


# Convert to numpy arrays
overlaps = np.array(overlaps)
gaps = np.array(gaps)


# ===== FIGURE =====
fig_gap, ax_gap = plt.subplots(figsize=(10,6))

ax_gap.plot(overlaps, gaps, color="purple", linewidth=2)

ax_gap.set_xlabel(r"Overlap Area $\int \min(P_U, P_A) \, dx$", fontsize=16)
ax_gap.set_ylabel(r"$|T_{\text{opt}} - T_{\text{EER}}|$", fontsize=16)
ax_gap.tick_params(axis="both", labelsize=16)

ax_gap.grid(True, linestyle=":")

plt.tight_layout()
plt.savefig("fig_gap_vs_overlap.pdf", format="pdf")
