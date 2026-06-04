import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
# ============================================================
#                 BASE PDF DEFINITION
# ============================================================
REPO_ROOT = Path("/Users/dormalka/Desktop/Dor/Paper").resolve()
OUTPUT_DIR = REPO_ROOT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_TXT = OUTPUT_DIR / "uniform_pdfs.txt"

def uniform_pdf(x, a, b):
    return np.where((x >= a) & (x <= b), 1 / (b - a), 0)

def compute_success_gap(u1, u2, a1, a2, N=2000):
    x = np.linspace(min(u1, a1), max(u2, a2), N)

    user_pdf = uniform_pdf(x, u1, u2)
    attacker_pdf = uniform_pdf(x, a1, a2)

    # FRR
    FRR = np.array([
        0.0 if T <= u1 else
        1.0 if T >= u2 else
        np.trapezoid(
            user_pdf[(x >= u1) & (x <= T)],
            x[(x >= u1) & (x <= T)]
        )
        for T in x
    ])

    # FAR
    FAR = np.array([
        1.0 if T <= a1 else
        0.0 if T >= a2 else
        np.trapezoid(
            attacker_pdf[(x >= T) & (x <= a2)],
            x[(x >= T) & (x <= a2)]
        )
        for T in x
    ])

    # Success
    loss  = FRR * (1 - FAR)
    leak  = FAR * (1 - FRR)
    theft = FRR * FAR
    safe  = 1 - loss - leak - theft

    # Optimal
    opt_idx = np.argmax(safe)
    P_opt = safe[opt_idx]

    # EER
    eer_idx = np.argmin(np.abs(FAR - FRR))
    P_eer = safe[eer_idx]

    return P_opt - P_eer

# ============================================================
#    COMPUTE T_opt, T_EER and GAP for uniform PDFs (ATTACKER LEFT)
# ============================================================
def compute_Topt_Teer(u1, u2, a1, a2, N=5000):
    # x covers from attacker left edge to user right edge
    T = np.linspace(min(u1, a1), max(u2, a2), N)

    user_pdf = uniform_pdf(x, u1, u2)
    attacker_pdf = uniform_pdf(x, a1, a2)

    # FRR: probability that a genuine user is rejected
    # threshold T: accept if score >= T
    # FRR(T) = ∫_{T}^{u2} f_user(t) dt  (for T<u2)
    FRR = np.array([
        0.0 if T <= u1 else
        1.0 if T >= u2 else
        np.trapezoid(
            user_pdf[(x >= u1) & (x <= T)],
            x[(x >= u1) & (x <= T)]
        )
        for T in x
    ])

    # ------------------------------------------------------------
    # CORRECT FAR(T)  (attacker region [a1_asym, a2_asym])
    # ------------------------------------------------------------
    FAR = np.array([
        1.0 if T <= a1 else
        0.0 if T >= a2 else
        np.trapezoid(
            attacker_pdf[(x >= T) & (x <= a2)],
            x[(x >= T) & (x <= a2)]
        )
        for T in x
    ])

    # Success function
    loss = FRR * (1 - FAR)
    leak = FAR * (1 - FRR)
    theft = FRR * FAR
    safe = 1 - loss - leak - theft

    max_idx = np.argmax(safe)
    T_opt = x[max_idx]
    T_eer = x[np.argmin(np.abs(FAR - FRR))]
    gap = abs(T_opt - T_eer)
    var = ((a2 - a1) ** 2) / 12

    return T_opt, T_eer, gap, var


# ============================================================
#               INTERVALS (ATTACKER LEFT + OVERLAP)
# ============================================================

# User interval (higher scores)
u1, u2 = 0.30, 0.70             # length = 0.40

# --- Symmetric case: same height → same length as user ---
# choose attacker left but overlapping:
# attacker: [0.10, 0.50]  (length 0.40, same as user)
a1_sym,  a2_sym  = 0.10, 0.50   # symmetric: same width as user

# --- Asymmetric case: different width, still overlapping and left ---
a1_asym, a2_asym = 0.05, 0.35   # attacker narrower & left, with overlap

x = np.linspace(0, 1, 1000)

user_pdf_sym       = uniform_pdf(x, u1, u2)
attacker_pdf_sym   = uniform_pdf(x, a1_sym, a2_sym)
attacker_pdf_asym  = uniform_pdf(x, a1_asym, a2_asym)

# Heights
max_u   = 1 / (u2      - u1)        # = 1 / 0.4
max_sym = 1 / (a2_sym  - a1_sym)    # same as max_u by construction
att_max = 1 / (a2_asym - a1_asym)

PU = uniform_pdf(x, u1, u2)
PA_sym = uniform_pdf(x, a1_sym, a2_sym)
PA_asym = uniform_pdf(x, a1_asym, a2_asym)

DATA_TXT = OUTPUT_DIR / "figs" / "fig_uniform" / "uniform_graphs.txt"
DATA_TXT.parent.mkdir(parents=True, exist_ok=True)

DATA_TXT_2 = OUTPUT_DIR / "figs" / "fig_uniform" / "uniform_graphs_2.txt"
DATA_TXT_2.parent.mkdir(parents=True, exist_ok=True)

with open(DATA_TXT, "w") as f:
    f.write("s PU PA\n")
    for xi, pu, pa in zip(x, PU, PA_asym):
        f.write(f"{xi:.4f} {pu:.4f} {pa:.4f}\n")

with open(DATA_TXT_2, "w") as f:
    f.write("s PU PA_sym PA_asym\n")
    for xi, pu, pa_sym, pa_asym in zip(x, PU, PA_sym, PA_asym):
        f.write(f"{xi:.4f} {pu:.4f} {pa_sym:.4f} {pa_asym:.4f}\n")

print(f"[i] Data saved to: {DATA_TXT}")
print(f"[i] Data saved to: {DATA_TXT_2}")

# ============================================================
#         FIGURE 1 – SYMMETRIC & ASYMMETRIC UNIFORMS
# ============================================================

a1_asym, a2_asym = 0.05, 0.42

plt.figure(figsize=(10, 5))

# --------------------------
# 1. SYMMETRIC CASE
# --------------------------
plt.subplot(1, 2, 1)
plt.plot(x, user_pdf_sym, label=r"$P_U(s)$", color='blue', linewidth=2)
plt.plot(x, attacker_pdf_sym, label=r"$P_A(s)$", color='red',  linewidth=2)
plt.xlabel("Matching Score")
plt.ylabel("Probability Density", labelpad=-60)
plt.legend(loc='upper right')
plt.grid(True)

plt.xticks(
    ticks=[0.0, u1, u2, a1_sym, a2_sym],
    labels=["0", r"$u_1$", r"$u_2$", r"$a_1$", r"$a_2$"]
)

plt.yticks(
    ticks=[0.0, max_u],
    labels=["0", r"$\frac{1}{u_2-u_1}=\frac{1}{a_2-a_1}$"]
)

# --------------------------
# 2. ASYMMETRIC CASE
# --------------------------
plt.subplot(1, 2, 2)
plt.plot(x, user_pdf_sym, label=r"$P_U(s)$", color='blue', linewidth=2)
plt.plot(x, attacker_pdf_asym, label=r"$P_A(s)$", color='red',  linewidth=2)
plt.xlabel("Matching Score")
plt.ylabel("Probability Density", labelpad=-20)
plt.legend(loc='upper right')
plt.grid(True)

plt.xticks(
    ticks=[0.0, u1, u2, a1_asym, a2_asym],
    labels=["0", r"$u_1$", r"$u_2$", r"$a_1$", r"$a_2$"]
)

plt.yticks(
    ticks=[0.0, max_u, att_max],
    labels=["0", r"$\frac{1}{u_2-u_1}$", r"$\frac{1}{a_2-a_1}$"]
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figs" / "fig_uniform" / "fig_uniforms.pdf")


# ============================================================
#                SUCCESS FUNCTION (ATTACKER LEFT)
# ============================================================

attacker_pdf_asym = uniform_pdf(x, a1_asym, a2_asym)

# ------------------------------------------------------------
# CORRECT FRR(T)  (user region [u1, u2])
# ------------------------------------------------------------
FRR = np.array([
    0.0 if T <= u1 else
    1.0 if T >= u2 else
    np.trapezoid(
        user_pdf_sym[(x >= u1) & (x <= T)],
        x[(x >= u1) & (x <= T)]
    )
    for T in x
])

# ------------------------------------------------------------
# CORRECT FAR(T)  (attacker region [a1_asym, a2_asym])
# ------------------------------------------------------------
FAR = np.array([
    1.0 if T <= a1_asym else
    0.0 if T >= a2_asym else
    np.trapezoid(
        attacker_pdf_asym[(x >= T) & (x <= a2_asym)],
        x[(x >= T) & (x <= a2_asym)]
    )
    for T in x
])

# ------------------------------------------------------------
# SYMMETRIC FAR/FRR CURVE
# ------------------------------------------------------------
attacker_pdf_sym = uniform_pdf(x, a1_sym, a2_sym)

FRR_sym = np.array([
    0.0 if T <= u1 else
    1.0 if T >= u2 else
    np.trapezoid(
        user_pdf_sym[(x >= u1) & (x <= T)],
        x[(x >= u1) & (x <= T)]
    )
    for T in x
])

FAR_sym = np.array([
    1.0 if T <= a1_sym else
    0.0 if T >= a2_sym else
    np.trapezoid(
        attacker_pdf_sym[(x >= T) & (x <= a2_sym)],
        x[(x >= T) & (x <= a2_sym)]
    )
    for T in x
])

loss  = FRR * (1 - FAR)
leak  = FAR * (1 - FRR)
theft = FRR * FAR
safe  = 1 - loss - leak - theft

max_idx    = np.argmax(safe)
best_T     = x[max_idx]
best_safe  = safe[max_idx]

FAR_opt = FAR[max_idx]
FRR_opt = FRR[max_idx]

# ---- Compute EER point ----
eer_idx = np.argmin(np.abs(FAR - FRR))
FRR_eer = FRR[eer_idx]
FAR_eer = FAR[eer_idx]

loss_sym  = FRR_sym * (1 - FAR_sym)
leak_sym  = FAR_sym * (1 - FRR_sym)
theft_sym = FRR_sym * FAR_sym
safe_sym  = 1 - loss_sym - leak_sym - theft_sym
max_idx_sym  = np.argmax(safe_sym)
FAR_opt_sym = FAR_sym[max_idx_sym]
FRR_opt_sym = FRR_sym[max_idx_sym]

# ---- Compute EER point ----
eer_idx_2 = np.argmin(np.abs(FAR_sym - FRR_sym))
FRR_eer_sym = FRR_sym[eer_idx_2]
FAR_eer_sym = FAR_sym[eer_idx_2]

SUCCESS_DATA_TXT = OUTPUT_DIR / "figs" / "fig_uniform" / "success_uniform_data.txt"
SUCCESS_DATA_TXT.parent.mkdir(parents=True, exist_ok=True)

with open(SUCCESS_DATA_TXT, "w") as f:
    f.write(
        "s FRR FAR FRR_sym FAR_sym "
        "loss leak theft safe "
        "loss_sym leak_sym theft_sym safe_sym "
        "FAR_opt FRR_opt FAR_eer FRR_eer "
        "FAR_opt_sym FRR_opt_sym FAR_eer_sym FRR_eer_sym\n"
    )
    for xi, frr, far, frr_s, far_s, lo, le, th, sa, lo_s, le_s, th_s, sa_s in zip(
        x, FRR, FAR, FRR_sym, FAR_sym,
        loss, leak, theft, safe,
        loss_sym, leak_sym, theft_sym, safe_sym
    ):
        f.write(
            f"{xi:.4f} "
            f"{frr:.4f} "
            f"{far:.4f} "
            f"{frr_s:.4f} "
            f"{far_s:.4f} "
            f"{lo:.4f} "
            f"{le:.4f} "
            f"{th:.4f} "
            f"{sa:.4f} "
            f"{lo_s:.4f} "
            f"{le_s:.4f} "
            f"{th_s:.4f} "
            f"{sa_s:.4f} "
            f"{FAR_opt:.4f} "
            f"{FRR_opt:.4f} "
            f"{FAR_eer:.4f} "
            f"{FRR_eer:.4f} "
            f"{FAR_opt_sym:.4f} "
            f"{FRR_opt_sym:.4f} "
            f"{FAR_eer_sym:.4f} "
            f"{FRR_eer_sym:.4f}\n"
        )

plt.figure(figsize=(10, 5))
plt.plot(x, safe, label=r"$P_{\text{success}}(T)$", color="purple")
plt.xlabel(r"$T$")
plt.ylabel(r"$P_{\text{success}}(T)$", labelpad=-80)
plt.legend()
plt.grid(True)

plt.xticks(
    ticks=[0.0, u1, u2, a1_asym, best_T, a2_asym],
    labels=["0", r"$u_1$", r"$u_2$", r"$a_1$", r"$T_{opt}$", r"$a_2$"]
)

plt.yticks(
    ticks=[0, best_safe,],
    labels=['0',
        r"$\frac{(a_1 - u_2)^2}{4 (a_1 - a_2) (u_1 - u_2)}$"
    ],
    fontsize=14
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figs" / "fig_uniform" / "fig_success_uniform.pdf")


# ============================================================
#                FAR vs FRR CURVE (WITH EER POINT)
# ============================================================

plt.figure(figsize=(8, 4))
plt.plot(FRR, FAR, color='purple', linewidth=2, label = "Operating Curve")

# ---- Plot Optimal Point ----
plt.scatter(FRR_opt, FAR_opt, color='blue', s=50, zorder=5,label = "Optimal Point")
plt.annotate(
    'Optimal',
    xy=(FRR_opt, FAR_opt),
    xytext=(FRR_opt + 0.015, FAR_opt + 0.015),
    fontsize=12
)

# ---- Plot EER Point ----
plt.scatter(FRR_eer, FAR_eer, color='red', s=50, zorder=5, label="EER Point")
plt.annotate(
    'EER',
    xy=(FRR_eer, FAR_eer),
    xytext=(FRR_eer - 0.05, FAR_eer - 0.07),
    fontsize=12,
    color='black'
)
plt.legend()
plt.xlabel(r"$\text{FRR}$")
plt.ylabel(r"$\text{FAR}$", labelpad=-10)
plt.grid(True)

# ---- Only show ticks 0 and 1 ----
plt.xticks([0, 1], ["0", "1"], fontsize=12)
plt.yticks([0, 1], ["0", "1"], fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figs" / "fig_uniform" / "fig_FARvFRR_uniform.pdf")


# ============================================================
#       GAP vs VARIANCE (attacker always LEFT & overlapping)
# ============================================================

beta2 = 0.65
# attacker expands to the right but stays overlapping & mostly left of user
beta1_values = np.linspace(0.0, 0.25, 30)
gaps = []
vars_ = []

for beta1 in beta1_values:
    Topt, Teer, gap, var = compute_Topt_Teer(u1, u2, beta1, beta2)
    gaps.append(gap)
    vars_.append(var)

plt.figure(figsize=(7, 4))
plt.plot(vars_, gaps, linewidth=2, color="darkred")
plt.scatter(vars_, gaps, color="black")
plt.title("Gap Between $T_{opt}$ and $T_{EER}$ vs Variance (Uniform PDFs)")
plt.xlabel("Variance of Attacker PDF")
plt.ylabel(r"$|T_{opt} - T_{EER}|$")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figs" / "fig_uniform" / "fig_gap_uniform.pdf")


# ============================================================
#   VISUALIZE USER PDF AND ALL ATTACKER PDFs USED IN SWEEP
# ============================================================

plt.figure(figsize=(10, 5))

# Plot user PDF once
plt.plot(x, user_pdf_sym, label="User PDF $U[u_1,u_2]$", color="blue", linewidth=3)

colors = plt.cm.Reds(np.linspace(0.3, 1, len(beta1_values)))

for (beta1, c) in zip(beta1_values, colors):
    att_pdf = uniform_pdf(x, beta1, beta2)
    plt.plot(x, att_pdf, color=c, linewidth=1.5)

    # Shade overlap region between user and this attacker
    left_overlap  = max(u1, beta1)
    right_overlap = min(u2, beta2)

    if left_overlap < right_overlap:
        plt.fill_between(
            x,
            uniform_pdf(x, beta1, beta2),
            user_pdf_sym,
            where=(x >= left_overlap) & (x <= right_overlap),
            color=c,
            alpha=0.15
        )

plt.title("User PDF and Attacker PDFs (Sweep)")
plt.xlabel("t")
plt.ylabel("Probability Density")
plt.grid(True)

legend_elements = [
    Line2D([0], [0], color='blue', lw=3, label='User PDF'),
    Line2D([0], [0], color='red',  lw=2, label='Attacker PDFs (sweep)')
]
plt.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figs" / "fig_uniform" / "fig_uniforms_sweep.pdf")

u1_values = np.linspace(0.05, 0.75, 40)
deltas_u1 = []

u2 = 0.9
a1 = 0.1
a2 = 0.5

for u1 in u1_values:
    delta = compute_success_gap(u1, u2, a1, a2)
    deltas_u1.append(delta)

plt.figure(figsize=(7, 4))

plt.plot(u1_values, deltas_u1, linewidth=2, label=r"$\Delta$ vs $u_1$")
plt.scatter(u1_values, deltas_u1)

plt.xlabel(r"$u_1$")
plt.ylabel(r"$P_{\max} - P_{\mathrm{EER}}$")
plt.title("Improvement vs $u_1$")
plt.grid(True)

plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figs" / "fig_uniform" / "fig_delta_vs_u1.pdf")

a2_values = np.linspace(0.4, 0.75, 40)  
deltas_a2 = []

u1 = 0.2
u2 = 0.9
a1 = 0.05

for a2 in a2_values:
    delta = compute_success_gap(u1, u2, a1, a2)
    deltas_a2.append(delta)

plt.figure(figsize=(7, 4))

plt.plot(a2_values, deltas_a2, linewidth=2, label=r"$\Delta$ vs $a_2$")
plt.scatter(a2_values, deltas_a2)

plt.xlabel(r"$a_2$")
plt.ylabel(r"$P_{\max} - P_{\mathrm{EER}}$")
plt.title("Improvement vs decreasing $a_2$")
plt.grid(True)

plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figs" / "fig_uniform" / "fig_delta_vs_a2.pdf")

DELTA_U1_TXT = OUTPUT_DIR / "figs" / "fig_uniform" / "delta_vs_u1.txt"
DELTA_U1_TXT.parent.mkdir(parents=True, exist_ok=True)

DELTA_A2_TXT = OUTPUT_DIR / "figs" / "fig_uniform" / "delta_vs_a2.txt"
DELTA_A2_TXT.parent.mkdir(parents=True, exist_ok=True)

with open(DELTA_U1_TXT, "w") as f:
    f.write("u1 delta\n")
    for ui, di in zip(u1_values, deltas_u1):
        f.write(f"{ui:.4f} {di:.4f}\n")

with open(DELTA_A2_TXT, "w") as f:
    f.write("a2 delta\n")
    for ai, di in zip(a2_values, deltas_a2):
        f.write(f"{ai:.4f} {di:.4f}\n")


CASES_DATA_TXT = OUTPUT_DIR / "figs" / "fig_uniform" / "uniform_three_cases.txt"
CASES_DATA_TXT.parent.mkdir(parents=True, exist_ok=True)

# Each case is: (title, u1, u2, a1, a2, threshold t)
cases = [
    (
        r"(1) $a_2 < u_1$",
        0.60, 0.85,   # u1, u2
        0.15, 0.40,   # a1, a2
        0.50          # t, chosen such that a2 <= t <= u1
    ),
    (
        r"(2) $u_1 < a_1 < a_2 < u_2$",
        0.20, 0.85,   # u1, u2
        0.40, 0.65,   # a1, a2
        0.525         # t, chosen inside attacker interval
    ),
    (
        r"(3) $a_1 < u_1 < a_2 < u_2$",
        0.40, 0.85,   # u1, u2
        0.15, 0.65,   # a1, a2
        0.525         # t, chosen inside overlap
    ),
]

# ------------------------------------------------------------
# Save data for TikZ/PGFPlots if needed later
# ------------------------------------------------------------
with open(CASES_DATA_TXT, "w") as f:
    f.write(
        "s "
        "PU_case1 PA_case1 "
        "PU_case2 PA_case2 "
        "PU_case3 PA_case3\n"
    )

    pdf_values = []
    for _, u1_c, u2_c, a1_c, a2_c, _ in cases:
        PU_c = uniform_pdf(x, u1_c, u2_c)
        PA_c = uniform_pdf(x, a1_c, a2_c)
        pdf_values.append((PU_c, PA_c))

    for i, xi in enumerate(x):
        f.write(
            f"{xi:.4f} "
            f"{pdf_values[0][0][i]:.4f} {pdf_values[0][1][i]:.4f} "
            f"{pdf_values[1][0][i]:.4f} {pdf_values[1][1][i]:.4f} "
            f"{pdf_values[2][0][i]:.4f} {pdf_values[2][1][i]:.4f}\n"
        )

# ============================================================
#   P_MAX AND P_EER AS FUNCTIONS OF OVERLAP AREA
#   Output only: document/data txt file for TikZ
# ============================================================

def uniform_FRR(T, u1, u2):
    """
    FRR(T) = Pr[user score < T]
    Accept iff score >= T.
    """
    if T <= u1:
        return 0.0
    if T >= u2:
        return 1.0
    return (T - u1) / (u2 - u1)


def uniform_FAR(T, a1, a2):
    """
    FAR(T) = Pr[attacker score >= T]
    Accept iff score >= T.
    """
    if T <= a1:
        return 1.0
    if T >= a2:
        return 0.0
    return (a2 - T) / (a2 - a1)


def compute_uniform_success_points(u1, u2, a1, a2, N=5000):
    """
    Computes P_max and P_EER for two uniform distributions.
    """
    T_grid = np.linspace(min(a1, u1), max(a2, u2), N)

    FRR = np.array([uniform_FRR(T, u1, u2) for T in T_grid])
    FAR = np.array([uniform_FAR(T, a1, a2) for T in T_grid])

    # Single biometric credential success:
    # accept user and reject attacker
    P_success = (1 - FRR) * (1 - FAR)

    opt_idx = np.argmax(P_success)
    eer_idx = np.argmin(np.abs(FAR - FRR))

    P_max = P_success[opt_idx]
    P_eer = P_success[eer_idx]

    T_opt = T_grid[opt_idx]
    T_eer = T_grid[eer_idx]

    return P_max, P_eer, T_opt, T_eer


def support_overlap_length(u1, u2, a1, a2):
    """
    Length of the overlap between [u1,u2] and [a1,a2].
    """
    return max(0.0, min(u2, a2) - max(u1, a1))


# ============================================================
#   Sweep overlap area and save only TXT data
# ============================================================

OVERLAP_SUCCESS_TXT = OUTPUT_DIR / "figs" / "fig_uniform" / "overlap_success_data.txt"
OVERLAP_SUCCESS_TXT.parent.mkdir(parents=True, exist_ok=True)

# Fixed user distribution
user_width = 0.5
# Fixed attacker width
a1_overlap = 0.05
a2_overlap = 0.35

# Sweep overlap length from no overlap to full attacker-support overlap
overlap_lengths = np.linspace(0.0, 0.3, 100)

with open(OVERLAP_SUCCESS_TXT, "w") as f:
    f.write("overlap_area overlap_length P_max P_eer P_gap T_opt T_eer a1 a2 u1 u2\n")

    for overlap_len in overlap_lengths:
        u1_overlap = a2_overlap - overlap_len
        u2_overlap = u1_overlap + user_width
        

        overlap_length = support_overlap_length(
            u1_overlap, u2_overlap,
            a1_overlap, a2_overlap
        )

        overlap_area = overlap_length / 0.3

        P_max, P_eer, T_opt, T_eer = compute_uniform_success_points(
            u1_overlap, u2_overlap,
            a1_overlap, a2_overlap
        )

        P_gap = P_max - P_eer

        f.write(
            f"{overlap_area:.4f} "
            f"{overlap_length:.4f} "
            f"{P_max:.4f} "
            f"{P_eer:.4f} "
            f"{P_gap:.4f} "
            f"{T_opt:.4f} "
            f"{T_eer:.4f} "
            f"{a1_overlap:.4f} "
            f"{a2_overlap:.4f} "
            f"{u1_overlap:.4f} "
            f"{u2_overlap:.4f}\n"
        )

print(f"[i] Data saved to: {OVERLAP_SUCCESS_TXT}")
print(f"[i] Data saved to: {CASES_DATA_TXT}")
print(f"[i] Data saved to: {DELTA_U1_TXT}")
print(f"[i] Data saved to: {DELTA_A2_TXT}")
print("Saved all uniform figures:")
print("  - fig_uniforms.pdf")
print("  - fig_success_uniform.pdf")
print("  - fig_FARvFRR_uniform.pdf")
print("  - fig_gap_uniform.pdf")
print("  - fig_uniforms_sweep.pdf")
print("  - fig_delta_vs_u1.pdf")