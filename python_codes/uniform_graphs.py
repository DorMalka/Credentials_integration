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
        f.write(f"{xi:.6f} {pu:.6f} {pa:.6f}\n")

with open(DATA_TXT_2, "w") as f:
    f.write("s PU PA_sym PA_asym\n")
    for xi, pu, pa_sym, pa_asym in zip(x, PU, PA_sym, PA_asym):
        f.write(f"{xi:.6f} {pu:.6f} {pa_sym:.6f} {pa_asym:.6f}\n")

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

SUCCESS_DATA_TXT = OUTPUT_DIR / "figs" / "fig_uniform" / "success_uniform_data.txt"
SUCCESS_DATA_TXT.parent.mkdir(parents=True, exist_ok=True)

with open(SUCCESS_DATA_TXT, "w") as f:
    f.write("s FRR FAR loss leak theft safe FAR_opt FRR_opt FAR_eer FRR_eer\n")
    for xi, frr, far, lo, le, th, sa in zip(x, FRR, FAR, loss, leak, theft, safe):
        f.write(
            f"{xi:.6f} "
            f"{frr:.6f} "
            f"{far:.6f} "
            f"{lo:.6f} "
            f"{le:.6f} "
            f"{th:.6f} "
            f"{sa:.6f} "
            f"{FAR_opt:.6f} "
            f"{FRR_opt:.6f} "
            f"{FAR_eer:.6f} "
            f"{FRR_eer:.6f}\n"
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

print("Saved all uniform figures:")
print("  - fig_uniforms.pdf")
print("  - fig_success_uniform.pdf")
print("  - fig_FARvFRR_uniform.pdf")
print("  - fig_gap_uniform.pdf")
print("  - fig_uniforms_sweep.pdf")
