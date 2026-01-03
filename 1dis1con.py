import numpy as np
import matplotlib.pyplot as plt

# =========================
# Roots (uniform supports)
# =========================
u1, u2 = 0.50, 0.95   # user
a1, a2 = 0.45, 0.80   # attacker

# =========================
# Wallet-2 parameters (original: safe/loss/leak)
# =========================
k1, k2, k3 = 0.50, 0.05, 0.45   # safe2, loss2, leak2

# =========================
# Wallet-2 variant A: safe/loss only
# (move leak mass into loss)
# =========================
k1_sl = k1
k2_sl = k2 + k3
k3_sl = 0.0

# =========================
# Wallet-2 variant B: safe/leak only
# (move loss mass into leak)
# =========================
k1_sk = k1
k2_sk = 0.0
k3_sk = k3 + k2

# =========================
# Wallet-2 variant C: safe/theft only
# (loss+leak mass becomes "theft", which does NOT contribute to success)
# =========================
k1_st = k1
k2_st = 0.0
k3_st = 0.0
kT_st = 1.0 - k1_st  # theft (tracked for completeness)

# =========================
# FAR / FRR
# =========================
def FAR(T):
    T = np.asarray(T)
    out = np.zeros_like(T, dtype=float)
    out[T <= a1] = 1.0
    mid = (T > a1) & (T <= a2)
    out[mid] = (a2 - T[mid]) / (a2 - a1)
    out[T > a2] = 0.0
    return out

def FRR(T):
    T = np.asarray(T)
    out = np.zeros_like(T, dtype=float)
    out[T <= u1] = 0.0
    mid = (T > u1) & (T <= u2)
    out[mid] = (T[mid] - u1) / (u2 - u1)
    out[T > u2] = 1.0
    return out

# =========================
# Wallet 1
# =========================
def safe1(T):
    return (1 - FAR(T)) * (1 - FRR(T))

def loss1(T):
    return FRR(T) * (1 - FAR(T))

def leak1(T):
    return FAR(T) * (1 - FRR(T))

# =========================
# AND / OR success — original wallet-2
# =========================
def P_AND(T):
    return safe1(T) * k1 + safe1(T) * k3 + k1 * leak1(T)

def P_OR(T):
    return safe1(T) * k1 + safe1(T) * k2 + k1 * loss1(T)

# =========================
# AND / OR success — variant A (safe/loss only)
# =========================
def P_AND_SL(T):
    return safe1(T) * k1_sl + safe1(T) * k3_sl + k1_sl * leak1(T)

def P_OR_SL(T):
    return safe1(T) * k1_sl + safe1(T) * k2_sl + k1_sl * loss1(T)

# =========================
# AND / OR success — variant B (safe/leak only)
# =========================
def P_AND_SK(T):
    return safe1(T) * k1_sk + safe1(T) * k3_sk + k1_sk * leak1(T)

def P_OR_SK(T):
    return safe1(T) * k1_sk + safe1(T) * k2_sk + k1_sk * loss1(T)

# =========================
# AND / OR success — variant C (safe/theft only)
# =========================
def P_AND_ST(T):
    return safe1(T) * k1_st + safe1(T) * k3_st + k1_st * leak1(T)

def P_OR_ST(T):
    return safe1(T) * k1_st + safe1(T) * k2_st + k1_st * loss1(T)

# =========================
# Threshold sweep
# =========================
T = np.linspace(min(a1, u1) - 0.1, max(a2, u2) + 0.1, 4001)
fa, fr = FAR(T), FRR(T)

# =========================
# EER
# =========================
idx_eer = np.argmin(np.abs(fa - fr))
T_eer = T[idx_eer]

# =========================
# Compute curves + optima
# =========================
p_or     = P_OR(T)
p_and    = P_AND(T)

p_or_sl  = P_OR_SL(T)
p_and_sl = P_AND_SL(T)

p_or_sk  = P_OR_SK(T)
p_and_sk = P_AND_SK(T)

p_or_st  = P_OR_ST(T)
p_and_st = P_AND_ST(T)

idx_opt_or = np.argmax(p_or)
T_opt_or, P_opt_or = T[idx_opt_or], p_or[idx_opt_or]

idx_opt_and = np.argmax(p_and)
T_opt_and, P_opt_and = T[idx_opt_and], p_and[idx_opt_and]

idx_opt_or_sl = np.argmax(p_or_sl)
T_opt_or_sl, P_opt_or_sl = T[idx_opt_or_sl], p_or_sl[idx_opt_or_sl]

idx_opt_and_sl = np.argmax(p_and_sl)
T_opt_and_sl, P_opt_and_sl = T[idx_opt_and_sl], p_and_sl[idx_opt_and_sl]

idx_opt_or_sk = np.argmax(p_or_sk)
T_opt_or_sk, P_opt_or_sk = T[idx_opt_or_sk], p_or_sk[idx_opt_or_sk]

idx_opt_and_sk = np.argmax(p_and_sk)
T_opt_and_sk, P_opt_and_sk = T[idx_opt_and_sk], p_and_sk[idx_opt_and_sk]

idx_opt_or_st = np.argmax(p_or_st)
T_opt_or_st, P_opt_or_st = T[idx_opt_or_st], p_or_st[idx_opt_or_st]

idx_opt_and_st = np.argmax(p_and_st)
T_opt_and_st, P_opt_and_st = T[idx_opt_and_st], p_and_st[idx_opt_and_st]

# =========================================================
# (1) FIGURE — ORIGINAL OR WALLET (unchanged filename)
# =========================================================
plt.figure(figsize=(6.5, 4.5))
plt.plot(T, p_or, label="P_success_OR(T)")
plt.scatter(T_eer, p_or[idx_eer], zorder=5)
plt.scatter(T_opt_or, P_opt_or, zorder=6, label="Optimal")

plt.xlabel("Threshold T")
plt.ylabel("Probability of success")
plt.title("OR Wallet — Success Probability")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("OR_wallet_1dis_1con_success.pdf")
plt.close()

# =========================================================
# (2) FIGURE — ORIGINAL AND WALLET (unchanged filename)
# =========================================================
plt.figure(figsize=(6.5, 4.5))
plt.plot(T, p_and, label="P_success_AND(T)")
plt.scatter(T_eer, p_and[idx_eer], zorder=5)
plt.scatter(T_opt_and, P_opt_and, zorder=6, label="Optimal")

plt.xlabel("Threshold T")
plt.ylabel("Probability of success")
plt.title("AND Wallet — Success Probability")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("AND_wallet_1dis_1con_success.pdf")
plt.close()

# =========================================================
# (3) FIGURE — OR WALLET (original vs safe/loss only)
# =========================================================
plt.figure(figsize=(6.5, 4.5))
plt.plot(T, p_or, label="OR: Wallet-2 (safe/loss/leak)")
plt.plot(T, p_or_sl, label="OR: Wallet-2 (safe/loss only)")
plt.scatter(T_eer, p_or[idx_eer], zorder=5, label="EER (on original OR)")
plt.scatter(T_opt_or, P_opt_or, zorder=6, label="Optimal (orig OR)")
plt.scatter(T_opt_or_sl, P_opt_or_sl, zorder=6, label="Optimal (safe/loss OR)")

plt.xlabel("Threshold T")
plt.ylabel("Probability of success")
plt.title("OR Wallet — Original vs Safe/Loss-Only Wallet-2")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("OR_wallet_compare_safe_loss_only.pdf")
plt.close()

# =========================================================
# (4) FIGURE — AND WALLET (original vs safe/loss only)
# =========================================================
plt.figure(figsize=(6.5, 4.5))
plt.plot(T, p_and, label="AND: Wallet-2 (safe/loss/leak)")
plt.plot(T, p_and_sl, label="AND: Wallet-2 (safe/loss only)")
plt.scatter(T_eer, p_and[idx_eer], zorder=5, label="EER (on original AND)")
plt.scatter(T_opt_and, P_opt_and, zorder=6, label="Optimal (orig AND)")
plt.scatter(T_opt_and_sl, P_opt_and_sl, zorder=6, label="Optimal (safe/loss AND)")

plt.xlabel("Threshold T")
plt.ylabel("Probability of success")
plt.title("AND Wallet — Original vs Safe/Loss-Only Wallet-2")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("AND_wallet_compare_safe_loss_only.pdf")
plt.close()

# =========================================================
# (5) FIGURE — OR WALLET (original vs safe/leak only)
# =========================================================
plt.figure(figsize=(6.5, 4.5))
plt.plot(T, p_or, label="OR: Wallet-2 (safe/loss/leak)")
plt.plot(T, p_or_sk, label="OR: Wallet-2 (safe/leak only)")
plt.scatter(T_eer, p_or[idx_eer], zorder=5, label="EER (on original OR)")
plt.scatter(T_opt_or, P_opt_or, zorder=6, label="Optimal (orig OR)")
plt.scatter(T_opt_or_sk, P_opt_or_sk, zorder=6, label="Optimal (safe/leak OR)")

plt.xlabel("Threshold T")
plt.ylabel("Probability of success")
plt.title("OR Wallet — Original vs Safe/Leak-Only Wallet-2")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("OR_wallet_compare_safe_leak_only.pdf")
plt.close()

# =========================================================
# (6) FIGURE — AND WALLET (original vs safe/leak only)
# =========================================================
plt.figure(figsize=(6.5, 4.5))
plt.plot(T, p_and, label="AND: Wallet-2 (safe/loss/leak)")
plt.plot(T, p_and_sk, label="AND: Wallet-2 (safe/leak only)")
plt.scatter(T_eer, p_and[idx_eer], zorder=5, label="EER (on original AND)")
plt.scatter(T_opt_and, P_opt_and, zorder=6, label="Optimal (orig AND)")
plt.scatter(T_opt_and_sk, P_opt_and_sk, zorder=6, label="Optimal (safe/leak AND)")

plt.xlabel("Threshold T")
plt.ylabel("Probability of success")
plt.title("AND Wallet — Original vs Safe/Leak-Only Wallet-2")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("AND_wallet_compare_safe_leak_only.pdf")
plt.close()

# =========================================================
# (7) FIGURE — OR WALLET (original vs safe/theft only)
# =========================================================
plt.figure(figsize=(6.5, 4.5))
plt.plot(T, p_or, label="OR: Wallet-2 (safe/loss/leak)")
plt.plot(T, p_or_st, label="OR: Wallet-2 (safe/theft only)")
plt.scatter(T_eer, p_or[idx_eer], zorder=5, label="EER (on original OR)")
plt.scatter(T_opt_or, P_opt_or, zorder=6, label="Optimal (orig OR)")
plt.scatter(T_opt_or_st, P_opt_or_st, zorder=6, label="Optimal (safe/theft OR)")

plt.xlabel("Threshold T")
plt.ylabel("Probability of success")
plt.title("OR Wallet — Original vs Safe/Theft-Only Wallet-2")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("OR_wallet_compare_safe_theft_only.pdf")
plt.close()

# =========================================================
# (8) FIGURE — AND WALLET (original vs safe/theft only)
# =========================================================
plt.figure(figsize=(6.5, 4.5))
plt.plot(T, p_and, label="AND: Wallet-2 (safe/loss/leak)")
plt.plot(T, p_and_st, label="AND: Wallet-2 (safe/theft only)")
plt.scatter(T_eer, p_and[idx_eer], zorder=5, label="EER (on original AND)")
plt.scatter(T_opt_and, P_opt_and, zorder=6, label="Optimal (orig AND)")
plt.scatter(T_opt_and_st, P_opt_and_st, zorder=6, label="Optimal (safe/theft AND)")

plt.xlabel("Threshold T")
plt.ylabel("Probability of success")
plt.title("AND Wallet — Original vs Safe/Theft-Only Wallet-2")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("AND_wallet_compare_safe_theft_only.pdf")
plt.close()


a1_base, a2_base = a1, a2
mid_attacker = 0.5 * (a1_base + a2_base)

# Ordered widths: start small and grow each iteration (>=10)
widths = np.linspace(0.10, 0.60, 10)  # a2-a1 increases monotonically

attacker_roots = []
for w in widths:
    a1_i = mid_attacker - 0.5 * w
    a2_i = mid_attacker + 0.5 * w
    # clamp to [0,1] just in case
    a1_i = max(0.0, a1_i)
    a2_i = min(1.0, a2_i)
    if a2_i - a1_i > 1e-9:
        attacker_roots.append((a1_i, a2_i))

# Common T grid covering all intervals
a1_min = min(x for x, _ in attacker_roots)
a2_max = max(y for _, y in attacker_roots)
T_sweep = np.linspace(min(a1_min, u1) - 0.1, max(a2_max, u2) + 0.1, 5001)

def FAR_uniform_roots(T, a1_i, a2_i):
    T = np.asarray(T, dtype=float)
    out = np.zeros_like(T, dtype=float)
    out[T <= a1_i] = 1.0
    mid = (T > a1_i) & (T <= a2_i)
    out[mid] = (a2_i - T[mid]) / (a2_i - a1_i)
    out[T > a2_i] = 0.0
    return out

def safe1_roots(T, a1_i, a2_i):
    return (1 - FAR_uniform_roots(T, a1_i, a2_i)) * (1 - FRR(T))

def loss1_roots(T, a1_i, a2_i):
    return FRR(T) * (1 - FAR_uniform_roots(T, a1_i, a2_i))

def leak1_roots(T, a1_i, a2_i):
    return FAR_uniform_roots(T, a1_i, a2_i) * (1 - FRR(T))

def P_OR_roots(T, a1_i, a2_i):
    s1 = safe1_roots(T, a1_i, a2_i)
    l1 = loss1_roots(T, a1_i, a2_i)
    return s1 * k1 + s1 * k2 + k1 * l1

def P_AND_roots(T, a1_i, a2_i):
    s1 = safe1_roots(T, a1_i, a2_i)
    lk1 = leak1_roots(T, a1_i, a2_i)
    return s1 * k1 + s1 * k3 + k1 * lk1

plt.figure(figsize=(7.4, 4.9))
for (a1_i, a2_i) in attacker_roots:
    p = P_OR_roots(T_sweep, a1_i, a2_i)
    idx_opt = np.argmax(p)
    T_opt, P_opt = T_sweep[idx_opt], p[idx_opt]

    w = a2_i - a1_i
    plt.plot(T_sweep, p, label=f"U[{a1_i:.3f},{a2_i:.3f}] (w={w:.2f})")
    plt.scatter(T_opt, P_opt, zorder=6)

plt.xlabel("Threshold T")
plt.ylabel("Probability of success")
plt.title("OR Wallet — Success vs Attacker Uniform Width (ordered small→large)")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("OR_wallet_attacker_roots_ordered_width.pdf")
plt.close()

# =========================================================
# (10) FIGURE — AND wallet: attacker roots sweep (uniform, ordered width small→large)
# =========================================================
plt.figure(figsize=(7.4, 4.9))
for (a1_i, a2_i) in attacker_roots:
    p = P_AND_roots(T_sweep, a1_i, a2_i)
    idx_opt = np.argmax(p)
    T_opt, P_opt = T_sweep[idx_opt], p[idx_opt]

    w = a2_i - a1_i
    plt.plot(T_sweep, p, label=f"U[{a1_i:.3f},{a2_i:.3f}] (w={w:.2f})")
    plt.scatter(T_opt, P_opt, zorder=6)

plt.xlabel("Threshold T")
plt.ylabel("Probability of success")
plt.title("AND Wallet — Success vs Attacker Uniform Width (ordered small→large)")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("AND_wallet_attacker_roots_ordered_width.pdf")
plt.close()

# Restore original attacker roots (optional)
a1, a2 = a1_base, a2_base