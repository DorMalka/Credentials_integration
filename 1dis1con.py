import numpy as np
import matplotlib.pyplot as plt

# =========================
# Roots (uniform supports)
# =========================
u1, u2 = 0.50, 0.95   # user support
a1, a2 = 0.45, 0.8   # attacker support

# =========================
# Wallet-2 parameters
# =========================
k1, k2, k3 = 0.45, 0.10, 0.45   # safe2, loss2

# =========================
# FAR / FRR
# =========================
def FAR(T):
    T = np.asarray(T)
    out = np.zeros_like(T)
    out[T <= a1] = 1.0
    mid = (T > a1) & (T <= a2)
    out[mid] = (a2 - T[mid]) / (a2 - a1)
    out[T > a2] = 0.0
    return out

def FRR(T):
    T = np.asarray(T)
    out = np.zeros_like(T)
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
# AND / OR success
# =========================
def P_AND(T):
    return safe1(T) * k1 + safe1(T) * k3 + k1 * leak1(T)

def P_OR(T):
    return safe1(T) * k1 + safe1(T) * k2 + k1 * loss1(T)

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
# AND wallet
# =========================
p_and = P_AND(T)
idx_opt_and = np.argmax(p_and)
T_opt_and, P_opt_and = T[idx_opt_and], p_and[idx_opt_and]

# =========================
# OR wallet
# =========================
p_or = P_OR(T)
idx_opt_or = np.argmax(p_or)
T_opt_or, P_opt_or = T[idx_opt_or], p_or[idx_opt_or]

# =========================================================
# FIGURE 1 — OR WALLET
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
# FIGURE 2 — AND WALLET
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