import numpy as np
import matplotlib.pyplot as plt
uq, uw = 0.3, 0.7  
a1_sym, a2_sym = 0.5, 0.9  
a1_asym, a2_asym = 0.4, 0.95  
x = np.linspace(0, 1, 1000)
def uniform_pdf(x, a, b):
    return np.where((x >= a) & (x <= b), 1 / (b - a), 0)

user_pdf_sym = uniform_pdf(x, uq, uw)
attacker_pdf_sym = uniform_pdf(x, a1_sym, a2_sym)


attacker_pdf_asym = uniform_pdf(x, a1_asym, a2_asym)


plt.figure(figsize=(10, 5))
max_u = 1/0.4
max_a = 1/0.4
#symmetric
plt.subplot(1, 2, 1)
plt.plot(x, user_pdf_sym, label="User PDF", color='blue', linewidth=2)
plt.plot(x, attacker_pdf_sym, label="Attacker PDF", color='red', linewidth=2)
plt.title("a. Symmetric PDFs")
plt.xlabel("t")
plt.ylabel("Probability Density",labelpad = -60)
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(
    ticks=[0.0, uq, uw, a1_sym, a2_sym],
    labels=["0", r"$u_1$", r"$u_2$", r"$a_1$", r"$a_2$"]
)
plt.yticks(
    ticks=[0.0, max_u],
    labels=["0", r"$\frac{1}{u_2-u_1}$=$\frac{1}{a_2-a_1}$"]
) 


map_a = 1/0.55
#asymmetric
plt.subplot(1, 2, 2)
plt.plot(x, user_pdf_sym, label="User PDF", color='blue', linewidth=2)
plt.plot(x, attacker_pdf_asym, label="Attacker PDF", color='red', linewidth=2)
plt.title("b. Asymmetric PDFs")
plt.xlabel("t")
plt.ylabel("Probability Density",labelpad = -20)
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(
    ticks=[0.0, uq, uw, a1_asym, a2_asym],
    labels=["0", r"$u_1$", r"$u_2$", r"$a_1$", r"$a_2$"]
)
plt.yticks(
    ticks=[0.0, max_u, map_a],
    labels=["0", r"$\frac{1}{u_2-u_1}$", r"$\frac{1}{a_2-a_1}$"]
) 

plt.tight_layout()
plt.savefig('fig_uniforms.pdf') 

a1_asym = 0.47
a2_asym = 0.81
attacker_pdf_asym = uniform_pdf(x, a1_asym, a2_asym)
FRR = np.array([
    np.trapezoid(user_pdf_sym[(x >= T) & (x <= uw)], x[(x >= T) & (x <= uw)]) if T < uw else 0.0
    for T in x
])

FAR = np.array([
    np.trapezoid(attacker_pdf_asym[(x >= a1_asym) & (x <= T)], x[(x >= a1_asym) & (x <= T)]) if T > a1_asym else 0.0
    for T in x
])

loss = FRR * (1 - FAR)
leak = FAR * (1 - FRR)
theft = FRR * FAR
safe = 1 - loss - leak - theft
max_idx = np.argmax(safe)
best_T = x[max_idx]
best_success = safe[max_idx] * 100
idx_safe_1 = np.argmin(np.abs(x - a1_asym))
idx_safe_2 = np.argmin(np.abs(x - uw))
safe_1 = safe[idx_safe_1]
safe_2 = safe[idx_safe_2]
# Optional: Plot the success curve
plt.figure(figsize=(10, 5))
plt.plot(x, safe, label="Wallet Success", color="purple")
plt.title("Uniform - Success Function")
plt.xlabel("Threshold T")
plt.ylabel("Success",labelpad=-100)
plt.legend()
plt.xticks(
    ticks=[0.0, uq, uw, a1_asym , best_T, a2_asym],
    labels=["0", r"$u_1$", r"$u_2$",r"$a_1$",r"$T_\text{opt}$", r"$a_2$"]
)
plt.yticks(
    ticks=[safe_1.item(), safe[max_idx], safe_2.item()],
    labels=[
        r"$\frac{a_1 - u_1}{u_2 - u_1}$",
        r"$\frac{(a_2 - u_1)^2}{4(u_2 - u_1) \cdot(a_2 - a_1)} $",
        r"$\frac{a_2 - u_2}{a_2 - a_1}$",
    ],
    fontsize = 14
)
plt.grid(True)
plt.tight_layout()
plt.savefig('fig_success_uniform.pdf') 

FAR_idx = np.argmin(np.abs(x - best_T))
FRR_idx = np.argmin(np.abs(x - best_T))
FAR_idx_r = np.argmin(np.abs(x - uw))
FRR_idx_r = np.argmin(np.abs(x - a1_asym))
FAR_opt = FAR[FAR_idx]
FRR_opt = FRR[FRR_idx]
FAR_root = FAR[FAR_idx_r]
FRR_root = FRR[FRR_idx_r]
plt.figure(figsize=(8, 4))
plt.plot(FRR, FAR, label="FAR vs FRR", color='purple', linewidth=2)
plt.scatter(FRR_opt, FAR_opt, color='black', s=80, zorder=5, label='Optimal Point')
plt.annotate(
    r'$(FRR_{opt}, FAR_{opt})$',
    xy=(FRR_opt, FAR_opt),
    xytext=(FRR_opt + 0.025, FAR_opt + 0.025),
    fontsize=12
)
plt.title("FAR vs FRR Curve")
plt.xlabel("FRR")
plt.ylabel("FAR",labelpad=-55)
plt.legend()
plt.xticks(
    ticks=[0.0, FRR_root, FRR_opt, 1.0],
    labels=["0", r"$\frac{u_2 - a_1}{u_2 - u_1}$", r"$\frac{u_2-\frac{a_2}{2}-\frac{u_1}{2}}{u_2-u_1}$", "1"],fontsize = 14
)
plt.yticks(
    ticks=[0.0, FAR_root, FAR_opt, 1.0],
    labels=["0", r"$\frac{u_2 - a_1}{a_2 - a_1}$", r"$\frac{\frac{a_2}{2}+\frac{u_1}{2}-a_1}{a_2-a_1}$", "1"
    ],
    fontsize = 14
)
plt.grid(True)
plt.tight_layout()
plt.savefig('fig_FARvFRR_uniform.pdf') 
