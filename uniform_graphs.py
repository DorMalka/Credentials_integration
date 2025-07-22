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

#symmetric
plt.subplot(1, 2, 1)
plt.plot(x, user_pdf_sym, label="User PDF", color='blue', linewidth=2)
plt.plot(x, attacker_pdf_sym, label="Attacker PDF", color='red', linewidth=2)
plt.title("a. Symmetric PDFs")
plt.xlabel("t")
plt.ylabel("Probability Density")
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(
    ticks=[0.0, uq, uw, a1_sym, a2_sym, 1.0],
    labels=["0", r"$u_1$", r"$a_1$", r"$u_2$", r"$a_2$", "1"]
)

#asymmetric
plt.subplot(1, 2, 2)
plt.plot(x, user_pdf_sym, label="User PDF", color='blue', linewidth=2)
plt.plot(x, attacker_pdf_asym, label="Attacker PDF", color='red', linewidth=2)
plt.title("b. Asymmetric PDFs")
plt.xlabel("t")
plt.ylabel("Probability Density")
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(
    ticks=[0.0, uq, uw, a1_asym, a2_asym, 1.0],
    labels=["0", r"$u_1$", r"$a_1$", r"$u_2$", r"$a_2$", "1"]
)

plt.tight_layout()
plt.savefig('fig_uniforms.pdf') 
