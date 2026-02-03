import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Union, Any, Optional, Literal

from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad

import matplotlib.pyplot as plt


Mode = Literal["AND", "OR"]


@dataclass(frozen=True)
class GaussianPair:
    # user ~ N(mu_u, sigma_u), attacker ~ N(mu_a, sigma_a)
    mu_u: float
    sigma_u: float
    mu_a: float
    sigma_a: float


# =========================
# Core math
# =========================
def _check_pair(p: GaussianPair) -> None:
    if p.sigma_u <= 0 or p.sigma_a <= 0:
        raise ValueError("sigmas must be > 0")


def far_frr(T: Union[float, np.ndarray], p: GaussianPair) -> Tuple[np.ndarray, np.ndarray]:
    """
    FAR(T) = ∫_{T}^{∞} f_attacker(x) dx = 1 - CDF_a(T)
    FRR(T) = ∫_{-∞}^{T} f_user(x) dx = CDF_u(T)
    """
    T = np.asarray(T, dtype=float)
    FAR = 1.0 - norm.cdf(T, loc=p.mu_a, scale=p.sigma_a)
    FRR = norm.cdf(T, loc=p.mu_u, scale=p.sigma_u)
    return np.clip(FAR, 0.0, 1.0), np.clip(FRR, 0.0, 1.0)


def safe_leak_loss(
    FAR: Union[float, np.ndarray],
    FRR: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    safe = (1-FAR)*(1-FRR)
    leak = FAR*(1-FRR)
    loss = FRR*(1-FAR)
    """
    FAR = np.asarray(FAR, dtype=float)
    FRR = np.asarray(FRR, dtype=float)
    safe = (1.0 - FAR) * (1.0 - FRR)
    leak = FAR * (1.0 - FRR)
    loss = FRR * (1.0 - FAR)
    return safe, leak, loss


def success_probability(T1: float, T2: float, p1: GaussianPair, p2: GaussianPair, mode: Mode) -> float:
    FAR1, FRR1 = far_frr(T1, p1)
    FAR2, FRR2 = far_frr(T2, p2)

    safe1, leak1, loss1 = safe_leak_loss(FAR1, FRR1)
    safe2, leak2, loss2 = safe_leak_loss(FAR2, FRR2)

    safe1, leak1, loss1 = float(safe1), float(leak1), float(loss1)
    safe2, leak2, loss2 = float(safe2), float(leak2), float(loss2)

    if mode == "AND":
        # P^AND_success = safe1*safe2 + safe1*leak2 + safe2*leak1
        return safe1 * safe2 + safe1 * leak2 + safe2 * leak1
    elif mode == "OR":
        # P^OR_success = safe1*safe2 + safe1*loss2 + safe2*loss1
        return safe1 * safe2 + safe1 * loss2 + safe2 * loss1
    else:
        raise ValueError("mode must be 'AND' or 'OR'")


def _derivatives_wrt_T(T: float, p: GaussianPair) -> Dict[str, float]:
    """
    Derivatives wrt T:
      dFAR/dT = -pdf_attacker(T)
      dFRR/dT =  pdf_user(T)
    """
    pdf_a = float(norm.pdf(T, loc=p.mu_a, scale=p.sigma_a))
    pdf_u = float(norm.pdf(T, loc=p.mu_u, scale=p.sigma_u))

    FAR, FRR = far_frr(T, p)
    FAR, FRR = float(FAR), float(FRR)

    dFAR = -pdf_a
    dFRR = +pdf_u

    omFAR = 1.0 - FAR
    omFRR = 1.0 - FRR

    safe = omFAR * omFRR
    leak = FAR * omFRR
    loss = FRR * omFAR

    dsafe = (-dFAR) * omFRR + omFAR * (-dFRR)
    dleak = dFAR * omFRR + FAR * (-dFRR)
    dloss = dFRR * omFAR + FRR * (-dFAR)

    return {
        "FAR": FAR,
        "FRR": FRR,
        "safe": safe,
        "leak": leak,
        "loss": loss,
        "dsafe": dsafe,
        "dleak": dleak,
        "dloss": dloss,
    }


def optimize_thresholds_lbfgs(
    p1: GaussianPair,
    p2: GaussianPair,
    mode: Mode = "AND",
    x0: Optional[Tuple[float, float]] = None,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Maximize P_success(T1,T2) using L-BFGS-B (2 variables).
    We minimize -P_success.
    """
    _check_pair(p1)
    _check_pair(p2)

    if x0 is None:
        x0 = ((p1.mu_u + p1.mu_a) / 2.0, (p2.mu_u + p2.mu_a) / 2.0)

    if bounds is None:
        def auto_bounds(p: GaussianPair) -> Tuple[float, float]:
            s = 8.0 * max(p.sigma_u, p.sigma_a)
            lo = min(p.mu_u, p.mu_a) - s
            hi = max(p.mu_u, p.mu_a) + s
            return (lo, hi)
        bounds = (auto_bounds(p1), auto_bounds(p2))

    def obj(x: np.ndarray) -> float:
        T1, T2 = float(x[0]), float(x[1])
        return -success_probability(T1, T2, p1, p2, mode)

    def jac(x: np.ndarray) -> np.ndarray:
        T1, T2 = float(x[0]), float(x[1])

        d1 = _derivatives_wrt_T(T1, p1)
        d2 = _derivatives_wrt_T(T2, p2)

        safe1, leak1, loss1 = d1["safe"], d1["leak"], d1["loss"]
        safe2, leak2, loss2 = d2["safe"], d2["leak"], d2["loss"]

        dsafe1, dleak1, dloss1 = d1["dsafe"], d1["dleak"], d1["dloss"]
        dsafe2, dleak2, dloss2 = d2["dsafe"], d2["dleak"], d2["dloss"]

        if mode == "AND":
            # P = safe1*safe2 + safe1*leak2 + safe2*leak1
            dP_dT1 = dsafe1 * safe2 + dsafe1 * leak2 + safe2 * dleak1
            dP_dT2 = safe1 * dsafe2 + safe1 * dleak2 + dsafe2 * leak1
        elif mode == "OR":
            # P = safe1*safe2 + safe1*loss2 + safe2*loss1
            dP_dT1 = dsafe1 * safe2 + dsafe1 * loss2 + safe2 * dloss1
            dP_dT2 = safe1 * dsafe2 + safe1 * dloss2 + dsafe2 * loss1
        else:
            raise ValueError("mode must be 'AND' or 'OR'")

        return np.array([-dP_dT1, -dP_dT2], dtype=float)

    res = minimize(
        fun=obj,
        x0=np.array(x0, dtype=float),
        jac=jac,
        method="L-BFGS-B",
        bounds=[bounds[0], bounds[1]],
        options={"maxiter": 500, "ftol": 1e-12},
    )

    T1_opt, T2_opt = float(res.x[0]), float(res.x[1])
    P_opt = success_probability(T1_opt, T2_opt, p1, p2, mode)

    FAR1, FRR1 = far_frr(T1_opt, p1)
    FAR2, FRR2 = far_frr(T2_opt, p2)

    return {
        "mode": mode,
        "T_opt": (T1_opt, T2_opt),
        "P_success_opt": float(P_opt),
        "FAR_opt": (float(FAR1), float(FAR2)),
        "FRR_opt": (float(FRR1), float(FRR2)),
        "optimizer_success": bool(res.success),
        "message": res.message,
        "nit": int(getattr(res, "nit", -1)),
        "fun": float(res.fun),
    }


# =========================
# Diagnostics
# =========================
def check_normalization(p: GaussianPair) -> None:
    f_u = lambda x: norm.pdf(x, p.mu_u, p.sigma_u)
    f_a = lambda x: norm.pdf(x, p.mu_a, p.sigma_a)
    iu, _ = quad(f_u, -np.inf, np.inf)
    ia, _ = quad(f_a, -np.inf, np.inf)
    print("User integral:", iu)
    print("Attacker integral:", ia)


# =========================
# EER helpers (per-credential)
# =========================
def auto_T_grid_1d(p: GaussianPair, k: float = 6.0, n: int = 2000) -> np.ndarray:
    s = k * max(p.sigma_u, p.sigma_a)
    lo = min(p.mu_u, p.mu_a) - s
    hi = max(p.mu_u, p.mu_a) + s
    return np.linspace(lo, hi, n)


def eer_from_grid(p: GaussianPair, T: Optional[np.ndarray] = None) -> Dict[str, float]:
    if T is None:
        T = auto_T_grid_1d(p)
    FAR, FRR = far_frr(T, p)
    idx = int(np.argmin(np.abs(FAR - FRR)))
    return {"T_eer": float(T[idx]), "FAR_eer": float(FAR[idx]), "FRR_eer": float(FRR[idx])}


# =========================
# Plotting requested: subplots for Gaussian PDFs and FAR-vs-FRR
# =========================
def shared_x_grid(p1: GaussianPair, p2: GaussianPair, k: float = 6.0, n: int = 2000) -> np.ndarray:
    sig_max = max(p1.sigma_u, p1.sigma_a, p2.sigma_u, p2.sigma_a)
    mu_min = min(p1.mu_u, p1.mu_a, p2.mu_u, p2.mu_a)
    mu_max = max(p1.mu_u, p1.mu_a, p2.mu_u, p2.mu_a)
    lo = mu_min - k * sig_max
    hi = mu_max + k * sig_max
    return np.linspace(lo, hi, n)


def plot_gaussians_subplots_2cred(
    p1: GaussianPair,
    p2: GaussianPair,
    out_pdf: str = "01_gaussians.pdf",
) -> None:
    x = shared_x_grid(p1, p2)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax = axes[0]
    ax.plot(x, norm.pdf(x, loc=p1.mu_u, scale=p1.sigma_u), label="User PDF")
    ax.plot(x, norm.pdf(x, loc=p1.mu_a, scale=p1.sigma_a), label="Attacker PDF")
    ax.set_title("Credential 1: Gaussian PDFs")
    ax.set_ylabel("density")
    ax.legend()

    ax = axes[1]
    ax.plot(x, norm.pdf(x, loc=p2.mu_u, scale=p2.sigma_u), label="User PDF")
    ax.plot(x, norm.pdf(x, loc=p2.mu_a, scale=p2.sigma_a), label="Attacker PDF")
    ax.set_title("Credential 2: Gaussian PDFs")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_far_vs_frr_subplots_2cred(
    p1: GaussianPair,
    p2: GaussianPair,
    out_pdf: str = "02_far_vs_frr.pdf",
) -> None:
    # Curves per credential
    T1 = auto_T_grid_1d(p1)
    T2 = auto_T_grid_1d(p2)
    FAR1, FRR1 = far_frr(T1, p1)
    FAR2, FRR2 = far_frr(T2, p2)

    # Joint optimization gives (T1*,T2*) for AND and OR
    out_and = optimize_thresholds_lbfgs(p1, p2, mode="AND")
    out_or = optimize_thresholds_lbfgs(p1, p2, mode="OR")
    T1_and, T2_and = out_and["T_opt"]
    T1_or, T2_or = out_or["T_opt"]

    FAR1_and, FRR1_and = far_frr(T1_and, p1)
    FAR1_or, FRR1_or = far_frr(T1_or, p1)
    FAR2_and, FRR2_and = far_frr(T2_and, p2)
    FAR2_or, FRR2_or = far_frr(T2_or, p2)

    # EER per credential
    eer1 = eer_from_grid(p1, T1)
    eer2 = eer_from_grid(p2, T2)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)

    ax = axes[0]
    ax.plot(FAR1, FRR1, label="Operating curve")
    ax.scatter([float(FAR1_and)], [float(FRR1_and)], marker="o", label="Opt AND")
    ax.scatter([float(FAR1_or)], [float(FRR1_or)], marker="x", label="Opt OR")
    ax.scatter([eer1["FAR_eer"]], [eer1["FRR_eer"]], marker="s", label="EER")
    ax.set_title("Credential 1: FAR vs FRR")
    ax.set_ylabel("FRR")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 0.5)
    ax.legend()

    ax = axes[1]
    ax.plot(FAR2, FRR2, label="Operating curve")
    ax.scatter([float(FAR2_and)], [float(FRR2_and)], marker="o", label="Opt AND")
    ax.scatter([float(FAR2_or)], [float(FRR2_or)], marker="x", label="Opt OR")
    ax.scatter([eer2["FAR_eer"]], [eer2["FRR_eer"]], marker="s", label="EER")
    ax.set_title("Credential 2: FAR vs FRR")
    ax.set_xlabel("FAR")
    ax.set_ylabel("FRR")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


# =========================
# Success surfaces (kept as you had it)
# =========================
def plot_success_surfaces_2cred(p1: GaussianPair, p2: GaussianPair, out_pdf: str = "03_success_surfaces.pdf") -> None:
    # 2D grids for (T1,T2)
    T1 = auto_T_grid_1d(p1, n=220)
    T2 = auto_T_grid_1d(p2, n=220)
    T1g, T2g = np.meshgrid(T1, T2, indexing="xy")

    P_and = np.empty_like(T1g, dtype=float)
    P_or = np.empty_like(T1g, dtype=float)

    for i in range(T2g.shape[0]):
        for j in range(T1g.shape[1]):
            t1 = float(T1g[i, j])
            t2 = float(T2g[i, j])
            P_and[i, j] = success_probability(t1, t2, p1, p2, "AND")
            P_or[i, j] = success_probability(t1, t2, p1, p2, "OR")

    out_and = optimize_thresholds_lbfgs(p1, p2, mode="AND")
    out_or = optimize_thresholds_lbfgs(p1, p2, mode="OR")
    T1_and, T2_and = out_and["T_opt"]
    T1_or, T2_or = out_or["T_opt"]

    eer1 = eer_from_grid(p1, auto_T_grid_1d(p1))
    eer2 = eer_from_grid(p2, auto_T_grid_1d(p2))
    T1_eer, T2_eer = eer1["T_eer"], eer2["T_eer"]

    fig = plt.figure(figsize=(10, 4))

    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(
        P_and,
        origin="lower",
        aspect="auto",
        extent=[T1.min(), T1.max(), T2.min(), T2.max()],
    )
    ax1.scatter([T1_and], [T2_and], marker="o", label="Opt AND")
    ax1.scatter([T1_eer], [T2_eer], marker="s", label="EER pair")
    ax1.set_xlabel("T1")
    ax1.set_ylabel("T2")
    ax1.set_title("AND success surface")
    ax1.legend()
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(
        P_or,
        origin="lower",
        aspect="auto",
        extent=[T1.min(), T1.max(), T2.min(), T2.max()],
    )
    ax2.scatter([T1_or], [T2_or], marker="x", label="Opt OR")
    ax2.scatter([T1_eer], [T2_eer], marker="s", label="EER pair")
    ax2.set_xlabel("T1")
    ax2.set_ylabel("T2")
    ax2.set_title("OR success surface")
    ax2.legend()
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Two credentials => TWO Gaussian pairs
    p1 = GaussianPair(mu_u=0.8, sigma_u=0.12, mu_a=0.3, sigma_a=0.15)
    p2 = GaussianPair(mu_u=1.0, sigma_u=0.20, mu_a=0.4, sigma_a=0.18)

    print("Credential 1 normalization:")
    check_normalization(p1)
    print("Credential 2 normalization:")
    check_normalization(p2)

    out_and = optimize_thresholds_lbfgs(p1, p2, mode="AND")
    out_or = optimize_thresholds_lbfgs(p1, p2, mode="OR")
    print("AND:", out_and)
    print("OR :", out_or)

    # Requested: each pair in its own subplot
    plot_gaussians_subplots_2cred(p1, p2, out_pdf="01_gaussians.pdf")
    plot_far_vs_frr_subplots_2cred(p1, p2, out_pdf="02_far_vs_frr.pdf")

    # Kept: success surface plot (AND+OR)
    plot_success_surfaces_2cred(p1, p2, out_pdf="03_success_surfaces.pdf")

    print("Saved: 01_gaussians.pdf, 02_far_vs_frr.pdf, 03_success_surfaces.pdf")
