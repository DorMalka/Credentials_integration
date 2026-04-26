import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Union, Any, Optional, Literal, List

from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad

import matplotlib.pyplot as plt
from pathlib import Path


Mode = Literal["AND", "OR"]
REPO_ROOT = Path("/Users/dormalka/Desktop/Dor/Paper").resolve()
OUTPUT_DIR = REPO_ROOT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class GaussianPair:
    # user ~ N(mu_u, sigma_u), attacker ~ N(mu_a, sigma_a)
    mu_u: float
    sigma_u: float
    mu_a: float
    sigma_a: float


# =========================
# NEW: Uniform + Parabolic (no figures)
# =========================
@dataclass(frozen=True)
class UniformPair:
    # user ~ U(a_u, b_u), attacker ~ U(a_a, b_a)
    a_u: float
    b_u: float
    a_a: float
    b_a: float


@dataclass(frozen=True)
class ParabolicPair:
    # Parabolic (concave) on [m-r, m+r]:
    # pdf(x)=3/(4r)*(1-((x-m)/r)^2) for |x-m|<=r else 0
    m_u: float
    r_u: float
    m_a: float
    r_a: float


DistPair = Union[GaussianPair, UniformPair, ParabolicPair]


# =========================
# Core math
# =========================
def _check_pair(p: GaussianPair) -> None:
    if p.sigma_u <= 0 or p.sigma_a <= 0:
        raise ValueError("sigmas must be > 0")


def _check_pair_generic(p: DistPair) -> None:
    if isinstance(p, GaussianPair):
        _check_pair(p)
    elif isinstance(p, UniformPair):
        if not (p.b_u > p.a_u and p.b_a > p.a_a):
            raise ValueError("Uniform must satisfy b_u>a_u and b_a>a_a")
    elif isinstance(p, ParabolicPair):
        if p.r_u <= 0 or p.r_a <= 0:
            raise ValueError("Parabolic radii must be > 0")
    else:
        raise TypeError("Unknown distribution pair type")


def _pdf_cdf_user(x: np.ndarray, p: DistPair) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(p, GaussianPair):
        return norm.pdf(x, loc=p.mu_u, scale=p.sigma_u), norm.cdf(x, loc=p.mu_u, scale=p.sigma_u)

    if isinstance(p, UniformPair):
        a, b = p.a_u, p.b_u
        pdf = np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)
        cdf = np.where(x < a, 0.0, np.where(x > b, 1.0, (x - a) / (b - a)))
        return pdf, cdf

    if isinstance(p, ParabolicPair):
        m, r = p.m_u, p.r_u
        z = (x - m) / r
        pdf = np.where(np.abs(z) <= 1.0, (3.0 / (4.0 * r)) * (1.0 - z**2), 0.0)
        # CDF: 0 for z<=-1, 1 for z>=1, else:
        # F(z)= (3/4)*( z - z^3/3 + 2/3 )
        cdf_inside = (3.0 / 4.0) * (z - (z**3) / 3.0 + 2.0 / 3.0)
        cdf = np.where(z <= -1.0, 0.0, np.where(z >= 1.0, 1.0, cdf_inside))
        return pdf, cdf

    raise TypeError("Unknown distribution pair type")


def _pdf_cdf_attacker(x: np.ndarray, p: DistPair) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(p, GaussianPair):
        return norm.pdf(x, loc=p.mu_a, scale=p.sigma_a), norm.cdf(x, loc=p.mu_a, scale=p.sigma_a)

    if isinstance(p, UniformPair):
        a, b = p.a_a, p.b_a
        pdf = np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)
        cdf = np.where(x < a, 0.0, np.where(x > b, 1.0, (x - a) / (b - a)))
        return pdf, cdf

    if isinstance(p, ParabolicPair):
        m, r = p.m_a, p.r_a
        z = (x - m) / r
        pdf = np.where(np.abs(z) <= 1.0, (3.0 / (4.0 * r)) * (1.0 - z**2), 0.0)
        cdf_inside = (3.0 / 4.0) * (z - (z**3) / 3.0 + 2.0 / 3.0)
        cdf = np.where(z <= -1.0, 0.0, np.where(z >= 1.0, 1.0, cdf_inside))
        return pdf, cdf

    raise TypeError("Unknown distribution pair type")


def far_frr(T: Union[float, np.ndarray], p: DistPair) -> Tuple[np.ndarray, np.ndarray]:
    """
    FAR(T) = ∫_{T}^{∞} f_attacker(x) dx = 1 - CDF_a(T)
    FRR(T) = ∫_{-∞}^{T} f_user(x) dx = CDF_u(T)
    Works for Gaussian / Uniform / Parabolic.
    """
    _check_pair_generic(p)
    T = np.asarray(T, dtype=float)
    _, cdf_a = _pdf_cdf_attacker(T, p)
    _, cdf_u = _pdf_cdf_user(T, p)
    FAR = 1.0 - cdf_a
    FRR = cdf_u
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


def success_probability(T1: float, T2: float, p1: DistPair, p2: DistPair, mode: Mode) -> float:
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


def _derivatives_wrt_T(T: float, p: DistPair) -> Dict[str, float]:
    """
    Derivatives wrt T:
      dFAR/dT = -pdf_attacker(T)
      dFRR/dT =  pdf_user(T)
    """
    _check_pair_generic(p)

    T_arr = np.asarray([T], dtype=float)
    pdf_a, _ = _pdf_cdf_attacker(T_arr, p)
    pdf_u, _ = _pdf_cdf_user(T_arr, p)
    pdf_a = float(pdf_a[0])
    pdf_u = float(pdf_u[0])

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


# =========================
# EER helpers (per-credential)
# =========================
def auto_T_grid_1d(p: DistPair, k: float = 6.0, n: int = 500) -> np.ndarray:
    """
    Gaussian: mu±k*sigma range.
    Uniform/Parabolic: cover full support (user ∪ attacker).
    """
    _check_pair_generic(p)
    if isinstance(p, GaussianPair):
        s = k * max(p.sigma_u, p.sigma_a)
        lo = min(p.mu_u, p.mu_a) - s
        hi = max(p.mu_u, p.mu_a) + s
        return np.linspace(lo, hi, n)
    if isinstance(p, UniformPair):
        lo = min(p.a_u, p.a_a)
        hi = max(p.b_u, p.b_a)
        return np.linspace(lo, hi, n)
    if isinstance(p, ParabolicPair):
        lo = min(p.m_u - p.r_u, p.m_a - p.r_a)
        hi = max(p.m_u + p.r_u, p.m_a + p.r_a)
        return np.linspace(lo, hi, n)
    raise TypeError("Unknown distribution pair type")


def eer_from_grid(p: DistPair, T: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    IMPORTANT: EER point IS on the operating curve by construction,
    because it's computed as FAR(T), FRR(T) for a single threshold T.
    """
    if T is None:
        T = auto_T_grid_1d(p)
    FAR, FRR = far_frr(T, p)
    idx = int(np.argmin(np.abs(FAR - FRR)))
    return {"T_eer": float(T[idx]), "FAR_eer": float(FAR[idx]), "FRR_eer": float(FRR[idx])}


def safe_opt_from_grid(p: DistPair, T: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Standalone optimal point for a single credential:
      maximize safe(T) = (1 - FAR(T)) * (1 - FRR(T))
    Returned point is on the FAR-vs-FRR operating curve.
    """
    if T is None:
        T = auto_T_grid_1d(p)

    FAR, FRR = far_frr(T, p)
    safe, _, _ = safe_leak_loss(FAR, FRR)

    idx = int(np.argmax(safe))
    return {
        "T_safe_opt": float(T[idx]),
        "safe_opt": float(safe[idx]),
        "FAR_safe_opt": float(FAR[idx]),
        "FRR_safe_opt": float(FRR[idx]),
    }


# =========================
# Joint optimization (2 creds)
# =========================
def optimize_thresholds_lbfgs(
    p1: DistPair,
    p2: DistPair,
    mode: Mode = "AND",
    x0: Optional[Tuple[float, float]] = None,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Maximize P_success(T1,T2) using L-BFGS-B (2 variables).
    We minimize -P_success.
    """
    _check_pair_generic(p1)
    _check_pair_generic(p2)

    if x0 is None:
        def init(p: DistPair) -> float:
            if isinstance(p, GaussianPair):
                return (p.mu_u + p.mu_a) / 2.0
            if isinstance(p, UniformPair):
                return 0.5 * ((p.a_u + p.b_u) / 2.0 + (p.a_a + p.b_a) / 2.0)
            if isinstance(p, ParabolicPair):
                return (p.m_u + p.m_a) / 2.0
            raise TypeError
        x0 = (init(p1), init(p2))

    if bounds is None:
        def auto_bounds(p: DistPair) -> Tuple[float, float]:
            if isinstance(p, GaussianPair):
                s = 8.0 * max(p.sigma_u, p.sigma_a)
                lo = min(p.mu_u, p.mu_a) - s
                hi = max(p.mu_u, p.mu_a) + s
                return (lo, hi)
            if isinstance(p, UniformPair):
                return (min(p.a_u, p.a_a), max(p.b_u, p.b_a))
            if isinstance(p, ParabolicPair):
                return (min(p.m_u - p.r_u, p.m_a - p.r_a), max(p.m_u + p.r_u, p.m_a + p.r_a))
            raise TypeError
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
            dP_dT1 = dsafe1 * safe2 + dsafe1 * leak2 + safe2 * dleak1
            dP_dT2 = safe1 * dsafe2 + safe1 * dleak2 + dsafe2 * leak1
        elif mode == "OR":
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
# Diagnostics (Gaussian only, as you had)
# =========================
def check_normalization(p: GaussianPair) -> None:
    f_u = lambda x: norm.pdf(x, p.mu_u, p.sigma_u)
    f_a = lambda x: norm.pdf(x, p.mu_a, p.sigma_a)
    iu, _ = quad(f_u, -np.inf, np.inf)
    ia, _ = quad(f_a, -np.inf, np.inf)
    print("User integral:", iu)
    print("Attacker integral:", ia)


# =========================
# Plotting: Gaussian PDFs and FAR-vs-FRR (unchanged behavior)
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
    out_pdf: str = OUTPUT_DIR / "figs" / "fig_2continuous" / "01_gaussians.pdf",
) -> None:
    x = shared_x_grid(p1, p2)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax = axes[0]
    ax.plot(x, norm.pdf(x, loc=p1.mu_u, scale=p1.sigma_u), label=r"$P_U(s)$", color = "blue")
    ax.plot(x, norm.pdf(x, loc=p1.mu_a, scale=p1.sigma_a), label=r"$P_A(s)$", color = "red")
    ax.set_title("Credential 1: Gaussian PDFs")
    ax.set_ylabel(r"$\text{Density}$")
    ax.legend()

    ax = axes[1]
    ax.plot(x, norm.pdf(x, loc=p2.mu_u, scale=p2.sigma_u), label=r"$P_U(s)$", color = "blue")
    ax.plot(x, norm.pdf(x, loc=p2.mu_a, scale=p2.sigma_a), label=r"$P_A(s)$", color = "red")
    ax.set_title("Credential 2: Gaussian PDFs")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$\text{Density}$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

def export_gaussians_2cred_data(
    p1: GaussianPair,
    p2: GaussianPair,
    out_txt = OUTPUT_DIR / "figs" / "fig_2continuous" / "gaussians_2cred_data.txt",
) -> None:
    x = shared_x_grid(p1, p2)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    PU1 = norm.pdf(x, loc=p1.mu_u, scale=p1.sigma_u)
    PA1 = norm.pdf(x, loc=p1.mu_a, scale=p1.sigma_a)

    PU2 = norm.pdf(x, loc=p2.mu_u, scale=p2.sigma_u)
    PA2 = norm.pdf(x, loc=p2.mu_a, scale=p2.sigma_a)

    with open(out_txt, "w") as f:
        f.write("s PU1 PA1 PU2 PA2\n")
        for xi, pu1, pa1, pu2, pa2 in zip(x, PU1, PA1, PU2, PA2):
            f.write(
                f"{xi:.4f} {pu1:.4f} {pa1:.4f} {pu2:.4f} {pa2:.4f}\n"
            )

def plot_far_vs_frr_subplots_2cred(
    p1: GaussianPair,
    p2: GaussianPair,
    out_pdf: str = OUTPUT_DIR / "figs" / "fig_2continuous" / "02_far_vs_frr.pdf",
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

    # standalone safe-opt per credential
    safeopt1 = safe_opt_from_grid(p1, T1)
    safeopt2 = safe_opt_from_grid(p2, T2)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)

    ax = axes[0]
    ax.plot(FAR1, FRR1, label="Operating curve")
    ax.scatter([float(FAR1_and)], [float(FRR1_and)], marker="o", label="AND Optimum")
    ax.scatter([float(FAR1_or)], [float(FRR1_or)], marker="x", label="OR Optimum")
    ax.scatter([eer1["FAR_eer"]], [eer1["FRR_eer"]], marker="s", label="EER")
    ax.scatter([safeopt1["FAR_safe_opt"]], [safeopt1["FRR_safe_opt"]], marker="^", label="Safe-opt (standalone)")
    ax.set_title("Credential 1: FAR vs FRR")
    ax.set_ylabel("FRR")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 0.5)
    ax.legend()

    ax = axes[1]
    ax.plot(FAR2, FRR2, label="Operating curve")
    ax.scatter([float(FAR2_and)], [float(FRR2_and)], marker="o", label="AND Optimum")
    ax.scatter([float(FAR2_or)], [float(FRR2_or)], marker="x", label="OR Optimum")
    ax.scatter([eer2["FAR_eer"]], [eer2["FRR_eer"]], marker="s", label="EER")
    ax.scatter([safeopt2["FAR_safe_opt"]], [safeopt2["FRR_safe_opt"]], marker="^", label="Safe-opt (standalone)")
    ax.set_title("Credential 2: FAR vs FRR")
    ax.set_xlabel(r"$FAR$")
    ax.set_ylabel(r"$FRR$")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

def export_far_vs_frr_2cred_data(
    p1: GaussianPair,
    p2: GaussianPair,
    out_txt=OUTPUT_DIR / "figs" / "fig_2continuous" / "far_vs_frr_2cred_data.txt",
    out_points=OUTPUT_DIR / "figs" / "fig_2continuous" / "far_vs_frr_2cred_points.txt",
) -> None:
    T1 = auto_T_grid_1d(p1)
    T2 = auto_T_grid_1d(p2)

    FAR1, FRR1 = far_frr(T1, p1)
    FAR2, FRR2 = far_frr(T2, p2)

    out_and = optimize_thresholds_lbfgs(p1, p2, mode="AND")
    out_or = optimize_thresholds_lbfgs(p1, p2, mode="OR")

    T1_and, T2_and = out_and["T_opt"]
    T1_or, T2_or = out_or["T_opt"]

    FAR1_and, FRR1_and = far_frr(T1_and, p1)
    FAR1_or, FRR1_or = far_frr(T1_or, p1)
    FAR2_and, FRR2_and = far_frr(T2_and, p2)
    FAR2_or, FRR2_or = far_frr(T2_or, p2)

    eer1 = eer_from_grid(p1, T1)
    eer2 = eer_from_grid(p2, T2)

    safeopt1 = safe_opt_from_grid(p1, T1)
    safeopt2 = safe_opt_from_grid(p2, T2)

    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(out_txt, "w") as f:
        f.write("T1 FAR1 FRR1 T2 FAR2 FRR2\n")
        for t1, far1, frr1, t2, far2, frr2 in zip(T1, FAR1, FRR1, T2, FAR2, FRR2):
            f.write(
                f"{t1:.4f} {far1:.4f} {frr1:.4f} "
                f"{t2:.4f} {far2:.4f} {frr2:.4f}\n"
            )

    with open(out_points, "w") as f:
        f.write(
            "FAR1_and FRR1_and FAR1_or FRR1_or FAR1_eer FRR1_eer FAR1_safe FRR1_safe "
            "FAR2_and FRR2_and FAR2_or FRR2_or FAR2_eer FRR2_eer FAR2_safe FRR2_safe\n"
        )
        f.write(
            f"{float(FAR1_and):.4f} {float(FRR1_and):.4f} "
            f"{float(FAR1_or):.4f} {float(FRR1_or):.4f} "
            f"{eer1['FAR_eer']:.4f} {eer1['FRR_eer']:.4f} "
            f"{safeopt1['FAR_safe_opt']:.4f} {safeopt1['FRR_safe_opt']:.4f} "
            f"{float(FAR2_and):.4f} {float(FRR2_and):.4f} "
            f"{float(FAR2_or):.4f} {float(FRR2_or):.4f} "
            f"{eer2['FAR_eer']:.4f} {eer2['FRR_eer']:.4f} "
            f"{safeopt2['FAR_safe_opt']:.4f} {safeopt2['FRR_safe_opt']:.4f}\n"
        )

# =========================
# Table for paper: Gaussian + Uniform + Parabolic (NO extra figures)
# =========================
def _fmt(x: float, nd: int = 6) -> str:
    if np.isfinite(x):
        return f"{x:.{nd}f}"
    return r"\infty"


def gaussian_to_uniform(p: GaussianPair) -> UniformPair:
    # U[a,b] moment-match to Gaussian: mean=mu, var=sigma^2=(b-a)^2/12 => halfwidth=sqrt(3)*sigma
    hu = np.sqrt(3.0) * p.sigma_u
    ha = np.sqrt(3.0) * p.sigma_a
    return UniformPair(
        a_u=p.mu_u - hu, b_u=p.mu_u + hu,
        a_a=p.mu_a - ha, b_a=p.mu_a + ha,
    )


def gaussian_to_parabolic(p: GaussianPair) -> ParabolicPair:
    # Parabolic moment-match: var = r^2/5 => r = sqrt(5)*sigma
    ru = np.sqrt(5.0) * p.sigma_u
    ra = np.sqrt(5.0) * p.sigma_a
    return ParabolicPair(
        m_u=p.mu_u, r_u=ru,
        m_a=p.mu_a, r_a=ra,
    )


def _wallet_metrics_at_reference_points(
    p1: DistPair,
    p2: DistPair,
    n_grid: int,
) -> Tuple[float, float, float, float, float, float]:
    # Grids
    T1_grid = auto_T_grid_1d(p1, n=n_grid)
    T2_grid = auto_T_grid_1d(p2, n=n_grid)

    # EER thresholds (per credential)
    eer1 = eer_from_grid(p1, T1_grid)
    eer2 = eer_from_grid(p2, T2_grid)
    T1_eer, T2_eer = eer1["T_eer"], eer2["T_eer"]

    # Standalone safe-opt thresholds (per credential)
    sopt1 = safe_opt_from_grid(p1, T1_grid)
    sopt2 = safe_opt_from_grid(p2, T2_grid)
    T1_sopt, T2_sopt = sopt1["T_safe_opt"], sopt2["T_safe_opt"]

    # Joint optima (each wallet at its own optimum)
    out_and = optimize_thresholds_lbfgs(p1, p2, mode="AND")
    out_or  = optimize_thresholds_lbfgs(p1, p2, mode="OR")
    T1_and, T2_and = out_and["T_opt"]
    T1_or,  T2_or  = out_or["T_opt"]

    # Wallet success at EER pair
    P_and_eer = success_probability(T1_eer, T2_eer, p1, p2, "AND")
    P_or_eer  = success_probability(T1_eer, T2_eer, p1, p2, "OR")

    # Wallet success at optimal pairs
    P_and_opt = success_probability(T1_and, T2_and, p1, p2, "AND")
    P_or_opt  = success_probability(T1_or,  T2_or,  p1, p2, "OR")

    # Wallet success at standalone pair
    P_and_sopt = success_probability(T1_sopt, T2_sopt, p1, p2, "AND")
    P_or_sopt  = success_probability(T1_sopt, T2_sopt, p1, p2, "OR")

    return P_and_eer, P_or_eer, P_and_opt, P_or_opt, P_and_sopt, P_or_sopt


def build_paper_table_gaussian_uniform_parabolic_2cred(
    p1: GaussianPair,
    p2: GaussianPair,
    n_grid: int = 2000,
    out_tex: str = "05_wallet_success_table.tex",
    nd: int = 6,
) -> str:
    """
    Produces LaTeX table with 3 rows:
      Gaussian (given), Uniform (moment-matched), Parabolic (moment-matched).

    Columns:
      P_AND @ EER, P_OR @ EER,
      P_AND @ optimal, P_OR @ optimal,
      P_AND @ standalone, P_OR @ standalone
    """
    # Moment-matched alternative distributions (per-credential)
    p1_u = gaussian_to_uniform(p1)
    p2_u = gaussian_to_uniform(p2)
    p1_p = gaussian_to_parabolic(p1)
    p2_p = gaussian_to_parabolic(p2)

    rows: List[List[str]] = []

    for name, a1, a2 in [
        ("Gaussian", p1, p2),
        ("Uniform",  p1_u, p2_u),
        ("Parabolic", p1_p, p2_p),
    ]:
        vals = _wallet_metrics_at_reference_points(a1, a2, n_grid=n_grid)
        rows.append([name] + [_fmt(v, nd) for v in vals])

    header_cols = [
        "Distribution",
        r"$P_{\mathrm{AND}}$ @ EER",
        r"$P_{\mathrm{OR}}$ @ EER",
        r"$P_{\mathrm{AND}}$ @ optimal",
        r"$P_{\mathrm{OR}}$ @ optimal",
        r"$P_{\mathrm{AND}}$ @ standalone",
        r"$P_{\mathrm{OR}}$ @ standalone",
    ]

    col_format = "l" + "c" * (len(header_cols) - 1)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_format + r"}")
    lines.append(r"\hline")
    lines.append(" & ".join(header_cols) + r" \\")
    lines.append(r"\hline")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{Wallet success under score distributions at three reference operating points: "
        r"EER-pair $(T_1^{\mathrm{eer}},T_2^{\mathrm{eer}})$, joint optima (AND at $(T_1^{*},T_2^{*})_{\mathrm{AND}}$ "
        r"and OR at $(T_1^{*},T_2^{*})_{\mathrm{OR}}$), and standalone safe-opt pair $(T_1^{\mathrm{safe}},T_2^{\mathrm{safe}})$. "
        r"Uniform and Parabolic are moment-matched to the Gaussian credentials (same mean and variance per credential).}"
    )
    lines.append(r"\label{tab:wallet_success_dist}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(latex)
    return latex


# =========================
# Success function plots (1D sweeps) for AND/OR wallets (Gaussian as before)
# =========================
def plot_success_functions_2cred(
    p1: GaussianPair,
    p2: GaussianPair,
    out_pdf: str = OUTPUT_DIR / "figs" / "fig_2continuous" / "04_success_functions.pdf",
) -> None:
    # Grids
    T1_grid = auto_T_grid_1d(p1)
    T2_grid = auto_T_grid_1d(p2)

    # Optimize thresholds for both modes
    out_and = optimize_thresholds_lbfgs(p1, p2, mode="AND")
    out_or = optimize_thresholds_lbfgs(p1, p2, mode="OR")

    T1_and, T2_and = out_and["T_opt"]
    T1_or, T2_or = out_or["T_opt"]

    # EER thresholds (per-credential)
    eer1 = eer_from_grid(p1, T1_grid)
    eer2 = eer_from_grid(p2, T2_grid)
    T1_eer, T2_eer = eer1["T_eer"], eer2["T_eer"]

    # Sweep T1 (fix T2 at each mode's optimum)
    P_and_sweep_T1 = np.array([success_probability(float(t1), T2_and, p1, p2, "AND") for t1 in T1_grid])
    P_or_sweep_T1  = np.array([success_probability(float(t1), T2_or,  p1, p2, "OR")  for t1 in T1_grid])

    # Sweep T2 (fix T1 at each mode's optimum)
    P_and_sweep_T2 = np.array([success_probability(T1_and, float(t2), p1, p2, "AND") for t2 in T2_grid])
    P_or_sweep_T2  = np.array([success_probability(T1_or,  float(t2), p1, p2, "OR")  for t2 in T2_grid])

    P_and_at_T1eer_on_sweep = success_probability(T1_eer, T2_and, p1, p2, "AND")
    P_or_at_T1eer_on_sweep  = success_probability(T1_eer, T2_or,  p1, p2, "OR")

    P_and_at_T2eer_on_sweep = success_probability(T1_and, T2_eer, p1, p2, "AND")
    P_or_at_T2eer_on_sweep  = success_probability(T1_or,  T2_eer, p1, p2, "OR")

    # Markers for "full optimum"
    P_and_at_opt = success_probability(T1_and, T2_and, p1, p2, "AND")
    P_or_at_opt  = success_probability(T1_or,  T2_or,  p1, p2, "OR")

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False, sharey=True)

    ax = axes[0]
    ax.plot(T1_grid, P_and_sweep_T1, label=r"$P_{\text{success}}^\mathrm{AND}$")
    ax.plot(T1_grid, P_or_sweep_T1,  label=r"$P_{\text{success}}^\mathrm{OR}$")
    ax.scatter([T1_and], [P_and_at_opt], marker="o", label=r"$\text{AND Optimum}$")
    ax.scatter([T1_or],  [P_or_at_opt],  marker="x", label=r"$\text{OR Optimum}$")
    ax.scatter([T1_eer], [P_and_at_T1eer_on_sweep], marker="s", label=r"$P_{\text{success}}^\mathrm{AND}(T_{1,\mathrm{eer}})$")
    ax.scatter([T1_eer], [P_or_at_T1eer_on_sweep],  marker="D", label=r"$P_{\text{success}}^\mathrm{OR}(T_{1,\mathrm{eer}})$")
    ax.set_title("Success vs threshold (Credential 1 sweep)")
    ax.set_xlabel(r"$T_1$")
    ax.set_ylabel(r"$P_{\text{success}}$")
    ax.legend()

    ax = axes[1]
    ax.plot(T2_grid, P_and_sweep_T2, label=r"$P_{\text{success}}^\mathrm{AND}$")
    ax.plot(T2_grid, P_or_sweep_T2,  label=r"$P_{\text{success}}^\mathrm{OR}$")
    ax.scatter([T2_and], [P_and_at_opt], marker="o", label=r"$\text{AND Optimum}$")
    ax.scatter([T2_or],  [P_or_at_opt],  marker="x", label=r"$\text{OR Optimum}$")
    ax.scatter([T2_eer], [P_and_at_T2eer_on_sweep], marker="s", label=r"$P_{\text{success}}^\mathrm{AND}(T_{2,\mathrm{eer}})$")
    ax.scatter([T2_eer], [P_or_at_T2eer_on_sweep],  marker="D", label=r"$P_{\text{success}}^\mathrm{OR}(T_{2,\mathrm{eer}})$")
    ax.set_title("Success vs threshold (Credential 2 sweep)")
    ax.set_xlabel(r"$T_2$")
    ax.set_ylabel(r"$P_{\text{success}}$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

def export_success_functions_2cred_data(
    p1: GaussianPair,
    p2: GaussianPair,
    out_data=OUTPUT_DIR / "figs" / "fig_2continuous" / "success_functions_2cred_data.txt",
    out_points=OUTPUT_DIR / "figs" / "fig_2continuous" / "success_functions_2cred_points.txt",
) -> None:
    T1_grid = auto_T_grid_1d(p1)
    T2_grid = auto_T_grid_1d(p2)

    out_and = optimize_thresholds_lbfgs(p1, p2, mode="AND")
    out_or = optimize_thresholds_lbfgs(p1, p2, mode="OR")

    T1_and, T2_and = out_and["T_opt"]
    T1_or, T2_or = out_or["T_opt"]

    eer1 = eer_from_grid(p1, T1_grid)
    eer2 = eer_from_grid(p2, T2_grid)
    T1_eer, T2_eer = eer1["T_eer"], eer2["T_eer"]

    P_and_T1 = np.array([
        success_probability(float(t1), T2_and, p1, p2, "AND")
        for t1 in T1_grid
    ])
    P_or_T1 = np.array([
        success_probability(float(t1), T2_or, p1, p2, "OR")
        for t1 in T1_grid
    ])

    P_and_T2 = np.array([
        success_probability(T1_and, float(t2), p1, p2, "AND")
        for t2 in T2_grid
    ])
    P_or_T2 = np.array([
        success_probability(T1_or, float(t2), p1, p2, "OR")
        for t2 in T2_grid
    ])

    P_and_at_opt = success_probability(T1_and, T2_and, p1, p2, "AND")
    P_or_at_opt = success_probability(T1_or, T2_or, p1, p2, "OR")

    P_and_T1_eer = success_probability(T1_eer, T2_and, p1, p2, "AND")
    P_or_T1_eer = success_probability(T1_eer, T2_or, p1, p2, "OR")

    P_and_T2_eer = success_probability(T1_and, T2_eer, p1, p2, "AND")
    P_or_T2_eer = success_probability(T1_or, T2_eer, p1, p2, "OR")

    out_data.parent.mkdir(parents=True, exist_ok=True)

    with open(out_data, "w") as f:
        f.write("T1 P_and_T1 P_or_T1 T2 P_and_T2 P_or_T2\n")
        for t1, pa1, po1, t2, pa2, po2 in zip(
            T1_grid, P_and_T1, P_or_T1,
            T2_grid, P_and_T2, P_or_T2
        ):
            f.write(
                f"{t1:.4f} {pa1:.4f} {po1:.4f} "
                f"{t2:.4f} {pa2:.4f} {po2:.4f}\n"
            )

    with open(out_points, "w") as f:
        f.write(
            "T1_and P_and_opt T1_or P_or_opt T1_eer P_and_T1_eer P_or_T1_eer "
            "T2_and P_and_opt2 T2_or P_or_opt2 T2_eer P_and_T2_eer P_or_T2_eer\n"
        )
        f.write(
            f"{T1_and:.4f} {P_and_at_opt:.4f} "
            f"{T1_or:.4f} {P_or_at_opt:.4f} "
            f"{T1_eer:.4f} {P_and_T1_eer:.4f} {P_or_T1_eer:.4f} "
            f"{T2_and:.4f} {P_and_at_opt:.4f} "
            f"{T2_or:.4f} {P_or_at_opt:.4f} "
            f"{T2_eer:.4f} {P_and_T2_eer:.4f} {P_or_T2_eer:.4f}\n"
        )

# =========================
# Success surfaces (Gaussian as before)
# =========================
def plot_success_surfaces_2cred(p1: GaussianPair, p2: GaussianPair, out_pdf: str = OUTPUT_DIR / "figs" / "fig_2continuous" / "03_success_surfaces.pdf") -> None:
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

    # Gaussian figures (as before)
    plot_gaussians_subplots_2cred(p1, p2, out_pdf=OUTPUT_DIR / "figs" / "fig_2continuous" /"01_gaussians.pdf")
    export_gaussians_2cred_data(p1, p2)
    plot_far_vs_frr_subplots_2cred(p1, p2, out_pdf=OUTPUT_DIR / "figs" / "fig_2continuous" / "02_far_vs_frr.pdf")
    export_far_vs_frr_2cred_data(p1, p2)
    plot_success_surfaces_2cred(p1, p2, out_pdf=OUTPUT_DIR / "figs" / "fig_2continuous" / "03_success_surfaces.pdf")
    plot_success_functions_2cred(p1, p2, out_pdf=OUTPUT_DIR / "figs" / "fig_2continuous" / "04_success_functions.pdf")
    export_success_functions_2cred_data(p1, p2)

    # Table now includes Gaussian + Uniform + Parabolic (no extra figures for the new distributions)
    latex = build_paper_table_gaussian_uniform_parabolic_2cred(
        p1, p2, out_tex=OUTPUT_DIR / "figs" / "fig_2continuous" / "05_wallet_success_table.tex", nd=6
    )
    print(latex)

    print("Saved: 01_gaussians.pdf, 02_far_vs_frr.pdf, 03_success_surfaces.pdf, 04_success_functions.pdf, 05_wallet_success_table.tex")