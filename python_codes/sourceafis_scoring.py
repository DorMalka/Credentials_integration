import csv
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# =========================
# Config
# =========================
REPO_ROOT = Path("/Users/dormalka/Desktop/Dor/Paper").resolve()
SOURCEAFIS_DIR = REPO_ROOT / "sourceafis-demo"

# Choose dataset here:
DATA_DIR = SOURCEAFIS_DIR / "fvc2002_png" / "DB1_B"
# DATA_DIR = SOURCEAFIS_DIR / "fvc2004_png" / "DB1_B"

USER_ID = 103
USER_SAMPLES = {2,3,5}

OUTPUT_DIR = REPO_ROOT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORES_CSV = OUTPUT_DIR / "sourceafis_scores.csv"

MAX_IMPOSTER_SCORES = 20000   # set to None for all

PDF_FINE_STEP = 0.1
SMOOTH_SIGMA_POINTS = 3.0

# Fixed normalized bins: 0..100
HIST_BINS = np.arange(0, 101, 2)


# =========================
# SourceAFIS batch scoring
# =========================
def run_sourceafis_batch():
    probe_csv = ",".join(str(x) for x in sorted(USER_SAMPLES))
    max_imp = -1 if MAX_IMPOSTER_SCORES is None else int(MAX_IMPOSTER_SCORES)

    cmd = [
        "mvn",
        "-q",
        "-DskipTests",
        "compile",
        "exec:java",
        "-Dexec.mainClass=BatchScorer",
        f"-Dexec.args={DATA_DIR} {USER_ID} {probe_csv} {SCORES_CSV} {max_imp}",
    ]

    print("[i] Running SourceAFIS batch scorer...")
    print("[i] Working dir:", SOURCEAFIS_DIR)
    print("[i] Command:", " ".join(map(str, cmd)))

    subprocess.run(
        cmd,
        cwd=SOURCEAFIS_DIR,
        check=True,
        text=True,
    )


def load_scores_from_csv():
    genuine = []
    impostor = []

    with open(SCORES_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kind = row["kind"].strip().lower()
            score = float(row["score"])
            if kind == "genuine":
                genuine.append(score)
            elif kind == "impostor":
                impostor.append(score)

    genuine = np.array(genuine, dtype=float)
    impostor = np.array(impostor, dtype=float)

    if len(genuine) == 0:
        raise ValueError("No genuine scores loaded.")
    if len(impostor) == 0:
        raise ValueError("No impostor scores loaded.")

    return genuine, impostor


# =========================
# Normalize raw SourceAFIS scores to [0,100]
# =========================
def normalize_scores_to_100(genuine_scores, impostor_scores):
    smax = max(np.max(genuine_scores), np.max(impostor_scores))
    if smax <= 0:
        raise ValueError("Maximum raw score must be positive for normalization.")

    genuine_norm = 100.0 * genuine_scores / smax
    impostor_norm = 100.0 * impostor_scores / smax

    genuine_norm = np.clip(genuine_norm, 0.0, 100.0)
    impostor_norm = np.clip(impostor_norm, 0.0, 100.0)

    return genuine_norm, impostor_norm, smax


# =========================
# Plot histogram + raw hist-PDF
# =========================
def plot_histograms(genuine_scores, impostor_scores):
    plt.figure(figsize=(8, 6))
    if len(impostor_scores) > 0:
        plt.hist(impostor_scores, bins=HIST_BINS, alpha=0.6, label="Impostor")
    if len(genuine_scores) > 0:
        plt.hist(genuine_scores, bins=HIST_BINS, alpha=0.6, label="Genuine")
    plt.xlabel("Normalized similarity score (%)")
    plt.ylabel("Count")
    plt.title(f"Score Histogram (user={USER_ID}, probe={sorted(USER_SAMPLES)})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sourceafis_histogram.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def build_hist_pdf(scores, bins):
    counts, edges = np.histogram(scores, bins=bins)
    bw = float(edges[1] - edges[0])
    centers = (edges[:-1] + edges[1:]) / 2.0
    pdf = counts / (len(scores) * bw)
    return centers, pdf, bw


# =========================
# Peak-to-peak interpolation + smoothing
# =========================
def interp_and_smooth_pdf(centers, pdf, step=0.1, sigma_points=3.0, eps=0.0):
    centers = np.asarray(centers, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    x_fine = np.arange(float(centers[0]), float(centers[-1]) + 1e-9, step)
    y_fine = np.zeros_like(x_fine)

    support_mask = pdf > eps
    if np.count_nonzero(support_mask) < 2:
        y_fine = np.interp(x_fine, centers, pdf)
    else:
        c_sup = centers[support_mask]
        p_sup = pdf[support_mask]

        x0 = float(c_sup[0])
        x1 = float(c_sup[-1])

        inside = (x_fine >= x0) & (x_fine <= x1)
        y_fine[inside] = np.interp(x_fine[inside], c_sup, p_sup)

    y_s = gaussian_filter1d(y_fine, sigma=sigma_points, mode="nearest")

    area = np.trapezoid(y_s, x_fine)
    if area > 0:
        y_s = y_s / area

    return x_fine, y_s


def plot_smoothed_pdfs(genuine_scores, impostor_scores, step=0.1, sigma_points=3.0):
    plt.figure(figsize=(8, 6))

    cg, pg, _ = build_hist_pdf(genuine_scores, HIST_BINS)
    ci, pi, _ = build_hist_pdf(impostor_scores, HIST_BINS)

    xg, yg = interp_and_smooth_pdf(cg, pg, step=step, sigma_points=sigma_points)
    xi, yi = interp_and_smooth_pdf(ci, pi, step=step, sigma_points=sigma_points)

    plt.plot(xi, yi, linewidth=2, label="Impostor PDF (smoothed)")
    plt.plot(xg, yg, linewidth=2, label="Genuine PDF (smoothed)")

    plt.xlabel("Normalized similarity score (%)")
    plt.ylabel("Probability Density")
    plt.title("Smoothed PDFs (hist → peak-to-peak → gaussian smooth)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sourceafis_smoothed_pdf.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# FAR/FRR from smoothed PDFs + EER
# =========================
def compute_far_frr_from_smoothed_pdf(genuine_scores, impostor_scores, step=0.1, sigma_points=3.0):
    cg, pg, _ = build_hist_pdf(genuine_scores, HIST_BINS)
    ci, pi, _ = build_hist_pdf(impostor_scores, HIST_BINS)

    xg, yg = interp_and_smooth_pdf(cg, pg, step=step, sigma_points=sigma_points)
    xi, yi = interp_and_smooth_pdf(ci, pi, step=step, sigma_points=sigma_points)

    if not np.allclose(xg, xi):
        x_min = max(xg[0], xi[0])
        x_max = min(xg[-1], xi[-1])
        x = np.arange(x_min, x_max + 1e-9, step)
        yg = np.interp(x, xg, yg)
        yi = np.interp(x, xi, yi)
    else:
        x = xg

    dx = float(step)

    cdf_g = np.cumsum(yg) * dx
    surv_i = np.flip(np.cumsum(np.flip(yi))) * dx

    thresholds = x
    frrs = np.clip(cdf_g, 0.0, 1.0)
    fars = np.clip(surv_i, 0.0, 1.0)

    return thresholds, fars, frrs


def compute_eer_intersection(thresholds, fars, frrs):
    d = fars - frrs

    exact = np.where(d == 0)[0]
    if len(exact) > 0:
        i = int(exact[0])
        return float(fars[i]), float(thresholds[i])

    sc = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if len(sc) == 0:
        i = int(np.argmin(np.abs(d)))
        return float(0.5 * (fars[i] + frrs[i])), float(thresholds[i])

    i = int(sc[0])
    t0, t1 = float(thresholds[i]), float(thresholds[i + 1])
    d0, d1 = float(d[i]), float(d[i + 1])

    alpha = d0 / (d0 - d1)
    t_star = t0 + alpha * (t1 - t0)

    far_star = float(fars[i] + alpha * (fars[i + 1] - fars[i]))
    frr_star = float(frrs[i] + alpha * (frrs[i + 1] - frrs[i]))
    eer = 0.5 * (far_star + frr_star)

    return eer, t_star


def compute_p_success(thresholds, fars, frrs):
    p_success = (1 - fars) * (1 - frrs)
    idx_max = np.argmax(p_success)
    max_threshold = thresholds[idx_max]
    max_success = p_success[idx_max]
    return p_success, max_success, max_threshold


def plot_far_frr(thresholds, fars, frrs, eer, eer_threshold):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, fars, label="FAR (smoothed-PDF integral)")
    plt.plot(thresholds, frrs, label="FRR (smoothed-PDF integral)")
    plt.scatter(eer_threshold, eer, label=f"EER≈{eer:.4f} @ T≈{eer_threshold:.2f}", zorder=3)
    plt.xlabel("Threshold (%)")
    plt.ylabel("Error Rate")
    plt.title("FAR / FRR vs Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sourceafis_far_frr.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def compute_success_and_or(
    thresholds,
    fars,
    frrs,
    *,
    eer_threshold,
    P_safe=0.5,
    P_leak=0.4,
    P_loss=0.1,
    P_theft=0.0,
):
    if not np.isclose(P_safe + P_leak + P_loss + P_theft, 1.0):
        raise ValueError("P_safe + P_leak + P_loss + P_theft must sum to 1")

    p_and = (1 - frrs) * (P_safe + P_leak * (1 - fars))
    p_or = (1 - fars) * (P_safe + P_loss * (1 - frrs))

    idx_and = int(np.argmax(p_and))
    idx_or = int(np.argmax(p_or))

    p_and_eer = float(np.interp(eer_threshold, thresholds, p_and))
    p_or_eer = float(np.interp(eer_threshold, thresholds, p_or))

    return p_and, p_or, idx_and, idx_or, p_and_eer, p_or_eer


def plot_p_success(thresholds, p_success, eer_threshold, max_success, max_threshold):
    plt.figure(figsize=(8, 6))

    plt.plot(thresholds, p_success, label="P_success(t)")
    p_eer = float(np.interp(eer_threshold, thresholds, p_success))

    plt.scatter(
        eer_threshold,
        p_eer,
        zorder=3,
        label=f"P_success@EER={p_eer:.4f} (T≈{eer_threshold:.2f})"
    )

    plt.scatter(
        max_threshold,
        max_success,
        color="red",
        zorder=3,
        label=f"Max P_success={max_success:.4f} (T={max_threshold:.2f})"
    )

    plt.xlabel("Threshold (%)")
    plt.ylabel("P_success")
    plt.title("Success Probability vs Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sourceafis_p_success.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_success_and_or(
    thresholds,
    p_and,
    p_or,
    idx_and,
    idx_or,
    eer_threshold,
    p_and_eer,
    p_or_eer,
):
    plt.figure(figsize=(8, 6))

    plt.plot(thresholds, p_and, label="P_success_AND")
    plt.plot(thresholds, p_or, label="P_success_OR")

    plt.scatter(
        thresholds[idx_and],
        p_and[idx_and],
        label=f"AND max, T={thresholds[idx_and]:.2f}, {p_and[idx_and]:.3f}",
        zorder=3,
    )

    plt.scatter(
        thresholds[idx_or],
        p_or[idx_or],
        label=f"OR max, T={thresholds[idx_or]:.2f}, {p_or[idx_or]:.3f}",
        zorder=3,
    )

    plt.scatter(
        eer_threshold,
        p_and_eer,
        label=f"AND@EER, {p_and_eer:.3f}",
        zorder=4,
    )

    plt.scatter(
        eer_threshold,
        p_or_eer,
        label=f"OR@EER, {p_or_eer:.3f}",
        zorder=4,
    )

    plt.xlabel("Threshold (%)")
    plt.ylabel("Success Probability")
    plt.title("Integrated Success vs Threshold (AND / OR)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sourceafis_success_and_or.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Main
# =========================
if __name__ == "__main__":
    run_sourceafis_batch()

    genuine_raw, impostor_raw = load_scores_from_csv()

    print(f"[i] Raw genuine scores:  n={len(genuine_raw)}  min={genuine_raw.min():.4f}  max={genuine_raw.max():.4f}  mean={genuine_raw.mean():.4f}")
    print(f"[i] Raw impostor scores: n={len(impostor_raw)} min={impostor_raw.min():.4f} max={impostor_raw.max():.4f} mean={impostor_raw.mean():.4f}")

    genuine, impostor, raw_max = normalize_scores_to_100(genuine_raw, impostor_raw)

    print(f"[i] Normalization factor (raw max) = {raw_max:.4f}")
    print(f"[i] Normalized genuine scores:  min={genuine.min():.4f}  max={genuine.max():.4f}  mean={genuine.mean():.4f}")
    print(f"[i] Normalized impostor scores: min={impostor.min():.4f} max={impostor.max():.4f} mean={impostor.mean():.4f}")

    plot_histograms(genuine, impostor)
    plot_smoothed_pdfs(genuine, impostor, step=PDF_FINE_STEP, sigma_points=SMOOTH_SIGMA_POINTS)

    thresholds, fars, frrs = compute_far_frr_from_smoothed_pdf(
        genuine,
        impostor,
        step=PDF_FINE_STEP,
        sigma_points=SMOOTH_SIGMA_POINTS
    )

    eer, eer_threshold = compute_eer_intersection(thresholds, fars, frrs)

    print(f"[i] EER = {eer:.6f}")
    print(f"[i] EER threshold ≈ {eer_threshold:.4f}")

    plot_far_frr(thresholds, fars, frrs, eer, eer_threshold)

    p_success, max_success, max_threshold = compute_p_success(thresholds, fars, frrs)

    print(f"[i] Max P_success = {max_success:.4f}")
    print(f"[i] Max P_success threshold = {max_threshold:.4f}")

    plot_p_success(thresholds, p_success, eer_threshold, max_success, max_threshold)

    p_and, p_or, idx_and, idx_or, p_and_eer, p_or_eer = compute_success_and_or(
        thresholds,
        fars,
        frrs,
        eer_threshold=eer_threshold,
        P_safe=0.9,
        P_leak=0.04,
        P_loss=0.04,
        P_theft=0.02,
    )

    print(f"[i] AND max P_success = {p_and[idx_and]:.4f} at T={thresholds[idx_and]:.4f}")
    print(f"[i] OR  max P_success = {p_or[idx_or]:.4f} at T={thresholds[idx_or]:.4f}")
    print(f"[i] AND at EER = {p_and_eer:.4f}")
    print(f"[i] OR  at EER = {p_or_eer:.4f}")

    plot_success_and_or(
        thresholds,
        p_and,
        p_or,
        idx_and,
        idx_or,
        eer_threshold,
        p_and_eer,
        p_or_eer,
    )