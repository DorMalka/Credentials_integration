import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# =========================
# Config (LivDet 2015, BMP)
# =========================
USER_PREFIX = "002_0"

LIVDET_ROOT = Path("/Users/dormalka/Desktop/Dor/Paper/livdet_iso")

TRAIN_LIVE_DIR = LIVDET_ROOT / "Training" / "Hi_Scan" / "Live"
TEST_LIVE_DIR  = LIVDET_ROOT / "Testing"  / "Hi_Scan" / "Live"
TEST_FAKE_DIR  = LIVDET_ROOT / "Testing"  / "Hi_Scan" / "Fake" / "Latex"

IMG_EXTS = {".iso", ".ISO"}   # <-- important

OUTPUT_DIR = Path("/Users/dormalka/Desktop/Dor/Paper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: if your binary is under build/, change this to:
# OPENAFIS_CLI = Path(".").resolve() / "build" / "cli" / "openafis-cli"
OPENAFIS_CLI = Path("/Users/dormalka/Desktop/Dor/Paper/measurements/openafis/cli/openafis-cli")

# Histogram bins: similarity is percent (0..100)
HIST_BINS = np.arange(0, 101, 2)

# Smoothing controls (for histogram-PDF smoothing)
PDF_FINE_STEP = 0.1
SMOOTH_SIGMA_POINTS = 3.0

# Optional subsampling (None = use all pairs)
MAX_GENUINE_SCORES = None
MAX_IMPOSTER_SCORES = None
RANDOM_SEED = 123


# =========================
# Helpers
# =========================
def _parse_similarity_from_cli_output(out: str) -> int:
    m = re.search(r"Similarity.*?:\s*(\d+)\s*%", out)
    if m:
        return int(m.group(1))
    m = re.search(r"with\s+(\d+)\s*%\s+similarity", out, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not parse similarity from CLI output:\n{out}")


def run_one_to_one_similarity(root: Path, f1: Path, f2: Path) -> int:
    if not OPENAFIS_CLI.exists():
        raise FileNotFoundError(
            f"Cannot find {OPENAFIS_CLI}. If you built with CMake, it may be under build/cli/. "
            f"Try setting OPENAFIS_CLI = Path('.').resolve() / 'build' / 'cli' / 'openafis-cli'."
        )

    # openafis-cli expects paths relative to --path
    f1_rel = f1.resolve().relative_to(root.resolve())
    f2_rel = f2.resolve().relative_to(root.resolve())

    cmd = [
        str(OPENAFIS_CLI),
        "one",
        "--path", str(root),
        "--f1", str(f1_rel),
        "--f2", str(f2_rel),
        "--f3", str(f2_rel),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    return _parse_similarity_from_cli_output(out)


def _collect_user_files(dir_path: Path, user_prefix: str) -> List[Path]:
    files = sorted(dir_path.glob(f"{user_prefix}_*.jpg"))
    if not files:
        raise FileNotFoundError(f"No JPG files found for user_prefix={user_prefix} under: {dir_path}")
    return files


def _maybe_subsample(pairs: List[Tuple[Path, Path]], max_n: Optional[int]) -> List[Tuple[Path, Path]]:
    if max_n is None or len(pairs) <= max_n:
        return pairs
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(pairs), size=max_n, replace=False)
    return [pairs[i] for i in idx]


# =========================
# Main scoring logic (LivDet)
# =========================
def build_scores_livdet():
    train_live = _collect_user_files(TRAIN_LIVE_DIR, USER_PREFIX)  # enrollment / probes
    test_live  = _collect_user_files(TEST_LIVE_DIR,  USER_PREFIX)  # genuine targets
    test_fake  = _collect_user_files(TEST_FAKE_DIR,  USER_PREFIX)  # impostor targets (Latex)

    print(f"[i] Train Live: {len(train_live)} files")
    print(f"    e.g. {[p.name for p in train_live[:10]]}")
    print(f"[i] Test  Live: {len(test_live)} files")
    print(f"    e.g. {[p.name for p in test_live[:10]]}")
    print(f"[i] Test  Fake(Latex): {len(test_fake)} files")
    print(f"    e.g. {[p.name for p in test_fake[:10]]}")

    genuine_pairs = [(a, b) for a in train_live for b in test_live]
    imposter_pairs = [(a, b) for a in train_live for b in test_fake]

    genuine_pairs = _maybe_subsample(genuine_pairs, MAX_GENUINE_SCORES)
    imposter_pairs = _maybe_subsample(imposter_pairs, MAX_IMPOSTER_SCORES)

    print(f"[i] Genuine pairs: {len(genuine_pairs)}")
    print(f"[i] Impostor pairs: {len(imposter_pairs)}")

    genuine_scores = [run_one_to_one_similarity(LIVDET_ROOT, a, b) for a, b in genuine_pairs]
    imposter_scores = [run_one_to_one_similarity(LIVDET_ROOT, a, b) for a, b in imposter_pairs]

    return np.array(genuine_scores, dtype=float), np.array(imposter_scores, dtype=float)


# =========================
# Plot histogram + raw hist-PDF
# =========================
def plot_histograms(genuine_scores, imposter_scores):
    plt.figure(figsize=(8, 6))
    if len(imposter_scores) > 0:
        plt.hist(imposter_scores, bins=HIST_BINS, alpha=0.6, label="Impostor (Train Live vs Test Fake/Latex)")
    if len(genuine_scores) > 0:
        plt.hist(genuine_scores, bins=HIST_BINS, alpha=0.6, label="Genuine (Train Live vs Test Live)")
    plt.xlabel("Similarity score (%)")
    plt.ylabel("Count")
    plt.title(f"Score Histogram (LivDet2015 Hi_Scan, user={USER_PREFIX})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "livdet2015_histogram.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def build_hist_pdf(scores, bins):
    counts, edges = np.histogram(scores, bins=bins)
    bw = float(edges[1] - edges[0])
    centers = (edges[:-1] + edges[1:]) / 2.0
    pdf = counts / (len(scores) * bw)
    return centers, pdf, bw


# =========================
# Peak-to-peak interpolation + smoothing (renormalized)
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


def plot_smoothed_pdfs(genuine_scores, imposter_scores, step=0.1, sigma_points=3.0):
    plt.figure(figsize=(8, 6))

    cg, pg, _ = build_hist_pdf(genuine_scores, HIST_BINS)
    ci, pi, _ = build_hist_pdf(imposter_scores, HIST_BINS)

    xg, yg = interp_and_smooth_pdf(cg, pg, step=step, sigma_points=sigma_points)
    xi, yi = interp_and_smooth_pdf(ci, pi, step=step, sigma_points=sigma_points)

    plt.plot(xi, yi, linewidth=2, label="Impostor PDF (smoothed)")
    plt.plot(xg, yg, linewidth=2, label="Genuine PDF (smoothed)")

    plt.xlabel("Similarity score (%)")
    plt.ylabel("Probability Density")
    plt.title("Smoothed PDFs (hist → peak-to-peak → gaussian smooth)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "livdet2015_smoothed_pdf.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# =========================
# FAR/FRR/EER + plots
# =========================
def compute_far_frr_from_smoothed_pdf(genuine_scores, imposter_scores, step=0.1, sigma_points=3.0):
    cg, pg, _ = build_hist_pdf(genuine_scores, HIST_BINS)
    ci, pi, _ = build_hist_pdf(imposter_scores, HIST_BINS)

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


def plot_far_frr(thresholds, fars, frrs, eer, eer_threshold):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, fars, label="FAR (smoothed-PDF integral)")
    plt.plot(thresholds, frrs, label="FRR (smoothed-PDF integral)")
    plt.scatter(eer_threshold, eer, label=f"EER≈{eer:.4f} @ T≈{eer_threshold:.2f}", zorder=3)
    plt.xlabel("Threshold (%)")
    plt.ylabel("Error Rate")
    plt.title("FAR / FRR vs Threshold (from smoothed histogram-PDF)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "livdet2015_far_frr.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def compute_p_success(thresholds, fars, frrs):
    p_success = (1 - fars) * (1 - frrs)
    idx_max = int(np.argmax(p_success))
    return p_success, float(p_success[idx_max]), float(thresholds[idx_max])


def plot_p_success(thresholds, p_success, eer_threshold, max_success, max_threshold):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, p_success, label="P_success(t)")

    p_eer = float(np.interp(eer_threshold, thresholds, p_success))
    plt.scatter(eer_threshold, p_eer, zorder=3,
                label=f"P_success@EER={p_eer:.4f} (T≈{eer_threshold:.2f})")

    plt.scatter(max_threshold, max_success, color="red", zorder=3,
                label=f"Max P_success={max_success:.4f} (T={max_threshold:.2f})")

    plt.xlabel("Threshold (%)")
    plt.ylabel("P_success")
    plt.title("Success Probability vs Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "livdet2015_p_success.pdf", dpi=300, bbox_inches="tight")
    plt.show()
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
    p_or  = (1 - fars) * (P_safe + P_loss * (1 - frrs))

    idx_and = int(np.argmax(p_and))
    idx_or  = int(np.argmax(p_or))

    p_and_eer = float(np.interp(eer_threshold, thresholds, p_and))
    p_or_eer  = float(np.interp(eer_threshold, thresholds, p_or))
    return p_and, p_or, idx_and, idx_or, p_and_eer, p_or_eer


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

    plt.scatter(thresholds[idx_and], p_and[idx_and],
                label=f"AND max, T={thresholds[idx_and]:.2f}, {p_and[idx_and]:.3f}", zorder=3)
    plt.scatter(thresholds[idx_or], p_or[idx_or],
                label=f"OR max, T={thresholds[idx_or]:.2f}, {p_or[idx_or]:.3f}", zorder=3)

    plt.scatter(eer_threshold, p_and_eer, label=f"AND@EER, {p_and_eer:.3f}", zorder=4)
    plt.scatter(eer_threshold, p_or_eer,  label=f"OR@EER, {p_or_eer:.3f}",  zorder=4)

    plt.xlabel("Threshold (%)")
    plt.ylabel("Success Probability")
    plt.title("Integrated Success vs Threshold (AND / OR)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "livdet2015_success_and_or.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# =========================
# Main
# =========================
if __name__ == "__main__":
    genuine, imposter = build_scores_livdet()

    print(f"[i] Genuine scores:  n={len(genuine)}  min={genuine.min():.0f}  max={genuine.max():.0f}  mean={genuine.mean():.2f}")
    print(f"[i] Impostor scores: n={len(imposter)} min={imposter.min():.0f} max={imposter.max():.0f} mean={imposter.mean():.2f}")

    plot_histograms(genuine, imposter)
    plot_smoothed_pdfs(genuine, imposter, step=PDF_FINE_STEP, sigma_points=SMOOTH_SIGMA_POINTS)

    thresholds, fars, frrs = compute_far_frr_from_smoothed_pdf(
        genuine, imposter, step=PDF_FINE_STEP, sigma_points=SMOOTH_SIGMA_POINTS
    )

    eer, eer_threshold = compute_eer_intersection(thresholds, fars, frrs)
    print(f"[i] EER (smoothed-PDF integral) = {eer:.6f}")
    print(f"[i] EER threshold ≈ {eer_threshold:.3f}")

    plot_far_frr(thresholds, fars, frrs, eer, eer_threshold)

    p_success, max_success, max_threshold = compute_p_success(thresholds, fars, frrs)
    print(f"[i] Max P_success = {max_success:.4f}")
    print(f"[i] Max P_success threshold = {max_threshold:.2f}%")

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

    print(f"[i] AND max P_success = {p_and[idx_and]:.4f} at T={thresholds[idx_and]:.2f}")
    print(f"[i] OR  max P_success = {p_or[idx_or]:.4f} at T={thresholds[idx_or]:.2f}")
    print(f"[i] AND at EER = {p_and_eer:.4f}")
    print(f"[i] OR  at EER = {p_or_eer:.4f}")

    plot_success_and_or(
        thresholds, p_and, p_or, idx_and, idx_or, eer_threshold, p_and_eer, p_or_eer
    )