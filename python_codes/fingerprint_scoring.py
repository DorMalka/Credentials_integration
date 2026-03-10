import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# =========================
# Config
# =========================
REPO_ROOT = Path(".").resolve()  # run from repo root
DATA_DIR = REPO_ROOT / "data" / "valid" / "fvc2002" / "DB1_B"
USER_ID = 105
OUTPUT_DIR = Path("/Users/dormalka/Desktop/Dor/Paper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Enrolled templates for the user (probe set)
USER_SAMPLES = {2,7}  # e.g., 105_2.iso, 105_7.iso

# NOTE: if your binary is under build/, change this to:
# OPENAFIS_CLI = REPO_ROOT / "build" / "cli" / "openafis-cli"
OPENAFIS_CLI = REPO_ROOT / "cli" / "openafis-cli"

MAX_IMPOSTER_SCORES = 20000   # set None for "all"
RANDOM_SEED = 123

# Histogram bins: similarity is percent (0..100)
HIST_BINS = np.arange(0, 101, 2)

# Smoothing controls (for histogram-PDF smoothing)
PDF_FINE_STEP = 0.1         # dense grid step (score units)
SMOOTH_SIGMA_POINTS = 3.0   # sigma in *grid points* (NOT score units)


# =========================
# Helpers
# =========================
def parse_finger_id(p: Path) -> int:
    m = re.match(r"^(\d+)_\d+\.iso$", p.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {p.name}")
    return int(m.group(1))


def parse_sample_id(p: Path) -> int:
    m = re.match(r"^\d+_(\d+)\.iso$", p.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {p.name}")
    return int(m.group(1))


def _parse_similarity_from_cli_output(out: str) -> int:
    m = re.search(r"Similarity.*?:\s*(\d+)\s*%", out)
    if m:
        return int(m.group(1))

    m = re.search(r"with\s+(\d+)\s*%\s+similarity", out, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not parse similarity from CLI output:\n{out}")


def run_one_to_one_similarity(f1: Path, f2: Path) -> int:
    if not OPENAFIS_CLI.exists():
        raise FileNotFoundError(
            f"Cannot find {OPENAFIS_CLI}. If you built with CMake, it may be under build/cli/. "
            f"Try setting OPENAFIS_CLI = REPO_ROOT / 'build' / 'cli' / 'openafis-cli'."
        )

    cmd = [
        str(OPENAFIS_CLI),
        "one",
        "--path", str(DATA_DIR),
        "--f1", str(f1.name),
        "--f2", str(f2.name),
        "--f3", str(f2.name),
    ]

    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    return _parse_similarity_from_cli_output(out)


def collect_templates() -> dict[int, list[Path]]:
    files = sorted(DATA_DIR.glob("*.iso"))
    if not files:
        raise FileNotFoundError(f"No .iso templates found under: {DATA_DIR}")

    by_id: dict[int, list[Path]] = {}
    for p in files:
        fid = parse_finger_id(p)
        by_id.setdefault(fid, []).append(p)
    return by_id


# =========================
# Main scoring logic
# =========================
def build_scores():
    by_id = collect_templates()

    if USER_ID not in by_id:
        raise ValueError(f"USER_ID={USER_ID} not found. Available IDs: {sorted(by_id.keys())[:20]} ...")

    all_user = sorted(by_id[USER_ID])

    probe_templates = [p for p in all_user if parse_sample_id(p) in USER_SAMPLES]
    genuine_targets = [p for p in all_user if parse_sample_id(p) not in USER_SAMPLES]

    if len(probe_templates) != len(USER_SAMPLES):
        existing = sorted(parse_sample_id(p) for p in all_user)
        raise ValueError(
            f"Missing requested USER_SAMPLES={sorted(USER_SAMPLES)} for user {USER_ID}. "
            f"Existing samples for {USER_ID}: {existing}"
        )
    if len(genuine_targets) == 0:
        raise ValueError(f"No remaining genuine targets for user {USER_ID} after excluding {sorted(USER_SAMPLES)}.")

    other_templates = []
    for fid, paths in by_id.items():
        if fid != USER_ID:
            other_templates.extend(paths)
    other_templates = sorted(other_templates)

    print(f"[i] Probe templates (enrollment): {[p.name for p in probe_templates]}")
    print(f"[i] Genuine targets (other {USER_ID}_*): {[p.name for p in genuine_targets]}")
    print(f"[i] Other-finger templates (impostors): {len(other_templates)} templates across {len(by_id)-1} fingers")

    # Genuine: probe vs genuine_targets
    genuine_pairs = [(u, g) for u in probe_templates for g in genuine_targets]
    print(f"[i] Genuine pairs (probe vs other-{USER_ID}): {len(genuine_pairs)}")
    genuine_scores = [run_one_to_one_similarity(a, b) for a, b in genuine_pairs]

    # Impostor: probe vs all other fingers
    imposter_pairs = [(u, o) for u in probe_templates for o in other_templates]
    print(f"[i] Raw impostor pairs (probe vs others): {len(imposter_pairs)}")

    if MAX_IMPOSTER_SCORES is not None and len(imposter_pairs) > MAX_IMPOSTER_SCORES:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(imposter_pairs), size=MAX_IMPOSTER_SCORES, replace=False)
        imposter_pairs = [imposter_pairs[i] for i in idx]
        print(f"[i] Subsampled impostor pairs to: {len(imposter_pairs)}")

    imposter_scores = [run_one_to_one_similarity(u, o) for u, o in imposter_pairs]

    return np.array(genuine_scores, dtype=float), np.array(imposter_scores, dtype=float)


# =========================
# Plot histogram + raw hist-PDF
# =========================
def plot_histograms(genuine_scores, imposter_scores):
    plt.figure(figsize=(8, 6))
    if len(imposter_scores) > 0:
        plt.hist(imposter_scores, bins=HIST_BINS, alpha=0.6, label="Impostor")
    if len(genuine_scores) > 0:
        plt.hist(genuine_scores, bins=HIST_BINS, alpha=0.6, label="Genuine")
    plt.xlabel("Similarity score (%)")
    plt.ylabel("Count")
    plt.title(f"Score Histogram (user={USER_ID}, probe={sorted(USER_SAMPLES)})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "real_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def build_hist_pdf(scores, bins):
    counts, edges = np.histogram(scores, bins=bins)
    bw = float(edges[1] - edges[0])
    centers = (edges[:-1] + edges[1:]) / 2.0
    pdf = counts / (len(scores) * bw)  # density; sum(pdf*bw)=1
    return centers, pdf, bw


def plot_empirical_pdf_from_hist(genuine_scores, imposter_scores):
    plt.figure(figsize=(8, 6))

    if len(imposter_scores) > 0:
        ci, pi, bw = build_hist_pdf(imposter_scores, HIST_BINS)
        plt.plot(ci, pi, marker="o", label="Impostor PDF (hist)")

    if len(genuine_scores) > 0:
        cg, pg, bw = build_hist_pdf(genuine_scores, HIST_BINS)
        plt.plot(cg, pg, marker="o", label="Genuine PDF (hist)")

    plt.xlabel("Similarity score (%)")
    plt.ylabel("Probability Density")
    plt.title(f"Empirical PDF from Histogram (user={USER_ID})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================
# Peak-to-peak interpolation + smoothing (renormalized)
# =========================
def interp_and_smooth_pdf(centers, pdf, step=0.1, sigma_points=3.0, eps=0.0):
    """
    Interpolate ONLY between non-zero bins in the middle ("peak-to-peak across support"),
    and force 0 only in the tails (outside support).
    Then Gaussian-smooth + renormalize area to 1.

    eps: treat pdf<=eps as zero (useful if you later have tiny numerical noise)
    """

    centers = np.asarray(centers, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    x_fine = np.arange(float(centers[0]), float(centers[-1]) + 1e-9, step)
    y_fine = np.zeros_like(x_fine)

    support_mask = pdf > eps
    if np.count_nonzero(support_mask) < 2:
        # Not enough support points to interpolate meaningfully.
        # Fall back to standard interpolation on all centers.
        y_fine = np.interp(x_fine, centers, pdf)
    else:
        c_sup = centers[support_mask]
        p_sup = pdf[support_mask]

        # define support interval [first_nonzero, last_nonzero]
        x0 = float(c_sup[0])
        x1 = float(c_sup[-1])

        inside = (x_fine >= x0) & (x_fine <= x1)

        # interpolate only using non-zero bins across the support
        y_fine[inside] = np.interp(x_fine[inside], c_sup, p_sup)

        # tails remain 0 (already)

    # smooth
    y_s = gaussian_filter1d(y_fine, sigma=sigma_points, mode="nearest")

    # renormalize (numpy 3.14+: trapezoid)
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
    plt.savefig(OUTPUT_DIR / "real_smoothed_pdf.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# =========================
# FAR/FRR from smoothed PDFs + EER intersection
# =========================
def compute_far_frr_from_smoothed_pdf(genuine_scores, imposter_scores, step=0.1, sigma_points=3.0):
    cg, pg, _ = build_hist_pdf(genuine_scores, HIST_BINS)
    ci, pi, _ = build_hist_pdf(imposter_scores, HIST_BINS)

    xg, yg = interp_and_smooth_pdf(cg, pg, step=step, sigma_points=sigma_points)
    xi, yi = interp_and_smooth_pdf(ci, pi, step=step, sigma_points=sigma_points)

    # Use common grid (they should match if centers range matches; enforce same x)
    if not np.allclose(xg, xi):
        # Align by intersection grid
        x_min = max(xg[0], xi[0])
        x_max = min(xg[-1], xi[-1])
        x = np.arange(x_min, x_max + 1e-9, step)
        yg = np.interp(x, xg, yg)
        yi = np.interp(x, xi, yi)
    else:
        x = xg

    dx = float(step)

    # FRR(T) = ∫_{-∞}^{T} f_g(s) ds  (left tail)
    # FAR(T) = ∫_{T}^{∞} f_i(s) ds   (right tail)
    cdf_g = np.cumsum(yg) * dx
    # survival_i(T) = integral from T to end
    surv_i = np.flip(np.cumsum(np.flip(yi))) * dx

    # thresholds are the x grid
    thresholds = x
    frrs = cdf_g
    fars = surv_i

    # numeric safety (tiny drift)
    frrs = np.clip(frrs, 0.0, 1.0)
    fars = np.clip(fars, 0.0, 1.0)

    return thresholds, fars, frrs


def compute_eer_intersection(thresholds, fars, frrs):
    d = fars - frrs

    # exact
    exact = np.where(d == 0)[0]
    if len(exact) > 0:
        i = int(exact[0])
        return float(fars[i]), float(thresholds[i])

    # sign change
    sc = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if len(sc) == 0:
        i = int(np.argmin(np.abs(d)))
        return float(0.5 * (fars[i] + frrs[i])), float(thresholds[i])

    i = int(sc[0])
    t0, t1 = float(thresholds[i]), float(thresholds[i + 1])
    d0, d1 = float(d[i]), float(d[i + 1])

    alpha = d0 / (d0 - d1)  # where d crosses 0
    t_star = t0 + alpha * (t1 - t0)

    far_star = float(fars[i] + alpha * (fars[i + 1] - fars[i]))
    frr_star = float(frrs[i] + alpha * (frrs[i + 1] - frrs[i]))
    eer = 0.5 * (far_star + frr_star)

    return eer, t_star

def compute_p_success(thresholds, fars, frrs):
    p_success = (1 - fars) * (1 - frrs)

    # Maximum success
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
    plt.title("FAR / FRR vs Threshold (from smoothed histogram-PDF)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "real_far_frr.png", dpi=300, bbox_inches="tight")
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

    # EER point values (eer_threshold may be float -> interpolate)
    p_and_eer = float(np.interp(eer_threshold, thresholds, p_and))
    p_or_eer  = float(np.interp(eer_threshold, thresholds, p_or))

    return p_and, p_or, idx_and, idx_or, p_and_eer, p_or_eer

def plot_p_success(thresholds, p_success, eer, eer_threshold, max_success, max_threshold):
    plt.figure(figsize=(8, 6))

    plt.plot(thresholds, p_success, label="P_success(t)")

    # interpolate P_success at the (float) EER threshold
    p_eer = float(np.interp(eer_threshold, thresholds, p_success))

    # mark EER point (no vertical lines)
    plt.scatter(eer_threshold, p_eer, zorder=3, label=f"P_success@EER={p_eer:.4f} (T≈{eer_threshold:.2f})")

    # mark maximum point (max_threshold is integer threshold, so indexing is OK)
    plt.scatter(max_threshold, max_success, color="red", zorder=3,
                label=f"Max P_success={max_success:.4f} (T={max_threshold:.2f})")

    plt.xlabel("Threshold (%)")
    plt.ylabel("P_success")
    plt.title("Success Probability vs Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "real_p_success.png", dpi=300, bbox_inches="tight")
    plt.show()
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

    # Mark AND maximum
    plt.scatter(thresholds[idx_and], p_and[idx_and],label=f"  AND max, T={thresholds[idx_and]:.2f},{p_and[idx_and]:.3f}", zorder=3)

    # Mark OR maximum
    plt.scatter(thresholds[idx_or], p_or[idx_or],label=f"  OR max, T={thresholds[idx_or]:.2f}, {p_or[idx_or]:.3f}", zorder=3)

    # Mark EER points
    plt.scatter(eer_threshold, p_and_eer, label=f"  AND@EER, {p_and_eer:.3f}", zorder=4)

    plt.scatter(eer_threshold, p_or_eer, label=f"  OR@EER, {p_or_eer:.3f}", zorder=4)

    plt.xlabel("Threshold (%)")
    plt.ylabel("Success Probability")
    plt.title("Integrated Success vs Threshold (AND / OR)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "real_success_and_or.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# =========================
# Main
# =========================
if __name__ == "__main__":
    genuine, imposter = build_scores()

    print(f"[i] Genuine scores:  n={len(genuine)}  min={genuine.min():.0f}  max={genuine.max():.0f}  mean={genuine.mean():.2f}")
    print(f"[i] Impostor scores: n={len(imposter)} min={imposter.min():.0f} max={imposter.max():.0f} mean={imposter.mean():.2f}")

    plot_histograms(genuine, imposter)

    # Smoothed PDFs (peak-to-peak + gaussian)
    plot_smoothed_pdfs(genuine, imposter, step=PDF_FINE_STEP, sigma_points=SMOOTH_SIGMA_POINTS)

    # FAR/FRR/EER from smoothed PDF integration
    thresholds, fars, frrs = compute_far_frr_from_smoothed_pdf(
        genuine, imposter,
        step=PDF_FINE_STEP,
        sigma_points=SMOOTH_SIGMA_POINTS
    )

    eer, eer_threshold = compute_eer_intersection(thresholds, fars, frrs)

    print(f"[i] EER (smoothed-PDF integral) = {eer:.6f}")
    print(f"[i] EER threshold ≈ {eer_threshold:.3f}")

    plot_far_frr(thresholds, fars, frrs, eer, eer_threshold)

    p_success, max_success, max_threshold = compute_p_success(thresholds, fars, frrs)

    print(f"[i] Max P_success = {max_success:.4f}")
    print(f"[i] Max P_success threshold = {max_threshold}%")

    plot_p_success(thresholds,p_success,eer,eer_threshold,max_success,max_threshold)

    # Integrated success (discrete + biometric)
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

    print(f"[i] AND max P_success = {p_and[idx_and]:.4f} at T={thresholds[idx_and]}")
    print(f"[i] OR  max P_success = {p_or[idx_or]:.4f} at T={thresholds[idx_or]}")
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