import csv
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple

# =========================
# Config
# =========================
REPO_ROOT = Path("/Users/dormalka/Desktop/Dor/Paper").resolve()
SOURCEAFIS_DIR = REPO_ROOT / "sourceafis-demo"
LIVDET_ROOT = SOURCEAFIS_DIR / "livdet_preproc_png"

# Main source dirs
TRAIN_LIVE_DIR = LIVDET_ROOT / "Training" / "Digital_Persona" / "Live"
TEST_LIVE_DIR = LIVDET_ROOT / "Testing" / "Digital_Persona" / "Live"

# Fake material names to include from BOTH training and testing
FAKE_MATERIALS = [
    "Gelatine",
    "Woodglue",
    "Ecoflex",
    "Latex",
]

# User selection / probe selection
# You can change these later for a different user
PROBE_FILES = [
    "002_0_0.png",
    "002_0_1.png",
    "002_0_2.png",
    "002_0_3.png",
    "002_0_4.png",
    "002_0_5.png",
]

# Which genuine files belong to the same identity
GENUINE_GLOB = "002_0_*.png"
FAKE_GLOB = "002_0_*.png"

OUTPUT_DIR = REPO_ROOT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORES_CSV = OUTPUT_DIR / "figs" / "fig_spoofed_users" /"sourceafis_livdet_scores_best_of_probes_raw.csv"
AGGREGATED_SCORES_CSV = OUTPUT_DIR / "figs" / "fig_spoofed_users" /"sourceafis_livdet_scores_best_of_probes_aggregated.csv"

PDF_FINE_STEP = 0.1
SMOOTH_SIGMA_POINTS = 3.0
HIST_BINS = np.arange(0, 101, 2)


# =========================
# Small path helpers
# =========================
def source_tag(p: Path) -> str:
    """
    Produce a unique staged filename prefix based on original path.
    """
    parts = list(p.parts)
    try:
        i = parts.index("livdet_preproc_png")
        rel = Path(*parts[i + 1:])
    except ValueError:
        rel = p.name

    rel_str = str(rel).replace("/", "__").replace("\\", "__").replace(" ", "_")
    digest = hashlib.md5(str(p).encode("utf-8")).hexdigest()[:8]
    return f"{rel_str}__{digest}"


def stage_files(files, dst_dir: Path):
    """
    Stage files into dst_dir with unique names so train/test/fake folders
    can be merged safely into one candidate directory.
    Returns:
        staged_dir, mapping(staged_name -> original_path)
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}

    for src in files:
        src = Path(src).resolve()
        staged_name = f"{source_tag(src)}__{src.name}"
        dst = dst_dir / staged_name
        mapping[staged_name] = src
        try:
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)

    return dst_dir, mapping


def collect_matching_files(src_dir: Path, pattern: str):
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        return []
    return sorted(f.resolve() for f in src_dir.glob(pattern) if f.is_file())


def collect_probe_files(probe_dir: Path, probe_files):
    found = []
    for name in probe_files:
        p = (probe_dir / name).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Probe file not found: {p}")
        found.append(p)
    return found


def collect_genuine_candidates(train_live_dir, test_live_dir, genuine_glob, probe_paths):
    probe_set = {p.resolve() for p in probe_paths}

    candidates = []
    candidates.extend(collect_matching_files(train_live_dir, genuine_glob))
    candidates.extend(collect_matching_files(test_live_dir, genuine_glob))

    # Exclude exact probe files
    candidates = [p for p in candidates if p.resolve() not in probe_set]

    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def collect_fake_candidates(livdet_root: Path, materials, fake_glob):
    """
    Collect fake candidates from BOTH Training/Fake/* and Testing/Fake/*,
    matching the same identity glob as the genuine user.
    """
    candidates = []

    for split in ["Training", "Testing"]:
        for material in materials:
            fake_dir = livdet_root / split / "Digital_Persona" / "Fake" / material
            if fake_dir.is_dir():
                candidates.extend(collect_matching_files(fake_dir, fake_glob))

    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


# =========================
# SourceAFIS batch scoring
# =========================
def run_sourceafis_batch_best_of_probes():
    print("[i] TRAIN_LIVE_DIR =", TRAIN_LIVE_DIR)
    print("[i] TEST_LIVE_DIR  =", TEST_LIVE_DIR)
    print("[i] PROBE_FILES    =", PROBE_FILES)
    print("[i] GENUINE_GLOB   =", GENUINE_GLOB)
    print("[i] FAKE_MATERIALS =", FAKE_MATERIALS)
    print("[i] FAKE_GLOB      =", FAKE_GLOB)
    print("[i] RAW SCORES CSV =", SCORES_CSV)

    if not TRAIN_LIVE_DIR.is_dir():
        raise FileNotFoundError(f"Training live directory not found: {TRAIN_LIVE_DIR}")
    if not TEST_LIVE_DIR.is_dir():
        raise FileNotFoundError(f"Testing live directory not found: {TEST_LIVE_DIR}")

    probe_paths = collect_probe_files(TRAIN_LIVE_DIR, PROBE_FILES)
    genuine_candidates = collect_genuine_candidates(
        TRAIN_LIVE_DIR,
        TEST_LIVE_DIR,
        GENUINE_GLOB,
        probe_paths,
    )
    fake_candidates = collect_fake_candidates(
        LIVDET_ROOT,
        FAKE_MATERIALS,
        FAKE_GLOB,
    )

    if not probe_paths:
        raise ValueError("No probe files were found.")
    if not genuine_candidates:
        raise ValueError("No genuine candidates were found.")
    if not fake_candidates:
        raise ValueError("No fake candidates were found.")

    print(f"[i] Probe files count      : {len(probe_paths)}")
    print(f"[i] Genuine candidates    : {len(genuine_candidates)}")
    print(f"[i] Fake candidates       : {len(fake_candidates)}")

    with tempfile.TemporaryDirectory(prefix="livdet_best_probe_") as tmp:
        tmp_path = Path(tmp)

        staged_probe_dir, probe_map = stage_files(probe_paths, tmp_path / "probes")
        staged_genuine_dir, genuine_map = stage_files(genuine_candidates, tmp_path / "genuine_candidates")
        staged_fake_dir, fake_map = stage_files(fake_candidates, tmp_path / "fake_candidates")

        print("[i] Staged probe dir      :", staged_probe_dir)
        print("[i] Staged genuine dir    :", staged_genuine_dir)
        print("[i] Staged fake dir       :", staged_fake_dir)

        # We pass "*.png" because we staged only the intended probes into the probe dir
        exec_args = (
            f'"{staged_probe_dir}" '
            f'"*.png" '
            f'"{staged_genuine_dir}" '
            f'"{staged_fake_dir}" '
            f'"{SCORES_CSV}"'
        )

        cmd = [
            "mvn",
            "-DskipTests",
            "compile",
            "exec:java",
            "-Dexec.mainClass=LivDetBatchScorer",
            f"-Dexec.args={exec_args}",
        ]

        print("[i] Running SourceAFIS LivDet batch scorer...")
        print("[i] Working dir:", SOURCEAFIS_DIR)
        print("[i] Command:", " ".join(map(str, cmd)))

        subprocess.run(
            cmd,
            cwd=SOURCEAFIS_DIR,
            check=True,
            text=True,
        )

        return probe_map, genuine_map, fake_map


# =========================
# CSV parsing / aggregation
# =========================
def find_first_existing(row, candidates, required=True):
    for c in candidates:
        if c in row and str(row[c]).strip() != "":
            return row[c]
    if required:
        raise KeyError(f"Could not find any of columns: {candidates}")
    return None


def load_scores_from_csv_best_per_candidate():
    """
    Each CSV row is one comparison:
        probe (stored template) vs target (candidate)

    We want:
        final_score(target) = max score over all probes

    Group by:
        - kind
        - target
    """

    genuine_best = {}
    impostor_best = {}

    with open(SCORES_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)

        expected = {"kind", "probe", "target", "score"}
        got = set(reader.fieldnames or [])
        if not expected.issubset(got):
            raise ValueError(
                f"CSV columns mismatch. Expected at least {expected}, got {reader.fieldnames}"
            )

        rows_seen = 0
        for row in reader:
            rows_seen += 1

            kind = row["kind"].strip().lower()
            probe = row["probe"].strip()
            target = row["target"].strip()
            score = float(row["score"])

            # target is the candidate being evaluated
            if kind == "genuine":
                genuine_best[target] = max(score, genuine_best.get(target, -np.inf))
            elif kind in ("impostor", "imposter"):
                impostor_best[target] = max(score, impostor_best.get(target, -np.inf))
            else:
                raise ValueError(f"Unknown kind in CSV: {kind!r}")

    if rows_seen == 0:
        raise ValueError(f"CSV is empty: {SCORES_CSV}")

    genuine = np.array(list(genuine_best.values()), dtype=float)
    impostor = np.array(list(impostor_best.values()), dtype=float)

    if len(genuine) == 0:
        raise ValueError("No genuine scores loaded after best-over-probes aggregation.")
    if len(impostor) == 0:
        raise ValueError("No impostor scores loaded after best-over-probes aggregation.")

    return genuine, impostor, genuine_best, impostor_best


def save_aggregated_scores_csv(genuine_best, impostor_best):
    with open(AGGREGATED_SCORES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "target", "best_score_over_all_probes"])

        for target, score in sorted(genuine_best.items()):
            writer.writerow(["genuine", target, score])

        for target, score in sorted(impostor_best.items()):
            writer.writerow(["impostor", target, score])


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
# Plot histogram + smoothed PDF
# =========================
def plot_histograms(genuine_scores, impostor_scores):
    plt.figure(figsize=(8, 6))
    if len(impostor_scores) > 0:
        plt.hist(impostor_scores, bins=HIST_BINS, alpha=0.6, label="Impostor")
    if len(genuine_scores) > 0:
        plt.hist(genuine_scores, bins=HIST_BINS, alpha=0.6, label="Genuine")
    plt.xlabel("Normalized similarity score (%)")
    plt.ylabel("Count")
    plt.title("LivDet Score Histogram (best score over probes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figs" / "fig_spoofed_users" / "livdet_sourceafis_histogram_best_of_probes.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def export_histograms(genuine_scores, impostor_scores):
    out_dir = OUTPUT_DIR / "figs" / "fig_spoofed_users"
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_genuine, edges = np.histogram(genuine_scores, bins=HIST_BINS)
    hist_impostor, _ = np.histogram(impostor_scores, bins=HIST_BINS)

    centers = (edges[:-1] + edges[1:]) / 2.0

    with open(out_dir / "livdet_sourceafis_histogram_best_of_probes_data.txt", "w") as f:
        f.write("score genuine impostor\n")
        for s, g, i in zip(centers, hist_genuine, hist_impostor):
            f.write(f"{s:.6f} {g:d} {i:d}\n")

def build_hist_pdf(scores, bins):
    counts, edges = np.histogram(scores, bins=bins)
    bw = float(edges[1] - edges[0])
    centers = (edges[:-1] + edges[1:]) / 2.0
    pdf = counts / (len(scores) * bw)
    return centers, pdf, bw


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
    plt.title("LivDet Smoothed PDFs (best score over probes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figs" / "fig_spoofed_users" /"livdet_sourceafis_smoothed_pdf_best_of_probes.pdf", dpi=300, bbox_inches="tight")
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
    plt.plot(thresholds, fars, label="FAR")
    plt.plot(thresholds, frrs, label="FRR")
    plt.scatter(eer_threshold, eer, label=f"EER≈{eer:.4f} @ T≈{eer_threshold:.2f}", zorder=3)
    plt.xlabel("Threshold (%)")
    plt.ylabel("Error Rate")
    plt.title("LivDet FAR / FRR vs Threshold (best score over probes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figs" / "fig_spoofed_users" /"livdet_sourceafis_far_frr_best_of_probes.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def export_far_frr(thresholds, fars, frrs, eer, eer_threshold):
    out_dir = OUTPUT_DIR / "figs" / "fig_spoofed_users"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "livdet_sourceafis_far_frr_best_of_probes_data.txt", "w") as f:
        f.write("T FAR FRR\n")
        for t, fa, fr in zip(thresholds, fars, frrs):
            f.write(f"{t:.6f} {fa:.6f} {fr:.6f}\n")

    with open(out_dir / "livdet_sourceafis_far_frr_best_of_probes_points.txt", "w") as f:
        f.write("T_eer EER\n")
        f.write(f"{eer_threshold:.6f} {eer:.6f}\n")

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
    plt.title("LivDet Success Probability vs Threshold (best score over probes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figs" / "fig_spoofed_users" /"livdet_sourceafis_p_success_best_of_probes.pdf", dpi=300, bbox_inches="tight")
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

    plt.scatter(eer_threshold, p_and_eer, label=f"AND@EER, {p_and_eer:.3f}", zorder=4)
    plt.scatter(eer_threshold, p_or_eer, label=f"OR@EER, {p_or_eer:.3f}", zorder=4)

    plt.xlabel("Threshold (%)")
    plt.ylabel("Success Probability")
    plt.title("LivDet Integrated Success vs Threshold (AND / OR)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figs" / "fig_spoofed_users" /"livdet_sourceafis_success_and_or_best_of_probes.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def export_p_success(thresholds, p_success, eer_threshold, max_success, max_threshold):
    out_dir = OUTPUT_DIR / "figs" / "fig_spoofed_users"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_eer = float(np.interp(eer_threshold, thresholds, p_success))

    with open(out_dir / "livdet_sourceafis_p_success_best_of_probes_data.txt", "w") as f:
        f.write("T P_success\n")
        for t, p in zip(thresholds, p_success):
            f.write(f"{t:.6f} {p:.6f}\n")

    with open(out_dir / "livdet_sourceafis_p_success_best_of_probes_points.txt", "w") as f:
        f.write("T_eer P_eer T_opt P_opt\n")
        f.write(f"{eer_threshold:.6f} {p_eer:.6f} {max_threshold:.6f} {max_success:.6f}\n")


def export_success_and_or(
    thresholds,
    p_and,
    p_or,
    idx_and,
    idx_or,
    eer_threshold,
    p_and_eer,
    p_or_eer,
):
    out_dir = OUTPUT_DIR / "figs" / "fig_spoofed_users"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "livdet_sourceafis_success_and_or_best_of_probes_data.txt", "w") as f:
        f.write("T P_and P_or\n")
        for t, pa, po in zip(thresholds, p_and, p_or):
            f.write(f"{t:.6f} {pa:.6f} {po:.6f}\n")

    with open(out_dir / "livdet_sourceafis_success_and_or_best_of_probes_points.txt", "w") as f:
        f.write("T_and_opt P_and_opt T_or_opt P_or_opt T_eer P_and_eer P_or_eer\n")
        f.write(
            f"{thresholds[idx_and]:.6f} {p_and[idx_and]:.6f} "
            f"{thresholds[idx_or]:.6f} {p_or[idx_or]:.6f} "
            f"{eer_threshold:.6f} {p_and_eer:.6f} {p_or_eer:.6f}\n"
        )

# =========================
# Sweep over P_safe configurations
# =========================
def generate_psafe_configs(
    safe_start: float = 0.50,
    safe_end: float = 0.90,
    safe_step: float = 0.01,
    *,
    leak_case_base: Tuple[float, float, float, float] = (0.50, 0.45, 0.04, 0.01),
    loss_case_base: Tuple[float, float, float, float] = (0.50, 0.04, 0.45, 0.01),
) -> List[Dict[str, float]]:
    """
    Generate two families of configurations:

    1) leak_case:
       P_safe increases from 0.50 to 0.90,
       P_leak decreases by the same amount,
       P_loss and P_theft remain fixed.

    2) loss_case:
       P_safe increases from 0.50 to 0.90,
       P_loss decreases by the same amount,
       P_leak and P_theft remain fixed.

    Each family must start from a valid base tuple:
        (P_safe, P_leak, P_loss, P_theft)

    Returns a list of dicts, one dict per configuration.
    """
    configs: List[Dict[str, float]] = []

    safe_values = np.round(np.arange(safe_start, safe_end + 1e-12, safe_step), 2)

    # ---- leak-decreasing family ----
    base_safe, base_leak, base_loss, base_theft = leak_case_base
    for psafe in safe_values:
        delta = round(psafe - base_safe, 10)
        pleak = round(base_leak - delta, 10)
        ploss = round(base_loss, 10)
        ptheft = round(base_theft, 10)

        if pleak < -1e-12:
            raise ValueError(
                f"Leak-case invalid: P_leak became negative for P_safe={psafe:.2f}. "
                f"Choose a larger base leak or a smaller safe range."
            )

        total = psafe + pleak + ploss + ptheft
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Leak-case probabilities do not sum to 1: "
                f"P_safe={psafe}, P_leak={pleak}, P_loss={ploss}, P_theft={ptheft}, total={total}"
            )

        configs.append({
            "family": "decrease_leak",
            "P_safe": psafe,
            "P_leak": pleak,
            "P_loss": ploss,
            "P_theft": ptheft,
        })

    # ---- loss-decreasing family ----
    base_safe, base_leak, base_loss, base_theft = loss_case_base
    for psafe in safe_values:
        delta = round(psafe - base_safe, 10)
        ploss = round(base_loss - delta, 10)
        pleak = round(base_leak, 10)
        ptheft = round(base_theft, 10)

        if ploss < -1e-12:
            raise ValueError(
                f"Loss-case invalid: P_loss became negative for P_safe={psafe:.2f}. "
                f"Choose a larger base loss or a smaller safe range."
            )

        total = psafe + pleak + ploss + ptheft
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Loss-case probabilities do not sum to 1: "
                f"P_safe={psafe}, P_leak={pleak}, P_loss={ploss}, P_theft={ptheft}, total={total}"
            )

        configs.append({
            "family": "decrease_loss",
            "P_safe": psafe,
            "P_leak": pleak,
            "P_loss": ploss,
            "P_theft": ptheft,
        })

    return configs


def evaluate_config_success(
    thresholds: np.ndarray,
    fars: np.ndarray,
    frrs: np.ndarray,
    eer_threshold: float,
    cfg: Dict[str, float],
) -> Dict[str, float]:
    """
    Evaluate one configuration and return:
      - maximal AND
      - maximal OR
      - EER AND
      - EER OR
      - best maximal config
      - improvement in failure percentage as requested
    """
    p_and, p_or, idx_and, idx_or, p_and_eer, p_or_eer = compute_success_and_or(
        thresholds,
        fars,
        frrs,
        eer_threshold=eer_threshold,
        P_safe=cfg["P_safe"],
        P_leak=cfg["P_leak"],
        P_loss=cfg["P_loss"],
        P_theft=cfg["P_theft"],
    )

    max_and = float(p_and[idx_and])
    max_or = float(p_or[idx_or])
    eer_and = float(p_and_eer)
    eer_or = float(p_or_eer)

    if max_and >= max_or:
        best_kind = "AND"
        best_max = max_and
        eer_of_maximal_config = eer_and
        best_threshold = float(thresholds[idx_and])
    else:
        best_kind = "OR"
        best_max = max_or
        eer_of_maximal_config = eer_or
        best_threshold = float(thresholds[idx_or])

    denom_reference = eer_of_maximal_config
    denom_failure = 1.0 - denom_reference

    if np.isclose(denom_failure, 0.0):
        improvement_failure_ratio = np.nan
    else:
        improvement_failure_ratio = (best_max - denom_reference) / denom_failure

    return {
        "family": cfg["family"],
        "P_safe": float(cfg["P_safe"]),
        "P_leak": float(cfg["P_leak"]),
        "P_loss": float(cfg["P_loss"]),
        "P_theft": float(cfg["P_theft"]),
        "max_AND": max_and,
        "max_OR": max_or,
        "eer_AND": eer_and,
        "eer_OR": eer_or,
        "best_kind": best_kind,
        "best_max": best_max,
        "eer_of_maximal_config": eer_of_maximal_config,
        "best_threshold": best_threshold,
        "failure_improvement_ratio": float(improvement_failure_ratio),
    }


def sweep_psafe_success(
    thresholds: np.ndarray,
    fars: np.ndarray,
    frrs: np.ndarray,
    eer_threshold: float,
    *,
    safe_start: float = 0.50,
    safe_end: float = 0.90,
    safe_step: float = 0.01,
    leak_case_base: Tuple[float, float, float, float] = (0.50, 0.45, 0.04, 0.01),
    loss_case_base: Tuple[float, float, float, float] = (0.50, 0.04, 0.45, 0.01),
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Sweep all requested configurations and return:
      1) all_rows: detailed results for every configuration
      2) family_best: best row in each family according to best_max
      3) overall_best: best row overall according to best_max
    """
    configs = generate_psafe_configs(
        safe_start=safe_start,
        safe_end=safe_end,
        safe_step=safe_step,
        leak_case_base=leak_case_base,
        loss_case_base=loss_case_base,
    )

    all_rows: List[Dict[str, float]] = []
    for cfg in configs:
        row = evaluate_config_success(
            thresholds=thresholds,
            fars=fars,
            frrs=frrs,
            eer_threshold=eer_threshold,
            cfg=cfg,
        )
        all_rows.append(row)

    leak_rows = [r for r in all_rows if r["family"] == "decrease_leak"]
    loss_rows = [r for r in all_rows if r["family"] == "decrease_loss"]

    family_best = {
        "decrease_leak": max(leak_rows, key=lambda r: r["best_max"]),
        "decrease_loss": max(loss_rows, key=lambda r: r["best_max"]),
    }

    overall_best = max(all_rows, key=lambda r: r["best_max"])

    return all_rows, family_best, overall_best


def print_psafe_sweep_summary(
    all_rows: List[Dict[str, float]],
    family_best: Dict[str, Dict[str, float]],
    overall_best: Dict[str, float],
) -> None:
    """
    Print:
      - overall 4 requested results from the overall-best configuration
      - family-best summaries
    """
    print("\n" + "=" * 80)
    print("[SWEEP SUMMARY] OVERALL BEST CONFIGURATION")
    print("=" * 80)
    print(f"[i] Family                    : {overall_best['family']}")
    print(f"[i] P_safe                    : {overall_best['P_safe']:.2f}")
    print(f"[i] P_leak                    : {overall_best['P_leak']:.2f}")
    print(f"[i] P_loss                    : {overall_best['P_loss']:.2f}")
    print(f"[i] P_theft                   : {overall_best['P_theft']:.2f}")
    print(f"[i] Maximal AND               : {overall_best['max_AND']:.6f}")
    print(f"[i] Maximal OR                : {overall_best['max_OR']:.6f}")
    print(f"[i] EER AND                   : {overall_best['eer_AND']:.6f}")
    print(f"[i] EER OR                    : {overall_best['eer_OR']:.6f}")
    print(f"[i] Best configuration kind   : {overall_best['best_kind']}")
    print(f"[i] Best threshold            : {overall_best['best_threshold']:.6f}")
    print(f"[i] Best maximal success      : {overall_best['best_max']:.6f}")
    print(f"[i] EER of maximal config     : {overall_best['eer_of_maximal_config']:.6f}")
    print(f"[i] Failure improvement ratio : {overall_best['failure_improvement_ratio']:.6f}")

    print("\n" + "=" * 80)
    print("[SWEEP SUMMARY] BEST PER FAMILY")
    print("=" * 80)
    for family_name, row in family_best.items():
        print(f"\n[i] Family                    : {family_name}")
        print(f"[i] P_safe                    : {row['P_safe']:.2f}")
        print(f"[i] P_leak                    : {row['P_leak']:.2f}")
        print(f"[i] P_loss                    : {row['P_loss']:.2f}")
        print(f"[i] P_theft                   : {row['P_theft']:.2f}")
        print(f"[i] Maximal AND               : {row['max_AND']:.6f}")
        print(f"[i] Maximal OR                : {row['max_OR']:.6f}")
        print(f"[i] EER AND                   : {row['eer_AND']:.6f}")
        print(f"[i] EER OR                    : {row['eer_OR']:.6f}")
        print(f"[i] Best configuration kind   : {row['best_kind']}")
        print(f"[i] Best threshold            : {row['best_threshold']:.6f}")
        print(f"[i] Best maximal success      : {row['best_max']:.6f}")
        print(f"[i] EER of maximal config     : {row['eer_of_maximal_config']:.6f}")
        print(f"[i] Failure improvement ratio : {row['failure_improvement_ratio']:.6f}")


def save_psafe_sweep_csv(
    all_rows: List[Dict[str, float]],
    out_csv: Path,
) -> None:
    """
    Save all sweep results to CSV.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "family",
        "P_safe",
        "P_leak",
        "P_loss",
        "P_theft",
        "max_AND",
        "max_OR",
        "eer_AND",
        "eer_OR",
        "best_kind",
        "best_max",
        "eer_of_maximal_config",
        "best_threshold",
        "failure_improvement_ratio",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

def print_psafe_summary_simple(csv_path: Path):
    """
    Prints ONLY:
    - overall improvement range
    - best improvement + values
    - minimal improvement + values
    """

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "P_safe": float(row["P_safe"]),
                "P_leak": float(row["P_leak"]),
                "P_loss": float(row["P_loss"]),
                "P_theft": float(row["P_theft"]),
                "max_AND": float(row["max_AND"]),
                "max_OR": float(row["max_OR"]),
                "eer_AND": float(row["eer_AND"]),
                "eer_OR": float(row["eer_OR"]),
                "best_max": float(row["best_max"]),
                "failure_improvement_ratio": float(row["failure_improvement_ratio"]),
            })

    if not rows:
        print("No data found.")
        return

    # Convert to percentages
    for r in rows:
        r["improvement_pct"] = 100 * r["failure_improvement_ratio"]

    # Find best and worst
    best_row = max(rows, key=lambda r: r["improvement_pct"])
    worst_row = min(rows, key=lambda r: r["improvement_pct"])

    min_imp = worst_row["improvement_pct"]
    max_imp = best_row["improvement_pct"]

    # ===== PRINT =====
    print(f"Overall improvement range: {min_imp:.2f}% - {max_imp:.2f}%\n")

    print("Best improvement:")
    print(f"{max_imp:.2f}%")
    print(f"Psafe: {best_row['P_safe']:.2f}")
    print(f"Pleak: {best_row['P_leak']:.2f}")
    print(f"Ploss: {best_row['P_loss']:.2f}")
    print(f"Ptheft: {best_row['P_theft']:.2f}")
    print(f"Max AND: {best_row['max_AND']:.6f}")
    print(f"Max OR: {best_row['max_OR']:.6f}")
    print(f"EER AND: {best_row['eer_AND']:.6f}")
    print(f"EER OR: {best_row['eer_OR']:.6f}")

    print("\nMinimal improvement:")
    print(f"{min_imp:.2f}%")
    print(f"Psafe: {worst_row['P_safe']:.2f}")
    print(f"Pleak: {worst_row['P_leak']:.2f}")
    print(f"Ploss: {worst_row['P_loss']:.2f}")
    print(f"Ptheft: {worst_row['P_theft']:.2f}")
    print(f"Max AND: {worst_row['max_AND']:.6f}")
    print(f"Max OR: {worst_row['max_OR']:.6f}")
    print(f"EER AND: {worst_row['eer_AND']:.6f}")
    print(f"EER OR: {worst_row['eer_OR']:.6f}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    run_sourceafis_batch_best_of_probes()

    genuine_raw, impostor_raw, genuine_best, impostor_best = load_scores_from_csv_best_per_candidate()
    save_aggregated_scores_csv(genuine_best, impostor_best)

    print(f"[i] Aggregated genuine scores : n={len(genuine_raw)}  min={genuine_raw.min():.4f}  max={genuine_raw.max():.4f}  mean={genuine_raw.mean():.4f}")
    print(f"[i] Aggregated impostor scores: n={len(impostor_raw)} min={impostor_raw.min():.4f} max={impostor_raw.max():.4f} mean={impostor_raw.mean():.4f}")
    print(f"[i] Aggregated CSV saved to   : {AGGREGATED_SCORES_CSV}")

    genuine, impostor, raw_max = normalize_scores_to_100(genuine_raw, impostor_raw)

    print(f"[i] Normalization factor (raw max) = {raw_max:.4f}")
    print(f"[i] Normalized genuine scores : min={genuine.min():.4f}  max={genuine.max():.4f}  mean={genuine.mean():.4f}")
    print(f"[i] Normalized impostor scores: min={impostor.min():.4f} max={impostor.max():.4f} mean={impostor.mean():.4f}")

    plot_histograms(genuine, impostor)
    export_histograms(genuine, impostor)
    plot_smoothed_pdfs(genuine, impostor, step=PDF_FINE_STEP, sigma_points=SMOOTH_SIGMA_POINTS)

    thresholds, fars, frrs = compute_far_frr_from_smoothed_pdf(
        genuine,
        impostor,
        step=PDF_FINE_STEP,
        sigma_points=SMOOTH_SIGMA_POINTS,
    )

    eer, eer_threshold = compute_eer_intersection(thresholds, fars, frrs)

    print(f"[i] EER = {eer:.6f}")
    print(f"[i] EER threshold ≈ {eer_threshold:.4f}")

    plot_far_frr(thresholds, fars, frrs, eer, eer_threshold)
    export_far_frr(thresholds, fars, frrs, eer, eer_threshold)
    p_success, max_success, max_threshold = compute_p_success(thresholds, fars, frrs)

    print(f"[i] Max P_success = {max_success:.4f}")
    print(f"[i] Max P_success threshold = {max_threshold:.4f}")

    plot_p_success(thresholds, p_success, eer_threshold, max_success, max_threshold)

    p_and, p_or, idx_and, idx_or, p_and_eer, p_or_eer = compute_success_and_or(
        thresholds,
        fars,
        frrs,
        eer_threshold=eer_threshold,
        P_safe=0.8,
        P_leak=0.15,
        P_loss=0.04,
        P_theft=0.01,
    )
    export_p_success(thresholds, p_success, eer_threshold, max_success, max_threshold)
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
    export_success_and_or(
        thresholds,
        p_and,
        p_or,
        idx_and,
        idx_or,
        eer_threshold,
        p_and_eer,
        p_or_eer,
    )
        # =========================
    # New sweep requested by user
    # =========================
    SWEEP_RESULTS_CSV = OUTPUT_DIR / "figs" / "fig_spoofed_users" / "psafe_sweep_results.csv"

    all_rows, family_best, overall_best = sweep_psafe_success(
        thresholds,
        fars,
        frrs,
        eer_threshold,
        safe_start=0.55,
        safe_end=0.90,
        safe_step=0.01,
        # Case 1: increasing P_safe comes from decreasing P_leak
        leak_case_base=(0.55, 0.4, 0.04, 0.01),
        # Case 2: increasing P_safe comes from decreasing P_loss
        loss_case_base=(0.55, 0.04, 0.4, 0.01),
    )

    print_psafe_sweep_summary(all_rows, family_best, overall_best)
    save_psafe_sweep_csv(all_rows, SWEEP_RESULTS_CSV)
    print(f"[i] Sweep CSV saved to: {SWEEP_RESULTS_CSV}")
    print_psafe_summary_simple(SWEEP_RESULTS_CSV)   