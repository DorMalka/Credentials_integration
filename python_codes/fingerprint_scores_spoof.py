import csv
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


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
]

# Which genuine files belong to the same identity
GENUINE_GLOB = "002_0_*.png"
FAKE_GLOB = "002_0_*.png"

OUTPUT_DIR = REPO_ROOT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORES_CSV = OUTPUT_DIR / "sourceafis_livdet_scores_best_of_probes_raw.csv"
AGGREGATED_SCORES_CSV = OUTPUT_DIR / "sourceafis_livdet_scores_best_of_probes_aggregated.csv"

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
    plt.savefig(OUTPUT_DIR / "livdet_sourceafis_histogram_best_of_probes.pdf", dpi=300, bbox_inches="tight")
    plt.close()


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
    plt.savefig(OUTPUT_DIR / "livdet_sourceafis_smoothed_pdf_best_of_probes.pdf", dpi=300, bbox_inches="tight")
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
    plt.savefig(OUTPUT_DIR / "livdet_sourceafis_far_frr_best_of_probes.pdf", dpi=300, bbox_inches="tight")
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
    plt.title("LivDet Success Probability vs Threshold (best score over probes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "livdet_sourceafis_p_success_best_of_probes.pdf", dpi=300, bbox_inches="tight")
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
    plt.savefig(OUTPUT_DIR / "livdet_sourceafis_success_and_or_best_of_probes.pdf", dpi=300, bbox_inches="tight")
    plt.close()


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