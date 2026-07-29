import csv
import re
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

# Choose one dataset here.
#DATA_DIR = SOURCEAFIS_DIR / "fvc2002_png" / "DB1_B"
DATA_DIR = SOURCEAFIS_DIR / "fvc2004_png" / "DB1_B"

# Multi-user evaluation:
# - Set USER_IDS to an explicit list, e.g. [101, 102, 103, 104, 105], or
# - Leave it as None and the script will use the first NUMBER_OF_USERS found.
USER_IDS = None
NUMBER_OF_USERS = 10
PROBES_PER_USER = 3

# Use only the selected users to construct the impostor distribution.
# For every unordered pair of selected users {u, v}, only one direction is kept:
# the smaller user ID is enrolled and templates of the larger user ID are candidates.
# Example: keep 101 -> 102 and discard 102 -> 101.
IMPOSTORS_ONLY_AMONG_SELECTED_USERS = True

OUTPUT_DIR = REPO_ROOT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORES_DIR = OUTPUT_DIR / "sourceafis_multiuser_scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)

# Maximum raw impostor comparisons generated for EACH enrolled user.
# Set to None to use all comparisons.
MAX_IMPOSTER_SCORES_PER_USER = 20000

PDF_FINE_STEP = 0.1
SMOOTH_SIGMA_POINTS = 3.0

# Fixed normalized bins: 0..100
HIST_BINS = np.arange(0, 101, 2)

# Matches filenames such as 101_1.png or 101-1.png.
FILENAME_PATTERN = re.compile(r"^(?P<user>\d+)[_-](?P<sample>\d+)")


def discover_dataset_samples():
    """
    Return:
        {
            user_id: {
                sample_id: Path(...)
            }
        }

    The dataset images must have names such as 101_1.png.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {DATA_DIR}")

    users = {}

    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file():
            continue

        match = FILENAME_PATTERN.match(path.stem)
        if match is None:
            continue

        user_id = int(match.group("user"))
        sample_id = int(match.group("sample"))
        users.setdefault(user_id, {})[sample_id] = path

    if not users:
        raise ValueError(
            f"No fingerprint images were discovered in {DATA_DIR}. "
            "Expected filenames such as 101_1.png."
        )

    return users


def select_users_and_probes():
    """
    Select several users and the first three available samples of each user
    as probe/enrolment templates.
    """
    dataset_users = discover_dataset_samples()

    if USER_IDS is None:
        selected_user_ids = sorted(dataset_users)[:NUMBER_OF_USERS]
    else:
        selected_user_ids = list(USER_IDS)

    if not selected_user_ids:
        raise ValueError("No users were selected.")

    selections = {}

    for user_id in selected_user_ids:
        if user_id not in dataset_users:
            raise ValueError(f"User {user_id} was not found in {DATA_DIR}")

        available_samples = sorted(dataset_users[user_id])

        if len(available_samples) <= PROBES_PER_USER:
            raise ValueError(
                f"User {user_id} has only {len(available_samples)} samples. "
                f"At least {PROBES_PER_USER + 1} are required: "
                f"{PROBES_PER_USER} probes and at least one genuine candidate."
            )

        selections[user_id] = tuple(available_samples[:PROBES_PER_USER])

    return selections


# =========================
# SourceAFIS batch scoring
# =========================
def run_sourceafis_batch(user_id, probe_samples, scores_csv):
    """
    Run the existing Java BatchScorer for one enrolled user.

    The Java program is expected to:
      1. Compare every candidate against all selected probes.
      2. Write one CSV row per probe-candidate comparison.
      3. Mark each row as genuine or impostor.
    """
    probe_csv = ",".join(str(x) for x in sorted(probe_samples))
    max_imp = (
        -1
        if MAX_IMPOSTER_SCORES_PER_USER is None
        else int(MAX_IMPOSTER_SCORES_PER_USER)
    )

    cmd = [
        "mvn",
        "-q",
        "-DskipTests",
        "compile",
        "exec:java",
        "-Dexec.mainClass=BatchScorer",
        (
            f"-Dexec.args={DATA_DIR} {user_id} {probe_csv} "
            f"{scores_csv} {max_imp}"
        ),
    ]

    print()
    print(f"[i] Running SourceAFIS for user {user_id}")
    print(f"[i] Probe samples: {sorted(probe_samples)}")
    print("[i] Working dir:", SOURCEAFIS_DIR)
    print("[i] Command:", " ".join(map(str, cmd)))

    subprocess.run(
        cmd,
        cwd=SOURCEAFIS_DIR,
        check=True,
        text=True,
    )


def extract_user_and_sample(candidate):
    """
    Try to parse a candidate identifier such as:
      /some/path/103_4.png
      103_4.png
      103-4

    Returns (user_id, sample_id), or (None, None) when parsing fails.
    """
    stem = Path(candidate).stem
    match = FILENAME_PATTERN.match(stem)

    if match is None:
        return None, None

    return int(match.group("user")), int(match.group("sample"))


def load_best_scores_from_csv(
    scores_csv,
    enrolled_user_id,
    probe_samples,
    selected_user_ids,
):
    """
    Collapse all comparisons between the same candidate and the three probes
    into one score:

        candidate_score = max(score(probe_1, candidate),
                              score(probe_2, candidate),
                              score(probe_3, candidate))

    Returns:
      - one genuine score per non-probe template of the enrolled user;
      - one impostor score per candidate template after removing mirrored
        user-pair comparisons.

    For selected users, only one direction is retained for each unordered pair.
    With users 101 and 102, this keeps:
        enrolled 101 -> candidate templates of 102
    and discards:
        enrolled 102 -> candidate templates of 101
    """
    genuine_best = {}
    impostor_best = {}
    selected_user_ids = set(selected_user_ids)

    with open(scores_csv, "r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {scores_csv}")

        required = {"kind", "score"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV {scores_csv} is missing columns {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        candidate_col = next(
            (col for col in ("target", "candidate") if col in reader.fieldnames),
            None,
        )

        if candidate_col is None:
            raise ValueError(
                f"CSV must contain a candidate identifier column named "
                f"'target' or 'candidate'. Found: {reader.fieldnames}"
            )

        for row in reader:
            kind = row["kind"].strip().lower()
            score = float(row["score"])
            candidate = row[candidate_col].strip()

            candidate_user, candidate_sample = extract_user_and_sample(candidate)

            # Do not let an enrolment/probe image become a genuine test sample.
            if (
                candidate_user == enrolled_user_id
                and candidate_sample in probe_samples
            ):
                continue

            if kind == "genuine":
                old_score = genuine_best.get(candidate)
                if old_score is None or score > old_score:
                    genuine_best[candidate] = score

            elif kind in {"impostor", "imposter"}:
                if candidate_user is None:
                    raise ValueError(
                        f"Could not parse impostor candidate identifier: {candidate}"
                    )

                # Optionally restrict the impostor population to the selected
                # evaluation cohort.
                if (
                    IMPOSTORS_ONLY_AMONG_SELECTED_USERS
                    and candidate_user not in selected_user_ids
                ):
                    continue

                # Remove mirrored user-pair comparisons.
                #
                # Keep only:
                #     enrolled_user_id < candidate_user
                #
                # Example:
                #   keep 101 probes against user 102 templates;
                #   discard 102 probes against user 101 templates.
                if candidate_user <= enrolled_user_id:
                    continue

                old_score = impostor_best.get(candidate)
                if old_score is None or score > old_score:
                    impostor_best[candidate] = score

    if not genuine_best:
        raise ValueError(
            f"No genuine candidate scores were loaded for user {enrolled_user_id} "
            f"from {scores_csv}"
        )

    print(
        f"[i] User {enrolled_user_id}: "
        f"{len(genuine_best)} genuine candidates, "
        f"{len(impostor_best)} non-mirrored impostor candidates"
    )

    return list(genuine_best.values()), list(impostor_best.values())


def collect_multiuser_scores():
    """
    Run the experiment for every selected user and pool the resulting scores.

    Mirrored impostor directions are removed. For each unordered pair of
    selected users, only the direction from the smaller user ID to the larger
    user ID is retained.
    """
    selections = select_users_and_probes()

    print("[i] Selected users and probes:")
    for user_id, probes in selections.items():
        print(f"    user {user_id}: {list(probes)}")

    all_genuine = []
    all_impostor = []

    for user_id, probe_samples in selections.items():
        scores_csv = SCORES_DIR / f"sourceafis_scores_user_{user_id}.csv"

        run_sourceafis_batch(
            user_id=user_id,
            probe_samples=probe_samples,
            scores_csv=scores_csv,
        )

        genuine_scores, impostor_scores = load_best_scores_from_csv(
            scores_csv=scores_csv,
            enrolled_user_id=user_id,
            probe_samples=set(probe_samples),
            selected_user_ids=selections.keys(),
        )

        all_genuine.extend(genuine_scores)
        all_impostor.extend(impostor_scores)

    genuine = np.asarray(all_genuine, dtype=float)
    impostor = np.asarray(all_impostor, dtype=float)

    if genuine.size == 0:
        raise ValueError("The pooled genuine distribution is empty.")
    if impostor.size == 0:
        raise ValueError("The pooled impostor distribution is empty.")

    print()
    print(f"[i] Total enrolled users: {len(selections)}")
    print(f"[i] Total pooled genuine scores: {len(genuine)}")
    print(f"[i] Total pooled impostor scores: {len(impostor)}")
    print(
        "[i] Mirrored selected-user comparisons were removed "
        "(for example, 102 -> 101 is discarded when 101 -> 102 is kept)."
    )

    return genuine, impostor, selections


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
    plt.title("Multi-user score histogram (three probes per user)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figs" / "fig_different_users_sourceafsi" / "sourceafis_histogram.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def export_histograms(genuine_scores, impostor_scores):
    out_dir = OUTPUT_DIR / "figs" / "fig_different_users_sourceafsi"
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = HIST_BINS

    # Compute histograms
    hist_g, edges = np.histogram(genuine_scores, bins=bins)
    hist_i, _     = np.histogram(impostor_scores, bins=bins)

    # Bin centers
    centers = (edges[:-1] + edges[1:]) / 2

    with open(out_dir / "sourceafis_histogram_data.txt", "w") as f:
        f.write("bin genuine impostor\n")
        for c, g, i in zip(centers, hist_g, hist_i):
            f.write(f"{c:.4f} {g} {i}\n")

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
    plt.savefig(OUTPUT_DIR / "figs" / "fig_different_users_sourceafsi" / "sourceafis_smoothed_pdf.pdf", dpi=300, bbox_inches="tight")
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
    idx_eer = np.argmin(np.abs(fars - frrs))
    max_threshold = thresholds[idx_max]
    max_success = p_success[idx_max]
    eer_success = p_success[idx_eer]
    return p_success,eer_success, max_success, max_threshold


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
    plt.savefig(OUTPUT_DIR / "figs" / "fig_different_users_sourceafsi" / "sourceafis_far_frr.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def export_far_frr(thresholds, fars, frrs, eer, eer_threshold):
    out_dir = OUTPUT_DIR / "figs" / "fig_different_users_sourceafsi"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Curve data
    with open(out_dir / "sourceafis_far_frr_data.txt", "w") as f:
        f.write("T FAR FRR\n")
        for t, fa, fr in zip(thresholds, fars, frrs):
            f.write(f"{t:.6f} {fa:.6f} {fr:.6f}\n")

    # EER point
    with open(out_dir / "sourceafis_far_frr_points.txt", "w") as f:
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
    plt.title("Success Probability vs Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figs" / "fig_different_users_sourceafsi" / "sourceafis_p_success.pdf", dpi=300, bbox_inches="tight")
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
    plt.savefig(OUTPUT_DIR / "figs" / "fig_different_users_sourceafsi" / "sourceafis_success_and_or.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def export_p_success(thresholds, p_success, eer_threshold, max_success, max_threshold):
    out_dir = OUTPUT_DIR / "figs" / "fig_different_users_sourceafsi"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_eer = float(np.interp(eer_threshold, thresholds, p_success))

    with open(out_dir / "sourceafis_p_success_data.txt", "w") as f:
        f.write("T P_success\n")
        for t, p in zip(thresholds, p_success):
            f.write(f"{t:.6f} {p:.6f}\n")

    with open(out_dir / "sourceafis_p_success_points.txt", "w") as f:
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
    out_dir = OUTPUT_DIR / "figs" / "fig_different_users_sourceafsi"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "sourceafis_success_and_or_data.txt", "w") as f:
        f.write("T P_and P_or\n")
        for t, pa, po in zip(thresholds, p_and, p_or):
            f.write(f"{t:.6f} {pa:.6f} {po:.6f}\n")

    with open(out_dir / "sourceafis_success_and_or_points.txt", "w") as f:
        f.write("T_and_opt P_and_opt T_or_opt P_or_opt T_eer P_and_eer P_or_eer\n")
        f.write(
            f"{thresholds[idx_and]:.6f} {p_and[idx_and]:.6f} "
            f"{thresholds[idx_or]:.6f} {p_or[idx_or]:.6f} "
            f"{eer_threshold:.6f} {p_and_eer:.6f} {p_or_eer:.6f}\n"
        )

# =========================
# Main
# =========================
if __name__ == "__main__":
    genuine_raw, impostor_raw, selections = collect_multiuser_scores()

    print(f"[i] Raw genuine scores:  n={len(genuine_raw)}  min={genuine_raw.min():.4f}  max={genuine_raw.max():.4f}  mean={genuine_raw.mean():.4f}")
    print(f"[i] Raw impostor scores: n={len(impostor_raw)} min={impostor_raw.min():.4f} max={impostor_raw.max():.4f} mean={impostor_raw.mean():.4f}")

    genuine, impostor, raw_max = normalize_scores_to_100(genuine_raw, impostor_raw)

    print(f"[i] Normalization factor (raw max) = {raw_max:.4f}")
    print(f"[i] Normalized genuine scores:  min={genuine.min():.4f}  max={genuine.max():.4f}  mean={genuine.mean():.4f}")
    print(f"[i] Normalized impostor scores: min={impostor.min():.4f} max={impostor.max():.4f} mean={impostor.mean():.4f}")

    plot_histograms(genuine, impostor)
    export_histograms(genuine, impostor)
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
    export_far_frr(thresholds, fars, frrs, eer, eer_threshold)
    p_success,eer_success, max_success, max_threshold = compute_p_success(thresholds, fars, frrs)

    print(f"[i] Max P_success = {max_success:.4f}")
    print(f"[i] EER P_success = {eer_success:.4f}")
    print(f"[i] Max P_success threshold = {max_threshold:.4f}")

    plot_p_success(thresholds, p_success, eer_threshold, max_success, max_threshold)
    export_p_success(thresholds, p_success, eer_threshold, max_success, max_threshold)
    p_and, p_or, idx_and, idx_or, p_and_eer, p_or_eer = compute_success_and_or(
        thresholds,
        fars,
        frrs,
        eer_threshold=eer_threshold,
        P_safe=0.75,
        P_leak=0.1,
        P_loss=0.1,
        P_theft=0.05,
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