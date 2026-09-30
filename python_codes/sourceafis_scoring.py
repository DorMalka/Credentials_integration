import csv
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
REPO_ROOT = Path("/Users/dormalka/Desktop/Dor/Paper").resolve()
SOURCEAFIS_DIR = REPO_ROOT / "sourceafis-demo"
# LivDet Live images only. No file under any Fake directory is used.
LIVDET_ROOT = SOURCEAFIS_DIR / "livdet_preproc_png"
LIVE_DIRS_BY_SPLIT = {
    "Training": LIVDET_ROOT / "Training" / "Digital_Persona" / "Live",
    "Testing": LIVDET_ROOT / "Testing" / "Digital_Persona" / "Live",
}
# Enroll eligible identities independently in both partitions. Genuine
# comparisons stay inside each partition. Testing never supplies impostors.
# Optional subset: [("Training", "002_0"), ("Testing", "003_0")].
SELECTED_COHORTS = None
# Optional enrollment limit per partition. None enrolls every eligible identity.
# Impostor candidates still come from all other Training identities.
NUMBER_OF_USERS_PER_SPLIT = None
PROBES_PER_USER = 5
# Compare each enrolled identity against every other Training identity.
# Set True only if you want one direction per unordered identity pair.
REMOVE_MIRRORED_IMPOSTOR_DIRECTIONS = False
OUTPUT_DIR = REPO_ROOT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCORES_DIR = OUTPUT_DIR / "sourceafis_livdet_training_live_one_scan_impostor_scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figs" / "fig_different_users_sourceafsi"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
HISTOGRAM_BIN_WIDTH = 2.5
HIST_BINS = np.arange(
    0.0,
    100.0 + HISTOGRAM_BIN_WIDTH,
    HISTOGRAM_BIN_WIDTH,
)
# The previous FVC2004 KS parameters are deliberately not reused. Run the KS
# fitting script on the new LivDet histogram before adding theoretical curves.
# A filename such as 002_0_5.png belongs to identity 002_0, sample 5.
LIVDET_FILENAME_PATTERN = re.compile(
    r"^(?P<identity>\d+_\d+)_(?P<sample>\d+)$"
)

def identity_sort_key(identity):
    return tuple(int(part) for part in identity.split("_"))

def sample_sort_key(path):
    match = LIVDET_FILENAME_PATTERN.match(Path(path).stem)
    if match is None:
        return float("inf")
    return int(match.group("sample"))

def cohort_sort_key(cohort):
    split, identity = cohort
    return {"Training": 0, "Testing": 1}[split], identity_sort_key(identity)

def discover_live_samples():
    """Discover Live samples separately in Training and Testing."""
    cohorts = {}
    for split, live_dir in LIVE_DIRS_BY_SPLIT.items():
        if not live_dir.is_dir():
            raise FileNotFoundError(
                f"{split} Live directory does not exist: {live_dir}"
            )
        for path in sorted(live_dir.glob("*.png")):
            if not path.is_file():
                continue
            match = LIVDET_FILENAME_PATTERN.match(path.stem)
            if match is None:
                print(f"[!] Ignoring unrecognized LivDet filename: {path.name}")
                continue
            identity = match.group("identity")
            cohorts.setdefault((split, identity), []).append(path.resolve())
    if not cohorts:
        raise ValueError(
            "No LivDet Live images were found. Expected filenames such as "
            "002_0_5.png."
        )
    return {
        cohort: sorted(paths, key=sample_sort_key)
        for cohort, paths in cohorts.items()
    }

def select_users_and_probes():
    """Pool partition-local genuine scores; generate impostors from Training only."""
    dataset_cohorts = discover_live_samples()
    if SELECTED_COHORTS is None:
        selected_cohorts = []
        for split in LIVE_DIRS_BY_SPLIT:
            split_cohorts = sorted(
                (
                    cohort
                    for cohort in dataset_cohorts
                    if cohort[0] == split
                ),
                key=cohort_sort_key,
            )
            if NUMBER_OF_USERS_PER_SPLIT is not None:
                split_cohorts = split_cohorts[:NUMBER_OF_USERS_PER_SPLIT]
            selected_cohorts.extend(split_cohorts)
    else:
        selected_cohorts = list(dict.fromkeys(tuple(cohort) for cohort in SELECTED_COHORTS))
        if any(cohort[0] not in LIVE_DIRS_BY_SPLIT for cohort in selected_cohorts):
            raise ValueError("SELECTED_COHORTS must use Training or Testing.")
    selections = {}
    all_dataset_paths = {
        cohort: list(paths) for cohort, paths in dataset_cohorts.items()
        if cohort[0] == "Training"
    }
    for cohort in selected_cohorts:
        if cohort not in dataset_cohorts:
            raise ValueError(f"LivDet cohort was not found: {cohort}")
        split, identity = cohort
        live_files = dataset_cohorts[cohort]
        if len(live_files) <= PROBES_PER_USER:
            print(
                f"[!] Skipping {split}/{identity}: only {len(live_files)} "
                f"Live images are available; {PROBES_PER_USER} references "
                "plus at least one genuine candidate are required."
            )
            continue
        probes = live_files[:PROBES_PER_USER]
        genuine_candidates = live_files[PROBES_PER_USER:]
        if split == "Testing":
            impostor_candidates = []
            keep_impostor_scores = False
            # Existing LivDetBatchScorer requires a nonempty fourth directory.
            # Use one SAME-IDENTITY genuine image as a compatibility placeholder.
            # These duplicate genuine comparisons are discarded by the loader;
            # no cross-identity Testing comparisons enter the impostor histogram.
            score_impostor_candidates = genuine_candidates[:1]
        else:
            # Select one scan per other identity: the lowest numeric sample number.
            # Each selected scan is scored against all enrollment references; its
            # maximum score contributes one impostor observation for this identity.
            impostor_candidates = []
            all_other_live_candidates = []
            for candidate_cohort, candidate_paths in all_dataset_paths.items():
                if candidate_cohort == cohort:
                    continue
                candidate_scan = candidate_paths[0]
                all_other_live_candidates.append(candidate_scan)
                if (
                    REMOVE_MIRRORED_IMPOSTOR_DIRECTIONS
                    and cohort_sort_key(candidate_cohort)
                    <= cohort_sort_key(cohort)
                ):
                    continue
                impostor_candidates.append(candidate_scan)
            # LivDetBatchScorer receives a nonempty impostor directory. For the
            # final ordered cohort, whose reverse directions are all discarded,
            # score another cohort only as a temporary placeholder and discard
            # those impostor rows after loading. Its genuine rows are still kept.
            score_impostor_candidates = impostor_candidates
            keep_impostor_scores = True
            if not score_impostor_candidates:
                score_impostor_candidates = all_other_live_candidates[:1]
                keep_impostor_scores = False
            if not score_impostor_candidates:
                raise ValueError(
                    f"No cross-user Live candidates exist for {cohort}."
                )
        key = f"{split.lower()}__{identity}"
        selections[key] = {
            "split": split,
            "identity": identity,
            "probes": probes,
            "genuine": genuine_candidates,
            "impostor": impostor_candidates,
            "score_impostor": score_impostor_candidates,
            "keep_impostor_scores": keep_impostor_scores,
        }
    if not selections:
        raise ValueError("No eligible LivDet Live cohorts were selected.")
    return selections

def source_tag(path: Path) -> str:
    """Create a unique staged name that retains the source partition."""
    path = Path(path).resolve()
    try:
        relative = path.relative_to(LIVDET_ROOT)
    except ValueError:
        relative = Path(path.name)
    readable = str(relative).replace("/", "__").replace("\\", "__")
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{readable}__{digest}"

def stage_files(files, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        source = Path(source).resolve()
        staged = destination / f"{source_tag(source)}__{source.name}"
        try:
            staged.symlink_to(source)
        except Exception:
            shutil.copy2(source, staged)
    return destination

# =========================
# SourceAFIS batch scoring
# =========================

def run_sourceafis_batch(
    cohort_key,
    probe_paths,
    genuine_candidates,
    impostor_candidates,
    scores_csv,
):
    """Score one LivDet Live enrollment cohort with LivDetBatchScorer."""
    print()
    print(f"[i] Running SourceAFIS for cohort {cohort_key}")
    print(f"[i] Enrollment references: {len(probe_paths)}")
    print(f"[i] Genuine candidates:    {len(genuine_candidates)}")
    print(f"[i] Impostor candidates:   {len(impostor_candidates)}")
    with tempfile.TemporaryDirectory(prefix=f"livdet_live_{cohort_key}_") as tmp:
        temporary_root = Path(tmp)
        probe_dir = stage_files(probe_paths, temporary_root / "probes")
        genuine_dir = stage_files(
            genuine_candidates,
            temporary_root / "genuine_candidates",
        )
        impostor_dir = stage_files(
            impostor_candidates,
            temporary_root / "impostor_candidates",
        )
        # LivDetBatchScorer labels its fourth directory as impostor input. The
        # files placed there here are real Live scans of different identities,
        # not spoof images.
        exec_args = (
            f'"{probe_dir}" '
            f'"*.png" '
            f'"{genuine_dir}" '
            f'"{impostor_dir}" '
            f'"{scores_csv}"'
        )
        cmd = [
            "mvn",
            "-DskipTests",
            "compile",
            "exec:java",
            "-Dexec.mainClass=LivDetBatchScorer",
            f"-Dexec.args={exec_args}",
        ]
        print("[i] Working dir:", SOURCEAFIS_DIR)
        print("[i] Command:", " ".join(map(str, cmd)))
        subprocess.run(
            cmd,
            cwd=SOURCEAFIS_DIR,
            check=True,
            text=True,
        )

def load_best_scores_from_csv(scores_csv, *, keep_impostor_scores=True):
    """Keep one best-over-references score for every candidate image."""
    genuine_best = {}
    impostor_best = {}
    with open(scores_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"kind", "target", "score"}
        columns = set(reader.fieldnames or [])
        if not expected.issubset(columns):
            raise ValueError(
                f"CSV columns mismatch. Expected at least {expected}, "
                f"found {reader.fieldnames}."
            )
        for row in reader:
            kind = row["kind"].strip().lower()
            target = row["target"].strip()
            score = float(row["score"])
            if kind == "genuine":
                genuine_best[target] = max(
                    score,
                    genuine_best.get(target, -np.inf),
                )
            elif kind in {"impostor", "imposter"}:
                if not keep_impostor_scores:
                    continue
                impostor_best[target] = max(
                    score,
                    impostor_best.get(target, -np.inf),
                )
            else:
                raise ValueError(f"Unknown CSV comparison kind: {kind!r}")
    if not genuine_best:
        raise ValueError(f"No genuine scores were loaded from {scores_csv}.")
    return list(genuine_best.values()), list(impostor_best.values())

def collect_multiuser_scores():
    """Run and pool every independent LivDet Live enrollment cohort."""
    selections = select_users_and_probes()
    print("[i] Impostors: one lowest-numbered Training Live scan per other identity")
    print("[i] Selected LivDet Live cohorts:")
    for key, files in selections.items():
        print(
            f"    {files['split']}/{files['identity']}: "
            f"{len(files['probes'])} references, "
            f"{len(files['genuine'])} genuine candidates, "
            f"{len(files['impostor'])} cross-user impostor candidates"
        )
    all_genuine = []
    all_impostor = []
    for key, files in selections.items():
        scores_csv = SCORES_DIR / f"sourceafis_scores_{key}.csv"
        run_sourceafis_batch(
            cohort_key=key,
            probe_paths=files["probes"],
            genuine_candidates=files["genuine"],
            impostor_candidates=files["score_impostor"],
            scores_csv=scores_csv,
        )
        genuine_scores, impostor_scores = load_best_scores_from_csv(
            scores_csv,
            keep_impostor_scores=files["keep_impostor_scores"],
        )
        all_genuine.extend(genuine_scores)
        all_impostor.extend(impostor_scores)
        print(
            f"[i] {key}: pooled {len(genuine_scores)} genuine and "
            f"{len(impostor_scores)} impostor scores"
        )
    genuine = np.asarray(all_genuine, dtype=float)
    impostor = np.asarray(all_impostor, dtype=float)
    if genuine.size == 0:
        raise ValueError("The pooled genuine distribution is empty.")
    if impostor.size == 0:
        raise ValueError("The pooled impostor distribution is empty.")
    print()
    print(f"[i] Total enrolled cohorts: {len(selections)}")
    print(f"[i] Total pooled genuine scores: {len(genuine)}")
    print(f"[i] Total pooled impostor scores: {len(impostor)}")
    print("[i] Testing contributes genuine scores only; Training supplies all impostors")
    print("[i] Fake-directory images used: 0")
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
    plt.title(f"Live histogram ({PROBES_PER_USER} references per identity)")
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

# =========================
# FAR/FRR from discrete histogram counts + theoretical models
# =========================

def compute_far_frr_from_histogram(genuine_scores, impostor_scores, bins=HIST_BINS):
    """Compute empirical FAR/FRR only at the histogram-bin centers.
    A score is accepted when score >= threshold.  Each histogram bin is
    treated as a discrete mass located at its center.  Therefore, at threshold
    t:
        FAR(t) = sum of impostor counts at centers >= t / N_impostor
        FRR(t) = sum of genuine counts at centers <  t / N_genuine
    There is no interpolation, density estimation, or smoothing here.
    """
    genuine_counts, edges = np.histogram(genuine_scores, bins=bins)
    impostor_counts, _ = np.histogram(impostor_scores, bins=bins)
    if genuine_counts.sum() == 0 or impostor_counts.sum() == 0:
        raise ValueError("Both genuine and impostor histograms must be non-empty.")
    thresholds = (edges[:-1] + edges[1:]) / 2.0
    # Counts strictly below each threshold; the current bin is accepted.
    genuine_below = np.concatenate(([0], np.cumsum(genuine_counts)[:-1]))
    # Counts at or above each threshold; the current bin is accepted.
    impostor_at_or_above = np.cumsum(impostor_counts[::-1])[::-1]
    frrs = genuine_below / genuine_counts.sum()
    fars = impostor_at_or_above / impostor_counts.sum()
    return thresholds, fars.astype(float), frrs.astype(float)

def compute_ks_theoretical_far_frr(thresholds):
    """Return no model curves until KS fits are computed for this dataset."""
    return {}

def compute_eer_intersection(thresholds, fars, frrs):
    """Return the closest discrete EER point without interpolation."""
    i = int(np.argmin(np.abs(fars - frrs)))
    eer = 0.5 * (float(fars[i]) + float(frrs[i]))
    return eer, float(thresholds[i])

def compute_p_success(thresholds, fars, frrs):
    p_success = (1 - fars) * (1 - frrs)
    idx_max = np.argmax(p_success)
    idx_eer = np.argmin(np.abs(fars - frrs))
    max_threshold = thresholds[idx_max]
    max_success = p_success[idx_max]
    eer_success = p_success[idx_eer]
    return p_success,eer_success, max_success, max_threshold

def plot_far_frr(thresholds, fars, frrs, theoretical, eer, eer_threshold):
    plt.figure(figsize=(8, 6))
    # Empirical results are discrete histogram integrals.  The step rendering
    # and markers make it explicit that values exist only at these thresholds.
    plt.step(
        thresholds,
        fars,
        where="post",
        color="tab:red",
        marker="o",
        label="Empirical FAR (histogram)",
    )
    plt.step(
        thresholds,
        frrs,
        where="post",
        color="tab:blue",
        marker="o",
        label="Empirical FRR (histogram)",
    )
    # The old FVC2004 theoretical fits are intentionally omitted. Re-enable
    # model curves only after fitting the newly generated LivDet histogram.
    plt.scatter(
        eer_threshold,
        eer,
        color="black",
        label=f"Discrete EER≈{eer:.4f} @ T={eer_threshold:.2f}",
        zorder=5,
    )
    plt.xlabel("Threshold (%)")
    plt.ylabel("Error Rate")
    plt.title("Discrete empirical FAR / FRR")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figs" / "fig_different_users_sourceafsi" / "sourceafis_far_frr.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def export_far_frr(thresholds, fars, frrs, theoretical, eer, eer_threshold):
    out_dir = OUTPUT_DIR / "figs" / "fig_different_users_sourceafsi"
    out_dir.mkdir(parents=True, exist_ok=True)
    # One row per histogram threshold. The FVC theoretical columns are omitted
    # because their parameters do not describe the new LivDet distributions.
    with open(out_dir / "sourceafis_far_frr_data.txt", "w") as f:
        f.write("T FAR FRR\n")
        for i, (t, fa, fr) in enumerate(zip(thresholds, fars, frrs)):
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
    thresholds, fars, frrs = compute_far_frr_from_histogram(
        genuine,
        impostor,
    )
    theoretical = compute_ks_theoretical_far_frr(thresholds)
    eer, eer_threshold = compute_eer_intersection(thresholds, fars, frrs)
    print(f"[i] EER = {eer:.6f}")
    print(f"[i] Discrete EER threshold = {eer_threshold:.4f}")
    print(
        "[i] KS theoretical curves are disabled until the new LivDet Live "
        "histogram is fitted."
    )
    plot_far_frr(thresholds, fars, frrs, theoretical, eer, eer_threshold)
    export_far_frr(thresholds, fars, frrs, theoretical, eer, eer_threshold)
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
