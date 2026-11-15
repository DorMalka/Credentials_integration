#!/usr/bin/env python3
"""Regenerate AND/OR success curves for both scenarios, without rerunning SourceAFIS."""
from pathlib import Path
import numpy as np

REPO_ROOT = Path("/Users/dormalka/Desktop/Dor/Paper")

P_SAFE, P_LEAK, P_LOSS, P_THEFT = 0.7, 0.15, 0.1, 0.05

if not np.isclose(P_SAFE + P_LEAK + P_LOSS + P_THEFT, 1.0):
    raise ValueError("State probabilities must sum to 1.")


def write_outputs(out_dir, prefix, thresholds, fars, frrs, eer_threshold):
    p_and = (1 - frrs) * (P_SAFE + P_LEAK * (1 - fars))
    p_or = (1 - fars) * (P_SAFE + P_LOSS * (1 - frrs))

    idx_and = int(np.argmax(p_and))
    idx_or = int(np.argmax(p_or))
    p_and_eer = float(np.interp(eer_threshold, thresholds, p_and))
    p_or_eer = float(np.interp(eer_threshold, thresholds, p_or))

    with open(out_dir / f"{prefix}_data.txt", "w") as f:
        f.write("T P_and P_or\n")
        for t, pa, po in zip(thresholds, p_and, p_or):
            f.write(f"{t:.6f} {pa:.6f} {po:.6f}\n")

    with open(out_dir / f"{prefix}_points.txt", "w") as f:
        f.write("T_and_opt P_and_opt T_or_opt P_or_opt T_eer P_and_eer P_or_eer\n")
        f.write(
            f"{thresholds[idx_and]:.6f} {p_and[idx_and]:.6f} "
            f"{thresholds[idx_or]:.6f} {p_or[idx_or]:.6f} "
            f"{eer_threshold:.6f} {p_and_eer:.6f} {p_or_eer:.6f}\n"
        )

    print(f"  AND max {p_and[idx_and]:.4f} at t={thresholds[idx_and]:.4f}")
    print(f"  OR  max {p_or[idx_or]:.4f} at t={thresholds[idx_or]:.4f}")
    print(f"  AND at EER {p_and_eer:.4f}")
    print(f"  OR  at EER {p_or_eer:.4f}")


# ---- Spoofing attack: read the saved checkpoint ----
spoof_dir = REPO_ROOT / "figs" / "fig_spoofed_users"
with np.load(spoof_dir / "psafe_sweep_inputs.npz", allow_pickle=False) as d:
    thresholds = np.asarray(d["thresholds"], dtype=float)
    fars = np.asarray(d["fars"], dtype=float)
    frrs = np.asarray(d["frrs"], dtype=float)
    eer_threshold = float(np.asarray(d["eer_threshold"]).item())

print("Spoofing attack:")
write_outputs(
    spoof_dir,
    "livdet_sourceafis_success_and_or_best_of_probes",
    thresholds, fars, frrs, eer_threshold,
)

# ---- Different persons: read the exported FAR/FRR tables ----
diff_dir = REPO_ROOT / "figs" / "fig_different_users_sourceafsi"
table = np.loadtxt(diff_dir / "sourceafis_far_frr_data.txt", skiprows=1)
thresholds, fars, frrs = table[:, 0], table[:, 1], table[:, 2]
eer_threshold = float(
    np.loadtxt(diff_dir / "sourceafis_far_frr_points.txt", skiprows=1)[0]
)

print("\nDifferent persons:")
write_outputs(
    diff_dir,
    "sourceafis_success_and_or",
    thresholds, fars, frrs, eer_threshold,
)