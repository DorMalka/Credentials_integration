#!/usr/bin/env python3
"""Fit distributions to genuine and impostor score histograms using KS.

Expected input columns:

    score genuine impostor

Each row gives a score (or bin center) and the number of genuine and
impostor observations at that score. Commas, tabs, and spaces are accepted.

The script reconstructs each sample, fits several candidate probability
distributions independently, and ranks them by their one-sample KS statistic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from scipy import stats


DISTRIBUTIONS = {
    "normal": stats.norm,
    "quadratic": stats.rdist,
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
    "weibull": stats.weibull_min,
    "beta": stats.beta,
    "exponential": stats.expon,
    "logistic": stats.logistic,
    "laplace": stats.laplace,
    "uniform": stats.uniform,
}

PARAMETER_NAMES = {
    "normal": ("loc", "scale"),
    "quadratic": ("shape", "loc", "scale"),
    "lognormal": ("shape", "loc", "scale"),
    "gamma": ("shape", "loc", "scale"),
    "weibull": ("shape", "loc", "scale"),
    "beta": ("a", "b", "loc", "scale"),
    "exponential": ("loc", "scale"),
    "logistic": ("loc", "scale"),
    "laplace": ("loc", "scale"),
    "uniform": ("loc", "scale"),
}


@dataclass(frozen=True)
class FitResult:
    name: str
    statistic: float
    pvalue: float
    parameters: tuple[float, ...]


def load_histogram(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read score, genuine-count, and impostor-count columns."""
    rows: list[tuple[float, float, float]] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.replace(",", " ").split()
            if len(fields) < 3:
                raise ValueError(
                    f"Line {line_number}: expected at least 3 columns, "
                    f"but found {len(fields)}."
                )

            try:
                row = tuple(float(value) for value in fields[:3])
            except ValueError:
                # Permit one header row such as: score genuine impostor
                if not rows:
                    continue
                raise ValueError(
                    f"Line {line_number}: the first three values must be numeric."
                ) from None

            rows.append(row)  # type: ignore[arg-type]

    if not rows:
        raise ValueError(f"No numeric histogram rows found in {path}.")

    data = np.asarray(rows, dtype=float)
    scores = data[:, 0]
    genuine_counts = data[:, 1]
    impostor_counts = data[:, 2]

    for label, counts in (
        ("genuine", genuine_counts),
        ("impostor", impostor_counts),
    ):
        if np.any(~np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError(f"{label} counts must be finite and non-negative.")
        if np.any(counts != np.floor(counts)):
            raise ValueError(f"{label} counts must be whole numbers.")

    if np.any(~np.isfinite(scores)):
        raise ValueError("Scores must be finite numbers.")

    return scores, genuine_counts.astype(int), impostor_counts.astype(int)


def reconstruct_sample(scores: np.ndarray, counts: np.ndarray, label: str) -> np.ndarray:
    """Expand histogram counts into a sample located at the score/bin centers."""
    sample = np.repeat(scores, counts)
    if sample.size == 0:
        raise ValueError(f"The {label} histogram contains no observations.")
    return sample


def maximum_cdf_gap(
    scores: np.ndarray,
    genuine_counts: np.ndarray,
    impostor_counts: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return score, signed gap, genuine CDF, and impostor CDF at max |gap|."""
    order = np.argsort(scores)
    sorted_scores = scores[order]
    genuine_cdf = np.cumsum(genuine_counts[order]) / genuine_counts.sum()
    impostor_cdf = np.cumsum(impostor_counts[order]) / impostor_counts.sum()
    gaps = genuine_cdf - impostor_cdf
    index = int(np.argmax(np.abs(gaps)))
    return (
        float(sorted_scores[index]),
        float(gaps[index]),
        float(genuine_cdf[index]),
        float(impostor_cdf[index]),
    )


def fit_distribution(sample: np.ndarray, name: str) -> FitResult:
    """Fit one SciPy distribution and calculate its one-sample KS statistic."""
    distribution = DISTRIBUTIONS[name]

    # Some flexible distributions emit harmless optimization warnings while
    # searching for their maximum-likelihood parameters. A failed fit is still
    # caught below and reported to the user.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if name == "quadratic":
            # scipy.stats.rdist with shape c=4 is the normalized parabolic
            # distribution 3/(4*scale) * (1-z^2), where
            # z=(x-loc)/scale and |z|<=1. Keep c fixed and fit loc/scale.
            fitted = distribution.fit(sample, f0=4.0)
        else:
            fitted = distribution.fit(sample)
        parameters = tuple(float(value) for value in fitted)

    result = stats.kstest(
        sample,
        distribution.cdf,
        args=parameters,
        alternative="two-sided",
        method="asymp",
    )
    return FitResult(
        name=name,
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        parameters=parameters,
    )


def fit_candidates(sample: np.ndarray, names: list[str]) -> tuple[list[FitResult], list[str]]:
    """Fit all requested distributions and return successful and failed fits."""
    fitted: list[FitResult] = []
    failures: list[str] = []

    for name in names:
        try:
            fitted.append(fit_distribution(sample, name))
        except Exception as error:  # Continue so one difficult model does not stop the run.
            failures.append(f"{name}: {error}")

    fitted.sort(key=lambda item: item.statistic)
    return fitted, failures


def format_parameters(name: str, parameters: tuple[float, ...]) -> str:
    labels = PARAMETER_NAMES[name]
    return ", ".join(
        f"{label}={value:.6g}" for label, value in zip(labels, parameters)
    )


def print_fit_table(label: str, sample: np.ndarray, fitted: list[FitResult]) -> None:
    """Print candidate models ordered from smallest to largest KS statistic."""
    print()
    print(f"{label.upper()} DISTRIBUTION FIT (n = {sample.size})")
    print("Rank  Distribution       KS D          Approx. p-value   Fitted parameters")
    print("----  -----------------  ------------  ----------------  -----------------")
    for rank, result in enumerate(fitted, start=1):
        print(
            f"{rank:<4}  {result.name:<17}  {result.statistic:<12.8f}  "
            f"{result.pvalue:<16.8g}  "
            f"{format_parameters(result.name, result.parameters)}"
        )

    if fitted:
        print(
            f"Best candidate by KS distance: {fitted[0].name} "
            f"(D = {fitted[0].statistic:.8f})"
        )


def print_two_sample_test(
    scores: np.ndarray,
    genuine_counts: np.ndarray,
    impostor_counts: np.ndarray,
    genuine: np.ndarray,
    impostor: np.ndarray,
    alpha: float,
) -> None:
    """Optionally compare the genuine and impostor samples directly."""
    result = stats.ks_2samp(
        genuine,
        impostor,
        alternative="two-sided",
        method="asymp",
    )
    score_at_max, signed_gap, genuine_cdf, impostor_cdf = maximum_cdf_gap(
        scores, genuine_counts, impostor_counts
    )

    print()
    print("GENUINE-vs-IMPOSTOR TWO-SAMPLE KS TEST")
    print(f"KS statistic D:      {result.statistic:.8f}")
    print(f"p-value:             {result.pvalue:.8g}")
    print(f"Maximum gap score:   {score_at_max:g}")
    print(f"Genuine CDF there:   {genuine_cdf:.8f}")
    print(f"Impostor CDF there:  {impostor_cdf:.8f}")
    print(f"Signed CDF gap:      {signed_gap:.8f}")
    print(f"Significance level:  {alpha:g}")

    if result.pvalue < alpha:
        print("Conclusion: reject the hypothesis that both samples come from")
        print("            the same underlying distribution.")
    else:
        print("Conclusion: do not reject the hypothesis that both samples")
        print("            come from the same underlying distribution.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit candidate distributions independently to genuine and impostor "
            "score histograms using the KS statistic."
        )
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=Path("livdet_sourceafis_histogram_best_of_probes_data.txt"),
        help="three-column histogram file (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="significance level used for the conclusion (default: %(default)s)",
    )
    parser.add_argument(
        "--distributions",
        default=",".join(DISTRIBUTIONS),
        help=(
            "comma-separated candidate names; available: "
            + ", ".join(DISTRIBUTIONS)
        ),
    )
    parser.add_argument(
        "--compare-samples",
        action="store_true",
        help="also run the direct genuine-vs-impostor two-sample KS test",
    )
    args = parser.parse_args()

    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be strictly between 0 and 1.")

    requested_distributions = [
        name.strip().lower()
        for name in args.distributions.split(",")
        if name.strip()
    ]
    unknown = [name for name in requested_distributions if name not in DISTRIBUTIONS]
    if unknown:
        parser.error(
            "unknown distribution(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(DISTRIBUTIONS)
        )
    if not requested_distributions:
        parser.error("--distributions must contain at least one name.")

    scores, genuine_counts, impostor_counts = load_histogram(args.file)
    genuine = reconstruct_sample(scores, genuine_counts, "genuine")
    impostor = reconstruct_sample(scores, impostor_counts, "impostor")

    print(f"Input file:          {args.file}")
    print(f"Genuine sample size: {genuine.size}")
    print(f"Impostor sample size:{impostor.size:>7}")

    genuine_fits, genuine_failures = fit_candidates(
        genuine, requested_distributions
    )
    impostor_fits, impostor_failures = fit_candidates(
        impostor, requested_distributions
    )
    print_fit_table("genuine", genuine, genuine_fits)
    print_fit_table("impostor", impostor, impostor_fits)

    failures = genuine_failures + impostor_failures
    if failures:
        print()
        print("Fits that could not be computed:")
        for failure in failures:
            print(f"  - {failure}")

    if args.compare_samples:
        print_two_sample_test(
            scores,
            genuine_counts,
            impostor_counts,
            genuine,
            impostor,
            args.alpha,
        )

    print()
    print("Interpretation: the smallest KS statistic D indicates the closest")
    print("candidate among those tested; it does not prove that the model is true.")
    print("Caution: these are approximate goodness-of-fit results because histogram")
    print("bin centers replace the raw scores, causing ties. In addition, parameters")
    print("were estimated from the tested data, so the displayed standard KS p-values")
    print("should not be used as formally calibrated significance values.")


if __name__ == "__main__":
    main()
