#!/usr/bin/env python3
"""Fit distributions to genuine and impostor score histograms using KS.

Expected input columns:

    score genuine impostor

Each row gives a score (or bin center) and the number of genuine and
impostor observations at that score. Commas, tabs, and spaces are accepted.

The script reconstructs each sample, fits several candidate probability
distributions independently, and ranks them by their one-sample KS statistic.
By default parameters directly minimize KS distance (uniform uses linear
feasibility; other models use a numerical search with an MLE fallback).
--fit-method mle reproduces the previous fitting criterion. Bin centers remain
an approximation to raw scores; no outliers are discarded. The script only
prints fit results and does not modify histogram or FAR/FRR files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution, linprog


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
    parameters: tuple[float, ...]
    mle_statistic: float
    method: str


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


def empirical_cdf_sides(sample):
    """CDF immediately before/after each jump; handle ties exactly."""
    x, counts = np.unique(sample, return_counts=True)
    right = np.cumsum(counts) / sample.size
    left = right - counts / sample.size
    return x, left, right


def ks_distance(distribution, parameters, x, left, right):
    cdf = distribution.cdf(x, *parameters)
    if np.any(~np.isfinite(cdf)):
        return float("inf")
    return float(max(np.max(right - cdf), np.max(cdf - left)))


def fit_uniform_ks(sample):
    """Minimum KS distance via bisection and linear feasibility, using all data.

    F(x) = clip(q*z + r, 0, 1), where q > 0 and z rescales x to [0,1].
    For a candidate D, require right-D <= F(x) <= left+D at every jump.
    Only positive lower bounds and upper bounds below one impose constraints
    on the unclipped line. No requirement forces all observations into support.
    """
    x, left, right = empirical_cdf_sides(sample)
    if x.size == 1:
        return (float(x[0] - 0.5), 1.0)
    origin, span = float(x[0]), float(x[-1] - x[0])
    z = (x - origin) / span
    best = (origin, span)
    upper = ks_distance(stats.uniform, best, x, left, right)
    lower = float(np.max(right - left) / 2)
    for _ in range(50):
        d = (lower + upper) / 2
        lo, hi = right - d, left + d
        mask_lo, mask_hi = lo > 0, hi < 1
        matrix = np.vstack([
            np.column_stack([-z[mask_lo], -np.ones(mask_lo.sum())]),
            np.column_stack([z[mask_hi], np.ones(mask_hi.sum())]),
        ])
        bounds = np.concatenate([-lo[mask_lo], hi[mask_hi]])
        result = linprog(
            [0.0, 0.0], A_ub=matrix, b_ub=bounds,
            bounds=[(1e-8, None), (None, None)], method="highs",
            options={"primal_feasibility_tolerance": 1e-9},
        )
        if result.success:
            q, r = result.x
            parameters = (origin - span * r / q, span / q)
            actual = ks_distance(stats.uniform, parameters, x, left, right)
            if actual <= upper + 1e-8:
                best = parameters
            upper = d
        elif result.status == 2:
            lower = d
        else:
            raise RuntimeError(f"Uniform fit failed: {result.message}")
    # Account for LP tolerance; the returned fit must never worsen MLE.
    mle = (origin, span)
    return min([best, mle], key=lambda p: ks_distance(stats.uniform, p, x, left, right))


def fit_distribution(sample, name, *, fit_method="ks", maxiter=300, seed=0):
    """Use an MLE baseline, then directly minimize KS distance by default."""
    distribution = DISTRIBUTIONS[name]
    x, left, right = empirical_cdf_sides(sample)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if name == "uniform" and np.ptp(sample) == 0:
            # Degenerate MLE has zero scale; use a finite interval attaining
            # the continuous-CDF lower bound D=0.5 for a point-mass sample.
            fitted = (float(sample[0]) - 0.5, 1.0)
        else:
            fitted = (distribution.fit(sample, f0=4.0) if name == "quadratic"
                      else distribution.fit(sample))
    mle = tuple(float(v) for v in fitted)
    mle_d = ks_distance(distribution, mle, x, left, right)
    if not np.isfinite(mle_d):
        raise ValueError("Maximum-likelihood baseline is invalid.")
    parameters, method = mle, "MLE"
    if fit_method == "ks":
        if name == "uniform":
            parameters, method = fit_uniform_ks(sample), "KS/LP"
        else:
            # Broad numerical search bounds include the MLE parameters.
            # Scale is optimized logarithmically to cover narrow and wide fits.
            span = max(float(np.ptp(sample)), 1.0)
            loc0, scale0 = mle[-2:]
            search_bounds = [
                (min(0.05, value / 2), max(30.0, value * 2))
                for value in mle[:-2]
            ] if name != "quadratic" else []
            search_bounds += [
                (min(float(x[0]) - span, loc0 - span),
                 max(float(x[-1]) + span, loc0 + span)),
                (np.log(min(span * 1e-5, scale0 / 2)),
                 np.log(max(span * 10, scale0 * 2))),
            ]
            def unpack(v):
                loc_scale = (float(v[-2]), float(np.exp(v[-1])))
                shapes = (4.0,) if name == "quadratic" else tuple(v[:-2])
                return shapes + loc_scale
            def objective(v):
                return ks_distance(distribution, unpack(v), x, left, right)
            initial = list(mle[:-2]) if name != "quadratic" else []
            initial += [loc0, np.log(scale0)]
            result = differential_evolution(
                objective, search_bounds, seed=seed, x0=initial,
                maxiter=maxiter, popsize=20, tol=1e-8, polish=True,
            )
            candidate = unpack(result.x)
            if ks_distance(distribution, candidate, x, left, right) < mle_d:
                parameters = candidate
            method = "KS/search" if result.success else "KS/budget"
    d = ks_distance(distribution, parameters, x, left, right)
    return FitResult(name, d, parameters, mle_d, method)


def fit_candidates(sample, names, *, fit_method="ks", maxiter=300, seed=0):
    """Rank by actual KS distance; preserve failures for reporting."""
    fitted, failures = [], []
    for name in names:
        try:
            fitted.append(fit_distribution(
                sample, name, fit_method=fit_method, maxiter=maxiter, seed=seed
            ))
        except Exception as error:
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
    print("Rank  Distribution       KS D          MLE KS D      Method      Fitted parameters")
    print("----  -----------------  ------------  ----------------  -----------------")
    for rank, result in enumerate(fitted, start=1):
        print(
            f"{rank:<4}  {result.name:<17}  {result.statistic:<12.8f}  "
            f"{result.mle_statistic:<12.8f}  {result.method:<10}  "
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
    parser.add_argument(
        "--fit-method", choices=("ks", "mle"), default="ks",
        help="direct KS-distance fitting (default) or previous MLE fitting",
    )
    parser.add_argument("--maxiter", type=int, default=300,
                        help="numerical optimizer iteration budget (default: 300)")
    parser.add_argument("--seed", type=int, default=0,
                        help="reproducible optimizer seed (default: 0)")
    args = parser.parse_args()
    if args.maxiter < 1 or args.seed < 0:
        parser.error("--maxiter must be positive and --seed non-negative.")

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
        genuine, requested_distributions, fit_method=args.fit_method,
        maxiter=args.maxiter, seed=args.seed
    )
    impostor_fits, impostor_failures = fit_candidates(
        impostor, requested_distributions, fit_method=args.fit_method,
        maxiter=args.maxiter, seed=args.seed
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
    print("Uniform KS/LP minimizes distance by linear feasibility and bisection.")
    print("Other KS fits are numerical searches within finite bounds; a global")
    print("minimum is not guaranteed. KS/budget means the iteration limit was hit.")
    print("Returned KS fits never worsen their MLE baseline. No observations are removed.")
    print("Interpretation: the smallest KS statistic D indicates the closest")
    print("candidate among those tested; it does not prove that the model is true.")
    print("Caution: these are approximate goodness-of-fit results because histogram")
    print("bin centers replace the raw scores, causing ties. In addition, parameters")
    print("were estimated from the tested data. Uncalibrated one-sample KS p-values")
    print("are intentionally omitted. Two-sample p-values, if requested, are also")
    print("approximate for tied/bin-center data and assume independent observations.")


if __name__ == "__main__":
    main()

