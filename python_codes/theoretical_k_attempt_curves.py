#!/usr/bin/env python3
"""Generate k-attempt authentication curves for theoretical score models.

For each of the uniform, quadratic (parabolic), and Gaussian model families,
the program jointly optimizes t_1, ..., t_k to maximize

    P_success = (1 - product_j FRR(t_j)) * product_j (1 - FAR(t_j)).

It exports one whitespace-delimited TXT table and one TikZ/PGFPlots figure per
model. The default model parameters reproduce the earlier uniform supports
U:[0.30, 0.70] and A:[0.10, 0.50], while matching the mean and variance across
all three model families.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import ndtr


DEFAULT_SIGMA = 0.4 / np.sqrt(12.0)
GAUSSIAN_USER_MEAN = 27.9694
GAUSSIAN_USER_SIGMA = 20.4007
GAUSSIAN_ATTACKER_MEAN = 5.04261
GAUSSIAN_ATTACKER_SIGMA = 4.95845


@dataclass(frozen=True)
class ScoreModel:
    name: str
    user_cdf: Callable[[np.ndarray], np.ndarray]
    attacker_cdf: Callable[[np.ndarray], np.ndarray]
    threshold_bounds: tuple[float, float]
    description: str


@dataclass(frozen=True)
class KResult:
    k: int
    thresholds: np.ndarray
    p_success_max: float
    product_frr: float
    product_one_minus_far: float
    p_attacker: float
    p_failure: float


def uniform_cdf(mean: float, sigma: float) -> tuple[Callable, tuple[float, float]]:
    """Return the CDF and support of a uniform law with given mean/std."""
    half_width = np.sqrt(3.0) * sigma
    lower = mean - half_width
    upper = mean + half_width

    def cdf(x):
        x = np.asarray(x, dtype=float)
        return np.clip((x - lower) / (upper - lower), 0.0, 1.0)

    return cdf, (lower, upper)


def quadratic_cdf(mean: float, sigma: float) -> tuple[Callable, tuple[float, float]]:
    r"""Return the CDF and support of the moment-matched quadratic PDF.

    On |s-mean| <= r, where r = sqrt(5)*sigma, the PDF is

        f(s) = 3/(4r) * (1 - ((s-mean)/r)^2).

    It is zero outside that interval.
    """
    radius = np.sqrt(5.0) * sigma
    lower = mean - radius
    upper = mean + radius

    def cdf(x):
        x = np.asarray(x, dtype=float)
        z = (x - mean) / radius
        inside = 0.5 + 0.75 * z - 0.25 * np.power(z, 3)
        return np.where(x <= lower, 0.0, np.where(x >= upper, 1.0, inside))

    return cdf, (lower, upper)


def gaussian_cdf(mean: float, sigma: float) -> Callable:
    """Return a Gaussian CDF with the given mean and standard deviation."""
    def cdf(x):
        return ndtr((np.asarray(x, dtype=float) - mean) / sigma)

    return cdf


def build_models(
    user_mean: float,
    user_sigma: float,
    attacker_mean: float,
    attacker_sigma: float,
) -> list[ScoreModel]:
    if user_sigma <= 0.0 or attacker_sigma <= 0.0:
        raise ValueError("Both standard deviations must be positive.")
    if attacker_mean >= user_mean:
        raise ValueError("attacker_mean must be lower than user_mean.")

    uniform_user, uniform_user_support = uniform_cdf(
    50.0, 97.5 / np.sqrt(12.0))
    uniform_attacker, uniform_attacker_support = uniform_cdf(
    15.0, 27.5 / np.sqrt(12.0))

    quadratic_user, quadratic_user_support = quadratic_cdf(
        user_mean, user_sigma
    )
    quadratic_attacker, quadratic_attacker_support = quadratic_cdf(
        attacker_mean, attacker_sigma
    )
    gaussian_user = gaussian_cdf(
        GAUSSIAN_USER_MEAN, GAUSSIAN_USER_SIGMA
    )
    gaussian_attacker = gaussian_cdf(
        GAUSSIAN_ATTACKER_MEAN, GAUSSIAN_ATTACKER_SIGMA
    )

    gaussian_lower = min(
        GAUSSIAN_USER_MEAN - 8.0 * GAUSSIAN_USER_SIGMA,
        GAUSSIAN_ATTACKER_MEAN - 8.0 * GAUSSIAN_ATTACKER_SIGMA,
    )
    gaussian_upper = max(
        GAUSSIAN_USER_MEAN + 8.0 * GAUSSIAN_USER_SIGMA,
        GAUSSIAN_ATTACKER_MEAN + 8.0 * GAUSSIAN_ATTACKER_SIGMA,
    )

    return [
        ScoreModel(
            name="uniform",
            user_cdf=uniform_user,
            attacker_cdf=uniform_attacker,
            threshold_bounds=(
                min(uniform_user_support[0], uniform_attacker_support[0]),
                max(uniform_user_support[1], uniform_attacker_support[1]),
            ),
            description=(
                "Uniform genuine and attacker score distributions with "
                "matched input moments"
            ),
        ),
        ScoreModel(
            name="quadratic",
            user_cdf=quadratic_user,
            attacker_cdf=quadratic_attacker,
            threshold_bounds=(
                min(quadratic_user_support[0], quadratic_attacker_support[0]),
                max(quadratic_user_support[1], quadratic_attacker_support[1]),
            ),
            description=(
                "Quadratic (parabolic) genuine and attacker score "
                "distributions with matched input moments"
            ),
        ),
        ScoreModel(
            name="gaussian",
            user_cdf=gaussian_user,
            attacker_cdf=gaussian_attacker,
            threshold_bounds=(gaussian_lower, gaussian_upper),
            description=(
                "Gaussian genuine and attacker score distributions"
            ),
        ),
    ]


def evaluate_thresholds(
    model: ScoreModel,
    thresholds,
) -> tuple[float, float, float]:
    """Return P_success, product(FRR), and product(1-FAR)."""
    thresholds = np.asarray(thresholds, dtype=float)
    frr = np.clip(model.user_cdf(thresholds), 0.0, 1.0)

    # FAR(t) = P_A[S >= t], hence 1-FAR(t) = F_A(t) for continuous laws.
    one_minus_far = np.clip(model.attacker_cdf(thresholds), 0.0, 1.0)

    product_frr = float(np.prod(frr))
    product_one_minus_far = float(np.prod(one_minus_far))
    p_success = float((1.0 - product_frr) * product_one_minus_far)
    return p_success, product_frr, product_one_minus_far


def optimize_for_k(
    model: ScoreModel,
    k: int,
    *,
    optimizer_starts: int,
    de_maxiter: int,
    seed: int,
) -> KResult:
    """Jointly optimize all k thresholds with global and local searches."""
    if k < 1:
        raise ValueError("k must be positive.")
    if optimizer_starts < 1:
        raise ValueError("optimizer_starts must be positive.")
    if de_maxiter < 1:
        raise ValueError("de_maxiter must be positive.")

    lower, upper = model.threshold_bounds
    bounds = [(lower, upper)] * k

    def objective(candidate):
        return -evaluate_thresholds(model, candidate)[0]

    # A dense equal-threshold grid supplies a deterministic baseline and a
    # strong local-search initialization.
    grid = np.linspace(lower, upper, 4001)
    equal_success = np.asarray(
        [evaluate_thresholds(model, np.full(k, threshold))[0] for threshold in grid]
    )
    equal_threshold = float(grid[int(np.argmax(equal_success))])
    best_thresholds = np.full(k, equal_threshold)
    best_success = float(np.max(equal_success))

    global_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        popsize=15,
        maxiter=de_maxiter,
        tol=1e-10,
        polish=False,
        workers=1,
        updating="immediate",
    )
    global_thresholds = np.clip(
        np.asarray(global_result.x, dtype=float), lower, upper
    )
    global_success = evaluate_thresholds(model, global_thresholds)[0]
    if global_success > best_success:
        best_thresholds = global_thresholds
        best_success = global_success

    rng = np.random.default_rng(seed + 10_000)
    starts = [
        global_thresholds,
        np.full(k, equal_threshold),
        np.full(k, 0.5 * (lower + upper)),
        np.linspace(lower, upper, k),
        np.linspace(lower, upper, k)[::-1],
    ]
    while len(starts) < optimizer_starts:
        starts.append(rng.uniform(lower, upper, size=k))

    for initial in starts[:optimizer_starts]:
        result = minimize(
            objective,
            x0=np.clip(initial, lower, upper),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 2000,
                "ftol": 1e-15,
                "gtol": 1e-10,
                "maxls": 50,
            },
        )
        candidate = np.clip(np.asarray(result.x, dtype=float), lower, upper)
        candidate_success = evaluate_thresholds(model, candidate)[0]
        if candidate_success > best_success:
            best_thresholds = candidate
            best_success = candidate_success

    # Attempt order cannot change the symmetric objective. Sorting only makes
    # the exported diagnostic thresholds deterministic.
    best_thresholds = np.sort(best_thresholds)
    p_success, product_frr, product_one_minus_far = evaluate_thresholds(
        model, best_thresholds
    )
    return KResult(
        k=k,
        thresholds=best_thresholds,
        p_success_max=p_success,
        product_frr=product_frr,
        product_one_minus_far=product_one_minus_far,
        p_attacker=1.0 - product_one_minus_far,
        p_failure=1.0 - p_success,
    )


def export_txt(results: list[KResult], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=" ", lineterminator="\n")
        writer.writerow(
            [
                "k",
                "P_success_max",
                "product_FRR",
                "product_1_minus_FAR",
                "P_attacker",
                "P_failure",
                "threshold_min",
                "threshold_max",
                "threshold_spread",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.k,
                    f"{result.p_success_max:.12f}",
                    f"{result.product_frr:.12f}",
                    f"{result.product_one_minus_far:.12f}",
                    f"{result.p_attacker:.12f}",
                    f"{result.p_failure:.12f}",
                    f"{float(np.min(result.thresholds)):.12f}",
                    f"{float(np.max(result.thresholds)):.12f}",
                    f"{float(np.ptp(result.thresholds)):.12e}",
                ]
            )


def export_tikz(model: ScoreModel, data_file: Path, tikz_file: Path) -> None:
    """Write a standalone figure environment that reads the exported table."""
    data_path = data_file.as_posix()
    title = model.name.capitalize()
    tikz = rf"""% Requires: \usepackage{{pgfplots}}
%           \pgfplotsset{{compat=1.18}}
\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.95\columnwidth,
    height=0.66\columnwidth,
    xlabel={{Number of attempts, $k$}},
    ylabel={{Probability}},
    ymin=0,
    ymax=1,
    xtick=data,
    grid=major,
    line width=1pt,
    mark size=2.4pt,
    legend style={{
        at={{(0.5,1.03)}},
        anchor=south,
        legend columns=3,
        draw=black,
        fill=white,
        font=\scriptsize
    }}
]

\addplot[green!55!black, mark=*]
table[x=k, y=P_success_max] {{{data_path}}};
\addlegendentry{{$P_{{\mathrm{{success}}}}^{{\max}}$}}

\addplot[blue, mark=triangle*]
table[x=k, y=product_FRR] {{{data_path}}};
\addlegendentry{{$\prod_{{j=1}}^{{k}}\mathrm{{FRR}}_j$}}

\addplot[red, mark=square*]
table[x=k, y=product_1_minus_FAR] {{{data_path}}};
\addlegendentry{{$\prod_{{j=1}}^{{k}}(1-\mathrm{{FAR}}_j)$}}

\end{{axis}}
\end{{tikzpicture}}
\caption{{Maximum success probability and its two cumulative factors for
the {title} score distributions as functions of the number of attempts~$k$.}}
\label{{fig:{model.name}_k_attempt_curves}}
\end{{figure}}
"""
    tikz_file.parent.mkdir(parents=True, exist_ok=True)
    tikz_file.write_text(tikz, encoding="utf-8")


def export_failure_comparison_tikz(
    data_files: dict[str, Path],
    tikz_file: Path,
) -> None:
    """Export one TikZ axis containing both new curves for every model."""
    styles = {
        "uniform": ("Uniform", "solid", "*"),
        "quadratic": ("Quadratic", "dashed", "square*"),
        "gaussian": ("Gaussian", "densely dotted", "triangle*"),
    }
    plot_lines = []
    for model_name in ("uniform", "quadratic", "gaussian"):
        if model_name not in data_files:
            continue
        label, line_style, marker = styles[model_name]
        data_path = data_files[model_name].as_posix()
        plot_lines.extend(
            [
                (
                    f"\\addplot[red, {line_style}, mark={marker}] "
                    f"table[x=k, y=P_attacker] {{{data_path}}};"
                ),
                (
                    f"\\addlegendentry{{{label} "
                    "$1-\\prod_{j=1}^{k}(1-\\mathrm{FAR}_j)$}"
                ),
                (
                    f"\\addplot[blue, {line_style}, mark={marker}] "
                    f"table[x=k, y=P_failure] {{{data_path}}};"
                ),
                (
                    f"\\addlegendentry{{{label} "
                    "$1-P_{\\mathrm{success}}^{\\max}$}"
                ),
                "",
            ]
        )

    plots = "\n".join(plot_lines).rstrip()
    tikz = rf"""% Requires: \usepackage{{pgfplots}}
%           \pgfplotsset{{compat=1.18}}
\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=0.98\columnwidth,
    height=0.75\columnwidth,
    xlabel={{Number of attempts, $k$}},
    ylabel={{Probability}},
    xmin=1,
    ymin=0,
    ymax=1,
    xtick=data,
    grid=major,
    line width=1pt,
    mark size=2.5pt,
    legend style={{
        at={{(0.5,-0.25)}},
        anchor=north,
        legend columns=3,
        draw=black,
        fill=white,
        font=\scriptsize
    }}
]
{plots}
\end{{axis}}
\end{{tikzpicture}}
\caption{{The probability that the attacker is accepted in at least one
attempt and the mechanism failure probability as functions of the number of
authentication attempts~$k$ for the uniform, quadratic, and Gaussian score
distributions.}}
\label{{fig:theoretical_k_attempts_failure_comparison}}
\end{{figure}}
"""
    tikz_file.parent.mkdir(parents=True, exist_ok=True)
    tikz_file.write_text(tikz, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize k-attempt success for uniform, quadratic, and Gaussian "
            "score distributions and export TikZ-ready curves."
        )
    )
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("uniform", "quadratic", "gaussian"),
        default=["uniform", "quadratic", "gaussian"],
    )
    parser.add_argument("--user-mean", type=float, default=0.50)
    parser.add_argument("--user-sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--attacker-mean", type=float, default=0.30)
    parser.add_argument("--attacker-sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--optimizer-starts", type=int, default=24)
    parser.add_argument("--de-maxiter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figs/fig_theoretical_k_attempts"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.max_k < 1:
        raise ValueError("--max-k must be positive.")

    models = build_models(
        args.user_mean,
        args.user_sigma,
        args.attacker_mean,
        args.attacker_sigma,
    )
    selected = [model for model in models if model.name in args.models]

    print(
        "[i] Quadratic moment parameters: "
        f"user=(mean={args.user_mean:.8f}, sigma={args.user_sigma:.8f}), "
        f"attacker=(mean={args.attacker_mean:.8f}, "
        f"sigma={args.attacker_sigma:.8f})"
    )
    print(
        "[i] Gaussian KS-fit parameters: "
        f"user=(mean={GAUSSIAN_USER_MEAN:.8f}, "
        f"sigma={GAUSSIAN_USER_SIGMA:.8f}), "
        f"attacker=(mean={GAUSSIAN_ATTACKER_MEAN:.8f}, "
        f"sigma={GAUSSIAN_ATTACKER_SIGMA:.8f})"
    )

    data_files = {}
    for model_index, model in enumerate(selected):
        print(f"\n[{model.name.upper()}]")
        results = []
        for k in range(1, args.max_k + 1):
            result = optimize_for_k(
                model,
                k,
                optimizer_starts=args.optimizer_starts,
                de_maxiter=args.de_maxiter,
                seed=args.seed + 1000 * model_index + k,
            )
            results.append(result)
            print(
                f"k={k:2d}: P_success_max={result.p_success_max:.10f}, "
                f"product_FRR={result.product_frr:.10f}, "
                "product_1_minus_FAR="
                f"{result.product_one_minus_far:.10f}, "
                f"P_attacker={result.p_attacker:.10f}, "
                f"P_failure={result.p_failure:.10f}, "
                f"thresholds={np.array2string(result.thresholds, precision=6)}"
            )

        data_file = args.output_dir / f"{model.name}_k_attempts.txt"
        tikz_file = args.output_dir / f"{model.name}_k_attempts.tex"
        export_txt(results, data_file)
        export_tikz(model, data_file, tikz_file)
        data_files[model.name] = data_file
        print(f"[i] Data saved to: {data_file}")
        print(f"[i] TikZ saved to: {tikz_file}")

    comparison_tikz_file = (
        args.output_dir / "failure_probabilities_comparison.tex"
    )
    export_failure_comparison_tikz(data_files, comparison_tikz_file)
    print(f"[i] Combined failure TikZ saved to: {comparison_tikz_file}")


if __name__ == "__main__":
    main()
