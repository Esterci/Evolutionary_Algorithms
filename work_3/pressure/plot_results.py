"""Create convergence plots for the independent DE runs."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASE_DIRECTORY = Path(__file__).resolve().parent
CURVES_DIRECTORY = BASE_DIRECTORY / "evolution_curves"
RESULTS_DIRECTORY = BASE_DIRECTORY / "results"
PLOTS_DIRECTORY = BASE_DIRECTORY / "evolution_plots"
MAX_PLOT_POINTS = 600

CURVE_FILENAME = re.compile(
    r"seed-(?P<seed>\d+)"
    r"--pop-(?P<population_size>\d+)"
    r"--maxeval-(?P<max_fitness_evaluations>\d+)"
    r"--step-(?P<evaluation_step>\d+)\.csv"
)

REQUIRED_COLUMNS = {
    "seed",
    "fitness_evaluations",
    "fit_mean",
    "fit_std",
}

VIOLATION_COLUMN = re.compile(
    r"constraint_(?P<constraint>\d+)_violation_(?P<statistic>mean|std)"
)

RESULT_COLUMNS = {
    "best_f",
    "best_objective",
    "max_constraint_violation",
    "is_feasible",
    "population_size",
    "max_fitness_evaluations",
    "evaluation_step",
    "seed",
    "mean_differential_weight",
    "std_differential_weight",
    "mean_crossover_rate",
    "std_crossover_rate",
}


def load_curve_groups(curves_directory):
    """Load curve files and group them by experimental configuration."""

    curve_groups = {}

    for curve_path in sorted(Path(curves_directory).glob("*.csv")):
        match = CURVE_FILENAME.fullmatch(curve_path.name)
        if match is None:
            print(f"Skipping unrecognized curve file: {curve_path.name}")
            continue

        with curve_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            missing_columns = REQUIRED_COLUMNS.difference(
                reader.fieldnames or ()
            )
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(
                    f"{curve_path.name} is missing columns: {missing}"
                )
            rows = list(reader)

        if not rows:
            raise ValueError(f"{curve_path.name} contains no curve points")

        curve = {
            "seed": np.array([int(row["seed"]) for row in rows]),
            "fitness_evaluations": np.array(
                [int(row["fitness_evaluations"]) for row in rows]
            ),
            "fit_mean": np.array(
                [float(row["fit_mean"]) for row in rows]
            ),
            "fit_std": np.array(
                [float(row["fit_std"]) for row in rows]
            ),
        }
        violation_columns = sorted(
            column for column in (reader.fieldnames or ())
            if VIOLATION_COLUMN.fullmatch(column)
        )
        for column in violation_columns:
            curve[column] = np.array([float(row[column]) for row in rows])

        seed = int(match.group("seed"))
        recorded_seeds = set(curve["seed"].tolist())
        if recorded_seeds != {seed}:
            raise ValueError(
                f"{curve_path.name} contains seeds {sorted(recorded_seeds)}"
            )

        configuration = (
            int(match.group("population_size")),
            int(match.group("max_fitness_evaluations")),
            int(match.group("evaluation_step")),
        )
        curve_groups.setdefault(configuration, []).append((seed, curve))

    if not curve_groups:
        raise FileNotFoundError(
            f"No valid evolution curves found in {curves_directory}"
        )

    return curve_groups


def reduce_curve(curve, max_plot_points):
    """Keep evenly spaced rows when a curve is too dense for plotting."""

    curve_length = len(curve["fitness_evaluations"])
    if curve_length <= max_plot_points:
        return curve

    indices = np.linspace(
        0,
        curve_length - 1,
        max_plot_points,
        dtype=int,
    )
    indices = np.unique(indices)
    return {column: values[indices] for column, values in curve.items()}


def load_best_result(configuration, results_directory, require_penalty_weight=False):
    """Load the best recorded run for one experimental configuration."""

    population_size, max_evaluations, evaluation_step = configuration
    candidates = []

    for result_path in sorted(Path(results_directory).glob("*.csv")):
        with result_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if not RESULT_COLUMNS.issubset(reader.fieldnames or ()):
                continue

            for row in reader:
                if require_penalty_weight and "penalty_weight" not in row:
                    continue
                row_configuration = (
                    int(row["population_size"]),
                    int(row["max_fitness_evaluations"]),
                    int(row["evaluation_step"]),
                )
                if row_configuration == configuration:
                    candidates.append(row)

    if not candidates:
        raise FileNotFoundError(
            "No result row matches configuration "
            f"population={population_size}, max evaluations={max_evaluations}, "
            f"step={evaluation_step}"
        )

    current_schema_candidates = [
        row for row in candidates
        if "final_population_feasible_percentage" in row
    ]
    if current_schema_candidates:
        candidates = current_schema_candidates

    feasible_candidates = [
        row for row in candidates
        if row["is_feasible"].strip().lower() == "true"
    ]
    if feasible_candidates:
        return min(
            feasible_candidates,
            key=lambda row: float(row["best_objective"]),
        )

    return min(candidates, key=lambda row: float(row["best_f"]))


def plot_configuration(configuration, seed_curves, results_directory,
                       plots_directory):
    """Plot the population-fitness distribution of the best run."""

    population_size, max_evaluations, evaluation_step = configuration
    curves_by_seed = dict(seed_curves)
    curves_have_violations = any(
        VIOLATION_COLUMN.fullmatch(column)
        for curve in curves_by_seed.values()
        for column in curve
    )
    best_result = load_best_result(
        configuration,
        results_directory,
        require_penalty_weight=curves_have_violations,
    )
    best_seed = int(best_result["seed"])
    if best_seed not in curves_by_seed:
        raise FileNotFoundError(
            f"Evolution curve for best seed {best_seed} was not found"
        )

    curve = reduce_curve(curves_by_seed[best_seed], MAX_PLOT_POINTS)
    evaluations = curve["fitness_evaluations"]
    mean_fitness = curve["fit_mean"]
    fitness_std = curve["fit_std"]
    feasible_percentage = best_result.get(
        "final_population_feasible_percentage"
    )

    constraint_numbers = sorted(
        int(match.group("constraint"))
        for column in curve
        if (match := VIOLATION_COLUMN.fullmatch(column))
        and match.group("statistic") == "mean"
        and f"constraint_{match.group('constraint')}_violation_std" in curve
    )
    number_of_panels = 2 if constraint_numbers else 1
    figure, axes = plt.subplots(
        number_of_panels,
        1,
        figsize=(12, 6 * number_of_panels),
    )
    axis = np.atleast_1d(axes)[0]
    axis.plot(
        evaluations,
        mean_fitness,
        color="tab:blue",
        linewidth=2.0,
        label="Aptidão penalizada média da população",
    )
    axis.fill_between(
        evaluations,
        mean_fitness - fitness_std,
        mean_fitness + fitness_std,
        color="tab:blue",
        alpha=0.2,
        label="± 1 desvio-padrão da população",
    )

    axis.set_xlabel("Avaliações da função objetivo")
    axis.set_ylabel("Aptidão penalizada")
    axis.set_title("Evolução da aptidão penalizada na melhor execução")
    axis.set_xlim(1, 700)
    axis.grid(True, alpha=0.3)
    axis.legend()

    if constraint_numbers:
        violation_axis = np.atleast_1d(axes)[1]
        for constraint in constraint_numbers:
            mean = curve[f"constraint_{constraint}_violation_mean"]
            std = curve[f"constraint_{constraint}_violation_std"]
            line, = violation_axis.plot(
                evaluations, mean, linewidth=1.5, label=f"Restrição {constraint}"
            )
            violation_axis.fill_between(
                evaluations,
                np.maximum(mean - std, 0.0),
                mean + std,
                color=line.get_color(),
                alpha=0.12,
            )
        violation_axis.set_xlabel("Avaliações da função objetivo")
        violation_axis.set_ylabel("Violação")
        violation_axis.set_title(
            "Violações médias por restrição (faixa: ± 1 desvio-padrão)"
        )
        violation_axis.grid(True, alpha=0.3)
        violation_axis.legend(ncol=3, fontsize=9)
        violation_axis.set_xlim(1, 700)

    figure.suptitle(
        f"Melhor seed: {best_seed} | Aptidão: "
        f"{float(best_result['best_f']):.6g} \n"
        f"F = {float(best_result['mean_differential_weight']):.4f} ± "
        f"{float(best_result['std_differential_weight']):.4f} | "
        f"CR = {float(best_result['mean_crossover_rate']):.4f} ± "
        f"{float(best_result['std_crossover_rate']):.4f}"
        + (
            f" | População viável = {float(feasible_percentage):.2f}%"
            if feasible_percentage is not None
            else ""
        ),
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9 if constraint_numbers else 0.87))

    plots_directory = Path(plots_directory)
    plots_directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"melhor_execucao--pop-{population_size}"
        f"--maxeval-{max_evaluations}--step-{evaluation_step}"
    )
    figure_path = plots_directory / f"{stem}.png"

    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Gráfico salvo em: {figure_path}")


def create_plots(curves_directory=CURVES_DIRECTORY,
                 results_directory=RESULTS_DIRECTORY,
                 plots_directory=PLOTS_DIRECTORY):
    """Create one convergence figure for every stored configuration."""

    curve_groups = load_curve_groups(curves_directory)
    for configuration, seed_curves in sorted(curve_groups.items()):
        plot_configuration(
            configuration,
            seed_curves,
            results_directory,
            plots_directory,
        )


if __name__ == "__main__":
    create_plots()
