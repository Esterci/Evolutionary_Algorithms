"""Self-adaptive DE/rand/1/bin for the constrained pressure vessel."""

import csv
from pathlib import Path

import torch as tc

POPULATION_SIZE = 30
MAX_FITNESS_EVALUATIONS = 80000
EVALUATION_STEP = 30
NUMBER_OF_RUNS = 2
NUMBER_OF_CONSTRAINTS = 4
FEASIBILITY_TOLERANCE = 1.0e-8
PENALTY_WEIGHT = 1e4
THICKNESS_INCREMENT = 0.0625

# The source prints 0.00625 as the lower thickness bound while requiring
# 0.0625 increments. Use the first positive value on the specified grid.
LOWER_BOUNDS = tc.tensor([0.0625, 0.0625, 10.0, 10.0], dtype=tc.float64)

UPPER_BOUNDS = tc.tensor([5.0, 5.0, 200.0, 200.0], dtype=tc.float64)


def pressure_vessel_objective(x):
    """Return vessel fabrication cost for one solution or a population."""

    # The equation printed in the source omits L-dependent terms. This
    # standard complete form reproduces its reported reference cost 6059.7143.
    return (
        0.6224 * x[..., 0] * x[..., 2] * x[..., 3]
        + 1.7781 * x[..., 1] * x[..., 2] ** 2
        + 3.1661 * x[..., 0] ** 2 * x[..., 3]
        + 19.84 * x[..., 0] ** 2 * x[..., 2]
    )


def constraint_values(x):
    """Return constraints g(x), all feasible when g(x) <= 0."""

    vessel_volume = (
        tc.pi * x[..., 2] ** 2 * x[..., 3]
        + (4.0 / 3.0) * tc.pi * x[..., 2] ** 3
    )
    # Normalize the source constraints without changing their feasible set.
    return tc.stack(
        (
            0.0193 * x[..., 2] / x[..., 0] - 1.0,
            0.00954 * x[..., 2] / x[..., 1] - 1.0,
            1_296_000.0 / vessel_volume - 1.0,
            x[..., 3] / 240.0 - 1.0,
        ),
        dim=-1,
    )


def evaluate_pressure_vessel(x, penalty_weight=PENALTY_WEIGHT):
    """Return penalized fitness, objective, and nonnegative violations."""

    if penalty_weight < 0:
        raise ValueError("penalty_weight must be nonnegative")

    objective = pressure_vessel_objective(x)
    violations = tc.clamp(constraint_values(x), min=0.0)
    fitness = objective + tc.sum(penalty_weight * violations, dim=-1)
    return fitness, objective, violations


def repair_design(x, lower_bounds, upper_bounds):
    """Enforce the box constraints of the design variables."""
    repaired = tc.maximum(tc.minimum(x, upper_bounds), lower_bounds)

    repaired[..., :2] = (
        tc.round(repaired[..., :2] / THICKNESS_INCREMENT)
        * THICKNESS_INCREMENT
    )
    repaired = tc.maximum(tc.minimum(repaired, upper_bounds), lower_bounds)

    return repaired


def population_statistics(fitness, violations):
    """Return population mean and population standard deviation."""

    statistics = {
        "fit_mean": fitness.mean().item(),
        "fit_std": fitness.std(unbiased=False).item(),
    }
    for index in range(violations.shape[-1]):
        constraint_violations = violations[:, index]
        prefix = f"constraint_{index + 1}_violation"
        statistics[f"{prefix}_mean"] = constraint_violations.mean().item()
        statistics[f"{prefix}_std"] = constraint_violations.std(unbiased=False).item()
    return statistics


def select_best_index(fitness, objectives, violations):
    """Select the lowest-objective feasible solution, with a penalized fallback."""

    feasible_mask = tc.all(violations <= FEASIBILITY_TOLERANCE, dim=-1)
    feasible_indices = tc.nonzero(feasible_mask, as_tuple=False).flatten()
    if feasible_indices.numel() > 0:
        local_best = tc.argmin(objectives[feasible_indices])
        return feasible_indices[local_best], feasible_mask

    return tc.argmin(fitness), feasible_mask


def save_evolution_curve(
    curve,
    output_directory,
    population_size,
    seed,
    max_fitness_evaluations,
    evaluation_step,
):
    """Save one execution's convergence curve as CSV."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"seed-{seed}--pop-{population_size}"
        f"--maxeval-{max_fitness_evaluations}--step-{evaluation_step}.csv"
    )
    output_path = output_directory / filename

    with output_path.open("w", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=curve[0].keys(),
        )
        writer.writeheader()
        writer.writerows(curve)

    return output_path


def save_final_result(
    result, output_directory, population_size, max_fitness_evaluations, evaluation_step
):
    """Append one execution's final result to the configuration CSV."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"pop-{population_size}"
        f"--maxeval-{max_fitness_evaluations}--step-{evaluation_step}.csv"
    )
    candidate_names = (
        filename,
        filename.replace(".csv", "--penalty-weight.csv"),
        filename.replace(".csv", "--feasibility.csv"),
    )
    output_path = None
    for candidate_name in candidate_names:
        candidate_path = output_directory / candidate_name
        if not candidate_path.exists() or candidate_path.stat().st_size == 0:
            output_path = candidate_path
            break
        with candidate_path.open("r", newline="", encoding="utf-8") as csv_file:
            existing_fields = csv.DictReader(csv_file).fieldnames or []
        if list(result.keys()) == existing_fields:
            output_path = candidate_path
            break
    if output_path is None:
        raise ValueError("no compatible final-result CSV schema is available")
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    return output_path


def differential_evolution(
    seed,
    population_size=POPULATION_SIZE,
    max_fitness_evaluations=MAX_FITNESS_EVALUATIONS,
    evaluation_step=EVALUATION_STEP,
    lower_bound=LOWER_BOUNDS,
    upper_bound=UPPER_BOUNDS,
    penalty_weight=PENALTY_WEIGHT,
    output_directory=None,
    results_directory=None,
):
    """Minimize pressure-vessel cost with a quadratic exterior penalty."""

    # Checking input values
    if population_size < 4:
        raise ValueError("population_size must be at least 4")

    if max_fitness_evaluations < population_size:
        raise ValueError("max_fitness_evaluations must be at least population_size")
    if evaluation_step < 1:
        raise ValueError("evaluation_step must be positive")

    dimension = len(lower_bound)

    if lower_bound.shape != upper_bound.shape:
        raise ValueError("bounds must have the same dimension sizes")
    if dimension != 4:
        raise ValueError("the pressure-vessel problem requires four variables")

    if tc.any(lower_bound >= upper_bound):
        raise ValueError("each lower bound must be smaller than its upper bound")

    # Set seed
    generator = tc.Generator()
    generator.manual_seed(seed)

    # initialize random population
    population = lower_bound + (upper_bound - lower_bound) * tc.rand(
        population_size,
        dimension,
        dtype=tc.float64,
        generator=generator,
    )
    population = repair_design(population, lower_bound, upper_bound)

    fitness, objectives, violations = evaluate_pressure_vessel(
        population, penalty_weight
    )
    fitness_evaluations = population_size
    initial_statistics = population_statistics(fitness, violations)
    evolution_curve = [
        {
            "seed": seed,
            "fitness_evaluations": fitness_evaluations,
            **initial_statistics,
        }
    ]
    next_curve_evaluation = fitness_evaluations + evaluation_step

    # Each population member carries its own F and CR values.
    differential_weights = 0.3 + 0.6 * tc.rand(
        population_size, dtype=tc.float64, generator=generator
    )
    crossover_rates = 0.9 + 0.1 * tc.rand(
        population_size, dtype=tc.float64, generator=generator
    )
    completed_generations = 0
    partial_generation_evaluations = 0

    while fitness_evaluations < max_fitness_evaluations:
        evaluations_this_generation = min(
            population_size,
            max_fitness_evaluations - fitness_evaluations,
        )
        new_population = population.clone()
        new_fitness = fitness.clone()
        new_objectives = objectives.clone()
        new_violations = violations.clone()
        new_differential_weights = differential_weights.clone()
        new_crossover_rates = crossover_rates.clone()

        if evaluations_this_generation < population_size:
            target_indices = tc.randperm(population_size, generator=generator)[
                :evaluations_this_generation
            ].tolist()
        else:
            target_indices = range(population_size)

        for i in target_indices:
            available = tc.arange(population_size)
            available = available[available != i]
            donor_indices = available[
                tc.randperm(population_size - 1, generator=generator)[:3]
            ]

            differential_weight = differential_weights[i]
            crossover_rate = crossover_rates[i]

            mutant = population[donor_indices[0]] + differential_weight * (
                population[donor_indices[1]] - population[donor_indices[2]]
            )
            mutant = repair_design(mutant, lower_bound, upper_bound)

            crossover_mask = tc.rand(dimension, generator=generator) < crossover_rate
            forced_dimension = tc.randint(dimension, (1,), generator=generator)
            crossover_mask[forced_dimension] = True
            trial = tc.where(crossover_mask, mutant, population[i])
            trial = repair_design(trial, lower_bound, upper_bound)

            # Mutate encoded parameters only when all decision variables come
            # from the mutant; otherwise the target parameters are inherited.
            if crossover_mask.all():
                child_differential_weight = (
                    differential_weights[donor_indices[0]]
                    + differential_weight
                    * (
                        differential_weights[donor_indices[1]]
                        - differential_weights[donor_indices[2]]
                    )
                ).clamp(0.3, 0.9)
                child_crossover_rate = (
                    crossover_rates[donor_indices[0]]
                    + differential_weight
                    * (
                        crossover_rates[donor_indices[1]]
                        - crossover_rates[donor_indices[2]]
                    )
                ).clamp(0.9, 1.0)
            else:
                child_differential_weight = differential_weight
                child_crossover_rate = crossover_rate

            trial_fitness, trial_objective, trial_violations = evaluate_pressure_vessel(
                trial, penalty_weight
            )
            fitness_evaluations += 1

            if trial_fitness < fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                new_objectives[i] = trial_objective
                new_violations[i] = trial_violations
                new_differential_weights[i] = child_differential_weight
                new_crossover_rates[i] = child_crossover_rate
            else:
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]
                new_differential_weights[i] = differential_weight
                new_crossover_rates[i] = crossover_rate

            if fitness_evaluations == next_curve_evaluation:
                statistics = population_statistics(new_fitness, new_violations)
                evolution_curve.append(
                    {
                        "seed": seed,
                        "fitness_evaluations": fitness_evaluations,
                        **statistics,
                    }
                )
                next_curve_evaluation += evaluation_step

        population = new_population
        fitness = new_fitness
        objectives = new_objectives
        violations = new_violations
        differential_weights = new_differential_weights
        crossover_rates = new_crossover_rates

        if evaluations_this_generation == population_size:
            completed_generations += 1
        else:
            partial_generation_evaluations = evaluations_this_generation

    best_index, feasible_mask = select_best_index(
        fitness, objectives, violations
    )

    feasible_percentage = 100.0 * feasible_mask.to(tc.float64).mean().item()

    if evolution_curve[-1]["fitness_evaluations"] != fitness_evaluations:
        final_statistics = population_statistics(fitness, violations)
        evolution_curve.append(
            {
                "seed": seed,
                "fitness_evaluations": fitness_evaluations,
                **final_statistics,
            }
        )

    if output_directory is None:
        output_directory = Path(__file__).resolve().parent / "evolution_curves"

    curve_path = save_evolution_curve(
        evolution_curve,
        output_directory,
        population_size,
        seed,
        max_fitness_evaluations,
        evaluation_step,
    )

    result = {
        "algorithm": (
            "Self-adaptive DE/rand/1/bin for pressure vessel "
            "with quadratic penalty"
        ),
        "best_f": fitness[best_index].item(),
        "best_objective": objectives[best_index].item(),
        "penalty": (fitness[best_index] - objectives[best_index]).item(),
        "max_constraint_violation": violations[best_index].max().item(),
        "is_feasible": bool(feasible_mask[best_index].item()),
        "final_population_feasible_percentage": feasible_percentage,
        "population_size": population_size,
        "dimension": dimension,
        "max_fitness_evaluations": max_fitness_evaluations,
        "evaluation_step": evaluation_step,
        "completed_generations": completed_generations,
        "partial_generation_evaluations": partial_generation_evaluations,
        "seed": seed,
        "mean_differential_weight": differential_weights.mean().item(),
        "std_differential_weight": differential_weights.std(unbiased=False).item(),
        "mean_crossover_rate": crossover_rates.mean().item(),
        "std_crossover_rate": crossover_rates.std(unbiased=False).item(),
        "fitness_evaluations": fitness_evaluations,
        "dtype": "float64",
        "device": "cpu",
        "evolution_curve_file": str(curve_path),
        "penalty_weight": penalty_weight,
    }
    for index, value in enumerate(population[best_index].tolist()):
        result[f"best_x_{index}"] = value

    if results_directory is None:
        results_directory = Path(__file__).resolve().parent / "results"
    result_path = save_final_result(
        result,
        results_directory,
        population_size,
        max_fitness_evaluations,
        evaluation_step,
    )

    print(f"Evolution curve saved to: {curve_path}")
    print(f"Final result saved to: {result_path}")


def main():
    for run in range(1, NUMBER_OF_RUNS + 1):
        print(f"Run {run}/{NUMBER_OF_RUNS} with seed {run}")
        differential_evolution(seed=run)


if __name__ == "__main__":
    main()
