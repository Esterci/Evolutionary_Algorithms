"""Self-adaptive DE/rand/1/bin for the constrained speed reducer."""

import csv
from pathlib import Path

import torch as tc


POPULATION_SIZE = 30
DIMENSION = 7
MAX_FITNESS_EVALUATIONS = 3030
EVALUATION_STEP = 30
NUMBER_OF_RUNS = 20
PENALTY_FACTOR = 1.0e7

LOWER_BOUNDS = tc.tensor(
    [2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0], dtype=tc.float64
)
UPPER_BOUNDS = tc.tensor(
    [3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5], dtype=tc.float64
)


def speed_reducer_objective(x):
    """Return the reducer weight for one solution or a population."""

    x1, x2, x3, x4, x5, x6, x7 = tc.unbind(x, dim=-1)
    return (
        0.7854 * x1 * x2**2
        * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
        - 1.508 * x1 * (x6**2 + x7**2)
        + 7.4777 * (x6**3 + x7**3)
        + 0.7854 * (x4 * x6**2 + x5 * x7**2)
    )


def constraint_values(x):
    """Return normalized constraints g(x), all feasible when g(x) <= 0."""

    x1, x2, x3, x4, x5, x6, x7 = tc.unbind(x, dim=-1)
    return tc.stack(
        (
            27.0 / (x1 * x2**2 * x3) - 1.0,
            397.5 / (x1 * x2**2 * x3**2) - 1.0,
            1.93 * x4**3 / (x2 * x3 * x6**4) - 1.0,
            1.93 * x5**3 / (x2 * x3 * x7**4) - 1.0,
            tc.sqrt((745.0 * x4 / (x2 * x3))**2 + 16.9e6)
            / (110.0 * x6**3) - 1.0,
            tc.sqrt((745.0 * x5 / (x2 * x3))**2 + 157.5e6)
            / (85.0 * x7**3) - 1.0,
            x2 * x3 / 40.0 - 1.0,
            5.0 * x2 / x1 - 1.0,
            x1 / (12.0 * x2) - 1.0,
            (1.5 * x6 + 1.9) / x4 - 1.0,
            (1.1 * x7 + 1.9) / x5 - 1.0,
        ),
        dim=-1,
    )


def evaluate_speed_reducer(x, penalty_factor=PENALTY_FACTOR):
    """Return penalized fitness, objective, and nonnegative violations."""

    objective = speed_reducer_objective(x)
    violations = tc.clamp(constraint_values(x), min=0.0)
    fitness = objective + penalty_factor * tc.sum(violations**2, dim=-1)
    return fitness, objective, violations


def repair_design(x, lower_bounds, upper_bounds):
    """Enforce the box constraints of the design variables."""

    return tc.maximum(tc.minimum(x, upper_bounds), lower_bounds)


def population_statistics(fitness):
    """Return population mean and population standard deviation."""

    return {
        "fit_mean": fitness.mean().item(),
        "fit_std": fitness.std(unbiased=False).item(),
    }


def save_evolution_curve(curve, output_directory, population_size, seed,
                         max_fitness_evaluations, evaluation_step):
    """Save one execution's convergence curve as CSV."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"seed-{seed}--pop-{population_size}"
        f"--maxeval-{max_fitness_evaluations}--step-{evaluation_step}.csv"
    )
    output_path = output_directory / filename

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "seed",
                "fitness_evaluations",
                "fit_mean",
                "fit_std",
            ),
        )
        writer.writeheader()
        writer.writerows(curve)

    return output_path


def save_final_result(result, output_directory, population_size,
                      max_fitness_evaluations, evaluation_step):
    """Append one execution's final result to the configuration CSV."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"pop-{population_size}"
        f"--maxeval-{max_fitness_evaluations}--step-{evaluation_step}.csv"
    )
    output_path = output_directory / filename
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    return output_path


def differential_evolution(
    population_size,
    dimension,
    max_fitness_evaluations,
    evaluation_step,
    seed,
    lower_bound=LOWER_BOUNDS,
    upper_bound=UPPER_BOUNDS,
    output_directory=None,
    results_directory=None,
):
    """Minimize speed-reducer weight with a quadratic exterior penalty."""

    if population_size < 4:
        raise ValueError("population_size must be at least 4")
    if dimension != DIMENSION:
        raise ValueError(f"dimension must be {DIMENSION}")
    if max_fitness_evaluations < population_size:
        raise ValueError(
            "max_fitness_evaluations must be at least population_size"
        )
    if evaluation_step < 1:
        raise ValueError("evaluation_step must be positive")
    lower_bound = tc.as_tensor(lower_bound, dtype=tc.float64)
    upper_bound = tc.as_tensor(upper_bound, dtype=tc.float64)
    if lower_bound.shape != (dimension,) or upper_bound.shape != (dimension,):
        raise ValueError("bounds must contain one value per dimension")
    if tc.any(lower_bound >= upper_bound):
        raise ValueError("each lower bound must be smaller than its upper bound")
    generator = tc.Generator()
    generator.manual_seed(seed)

    population = lower_bound + (upper_bound - lower_bound) * tc.rand(
        population_size,
        dimension,
        dtype=tc.float64,
        generator=generator,
    )
    population = repair_design(population, lower_bound, upper_bound)
    fitness, objectives, violations = evaluate_speed_reducer(population)
    fitness_evaluations = population_size
    initial_statistics = population_statistics(fitness)
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
            target_indices = tc.randperm(
                population_size, generator=generator
            )[:evaluations_this_generation].tolist()
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
                population[donor_indices[1]]
                - population[donor_indices[2]]
            )
            mutant = repair_design(mutant, lower_bound, upper_bound)

            crossover_mask = (
                tc.rand(dimension, generator=generator) < crossover_rate
            )
            forced_dimension = tc.randint(
                dimension, (1,), generator=generator
            )
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

            trial_fitness, trial_objective, trial_violations = (
                evaluate_speed_reducer(trial)
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
                statistics = population_statistics(new_fitness)
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

    best_index = tc.argmin(fitness)

    if evolution_curve[-1]["fitness_evaluations"] != fitness_evaluations:
        final_statistics = population_statistics(fitness)
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
        "algorithm": "Self-adaptive DE/rand/1/bin with quadratic penalty",
        "best_f": fitness[best_index].item(),
        "best_objective": objectives[best_index].item(),
        "penalty": (fitness[best_index] - objectives[best_index]).item(),
        "max_constraint_violation": violations[best_index].max().item(),
        "is_feasible": bool(tc.all(violations[best_index] <= 1.0e-8)),
        "penalty_factor": PENALTY_FACTOR,
        "population_size": population_size,
        "dimension": dimension,
        "max_fitness_evaluations": max_fitness_evaluations,
        "evaluation_step": evaluation_step,
        "completed_generations": completed_generations,
        "partial_generation_evaluations": partial_generation_evaluations,
        "seed": seed,
        "mean_differential_weight": differential_weights.mean().item(),
        "std_differential_weight": differential_weights.std(
            unbiased=False
        ).item(),
        "mean_crossover_rate": crossover_rates.mean().item(),
        "std_crossover_rate": crossover_rates.std(unbiased=False).item(),
        "fitness_evaluations": fitness_evaluations,
        "dtype": "float64",
        "device": "cpu",
        "evolution_curve_file": str(curve_path),
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
        differential_evolution(
            population_size=POPULATION_SIZE,
            dimension=DIMENSION,
            max_fitness_evaluations=MAX_FITNESS_EVALUATIONS,
            evaluation_step=EVALUATION_STEP,
            seed=run,
        )


if __name__ == "__main__":
    main()
